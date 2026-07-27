# current_development/plus_minus_ratings/r00_data_qa.jl
#
# RUNNER. WP1 — the data-integrity gate for the RAPM stream.
#
# Everything downstream rests on one claim: that we can reconstruct, for every match, exactly
# which 22 players were on the pitch during every interval of play. This file tries to break
# that claim before we build 5,000 lines on top of it.
#
# GATES (each prints a table; paste the verdict into NOTES.md):
#   G1 coverage      — matches / lineups / incidents / live_text per tier × season
#   G2 id resolution — do substitution player ids resolve to a lineup row in that match?
#   G3 minutes       — incident-reconstructed minutes vs `minutes_played`, on 23/24–24/25 only
#                      (the seasons where the column is trustworthy). THE decisive gate: it
#                      validates the incident route on the only data where it can be checked,
#                      before we rely on it for seasons where `minutes_played` is dead.
#   G4 hole fill     — can BBC live_text cover tier 56's incident holes? (needs text parsing;
#                      <95% parse ⇒ drop those seasons rather than half-fill them)
#   G5 goals         — incident goal times reconcile with the final score
#   G6 positions     — position coverage, and the true-vs-defaulted split the old extractor hid
#
# Run:
#   include(".../plus_minus_ratings/r00_data_qa.jl")
# Results land in the global `QA` NamedTuple for interactive poking.

using DataFrames
using Statistics
using Printf

include(joinpath(@__DIR__, "l00_pm_data.jl"))

const DAT = ensure_pm_data!()
const LU  = DAT.lineups
const IN  = DAT.incidents
const LT  = DAT.livetext

_hdr(s) = println("\n", "="^78, "\n", s, "\n", "="^78)

# Boolean masks over columns that may hold `missing` — DataFrames refuses to index with
# `Union{Missing,Bool}`, and a missing here always means "not a match".
_mask(v) = coalesce.(v, false)

# ==========================================
# G1 — COVERAGE
# ==========================================
_hdr("G1 — coverage by tier × season")

lu_by_match = combine(groupby(LU, [:match_id, :tournament_id, :season]), nrow => :n_lineup)
sub_matches = Set(IN.match_id[IN.incident_type .== "substitution"])
lt_matches  = Set(LT.match_id)

lu_by_match.has_subs = [m in sub_matches for m in lu_by_match.match_id]
lu_by_match.has_lt   = [m in lt_matches  for m in lu_by_match.match_id]

g1 = combine(groupby(lu_by_match, [:tournament_id, :season]),
             nrow                    => :matches,
             :has_subs => sum         => :m_incidents,
             :has_lt   => sum         => :m_livetext,
             :n_lineup => mean        => :mean_teamsheet)
sort!(g1, [:tournament_id, :season])
g1.pct_incidents = round.(100 .* g1.m_incidents ./ g1.matches, digits = 1)
g1.pct_livetext  = round.(100 .* g1.m_livetext  ./ g1.matches, digits = 1)
g1.mean_teamsheet = round.(g1.mean_teamsheet, digits = 1)
println(g1)

# ==========================================
# G2 — SUBSTITUTION ID RESOLUTION
# ==========================================
_hdr("G2 — do substitution player ids resolve to a lineup row in that match?")

# (match_id, player_id) -> was on the teamsheet
sheet = Set((Int(r.match_id), Int(r.player_id)) for r in eachrow(LU) if !ismissing(r.player_id))
starters = Set((Int(r.match_id), Int(r.player_id)) for r in eachrow(LU)
               if !ismissing(r.player_id) && r.is_starter === true)

subs = IN[IN.incident_type .== "substitution", :]
subs = subs[.!ismissing.(subs.player_in_id) .& .!ismissing.(subs.player_out_id), :]

subs.in_on_sheet   = [(Int(r.match_id), Int(r.player_in_id))  in sheet    for r in eachrow(subs)]
subs.out_on_sheet  = [(Int(r.match_id), Int(r.player_out_id)) in sheet    for r in eachrow(subs)]
# A player coming ON should NOT be a starter; a player going OFF may be a starter or an
# earlier substitute (a sub being subbed). Both are legal — we only flag the first.
subs.in_was_starter = [(Int(r.match_id), Int(r.player_in_id)) in starters for r in eachrow(subs)]

g2 = combine(groupby(subs, :tournament_id),
             nrow                     => :n_subs,
             :in_on_sheet    => mean  => :pct_in_resolved,
             :out_on_sheet   => mean  => :pct_out_resolved,
             :in_was_starter => mean  => :pct_in_was_starter)
for c in (:pct_in_resolved, :pct_out_resolved, :pct_in_was_starter)
    g2[!, c] = round.(100 .* g2[!, c], digits = 2)
end
println(g2)
println("\nNULL-id substitution rows dropped: ",
        sum(IN.incident_type .== "substitution") - nrow(subs))

# WHY do ~3-4% of lower-tier sub ids miss the teamsheet? The plausible cause is an incomplete
# teamsheet (bench not fully scraped), which would be visible as a smaller teamsheet on exactly
# those matches. If instead the failures are spread evenly over normal-sized teamsheets, the
# cause is id drift and is far more serious.
sheet_size = Dict(Int(r.match_id) => r.n_lineup for r in eachrow(lu_by_match))
subs.sheet_n = [get(sheet_size, Int(r.match_id), 0) for r in eachrow(subs)]
subs.unresolved = .!(subs.in_on_sheet .& subs.out_on_sheet)
g2b = combine(groupby(subs[subs.tournament_id .∈ Ref([56, 57]), :], :unresolved),
              nrow                => :n_subs,
              :sheet_n => mean    => :mean_teamsheet_size,
              :sheet_n => minimum => :min_teamsheet_size)
g2b.mean_teamsheet_size = round.(g2b.mean_teamsheet_size, digits = 1)
println("\nUnresolved-vs-teamsheet-size diagnostic (tiers 56/57):")
println(g2b)
println("Smaller teamsheets on the unresolved rows ⇒ incomplete bench scrape (tolerable:")
println("treat the player as an unmodelled entrant). Equal sizes ⇒ id drift (must be fixed).")

# ==========================================
# G3 — MINUTES RECONSTRUCTION  (the decisive WP1 gate)
# ==========================================
_hdr("G3 — incident-reconstructed minutes vs `minutes_played` (23/24 & 24/25 only)")

"""
Reconstruct on-pitch minutes for every teamsheet player of one match from its substitutions
and dismissals. Starters begin at 0; substitutes begin when they come on; anyone still on at
the whistle ends at `full`. Returns Dict(player_id => minutes).

Deliberately ignores stoppage time (`full` defaults to 90) so it is comparable with SofaScore's
`minutes_played`, which is also capped at 90.
"""
function reconstruct_minutes(match_lu::AbstractDataFrame, match_in::AbstractDataFrame;
                             full::Float64 = 90.0)
    on  = Dict{Int, Float64}()   # player -> minute they came on
    off = Dict{Int, Float64}()   # player -> minute they went off

    for r in eachrow(match_lu)
        ismissing(r.player_id) && continue
        r.is_starter === true && (on[Int(r.player_id)] = 0.0)
    end

    for r in eachrow(match_in)
        t = pm_full_time(r.time, r.added_time)
        isnan(t) && continue
        t = clamp(t, 0.0, full)
        if r.incident_type == "substitution"
            ismissing(r.player_in_id)  || (on[Int(r.player_in_id)]   = t)
            ismissing(r.player_out_id) || (off[Int(r.player_out_id)] = t)
        elseif r.incident_type == "card" &&
               (r.incident_class == "red" || r.incident_class == "yellowRed")
            ismissing(r.player_id) || (off[Int(r.player_id)] = t)
        end
    end

    out = Dict{Int, Float64}()
    for (pid, t_on) in on
        out[pid] = max(0.0, get(off, pid, full) - t_on)
    end
    # Named-but-unused substitutes played nothing.
    for r in eachrow(match_lu)
        ismissing(r.player_id) && continue
        pid = Int(r.player_id)
        haskey(out, pid) || (out[pid] = 0.0)
    end
    return out
end

good_seasons = ["23/24", "24/25"]
lu_g3 = LU[in.(LU.season, Ref(good_seasons)) .& .!ismissing.(LU.minutes_played), :]
in_by_match = groupby(IN, :match_id)
in_index = Dict(Int(k.match_id) => v for (k, v) in pairs(in_by_match))

rows = NamedTuple[]
for grp in groupby(lu_g3, :match_id)
    mid = Int(grp.match_id[1])
    haskey(in_index, mid) || continue
    rec = reconstruct_minutes(grp, in_index[mid])
    for r in eachrow(grp)
        ismissing(r.player_id) && continue
        pid = Int(r.player_id)
        haskey(rec, pid) || continue
        push!(rows, (tournament_id = r.tournament_id, season = r.season,
                     is_starter = r.is_starter === true,
                     recon = rec[pid], actual = Float64(r.minutes_played)))
    end
end
m3 = DataFrame(rows)
m3.err = m3.recon .- m3.actual

g3 = combine(groupby(m3, [:tournament_id, :is_starter]),
             nrow                                  => :n,
             :err => (e -> mean(abs.(e)))          => :mae_min,
             :err => (e -> mean(abs.(e) .<= 1.0))  => :pct_within_1,
             :err => (e -> mean(abs.(e) .<= 3.0))  => :pct_within_3,
             :err => (e -> quantile(abs.(e), 0.95)) => :p95_abs_err)
for c in (:mae_min, :p95_abs_err)
    g3[!, c] = round.(g3[!, c], digits = 2)
end
for c in (:pct_within_1, :pct_within_3)
    g3[!, c] = round.(100 .* g3[!, c], digits = 1)
end
sort!(g3, [:tournament_id, :is_starter])
println(g3)
println("\nGATE: starters should be ≳95% within ±1 minute. Systematic error on SUBSTITUTES is")
println("expected and benign (SofaScore rounds the minute they entered); systematic error on")
println("STARTERS means the segment builder would be wrong.")

# ==========================================
# G4 — CAN LIVE_TEXT FILL TIER-56'S INCIDENT HOLES?
# ==========================================
_hdr("G4 — tier-56 incident holes vs live_text availability")

holes = lu_by_match[(lu_by_match.tournament_id .== 56) .& .!lu_by_match.has_subs, :]
g4 = combine(groupby(holes, :season),
             nrow            => :matches_missing_incidents,
             :has_lt => sum  => :of_which_have_livetext)
sort!(g4, :season)
println(g4)

# BBC substitution rows carry no player column — the names are only in `text`. Measure how
# often the standard phrasing parses, since that determines whether the fallback is viable.
lt_subs = LT[_mask((LT.event_type .== "substitution") .& (LT.tournament_id .== 56)), :]
const SUB_RE = r"Substitution,\s*(.+?)\.\s*(.+?)\s+replaces\s+(.+?)\."
lt_subs.parsed = [!ismissing(t) && occursin(SUB_RE, String(t)) for t in lt_subs.text]
@printf("\nlive_text substitution rows (tier 56): %d, parseable by the standard pattern: %.1f%%\n",
        nrow(lt_subs), 100 * mean(lt_subs.parsed))
println("Unparsed examples:")
for t in first(skipmissing(lt_subs.text[.!lt_subs.parsed]), 5)
    println("  · ", t)
end
println("\nGATE: <95% parse ⇒ drop tier-56's holed seasons rather than half-fill them.")
println("NOTE: even a clean parse yields NAMES, not ids — name→player_id matching against the")
println("teamsheet is a second, separate risk. Report both before trusting this route.")

# ==========================================
# G5 — GOAL RECONCILIATION
# ==========================================
_hdr("G5 — incident goal counts vs the final score")

goals = IN[IN.incident_type .== "goal", :]
gc = combine(groupby(goals, :match_id),
             :is_home => (h -> sum(skipmissing(h)))        => :inc_home,
             :is_home => (h -> sum(.!collect(skipmissing(h)))) => :inc_away)
meta = pm_match_meta(LU)
gj = innerjoin(meta, gc, on = :match_id)
gj.ok = (gj.inc_home .== gj.home_score) .& (gj.inc_away .== gj.away_score)

g5 = combine(groupby(gj, [:tournament_id, :season]),
             nrow      => :matches_with_goal_incidents,
             :ok => (o -> round(100 * mean(o), digits = 1)) => :pct_score_reconciles)
sort!(g5, [:tournament_id, :season])
println(g5)
println("\nNOTE: own goals are recorded on the side that BENEFITS in SofaScore's `is_home`, so a")
println("clean reconcile here also confirms own-goal handling. Mismatches must be listed and")
println("excluded, not silently tolerated.")

# ==========================================
# G6 — POSITIONS
# ==========================================
_hdr("G6 — position coverage (true vs unknown, NOT defaulted to M)")

LU.pos_clean = pm_clean_position.(LU.position)
g6 = combine(groupby(LU, [:tournament_id, :pos_clean]), nrow => :n)
g6 = unstack(g6, :tournament_id, :pos_clean, :n)
println(g6)
println("\nGATE: the `U` column is what `src/features/.../player_extractors.jl` would have")
println("silently folded into midfield. It must be small, and it must be reported.")

# ==========================================
# SUMMARY OBJECT
# ==========================================
const QA = (coverage = g1, id_resolution = g2, minutes = g3, minutes_raw = m3,
            holes = g4, goals = g5, positions = g6)

_hdr("WP1 done — inspect `QA`, then write the verdict into NOTES.md")
