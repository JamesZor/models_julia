# current_development/orderbook_layer2/r00_preflight.jl
#
# WP0. Build the corpus, measure the feed, and check that Ireland 718 can carry the engine.
#
# Run this BEFORE anything else in this stream. Nothing downstream is meaningful if the corpus
# is smaller than it looks, the cadence is not what the table name claims, or 718 turns out to
# lack the features the player-level engine needs.
#
# ---------------------------------------------------------------------------------------------
# GATES (stated before running, so a miss is a finding rather than a renegotiation)
# ---------------------------------------------------------------------------------------------
#
#   G1  >= 75 of the 81 known order-book fixtures enter the corpus
#   G2  >= 8 usable market types on the median fixture
#   G3  a measured cadence, with p90 not more than 2x the median  (a stable collector)
#   G4  718 carries xG and player ratings across the training window, with no all-zero xG split
#
# G4 is the one that can genuinely fail. Present-but-zero xG drives the Gamma pillar to -Inf,
# which does not error -- it makes NUTS initialisation fail, and the queued trainer then drops
# that split SILENTLY. The symptom is a short `training_results.items`, never an exception, so
# the count is checked explicitly rather than trusted.
#
# If G4 fails, 718 leaves the programme and the corpus halves to 38 fixtures. Say so loudly:
# every downstream confidence interval roughly sqrt(2) wider is a different research programme,
# not a detail.
#
# ---------------------------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------------------------
#
#   julia --project -t 32
#   using BayesianFootball, Revise
#   include("current_development/orderbook_layer2/r00_preflight.jl")

using BayesianFootball
using DataFrames, Dates, Statistics, Printf

include(joinpath(@__DIR__, "l00_corpus.jl"))

const D = BayesianFootball.Data

# ===================================================================
# 0. Scope
# ===================================================================

const IRE_TOURNAMENTS = [79, 718]          # Premier Division, First Division
const FROM = Date(2026, 5, 20)             # a week before the first known order-book tick
const TO   = Date(2026, 8, 10)             # the day after the last

println("\n", "="^95)
println("WP0 PREFLIGHT  —  Ireland order-book corpus")
println("="^95)

# ===================================================================
# 1. Build and measure the corpus
# ===================================================================

println("\n[1] Building corpus via MatchDay.MatchMetaCrosswalk ...")
corpus = build_corpus("ireland", IRE_TOURNAMENTS; from = FROM, to = TO)

display(corpus)
println()

println("\n--- per-league coverage ---")
display(corpus_report(corpus))

cad = measure_cadence(corpus)
@printf("\n--- cadence ---\nmedian %.2f min   p10 %.2f   p90 %.2f   -> recommended fine step %s\n",
        cad.median_min, cad.p10, cad.p90, cad.recommended_step)

if !isempty(corpus.excluded)
    println("\n--- excluded fixtures, by reason ---")
    display(combine(groupby(corpus.excluded, :reason), nrow => :n))
end

# The lead-time distribution decides how far back the snapshot grid can reach. Reported as
# quantiles rather than a mean because the grid must be honest for the WORST-covered fixture:
# a grid reaching past a fixture's first tick produces snapshots with no book, which the gate
# correctly reports as `no quotes retrieved` and which reads like a broken pipeline.
if !isempty(corpus.coverage)
    fl = corpus.coverage.first_lead_min
    println("\n--- pre-kickoff depth (minutes of book before KO) ---")
    @printf("  min %.0f | p10 %.0f | median %.0f | p90 %.0f | max %.0f\n",
            minimum(fl), quantile(fl, 0.1), median(fl), quantile(fl, 0.9), maximum(fl))
    for h in (60, 120, 180, 240, 360, 720)
        @printf("  fixtures with >= T-%-4d min of book: %3d / %3d\n",
                h, count(>=(h), fl), length(fl))
    end
end

# ===================================================================
# 2. Gates G1–G3
# ===================================================================

println("\n", "-"^95)
println("GATES")
println("-"^95)

const KNOWN_ORDERBOOK_FIXTURES = 81        # measured directly against betdb, 2026-08-11

g1 = length(corpus.fixtures) >= 75
@printf("G1  corpus size          %3d / %d known          %s\n",
        length(corpus.fixtures), KNOWN_ORDERBOOK_FIXTURES, g1 ? "PASS" : "FAIL")

med_mkts = isempty(corpus.coverage) ? 0.0 : median(corpus.coverage.n_markets)
g2 = med_mkts >= 8
@printf("G2  median market types  %4.1f                    %s\n", med_mkts, g2 ? "PASS" : "FAIL")

g3 = !isnan(cad.median_min) && cad.p90 <= 2 * cad.median_min
@printf("G3  cadence stability    p90 %.1f <= 2x med %.1f   %s\n",
        cad.p90, cad.median_min, g3 ? "PASS" : "FAIL")

# ===================================================================
# 3. G4 — can 718 carry the player-level engine?
# ===================================================================
#
# The engine is `DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel`, which needs BOTH a
# team-match xG pillar and per-player ratings. 718's own segment notes record that its xG starts
# in 2023 and that it has no bigChance column; neither is fatal, but an xG column that is present
# and ZERO is, for the reason in the header.

println("\n", "-"^95)
println("G4  Ireland 718 feature preflight")
println("-"^95)

"""
    feature_preflight(ds, label) -> DataFrame

Season-by-season xG and player-rating coverage for one DataStore.

xG comes from `ds.statistics` (`expectedGoals_home/away`, `period == "ALL"`) and ratings from
`ds.lineups` (`:rating`) — the same two places `XGFeature` and `PlayerRatingsFeature` read, so a
pass here means the extractors will find what they need rather than merely that the DB has rows.

PRESENT-BUT-ZERO is reported separately from MISSING because they fail in completely different
ways: a missing xG is routed around by the model's `findall(!isnan, ...)` split, whereas a
present zero reaches the Gamma pillar and returns -Inf, which kills NUTS initialisation and
makes the queued trainer drop the split without raising.
"""
function feature_preflight(ds, label::String)
    m     = ds.matches
    stats = filter(:period => ==("ALL"), ds.statistics)
    xg    = leftjoin(select(m, :match_id, :season),
                     select(stats, :match_id, :expectedGoals_home, :expectedGoals_away),
                     on = :match_id)

    lu = leftjoin(select(ds.lineups, :match_id, :rating),
                  select(m, :match_id, :season), on = :match_id)

    rows = NamedTuple[]
    for s in sort(unique(skipmissing(m.season)))
        sub  = filter(:season => ==(s), xg)
        vals = vcat(collect(skipmissing(sub.expectedGoals_home)),
                    collect(skipmissing(sub.expectedGoals_away)))
        lsub = filter(:season => ==(s), lu)
        rats = collect(skipmissing(lsub.rating))
        rats = filter(!isnan, rats)

        push!(rows, (season      = s,
                     matches     = nrow(sub),
                     xg_present  = length(vals),
                     xg_zero     = count(==(0.0), vals),
                     xg_pct      = 100 * length(vals) / max(1, 2 * nrow(sub)),
                     rating_rows = nrow(lsub),
                     rating_pct  = 100 * length(rats) / max(1, nrow(lsub))))
    end

    df = DataFrame(rows)
    println("\n$label — xG and ratings by season:")
    display(df)
    return df
end

println("\nLoading DataStores (uses .cache/ when warm) ...")
ds79  = D.load_datastore_cached(D.Ireland())
ds718 = D.load_datastore_cached(D.IrelandFirstDivision())

xg79  = feature_preflight(ds79,  "Ireland Premier (79)")
xg718 = feature_preflight(ds718, "Ireland First Division (718)")

# The training window is target 2025/2026 with 2 seasons of history, so 2023 onward must be sound.
train_seasons(df) = filter(:season => s -> string(s) >= "2023", df)

bad718 = filter(r -> r.xg_pct < 50.0 || r.xg_zero > 0 || r.rating_pct < 50.0,
                collect(eachrow(train_seasons(xg718))))
g4 = isempty(bad718)
@printf("\nG4  718 xG + ratings usable across 2023+          %s\n", g4 ? "PASS" : "FAIL")
if !g4
    println("    offending seasons:")
    for r in bad718
        @printf("      %s: xG %.0f%% present (%d zero), ratings %.0f%%\n",
                r.season, r.xg_pct, r.xg_zero, r.rating_pct)
    end
    println("\n    => 718 cannot carry the player-level engine as configured.")
    println("       The corpus halves to the 38 fixtures in tournament 79 and every downstream")
    println("       interval widens by roughly sqrt(2). Do not proceed silently.")
end

# ===================================================================
# 4. Verdict
# ===================================================================

println("\n", "="^95)
all_pass = g1 && g2 && g3 && g4
println(all_pass ? "WP0 PREFLIGHT: ALL GATES PASS" : "WP0 PREFLIGHT: GATES FAILED — see above")
println("="^95)

@printf("""

Carry forward into WP3 (corpus replay):
  fine step inside T-60      %s
  coarse step T-360 .. T-60  %s
  slates                     %d settlement windows
  fixtures                   %d  (79: %d, 718: %d)
""",
    cad.recommended_step, Minute(15), length(corpus_slates(corpus)), length(corpus.fixtures),
    count(f -> f.tournament_id == 79,  corpus.fixtures),
    count(f -> f.tournament_id == 718, corpus.fixtures))

corpus_79  = subset_corpus(corpus, 79)
corpus_718 = subset_corpus(corpus, 718)

nothing
