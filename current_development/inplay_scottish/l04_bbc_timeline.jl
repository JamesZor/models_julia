#=
l04_bbc_timeline.jl — WP-A loader: the WIDE BBC commentary timeline for 56/57.

`ds.bbc_events` (src/Data/fetchers/sql/bbc_events.jl) deliberately fetches ONLY the 7
shot-bearing event types, because the plus-minus rating family is the only consumer and
the APM port was verified bit-faithful against it (ρ = 1.000000, max abs diff 1.2e-5).
Widening that domain would silently move every PM rating for no MVP benefit, so the
in-play stream builds its own wider view here, by direct LibPQ query, and `src/` is left
alone. (WP-H sketches what a later graduation would change.)

What this adds over `ds.bbc_events`: goals with the RUNNING SCORE, red / second-yellow
cards, substitutions and corners — i.e. everything `l01`'s `goals_of` / `reds_of` read
out of `ds.incidents`, plus two new event streams the MVPs need.

Two conventions that are load-bearing:

1. **`is_home_event` is the fetcher's three-way CASE, verbatim, with `-fc` slug
   normalisation.** The naive boolean form (`lt.team = mm.bbc_home_slug`) returns FALSE
   rather than NULL for an unmatched slug, so it dumps every unmatched row on AWAY: that
   hit 15.6% of shot rows in the PM research and REVERSED the measured home shot
   advantage. Do not "simplify" it.

2. **Goal side comes from the RUNNING SCORE, not the slug.** For goals only, the slug is
   actively wrong: BBC attributes an own goal to the OFFENDING player's team, while
   `home_score`/`away_score` increment for the team that BENEFITS — which is the side a
   scoring-intensity model must credit. All 67 slug-less goals in 56/57 are own goals,
   and where both routes resolve they agree on 2,898/2,898. This single change moves goal
   reconciliation against the final score from 93.8% to 99.8% (see Gate A in NOTES.md).
   The `src` fetcher's "do NOT infer the side from the running score" warning is about
   SHOTS, where there is no score to difference; it does not apply to goal rows.

Minute convention: BBC carries a genuine `added_time` (1..8) on stoppage events, where
this SofaScore feed clamps all stoppage goals to exactly mm=45 / mm=90 with added_time=0
(r00 finding). `bbc_minute` therefore returns time + added_time and will sit LATER than
the incident clock in the stoppage tail — that is BBC being more accurate, not a defect,
but any cross-source comparison must be done on `bbc_base_minute` to be apples-to-apples.
=#

using DataFrames, LibPQ, Statistics

# ---------------------------------------------------------------------------
# 1. Event vocabulary
# ---------------------------------------------------------------------------

# Identical to src's BBC_SHOT_EVENTS; re-declared so the prototype does not depend on an
# unexported const and stays pinned if src's list ever moves.
const L04_SHOT_EVENTS   = ["goal", "attempt_missed", "attempt_saved", "attempt_blocked",
                           "post", "penalty_missed", "penalty_saved"]
const L04_RED_EVENTS    = ["red_card", "second_yellow_card"]
const L04_OTHER_EVENTS  = ["substitution", "corner"]
const L04_TIMELINE_EVENTS = vcat(L04_SHOT_EVENTS, L04_RED_EVENTS, L04_OTHER_EVENTS)

bbc_conn() = LibPQ.Connection(ENV["BF_DB_URL"])

# ---------------------------------------------------------------------------
# 2. Fetch
# ---------------------------------------------------------------------------

"""
    fetch_bbc_timeline(conn, t_ids) -> DataFrame

One row per timeline event for the given tournaments, ordered by (match_id, post_index).
Columns: match_id, tournament_id, season_id, post_index, time, added_time, event_type,
is_home_event, player, home_score, away_score, text.

`is_home_event` is missing wherever the BBC team slug matches neither side — 438 `post`
(woodwork) rows carry no team at all, and own-goal rows carry the offender's team.
`resolve_sides!` repairs the goal rows; the rest stay missing and must be dropped.
"""
function fetch_bbc_timeline(conn::LibPQ.Connection, t_ids::Vector{Int})
    ev_in = join(("'" * e * "'" for e in L04_TIMELINE_EVENTS), ", ")
    query = """
    SELECT lt.match_id,
           m.tournament_id,
           m.season_id,
           lt.post_index,
           lt.time,
           lt.added_time,
           lt.event_type,
           -- THREE-WAY, NOT BOOLEAN — see the header note. Copied verbatim from
           -- src/Data/fetchers/sql/bbc_events.jl.
           CASE
             WHEN regexp_replace(lt.team, '-fc\$', '') =
                  regexp_replace(mm.bbc_home_slug, '-fc\$', '') THEN true
             WHEN regexp_replace(lt.team, '-fc\$', '') =
                  regexp_replace(mm.bbc_away_slug, '-fc\$', '') THEN false
             ELSE NULL
           END AS is_home_event,
           lt.player,
           lt.home_score,
           lt.away_score,
           lt.text
    FROM bbc.live_text lt
    JOIN sofascore.matches m ON m.match_id = lt.match_id
    JOIN bbc.match_meta mm   ON mm.match_id = lt.match_id
    WHERE m.tournament_id = ANY(\$1)
      AND lt.event_type IN ($ev_in)
    ORDER BY lt.match_id, lt.post_index
    """
    return DataFrame(LibPQ.execute(conn, query, [t_ids]))
end

# ---------------------------------------------------------------------------
# 3. Minutes + side resolution
# ---------------------------------------------------------------------------

"BBC match minute including stoppage (time + added_time). BBC's added_time is real."
bbc_minute(r)::Float64 = Float64(r.time) + Float64(coalesce(r.added_time, 0))

"BBC base minute, ignoring stoppage — the only clock comparable to `ds.incidents`."
bbc_base_minute(r)::Float64 = Float64(r.time)

"""
    resolve_sides!(tl) -> tl

Adds `:side` (`Union{Missing,Bool}`, true = home) and `:minute` / `:base_minute`.

Goal rows are resolved by differencing the running score within each match (ordered by
post_index): the side whose tally increments by exactly 1 owns the goal. This is exact for
own goals and needs no team slug. Rows where neither tally increments by exactly 1 fall
back to the slug and are reported by `timeline_qa` as a running-score break.

Every other event type keeps the slug-derived `is_home_event`.
"""
function resolve_sides!(tl::DataFrame)
    side = Vector{Union{Missing, Bool}}(tl.is_home_event)
    breaks = 0
    for g in groupby(tl, :match_id)
        ph, pa = 0, 0
        for i in 1:nrow(g)
            g.event_type[i] == "goal" || continue
            h = Int(coalesce(g.home_score[i], -1)); a = Int(coalesce(g.away_score[i], -1))
            row = parentindices(g)[1][i]
            if h == ph + 1 && a == pa
                side[row] = true
            elseif a == pa + 1 && h == ph
                side[row] = false
            else
                breaks += 1   # keep the slug value, whatever it is
            end
            ph, pa = max(h, ph), max(a, pa)
        end
    end
    tl.side = side
    tl.minute = [bbc_minute(r) for r in eachrow(tl)]
    tl.base_minute = [bbc_base_minute(r) for r in eachrow(tl)]
    metadata!(tl, "score_breaks", breaks; style = :note)
    return tl
end

# ---------------------------------------------------------------------------
# 4. Per-match event sequences
# ---------------------------------------------------------------------------

# `t` is the stoppage-inclusive minute (what the NHPP slicer consumes); `tb` is the base
# minute, the only clock comparable with `ds.incidents`. l01's `build_slices` touches only
# `.t` / `.home`, so the extra field is inert there.
const TSeq = Vector{@NamedTuple{t::Float64, tb::Float64, home::Bool}}

_seq(sub::AbstractDataFrame, types)::TSeq = begin
    out = TSeq()
    for r in eachrow(sub)
        (r.event_type in types && !ismissing(r.side)) || continue
        push!(out, (t = r.minute, tb = r.base_minute, home = Bool(r.side)))
    end
    sort!(out, by = x -> x.t)
    out
end

"""
    build_event_seqs(tl) -> Dict{Int, NamedTuple}

Per match: `(goals, reds, subs, shots, corners)`, each a sorted `Vector{(t, home)}` —
exactly the shape `l01`'s `goals_of` / `reds_of` return, so `build_slices` consumes either
source unchanged.

Note `goals ⊂ shots`: a goal is a shot that went in, and the shot-flow NHPP (WP-C) needs
the full attempt count including goals.
"""
function build_event_seqs(tl::DataFrame)
    hasproperty(tl, :side) || error("call resolve_sides!(tl) first")
    out = Dict{Int, @NamedTuple{goals::TSeq, reds::TSeq, subs::TSeq,
                                shots::TSeq, corners::TSeq}}()
    for g in groupby(tl, :match_id)
        out[Int(g.match_id[1])] = (
            goals   = _seq(g, ("goal",)),
            reds    = _seq(g, L04_RED_EVENTS),
            subs    = _seq(g, ("substitution",)),
            shots   = _seq(g, L04_SHOT_EVENTS),
            corners = _seq(g, ("corner",)),
        )
    end
    return out
end

"""
    bbc_goals_of(seqs, mid) / bbc_reds_of(seqs, mid)

Drop-in replacements for `l01`'s `goals_of(ds, mid)` / `reds_of(ds, mid)`, so
`assemble_nhpp_matches` can be pointed at BBC (race arm 0b) with no other change.
"""
bbc_goals_of(seqs, mid)::TSeq = haskey(seqs, mid) ? seqs[mid].goals : TSeq()
bbc_reds_of(seqs, mid)::TSeq  = haskey(seqs, mid) ? seqs[mid].reds  : TSeq()

# ---------------------------------------------------------------------------
# 5. Gate A QA
# ---------------------------------------------------------------------------

"""
    reconcile_goals(seqs, matches) -> DataFrame

Per match: BBC goal counts vs the SofaScore final score. `ok` is the reconciliation gate;
the plan's blocking threshold is ≥ 92% with the failures classified.
"""
function reconcile_goals(seqs, matches::DataFrame)
    rows = DataFrame(match_id = Int[], tournament_id = Int[], season = String[],
                     bh = Int[], ba = Int[], fh = Int[], fa = Int[], ok = Bool[])
    for r in eachrow(matches)
        haskey(seqs, r.match_id) || continue
        (ismissing(r.home_score) || ismissing(r.away_score)) && continue
        g = seqs[r.match_id].goals
        bh = count(x -> x.home, g); ba = count(x -> !x.home, g)
        push!(rows, (Int(r.match_id), Int(r.tournament_id), String(r.season),
                     bh, ba, Int(r.home_score), Int(r.away_score),
                     bh == r.home_score && ba == r.away_score))
    end
    return rows
end

"""
    cross_check_events(seqs, ds, kind; tol = 2.0) -> (agree, minute_stats, per_match)

Match BBC events against `ds.incidents` on matches where both sources exist, greedily
pairing same-side events by nearest minute. `kind` is `:red` or `:sub`.

Compared on the BASE minute (stoppage excluded): BBC carries real added_time where this
SofaScore feed clamps stoppage to 45/90, so a full-minute comparison would score a clock
convention difference as a data disagreement.
"""
function cross_check_events(seqs, ds, kind::Symbol; tol::Float64 = 2.0)
    isred(x) = !ismissing(x) && (occursin("red", lowercase(String(x))))
    inc_of = function (mid)
        if kind === :red
            c = subset(ds.incidents, :match_id => ByRow(==(mid)),
                                     :incident_type => ByRow(==("card")))
            return [(t = Float64(r.time), home = Bool(r.is_home)) for r in eachrow(c)
                    if isred(r.incident_class) && !ismissing(r.is_home)]
        else
            c = subset(ds.incidents, :match_id => ByRow(==(mid)),
                                     :incident_type => ByRow(==("substitution")))
            return [(t = Float64(r.time), home = Bool(r.is_home)) for r in eachrow(c)
                    if !ismissing(r.is_home)]
        end
    end
    inc_mids = Set(unique(ds.incidents.match_id))
    per = DataFrame(match_id = Int[], n_bbc = Int[], n_inc = Int[], n_pair = Int[])
    diffs = Float64[]
    for (mid, s) in seqs
        mid in inc_mids || continue
        b = kind === :red ? s.reds : s.subs
        i = inc_of(mid)
        used = falses(length(i)); npair = 0
        for e in b
            best = 0; bd = Inf
            for (j, x) in enumerate(i)
                (used[j] || x.home != e.home) && continue
                d = abs(x.t - e.tb)
                # NB: `d < bd && (best, bd = j, d)` parses as a tuple expression, not an
                # assignment — it silently paired nothing. Keep the explicit block.
                if d < bd
                    best = j; bd = d
                end
            end
            if best > 0
                used[best] = true; npair += 1
                # `e.tb` is BBC's base minute — the incident clock clamps stoppage to
                # 45/90, so pairing on the stoppage-inclusive minute would score a clock
                # convention difference as a data disagreement.
                push!(diffs, abs(e.tb - i[best].t))
            end
        end
        push!(per, (mid, length(b), length(i), npair))
    end
    agree = (matches = nrow(per), n_bbc = sum(per.n_bbc), n_inc = sum(per.n_inc),
             n_paired = sum(per.n_pair),
             recall_vs_inc = sum(per.n_pair) / max(sum(per.n_inc), 1),
             precision_vs_bbc = sum(per.n_pair) / max(sum(per.n_bbc), 1),
             count_exact = mean(per.n_bbc .== per.n_inc))
    mstat = isempty(diffs) ? (mae = NaN, p90 = NaN, frac_gt_tol = NaN) :
            (mae = mean(diffs), p90 = quantile(diffs, 0.9),
             frac_gt_tol = mean(diffs .> tol))
    return agree, mstat, per
end
