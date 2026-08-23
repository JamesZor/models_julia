# current_development/matchday_2026_08_08/l02_slate_replay.jl
#
# Post-match-day forensics for a WHOLE SLATE, replayed minute by minute into kick-off.
#
# ---------------------------------------------------------------------------------------------
# WHY THIS FILE EXISTS
# ---------------------------------------------------------------------------------------------
#
# `r02_price_tonight.jl` answers "what do I bet in the next hour". It cannot answer any of the
# questions you actually want answered on Sunday morning:
#
#   * did the portfolio get BIGGER or SMALLER as the book firmed up?
#   * would I have done better firing at T-60 or at T-0?
#   * did the market come toward the model or run away from it?
#   * could the sizes on the sheet have been filled at all?
#   * which family paid, and was the answer the same at every entry time?
#
# All five need the same primitive: run the identical pipeline at many `as_of` instants against
# a book that is now history, then grade the lot against a result that is now known.
# `betfair_live.order_book_1m` makes that possible; `as_of` being a call-site argument rather
# than a clock read inside a stage is what makes it HONEST.
#
# ---------------------------------------------------------------------------------------------
# WHAT THE 2026-08-08 SCOTTISH LOWER SLATE ACTUALLY OFFERS (measured, not assumed)
# ---------------------------------------------------------------------------------------------
#
#   fixtures            10  (56 x5, 57 x5), ALL kicking off 13:00 UTC -> ONE settlement window
#   crosswalk           10/10 resolved, is_verified, 11 markets each
#   order book          first pre-KO tick 12:00 UTC, i.e. T-60 ONLY -- not T-24h, not T-6h
#   cadence             3 minutes, despite the table being called order_book_1m -> 21 snapshots
#   in-play             the feed keeps running to ~16:00; `ExplicitFixtures` drops a fixture once
#                       `as_of > kickoff`, which is the guard that keeps in-play ticks out
#   last_price_traded   32% populated (the ARCHITECTURE's "NULL in 100% of rows" is now stale)
#
# So the honest framing is: this measures the FINAL HOUR, at 3-minute resolution, on one slate.
# It is a mechanism, not a result. n = 10 matches.
#
# ---------------------------------------------------------------------------------------------
# THE ONE STRUCTURAL FACT THAT SHAPES EVERY READING
# ---------------------------------------------------------------------------------------------
#
# `DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel` needs no lineups and no ratings. Its only
# injectable feature is `:league_lookup`, materialised from `tournament_id`, which does not
# depend on the clock. Therefore **the latents are identical at every snapshot** and 100% of the
# movement in the trace below is the BOOK moving, plus Kelly's response to it.
#
# That is a gift for interpretation and a trap for anyone who copies this file to a player-level
# engine, where `RatingsFromTracker` does move with the lineup. `latents_invariant` asserts it
# rather than assuming it.
#
# ---------------------------------------------------------------------------------------------
# DEFECT FOUND WHILE WRITING THIS (see `venue_leg`)
# ---------------------------------------------------------------------------------------------
#
# `MatchDay.order_ticket` mis-names every synthetic leg. `Instrument.key` is the position you
# WANT; for a synthetic the venue action is on its COMPLEMENT. The ticket emits
# `(selection = position, side = :lay, price = complement's lay price)`, so executing it places
# the opposite position at a price that belongs to the other selection. 14 of the 48 legs on
# this slate's closing sheet are synthetics. `venue_leg` below is the corrected ticket, and the
# src fix is to give `Instrument` a `venue_key` field.

using DataFrames, Dates, Statistics, Printf

# Aliases are local to each function rather than top-level `const`s: this file gets re-included
# into warm REPL sessions where `const MD = ...` collides with whatever the last runner bound.
_md() = BayesianFootball.MatchDay
_pf() = BayesianFootball.Portfolio
_dd() = BayesianFootball.Data

# ===================================================================
# 1. The slate, and its result
# ===================================================================

"""
    slate_from_db(tournament_ids, day) -> (fixtures, results)

Every fixture in `tournament_ids` kicking off on `day` (UTC), as `MatchDay.Fixture` objects,
plus a `match_id -> (home_score, away_score)` map for the ones that have finished.

`ExplicitFixtures`, not `SofaScoreEvents`: the live source filters on `status_type =
'notstarted'`, so the moment a match kicks off it becomes invisible to it. A replay of a played
day therefore CANNOT use the live fixture source, and reaching for it is the first thing that
goes wrong when someone adapts `r02` into a post-mortem.

Window is expressed in epoch seconds because `sofascore.events.start_timestamp` is an integer.
"""
function slate_from_db(tournament_ids::Vector{Int}, day::Date)
    MD = _md()
    lo = Int(round(datetime2unix(DateTime(day))))
    hi = Int(round(datetime2unix(DateTime(day) + Day(1))))

    df = MD._query("""
        SELECT e.match_id, e.tournament_id, e.home_team, e.away_team, e.start_timestamp,
               e.status_type, m.home_score, m.away_score
        FROM sofascore.events e
        LEFT JOIN sofascore.matches m USING (match_id)
        WHERE e.tournament_id = ANY(\$1)
          AND e.start_timestamp >= \$2 AND e.start_timestamp < \$3
        ORDER BY e.start_timestamp, e.match_id;
        """, (tournament_ids, lo, hi))

    isempty(df) && error("slate_from_db: no fixtures for $tournament_ids on $day")

    fx = MD.Fixture[MD.Fixture(Int(r.match_id), String(r.home_team), String(r.away_team),
                               unix2datetime(r.start_timestamp), Int(r.tournament_id))
                    for r in eachrow(df)]

    results = Dict{Int,Tuple{Int,Int}}()
    for r in eachrow(df)
        (ismissing(r.home_score) || ismissing(r.away_score)) && continue
        results[Int(r.match_id)] = (Int(r.home_score), Int(r.away_score))
    end

    n_unfinished = count(s -> coalesce(s, "") != "finished", df.status_type)
    n_unfinished == 0 || @warn "slate_from_db: $n_unfinished fixture(s) not finished — they " *
                               "will be priced but cannot be graded"
    return fx, results
end

"""
    book_coverage(fixtures) -> DataFrame

How deep the replayable window actually is, per fixture: first and last pre-kick-off tick, how
many distinct snapshot instants exist, and how much was matched by the close.

**Run this before choosing a snapshot grid.** A grid that reaches back further than the feed
does produces snapshots with no book, which the gate reports as `no quotes retrieved` — correct
behaviour that reads like a broken pipeline if you were expecting prices.
"""
function book_coverage(fixtures)
    MD = _md()
    rows = NamedTuple[]
    for f in fixtures
        id = MD.resolve(MD.MatchMetaCrosswalk(), f)
        if !(id isa MD.Resolved)
            push!(rows, (match_id = f.m_id, fixture = "$(f.home) v $(f.away)",
                         event = "-", n_mkts = 0, first_tick = nothing, last_tick = nothing,
                         n_snaps = 0, matched_close = NaN))
            continue
        end
        ids = collect(values(id.market_ids))
        df = MD._query("""
            SELECT min(ts) AS first_ts, max(ts) AS last_ts,
                   count(DISTINCT ts) AS n_snaps, max(market_matched) AS mm
            FROM betfair_live.order_book_1m
            WHERE market_id = ANY(\$1) AND ts <= \$2;
            """, (ids, f.kickoff))
        r = first(df)
        push!(rows, (match_id = f.m_id, fixture = "$(f.home) v $(f.away)",
                     event = id.bf_event_id, n_mkts = length(ids),
                     first_tick = ismissing(r.first_ts) ? nothing : DateTime(r.first_ts),
                     last_tick  = ismissing(r.last_ts)  ? nothing : DateTime(r.last_ts),
                     n_snaps = ismissing(r.n_snaps) ? 0 : Int(r.n_snaps),
                     matched_close = ismissing(r.mm) ? NaN : Float64(r.mm) / 10_000))
    end
    return DataFrame(rows)
end

"""
    snapshot_grid(fixtures; lookback = Minute(60), step = Minute(3)) -> Vector{DateTime}

Decision instants, from `earliest_kickoff - lookback` up to and including `earliest_kickoff`.

Anchored on the EARLIEST kick-off so that no snapshot sits after a fixture has started —
`ExplicitFixtures` silently drops a started fixture (`kickoff >= as_of`), which would otherwise
shrink the slate mid-trace and make the exposure series incomparable across snapshots.

`step` should match the feed's real cadence. Despite the table name, `order_book_1m` carries
this slate at **3-minute** resolution; a 1-minute grid just repeats the same tick three times
and triples the run time for no extra information.
"""
function snapshot_grid(fixtures; lookback::Period = Minute(60), step::Period = Minute(3))
    ko = minimum(f.kickoff for f in fixtures)
    return collect(ko - lookback : step : ko)
end

# ===================================================================
# 2. The spec
# ===================================================================

"""
    replay_spec(fixtures; rounding = NoMinimum(), min_matched = nothing, book_age = Minute(10))

The `MatchDaySpec` for a played day.

Three deliberate differences from `r02`'s live spec:

* `ExplicitFixtures` — see `slate_from_db`.
* `SourceChain()` (empty) — the funnel engine reads no lineup, and an empty chain returns
  `nothing` for every fixture, which is the honest statement. Handing it `ProvisionalDB()` would
  fetch an XI that nothing downstream consumes and put a `MaxLineupAge` reason in the blocked
  report that has no bearing on the price.
* `MatchMetaCrosswalk` alone, no `LiveNameMatch` fallback — the crosswalk answers 10/10 here, and
  a fallback that never fires is a fallback you cannot audit. If it ever fires on a REPLAY it
  means the crosswalk was backfilled after the fact, which is a different (and worse) thing than
  it firing live.

`min_matched` is opt-in and blocking when supplied, because on a replay the interesting question
is what a liquidity gate WOULD have refused, not what it silently permitted.
"""
function replay_spec(fixtures; rounding = nothing, min_matched = nothing,
                     book_age::Period = Minute(10))
    MD = _md()
    gates = MD.AbstractReadinessGate[MD.IdentityResolved(), MD.MaxBookAge(book_age)]
    min_matched === nothing || push!(gates, MD.MinMatched(Float64(min_matched), true))

    return MD.MatchDaySpec(
        fixtures = MD.ExplicitFixtures(fixtures),
        identity = MD.MatchMetaCrosswalk(),
        lineups  = MD.SourceChain(),
        rounding = rounding === nothing ? MD.NoMinimum() : rounding,
        gate     = MD.GateChain(gates...))
end

"""
    latents_invariant(spec, expr, ds, fixtures, t1, t2) -> (ok, max_abs_diff, worst_col)

Does the model's view of these fixtures depend on `as_of`?

For the funnel engine it must not: no lineup, no ratings, only `:league_lookup`, which is a
function of `tournament_id`. If this ever returns `ok = false` on this engine, something is
reading the clock that should not be, and the whole "all movement is market movement" reading
of the trace below collapses.

Copy this call into any adaptation to a player-level engine, where the expected answer flips to
`false` — there `RatingsFromTracker` genuinely does move with the announced XI, and the trace
then confounds model drift with market drift unless the two are separated.
"""
function latents_invariant(spec, expr, ds, fixtures, t1::DateTime, t2::DateTime)
    MD = _md()
    l1, _ = MD.matchday_latents(spec, expr, ds, MD.build_cards(spec, nothing, t1),
                                DataFrame(), t1)
    l2, _ = MD.matchday_latents(spec, expr, ds, MD.build_cards(spec, nothing, t2),
                                DataFrame(), t2)
    (isempty(l1) || isempty(l2)) && return (false, NaN, :empty)

    a = sort(l1, :match_id); b = sort(l2, :match_id)
    a.match_id == b.match_id || return (false, NaN, :match_id)

    worst, wcol = 0.0, :none
    for c in names(a)
        v1, v2 = a[!, c], b[!, c]
        eltype(v1) <: Number || continue
        d = maximum(abs.(Float64.(v1) .- Float64.(v2)))
        d > worst && (worst = d; wcol = Symbol(c))
    end
    return (worst < 1e-10, worst, wcol)
end

# ===================================================================
# 3. The replay
# ===================================================================

"""
    replay(spec, sys, segment, expr, ds, snaps; bankroll, results, capture_depth = true)

Run `MatchDay.match_day` once per snapshot and keep everything each run produced.

Deliberately calls the real `match_day` rather than a cheaper hand-rolled loop with the latents
hoisted out. The whole value of a post-mortem is that it exercises the SAME code path that will
run live; an optimised replica that drifts from it measures the replica. At ~9s per snapshot,
21 snapshots is ~3 minutes, which is not worth a fork of the pipeline.

Returns a NamedTuple:

    snaps    the grid, as given
    legs     every staked leg at every snapshot, graded            <- the main table
    quotes   every QUOTED selection at every snapshot (staked or not)
    depth    top-of-book prices and sizes per selection per snapshot
    blocked  every gate refusal, per snapshot
    slate    one row per snapshot: exposure, k_risk, P/L, ROI
    close    the inputs at the final snapshot, for policy sweeps
"""
function replay(spec, sys, segment, expr, ds, snaps::Vector{DateTime};
                bankroll::Real = 1000.0, results::Dict{Int,Tuple{Int,Int}} = Dict{Int,Tuple{Int,Int}}(),
                capture_depth::Bool = true)
    MD, PF = _md(), _pf()
    ko = Dict(f.m_id => f.kickoff for f in spec.fixtures.list)

    legs, quotes, depth, blocked = DataFrame[], DataFrame[], DataFrame[], DataFrame[]
    close_inputs = nothing

    for (i, t) in enumerate(snaps)
        res = MD.match_day(spec, sys, segment, expr, ds; as_of = t, bankroll = bankroll)

        if !isempty(res.sheet)
            s = copy(res.sheet)
            s.as_of = fill(t, nrow(s))
            s.mins_to_ko = [Dates.value(ko[m] - t) ÷ 60_000 for m in s.match_id]
            grade!(s, results, sys)
            push!(legs, s)
        end

        if !isempty(res.odds)
            o = copy(res.odds)
            o.as_of = fill(t, nrow(o))
            push!(quotes, o)
        end

        br = MD.blocked_report(res)
        isempty(br) || push!(blocked, insertcols!(br, 1, :as_of => t))

        capture_depth && push!(depth, _depth_snapshot(spec, res, t))

        # The last snapshot's inputs are kept so a policy sweep can re-stake the same book
        # without re-running inference 24 more times.
        if i == length(snaps)
            passed = [c for c in res.cards if MD.is_ready(c.readiness)]
            lat, _ = MD.matchday_latents(spec, expr, ds, passed, res.odds, t)
            close_inputs = (as_of = t, latents = lat, odds = res.odds,
                            fixtures = MD.fixture_info(passed), instruments = res.instruments)
        end
    end

    legs_df   = isempty(legs)    ? DataFrame() : reduce(vcat, legs)
    quotes_df = isempty(quotes)  ? DataFrame() : reduce(vcat, quotes)
    depth_df  = isempty(depth)   ? DataFrame() : reduce(vcat, depth)
    blk_df    = isempty(blocked) ? DataFrame() : reduce(vcat, blocked)

    return (snaps = snaps, legs = legs_df, quotes = quotes_df, depth = depth_df,
            blocked = blk_df, bankroll = Float64(bankroll), kickoffs = ko,
            slate = slate_trace(legs_df, Float64(bankroll)), close = close_inputs)
end

"Top-of-book prices AND sizes for every quoted selection — the input to `fill_report`."
function _depth_snapshot(spec, res, t::DateTime)
    MD = _md()
    rows = NamedTuple[]
    for c in res.cards
        MD.resolved(c) || continue
        book = MD.quotes(spec.book, c.identity, t)
        for (k, b) in book
            push!(rows, (as_of = t, match_id = c.fixture.m_id, group = k.group, line = k.line,
                         selection = k.selection,
                         back = MD.best_back(b), back_size = isempty(b.back_size) ? 0.0 : b.back_size[1],
                         lay  = MD.best_lay(b),  lay_size  = isempty(b.lay_size)  ? 0.0 : b.lay_size[1],
                         matched = b.matched, tick = b.ts))
        end
    end
    return isempty(rows) ? DataFrame(as_of = DateTime[], match_id = Int[], group = String[],
                                     line = Float64[], selection = Symbol[], back = Float64[],
                                     back_size = Float64[], lay = Float64[], lay_size = Float64[],
                                     matched = Float64[], tick = DateTime[]) : DataFrame(rows)
end

# ===================================================================
# 4. Grading
# ===================================================================

"""
    grade!(sheet, results, sys) -> sheet

Attach `:graded`, `:unit_payoff` and `:pnl`.

Two things here are easy to get wrong and both were wrong in earlier hand-rolled versions:

1. **Stake `:risk`, not `:stake`.** `stake` is what Portfolio wanted; `risk` is what survived
   `AbstractStakeRounding`. With `NoMinimum` they coincide, so a bug here is invisible until the
   day someone switches to `FloorOrDrop` and the reported P/L silently belongs to a book that
   was never placeable.
2. **Grade at `:odds`, not `:venue_odds`.** `odds` are the EFFECTIVE odds — the whole point of
   the morphism is that a lay expressed in risk units settles exactly like a back at `d/(d-1)`,
   so one formula covers both sides. `venue_odds` is the number you type into the exchange and
   settles nothing.

A `missing` grade is a PUSH: the stake comes back, so the unit payoff is 0.0, not −1.0.
"""
function grade!(sheet::DataFrame, results::Dict{Int,Tuple{Int,Int}}, sys)
    DD, PF = _dd(), _pf()
    comm = sys.book.exec.commission

    g = Vector{Union{Missing,Bool}}(undef, nrow(sheet))
    for (i, r) in enumerate(eachrow(sheet))
        sc = get(results, r.match_id, nothing)
        g[i] = sc === nothing ? missing :
               DD.grade_selection(r.group, r.line, r.selection, sc[1], sc[2])
    end
    sheet.graded = g
    sheet.unit_payoff = [ismissing(x) ? 0.0 : (x ? PF.net_return(comm, r.odds) : -1.0)
                         for (x, r) in zip(g, eachrow(sheet))]
    sheet.pnl = sheet.risk .* sheet.unit_payoff
    return sheet
end

"""
    venue_leg(row) -> (selection, side, price, stake, liability)

The venue leg, derived from the sheet row ALONE.

This was written when `MatchDay.order_ticket` mis-named every synthetic — `Instrument` had no
field for the runner the order touches, so the ticket emitted the position's selection beside
the complement's side and price. That is fixed in `src`: `Instrument` now carries `venue_key`,
the sheet carries `venue_selection`, and `order_ticket` reads it.

This function is kept as an INDEPENDENT CHECK, not as a workaround. It reconstructs the venue
runner structurally, from `(group, line, selection)`, without consulting the `Instrument` that
produced the row — so `venue_leg(row).selection == order_ticket(row).selection` is a real
cross-validation of the src path rather than a restatement of it. `r03` asserts exactly that.

It also re-tickets a saved CSV from a run that predates the fix, which is the other reason to
keep it.

1X2 has three outcomes and therefore no complement, so a 1X2 row is always a direct back and
this function is the identity on it.
"""
function venue_leg(row)
    row.side === :back && return (selection = row.selection, side = :back,
                                  price = row.venue_odds, stake = row.venue_stake,
                                  liability = row.venue_stake)
    comp = _complement_selection(row.group, row.line, row.selection)
    comp === nothing && error("venue_leg: :lay row on $(row.group) $(row.line) " *
                              "$(row.selection), which has no complement — this should be " *
                              "unreachable, `instrument` cannot build a synthetic without one")
    return (selection = comp, side = :lay, price = row.venue_odds,
            stake = row.venue_stake, liability = row.risk)
end

"Structural complement inside a two-outcome group. Mirrors `MatchDay.complement_of` without
needing the book, so a saved CSV can be re-ticketed after the fact."
function _complement_selection(group::AbstractString, line::Real, sel::Symbol)
    if group == "BTTS"
        sel === :btts_yes && return :btts_no
        sel === :btts_no  && return :btts_yes
    elseif group == "OverUnder"
        suffix = replace(@sprintf("%.1f", line), "." => "")
        startswith(String(sel), "over_")  && return Symbol("under_", suffix)
        startswith(String(sel), "under_") && return Symbol("over_",  suffix)
    end
    return nothing
end

# ===================================================================
# 5. The five views
# ===================================================================

"""
    slate_trace(legs, bankroll, kickoffs) -> DataFrame

**View 1 — how the portfolio adapts as t -> kick-off.** One row per decision instant.

`exposure` is `sum(risk)/bankroll`, computed here rather than taken from `slate_exposure`,
because the sheet's own column is pre-rounding: with `FloorOrDrop` the two diverge and the one
that matters is the one you could actually have had on.

`capped` false with a low `k_risk` means the DRAWDOWN budget is binding, not the hard cap — a
different lever, and the one to move if you want more or less on. See ARCHITECTURE §9.1: trust
and shrinkage cannot resize the book once `risk_factor` binds, they only reshape it.
"""
function slate_trace(legs::DataFrame, bankroll::Float64)
    isempty(legs) && return DataFrame()
    rows = NamedTuple[]
    for g in groupby(legs, :as_of)
        t = first(g.as_of)
        staked = sum(g.risk)
        pnl = "pnl" in names(g) ? sum(skipmissing(g.pnl)) : NaN
        push!(rows, (as_of = t,
                     mins_to_ko = minimum(g.mins_to_ko),
                     fixtures = length(unique(g.match_id)),
                     legs = nrow(g),
                     lays = count(==(:lay), g.side),
                     exposure = staked / bankroll,
                     k_risk = first(g.k_risk),
                     capped = first(g.capped),
                     mean_edge = mean(g.edge),
                     risk_wtd_edge = sum(g.edge .* g.risk) / staked,
                     staked = staked,
                     pnl = pnl,
                     roi = pnl / staked))
    end
    return sort!(DataFrame(rows), :as_of)
end

"""
    family_trace(legs) -> DataFrame

**View 1b.** The same trace split by market family. This is where the curation question is
answered empirically: if 1X2 carries most of the risk and all of the loss at every entry time,
that is the per-line trust result reproducing itself on live data rather than in simulation.
"""
function family_trace(legs::DataFrame)
    isempty(legs) && return DataFrame()
    g = combine(groupby(legs, [:as_of, :group]),
                nrow => :legs, :risk => sum => :risk, :pnl => sum => :pnl)
    g.roi = g.pnl ./ g.risk
    return sort!(g, [:as_of, :group])
end

"""
    churn(legs) -> DataFrame

**View 2 — would re-pricing every 3 minutes have whipsawed you?**

`jaccard` is set overlap of the staked legs between consecutive snapshots; `risk_turnover` is
`sum(|Δrisk|) / sum(risk)`, i.e. how much of the book you would have had to trade to move from
one snapshot's allocation to the next.

A high turnover with a flat P/L curve means the allocation is noise-sensitive and the re-pricing
cadence is costing you spread for nothing. A low turnover says one shot at any time in the window
would have given you substantially the same book — which is the operationally useful answer,
because you cannot actually stand at the terminal for an hour.
"""
function churn(legs::DataFrame)
    isempty(legs) && return DataFrame()
    key(r) = (r.match_id, r.group, r.line, r.selection)
    snaps = sort(unique(legs.as_of))
    rows = NamedTuple[]
    prev = nothing
    for t in snaps
        g = legs[legs.as_of .== t, :]
        cur = Dict(key(r) => r.risk for r in eachrow(g))
        if prev !== nothing
            ka, kb = Set(keys(prev)), Set(keys(cur))
            uni = length(union(ka, kb))
            turnover = sum(abs(get(cur, k, 0.0) - get(prev, k, 0.0)) for k in union(ka, kb))
            push!(rows, (as_of = t, legs = length(cur),
                         entered = length(setdiff(kb, ka)), exited = length(setdiff(ka, kb)),
                         jaccard = uni == 0 ? NaN : length(intersect(ka, kb)) / uni,
                         risk_turnover = turnover / max(sum(values(cur)), eps())))
        end
        prev = cur
    end
    return DataFrame(rows)
end

"""
    clv_vs_close(out) -> (per_leg, by_family)

**View 3 — did the market come toward us?**

For every leg staked at snapshot `t`, the price of the SAME selection in the final (closing)
book. Convention: a price that SHORTENS after we took it means the market moved our way, so
`move_pct < 0` is `+CLV`.

Both prices are EFFECTIVE odds, so the comparison is valid even when `BestOfBackLay` flips a leg
from a direct back to a synthetic between the two instants — which it does, and which would make
a venue-price comparison meaningless.

The reference is the closing **quote table**, not the closing sheet: a leg we stopped staking
still has a price, and dropping it would condition the CLV sample on our own later decisions.

CLV is the measurement that matters at n = 10 matches. P/L over one slate cannot separate a real
edge from luck — the Portfolio backtest's ROI interval includes zero over 628 matches.
"""
function clv_vs_close(out)
    isempty(out.legs) && return (DataFrame(), DataFrame())
    t_close = maximum(out.quotes.as_of)
    close_book = out.quotes[out.quotes.as_of .== t_close, :]
    ref = Dict((r.match_id, r.market_name, r.market_line, r.selection) => r.odds_close
               for r in eachrow(close_book))

    rows = NamedTuple[]
    for r in eachrow(out.legs)
        r.as_of == t_close && continue
        oc = get(ref, (r.match_id, r.group, r.line, r.selection), nothing)
        oc === nothing && continue
        push!(rows, (as_of = r.as_of, mins_to_ko = r.mins_to_ko, match_id = r.match_id,
                     group = r.group, line = r.line, selection = r.selection,
                     side = r.side, risk = r.risk, odds_taken = r.odds, odds_close = oc,
                     move_pct = 100 * (oc / r.odds - 1), toward_us = oc < r.odds,
                     pnl = r.pnl))
    end
    isempty(rows) && return (DataFrame(), DataFrame())
    per_leg = DataFrame(rows)

    by_family = combine(groupby(per_leg, :group),
                        nrow => :legs,
                        :toward_us => mean => :pct_toward_us,
                        :move_pct => median => :median_move,
                        [:move_pct, :risk] => ((m, w) -> sum(m .* w) / sum(w)) => :risk_wtd_move)
    return per_leg, by_family
end

"""
    fill_report(out) -> (per_leg, by_snapshot)

**View 4 — could the sheet have been filled?**

`BestAvailable` reads the best PRICE and nothing else; `max_leverage` rejects implausible
synthetics on price alone, which the ARCHITECTURE argues makes skipping the depth check safe.
That argument is about not taking a fake price. It says nothing about SIZE, and size is the
binding constraint on Scottish League One and Two: the closing book on this slate matched
~£34 on a median BTTS market and ~£44 on O/U 2.5, against ~£844 on MATCH_ODDS.

For a back leg the constraint is `back_size` at the top of book; for a synthetic it is the
COMPLEMENT's `lay_size`, because that is the runner the order actually touches.

`fill_ratio` is `min(1, available / venue_stake)`. `capacity` scales it to the whole book: the
bankroll multiple at which the marginal leg stops being fillable at the top of book.
"""
function fill_report(out)
    (isempty(out.legs) || isempty(out.depth)) && return (DataFrame(), DataFrame())

    dep = Dict((r.as_of, r.match_id, r.group, r.line, r.selection) => r for r in eachrow(out.depth))

    rows = NamedTuple[]
    for r in eachrow(out.legs)
        if r.side === :back
            d = get(dep, (r.as_of, r.match_id, r.group, r.line, r.selection), nothing)
            avail = d === nothing ? NaN : d.back_size
            venue_sel = r.selection
        else
            comp = _complement_selection(r.group, r.line, r.selection)
            d = comp === nothing ? nothing :
                get(dep, (r.as_of, r.match_id, r.group, r.line, comp), nothing)
            avail = d === nothing ? NaN : d.lay_size
            venue_sel = comp
        end
        push!(rows, (as_of = r.as_of, mins_to_ko = r.mins_to_ko, match_id = r.match_id,
                     group = r.group, line = r.line, selection = r.selection,
                     venue_selection = venue_sel, side = r.side,
                     venue_stake = r.venue_stake, available = avail,
                     fill_ratio = isnan(avail) ? NaN : min(1.0, avail / max(r.venue_stake, eps())),
                     risk = r.risk))
    end
    per_leg = DataFrame(rows)

    _meanfin(x) = (v = filter(!isnan, x); isempty(v) ? NaN : mean(v))
    by_snapshot = combine(groupby(per_leg, :as_of),
                          nrow => :legs,
                          :fill_ratio => _meanfin => :mean_fill,
                          :fill_ratio => (x -> count(y -> !isnan(y) && y >= 0.999, x)) => :fully_fillable,
                          [:fill_ratio, :risk] =>
                              ((f, w) -> sum(ifelse.(isnan.(f), 0.0, f) .* w) / sum(w)) => :risk_wtd_fill)
    return per_leg, sort!(by_snapshot, :as_of)
end

"""
    divergence_vs_experience(out, ds, fixtures, as_of) -> DataFrame

**View 5 — is the biggest "edge" just the model's ignorance?**

Per fixture: the size of the model's disagreement with the market, next to how many matches each
team actually has inside the training window.

The motivating case is on this very slate. Ross County were relegated 55 -> 56 and Airdrieonians
with them, so inside a `ScottishLower [56, 57]` DataStore they have exactly ONE match of history
at the time of pricing. The model therefore prices them near the pooled prior — a mid-table
League One side — while the market prices a relegated Championship club. The result is the
second-largest position on the closing sheet: away @ 15.0 with `p_model` 0.265 against
`p_market` 0.066, a 4x disagreement built on one match of evidence.

That is not an edge, it is a cold start, and it is systematically WORST in the opening weeks of
a season and for exactly the promoted and relegated teams that pooling was introduced to handle.
A ranking of `max_edge` against `min_team_matches` is the cheapest available detector.
"""
function divergence_vs_experience(out, ds, fixtures, as_of::DateTime)
    isempty(out.legs) && return DataFrame()
    close_legs = out.legs[out.legs.as_of .== maximum(out.legs.as_of), :]

    m = ds.matches
    played = Dict{String,Int}()
    for i in 1:nrow(m)
        Date(m.match_date[i]) < Date(as_of) || continue
        for team in (m.home_team[i], m.away_team[i])
            played[team] = get(played, team, 0) + 1
        end
    end

    fx = Dict(f.m_id => f for f in fixtures)
    rows = NamedTuple[]
    for g in groupby(close_legs, :match_id)
        f = fx[first(g.match_id)]
        push!(rows, (match_id = f.m_id, fixture = "$(f.home) v $(f.away)",
                     tournament = f.tournament_id,
                     home_matches = get(played, f.home, 0),
                     away_matches = get(played, f.away, 0),
                     min_team_matches = min(get(played, f.home, 0), get(played, f.away, 0)),
                     legs = nrow(g), risk = sum(g.risk),
                     max_abs_edge = maximum(abs.(g.edge)),
                     pnl = sum(g.pnl)))
    end
    return sort!(DataFrame(rows), :max_abs_edge, rev = true)
end

# ===================================================================
# 6. Policy sweep at the close
# ===================================================================

"""
    policy_sweep(sys, close, results, grid) -> DataFrame

**View 6 — was the policy the problem, or the model?**

Re-stakes the CLOSING book under a list of `PolicySpec`s and grades each. Cheap because trust,
risk, cap and filter all sit BELOW the `MatchBook` cache line (ARCHITECTURE §3): they are pure
multipliers on a book that has already been built, so a sweep costs milliseconds while the book
itself cost ~9 seconds.

`grid` is a `Vector{Pair{String,PolicySpec}}` — label first so the output table reads.

Growth, not ROI, is the number to compare. ROI is blind to flat trust (a uniform scaling cancels
in `P/L / stake`), so a sweep judged on ROI will report that half the cells are identical when
their bankroll outcomes differ by a factor of three.
"""
function policy_sweep(sys, expr, close, results::Dict{Int,Tuple{Int,Int}},
                      grid::Vector{<:Pair}; bankroll::Real = 1000.0, instruments = nothing)
    PF, MD = _pf(), _md()
    rows = NamedTuple[]
    for (label, policy) in grid
        s = PF.stake_sheet(PF.PortfolioSystem(sys.book, policy), close.latents, expr,
                           close.odds, close.fixtures; bankroll = bankroll)
        if isempty(s)
            push!(rows, (policy = label, legs = 0, staked = 0.0, exposure = 0.0,
                         pnl = 0.0, roi = NaN, growth = 0.0, k_risk = NaN, capped = false))
            continue
        end
        instruments === nothing || MD._attach_instruments!(s, instruments, MD.NoMinimum())
        "risk" in names(s) || (s.risk = s.stake)
        grade!(s, results, sys)
        staked, pnl = sum(s.risk), sum(s.pnl)
        push!(rows, (policy = label, legs = nrow(s), staked = staked,
                     exposure = staked / bankroll, pnl = pnl, roi = pnl / staked,
                     growth = log(1 + pnl / bankroll), k_risk = first(s.k_risk),
                     capped = first(s.capped)))
    end
    return DataFrame(rows)
end

"""
    family_trust(; w_1x2 = 0.0, w_totals = 0.5, w_btts = 0.5)

The curation result expressed as a TRUST model rather than as a filter.

`MarketWhitelist` is applied last in `stake_slate`, after `apply_cap`, so it can only zero a
stake — the legs that survive keep sizes that were solved for a portfolio still containing the
ones removed, and the freed exposure is left on the table. `SelectionTrust` multiplies
`a_kelly` *before* `risk_factor`, so the drawdown budget re-solves against the curated book and
re-expands what is left.

Same intent, different arithmetic, and the difference is exactly the capacity the whitelist
throws away. `strict = false` because the table below is keyed by family shape rather than
enumerated, and a `KeyError` on an unlisted line is not the failure mode worth having here.
"""
function family_trust(; w_1x2::Float64 = 0.0, w_totals::Float64 = 0.5, w_btts::Float64 = 0.5)
    PF = _pf()
    t = Dict{Tuple{String,Float64,Symbol},Float64}()
    for s in (:home, :draw, :away)
        t[("1X2", 0.0, s)] = w_1x2
    end
    t[("BTTS", 0.0, :btts_yes)] = w_btts
    t[("BTTS", 0.0, :btts_no)]  = w_btts
    for i in 0:5
        line = i + 0.5
        sfx  = replace(string(line), "." => "")
        t[("OverUnder", line, Symbol("over_",  sfx))] = w_totals
        t[("OverUnder", line, Symbol("under_", sfx))] = w_totals
    end
    return PF.SelectionTrust(t; default = w_totals, strict = false)
end

"Convenience: the totals+BTTS whitelist the curation work settled on."
function totals_btts_whitelist()
    PF = _pf()
    keys_ = vcat([("BTTS", 0.0, :btts_yes), ("BTTS", 0.0, :btts_no)],
                 [("OverUnder", i + 0.5, Symbol("over_",  replace(string(i + 0.5), "." => ""))) for i in 0:4],
                 [("OverUnder", i + 0.5, Symbol("under_", replace(string(i + 0.5), "." => ""))) for i in 0:4])
    return PF.MarketWhitelist(Set{Tuple{String,Float64,Symbol}}(keys_))
end
