# current_development/orderbook_layer2/l07_route2.jl
#
# WP9 (Route 2). The Layer-2 studies, moved onto the sample where the model actually has an edge.
#
# ---------------------------------------------------------------------------------------------
# WHY THIS FILE EXISTS
# ---------------------------------------------------------------------------------------------
#
# WP4-WP8 were run on the 81-match order-book corpus and concluded, among other things, that the
# engine was uninformative and that per-market trust had nothing stable to stand on. Measured
# with r21's own metric against r21's own benchmark, on the SAME experiment this stream trained:
#
#     set                       obs    matches   model LL   market LL      diff
#     ALL OOS                  5859        293    0.42454     0.44231   -0.01778   model wins
#     2025, outside OB window  3580        180    0.41880     0.43360   -0.01479   model wins
#     2026, outside OB window  1789         92    0.44402     0.47411   -0.03009   model wins
#     2026, INSIDE OB window    490         21    0.39528     0.38990   +0.00538   model loses
#
# The order-book window is the ONLY subset where the model loses, it is 21 matches in the
# overlap, and the market's own log loss there is 0.390 against 0.474 on the rest of the same
# season — an unusually sharp market on unusually predictable fixtures. The Layer-2 conclusions
# drawn on it were therefore measured on a sample with no edge to allocate, which makes them
# uninformative about Layer 2 rather than damning of Layer 1.
#
# So: same questions, same estimators, 293 matches instead of 76.
#
# ---------------------------------------------------------------------------------------------
# WHAT ROUTE 2 TRADES AWAY
# ---------------------------------------------------------------------------------------------
#
# `summarize_betfair_market` gives two prices per selection — an open and a close — rather than a
# 3-minute tick history, and no ladder. That costs:
#
#   * the fine entry-time axis   -> only a coarse OPEN vs CLOSE contrast survives
#   * FillCost / depth           -> gone entirely; there is no size question on this path
#   * executable venue prices    -> these are summary closes, not the best back price a stake
#                                   would actually have hit
#
# What it buys is the only thing that matters right now: a sample in which the model has a
# measured edge, so that a staking layer has something to allocate. Findings here are about
# ALLOCATION; the order-book stream remains the authority on EXECUTION.
#
# ---------------------------------------------------------------------------------------------
# THE PIPELINE IS THE REPO'S, NOT A REPLICA
# ---------------------------------------------------------------------------------------------
#
#     extract_oos_predictions -> build_books -> group -> simulate -> report
#
# exactly as `portfolio_runbook/r01_quickstart.jl` runs it. No staking, Kelly, shrinkage or
# settlement maths is reimplemented here. In particular the trust race now varies a real
# `PolicySpec` and re-runs `simulate`, rather than rescaling a staked ledger as WP5 had to on the
# thin corpus — so the drawdown solve and the cap re-bind properly instead of being approximated.

using DataFrames, Dates, Statistics, Printf

# ===================================================================
# 1. Books, once
# ===================================================================

"""
    route2_setup(ds, expr; price = :close) -> (odds, latents, books)

The expensive half: OOS latents, the Betfair open/close summary, and one `MatchBook` per match.

`price = :open` swaps `odds_open` into the `odds_close` column before building. That is the whole
open-vs-close contrast: `extract_selections` reads `odds_close` and nothing else, so re-pointing
that one column prices the identical book at the earlier instant with no other change.

Windows are r21's: open `(-100000, -10)` minutes, close `(-20, 0)`. Keeping them identical is
what makes this comparable to the log-loss table above.
"""
function route2_setup(ds, expr; price::Symbol = :close)
    DD, PF, EE = _dd(), _pf(), BayesianFootball.Experiments

    odds = DD.summarize_betfair_market(ds; open_window = (-100000.0, -10.0),
                                           close_window = (-20.0, 0.0))
    if price === :open
        odds = copy(odds)
        odds.odds_close = odds.odds_open
        odds.prob_fair_close = odds.prob_fair_open
    end

    ds1 = DD.DataStore(ds.segment, ds.matches, ds.statistics, odds,
                       ds.lineups, ds.incidents, ds.betfair_odds)
    latents = EE.extract_oos_predictions(ds1, expr)
    books   = PF.build_books(reference_spec(), latents, expr, odds, ds1)
    return (odds = odds, latents = latents, books = books, ds1 = ds1)
end

"""
    reference_spec() -> BookSpec

The book half of the reference system. Held fixed across every study in this file — Route 2
varies POLICY (trust, filter, cap, risk), never the book, so `build_books` runs once and every
arm is a re-`simulate` over the same `MatchBook` objects.

That is not merely an optimisation: it means two arms cannot differ because they priced different
books, which is the failure mode a per-arm rebuild invites.
"""
reference_spec() = _pf().BookSpec(
    markets = _dd().MarketConfig(_dd().AbstractMarket[
        _dd().Market1X2(), _dd().MarketBTTS(),
        (_dd().MarketOverUnder(l) for l in (0.5, 1.5, 2.5, 3.5, 4.5))...]))

"""
    reference_policy(; trust, filter, risk, cap) -> PolicySpec

The runbook default, with the pieces this stream varies exposed as keywords.

`FixedCap(0.25)` rather than the quickstart's 0.10: the WP5 homogeneity result says a binding
`SlateDrawdown` makes absolute stake levels the risk model's business, so the cap is set where
the staking-layer stream measured it rather than at the runbook's more conservative value.
"""
reference_policy(; trust = _pf().FlatTrust(0.25),
                   filter = _pf().KeepAll(),
                   risk = _pf().SlateDrawdown(23.0),
                   cap = _pf().FixedCap(0.25)) =
    _pf().PolicySpec(trust = trust, risk = risk, cap = cap, filter = filter,
                     grouping = _pf().DailySlate())

# ===================================================================
# 1b. Two filters src does not ship
# ===================================================================
#
# `src/Portfolio/implementations/filters.jl` has `MinOdds`, `MinEdge`, `MarketWhitelist`,
# `KeepAll` and `FilterChain` — but nothing that caps odds from ABOVE, and nothing that caps the
# model's claimed disagreement. Those are exactly the two cuts the curse curve points at, so they
# are defined here and graduate to src only if they earn it.
#
# ⚠️ The methods are added to `BayesianFootball.Portfolio.keep`, fully qualified. Writing
# `keep(f::MaxOdds, ...)` unqualified defines a NEW function in `Main`, the simulator keeps
# calling `Portfolio.keep`, no method matches this type, and the filter is a silent no-op — the
# arm then "loses" for reasons that have nothing to do with the filter.

"""
    MaxOdds(o)

Skip selections priced above `o`. The favourite–longshot cut: long prices are systematically
overpriced, so a model's positive claim on them is more often the market's margin than
information. WP5 measured beat rates of 28.9% / 14.0% above 6.0 on the order-book corpus.
"""
struct MaxOdds <: BayesianFootball.Portfolio.AbstractSelectionFilter
    o::Float64
end
BayesianFootball.Portfolio.keep(f::MaxOdds, s, ::Real, ctx) = s.odds_used <= f.o

"""
    MaxClaim(c)

Skip selections where the model claims more than `c` above the de-vigged market probability.

The filter form of the curse curve. `MinEdge` keeps the legs with the LARGEST disagreement;
this discards them, which is the opposite intervention and the one the measurement points at if
skill really does fall in the upper tail.
"""
struct MaxClaim <: BayesianFootball.Portfolio.AbstractSelectionFilter
    c::Float64
end
BayesianFootball.Portfolio.keep(f::MaxClaim, s, ::Real, ctx) = (s.p_model - s.p_market) <= f.c

# ===================================================================
# 2. The full book as a scored frame
# ===================================================================

"""
    books_frame(books, ds) -> DataFrame

Every selection of every book, with the model probability, the de-vigged market probability and
the realised outcome — the Route-2 analogue of WP8's `full_book_close`.

Taken from `MatchBook.sels` and `MatchBook.settle` rather than re-derived, so it is exactly the
book the simulator staked. `settle` is the realised per-unit payoff, so `is_winner` is
`settle > 0` — a push (settle == 0) is neither, and is dropped by `usable` downstream because
it carries no information about a probability forecast.
"""
function books_frame(books, ds)
    rows = NamedTuple[]
    for b in books
        b.settle === nothing && continue
        for (j, s) in enumerate(b.sels)
            push!(rows, (match_id = b.m_id, date = b.date,
                         family = s.family, group = s.group, line = s.line,
                         selection = s.selection,
                         odds = s.odds_used, odds_quoted = s.odds_quoted,
                         p_model = s.p_model, p_market = s.p_market,
                         settle = b.settle[j],
                         is_winner = b.settle[j] > 0))
        end
    end
    isempty(rows) && return DataFrame()
    df = DataFrame(rows)
    df.claim      = df.p_model .- df.p_market
    df.fair_close = df.p_market        # so l05/l06's cuts apply unchanged
    df.season     = Dates.year.(df.date)
    df.stake      = zeros(nrow(df))    # placeholder; skill cuts do not use it
    df.pnl        = zeros(nrow(df))
    return df
end

# ===================================================================
# 3. Running a policy
# ===================================================================

"""
    run_policy(books, policy; label) -> NamedTuple

Group, simulate, and report. One row of any race in this file.

`n_slates` is carried because the path metrics (Calmar, Sterling, Burke, Sortino) are only
meaningful once there are enough settlement windows — WP1 set that bar at 25, and the order-book
corpus never cleared it. Route 2 has 293 matches across a full season and a half, so for the
first time in this stream those metrics are worth reading.
"""
function run_policy(books, policy; label::AbstractString = "policy", metrics = [])
    PF = _pf()
    slates = PF.group(policy.grouping, books)
    traj   = PF.simulate(policy, slates)
    rep    = PF.report(traj, metrics)
    return (label = label, traj = traj, n_slates = length(slates),
            n_bets = nrow(traj.bets),
            final = round(traj.bankroll[end], digits = 4),
            roi = round(100 * traj.total_pl / max(traj.total_stake, eps()), digits = 2),
            roi_lo = round(rep.roi_ci_lo, digits = 2),
            roi_hi = round(rep.roi_ci_hi, digits = 2),
            growth = round(mean(log.(1.0 .+ traj.slate_pl)), digits = 5),
            mdd = round(rep.mdd, digits = 2),
            report = rep)
end

"""
    race(books, arms; metrics) -> DataFrame

Run several policies over the same books and stack the summaries.

Same `books` for every arm by construction — see `reference_spec`.
"""
function race(books, arms::Vector; metrics = [])
    rows = NamedTuple[]
    for (label, pol) in arms
        r = run_policy(books, pol; label = label, metrics = metrics)
        push!(rows, (label = r.label, slates = r.n_slates, bets = r.n_bets,
                     final = r.final, roi = r.roi, roi_lo = r.roi_lo, roi_hi = r.roi_hi,
                     growth = r.growth, mdd = r.mdd))
    end
    return DataFrame(rows)
end

# ===================================================================
# 4. Curation derived out of sample
# ===================================================================

"""
    split_books(books, cutoff) -> (before, after)

Temporal split for deriving curation on history and testing it forward.

Temporal, not cross-league, because Route 2 has the sample size to afford it and because it is
the split that matches how the rule would actually be used: you fit on what has settled and apply
it to what has not. WP5 had to use cross-league only because 267 legs could not be halved.
"""
split_books(books, cutoff::Date) =
    (filter(b -> b.date < cutoff, books), filter(b -> b.date >= cutoff, books))

"""
    trust_from_frame(df; min_legs, default, w_hi, w_lo) -> SelectionTrust

`derive_trust` (l05) against a Route-2 books frame.

Identical decision rule — a three-level step on a match-clustered sign test, not a fitted weight
per family. The reason has not changed with the sample size: fifteen free parameters will find
structure in noise, and the step function makes one decision per family that a bootstrap interval
can actually support.
"""
trust_from_frame(df::AbstractDataFrame; kw...) = derive_trust(df; kw...)
