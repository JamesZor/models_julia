# _setup.jl -- shared loading for the MatchDay runbook.
#
# Not a runner: every rXX file starts with `include("_setup.jl")`. Guarded on a sentinel so
# re-including in a warm REPL is free.
#
# TARGET: Ireland Premier Division (tournament 79) with `src_sup40_sw40`.
#
# "Ireland" is TWO segments and only one has a trained engine:
#     Data.Ireland()              = [79]   Premier Division    <- src_sup40_sw40 was trained here
#     Data.IrelandFirstDivision() = [718]  First Division      <- no engine
# The live Betfair feed actually carries MORE 718 than 79 (38 events vs 33), and 718 is a
# materially different dispersion regime (V/M ~1.14, NB beating Poisson by 9-12 AIC, against 79
# sitting near-Poisson). Pricing 718 with the 79 posterior would under-price totals tails,
# silently, which is the direction that hurts most on an unders-leaning book. So: 79 only.

using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization

const MD = BayesianFootball.MatchDay
const PF = BayesianFootball.Portfolio
const DD = BayesianFootball.Data
const EE = BayesianFootball.Experiments

MATCHDAY_TAG = :ireland_premier
_reload = !isdefined(Main, :MATCHDAY_READY) || get(ENV, "RUNBOOK_RELOAD", "0") == "1"

# The engine's experiment tree. Normally lives on the server; point this elsewhere if you
# copied the payload somewhere else.
EXP_ROOT = joinpath(pkgdir(BayesianFootball), "data", "double_poisson_smile_src_grid")
EXP_CELL = "src_sup40_sw40"

# -------------------------------------------------------------------
# 1. DataStore, with the Betfair market pillar swapped in
# -------------------------------------------------------------------
# The smile engine took market odds as a *training* feature, fitted against the BETFAIR summary
# in the `odds` slot rather than the SofaScore bookmaker feed. Rebuilding the feature collection
# against an un-swapped DataStore gives a FeatureSet conditioned on a different market pillar
# than the chain was fitted with -- silently wrong, no error.
#
# NOTE this matters for feature construction only. `extract_parameters` for this engine reads
# just home_team / away_team / match_id / season_idx / month_idx -- it consumes NO market
# feature at inference. So the live book does not feed the model; it only prices the bets.
if _reload || !isdefined(Main, :ds)
    @info "loading Ireland DataStore (uses .cache/ if warm)"
    _raw = DD.load_datastore_cached(DD.Ireland())
    _bf  = DD.summarize_betfair_market(_raw, open_window = (-100000.0, -10.0),
                                             close_window = (-20.0, 0.0))
    global ds = DD.DataStore(_raw.segment, _raw.matches, _raw.statistics, _bf,
                             _raw.lineups, _raw.incidents, _raw.betfair_odds,
                             _raw.bbc, _raw.bbc_events)
    @info "market pillar swapped to Betfair" n_quotes = nrow(_bf)
end

# -------------------------------------------------------------------
# 2. The trained engine
# -------------------------------------------------------------------
if _reload || !isdefined(Main, :expr)
    isdir(EXP_ROOT) || error("""
        No experiment tree at $(EXP_ROOT).
        Produced by current_development/split_market_pillar/r21_grid_search_src_smile.jl and
        normally living on the server. Copy the `$(EXP_CELL)_*` payload across, or point
        EXP_ROOT at wherever it is.""")
    _hits = filter(isdir, readdir(EXP_ROOT, join = true))
    _i = findfirst(p -> occursin(EXP_CELL, p), _hits)
    _i === nothing && error("No `$(EXP_CELL)` payload under $(EXP_ROOT). Found: $(basename.(_hits))")
    @info "loading engine" cell = basename(_hits[_i])
    global expr = EE.load_experiment(_hits[_i])
end

# -------------------------------------------------------------------
# 3. The staking system
# -------------------------------------------------------------------
# MatchDay does no staking maths. Everything below this line is src/Portfolio, unchanged from
# the portfolio_runbook -- which is the point of the split.
MARKETS = DD.MarketConfig(reduce(vcat, (
    DD.AbstractMarket[DD.Market1X2(), DD.MarketBTTS()],
    [DD.MarketOverUnder(i + 0.5) for i in 0:4])))

SYS = PF.PortfolioSystem(PF.BookSpec(markets = MARKETS), PF.PolicySpec())

# -------------------------------------------------------------------
# 4. A MatchDaySpec you can actually run today
# -------------------------------------------------------------------
# `LastHistorical` needs the DataStore, so the lineup chain cannot be a default -- it is the one
# component that has to be built here rather than taken from MatchDaySpec's defaults.
LINEUPS = MD.SourceChain(MD.ProvisionalDB(), MD.LastHistorical(ds))

GATES = MD.GateChain(MD.IdentityResolved(),
                     MD.MaxBookAge(Minute(30)),
                     MD.MaxLineupAge(max_age = Hour(2), blocking = false),
                     MD.MinMatched(minimum = 500.0, blocking = false))

"Live spec: asks sofascore.events what is on. Returns nothing while the season is between rounds."
live_spec() = MD.MatchDaySpec(fixtures = MD.SofaScoreEvents(horizon = Hour(36)),
                              lineups = LINEUPS, gate = GATES, markets = MARKETS)

"""
    replay_spec(day) -> MatchDaySpec

Replay spec for a past match day. Fixtures come from `ds.matches` because `sofascore.events`
no longer calls a finished match `notstarted` -- which is exactly why `ExplicitFixtures` exists.
"""
function replay_spec(day::Date; kickoff = Time(18, 45))
    rows = filter(r -> Date(r.match_date) == day, ds.matches)
    fx = MD.Fixture[MD.Fixture(Int(r.match_id), String(r.home_team), String(r.away_team),
                               DateTime(day, kickoff), 79) for r in eachrow(rows)]
    return MD.MatchDaySpec(fixtures = MD.ExplicitFixtures(fx), lineups = LINEUPS,
                           gate = GATES, markets = MARKETS)
end

MATCHDAY_READY = true
@info "matchday runbook ready" league = MATCHDAY_TAG n_matches = nrow(ds.matches) markets = length(MARKETS.markets)

# ===================================================================
# THE REPLAY CORPUS -- read this before planning any validation
# ===================================================================
#
# Replay needs an order book AND a resolvable match_id. The intersection is currently:
#
#     35 matches, 2026-05-29 .. 2026-06-26
#
# bounded at the near end by the identity resolver dying (~2026-06-27, after which resolution is
# 0%, having been 100% before) and at the far end by order_book_1m starting 2026-05-29.
#
# The corpus GROWS every match week now that order_book_1m is the source, so it is written as a
# query in r02 rather than as a fixed list. But today it is enough to debug the pipeline and
# nowhere near enough to establish an edge.
