# _setup.jl -- shared data loading for the runbook.
#
# Not a runner: every rXX file starts with `include("_setup.jl")`. Everything is guarded on
# `isdefined`, so re-including it in a warm REPL is free.
#
# What we need, and why:
#   ds          the DataStore -- we use ds.matches for final scores and kick-off dates
#   odds        Betfair closing prices, summarised into one row per (match, market, selection)
#   expr        a trained L1 experiment; carries the model we ask for score matrices
#   latents_df  the OUT-OF-SAMPLE posterior summaries, one row per match
#
# `latents_df` is the expensive one (minutes), so it is cached to .jls.

using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization

const PF = BayesianFootball.Portfolio       # the module under demonstration
const DD = BayesianFootball.Data
const EE = BayesianFootball.Experiments
const BT = BayesianFootball.BackTesting

const RUNBOOK_CACHE = joinpath(@__DIR__, "_data_cache.jls")

if !isdefined(Main, :ds)
    @info "loading DataStore (uses .cache/ if warm)"
    global ds = DD.load_datastore_cached(DD.ScottishLower())
end

if !isdefined(Main, :expr)
    # index 3 = funnel_apm_xg, the champion L1 engine for this league.
    # Call list_experiments with no index to see what else is available.
    global expr = EE.load_experiment(
        EE.list_experiments("./data/experiments/plus_minus_biweek", data_dir = ""), 3)
end

if !isdefined(Main, :odds) || !isdefined(Main, :latents_df)
    if isfile(RUNBOOK_CACHE)
        @info "restoring odds + latents from cache" RUNBOOK_CACHE
        _w = deserialize(RUNBOOK_CACHE)
        global odds       = _w.odds
        global latents_df = _w.latents_df
    else
        @info "building odds summary + OOS latents (slow, cached afterwards)"
        # close_window = (-20, 0) means "the last 20 minutes before kick-off".
        # minutes_to_kickoff is NEGATIVE pre-match, so this excludes all in-play ticks.
        global odds = DD.summarize_betfair_market(ds, open_window = (-100000.0, -10.0),
                                                  close_window = (-20.0, 0.0))
        global latents_df = EE.extract_oos_predictions(ds, expr).df
        serialize(RUNBOOK_CACHE, (odds = odds, latents_df = latents_df))
    end
end

# The book of markets we are willing to price. Adding a line here is the ONLY change needed to
# bet it -- masks and settlement are both derived from Data.grade_selection, so the module does
# not need to learn a new market.
const MARKETS = DD.MarketConfig(reduce(vcat, (
    DD.AbstractMarket[DD.Market1X2(), DD.MarketBTTS()],
    [DD.MarketOverUnder(i + 0.5) for i in 0:4],
)))

@info "runbook setup ready" n_matches = nrow(latents_df) n_quotes = nrow(odds) markets = length(MARKETS.markets)
