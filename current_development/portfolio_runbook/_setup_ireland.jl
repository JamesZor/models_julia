# _setup_ireland.jl -- Ireland Premier Division variant of `_setup.jl`.
#
# Drop-in replacement: `include("_setup_ireland.jl")` instead of `include("_setup.jl")` and every
# runner (r01-r05) works unchanged. Same variable names on exit: ds, expr, odds, latents_df,
# MARKETS, PF/DD/EE/BT.
#
# WHY IRELAND -- measured on betdb, not assumed:
#
#   tournament          matches w/ betfair   mean ticks   MEDIAN ticks   % empty series
#   56 League One             891               14.9            5             14.4%
#   57 League Two             750               13.7            4             15.9%
#   79 Ireland Premier       1012               39.5           20              1.0%
#   718 Ireland First Div     706               18.8            6             18.4%
#
# 4x the median tick count and 1% empty series versus 14-16%. On ScottishLower the median O/U
# and BTTS market had ONE trade in the 20-minute closing window, which is the single biggest
# reason to distrust that backtest. Ireland Premier is a genuinely priced book.
#
# TWO DIFFERENCES FROM `_setup.jl` THAT MATTER
# ============================================
#
# 1. THE MARKET-PILLAR SWAP. The smile engine takes market odds as a *model feature*
#    (`market_feature_config`). `ds.odds` is the SofaScore bookmaker feed; the engine was
#    trained against the BETFAIR summary swapped into that slot. If you call
#    `extract_oos_predictions` on the un-swapped DataStore you get latents conditioned on a
#    different market pillar than the one the model was fitted with -- silently wrong, no error.
#    Replicated verbatim from `staking_layer/preflight_real.jl`.
#
# 2. THE ENGINE IS DIFFERENT. ScottishLower used `funnel_apm_xg` (index 3 of
#    plus_minus_biweek). Ireland uses `src_sup40_sw40` -- the graduated smile engine from
#    split_market_pillar r21, and the same cell the staking_layer work validated on 275 Ireland
#    Betfair-close matches. It lives in its own experiment tree, loaded by name not by index.
#
# NOT YET RUN: written from the split_market_pillar / preflight_real recipe, but the kaimon
# server was unavailable when this was authored, so it has not been executed end to end.
# Expect to fix a path. See the CHECKS section at the bottom.

using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization

const PF = BayesianFootball.Portfolio
const DD = BayesianFootball.Data
const EE = BayesianFootball.Experiments
const BT = BayesianFootball.BackTesting

# -------------------------------------------------------------------
# 0. League switching
# -------------------------------------------------------------------
# Guarding on `isdefined(Main, :ds)` alone is WRONG: after running the other league's setup
# everything is already defined, so every block below would be skipped and you would silently
# keep the previous league's data. Guard on WHICH league is loaded instead.
#
#   include("_setup.jl")          -> ScottishLower
#   include("_setup_ireland.jl")  -> Ireland          (no restart needed, just include)
#   ENV["RUNBOOK_RELOAD"]="1"     -> force a rebuild of the current league

LEAGUE_TAG = :ireland

_reload = !isdefined(Main, :PORTFOLIO_LEAGUE) ||
          Main.PORTFOLIO_LEAGUE !== LEAGUE_TAG ||
          get(ENV, "RUNBOOK_RELOAD", "0") == "1"

if _reload && isdefined(Main, :PORTFOLIO_LEAGUE) && Main.PORTFOLIO_LEAGUE !== LEAGUE_TAG
    @warn "switching league -- rebuilding ds/expr/odds/latents" from=Main.PORTFOLIO_LEAGUE to=LEAGUE_TAG
end

RUNBOOK_CACHE = joinpath(@__DIR__, "_data_cache_ireland.jls")

# where r21 wrote the src smile grid; each cell is its own subdirectory
const IRELAND_EXP_ROOT = joinpath(pkgdir(BayesianFootball), "data", "double_poisson_smile_src_grid")
const IRELAND_CELL     = "src_sup40_sw40"      # sup=0.4, smile=0.4 -- the validated cell

# -------------------------------------------------------------------
# 1. DataStore, with the Betfair market pillar swapped in
# -------------------------------------------------------------------
if _reload || !isdefined(Main, :ds)
    @info "loading Ireland DataStore (uses .cache/ if warm)"
    _ds_raw = DD.load_datastore_cached(DD.Ireland())

    # The Betfair closing summary, in the ds.odds schema.
    _bf = DD.summarize_betfair_market(_ds_raw, open_window = (-100000.0, -10.0),
                                                close_window = (-20.0, 0.0))

    # Rebuild the DataStore with Betfair in the `odds` slot. This is what the engine was
    # trained against -- see note 1 in the header.
    global ds = DD.DataStore(_ds_raw.segment, _ds_raw.matches, _ds_raw.statistics,
                             _bf, _ds_raw.lineups, _ds_raw.incidents, _ds_raw.betfair_odds)
    global odds = _bf
    @info "market pillar swapped to Betfair" n_quotes = nrow(_bf)
end

# -------------------------------------------------------------------
# 2. The trained engine
# -------------------------------------------------------------------
if _reload || !isdefined(Main, :expr)
    isdir(IRELAND_EXP_ROOT) || error("""
        No experiment tree at $(IRELAND_EXP_ROOT).
        The src smile grid is produced by
        current_development/split_market_pillar/r21_grid_search_src_smile.jl
        and normally lives on the server. Copy the `$(IRELAND_CELL)_*` payload across, or point
        IRELAND_EXP_ROOT at wherever it is.""")

    _hits = filter(isdir, readdir(IRELAND_EXP_ROOT, join = true))
    _cell = findfirst(p -> occursin(IRELAND_CELL, p), _hits)
    _cell === nothing && error("""
        No `$(IRELAND_CELL)` payload under $(IRELAND_EXP_ROOT).
        Found: $(basename.(_hits))""")

    @info "loading engine" cell = basename(_hits[_cell])
    global expr = EE.load_experiment(_hits[_cell])
end

# -------------------------------------------------------------------
# 3. OOS latents  (expensive -> cached)
# -------------------------------------------------------------------
if _reload || !isdefined(Main, :latents_df)
    if isfile(RUNBOOK_CACHE)
        @info "restoring latents from cache" RUNBOOK_CACHE
        global latents_df = deserialize(RUNBOOK_CACHE).latents_df
    else
        @info "extracting OOS latents (slow, cached afterwards)"
        global latents_df = EE.extract_oos_predictions(ds, expr).df   # note: SWAPPED ds
        serialize(RUNBOOK_CACHE, (odds = odds, latents_df = latents_df))
    end
end

# -------------------------------------------------------------------
# 4. The book of markets
# -------------------------------------------------------------------
# Same core seven as ScottishLower. Ireland has NO AsianHandicap / DrawNoBet / DoubleChance in
# the graded Betfair feed, so do not add them here -- they produce empty groups, which the
# completeness guard then silently drops. CorrectScore exists but was a -20% ROI drag in the
# staking_layer work; leave it out until something says otherwise.
MARKETS = DD.MarketConfig(reduce(vcat, (
    DD.AbstractMarket[DD.Market1X2(), DD.MarketBTTS()],
    [DD.MarketOverUnder(i + 0.5) for i in 0:4],
)))

PORTFOLIO_LEAGUE = LEAGUE_TAG
@info "Ireland runbook setup ready" league=LEAGUE_TAG n_matches = nrow(latents_df) n_quotes = nrow(odds) markets = length(MARKETS.markets)

# ===================================================================
# CHECKS -- run these before trusting anything downstream
# ===================================================================
#
#   # a) did the market-pillar swap actually take?
#   ds.odds === odds                       # true
#   names(ds.odds)                         # must contain :odds_close, :prob_fair_close, :is_winner
#
#   # b) is the tick density really better than ScottishLower?
#   using DataFrames
#   bf  = filter(r -> -20.0 <= r.minutes_to_kickoff <= 0.0, ds.betfair_odds)
#   tk  = combine(groupby(bf, [:match_id,:market_name,:market_line,:selection]), nrow => :ticks)
#   combine(groupby(tk, :market_name), :ticks => median => :med,
#           :ticks => (x -> mean(x .== 1)) => :frac_single_tick)
#   #   ScottishLower reference: O/U median 1 tick, 58.5% single-tick
#   #   Ireland should be materially better. If it is not, the switch bought nothing.
#
#   # c) does the engine actually price these matches?
#   spec  = PF.BookSpec(markets = MARKETS)
#   books = PF.build_books(spec, first(latents_df, 20), expr, odds, ds)
#   length(books)                          # expect most of 20; 0 means a schema/id mismatch
#   maximum(b.kkt for b in books) < 1e-4   # solver health
#
# Then: include("r01_quickstart.jl") and everything proceeds as on ScottishLower.
