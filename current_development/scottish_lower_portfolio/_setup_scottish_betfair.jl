# current_development/scottish_lower_portfolio/_setup_scottish_betfair.jl
#
# Shared data & experiment loader for Scottish Lower BETFAIR EXCHANGE portfolio simulations.
#
# Loads:
#   ds          - DataStore(ScottishLower())
#   odds        - Betfair Exchange closing summary with 20-minute close window (12,286 quotes)
#   all_exprs   - Dict of all 5 trained Scottish models
#   expr_champ  - Champion model (funnel_pxg_apm_hl365_hs2)
#   latents_map - Dict(model_name => oos_latents_df)
#   MARKETS     - 1X2, BTTS, and Over/Under (0.5 to 4.5)
#   PF / DD / EE / BT module aliases

using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization

const PF = BayesianFootball.Portfolio
const DD = BayesianFootball.Data
const EE = BayesianFootball.Experiments
const BT = BayesianFootball.BackTesting

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_proxy_xg/l02_pxg_engines.jl"))

LEAGUE_TAG = :scottish_lower_betfair

CACHE_DIR = joinpath(@__DIR__, "cache")
mkpath(CACHE_DIR)
LATENTS_CACHE = joinpath(CACHE_DIR, "latents_map_scottish.jls")
ODDS_CACHE    = joinpath(CACHE_DIR, "betfair_summary_odds.jls")

# -------------------------------------------------------------------
# 1. Load DataStore
# -------------------------------------------------------------------
@info "Loading ScottishLower DataStore (cached)..."
global ds = DD.load_datastore_cached(DD.ScottishLower(), max_age_hours = 720)

# -------------------------------------------------------------------
# 2. Summarize Betfair Exchange Closing Odds
# -------------------------------------------------------------------
if isfile(ODDS_CACHE)
    @info "Restoring Betfair closing summary from cache" ODDS_CACHE
    global odds = deserialize(ODDS_CACHE)
else
    @info "Building Betfair closing summary [-20min, 0min] (cached)..."
    global odds = DD.summarize_betfair_market(ds, open_window = (-100000.0, -10.0), close_window = (-20.0, 0.0))
    serialize(ODDS_CACHE, odds)
end
@info "Betfair Odds loaded" n_matches=length(unique(odds.match_id)) n_quotes=nrow(odds)

# -------------------------------------------------------------------
# 3. Load the 5 Trained Grid Models
# -------------------------------------------------------------------
grid_dir = joinpath(ROOT, "data/scottish_pxg_grid")
folders = EE.list_experiments("scottish_pxg_grid"; data_dir = joinpath(ROOT, "data"))
loaded_list = EE.load_experiments(folders)

global all_exprs = Dict{String, Any}()
for exp in loaded_list
    all_exprs[exp.config.name] = exp
end

champ_keys = filter(k -> startswith(k, "funnel_pxg_apm"), collect(keys(all_exprs)))
isempty(champ_keys) && error("Champion funnel_pxg_apm model not found in $grid_dir")
global expr_champ = all_exprs[champ_keys[1]]

# -------------------------------------------------------------------
# 4. Extract / Restore Out-of-Sample Latents
# -------------------------------------------------------------------
if isfile(LATENTS_CACHE)
    @info "Restoring OOS latents from cache" LATENTS_CACHE
    global latents_map = deserialize(LATENTS_CACHE)
else
    @info "Extracting OOS predictions for all 5 models..."
    global latents_map = Dict{String, DataFrame}()
    for (m_name, exp) in all_exprs
        latents_map[m_name] = EE.extract_oos_predictions(ds, exp).df
    end
    serialize(LATENTS_CACHE, latents_map)
end
global latents_df = latents_map[expr_champ.config.name]

# -------------------------------------------------------------------
# 5. Betting Market Universe
# -------------------------------------------------------------------
MARKETS = DD.MarketConfig(reduce(vcat, (
    DD.AbstractMarket[DD.Market1X2(), DD.MarketBTTS()],
    [DD.MarketOverUnder(i + 0.5) for i in 0:4],
)))

PORTFOLIO_LEAGUE = LEAGUE_TAG
@info "Scottish Lower Betfair Setup Ready" league=LEAGUE_TAG n_models=length(all_exprs) markets=length(MARKETS.markets)
