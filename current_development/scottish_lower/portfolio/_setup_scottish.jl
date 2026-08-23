# current_development/scottish_lower_portfolio/_setup_scottish.jl
#
# Shared data & experiment loader for Scottish Lower portfolio simulations.
#
# Loads:
#   ds          - DataStore(ScottishLower())
#   odds        - Bet365/SofaScore closing odds (ds.odds)
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

LEAGUE_TAG = :scottish_lower

_reload = !isdefined(Main, :PORTFOLIO_LEAGUE) ||
          Main.PORTFOLIO_LEAGUE !== LEAGUE_TAG ||
          get(ENV, "RUNBOOK_RELOAD", "0") == "1"

CACHE_DIR = joinpath(@__DIR__, "cache")
mkpath(CACHE_DIR)
LATENTS_CACHE = joinpath(CACHE_DIR, "latents_map_scottish.jls")

# -------------------------------------------------------------------
# 1. Load DataStore
# -------------------------------------------------------------------
if _reload || !isdefined(Main, :ds)
    @info "Loading ScottishLower DataStore (cached)..."
    global ds = DD.load_datastore_cached(DD.ScottishLower(), max_age_hours = 720)
    global odds = ds.odds  # Bet365 / SofaScore closing odds book
    @info "DataStore loaded" n_matches = nrow(ds.matches) n_odds = nrow(odds)
end

# -------------------------------------------------------------------
# 2. Load the 5 Trained Grid Models
# -------------------------------------------------------------------
if _reload || !isdefined(Main, :all_exprs)
    @info "Loading 5 Scottish Grid experiments from data/scottish_pxg_grid/..."
    grid_dir = joinpath(ROOT, "data/scottish_pxg_grid")
    folders = EE.list_experiments("scottish_pxg_grid"; data_dir = joinpath(ROOT, "data"))
    loaded_list = EE.load_experiments(folders)
    
    global all_exprs = Dict{String, Any}()
    for exp in loaded_list
        all_exprs[exp.config.name] = exp
    end
    
    # Identify the champion model (funnel_pxg_apm)
    champ_keys = filter(k -> startswith(k, "funnel_pxg_apm"), collect(keys(all_exprs)))
    isempty(champ_keys) && error("Champion funnel_pxg_apm model not found in $grid_dir")
    global expr_champ = all_exprs[champ_keys[1]]
    @info "Loaded $(length(all_exprs)) models. Champion: $(expr_champ.config.name)"
end

# -------------------------------------------------------------------
# 3. Extract / Restore Out-of-Sample Latents for All Models
# -------------------------------------------------------------------
if _reload || !isdefined(Main, :latents_map)
    if isfile(LATENTS_CACHE)
        @info "Restoring OOS latents from cache" LATENTS_CACHE
        global latents_map = deserialize(LATENTS_CACHE)
    else
        @info "Extracting OOS predictions for all 5 models (one-time, cached)..."
        global latents_map = Dict{String, DataFrame}()
        for (m_name, exp) in all_exprs
            @info "  -> Extracting latents for: $m_name"
            latents_map[m_name] = EE.extract_oos_predictions(ds, exp).df
        end
        serialize(LATENTS_CACHE, latents_map)
        @info "Cached latents map to $LATENTS_CACHE"
    end
    global latents_df = latents_map[expr_champ.config.name]
end

# -------------------------------------------------------------------
# 4. Betting Market Universe
# -------------------------------------------------------------------
MARKETS = DD.MarketConfig(reduce(vcat, (
    DD.AbstractMarket[DD.Market1X2(), DD.MarketBTTS()],
    [DD.MarketOverUnder(i + 0.5) for i in 0:4],
)))

PORTFOLIO_LEAGUE = LEAGUE_TAG
@info "Scottish Lower Portfolio Setup Ready" league=LEAGUE_TAG n_models=length(all_exprs) markets=length(MARKETS.markets)
