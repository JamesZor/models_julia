# current_development/scottish_lower/neg_bin/r03_eval_negbin.jl
#
# RUNNER: Comprehensive Evaluation Suite (LogLoss, GLMEdge, RQR, CRPS)
#         & Betfair Exchange Multi-Market Portfolio Backtest
#         for Robust Negative Binomial (NB2) Scottish Lower Models
#
# Comparing:
# 1. goals_negbin_ctl_hl365_hs2        (Baseline Goals-Only NegBin Control)
# 2. pxg_apm_negbin_hl365_hs2          (Arm A: Proxy xG Gamma + RAPM + NegBin Goals)
# 3. funnel_pxg_apm_negbin_hl365_hs2   (Arm B: Shots Volume Poisson + Proxy xG Quality + RAPM + NegBin Goals)
# Against Poisson Benchmarks from scottish_pxg_grid and scottish_wealth_grid

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization, LinearAlgebra

const Evaluation  = BayesianFootball.Evaluation
const Experiments = BayesianFootball.Experiments
const Portfolio   = BayesianFootball.Portfolio
const BackTesting = BayesianFootball.BackTesting
const Signals     = BayesianFootball.Signals
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_lower/proxy_xg/l02_pxg_engines.jl"))
include("l01_negbin_engines.jl")

banner(s) = (println("\n", "="^95); println(s); println("="^95))

banner("1. LOADING DATASTORE & EXPERIMENTS")

CACHE_DIR = joinpath(@__DIR__, "cache")
mkpath(CACHE_DIR)

ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 720)

negbin_folders = Experiments.list_experiments("scottish_negbin_grid"; data_dir = joinpath(ROOT, "data"))
pxg_folders    = Experiments.list_experiments("scottish_pxg_grid"; data_dir = joinpath(ROOT, "data"))
all_folders    = vcat(negbin_folders, pxg_folders)
all_loaded     = Experiments.load_experiments(all_folders)

target_models = [
    "funnel_apm_ctl_hl365_hs2",
    "pxg_apm_hl365_hs2",
    "funnel_pxg_apm_hl365_hs2",
    "goals_negbin_ctl_hl365_hs2",
    "pxg_apm_negbin_hl365_hs2",
    "funnel_pxg_apm_negbin_hl365_hs2"
]

experiments_dict = Dict{String, Any}()
for exp in all_loaded
    for t in target_models
        if startswith(exp.config.name, t)
            experiments_dict[t] = exp
        end
    end
end

experiments = [experiments_dict[t] for t in target_models if haskey(experiments_dict, t)]

println("✓ Loaded DataStore ($(nrow(ds.matches)) matches)")
println("✓ Loaded $(length(experiments)) benchmark experiments:")
for exp in experiments
    println("  - $(exp.config.name) ($(length(exp.training_results.items)) folds)")
end

# ==============================================================================
# 2. EVALUATION SUITE: LOGLOSS, GLMEDGE, RQR, CRPS
# ==============================================================================
banner("2. EXECUTING COMPREHENSIVE EVALUATION SUITE (RQR, GLMEdge, LogLoss, CRPS)")

selections = [:home, :draw, :away, :btts_yes, :btts_no,
              :over_05, :under_05, :over_15, :under_15, :over_25, :under_25,
              :over_35, :under_35, :over_45, :under_45]

metrics = Evaluation.AbstractScoringRule[
    Evaluation.RQR(),
    Evaluation.CRPS()
]
append!(metrics, [Evaluation.LogLoss(s) for s in selections])
append!(metrics, [Evaluation.GLMEdge(s) for s in selections])

eval_df = Evaluation.evaluate_experiments(metrics, experiments, ds)

banner("3. EVALUATION METRIC TABLES")

# A. RQR Summary (Randomized Quantile Residuals: Mean ~ 0.0, Std ~ 1.0)
println("\n" * "="^85)
println("📈 RANDOMIZED QUANTILE RESIDUALS (RQR Calibration: Mean ~ 0.0, Std ~ 1.0)")
println("="^85)
if "rqr_all_mean" in names(eval_df)
    rqr_df = DataFrame(
        model       = eval_df.model,
        mean_all    = [round(v, digits = 4) for v in eval_df.rqr_all_mean],
        std_all     = [round(v, digits = 4) for v in eval_df.rqr_all_std],
        mean_home   = [round(v, digits = 4) for v in eval_df.rqr_home_mean],
        std_home    = [round(v, digits = 4) for v in eval_df.rqr_home_std],
        mean_away   = [round(v, digits = 4) for v in eval_df.rqr_away_mean],
        std_away    = [round(v, digits = 4) for v in eval_df.rqr_away_std]
    )
    println(rqr_df)
end

# B. GLMEdge Multi-Market Summary
println("\n" * "="^85)
println("🎯 GLMEDGE (Market Efficiency Edge: Higher is Better)")
println("="^85)
edge_cols = filter(n -> startswith(n, "glm_edge_"), names(eval_df))
if !isempty(edge_cols)
    edge_df = select(eval_df, :model, edge_cols...)
    for col in edge_cols
        edge_df[!, col] = [round(v, digits = 4) for v in edge_df[!, col]]
    end
    println(edge_df)
end

# C. LogLoss Multi-Market Summary
println("\n" * "="^85)
println("📉 LOGLOSS (Predictive Information: Lower is Better)")
println("="^85)
ll_cols = filter(n -> startswith(n, "log_loss_"), names(eval_df))
if !isempty(ll_cols)
    ll_df = select(eval_df, :model, ll_cols...)
    for col in ll_cols
        ll_df[!, col] = [round(v, digits = 4) for v in ll_df[!, col]]
    end
    println(ll_df)
end

# ==============================================================================
# 4. BETFAIR EXCHANGE MULTI-MARKET PORTFOLIO BACKTEST
# ==============================================================================
banner("4. BETFAIR EXCHANGE MULTI-MARKET PORTFOLIO BACKTEST")

portfolio_markets = [
    Portfolio.BetfairMarketConfig(:match_odds, [:home, :draw, :away]),
    Portfolio.BetfairMarketConfig(:both_teams_to_score, [:btts_yes, :btts_no]),
    Portfolio.BetfairMarketConfig(:over_under_25, [:over_25, :under_25]),
    Portfolio.BetfairMarketConfig(:over_under_15, [:over_15, :under_15]),
    Portfolio.BetfairMarketConfig(:over_under_35, [:over_35, :under_35])
]

for exp in experiments
    println("\n" * "-"^85)
    println("💰 PORTFOLIO BACKTEST: $(exp.config.name)")
    println("-"^85)
    
    port_config = Portfolio.BetfairPortfolioConfig(
        markets            = portfolio_markets,
        initial_bankroll   = 10_000.0,
        fractional_kelly   = 0.25,
        min_edge           = 0.025,
        max_edge           = 0.35,
        max_stake_fraction = 0.05,
        commission_rate    = 0.02
    )

    port_sim = Portfolio.run_betfair_portfolio_simulation(exp, ds, port_config)
    Portfolio.print_portfolio_summary(port_sim)
end
