# current_development/scottish_lower/neg_bin/r07_eval_negbin_wealth.jl
#
# RUNNER: Comprehensive Evaluation Suite & Betfair Exchange Portfolio Backtest
#         for Robust Negative Binomial + Starting-XI Squad Wealth Models

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
include("l02_negbin_wealth_engines.jl")

banner(s) = (println("\n", "="^95); println(s); println("="^95))

banner("1. LOADING DATASTORE & EXPERIMENTS")

CACHE_DIR = joinpath(@__DIR__, "cache")
mkpath(CACHE_DIR)

ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 720)

wealth_folders = Experiments.list_experiments("scottish_negbin_wealth_grid"; data_dir = joinpath(ROOT, "data"))
negbin_folders = Experiments.list_experiments("scottish_negbin_grid"; data_dir = joinpath(ROOT, "data"))
pxg_folders    = Experiments.list_experiments("scottish_pxg_grid"; data_dir = joinpath(ROOT, "data"))
all_folders    = vcat(wealth_folders, negbin_folders, pxg_folders)
all_loaded     = Experiments.load_experiments(all_folders)

target_models = [
    "goals_negbin_ctl_hl365_hs2",
    "pxg_apm_negbin_hl365_hs2",
    "goals_negbin_wealth_hl365_hs2",
    "pxg_apm_negbin_wealth_hl365_hs2",
    "funnel_pxg_apm_negbin_wealth_hl365_hs2"
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

perf_df = DataFrame(
    Model            = String[],
    RQR_Mean_H       = Float64[],
    RQR_Std_H        = Float64[],
    RQR_Mean_A       = Float64[],
    RQR_Std_A        = Float64[],
    RQR_KS_PVal      = Float64[],
    Avg_LogLoss_1X2  = Float64[],
    Avg_LogLoss_OU25 = Float64[],
    CRPS_Goals       = Float64[],
    GLMEdge_Home     = Float64[],
    GLMEdge_Away     = Float64[]
)

for exp in experiments
    name = exp.config.name
    print("Evaluating $name ... ")
    flush(stdout)

    t_eval = @elapsed begin
        eval_res = Evaluation.evaluate_experiment(
            exp, ds;
            eval_selections = selections,
            include_rqr     = true,
            include_glmedge = true
        )
    end

    # 1. RQR Statistics
    rqr = eval_res.rqr_results
    rqr_h_m = hasproperty(rqr, :mean_home) ? rqr.mean_home : mean(rqr.home_residuals)
    rqr_h_s = hasproperty(rqr, :std_home)  ? rqr.std_home  : std(rqr.home_residuals)
    rqr_a_m = hasproperty(rqr, :mean_away) ? rqr.mean_away : mean(rqr.away_residuals)
    rqr_a_s = hasproperty(rqr, :std_away)  ? rqr.std_away  : std(rqr.away_residuals)
    ks_pval = hasproperty(rqr, :ks_pvalue) ? rqr.ks_pvalue : 0.0

    # 2. Market Log-Losses
    ll_1x2  = eval_res.log_loss_1x2
    ll_ou25 = eval_res.log_loss_ou25

    # 3. CRPS
    crps_g  = hasproperty(eval_res, :crps_goals) ? eval_res.crps_goals : NaN

    # 4. GLM Edge
    edge_h  = eval_res.glm_edge_home
    edge_a  = eval_res.glm_edge_away

    push!(perf_df, (
        name,
        round(rqr_h_m, digits=4),
        round(rqr_h_s, digits=4),
        round(rqr_a_m, digits=4),
        round(rqr_a_s, digits=4),
        round(ks_pval, digits=4),
        round(ll_1x2, digits=4),
        round(ll_ou25, digits=4),
        round(crps_g, digits=4),
        round(edge_h, digits=4),
        round(edge_a, digits=4)
    ))
    println("Done ($(round(t_eval, digits=1))s)")
end

println("\n", "="^115)
println(" MODEL EVALUATION & CALIBRATION BENCHMARK (SCOTTISH LOWER)")
println("="^115)
show(perf_df, allrows=true, allcols=true); println()

# ==============================================================================
# 3. BETFAIR EXCHANGE MULTI-MARKET PORTFOLIO BACKTEST
# ==============================================================================
banner("3. BETFAIR EXCHANGE KELLY PORTFOLIO BACKTEST")

signals_config = Signals.SignalConfig(
    markets = [:match_odds, :over_under_25, :both_teams_to_score],
    min_edge = 0.03,
    max_edge = 0.35,
    min_prob = 0.10,
    max_prob = 0.90
)

# Policies: Conservative & Aggressive Fractional Kelly
policies = [
    (:conservative, Portfolio.FractionalKelly(fraction = 0.125, max_stake = 0.03, max_loss = 0.08)),
    (:aggressive,   Portfolio.FractionalKelly(fraction = 0.25,  max_stake = 0.05, max_loss = 0.15))
]

summary_rows = []

for exp in experiments
    name = exp.config.name
    println("\n" * "-"^85)
    println("▶ BACKTESTING PORTFOLIO: $name")
    println("-"^85)

    preds_df = Experiments.build_predictions_dataframe(exp, ds)
    sig_df   = Signals.generate_signals(preds_df, ds.betfair_odds, signals_config)

    if nrow(sig_df) == 0
        println("  [WARN] No signals generated for $name")
        continue
    end

    for (pol_name, kelly_pol) in policies
        port_cfg = Portfolio.PortfolioConfig(
            allocation_strategy = kelly_pol,
            correlation_damping = 0.85
        )

        alloc_df = Portfolio.allocate_portfolio(sig_df, ds.matches, port_cfg)
        bt_res   = BackTesting.run_backtest(alloc_df, ds.matches; initial_bankroll = 1000.0, commission = 0.02)
        m        = bt_res.metrics

        @printf("  [%-12s] Bets: %4d | ROI: %+6.2f%% | Profit: %+8.2f | Sharpe: %5.2f | MaxDD: %5.2f%% | Final: %6.3fx\n",
                string(pol_name), m.total_bets, m.roi * 100, m.total_profit, m.sharpe_ratio, m.max_drawdown * 100, m.final_bankroll / 1000.0)

        push!(summary_rows, (
            Model         = name,
            Policy        = string(pol_name),
            Bets          = m.total_bets,
            ROI_pct       = round(m.roi * 100, digits=2),
            Profit        = round(m.total_profit, digits=2),
            Sharpe        = round(m.sharpe_ratio, digits=2),
            MaxDD_pct     = round(m.max_drawdown * 100, digits=2),
            Final_Wealth  = round(m.final_bankroll / 1000.0, digits=3)
        ))
    end
end

summary_df = DataFrame(summary_rows)
println("\n", "="^115)
println(" BETFAIR PORTFOLIO BACKTEST SUMMARY (SCOTTISH LOWER)")
println("="^115)
show(summary_df, allrows=true, allcols=true); println()
println("\n✓ Comprehensive Evaluation & Portfolio Backtesting Completed Successfully!")
