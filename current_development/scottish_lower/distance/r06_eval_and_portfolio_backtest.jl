# current_development/scottish_lower/distance/r06_eval_and_portfolio_backtest.jl
#
# RUNNER: Comprehensive Leaderboard Evaluation (LogLoss, RQR, CRPS)
#         & Betfair Exchange Multi-Market Kelly Portfolio Backtest
#         Comparing Baseline, Wealth, Distance, and Grand Champion Models.

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
include(joinpath(ROOT, "current_development/scottish_lower/neg_bin/l01_negbin_engines.jl"))
include(joinpath(ROOT, "current_development/scottish_lower/neg_bin/l02_negbin_wealth_engines.jl"))
include("l01_distance_features.jl")
include("l02_negbin_distance_engines.jl")
include("l03_negbin_wealth_distance_engines.jl")

banner(s) = (println("\n", "="^95); println(s); println("="^95))

banner("1. LOADING DATASTORE & EXPERIMENTS")

CACHE_DIR = joinpath(@__DIR__, "cache")
mkpath(CACHE_DIR)

ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)

wealth_folders   = Experiments.list_experiments("scottish_negbin_wealth_grid"; data_dir = joinpath(ROOT, "data"))
negbin_folders   = Experiments.list_experiments("scottish_negbin_grid"; data_dir = joinpath(ROOT, "data"))
dist_folders     = Experiments.list_experiments("scottish_distance_negbin_grid"; data_dir = joinpath(ROOT, "data"))
champ_folders    = Experiments.list_experiments("scottish_full_champion_grid"; data_dir = joinpath(ROOT, "data"))
all_folders      = vcat(wealth_folders, negbin_folders, dist_folders, champ_folders)
all_loaded       = Experiments.load_experiments(all_folders)

target_models = [
    "goals_negbin_ctl_hl365_hs2",
    "goals_negbin_dist_hl365_hs2",
    "goals_negbin_wealth_hl365_hs2",
    "goals_negbin_wealth_dist_hl365_hs2",
    "pxg_apm_negbin_hl365_hs2",
    "pxg_apm_negbin_dist_hl365_hs2",
    "pxg_apm_negbin_wealth_hl365_hs2",
    "pxg_apm_negbin_wealth_dist_hl365_hs2"
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

# B. CRPS Summary
println("\n" * "="^85)
println("🎯 CONTINUOUS RANKED PROBABILITY SCORE (CRPS - Lower is Better)")
println("="^85)
if "crps_goals_mean" in names(eval_df)
    crps_df = DataFrame(
        model      = eval_df.model,
        crps_goals = [round(v, digits = 4) for v in eval_df.crps_goals_mean],
        crps_home  = [round(v, digits = 4) for v in eval_df.crps_home_mean],
        crps_away  = [round(v, digits = 4) for v in eval_df.crps_away_mean]
    )
    sort!(crps_df, :crps_goals)
    println(crps_df)
end

# ==============================================================================
# 4. BETFAIR EXCHANGE KELLY PORTFOLIO BACKTEST
# ==============================================================================
banner("4. BETFAIR EXCHANGE MULTI-MARKET PORTFOLIO BACKTEST (2% Commission)")

include(joinpath(ROOT, "current_development/scottish_lower/portfolio/_setup_scottish_betfair.jl"))

bcache_path = joinpath(CACHE_DIR, "scottish_benchmark_books.jls")
books_by_model = if isfile(bcache_path)
    println("✓ Loading pre-built MatchBooks from cache...")
    deserialize(bcache_path)
else
    println("Building MatchBooks for $(length(experiments)) models (800 draws, shrinkage)...")
    b_dict = Dict{String, Any}()
    for exp in experiments
        mname = exp.config.name
        println("  -> Building MatchBooks for: $mname")
        b_dict[mname] = Portfolio.build_all_match_books(
            exp, ds;
            source            = :betfair_summary,
            selections        = selections,
            n_draws           = 800,
            shrinkage_lambda  = 1.0,
            commission        = 0.02
        )
    end
    serialize(bcache_path, b_dict)
    b_dict
end

policies = [
    ("Conservative", Portfolio.RiskAverse(bankroll_cap = 0.10, risk_aversion = 23.0)),
    ("Balanced",     Portfolio.BalancedGrowth(bankroll_cap = 0.15, risk_aversion = 15.0)),
    ("Aggressive",   Portfolio.AggressiveGrowth(bankroll_cap = 0.20, risk_aversion = 10.0))
]

for (pname, policy) in policies
    println("\n" * "="^85)
    println("💼 BETFAIR PORTFOLIO: $pname ($(typeof(policy).name.name))")
    println("="^85)

    port_df = DataFrame(
        model        = String[],
        final_wealth = Float64[],
        growth_slate = Float64[],
        roi_pct      = Float64[],
        mean_expo    = Float64[],
        mdd_pct      = Float64[],
        sharpe       = Float64[],
        n_bets       = Int[]
    )

    for exp in experiments
        mname = exp.config.name
        haskey(books_by_model, mname) || continue
        mbooks = books_by_model[mname]

        sim_res = Portfolio.simulate_portfolio(mbooks, policy)
        ts = BackTesting.generate_tearsheet(sim_res.wealth_series, ds.matches; benchmark = :flat_stake)

        fw = last(sim_res.wealth_series.wealth)
        g_slate = sim_res.metrics.per_slate_growth_rate
        roi = sim_res.metrics.roi * 100.0
        mdd = sim_res.metrics.max_drawdown * 100.0
        expo = mean(sim_res.portfolio_exposure) * 100.0
        total_bets = sum(length(mb.positions) for mb in sim_res.settled_books)

        ret_series = diff(sim_res.wealth_series.wealth) ./ sim_res.wealth_series.wealth[1:(end - 1)]
        sh = (std(ret_series) > 0) ? (mean(ret_series) / std(ret_series)) * sqrt(38) : 0.0

        push!(port_df, (
            mname,
            round(fw, digits = 3),
            round(g_slate, digits = 5),
            round(roi, digits = 2),
            round(expo, digits = 1),
            round(mdd, digits = 2),
            round(sh, digits = 2),
            total_bets
        ))
    end

    sort!(port_df, :final_wealth, rev = true)
    show(port_df, allrows=true, allcols=true); println()
end

println("\n", "="^95)
println("✓ EVALUATION & BETFAIR PORTFOLIO BACKTEST COMPLETE!")
println("="^95)
