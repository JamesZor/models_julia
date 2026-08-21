# current_development/scottish_lower/open_play/r07_eval_recomb_benchmark.jl
#
# EVALUATION & BETFAIR BACKTEST: Recombination Head-to-Head Comparison
#
# Compares:
# 1. goals_negbin_ctl_hl365_hs2: Baseline Gross Goals Control (All Goals)
# 2. goals_negbin_open_play_hl365_hs2: Pure Open-Play NegBin (Un-recombined)
# 3. goals_pois_open_play_hl365_hs2: Pure Open-Play Poisson (Un-recombined)
# 4. recomb_pois_empirical_bayes: Open-Play Poisson + Analytical Empirical Bayes Recombination
# 5. recomb_pois_integrated_hl365_hs2: Integrated Co-Trained Turing MCMC Recombination

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, Printf, LinearAlgebra

const PreGame     = BayesianFootball.Models.PreGame
const Experiments = BayesianFootball.Experiments
const Evaluation  = BayesianFootball.Evaluation
const Portfolio   = BayesianFootball.Portfolio
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include("l01_open_play_feature.jl")
include("l02_open_play_engines.jl")
include("l03_recombination_models.jl")

println("\n", "="^95)
println("🔍 EVALUATION & BETFAIR BACKTEST: RECOMBINATION VS ALL-GOALS BASELINE")
println("="^95)

# 1. Load DataStore
ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)
println("✓ Loaded DataStore ($(nrow(ds.matches)) matches)")

# 2. Discover & Load Experiments
exp_dirs = [
    joinpath(ROOT, "data/scottish_open_play_grid"),
    joinpath(ROOT, "data/scottish_negbin_grid")
]

target_names = [
    "goals_negbin_ctl_hl365_hs2",
    "goals_negbin_open_play_hl365_hs2",
    "goals_pois_open_play_hl365_hs2",
    "recomb_pois_integrated_hl365_hs2"
]

loaded_experiments = []
for name in target_names
    found = false
    for d in exp_dirs
        !isdir(d) && continue
        matching_dirs = sort(filter(x -> startswith(x, name) && isdir(joinpath(d, x)), readdir(d)))
        if !isempty(matching_dirs)
            target_path = joinpath(d, matching_dirs[end])
            exp = Experiments.load_experiment(target_path)
            push!(loaded_experiments, exp)
            println("✓ Loaded: $(name) ($(length(exp.results)) folds) from $(basename(target_path))")
            found = true
            break
        end
    end
    if !found
        println("⚠️ Note: Experiment $name not found yet (will be evaluated once trained)")
    end
end

if isempty(loaded_experiments)
    println("No experiments loaded. Exiting.")
    exit(0)
end

# 3. Standard Evaluation Suite
println("\n", "="^95)
println("📊 RUNNING BATCH EVALUATION SUITE (RQR, CRPS, 1X2 LogLoss, Totals)")
println("="^95)

eval_metrics = [
    Evaluation.RQRCalibration(),
    Evaluation.CRPSMetric(),
    Evaluation.LogLossDiffSelection(:x12),
    Evaluation.LogLossDiffFamily([:x12, :btts, :totals])
]

eval_df = Evaluation.evaluate_experiments(eval_metrics, loaded_experiments, ds)

# 4. Betfair Exchange Portfolio Backtest
println("\n", "="^95)
println("💰 BETFAIR EXCHANGE MULTI-MARKET KELLY SIMULATION (2% Commission, BM 800 Draws)")
println("="^95)

bf_summary = Data.summarize_betfair_market(ds; pre_match_window_minutes=(20, 0))

policies = [
    ("Conservative (Cap 10%, λ=23)", Portfolio.PortfolioSimulationConfig(max_leverage=0.10, lambda_penalty=23.0)),
    ("Balanced Growth (Cap 15%, λ=15)", Portfolio.PortfolioSimulationConfig(max_leverage=0.15, lambda_penalty=15.0)),
    ("Aggressive (Cap 25%, λ=10)", Portfolio.PortfolioSimulationConfig(max_leverage=0.25, lambda_penalty=10.0))
]

for (p_name, p_cfg) in policies
    println("\n" * "="^95)
    println("BETFAIR PORTFOLIO: $p_name")
    println("="^95)
    
    rows = []
    for exp in loaded_experiments
        oos_latents = Experiments.extract_oos_predictions(ds, exp)
        books = Portfolio.build_books(exp.model, oos_latents, bf_summary, ds; n_draws_joint=800)
        slates = Portfolio.group(Portfolio.DailySlate(), books)
        sim = Portfolio.simulate(slates, p_cfg)
        
        n_bets = sum(length(s.orders) for s in sim.slates)
        mdd = isempty(sim.drawdowns) ? 0.0 : minimum(sim.drawdowns) * 100.0
        
        push!(rows, (
            model        = exp.name,
            final_wealth = round(sim.final_wealth, digits=3),
            growth_slate = round(sim.mean_slate_growth, digits=5),
            roi_pct      = round(sim.roi * 100.0, digits=2),
            mean_expo    = round(sim.mean_exposure * 100.0, digits=1),
            mdd_pct      = round(mdd, digits=2),
            sharpe       = round(sim.sharpe, digits=2),
            n_bets       = n_bets
        ))
    end
    
    res_df = DataFrame(rows)
    sort!(res_df, :final_wealth, rev=true)
    show(stdout, MIME("text/plain"), res_df)
    println()
end

println("\n", "="^95)
println("✓ Recombination Benchmark Evaluation Complete!")
println("="^95)
