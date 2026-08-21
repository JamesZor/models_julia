# current_development/scottish_lower/open_play/r07_eval_recomb_benchmark.jl
#
# EVALUATION & BETFAIR BACKTEST: Recombination Head-to-Head Comparison
#
# Compares:
# 1. goals_negbin_ctl_hl365_hs2: Baseline Gross Goals Control (All Goals)
# 2. goals_negbin_open_play_hl365_hs2: Pure Open-Play NegBin (Un-recombined)
# 3. goals_pois_open_play_hl365_hs2: Pure Open-Play Poisson (Un-recombined)
# 4. recomb_pois_integrated_hl365_hs2: Integrated Co-Trained Turing MCMC Recombination

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

function banner(msg::String)
    println("\n", "="^95)
    println("  " * msg)
    println("="^95)
end

banner("🔍 EVALUATION & BETFAIR BACKTEST: RECOMBINATION VS ALL-GOALS BASELINE")

# 1. Load DataStore
ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)
println("✓ Loaded DataStore ($(nrow(ds.matches)) matches)")

# 2. Discover & Load Experiments
ctl_folders = Experiments.list_experiments("scottish_negbin_grid"; data_dir = joinpath(ROOT, "data"))
op_folders  = Experiments.list_experiments("scottish_open_play_grid"; data_dir = joinpath(ROOT, "data"))
all_folders = vcat(ctl_folders, op_folders)
all_loaded  = Experiments.load_experiments(all_folders)

target_models = [
    "goals_negbin_ctl_hl365_hs2",
    "goals_negbin_open_play_hl365_hs2",
    "goals_pois_open_play_hl365_hs2",
    "recomb_pois_integrated_hl365_hs2"
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

println("✓ Loaded $(length(experiments))/$(length(target_models)) target experiments:")
for exp in experiments
    println("  - $(exp.config.name) ($(length(exp.training_results.items)) folds)")
end

# 3. Standard Evaluation Suite
banner("📊 RUNNING COMPREHENSIVE EVALUATION SUITE (RQR, CRPS, LogLoss, GLMEdge)")

selections = [:home, :draw, :away, :btts_yes, :btts_no,
              :over_05, :under_05, :over_15, :under_15, :over_25, :under_25,
              :over_35, :under_35, :over_45, :under_45]

fam = Dict(
    :x12    => [:home, :draw, :away],
    :btts   => [:btts_yes, :btts_no],
    :totals => [:over_05, :under_05, :over_15, :under_15, :over_25, :under_25, :over_35, :under_35, :over_45, :under_45],
)

_col(df, model, colname) = begin
    colname in names(df) || return NaN
    r = df[df.model .== model, colname]
    (isempty(r) || ismissing(r[1])) ? NaN : round(Float64(r[1]), digits = 4)
end

metrics = Evaluation.AbstractScoringRule[
    Evaluation.RQR(),
    Evaluation.CRPS()
]
append!(metrics, [Evaluation.LogLoss(s) for s in selections])
append!(metrics, [Evaluation.GLMEdge(s) for s in selections])

eval_df = Evaluation.evaluate_experiments(metrics, experiments, ds)
present_models = unique(eval_df.model)

banner("📈 1. RANDOMIZED QUANTILE RESIDUALS (RQR Calibration: Mean ~ 0.0, Std ~ 1.0)")
if "rqr_all_mean" in names(eval_df)
    rqr_df = DataFrame(
        model       = eval_df.model,
        mean_all    = [round(v, digits = 4) for v in eval_df.rqr_all_mean],
        std_all     = [round(v, digits = 4) for v in eval_df.rqr_all_std],
        mean_home   = [round(v, digits = 4) for v in eval_df.rqr_home_mean],
        std_home    = [round(v, digits = 4) for v in eval_df.rqr_home_std],
        mean_away   = [round(v, digits = 4) for v in eval_df.rqr_away_mean],
        std_away    = [round(v, digits = 4) for v in eval_df.rqr_away_std],
    )
    show(stdout, MIME("text/plain"), rqr_df)
    println()
end

banner("📉 2. CRPS & FAMILY-POOLED LOG LOSS DIFFERENCES (Lower / More Negative vs Market = Better)")
rows = []
for m in present_models
    crps_val = _col(eval_df, m, "crps_all_score")
    
    # 1X2 LogLoss diff vs market close
    h_ll = _col(eval_df, m, "logloss_home_diff")
    d_ll = _col(eval_df, m, "logloss_draw_diff")
    a_ll = _col(eval_df, m, "logloss_away_diff")
    
    # Family pooled
    x12_vals = [_col(eval_df, m, "logloss_$(s)_diff") for s in fam[:x12]]
    btts_vals = [_col(eval_df, m, "logloss_$(s)_diff") for s in fam[:btts]]
    tot_vals = [_col(eval_df, m, "logloss_$(s)_diff") for s in fam[:totals]]
    
    mean_x12  = round(mean(filter(!isnan, x12_vals)), digits = 5)
    mean_btts = round(mean(filter(!isnan, btts_vals)), digits = 5)
    mean_tot  = round(mean(filter(!isnan, tot_vals)), digits = 5)
    
    push!(rows, (
        model     = m,
        crps      = crps_val,
        LL_Home   = h_ll,
        LL_Draw   = d_ll,
        LL_Away   = a_ll,
        LL_1X2    = mean_x12,
        LL_BTTS   = mean_btts,
        LL_Totals = mean_tot
    ))
end
ll_summary_df = DataFrame(rows)
show(stdout, MIME("text/plain"), ll_summary_df)
println()

# 4. Betfair Exchange Portfolio Backtest
banner("💰 3. BETFAIR EXCHANGE MULTI-MARKET KELLY SIMULATION (2% Commission, BM 800 Draws)")

bf_summary = Data.summarize_betfair_market(ds)

policies = [
    ("Conservative (Cap 10%, λ=23)", Portfolio.PortfolioSimulationConfig(max_leverage=0.10, lambda_penalty=23.0)),
    ("Balanced Growth (Cap 15%, λ=15)", Portfolio.PortfolioSimulationConfig(max_leverage=0.15, lambda_penalty=15.0)),
    ("Aggressive (Cap 25%, λ=10)", Portfolio.PortfolioSimulationConfig(max_leverage=0.25, lambda_penalty=10.0))
]

for (p_name, p_cfg) in policies
    println("\n" * "="^95)
    println("BETFAIR PORTFOLIO: $p_name")
    println("="^95)
    
    sim_rows = []
    for exp in experiments
        oos_latents = Experiments.extract_oos_predictions(ds, exp)
        books = Portfolio.build_books(exp.config.model, oos_latents, bf_summary, ds; n_draws_joint=800)
        slates = Portfolio.group(Portfolio.DailySlate(), books)
        sim = Portfolio.simulate(slates, p_cfg)
        
        n_bets = sum(length(s.orders) for s in sim.slates)
        mdd = isempty(sim.drawdowns) ? 0.0 : minimum(sim.drawdowns) * 100.0
        
        push!(sim_rows, (
            model        = exp.config.name,
            final_wealth = round(sim.final_wealth, digits=3),
            growth_slate = round(sim.mean_slate_growth, digits=5),
            roi_pct      = round(sim.roi * 100.0, digits=2),
            mean_expo    = round(sim.mean_exposure * 100.0, digits=1),
            mdd_pct      = round(mdd, digits=2),
            sharpe       = round(sim.sharpe, digits=2),
            n_bets       = n_bets
        ))
    end
    
    res_df = DataFrame(sim_rows)
    sort!(res_df, :final_wealth, rev=true)
    show(stdout, MIME("text/plain"), res_df)
    println()
end

banner("✓ Recombination Benchmark Evaluation Complete!")
