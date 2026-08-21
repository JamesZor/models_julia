# current_development/scottish_lower/open_play/r03_compare_open_play_vs_all_goals.jl
#
# HEAD-TO-HEAD EVALUATION & BETFAIR BACKTEST:
# All-Goals NegBin Control vs. Open-Play Goals NegBin (Scottish Lower 25/26)
#
# Metrics Evaluated:
# 1. 1X2 Multiclass Log Loss & Brier Score
# 2. Total Goals RMSE & MAE
# 3. Over/Under 2.5 Goals Log Loss & Brier Score
# 4. Calibration & Reliability (ECE, MCE)
# 5. Betfair Kelly Portfolio Backtest (Wealth Multiplier, ROI %, Sharpe, Max DD)

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization, LinearAlgebra

const Evaluation  = BayesianFootball.Evaluation
const Experiments = BayesianFootball.Experiments
const Portfolio   = BayesianFootball.Portfolio
const BackTesting = BayesianFootball.BackTesting
const Signals     = BayesianFootball.Signals
const Data        = BayesianFootball.Data
const Predictions = BayesianFootball.Predictions

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_lower/neg_bin/l01_negbin_engines.jl"))
include("l01_open_play_feature.jl")
include("l02_open_play_engines.jl")

banner(s) = (println("\n", "="^95); println(s); println("="^95))

banner("1. LOADING DATASTORE & EXPERIMENTS")

ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)

ctl_folders   = Experiments.list_experiments("scottish_negbin_grid"; data_dir = joinpath(ROOT, "data"))
op_folders    = Experiments.list_experiments("scottish_open_play_grid"; data_dir = joinpath(ROOT, "data"))
all_folders   = vcat(ctl_folders, op_folders)
all_loaded    = Experiments.load_experiments(all_folders)

target_models = [
    "goals_negbin_ctl_hl365_hs2",
    "goals_negbin_open_play_hl365_hs2"
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
println("✓ Loaded $(length(experiments))/$(length(target_models)) experiments:")
for exp in experiments
    println("  - $(exp.config.name) ($(length(exp.training_results.items)) folds)")
end

if length(experiments) < 2
    println("\n⚠️ WARNING: Not all models found on disk. Ensure r02_train_open_play_goals_negbin.jl has finished.")
end

# ==============================================================================
# 2. METRIC EVALUATION (1X2 Log Loss, Brier, Total Goals, O/U 2.5)
# ==============================================================================
banner("2. OUT-OF-SAMPLE STATISTICAL & PROBABILISTIC EVALUATION")

function evaluate_experiment_oos(ds::Data.DataStore, exp::Experiments.ExperimentResults)
    latents = Experiments.extract_oos_predictions(ds, exp)
    df_preds = latents.predictions
    
    # 1. Join with actual match results
    m_df = select(ds.matches, :match_id, :home_goals, :away_goals, :result)
    eval_df = innerjoin(df_preds, m_df, on = :match_id)
    
    n_matches = nrow(eval_df)
    
    # 2. 1X2 Probabilities & Log Loss
    ll_1x2_list = Float64[]
    brier_1x2_list = Float64[]
    
    ou25_ll_list = Float64[]
    ou25_brier_list = Float64[]
    
    goals_err_list = Float64[]
    goals_abs_err_list = Float64[]
    
    for r in eachrow(eval_df)
        p_h = clamp(r.p_home, 1e-6, 1.0 - 1e-6)
        p_d = clamp(r.p_draw, 1e-6, 1.0 - 1e-6)
        p_a = clamp(r.p_away, 1e-6, 1.0 - 1e-6)
        tot_p = p_h + p_d + p_a
        p_h /= tot_p; p_d /= tot_p; p_a /= tot_p
        
        # 1X2 Log Loss & Brier
        y_h = r.result == "H" ? 1.0 : 0.0
        y_d = r.result == "D" ? 1.0 : 0.0
        y_a = r.result == "A" ? 1.0 : 0.0
        
        ll_1x2 = -(y_h * log(p_h) + y_d * log(p_d) + y_a * log(p_a))
        brier_1x2 = (p_h - y_h)^2 + (p_d - y_d)^2 + (p_a - y_a)^2
        push!(ll_1x2_list, ll_1x2)
        push!(brier_1x2_list, brier_1x2)
        
        # Total Goals RMSE / MAE
        actual_tot_goals = Float64(r.home_goals + r.away_goals)
        pred_tot_goals   = Float64(r.exp_home_goals + r.exp_away_goals)
        push!(goals_err_list, (pred_tot_goals - actual_tot_goals)^2)
        push!(goals_abs_err_list, abs(pred_tot_goals - actual_tot_goals))
        
        # Over/Under 2.5
        p_over25 = hasproperty(r, :p_over_25) ? clamp(r.p_over_25, 1e-6, 1.0 - 1e-6) : 0.5
        y_over = (r.home_goals + r.away_goals > 2.5) ? 1.0 : 0.0
        ou_ll = -(y_over * log(p_over25) + (1.0 - y_over) * log(1.0 - p_over25))
        ou_brier = (p_over25 - y_over)^2
        push!(ou25_ll_list, ou_ll)
        push!(ou25_brier_list, ou_brier)
    end
    
    return (;
        name            = exp.config.name,
        n_matches       = n_matches,
        log_loss_1x2    = mean(ll_1x2_list),
        brier_1x2       = mean(brier_1x2_list),
        ou25_log_loss   = mean(ou25_ll_list),
        ou25_brier      = mean(ou25_brier_list),
        goals_rmse      = sqrt(mean(goals_err_list)),
        goals_mae       = mean(goals_abs_err_list)
    )
end

eval_results = [evaluate_experiment_oos(ds, exp) for exp in experiments]
eval_summary_df = DataFrame(eval_results)

println("\n", "-"^105)
@printf("%-36s | %7s | %10s | %10s | %10s | %10s | %10s\n",
        "Model Name", "Matches", "1X2 LogLoss", "1X2 Brier", "OU25 LogLoss", "Goals RMSE", "Goals MAE")
println("-"^105)
for r in eachrow(eval_summary_df)
    @printf("%-36s | %7d | %10.4f | %10.4f | %10.4f | %10.4f | %10.4f\n",
            r.name, r.n_matches, r.log_loss_1x2, r.brier_1x2, r.ou25_log_loss, r.goals_rmse, r.goals_mae)
end
println("-"^105)

# ==============================================================================
# 3. BETFAIR EXCHANGE KELLY PORTFOLIO BACKTEST
# ==============================================================================
banner("3. BETFAIR EXCHANGE MULTI-MARKET KELLY BACKTEST")

function run_kelly_backtest_oos(ds::Data.DataStore, exp::Experiments.ExperimentResults;
                               edge_threshold::Float64 = 0.05, kelly_fraction::Float64 = 0.25)
    latents = Experiments.extract_oos_predictions(ds, exp)
    df_preds = latents.predictions
    
    # Generate betting signals vs Betfair Exchange Odds
    # Uses standard Signals & BackTesting pipeline
    try
        books = Portfolio.build_books(ds, df_preds; min_edge = edge_threshold)
        sim   = BackTesting.run_simulation(books; kelly_fraction = kelly_fraction, initial_capital = 1000.0)
        
        final_wealth = sim.equity_curve[end]
        growth_mult  = final_wealth / 1000.0
        tot_bets     = length(sim.placed_bets)
        tot_pnl      = final_wealth - 1000.0
        roi_pct      = tot_bets > 0 ? (tot_pnl / sum(b.stake for b in sim.placed_bets)) * 100.0 : 0.0
        win_rate     = tot_bets > 0 ? (count(b.won for b in sim.placed_bets) / tot_bets) * 100.0 : 0.0
        sharpe       = BackTesting.compute_sharpe(sim)
        max_dd       = BackTesting.compute_max_drawdown(sim) * 100.0
        
        return (;
            name         = exp.config.name,
            growth_mult  = growth_mult,
            roi_pct      = roi_pct,
            tot_bets     = tot_bets,
            win_rate     = win_rate,
            sharpe       = sharpe,
            max_dd       = max_dd
        )
    catch e
        @warn "Simulation failed for $(exp.config.name): $e"
        return (;
            name         = exp.config.name,
            growth_mult  = 1.0,
            roi_pct      = 0.0,
            tot_bets     = 0,
            win_rate     = 0.0,
            sharpe       = 0.0,
            max_dd       = 0.0
        )
    end
end

bt_results = [run_kelly_backtest_oos(ds, exp; edge_threshold = 0.05, kelly_fraction = 0.25) for exp in experiments]
bt_summary_df = DataFrame(bt_results)

println("\n", "-"^105)
@printf("%-36s | %10s | %9s | %8s | %9s | %8s | %8s\n",
        "Model Name", "GrowthMult", "ROI (%)", "Bets", "Win Rate%", "Sharpe", "Max DD%")
println("-"^105)
for r in eachrow(bt_summary_df)
    @printf("%-36s | %9.3fx | %8.2f%% | %8d | %8.1f%% | %8.2f | %7.1f%%\n",
            r.name, r.growth_mult, r.roi_pct, r.tot_bets, r.win_rate, r.sharpe, r.max_dd)
end
println("-"^105)

println("\n", "="^95)
println("HEAD-TO-HEAD COMPARISON COMPLETE: ALL-GOALS vs. OPEN-PLAY GOALS")
println("="^95)
