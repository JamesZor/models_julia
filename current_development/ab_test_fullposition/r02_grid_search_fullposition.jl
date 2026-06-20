# current_development/ab_test_fullposition/r02_grid_search_fullposition.jl
#
# HALF-LIFE GRID SEARCH for the FULL-POSITION (G/D/M/F) time-decay Dixon-Coles
# player engine. Mirrors r06_grid_search_dynamics.jl but swaps the OUTFIELD model
# for the FULL-POSITION model (8 global positional weights via
# PositionalPlayerDynamics) and writes to its own save location.

using Revise
using BayesianFootball
using DataFrames
using Distributions
using ThreadPinning
using ProgressMeter

# Pin threads for maximum performance
pinthreads(:cores)

const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Evaluation = BayesianFootball.Evaluation
const BackTesting = BayesianFootball.BackTesting
const Data = BayesianFootball.Data

# ==========================================
# 1. SETUP & DATA
# ==========================================
println("[INFO] Loading Ireland DataStore...")
ds = Data.load_datastore_cached(Data.Ireland())

save_dir::String = "./data/fullposition_halflife_grid/"
mkpath(save_dir)

# ==========================================
# 2. SHARED COMPONENT CONFIGURATION
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()

tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

# Fast experiment parameters matching the Double Poisson / Outfield tests
samples = 1000
warmup  = 300
chains  = 4
target_seasons = ["2025", "2026"]
# Using match_biweek to align with HierarchicalMonthlyInterception
dynamics_col = :match_biweek

# ==========================================
# 3. GRID SEARCH SETUP
# ==========================================
half_lives = [14.0, 30.0, 45.0, 60.0, 120.0]
tasks = []
all_results = []

for hl in half_lives
    println("\n[INFO] Creating Model Task for Half-Life: $(hl) days")

    # Full-position dynamics: 8 global positional weights (G/D/M/F att & def)
    dyn_cfg = PreGame.PositionalPlayerDynamics(days_half_life=hl)

    model_dc_fp = PreGame.DynamicDixonColesXGFullPositionPlayerTimeDecayModel(
        interception_config    = inter_cfg,
        player_dynamics_config = dyn_cfg,
        dispersion_config      = disp_cfg,
        homeadvantage_config   = ha_cfg,
        kappa_config           = kap_cfg,
        dixon_coles_config     = PreGame.HierarchicalTeamDixonColesConfig(),
        player_ratings_feature = feature_cfg_bayes,
        market_feature_config  = Features.DixonColesMarketFeature(),
        market_weight          = 0.4
    )

    model_name = "DCFP_HalfLife_$(Int(hl))"

    task = Experiments.create_experiment_task(
        ds, model_dc_fp, model_name, save_dir;
        target_seasons=target_seasons, dynamics_col=dynamics_col,
        warmup_period=0, samples=samples, warmup=warmup, chains=chains, use_queue=true,
    )
    push!(tasks, task)
end

# ==========================================
# 4. RUN EXPERIMENTS
# ==========================================
for task in tasks
    println("\n--- Running Experiment: $(task.config.name) ---")
    res = Experiments.run_experiment(task)
    Experiments.save_experiment(res)
    push!(all_results, res)
end

# NOTE: If you have already run the above models and have them loaded in your REPL,
# you can comment out the run loop and use the load logic below instead:
saved_files = Experiments.list_experiments(save_dir, data_dir="")
all_results = [Experiments.load_experiment(saved_files, i) for i in 1:length(half_lives)]

# ==========================================
# 5. EVALUATION & BACKTESTING
# ==========================================

# Evaluate with Betfair Odds
odds = Data.summarize_betfair_market(
    ds,
    open_window=(-100000.0, -10.0),
    close_window=(-20.0, 0.0)
)
ds1 = Data.DataStore(
  ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds
)

println("\n===========================================")
println("📊 GLM Edge Evaluation (Betfair Odds)")
println("===========================================")
eval_glmedge = Evaluation.evaluate_experiments(Evaluation.GLMEdge(), all_results, ds1)
Evaluation.display_summary_metric(eval_glmedge, :glmedge)

println("\n===========================================")
println("📉 LogLoss Evaluation (Betfair Odds)")
println("===========================================")
eval_logloss = Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds1)
Evaluation.display_summary_metric(eval_logloss, :logloss)

println("\n===========================================")
println("💰 Backtesting Strategy (Kelly)")
println("===========================================")
ledger = BackTesting.run_backtest(
    ds1,
    all_results,
    [BayesianFootball.Signals.BayesianKelly()];
    market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG
)

tearsheet = BackTesting.generate_tearsheet(ledger)

println("\n>>> Backtest Comparison Summary:")
cols_to_show = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover, :profit, :roi_pct, :win_rate_pct, :hurdle_G_emp, :hurdle_scale, :hurdle_shape, :hurdle_n_bets, :hurdle_avg_stake, :hurdle_E_R, :hurdle_sharpe, :hurdle_p, :hurdle_G]
show(tearsheet[:, cols_to_show], allrows=true)

println("\nDone! Full-Position Half-life Grid Search complete.")

model_names = unique(tearsheet.selection)

for m_name in model_names
    println("\nStats for: $m_name")
    sub = subset(tearsheet, :selection => ByRow(isequal(m_name)))
    show(sub[!, cols_to_show]; truncate=0)
end




chains_df_all = Experiments.Diagnostics.extract_chains(ds, all_results[1])

println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv_diag_all = Experiments.Diagnostics.check_convergence(chains_df_all)

println("\n--- Temporal Stability Diagnostics (ADF Stationarity) ---")
stab_diag_all = Experiments.Diagnostics.check_stability(chains_df_all)

