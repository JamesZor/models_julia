# current_development/split_market_pillar/r01_smoke_and_grid.jl
#
# RUNNER for the axis-split market pillar. Two phases:
#   PHASE 1 — SMOKE: tiny single experiment to prove the engine compiles, initialises
#             (no -Inf), samples, and that extract_parameters + PPD + backtest run.
#   PHASE 2 — GRID: sweep σ_supremacy × σ_level for a chosen rung, then evaluate.
#
# Run on the server (kaimon REPL). See NOTES.md.

using Revise
using BayesianFootball
using DataFrames
using Distributions
using ThreadPinning
using ProgressMeter

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Evaluation  = BayesianFootball.Evaluation
const BackTesting = BayesianFootball.BackTesting
const Data        = BayesianFootball.Data
const Signals     = BayesianFootball.Signals

# Load the loader (defines the SplitMarket* engines + methods)
include(joinpath(@__DIR__, "l01_split_market.jl"))

# ==========================================
# 1. DATA  (+ Betfair swapped into ds.odds for the TRAINING market pillar)
# ==========================================
println("[INFO] Loading Ireland DataStore...")
ds_raw = Data.load_datastore_cached(Data.Ireland())

# IMPORTANT (outfield-xg-engine-gotchas): the market feature inverts ds.odds, so the
# supremacy anchor must be built from Betfair, not SofaScore 1X2. Swap BEFORE training.
odds = Data.summarize_betfair_market(ds_raw, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds = Data.DataStore(
    ds_raw.segment, ds_raw.matches, ds_raw.statistics, odds,
    ds_raw.lineups, ds_raw.incidents, ds_raw.betfair_odds
)

save_dir = "./data/split_market_grid/"
mkpath(save_dir)

# ==========================================
# 2. SHARED COMPONENT CONFIG  (matches r06)
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()

tracker_bayes     = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

target_seasons = ["2025", "2026"]
dynamics_col   = :match_biweek
half_life      = 60.0

# Build a rung by name → struct constructor with given σ's.
function make_model(rung::Symbol, σ_sup::Float64, σ_level::Float64)
    dyn = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=half_life)
    common = (
        interception_config    = inter_cfg,
        player_dynamics_config = dyn,
        dispersion_config      = disp_cfg,
        homeadvantage_config   = ha_cfg,
        kappa_config           = kap_cfg,
        player_ratings_feature = feature_cfg_bayes,
        σ_supremacy            = σ_sup,
        σ_level                = σ_level,
    )
    if rung == :poisson_goals
        return SplitMarketPoissonGoalsModel(; market_feature_config=Features.DoublePoissonMarketFeature(), common...)
    elseif rung == :poisson_xg
        return SplitMarketPoissonXGModel(;    market_feature_config=Features.DoublePoissonMarketFeature(), common...)
    elseif rung == :negbin_xg
        return SplitMarketNegBinXGModel(;     market_feature_config=Features.DoublePoissonMarketFeature(), common...)
    elseif rung == :dixoncoles_xg
        return SplitMarketDixonColesXGModel(; market_feature_config=Features.DixonColesMarketFeature(),
                                              dixon_coles_config=PreGame.HierarchicalTeamDixonColesConfig(), common...)
    else
        error("unknown rung $rung")
    end
end

# ==========================================
# 3. PHASE 1 — SMOKE TEST  (rung 1, supremacy-only, tiny)
# ==========================================
println("\n===== SMOKE TEST: SplitMarketPoissonGoalsModel (σ_sup=0.1, σ_level=Inf) =====")
smoke_model = make_model(:poisson_goals, 0.1, Inf)

smoke_task = Experiments.create_experiment_task(
    ds, smoke_model, "SMOKE_R1_supremacy_only", save_dir;
    target_seasons=target_seasons, dynamics_col=dynamics_col,
    warmup_period=0, samples=50, warmup=50, chains=2, use_queue=true,
)
smoke_res = Experiments.run_experiment(smoke_task)
println("[SMOKE] sampling complete. Checking PPD + backtest wiring...")

# Exercises extract_parameters + the prediction dispatch override end-to-end.
smoke_ppd = BayesianFootball.Predictions.model_inference(ds, smoke_res)
println("[SMOKE] PPD rows: ", nrow(smoke_ppd.df), "  (dispatch OK if no :r ArgumentError)")

# ==========================================
# 4. PHASE 2 — σ GRID  (choose the rung here)
# ==========================================
RUNG = :poisson_goals     # :poisson_goals | :poisson_xg | :negbin_xg | :dixoncoles_xg

σ_sups   = [0.05, 0.1, 0.2]
σ_levels = [0.5, 1.0, Inf]

samples, warmup, chains = 800, 300, 4

grid = [(s, l) for s in σ_sups for l in σ_levels]
tasks = []
for (σs, σl) in grid
    lbl = "SPLIT_$(RUNG)_sup$(replace(string(σs),"."=>""))_lev$(isinf(σl) ? "Inf" : replace(string(σl),"."=>""))"
    println("[INFO] Task: $lbl")
    m = make_model(RUNG, σs, σl)
    push!(tasks, Experiments.create_experiment_task(
        ds, m, lbl, save_dir;
        target_seasons=target_seasons, dynamics_col=dynamics_col,
        warmup_period=0, samples=samples, warmup=warmup, chains=chains, use_queue=true,
    ))
end

all_results = []
for task in tasks
    println("\n--- Running: $(task.config.name) ---")
    res = Experiments.run_experiment(task)
    Experiments.save_experiment(res)
    push!(all_results, res)
end

# To re-load instead of re-running:
# saved = Experiments.list_experiments(save_dir, data_dir="")
# all_results = [Experiments.load_experiment(saved, i) for i in 1:length(grid)]

# ==========================================
# 5. EVALUATION  (mirrors r06)
# ==========================================
println("\n=== LogLoss (Betfair) ===")
eval_logloss = Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds)
Evaluation.display_summary_metric(eval_logloss, :logloss)

println("\n=== GLM Edge (Betfair) ===")
eval_glmedge = Evaluation.evaluate_experiments(Evaluation.GLMEdge(), all_results, ds)
Evaluation.display_summary_metric(eval_glmedge, :glmedge)

println("\n=== Backtest (Kelly) ===")
ledger = BackTesting.run_backtest(
    ds, all_results, [Signals.BayesianKelly()];
    market_config = Data.Markets.DEFAULT_MARKET_CONFIG
)
tearsheet = BackTesting.generate_tearsheet(ledger)

cols = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover,
        :profit, :roi_pct, :win_rate_pct, :hurdle_n_bets, :hurdle_E_R, :hurdle_sharpe, :hurdle_G]
println("\n>>> Backtest summary (per σ-cell × selection):")
show(tearsheet[:, cols], allrows=true, truncate=0)

# Per-selection comparison across the σ grid (find the winning (σ_sup, σ_level) per market).
for sel in unique(tearsheet.selection)
    println("\nSelection: $sel")
    sub = subset(tearsheet, :selection => ByRow(isequal(sel)))
    show(sort(sub[:, cols], :hurdle_G, rev=true); truncate=0)
end

println("\nDone — record the winning σ's per market family in NOTES.md.")
