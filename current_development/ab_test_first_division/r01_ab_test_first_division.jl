# current_development/ab_test_first_division/r01_ab_test_first_division.jl
#
# Outfield-player A/B for Ireland First Division (718), seasons 2025+2026.
# Models (pillars: goals + xG + market):
#   1. Double Poisson  (DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel)
#   2. Dixon-Coles     (DynamicDixonColesXGOutfieldPlayerTimeDecayModel)
#   (Double Negative-Binomial engine to be added next — does not exist yet.)
#
# KEY vs r02_ab_test_ireland.jl: the market pillar is anchored to the **Betfair**
# market, NOT SofaScore. SofaScore `ds.odds` only carries the 1x2 result market,
# which under-identifies the home/away goal rates; Betfair carries 1x2 + BTTS +
# Over/Under lines, giving a far better market-implied (λ_h, λ_a). We therefore
# swap `summarize_betfair_market(ds)` into the `.odds` slot BEFORE task creation,
# so the SAME Betfair frame drives both the training market pillar and evaluation.
#
# Sync first: local edits reach mcmc-beast only via git push then `git pull` in
# /root/BayesianFootball. No src/ change here → no REPL restart needed; just include.

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using ThreadPinning
using ProgressMeter

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Evaluation  = BayesianFootball.Evaluation
const BackTesting = BayesianFootball.BackTesting
const Data        = BayesianFootball.Data

# ==========================================
# 1. SETUP & DATA  (718 First Division)
# ==========================================
println("[INFO] Loading Ireland First Division (718) DataStore...")
ds_raw = Data.load_datastore_cached(Data.IrelandFirstDivision())

save_dir::String = "./data/first_division_ab/"
mkpath(save_dir)

# --- Betfair market as the odds source (for BOTH training pillar and eval) ---
# close_window = closing line (what the market feature reads via prob_fair_close).
odds_bf = Data.summarize_betfair_market(
    ds_raw,
    open_window  = (-100000.0, -10.0),
    close_window = (-20.0, 0.0),
)
ds = Data.DataStore(
    ds_raw.segment, ds_raw.matches, ds_raw.statistics,
    odds_bf, ds_raw.lineups, ds_raw.incidents, ds_raw.betfair_odds,
)

# --- Sanity: confirm Betfair provides the selections the inversion needs ---
println("[CHECK] Betfair summary rows: ", nrow(odds_bf))
if nrow(odds_bf) > 0
    println("[CHECK] selections present: ", sort(unique(string.(odds_bf.selection))))
    n_mkts = combine(groupby(odds_bf, :match_id), nrow => :n)
    println("[CHECK] matches with ≥1 betfair market row: ", nrow(n_mkts),
            " (median markets/match ", Int(round(median(n_mkts.n))), ")")
end

#= RESULT — data/coverage

=#

# ==========================================
# 2. SHARED COMPONENT CONFIGURATION
# ==========================================
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life = 60.0)

tracker_bayes     = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

samples = 800
warmup  = 300
chains  = 4
target_seasons = ["2025", "2026"]

# ==========================================
# 3. MODEL INITIALIZATION  (goals + xG + market pillars)
# ==========================================
# Model 1: Double Poisson + Market
model_dp_m = PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight          = 0.4,
)

# Model 2: Dixon-Coles + Market
model_dc_m = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DixonColesMarketFeature(),
    market_weight          = 0.4,
)

# ==========================================
# 4. TASK CREATION
# ==========================================
task_dp_m = Experiments.create_experiment_task(
    ds, model_dp_m, "FD_DoublePoisson_Market", save_dir;
    target_seasons = target_seasons, dynamics_col = :match_biweek,
    warmup_period = 0, samples = samples, warmup = warmup, chains = chains, use_queue = true,
)

task_dc_m = Experiments.create_experiment_task(
    ds, model_dc_m, "FD_DixonColes_Market", save_dir;
    target_seasons = target_seasons, dynamics_col = :match_biweek,
    warmup_period = 0, samples = samples, warmup = warmup, chains = chains, use_queue = true,
)

# ==========================================
# 5. RUN EXPERIMENTS
# ==========================================
println("--- Running FD Double Poisson Market ---")
res_dp_m = Experiments.run_experiment(task_dp_m)
Experiments.save_experiment(res_dp_m)

println("--- Running FD Dixon Coles Market ---")
res_dc_m = Experiments.run_experiment(task_dc_m)
Experiments.save_experiment(res_dc_m)

all_results = [res_dp_m, res_dc_m]

# ==========================================
# 6. EVALUATION & BACKTESTING  (Betfair odds)
# ==========================================
# `ds` already carries the Betfair odds frame, so it doubles as the eval store.
ds1 = ds

println("\n===========================================")
println("📊 GLM Edge Evaluation (Betfair Odds)")
println("===========================================")
eval_glmedge = Evaluation.evaluate_experiments(Evaluation.GLMEdge(), all_results, ds1)
Evaluation.display_summary_metric(eval_glmedge, :glmedge)

#= RESULT — GLM Edge

=#

println("\n===========================================")
println("📉 LogLoss Evaluation (Betfair Odds)")
println("===========================================")
eval_logloss = Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds1)
Evaluation.display_summary_metric(eval_logloss, :logloss)

#= RESULT — LogLoss

=#

println("\n===========================================")
println("💰 Backtesting Strategy (Kelly)")
println("===========================================")
ledger = BackTesting.run_backtest(
    ds1,
    all_results,
    [BayesianFootball.Signals.BayesianKelly()];
    market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG,
)
tearsheet = BackTesting.generate_tearsheet(ledger)

println("\n>>> Backtest Comparison Summary:")
cols_to_show = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed,
                :turnover, :profit, :roi_pct, :win_rate_pct]
show(tearsheet[:, cols_to_show], allrows = true)

#= RESULT — Backtest

=#

println("\nDone! First Division outfield A/B (DP + DC) complete.")
