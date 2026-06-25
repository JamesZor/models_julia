#=
OVERNIGHT GRID — double-Poisson market-pillar comparison (Ireland). Modelled on
ab_test_dixon_coles/r07_grid_search_market_weight.jl.

Question: does the SPLIT (level/supremacy) market pillar buy anything over (a) NO market and
(b) the OLD isotropic market pillar — on OOS LogLoss / CLV / Kelly growth? The κ-spread analysis
suggested market-informed supremacy (which reaches the score grid via κ) might sharpen 1X2
(top-vs-bottom) at the cost of totals; this tests it.

7 cells (~3-4h):
  - dp_nomarket                : no market pillar (baseline)
  - dp_old_mw{50,100}          : old isotropic market, market_weight 0.5 / 1.0
  - dp_split_lw{0,25,50,100}   : split market, supremacy_weight=1.0 FIXED, level_weight 0/.25/.5/1
                                 (sw is fixed at the natural 1.0; sw>1 just pumps the κ distortion)

Train pillar from plain ds (SofaScore odds); eval CLV vs Betfair (ds1) => no CLV leakage.

Run after git pull + REPL restart:
    include("current_development/split_market_pillar/r05_grid_search_double_poisson.jl")
=#

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

include("l02_split_market_poisson.jl")

# ==========================================
# 1. DATA  (train pillar = SofaScore ds.odds)
# ==========================================
println("[INFO] Loading Ireland DataStore...")
ds = Data.load_datastore_cached(Data.Ireland())

save_dir = "./data/double_poisson_market_grid/"
mkpath(save_dir)

# ==========================================
# 2. SHARED CONFIG (matches r07)
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
feature_cfg_bayes = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))

const HALF_LIFE = 60.0
dyn_cfg = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=HALF_LIFE)

samples        = 800
warmup         = 300
chains         = 4
target_seasons = ["2025", "2026"]
dynamics_col   = :match_biweek

# ==========================================
# 3. MODEL SPECS  (name => model)
# ==========================================
_nomarket() = PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
)

_old(mw) = PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    market_weight          = mw,
)

# supremacy_weight fixed at the natural 1.0; sweep level_weight only.
_split(lw) = SplitMarketDoublePoissonModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    market_on              = true,
    supremacy_weight       = 1.0,
    level_weight           = lw,
)

# --- grid vectors (trim/expand here) ---
old_market_weights = [0.5, 1.0]
split_level_weights = [0.0, 0.25, 0.5, 1.0]

specs = Tuple{String, Any}[]
push!(specs, ("dp_nomarket", _nomarket()))
for mw in old_market_weights
    push!(specs, ("dp_old_mw$(Int(round(mw*100)))", _old(mw)))
end
for lw in split_level_weights
    push!(specs, ("dp_split_lw$(Int(round(lw*100)))", _split(lw)))
end

println("[INFO] Grid: $(length(specs)) cells -> ", join(first.(specs), ", "))

# ==========================================
# 4. PHASE 1 — RUN + SAVE (robust; each persisted)
# ==========================================
all_results = Any[]
failures = String[]
t0 = time()
for (name, model) in specs
    println("\n", "#"^70, "\n# RUN: $name\n", "#"^70)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons  = target_seasons,
            history_seasons = 2,
            warmup_period   = 0,
            dynamics_col    = dynamics_col,
            samples         = samples,
            warmup          = warmup,
            chains          = chains,
            use_queue       = true,
            max_depth       = 10,
        )
        res = Experiments.run_experiment(task)
        Experiments.save_experiment(res)
        push!(all_results, res)
    catch e
        @error "FAILED: $name" exception=(e, catch_backtrace())
        push!(failures, name)
    end
end
println("\n[INFO] Phase 1 done in $(round((time()-t0)/60, digits=1)) min. " *
        "OK: $(length(all_results))/$(length(specs)). Failures: $(isempty(failures) ? "none" : join(failures, ", "))")

# Re-eval from disk without re-running (uncomment):
# saved = Experiments.list_experiments(save_dir, data_dir="")
# all_results = [Experiments.load_experiment(saved, i) for i in 1:length(saved)]

# ==========================================
# 5. PHASE 2 — EVAL (CLV vs Betfair)
# ==========================================
try
    odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
    ds1 = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)

    println("\n", "="^60, "\n📊 GLM Edge (Betfair)\n", "="^60)
    eval_glmedge = Evaluation.evaluate_experiments(Evaluation.GLMEdge(), all_results, ds1)
    Evaluation.display_summary_metric(eval_glmedge, :glmedge)

    println("\n", "="^60, "\n📉 LogLoss (Betfair)\n", "="^60)
    eval_logloss = Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds1)
    Evaluation.display_summary_metric(eval_logloss, :logloss)

    println("\n", "="^60, "\n💰 Backtest (BayesianKelly)\n", "="^60)
    ledger = BackTesting.run_backtest(
        ds1, all_results, [Signals.BayesianKelly()];
        market_config = Data.Markets.DEFAULT_MARKET_CONFIG
    )
    tearsheet = BackTesting.generate_tearsheet(ledger)
    cols_to_show = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover,
                    :profit, :roi_pct, :win_rate_pct, :hurdle_G_emp, :hurdle_n_bets, :hurdle_avg_stake,
                    :hurdle_E_R, :hurdle_sharpe, :hurdle_p, :hurdle_G]
    println("\n>>> Backtest by market selection (hurdle_G higher = better):")
    for m_name in unique(tearsheet.selection)
        println("\nSelection: $m_name")
        show(subset(tearsheet, :selection => ByRow(isequal(m_name)))[!, cols_to_show]; truncate=0)
        println()
    end
catch e
    @error "Eval phase failed (experiments are still saved on disk — re-run Phase 2 from the load block)" exception=(e, catch_backtrace())
end

println("\n[INFO] r05 grid complete.")
