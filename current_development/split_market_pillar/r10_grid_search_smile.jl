#=
OVERNIGHT GRID — local-intensity SMILE (Poisson) vs the r05 Poisson grid (Ireland).

Trains a small grid of the SMILE model (l03 LocalIntensitySmileDoublePoissonModel, Kmax=4 — the
high strikes K=5,6 are thin/selection-biased per r07, n→0) and evaluates it with the SAME eval as
r05 (GLMEdge / LogLoss / Kelly tearsheet), loading the r05 grid alongside so the smile cells sit
directly next to baseline / old / sup+level in one comparison.

EXPECTATION (be honest): we proved the Ireland market+realized smile is FLAT (φ≈1, V/M 0.94, see
[[no-pregame-intensity-smile]]). So the smile cells should land ~on top of the supremacy-only cells —
this run CONFIRMS that OOS on the backtest, it is unlikely to find lift. Edge, if any, is on
BTTS/totals tails, not 1X2.

Grid (smile_weight sweep, supremacy_weight fixed unless noted; Kmax=4):
  - li_sup_only   : sup=1, smile=0     (supremacy-only control; ≈ r05 dp_split_lw0 but l03)
  - li_smile50    : sup=1, smile=0.5
  - li_smile100   : sup=1, smile=1.0
  - li_smile_only : sup=0, smile=1.0   (totals-smile anchor alone, no who-wins)

Train pillar from plain ds (SofaScore); eval CLV vs Betfair ds1 — no CLV leakage (matches r05).

Run after git pull + REPL restart:
    include("current_development/split_market_pillar/r10_grid_search_smile.jl")
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

# l03 = smile model; l02 = the split (sup+level) cells in the r05 grid must be loadable from disk
# (JLD2 needs the SplitMarketDoublePoissonModel type defined). Including both prints a few
# "redefinition of constant" warnings (shared SUP_PRIOR etc.) — harmless.
include("l02_split_market_poisson.jl")
include("l03_local_intensity_poisson.jl")

# ==========================================
# 1. DATA  (train pillar = SofaScore ds.odds, as r05)
# ==========================================
println("[INFO] Loading Ireland DataStore...")
ds = Data.load_datastore_cached(Data.Ireland())

smile_dir = "./data/double_poisson_smile_grid/"
r05_dir   = "./data/double_poisson_market_grid/"   # the existing baseline/old/sup+level grid
mkpath(smile_dir)

# ==========================================
# 2. SHARED CONFIG (IDENTICAL to r05 for comparability)
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
feature_cfg_bayes = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))
dyn_cfg = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

samples        = 800
warmup         = 300
chains         = 4
target_seasons = ["2025", "2026"]
dynamics_col   = :match_biweek
KMAX           = 4

_smile(sup, sw) = LocalIntensitySmileDoublePoissonModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    smile_feature          = MarketSmileFeature(Kmax = KMAX),
    market_on              = true,
    supremacy_weight       = sup,
    smile_weight           = sw,
)

specs = Tuple{String, Any}[
    ("li_sup_only",   _smile(1.0, 0.0)),
    ("li_smile50",    _smile(1.0, 0.5)),
    ("li_smile100",   _smile(1.0, 1.0)),
    ("li_smile_only", _smile(0.0, 1.0)),
]
println("[INFO] Smile grid: $(length(specs)) cells (Kmax=$KMAX) -> ", join(first.(specs), ", "))

# ==========================================
# 3. PHASE 1 — RUN + SAVE the smile grid (robust)
# ==========================================
smile_results = Any[]
failures = String[]
t0 = time()
for (name, model) in specs
    println("\n", "#"^70, "\n# RUN: $name\n", "#"^70)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, smile_dir;
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
        push!(smile_results, res)
    catch e
        @error "FAILED: $name" exception=(e, catch_backtrace())
        push!(failures, name)
    end
end
println("\n[INFO] Phase 1 done in $(round((time()-t0)/60, digits=1)) min. " *
        "OK: $(length(smile_results))/$(length(specs)). Failures: $(isempty(failures) ? "none" : join(failures, ", "))")

# ==========================================
# 4. PHASE 2 — EVAL smile grid + r05 grid together (CLV vs Betfair)
# ==========================================
try
    # Load the existing r05 grid (baseline / old / sup+level) from disk and combine.
    r05_results = Any[]
    if isdir(r05_dir)
        r05_folders = Experiments.list_experiments(r05_dir; data_dir="")
        r05_results = Experiments.load_experiments(r05_folders)
        println("[INFO] Loaded $(length(r05_results)) r05 cells: ", join([r.config.name for r in r05_results], ", "))
    else
        @warn "r05 grid dir not found ($r05_dir) — evaluating smile grid only."
    end
    all_results = vcat(smile_results, r05_results)

    odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
    ds1  = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)

    println("\n", "="^60, "\n📈 GLM Edge (Betfair) — smile vs r05\n", "="^60)
    Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.GLMEdge(), all_results, ds1), :glmedge)

    println("\n", "="^60, "\n📉 LogLoss (Betfair) — smile vs r05\n", "="^60)
    Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds1), :logloss)

    println("\n", "="^60, "\n💰 Backtest (BayesianKelly) — smile vs r05\n", "="^60)
    ledger = BackTesting.run_backtest(ds1, all_results, [Signals.BayesianKelly()];
                                      market_config = Data.Markets.DEFAULT_MARKET_CONFIG)
    tearsheet = BackTesting.generate_tearsheet(ledger)
    cols = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed, :turnover,
            :profit, :roi_pct, :win_rate_pct, :hurdle_G_emp, :hurdle_n_bets, :hurdle_avg_stake,
            :hurdle_E_R, :hurdle_sharpe, :hurdle_p, :hurdle_G]
    println("\n>>> Backtest by market selection (hurdle_G higher = better):")
    for m_name in unique(tearsheet.selection)
        println("\nSelection: $m_name")
        show(subset(tearsheet, :selection => ByRow(isequal(m_name)))[!, cols]; truncate=0)
        println()
    end
catch e
    @error "Eval phase failed (smile experiments are saved on disk — re-run Phase 2 from the load block)" exception=(e, catch_backtrace())
end

println("\n[INFO] r10 smile grid complete. Compare li_* against dp_split_lw* / dp_nomarket / dp_old_mw*.")
