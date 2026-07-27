#=
OVERNIGHT GRID — NegBin local-intensity SMILE (mirrors r10's Poisson smile grid for the `nb` row).

Trains the smile-weight sweep of the NegBin smile engine (l07 LocalIntensitySmileNegBinModel, Kmax=4)
and evaluates it with the SAME frame as r10/r05 (GLMEdge / LogLoss / Kelly tearsheet). The Poisson
smile grid (r10, ./data/double_poisson_smile_grid/) is loaded alongside so the nb-smile cells sit
directly next to the pois-smile cells in one comparison.

THESIS (the reason to build this — l07 header): the smile and the NegBin r attack the SAME thin-tail
problem on DIFFERENT markets, so they compose without double-counting:
  • smile φ(K) → totals SHAPE → O/U   • NegBin r → marginal P(0)/tail → BTTS / correct-score.
So vs the pois-smile we expect the EXTRA lift (if any) on BTTS / correct-score, while O/U stays ≈ the
pois-smile (same Poisson-per-strike O/U pricing). Judge per-line GLMEdge (r13) + pooled totals (r14).

LEAGUE (= r15 caveat): NegBin's r is INERT on near-Poisson Ireland-79 (V/M 0.94 → r→large ⇒ nb-smile ≈
pois-smile). It only BITES on Ireland First Division 718 (V/M≈1.14), where the BTTS/CS lift should show.
  • SEGMENT = Ireland()            → comparable to the saved pois-smile/dp grids; expect ≈ pois-smile.
  • SEGMENT = IrelandFirstDivision() → where the NegBin smile should actually differ (the real test).

Train pillar from SofaScore ds.odds (= r10); eval CLV vs Betfair (ds1).

Cells (canonical, disp=nb; mirror r10's 4):
  - smile_nb_sw0    : sup=1, smile=0     (supremacy-only NegBin control; ≈ split_nb_lw0)
  - smile_nb_sw50   : sup=1, smile=0.5
  - smile_nb_sw100  : sup=1, smile=1.0
  - smile_nb_sup0   : sup=0, smile=1.0   (totals-smile alone, no who-wins)

Run after git pull + REPL restart:
    include("current_development/split_market_pillar/r16_grid_search_smile_negbin.jl")
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

# l03 first (MarketSmileFeature + SmileScoreMatrix + O/U override), then l07 (the NegBin smile model).
include("current_development/split_market_pillar/l03_local_intensity_poisson.jl")
include("current_development/split_market_pillar/l07_local_intensity_negbin.jl")

# ==========================================
# 1. DATA  (train pillar = SofaScore ds.odds, as r10)
# ==========================================
SEGMENT = Data.Ireland()                     # ← swap to Data.IrelandFirstDivision() for the NB-active league
seg_tag = lowercase(string(nameof(typeof(SEGMENT))))
println("[INFO] Loading $(seg_tag) DataStore...")
ds = Data.load_datastore_cached(SEGMENT)

smile_nb_dir = "./data/negbin_smile_grid_$(seg_tag)/"
pois_dir     = "./data/double_poisson_smile_grid/"   # r10 pois-smile grid (for side-by-side, Ireland only)
mkpath(smile_nb_dir)

# ==========================================
# 2. SHARED CONFIG (IDENTICAL to r10 for comparability)
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()      # NegBin r (now USED)
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

_smile_nb(sup, sw) = LocalIntensitySmileNegBinModel(
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
    ("smile_nb_sw0",   _smile_nb(1.0, 0.0)),
    ("smile_nb_sw50",  _smile_nb(1.0, 0.5)),
    ("smile_nb_sw100", _smile_nb(1.0, 1.0)),
    ("smile_nb_sup0",  _smile_nb(0.0, 1.0)),
]
println("[INFO] NegBin smile grid ($seg_tag): $(length(specs)) cells (Kmax=$KMAX) -> ",
        join(first.(specs), ", "))

# ==========================================
# 3. PHASE 1 — RUN + SAVE (robust)
# ==========================================
smile_results = Any[]
failures = String[]
t0 = time()
for (name, model) in specs
    println("\n", "#"^70, "\n# RUN: $name\n", "#"^70)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, smile_nb_dir;
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
# 4. PHASE 2 — EVAL nb-smile + pois-smile together (CLV vs Betfair)
# ==========================================
pois_results = Any[]
if seg_tag == "ireland" && isdir(pois_dir)   # the r10 pois-smile grid is Ireland-only
    pois_folders = Experiments.list_experiments(pois_dir; data_dir="")
    pois_results = Experiments.load_experiments(pois_folders)
    println("[INFO] Loaded $(length(pois_results)) pois-smile cells: ",
            join([r.config.name for r in pois_results], ", "))
end
all_results = convert(Vector{Experiments.ExperimentResults}, vcat(smile_results, pois_results))

try
    odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
    ds1  = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)

    println("\n", "="^60, "\n📊 GLM Edge (Betfair) — nb-smile vs pois-smile\n", "="^60)
    Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.GLMEdge(), all_results, ds1), :glmedge)

    println("\n", "="^60, "\n📉 LogLoss (Betfair) — nb-smile vs pois-smile\n", "="^60)
    Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds1), :logloss)

    println("\n", "="^60, "\n💰 Backtest (BayesianKelly)\n", "="^60)
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
    @error "Eval phase failed (experiments still saved on disk)" exception=(e, catch_backtrace())
end

println("""

[INFO] r16 NegBin smile grid complete ($seg_tag).
 • On Ireland-79 expect nb-smile ≈ pois-smile (r→large). The real test is IrelandFirstDivision-718,
   where the NegBin r bites → look for the EXTRA lift on BTTS / correct-score (NOT O/U, which shares
   the pois-smile's Poisson-per-strike pricing).
 • Next: point r13_per_line_bias_edge_smile.jl / r14_pooled_totals_edge.jl at $(smile_nb_dir) to score
   per-line GLMEdge + pooled totals against the pois-smile and dp/nb grids.
""")
