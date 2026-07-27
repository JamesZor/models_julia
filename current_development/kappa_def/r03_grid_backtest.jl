#=
FULL-CV BACKTEST GRID — κ-structure cells vs the src control (mirrors r15's structure).

Question: do the log-additive κ reparameterizations (net / attdef) buy anything over the
battle-tested src attack-only control on a FULL cross-validated run — judged vs the Betfair
close (GLMEdge / LogLoss / LPD) + the BayesianKelly backtest per market selection?
Plus the stream-specific read: POOLED τ posteriors across all CV splits — each split is a
near-independent look at the population spread, so pooling ~29 splits is the biggest
within-league power gain available for the "is τ_def > 0" question (single-split r01 was
uninformative: posterior ≈ prior).

Cells (all market OFF — kappa_def stream convention; see NOTES.md):
  - kd_none_src : src DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel
                  (the production attack-only κ; converged control — the l01 :attack_only
                  twin is RETIRED after the r01 metastability, see EXPERIMENTS.md)
  - kd_net      : l01 KappaDefDoublePoissonModel(:net)     — κ0 + τ·(δ[home]−δ[away])
  - kd_attdef   : l01 KappaDefDoublePoissonModel(:attdef)  — κ0 + τ_att·z_att[h] − τ_def·z_def[a]

GATES before trusting any verdict here: r00 persistence EDA (is defensive persistence real?)
and r02 (718 shakedown). If r00 fails, this grid only measures noise-fitting.

LEAGUE: Ireland-79 by default — directly comparable to the saved dp grid (r05 dp_nomarket ≈
kd_none_src, same configs) and the r15 nb row. Flip SEGMENT to IrelandFirstDivision() for the
dispersion-active league (the more interesting test; no dp grid to compare there yet).
Saves keyed by segment, no collisions.

Run after git pull + REPL restart:
    include("current_development/kappa_def/r03_grid_backtest.jl")
=#

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
const Signals     = BayesianFootball.Signals

include("current_development/kappa_def/l01_kappa_def_models.jl")

# ==========================================
# 1. DATA
# ==========================================
SEGMENT = Data.Ireland()                 # ← flip to Data.IrelandFirstDivision() for 718
seg_tag = lowercase(string(nameof(typeof(SEGMENT))))
println("[INFO] Loading $(seg_tag) DataStore...")
ds = Data.load_datastore_cached(SEGMENT)

save_dir = "./data/kappa_def_grid_$(seg_tag)/"
mkpath(save_dir)

# ==========================================
# 2. SHARED CONFIG (= r05/r15 grid conventions for comparability)
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
feature_cfg_bayes = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

samples        = 800
warmup         = 300
chains         = 4
target_seasons = ["2025", "2026"]
dynamics_col   = :match_biweek

# ==========================================
# 3. MODEL SPECS
# ==========================================
_src_control() = PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
)

_kd(mode) = KappaDefDoublePoissonModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,          # unused outside :attack_only
    player_ratings_feature = feature_cfg_bayes,
    kappa_mode             = mode,
)

specs = Tuple{String, Any}[
    ("kd_none_src", _src_control()),
    ("kd_net",      _kd(:net)),
    ("kd_attdef",   _kd(:attdef)),
    # ("kd_attack_only", _kd(:attack_only)),   # RETIRED: metastable (r01, R-hat 1.53) — src control covers it
]
println("[INFO] kappa_def grid ($seg_tag): $(length(specs)) cells -> ", join(first.(specs), ", "))

# ==========================================
# 4. PHASE 1 — RUN + SAVE (robust; each persisted immediately)
# ==========================================
all_results = Any[]
runs        = Tuple{String, Any, Any}[]
failures    = String[]
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
        push!(runs, (name, model, res))
    catch e
        @error "FAILED: $name" exception=(e, catch_backtrace())
        push!(failures, name)
    end
end
println("\n[INFO] Phase 1 done in $(round((time()-t0)/60, digits=1)) min. " *
        "OK: $(length(all_results))/$(length(specs)). Failures: $(isempty(failures) ? "none" : join(failures, ", "))")

# Re-eval from disk without re-running (uncomment; NB include l01 first so the structs deserialize):
# saved = Experiments.list_experiments(save_dir, data_dir="")
# all_results = [Experiments.load_experiment(saved, i) for i in 1:length(saved)]

# ==========================================
# 5. PHASE 2 — EVAL vs Betfair close (identical frame to r05/r15/r19)
# ==========================================
odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds1 = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)

println("\n", "="^60, "\n📊 GLM Edge (Betfair) — spread coef>0 & p<0.1 = edge beyond market\n", "="^60)
Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.GLMEdge(), all_results, ds1), :glmedge)

println("\n", "="^60, "\n📉 LogLoss (Betfair) — lower diff_ll = better than market\n", "="^60)
Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds1), :logloss)

println("\n", "="^60, "\n🎯 LPD (Betfair) — higher diff_lpd / elpd = better\n", "="^60)
Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.LPD(), all_results, ds1), :lpd)

# ==========================================
# 6. PHASE 3 — BACKTEST (BayesianKelly) per market selection
# ==========================================
println("\n", "="^60, "\n💰 Backtest (BayesianKelly) — kappa_def grid\n", "="^60)
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

# ==========================================
# 7. PHASE 4 — POOLED τ ACROSS SPLITS (the power read r01 couldn't give)
# ==========================================
println("\n", "█"^70, "\n  POOLED τ POSTERIORS ACROSS CV SPLITS ($seg_tag)\n", "█"^70)
println("prior reference: half-N(0, 0.1) → mean 0.0798, P(τ>0.05)=0.617, P(τ>0.03)=0.764\n")
function _pool_param(res, s)
    out = Float64[]
    for it in res.training_results.items
        ch = it[1]
        Symbol(s) in keys(ch) && append!(out, vec(Array(ch[Symbol(s)])))
    end
    return out
end
for (name, model, res) in runs
    model isa KappaDefDoublePoissonModel || continue
    syms = model.kappa_mode === :net ? ["τ_net"] :
           model.kappa_mode === :attdef ? ["τ_att", "τ_def"] : String[]
    n_items = length(res.training_results.items)
    for s in syms
        t = _pool_param(res, s)
        isempty(t) && (println("$name  $s: (not found in chains)"); continue)
        println(rpad(name, 12), rpad(s, 8),
                " splits=", rpad(n_items, 4),
                " mean=", rpad(round(mean(t), digits=4), 8),
                " 90%=[", round(quantile(t, 0.05), digits=4), ",", round(quantile(t, 0.95), digits=4), "]",
                "  P(τ>0.05)=", round(mean(t .> 0.05), digits=3),
                "  P(τ>0.03)=", round(mean(t .> 0.03), digits=3))
    end
end

println("""

[READ] Verdict logic:
 • Scores (Phases 2–3): compare kd_net / kd_attdef against kd_none_src per market family —
   the κ structure should move BTTS and tails if anywhere; 1X2 movement is likely bias,
   not edge (r13 precedent). Judge per-line, not grouped P/L.
 • Pooled τ (Phase 4): pooling ~29 splits is the real test of τ_def > 0 within one league.
   Pulled clearly below prior (P(τ>0.05) ≪ 0.62) ⇒ hierarchy learned "no structure" — park.
   Still ≈ prior ⇒ even full-CV is underpowered here ⇒ the pooled cross-league τ design is
   the only remaining within-model route (or trust r00's un-decayed EDA).
   NB: splits overlap in training data (rolling folds), so pooled draws are correlated —
   treat the pooled CI as optimistic, not exact.
 • Cross-reference: on Ireland-79, kd_none_src ≈ the saved dp_nomarket cell (r05 grid) —
   sanity-check GLMEdge/LogLoss roughly match before reading anything else.
 • Record verdicts in EXPERIMENTS.md + NOTES.md findings log.
""")
