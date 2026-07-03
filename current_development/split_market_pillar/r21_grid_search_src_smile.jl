#=
r21 — SMALL 2-D GRID over (supremacy_weight × smile_weight) for the GRADUATED SRC smile engine.

Motivation
----------
r10 only swept `smile_weight` with `supremacy_weight` PINNED at 1.0 (keeper = `li_smile50`,
sup=1.0 / smile=0.5). But the old ISOTROPIC market pillar lived around market_weight≈0.4
([[betfair-vs-bet365-market-anchor]], r19 ISO_MW=0.4), so a hard sup=1.0 anchor is probably
over-tight for the supremacy axis too. This runner adds the missing axis: it crosses a light-ish
supremacy anchor (≈0.4, echoing the old market weight) up to the keeper's 1.0, against the
smile_weight band that r10/RESULTS flagged as best (0.4–0.5, "some smile, not much"; §5/§6.2).

Built FROM SRC ONLY — like r20, this does NOT include any l0X loader to construct the grid
cells. It uses `PreGame.DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel` directly. The
l02/l03 loaders are included ONLY so the previously-saved r05 (`dp_*`) and r10 (`li_*`) grids can
be deserialized for a side-by-side comparison (their JLD2 payloads need the loader structs). The
src cells are qualified `Features.MarketSmileFeature` throughout, so they never touch the loader's
same-named struct.

Grid (Kmax=4, everything else IDENTICAL to r10 for comparability):
  supremacy_weight ∈ {0.4, 0.7, 1.0}  ×  smile_weight ∈ {0.4, 0.5}   → 6 cells
    src_sup40_sw40 … src_sup100_sw50
  (src_sup100_sw50 reproduces r10's `li_smile50` keeper from src → cross-check.)

Expectation (be honest, per RESULTS §5): supremacy drives 1X2, smile drives totals/BTTS. Loosening
the supremacy anchor (sup<1) should help the 1X2 P/L that hard-anchoring wrecks, at little cost to
the BTTS/totals edge (which the smile owns). Judge PER MARKET FAMILY (GLMEdge / LogLoss / LPD /
hurdle_G), not a single blended number. LPD (full-posterior proper score, as r19) is added so a cell
is scored over its whole PPD, not just the collapsed mean.

Run after git pull + REPL restart (structs change → Revise won't pick them up):
    include("current_development/split_market_pillar/r21_grid_search_src_smile.jl")
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

# Loaders included ONLY to deserialize the saved dp_*/li_* comparison grids (not to build src cells).
# Harmless "redefinition of constant" warnings (shared SUP_PRIOR etc.) are expected.
include("current_development/split_market_pillar/l02_split_market_poisson.jl")
include("current_development/split_market_pillar/l03_local_intensity_poisson.jl")

# ==========================================
# 1. DATA  (train pillar = SofaScore ds.odds, as r05/r10; eval CLV vs Betfair ds1 — no leakage)
# ==========================================
println("[INFO] Loading Ireland DataStore...")
ds = Data.load_datastore_cached(Data.Ireland())

src_dir   = "./data/double_poisson_smile_src_grid/"   # NEW: this runner's src cells
smile_dir = "./data/double_poisson_smile_grid/"        # r10 li_* grid
r05_dir   = "./data/double_poisson_market_grid/"       # r05 baseline/old/sup+level grid
mkpath(src_dir)

# ==========================================
# 2. SHARED CONFIG (IDENTICAL to r05/r10 for comparability)
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

# SRC engine builder — keeper defaults except the two grid knobs.
_src_smile(sup, sw) = PreGame.DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    smile_feature          = Features.MarketSmileFeature(Kmax = KMAX),
    market_on              = true,
    supremacy_weight       = sup,
    smile_weight           = sw,
)

_tag(x) = string(Int(round(x * 100)))   # 0.4 -> "40", 1.0 -> "100"
sup_grid = [0.4, 0.7, 1.0]
sw_grid  = [0.4, 0.5]

specs = Tuple{String, Any}[]
for sup in sup_grid, sw in sw_grid
    push!(specs, ("src_sup$(_tag(sup))_sw$(_tag(sw))", _src_smile(sup, sw)))
end
println("[INFO] SRC smile grid: $(length(specs)) cells (Kmax=$KMAX) -> ", join(first.(specs), ", "))

# ==========================================
# 3. PHASE 1 — RUN + SAVE the src grid (robust; one failure doesn't sink the rest)
# ==========================================
src_results = Any[]
failures = String[]
t0 = time()
for (name, model) in specs
    println("\n", "#"^70, "\n# RUN: $name\n", "#"^70)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, src_dir;
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
        push!(src_results, res)
    catch e
        @error "FAILED: $name" exception=(e, catch_backtrace())
        push!(failures, name)
    end
end
println("\n[INFO] Phase 1 done in $(round((time()-t0)/60, digits=1)) min. " *
        "OK: $(length(src_results))/$(length(specs)). Failures: $(isempty(failures) ? "none" : join(failures, ", "))")

# ==========================================
# 4. PHASE 2 — EVAL src grid + r10 li_* grid + r05 dp_* grid together (CLV vs Betfair)
# ==========================================
_load_dir(d) = isdir(d) ? Experiments.load_experiments(Experiments.list_experiments(d, data_dir="")) : Any[]

src_results = _load_dir(src_dir)
li_results  = _load_dir(smile_dir)
r05_results = _load_dir(r05_dir)
isdir(smile_dir) || @warn "r10 smile grid dir not found ($smile_dir) — skipping li_* comparison."
isdir(r05_dir)   || @warn "r05 grid dir not found ($r05_dir) — skipping dp_* comparison."
println("[INFO] cells -> src: $(length(src_results)), li_*: $(length(li_results)), dp_*: $(length(r05_results))")

all_results = convert(Vector{BayesianFootball.Experiments.ExperimentResults},
                      vcat(src_results, li_results, r05_results))

odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds1  = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)

println("\n", "="^60, "\n📈 GLM Edge (Betfair) — src grid vs li_*/dp_*\n", "="^60)
Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.GLMEdge(), all_results, ds1), :glmedge)

println("\n", "="^60, "\n📉 LogLoss (Betfair) — src grid vs li_*/dp_*\n", "="^60)
Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds1), :logloss)

# LPD = full-PPD log predictive density (scored over the WHOLE posterior sample vector, not the
# collapsed point prob). diff_lpd > 0 ⇒ model beats the market fair-odds baseline. Same interface as
# r19's hier-iso grid (which ranked cells on LPD diff). Higher = better (opposite sign to LogLoss diff).
println("\n", "="^60, "\n📊 LPD (Betfair, full-posterior) — src grid vs li_*/dp_*\n", "="^60)
Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.LPD(), all_results, ds1), :lpd)

println("\n", "="^60, "\n💰 Backtest (BayesianKelly) — src grid vs li_*/dp_*\n", "="^60)
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

println("\n[INFO] r21 src smile grid complete.")
println("       Read PER MARKET FAMILY: supremacy_weight ↓ should recover 1X2 (home/away/draw) P/L that")
println("       sup=1.0 wrecks; smile_weight owns totals/BTTS. Cross-check: src_sup100_sw50 ≈ r10 li_smile50.")
