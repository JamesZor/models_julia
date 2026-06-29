#=
BACKTEST GRID — hierarchical-σ ISO market pillar (l09 HierIsoDoublePoissonModel).
Run AFTER r18 smoke passes (convergence confirmed). Fixes the market weight to the better iso weight and
sweeps the σ-hierarchy STRUCTURE, scoring each cell per-line with GLMEdge + LogLoss vs the Betfair close
(the proper-scoring "backtest" used across this stream — judge vs market, NOT grouped P/L; see
[[totals-compression-is-denoising]] / [[calibrate-centre-edge-in-tails]]).

4 cells, all at market_weight = ISO_MW:
  - iso_flat    : neither flag → global scalar σ (≈ src iso, the reference)
  - iso_perteam : + δ[team]      (per-team anchor tightness)
  - iso_perside : + ±δ_side      (home vs away anchor tightness)
  - iso_both    : both

>>> SET ISO_MW to whichever iso market_weight won the double-Poisson grid (r05/r15: iso_pois_mw50 vs mw100).
    Default 0.5 (consistent with the recurring "lower market weight optimal" finding). <<<

Ireland by default; flip SEGMENT to IrelandFirstDivision() (718, V/M 1.14) where dispersion/heterogeneity
is likelier to bite. STANDALONE — include l09 only.

Run after git pull + REPL restart:
    include("current_development/split_market_pillar/r19_grid_hier_iso.jl")
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using ThreadPinning

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Evaluation  = BayesianFootball.Evaluation
const Data        = BayesianFootball.Data

include("current_development/split_market_pillar/l09_hier_iso_poisson.jl")

# ==========================================
# 1. DATA
# ==========================================
SEGMENT = Data.Ireland()                       # flip to Data.IrelandFirstDivision() to test 718
seg_tag = lowercase(string(nameof(typeof(SEGMENT))))
println("[INFO] Loading $(seg_tag) DataStore...")
ds = Data.load_datastore_cached(SEGMENT)

save_dir = "./data/hier_iso_grid_$(seg_tag)/"
mkpath(save_dir)

# ==========================================
# 2. CONFIG  (= r15/r16 grid conventions)
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
feature_cfg_bayes = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))
dyn_cfg = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

samples        = 800
warmup         = 300
target_seasons  = ["2025", "2026"]
dynamics_col    = :match_biweek
chains         = 4
warmup_period   = 21
ISO_MW         = 0.5        # <<< better iso market weight from the dp grid — CONFIRM/ADJUST

_hiso(per_team, per_side) = HierIsoDoublePoissonModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    market_on              = true,
    market_weight          = ISO_MW,
    sigma_per_team         = per_team,
    sigma_per_side         = per_side,
)

specs = Tuple{String, Any}[
    ("iso_flat",    _hiso(false, false)),   # global scalar σ = reference
    ("iso_perteam", _hiso(true,  false)),
    ("iso_perside", _hiso(false, true)),
    ("iso_both",    _hiso(true,  true)),
]
println("[INFO] Hier-iso GRID ($seg_tag) @ ISO_MW=$ISO_MW: $(length(specs)) cells -> ", join(first.(specs), ", "))

# ==========================================
# 3. RUN
# ==========================================
all_results = Any[]
runs        = Tuple{String,Any,Any}[]
for (name, model) in specs
    println("\n", "#"^72, "\n# RUN: $name\n", "#"^72)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons  = target_seasons,
            history_seasons = 2,
            warmup_period   = warmup_period,
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
    end
end

# ==========================================
# 4. BACKTEST GRID — per-line GLMEdge + LogLoss vs Betfair close
# ==========================================
try
    odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
    ds1  = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)

    println("\n", "█"^72, "\n  HIER-ISO GRID @ ISO_MW=$ISO_MW  ($seg_tag)\n", "█"^72)
    println("\n", "="^60, "\n📊 GLM Edge (Betfair) — spread coef>0 & p<0.1 = edge beyond market\n", "="^60)
    Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.GLMEdge(), all_results, ds1), :glmedge)
    println("\n", "="^60, "\n📉 LogLoss (Betfair) — lower diff_ll = better than market\n", "="^60)
    Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds1), :logloss)
    println("\n", "="^60, "\n🎯 LPD (Betfair) — higher diff_lpd / elpd = better\n", "="^60)
    Evaluation.display_summary_metric(Evaluation.evaluate_experiments(Evaluation.LPD(), all_results, ds1), :lpd)
catch e
    @error "Eval phase failed (chains are saved)" exception=(e, catch_backtrace())
end

# ==========================================
# 4b. WHAT EACH CELL LEARNED — σ-hierarchy read (raw chains)
# ==========================================
_chain1(res) = res.training_results.items[1][1]
_has(res, s) = Symbol(s) in keys(_chain1(res))
function _pool(res, s)
    out = Float64[]
    for it in res.training_results.items
        ch = it[1]
        Symbol(s) in keys(ch) && append!(out, vec(Array(ch[Symbol(s)])))
    end
    return out
end
_count(res, stem) = count(i -> _has(res, "$stem[$i]"), 1:999)
_f(v) = (m = round(mean(v), digits=3), lo = round(quantile(v, 0.05), digits=3), hi = round(quantile(v, 0.95), digits=3))

println("\n", "█"^72, "\n  WHAT EACH CELL LEARNED\n", "█"^72)
for (name, model, res) in runs
    println("\n--- $name (per_team=$(model.sigma_per_team), per_side=$(model.sigma_per_side)) ---")
    if !_has(res, "log_σ_base"); println("  (no log_σ_base)"); continue; end
    g = _f(exp.(_pool(res, "log_σ_base")))
    println("  σ_base mean=$(g.m) 90%=[$(g.lo),$(g.hi)]")
    if model.sigma_per_team
        tt = _pool(res, "τ_team"); ft = _f(tt)
        nT = _count(res, "z_team"); zbar = [mean(_pool(res, "z_team[$t]")) for t in 1:nT]
        mult = exp.(mean(tt) .* zbar)
        println("  τ_team mean=$(ft.m) 90%=[$(ft.lo),$(ft.hi)]  P(τ>0.05)=$(round(mean(tt.>0.05),digits=2))  " *
                "team σ-mult [$(round(minimum(mult),digits=3)),$(round(maximum(mult),digits=3))]")
    end
    if model.sigma_per_side
        ds_ = _pool(res, "δ_side"); fd = _f(ds_)
        println("  δ_side mean=$(fd.m) 90%=[$(fd.lo),$(fd.hi)]")
    end
end

println("""

[READ] Verdict logic:
 • Compare GLMEdge spread-coef / LogLoss diff_ll across {iso_flat → perteam/perside/both}. If the hierarchy
   cells DON'T beat iso_flat, the extra σ structure isn't earning its keep on this league (keep flat).
 • Cross-check 4b: if τ_team sits near 0 / team σ-mult ≈ 1.0, the hierarchy collapsed to the global σ
   (same outcome as the smile-σ smoke on Ireland) → no surprise the scores don't move.
 • The hierarchy is likeliest to bite on a genuinely heterogeneous league — re-run with SEGMENT =
   IrelandFirstDivision() (718) before concluding.
""")
