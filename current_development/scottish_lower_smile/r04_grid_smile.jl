#=
r04 — GRID B (FAST-RANK REDESIGN 2026-07-13): supremacy_weight × smile_weight on the team
smile engine (ScottishLower), at the Grid-A winner hl365_hs2.

WHY the redesign: r01b depth probe — full spec (60 folds × 4 chains, depth 10) ≈ 25h/CELL
(smile geometry: ~127 leapfrogs/iter). Depth 6 halves tree cost and is PROVEN UNBIASED
(σ_smile/σ_sup/δ_gap match the depth-10 reference to 4 decimals) but mixes sluggishly
(fold-1 log_φ R-hat 1.077, ESS≈45 at 600 samples → samples=1200 doubles ESS). Depth 5 is
broken (R-hat 1.38). USER DECISION: depth-6 cells are for RANKING ONLY; the per-family
winner from r05 is re-run at depth 10 / full spec in r04b_winner_confirm.jl, where the
hard gate applies before graduation.

Cells (canonical naming; all at hl365_hs2; sw=0.4 column dropped — 0.4 vs 0.5 never a live axis):
  smile_pois_sup{40,70,100}_sw50   — depth 6, samples 1200  (ranking gate: R-hat ≤ 1.05 new params)
  smile_pois_sup{40,70,100}_sw0    — depth 10, samples 1200 (supremacy-only rung; loose geometry,
                                     hard gate — no smile pillar so no depth problem)
  iso_pois_mw100                   — depth 10 control (old pillar vs smile A/B on same base)
  none_pois ctl                    — depth 10 structural control, RE-RUN at THIS spec (the Grid-A
                                     cell pools 3 target seasons → not comparable to 2-season rows)

Settings: targets 24/25→25/26 (2 seasons → ~40 folds), GroupedCVConfig [56,57], match_biweek,
1200/300 × 3 chains. Budget @16 threads: 3×~6.6h (sw50) + 3×~2.8h (sw0) + ~2.5h (iso) + ~1h
(none) ≈ 30h ≈ 1.5 nights. (-t 32 shaves ~30%.)

Convergence gate per cell → r04_convergence.txt: reports BOTH %folds all-param R-hat≤1.01
(hard) and ≤1.05 (ranking). Depth-6 cells are expected to fail the hard line — that is priced
in; they must pass the ranking line. Controls must pass the hard line.

Run on the server (kaimon REPL) after git pull:
    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower_smile/r04_grid_smile.jl"))
=#

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using MCMCChains
using ThreadPinning
using Dates

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include(joinpath(ROOT, "current_development/scottish_lower_smile/l01_team_dp_league.jl"))

# ==========================================
# 0. GRID-A WINNER (r03, 2026-07-13)
# ==========================================
const BEST_HL = 365.0    # r03 winner: best family-pooled LogLoss on ALL of x12/btts/totals
const BEST_HS = 2        # r03 winner: hs3 adds nothing, hs1 truncates the decay

# ==========================================
# 1. DATA + GRID SPEC
# ==========================================
println("[INFO] Loading ScottishLower DataStore...")
ds = Data.load_datastore_cached(Data.ScottishLower())
save_dir = joinpath(ROOT, "data/scottish_smile_grid/")
mkpath(save_dir)

TARGETS = ["24/25", "25/26"]   # trimmed: 2 target seasons (~40 folds)
DYN_COL = :match_biweek
SAMPLES = 1200                 # doubled vs Grid A: recovers ESS at depth 6 (r01b: ESS≈45 @600)
WARMUP  = 300
CHAINS  = 3

DEPTH_SMILE = 6                # ranking-only cells (sw>0)
DEPTH_LOOSE = 10               # sw=0 / iso / none — loose geometry, hard gate

dyn_cfg = PreGame.TimeDecayDynamics(days_half_life = BEST_HL)
_tag = "hl$(Int(BEST_HL))_hs$(BEST_HS)"

# (name, model, max_depth)
specs = Tuple{String, Any, Int}[]
for sup in (0.4, 0.7, 1.0), sw in (0.0, 0.5)
    depth = sw > 0 ? DEPTH_SMILE : DEPTH_LOOSE
    push!(specs, ("smile_pois_sup$(Int(100sup))_sw$(Int(100sw))_$(_tag)",
                  TeamSmileDPGoalsModel(dynamics_config = dyn_cfg,
                                        supremacy_weight = sup, smile_weight = sw),
                  depth))
end
push!(specs, ("iso_pois_mw100_$(_tag)",
              TeamIsoDPGoalsModel(dynamics_config = dyn_cfg, market_weight = 1.0), DEPTH_LOOSE))
push!(specs, ("none_pois_$(_tag)_ctl", TeamDPGoalsModel(dynamics_config = dyn_cfg), DEPTH_LOOSE))

println("[INFO] Grid B: $(length(specs)) cells (hs=$BEST_HS, targets=$(TARGETS)) -> ",
        join([s[1] * " [d$(s[3])]" for s in specs], ", "))

# ==========================================
# 2. RUN — cheap depth-10 cells FIRST (fast signal), then the depth-6 smile cells
# ==========================================
sort!(specs, by = s -> s[3] == DEPTH_SMILE)   # loose cells first

function _fold_convergence(res)
    n = length(res.training_results.items); n_hard = 0; n_rank = 0; worst = 0.0
    for it in res.training_results.items
        er = DataFrame(MCMCChains.ess_rhat(it[1]))
        rcol = :rhat in propertynames(er) ? :rhat :
               first(filter(c -> occursin("rhat", lowercase(string(c))), propertynames(er)))
        vals = collect(skipmissing(replace(er[!, rcol], NaN => missing)))
        mr = isempty(vals) ? NaN : maximum(vals)
        isnan(mr) && continue
        worst = max(worst, mr)
        mr <= 1.01 && (n_hard += 1)
        mr <= 1.05 && (n_rank += 1)
    end
    return n, n_hard, n_rank, worst
end

gate_lines = String[]
t_start = time()
for (name, model, depth) in specs
    println("\n", "#"^72, "\n# CELL: $name  depth=$depth  (elapsed $(round((time()-t_start)/60, digits=1)) min)\n", "#"^72)
    try
        task = Experiments.create_experiment_task(
            ds, model, name, save_dir;
            target_seasons  = TARGETS,
            history_seasons = BEST_HS,
            warmup_period   = 0,   # match r02: include season-start folds (the operational regime)
            dynamics_col    = DYN_COL,
            samples         = SAMPLES,
            warmup          = WARMUP,
            chains          = CHAINS,
            use_queue       = true,
            max_depth       = depth,
        )
        res = Experiments.run_experiment(task)
        Experiments.save_experiment(res)

        n, n_hard, n_rank, worst = _fold_convergence(res)
        p_hard = n == 0 ? 0.0 : round(100n_hard / n, digits=1)
        p_rank = n == 0 ? 0.0 : round(100n_rank / n, digits=1)
        is_rank_cell = depth == DEPTH_SMILE
        ok = n > 0 && (is_rank_cell ? p_rank >= 95 : p_hard >= 95)
        gate = "$name [d$depth]: folds=$n  ≤1.01: $n_hard ($(p_hard)%)  ≤1.05: $n_rank ($(p_rank)%)  worst=$(round(worst, digits=4))" *
               (n == 0 ? "  ⚠ SILENT DROP — no items!" :
                ok ? (is_rank_cell ? "  ✅ (ranking gate)" : "  ✅ (hard gate)") : "  ⚠ BELOW GATE")
        println("[GATE] ", gate)
        push!(gate_lines, gate)
    catch e
        msg = "$name: FAILED ($(typeof(e)))"
        println("[GATE] ", msg)
        push!(gate_lines, msg)
        @error "cell failed: $name" exception=(e, catch_backtrace())
    end
end

open(joinpath(ROOT, "current_development/scottish_lower_smile/r04_convergence.txt"), "w") do io
    println(io, "r04 Grid B convergence gate — ", string(now()),
            "  (hl=$(BEST_HL), hs=$(BEST_HS), targets=$(TARGETS), $(SAMPLES)/$(WARMUP)×$(CHAINS))")
    println(io, "Gate: depth-10 cells HARD (≥95% folds ≤1.01); depth-6 smile cells RANKING (≥95% ≤1.05,")
    println(io, "r01b-justified: depth-6 posteriors unbiased vs depth-10, winner re-confirmed in r04b).")
    foreach(l -> println(io, l), gate_lines)
end

println("\n[INFO] Grid B complete in $(round((time()-t_start)/3600, digits=2)) h. ",
        "Gate written to r04_convergence.txt. Next: r05_eval_smile.jl, then r04b_winner_confirm.jl")
