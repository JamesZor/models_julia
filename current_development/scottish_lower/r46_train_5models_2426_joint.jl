# ==============================================================================
# r46 — Two-arm joint model: 40-fold walk-forward grid, seasons 24/25 + 25/26
# Scottish Lower (tiers 56/57) · BayesianFootball.jl Unified V2 stack
# ==============================================================================
#
# WHAT THIS IS
#   The overnight grid. Five joint arms and one single-arm control, fitted on genuine
#   walk-forward folds across two seasons and scored out of sample.
#
#   QUESTION. Does adding a second likelihood — `pxg ~ Gamma(ν, μ/ν)` on the matches
#   that have BBC proxy xG — improve genuine walk-forward log loss over the same model
#   fitted on goals alone?
#
#   CONTROL.       m00_poisson_control, the identical spine with no Gamma arm. This is
#                  the comparison the whole work package rests on: it isolates the
#                  second likelihood from every other difference.
#   REFERENCE.     m00_joint_baseline, the joint model with no covariates.
#   HELD FIXED.    Interception, dynamics half-life, home advantage, splitter, sampler,
#                  seed and proxy feed are identical across every arm. The only
#                  differences are the observation layer and the covariate list.
#   DECISION RULE. An arm must (a) converge on every fold, and (b) beat
#                  m00_poisson_control on fold-averaged out-of-sample log loss. A joint
#                  arm that converges and does not beat the control has not earned its
#                  two extra parameters and does not graduate to `src`.
#
# WHAT THIS IS NOT
#   Not a staking or allocation study. No bankroll, no Kelly, no Betfair book. An
#   improvement in log loss here is a necessary condition for a betting edge, never
#   evidence of one — r22 and r33 are where money enters. The portfolio section at the
#   end of r20 is deliberately NOT reproduced here.
#
#   It also does not claim the proxy arm is informative on every fold. BBC live text
#   starts in 23/24; folds whose history predates that carry a thin Gamma arm and the
#   preflight in section 6 says so per fold. Read that table before the leaderboard.
#
# FILTRATION CONTRACT
#   The proxy arm is an OBSERVATION of the match it belongs to, so it raises no
#   point-in-time question — a match may always see itself. The one fitted object is
#   the shot-xG cell table, refitted per fold from `history_match_ids` and carrying no
#   team or player identity. Covariates keep their own existing contracts.
#
#   `MatchProxyXGFeature` REFUSES the ladder's goals rung. Feeding goals to the Gamma
#   arm would hand it the counts the Poisson arm already reads and double-count every
#   goal; the feature errors rather than allowing it.
#
#   Required invariant, unchanged from the rest of the grid family:
#       training kickoff < prediction cutoff <= held-out kickoff
#
# PERSISTENCE CAVEAT
#   Each arm writes its own `Fit` under R46_SAVE_ROOT/<arm>/. `save_fit` is atomic and
#   the runner is restart-safe at arm granularity: re-running skips nothing, so kill it
#   between arms rather than mid-arm if you need to stop.
#
# USAGE — mcm-beast, overnight
#   julia --project -t 32
#   julia> include("current_development/scottish_lower/r46_train_5models_2426_joint.jl")
#
#   Run r45_smoke_joint_gamma_poisson.jl FIRST. Its gates exist to stop this file from
#   spending a night discovering something one fold would have shown in five minutes.
#
#   Environment overrides:
#     R46_SAMPLES, R46_WARMUP, R46_CHAINS   sampler budget
#     R46_SAVE_ROOT                          output directory
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Distributions
using LinearAlgebra
using MCMCChains
using Printf
using Statistics
using ThreadPinning

include(joinpath(@__DIR__, "l45_joint_gamma_poisson.jl"))

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================

const R46_TOURNAMENT_GROUPS = [[56, 57]]      # Scottish League One + League Two, pooled
const R46_TARGET_SEASONS    = ["24/25", "25/26"]
const R46_HISTORY_SEASONS   = 2
const R46_DYNAMICS_COL      = :match_biweek
const R46_HALF_LIFE_DAYS    = 180.0

# The full walk-forward: every biweek of both target seasons.
const R46_END_DYNAMICS      = 0
const R46_STOP_EARLY        = true

const R46_SAMPLES     = parse(Int, get(ENV, "R46_SAMPLES", "800"))
const R46_WARMUP      = parse(Int, get(ENV, "R46_WARMUP",  "800"))
const R46_CHAINS      = parse(Int, get(ENV, "R46_CHAINS",  "4"))
const R46_ACCEPT_RATE = 0.65
const R46_QUEUE_TASKS = Threads.nthreads()

# The proxy arm's feed, identical to r45's so the smoke gates transfer.
const R46_PXG_FALLBACK = :shots
const R46_PXG_CELL_K   = 25.0

const R46_MIN_DECAYED_MASK = 0.10

const R46_SAVE_ROOT = get(ENV, "R46_SAVE_ROOT", "./data/scottish_lower_2426_joint")
const R46_BASELINE  = "m00_poisson_control"

# %%
# ==============================================================================
# 3. Runtime and output directory
# ==============================================================================

mkpath(R46_SAVE_ROOT)

println("\n" * "="^110)
println(" r46 · TWO-ARM JOINT MODEL WALK-FORWARD GRID (24/25 + 25/26)")
println("="^110)
println("  mode        : FULL WALK-FORWARD")
println("  target      : ", join(R46_TARGET_SEASONS, ", "), "  · history ", R46_HISTORY_SEASONS, " seasons")
println("  sampler     : NUTS ", R46_SAMPLES, " samples / ", R46_WARMUP, " warmup / ", R46_CHAINS, " chains")
println("  execution   : QueuedExecution(", R46_QUEUE_TASKS, " tasks)")
println("  proxy feed  : MatchProxyXGFeature(fallback = :", R46_PXG_FALLBACK, ", k = ", R46_PXG_CELL_K, ")")
println("  threads     : ", Threads.nthreads())
println("  output      : ", R46_SAVE_ROOT)
println("  started     : ", Dates.now())

# %%
# ==============================================================================
# 4. Data snapshot and temporal splits
# ==============================================================================

println("\n[1/7] Loading DataStore ...")
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)

@printf("  matches %d | lineups %d | odds %d | bbc %d | bbc_events %d\n",
        nrow(ds.matches), nrow(ds.lineups), nrow(ds.odds),
        nrow(ds.bbc), nrow(ds.bbc_events))

splitter = Data.GroupedCVConfig(
    tournament_groups = R46_TOURNAMENT_GROUPS,
    target_seasons    = R46_TARGET_SEASONS,
    history_seasons   = R46_HISTORY_SEASONS,
    dynamics_col      = R46_DYNAMICS_COL,
    warmup_period     = 0,
    end_dynamics      = R46_END_DYNAMICS,
    stop_early        = R46_STOP_EARLY,
)

boundaries = Data.create_id_boundaries(ds, splitter)
println("  walk-forward folds: ", length(boundaries))
@printf("  fitted matches (fold 1 .. fold %d): %d .. %d\n",
        length(boundaries),
        length(first(boundaries[1]).history_match_ids),
        length(first(boundaries[end]).history_match_ids))
@printf("  held-out matches, summed over folds: %d\n",
        sum(length(first(b).target_match_ids) for b in boundaries))

# %%
# ==============================================================================
# 5. Model construction
# ==============================================================================

println("\n[2/7] Assembling the five joint arms and the single-arm control ...")

r46_proxy_feature = MatchProxyXGFeature(
    k        = R46_PXG_CELL_K,
    fallback = R46_PXG_FALLBACK,
)

r46_observation = JointGammaPoissonObservation(
    feature         = r46_proxy_feature,
    shape_prior     = truncated(Normal(4.0, 1.5), 0.5, Inf),
    log_kappa_prior = Normal(0.0, 0.2),
)

r46_models = l45_joint_arms(
    half_life_days = R46_HALF_LIFE_DAYS,
    observation    = r46_observation,
)

# The decision comparison. Identical spine, no Gamma arm.
push!(r46_models, (R46_BASELINE,
                   l45_poisson_control(half_life_days = R46_HALF_LIFE_DAYS)))

for (name, model) in r46_models
    sites = join(string.(Models.PreGame.cb_covariate_names(model)), ", ")
    @printf("  %-28s observation %-30s covariates: %s\n",
            name, string(nameof(typeof(model.observation))), isempty(sites) ? "none" : sites)
end

# %%
# ==============================================================================
# 6. Feature construction and preflight gates
# ==============================================================================

println("\n[3/7] Proxy-xG observation coverage by season ...")
l45_print_observation_coverage(l45_observation_coverage(ds, r46_proxy_feature))

println("\n[4/7] Per-fold proxy-arm preflight (all $(length(boundaries)) folds) ...")
r46_preflight = l45_arm_preflight(ds, r46_models[1][2], splitter)
l45_print_arm_preflight(r46_preflight; min_decayed_share = R46_MIN_DECAYED_MASK)

# %%
# ==============================================================================
# 7. Training
# ==============================================================================

sampler_config = QueuedNUTSConfig(
    n_samples   = R46_SAMPLES,
    n_warmup    = R46_WARMUP,
    n_chains    = R46_CHAINS,
    accept_rate = R46_ACCEPT_RATE,
)

println("\n[5/7] Fitting the walk-forward grid ...")

r46_fits = Dict{String, Fit}()
r46_rows = NamedTuple[]

for (name, model) in r46_models
    println("\n" * "="^90)
    println(" GRID: $name   (", Dates.now(), ")")
    println("="^90)

    fit_config = FitConfig(
        name      = name,
        model     = model,
        splitter  = splitter,
        sampler   = sampler_config,
        execution = QueuedExecution(max_concurrent_tasks = R46_QUEUE_TASKS),
        save_dir  = joinpath(R46_SAVE_ROOT, name),
    )

    started = time()
    fit = fit_model(fit_config, ds; quiet = false)
    elapsed = time() - started

    r46_fits[name] = fit
    saved = save_fit(fit)
    println("  saved: ", saved)

    push!(r46_rows, l45_summarise_fit(name, fit, ds, elapsed))
end

# %%
# ==============================================================================
# 8. Convergence diagnostics
# ==============================================================================

println("\n[6/7] Convergence across all folds ...")
for row in r46_rows
    @printf("  %-28s  passed=%-5s  max R̂ %s  min ESS %s  divergences %d\n",
            row.name, string(row.passed),
            isnan(row.max_rhat) ? "—" : @sprintf("%.4f", row.max_rhat),
            isnan(row.min_ess) ? "—" : string(Int(round(row.min_ess))),
            row.divergences)
end

# %%
# ==============================================================================
# 9. Final report
# ==============================================================================
#
# The decision quantity is ΔLogLoss against m00_poisson_control. A joint arm that does
# not beat it has not earned the second likelihood, however well it converged.

println("\n[7/7] Walk-forward leaderboard ...")
r46_table = l45_print_leaderboard(r46_rows; baseline = R46_BASELINE)

println("\n  κ (finishing factor) by arm — the joint model's own readout:")
for row in r46_rows
    isnan(row.kappa) && continue
    @printf("    %-28s κ = %.4f   ν = %.3f\n", row.name, row.kappa, row.nu)
end

println("\n  finished: ", Dates.now())
println("  fits written under: ", R46_SAVE_ROOT)
