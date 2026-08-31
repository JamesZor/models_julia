# ==============================================================================
# r45 — Single-fold smoke test: the two-arm (Gamma pxG + Poisson goals) joint model
# Scottish Lower (tiers 56/57) · BayesianFootball.jl Unified V2 stack
# ==============================================================================
#
# WHAT THIS IS
#   A shakedown, not an experiment. It fits five joint arms on ONE fold and asks
#   whether the new observation layer is safe to spend a 40-fold overnight grid on.
#
#   QUESTION. Does `JointGammaPoissonObservation` sample, converge, and produce the
#   two parameters it claims — with the proxy arm actually carrying evidence?
#
#   THE MODEL. One shared latent μ = exp(η) read by two likelihoods at once:
#       ARM 1   pxg ~ Gamma(ν, μ/ν)     where a BBC proxy-xG measurement exists (23/24+)
#       ARM 2   goals ~ Poisson(κ·μ)    on every match in the fold
#   ν is the proxy arm's precision; κ is the league finishing factor.
#
#   HELD FIXED. Interception, dynamics half-life, home advantage, splitter, sampler
#   and seed are identical across all five arms and across the Poisson control. The
#   ONLY differences are which covariate is attached and which observation is used.
#
#   THE GATES (all must pass before r46 is worth starting):
#     * R̂ < 1.05 and zero divergences on every arm.
#     * κ inside [0.60, 1.60]. pxG is already calibrated in goal units, so κ near 1
#       is the expected answer; far from 1 is either a real finishing effect or a
#       units bug in the shot-xG cell table, and only a human can tell which.
#     * ν inside [1.0, 12.0]. At the prior mean the Gamma arm gives a 1.5 xG
#       performance an sd near 0.75 goals, which is the right order of magnitude.
#     * λ = κ·μ on extraction. If λ and μ come back equal, the joint `_cb_rates`
#       method did not fire and the diagnostics are mirroring the wrong quantity.
#
# WHAT THIS IS NOT
#   Not a predictive comparison and not a staking study. One fold cannot rank five
#   models, and the leaderboard printed at the end is a smoke-test readout, not a
#   result. r46 is where the walk-forward grid runs; r22/r33 are where money enters.
#
#   It also does not claim the proxy arm is informative on every fold. BBC live text
#   starts in 23/24, so a fold whose history predates that has an empty or thin Gamma
#   arm. Section 6 measures that BEFORE sampling and says so.
#
# FILTRATION CONTRACT
#   The proxy arm is an OBSERVATION of the match it belongs to, not a covariate, so
#   it carries no point-in-time question of its own — a match may always see itself.
#   The one fitted object is the shot-xG cell table, which `MatchProxyXGFeature`
#   refits from the fold's `history_match_ids` when the builder supplies them. It
#   carries no team or player identity. The goals arm is unchanged from the baseline.
#
#   The proxy arm deliberately REFUSES the measurement ladder's goals rung: feeding
#   goals to the Gamma arm would hand it the same counts the Poisson arm reads and
#   double-count every goal. `MatchProxyXGFeature(fallback = :goals)` is an error.
#
# USAGE
#   julia --project -t 32
#   julia> include("current_development/scottish_lower/r45_smoke_joint_gamma_poisson.jl")
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

const R45_TOURNAMENT_GROUPS = [[56, 57]]      # Scottish League One + League Two, pooled
const R45_TARGET_SEASONS    = ["24/25"]
const R45_HISTORY_SEASONS   = 2
const R45_DYNAMICS_COL      = :match_biweek
const R45_HALF_LIFE_DAYS    = 180.0

# One fold. `end_dynamics = 1` with `stop_early` stops after the first time step.
const R45_END_DYNAMICS      = 1
const R45_STOP_EARLY        = true

const R45_SAMPLES           = 500
const R45_WARMUP            = 500
const R45_CHAINS            = 4
const R45_ACCEPT_RATE       = 0.65

# The proxy arm's feed. `:shots` reaches matches with a BBC scoreline page but no live
# text, which roughly doubles coverage on 23/24. `:goals` is refused by the feature.
const R45_PXG_FALLBACK      = :shots
const R45_PXG_CELL_K        = 25.0

# Gate bands. See the header for why each one is where it is.
const R45_MAX_RHAT          = 1.05
const R45_KAPPA_BAND        = (0.60, 1.60)
const R45_NU_BAND           = (1.0, 12.0)
const R45_MIN_DECAYED_MASK  = 0.10

const R45_SAVE_ROOT   = "/tmp/scottish_lower_joint_gamma_poisson_smoke"
const R45_BASELINE    = "m00_joint_baseline"

# %%
# ==============================================================================
# 3. Runtime and output directory
# ==============================================================================

mkpath(R45_SAVE_ROOT)

println("\n" * "="^100)
println(" r45 · TWO-ARM JOINT MODEL SMOKE TEST (Gamma proxy xG + Poisson goals)")
println("="^100)
println("  mode        : SMOKE — fold 1 only")
println("  target      : ", join(R45_TARGET_SEASONS, ", "), "  · history ", R45_HISTORY_SEASONS, " seasons")
println("  sampler     : NUTS ", R45_SAMPLES, " samples / ", R45_WARMUP, " warmup / ", R45_CHAINS, " chains")
println("  proxy feed  : MatchProxyXGFeature(fallback = :", R45_PXG_FALLBACK, ", k = ", R45_PXG_CELL_K, ")")
println("  threads     : ", Threads.nthreads())
println("  output      : ", R45_SAVE_ROOT)

# %%
# ==============================================================================
# 4. Data snapshot and temporal splits
# ==============================================================================

println("\n[1/7] Loading DataStore ...")
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)

@printf("  matches %d | lineups %d | incidents %d | bbc %d | bbc_events %d\n",
        nrow(ds.matches), nrow(ds.lineups), nrow(ds.incidents),
        nrow(ds.bbc), nrow(ds.bbc_events))

splitter = Data.GroupedCVConfig(
    tournament_groups = R45_TOURNAMENT_GROUPS,
    target_seasons    = R45_TARGET_SEASONS,
    history_seasons   = R45_HISTORY_SEASONS,
    dynamics_col      = R45_DYNAMICS_COL,
    warmup_period     = 0,
    end_dynamics      = R45_END_DYNAMICS,
    stop_early        = R45_STOP_EARLY,
)

boundaries = Data.create_id_boundaries(ds, splitter)
println("  folds: ", length(boundaries))
for (i, (boundary, meta)) in enumerate(boundaries)
    @printf("    fold %d — history %5d · target %5d · %s\n",
            i, length(boundary.history_match_ids), length(boundary.target_match_ids),
            string(meta))
end

# %%
# ==============================================================================
# 5. Model construction
# ==============================================================================

println("\n[2/7] Assembling the five joint arms and the single-arm control ...")

r45_proxy_feature = MatchProxyXGFeature(
    k        = R45_PXG_CELL_K,
    fallback = R45_PXG_FALLBACK,
)

# The work package's priors. ν is truncated well above zero because a Gamma shape at 0
# is a density with no mode; log κ is tight around 0 because pxG is already expressed
# in goal units and κ is a correction, not a free scale.
r45_observation = JointGammaPoissonObservation(
    feature         = r45_proxy_feature,
    shape_prior     = truncated(Normal(4.0, 1.5), 0.5, Inf),
    log_kappa_prior = Normal(0.0, 0.2),
)

r45_models = l45_joint_arms(
    half_life_days = R45_HALF_LIFE_DAYS,
    observation    = r45_observation,
)

# The control differs from m00_joint_baseline ONLY by having no Gamma arm. It is what
# makes "the joint model converged" into "the joint model was worth fitting".
push!(r45_models, ("m00_poisson_control",
                   l45_poisson_control(half_life_days = R45_HALF_LIFE_DAYS)))

for (name, model) in r45_models
    sites = join(string.(Models.PreGame.cb_covariate_names(model)), ", ")
    @printf("  %-28s observation %-30s covariates: %s\n",
            name, string(nameof(typeof(model.observation))), isempty(sites) ? "none" : sites)
end

# %%
# ==============================================================================
# 6. Feature construction and preflight gates
# ==============================================================================
#
# Nothing is sampled until the proxy arm's evidence has been looked at. An empty mask
# means the joint model IS the baseline carrying ν and log κ as unidentified
# parameters, and κ read off such a fold means nothing.

println("\n[3/7] Proxy-xG observation coverage by season ...")
l45_print_observation_coverage(l45_observation_coverage(ds, r45_proxy_feature))

println("\n[4/7] Per-fold proxy-arm preflight ...")
r45_preflight = l45_arm_preflight(ds, r45_models[1][2], splitter)
l45_print_arm_preflight(r45_preflight; min_decayed_share = R45_MIN_DECAYED_MASK)

# The feature sets and held-out frames the latent gate will reuse, built once from the
# same splitter the fits use so the gate cannot accidentally see a different fold.
r45_feature_sets = Features.create_features(boundaries, ds, r45_models[1][2], splitter)
r45_oos_fixtures = l45_fold_fixtures(ds, first(boundaries[1]))
@printf("  fold 1 held-out fixtures for the latent gate: %d\n", nrow(r45_oos_fixtures))

# %%
# ==============================================================================
# 7. Training
# ==============================================================================

sampler_config = QueuedNUTSConfig(
    n_samples   = R45_SAMPLES,
    n_warmup    = R45_WARMUP,
    n_chains    = R45_CHAINS,
    accept_rate = R45_ACCEPT_RATE,
)

println("\n[5/7] Fitting fold 1 ...")

r45_fits = Dict{String, Fit}()
r45_rows = NamedTuple[]

for (name, model) in r45_models
    println("\n" * "-"^90)
    println(" SMOKE FIT: $name")
    println("-"^90)

    fit_config = FitConfig(
        name      = name,
        model     = model,
        splitter  = splitter,
        sampler   = sampler_config,
        execution = QueuedExecution(),
        save_dir  = joinpath(R45_SAVE_ROOT, name),
    )

    started = time()
    fit = fit_model(fit_config, ds; quiet = false)
    elapsed = time() - started

    r45_fits[name] = fit
    push!(r45_rows, l45_summarise_fit(name, fit, ds, elapsed))
end

# %%
# ==============================================================================
# 8. Convergence and joint-arm gates
# ==============================================================================

println("\n[6/7] Gates ...")

r45_gates = []
for (name, model) in r45_models
    model.observation isa JointGammaPoissonObservation || continue
    fit = r45_fits[name]
    append!(r45_gates, l45_smoke_gates(name, fit;
                                       max_rhat   = R45_MAX_RHAT,
                                       kappa_band = R45_KAPPA_BAND,
                                       nu_band    = R45_NU_BAND))
    push!(r45_gates, l45_latent_gate(name, model, fit.folds[1].chain,
                                     first(r45_feature_sets[1]), r45_oos_fixtures))
end

r45_passed = l45_print_gates(r45_gates)

# %%
# ==============================================================================
# 9. Fold-1 readout
# ==============================================================================
#
# ONE FOLD CANNOT RANK FIVE MODELS. This table exists so a human can see that the
# numbers are the right shape — κ near 1, ν away from its prior mean where the mask is
# thick, log loss in the same neighbourhood as the control — not to pick a winner.

println("\n[7/7] Fold-1 readout (NOT a ranking) ...")
r45_table = l45_print_leaderboard(r45_rows; baseline = R45_BASELINE)

println()
if r45_passed
    println(" SMOKE TEST PASSED — r46_train_5models_2426_joint.jl is safe to start.")
else
    println(" SMOKE TEST FAILED — fix the failing gate(s) before starting the grid run.")
end
