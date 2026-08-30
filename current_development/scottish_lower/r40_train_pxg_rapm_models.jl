# ==============================================================================
# r40 — Orthogonal benchmark: wealth, tuned pxG, bench depth, and RAPM control
# Scottish Lower (tiers 56/57) · BayesianFootball.jl Unified V2 stack
# ==============================================================================
#
# WHAT THIS IS
#   A convergence and predictive-fit experiment over six composable count models,
#   fitted on genuine walk-forward folds and scored out of sample.
#
#   QUESTION. Which independently useful squad and chance-quality signals survive
#   beside dynamic team ratings, and does RAPM still add anything in the all-in control?
#
#   CONTROL.       m00_baseline, the clean team-level arm.
#   REFERENCE.     m05_prod_wealth, the covariate that currently earns its keep.
#   HELD FIXED.    Interception, dynamics half-life, home advantage, observation
#                  density, splitter, sampler and seed are identical across all six
#                  arms. The ONLY difference is which covariate is attached.
#   DECISION RULE. An arm must (a) converge, and (b) beat m00 on fold-averaged
#                  out-of-sample log loss. An arm that converges and does not beat
#                  m00 has not earned its parameter and does not graduate.
#
# WHAT THIS IS NOT
#   Not a staking or allocation study — no bankroll, no Kelly, no Betfair book. An
#   improvement in log loss here is a necessary condition for a betting edge, never
#   evidence of one. r22 is where money enters.
#
#   It also does not claim the pxG covariate measures chance quality on every fixture:
#   BBC live text only starts in 23/24, and older matches fall back down the ladder to
#   shot counts and then to goals. Section 5 prints that mix; read it before reading
#   the leaderboard.
#
# FILTRATION CONTRACT
#   Both new features are point-in-time by construction:
#     * pxG form is built from matches that kicked off STRICTLY EARLIER, and same-day
#       fixtures cannot see each other.
#     * The RAPM ridge is fit on the fold's FROZEN HISTORY BLOCK (`fit_on = :history`),
#       and for the `:y_xg` target the shot-xG cell table is refitted from those same
#       permitted matches.
#   Ratings are then applied to future teamsheets, which is the pre-match rating being
#   tested, not a leak. See src/features/pxg.jl and src/features/pxg_rapm.jl.
#
# USAGE
#   julia --project -t 32
#   julia> include("current_development/scottish_lower/r40_train_pxg_rapm_models.jl")
#
#   Set R40_SMOKE=1 for a fast single-fold shakedown before committing a grid run.
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

include(joinpath(@__DIR__, "l40_pxg_rapm_bench.jl"))

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================

const R40_SMOKE = get(ENV, "R40_SMOKE", "0") == "1"

const R40_TOURNAMENT_GROUPS = [[56, 57]]     # Scottish League One + League Two, pooled
const R40_TARGET_SEASONS    = ["24/25"]
const R40_HISTORY_SEASONS   = 2
const R40_DYNAMICS_COL      = :match_biweek
const R40_HALF_LIFE_DAYS    = 180.0

# A smoke run stops after the first fold; the full run walks the season forward.
const R40_END_DYNAMICS      = R40_SMOKE ? 1 : 0
const R40_STOP_EARLY        = R40_SMOKE

const R40_SAMPLES           = R40_SMOKE ? 500 : 1_000
const R40_WARMUP            = R40_SMOKE ? 500 : 1_000
const R40_CHAINS            = 4
const R40_ACCEPT_RATE       = 0.65

const R40_SAVE_ROOT = joinpath("/tmp",
    R40_SMOKE ? "scottish_lower_pxg_rapm_smoke" : "scottish_lower_pxg_rapm")
const R40_BASELINE_NAME = "m00_baseline"

# %%
# ==============================================================================
# 3. Runtime and output directory
# ==============================================================================

mkpath(R40_SAVE_ROOT)

println("\n" * "="^100)
println(" r40 · WEALTH, TUNED pxG, BENCH DEPTH, AND RAPM CONTROL")
println("="^100)
println("  mode        : ", R40_SMOKE ? "SMOKE (fold 1 only)" : "FULL WALK-FORWARD")
println("  target      : ", join(R40_TARGET_SEASONS, ", "), "  · history ", R40_HISTORY_SEASONS, " seasons")
println("  sampler     : NUTS ", R40_SAMPLES, " samples / ", R40_WARMUP, " warmup / ", R40_CHAINS, " chains")
println("  threads     : ", Threads.nthreads())
println("  output      : ", R40_SAVE_ROOT)

# %%
# ==============================================================================
# 4. Data snapshot and temporal splits
# ==============================================================================

println("\n[1/6] Loading DataStore ...")
ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)

@printf("  matches %d | lineups %d | incidents %d | bbc %d | bbc_events %d\n",
        nrow(ds.matches), nrow(ds.lineups), nrow(ds.incidents),
        nrow(ds.bbc), nrow(ds.bbc_events))

# The splitter defines the filtration. Feature construction (section 6) is a strict
# consumer of it; nothing below reaches around it to touch the target block.
splitter = Data.GroupedCVConfig(
    tournament_groups = R40_TOURNAMENT_GROUPS,
    target_seasons    = R40_TARGET_SEASONS,
    history_seasons   = R40_HISTORY_SEASONS,
    dynamics_col      = R40_DYNAMICS_COL,
    warmup_period     = 0,
    end_dynamics      = R40_END_DYNAMICS,
    stop_early        = R40_STOP_EARLY,
)

# %%
# ==============================================================================
# 5. Model construction
# ==============================================================================
#
# One structural spine, six orthogonal covariate configurations. The spine is written out in
# full on every arm rather than factored into a helper, because the comparability
# claim of this experiment IS that these lines are identical.

println("\n[2/6] Assembling the six arms ...")

# Tuned team-form kernel from the held-out feature-discovery gauntlet.
r40_pxg_feature = PxGFeature(
    decay = :exponential,
    half_life_matches = 16.0,
    prior_weight = 3.0,
    min_matches = 3,
    fallback = :goals,
    scale = 1.0,
)

# The pxG-APM feature. `:y_xg` is the least team-loaded of the four plus-minus targets
# (club R² 0.212 against 0.389 for shots), which matters here precisely because the
# engine already carries team strength in dyn.α/dyn.β — a covariate that re-derives it
# is fighting its own model. λ = 200 is that target's tuned cell.
r40_rapm_feature = PxGRapmFeature(
    target = :y_xg,
    lambda = 200.0,
    w_sim = 0.0,                 # plain ridge; w_sim > 0 buys Brier by becoming team-like
    half_life_days = 730.0,
    fit_on = :history,           # Gate 2. Do not relax this for a scored run.
    shrink_segments = 20.0,
    min_rated_per_side = 3,
    scale = nothing,             # standardise on the permitted matches' own spread
)

m00 = CountModelBuilder(:m00_baseline) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = R40_HALF_LIFE_DAYS)) |>
    add(GlobalHomeAdvantage()) |>
    add(PoissonObservation()) |>
    build

m05 = CountModelBuilder(:m05_prod_wealth) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = R40_HALF_LIFE_DAYS)) |>
    add(GlobalHomeAdvantage()) |>
    add(ProductionWealthCovariate(
        feature = ProductionWealthFeature(curve = RichardsSigmoid(23.0, 0.80, 2.0)),
        prior   = truncated(Normal(0.10, 0.05), lower = 0.0),
    )) |>
    add(PoissonObservation()) |>
    build

m06 = CountModelBuilder(:m06_pxg_exponential) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = R40_HALF_LIFE_DAYS)) |>
    add(GlobalHomeAdvantage()) |>
    add(PxGCovariate(
        feature = r40_pxg_feature,
        prior   = truncated(Normal(0.15, 0.10), lower = 0.0),
        role    = SupremacyRole(),
    )) |>
    add(PoissonObservation()) |>
    build

m07 = CountModelBuilder(:m07_bench_depth) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = R40_HALF_LIFE_DAYS)) |>
    add(GlobalHomeAdvantage()) |>
    add(BenchDepthCovariate(log_transform = true)) |>
    add(PoissonObservation()) |>
    build

m08 = CountModelBuilder(:m08_composite_allstar) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = R40_HALF_LIFE_DAYS)) |>
    add(GlobalHomeAdvantage()) |>
    add(ProductionWealthCovariate(
        feature = ProductionWealthFeature(curve = RichardsSigmoid(23.0, 0.80, 2.0)))) |>
    add(PxGCovariate(feature = r40_pxg_feature, role = SupremacyRole())) |>
    add(BenchDepthCovariate(log_transform = true)) |>
    add(PoissonObservation()) |>
    build

m09 = CountModelBuilder(:m09_rapm_control) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = R40_HALF_LIFE_DAYS)) |>
    add(GlobalHomeAdvantage()) |>
    add(ProductionWealthCovariate(
        feature = ProductionWealthFeature(curve = RichardsSigmoid(23.0, 0.80, 2.0)))) |>
    add(PxGCovariate(feature = r40_pxg_feature, role = SupremacyRole())) |>
    add(PxGRapmCovariate(feature = r40_rapm_feature, role = SupremacyRole())) |>
    add(PoissonObservation()) |>
    build

r40_models = [
    (R40_BASELINE_NAME,          m00),
    ("m05_prod_wealth",         m05),
    ("m06_pxg_exponential",     m06),
    ("m07_bench_depth",         m07),
    ("m08_composite_allstar",   m08),
    ("m09_rapm_control",        m09),
]

for (name, model) in r40_models
    sites = join(string.(Models.PreGame.cb_covariate_names(model)), ", ")
    println("  $name — covariates: ", isempty(sites) ? "none" : sites)
end

# %%
# ==============================================================================
# 6. Feature construction and preflight gates
# ==============================================================================
#
# Nothing is sampled until the design columns have been looked at. A constant column
# means the feature found no coverage and its weight is unidentified; a mostly-neutral
# column means the arm is quietly the baseline on most fixtures.

println("\n[3/6] pxG measurement coverage ...")
l40_print_pxg_coverage(l40_pxg_coverage(ds, r40_pxg_feature))

println("\n[4/6] Per-fold design column preflight and leak gate ...")
for (name, model) in r40_models
    println("\n  $name")
    l40_print_preflight(l40_covariate_preflight(ds, model, splitter))
end

# %%
# ==============================================================================
# 7. Training
# ==============================================================================

sampler_config = QueuedNUTSConfig(
    n_samples   = R40_SAMPLES,
    n_warmup    = R40_WARMUP,
    n_chains    = R40_CHAINS,
    accept_rate = R40_ACCEPT_RATE,
)

println("\n[5/6] Fitting ...")
r40_fits = Dict{String, Fit}()
r40_rows = NamedTuple[]

for (name, model) in r40_models
    println("\n" * "-"^90)
    println(" FIT: $name")
    println("-"^90)

    fit_config = FitConfig(
        name      = name,
        model     = model,
        splitter  = splitter,
        sampler   = sampler_config,
        execution = QueuedExecution(),
        save_dir  = joinpath(R40_SAVE_ROOT, name),
    )

    started = time()
    fit = fit_model(fit_config, ds; quiet = false)
    elapsed = time() - started

    r40_fits[name] = fit
    push!(r40_rows, l40_summarise_fit(name, fit, ds, elapsed))
end

# %%
# ==============================================================================
# 8. Convergence diagnostics and out-of-sample report
# ==============================================================================

println("\n[6/6] Leaderboard\n")
r40_leaderboard = l40_print_leaderboard(r40_rows; baseline = R40_BASELINE_NAME)

# %%
# ==============================================================================
# 9. The decision
# ==============================================================================
#
# Stated explicitly so the run answers the question it opened with, rather than
# leaving the reader to eyeball a table.

let
    baseline = only(filter(r -> r.name == R40_BASELINE_NAME, r40_rows))
    println("\nDecision against $(R40_BASELINE_NAME) (log loss $(round(baseline.logloss, digits = 4))):")
    for row in r40_rows
        row.name == R40_BASELINE_NAME && continue
        delta = row.logloss - baseline.logloss
        verdict = if !row.passed
            "NOT COMPARABLE — convergence gate failed"
        elseif delta < 0
            @sprintf("EARNS ITS PARAMETER (%.4f better)", -delta)
        else
            @sprintf("does not earn its parameter (%.4f worse)", delta)
        end
        @printf("  %-26s : %s\n", row.name, verdict)
    end
    println("\n  Reminder: this is predictive fit only. Allocation is r22's question.")
end

r40_leaderboard
