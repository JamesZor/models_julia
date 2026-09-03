# ==============================================================================
# r65 — Production 40-fold hierarchical team-kappa grid
# ==============================================================================
#
# WHAT THIS IS. The production walk-forward grid for the two candidates that passed
# the extended r64 smoke test:
#
#   m05_hierarchical_kappa   team-state control
#   m12_hierarchical_kappa   production wealth + shots-RAPM lineup hybrid
#
# WHAT IS HELD FIXED. Both candidates use the same 40-boundary Scottish Lower
# splitter, two-arm observation, priors, feature recipes, book and policy as the
# Experiment 06 shared-kappa controls. The hierarchical sampler keeps 4 chains,
# 800 warmup and 800 retained draws, with target acceptance 0.90 as established by
# r64's zero-divergence gate.
#
# EXECUTION CONTRACT. `QueuedExecution()` flattens 40 folds × 4 chains into the
# native queue. Candidates run sequentially so one production grid owns the host.
# Checkpoints are retained after completion; PostgreSQL is the system of record.
#
# SAFETY. The file defaults to PREPARE ONLY: it builds and audits every FeatureSet,
# reports the exact task inventory, and checks for an already-completed recipe.
# Launch sampling deliberately with:
#
#   L65_RUN_GRID=true /root/.juliaup/bin/julia --project -t 16 \
#     experiments/scottish_lower/06_joint_player_lineup_fusion/r65_train_hierarchical_kappa_40fold.jl
# ================================================================================

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================
using BayesianFootball
using DataFrames
using Dates
using LinearAlgebra
using Printf
using ThreadPinning

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "l64_hierarchical_kappa_loader.jl"))

const R65_FEATURES = BayesianFootball.Features

# %%
# ==============================================================================
# 2. Configuration and promotion thresholds
# ==============================================================================
const R65_EXPECTED_BOUNDARIES = 40
const R65_EXPECTED_OOS_FIXTURES = 710
const R65_MIN_ESS_BULK = 400.0
const R65_MIN_ESS_TAIL = 300.0
const R65_RUN_GRID = lowercase(get(ENV, "L65_RUN_GRID", "false")) in ("1", "true", "yes")

const R65_THRESHOLDS = ConvergenceThresholds(
    max_rhat = 1.05,
    min_ess = R65_MIN_ESS_TAIL,
    max_divergence_rate = eps(Float64),
    min_bfmi = 0.30,
    max_treedepth_rate = 0.05,
)

function r65_assert_promotion(name, diagnostics)
    diagnostics.passed || error(
        "$name failed the production convergence audit: $(join(diagnostics.failures, "; "))")
    diagnostics.max_rhat < 1.05 || error("$name max R-hat is $(diagnostics.max_rhat)")
    diagnostics.min_ess_bulk > R65_MIN_ESS_BULK || error(
        "$name min bulk ESS is $(diagnostics.min_ess_bulk); require > $R65_MIN_ESS_BULK")
    diagnostics.min_ess_tail > R65_MIN_ESS_TAIL || error(
        "$name min tail ESS is $(diagnostics.min_ess_tail); require > $R65_MIN_ESS_TAIL")
    diagnostics.n_divergent == 0 || error(
        "$name has $(diagnostics.n_divergent) divergent transitions; require zero")
    return nothing
end

# %%
# ==============================================================================
# 3. Data snapshot, temporal splits and queue inventory
# ==============================================================================
println("\n" * "="^112)
println(" EXPERIMENT 06 · PRODUCTION HIERARCHICAL TEAM-KAPPA GRID · 24/25 + 25/26")
println("="^112)
@printf("  sampler   : 4 chains × 800 warmup × 800 retained · target acceptance %.2f\n",
        l64_production_sampler.accept_rate)
println("  execution : QueuedExecution()")
println("  threads   : ", Threads.nthreads())
println("  database  : ", db)
println("  mode      : ", R65_RUN_GRID ? "PRODUCTION SAMPLING" : "PREPARE ONLY")
println("  started   : ", Dates.now())

Threads.nthreads() >= 16 || error(
    "r65 is staged for mcmc-beast with 16 Julia threads; got $(Threads.nthreads())")

boundaries = Data.create_id_boundaries(ds, l64_production_splitter)
length(boundaries) == R65_EXPECTED_BOUNDARIES || error(
    "production splitter produced $(length(boundaries)) boundaries; expected " *
    "$R65_EXPECTED_BOUNDARIES. Do not sample until datastore/splitter drift is resolved.")

nonempty_boundary_states = count(pair -> !isempty(first(pair).target_match_ids), boundaries)
@printf("  boundaries: %d total | %d with non-empty cumulative target state\n",
        length(boundaries), nonempty_boundary_states)
@printf("  queue     : %d fold-chain tasks per candidate | %d total tasks\n",
        length(boundaries) * l64_production_sampler.n_chains,
        length(boundaries) * l64_production_sampler.n_chains * length(l64_candidate_models))

# %%
# ==============================================================================
# 4. Full FeatureSet and filtration preflight
# ==============================================================================
feature_sets_by_name = Dict{String,Any}()
oos_by_name = Dict{String,Any}()
completed_runs = Dict{String,Union{Nothing,String}}()

for (name, model) in l64_candidate_models
    println("\n  PREPARE $name")
    feature_sets = R65_FEATURES.create_features(
        boundaries, ds, model, l64_production_splitter)
    oos = [Data.get_next_matches(ds, feature_sets[i], l64_production_splitter)
           for i in eachindex(feature_sets)]

    length(feature_sets) == R65_EXPECTED_BOUNDARIES || error(
        "$name built $(length(feature_sets)) FeatureSets; expected $R65_EXPECTED_BOUNDARIES")
    sum(nrow, oos) == R65_EXPECTED_OOS_FIXTURES || error(
        "$name resolves $(sum(nrow, oos)) held-out rows; expected " *
        "$R65_EXPECTED_OOS_FIXTURES from the canonical Experiment 06 match set")
    all(frame -> nrow(frame) > 0, oos) || error(
        "$name has an empty OOS fold; the 40-fold queue must price every boundary")

    for (i, feature_set) in enumerate(feature_sets)
        n_teams = Int(first(feature_set).data[:n_teams])
        observed = cb_parameter_count(model, n_teams)
        expected = l64_expected_params(name, n_teams)
        observed == expected || error(
            "$name boundary $i has $observed parameters; structural contract expects $expected")
    end

    feature_sets_by_name[name] = feature_sets
    oos_by_name[name] = oos
    completed_runs[name] = l64_completed_run_id(db, l64_configs[name])
    println("    FeatureSets: $(length(feature_sets)) | OOS fixtures: $(sum(nrow, oos))")
    println("    exact completed recipe: ", something(completed_runs[name], "none"))
end

println("\n  PREPARE PASS: all 40 boundaries built for both candidates; structural " *
        "parameter counts and OOS coverage match the production splitter.")

# %%
# ==============================================================================
# 5. Production training and PostgreSQL persistence
# ==============================================================================
run_ids = Dict{String,Any}()
rows = NamedTuple[]

if !R65_RUN_GRID
    println("\nPREPARE ONLY complete. No MCMC was launched.")
    println("Set L65_RUN_GRID=true on an idle mcmc-beast to start the 40-fold queue.")
else
    for (name, _) in l64_candidate_models
        config = l64_configs[name]
        existing = completed_runs[name]
        if existing !== nothing
            println("\n  SKIP $name: exact completed recipe already exists at run $existing")
            run_ids[name] = existing
            continue
        end

        println("\n" * "-"^100)
        println(" GRID: $name · ", Dates.now())
        println("-"^100)

        checkpoint_dir = joinpath(config.save_dir, "checkpoints")
        started = time()
        fit = fit_model(
            config;
            feature_sets = feature_sets_by_name[name],
            oos_fixtures = oos_by_name[name],
            thresholds = R65_THRESHOLDS,
            checkpoint_dir = checkpoint_dir,
            cleanup_checkpoints = false,
            quiet = false,
        )
        elapsed = time() - started

        length(fit.folds) == R65_EXPECTED_BOUNDARIES || error(
            "$name returned $(length(fit.folds)) folds; expected $R65_EXPECTED_BOUNDARIES")
        fit.latents isa CountLatents || error(
            "$name returned $(typeof(fit.latents)); PostgreSQL requires CountLatents")
        n_matches(fit.latents) == R65_EXPECTED_OOS_FIXTURES || error(
            "$name returned $(n_matches(fit.latents)) OOS latents; expected " *
            "$R65_EXPECTED_OOS_FIXTURES")
        r65_assert_promotion(name, fit.diagnostics)

        run_id = save_fit(fit, db)
        run_ids[name] = run_id
        push!(rows, (
            name = name,
            elapsed = elapsed,
            rhat = fit.diagnostics.max_rhat,
            ess_bulk = fit.diagnostics.min_ess_bulk,
            ess_tail = fit.diagnostics.min_ess_tail,
            divergences = fit.diagnostics.n_divergent,
            n_latents = n_matches(fit.latents),
            run_id = run_id,
        ))
        @printf("  persisted %s in %.1f minutes · run %s\n",
                name, elapsed / 60, string(run_id))
    end
end

# %%
# ==============================================================================
# 6. Final report
# ==============================================================================
if !isempty(rows)
    println("\n" * "="^155)
    @printf(" %-28s | %9s | %7s | %8s | %8s | %4s | %8s | %36s\n",
            "Model", "Minutes", "R-hat", "ESS bulk", "ESS tail", "Div", "Latents", "Run UUID")
    println("-"^155)
    for row in rows
        @printf(" %-28s | %9.1f | %7.4f | %8.0f | %8.0f | %4d | %8d | %36s\n",
                row.name, row.elapsed / 60, row.rhat, row.ess_bulk, row.ess_tail,
                row.divergences, row.n_latents, string(row.run_id))
    end
    println("="^155)
end
println("Finished: ", Dates.now())
