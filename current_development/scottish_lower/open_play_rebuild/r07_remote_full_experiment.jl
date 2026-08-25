# current_development/scottish_lower/open_play_rebuild/r07_remote_full_experiment.jl
#
# RUNNER. Train and infer the clean Scottish Lower NP-NOG rebuild over every genuine
# walk-forward fold using the project's native flattened fold × chain NUTS queue.
#
# =============================================================================================
# WHAT THIS IS AND IS NOT
# =============================================================================================
#
# This is the Stage 8 full-posterior and genuine next-step OOS generation run for pooled
# tournaments 56 and 57, covering target seasons 24/25 and 25/26.
#
# It DOES:
#   - translate generic cumulative split boundaries into the rebuild's fit/holdout contract;
#   - enforce strict kickoff-time filtration around postponed fixtures;
#   - run four NUTS chains per fold through BayesianFootball's native global queue;
#   - hard-gate every fold on convergence before promoting OOS predictions;
#   - persist resumable chains, diagnostics, latent states, and PPDs.
#
# It DOES NOT:
#   - evaluate predictions against outcomes;
#   - create a leaderboard or backtest;
#   - reuse legacy open-play, pxG, wealth, Layer-2, or portfolio results.
#
# =============================================================================================
# FILTRATION CONTRACT
# =============================================================================================
#
# `SplitBoundary.target_match_ids` are cumulative observations through time step t; they are
# not held-out fixtures. For each fold this runner constructs an explicit rebuild boundary:
#
#   fitted history = eligible observations through t with kickoff < earliest t+1 kickoff
#   held-out target = genuine next-step fixtures returned by Data.get_next_matches(...)
#
# The date filter matters because postponed matches can make adjacent `match_biweek` groups
# overlap in calendar time. Excluded not-yet-played IDs are retained in fold provenance.
#
# =============================================================================================
# PERSISTENCE AND RESUME
# =============================================================================================
#
# Native queue checkpoints are written atomically as:
#
#   queued_checkpoints/split_001.jls ... split_038.jls
#
# Accepted fold outputs live under `fold_XX/fold_result.jls`. Prototype artifacts reference
# `RebuildExtractionRecombination`; include l05 before deserializing them in another session.
#
# =============================================================================================
# USAGE — mcmc-beast only
# =============================================================================================
#
# Full run or exact resume:
#
#   STAGE8_RESUME_DIR=/absolute/run/path \
#   STAGE8_MAX_CONCURRENT_TASKS=16 \
#   julia --project -t16 -e \
#     'include("current_development/scottish_lower/open_play_rebuild/r07_remote_full_experiment.jl")'
#
# Preflight without sampling:
#
#   STAGE8_PREPARE_ONLY=1 julia --project -t16 -e 'include(...)'
#
# Style reference: docs/prototype_runner_style_guide.md

# %%
# ===================================================================
# 1. Packages and rebuild implementation
# ===================================================================

using BayesianFootball
using DataFrames, Dates, LinearAlgebra, MCMCChains, Printf
using Random, Serialization, Statistics, ThreadPinning, UUIDs

include(joinpath(@__DIR__, "l07_rebuild_full_experiment.jl"))
using .RebuildFullExperiment
using .RebuildFullExperiment.RebuildSampling

include(joinpath(@__DIR__, "l05_rebuild_extraction_recombination.jl"))
using .RebuildExtractionRecombination
using .RebuildExtractionRecombination.RebuildEquations
using .RebuildExtractionRecombination.RebuildFeatures

const SL8_Data        = BayesianFootball.Data
const SL8_Features    = BayesianFootball.Features
const SL8_PreGame     = BayesianFootball.Models.PreGame
const SL8_Predictions = BayesianFootball.Predictions
const SL8_Training    = BayesianFootball.Training
const SL8_Samplers    = BayesianFootball.Samplers

# %%
# ===================================================================
# 2. Experiment configuration
# ===================================================================

const SL8_CONFIG = stage8_config_from_env()

const SL8_TARGET_SEASONS = ["24/25", "25/26"]
const SL8_TOURNAMENTS    = [56, 57]
const SL8_DYNAMICS_COL   = :match_biweek
const SL8_HISTORY_DEPTH  = 2

function sl8_splitter()
    return SL8_Data.GroupedCVConfig(
        tournament_groups = [SL8_TOURNAMENTS],
        target_seasons = SL8_TARGET_SEASONS,
        history_seasons = SL8_HISTORY_DEPTH,
        dynamics_col = SL8_DYNAMICS_COL,
        warmup_period = 0,
        stop_early = true,
    )
end

function sl8_sampler(config)
    return SL8_Samplers.QueuedNUTSConfig(
        n_samples = config.samples,
        n_warmup = config.warmup,
        n_chains = config.chains,
        accept_rate = 0.8,
        max_depth = config.max_depth,
        initialisation = PriorInit(),
        show_progress = false,
    )
end

# %%
# ===================================================================
# 3. Remote runtime and output directory
# ===================================================================

Threads.nthreads() == 16 ||
    error("Stage 8 requires `julia --project -t16`; got $(Threads.nthreads()) threads")

SL8_CONFIG.max_concurrent_tasks <= Threads.nthreads() ||
    error("queue concurrency exceeds available Julia threads")

haskey(ENV, "BF_DB_URL") ||
    error("BF_DB_URL is required for read-only canonical identity access")

BLAS.set_num_threads(1)
BLAS.get_num_threads() == 1 || error("Stage 8 requires one BLAS thread")
ThreadPinning.pinthreads(:cores)

const SL8_OUT_DIR = stage8_output_directory(SL8_CONFIG)

println("\n", "="^90)
println("STAGE 8 — clean Scottish Lower walk-forward experiment")
println("="^90)
@printf("Julia threads %-3d  BLAS threads %-2d  queue concurrency %-2d\n",
    Threads.nthreads(), BLAS.get_num_threads(), SL8_CONFIG.max_concurrent_tasks)
@printf("NUTS per chain: %d warmup + %d retained; %d chains per fold\n",
    SL8_CONFIG.warmup, SL8_CONFIG.samples, SL8_CONFIG.chains)
println("Output: ", SL8_OUT_DIR)

# %%
# ===================================================================
# 4. Data snapshot and genuine next-step fold inventory
# ===================================================================

const SL8_DATASTORE = SL8_Data.load_datastore_cached(
    SL8_Data.ScottishLower();
    max_age_hours = 10_000,
)

const SL8_SPLITTER = sl8_splitter()
const SL8_BOUNDARIES = SL8_Data.create_id_boundaries(SL8_DATASTORE, SL8_SPLITTER)

length(SL8_BOUNDARIES) == SL8_CONFIG.expected_folds ||
    error("expected $(SL8_CONFIG.expected_folds) folds, got $(length(SL8_BOUNDARIES))")

const SL8_OOS_FOLDS = true_oos_inventory(
    SL8_DATASTORE,
    SL8_BOUNDARIES,
    SL8_SPLITTER,
)
const SL8_INVENTORY = fold_inventory(SL8_OOS_FOLDS)

const SL8_REGISTRY_IDS = sort!(unique(vcat([
    inference_ids(fold.boundary, fold.ids) for fold in SL8_OOS_FOLDS
]...)))

@printf("Built %d folds: %d genuine OOS matches, %d unique registry matches\n",
    length(SL8_OOS_FOLDS),
    sum(fold.count for fold in SL8_OOS_FOLDS),
    length(SL8_REGISTRY_IDS))

# %%
# ===================================================================
# 5. Canonical registry and model construction
# ===================================================================

const SL8_REGISTRY = fetch_canonical_registry(SL8_REGISTRY_IDS)
const SL8_REGISTRY_SHA = registry_fingerprint(SL8_REGISTRY)

function sl8_model(registry)
    return ScottishLowerNPNOGRecombinedPoissonModel(
        registry;
        half_life_days = 365.0,
        own_goal_policy = :beneficiary,
    )
end

const SL8_MODEL = sl8_model(SL8_REGISTRY)

registry_fingerprint(SL8_MODEL.registry) == SL8_REGISTRY_SHA ||
    error("global model registry changed during construction")

println("Global canonical registry SHA256: ", SL8_REGISTRY_SHA)

# %%
# ===================================================================
# 6. Run manifests
# ===================================================================

function sl8_git_commit()
    try
        return readchomp(`git rev-parse HEAD`)
    catch
        return "unavailable"
    end
end

function sl8_write_manifests!(config, outdir, registry_sha, inventory)
    commit = sl8_git_commit()
    sampler_manifest = (
        samples = config.samples,
        warmup = config.warmup,
        chains = config.chains,
        concurrency = config.max_concurrent_tasks,
        init = "PriorInit",
        max_depth = config.max_depth,
        seed = config.queue_seed,
    )
    run_manifest = immutable_manifest(
        run_id = config.run_id,
        git_commit = commit,
        julia_version = string(VERSION),
        threads = Threads.nthreads(),
        blas_threads = BLAS.get_num_threads(),
        splitter = (
            tournament_groups = [SL8_TOURNAMENTS],
            target_seasons = SL8_TARGET_SEASONS,
            history_seasons = SL8_HISTORY_DEPTH,
            dynamics_col = SL8_DYNAMICS_COL,
            stop_early = true,
        ),
        expected_folds = config.expected_folds,
        registry_fingerprint = registry_sha,
        sampler = sampler_manifest,
        folds = inventory,
    )

    run_path = joinpath(outdir, "run_manifest.jls")
    if isfile(run_path)
        old = deserialize(run_path)
        same_sampler = old.sampler.samples == config.samples &&
            old.sampler.warmup == config.warmup &&
            old.sampler.chains == config.chains &&
            old.sampler.init == "PriorInit" &&
            old.sampler.max_depth == config.max_depth
        exact = old.stage == 8 &&
            old.registry_fingerprint == registry_sha &&
            old.folds == inventory &&
            same_sampler
        exact || error("resume manifest is not model/split/sampler exact")
    else
        atomic_serialize(run_path, run_manifest)
    end

    queue_path = joinpath(outdir, "native_queue_manifest.jls")
    if isfile(queue_path)
        old_queue = deserialize(queue_path)
        changed = get(old_queue, :git_commit, nothing) != commit ||
            get(old_queue, :max_concurrent_tasks, nothing) != config.max_concurrent_tasks
        changed && mv(queue_path, queue_path * ".superseded-" * string(uuid4()))
    end
    if !isfile(queue_path)
        atomic_serialize(queue_path, (
            stage = 8,
            git_commit = commit,
            created_utc = string(now(UTC)),
            max_concurrent_tasks = config.max_concurrent_tasks,
            queue_seed = config.queue_seed,
            source_run_manifest = basename(run_path),
            registry_fingerprint = registry_sha,
        ))
    end
    return (run_manifest = run_path, queue_manifest = queue_path)
end

const SL8_MANIFESTS = sl8_write_manifests!(
    SL8_CONFIG,
    SL8_OUT_DIR,
    SL8_REGISTRY_SHA,
    SL8_INVENTORY,
)

# %%
# ===================================================================
# 7. Leakage-safe FeatureSets and checkpoint preflight
# ===================================================================

function sl8_build_fold_context(fold, ds, model, registry_sha, match_dates)
    cutoff = minimum(RebuildFeatures._date(row) for row in eachrow(fold.rows))
    nominal_fit_ids = boundary_ids(fold.boundary)

    # A nominally earlier biweek can contain a postponed match that had not yet happened.
    fitted_ids = sort!(Int[
        match_id for match_id in nominal_fit_ids if match_dates[match_id] < cutoff
    ])
    excluded_ids = sort!(collect(setdiff(Set(nominal_fit_ids), Set(fitted_ids))))

    fit_boundary = SL8_Data.SplitBoundary(
        fold.boundary.fold_id,
        fold.boundary.target_step,
        fitted_ids,
        fold.ids,
    )
    features = SL8_Features.create_features(
        fit_boundary,
        ds,
        model,
        SL8_DYNAMICS_COL,
    )
    validate_feature_set(features)
    features[:model_registry_fingerprint] == registry_sha ||
        error("fold $(fold.fold) lost global registry identity")

    boundary_hash = boundary_sha256(fit_boundary)
    oos_provenance = (
        true_oos_ids = Int.(fold.ids),
        true_oos_ids_sha256 = fold.ids_sha256,
        prediction_step = Int(fold.prediction_step),
        target_season = fold.meta.target_season,
        target_time_step = Int(fold.meta.time_step),
        cutoff_date = cutoff,
        excluded_not_yet_played_ids = excluded_ids,
    )
    checkpoint_metadata = (
        original_meta = fold.meta,
        fold_index = Int(fold.fold),
        boundary_sha256 = boundary_hash,
        oos_provenance = oos_provenance,
    )
    return (
        x = fold,
        fit_boundary = fit_boundary,
        fs = features,
        meta = checkpoint_metadata,
        boundary_sha256 = boundary_hash,
        excluded_fit_ids = excluded_ids,
    )
end

function sl8_build_contexts(ds, folds, model, registry_sha)
    match_dates = Dict(
        Int(row.match_id) => RebuildFeatures._date(row) for row in eachrow(ds.matches)
    )
    contexts = [
        sl8_build_fold_context(fold, ds, model, registry_sha, match_dates)
        for fold in folds
    ]
    length(unique(typeof(context.meta) for context in contexts)) == 1 ||
        error("FeatureCollection metadata must be homogeneous")
    return contexts
end

function sl8_feature_collection(contexts)
    items = [(context.fs, context.meta) for context in contexts]
    return SL8_Features.FeatureCollection(items)
end

if !SL8_CONFIG.dry_run
    const SL8_CONTEXTS = sl8_build_contexts(
        SL8_DATASTORE,
        SL8_OOS_FOLDS,
        SL8_MODEL,
        SL8_REGISTRY_SHA,
    )
    const SL8_FEATURES = sl8_feature_collection(SL8_CONTEXTS)
    const SL8_CHECKPOINT_DIR = joinpath(SL8_OUT_DIR, "queued_checkpoints")
    const SL8_CHECKPOINT_REPORT = prepare_native_checkpoints!(
        SL8_CONTEXTS,
        SL8_CHECKPOINT_DIR,
        SL8_OUT_DIR,
        SL8_CONFIG.samples,
        validate_primitive_chain,
    )

    @printf("Checkpoint preflight: %d/%d valid", SL8_CHECKPOINT_REPORT.valid,
        SL8_CHECKPOINT_REPORT.total)
    isempty(SL8_CHECKPOINT_REPORT.recovered) ||
        @printf("; recovered %d", length(SL8_CHECKPOINT_REPORT.recovered))
    isempty(SL8_CHECKPOINT_REPORT.migrated) ||
        @printf("; migrated %d", length(SL8_CHECKPOINT_REPORT.migrated))
    println()
end

# %%
# ===================================================================
# 8. Native flattened fold × chain queue
# ===================================================================

function sl8_training_config(config, checkpoint_dir)
    strategy = SL8_Training.Independent(
        parallel = true,
        max_concurrent_tasks = config.max_concurrent_tasks,
    )
    return SL8_Training.TrainingConfig(
        sl8_sampler(config),
        strategy,
        checkpoint_dir,
        false,
    )
end

function sl8_train(model, features, config, checkpoint_dir)
    training_config = sl8_training_config(config, checkpoint_dir)
    Random.seed!(config.queue_seed)
    println("\n", "="^90)
    @printf("NATIVE QUEUE: %d folds × %d chains; at most %d live chain tasks\n",
        length(features), config.chains, config.max_concurrent_tasks)
    println("="^90)
    return SL8_Training.train(model, training_config, features)
end

# %%
# ===================================================================
# 9. Convergence gates and genuine OOS inference
# ===================================================================

function sl8_fold_manifest(context, chain, diagnostics_report, config, registry_sha, index)
    return (
        stage = 8,
        fold = context.x.fold,
        boundary_sha256 = context.boundary_sha256,
        registry_fingerprint = context.fs[:registry_fingerprint],
        model_registry_fingerprint = registry_sha,
        fitted_match_ids = Int.(context.fit_boundary.history_match_ids),
        excluded_not_yet_played_ids = context.excluded_fit_ids,
        true_oos_ids = Int.(context.x.ids),
        true_oos_ids_sha256 = context.x.ids_sha256,
        sampler = sampler_metadata(sl8_sampler(config)),
        diagnostics = diagnostics_report,
        native_checkpoint = joinpath(
            "queued_checkpoints",
            "split_$(lpad(index, 3, '0')).jls",
        ),
    )
end

function sl8_latent_rows(predictions, context)
    return [(
        match_id = match_id,
        league_id = values[:league_id],
        home_team_status = values[:home_team_status],
        away_team_status = values[:away_team_status],
        provenance = "stage8_true_next_step_metadata_only",
        fold_index = context.x.fold,
        boundary_sha256 = context.boundary_sha256,
        lambda_h = values[:lambda_h],
        lambda_a = values[:lambda_a],
        lambda_Y_home = values[:lambda_Y_home],
        lambda_Y_away = values[:lambda_Y_away],
        lambda_converted_penalty_home = values[:lambda_converted_penalty_home],
        lambda_converted_penalty_away = values[:lambda_converted_penalty_away],
        lambda_og_home = values[:lambda_og_home],
        lambda_og_away = values[:lambda_og_away],
    ) for (match_id, values) in predictions]
end

function sl8_validate_score_matrices!(latent, model, fold)
    for row in eachrow(latent)
        parameters = SL8_Predictions.extract_params(model, row)
        score = SL8_Predictions.compute_score_matrix(model, parameters; max_goals = 12)
        tensor = SL8_Predictions.score_matrix_data(score)
        all(isfinite, tensor) || error("fold $fold has non-finite score mass")
        all(>=(0), tensor) || error("fold $fold has negative score mass")
        normalized = all(
            isapprox(sum(tensor[:, :, draw]), 1; atol = 1e-10)
            for draw in axes(tensor, 3)
        )
        normalized || error("fold $fold score matrix is not normalized")
    end
end

function sl8_process_fold!(index, context, model, config, checkpoint_dir, outdir, registry_sha)
    fold = context.x.fold
    fold_dir = joinpath(outdir, "fold_$(lpad(fold, 2, '0'))")
    mkpath(fold_dir)

    result_path = joinpath(fold_dir, "fold_result.jls")
    isfile(result_path) && return :already_passed

    checkpoint_path = joinpath(checkpoint_dir, "split_$(lpad(index, 3, '0')).jls")
    saved = valid_native_checkpoint(
        checkpoint_path,
        context,
        config.samples,
        validate_primitive_chain,
    )
    isnothing(saved) && return :incomplete

    chain, stored_metadata = saved
    checkpoint_metadata_matches(stored_metadata, context.meta) ||
        error("fold $fold checkpoint metadata mismatch")

    diagnostics_report = diagnostics(chain; max_depth = config.max_depth)
    manifest = sl8_fold_manifest(
        context,
        chain,
        diagnostics_report,
        config,
        registry_sha,
        index,
    )

    diagnostics_path = joinpath(fold_dir, "diagnostics.jls")
    isfile(diagnostics_path) || atomic_serialize(diagnostics_path, manifest)

    if !hard_smoke_pass(diagnostics_report)
        failure_path = joinpath(fold_dir, "hard_gate_failure.jls")
        isfile(failure_path) || atomic_serialize(
            failure_path,
            (; fold_manifest = manifest, status = :hard_gate_failed),
        )
        return :hardfail
    end

    predictions = SL8_PreGame.extract_parameters(
        model,
        context.x.rows,
        context.fs,
        chain,
    )
    rows = sl8_latent_rows(predictions, context)
    length(rows) == length(context.x.ids) || error("fold $fold has missing OOS extraction")

    latent = generic_dataframe(rows)
    dataframe_roundtrip_ok(latent) || error("fold $fold latent DataFrame roundtrip failed")
    sl8_validate_score_matrices!(latent, model, fold)

    states = BayesianFootball.Experiments.LatentStates(latent, model)
    ppd = SL8_Predictions.model_inference(states; verbose = false)
    expected_draws = config.chains * config.samples
    all(length(distribution) == expected_draws for distribution in ppd.df.distribution) ||
        error("fold $fold PPD draw count mismatch")

    statuses = vcat(Symbol.(latent.home_team_status), Symbol.(latent.away_team_status))
    fallback = count(==(:target_only_population_fallback), statuses)
    fallback_audit = (
        fallback = fallback,
        total = length(statuses),
        rate = fallback / length(statuses),
    )
    atomic_serialize(
        result_path,
        (; fold_manifest = manifest, status = :pass, latent, ppd, fallback_audit),
    )
    return :pass
end

function sl8_process_all_folds!(contexts, model, config, checkpoint_dir, outdir, registry_sha)
    statuses = Symbol[]
    for (index, context) in enumerate(contexts)
        try
            status = sl8_process_fold!(
                index,
                context,
                model,
                config,
                checkpoint_dir,
                outdir,
                registry_sha,
            )
            push!(statuses, status)
            @printf("fold %2d  %s\n", context.x.fold, uppercase(string(status)))
        catch err
            io = IOBuffer()
            Base.show_backtrace(io, catch_backtrace())
            artifact = (
                stage = 8,
                fold = context.x.fold,
                status = :post_queue_error,
                error = sanitized_error(err),
                backtrace = sanitized_error(String(take!(io))),
            )
            fold_dir = joinpath(outdir, "fold_$(lpad(context.x.fold, 2, '0'))")
            mkpath(fold_dir)
            error_path = joinpath(fold_dir, "error.jls")
            isfile(error_path) || atomic_serialize(error_path, artifact)
            push!(statuses, :error)
            @printf("fold %2d  ERROR: %s\n", context.x.fold, artifact.error)
        end
    end
    return statuses
end

# %%
# ===================================================================
# 10. Execute selected mode
# ===================================================================

if SL8_CONFIG.dry_run
    @printf("FINAL_STAGE8_DRY_RUN folds=%d registry=%s output=%s\n",
        length(SL8_BOUNDARIES), SL8_REGISTRY_SHA, SL8_OUT_DIR)
elseif SL8_CONFIG.prepare_only
    @printf("FINAL_STAGE8_PREPARE_ONLY folds=%d valid_checkpoints=%d queue_tasks=%d concurrency=%d output=%s\n",
        length(SL8_FEATURES),
        SL8_CHECKPOINT_REPORT.valid,
        SL8_CONFIG.chains * length(SL8_FEATURES),
        SL8_CONFIG.max_concurrent_tasks,
        SL8_OUT_DIR)
else
    const SL8_TRAINING_RESULTS = sl8_train(
        SL8_MODEL,
        SL8_FEATURES,
        SL8_CONFIG,
        SL8_CHECKPOINT_DIR,
    )
    const SL8_FOLD_STATUSES = sl8_process_all_folds!(
        SL8_CONTEXTS,
        SL8_MODEL,
        SL8_CONFIG,
        SL8_CHECKPOINT_DIR,
        SL8_OUT_DIR,
        SL8_REGISTRY_SHA,
    )
    const SL8_FINAL_SUMMARY = stage8_progress!(SL8_OUT_DIR, SL8_INVENTORY)

    println("\n", "="^90)
    @printf("FINAL_STAGE8_SUMMARY pass=%d hardfail=%d error=%d pending=%d output=%s\n",
        SL8_FINAL_SUMMARY.pass,
        SL8_FINAL_SUMMARY.hardfail,
        SL8_FINAL_SUMMARY.error,
        SL8_FINAL_SUMMARY.pending,
        SL8_OUT_DIR)
    println("="^90)
end

nothing
