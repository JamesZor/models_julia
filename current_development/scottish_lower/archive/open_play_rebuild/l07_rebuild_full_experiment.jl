module RebuildFullExperiment

using BayesianFootball, DataFrames, MCMCChains, Serialization, UUIDs, SHA, Dates, Random
include(joinpath(@__DIR__, "l06_rebuild_sampling.jl"))
using .RebuildSampling

export Stage8RunConfig, stage8_config_from_env, stage8_output_directory,
       boundary_sha256, registry_subset, atomic_replace_serialize, sanitized_error,
       checkpoint_valid, checkpoint_metadata_matches, valid_native_checkpoint,
       prepare_native_checkpoints!, stage8_progress!, fold_inventory,
       true_oos_inventory, immutable_manifest, boundary_ids, inference_ids

"""Credential-free stable identity for a temporal boundary's training observations."""
function boundary_sha256(boundary)
    payload = join(vcat("H:" .* string.(sort(Int.(boundary.history_match_ids))),
                        "T:" .* string.(sort(Int.(boundary.target_match_ids)))), "\n")
    bytes2hex(sha256(codeunits(payload)))
end
boundary_ids(boundary) = sort!(unique(vcat(Int.(boundary.history_match_ids), Int.(boundary.target_match_ids))))
inference_ids(boundary, oos_ids) = sort!(unique(vcat(boundary_ids(boundary), Int.(oos_ids))))
_ids_sha256(ids) = bytes2hex(sha256(codeunits(join(string.(sort(Int.(ids))), "\n"))))

"""Take an exact registry snapshot; builders reject both missing and extra rows."""
function registry_subset(registry::DataFrame, ids::AbstractVector{<:Integer})
    wanted = sort!(unique(Int.(ids)))
    out = filter(:match_id => x -> Int(x) in Set(wanted), registry)
    nrow(out) == length(wanted) || throw(ArgumentError("registry snapshot lacks requested match IDs"))
    sort!(out, :match_id)
    Int.(out.match_id) == wanted || throw(ArgumentError("registry snapshot is not an exact requested-ID registry"))
    out
end
registry_subset(registry::DataFrame, boundary::BayesianFootball.Data.SplitBoundary) = registry_subset(registry, boundary_ids(boundary))

"""Return and validate genuine next-step OOS metadata for every `(boundary, meta)` fold."""
function true_oos_inventory(ds, boundaries, splitter)
    out = NamedTuple[]
    for (fold, fold_tuple) in enumerate(boundaries)
        boundary, meta = fold_tuple
        rows = DataFrame(BayesianFootball.Data.get_next_matches(ds, fold_tuple, splitter))
        nrow(rows) > 0 || throw(ArgumentError("fold $fold has no true next-step OOS rows"))
        required = (:match_id, :tournament_id, :season, splitter.dynamics_col)
        all(x -> x in propertynames(rows), required) || throw(ArgumentError("fold $fold OOS metadata lacks required columns"))
        ids = Int.(rows.match_id)
        length(unique(ids)) == length(ids) || throw(ArgumentError("fold $fold OOS IDs are duplicated"))
        observed_tournaments = Set(Int.(rows.tournament_id))
        expected_tournaments = Set(Int.(meta.tournament_ids))
        observed_tournaments ⊆ expected_tournaments || throw(ArgumentError("fold $fold OOS tournament misalignment"))
        all(==(meta.target_season), rows.season) || throw(ArgumentError("fold $fold OOS season misalignment"))
        all(==(meta.time_step + 1), rows[!, splitter.dynamics_col]) || throw(ArgumentError("fold $fold OOS prediction-step misalignment"))
        overlap = intersect(Set(ids), Set(boundary_ids(boundary)))
        isempty(overlap) || throw(ArgumentError("fold $fold true OOS overlaps boundary history/target: $(sort!(collect(overlap)))"))
        metadata_rows = select(rows, :match_id, :tournament_id, :home_team, :away_team, :match_date)
        push!(out, (; fold, boundary, meta, rows=metadata_rows, ids=sort!(ids), count=length(ids),
            ids_sha256=_ids_sha256(ids), prediction_step=meta.time_step + 1))
    end
    out
end

"""Replace mutable progress safely.  This is intentionally distinct from immutable atomic_serialize."""
function atomic_replace_serialize(path::AbstractString, value)
    mkpath(dirname(path)); tmp = path * ".tmp-" * string(uuid4())
    try
        serialize(tmp, value)
        mv(tmp, path; force=true)
    finally
        ispath(tmp) && rm(tmp; force=true)
    end
    path
end

function sanitized_error(err)
    msg = sprint(showerror, err)
    url = get(ENV, "BF_DB_URL", "")
    !isempty(url) && (msg = replace(msg, url => "[REDACTED_DB_URL]"))
    replace(msg, r"://[^\s/@]+:[^\s/@]+@" => "://[REDACTED]@")
end

function checkpoint_valid(path, boundary_hash::AbstractString, chain_id::Int, samples::Int, J::Int, validate_chain)
    isfile(path) || return nothing
    x = try deserialize(path) catch; return nothing end
    (x isa NamedTuple && get(x, :chain_id, nothing) == chain_id &&
     get(x, :boundary_sha256, nothing) == boundary_hash && get(x, :samples, nothing) == samples &&
     haskey(x, :chain)) || return nothing
    try
        validate_chain(x.chain, J)
        size(x.chain, 1) == samples || return nothing
        return x
    catch
        return nothing
    end
end

function fold_inventory(oos_folds)
    [(fold=x.fold, history_count=length(x.boundary.history_match_ids), target_count=length(x.boundary.target_match_ids),
      sha256=boundary_sha256(x.boundary), oos_count=x.count, oos_ids_sha256=x.ids_sha256,
      target_season=x.meta.target_season, target_time_step=x.meta.time_step, prediction_step=x.prediction_step,
      tournament_ids=sort(Int.(x.meta.tournament_ids))) for x in oos_folds]
end
immutable_manifest(; kwargs...) = (; schema_version=1, stage=8, created_utc=string(now(UTC)), kwargs...)

# -------------------------------------------------------------------
# Human-facing Stage 8 runner configuration
# -------------------------------------------------------------------

Base.@kwdef struct Stage8RunConfig
    samples::Int = 800
    warmup::Int = 800
    chains::Int = 4
    max_depth::Int = 10
    expected_folds::Int = 38
    max_concurrent_tasks::Int = 16
    queue_seed::Int = 80_808
    dry_run::Bool = false
    prepare_only::Bool = false
    run_id::String = Dates.format(now(UTC), "yyyymmddTHHMMSS") * "_" * string(rand(UInt), base=16)
    output_root::String = abspath(joinpath("data", "scottish_open_play_rebuild"))
    resume_directory::Union{Nothing,String} = nothing
end

_env_bool(name, default="0") = get(ENV, name, default) == "1"
_env_int(name, default) = parse(Int, get(ENV, name, string(default)))

function stage8_config_from_env()
    resume = get(ENV, "STAGE8_RESUME_DIR", "")
    config = Stage8RunConfig(
        samples = _env_int("STAGE8_SAMPLES", 800),
        warmup = _env_int("STAGE8_WARMUP", 800),
        expected_folds = _env_int("STAGE8_EXPECTED_FOLDS", 38),
        max_concurrent_tasks = _env_int("STAGE8_MAX_CONCURRENT_TASKS", 16),
        queue_seed = _env_int("STAGE8_SEED", 80_808),
        dry_run = _env_bool("STAGE8_DRY_RUN"),
        prepare_only = _env_bool("STAGE8_PREPARE_ONLY"),
        run_id = get(ENV, "STAGE8_RUN_ID",
            Dates.format(now(UTC), "yyyymmddTHHMMSS") * "_" * string(rand(UInt), base=16)),
        output_root = abspath(get(ENV, "STAGE8_OUTPUT_DIR",
            joinpath("data", "scottish_open_play_rebuild"))),
        resume_directory = isempty(resume) ? nothing : abspath(resume),
    )
    config.samples > 0 || throw(ArgumentError("samples must be positive"))
    config.warmup > 0 || throw(ArgumentError("warmup must be positive"))
    config.expected_folds > 0 || throw(ArgumentError("expected folds must be positive"))
    config.max_concurrent_tasks > 0 || throw(ArgumentError("queue concurrency must be positive"))
    return config
end

function stage8_output_directory(config::Stage8RunConfig)
    if isnothing(config.resume_directory)
        outdir = joinpath(config.output_root, "stage8_" * config.run_id)
        ispath(outdir) && throw(ArgumentError("run directory already exists: $outdir"))
        mkpath(outdir)
        return outdir
    end
    isdir(config.resume_directory) ||
        throw(ArgumentError("resume directory does not exist: $(config.resume_directory)"))
    return config.resume_directory
end

# -------------------------------------------------------------------
# Native queued-checkpoint validation and recovery
# -------------------------------------------------------------------

function checkpoint_metadata_matches(stored, expected)
    stored isa NamedTuple || return false
    return get(stored, :fold_index, nothing) == expected.fold_index &&
        get(stored, :boundary_sha256, nothing) == expected.boundary_sha256 &&
        get(stored, :oos_provenance, nothing) == expected.oos_provenance
end

function valid_native_checkpoint(path, context, samples::Int, validate_chain)
    isfile(path) || return nothing
    saved = try
        deserialize(path)
    catch
        return nothing
    end
    saved isa Tuple && length(saved) == 2 || return nothing
    chain, metadata = saved
    chain isa Chains || return nothing
    checkpoint_metadata_matches(metadata, context.meta) || return nothing
    try
        validate_chain(chain, Int(context.fs[:n_teams]))
        size(chain, 1) == samples || return nothing
        size(chain, 3) == 4 || return nothing
        return saved
    catch
        return nothing
    end
end

_checkpoint_name(index) = "split_$(lpad(index, 3, '0')).jls"
_fold_directory(outdir, fold) = joinpath(outdir, "fold_$(lpad(fold, 2, '0'))")

"""Validate, recover, or migrate native split checkpoints without sampling."""
function prepare_native_checkpoints!(contexts, checkpoint_dir, outdir, samples, validate_chain)
    mkpath(checkpoint_dir)
    recovered = Int[]
    migrated = Int[]
    invalidated = Int[]

    for (index, context) in enumerate(contexts)
        native_path = joinpath(checkpoint_dir, _checkpoint_name(index))
        if isfile(native_path) &&
                isnothing(valid_native_checkpoint(native_path, context, samples, validate_chain))
            mv(native_path, native_path * ".invalid-" * string(uuid4()))
            push!(invalidated, context.x.fold)
        end

        if !isfile(native_path)
            prefix = basename(native_path) * ".invalid-"
            candidates = filter(
                path -> startswith(basename(path), prefix),
                readdir(checkpoint_dir; join=true),
            )
            valid_candidates = filter(
                path -> !isnothing(valid_native_checkpoint(path, context, samples, validate_chain)),
                candidates,
            )
            length(valid_candidates) > 1 &&
                error("multiple valid recovery checkpoints for fold $(context.x.fold)")
            if length(valid_candidates) == 1
                mv(only(valid_candidates), native_path)
                push!(recovered, context.x.fold)
            end
        end

        isfile(native_path) && continue

        fold_dir = _fold_directory(outdir, context.x.fold)
        old_chain_path = joinpath(fold_dir, "combined_chain.jls")
        old_diagnostics_path = joinpath(fold_dir, "diagnostics.jls")
        isfile(old_chain_path) && isfile(old_diagnostics_path) || continue

        old_metadata = try deserialize(old_diagnostics_path) catch; nothing end
        old_chain = try deserialize(old_chain_path) catch; nothing end
        exact = old_metadata isa NamedTuple &&
            get(old_metadata, :boundary_sha256, nothing) == context.boundary_sha256 &&
            get(old_metadata, :registry_fingerprint, nothing) == context.fs[:registry_fingerprint]
        exact && old_chain isa Chains || continue

        try
            validate_chain(old_chain, Int(context.fs[:n_teams]))
            size(old_chain, 1) == samples || error("legacy retained-draw mismatch")
            size(old_chain, 3) == 4 || error("legacy chain-count mismatch")
            BayesianFootball.Training.save_split_checkpoint(
                checkpoint_dir,
                index,
                (old_chain, context.meta),
            )
            push!(migrated, context.x.fold)
        catch err
            @warn "did not migrate invalid legacy chain for fold $(context.x.fold)" exception=(err, catch_backtrace())
        end
    end

    valid = count(eachindex(contexts)) do index
        path = joinpath(checkpoint_dir, _checkpoint_name(index))
        !isnothing(valid_native_checkpoint(path, contexts[index], samples, validate_chain))
    end
    return (valid=valid, total=length(contexts), recovered, migrated, invalidated)
end

function stage8_progress!(outdir, inventory)
    states = Symbol[]
    for item in inventory
        fold_dir = _fold_directory(outdir, item.fold)
        state = if isfile(joinpath(fold_dir, "fold_result.jls"))
            :pass
        elseif isfile(joinpath(fold_dir, "hard_gate_failure.jls"))
            :hardfail
        elseif isfile(joinpath(fold_dir, "error.jls"))
            :error
        else
            :pending
        end
        push!(states, state)
    end
    summary = (
        updated_utc = string(now(UTC)),
        pass = count(==(:pass), states),
        hardfail = count(==(:hardfail), states),
        error = count(==(:error), states),
        pending = count(==(:pending), states),
        total = length(states),
    )
    atomic_replace_serialize(joinpath(outdir, "progress.jls"), (; summary, states))
    return summary
end

end
