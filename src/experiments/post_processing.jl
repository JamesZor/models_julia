# src/experiments/post_process.jl

using DataFrames
using ProgressMeter
using Base.Threads
using Serialization
using ..Data
using ..Features
using ..Models
using ..TypesInterfaces: AbstractFootballModel

# We don't export here, we rely on the main module to export

# ==============================================================================
# 1. THE WRAPPER (Holy Trait Pattern)
# ==============================================================================

struct LatentStates
    df::DataFrame
    model::AbstractFootballModel # You can now store the specific model used
end

# 1. Standard Base methods
Base.getindex(ls::LatentStates, args...) = getindex(ls.df, args...)
Base.setindex!(ls::LatentStates, val, args...) = setindex!(ls.df, val, args...)
Base.size(ls::LatentStates) = size(ls.df)
Base.size(ls::LatentStates, i) = size(ls.df, i)
Base.show(io::IO, ls::LatentStates) = show(io, ls.df)

# 2. DataFrames specific methods (Use DataFrames.nrow, not Base.nrow)
DataFrames.nrow(ls::LatentStates) = nrow(ls.df)
DataFrames.ncol(ls::LatentStates) = ncol(ls.df)


# ==============================================================================
# 2. CONSTANTS & CACHING HELPERS
# ==============================================================================

const OOS_LATENTS_FILENAME = "oos_latents.jls"

"""
    has_oos_predictions(exp_results::ExperimentResults; path=nothing)::Bool
    has_oos_predictions(path::String)::Bool

Check if serialized out-of-sample latent predictions (`oos_latents.jls`) exist on disk.
"""
function has_oos_predictions(path::String)::Bool
    return isfile(joinpath(path, OOS_LATENTS_FILENAME))
end

function has_oos_predictions(exp_results::ExperimentResults; path=nothing)::Bool
    target_path = isnothing(path) ? exp_results.save_path : path
    return has_oos_predictions(target_path)
end

"""
    load_oos_predictions(exp_results::ExperimentResults; path=nothing)::Union{LatentStates, Nothing}
    load_oos_predictions(path::String)::Union{LatentStates, Nothing}

Load cached out-of-sample latent predictions from disk. Returns `nothing` if the file
is missing or corrupted.
"""
function load_oos_predictions(path::String)::Union{LatentStates, Nothing}
    target_file = joinpath(path, OOS_LATENTS_FILENAME)
    if !isfile(target_file)
        return nothing
    end
    try
        obj = Serialization.deserialize(target_file)
        if obj isa LatentStates
            return obj
        else
            @warn "Cached file at $target_file is not of type LatentStates (got $(typeof(obj)))"
            return nothing
        end
    catch e
        @warn "Failed to deserialize cached OOS predictions from $target_file: $e"
        return nothing
    end
end

function load_oos_predictions(exp_results::ExperimentResults; path=nothing)::Union{LatentStates, Nothing}
    target_path = isnothing(path) ? exp_results.save_path : path
    return load_oos_predictions(target_path)
end

"""
    save_oos_predictions(exp_results::ExperimentResults, latents::LatentStates; path=nothing)::String
    save_oos_predictions(path::String, latents::LatentStates)::String

Atomically serialize out-of-sample latent predictions to disk (`.tmp` -> `mv`).
Returns the saved file path.
"""
function save_oos_predictions(path::String, latents::LatentStates)::String
    mkpath(path)
    target_file = joinpath(path, OOS_LATENTS_FILENAME)
    tmp_file = target_file * ".tmp." * string(rand(UInt64), base=16)
    try
        Serialization.serialize(tmp_file, latents)
        mv(tmp_file, target_file; force=true)
    catch e
        if isfile(tmp_file)
            rm(tmp_file; force=true)
        end
        rethrow(e)
    end
    return target_file
end

function save_oos_predictions(exp_results::ExperimentResults, latents::LatentStates; path=nothing)::String
    target_path = isnothing(path) ? exp_results.save_path : path
    return save_oos_predictions(target_path, latents)
end


# ==============================================================================
# 2. THE BRIDGE (Relational DataStore Pipeline)
# ==============================================================================

"""
    extract_oos_predictions(ds::Data.DataStore, exp_results::ExperimentResults; force::Bool = false)::LatentStates

Extract out-of-sample latent predictions across splits.
- Default (`force = false`): If `oos_latents.jls` exists on disk, it is loaded (~0.04s).
  Otherwise, predictions are computed across MCMC splits, atomically saved to `oos_latents.jls`, and returned.
- Recompute (`force = true`): Force recomputation from MCMC chains and overwrite `oos_latents.jls`.
"""
function extract_oos_predictions(ds::Data.DataStore, exp_results::ExperimentResults; force::Bool = false)::LatentStates
    if !force && has_oos_predictions(exp_results)
        cached = load_oos_predictions(exp_results)
        if cached !== nothing
            return cached
        end
    end

    config = exp_results.config
    
    # 1. Reconstruct Context using the NEW Relational Pipeline
    boundaries_with_meta = Data.create_id_boundaries(ds, config.splitter)
    
    # Extract the array of tuples from the TrainingResults object
    results_array = exp_results.training_results.items
    n_splits = length(results_array)

    # Safety guard: prevent DataStore drift corruption
    if length(boundaries_with_meta) != n_splits
        error("DataStore drift detected: Splitter generated $(length(boundaries_with_meta)) boundaries from DataStore, but ExperimentResults has $(n_splits) training splits. Please re-run experiment or align DataStore.")
    end

    feature_sets = Features.create_features(
        boundaries_with_meta, 
        ds, 
        config.model,
        config.splitter
    )

    # 2. Extract (Multi-threaded across splits)
    split_dfs = Vector{DataFrame}(undef, n_splits)
    Threads.@threads for i in 1:n_splits
        split_dfs[i] = _process_split(
            ds, 
            config.model, 
            config.splitter, 
            feature_sets[i], # This is now the Tuple: (FeatureSet, MetaData)
            results_array[i] # This is now the Tuple: (Chains, MetaData)
        )
    end

    # 3. Consolidate
    combined_df = isempty(split_dfs) ? DataFrame() : vcat(split_dfs...)
    latents = LatentStates(combined_df, config.model)

    # 4. Atomic Cache Persistence
    if !isempty(exp_results.save_path)
        try
            save_oos_predictions(exp_results, latents)
        catch e
            @warn "Failed to cache OOS predictions to $(exp_results.save_path): $e"
        end
    end

    return latents
end

function _process_split(ds, model, splitter, feature_tuple, result_tuple)
    # feature_tuple[2] contains the SplitMetaData, which our new catch-all 
    # get_next_matches wrapper expects!
    df_to_predict = Data.get_next_matches(ds, feature_tuple, splitter)
    
    if isempty(df_to_predict)
        return DataFrame()
    end
    
    # Pass the actual FeatureSet (feature_tuple[1]) and MCMCChains (result_tuple[1])
    raw_preds = Models.PreGame.extract_parameters(
        model, 
        df_to_predict, 
        feature_tuple[1],  
        result_tuple[1]    
    )

    return _latent_state_dict_to_df(raw_preds)
end

function _latent_state_dict_to_df(raw_preds::Dict)::AbstractDataFrame
    match_ids = collect(keys(raw_preds))
    if isempty(match_ids); return DataFrame(); end
    
    first_val = raw_preds[match_ids[1]]
    cols = Dict{Symbol, Vector{Any}}(:match_id => match_ids)
    for p in keys(first_val)
        cols[p] = [raw_preds[id][p] for id in match_ids]
    end
    return DataFrame(cols)
end


