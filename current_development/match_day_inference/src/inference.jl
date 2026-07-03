# current_development/match_day_inference/src/inference.jl

using DataFrames
using BayesianFootball
import BayesianFootball.Features: FeatureSet
import BayesianFootball.Experiments: LatentStates
import BayesianFootball.Predictions: model_inference

"""
    inject_matchday_features!(feature_set::FeatureSet, todays_ratings_map::Dict)

Injects matchday player ratings directly into the FeatureSet ratings map.
"""
function inject_matchday_features!(feature_set::FeatureSet, todays_ratings_map::Dict)
    # Get the existing lookup map (usually contains historical matches)
    ratings_map = feature_set.data[:player_ratings_map]
    
    # Merge today's matchday ratings into it
    for (mid, match_ratings) in todays_ratings_map
        ratings_map[mid] = match_ratings
    end
    
    println("└── [Features] Injected ratings map for ", length(todays_ratings_map), " matches into feature set.")
end

"""
    raw_preds_to_df(raw_preds)

Converts raw Turing extract_parameters dict to a DataFrame.
"""
function raw_preds_to_df(raw_preds)
    ids = collect(keys(raw_preds))
    cols = Dict{Symbol, Vector{Any}}(:match_id => ids)
    first_entry = raw_preds[ids[1]]
    
    for k in keys(first_entry)
        cols[Symbol(k)] = [raw_preds[i][k] for i in ids]
    end
    
    return DataFrame(cols)
end

"""
    find_experiment_path(saved_paths::Vector{String}, name_prefix::String)

Finds the experiment path whose folder name starts with `name_prefix` (e.g. the
`config.name` a task was created with, before the timestamp suffix). Used to pick a
specific model out of `Experiments.list_experiments` results instead of relying on
mtime-sort order when comparing multiple models trained in the same run.
"""
function find_experiment_path(saved_paths::Vector{String}, name_prefix::String)
    idx = findfirst(p -> startswith(basename(p), name_prefix), saved_paths)
    if idx === nothing
        error("No experiment found matching prefix '$name_prefix' among $(length(saved_paths)) saved paths.")
    end
    return saved_paths[idx]
end

"""
    compute_todays_matches_latents(ds::Data.DataStore, experiment, todays_matches::AbstractDataFrame, json_dir::String)

Runs the inference pipeline up to the LatentStates (per-match posterior parameter draws) for the
given matchday fixtures, WITHOUT collapsing to market probabilities. Exposed so callers that need
the raw λ draws (e.g. the unified structural-Kelly staking panel) can reuse the same posterior
without re-running the (expensive) MCMC parameter extraction.
"""
function compute_todays_matches_latents(ds::Data.DataStore, experiment, todays_matches::AbstractDataFrame, json_dir::String)
    println("└── [Inference] Starting inference pipeline...")

    model = experiment.config.model
    tracker = model.player_ratings_feature.tracker

    # 1. Build Ratings Map
    todays_ratings_map = build_matchday_ratings_map(ds, tracker, todays_matches, json_dir)

    # 2. Extract History/Target splits and create features
    boundaries_with_meta = BayesianFootball.Data.create_id_boundaries(ds, experiment.config.splitter)
    feature_collection = BayesianFootball.Features.create_features(boundaries_with_meta, ds, model)

    # 3. Extract the last training split indices, chain, and features
    last_split_idx = length(experiment.training_results)
    chain = experiment.training_results[last_split_idx][1]
    feature_set = feature_collection[last_split_idx][1]

    # 4. Inject today's matchday ratings
    inject_matchday_features!(feature_set, todays_ratings_map)

    # 5. Extract parameters for today's matches
    println("└── [MCMC] Extracting parameters across posterior chain samples...")
    raw_preds = BayesianFootball.Models.PreGame.extract_parameters(
        model, todays_matches, feature_set, chain
    )

    return LatentStates(raw_preds_to_df(raw_preds), model)
end

"""
    compute_todays_matches_pdds(ds::Data.DataStore, experiment, todays_matches::AbstractDataFrame, json_dir::String)

Runs the full inference pipeline for the given matchday fixtures and returns the PPD. Thin wrapper
around `compute_todays_matches_latents` + `model_inference`.
"""
function compute_todays_matches_pdds(ds::Data.DataStore, experiment, todays_matches::AbstractDataFrame, json_dir::String)
    latents = compute_todays_matches_latents(ds, experiment, todays_matches, json_dir)

    # Generate Posterior Predictive Distribution (PPD)
    println("└── [Predictions] Running model inference (PPD simulation)...")
    ppd = model_inference(latents)

    println("└── [Success] Match day inference complete!")
    return ppd
end
