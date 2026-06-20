# src/experiments/presets.jl

using ..Data
using ..Training
using ..Samplers
using ..Models

export create_experiment_task


"""
    create_experiment_task(ds::DataStore, model, name::String, save_dir::String; kwargs...)

Creates an `ExperimentTask` containing both the dataset and the configuration, ready to be run.
Defaults are provided for all major parameters to ensure a clean user experience.

# Arguments
- `ds::DataStore`: The dataset to run the experiment on.
- `model::AbstractFootballModel`: The Bayesian model engine.
- `name::String`: Name label for the experiment (e.g. "ab_std_player").
- `save_dir::String`: Directory path to save the experiment artifacts.

# Keywords
- `target_seasons`: A vector of season strings (e.g., `["2025", "2026"]` or `["24/25"]`). Must match the format of the `DataStore`.
- `history_seasons`: Number of historical seasons to include for burn-in (default `2`).
- `dynamics_col`: The time column used for grouped CV (default `:match_month`).
- `warmup_period`: Warmup period for evaluating early season matches (default `0`).
- `samples`: NUTS sampling steps (default `500`).
- `chains`: NUTS chains (default `4`).
- `warmup`: NUTS warmup steps (default `200`).
- `accept_rate`: NUTS acceptance rate (default `0.65`).
- `parallel`: Run cross-validation folds in parallel (default `true`).
"""
function create_experiment_task(
    ds::Data.DataStore, 
    model, 
    name::String, 
    save_dir::String;
    # --- Splitter Overrides ---
    target_seasons::Vector{String},
    history_seasons::Int = 2,
    dynamics_col::Symbol = :match_month,
    warmup_period::Int = 0,
    stop_early::Bool = false,
    
    # --- Sampler Defaults ---
    samples::Int = 500,
    chains::Int = 4,
    warmup::Int = 200,
    accept_rate::Float64 = 0.65,
    max_depth::Int = 10,
    show_progress = false, # :perchain
    
    # --- Training/Execution Defaults ---
    parallel::Bool = true,
    max_concurrent_splits::Int = 4,
    use_queue::Bool = true,
    max_concurrent_tasks::Int = -1
)

    cv_config = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons = target_seasons,
        history_seasons = history_seasons,
        dynamics_col = dynamics_col,
        warmup_period = warmup_period,
        stop_early = stop_early
    )

    if use_queue
        sampler_conf = Samplers.QueuedNUTSConfig(
            n_samples = samples, 
            n_chains = chains, 
            n_warmup = warmup, 
            accept_rate = accept_rate, 
            max_depth = max_depth,  
            initialisation = Samplers.UniformInit(-2.0, 2.0),
            show_progress = show_progress 
        )
        
        actual_tasks = max_concurrent_tasks == -1 ? Threads.nthreads() : max_concurrent_tasks
        train_cfg = Training.Independent(
            parallel = parallel,
            max_concurrent_tasks = actual_tasks
        )
    else
        sampler_conf = Samplers.NUTSConfig(
            samples, 
            chains, 
            warmup, 
            accept_rate, 
            max_depth,  
            Samplers.UniformInit(-2, 2),
            show_progress #  display the chain progress 
        )

        train_cfg = Training.Independent(
            parallel = parallel,
            max_concurrent_splits = max_concurrent_splits
        )
    end
    
    training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

    config = ExperimentConfig(
        name = name,
        model = model, 
        splitter = cv_config,
        training_config = training_config,
        save_dir = save_dir
    )

    return ExperimentTask(ds, config)
end
