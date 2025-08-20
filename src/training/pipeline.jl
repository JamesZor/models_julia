# src/training/pipeline.jl
# Core training pipeline implementation

"""
    train_all_splits(data_store, cv_config, model_config, sample_config, mapping_funcs; parallel=true)

Main training pipeline that applies the composed morphisms to all CV splits.
Returns an ExperimentResult containing all trained chains.
"""

function train_all_splits(
    data_store::DataStore,
    cv_config::TimeSeriesSplitsConfig,
    model_config::ModelConfig,
    sample_config::ModelSampleConfig,
    mapping_funcs::MappingFunctions;
    parallel::Bool = true
)::ExperimentResult
    
    start_time = time()
    
    # Create mapping once for all splits
    mapping = MappedData(data_store, mapping_funcs)
    
    # Create the composed training morphism
    training_morphism = compose_training_morphism(
        model_config,
        sample_config,
        mapping
    )
    
    # Generate splits
    splits = time_series_splits(data_store, cv_config)
    split_data = collect(splits)
    
    println("Training on $(length(split_data)) splits...")
    
    # Apply training morphism to all splits
    chains_sequence = if parallel
        println("Using parallel execution with $(Threads.nthreads()) threads")
        ThreadsX.map(split_data) do (data, info)
            println("  Training split: $info")
            training_morphism(data, info)
        end
    else
        println("Using sequential execution")
        map(split_data) do (data, info)
            println("  Training split: $info")
            training_morphism(data, info)
        end
    end
    
    # Hash config for tracking
    config_hash = hash((
        cv_config.base_seasons,
        cv_config.target_season,
        sample_config.steps,
        nameof(model_config.model)
    ))
    
    total_time = time() - start_time
    println("Training completed in $(round(total_time, digits=2)) seconds")
    
    return ExperimentResult(
        chains_sequence,
        mapping,
        config_hash,
        total_time
    )
end

"""
    sample_basic_maher_model(models, sample_config)

Execute MCMC sampling for both half-time and full-time models.
"""
function sample_basic_maher_model(
    models::BasicMaherModels,
    sample_config::ModelSampleConfig
)::ModelChain
    
    # Sample half-time model
    ht_chain = sample(
        models.ht, 
        NUTS(0.65),  # Acceptance rate target
        MCMCThreads(), 
        sample_config.steps, 
        4;  # Number of chains
        progress = sample_config.bar
    )
    
    # Sample full-time model
    ft_chain = sample(
        models.ft, 
        NUTS(0.65),
        MCMCThreads(), 
        sample_config.steps, 
        4;
        progress = sample_config.bar
    )
    
    return ModelChain(ht_chain, ft_chain)
end
