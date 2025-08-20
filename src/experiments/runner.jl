# src/experiments/runner.jl
# Experiment orchestration and management

"""
    run_experiment(config, data_store; kwargs...)

Execute a complete experiment with the given configuration.
"""
function run_experiment(
    config::ExperimentConfig,
    data_store::DataStore;
    parallel::Bool = true,
    save_path::Union{String, Nothing} = nothing,
    verbose::Bool = true
)::ExperimentResult
    
    verbose && println("="^60)
    verbose && println("Starting experiment: $(config.name)")
    verbose && println("="^60)
    verbose && println("Model: $(nameof(config.model_config.model))")
    verbose && println("CV: $(config.cv_config.base_seasons) -> $(config.cv_config.target_season)")
    verbose && println("MCMC steps: $(config.sample_config.steps)")
    verbose && println("-"^60)
    
    # Execute training pipeline
    result = train_all_splits(
        data_store,
        config.cv_config,
        config.model_config,
        config.sample_config,
        config.mapping_funcs;
        parallel = parallel
    )
    
    # Save if path provided
    if !isnothing(save_path)
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        filename = "$(config.name)_$(timestamp).jld2"
        filepath = joinpath(save_path, filename)
        save_experiment(result, config, filepath)
        verbose && println("Results saved to: $filepath")
    end
    
    # Print diagnostics summary
    if verbose
        print_diagnostics_summary(result)
    end
    
    return result
end

"""
    run_batch_experiments(configs, data_store; kwargs...)

Run multiple experiments in sequence or parallel.
"""
function run_batch_experiments(
    configs::Vector{ExperimentConfig},
    data_store::DataStore;
    save_path::String = "./experiments",
    parallel_experiments::Bool = false,
    parallel_training::Bool = true
)::Vector{ExperimentResult}
    
    println("Running batch of $(length(configs)) experiments")
    
    if parallel_experiments
        # Run experiments in parallel (careful with memory!)
        results = ThreadsX.map(configs) do config
            run_experiment(
                config, 
                data_store,
                parallel = parallel_training,
                save_path = save_path,
                verbose = false
            )
        end
    else
        # Run experiments sequentially
        results = map(configs) do config
            run_experiment(
                config,
                data_store,
                parallel = parallel_training,
                save_path = save_path,
                verbose = true
            )
        end
    end
    
    return results
end

"""
    create_experiment_config(name, model_variant, cv_params, mcmc_params)

Factory function to create experiment configurations.
"""
function create_experiment_config(
    name::String,
    model_variant::Symbol,
    base_seasons::Vector{String},
    target_season::String,
    mcmc_steps::Int = 2000;
    round_col::Symbol = :round
)::ExperimentConfig
    
    # Select model configuration based on variant
    model_config = if model_variant == :maher_basic
        ModelConfig(
            basic_maher_model_raw,
            feature_map_basic_maher_model
        )
    elseif model_variant == :maher_league
        ModelConfig(
            maher_model_with_league,
            feature_map_maher_league
        )
    # Add more model variants here
    else
        error("Unknown model variant: $model_variant")
    end
    
    cv_config = TimeSeriesSplitsConfig(
        base_seasons,
        target_season,
        round_col
    )
    
    sample_config = ModelSampleConfig(mcmc_steps, true)
    
    mapping_funcs = MappingFunctions(create_list_mapping)
    
    return ExperimentConfig(
        name,
        model_config,
        cv_config,
        sample_config,
        mapping_funcs
    )
end

"""
    print_diagnostics_summary(result)

Print a summary of convergence diagnostics for the experiment.
"""
function print_diagnostics_summary(result::ExperimentResult)
    println("\n" * "="^60)
    println("Diagnostics Summary")
    println("="^60)
    
    for (idx, chains) in enumerate(result.chains_sequence)
        println("\nSplit $idx ($(chains.round_info)):")
        println("  Samples: $(chains.n_samples)")
        
        # Get diagnostics for HT model
        ht_ess = mean(ess(chains.ht))
        ht_rhat = maximum(rhat(chains.ht))
        println("  HT - Mean ESS: $(round(ht_ess, digits=1)), Max R̂: $(round(ht_rhat, digits=3))")
        
        # Get diagnostics for FT model
        ft_ess = mean(ess(chains.ft))
        ft_rhat = maximum(rhat(chains.ft))
        println("  FT - Mean ESS: $(round(ft_ess, digits=1)), Max R̂: $(round(ft_rhat, digits=3))")
        
        # Warn if convergence issues
        if ht_rhat > 1.01 || ft_rhat > 1.01
            println("  ⚠ Warning: R̂ > 1.01, chains may not have converged")
        end
    end
    
    println("\nTotal training time: $(round(result.total_time/60, digits=2)) minutes")
    println("="^60)
end

"""
    create_comparison_experiments(base_name, data_store)

Create a set of experiments for model comparison.
"""
function create_comparison_experiments(
    base_name::String,
    base_seasons::Vector{String},
    target_season::String
)::Vector{ExperimentConfig}
    
    configs = ExperimentConfig[]
    
    # Different model variants
    for (variant, steps) in [
        (:maher_basic, 2000),
        (:maher_league, 2500),
        # Add more variants
    ]
        config = create_experiment_config(
            "$(base_name)_$(variant)",
            variant,
            base_seasons,
            target_season,
            steps
        )
        push!(configs, config)
    end
    
    return configs
end
