# Main execution script
using BayesianFootball

function main()
    println("Starting BayesianFootball experiment...")
    
    # Load data
    data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
    data_store = DataStore(data_files)
    println("Loaded $(nrow(data_store.matches)) matches")
    
    # Configure experiment manually (since create_experiment_config might not be complete)
    model_config = ModelConfig(
        basic_maher_model_raw,
        feature_map_basic_maher_model
    )
    
    cv_config = TimeSeriesSplitsConfig(
        ["20/21", "21/22", "22/23", "23/24"],
        "24/25",
        :round
    )
    
    sample_config = ModelSampleConfig(100, true)  # Start with fewer steps for testing
    
    mapping_funcs = MappingFunctions(create_list_mapping)
    
    config = ExperimentConfig(
        "maher_basic_test",
        model_config,
        cv_config,
        sample_config,
        mapping_funcs
    )
    
    # Run experiment
    println("Running experiment: $(config.name)")
    result = train_all_splits(
        data_store,
        config.cv_config,
        config.model_config,
        config.sample_config,
        config.mapping_funcs;
        parallel = false  # Start with sequential for debugging
    )
    
    println("Experiment completed!")
    println("Total time: $(round(result.total_time, digits=2)) seconds")
    println("Number of splits: $(length(result.chains_sequence))")
    
    # Save results
    save_path = "./experiments"
    !isdir(save_path) && mkpath(save_path)
    
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename = "$(config.name)_$(timestamp).jld2"
    filepath = joinpath(save_path, filename)
    
    save_experiment(result, config, filepath)
    
    return result
end

# Execute if run directly
if abspath(PROGRAM_FILE) == @__FILE__
    result = main()
end
