# Main execution script
using BayesianFootball

function main()
    # Load data
    data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
    data_store = DataStore(data_files)
    
    # Configure experiment
    config = ExperimentConfig(
        "maher_basic_v1",
        ModelConfig(basic_maher_model_raw, feature_map_basic_maher_model),
        TimeSeriesSplitsConfig(["20/21", "21/22", "22/23", "23/24"], "24/25", :round),
        ModelSampleConfig(2000, true),
        MappingFunctions(create_list_mapping)
    )
    
    # Run experiment
    result = run_experiment(
        config,
        data_store,
        parallel = true,
        save_path = "./experiments"
    )
    
    return result
end

# Execute if run directly
if abspath(PROGRAM_FILE) == @__FILE__
    result = main()
end
