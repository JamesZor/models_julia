using JLD2
# Save entire experiment as single file
function save_experiment(result::ExperimentResult, config::ExperimentConfig, filepath::String)
    # Create directory if it doesn't exist
    dir = dirname(filepath)
    !isdir(dir) && mkpath(dir)
    
    @save filepath result config
    println("Experiment saved to: $filepath")
    return filepath
end

# Load experiment
function load_experiment(filepath::String)
    @load filepath result config
    return result, config
end
