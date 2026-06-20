# src/experiments/runner.jl

using Dates
using DataFrames
using JLD2, JSON3

# Internal Imports
using ..Data
using ..Features
using ..Training
using ..Experiments: ExperimentConfig, ExperimentResults

export run_experiment, save_experiment, load_experiment, list_experiments

# ==============================================================================
# 0. TIME FORMATTING HELPER
# ==============================================================================
function _format_time(seconds::Float64)
    if seconds < 60
        return "$(round(seconds, digits=1))s"
    elseif seconds < 3600
        mins = floor(Int, seconds / 60)
        secs = round(Int, seconds % 60)
        return "$(mins)m $(secs)s"
    else
        hrs = floor(Int, seconds / 3600)
        mins = floor(Int, (seconds % 3600) / 60)
        return "$(hrs)h $(mins)m"
    end
end

# ==============================================================================
# 1. EXPERIMENT ORCHESTRATION
# ==============================================================================
# new Architecture

function run_experiment(task::ExperimentTask)
    return run_experiment(task.ds, task.config)
end

function run_experiment(data_store::Data.DataStore, config::ExperimentConfig)
    _log_header(config.name)
    
    # START TIMER
    start_time = time()

    # 1. Splits
    _log_step(2, "Generating Data Splits")
    boundaries_with_meta = Data.create_id_boundaries(data_store, config.splitter)
    _log_info("Generated $(length(boundaries_with_meta)) splits")

    # 2. Features
    _log_step(3, "Building Feature Sets")
    feature_sets = Features.create_features(
            boundaries_with_meta,
            data_store,
            config.model,                
            config.splitter.dynamics_col
        )

    # 3. Training
    _log_step(4, "Executing Training Strategy")
    training_results = Training.train(
        config.model, 
        config.training_config, 
        feature_sets
    )

    # END TIMER
    elapsed_time = time() - start_time
    time_str = _format_time(elapsed_time)
    
    push!(config.tags, "time:$time_str")

    _log_footer()
    _log_info("Experiment completed in $time_str")

    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    default_path = joinpath(config.save_dir, "$(config.name)_$(timestamp)")

    return ExperimentResults(
        config,
        training_results,
        nothing,
        default_path
    )
end

# ==============================================================================
# 2. PERSISTENCE (Saving)
# ==============================================================================

function save_experiment(results::ExperimentResults; path=nothing, quiet=false)
    target_path = isnothing(path) ? results.save_path : path
    mkpath(target_path)

    # 1. Save Binary (This works fine with Inf)
    jldsave(joinpath(target_path, "results.jld2"); results)

    # 2. Save Config (JSON) 
    safe_config = Dict(
        :name => results.config.name,
        :save_dir => results.config.save_dir,
        :model => string(results.config.model),     
        :splitter => results.config.splitter,
        :training_config => results.config.training_config,
        :tags => results.config.tags # Make sure tags get written to JSON!
    )

    open(joinpath(target_path, "config.json"), "w") do io
        JSON3.pretty(io, safe_config)
    end

    # Extract time from tags for the meta sidecar
    time_taken = "N/A"
    for tag in results.config.tags
        if startswith(tag, "time:")
            time_taken = replace(tag, "time:" => "")
            break
        end
    end

    # 3. Save Metadata Sidecar
    meta = Dict(
        :name => results.config.name,
        :model => string(nameof(typeof(results.config.model))),
        :splitter => string(nameof(typeof(results.config.splitter))),
        :sampler => string(nameof(typeof(results.config.training_config.sampler))),
        :timestamp => basename(target_path),
        :time_taken => time_taken # Easy extraction for the UI
    )
    
    open(joinpath(target_path, "meta.json"), "w") do io
        JSON3.pretty(io, meta)
    end

    if !quiet
        printstyled("\n[IO] Artifacts saved to: ", color=:green, bold=true)
        println(target_path)
    end
end

# ==============================================================================
# 3. DISCOVERY & LOADING
# ==============================================================================

function list_experiments(dir::String; data_dir::String="./data")
    base_dir = joinpath(data_dir, dir)

    if !isdir(base_dir)
        println("Directory not found: $base_dir")
        return String[]
    end

    subdirs = filter(isdir, readdir(base_dir, join=true))
    # Sort by modification time (newest first)
    sort!(subdirs, by=mtime, rev=true)

    if isempty(subdirs)
        println("No experiments found in $base_dir")
        return String[]
    end

    println("\n Experiments in: $base_dir")
    println("="^125)
    println(
        rpad("IDX", 4), " | ", 
        rpad("NAME", 25), " | ", 
        rpad("MODEL", 20), " | ", 
        rpad("SPLITTER", 18), " | ", 
        rpad("SAMPLER", 15), " | ", 
        rpad("TIME", 10), " | ", 
        "PATH ID"
    )
    println("-"^125)

    for (i, path) in enumerate(subdirs)
        m = _read_meta(path)

        idx_str = "[$i]"
        name_s = length(m.name) > 23 ? m.name[1:23] * ".." : m.name
        model_s = length(m.model) > 18 ? m.model[1:18] * ".." : m.model
        split_s = length(m.splitter) > 16 ? m.splitter[1:16] * ".." : m.splitter
        samp_s = length(m.sampler) > 13 ? m.sampler[1:13] * ".." : m.sampler
        time_s = length(m.time_taken) > 10 ? m.time_taken[1:10] : m.time_taken

        c = i == 1 ? :white : :light_black
        
        printstyled(rpad(idx_str, 4), " | ", color=:cyan, bold=(i==1))
        printstyled(rpad(name_s, 25), " | ", color=c, bold=(i==1))
        printstyled(rpad(model_s, 20), " | ", color=c)
        printstyled(rpad(split_s, 18), " | ", color=c)
        printstyled(rpad(samp_s, 15), " | ", color=c)
        printstyled(rpad(time_s, 10), " | ", color=:yellow, bold=false) # Pop the time in yellow!
        println(basename(path))
    end
    println("="^125, "\n")

    return subdirs
end

function load_experiment(path::String)
    file_path = endswith(path, ".jld2") ? path : joinpath(path, "results.jld2")
    if !isfile(file_path)
        error("Results file not found: $file_path")
    end
    printstyled("Loading: ", color=:green)
    println(basename(dirname(file_path)))
    
    data = load(file_path)
    return data["results"]
end

function load_experiment(experiment_list::Vector{String}, index::Int)
    if index < 1 || index > length(experiment_list)
        error("Index $index out of bounds.")
    end
    return load_experiment(experiment_list[index])
end


function load_experiments(saved_folders::Vector{String})
  loaded_results = Vector{ExperimentResults}([])
  for folder in saved_folders
      try
          res = load_experiment(folder)
          push!(loaded_results, res)
      catch e
          @warn "Could not load $folder: $e"
      end
  end

  if isempty(loaded_results)
      error("No results loaded! Did you run runner.jl?")
  end

  return loaded_results

end


# ==============================================================================
# 4. INTERNAL HELPERS
# ==============================================================================

    # Tell Julia how to upgrade the old JLD2.ReconstructedStatic into a modern AbstractExecutionStrategy
function Base.convert(::Type{Training.AbstractExecutionStrategy}, rs::JLD2.ReconstructedStatic)
	# Check if the reconstructed object was originally an "Independent" struct
	if typeof(rs).parameters[1] == Symbol("BayesianFootball.Training.Independent")

            # Safely extract the old fields (using defaults just in case)
            parallel = hasproperty(rs, :parallel) ? rs.parallel : false
            max_splits = hasproperty(rs, :max_concurrent_splits) ? rs.max_concurrent_splits : 1

            # Return the new, modern version of the struct (supplying a default for the new field)
            return BayesianFootball.Training.Independent(
		parallel = parallel,
		max_concurrent_splits = max_splits,
		max_concurrent_tasks = Threads.nthreads()
            )
	end

	# If it's something else, throw the normal error
	throw(MethodError(Base.convert, (BayesianFootball.Training.AbstractExecutionStrategy, rs)))
    end

function _read_meta(path)
    meta_path = joinpath(path, "meta.json")
    default = (name=basename(path), model="?", splitter="?", sampler="?", time_taken="N/A")

    if isfile(meta_path)
        try
            # Read meta and dynamically provide "N/A" if the JSON is from an older run
            cfg = JSON3.read(read(meta_path, String))
            return (
                name = get(cfg, :name, default.name),
                model = get(cfg, :model, default.model),
                splitter = get(cfg, :splitter, default.splitter),
                sampler = get(cfg, :sampler, default.sampler),
                time_taken = get(cfg, :time_taken, "N/A")
            )
        catch
            return default
        end
    end
    
    # Fallback to config.json if needed
    config_path = joinpath(path, "config.json")
    if isfile(config_path)
        try
            cfg = JSON3.read(read(config_path, String))
            
            # Check if older config.json has the time tag
            time_tag = "N/A"
            if haskey(cfg, :tags)
                for tag in cfg.tags
                    if startswith(tag, "time:")
                        time_tag = replace(tag, "time:" => "")
                    end
                end
            end

            return (
                name = get(cfg, :name, default.name),
                model = "Legacy Config",
                splitter = "?",
                sampler = "?",
                time_taken = time_tag
            )
        catch
        end
    end
    return default
end

function _log_header(name)
    printstyled("\n>> EXPERIMENT: ", color=:magenta, bold=true)
    printstyled(name, "\n", color=:white, bold=true)
    println("-"^60)
end

function _log_step(step_num, msg)
    printstyled(" [$step_num] ", color=:cyan, bold=true)
    println(msg, "...")
end

function _log_info(msg)
    printstyled("     > ", color=:light_black)
    println(msg)
end

function _log_footer()
    println("-"^60)
    printstyled("DONE.\n", color=:green, bold=true)
end
