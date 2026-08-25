 # current_development/orderbook_layer2/research_questions_explore/rqs_loader_01_get_latents.jl

using BayesianFootball
using DataFrames, Statistics, Serialization

const PF = BayesianFootball.Portfolio
const MD = BayesianFootball.MatchDay
const DD = BayesianFootball.Data
const EE = BayesianFootball.Experiments
const PP = BayesianFootball.Predictions

const ENGINE_DIR = "./data/l2_ireland_engines"

# ===================================================================
# 1. Loading the Data
# ===================================================================

"""
    load_engine_data(tag, prefix) -> (DataStore, ExperimentResults)

Loads the pinned datastore and the latest trained experiment results for a given tag.
"""
function load_engine_data(tag::String, prefix::String)
    # Load the Pinned DataStore
    p = joinpath(ENGINE_DIR, "ds_$(tag).jls")
    isfile(p) || error("Missing pinned store $p")
    ds = deserialize(p)
    
    # Load the latest matching experiment results directory
    dirs = filter(d -> startswith(basename(d), prefix),
                  [joinpath(ENGINE_DIR, d) for d in readdir(ENGINE_DIR) if isdir(joinpath(ENGINE_DIR, d))])
    isempty(dirs) && error("No experiment matching $prefix found in $ENGINE_DIR")
    
    latest_dir = sort(dirs, by = mtime, rev = true)[1]
    expr = EE.load_experiment(latest_dir)
    
    return ds, expr
end

# We will uncomment this once you have SCP'd the files over!
ds79, expr79 = load_engine_data("ire79", "l2_ire79_sup40_sw40")
println("Successfully loaded ds79 and expr79!")

