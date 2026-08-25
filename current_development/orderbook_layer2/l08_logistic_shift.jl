# current_development/orderbook_layer2/l08_logistic_shift.jl

using BayesianFootball
using DataFrames, Statistics, Serialization

const PF = BayesianFootball.Portfolio
const MD = BayesianFootball.MatchDay
const DD = BayesianFootball.Data
const EE = BayesianFootball.Experiments

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
    isfile(p) || error("Missing pinned store $p. Make sure you SCP'd the files from mcmc-beast!")
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
# ds79, expr79 = load_engine_data("ire79", "l2_ire79_sup40_sw40")
# println("Successfully loaded ds79 and expr79!")


# ===================================================================
# 2. Block 1.1: Multi-Class Softmax Pooling
# ===================================================================

"""
    log_linear_pool(p_model::Vector{Float64}, p_market::Vector{Float64}, w::Float64) -> Vector{Float64}

Applies a softmax (log-linear) blend between the model's probabilities and the market's probabilities.
- `w = 1.0` returns exactly `p_model`
- `w = 0.0` returns exactly `p_market`
"""
function log_linear_pool(p_model::Vector{Float64}, p_market::Vector{Float64}, w::Float64)
    # Step 1: Prevent log(0) by flooring at a tiny value
    # Step 2: Calculate the blended logits: w * log(model) + (1-w) * log(market)
    # Step 3: Exponentiate the logits
    # Step 4: Normalize so they sum to 1.0
end
