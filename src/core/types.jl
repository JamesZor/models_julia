# src/core/types.jl
module Types

using Distributions: Distribution
using MCMCChains: Chains

export AbstractModel, AbstractPredictor, AbstractEvaluator, AbstractDataTransform
export MaherModel, MaherPredictor
export train, predict

# Define your contract/interface using abstract types
abstract type AbstractModel end
abstract type AbstractPredictor end
abstract type AbstractEvaluator end
abstract type AbstractDataTransform end

# Required methods for each abstract type (interface contract)
function train(model::AbstractModel, data) 
    error("train not implemented for $(typeof(model))")
end

function predict(model::AbstractModel, data)
    error("predict not implemented for $(typeof(model))")
end

# Concrete implementations
struct MaherModel <: AbstractModel
    priors::Dict{Symbol, Distribution}
    extensions::Vector{Symbol}
end

struct MaherPredictor <: AbstractPredictor
    chain::Chains
    data_mapping::Dict
end

# Now you MUST implement the interface
function train(model::MaherModel, data)
    # Actual implementation - placeholder for now
    println("Training MaherModel...")
    return nothing
end

end # module Types
