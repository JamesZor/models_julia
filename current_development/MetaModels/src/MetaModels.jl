# current_development/MetaModels/src/MetaModels.jl

module MetaModels

include("types.jl")
include("components/dynamics.jl")
include("components/teams.jl")
# Core Engine and Inference
include("engines/mixture_engine.jl")
include("engines/affine_engine.jl")
include("training/workflow.jl")
include("staking.jl")
include("utils.jl")
include("metrics.jl")

end
