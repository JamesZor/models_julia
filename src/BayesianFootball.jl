# src/BayesianFootball.jl

module BayesianFootball

# --- Environment Setup ---
# Loads .env (e.g. BF_DB_URL) into ENV at module init. DotEnv v1's `config` does
# NOT mutate ENV; `load!(ENV, path)` is the call that actually populates it.
using DotEnv
const env_path = joinpath(pkgdir(@__MODULE__), ".env")
if isfile(env_path)
    DotEnv.load!(ENV, env_path)
end
# -------------------------

# 1. Interfaces contains all shared types and contracts.
include("./types-interfaces.jl") #
using .TypesInterfaces

include("./MyDistributions/MyDistributions-module.jl")
export MyDistributions


# 2. Data is self-contained.
# include("data/data-module.jl") #
# FIX: dev test 
include("./Data/data-module.jl")

# 4. Features depends on Data, TypesInterfaces, and Models.
include("features/features-module.jl") #

# 3. Models depends only on TypesInterfaces.
include("models/models-module.jl") #


# 5. Samplers provides core sampling algorithms.
include("samplers/samplers-module.jl") # *** ADDED RENAMED MODULE ***

# 6. Training orchestrates the training process using Models, Features, Samplers.
include("training/training-module.jl") # *** ADDED NEW MODULE ***

# 7. Other modules
include("./experiments/experiment-module.jl") #
include("./predictions/predictions-module.jl") #

include("./Calibration/calibration-module.jl")


include( "./signals/signals-module.jl")

include("./evaluation/evaluation-module.jl")

include("./synthetic/synthetic-data-module.jl")

include("./backtesting/backtesting-module.jl")

# Portfolio depends on BackTesting's metric interface, so it must come after it.
include("./Portfolio/portfolio-module.jl")

# Export the main modules and key functions/types for users
# *** UPDATED EXPORTS ***
export Data, Features, Models, Samplers, Training, Experiments, Predictions, Markets, Calibration, BackTesting, Evaluation, Portfolio
export AbstractFootballModel, Vocabulary, FeatureSet, required_mapping_keys

# Maybe export core config types too?
export NUTSConfig, ADVIConfig, MAPConfig # From Samplers
export TrainingConfig, Independent, SequentialPriorUpdate # From Training

# 
using .Data: Markets 
export Markets


# --- ADDED DEV HELPERS ---
include("./dev_helpers/dev-helpers.jl")
# --- EXPORT HELPER ---
export load_scottish_data

end
