# src/training/training-module.jl

module Training

using Turing
using ProgressMeter
using Printf
using Base.Threads
using ..Samplers
using ..Models
using ..Features
using ..Models.PreGame: build_turing_model

# 1. Configuration & Persistence
include("./types.jl")
include("./checkpointing.jl")

# 2. Strategies
include("./strategies/independent.jl")
# include("strategies/sequential.jl") # Placeholder for future
include("./method.jl")

# 3. The unified inference & fit lifecycle (split → sample → audit → extract → Fit).
#    Additive: everything above keeps working unchanged, and `Inference.compat` bridges
#    a `Fit` to and from `Experiments.ExperimentResults`.
include("./inference/inference-module.jl")
using .Inference

export Inference

# The lifecycle surface, re-exported so `Training.fit_model` and `BayesianFootball.Fit`
# both resolve.
export Fit, FoldFit, FitConfig, FitMetadata
export fit_model, sample_fold, run_folds, ReplaySampler
export fold_chains, fold_metas, total_draws, fit_name
export ConvergenceThresholds, ConvergenceSummary, FoldConvergence,
       audit_convergence, audit_fold, summarise_convergence, convergence_table
export AbstractExecution, AutoExecution, SequentialExecution, ThreadedExecution,
       QueuedExecution
export save_fit, load_fit, load_fits, list_fits, read_fit_meta,
       save_latents, load_latents
export merge_latents, extract_run_latents
export MatchState, kickoff_state, NHPPIntensityModel,
       IngameRatesWorkspace, LiveMatchRates,
       build_ingame_workspace, alloc_live_rates,
       solve_ingame_rates, solve_ingame_rates!
export upgrade_to_fit, fit_from_experiment, experiment_from_fit

end # module
