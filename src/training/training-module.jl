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
export AbstractStorageBackend, FileStorage, PostgresStorage, DualStorage
export save_fit, load_fit, load_fits, list_fits, read_fit_meta,
       save_latents, load_latents, ensure_schema!, config_hash,
       compress_draws, decompress_draws,
       save_config, save_model, save_splitter, save_sampler,
       save_book_spec, save_policy_spec,
       load_model, load_splitter, load_sampler, load_fit_config,
       load_book_spec, load_policy_spec, load_portfolio_spec, list_configs,
       explore_experiments, search_configs, show_config,
       preview_extension, extend_fit
export merge_latents, extract_run_latents
export MatchState, kickoff_state, NHPPIntensityModel,
       IngameRatesWorkspace, LiveMatchRates,
       build_ingame_workspace, alloc_live_rates,
       solve_ingame_rates, solve_ingame_rates!
export upgrade_to_fit, fit_from_experiment, experiment_from_fit

end # module
