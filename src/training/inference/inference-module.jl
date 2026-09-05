# src/training/inference/inference-module.jl
#
# The unified inference and fit lifecycle.
#
#     split → sample → audit → extract → Fit
#
# One transaction, one container, one audit that cannot be skipped. Graduated from
# `current_development/07_unified_inference_framework/`.
#
# WHAT THIS IS FOR, NEXT TO WHAT ALREADY EXISTS
#
# `Training.train` loops over splits calling `Samplers.run_sampler`;
# `Experiments.run_experiment` is a stopwatch around `Training.train`. Neither does
# anything the other could not, and the split costs three things: nested configs
# (`ExperimentConfig` wraps `TrainingConfig` purely to carry a sampler and a flag),
# nested results (one fold's chain is `res.training_results.items[i][1]` — four hops),
# and work done twice (`extract_oos_predictions` and `Diagnostics.extract_chains` each
# re-derive the boundaries and rebuild the feature sets from a live `DataStore`, because
# the run threw them away). And the convergence audit is OPTIONAL: a run whose R-hat is
# 1.4 saves, loads, prices and stakes exactly like one that converged.
#
#     fit = fit_model(ds, FitConfig(name = "run", model = m, splitter = s, sampler = nuts))
#
#     fit[1].chain                  # the fold's chain — one hop
#     fit.latents.λ_home            # typed OOS posterior, extracted by the run
#     fit.diagnostics.passed        # audited by the run; a field, not a call
#     fit.metadata.git_commit       # "c44af652-dirty"
#     save_fit(fit)
#
# The audit and the extraction happen while the feature sets are still in scope, so the
# second derivation and its drift guard are gone. `audit_convergence` and `load_fit` need
# no `DataStore` at all.
#
# BACKWARD COMPATIBILITY. `src/experiments/` and the legacy `Training.train` path are
# untouched and keep working exactly as they did. `compat.jl` adds a conversion in each
# direction — `fit_from_experiment` and `experiment_from_fit` — so a result can move
# between the two worlds without being re-run. A `Fit` additionally answers
# `res.training_results.items[i][1]`, `res.vocabulary` and
# `res.config.training_config.sampler` for call sites that read those directly.
#
# NOT GRADUATED HERE. The prototype's in-game FIT lifecycle (`InGameFit`,
# `InGameFitConfig`) and its live MARKET pricer (`LiveBook`, `price_live_market!`) stay
# in `current_development/`: the pricer belongs in `src/predictions/` rather than here,
# and the in-game lifecycle waits on `current_development/inplay_scottish` settling a
# single integrator convention. The zero-allocation RATE solver, which those depend on
# and which has no such open question, is in `ingame.jl`.

module Inference

using Base.Threads
using CodecZstd
using DataFrames
using Dates
using JLD2
using JSON3
using LibPQ
using MCMCChains
using Printf
using ProgressMeter
using Random
using SHA
using Serialization
using Statistics
using UUIDs

using ...TypesInterfaces
using ...Data
using ...Features
using ...Models
using ...Samplers

# 1. Containers. Every named data shape the lifecycle produces.
include("types.jl")

# 2. Telemetry. Depends on the containers and on `MCMCChains`, and on nothing else —
#    which is what makes it callable on a fit loaded from disk with no database.
include("convergence.jl")

# 3. Execution. Sampling, fold dispatch, latent extraction, checkpoints, `fit_model`.
include("engine.jl")

# 4. Persistence. Atomic writes, the scannable sidecar, discovery.
include("io.jl")
include("db_storage.jl")
include("extension.jl")

# 5. The zero-allocation live rate solver.
include("ingame.jl")

# 6. The bidirectional bridge to `Experiments.ExperimentResults`. Last, because it
#    resolves sibling modules that are not loaded until after `Training` is.
include("compat.jl")


# --- The lifecycle ------------------------------------------------------------
export Fit, FoldFit, FitConfig, FitMetadata
export fit_model, sample_fold, run_folds, ReplaySampler, default_save_path
export fold_chains, fold_metas, total_draws, fit_name, format_elapsed, git_commit_id

# --- The model split ----------------------------------------------------------
export AbstractPreGameModel, AbstractInGameModel, is_pregame, is_ingame

# --- Fold dispatch ------------------------------------------------------------
export AbstractExecution, AutoExecution, SequentialExecution, ThreadedExecution,
       QueuedExecution, resolve_execution, execution_from_strategy

# --- Convergence telemetry ----------------------------------------------------
export ConvergenceThresholds, ConvergenceSummary, FoldConvergence,
       audit_convergence, audit_fold, summarise_convergence, convergence_table,
       diagnostics_line, bfmi

# --- Persistence --------------------------------------------------------------
export AbstractStorageBackend, FileStorage, PostgresStorage, DualStorage
export save_fit, load_fit, load_fits, list_fits, read_fit_meta,
       save_latents, load_latents, atomic_write, fit_meta, fit_config_json,
       ensure_schema!, config_hash, compress_draws, decompress_draws,
       save_config, save_model, save_splitter, save_sampler,
       save_book_spec, save_policy_spec, save_calibrator,
       load_model, load_splitter, load_sampler, load_fit_config,
       load_book_spec, load_policy_spec, load_calibrator, load_portfolio_spec, list_configs,
       explore_experiments, search_configs, show_config,
       preview_extension, extend_fit

# --- Latents ------------------------------------------------------------------
export merge_latents, extract_run_latents

# --- In-game live rates -------------------------------------------------------
export MatchState, kickoff_state, NHPPIntensityModel,
       IngameRatesWorkspace, LiveMatchRates,
       build_ingame_workspace, alloc_live_rates,
       solve_ingame_rates, solve_ingame_rates!,
       workspace_n_draws, workspace_n_bins, n_time_bins

# --- Legacy bridge ------------------------------------------------------------
export upgrade_to_fit, fit_from_experiment, experiment_from_fit, legacy_strategy,
       LegacyTrainingConfig, LegacyTrainingResults, legacy_training_results,
       legacy_checkpointing

end # module Inference
