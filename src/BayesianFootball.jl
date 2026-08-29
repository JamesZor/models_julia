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

# MatchDay hands its output to Portfolio.stake_sheet, so it must come after Portfolio.
include("./MatchDay/matchday-module.jl")

# Export the main modules and key functions/types for users
# *** UPDATED EXPORTS ***
export Data, Features, Models, Samplers, Training, Experiments, Predictions, Markets, Calibration, BackTesting, Evaluation, Portfolio, MatchDay
export AbstractFootballModel, Vocabulary, FeatureSet, required_mapping_keys

using .Models: AbstractPosteriorLatents, CountLatents, RecombLatents, SmileLatents,
               n_matches, n_draws, n_strikes, latent_match_ids, latent_matrices,
               match_index, latent_bytes, latent_allocations, observation_family,
               extract_latents, latent_family, latents_from_legacy_dataframe,
               to_legacy_dataframe,
               CountModelBuilder, PoissonCountModel, NegBinCountModel,
               ComposableCountModel, AbstractCovariateRole, SupremacyRole, LevelRole,
               AbstractCovariateConfig, LogSumWealthFeature,
               SLFPLogSumWealthFeature, WealthCovariate, DistanceCovariate,
               covariate_name, covariate_role, covariate_prior, covariate_features,
               covariate_column, covariate_oos, covariate_sides,
               AbstractRateGuard, ClampGuard, NoGuard,
               AbstractObservationConfig, PoissonObservation,
               NegativeBinomialObservation, DixonColesCorrelation,
               FrankCopulaCorrelation, add!, add, replace!, validate,
               build_count_model, cb_varinfo_sites, cb_chain_columns,
               cb_parameter_count
using .Predictions: GridWorkspace, SmileScoreGrid, alloc_score_grid,
                    alloc_smile_buffers, alloc_market_book, compute_score_grid!,
                    compute_score_grid, fill_smile_buffers!, price_market!,
                    price_market, market_keys
export AbstractPosteriorLatents, CountLatents, RecombLatents, SmileLatents
export n_matches, n_draws, n_strikes, latent_match_ids, latent_matrices,
       match_index, latent_bytes, latent_allocations, observation_family
export extract_latents, latent_family, latents_from_legacy_dataframe,
       to_legacy_dataframe
export CountModelBuilder, PoissonCountModel, NegBinCountModel, ComposableCountModel
export AbstractCovariateRole, SupremacyRole, LevelRole
export AbstractCovariateConfig, LogSumWealthFeature, SLFPLogSumWealthFeature,
       WealthCovariate, DistanceCovariate
export covariate_name, covariate_role, covariate_prior, covariate_features,
       covariate_column, covariate_oos, covariate_sides
export AbstractRateGuard, ClampGuard, NoGuard
export AbstractObservationConfig, PoissonObservation, NegativeBinomialObservation,
       DixonColesCorrelation, FrankCopulaCorrelation
export add!, add, replace!, validate, build_count_model
export cb_varinfo_sites, cb_chain_columns, cb_parameter_count
export GridWorkspace, SmileScoreGrid, alloc_score_grid, alloc_smile_buffers,
       alloc_market_book, compute_score_grid!, compute_score_grid,
       fill_smile_buffers!, price_market!, price_market, market_keys

# The unified inference & fit lifecycle (Training.Inference). Additive: `Experiments`
# and the legacy `Training.train` path are unchanged, and `fit_from_experiment` /
# `experiment_from_fit` bridge a result between the two.
using .Training: Inference,
                 Fit, FoldFit, FitConfig, FitMetadata,
                 fit_model, sample_fold, run_folds, ReplaySampler,
                 fold_chains, fold_metas, total_draws, fit_name,
                 ConvergenceThresholds, ConvergenceSummary, FoldConvergence,
                 audit_convergence, audit_fold, summarise_convergence,
                 convergence_table,
                 AbstractExecution, AutoExecution, SequentialExecution,
                 ThreadedExecution, QueuedExecution,
                 save_fit, load_fit, load_fits, list_fits, read_fit_meta,
                 save_latents, load_latents,
                 merge_latents, extract_run_latents,
                 MatchState, kickoff_state, NHPPIntensityModel,
                 IngameRatesWorkspace, LiveMatchRates,
                 build_ingame_workspace, alloc_live_rates,
                 solve_ingame_rates, solve_ingame_rates!,
                 upgrade_to_fit, fit_from_experiment, experiment_from_fit
export Inference
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

# The typed evaluation pipeline (Evaluation). Additive: the legacy metric kernels,
# `evaluate_experiments` and `to_dataframe_row` are unchanged, and `compat.jl` converts
# a legacy container's inputs into the typed evaluator's.
using .Evaluation: OddsView, MatchOutcomes, MarketProbabilities, EvaluationContext,
                   EvaluationRow, EvaluationWorkspace, EvaluationReport,
                   build_odds_view, extract_match_outcomes, build_evaluation_context,
                   evaluation_rows, verify_alignment, AlignmentReport,
                   pit_values, pit_uniformity, PITReport,
                   alloc_evaluation_workspace, price_match_markets!,
                   market_probabilities, prob_mean, prob_draws, outcome_of,
                   fit_latents,
                   marginals, crps_parameters,
                   evaluate_predictions, PredictionScores, PredictionScore,
                   brier_score, ranked_probability_score, calibration_curve,
                   CalibrationCurve, expected_calibration_error, max_calibration_error,
                   evaluate_fits, leaderboard, report_table, markdown_report,
                   as_typed_latents, as_fit, convergence_verdict, ConvergenceRefusal
export OddsView, MatchOutcomes, MarketProbabilities, EvaluationContext,
       EvaluationRow, EvaluationWorkspace, EvaluationReport
export build_odds_view, extract_match_outcomes, build_evaluation_context,
       evaluation_rows, verify_alignment, AlignmentReport
export pit_values, pit_uniformity, PITReport
export alloc_evaluation_workspace, price_match_markets!, market_probabilities
export prob_mean, prob_draws, outcome_of, fit_latents
export marginals, crps_parameters
export evaluate_predictions, PredictionScores, PredictionScore
export brier_score, ranked_probability_score, calibration_curve, CalibrationCurve,
       expected_calibration_error, max_calibration_error
export evaluate_fits, leaderboard, report_table, markdown_report
export as_typed_latents, as_fit, convergence_verdict, ConvergenceRefusal

# Maybe export core config types too?
export NUTSConfig, ADVIConfig, MAPConfig # From Samplers
export TrainingConfig, Independent, SequentialPriorUpdate # From Training

# 
using .Data: Markets
export Markets

end
