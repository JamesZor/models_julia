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
               SLFPLogSumWealthFeature, AbstractAgeWeightingCurve,
               RichardsSigmoid, ShiftedGamma, GaussianPrime, age_weight,
               ProductionWealthFeature, WealthCovariate,
               ProductionWealthCovariate, BenchDepthCovariate, DistanceCovariate,
               PxGCovariate, LateGameChanceCovariate, PxGRapmCovariate,
               covariate_name, covariate_role, covariate_prior, covariate_features,
               covariate_column, covariate_oos, covariate_sides,
               AbstractRateGuard, ClampGuard, NoGuard,
                AbstractObservationConfig, PoissonObservation,
                NegativeBinomialObservation, GlobalDispersion, HomeAwayDispersion,
                DixonColesCorrelation, FrankCopulaCorrelation,
                add!, add, replace!, validate,
               build_count_model, build, cb_varinfo_sites, cb_chain_columns,
               cb_parameter_count
using .Models: GlobalInterception, SeasonalInterception, HierarchicalMonthlyInterception,
               GlobalHomeAdvantage, HierarchicalTeamHomeAdvantage, HierarchicalLeagueHomeAdvantage,
               TimeDecayDynamics, StaticZeroDynamics, PositionalPlayerDynamics
export GlobalInterception, SeasonalInterception, HierarchicalMonthlyInterception,
       GlobalHomeAdvantage, HierarchicalTeamHomeAdvantage, HierarchicalLeagueHomeAdvantage,
       TimeDecayDynamics, StaticZeroDynamics, PositionalPlayerDynamics, build
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
       AbstractAgeWeightingCurve, RichardsSigmoid, ShiftedGamma, GaussianPrime,
       age_weight, ProductionWealthFeature, WealthCovariate,
       ProductionWealthCovariate, BenchDepthCovariate, DistanceCovariate,
       PxGCovariate, LateGameChanceCovariate, PxGRapmCovariate
# The two pxG covariate feeds live in `Features`, not the builder, because both reuse the
# plus-minus segment/shot machinery. Re-exported here so a runner assembling a model can name the
# feature and its covariate in the same breath, as it already can for production wealth.
using .Features: BenchDepthFeature, PxGFeature, PxGRapmFeature, LateGameChanceFeature
export BenchDepthFeature, PxGFeature, PxGRapmFeature, LateGameChanceFeature
export covariate_name, covariate_role, covariate_prior, covariate_features,
       covariate_column, covariate_oos, covariate_sides
export AbstractRateGuard, ClampGuard, NoGuard
export AbstractObservationConfig, PoissonObservation, NegativeBinomialObservation,
       GlobalDispersion, HomeAwayDispersion,
       DixonColesCorrelation, FrankCopulaCorrelation
export add!, add, replace!, validate, build, build_count_model
export cb_varinfo_sites, cb_chain_columns, cb_parameter_count
export GridWorkspace, SmileScoreGrid, alloc_score_grid, alloc_smile_buffers,
       alloc_market_book, compute_score_grid!, compute_score_grid,
       fill_smile_buffers!, price_market!, price_market, market_keys

using .Samplers: AbstractSamplerConfig, AbstractNUTSConfig, NUTSConfig, QueuedNUTSConfig,
                 MAPConfig, MLEConfig
export AbstractSamplerConfig, AbstractNUTSConfig, NUTSConfig, QueuedNUTSConfig,
       MAPConfig, MLEConfig

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

# The zero-allocation portfolio & staking path (Portfolio). Additive: `build_book`,
# `build_books`, `extract_selections`, `simulate`, `stake_slate`, `group`, `path_metrics`,
# `bootstrap_roi`, `report` and `stake_sheet`'s legacy signatures are unchanged, and
# `book_cache_key` still returns the same `UInt` so an existing book cache hits rather than
# silently rebuilding. What is new is one workspace per FOLD instead of one tensor per FIXTURE,
# the convergence gate in front of the bankroll, and the richer result / report objects.
#
# `MarketSelection`, `MarketBook`, `MatchedMarketOdds`, `PortfolioPolicy` and `LogUtility` are
# `const` ALIASES of `Selection`, `MatchBook`, `OddsIndex`, `PolicySpec` and `KellyLogUtility` --
# one type per name, never a second struct with the same shape. See `src/Portfolio/compat.jl`.
using .Portfolio: OddsIndex, MarketSlot, FallbackSlot, BookWorkspace, BuildReport,
                  DailyState, PortfolioSummary, BootstrapCI, PortfolioResult,
                  PortfolioReport,
                  Selection, MatchBook, Slate, SlateContext, SlateAllocation, Trajectory,
                  ExecutionConfig, BookSpec, PolicySpec, PortfolioSystem,
                  DeArb, Normalise, RawPrice,
                  PerBetCommission, TurnoverCommission,
                  IndependentKelly, BakerMcHale, NoShrinkage,
                  FlatTrust, StaticFamilyTrust, ShrinkToMarketTrust,
                  SlateDrawdown, MatchDrawdown, FixedFraction,
                  FixedCap, PerMatchCap,
                  DailySlate, WeeklySlate, MatchSlate,
                  MarketSelection, MarketBook, MatchedMarketOdds, PortfolioPolicy,
                  LogUtility, KellyLogUtility, UnsettledBooks,
                  build_odds_index, group_slates_by_day, fixture_table,
                  price_fixture!, fallback_probs, grid_shrink_factor,
                  workspace_bytes, fallback_market_names, n_skipped,
                  build_book, build_books, build_books_reported, build_slates,
                  extract_selections, selection_family, is_settled, unsettled_books,
                  simulate, simulate_portfolio, portfolio_summary, bootstrap_portfolio,
                  run_portfolio_simulation, states_frame, stake_sheet, stake_slate,
                  slate_summary, path_metrics, bootstrap_roi, attribution,
                  portfolio_report, display_portfolio, daily_returns_table,
                  portfolio_markdown, as_namedtuple, log_growth, book_cache_key,
                  book_match_id, book_date, book_selections, book_grid, book_payoff,
                  book_settle, book_alloc, book_shrink, book_kkt, book_converged,
                  sel_name, sel_odds_close, sel_odds_settle, sel_prob_model,
                  sel_prob_market, sel_edge
export OddsIndex, MarketSlot, FallbackSlot, BookWorkspace, BuildReport
export DailyState, PortfolioSummary, BootstrapCI, PortfolioResult, PortfolioReport
export Selection, MatchBook, Slate, SlateContext, SlateAllocation, Trajectory
export ExecutionConfig, BookSpec, PolicySpec, PortfolioSystem
export DeArb, Normalise, RawPrice
export PerBetCommission, TurnoverCommission
export IndependentKelly, BakerMcHale, NoShrinkage
export FlatTrust, StaticFamilyTrust, ShrinkToMarketTrust
export SlateDrawdown, MatchDrawdown, FixedFraction
export FixedCap, PerMatchCap
export DailySlate, WeeklySlate, MatchSlate
export MarketSelection, MarketBook, MatchedMarketOdds, PortfolioPolicy, LogUtility,
       KellyLogUtility, UnsettledBooks
export build_odds_index, group_slates_by_day, fixture_table
export price_fixture!, fallback_probs, grid_shrink_factor, workspace_bytes,
       fallback_market_names, n_skipped
export build_book, build_books, build_books_reported, build_slates
export extract_selections, selection_family, is_settled, unsettled_books
export simulate, simulate_portfolio, portfolio_summary, bootstrap_portfolio,
       run_portfolio_simulation, states_frame, stake_sheet, stake_slate, slate_summary,
       path_metrics, bootstrap_roi, attribution
export portfolio_report, display_portfolio, daily_returns_table, portfolio_markdown,
       as_namedtuple, log_growth, book_cache_key
export book_match_id, book_date, book_selections, book_grid, book_payoff, book_settle,
       book_alloc, book_shrink, book_kkt, book_converged
export sel_name, sel_odds_close, sel_odds_settle, sel_prob_model, sel_prob_market, sel_edge

# Maybe export core config types too?
export NUTSConfig, ADVIConfig, MAPConfig # From Samplers
export TrainingConfig, Independent, SequentialPriorUpdate # From Training

# 
using .Data: Markets
export Markets

end
