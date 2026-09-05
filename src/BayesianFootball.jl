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

include( "./signals/signals-module.jl")

include("./evaluation/evaluation-module.jl")

include("./synthetic/synthetic-data-module.jl")

include("./backtesting/backtesting-module.jl")

# Portfolio depends on BackTesting's metric interface, so it must come after it.
include("./Portfolio/portfolio-module.jl")

# Calibration is the Layer-2 tier BETWEEN inference and portfolio allocation, so it must
# come after both: `calibrate_fit` returns a `Training.Fit`, `CalibratedFit` extends
# `Portfolio.build_books_reported` and `Evaluation.fit_latents` onto its own type, and
# `save_calibration_db` writes beside `Portfolio.save_portfolio_db`. Nothing loaded before
# this line references `Calibration` (only `types-interfaces.jl`'s
# `AbstractLayerTwoModelConfig`, which is defined at the top and unaffected).
include("./Calibration/calibration-module.jl")

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
               AbstractPlayerAggregation, OutfieldPlayerAggregation,
               BenchWeightedPlayerAggregation, PositionalPlayerAggregation,
               MinuteWeightedPlayerAggregation, PlayerLineupPillar,
               PlayerLineupDynamics, AbstractPredictorTerm,
               AbstractCovariateConfig, LogSumWealthFeature,
               SLFPLogSumWealthFeature, AbstractAgeWeightingCurve,
               RichardsSigmoid, ShiftedGamma, GaussianPrime, age_weight,
               ProductionWealthFeature, WealthCovariate,
               ProductionWealthCovariate, BenchDepthCovariate, DistanceCovariate,
               PxGCovariate, LateGameChanceCovariate, PxGRapmCovariate,
               predictor_name, predictor_features, predictor_design, predictor_sites,
               predictor_extract, predictor_oos,
               covariate_name, covariate_role, covariate_prior, covariate_features,
               covariate_column, covariate_oos, covariate_sides,
               AbstractRateGuard, ClampGuard, NoGuard,
                AbstractObservationConfig, PoissonObservation,
                NegativeBinomialObservation, NegBinObservation,
                GlobalDispersion, HomeAwayDispersion,
                DixonColesCorrelation, FrankCopulaCorrelation,
                JointGammaPoissonObservation, JointGammaPoissonDesign,
                AbstractKappaMode, SharedKappa, HierarchicalKappa,
                SharedKappaJoint, HierarchicalKappaJoint, kappa_mode_width, cb_hpdi,
                observation_features, observation_design,
                add!, add, replace!, validate,
               build_count_model, build, cb_predictor_terms, cb_predictor_names,
               cb_covariates, cb_covariate_names, cb_varinfo_sites, cb_chain_columns,
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
export AbstractPlayerAggregation, OutfieldPlayerAggregation,
       BenchWeightedPlayerAggregation, PositionalPlayerAggregation,
       MinuteWeightedPlayerAggregation, PlayerLineupPillar, PlayerLineupDynamics
export AbstractPredictorTerm, AbstractCovariateConfig,
       LogSumWealthFeature, SLFPLogSumWealthFeature,
       AbstractAgeWeightingCurve, RichardsSigmoid, ShiftedGamma, GaussianPrime,
       age_weight, ProductionWealthFeature, WealthCovariate,
       ProductionWealthCovariate, BenchDepthCovariate, DistanceCovariate,
       PxGCovariate, LateGameChanceCovariate, PxGRapmCovariate
# The two pxG covariate feeds live in `Features`, not the builder, because both reuse the
# plus-minus segment/shot machinery. Re-exported here so a runner assembling a model can name the
# feature and its covariate in the same breath, as it already can for production wealth.
using .Features: BenchDepthFeature, PxGFeature, PxGRapmFeature, LateGameChanceFeature,
                 MatchProxyXGFeature
export BenchDepthFeature, PxGFeature, PxGRapmFeature, LateGameChanceFeature,
       MatchProxyXGFeature
export predictor_name, predictor_features, predictor_design, predictor_sites,
       predictor_extract, predictor_oos
export covariate_name, covariate_role, covariate_prior, covariate_features,
       covariate_column, covariate_oos, covariate_sides
export AbstractRateGuard, ClampGuard, NoGuard
export AbstractObservationConfig, PoissonObservation, NegativeBinomialObservation,
       NegBinObservation,
       GlobalDispersion, HomeAwayDispersion,
       DixonColesCorrelation, FrankCopulaCorrelation,
       JointGammaPoissonObservation, JointGammaPoissonDesign,
       AbstractKappaMode, SharedKappa, HierarchicalKappa,
       SharedKappaJoint, HierarchicalKappaJoint, kappa_mode_width, cb_hpdi,
       observation_features, observation_design
export add!, add, replace!, validate, build, build_count_model
export cb_predictor_terms, cb_predictor_names, cb_covariates,
       cb_covariate_names, cb_varinfo_sites, cb_chain_columns, cb_parameter_count
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
                 AbstractStorageBackend, FileStorage, PostgresStorage, DualStorage,
                 save_fit, load_fit, load_fits, list_fits, read_fit_meta,
                 save_latents, load_latents, ensure_schema!, config_hash,
                 compress_draws, decompress_draws,
                 save_config, save_model, save_splitter, save_sampler,
                 save_book_spec, save_policy_spec, save_calibrator,
                 load_model, load_splitter, load_sampler, load_fit_config,
                 load_book_spec, load_policy_spec, load_calibrator,
                 load_portfolio_spec, list_configs,
                 explore_experiments, search_configs, show_config,
                 preview_extension, extend_fit,
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
export AbstractStorageBackend, FileStorage, PostgresStorage, DualStorage
export save_fit, load_fit, load_fits, list_fits, read_fit_meta,
       save_latents, load_latents, ensure_schema!, config_hash,
       compress_draws, decompress_draws,
       save_config, save_model, save_splitter, save_sampler,
       save_book_spec, save_policy_spec, save_calibrator,
       load_model, load_splitter, load_sampler, load_fit_config,
       load_book_spec, load_policy_spec, load_calibrator,
       load_portfolio_spec, list_configs,
       explore_experiments, search_configs, show_config,
       preview_extension, extend_fit
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
                  FlatTrust, SelectionTrust, TieredTrust, CanonicalScottishLowerTrust,
                  ScheduledTrust, StaticFamilyTrust, ShrinkToMarketTrust,
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
                  save_portfolio_db, load_portfolio_db, portfolio_spec_hash,
                  extend_portfolio,
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
export FlatTrust, SelectionTrust, TieredTrust, CanonicalScottishLowerTrust, ScheduledTrust,
       StaticFamilyTrust, ShrinkToMarketTrust
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
export save_portfolio_db, load_portfolio_db, portfolio_spec_hash, extend_portfolio
export book_match_id, book_date, book_selections, book_grid, book_payoff, book_settle,
       book_alloc, book_shrink, book_kkt, book_converged
export sel_name, sel_odds_close, sel_odds_settle, sel_prob_model, sel_prob_market, sel_edge

# The Layer-2 calibrator tier (Calibration). Generative rate calibration: the tradeable
# book is inverted back to (lambda_mkt_h, lambda_mkt_a), every posterior log-rate draw is
# pooled with it, and the calibrated container is priced through the SAME score-grid
# kernels, evaluator and portfolio the raw one goes through — so 1X2, every totals line
# and BTTS are three partitions of one score tensor and cannot disagree.
#
# The legacy selection-level names (`BasicLogitShift`, `CalibrationConfig`,
# `train_calibrators`, `apply_calibrators`) are still exported and still work; they warn
# once per session. See `docs/architecture/rfc_layer2_calibration_v2.md`.
using .Calibration: AbstractCalibrator, AbstractGenerativeRateCalibrator,
                    AbstractCalibrationWeightLaw, StandardGaussianLaw, InverseGaussianLaw,
                    StaticGeometricLaw, calibration_weight, is_identity_law, law_label,
                    AbstractDispersionMap, PoolDispersion, PreservedDispersion,
                    ConjugateDispersion, SupremacyDispersion, residual_map, is_pool_map,
                    map_label,
                    GenerativeRateCalibrator, CalibratedLatents, CalibratedFit,
                    is_identity_calibrator, calibrator_label, calibrator_hash,
                    calibrator_json,
                    PointInTimeBookConfig, point_in_time_book, point_in_time_prices,
                    devig_book!, assert_book_as_of, expected_selection_count,
                    book_coverage, book_refusal_summary, book_drift, bet_clv, clv_summary,
                    closing_book, drop_one_sided_markets,
                    MarketInversionConfig, MarketRateFit, market_targets,
                    invert_market_rates, inversion_frame, inversion_refusals,
                    inversion_coverage, calibrate_latents, restrict_latents,
                    weight_summary, dispersion_summary,
                    calibrate_fit, calibrated_fit, calibration_context, calibration_scores,
                    coherence_report, calibration_summary, l2_tradeable_markets,
                    l2_full_direction_markets, l2_headline_selections,
                    save_calibration_db, load_calibration_db, list_calibration_runs,
                    link_portfolio_run, portfolio_runs_for_calibration,
                    CalibrationConfig, CalibrationResults, BasicLogitShift,
                    train_calibrators, apply_calibrators
export AbstractCalibrator, AbstractGenerativeRateCalibrator
export AbstractCalibrationWeightLaw, StandardGaussianLaw, InverseGaussianLaw,
       StaticGeometricLaw, calibration_weight, is_identity_law, law_label
export AbstractDispersionMap, PoolDispersion, PreservedDispersion, ConjugateDispersion,
       SupremacyDispersion, residual_map, is_pool_map, map_label
export GenerativeRateCalibrator, CalibratedLatents, CalibratedFit,
       is_identity_calibrator, calibrator_label, calibrator_hash, calibrator_json
export PointInTimeBookConfig, point_in_time_book, point_in_time_prices, devig_book!,
       assert_book_as_of, expected_selection_count, book_coverage, book_refusal_summary,
       book_drift, bet_clv, clv_summary, closing_book, drop_one_sided_markets
export MarketInversionConfig, MarketRateFit, market_targets, invert_market_rates,
       inversion_frame, inversion_refusals, inversion_coverage, calibrate_latents,
       restrict_latents, weight_summary, dispersion_summary
export calibrate_fit, calibrated_fit, calibration_context, calibration_scores
export coherence_report, calibration_summary, l2_tradeable_markets,
       l2_full_direction_markets, l2_headline_selections
export save_calibration_db, load_calibration_db, list_calibration_runs,
       link_portfolio_run, portfolio_runs_for_calibration
export CalibrationConfig, CalibrationResults, BasicLogitShift,
       train_calibrators, apply_calibrators

# Maybe export core config types too?
export NUTSConfig, ADVIConfig, MAPConfig # From Samplers
export TrainingConfig, Independent, SequentialPriorUpdate # From Training

# 
using .Data: Markets
export Markets

end
