module ScottishLowerProtocol

export AbstractSLModelAdapter, SLContract, SLFold,
    sl_contract, sl_hash, sl_protocol_contract, sl_artifact_hash, sl_splitter, sl_artifact_dir, sl_legacy_artifact_dir, sl_assert_not_sealed,
    sl_describe, sl_gate_table, sl_fold_table, sl_result,
    sl_model, sl_model_name, sl_required_features, sl_assert_model_contract,
    sl_build_turing_model, sl_params_from_varinfo, sl_equation_data, sl_equation_logjoint,
    sl_sampled_sites, sl_parameter_row, sl_synthetic_n_teams, sl_synthetic_draws, sl_synthetic_fixtures,
    sl_reference_extract, sl_extract_parameters, sl_extract_params, sl_compute_score_matrix,
    sl_referee_eval, sl_reference_grid, sl_marginal_cdf_bounds, sl_marginal_logpdf,
    sl_capabilities, sl_posterior_schema, sl_build_folds, sl_kickoff_map, sl_kickoffs, sl_last_kickoff,
    sl_first_kickoff, sl_gate_contract, sl_gate_config, sl_gate_features,
    sl_truncate_datastore, sl_featureset_equal, sl_prior_draw, sl_logdensity_fn,
    sl_gate_equation_parity, sl_gate_gradients, sl_experiment_config, sl_run_experiment,
    sl_load_experiment, sl_bfmi, sl_gate_convergence, sl_synthetic_chain,
    sl_gate_extraction_synthetic, sl_gate_extraction_real, sl_gate_extraction_fallbacks,
    sl_book_markets, sl_score_matrix, sl_gate_score_dispatch, sl_gate_score_grid,
    sl_gate_market_identities, sl_market_book, sl_drop_incomplete, sl_model_book,
    sl_join_books, sl_log_loss, sl_brier, sl_paired_delta, sl_gate_book_integrity,
    sl_gate_alignment, sl_score_table, sl_gate_shape, sl_gate_not_broken, sl_gate_evaluation,
    sl_betfair_odds_df, sl_bookmaker_odds_df, sl_book_spec, sl_keep_groups,
    sl_growth_policies, sl_gate_books, sl_gate_simulation, sl_pnl_concentration,
    sl_growth_table, sl_gate_growth

include("types.jl")
include("config.jl")
include("interface.jl")
include("reporting.jl")
include("folds.jl")
include("features.jl")
include("sampling.jl")
include("extraction.jl")
include("score_matrix.jl")
include("evaluation.jl")
include("growth.jl")

end # module
