# src/Calibration/calibration-module.jl

"""
    Calibration

Layer-2 calibration: the tier between L1/L2 inference and L4 portfolio allocation.

# The production path — generative rate calibration

```julia
book, refusals = point_in_time_book(ds; config = PointInTimeBookConfig(as_of_minutes = -25.0))

cal = GenerativeRateCalibrator(
    name = "scot_lower_t25_inv",
    law  = InverseGaussianLaw(w_base = 0.25, sigma = 0.35),
    book_as_of_minutes = -25.0,
)

cf = calibrate_fit(cal, fit, book)                       # -> CalibratedFit
result, books, rep = run_portfolio_simulation(spec, policy, cf, book, ds)
cal_run = save_calibration_db(cf, model_run_id, db; scores = calibration_scores(cf, book, ds.matches))
link_portfolio_run(result, model_run_id, cal_run, db; book_spec = spec, policy_spec = policy)
```

The shift is applied at the GENERATIVE INTENSITY, not at the selection: the market book is
inverted back to `(lambda_mkt_h, lambda_mkt_a)`, every posterior log-rate draw is pooled
with it, and the calibrated container is priced through the same score-grid kernels,
evaluator and portfolio the raw one goes through. 1X2, every totals line and BTTS are then
three partitions of ONE 12x12 score tensor, so derivative coherence is structural rather
than checked.

# What is deprecated

`BasicLogitShift` (`build_l2_training_df` -> `train_calibrators` -> `apply_calibrators`)
moves each selection's probability with its own GLM offset, so `P(over 2.5)` and
`P(under 2.5)` no longer sum to 1 and there is no scoreline distribution behind the
shifted board at all. It is retained and warned, not deleted.

# Where the evidence is

`current_development/calibration_generative_eda/README.md` — four phases, the gates each
passed or failed, and two published conclusions this module's own results retracted.
Design record: `docs/architecture/rfc_layer2_calibration_v2.md`.
"""
module Calibration

using DataFrames
using Base.Threads
using Dates
using GLM
using JSON3
using LibPQ
using Optim
using Printf
using SHA
using StatsModels
using StatsFuns: logit, logistic
using Statistics
using UUIDs

using ..TypesInterfaces          # `AbstractLayerTwoModelConfig`, named bare in types.jl

# Everything else is IMPORTED, not `using`ed, so every call site says where the name came
# from and no exported name from these modules (`n_matches`, `n_draws`, `add!`, `build`,
# `market_keys`, `calc_logloss`, ...) can shadow or be shadowed by one of ours. This is the
# same discipline `src/Portfolio/portfolio-module.jl` follows, and for the same reason.
#
#   Data        the market types, `summarize_odds`, `DataStore`
#   Features    the market-inversion primitives the Nelder-Mead objective is rebuilt from
#   Models      the typed posterior containers a calibrator maps between
#   Predictions `market_keys`, and the score-grid contract the calibrated container meets
#   Training    `Fit`, `PostgresStorage`, and the `_db_*` primitives the artefact tables share
#   Evaluation  `fit_latents` / `convergence_verdict` / the scoring context, reused verbatim
#               so calibration and staking cannot gate on two different verdicts
#   Portfolio   `BookSpec` and `build_books_reported`, extended onto `CalibratedFit` here
#               rather than edited there
import ..Data
import ..Features
import ..Models
import ..Predictions
import ..Training
import ..Evaluation
import ..Portfolio

# --- order matters -----------------------------------------------------------
# types before everything (the structs every other file dispatches on); book before
# rate_pool (the pool inverts a book frame); pipeline before diagnostics and db_storage
# (both dispatch on `CalibratedFit`).

include("types.jl")
include("book.jl")
include("rate_pool.jl")
include("pipeline.jl")
include("diagnostics.jl")
include("db_storage.jl")

# --- the deprecated selection-level path --------------------------------------
include("data_l2_prep.jl")
include("trainer.jl")
include("basic_metrics.jl")
include("shift_models/basic_logit.jl")
include("shift_models/fitted_logit.jl")

# --- public surface ----------------------------------------------------------

export
    # the hierarchy
    AbstractCalibrator, AbstractGenerativeRateCalibrator,
    AbstractCalibrationWeightLaw, StandardGaussianLaw, InverseGaussianLaw,
    StaticGeometricLaw, calibration_weight, is_identity_law, law_label,
    AbstractDispersionMap, PoolDispersion, PreservedDispersion, ConjugateDispersion,
    SupremacyDispersion, residual_map, is_pool_map, map_label,
    GenerativeRateCalibrator, CalibratedLatents, CalibratedFit,
    is_identity_calibrator, calibrator_label, calibrator_hash, calibrator_json,

    # the tradeable book
    PointInTimeBookConfig, point_in_time_book, point_in_time_prices, devig_book!,
    assert_book_as_of, expected_selection_count, book_coverage, book_refusal_summary,
    book_drift, bet_clv, clv_summary, closing_book, drop_one_sided_markets,

    # inversion and the pool
    MarketInversionConfig, MarketRateFit, L2_INVERSION_LINES,
    market_targets, invert_market_rates, inversion_frame, inversion_refusals,
    inversion_coverage, calibrate_latents, restrict_latents,
    weight_summary, dispersion_summary,

    # the pipeline
    calibrate_fit, calibrated_fit, base_fit, calibration_context, calibration_scores,

    # diagnostics
    coherence_report, market_family_label, calibration_summary,
    l2_tradeable_markets, l2_full_direction_markets, l2_headline_selections,

    # persistence
    save_calibration_db, load_calibration_db, list_calibration_runs,
    link_portfolio_run, portfolio_runs_for_calibration,

    # DEPRECATED — the selection-level path
    CalibrationConfig, CalibrationResults, BasicLogitShift,
    train_calibrators, apply_calibrators,
    build_evaluation_df, summarize_metrics, compare_models

end
