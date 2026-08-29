# ==============================================================================
# 09 — UNIFIED PORTFOLIO & STAKING FRAMEWORK : THE COMPATIBILITY BRIDGE
# ==============================================================================
#
# Loader. Opens `module UnifiedPortfolio`, whose include chain pulls in l03 → l02 →
# l01 and, through l01, the whole of `08_unified_evaluation_framework`,
# `07_unified_inference_framework`, `06_typed_posterior_latents` and
# `05_composable_count_builder`. One `include` loads everything.
#
# ------------------------------------------------------------------------------
# WHAT "100% BACKWARD COMPATIBLE" MEANS HERE, PRECISELY
# ------------------------------------------------------------------------------
#
# Stronger than in `07` or `08`, and for a structural reason: this framework does not
# redeclare a single domain or configuration type. A `MatchBook` built by
# `UnifiedPortfolio.build_books` IS a `BayesianFootball.Portfolio.MatchBook` — the same
# struct, from the same module, with the same fields — so there is no translation
# layer between old and new and therefore nothing for a translation layer to get
# wrong. See `l01_types.jl`'s header for why.
#
# Concretely, all four of these hold:
#
#   1. A book built here can be handed to any legacy function, including ones this
#      framework has never heard of.
#   2. A book UNSERIALISED FROM AN EXISTING `.jls` CACHE can be handed to any function
#      here. Neither direction needs a version check.
#   3. `book_cache_key(spec)` returns the same `UInt` as before, so an existing book
#      cache still HITS rather than silently rebuilding under a new key.
#   4. `simulate`, `stake_slate`, `group`, `path_metrics`, `bootstrap_roi`, `report`,
#      `attribution`, `slate_summary`, `calibrate_lambda` and `calibrate_scale` are
#      not reimplemented at all — they are the `src` functions, bound to the same
#      names. There is no number they could disagree on.
#
# THE ONE LINE A CALLER MUST CHANGE, and it is an import, not a body:
#
#     # before
#     using BayesianFootball
#     books = Portfolio.build_books(spec, latents.df, expr, odds_df, ds)
#     traj  = Portfolio.simulate(policy, Portfolio.group(policy.grouping, books))
#
#     # after — the same three lines, one different import
#     import BayesianFootball
#     using .UnifiedPortfolio.Legacy       # binds `Portfolio`
#     …identical body…
#
# The second line is needed because `BayesianFootball` EXPORTS the name `Portfolio`
# and Julia refuses to rebind an imported name. Nothing can make two modules answer to
# one name; the honest claim is that everything AFTER the import is untouched, and
# `r01_demo.jl` §10 proves it with a `LegacyCallSite` module whose body is copied
# verbatim from `current_development/scottish_lower/02_poisson_wealth/r03_growth_clv.jl`.
#
# ------------------------------------------------------------------------------
# THE TWO DELIBERATE BEHAVIOUR CHANGES
# ------------------------------------------------------------------------------
#
# 1. `build_books` over a `Fit` REFUSES an unconverged posterior by default. Every
#    other entry point — the typed-container method, the legacy five-argument
#    method — is ungated, because neither is handed anything to gate ON. A legacy
#    caller therefore sees no change; a caller who upgraded to a `Fit` opted into the
#    gate by doing so. See `l02_book_builder.jl` §9 for why the default is `true`.
#
# 2. A fixture the builder declines is COUNTED AND NAMED rather than silently
#    dropped. `src` returns `nothing` for five distinct causes and filters them out
#    (`book.jl:151`); `BuildReport` separates them. The books produced are identical —
#    this adds a second return value, it does not change the first.
#
# ==============================================================================

module UnifiedPortfolio

# The whole prototype. l03 chains down through l02 → l01, and l01 pulls in `08` and,
# through it, `07`, `06` and `05`.
include(joinpath(@__DIR__, "l03_stake_and_simulate.jl"))

using DataFrames
using Dates
using Printf
using Random
using Statistics


# ==============================================================================
# 1. THE NEW API
# ==============================================================================

# --- the builder --------------------------------------------------------------
export OddsIndex, build_odds_index, BookWorkspace, workspace_bytes,
       fallback_market_names, MarketSlot, FallbackSlot
export price_fixture!, fallback_probs, grid_shrink_factor
export build_book, build_books, build_books_reported, BuildReport, n_skipped
export fixture_table, is_settled, unsettled_books, selection_family, extract_selections

# --- staking and simulation ---------------------------------------------------
export DailyState, PortfolioSummary, BootstrapCI, PortfolioResult
export simulate_portfolio, portfolio_summary, bootstrap_portfolio, states_frame,
       as_namedtuple, log_growth, display_portfolio, run_portfolio_simulation
export stake_sheet

# --- the briefing's field names, as accessors ---------------------------------
export book_match_id, book_date, book_selections, book_grid, book_payoff, book_settle,
       book_alloc, book_shrink, book_kkt, book_converged
export sel_name, sel_odds_close, sel_odds_settle, sel_prob_model, sel_prob_market, sel_edge


# ==============================================================================
# 2. THE LEGACY SURFACE
# ==============================================================================
#
# Every name `src/Portfolio/` exports, bound here to the same object. A call site that
# swaps its import keeps every one of them.

# --- abstract seams -----------------------------------------------------------
export AbstractPricePolicy, AbstractCommissionModel, AbstractAllocator, AbstractShrinkage,
       AbstractTrustModel, AbstractRiskModel, AbstractExposureCap, AbstractSelectionFilter,
       AbstractSlateGrouping

# --- domain and configuration -------------------------------------------------
export Selection, MatchBook, Slate, SlateContext, SlateAllocation, Trajectory, FixtureInfo
export ExecutionConfig, BookSpec, PolicySpec, PortfolioSystem
export component_hash, book_cache_key

# --- seam methods -------------------------------------------------------------
export settlement_odds, net_return, allocate, shrink_factor, trust_for, trust_vector,
       risk_factor, apply_cap, keep, group
export payoff, payoff_matrix, settle_vector, grid_index

# --- pipeline stages ----------------------------------------------------------
export build_slates, stake_slate, simulate, path_metrics, bootstrap_roi, report,
       attribution, slate_summary, calibrate_lambda, calibrate_scale

# --- components ---------------------------------------------------------------
export PerBetCommission, NetMarketCommission, NoCommission
export DeArb, Normalise, RawPrice
export KellyLogUtility
export NoShrinkage, FractionalKelly, BakerMcHale
export FlatTrust, SelectionTrust, ScheduledTrust
export NoRisk, IsolatedDrawdown, SlateDrawdown, risk_lambda
export FixedCap, VolTargetCap
export KeepAll, MinEdge, MarketWhitelist, MinOdds, FilterChain
export DailySlate, SingleMatchSlate


# ==============================================================================
# 3. RE-EXPORTS FROM `06`, `07` AND `08`
# ==============================================================================
#
# So a portfolio runner needs ONE `using`. These are the names a staking call site
# reaches for immediately before and after building a book.

export Fit, FitConfig, FoldFit, ConvergenceSummary, ConvergenceRefusal,
       convergence_verdict, fit_latents, fit_name, as_typed_latents
export AbstractPosteriorLatents, CountLatents, RecombLatents, SmileLatents,
       n_matches, n_draws, latent_match_ids, extract_latents,
       to_legacy_dataframe, latents_from_legacy_dataframe
export GridWorkspace, alloc_score_grid, compute_score_grid, compute_score_grid!,
       price_market, price_market!, market_keys, market_arity, alloc_market_book,
       SmileScoreGrid


# ==============================================================================
# 4. THE PARITY HARNESS
# ==============================================================================
#
# Included here, inside the module, because it holds BOTH builders at once — the one
# above and `BayesianFootball.Portfolio`'s — and neither is reachable from the other's
# namespace. The briefing's file NUMBERING is kept; the include ORDER follows the
# dependency, as in `08`.

include(joinpath(@__DIR__, "l05_parity.jl"))

export LegacyExpr, legacy_build
export book_structure_checks, book_parity_rows, trajectory_parity_rows,
       trajectory_structure_checks, summary_parity_rows
export AllocationRow, allocation_table, scoring_allocations, baseline_allocations
export measure_build_cost, measure_pricing_cost
export portfolio_matches, portfolio_datastore, thin_quotes!, drop_market_leg!

# From `06`'s parity harness, re-exported because every table in `r01_demo.jl` uses
# them and a second ULP comparator in this repository would be one too many.
export ulp_distance, ParityRow, tpl_compare, tpl_parity_table
export CostRow, cost_table, speedup, shrink


# ==============================================================================
# 5. THE COLLIDING NAME
# ==============================================================================

"""
    UnifiedPortfolio.Legacy

The one name that cannot be bound in a scope that has done `using BayesianFootball`.

```julia
import BayesianFootball
using .UnifiedPortfolio.Legacy      # Portfolio
```

`Portfolio` is `UnifiedPortfolio` itself, so `Portfolio.build_books`,
`Portfolio.simulate`, `Portfolio.BookSpec` and `Portfolio.DailySlate` all resolve to
the bindings above — most of which ARE the `src` objects. Anything else the legacy
module offered resolves to whatever `UnifiedPortfolio` binds under that name, or
raises an `UndefVarError` naming it.

`parentmodule(@__MODULE__)` rather than `import ..UnifiedPortfolio`: this submodule is
elaborated while its parent's body is still executing, and `parentmodule` needs no
binding to already exist.
"""
module Legacy

const Portfolio = parentmodule(@__MODULE__)

export Portfolio

end # module Legacy

end # module UnifiedPortfolio
