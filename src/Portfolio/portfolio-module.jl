# src/Portfolio/portfolio-module.jl

"""
    Portfolio

Multi-market, multi-match Kelly staking manager.

Turns L1 posterior score grids plus exchange quotes into a sized, risk-controlled book of bets
across matches that settle simultaneously. Implements Jacot & Mochkovitch (2023) for the
non-mutually-exclusive Kelly portfolio, Baker & McHale (2013) for parameter-uncertainty
shrinkage, and Busseti, Ryu & Boyd (2016) for the drawdown constraint.

# Pipeline

    latents + quotes --build_books--> Vector{MatchBook} --group--> Vector{Slate}
                                                                        |
                                                              simulate(policy)
                                                                        |
                                                                   Trajectory --report--> metrics

# The two configs, and why they are separate

`BookSpec` (pricing, allocator, shrinkage, execution) determines the `MatchBook` and is the
**cache key**. `PolicySpec` (trust, risk, cap, filter, grouping) is a set of pure
post-multipliers on an already-built book. Sweeping a `PolicySpec` must never rebuild books --
that split is what makes walk-forward evaluation affordable.

# Extending

Every swappable component is an abstract type with one required method
(`src/Portfolio/interfaces.jl`). Adding a staking idea is "add a struct + one method"; no
existing file changes.

# Health warning

The generic `PolicySpec` remains league-agnostic and conservative. Scottish Lower production
uses the separately audited `CanonicalScottishLowerTrust()` through MatchDay's canonical policy
factory; league-specific directional findings must not become an implicit default for every
portfolio simulation.
"""
module Portfolio

using DataFrames
using Dates
using JSON3
using LibPQ
using Statistics
using LinearAlgebra
using Random
using Printf          # display.jl
using Optim
using SHA
using UUIDs

using ..Data
using ..Predictions
using ..Experiments
using ..BackTesting

# The market interface functions. `Data` re-exports the concrete market types but not
# `market_group` / `market_line` / `outcomes` / `AbstractMarket`, so those come from the submodule.
using ..Data.Markets: AbstractMarket, market_group, market_line, outcomes

# Qualified rather than `using`, so every call site says where the name comes from and no exported
# name from these modules (`n_matches`, `add!`, `replace!`, `report_table`, ...) can shadow or be
# shadowed by one of ours.
#
#   Models      the typed posterior containers the zero-allocation builder reads
#   Training    `Fit` -- the run container the convergence gate reads a FIELD off
#   Evaluation  `convergence_verdict` / `fit_latents` / `as_typed_latents`, reused verbatim so
#               staking and evaluation cannot gate on two different verdicts
import ..Models
import ..Training
import ..Evaluation
import ..TypesInterfaces

# --- order matters -----------------------------------------------------------
# types.jl declares the abstract types AND the config structs, whose @kwdef defaults name
# concrete types from implementations/. That is fine: @kwdef defaults are evaluated when a
# constructor is CALLED, not when the struct is defined.

include("types.jl")
include("interfaces.jl")

include("implementations/commission.jl")
include("implementations/pricing.jl")
include("implementations/allocators.jl")
include("implementations/shrinkage.jl")
include("implementations/trust.jl")
include("implementations/risk.jl")
include("implementations/caps.jl")
include("implementations/filters.jl")

include("payoff.jl")
include("book.jl")
include("slates.jl")
include("stake.jl")
include("simulate.jl")
include("matchday.jl")
include("metrics.jl")
include("calibrate.jl")

# --- the zero-allocation path ------------------------------------------------
# Additive to everything above: `book.jl`'s builder, `simulate.jl`'s trajectory and every legacy
# signature are unchanged. These files add the typed-container fast path (one workspace per FOLD
# rather than one tensor per FIXTURE), the convergence gate in front of the bankroll, and the
# richer result / report objects.
#
# `alignment.jl` before `pricing.jl` because the builder reads an `OddsIndex`; `pricing.jl` before
# `simulation.jl` because the simulator's `BuildReport` method needs the builder that produces one;
# `compat.jl` last of the four because its aliases resolve at definition time.

include("alignment.jl")
include("pricing.jl")
include("simulation.jl")
include("reporting.jl")
include("db_storage.jl")
include("extension.jl")
include("compat.jl")

# last: it dispatches on every type and component defined above
include("display.jl")

# --- public surface ----------------------------------------------------------

export
    # abstract seams
    AbstractPricePolicy, AbstractCommissionModel, AbstractAllocator, AbstractShrinkage,
    AbstractTrustModel, AbstractRiskModel, AbstractExposureCap, AbstractSelectionFilter,
    AbstractSlateGrouping,

    # concrete implementations
    DeArb, Normalise, RawPrice,
    PerBetCommission, TurnoverCommission,
    KellyLogUtility, IndependentKelly,
    BakerMcHale, NoShrinkage,
    FlatTrust, SelectionTrust, TieredTrust, CanonicalScottishLowerTrust, ScheduledTrust,
    StaticFamilyTrust, ShrinkToMarketTrust,
    SlateDrawdown, MatchDrawdown, FixedFraction,
    FixedCap, PerMatchCap,
    DailySlate, WeeklySlate, MatchSlate,

    # domain
    Selection, MatchBook, Slate, SlateContext, SlateAllocation, Trajectory,

    # config
    ExecutionConfig, BookSpec, PolicySpec, PortfolioSystem,

    # the zero-allocation path: alignment, workspace, build report
    OddsIndex, MarketSlot, FallbackSlot, BookWorkspace, BuildReport,
    build_odds_index, build_books, price_portfolio_books!, price_book!,
    simulate_portfolio, run_portfolio_simulation,

    # simulation results
    DailyState, PortfolioSummary, BootstrapCI, PortfolioResult, PortfolioReport,
    portfolio_summary, portfolio_report,

    # PostgreSQL persistence
    save_portfolio_db, load_portfolio_db, portfolio_spec_hash, extend_portfolio

end
