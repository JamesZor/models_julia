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

On the only out-of-sample evaluation available (ScottishLower, 628 matches), the default policy
returns a flat ROI whose match-clustered 95% bootstrap interval **includes zero**. `FlatTrust`
is the default because every attempt to *learn* per-selection trust lost money out of sample.
Treat the non-default components as slots for testing and rejecting ideas cheaply.
"""
module Portfolio

using DataFrames
using Dates
using Statistics
using LinearAlgebra
using Random
using Optim

using ..Data
using ..Predictions
using ..Experiments
using ..BackTesting

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

# last: it dispatches on every type and component defined above
include("display.jl")

# --- public surface ----------------------------------------------------------

export
    # abstract seams
    AbstractPricePolicy, AbstractCommissionModel, AbstractAllocator, AbstractShrinkage,
    AbstractTrustModel, AbstractRiskModel, AbstractExposureCap, AbstractSelectionFilter,
    AbstractSlateGrouping,

    # domain
    Selection, MatchBook, Slate, SlateContext, SlateAllocation, Trajectory,

    # config
    ExecutionConfig, BookSpec, PolicySpec, PortfolioSystem

end
