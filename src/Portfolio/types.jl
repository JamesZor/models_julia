# src/Portfolio/types.jl
#
# Abstract types first (the swappable seams), then the domain objects that flow through the
# pipeline, then the configuration structs.

# ===================================================================
# 1. Abstract types -- the swappable seams
# ===================================================================

"""
    AbstractPricePolicy

How a traded price becomes the price we are settled at.
Contract: `settlement_odds(policy, decimal_odds, group_overround) -> Float64`.
"""
abstract type AbstractPricePolicy end

"""
    AbstractCommissionModel

How exchange commission is charged.
Contract: `net_return(model, decimal_odds) -> Float64` (profit per unit staked on a win).
"""
abstract type AbstractCommissionModel end

"""
    AbstractAllocator

Turns a belief over outcomes plus a payoff matrix into stakes.
Contract: `allocate(alloc, p, R, exec) -> (a, kkt, converged)`.
"""
abstract type AbstractAllocator end

"""
    AbstractShrinkage

Parameter-uncertainty correction applied to the point-estimate allocation.
Contract: `shrink_factor(s, score_matrix, R, p, alloc, exec) -> Float64` in `[0, 1]`.
"""
abstract type AbstractShrinkage end

"""
    AbstractTrustModel

How much we believe the model over the market, per selection.
Contract: `trust_for(model, sel, ctx) -> Float64` in `[0, 1]`.
"""
abstract type AbstractTrustModel end

"""
    AbstractRiskModel

The drawdown budget.
Contract: `risk_factor(model, probs, rets) -> Float64` (one factor for the whole slate) or
`Vector{Float64}` (one per match).
"""
abstract type AbstractRiskModel end

"""
    AbstractExposureCap

Hard bound on stake settled simultaneously. There is deliberately no `NoCap`: a cap `< 1` is
what makes a non-positive bankroll unrepresentable rather than merely asserted against.
Contract: `apply_cap(cap, stakes) -> (stakes, capped::Bool)`.
"""
abstract type AbstractExposureCap end

"""
    AbstractSelectionFilter

Curation. Contract: `keep(filter, sel, stake, ctx) -> Bool`.
"""
abstract type AbstractSelectionFilter end

"""
    AbstractSlateGrouping

How matches are partitioned into simultaneous settlement windows.
Contract: `group(grouping, books) -> Vector{Slate}`.
"""
abstract type AbstractSlateGrouping end

# ===================================================================
# 2. Domain objects
# ===================================================================

"""
    Selection

One priced leg of one match.

`odds_quoted` is the price as traded; `odds_used` is what we are settled at after the
`AbstractPricePolicy` has been applied and is never above `odds_quoted` under `DeArb`.
`p_market` is the vig-removed market probability and is a *forecast benchmark only* -- it is
never used as a price.
"""
struct Selection
    family::String        # "1X2_home", "O/U 2.5_over_25" -- the trust key
    group::String         # "1X2" | "BTTS" | "OverUnder" | ...
    line::Float64
    selection::Symbol
    odds_quoted::Float64
    odds_used::Float64
    p_model::Float64
    p_market::Float64
end

"""
    MatchBook

Everything needed to stake and settle one match, and nothing that depends on a `PolicySpec`.
This is the cache boundary: a `MatchBook` is a pure function of the data and the `BookSpec`.

* `p_grid`  -- posterior-mean score grid, length `N = max_h * max_a`, normalised to 1.
* `R`       -- Jacot return matrix, `N x n`; wealth after the bets is `1 .+ R * a`.
* `settle`  -- realised per-unit payoff of each selection (win / push / lose), or `nothing`
              for a fixture that has not been played. An unsettled book can be STAKED but not
              SIMULATED; `simulate` refuses them.
* `a_kelly` -- allocation on the posterior mean, at full size.
* `k_shrink`-- parameter-uncertainty factor from the `AbstractShrinkage`.
* `kkt`     -- KKT residual of `a_kelly`; should be ~1e-6.
"""
struct MatchBook
    m_id::Int
    date::Date
    sels::Vector{Selection}
    p_grid::Vector{Float64}
    R::Matrix{Float64}
    settle::Union{Nothing,Vector{Float64}}
    a_kelly::Vector{Float64}
    k_shrink::Float64
    kkt::Float64
    converged::Bool
end

"A set of matches that settle together and therefore share one bankroll."
struct Slate
    window::Date
    books::Vector{MatchBook}
end

"Context handed to trust / risk / filter so bankroll- or time-dependent policies are possible."
struct SlateContext
    idx::Int
    date::Date
    bankroll::Float64
end

"Stakes for one slate plus the risk diagnostics that produced them."
struct SlateAllocation
    stakes::Vector{Vector{Float64}}
    k_risk::Float64
    exposure::Float64
    capped::Bool
end

"""
    Trajectory

The result of a simulation. `bets` carries `:match_id, :date, :family, :stake, :odds, :pnl`
so it can be handed straight to `BackTesting.generate_tearsheet`.
"""
struct Trajectory
    bankroll::Vector{Float64}
    dates::Vector{Date}
    slate_pl::Vector{Float64}
    k_risk::Vector{Float64}
    exposure::Vector{Float64}
    n_capped::Int
    total_stake::Float64
    total_pl::Float64
    bets::DataFrame
end

Base.show(io::IO, t::Trajectory) = print(io,
    "Trajectory($(length(t.slate_pl)) slates, final $(round(t.bankroll[end], digits=3))x, " *
    "$(nrow(t.bets)) bets)")

# ===================================================================
# 3. Configuration
# ===================================================================

"""
    ExecutionConfig

Exchange and sizing constraints. `budget` bounds `sum(a)` within a single match; the
simultaneous bound across a slate is the `AbstractExposureCap` in the `PolicySpec`.
"""
Base.@kwdef struct ExecutionConfig{C<:AbstractCommissionModel}
    commission::C            = PerBetCommission(0.02)
    max_selection_stake::Float64 = 0.50
    min_selection_stake::Float64 = 1e-4
    budget::Float64          = 0.99
    barrier_mu::Float64      = 1e-6
    require_complete_markets::Bool = true
end

"""
    BookSpec

Everything that changes a `MatchBook`. **This is the cache key** -- hash it. Changing any field
here invalidates built books; changing a `PolicySpec` does not.
"""
Base.@kwdef struct BookSpec{P<:AbstractPricePolicy,
                            A<:AbstractAllocator,
                            S<:AbstractShrinkage,
                            E<:ExecutionConfig}
    markets::Data.MarketConfig
    price::P     = DeArb()
    allocator::A = KellyLogUtility()
    shrink::S    = BakerMcHale()
    exec::E      = ExecutionConfig()
end

"""
    PolicySpec

Everything that is a pure post-multiplier on an already-built book. Free to sweep against a
cached `Vector{MatchBook}`.

Note: `risk` is homogeneous of degree 0 in the stakes it is handed, so once the drawdown
constraint binds, `trust` can only reshape the book -- it cannot rescale it. See `stake_slate`.
"""
Base.@kwdef struct PolicySpec{T<:AbstractTrustModel,
                              R<:AbstractRiskModel,
                              C<:AbstractExposureCap,
                              F<:AbstractSelectionFilter,
                              G<:AbstractSlateGrouping}
    trust::T    = FlatTrust(0.25)
    risk::R     = SlateDrawdown(23.0)
    cap::C      = FixedCap(0.25)
    filter::F   = KeepAll()
    grouping::G = DailySlate()
end

"A complete staking system: the book it builds and the policy it stakes with."
struct PortfolioSystem{B<:BookSpec, P<:PolicySpec}
    book::B
    policy::P
end
PortfolioSystem(b::BookSpec) = PortfolioSystem(b, PolicySpec())
