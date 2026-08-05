# src/Portfolio/interfaces.jl
#
# One required method per seam, each with an error fallback so a half-implemented component
# fails loudly at the call site rather than silently doing nothing. Mirrors
# src/backtesting/metrics/interfaces.jl.

export settlement_odds, net_return, allocate, shrink_factor,
       trust_for, trust_vector, risk_factor, apply_cap, keep, group

# ---------------------------------------------------------------- pricing

"""
    settlement_odds(policy::AbstractPricePolicy, d, overround) -> Float64

Map a traded decimal price `d`, in a market group whose implied probabilities sum to
`overround`, onto the price we are settled at.
"""
settlement_odds(p::AbstractPricePolicy, ::Real, ::Real) =
    error("settlement_odds not implemented for $(typeof(p))")

# ---------------------------------------------------------------- commission

"""
    net_return(model::AbstractCommissionModel, d) -> Float64

Profit per unit staked when a bet at decimal odds `d` wins, net of commission.
"""
net_return(c::AbstractCommissionModel, ::Real) =
    error("net_return not implemented for $(typeof(c))")

# ---------------------------------------------------------------- allocation

"""
    allocate(alloc::AbstractAllocator, p, R, exec) -> (a, kkt, converged)

Choose stakes `a` given a belief `p` over the `N` outcome states and the `N x n` payoff matrix
`R`, subject to `0 .<= a .<= exec.max_selection_stake` and `sum(a) <= exec.budget`.

`kkt` is the worst first-order-condition violation at the returned point and exists so callers
can audit the solve rather than trust a convergence flag.
"""
allocate(a::AbstractAllocator, ::AbstractVector, ::AbstractMatrix, ::ExecutionConfig) =
    error("allocate not implemented for $(typeof(a))")

# ---------------------------------------------------------------- shrinkage

"""
    shrink_factor(s::AbstractShrinkage, score_matrix, R, p, alloc, exec; seed_offset = 0) -> Float64

Scalar in `[0, 1]` correcting the point-estimate allocation for parameter uncertainty.

`seed_offset` is part of the contract, not an optional extra: any shrinkage that samples the
posterior must decorrelate its draws across matches, and doing that from the match id keeps a
book reproducible regardless of how many threads built it. Deterministic implementations accept
it and ignore it.
"""
shrink_factor(s::AbstractShrinkage, ::Any, ::AbstractMatrix, ::AbstractVector,
              ::AbstractAllocator, ::ExecutionConfig; seed_offset::Integer = 0) =
    error("shrink_factor not implemented for $(typeof(s))")

# ---------------------------------------------------------------- trust

"""
    trust_for(model::AbstractTrustModel, sel::Selection, ctx::SlateContext) -> Float64

Weight in `[0, 1]` on the model's own probability for this selection. Because the market
probabilities are vig-removed, blending `w*p_model + (1-w)*p_market` scales the marginal Kelly
edge by exactly `w`, so this is applied as a stake multiplier.
"""
trust_for(t::AbstractTrustModel, ::Selection, ::SlateContext) =
    error("trust_for not implemented for $(typeof(t))")

"""
    trust_vector(model, book::MatchBook, ctx) -> Vector{Float64}

Resolve the trust weight for every selection in a book in one pass. Doing it here rather than
inside the staking loop gives one place to assert coverage.
"""
function trust_vector(t::AbstractTrustModel, book::MatchBook, ctx::SlateContext)
    w = Vector{Float64}(undef, length(book.sels))
    @inbounds for j in eachindex(w)
        w[j] = trust_for(t, book.sels[j], ctx)
    end
    return w
end

# ---------------------------------------------------------------- risk

"""
    risk_factor(model::AbstractRiskModel, probs, rets) -> Float64 | Vector{Float64}

Shrinkage applied to a whole slate's stakes to respect a drawdown budget. `probs[t]` is match
`t`'s outcome distribution and `rets[t] = R_t * a_t` its portfolio return in each state.

Returning a `Vector` means "one factor per match" (isolated scoping).

NOTE this map is homogeneous of degree 0 in `rets`: it solves for the factor that makes the
stakes it is handed satisfy the constraint, so scaling the input leaves `factor .* stakes`
unchanged. That is why trust cannot rescale a book once the constraint binds.
"""
risk_factor(r::AbstractRiskModel, ::Vector, ::Vector) =
    error("risk_factor not implemented for $(typeof(r))")

# ---------------------------------------------------------------- cap

"""
    apply_cap(cap::AbstractExposureCap, stakes) -> (stakes, capped::Bool)

Scale a slate's stakes so total simultaneous exposure respects the cap.
"""
apply_cap(c::AbstractExposureCap, ::Vector{Vector{Float64}}) =
    error("apply_cap not implemented for $(typeof(c))")

# ---------------------------------------------------------------- filter

"""
    keep(f::AbstractSelectionFilter, sel::Selection, stake, ctx) -> Bool

Curation: return `false` to zero a stake that the allocator wanted to take.
"""
keep(f::AbstractSelectionFilter, ::Selection, ::Real, ::SlateContext) =
    error("keep not implemented for $(typeof(f))")

# ---------------------------------------------------------------- grouping

"""
    group(g::AbstractSlateGrouping, books::Vector{MatchBook}) -> Vector{Slate}

Partition chronologically ordered books into simultaneous settlement windows.
"""
group(g::AbstractSlateGrouping, ::Vector{MatchBook}) =
    error("group not implemented for $(typeof(g))")

# ---------------------------------------------------------------- display

for T in (:AbstractPricePolicy, :AbstractCommissionModel, :AbstractAllocator,
          :AbstractShrinkage, :AbstractTrustModel, :AbstractRiskModel,
          :AbstractExposureCap, :AbstractSelectionFilter, :AbstractSlateGrouping)
    @eval component_name(x::$T) = string(nameof(typeof(x)))
    @eval Base.show(io::IO, x::$T) = print(io, component_name(x))
end
