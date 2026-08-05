# src/Portfolio/implementations/risk.jl
#
# The drawdown budget, after Busseti, Ryu & Boyd (2016).
#
# The constraint is  E[(1 + k R)^-lambda] <= 1  with  lambda = log(beta)/log(D), which by a
# supermartingale argument bounds P(bankroll ever drops below D) <= beta over an infinite
# horizon. Applying it afresh in every period -- as here -- is what buys that guarantee.
#
# Two measured caveats worth carrying:
#   * the bound holds under the MODEL's measure. Realised drawdown overshoots the nominal by a
#     stable ~1.15x across lambda, so budget lambda for ~1.15x the drawdown you will accept
#     (lambda ~ 23 for a real 20% limit, hence the default).
#   * `:sequential` aggregation across a slate is ~2-3% looser in k than the exact simultaneous
#     Monte-Carlo. Cheap and close enough; `:joint` is available when it matters.

export NoRisk, IsolatedDrawdown, SlateDrawdown, risk_lambda

"""
    risk_lambda(D, beta) -> Float64

`lambda = log(beta)/log(D)` for a bankroll floor `D` (0.8 = 20% drawdown) breached with
probability at most `beta`.
"""
risk_lambda(D::Real, beta::Real) = log(beta) / log(D)

"No drawdown budget. The exposure cap is still applied -- that one is not optional."
struct NoRisk <: AbstractRiskModel end
risk_factor(::NoRisk, ::Vector, ::Vector) = 1.0

"""
    IsolatedDrawdown(lambda)

Solve the constraint separately for each match, from that match's own return distribution.

Bounds the drawdown of every bet in isolation, which is not the same as bounding the drawdown of
the six that settle together. On a programme of near-identical matches the two coincide (the sum
of `L` equal log-penalties is zero exactly when each is), so measured differences are small --
but that is a property of the fixture list, not of the method.
"""
struct IsolatedDrawdown <: AbstractRiskModel
    lambda::Float64
end

"""
    SlateDrawdown(lambda; mode = :sequential, joint_draws = 50_000)

One factor for the whole slate, solved against all `L` matches jointly.

`:sequential` solves `sum_t log E[(1 + k R_t)^-lambda] <= 0` (matches compounding one after the
other); `:joint` Monte-Carlos the true simultaneous sum `sum_t R_t`.
"""
Base.@kwdef struct SlateDrawdown <: AbstractRiskModel
    lambda::Float64      = 23.0
    mode::Symbol         = :sequential
    joint_draws::Int     = 50_000
    seed::Int            = 1
end
SlateDrawdown(lambda::Real) = SlateDrawdown(lambda = Float64(lambda))

# ---------------------------------------------------------------- solvers

"Bisect a monotone-crossing penalty `f` with `f(0) = 0`; returns the largest safe `k` in [0,1]."
function _bisect_k(f, iters::Int = 60)
    f(1.0) <= 0.0 && return 1.0
    lo, hi = 0.0, 1.0
    for _ in 1:iters
        mid = 0.5 * (lo + hi)
        f(mid) > 0.0 ? (hi = mid) : (lo = mid)
    end
    return lo
end

function _sequential_penalty(probs, rets, lambda)
    k -> begin
        tot = 0.0
        for t in eachindex(probs)
            s = 0.0
            @inbounds for i in eachindex(probs[t])
                s += probs[t][i] * (1.0 + k * rets[t][i])^(-lambda)
            end
            tot += log(s)
        end
        tot
    end
end

function _joint_penalty(probs, rets, lambda, n_draws, seed)
    rng  = Random.MersenneTwister(seed)
    cums = [cumsum(p ./ sum(p)) for p in probs]
    draws = zeros(n_draws)
    for m in 1:n_draws
        s = 0.0
        for t in eachindex(rets)
            idx = searchsortedfirst(cums[t], rand(rng))
            s += rets[t][min(idx, length(rets[t]))]
        end
        draws[m] = s
    end
    return k -> mean((1.0 .+ k .* draws) .^ (-lambda)) - 1.0
end

function risk_factor(r::SlateDrawdown, probs::Vector, rets::Vector)
    (r.lambda <= 0 || isempty(probs)) && return 1.0
    f = r.mode === :joint ? _joint_penalty(probs, rets, r.lambda, r.joint_draws, r.seed) :
                            _sequential_penalty(probs, rets, r.lambda)
    return _bisect_k(f)
end

function risk_factor(r::IsolatedDrawdown, probs::Vector, rets::Vector)
    (r.lambda <= 0 || isempty(probs)) && return ones(length(probs))
    return [_bisect_k(_sequential_penalty([probs[t]], [rets[t]], r.lambda))
            for t in eachindex(probs)]
end
