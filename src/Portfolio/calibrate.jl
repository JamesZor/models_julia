# src/Portfolio/calibrate.jl
#
# Putting two systems on the same footing before comparing them.

export calibrate_lambda, calibrate_scale

"""
    calibrate_lambda(policy, slates; target_exposure, use_shrink = true) -> Float64

Bisect `lambda` so that MEAN realised slate exposure equals `target_exposure`.

**This is the correct dial.** Exposure is monotone decreasing in lambda, and lambda is the only
knob that moves it once the drawdown constraint is active. Use this -- not `calibrate_scale` --
whenever you want to compare two risk models, two trust models or two allocators at equal risk;
otherwise the comparison is dominated by whichever one happens to lever harder.

Requires `policy.risk` to carry a `lambda` field (`SlateDrawdown` / `IsolatedDrawdown`).
"""
function calibrate_lambda(policy::PolicySpec, slates::Vector{Slate};
                          target_exposure::Float64 = 0.15, use_shrink::Bool = true,
                          lo::Float64 = 0.5, hi::Float64 = 200.0, iters::Int = 30)
    hasproperty(policy.risk, :lambda) ||
        throw(ArgumentError("$(typeof(policy.risk)) has no lambda to calibrate"))

    f(lam) = mean(simulate(_with_lambda(policy, lam), slates; use_shrink = use_shrink).exposure)

    f(lo) < target_exposure && return lo      # unreachable: lambda is already slack
    f(hi) > target_exposure && return hi
    for _ in 1:iters
        mid = sqrt(lo * hi)
        f(mid) > target_exposure ? (lo = mid) : (hi = mid)
    end
    return sqrt(lo * hi)
end

_with_lambda(p::PolicySpec, lam::Float64) =
    PolicySpec(trust = p.trust, risk = _relambda(p.risk, lam), cap = p.cap,
               filter = p.filter, grouping = p.grouping)
_relambda(r::SlateDrawdown, lam) =
    SlateDrawdown(lambda = lam, mode = r.mode, joint_draws = r.joint_draws, seed = r.seed)
_relambda(r::IsolatedDrawdown, lam) = IsolatedDrawdown(lam)

"""
    calibrate_scale(policy, slates; target_exposure, use_shrink = true) -> Float64

Bisect the global stake multiplier to hit a mean exposure.

!!! warning
    Only usable while the drawdown constraint is SLACK. An active constraint solves its factor
    against whatever stakes it is handed, so doubling the input halves the factor and realised
    exposure barely moves -- it saturates at the lambda-implied level and this search then runs
    away until the exposure cap starts binding, producing a nonsense portfolio.

    Measured at lambda = 20: multipliers of 0.25, 1.0 and 4.0 all give mean exposure 0.1088.

    That scale-invariance is the same mechanism that makes lambda subsume trust. Prefer
    [`calibrate_lambda`](@ref).
"""
function calibrate_scale(policy::PolicySpec, slates::Vector{Slate};
                         target_exposure::Float64 = 0.15, use_shrink::Bool = true,
                         iters::Int = 40)
    f(sc) = mean(simulate(policy, slates; use_shrink = use_shrink, scale = sc).exposure)
    lo, hi = 1e-4, 1.0
    while f(hi) < target_exposure && hi < 1e4
        hi *= 2
    end
    for _ in 1:iters
        mid = sqrt(lo * hi)
        f(mid) < target_exposure ? (lo = mid) : (hi = mid)
    end
    return sqrt(lo * hi)
end
