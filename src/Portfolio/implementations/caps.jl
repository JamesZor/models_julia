# src/Portfolio/implementations/caps.jl
#
# The hard bound on stake settled simultaneously.
#
# There is deliberately no `NoCap`. If the total staked on one settlement window can reach 1,
# a bad round drives the bankroll to zero or below and every downstream number -- drawdown,
# final wealth, growth -- becomes arithmetic on a negative bankroll. The prototype had no cap:
# its worst slate lost 129.5% of bankroll and the simulated wealth reached -0.697, after which
# the sign flipped on every subsequent compounding step.
#
# Making the cap mandatory and validating it in the constructor turns that failure from
# something you assert against into something you cannot express.

export FixedCap, VolTargetCap

"""
    FixedCap(c)

Total stake settled simultaneously may not exceed fraction `c` of bankroll. `0 < c < 1` is
enforced at construction, which is what makes `slate_pl > -1` a theorem.
"""
struct FixedCap <: AbstractExposureCap
    cap::Float64
    function FixedCap(c::Real)
        0.0 < c < 1.0 || throw(ArgumentError(
            "exposure cap must be in (0,1) -- a cap of $c permits a non-positive bankroll"))
        new(Float64(c))
    end
end

function apply_cap(c::FixedCap, stakes::Vector{Vector{Float64}})
    exposure = isempty(stakes) ? 0.0 : sum(sum(s) for s in stakes)
    exposure <= c.cap && return (stakes, false)
    exposure <= 0.0   && return (stakes, false)
    sc = c.cap / exposure
    for s in stakes
        s .*= sc
    end
    return (stakes, true)
end

"""
    VolTargetCap(target, floor, ceiling)

Research slot: scale exposure toward a target portfolio volatility, clamped to `[floor,
ceiling]`. `ceiling < 1` is enforced for the same reason as `FixedCap`. Not implemented.
"""
struct VolTargetCap <: AbstractExposureCap
    target::Float64
    floor::Float64
    ceiling::Float64
    function VolTargetCap(t::Real, f::Real, c::Real)
        0.0 < c < 1.0 || throw(ArgumentError("ceiling must be in (0,1): $c"))
        new(Float64(t), Float64(f), Float64(c))
    end
end
apply_cap(::VolTargetCap, ::Vector{Vector{Float64}}) =
    error("VolTargetCap is not implemented yet")
