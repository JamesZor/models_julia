#=
LOADER l03 — coherent pricing: per-unit trust blend → closed-form IPF grid tilt.

This is the middle of the one seam that matters:

    model targets p ─┐
    market probs   q ─┤→ blend_targets(p,q,w) → coherent_multiplier → tilted, coherent 144-grid
    trust weights  w ─┘

`blend_targets` mixes model and market per unit; `coherent_multiplier` I-projects the plain grid
onto those marginals (an IPF: cycle closed-form intercept tilts γ_j = logit(target) − logit(mass)
until every constrained marginal is hit). The result is one coherent grid on which a single Kelly
solve prices all markets consistently. `cycles=50` for real smile targets (three nested over-lines
overlap 1X2/BTTS; sim default 10 under-converges).

Extracted from staking_sim/l02_strategies.jl §trust-blend. Maths: docs/bets_multi/trust_blend_notes.md
§3 (blend), §4 (I-projection = tilt). Depends on l01 (SEL_MASKS, UNIT_REP_SEL).
=#

using LogExpFunctions: logit, logistic

"Blended per-unit targets w·p_model + (1−w)·q_mkt; 1X2 triple renormalized to sum to 1."
function blend_targets(pbar_sel, q_sel, w::AbstractVector)
    t = [w[u] * pbar_sel[UNIT_REP_SEL[u]] + (1.0 - w[u]) * q_sel[UNIT_REP_SEL[u]] for u in 1:7]
    s = t[1] + t[2] + t[3]
    t[1] /= s; t[2] /= s; t[3] /= s
    return t
end

# constrained masks: home, draw (away implied by 1X2 renorm), O1.5/2.5/3.5 over, btts_yes
const TILT_MASKS = (SEL_MASKS[1], SEL_MASKS[2], SEL_MASKS[4], SEL_MASKS[6], SEL_MASKS[8], SEL_MASKS[10])
const TILT_UNIT  = (1, 2, 4, 5, 6, 7)   # unit index of each constrained mask's target

"""
Closed-form IPF: cycle intercept tilts γ_j = logit(t_j) − logit(current mass) until all six
blended targets are hit (I-projection onto the target marginals — trust_blend_notes §4).
Returns the multiplier vector over the 144 states (apply to pbar and/or each draw column).
"""
function coherent_multiplier(pbar::Vector{Float64}, targets::Vector{Float64};
                             cycles=10, tol=1e-8)
    g = copy(pbar)
    mult = ones(length(g))
    for _ in 1:cycles
        moved = 0.0
        for j in 1:6
            m = TILT_MASKS[j]
            cur = sum(view(g, m))
            t = clamp(targets[TILT_UNIT[j]], 1e-9, 1 - 1e-9)
            δγ = logit(t) - logit(clamp(cur, 1e-9, 1 - 1e-9))
            e = exp(δγ)
            g[m] .*= e
            mult[m] .*= e
            z = sum(g)
            g ./= z
            mult ./= z
            moved = max(moved, abs(δγ))
        end
        moved < tol && break
    end
    return mult
end

normalize_mult(p::Vector{Float64}, mult) = (g = p .* mult; g ./ sum(g))

function apply_mult(P::Matrix{Float64}, mult)
    Q = P .* mult
    for j in 1:size(Q, 2)
        Q[:, j] ./= sum(view(Q, :, j))
    end
    return Q
end
