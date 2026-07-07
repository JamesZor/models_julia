#=
LOADER l04 — the trust-model interface (the "system API" for swapping estimators).

This is the extensible seam. A trust model answers ONE question: per market unit, how much should
we believe the model over the market? Everything downstream (blend → tilt → Kelly) only consumes
the resulting per-unit `w`, so the whole roadmap — EB (l05) → distributional → full Bayesian (l06)
→ hierarchical-per-team — is a matter of adding structs behind this interface. Nothing else changes.

Mirrors the src/ conventions (src/Calibration/types.jl AbstractLayerTwoModel with fit/apply;
src/signals/interfaces.jl AbstractSignal with compute_stake):

    abstract type AbstractTrustModel end          # the config / model spec
    fit_trust(model, hist)        -> Fitted…      # walk-forward fit on accumulated evidence
    trust_weights(fitted, match)  -> 7-vector     # POINT estimate of w
    trust_draws(fitted, match; D) -> 7 × D        # DISTRIBUTION of w (default: replicate the point)

`TrustHist` records (p_model, q_mkt, y) of each unit's representative selection, PLUS the home/away
team ids per observation so a hierarchical model can partition the evidence by team. Push AFTER the
match settles (no leakage). Depends on l01 (UNIT_REP_SEL, StakingMatch).
=#

# ---------- history ----------

"Per unit (1..7): (p_model, q_mkt, y) of the rep selection + the match's team ids per observation."
struct TrustHist
    p::Vector{Vector{Float64}}
    q::Vector{Vector{Float64}}
    y::Vector{Vector{Float64}}
    home::Vector{Vector{Int}}
    away::Vector{Vector{Int}}
end
TrustHist() = TrustHist([Float64[] for _ in 1:7], [Float64[] for _ in 1:7], [Float64[] for _ in 1:7],
                        [Int[] for _ in 1:7], [Int[] for _ in 1:7])

"Append one settled match. `model_sel` = 11-vector of model probs (rep selection read per unit)."
function push_hist!(h::TrustHist, m::StakingMatch, model_sel::AbstractVector{<:Real})
    for u in 1:7
        r = UNIT_REP_SEL[u]
        push!(h.p[u], model_sel[r]); push!(h.q[u], m.q_mkt[r]); push!(h.y[u], Float64(m.won[r]))
        push!(h.home[u], m.home);    push!(h.away[u], m.away)
    end
    return h
end

nobs(h::TrustHist) = length(h.y[1])

# ---------- interface ----------

"A trust-model spec. Concrete subtypes: FlatTrust, CuratedTrust, EBTrust (l05), BayesianTrust (l06)."
abstract type AbstractTrustModel end

"Fit `model` on accumulated `hist`; returns a model-specific fitted object."
fit_trust(model::AbstractTrustModel, ::TrustHist) =
    error("fit_trust not implemented for $(typeof(model))")

"Per-match per-unit point weights (7-vector)."
trust_weights(ft, ::StakingMatch) =
    error("trust_weights not implemented for $(typeof(ft))")

"Per-match per-unit weight draws (7 × D). Default: replicate the point estimate D times."
trust_draws(ft, m::StakingMatch; D::Int=64) = repeat(trust_weights(ft, m), 1, D)

"Match-free per-unit point weights — valid for non-hierarchical models (the 7 core units).
The extended book uses this (it has no per-team structure). Hierarchical models must be queried
per match via `trust_weights(ft, m)`."
trust_weights(ft) = error("trust_weights(ft) has no match-free form for $(typeof(ft))")

# ---------- constant trust (Flat / Curated) ----------

"Constant per-unit trust; ignores history. `FlatTrust(w)` or the E4 `CuratedTrust()`."
struct FlatTrust <: AbstractTrustModel
    w::Vector{Float64}
end
FlatTrust(x::Real) = FlatTrust(fill(Float64(x), 7))

"The sim-E4 curation: abstain on 1X2 (w=0, vig moat), half-trust totals + BTTS."
CuratedTrust() = FlatTrust([0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])

struct FittedConstantTrust
    w::Vector{Float64}
end
fit_trust(model::FlatTrust, ::TrustHist) = FittedConstantTrust(copy(model.w))
trust_weights(ft::FittedConstantTrust, ::StakingMatch) = ft.w
trust_weights(ft::FittedConstantTrust) = ft.w
