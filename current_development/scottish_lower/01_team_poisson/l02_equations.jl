# ==============================================================================
# Model 01 — INDEPENDENT EQUATIONS  (the parity reference)
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# This is a DELIBERATE SECOND IMPLEMENTATION of the baseline's mathematics,
# written from MODEL.md rather than from the Turing engine. Gate 3 checks the
# Turing model against it; gate 4 checks `extract_parameters` against it. If all
# three agree, the fitted model, the documented model, and the priced model are
# the same model.
#
#   DO NOT refactor this file to call the thing it is checking.
#   DO NOT "fix" it to match the engine — if they disagree, that is the finding.
#
# Scope of independence: the model STRUCTURE is reimplemented here — indexing,
# scaling, zero-sum centring, the intercept/home/attack/defence sum, the
# exponential link, and the decay weighting. Shared numerical primitives
# (`RobustNegativeBinomial`, `logpdf`) are NOT reimplemented; re-deriving those
# would test Distributions.jl, not this model.
#
# Written for the DEFAULT component set only. Any other components must fail
# loudly rather than silently compute something else — see `tp_assert_default`.
#
# ==============================================================================

using BayesianFootball
using Distributions
using Statistics

const MyDist = BayesianFootball.MyDistributions


# ==============================================================================
# 1. Parameter container
# ==============================================================================

"""
    TPParams

One posterior draw (or one prior draw) of the baseline's parameters, in the
model's own parameterisation — raw z-scores and scales, exactly as sampled.

Deriving `α` and `β` from these is the job of `tp_team_effects`, and doing that
derivation correctly is precisely what the 2026-08-24 audit found missing in the
archived prototype.
"""
Base.@kwdef struct TPParams
    μ::Float64                # inter.μ        — scoring level
    log_r::Float64            # disp.log_r     — NegBin dispersion, log scale
    γ::Float64                # ha.γ_global    — home advantage
    σ_a::Float64              # dyn.σ_a        — spread of attack ratings
    σ_d::Float64              # dyn.σ_d        — spread of defence ratings
    raw_a::Vector{Float64}    # dyn.raw_a[1:n] — attack z-scores
    raw_d::Vector{Float64}    # dyn.raw_d[1:n] — defence z-scores
end

n_teams(p::TPParams) = length(p.raw_a)


"""
    tp_assert_default(model)

Refuse to compute if the model is not the default component set this file was
written for. Silence here would defeat the entire purpose of the parity gate.
"""
function tp_assert_default(model)
    ok_inter = model.interception_config isa PG.GlobalInterception
    ok_disp  = model.dispersion_config   isa PG.GlobalDispersion
    ok_ha    = model.homeadvantage_config isa PG.GlobalHomeAdvantage
    ok_dyn   = model.dynamics_config     isa PG.TimeDecayDynamics

    if !(ok_inter && ok_disp && ok_ha && ok_dyn)
        error("""
        l02_equations.jl implements the DEFAULT component set only:
            GlobalInterception / GlobalDispersion / GlobalHomeAdvantage / TimeDecayDynamics
        Got:
            $(typeof(model.interception_config).name.name) /
            $(typeof(model.dispersion_config).name.name) /
            $(typeof(model.homeadvantage_config).name.name) /
            $(typeof(model.dynamics_config).name.name)
        Extend this file for the new components before trusting any gate.
        """)
    end
    return true
end


# ==============================================================================
# 2. The equations
# ==============================================================================

"""
    tp_team_effects(p) -> (α, β)

Non-centred hierarchy, then zero-sum:

    α_scaled = raw_a .* σ_a          α = α_scaled .- mean(α_scaled)
    β_scaled = raw_d .* σ_d          β = β_scaled .- mean(β_scaled)

The scale multiplication is the step whose omission inflated team effects in the
archived prototype. It is not optional and it is not cosmetic.
"""
function tp_team_effects(p::TPParams)
    α_scaled = p.raw_a .* p.σ_a
    β_scaled = p.raw_d .* p.σ_d

    α = α_scaled .- mean(α_scaled)
    β = β_scaled .- mean(β_scaled)

    return (α, β)
end


"""
    tp_intensities(p, home_idx, away_idx) -> (λ_h, λ_a)

    λ_h = exp( μ + γ + α[h] + β[a] )
    λ_a = exp( μ     + α[a] + β[h] )

`β` is a defensive LEAK: it enters the opponent's intensity, not its own.

With `GlobalInterception` the season term is a constant `μ` and the month term is
exactly zero, so neither index appears here. That is a property of the default
components, not of the engine — see MODEL.md.
"""
function tp_intensities(p::TPParams, home_idx::AbstractVector{Int}, away_idx::AbstractVector{Int})
    α, β = tp_team_effects(p)

    λ_h = exp.(p.μ .+ p.γ .+ α[home_idx] .+ β[away_idx])
    λ_a = exp.(p.μ        .+ α[away_idx] .+ β[home_idx])

    return (λ_h, λ_a)
end


"""
    tp_dispersion(p) -> (r_h, r_a)

`GlobalDispersion`: one shared `r`, with the engine's clamp reproduced.

The clamp is applied in the Turing model but NOT in `extract_dispersion`. Under
`log_r ~ Normal(3.1, 0.4)` it never binds, so the two agree — but gate 4 reports
the observed `|log_r|` range rather than assuming it. See MODEL.md § "Known
asymmetries".
"""
function tp_dispersion(p::TPParams)
    r = exp(clamp(p.log_r, -10.0, 10.0))
    return (r, r)
end


"""
    tp_weights(day_deltas, half_life_days) -> Vector{Float64}

    w_i = 0.5 ^ (Δ_i / H)

`Δ_i` is days before the boundary, so a match exactly one half-life old counts
half as much as one played today.
"""
function tp_weights(day_deltas::AbstractVector{<:Real}, half_life_days::Real)
    return 0.5 .^ (day_deltas ./ half_life_days)
end


# ==============================================================================
# 3. Log density
# ==============================================================================

"""
    tp_loglik(p, data, half_life_days) -> Float64

The time-decayed pseudo-likelihood:

    Σ_i w_i · logpdf(NegBin(r_h, λ_h(i)), y_h(i))
  + Σ_i w_i · logpdf(NegBin(r_a, λ_a(i)), y_a(i))

`data` is a NamedTuple with `home_idx`, `away_idx`, `home_goals`, `away_goals`,
`day_deltas` — plain vectors, no DataFrames, no FeatureSet.
"""
function tp_loglik(p::TPParams, data::NamedTuple, half_life_days::Real)
    λ_h, λ_a = tp_intensities(p, data.home_idx, data.away_idx)
    r_h, r_a = tp_dispersion(p)
    w        = tp_weights(data.day_deltas, half_life_days)

    ll_h = logpdf.(MyDist.RobustNegativeBinomial.(r_h, λ_h), data.home_goals)
    ll_a = logpdf.(MyDist.RobustNegativeBinomial.(r_a, λ_a), data.away_goals)

    return sum(ll_h .* w) + sum(ll_a .* w)
end


"""
    tp_logprior(p, model) -> Float64

Every prior the default configuration declares, in the model's ORIGINAL
(constrained) space — no Jacobian corrections.

That matters for the parity gate: DynamicPPL reports the log density in
unconstrained space when a VarInfo is linked, and `σ_a`/`σ_d` are positive-
constrained. Compare against an UNLINKED evaluation, or the Gamma terms will
differ by their log-Jacobians and the gate will fail for the wrong reason.
"""
function tp_logprior(p::TPParams, model)
    tp_assert_default(model)

    lp  = logpdf(model.interception_config.μ,   p.μ)
    lp += logpdf(model.dispersion_config.log_r, p.log_r)
    lp += logpdf(model.homeadvantage_config.γ_global, p.γ)
    lp += logpdf(model.dynamics_config.σ_att, p.σ_a)
    lp += logpdf(model.dynamics_config.σ_def, p.σ_d)

    lp += sum(logpdf.(Normal(0, 1), p.raw_a))
    lp += sum(logpdf.(Normal(0, 1), p.raw_d))

    return lp
end


"""
    tp_logjoint(p, data, model) -> Float64

Prior + time-decayed likelihood. This is the number gate 3 compares against
DynamicPPL's evaluation of the Turing model at the same parameter values.
"""
function tp_logjoint(p::TPParams, data::NamedTuple, model)
    tp_assert_default(model)
    return tp_logprior(p, model) + tp_loglik(p, data, model.dynamics_config.days_half_life)
end


# ==============================================================================
# 4. Adapters
# ==============================================================================

"""
    tp_equation_data(fs::FeatureSet) -> NamedTuple

Pull the plain vectors this file needs out of a `FeatureSet`, with types made
concrete. No conditional logic, no defaults, no `missing` handling — if a field
is absent or wrongly typed, this should fail here rather than produce a number.
"""
function tp_equation_data(fs)
    d = fs.data
    return (
        home_idx   = Vector{Int}(d[:flat_home_ids]),
        away_idx   = Vector{Int}(d[:flat_away_ids]),
        home_goals = Vector{Int}(d[:flat_home_goals]),
        away_goals = Vector{Int}(d[:flat_away_goals]),
        day_deltas = Vector{Float64}(d[:dates]),
    )
end
