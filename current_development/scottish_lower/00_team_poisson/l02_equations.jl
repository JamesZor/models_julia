# ==============================================================================
# Model 00 — INDEPENDENT LOG-POISSON EQUATIONS  (the parity reference)
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# This is a DELIBERATE SECOND IMPLEMENTATION of the pure Poisson model's maths,
# written from MODEL.md rather than from the Turing engine.
#
# Gate 3 checks the Turing model against it; gate 4 checks `extract_parameters`
# against it. If all three agree, the fitted model, the documented model, and the
# priced model are the same model.
#
#   DO NOT refactor this file to call the thing it is checking.
#   DO NOT "fix" it to match the engine — if they disagree, that is the finding.
#
# ==============================================================================

using BayesianFootball
using Distributions
using Statistics
using SpecialFunctions


# ==============================================================================
# 1. Parameter container
# ==============================================================================

"""
    TP00Params

One posterior draw of the pure Poisson baseline's parameters.
Notice that there is NO dispersion parameter (log_r).
"""
Base.@kwdef struct TP00Params
    μ::Float64                # inter.μ        — scoring level
    γ::Float64                # ha.γ_global    — home advantage
    σ_a::Float64              # dyn.σ_a        — spread of attack ratings
    σ_d::Float64              # dyn.σ_d        — spread of defence ratings
    raw_a::Vector{Float64}    # dyn.raw_a[1:n] — attack z-scores
    raw_d::Vector{Float64}    # dyn.raw_d[1:n] — defence z-scores
end

n_teams(p::TP00Params) = length(p.raw_a)

"""
    tp00_assert_default(model)

Refuse to compute if the model is not the default component set this file was
written for.
"""
function tp00_assert_default(model)
    ok_inter = model.interception_config isa PG.GlobalInterception
    ok_ha    = model.homeadvantage_config isa PG.GlobalHomeAdvantage
    ok_dyn   = model.dynamics_config     isa PG.TimeDecayDynamics

    if !(ok_inter && ok_ha && ok_dyn)
        error("""
        l02_equations.jl implements the DEFAULT component set only:
            GlobalInterception / GlobalHomeAdvantage / TimeDecayDynamics
        Got:
            $(typeof(model.interception_config).name.name) /
            $(typeof(model.homeadvantage_config).name.name) /
            $(typeof(model.dynamics_config).name.name)
        """)
    end
    return true
end


# ==============================================================================
# 2. The Equations
# ==============================================================================

"""
    tp00_team_effects(p) -> (α, β)

Non-centred hierarchy, then zero-sum:
    α_scaled = raw_a .* σ_a          α = α_scaled .- mean(α_scaled)
    β_scaled = raw_d .* σ_d          β = β_scaled .- mean(β_scaled)
"""
function tp00_team_effects(p::TP00Params)
    α_scaled = p.raw_a .* p.σ_a
    β_scaled = p.raw_d .* p.σ_d

    α = α_scaled .- mean(α_scaled)
    β = β_scaled .- mean(β_scaled)

    return (α, β)
end

"""
    tp00_log_intensities(p, home_idx, away_idx) -> (η_h, η_a)

Linear predictors in pure log space:
    η_h = μ + γ + α[h] + β[a]
    η_a = μ     + α[a] + β[h]
"""
function tp00_log_intensities(p::TP00Params, home_idx::AbstractVector{Int}, away_idx::AbstractVector{Int})
    α, β = tp00_team_effects(p)

    η_h = p.μ .+ p.γ .+ α[home_idx] .+ β[away_idx]
    η_a = p.μ        .+ α[away_idx] .+ β[home_idx]

    return (η_h, η_a)
end

"""
    tp00_intensities(p, home_idx, away_idx) -> (λ_h, λ_a)

    λ_h = exp(η_h)
    λ_a = exp(η_a)
"""
function tp00_intensities(p::TP00Params, home_idx::AbstractVector{Int}, away_idx::AbstractVector{Int})
    η_h, η_a = tp00_log_intensities(p, home_idx, away_idx)
    return (exp.(η_h), exp.(η_a))
end

"""
    tp00_weights(day_deltas, half_life_days) -> Vector{Float64}

    w_i = 0.5 ^ (Δ_i / H)
"""
function tp00_weights(day_deltas::AbstractVector{<:Real}, half_life_days::Real)
    return 0.5 .^ (day_deltas ./ half_life_days)
end


# ==============================================================================
# 3. Log Density
# ==============================================================================

"""
    tp00_loglik(p, data, half_life_days) -> Float64

The time-decayed log-Poisson pseudo-likelihood:
    log p(y | η) = y * η - exp(η) - log(y!)
"""
function tp00_loglik(p::TP00Params, data::NamedTuple, half_life_days::Real)
    η_h, η_a = tp00_log_intensities(p, data.home_idx, data.away_idx)
    w        = tp00_weights(data.day_deltas, half_life_days)

    log_fact_h = SpecialFunctions.loggamma.(Float64.(data.home_goals) .+ 1.0)
    log_fact_a = SpecialFunctions.loggamma.(Float64.(data.away_goals) .+ 1.0)

    ll_h = data.home_goals .* η_h .- exp.(η_h) .- log_fact_h
    ll_a = data.away_goals .* η_a .- exp.(η_a) .- log_fact_a

    return sum(ll_h .* w) + sum(ll_a .* w)
end

"""
    tp00_logprior(p, model) -> Float64

Every prior the default configuration declares, in the model's ORIGINAL
(unconstrained) space.
"""
function tp00_logprior(p::TP00Params, model)
    tp00_assert_default(model)

    lp  = logpdf(model.interception_config.μ,   p.μ)
    lp += logpdf(model.homeadvantage_config.γ_global, p.γ)
    lp += logpdf(model.dynamics_config.σ_att, p.σ_a)
    lp += logpdf(model.dynamics_config.σ_def, p.σ_d)

    lp += sum(logpdf.(Normal(0, 1), p.raw_a))
    lp += sum(logpdf.(Normal(0, 1), p.raw_d))

    return lp
end

"""
    tp00_logjoint(p, data, model) -> Float64

Prior + time-decayed log-Poisson likelihood.
"""
function tp00_logjoint(p::TP00Params, data::NamedTuple, model)
    tp00_assert_default(model)
    return tp00_logprior(p, model) + tp00_loglik(p, data, model.dynamics_config.days_half_life)
end


# ==============================================================================
# 4. Adapters
# ==============================================================================

"""
    tp00_equation_data(fs::FeatureSet) -> NamedTuple
"""
function tp00_equation_data(fs)
    d = fs.data
    return (
        home_idx   = Vector{Int}(d[:flat_home_ids]),
        away_idx   = Vector{Int}(d[:flat_away_ids]),
        home_goals = Vector{Int}(d[:flat_home_goals]),
        away_goals = Vector{Int}(d[:flat_away_goals]),
        day_deltas = Vector{Float64}(d[:dates]),
    )
end
