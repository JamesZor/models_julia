# ==============================================================================
# 05 — Composable Count Model Builder : THE PARITY REFERENCE
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# A DELIBERATE SECOND IMPLEMENTATION of the composable model's log-joint, written
# from the equations rather than from `engine.jl`. Same discipline as
# 00_team_poisson/l02_equations.jl:
#
#   DO NOT refactor this file to call the thing it is checking.
#   DO NOT "fix" it to match the engine — if they disagree, that is the finding.
#
# It is generic over the covariate tuple, which is the point: ONE reference covers
# all four arms, just as one engine does. If the reference had to be rewritten per
# covariate combination, the design would not have removed the duplication, only
# moved it.
#
# SCOPE. Valid for the component set the demo exercises — GlobalInterception,
# GlobalHomeAdvantage, TimeDecayDynamics, and a Poisson, scalar-dispersion NegBin
# or two-arm JointGammaPoisson observation.
# `cb_equation_data` refuses anything else rather than silently referencing the
# wrong maths, exactly as the arms' references do.
#
# ==============================================================================

# Dependencies and engine definitions are loaded by builder-module.jl.


# ==============================================================================
# 1. Parameter container
# ==============================================================================

"""
    CBParams

One draw of the composable model's parameters, in the model's own (unlinked) space.

`w` holds the covariate weights in BUILD ORDER — the same order the builder froze
into the tuple and the engine declared them in. `log_r` is `nothing` for a Poisson
observation.
"""
Base.@kwdef struct CBParams
    μ::Float64                      # inter.μ
    γ::Float64                      # ha.γ_global
    σ_a::Float64                    # dyn.σ_a
    σ_d::Float64                    # dyn.σ_d
    raw_a::Vector{Float64}          # dyn.raw_a[1:n_teams]
    raw_d::Vector{Float64}          # dyn.raw_d[1:n_teams]
    w::Vector{Float64}              # <covariate>.w, in build order
    log_r::Union{Nothing, Float64} = nothing   # disp.log_r (NegBin only)
    ν::Union{Nothing, Float64} = nothing       # obs.ν      (joint only)
    log_κ::Union{Nothing, Float64} = nothing   # obs.log_κ  (joint only)
end

"""
    cb_params_from_varinfo(model, vi) -> CBParams

Read one draw out of a `DynamicPPL.VarInfo` BY SITE NAME. Reading by name rather
than by position is what makes the parity test meaningful: if the engine ever
reorders its declarations, this still reads the right numbers and the log-joint
comparison still holds, while the separate θ-layout check in `r01_demo.jl` reports
the reordering.
"""
function cb_params_from_varinfo(model::ComposableCountModel, vi)
    v = Dict(string(k) => vi[k] for k in keys(vi))
    get_scalar(name) = Float64(v[name])

    return CBParams(
        μ     = get_scalar("inter.μ"),
        γ     = get_scalar("ha.γ_global"),
        σ_a   = get_scalar("dyn.σ_a"),
        σ_d   = get_scalar("dyn.σ_d"),
        raw_a = Float64.(v["dyn.raw_a"]),
        raw_d = Float64.(v["dyn.raw_d"]),
        w     = Float64[Float64(v["$(covariate_name(c)).w"]) for c in model.covariates],
        log_r = model.observation isa NegativeBinomialObservation ?
                Float64(v["disp.log_r"]) : nothing,
        ν     = model.observation isa JointGammaPoissonObservation ?
                Float64(v["obs.ν"]) : nothing,
        log_κ = model.observation isa JointGammaPoissonObservation ?
                Float64(v["obs.log_κ"]) : nothing,
    )
end

"Zero-sum team attack and defence effects, reconstructed from the non-centred draw."
function cb_team_effects(p::CBParams)
    a = p.raw_a .* p.σ_a
    d = p.raw_d .* p.σ_d
    return (a .- mean(a), d .- mean(d))
end


# ==============================================================================
# 2. Reference data
# ==============================================================================

"""
    cb_equation_data(model, feature_set) -> NamedTuple

The reference design, read straight out of the `FeatureSet` without calling
`cb_design`. Refuses component sets the reference maths does not cover.
"""
function cb_equation_data(model::ComposableCountModel, feature_set)
    model.interception   isa CB_PG.GlobalInterception  ||
        error("reference covers GlobalInterception only; got $(nameof(typeof(model.interception)))")
    model.home_advantage isa CB_PG.GlobalHomeAdvantage ||
        error("reference covers GlobalHomeAdvantage only; got $(nameof(typeof(model.home_advantage)))")
    model.dynamics       isa CB_PG.TimeDecayDynamics   ||
        error("reference covers TimeDecayDynamics only; got $(nameof(typeof(model.dynamics)))")
    (model.observation isa PoissonObservation ||
     model.observation isa JointGammaPoissonObservation ||
     (model.observation isa NegativeBinomialObservation &&
      model.observation.dispersion isa CB_PG.GlobalDispersion)) ||
        error("reference covers PoissonObservation, JointGammaPoissonObservation and " *
              "NegBin+GlobalDispersion only")

    d  = feature_set.data
    yh = Vector{Int}(d[:flat_home_goals])
    ya = Vector{Int}(d[:flat_away_goals])

    # Read straight out of the FeatureSet, NOT out of `observation_design` — the reference must
    # not borrow the thing it is checking. `nothing` where the observation has no proxy arm.
    joint = model.observation isa JointGammaPoissonObservation
    return (;
        home    = Vector{Int}(d[:flat_home_ids]),
        away    = Vector{Int}(d[:flat_away_ids]),
        yh, ya,
        pxg_h   = joint ? Vector{Float64}(d[:flat_pxg_home]) : nothing,
        pxg_a   = joint ? Vector{Float64}(d[:flat_pxg_away]) : nothing,
        mask    = joint ? Vector{Float64}(d[:flat_pxg_obs_available]) : nothing,
        weights = 0.5 .^ (Vector{Float64}(d[:dates]) ./ model.dynamics.days_half_life),
        # One column per covariate, in build order, plus its role.
        x       = [covariate_column(c, feature_set) for c in model.covariates],
        roles   = [covariate_role(c) for c in model.covariates],
        lfh     = loggamma.(Float64.(yh) .+ 1.0),
        lfa     = loggamma.(Float64.(ya) .+ 1.0),
    )
end


# ==============================================================================
# 3. Reference log-joint
# ==============================================================================

"""
    cb_logjoint(model, p::CBParams, data) -> Float64

log p(θ) + Σ_i weight_i · [ log p(y_h,i | η_h,i) + log p(y_a,i | η_a,i) ]

with

    η_h = μ + γ + α_{h} + β_{a} + Σ_k q_k,h
    η_a = μ     + α_{a} + β_{h} + Σ_k q_k,a
    (q_k,h, q_k,a) = covariate_sides(role_k, w_k · x_k)

both passed through the model's rate guard (`ClampGuard` or `NoGuard`).
"""
function cb_logjoint(model::ComposableCountModel, p::CBParams, data)
    α, β = cb_team_effects(p)
    n = length(data.home)

    q_h = zeros(Float64, n)
    q_a = zeros(Float64, n)
    for k in eachindex(data.x)
        qh, qa = covariate_sides(data.roles[k], p.w[k] .* data.x[k])
        q_h .+= qh
        q_a .+= qa
    end

    η_h = apply_guard(model.guard, p.μ .+ p.γ .+ α[data.home] .+ β[data.away] .+ q_h)
    η_a = apply_guard(model.guard, p.μ        .+ α[data.away] .+ β[data.home] .+ q_a)

    # --- priors ---------------------------------------------------------------
    lp = logpdf(model.interception.μ, p.μ) +
         logpdf(model.home_advantage.γ_global, p.γ) +
         logpdf(model.dynamics.σ_att, p.σ_a) +
         logpdf(model.dynamics.σ_def, p.σ_d) +
         sum(logpdf.(Normal(), p.raw_a)) +
         sum(logpdf.(Normal(), p.raw_d))
    for (k, c) in enumerate(model.covariates)
        lp += logpdf(covariate_prior(c), p.w[k])
    end
    if model.observation isa NegativeBinomialObservation
        lp += logpdf(model.observation.dispersion.log_r, p.log_r)
    end
    if model.observation isa JointGammaPoissonObservation
        lp += logpdf(model.observation.shape_prior, p.ν) +
              logpdf(model.observation.log_kappa_prior, p.log_κ)
    end

    return lp + cb_loglik(model.observation, p, data, η_h, η_a)
end

"Poisson log-likelihood in log-intensity space, time-decay weighted."
cb_loglik(::PoissonObservation, p::CBParams, data, η_h, η_a) =
    sum(data.weights .* (data.yh .* η_h .- exp.(η_h) .- data.lfh)) +
    sum(data.weights .* (data.ya .* η_a .- exp.(η_a) .- data.lfa))

"Negative-binomial log-likelihood with a single global dispersion, time-decay weighted."
function cb_loglik(::NegativeBinomialObservation, p::CBParams, data, η_h, η_a)
    r = exp(_cb_bound_dispersion_log(p.log_r))
    λ_h = exp.(η_h)
    λ_a = exp.(η_a)
    total_h = log.(r .+ λ_h)
    total_a = log.(r .+ λ_a)
    ll_h = loggamma.(data.yh .+ r) .- loggamma(r) .- data.lfh .+
           r .* (log(r) .- total_h) .+ data.yh .* (η_h .- total_h)
    ll_a = loggamma.(data.ya .+ r) .- loggamma(r) .- data.lfa .+
           r .* (log(r) .- total_a) .+ data.ya .* (η_a .- total_a)
    return sum(data.weights .* ll_h) + sum(data.weights .* ll_a)
end

"""
Two-arm joint log-likelihood, written from the DISTRIBUTIONS rather than from the engine's
hand-expanded form. This is the whole value of the file: `engine.jl` inlines

    log Gamma(x; ν, μ/ν) = (ν−1)·log x − ν·x·e^(−η) − ν·η + ν·log ν − log Γ(ν)

for the tape's sake, and an algebra slip there would produce a perfectly smooth, perfectly
differentiable, perfectly wrong posterior. `logpdf(Gamma(ν, μ/ν), x)` cannot make that slip.
"""
function cb_loglik(::JointGammaPoissonObservation, p::CBParams, data, η_h, η_a)
    μ_h = exp.(η_h)
    μ_a = exp.(η_a)
    κ = exp(p.log_κ)

    goals = sum(data.weights .* logpdf.(Poisson.(κ .* μ_h), data.yh)) +
            sum(data.weights .* logpdf.(Poisson.(κ .* μ_a), data.ya))

    proxy = sum(data.weights .* data.mask .* logpdf.(Gamma.(p.ν, μ_h ./ p.ν), data.pxg_h)) +
            sum(data.weights .* data.mask .* logpdf.(Gamma.(p.ν, μ_a ./ p.ν), data.pxg_a))

    return goals + proxy
end
