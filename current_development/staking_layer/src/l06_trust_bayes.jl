#=
LOADER l06 — BayesianTrust (Turing) + HierarchicalTrust (documented stub).

Roadmap step 2: replace the EB grid/marginal-likelihood search (l05) with a proper Turing model
that samples the trust weights jointly with the pooling hyperparameters. The output is an MCMC
chain of w's, so distributional staking (step 1) comes for free — `trust_draws` returns the chain.

    w0 ~ Normal(0, 1.5)                      # global logit-trust mean
    τ  ~ truncated(Normal(0,1); lower=0)     # between-unit spread (partial pooling)
    z_u ~ Normal(0,1),  w_u = logistic(w0 + τ·z_u)      # non-centred (7 units)
    y_i ~ Bernoulli(w_{u(i)}·p_i + (1−w_{u(i)})·q_i)    # blended-prob likelihood

EB (l05) is the MAP/EB shadow of exactly this model, so the posterior means should land near the
EB weights on the same history (verification 5). Vectorised, non-centred → AD-safe & well-mixing.

HierarchicalTrust (step 3, the per-team payoff) is a documented stub: the team-id plumbing already
lands (TrustHist.home/away, StakingMatch.home/away), so implementing it is adding the `@model`
below and a team-indexed `trust_weights`. Left unimplemented to honour the agreed scope.

Depends on l04 (interface), l01 (schema). Pulls Turing (a BayesianFootball dependency).
=#

using Turing
using LogExpFunctions: logistic
using Statistics
using Random
using LinearAlgebra: normalize

# ---------- flat (non-hierarchical) Bayesian trust ----------

"""
Full Bayesian per-unit trust via NUTS. `sampler`/`nsamples`/`nadapt` control sampling; `seed`
makes the fit reproducible. Small model (7 units, Bernoulli) → cheap to sample locally.
"""
Base.@kwdef struct BayesianTrust <: AbstractTrustModel
    nsamples::Int   = 800
    nadapt::Int     = 500
    accept::Float64 = 0.8
    seed::Int       = 20260707
    w0_cold::Float64 = 0.5
end

@model function _bayes_trust_model(p, q, y, unit, U)
    w0 ~ Normal(0.0, 1.5)
    τ  ~ truncated(Normal(0.0, 1.0); lower=0.0)
    z  ~ filldist(Normal(0.0, 1.0), U)
    w  = logistic.(w0 .+ τ .* z)                 # U-vector of unit weights
    p̃  = clamp.(w[unit] .* p .+ (1.0 .- w[unit]) .* q, 1e-9, 1 - 1e-9)
    y ~ product_distribution(Bernoulli.(p̃))      # DynamicPPL ≥0.35: no arrays-of-dists in .~
    return w
end

struct FittedBayesTrust
    w::Vector{Float64}          # posterior mean per unit (7)
    wdraws::Matrix{Float64}     # 7 × nsamples posterior draws of w
    chain::Any
end

"Flatten the per-unit history into (p, q, y, unit) obs vectors."
function _flatten_hist(h::TrustHist)
    p = Float64[]; q = Float64[]; y = Float64[]; unit = Int[]
    for u in 1:7, i in eachindex(h.y[u])
        push!(p, h.p[u][i]); push!(q, h.q[u][i]); push!(y, h.y[u][i]); push!(unit, u)
    end
    return p, q, y, unit
end

function fit_trust(model::BayesianTrust, h::TrustHist)
    if nobs(h) == 0
        wd = fill(model.w0_cold, 7, model.nsamples)
        return FittedBayesTrust(fill(model.w0_cold, 7), wd, nothing)
    end
    p, q, y, unit = _flatten_hist(h)
    m = _bayes_trust_model(p, q, y, unit, 7)
    rng = Xoshiro(model.seed)
    chn = sample(rng, m, NUTS(model.nadapt, model.accept), model.nsamples; progress=false)

    w0s = vec(Array(chn[:w0])); τs = vec(Array(chn[:τ]))
    ns = length(w0s)
    wdraws = Matrix{Float64}(undef, 7, ns)
    for u in 1:7
        zu = vec(Array(chn[Symbol("z[$u]")]))    # ns samples of z_u
        @inbounds for s in 1:ns
            wdraws[u, s] = logistic(w0s[s] + τs[s] * zu[s])
        end
    end
    w = vec(mean(wdraws, dims=2))
    return FittedBayesTrust(w, wdraws, chn)
end

trust_weights(ft::FittedBayesTrust, ::StakingMatch) = ft.w
trust_weights(ft::FittedBayesTrust) = ft.w

function trust_draws(ft::FittedBayesTrust, ::StakingMatch; D::Int=64, rng=nothing)
    ns = size(ft.wdraws, 2)
    idx = ns <= D ? (1:ns) : round.(Int, range(1, ns, length=D))
    return ft.wdraws[:, collect(idx)]
end

# ---------- hierarchical per-team trust (roadmap step 3) ----------

"""
Per-team hierarchical trust. Two pooling axes: per-unit global trust `w0_u`, and a per-team
random effect `b_{u,t}` that shrinks to zero when a team is thin (`σ_u` = between-team spread of
unit u — the key output: σ_u ≈ 0 ⇒ no real team variation on unit u, so hierarchy buys nothing
there; σ_u > 0 ⇒ that unit's trust genuinely differs by team).

    w0_u ~ Normal(0, 1.5)
    σ_u  ~ truncated(Normal(0, 0.75); lower=0)          # between-team spread, per unit
    z_{u,t} ~ Normal(0,1),  w_{u,t} = logistic(w0_u + σ_u·z_{u,t})   # non-centred
    y_i ~ Bernoulli( w_{u(i), team(i)}·p_i + (1 − w_{u(i), team(i)})·q_i )

Teams are grouped by the match's HOME team (the axis r05 measured). `trust_weights(ft, m)` reads
`m.home`; unseen teams fall back to the pooled `logistic(w0_u)`.
"""
Base.@kwdef struct HierarchicalTrust <: AbstractTrustModel
    nsamples::Int   = 800
    nadapt::Int     = 500
    accept::Float64 = 0.8
    seed::Int       = 20260707
    σ_prior::Float64 = 0.75      # half-Normal scale on the between-team spread σ_u
    w0_cold::Float64 = 0.5
end

@model function _hier_trust_model(p, q, y, unit, lin, U, UT, σ_prior)
    w0 ~ filldist(Normal(0.0, 1.5), U)
    σ  ~ filldist(truncated(Normal(0.0, σ_prior); lower=0.0), U)
    z  ~ filldist(Normal(0.0, 1.0), UT)
    w  = logistic.(w0[unit] .+ σ[unit] .* z[lin])       # per-observation weight
    p̃  = clamp.(w .* p .+ (1.0 .- w) .* q, 1e-9, 1 - 1e-9)
    y ~ product_distribution(Bernoulli.(p̃))
    return w
end

"""
Fitted hierarchical trust. `wmean` is 7 × T (dense teams); `teammap` sends a raw team id to its
dense column; `pooled_w`/`w0draws` are the fallback for unseen teams; `σ` = between-team spread
posterior mean per unit (the hierarchy verdict).
"""
struct FittedHierTrust
    w0::Vector{Float64}            # 7 posterior-mean global logit-trust
    σ::Vector{Float64}             # 7 between-team spread (posterior mean)
    wmean::Matrix{Float64}         # 7 × T posterior-mean per-team weights
    wdraws::Array{Float64,3}       # 7 × T × S per-team weight draws
    w0draws::Matrix{Float64}       # 7 × S pooled (unseen-team) weight draws = logistic(w0)
    pooled_w::Vector{Float64}      # 7 fallback point weights
    teammap::Dict{Int,Int}         # raw team id => dense column
    team_names_dense::Vector{Int}  # dense column => raw team id
end

"Flatten history using the HOME team as the grouping factor; returns dense team indexing."
function _flatten_hist_team(h::TrustHist)
    p = Float64[]; q = Float64[]; y = Float64[]; unit = Int[]; teamraw = Int[]
    for u in 1:7, i in eachindex(h.y[u])
        push!(p, h.p[u][i]); push!(q, h.q[u][i]); push!(y, h.y[u][i]); push!(unit, u); push!(teamraw, h.home[u][i])
    end
    teams = sort(unique(teamraw)); teammap = Dict(t => j for (j, t) in enumerate(teams))
    tdense = [teammap[t] for t in teamraw]
    return p, q, y, unit, tdense, teammap, teams
end

function fit_trust(model::HierarchicalTrust, h::TrustHist)
    if nobs(h) == 0
        w0d = fill(0.0, 7, model.nsamples)
        return FittedHierTrust(zeros(7), zeros(7), fill(model.w0_cold, 7, 1),
                               fill(model.w0_cold, 7, 1, model.nsamples), fill(model.w0_cold, 7, model.nsamples),
                               fill(model.w0_cold, 7), Dict{Int,Int}(), Int[])
    end
    p, q, y, unit, tdense, teammap, teams = _flatten_hist_team(h)
    U = 7; T = length(teams); UT = U * T
    lin = @. unit + (tdense - 1) * U                    # linear index into the U*T z-vector
    m = _hier_trust_model(p, q, y, unit, lin, U, UT, model.σ_prior)
    rng = Xoshiro(model.seed)
    chn = sample(rng, m, NUTS(model.nadapt, model.accept), model.nsamples; progress=false)

    W0 = reduce(hcat, [vec(Array(chn[Symbol("w0[$u]")])) for u in 1:U])'   # U × S
    Σ  = reduce(hcat, [vec(Array(chn[Symbol("σ[$u]")]))  for u in 1:U])'   # U × S
    S = size(W0, 2)
    Z = Array{Float64,3}(undef, U, T, S)
    for u in 1:U, t in 1:T
        Z[u, t, :] = vec(Array(chn[Symbol("z[$(u + (t - 1) * U)]")]))
    end
    wdraws = Array{Float64,3}(undef, U, T, S)
    for u in 1:U, t in 1:T, s in 1:S
        wdraws[u, t, s] = logistic(W0[u, s] + Σ[u, s] * Z[u, t, s])
    end
    w0draws = logistic.(W0)                              # U × S pooled fallback
    wmean = dropdims(mean(wdraws, dims=3), dims=3)       # U × T
    return FittedHierTrust(vec(mean(W0, dims=2)), vec(mean(Σ, dims=2)), wmean, wdraws, w0draws,
                           vec(mean(w0draws, dims=2)), teammap, teams)
end

function trust_weights(ft::FittedHierTrust, m::StakingMatch)
    t = get(ft.teammap, m.home, 0)
    return t == 0 ? ft.pooled_w : ft.wmean[:, t]
end
trust_weights(ft::FittedHierTrust) = ft.pooled_w     # match-free fallback = pooled

function trust_draws(ft::FittedHierTrust, m::StakingMatch; D::Int=64, rng=nothing)
    t = get(ft.teammap, m.home, 0)
    src = t == 0 ? ft.w0draws : @view ft.wdraws[:, t, :]
    ns = size(src, 2)
    idx = ns <= D ? (1:ns) : round.(Int, range(1, ns, length=D))
    return Array(src[:, collect(idx)])
end
