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

function trust_draws(ft::FittedBayesTrust, ::StakingMatch; D::Int=64, rng::AbstractRNG=Random.default_rng())
    ns = size(ft.wdraws, 2)
    idx = ns <= D ? (1:ns) : round.(Int, range(1, ns, length=D))
    return ft.wdraws[:, collect(idx)]
end

# ---------- hierarchical per-team trust (STUB — roadmap step 3) ----------

"""
Per-team hierarchical trust (NOT YET IMPLEMENTED — the immediate next experiment).

The team-id plumbing is already in place (`TrustHist.home/away`, `StakingMatch.home/away`), so
implementing this is: add the `@model` below and a team-indexed `trust_weights(ft, m)` that reads
`m.home`/`m.away`. Sketch (per unit u, team t):

    w0_u ~ Normal(0, 1.5);   σ_u ~ truncated(Normal(0,1); lower=0)
    b_{u,t} ~ Normal(0, σ_u)                      # team random effect, shrinks to 0 when thin
    w_{u,t} = logistic(w0_u + b_{u,t})
    y_i ~ Bernoulli( w_{u(i), team(i)}·p_i + (1 − w_{u(i), team(i)})·q_i )

Run the l05/l06 EB-vs-Bayes race first, then Step-0 EDA (r05) to confirm per-team w actually
separates before paying for this second (team) pooling axis.
"""
Base.@kwdef struct HierarchicalTrust <: AbstractTrustModel
    nsamples::Int = 800
    nadapt::Int   = 500
    accept::Float64 = 0.8
    seed::Int     = 20260707
end

fit_trust(::HierarchicalTrust, ::TrustHist) =
    error("HierarchicalTrust: not implemented — roadmap step 3. Team ids are already captured " *
          "in TrustHist.home/away; see the docstring for the @model to add.")
