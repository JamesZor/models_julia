#=
LOADER l07 — staking policies: how a match + a trust model become stakes.

A policy maps (match, model targets, fitted trust) → an 11-vector of stakes. Concrete policies:

  FlatPolicy         flat f on every model-edge ≥ min_edge selection.
  PerBetKellyPolicy  per-selection Kelly (a src/Signals AbstractSignal) on the model draws,
                     portfolio-capped Σa ≤ cap. The b21-comparable baseline.
  UnifiedPolicy      the structural (P) portfolio: trust-blend → coherent IPF tilt → capped Kelly.
                     `trust` is any AbstractTrustModel; `distributional=true` averages the Kelly
                     solve over trust_draws (trust uncertainty → stake shrinkage).

The runner (l10) owns the trust refit cadence and passes each policy its current fitted trust.
Add a staking system = add a struct + a `stake_for` method. Depends on l01–l06 + src/Signals.
=#

using Statistics
using Random
using BayesianFootball
const SIG = BayesianFootball.Signals

# ---------- shared per-match pieces ----------

match_return(a::Vector{Float64}, m::StakingMatch) =
    1.0 - sum(a) + sum(a[k] * m.d[k] * m.won[k] for k in 1:11)

"Rescale per-bet stakes if their total exceeds a portfolio guard."
function guard!(a::Vector{Float64}; cap::Float64=0.98)
    s = sum(a)
    s > cap && (a .*= cap / s)
    return a
end

# ---------- policies ----------

abstract type AbstractStakingPolicy end

Base.@kwdef struct FlatPolicy <: AbstractStakingPolicy
    f::Float64 = 0.01
    min_edge::Float64 = 0.03
end

Base.@kwdef struct PerBetKellyPolicy <: AbstractStakingPolicy
    signal::Any = SIG.BayesianKelly(0.03)
    cap::Float64 = 0.2
end

Base.@kwdef struct UnifiedPolicy <: AbstractStakingPolicy
    trust::AbstractTrustModel = CuratedTrust()
    cap::Float64 = 0.2
    distributional::Bool = false
    D::Int = 64
    cycles::Int = 50
    w0_start::Float64 = 0.5     # cold per-unit weight before the first refit
end

needs_trust(::AbstractStakingPolicy) = false
needs_trust(::UnifiedPolicy) = true

# ---------- stake_for ----------

"Flat f on selections whose model edge (model_sel − 1/d) clears min_edge."
function stake_for(p::FlatPolicy, m::StakingMatch, model_sel, model_dists, fitted; rng=Random.default_rng())
    a = zeros(11)
    for k in 1:11
        model_sel[k] - 1.0 / m.d[k] >= p.min_edge && (a[k] = p.f)
    end
    return guard!(a)
end

"Per-selection Kelly on the model draws, portfolio-capped."
function stake_for(p::PerBetKellyPolicy, m::StakingMatch, model_sel, model_dists, fitted; rng=Random.default_rng())
    a = zeros(11)
    for k in 1:11
        a[k] = SIG.compute_stake(p.signal, view(model_dists, k, :), m.d[k])
    end
    return guard!(a; cap=p.cap)
end

"Structural (P): trust-blend → coherent tilt → capped Kelly. Distributional = average over trust draws."
function stake_for(p::UnifiedPolicy, m::StakingMatch, model_sel, model_dists, fitted; rng=Random.default_rng())
    if !p.distributional
        w = trust_weights(fitted, m)
        mult = coherent_multiplier(m.pbar, blend_targets(model_sel, m.q_mkt, w); cycles=p.cycles)
        return solve_P(normalize_mult(m.pbar, mult), m.R; cap=p.cap)
    end
    W = trust_draws(fitted, m; D=p.D, rng=rng)
    a = zeros(size(m.R, 2))
    for d in 1:size(W, 2)
        mult = coherent_multiplier(m.pbar, blend_targets(model_sel, m.q_mkt, view(W, :, d)); cycles=p.cycles)
        a .+= solve_P(normalize_mult(m.pbar, mult), m.R; cap=p.cap)
    end
    a ./= size(W, 2)
    return a
end
