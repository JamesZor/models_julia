#=
LOADER l05 — EBTrust: empirical-Bayes partial-pooled per-unit trust.

The first concrete AbstractTrustModel and the PARITY target — a faithful port of the old
staking_sim/l02 `fit_trust_eb` (itself the Julia port of docs/bets_multi/_verify_trust.py T4).

Per unit u: a Bernoulli log-likelihood of the realised win/loss over a grid of w∈[0,1]; a shared
logit-normal prior N(logit w; w0, τ) whose (w0, τ) is chosen by marginal likelihood (partial
pooling — thin-history units are shrunk toward the common w0). The point estimate is the posterior
mean; the FULL grid posterior is retained so `trust_draws` can sample it (distributional staking).

`halflife` (in observations) exp2-down-weights old matches; Inf = full memory (r04: bias is
stationary → full memory wins). Depends on l04 (interface), l01 (UNIT_REP_SEL).
=#

using LogExpFunctions: logit
using Random

"""
EB partial-pooled per-unit trust. Grid `wgrid`; prior hyper-grid `(w0grid, τgrid)` selected by
marginal likelihood. `halflife` in observations (Inf = full memory).
"""
Base.@kwdef struct EBTrust <: AbstractTrustModel
    wgrid::Vector{Float64}       = collect(0.0:0.005:1.0)
    w0grid::Vector{Float64}      = collect(range(-2.0, 2.0, length=17))
    τgrid::Vector{Float64}       = [0.25, 0.5, 1.0, 2.0]
    halflife::Float64            = Inf
    w0_cold::Float64             = 0.5    # per-unit w before any data (returned when n==0)
end

"Fitted EBTrust: posterior means `w` (7) + per-unit grid posteriors `post` (7 × nw) + hyperparams."
struct FittedEBTrust
    w::Vector{Float64}
    post::Vector{Vector{Float64}}
    wgrid::Vector{Float64}
    w0::Float64
    τ::Float64
end

function fit_trust(model::EBTrust, h::TrustHist)
    wgrid, w0grid, τgrid, halflife = model.wgrid, model.w0grid, model.τgrid, model.halflife
    nw = length(wgrid)
    if nobs(h) == 0                                   # cold: no evidence → flat cold weights
        flat = fill(1.0 / nw, nw)
        return FittedEBTrust(fill(model.w0_cold, 7), [copy(flat) for _ in 1:7], wgrid, 0.0, 1.0)
    end

    LL = zeros(7, nw)
    for u in 1:7
        n = length(h.y[u])
        n == 0 && continue
        for (wi, w) in enumerate(wgrid)
            s = 0.0
            @inbounds for i in 1:n
                p̃ = clamp(w * h.p[u][i] + (1.0 - w) * h.q[u][i], 1e-9, 1 - 1e-9)
                ll = h.y[u][i] * log(p̃) + (1.0 - h.y[u][i]) * log1p(-p̃)
                s += isinf(halflife) ? ll : exp2(-(n - i) / halflife) * ll
            end
            LL[u, wi] = s
        end
    end

    zg = logit.(clamp.(wgrid, 1e-4, 1 - 1e-4))
    best, bw0, bτ = -Inf, 0.0, 1.0
    for w0 in w0grid, τ in τgrid
        lp = @. -0.5 * ((zg - w0) / τ)^2
        marg = 0.0
        for u in 1:7
            mx = maximum(view(LL, u, :))
            marg += mx + log(sum(exp.(view(LL, u, :) .- mx .+ lp)))
        end
        marg > best && ((best, bw0, bτ) = (marg, w0, τ))
    end

    lp = @. -0.5 * ((zg - bw0) / bτ)^2
    w = zeros(7)
    post = Vector{Vector{Float64}}(undef, 7)
    for u in 1:7
        mx = maximum(view(LL, u, :))
        p = exp.(view(LL, u, :) .- mx .+ lp)
        p ./= sum(p)
        post[u] = p
        w[u] = sum(p .* wgrid)
    end
    return FittedEBTrust(w, post, wgrid, bw0, bτ)
end

trust_weights(ft::FittedEBTrust, ::StakingMatch) = ft.w
trust_weights(ft::FittedEBTrust) = ft.w

"Sample each unit's grid posterior D times → 7 × D. Trust uncertainty flows into staking."
function trust_draws(ft::FittedEBTrust, ::StakingMatch; D::Int=64, rng::AbstractRNG=Random.default_rng())
    W = Matrix{Float64}(undef, 7, D)
    for u in 1:7
        cdf = cumsum(ft.post[u])
        @inbounds for d in 1:D
            r = rand(rng)
            k = searchsortedfirst(cdf, r)
            W[u, d] = ft.wgrid[min(k, length(ft.wgrid))]
        end
    end
    return W
end
