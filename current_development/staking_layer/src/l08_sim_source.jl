#=
LOADER l08 — SimSource: the simulated double-Poisson league (truth + market + model observers).

The synthetic match source. A "campaign" is a chained round-robin (Ireland-like: 10 teams) with
team strengths drifting by per-round zero-sum random walks. Three observers per match: TRUTH (the
score), MARKET (noisy λ → de-vigged quotes with optional favourite-longshot shading), MODEL (noisy
λ + per-line bias tilt → S posterior draws). Extracted verbatim from the old
staking_sim/l01_sim_market_model.jl, minus the schema (now l01) and solver (now l02); `sim_match`
now emits a `StakingMatch` (team ids threaded in) and returns `p_true` as a side-channel.

`SimSource(cfg, seed)` implements the AbstractMatchSource contract: `load_matches` →
(matches, model_sel, model_dists, teams, mids, p_true, prehist). Depends on l01–l02.
=#

using Distributions
using Random
using Statistics

# quoting groups: each is one market whose fair probs sum to 1 (vig/FLB applied per group)
const MKT_GROUPS = ((1, 2, 3), (4, 5), (6, 7), (8, 9), (10, 11))
const TOTG  = Float64.(HGRID .+ AGRID)
const BTTSG = Float64.((HGRID .>= 1) .& (AGRID .>= 1))

Base.@kwdef struct SimConfig
    n_teams::Int = 10
    n_matches::Int = 330
    n_prehist::Int = 270          # no-betting pre-history for trust warm-start (0 = cold)
    μ::Float64 = 0.0532
    ha::Float64 = 0.2459
    σ0::Float64 = 0.20
    σ_in::Float64 = 0.015
    σ_ha::Float64 = 0.01
    σ_mkt::Float64 = 0.08
    σ_mod::Float64 = 0.05
    σ_mod_lvl::Float64 = NaN
    σ_mod_sup::Float64 = NaN
    γ_tot::Float64 = -0.05
    γ_btts::Float64 = 0.10
    σ_post::Float64 = 0.05
    S::Int = 100
    O_1x2::Float64 = 1.0841
    O_ou::Float64 = 1.0491
    O_btts::Float64 = 1.0696
    ρ_flb::Float64 = 1.0
    devig_quotes::Bool = false
    ruin_floor::Float64 = 0.01
end

"Circle-method round robin: vector of rounds, each a vector of (home, away). n_teams even."
function round_robin_rounds(n_teams::Int)
    ts = collect(1:n_teams)
    rounds = Vector{Vector{Tuple{Int,Int}}}()
    for r in 1:(n_teams-1)
        pairs = Tuple{Int,Int}[]
        for i in 1:(n_teams ÷ 2)
            a, b = ts[i], ts[n_teams+1-i]
            push!(pairs, isodd(r) ? (a, b) : (b, a))
        end
        push!(rounds, pairs)
        ts = vcat(ts[1], ts[end], ts[2:end-1])
    end
    return rounds
end

"Double-Poisson 12×12 grid (flattened, renormalized)."
function dp_grid(λh::Float64, λa::Float64)
    ph = pdf.(Poisson(λh), 0:GG-1)
    pa = pdf.(Poisson(λa), 0:GG-1)
    g = vec(ph * pa')
    return g ./ sum(g)
end

"Apply the model's per-line bias tilt in place: g ∝ g·exp(γ_tot·(H+A) + γ_btts·1[BTTS])."
function bias_tilt!(g::Vector{Float64}, γ_tot::Float64, γ_btts::Float64)
    (γ_tot == 0.0 && γ_btts == 0.0) && return g
    g .*= exp.(γ_tot .* TOTG .+ γ_btts .* BTTSG)
    g ./= sum(g)
    return g
end

"Simulate one match given true rates + team ids. Returns (StakingMatch, p_true)."
function sim_match(cfg::SimConfig, λh::Float64, λa::Float64, home::Int, away::Int,
                   rng::AbstractRNG; S::Int=cfg.S)
    h = rand(rng, Poisson(λh)); a = rand(rng, Poisson(λa))

    gm = dp_grid(λh * exp(cfg.σ_mkt * randn(rng)), λa * exp(cfg.σ_mkt * randn(rng)))
    q = MMASK' * gm
    O = (cfg.O_1x2, cfg.O_ou, cfg.O_btts)
    d = Vector{Float64}(undef, 11)
    for grp in MKT_GROUPS
        Ofam = O[FAM_OF_SEL[grp[1]]]
        z = sum(q[m]^cfg.ρ_flb for m in grp)
        for m in grp
            d[m] = z / (q[m]^cfg.ρ_flb * Ofam)
        end
    end
    if cfg.devig_quotes
        for grp in MKT_GROUPS
            s = sum(1.0 / d[m] for m in grp)
            for m in grp
                q[m] = (1.0 / d[m]) / s
            end
        end
    end

    if isnan(cfg.σ_mod_lvl)
        λmh = λh * exp(cfg.σ_mod * randn(rng))
        λma = λa * exp(cfg.σ_mod * randn(rng))
    else
        εl = cfg.σ_mod_lvl * randn(rng)
        εs = cfg.σ_mod_sup * randn(rng)
        λmh = λh * exp(εl + εs)
        λma = λa * exp(εl - εs)
    end
    corr = exp(-cfg.σ_post^2 / 2)
    P = Matrix{Float64}(undef, GG * GG, S)
    for s in 1:S
        g = dp_grid(λmh * corr * exp(cfg.σ_post * randn(rng)),
                    λma * corr * exp(cfg.σ_post * randn(rng)))
        P[:, s] = bias_tilt!(g, cfg.γ_tot, cfg.γ_btts)
    end
    pbar = vec(mean(P, dims=2))

    R = return_matrix(d)
    gt = dp_grid(λh, λa)
    sm = StakingMatch(d, q, P, pbar, settle_score(h, a), R, home, away, (h, a))
    return sm, MMASK' * gt      # p_true side-channel
end

"""
Simulate `n` matches with drifting strengths. Returns (matches, p_true) as parallel vectors.
Truth drift mirrors SyntheticData: per-round N(0,σ_in) increments on α,β (zero-sum) + N(0,σ_ha) on ha.
"""
function simulate_campaign(cfg::SimConfig, rng::AbstractRNG; n_matches::Int=cfg.n_matches, S::Int=cfg.S)
    nt = cfg.n_teams
    α = cfg.σ0 .* randn(rng, nt); α .-= mean(α)
    β = cfg.σ0 .* randn(rng, nt); β .-= mean(β)
    ha = cfg.ha
    rounds = round_robin_rounds(nt)
    matches = Vector{StakingMatch}(undef, n_matches)
    ptrue   = Vector{Vector{Float64}}(undef, n_matches)
    k, leg = 0, 0
    while k < n_matches
        for rnd in rounds
            α .+= cfg.σ_in .* randn(rng, nt); α .-= mean(α)
            β .+= cfg.σ_in .* randn(rng, nt); β .-= mean(β)
            ha += cfg.σ_ha * randn(rng)
            for (hm, aw) in rnd
                k == n_matches && break
                home, away = isodd(leg) ? (aw, hm) : (hm, aw)
                λh = exp(cfg.μ + α[home] + β[away] + ha)
                λa = exp(cfg.μ + α[away] + β[home])
                k += 1
                matches[k], ptrue[k] = sim_match(cfg, λh, λa, home, away, rng; S=S)
            end
            k == n_matches && break
        end
        leg += 1
    end
    return matches, ptrue
end

# ---------- source contract ----------

abstract type AbstractMatchSource end

"Simulated match source. `n_prehist` no-bet matches are returned as a warm-start prehistory."
struct SimSource <: AbstractMatchSource
    cfg::SimConfig
    seed::Int
end
SimSource(cfg::SimConfig; seed::Int=1) = SimSource(cfg, seed)

"""
    load_matches(src) -> (; matches, model_sel, model_dists, teams, mids, p_true, prehist)

`model_sel[i]` = 11-vector of model probs (grid-derived), `model_dists[i]` = 11 × S draws,
`p_true[i]` = 11-vector of TRUE probs (diagnostics/oracle), `prehist` = (matches, model_sel) for
the no-bet warm-start (empty when n_prehist == 0).
"""
function load_matches(src::SimSource)
    cfg = src.cfg
    rng = Xoshiro(src.seed)

    prehist_matches = StakingMatch[]
    if cfg.n_prehist > 0
        prehist_matches, _ = simulate_campaign(cfg, rng; n_matches=cfg.n_prehist, S=16)
    end

    matches, ptrue = simulate_campaign(cfg, rng)
    model_sel   = [Vector(MMASK' * m.pbar) for m in matches]
    model_dists = [MMASK' * m.P for m in matches]
    prehist_sel = [Vector(MMASK' * m.pbar) for m in prehist_matches]
    teams = [(m.home, m.away) for m in matches]
    mids  = collect(1:length(matches))
    return (; matches, model_sel, model_dists, teams, mids, p_true=ptrue,
            prehist=(matches=prehist_matches, model_sel=prehist_sel))
end
