#=
LOADER — simulated double-Poisson league: truth + market observer + model observer + books.

A "campaign" is a chained round-robin schedule (Ireland Premier-like: 10 teams) with team
strengths drifting by per-round zero-sum-centred random walks. Drift maths mirrors
`src/synthetic/synthetic-data-module.jl` (which we deliberately do NOT call: it seeds the
global RNG — thread-unsafe for the MC loop — has no league-rate intercept μ, and round-trips
a DataStore we don't need). λ convention identical:
    λ_h = exp(μ + α_h + β_a + ha),   λ_a = exp(μ + α_a + β_h).

Three observers of each match:
  • TRUTH   λ_h, λ_a                        → the score  ~ Poisson each side.
  • MARKET  λ·exp(ε), ε~N(0,σ_mkt²) fresh   → its own 12×12 grid → de-vigged q_sel;
            quoted implied prob π ∝ q^ρ_flb within each market group, scaled to O_fam
            (ρ_flb = 1 recovers plain multiplicative vig; ρ_flb < 1 = favourite–longshot
            bias: longshot odds shaded shorter, favourites relatively better value).
            `devig_quotes=true` makes q_mkt the PROPORTIONAL de-vig of the quoted odds
            (what the real pipeline can observe) instead of the market's fair belief —
            under FLB the two differ, and the real pipeline gets the contaminated one.
  • MODEL   λ·exp(ε), ε~N(0,σ_mod²)         → grid → PER-LINE BIAS TILT applied
            (γ_tot totals compression + γ_btts BTTS boost, the li_smile50 signature from
            score_matrix_calibration/experiments.md); "posterior" = S mean-corrected
            lognormal λ draws, each gridded WITH the same tilt → P (144×S).
            Bias-before-draws makes Baker–McHale (variance-only) vs trust blend (bias-aware)
            distinguishable — the central experimental contrast.

Book per match: 11 selections — 1X2 (3) + O/U 1.5/2.5/3.5 (6) + BTTS (2), fixed order.
Settlement (`won`) uses true score semantics directly (avoids GG=12 truncation edge cases).

Depends on unified_staking/l01 (GG, HGRID, AGRID, proj_cap!, G_growth, solve_P) — included
below with a guard.
=#

using Distributions
using Random
using Statistics

if !@isdefined(solve_P)
    include(joinpath(@__DIR__, "..", "unified_staking", "l01_structural_kelly.jl"))
end

# ---------- fixed book layout (11 selections) ----------

const SEL_NAMES = ["home", "draw", "away",
                   "over_15", "under_15", "over_25", "under_25", "over_35", "under_35",
                   "btts_yes", "btts_no"]
const SEL_SPECS = [("1X2", 0.0, "home"), ("1X2", 0.0, "draw"), ("1X2", 0.0, "away"),
                   ("OverUnder", 1.5, "over"), ("OverUnder", 1.5, "under"),
                   ("OverUnder", 2.5, "over"), ("OverUnder", 2.5, "under"),
                   ("OverUnder", 3.5, "over"), ("OverUnder", 3.5, "under"),
                   ("BTTS", 0.0, "btts_yes"), ("BTTS", 0.0, "btts_no")]
const SEL_MASKS = [BitVector(mask_for(n, l, s)) for (n, l, s) in SEL_SPECS]
const MMASK = Float64.(hcat(SEL_MASKS...))          # 144 × 11
const FAM_OF_SEL = [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3]  # 1=1X2, 2=OU, 3=BTTS
# quoting groups: each is one market whose fair probs sum to 1 (vig/FLB applied per group)
const MKT_GROUPS = ((1, 2, 3), (4, 5), (6, 7), (8, 9), (10, 11))

# trust units: 7 lines (complement shares its unit's w); representative sel per unit
const UNIT_OF_SEL = [1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7]
const UNIT_REP_SEL = [1, 2, 3, 4, 6, 8, 10]          # home, draw, away, o15, o25, o35, btts_yes
const UNIT_NAMES = ["home", "draw", "away", "over_15", "over_25", "over_35", "btts_yes"]

const TOTG = Float64.(HGRID .+ AGRID)
const BTTSG = Float64.((HGRID .>= 1) .& (AGRID .>= 1))

# ---------- config ----------

Base.@kwdef struct SimConfig
    n_teams::Int = 10
    n_matches::Int = 330        # betting campaign length (66 rounds of 5)
    n_prehist::Int = 270        # no-betting pre-history for trust warm-start (0 = cold start)
    # league level — calibrated by r01_calibrate_ireland.jl (Ireland Premier 79, n=1013,
    # 2026-07-04: mean goals 1.404/1.098, HW 42.8%, draw 27.5%, over2.5 47.0%)
    μ::Float64 = 0.0532         # log base away rate = log(1.098) − σ0²
    ha::Float64 = 0.2459        # home advantage = log(1.404/1.098)
    σ0::Float64 = 0.20          # initial α,β spread (SyntheticData default)
    σ_in::Float64 = 0.015       # per-round strength RW vol (SyntheticData default)
    σ_ha::Float64 = 0.01        # per-round home-adv RW vol
    # observers
    σ_mkt::Float64 = 0.08       # market log-λ obs noise (fresh per match/side)
    σ_mod::Float64 = 0.05       # model log-λ obs noise (< σ_mkt ⇒ genuine info edge)
    # optional level/supremacy decomposition of the MODEL's noise (E4 sup-blind worlds):
    # ε_h = ε_lvl + ε_sup, ε_a = ε_lvl − ε_sup. i.i.d. σ_mod ⇔ σ_lvl = σ_sup = σ_mod/√2
    # (market i.i.d. σ_mkt ⇒ 0.0566 per component). NaN = use i.i.d. σ_mod (default).
    σ_mod_lvl::Float64 = NaN    # model noise on the level component (total intensity)
    σ_mod_sup::Float64 = NaN    # model noise on the supremacy component (home/away split)
    γ_tot::Float64 = -0.05      # model totals tilt on H+A (pure bias)
    γ_btts::Float64 = 0.10      # model BTTS-yes tilt (pure bias)
    σ_post::Float64 = 0.05      # posterior width on log-λ
    S::Int = 100                # posterior draws
    # market frictions — per-family closing overrounds (r01, Ireland empirical)
    O_1x2::Float64 = 1.0841
    O_ou::Float64 = 1.0491
    O_btts::Float64 = 1.0696
    ρ_flb::Float64 = 1.0        # favourite–longshot shading exponent (π ∝ q^ρ; 1 = off)
    devig_quotes::Bool = false  # q_mkt from proportional de-vig of quotes (realistic under FLB)
    ruin_floor::Float64 = 0.01  # wealth below this ⇒ ruined, betting frozen
end

# ---------- schedule ----------

"Circle-method round robin: vector of rounds, each a vector of (home, away). n_teams even."
function round_robin_rounds(n_teams::Int)
    ts = collect(1:n_teams)
    rounds = Vector{Vector{Tuple{Int,Int}}}()
    for r in 1:(n_teams-1)
        pairs = Tuple{Int,Int}[]
        for i in 1:(n_teams ÷ 2)
            a, b = ts[i], ts[n_teams+1-i]
            push!(pairs, isodd(r) ? (a, b) : (b, a))   # alternate orientation
        end
        push!(rounds, pairs)
        ts = vcat(ts[1], ts[end], ts[2:end-1])         # rotate all but the first
    end
    return rounds
end

# ---------- grids ----------

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

# ---------- one simulated match ----------

struct SimMatch
    d::Vector{Float64}       # quoted odds (11)
    q_mkt::Vector{Float64}   # de-vigged market prob per selection (11)
    P::Matrix{Float64}       # model posterior state grids (144 × S), bias-tilted
    pbar::Vector{Float64}    # mean over draws (144)
    won::Vector{Bool}        # settlement per selection (11), from true score
    R::Matrix{Float64}       # return matrix d.*mask .- 1 (144 × 11)
    p_true::Vector{Float64}  # TRUE selection probs (11) — diagnostics/oracle only
    score::Tuple{Int,Int}
end

function settle_score(h::Int, a::Int)
    Bool[h > a, h == a, h < a,
         h + a > 1.5, h + a < 1.5, h + a > 2.5, h + a < 2.5, h + a > 3.5, h + a < 3.5,
         h >= 1 && a >= 1, !(h >= 1 && a >= 1)]
end

"Simulate one match given true rates. Returns a SimMatch (books, posterior, settlement)."
function sim_match(cfg::SimConfig, λh::Float64, λa::Float64, rng::AbstractRNG; S::Int=cfg.S)
    h = rand(rng, Poisson(λh)); a = rand(rng, Poisson(λa))

    # market observer → fair probs → quoted odds: per-group π ∝ q^ρ_flb scaled to O_fam
    gm = dp_grid(λh * exp(cfg.σ_mkt * randn(rng)), λa * exp(cfg.σ_mkt * randn(rng)))
    q = MMASK' * gm
    O = (cfg.O_1x2, cfg.O_ou, cfg.O_btts)
    d = Vector{Float64}(undef, 11)
    for grp in MKT_GROUPS
        Ofam = O[FAM_OF_SEL[grp[1]]]
        z = sum(q[m]^cfg.ρ_flb for m in grp)     # = 1 when ρ_flb = 1 (fair probs sum to 1)
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

    # model observer: point belief + posterior draws, bias tilt on every grid
    if isnan(cfg.σ_mod_lvl)
        λmh = λh * exp(cfg.σ_mod * randn(rng))
        λma = λa * exp(cfg.σ_mod * randn(rng))
    else
        εl = cfg.σ_mod_lvl * randn(rng)
        εs = cfg.σ_mod_sup * randn(rng)
        λmh = λh * exp(εl + εs)
        λma = λa * exp(εl - εs)
    end
    corr = exp(-cfg.σ_post^2 / 2)                      # mean-corrected lognormal draws
    P = Matrix{Float64}(undef, GG * GG, S)
    for s in 1:S
        g = dp_grid(λmh * corr * exp(cfg.σ_post * randn(rng)),
                    λma * corr * exp(cfg.σ_post * randn(rng)))
        P[:, s] = bias_tilt!(g, cfg.γ_tot, cfg.γ_btts)
    end
    pbar = vec(mean(P, dims=2))

    R = d' .* MMASK .- 1.0
    gt = dp_grid(λh, λa)
    return SimMatch(d, q, P, pbar, settle_score(h, a), R, MMASK' * gt, (h, a))
end

# ---------- a campaign: drifting strengths + a stream of matches ----------

"""
Simulate `n_matches` matches with drifting strengths. Returns Vector{SimMatch}.
Truth drift mirrors SyntheticData: per-round N(0,σ_in) increments on α,β (zero-sum recentred)
and N(0,σ_ha) on ha. `S` overrides posterior draw count (pre-history uses fewer).
"""
function simulate_campaign(cfg::SimConfig, rng::AbstractRNG; n_matches::Int=cfg.n_matches,
                           S::Int=cfg.S)
    nt = cfg.n_teams
    α = cfg.σ0 .* randn(rng, nt); α .-= mean(α)
    β = cfg.σ0 .* randn(rng, nt); β .-= mean(β)
    ha = cfg.ha
    rounds = round_robin_rounds(nt)
    out = Vector{SimMatch}(undef, n_matches)
    k, leg = 0, 0
    while k < n_matches
        for rnd in rounds
            # drift per round
            α .+= cfg.σ_in .* randn(rng, nt); α .-= mean(α)
            β .+= cfg.σ_in .* randn(rng, nt); β .-= mean(β)
            ha += cfg.σ_ha * randn(rng)
            for (hm, aw) in rnd
                k == n_matches && break
                home, away = isodd(leg) ? (aw, hm) : (hm, aw)   # alternate venues per leg
                λh = exp(cfg.μ + α[home] + β[away] + ha)
                λa = exp(cfg.μ + α[away] + β[home])
                k += 1
                out[k] = sim_match(cfg, λh, λa, rng; S=S)
            end
            k == n_matches && break
        end
        leg += 1
    end
    return out
end
