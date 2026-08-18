# current_development/scottish_proxy_xg/l02_pxg_engines.jl
#
# LOADER 2/2 — two player-level engines built on BBC commentary PROXY xG, for ScottishLower (56/57).
#
#   ARM A  TeamPxGGoalsAPMModel        xG REPLACES shots      (the Ireland structure)
#   ARM B  TeamFunnelPxGGoalsAPMModel  xG JOINS shots         (3-layer volume -> quality -> goals)
#
# Both carry the same linear predictor as the incumbent
# src/.../player_level/time_decay/goals_funnel_plus_minus_league.jl — hierarchical monthly
# interception, hierarchical team HA, time-decay dynamics, zero-sum league offsets, and the RAPM
# player pillar w_att*R_h - w_def*R_a fed by Features.XGPlusMinusFeature.
#
# ------------------------------------------------------------------------------------------------
# ARM A
#     log mu = mu_base + d_month + d_league + ha + alpha_h + beta_a + (w_att*R_h - w_def*R_a)
#     xG  ~ Gamma(nu, mu/nu)      [masked]        mean mu, CV = 1/sqrt(nu)
#     G   ~ Poisson(kappa * mu)
#
# Julia's Gamma is shape-SCALE, so Gamma(nu, mu/nu) has mean nu*(mu/nu) = mu and variance mu^2/nu.
# `nu` is therefore a PRECISION (inverse squared CV), not a rate — do NOT write Gamma(nu, nu/mu).
#
# NO `shot_scale` OFFSET. The funnel needs one because it models ~10 shots against a UniformInit
# (-2, 2) that lives in VALUE space (bbc_xg_proxy/NOTES.md:174-204). Mean proxy xG here is ~1.22,
# so log(1.22) ~ 0.2 and the sampler already starts on scale. This is the one funnel gotcha that
# does not apply.
#
# `kappa` is the finishing multiplier: the wedge between expected and actual goals. It is centred
# on 1 (log_kappa ~ N(0, 0.2)) because the cell table is a CONVERSION-RATE table, so sum(proxy xG)
# ~ sum(goals) by construction. r00 gate 3 measures the deviation. It is GLOBAL, not per team:
# per-team conversion was the r04 null (sigma pulled to 1/4-1/7 of its prior, +/-4% relative).
#
# ------------------------------------------------------------------------------------------------
# ARM B
#     log lambda_s = shot_scale + (same linear predictor)              VOLUME
#     logit q      = q_raw + a_i - d_j                                 QUALITY (xG per shot)
#          mu      = lambda_s * q
#     S      ~ Poisson(lambda_s)                    [ds.bbc, all 6 seasons]
#     xG | S ~ Gamma(nu_q * S, q / nu_q)            [commentary, 23/24+]
#     G      ~ Poisson(kappa * lambda_s * q)
#
# ⚠ THE DESIGN POINT — WHY THE xG PILLAR IS CONDITIONAL ON S.
# Proxy xG is literally sum_{i=1..S} q_i over the SAME shots the volume pillar counts. A marginal
# Gamma on xG would therefore count the volume information twice, over-sharpening the posterior
# and over-weighting volume — the very axis the funnel already owns. Conditioning on the observed
# shot count strips the volume out and leaves the pillar carrying ONLY quality, which is the new
# information. Mean = nu_q*S * q/nu_q = S*q and variance = S*q^2/nu_q, i.e. CV = 1/sqrt(nu_q*S),
# shrinking with S exactly as a sum of S i.i.d. contributions should.
#
# Goals stay MARGINAL, Poisson(kappa*lambda_s*q). r06 proved this matters: routing goals through a
# conditional (cascade_weight = 1) makes them independent of lambda_s given the intermediate count
# and severs the goals -> team-strength gradient, which is what lost totals at r03. Only the xG
# pillar is conditional here; the pricing path is untouched.
#
# ⚠ TWO SHOT SERIES, ON PURPOSE. The volume pillar reads ds.bbc (BBC match pages, ~9.89/side, all
# six seasons); the conditioning reads the commentary EVENT count (~9.14/side, 23/24+). They must
# not be swapped: the conditioning count has to be the one the xG was actually summed over. The
# ~8% level gap is absorbed by the global kappa — valid only if the gap is not systematic by team,
# which is r00 gate 2.
#
# ⚠ EXPECT sigma_q TO BE SMALL. It is the same shape as the r04 hierarchical-conversion null.
# Measured cross-team shot-mix spread implies sigma_q ~ 0.05 on the logit. Treat the sigma_q
# POSTERIOR-VS-PRIOR ratio as a first-class result: it answers "is there team-level shot quality?"
# whether or not the engine wins anything.
#
# ------------------------------------------------------------------------------------------------
# DISPATCH ([[dixoncoles-prediction-dispatch-union]]): both structs subtype
# AbstractTimeDecayPlayerModel <: AbstractPlayerModel <: AbstractNegBinModel and return no `r`
# column, so loader-local Pred.extract_params / Pred.compute_score_matrix overrides ship at the
# bottom (plain Poisson grid). At graduation they must instead be added to the Union AND the import
# list in src/predictions/score_computation/poisson.jl.
#
# ⚠ Any runner that EVALUATES these experiments must include this file too, not just the runner
# that trained them — otherwise evaluate_experiments silently NaNs every row.

using Turing
using Distributions
using DataFrames
using Dates
using Statistics
using LogExpFunctions: log1pexp
using SpecialFunctions: loggamma
using StatsFuns: logit

include(joinpath(@__DIR__, "l01_proxy_xg_feature.jl"))

# ==========================================
# 0. SHARED PRIORS AND HELPERS
# ==========================================
# nu: WP1-E4 sets this from the data. The zone mix gives E[q] ~ 0.133, E[q^2] ~ 0.038 over ~9.14
# shots => CV ~ 0.49 => nu ~ 4.2. Ireland's truncated(Normal(3.0, 0.5)) is both mis-centred and too
# tight for a compound sum, so it is widened here.
const PXG_NU_PRIOR    = truncated(Normal(4.0, 1.5), lower = 0.5)
const PXG_LOGK_PRIOR  = Normal(0.0, 0.2)                      # kappa centred on 1
const PXG_Q_PRIOR     = Normal(logit(0.133), 0.5)             # xG per shot, logit scale
const PXG_SIGQ_PRIOR  = truncated(Normal(0.0, 0.15), lower = 0.0)
# theta for the LINEAR-VARIANCE form (cell 5): Gamma(mu/theta, theta) has mean mu and variance
# mu*theta. The compound-Poisson value is theta = E[q^2]/E[q] ~ 0.038/0.133 ~ 0.29, which implies a
# shape mu/theta ~ 1.22/0.29 ~ 4.2 — the SAME number PXG_NU_PRIOR is centred on, as it must be.
const PXG_THETA_PRIOR = truncated(Normal(0.29, 0.15), lower = 0.01)

# Mirrors PreGame._pm_outfield (goals_plus_minus_league.jl) and the extract_parameters aggregation
# in goals_funnel_plus_minus_league.jl:252-255. RAPM is zero-centred so `base` is 0.0 and the
# centring is a no-op, but it is kept explicit so a non-RAPM rating family drops in unchanged.
_pxg_outfield(D, M, F, base) = (D .+ M .+ F) .- 10.0 * base

_pxg_active(b::Bool) = b ? 1.0 : 0.0

function _pxg_core(data, config)
    date_deltas = Vector{Int}(data[:dates])
    return (;
        home_ids   = Vector{Int}(data[:flat_home_ids]),
        away_ids   = Vector{Int}(data[:flat_away_ids]),
        season_idx = Vector{Int}(data[:season_indices]),
        month_idx  = Vector{Int}(data[:flat_months]),
        league_idx = Vector{Int}(data[:flat_league_ids]),
        home_goals = Vector{Int}(data[:flat_home_goals]),
        away_goals = Vector{Int}(data[:flat_away_goals]),
        w          = 0.5 .^ (date_deltas ./ config.dynamics_config.days_half_life),
        n_teams    = Int(data[:n_teams]),
        n_seasons  = Int(data[:n_seasons]),
        n_months   = 12,
        n_leagues  = Int(data[:n_leagues]),
    )
end

"""RAPM outfield ratings for both sides, or zero vectors when the APM pillar is switched off."""
function _pxg_ratings(data, config, n::Int)
    config.apm_on || return (zeros(Float64, n), zeros(Float64, n))
    base = Features.rating_base(config.player_ratings_feature)
    h = _pxg_outfield(Vector{Float64}(data[:flat_home_D_rating]),
                      Vector{Float64}(data[:flat_home_M_rating]),
                      Vector{Float64}(data[:flat_home_F_rating]), base)
    a = _pxg_outfield(Vector{Float64}(data[:flat_away_D_rating]),
                      Vector{Float64}(data[:flat_away_M_rating]),
                      Vector{Float64}(data[:flat_away_F_rating]), base)
    return (h, a)
end

function _pxg_league_offsets(chain, n_leagues::Int, name::String)
    n_samples = size(chain, 1) * size(chain, 3)
    raw = zeros(n_samples, n_leagues)
    for i in 1:n_leagues
        raw[:, i] = vec(Array(chain[Symbol("$(name)[$i]")]))
    end
    return raw .- mean(raw, dims = 2)
end

# ==========================================
# 1. ARM A — TeamPxGGoalsAPMModel
# ==========================================
Base.@kwdef struct TeamPxGGoalsAPMModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    P<:Features.AbstractPlusMinusFeature
    } <: PreGame.AbstractTimeDecayPlayerModel
      interception_config::I    = PreGame.HierarchicalMonthlyInterception(prior_μ_base = Normal(0.0, 0.3))
      dynamics_config::T        = PreGame.TimeDecayDynamics()
      homeadvantage_config::H   = PreGame.HierarchicalTeamHomeAdvantage()
      player_ratings_feature::P = Features.XGPlusMinusFeature()
      proxy_feature::ProxyXGFeature = ProxyXGFeature()
      ν_prior::Distribution     = PXG_NU_PRIOR
      θ_prior::Distribution     = PXG_THETA_PRIOR
      log_κ_prior::Distribution = PXG_LOGK_PRIOR
      w_att_prior::Distribution = Normal(0.0, 0.3)
      w_def_prior::Distribution = Normal(0.0, 0.3)
      apm_on::Bool              = true      # false => the pxg_noapm isolation cell
      # :quadratic -> Gamma(nu, mu/nu),   Var = mu^2/nu   (the Ireland form)
      # :linear    -> Gamma(mu/theta, theta), Var = mu*theta (compound-Poisson-matched; cell 5)
      variance_law::Symbol      = :quadratic
      league_offset_sd::Float64 = 0.1
      league_ha_sd::Float64     = 0.1
      league_ha_on::Bool        = false
end

"""
Per-side sufficient statistics for Arm A.

    Gamma(nu, mu/nu):  logpdf = (nu-1)log x - nu*x/mu - nu*log mu + nu*log nu - loggamma(nu)
    Poisson(kappa*mu): loglik = g*log kappa + g*log mu - kappa*mu            [+ const]

so the weighted log-likelihood collapses onto four vectors and three scalars. The dropped Poisson
constant (log g!) is parameter-free — the posterior is EXACTLY unchanged, only the reported `lp`
shifts, so never compare `lp` across engines.

The xG terms carry the coverage mask; the goals terms carry NO mask, so every match informs the
rate exactly as in goals_funnel_league.jl. `log.(xg)` is safe because the extractor guarantees
xg >= 1e-3 even on masked slots (a 1.0 dummy).
"""
function _pxg_suff(xg::Vector{Float64}, mask::Vector{Float64}, goals::Vector{Int},
                   w::Vector{Float64})
    wm      = w .* mask
    c_x     = wm .* xg
    c_mlogx = wm .* log.(xg)
    c_g_lin = w .* goals
    return (
        # --- shared ---
        c_x      = c_x,                  # QUADRATIC: multiplies 1/mu
        c_m      = wm,                   # QUADRATIC: multiplies log mu
        S_m      = sum(wm),
        S_logx   = sum(c_mlogx),
        # --- LINEAR-variance form only (alpha = mu/theta is per match, so these must be vectors) ---
        c_mlogx  = c_mlogx,              # LINEAR: multiplies alpha
        S_x      = sum(c_x),             # LINEAR: multiplies 1/theta (free of mu)
        # --- goals ---
        c_g_lin  = c_g_lin,              # multiplies log mu
        c_g_rate = w,                    # multiplies mu
        S_g      = sum(c_g_lin),
    )
end

@model function build_pxg_goals_apm_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    rat_h::Vector{Float64}, rat_a::Vector{Float64},
    sx_h::NamedTuple, sx_a::NamedTuple,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64, apm_active::Float64,
    config
)
    # --- 1. COMPONENTS ---
    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(PreGame.build_dynamics(config.dynamics_config, n_teams))

    # Sampled even when apm_active = 0.0 so the model structure is fixed. With the pillar gated off
    # they simply draw their priors (which NUTS mixes perfectly) — IGNORE them in that cell.
    w_att ~ config.w_att_prior
    w_def ~ config.w_def_prior

    δ_league_raw ~ filldist(Normal(0.0, config.league_offset_sd), n_leagues)
    γ_league_raw ~ filldist(Normal(0.0, config.league_ha_sd), n_leagues)
    δ_league = δ_league_raw .- mean(δ_league_raw)
    γ_league = league_ha_active .* (γ_league_raw .- mean(γ_league_raw))

    # The variance law is fixed when the model instance is CONSTRUCTED, so this branch resolves
    # once and is baked into the compiled tape. It is not data-dependent control flow (the thing
    # the AD guide forbids) and the parameter set is one scalar either way.
    if config.variance_law === :linear
        θ_xg ~ config.θ_prior
    else
        ν_xg ~ config.ν_prior
    end
    log_κ ~ config.log_κ_prior

    # --- 2. THE LATENT xG RATE (no shot_scale offset — see the header) ---
    int_m = view(inter.μ_base, season_idx) .+ view(inter.δ_month, month_idx)
    lg    = view(δ_league, league_idx)
    γ_lg  = view(γ_league, league_idx)

    pillar_h = apm_active .* (w_att .* rat_h .- w_def .* rat_a)
    pillar_a = apm_active .* (w_att .* rat_a .- w_def .* rat_h)

    log_μ_h = clamp.(int_m .+ lg .+ view(ha, home_ids) .+ γ_lg .+
                     view(dyn.α, home_ids) .+ view(dyn.β, away_ids) .+ pillar_h, -10.0, 10.0)
    log_μ_a = clamp.(int_m .+ lg .+
                     view(dyn.α, away_ids) .+ view(dyn.β, home_ids) .+ pillar_a, -10.0, 10.0)

    # AD-safe rejection. `clamp` bounds Inf but PROPAGATES NaN, so sanitise the LOG-rate and derive
    # mu and 1/mu from it — rate, log-rate and inverse-rate must stay exactly consistent.
    bad_h  = isnan.(log_μ_h)
    bad_a  = isnan.(log_μ_a)
    is_bad = any(bad_h) || any(bad_a) || isnan(log_κ)
    log_μ_h = ifelse.(bad_h, zero.(log_μ_h), log_μ_h)
    log_μ_a = ifelse.(bad_a, zero.(log_μ_a), log_μ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    μ_h     = exp.(log_μ_h);   μ_a     = exp.(log_μ_a)
    inv_μ_h = exp.(.-log_μ_h); inv_μ_a = exp.(.-log_μ_a)
    κ       = exp(log_κ)

    # --- 3. PILLAR A: proxy xG (Gamma), via sufficient statistics ---
    if config.variance_law === :linear
        # Gamma(alpha, theta) with alpha = mu/theta: mean mu, VARIANCE mu*theta (LINEAR).
        #   logpdf = (alpha - 1)log x - x/theta - alpha*log theta - loggamma(alpha)
        # alpha is per match, so the log-x and loggamma terms stay vectors — this form is strictly
        # more expensive than the quadratic one and is only worth it if E4 says the law is linear.
        α_h = μ_h ./ θ_xg
        α_a = μ_a ./ θ_xg
        lθ  = log(θ_xg)
        ll_xg_h = sum(sx_h.c_mlogx .* α_h) - sx_h.S_logx - sx_h.S_x / θ_xg -
                  lθ * sum(sx_h.c_m .* α_h) - sum(sx_h.c_m .* loggamma.(α_h))
        ll_xg_a = sum(sx_a.c_mlogx .* α_a) - sx_a.S_logx - sx_a.S_x / θ_xg -
                  lθ * sum(sx_a.c_m .* α_a) - sum(sx_a.c_m .* loggamma.(α_a))
    else
        # Gamma(nu, mu/nu): mean mu, VARIANCE mu^2/nu (QUADRATIC — the Ireland form).
        #   logpdf = (nu-1)log x - nu*x/mu - nu*log mu + nu*log nu - loggamma(nu)
        cν = ν_xg * log(ν_xg) - loggamma(ν_xg)      # per-observation constant block
        ll_xg_h = (ν_xg - 1.0) * sx_h.S_logx - ν_xg * sum(sx_h.c_x .* inv_μ_h) -
                  ν_xg * sum(sx_h.c_m .* log_μ_h) + cν * sx_h.S_m
        ll_xg_a = (ν_xg - 1.0) * sx_a.S_logx - ν_xg * sum(sx_a.c_x .* inv_μ_a) -
                  ν_xg * sum(sx_a.c_m .* log_μ_a) + cν * sx_a.S_m
    end

    # --- 4. PILLAR B: goals (Poisson, kappa * mu) ---
    ll_g_h = sx_h.S_g * log_κ + sum(sx_h.c_g_lin .* log_μ_h) - κ * sum(sx_h.c_g_rate .* μ_h)
    ll_g_a = sx_a.S_g * log_κ + sum(sx_a.c_g_lin .* log_μ_a) - κ * sum(sx_a.c_g_rate .* μ_a)

    # One NaN guard for the whole block: a NaN shape scalar reaches loggamma rather than log_mu, so
    # the rejection above cannot see it. `-Inf + NaN == NaN`, hence ifelse rather than an add.
    total = ll_xg_h + ll_xg_a + ll_g_h + ll_g_a
    Turing.@addlogprob! ifelse(isnan(total), -Inf, total)
end

function Features.required_features(model::TeamPxGGoalsAPMModel)
    fs = Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), Features.LeagueFeature(), model.proxy_feature,
        Features.TimeIndicesFeature(),
    ]
    # Dropped entirely when the pillar is off — that also skips the RAPM ridge fit per fold.
    model.apm_on && push!(fs, model.player_ratings_feature)
    return fs
end

function PreGame.build_turing_model(config::TeamPxGGoalsAPMModel, feature_set)
    data = feature_set.data
    d    = _pxg_core(data, config)
    n    = length(d.home_ids)
    rat_h, rat_a = _pxg_ratings(data, config, n)

    xg_h   = Vector{Float64}(data[:flat_home_xg_proxy])
    xg_a   = Vector{Float64}(data[:flat_away_xg_proxy])
    mask_h = Vector{Float64}(data[:flat_pxg_mask_h])
    mask_a = Vector{Float64}(data[:flat_pxg_mask_a])

    return build_pxg_goals_apm_engine(
        d.home_ids, d.away_ids, d.season_idx, d.month_idx, d.league_idx,
        rat_h, rat_a,
        _pxg_suff(xg_h, mask_h, d.home_goals, d.w),
        _pxg_suff(xg_a, mask_a, d.away_goals, d.w),
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        _pxg_active(config.league_ha_on), _pxg_active(config.apm_on),
        config)
end

# ==========================================
# 2. ARM B — TeamFunnelPxGGoalsAPMModel
# ==========================================
Base.@kwdef struct TeamFunnelPxGGoalsAPMModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    P<:Features.AbstractPlusMinusFeature
    } <: PreGame.AbstractTimeDecayPlayerModel
      shot_scale::Float64       = log(10.0)   # needed HERE (volume ~10), unlike Arm A
      interception_config::I    = PreGame.HierarchicalMonthlyInterception(prior_μ_base = Normal(0.0, 0.3))
      dynamics_config::T        = PreGame.TimeDecayDynamics()
      homeadvantage_config::H   = PreGame.HierarchicalTeamHomeAdvantage()
      player_ratings_feature::P = Features.XGPlusMinusFeature()
      proxy_feature::ProxyXGFeature = ProxyXGFeature()
      ν_prior::Distribution     = PXG_NU_PRIOR      # per-shot precision (nu_q)
      log_κ_prior::Distribution = PXG_LOGK_PRIOR
      q_prior::Distribution     = PXG_Q_PRIOR
      σ_q_prior::Distribution   = PXG_SIGQ_PRIOR
      w_att_prior::Distribution = Normal(0.0, 0.3)
      w_def_prior::Distribution = Normal(0.0, 0.3)
      apm_on::Bool              = true
      team_quality_on::Bool     = true      # false => global q only (cheap sanity cell)
      league_offset_sd::Float64 = 0.1
      league_ha_sd::Float64     = 0.1
      league_ha_on::Bool        = false
end

"""
Per-side sufficient statistics for Arm B.

    Poisson(lambda_s):        loglik  = S*log lambda_s - lambda_s                         [+ const]
    Gamma(nu_q*S, q/nu_q):    logpdf  = (nu_q*S - 1)log x - nu_q*x/q - nu_q*S*log q
                                        + nu_q*S*log nu_q - loggamma(nu_q*S)
    Poisson(kappa*lambda_s*q):loglik  = g*log kappa + g*(log lambda_s + log q)
                                        - kappa*lambda_s*q                                [+ const]

`loggamma(nu_q * S)` cannot be collapsed (it mixes a sampled scalar with per-match data), so it
stays a broadcast — one loggamma per side per match, which ReverseDiff handles fine.

⚠ `n_ev` is forced to 1.0 wherever the mask is 0. loggamma(0) = Inf and `Inf * 0.0 == NaN` would
poison the gradient through a term that is supposed to contribute nothing — the exact failure mode
the AD guide warns about.
"""
function _pxg_funnel_suff(shots_bbc::Vector{Int}, mask_s::Vector{Float64},
                          xg::Vector{Float64}, n_ev::Vector{Int}, mask_x::Vector{Float64},
                          goals::Vector{Int}, w::Vector{Float64})
    ws      = w .* mask_s
    wx      = w .* mask_x
    logx    = log.(xg)
    n_safe  = [mask_x[i] > 0 && n_ev[i] > 0 ? Float64(n_ev[i]) : 1.0 for i in eachindex(n_ev)]
    cq_S    = wx .* n_safe
    c_g_lin = w .* goals
    return (
        # volume
        c_s_lin  = ws .* shots_bbc,
        c_s_rate = ws,
        # quality (conditional Gamma)
        cq_m     = wx,                       # multiplies loggamma(nu_q * S)
        cq_S     = cq_S,                     # multiplies log q
        cq_x     = wx .* xg,                 # multiplies 1/q
        n_ev     = n_safe,                   # DATA inside loggamma(nu_q * .)
        S_Slogx  = sum(cq_S .* logx),        # multiplies nu_q
        S_logx   = sum(wx .* logx),          # multiplies -1
        S_cq_S   = sum(cq_S),                # multiplies nu_q*log nu_q
        # goals
        c_g_lin  = c_g_lin,
        c_g_rate = w,
        S_g      = sum(c_g_lin),
    )
end

@model function build_funnel_pxg_goals_apm_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    rat_h::Vector{Float64}, rat_a::Vector{Float64},
    sf_h::NamedTuple, sf_a::NamedTuple,
    shot_scale::Float64,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64, apm_active::Float64, quality_active::Float64,
    config
)
    # --- 1. COMPONENTS ---
    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(PreGame.build_dynamics(config.dynamics_config, n_teams))

    w_att ~ config.w_att_prior
    w_def ~ config.w_def_prior

    δ_league_raw ~ filldist(Normal(0.0, config.league_offset_sd), n_leagues)
    γ_league_raw ~ filldist(Normal(0.0, config.league_ha_sd), n_leagues)
    δ_league = δ_league_raw .- mean(δ_league_raw)
    γ_league = league_ha_active .* (γ_league_raw .- mean(γ_league_raw))

    ν_q   ~ config.ν_prior
    log_κ ~ config.log_κ_prior
    q_raw ~ config.q_prior
    σ_q   ~ config.σ_q_prior

    # Non-centred + zero-sum team quality. Gated branch-free; when off, raw_aq/raw_dq/sigma_q draw
    # their priors and must be ignored.
    raw_aq ~ filldist(Normal(0.0, 1.0), n_teams)
    raw_dq ~ filldist(Normal(0.0, 1.0), n_teams)
    aq = quality_active .* (raw_aq .* σ_q); aq = aq .- mean(aq)
    dq = quality_active .* (raw_dq .* σ_q); dq = dq .- mean(dq)

    # --- 2. VOLUME (shot_scale offset; HA on the home side only, per the 2026-07-17 EDA) ---
    int_m = shot_scale .+ view(inter.μ_base, season_idx) .+ view(inter.δ_month, month_idx)
    lg    = view(δ_league, league_idx)
    γ_lg  = view(γ_league, league_idx)

    pillar_h = apm_active .* (w_att .* rat_h .- w_def .* rat_a)
    pillar_a = apm_active .* (w_att .* rat_a .- w_def .* rat_h)

    log_λ_h = clamp.(int_m .+ lg .+ view(ha, home_ids) .+ γ_lg .+
                     view(dyn.α, home_ids) .+ view(dyn.β, away_ids) .+ pillar_h, -10.0, 10.0)
    log_λ_a = clamp.(int_m .+ lg .+
                     view(dyn.α, away_ids) .+ view(dyn.β, home_ids) .+ pillar_a, -10.0, 10.0)

    # --- 3. QUALITY (xG per shot; logit scale, stable log-logistic) ---
    qlin_h  = q_raw .+ view(aq, home_ids) .- view(dq, away_ids)
    qlin_a  = q_raw .+ view(aq, away_ids) .- view(dq, home_ids)
    log_q_h = .-log1pexp.(.-qlin_h)
    log_q_a = .-log1pexp.(.-qlin_a)

    # AD-safe rejection on the log-rates (see Arm A).
    bad_h  = isnan.(log_λ_h) .| isnan.(log_q_h)
    bad_a  = isnan.(log_λ_a) .| isnan.(log_q_a)
    is_bad = any(bad_h) || any(bad_a) || isnan(ν_q) || isnan(log_κ)
    log_λ_h = ifelse.(bad_h, zero.(log_λ_h), log_λ_h)
    log_λ_a = ifelse.(bad_a, zero.(log_λ_a), log_λ_a)
    log_q_h = ifelse.(bad_h, zero.(log_q_h), log_q_h)
    log_q_a = ifelse.(bad_a, zero.(log_q_a), log_q_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    λ_h     = exp.(log_λ_h);   λ_a     = exp.(log_λ_a)
    q_h     = exp.(log_q_h);   q_a     = exp.(log_q_a)
    inv_q_h = exp.(.-log_q_h); inv_q_a = exp.(.-log_q_a)

    κ   = exp(log_κ)
    lνq = log(ν_q)

    # --- 4a. VOLUME likelihood ---
    ll_s_h = sum(sf_h.c_s_lin .* log_λ_h) - sum(sf_h.c_s_rate .* λ_h)
    ll_s_a = sum(sf_a.c_s_lin .* log_λ_a) - sum(sf_a.c_s_rate .* λ_a)

    # --- 4b. QUALITY likelihood, CONDITIONED on the observed event shot count ---
    ll_q_h = ν_q * sf_h.S_Slogx - sf_h.S_logx -
             ν_q * sum(sf_h.cq_x .* inv_q_h) -
             ν_q * sum(sf_h.cq_S .* log_q_h) +
             ν_q * lνq * sf_h.S_cq_S -
             sum(sf_h.cq_m .* loggamma.(ν_q .* sf_h.n_ev))
    ll_q_a = ν_q * sf_a.S_Slogx - sf_a.S_logx -
             ν_q * sum(sf_a.cq_x .* inv_q_a) -
             ν_q * sum(sf_a.cq_S .* log_q_a) +
             ν_q * lνq * sf_a.S_cq_S -
             sum(sf_a.cq_m .* loggamma.(ν_q .* sf_a.n_ev))

    # --- 4c. GOALS, MARGINAL (keeps the goals -> team-strength gradient; see the header) ---
    ll_g_h = sf_h.S_g * log_κ + sum(sf_h.c_g_lin .* (log_λ_h .+ log_q_h)) -
             κ * sum(sf_h.c_g_rate .* λ_h .* q_h)
    ll_g_a = sf_a.S_g * log_κ + sum(sf_a.c_g_lin .* (log_λ_a .+ log_q_a)) -
             κ * sum(sf_a.c_g_rate .* λ_a .* q_a)

    Turing.@addlogprob! ll_s_h + ll_s_a + ll_q_h + ll_q_a + ll_g_h + ll_g_a
end

function Features.required_features(model::TeamFunnelPxGGoalsAPMModel)
    fs = Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), Features.LeagueFeature(),
        Features.ShotsFunnelFeature(), model.proxy_feature,
        Features.TimeIndicesFeature(),
    ]
    model.apm_on && push!(fs, model.player_ratings_feature)
    return fs
end

function PreGame.build_turing_model(config::TeamFunnelPxGGoalsAPMModel, feature_set)
    data = feature_set.data
    d    = _pxg_core(data, config)
    n    = length(d.home_ids)
    rat_h, rat_a = _pxg_ratings(data, config, n)

    return build_funnel_pxg_goals_apm_engine(
        d.home_ids, d.away_ids, d.season_idx, d.month_idx, d.league_idx,
        rat_h, rat_a,
        _pxg_funnel_suff(Vector{Int}(data[:flat_home_shots_n]),
                         Vector{Float64}(data[:flat_funnel_mask_h]),
                         Vector{Float64}(data[:flat_home_xg_proxy]),
                         Vector{Int}(data[:flat_home_pxg_shots]),
                         Vector{Float64}(data[:flat_pxg_mask_h]),
                         d.home_goals, d.w),
        _pxg_funnel_suff(Vector{Int}(data[:flat_away_shots_n]),
                         Vector{Float64}(data[:flat_funnel_mask_a]),
                         Vector{Float64}(data[:flat_away_xg_proxy]),
                         Vector{Int}(data[:flat_away_pxg_shots]),
                         Vector{Float64}(data[:flat_pxg_mask_a]),
                         d.away_goals, d.w),
        config.shot_scale,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        _pxg_active(config.league_ha_on), _pxg_active(config.apm_on),
        _pxg_active(config.team_quality_on),
        config)
end

# ==========================================
# 3. EXTRACTORS
# ==========================================
"""Everything both arms share: the per-draw linear predictor, per match."""
function _pxg_extract_core(model, df, feature_set, chain)
    data      = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    n_leagues = Int(data[:n_leagues])
    team_map  = data[:team_map]
    league_lookup = data[:league_lookup]
    ratings_map   = get(data, :player_ratings_map, Dict{Int, Dict{Tuple{String, String}, Float64}}())

    inter_nt = PreGame.extract_interception(chain, model.interception_config, n_seasons)
    ha_mat   = PreGame.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    dyn_nt   = PreGame.extract_dynamics(chain, model.dynamics_config, "dyn", n_teams)
    δ_mat    = _pxg_league_offsets(chain, n_leagues, "δ_league_raw")

    n_samples = size(chain, 1) * size(chain, 3)
    γ_mat = model.league_ha_on ? _pxg_league_offsets(chain, n_leagues, "γ_league_raw") :
                                 zeros(n_samples, n_leagues)

    w_att = vec(Array(chain[:w_att]))
    w_def = vec(Array(chain[:w_def]))
    apm_a = _pxg_active(model.apm_on)
    base  = Features.rating_base(model.player_ratings_feature)

    out = Dict{Int, NamedTuple}()
    for row in eachrow(df)
        mid   = Int(row.match_id)
        h_idx = get(team_map, row.home_team, -1)
        a_idx = get(team_map, row.away_team, -1)
        l_idx = get(league_lookup, mid, 0)

        α_h = h_idx > 0 ? dyn_nt.α[:, h_idx] : zeros(n_samples)
        β_h = h_idx > 0 ? dyn_nt.β[:, h_idx] : zeros(n_samples)
        α_a = a_idx > 0 ? dyn_nt.α[:, a_idx] : zeros(n_samples)
        β_a = a_idx > 0 ? dyn_nt.β[:, a_idx] : zeros(n_samples)
        γ_h = h_idx > 0 ? ha_mat[:, h_idx] : zeros(n_samples)
        lg  = l_idx > 0 ? δ_mat[:, l_idx] : zeros(n_samples)
        γlg = l_idx > 0 ? γ_mat[:, l_idx] : zeros(n_samples)

        m_r = get(ratings_map, mid, Dict{Tuple{String, String}, Float64}())
        r_h = (get(m_r, ("home", "D"), 0.0) + get(m_r, ("home", "M"), 0.0) +
               get(m_r, ("home", "F"), 0.0)) - 10.0 * base
        r_a = (get(m_r, ("away", "D"), 0.0) + get(m_r, ("away", "M"), 0.0) +
               get(m_r, ("away", "F"), 0.0)) - 10.0 * base

        pillar_h = apm_a .* (w_att .* r_h .- w_def .* r_a)
        pillar_a = apm_a .* (w_att .* r_a .- w_def .* r_h)

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        m_idx = month(row.match_date)
        int_v = inter_nt.μ_base[:, s_idx] .+ inter_nt.δ_month[:, m_idx]

        out[mid] = (;
            h_idx, a_idx, n_samples,
            lin_h = clamp.(int_v .+ lg .+ γ_h .+ γlg .+ α_h .+ β_a .+ pillar_h, -10.0, 10.0),
            lin_a = clamp.(int_v .+ lg .+               α_a .+ β_h .+ pillar_a, -10.0, 10.0),
        )
    end
    return out, n_samples, n_teams
end

function PreGame.extract_parameters(model::TeamPxGGoalsAPMModel, df, feature_set, chain)
    core, _, _ = _pxg_extract_core(model, df, feature_set, chain)
    κ = exp.(vec(Array(chain[:log_κ])))
    # `xg_shape` is nu under the quadratic law and theta under the linear one — carried for
    # diagnostics only; nothing downstream prices off it.
    xg_shape = vec(Array(chain[model.variance_law === :linear ? :θ_xg : :ν_xg]))

    results = Dict{Int, NamedTuple}()
    for (mid, c) in core
        μ_h = exp.(c.lin_h); μ_a = exp.(c.lin_a)
        results[mid] = (; λ_h = κ .* μ_h, λ_a = κ .* μ_a,
                          true_xg_h = μ_h, true_xg_a = μ_a, κ = κ, xg_shape = xg_shape)
    end
    return results
end

function PreGame.extract_parameters(model::TeamFunnelPxGGoalsAPMModel, df, feature_set, chain)
    core, n_samples, n_teams = _pxg_extract_core(model, df, feature_set, chain)
    κ     = exp.(vec(Array(chain[:log_κ])))
    q_raw = vec(Array(chain[:q_raw]))
    σ_q   = vec(Array(chain[:σ_q]))
    qa    = _pxg_active(model.team_quality_on)

    # Reconstruct the zero-sum team quality effects EXACTLY as the @model does: scale by sigma,
    # apply the gate, THEN centre.
    aq = zeros(n_samples, n_teams); dq = zeros(n_samples, n_teams)
    for i in 1:n_teams
        aq[:, i] = qa .* (vec(Array(chain[Symbol("raw_aq[$i]")])) .* σ_q)
        dq[:, i] = qa .* (vec(Array(chain[Symbol("raw_dq[$i]")])) .* σ_q)
    end
    aq .-= mean(aq, dims = 2); dq .-= mean(dq, dims = 2)

    results = Dict{Int, NamedTuple}()
    for (mid, c) in core
        a_h = c.h_idx > 0 ? aq[:, c.h_idx] : zeros(n_samples)
        d_h = c.h_idx > 0 ? dq[:, c.h_idx] : zeros(n_samples)
        a_a = c.a_idx > 0 ? aq[:, c.a_idx] : zeros(n_samples)
        d_a = c.a_idx > 0 ? dq[:, c.a_idx] : zeros(n_samples)

        q_h = 1 ./ (1 .+ exp.(.-(q_raw .+ a_h .- d_a)))
        q_a = 1 ./ (1 .+ exp.(.-(q_raw .+ a_a .- d_h)))
        λ_s_h = exp.(model.shot_scale .+ c.lin_h)
        λ_s_a = exp.(model.shot_scale .+ c.lin_a)
        μ_h = λ_s_h .* q_h; μ_a = λ_s_a .* q_a

        results[mid] = (; λ_h = κ .* μ_h, λ_a = κ .* μ_a,
                          λ_s_h, λ_s_a, q_h, q_a,
                          true_xg_h = μ_h, true_xg_a = μ_a, κ = κ)
    end
    return results
end

# ==========================================
# 4. PREDICTION OVERRIDES (loader-local Poisson grid)
# ==========================================
const PxGEngines = Union{TeamPxGGoalsAPMModel, TeamFunnelPxGGoalsAPMModel}

function _pxg_poisson_score(λ_h, λ_a; max_goals::Int = 12)
    n = length(λ_h)
    S = zeros(Float64, max_goals, max_goals, n)
    p_h = zeros(Float64, max_goals); p_a = zeros(Float64, max_goals)
    goals = 0:(max_goals - 1)
    @inbounds for k in 1:n
        @. p_h = pdf(Poisson(λ_h[k]), goals)
        @. p_a = pdf(Poisson(λ_a[k]), goals)
        for j in 1:max_goals, i in 1:max_goals
            S[i, j, k] = p_h[i] * p_a[j]
        end
    end
    return Pred.ScoreMatrix(S)
end

Pred.extract_params(::PxGEngines, row) = (λ_h = row.λ_h, λ_a = row.λ_a)
Pred.compute_score_matrix(::PxGEngines, params; max_goals::Int = 12) =
    _pxg_poisson_score(params.λ_h, params.λ_a; max_goals)
