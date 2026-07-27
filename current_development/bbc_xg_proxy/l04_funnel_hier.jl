# current_development/bbc_xg_proxy/l04_funnel_hier.jl
#
# LOADER (temporary module). STAGE 2 of the funnel cascade: HIERARCHICAL CONVERSION.
#
#     Shots ~ Poisson(λ_s)                         λ_s from the l03 latent block (unchanged)
#     SoT   | Shots ~ Binomial(Shots, p₁_ij)       logit p₁_ij = p1_μ + a₁_i + d₁_j
#     Goals | SoT   ~ Binomial(SoT,   p₂_ij)       logit p₂_ij = p2_μ + a₂_i + d₂_j
#
# WHY: l03's GLOBAL p₁/p₂ force λ_goals = λ_s(α_i, β_j)·p₁·p₂, i.e. ALL team-strength variation
# in goals must come from shot VOLUME — shot counts are proportionally tighter than goal counts,
# so team strength gets compressed (r03's spread diagnostic measures exactly this). Per-team
# conversion is the term that restores the spread: a₁/d₁ = shooting quality vs opponent
# shot-blocking, a₂/d₂ = finishing vs goalkeeping.
#
# IDENTIFIABILITY: three separately OBSERVED layers pin the three parameter blocks — λ_s by
# shots, p₁ by SoT|shots, p₂ by goals|SoT. Power (56/57 pooled, ~20 teams, 2 seasons history):
# ≈350 shot trials/team-season for p₁ (well determined) and ≈150 SoT trials for p₂ (a genuine
# shrinkage estimate — hence the tight σ priors and non-centred zero-sum parameterisation,
# same idiom as build_dynamics in src/models/pregame/components/dynamics/team_level/time_decay.jl).
#
# NO home effect on conversion: EDA 2026-07-17 — p₁ 0.437 vs 0.443, p₂ 0.323 vs 0.324 (h/a).
# HA stays on log λ_s only, as in l03.
#
# Everything else is inherited from l03: BBCFunnelFeature, _unpack_funnel, _extract_funnel_core,
# the safe-dummy masking, and the Poisson-grid pricing (thinning still holds per match).

using Turing
using Distributions
using DataFrames
using Dates
using StatsFuns: logit
using LogExpFunctions: log1pexp

const ROOT_L04 = pkgdir(BayesianFootball)
include(joinpath(ROOT_L04, "current_development/bbc_xg_proxy/l03_funnel_cascade.jl"))

# ==========================================
# 1. ENGINE — TeamFunnelHierDPGoalsModel (funnel_hier_pois)
# ==========================================
Base.@kwdef struct TeamFunnelHierDPGoalsModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig
    } <: PreGame.AbstractTimeDecayTeamModel
      shot_scale::Float64     = log(10.0)          # see l03 — keeps μ_base O(0) for the init
      interception_config::I  = PreGame.HierarchicalMonthlyInterception(
                                    prior_μ_base = Normal(0.0, 0.3))
      dynamics_config::T      = PreGame.TimeDecayDynamics()
      homeadvantage_config::H = PreGame.HierarchicalTeamHomeAdvantage()
      p1_prior::Distribution  = Normal(logit(0.44), 0.5)
      p2_prior::Distribution  = Normal(logit(0.32), 0.5)
      # per-team conversion spread on the LOGIT scale. 0.3 ⇒ ±1sd spans ≈ p₁ 0.37–0.51,
      # p₂ 0.26–0.39 — wide enough to matter, tight enough to shrink a 150-trial estimate.
      σ_p1_prior::Distribution = truncated(Normal(0.0, 0.3), lower = 0.0)
      σ_p2_prior::Distribution = truncated(Normal(0.0, 0.3), lower = 0.0)
      p1_defence_on::Bool      = true    # opponent effect d₁ (shot blocking)
      p2_defence_on::Bool      = true    # opponent effect d₂ (goalkeeping)
      funnel_weight::Float64   = 1.0
      league_offset_sd::Float64 = 0.1
      league_ha_sd::Float64     = 0.1
      league_ha_on::Bool        = false
end

@model function build_team_funnel_hier_dp_goals_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    suff_h, suff_a,
    funnel_weight::Float64, shot_scale::Float64, d1_active::Float64, d2_active::Float64,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64,
    config
)
    # --- shot-rate block: IDENTICAL to l03 ---
    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(PreGame.build_dynamics(config.dynamics_config, n_teams))

    δ_league_raw ~ filldist(Normal(0.0, config.league_offset_sd), n_leagues)
    γ_league_raw ~ filldist(Normal(0.0, config.league_ha_sd), n_leagues)
    δ_league = δ_league_raw .- mean(δ_league_raw)
    γ_league = league_ha_active .* (γ_league_raw .- mean(γ_league_raw))

    int_m = shot_scale .+ view(inter.μ_base, season_idx) .+ view(inter.δ_month, month_idx)
    lg    = view(δ_league, league_idx)
    γ_lg  = view(γ_league, league_idx)

    log_λ_s_h = clamp.(int_m .+ lg .+ view(ha, home_ids) .+ γ_lg .+
                       view(dyn.α, home_ids) .+ view(dyn.β, away_ids), -10.0, 10.0)
    log_λ_s_a = clamp.(int_m .+ lg .+
                       view(dyn.α, away_ids) .+ view(dyn.β, home_ids), -10.0, 10.0)
    λ_s_h = exp.(log_λ_s_h)
    λ_s_a = exp.(log_λ_s_a)

    # --- hierarchical conversion (non-centred, zero-sum; defence gated branch-free) ---
    p1_μ ~ config.p1_prior
    p2_μ ~ config.p2_prior
    σ_p1 ~ config.σ_p1_prior
    σ_p2 ~ config.σ_p2_prior
    raw_a1 ~ filldist(Normal(0, 1), n_teams)
    raw_d1 ~ filldist(Normal(0, 1), n_teams)
    raw_a2 ~ filldist(Normal(0, 1), n_teams)
    raw_d2 ~ filldist(Normal(0, 1), n_teams)

    a1 = (raw_a1 .* σ_p1); a1 = a1 .- mean(a1)
    d1 = d1_active .* ((raw_d1 .* σ_p1) .- mean(raw_d1 .* σ_p1))
    a2 = (raw_a2 .* σ_p2); a2 = a2 .- mean(a2)
    d2 = d2_active .* ((raw_d2 .* σ_p2) .- mean(raw_d2 .* σ_p2))

    # per-match logits: attacker = the shooting side, defender = the opponent
    z1_h = clamp.(p1_μ .+ view(a1, home_ids) .+ view(d1, away_ids), -10.0, 10.0)
    z1_a = clamp.(p1_μ .+ view(a1, away_ids) .+ view(d1, home_ids), -10.0, 10.0)
    z2_h = clamp.(p2_μ .+ view(a2, home_ids) .+ view(d2, away_ids), -10.0, 10.0)
    z2_a = clamp.(p2_μ .+ view(a2, away_ids) .+ view(d2, home_ids), -10.0, 10.0)

    # stable log p / log(1-p) straight off the logit (no logistic round-trip)
    lp1_h, lq1_h = -log1pexp.(-z1_h), -log1pexp.(z1_h)
    lp1_a, lq1_a = -log1pexp.(-z1_a), -log1pexp.(z1_a)
    lp2_h, lq2_h = -log1pexp.(-z2_h), -log1pexp.(z2_h)
    lp2_a, lq2_a = -log1pexp.(-z2_a), -log1pexp.(z2_a)

    is_bad = any(isnan, λ_s_h) || any(isnan, λ_s_a) || any(isinf, λ_s_h) || any(isinf, λ_s_a) ||
             any(isnan, z1_h) || any(isnan, z1_a) || any(isnan, z2_h) || any(isnan, z2_a)
    λ_s_h = ifelse.(isnan.(λ_s_h) .| isinf.(λ_s_h), one.(λ_s_h), λ_s_h)
    λ_s_a = ifelse.(isnan.(λ_s_a) .| isinf.(λ_s_a), one.(λ_s_a), λ_s_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    # --- LIKELIHOOD via the SAME sufficient statistics as l03 (_unpack_funnel) ---
    # Per-team p only changes the contraction: where Stage 1 multiplies a scalar log-odds by
    # the summed weight S_*, Stage 2 dots the per-match log-odds against the weight VECTOR v_*.
    # Data-side constants (counts, masks, decay weights, log y!, binomial coefficients) are
    # still folded in by the builder — no logpdf kernels on the tape at all.
    ll_shots_h = sum(suff_h.c_shots_lin .* log_λ_s_h) - sum(suff_h.c_shots_rate .* λ_s_h)
    ll_shots_a = sum(suff_a.c_shots_lin .* log_λ_s_a) - sum(suff_a.c_shots_rate .* λ_s_a)

    ll_sot_h = sum(suff_h.v_sot .* lp1_h) + sum(suff_h.v_miss .* lq1_h)
    ll_sot_a = sum(suff_a.v_sot .* lp1_a) + sum(suff_a.v_miss .* lq1_a)

    ll_casc_h = sum(suff_h.v_goal .* lp2_h) + sum(suff_h.v_save .* lq2_h)
    ll_casc_a = sum(suff_a.v_goal .* lp2_a) + sum(suff_a.v_save .* lq2_a)

    # marginal Poisson(λ_s·p₁_ij·p₂_ij) route for own-goal violations / missing stats
    ll_marg_h = sum(suff_h.c_marg_lin .* (log_λ_s_h .+ lp1_h .+ lp2_h)) -
                sum(suff_h.c_marg_rate .* λ_s_h .* exp.(lp1_h .+ lp2_h))
    ll_marg_a = sum(suff_a.c_marg_lin .* (log_λ_s_a .+ lp1_a .+ lp2_a)) -
                sum(suff_a.c_marg_rate .* λ_s_a .* exp.(lp1_a .+ lp2_a))

    Turing.@addlogprob!(
        funnel_weight * (ll_shots_h + ll_shots_a + ll_sot_h + ll_sot_a) +
        ll_casc_h + ll_casc_a + ll_marg_h + ll_marg_a)
end

function Features.required_features(model::TeamFunnelHierDPGoalsModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), LeagueFeature(), Features.TimeIndicesFeature(),
        BBCFunnelFeature(),
    ]
end

function PreGame.build_turing_model(config::TeamFunnelHierDPGoalsModel, feature_set)
    d = _unpack_funnel(feature_set.data, config)     # from l03
    return build_team_funnel_hier_dp_goals_engine(
        d.home_ids, d.away_ids, d.season_idx, d.month_idx, d.league_idx,
        d.suff_h, d.suff_a,
        config.funnel_weight, config.shot_scale,
        config.p1_defence_on ? 1.0 : 0.0, config.p2_defence_on ? 1.0 : 0.0,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        config.league_ha_on ? 1.0 : 0.0, config)
end

# ==========================================
# 2. EXTRACTION — per-match conversion
# ==========================================
# [n_samples × n_teams], zero-sum centred per draw (reconstructs the @model math exactly).
function _extract_team_effect(chain, n_teams::Int, raw_stem::String, σ_sym::String, active::Bool)
    n_samples = size(chain, 1) * size(chain, 3)
    active || return zeros(n_samples, n_teams)
    σ = vec(Array(chain[Symbol(σ_sym)]))
    raw = zeros(n_samples, n_teams)
    for i in 1:n_teams
        raw[:, i] = vec(Array(chain[Symbol("$(raw_stem)[$i]")]))
    end
    scaled = raw .* σ
    return scaled .- mean(scaled, dims = 2)
end

_sigmoid(x) = 1 ./ (1 .+ exp.(-x))

function PreGame.extract_parameters(model::TeamFunnelHierDPGoalsModel, df, feature_set, chain)
    core, n_samples = _extract_funnel_core(model, df, feature_set, chain)   # from l03
    n_teams = Int(feature_set.data[:n_teams])

    p1_μ = vec(Array(chain[:p1_μ]));  p2_μ = vec(Array(chain[:p2_μ]))
    a1 = _extract_team_effect(chain, n_teams, "raw_a1", "σ_p1", true)
    d1 = _extract_team_effect(chain, n_teams, "raw_d1", "σ_p1", model.p1_defence_on)
    a2 = _extract_team_effect(chain, n_teams, "raw_a2", "σ_p2", true)
    d2 = _extract_team_effect(chain, n_teams, "raw_d2", "σ_p2", model.p2_defence_on)

    _col(M, idx) = idx > 0 ? M[:, idx] : zeros(n_samples)

    results = Dict{Int, NamedTuple}()
    for (mid, c) in core
        p1_h = _sigmoid(p1_μ .+ _col(a1, c.h_idx) .+ _col(d1, c.a_idx))
        p1_a = _sigmoid(p1_μ .+ _col(a1, c.a_idx) .+ _col(d1, c.h_idx))
        p2_h = _sigmoid(p2_μ .+ _col(a2, c.h_idx) .+ _col(d2, c.a_idx))
        p2_a = _sigmoid(p2_μ .+ _col(a2, c.a_idx) .+ _col(d2, c.h_idx))
        λ_h = c.λ_s_h .* p1_h .* p2_h
        λ_a = c.λ_s_a .* p1_a .* p2_a
        results[mid] = (; λ_h, λ_a, λ_s_h = c.λ_s_h, λ_s_a = c.λ_s_a,
                          p1_h, p1_a, p2_h, p2_a, true_xg_h = λ_h, true_xg_a = λ_a)
    end
    return results
end

# ==========================================
# 3. PREDICTION OVERRIDES (thinning still exact per match)
# ==========================================
Pred.extract_params(::TeamFunnelHierDPGoalsModel, row) = (λ_h = row.λ_h, λ_a = row.λ_a)
Pred.compute_score_matrix(::TeamFunnelHierDPGoalsModel, params; max_goals::Int = 12) =
    _poisson_score(params.λ_h, params.λ_a; max_goals)

println("[l04] hierarchical funnel loader ready: TeamFunnelHierDPGoalsModel (funnel_hier_pois) " *
        "— per-team p₁_ij/p₂_ij (attack + opponent defence, non-centred zero-sum). " *
        "Comparators: TeamFunnelDPGoalsModel (l03) / TeamDPGoalsModel (l01).")
