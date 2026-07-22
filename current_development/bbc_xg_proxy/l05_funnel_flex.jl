# current_development/bbc_xg_proxy/l05_funnel_flex.jl
#
# LOADER. FLEXIBLE funnel: same three data streams as l03 (shots, SoT, goals), same sufficient
# statistics, same global p₁/p₂ — but two knobs that decide HOW the data is used. No new
# parameters beyond l03 (95), so it runs at Stage-1 cost (~28 min), NOT Stage-2 cost (3h 40m —
# that was the 80 hierarchical team effects, which Stage 2 showed buy nothing).
#
# KNOB 1 — `cascade_weight` (cw): how the GOALS likelihood is routed.
#
#     goals term = cw·cm · logBin(G | T, p₂)  +  (1 − cw·cm) · logPois(G | λ_s·p₁·p₂)
#
#   cw = 1  → pure cascade (l03 Stage 1). Given T, goals are conditionally independent of λ_s,
#             so goals contribute ZERO gradient to team strength: λ_s is fitted to shots alone
#             and the goal LEVEL rides on a single global conversion constant. That won 1X2
#             (Δ −0.0071 vs none_pois) but lost totals (Δ +0.0078).
#   cw = 0  → goals go through the marginal Poisson, where they appear in the SAME term as λ_s.
#             Shots and goals then JOINTLY pin team strength — shots bring 7× the counts, goals
#             bring the actual target. This is the Ireland joint goals+xG structure with shots
#             playing the role of xG, and it is the only untested lever for the totals deficit.
#   0<cw<1 → geometric blend of the two routes.
#
#   (cm is the per-match cascade-validity mask: rows with goals > SoT or missing stats always
#   take the marginal route, so the effective marginal weight is 1 − cw·cm.)
#
# KNOB 2 — `sot_on`: whether the SoT layer is used at all.
#
#   true  → three-layer funnel; p₁ is identified by SoT|shots, p₂ by goals|SoT.
#   false → two-layer (shots → goals). The SoT term is dropped AND log p₁ is forced to 0
#           (p₁ ≡ 1) so p₂ alone carries the whole conversion — this avoids the p₁·p₂ ridge
#           that would otherwise make the two unidentifiable. Set p2_prior ≈ logit(0.145)
#           (goals per SHOT) in this mode.
#
#   Why this is worth testing rather than assumed: with a GLOBAL p₁, SoT is ANCILLARY for team
#   strength once shots are observed (T|S ~ Bin(S,p₁) depends only on p₁). Its team-level
#   content is exactly σ_p1, which Stage 2 measured at 0.034, CI [0.004, 0.074] — small but not
#   zero. `sot_on=false` vs `true` measures empirically what that is worth in prediction.
#
# Everything else (BBCFunnelFeature, _unpack_funnel, safe dummies, _extract_funnel_core,
# Poisson-grid pricing) is inherited from l03 unchanged.

using Turing
using Distributions
using DataFrames
using Dates
using StatsFuns: logit
using LogExpFunctions: log1pexp

const ROOT_L05 = pkgdir(BayesianFootball)
include(joinpath(ROOT_L05, "current_development/bbc_xg_proxy/l03_funnel_cascade.jl"))

Base.@kwdef struct TeamFunnelFlexDPGoalsModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig
    } <: PreGame.AbstractTimeDecayTeamModel
      shot_scale::Float64     = log(10.0)
      interception_config::I  = PreGame.HierarchicalMonthlyInterception(
                                    prior_μ_base = Normal(0.0, 0.3))
      dynamics_config::T      = PreGame.TimeDecayDynamics()
      homeadvantage_config::H = PreGame.HierarchicalTeamHomeAdvantage()
      p1_prior::Distribution  = Normal(logit(0.44), 0.5)
      p2_prior::Distribution  = Normal(logit(0.32), 0.5)   # sot_on=false ⇒ use logit(0.145)
      cascade_weight::Float64 = 1.0    # 1 = l03 cascade, 0 = joint shots+goals on λ_s
      sot_on::Bool            = true
      funnel_weight::Float64  = 1.0    # tempers the shots (+SoT) terms
      league_offset_sd::Float64 = 0.1
      league_ha_sd::Float64     = 0.1
      league_ha_on::Bool        = false
end

@model function build_team_funnel_flex_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    suff_h, suff_a,
    funnel_weight::Float64, shot_scale::Float64, cw::Float64, sot_active::Float64,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64,
    config
)
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

    p1_raw ~ config.p1_prior
    p2_raw ~ config.p2_prior
    # sot_active = 0 ⇒ log p₁ ≡ 0 (p₁ ≡ 1): p₂ alone carries the conversion, no p₁·p₂ ridge
    log_p1, log_q1 = sot_active * -log1pexp(-p1_raw), sot_active * -log1pexp(p1_raw)
    log_p2, log_q2 = -log1pexp(-p2_raw), -log1pexp(p2_raw)

    is_bad = any(isnan, λ_s_h) || any(isnan, λ_s_a) || any(isinf, λ_s_h) || any(isinf, λ_s_a) ||
             isnan(log_p1) || isnan(log_p2)
    λ_s_h = ifelse.(isnan.(λ_s_h) .| isinf.(λ_s_h), one.(λ_s_h), λ_s_h)
    λ_s_a = ifelse.(isnan.(λ_s_a) .| isinf.(λ_s_a), one.(λ_s_a), λ_s_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    lp12 = log_p1 + log_p2
    conv = exp(lp12)

    # Poisson shots (unchanged)
    ll_shots_h = sum(suff_h.c_shots_lin .* log_λ_s_h) - sum(suff_h.c_shots_rate .* λ_s_h)
    ll_shots_a = sum(suff_a.c_shots_lin .* log_λ_s_a) - sum(suff_a.c_shots_rate .* λ_s_a)
    # Binomial SoT | shots — gated by sot_active
    ll_sot_h = sot_active * (suff_h.S_sot * log_p1 + suff_h.S_miss * log_q1)
    ll_sot_a = sot_active * (suff_a.S_sot * log_p1 + suff_a.S_miss * log_q1)
    # goals, cascade route — weight cw·cm
    ll_casc_h = cw * (suff_h.S_goal * log_p2 + suff_h.S_save * log_q2)
    ll_casc_a = cw * (suff_a.S_goal * log_p2 + suff_a.S_save * log_q2)
    # goals, marginal Poisson route — weight (1−cm) always, plus (1−cw)·cm from the blend.
    # THIS is the term where goals and λ_s meet, i.e. where goals inform team strength.
    ll_marg_h = sum(suff_h.c_marg_lin .* log_λ_s_h) + suff_h.S_marg_goals * lp12 -
                conv * sum(suff_h.c_marg_rate .* λ_s_h) +
                (1 - cw) * (sum(suff_h.c_marg2_lin .* log_λ_s_h) + suff_h.S_marg2_goals * lp12 -
                            conv * sum(suff_h.c_marg2_rate .* λ_s_h))
    ll_marg_a = sum(suff_a.c_marg_lin .* log_λ_s_a) + suff_a.S_marg_goals * lp12 -
                conv * sum(suff_a.c_marg_rate .* λ_s_a) +
                (1 - cw) * (sum(suff_a.c_marg2_lin .* log_λ_s_a) + suff_a.S_marg2_goals * lp12 -
                            conv * sum(suff_a.c_marg2_rate .* λ_s_a))

    Turing.@addlogprob!(
        funnel_weight * (ll_shots_h + ll_shots_a + ll_sot_h + ll_sot_a) +
        ll_casc_h + ll_casc_a + ll_marg_h + ll_marg_a)
end

function Features.required_features(model::TeamFunnelFlexDPGoalsModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), LeagueFeature(), Features.TimeIndicesFeature(),
        BBCFunnelFeature(),
    ]
end

function PreGame.build_turing_model(config::TeamFunnelFlexDPGoalsModel, feature_set)
    d = _unpack_funnel(feature_set.data, config)
    return build_team_funnel_flex_engine(
        d.home_ids, d.away_ids, d.season_idx, d.month_idx, d.league_idx,
        d.suff_h, d.suff_a,
        config.funnel_weight, config.shot_scale, config.cascade_weight,
        config.sot_on ? 1.0 : 0.0,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        config.league_ha_on ? 1.0 : 0.0, config)
end

function PreGame.extract_parameters(model::TeamFunnelFlexDPGoalsModel, df, feature_set, chain)
    core, _ = _extract_funnel_core(model, df, feature_set, chain)
    p1 = model.sot_on ? _logistic_draws(chain, "p1_raw") :
                        ones(size(chain, 1) * size(chain, 3))
    p2 = _logistic_draws(chain, "p2_raw")
    conv = p1 .* p2
    results = Dict{Int, NamedTuple}()
    for (mid, c) in core
        λ_h = c.λ_s_h .* conv
        λ_a = c.λ_s_a .* conv
        results[mid] = (; λ_h, λ_a, λ_s_h = c.λ_s_h, λ_s_a = c.λ_s_a, p1, p2,
                          true_xg_h = λ_h, true_xg_a = λ_a)
    end
    return results
end

Pred.extract_params(::TeamFunnelFlexDPGoalsModel, row) = (λ_h = row.λ_h, λ_a = row.λ_a)
Pred.compute_score_matrix(::TeamFunnelFlexDPGoalsModel, params; max_goals::Int = 12) =
    _poisson_score(params.λ_h, params.λ_a; max_goals)

println("[l05] flexible funnel ready: TeamFunnelFlexDPGoalsModel — knobs cascade_weight " *
        "(1 = l03 cascade, 0 = goals joint with shots on λ_s) and sot_on (three- vs two-layer). " *
        "Same 95 params and Stage-1 runtime.")
