# current_development/bbc_xg_proxy/l06_funnel_iso.jl
#
# LOADER. FUNNEL + ISO MARKET PILLAR. The r06 full-spec verdict was: the two-layer
# shots→goals funnel (cascade_weight=0, sot_on=false) is a genuine STRUCTURAL improvement over
# none_pois (beats it on 1X2, ties totals), but the isotropic market pillar still owns totals
# (iso −0.0058 vs close). Those two levers are orthogonal — one sharpens the structural core,
# the other anchors it to the market. This loader combines them so we can finally ask:
#
#     does a sharper structural core add anything ONCE the iso market pillar sits on top?
#     i.e.  funnel+iso  vs  none+iso  (the stored iso_pois_mw25/mw40)   at matched market_weight.
#
# The engine = l05's flexible funnel likelihood (same 95 params, same sufficient statistics,
# same cw/sot knobs) PLUS the exact iso pillar from scottish_lower_smile/l01's
# TeamIsoDPGoalsModel: a soft anchor `log λ_goals ~ Normal(market_log, σ_market)` with σ_market
# SAMPLED (the release valve — never fix it) and a `market_weight` knob (mw).
#
# WHERE THE ANCHOR LANDS. The market feature (DoublePoissonMarketFeature) supplies the de-vigged
# implied GOAL log-rates. The funnel's goal rate is λ_goals = λ_s · p₁ · p₂, so its goal
# log-rate is log_λ_s + lp12 (lp12 = log p₁ + log p₂). We anchor THAT — the same quantity the
# none+iso engine anchors as log_λ_h. With sot_on=false, lp12 = log p₂ (log p₁ ≡ 0), so the
# pillar shapes per-match team strength (log_λ_s) while the global p₂ sets the level — exactly
# the intended division of labour: shots pin relative strength, the market refines per match.
#
# Everything else (BBCFunnelFeature, _unpack_funnel, safe dummies, sufficient statistics,
# _extract_funnel_core, Poisson-grid pricing, the market helpers _unpack_market / MARKET_PRIOR /
# _market_active) is inherited unchanged from l05 → l03 → scottish_lower_smile/l01.

using Turing
using Distributions
using StatsFuns: logit
using LogExpFunctions: log1pexp

const ROOT_L06 = pkgdir(BayesianFootball)
include(joinpath(ROOT_L06, "current_development/bbc_xg_proxy/l05_funnel_flex.jl"))

Base.@kwdef struct TeamFunnelIsoDPGoalsModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    M<:Features.AbstractMarketFeatureConfig
    } <: PreGame.AbstractTimeDecayTeamModel
      # --- funnel core (identical defaults to l05's winner: cw=0, sot off) ---
      shot_scale::Float64     = log(10.0)
      interception_config::I  = PreGame.HierarchicalMonthlyInterception(
                                    prior_μ_base = Normal(0.0, 0.3))
      dynamics_config::T      = PreGame.TimeDecayDynamics()
      homeadvantage_config::H = PreGame.HierarchicalTeamHomeAdvantage()
      p1_prior::Distribution  = Normal(logit(0.44), 0.5)
      p2_prior::Distribution  = Normal(logit(0.145), 0.5)   # goals per SHOT (sot_on=false default)
      cascade_weight::Float64 = 0.0     # r06 winner: goals joint with shots on λ_s
      sot_on::Bool            = false   # r06 winner: SoT layer worthless, drop it
      funnel_weight::Float64  = 1.0
      # --- iso market pillar (identical to TeamIsoDPGoalsModel) ---
      market_feature_config::M = Features.DoublePoissonMarketFeature()
      market_σ::Distribution   = MARKET_PRIOR      # SAMPLED release valve — never fix
      market_weight::Float64   = 0.4               # mw knob (Ireland/56-57 optimum 0.25–0.40)
      market_on::Bool          = true
      # --- league nuisance ---
      league_offset_sd::Float64 = 0.1
      league_ha_sd::Float64     = 0.1
      league_ha_on::Bool        = false
end

@model function build_team_funnel_iso_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    suff_h, suff_a, match_weights::Vector{Float64},
    market_log_h::Vector{Float64}, market_log_a::Vector{Float64}, market_mask::Vector{Float64},
    market_active::Float64,
    funnel_weight::Float64, shot_scale::Float64, cw::Float64, sot_active::Float64,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64,
    config
)
    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(PreGame.build_dynamics(config.dynamics_config, n_teams))
    σ_market ~ config.market_σ

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
    log_p1, log_q1 = sot_active * -log1pexp(-p1_raw), sot_active * -log1pexp(p1_raw)
    log_p2, log_q2 = -log1pexp(-p2_raw), -log1pexp(p2_raw)

    is_bad = any(isnan, λ_s_h) || any(isnan, λ_s_a) || any(isinf, λ_s_h) || any(isinf, λ_s_a) ||
             isnan(log_p1) || isnan(log_p2)
    λ_s_h = ifelse.(isnan.(λ_s_h) .| isinf.(λ_s_h), one.(λ_s_h), λ_s_h)
    λ_s_a = ifelse.(isnan.(λ_s_a) .| isinf.(λ_s_a), one.(λ_s_a), λ_s_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    lp12 = log_p1 + log_p2
    conv = exp(lp12)

    # --- funnel likelihood via sufficient statistics (IDENTICAL to l05 flex) ---
    ll_shots_h = sum(suff_h.c_shots_lin .* log_λ_s_h) - sum(suff_h.c_shots_rate .* λ_s_h)
    ll_shots_a = sum(suff_a.c_shots_lin .* log_λ_s_a) - sum(suff_a.c_shots_rate .* λ_s_a)
    ll_sot_h = sot_active * (suff_h.S_sot * log_p1 + suff_h.S_miss * log_q1)
    ll_sot_a = sot_active * (suff_a.S_sot * log_p1 + suff_a.S_miss * log_q1)
    ll_casc_h = cw * (suff_h.S_goal * log_p2 + suff_h.S_save * log_q2)
    ll_casc_a = cw * (suff_a.S_goal * log_p2 + suff_a.S_save * log_q2)
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

    # --- ISO MARKET PILLAR — anchor the GOAL log-rate log(λ_s·conv) = log_λ_s + lp12 ---
    # Written in the SAME sufficient-statistic arithmetic form as the funnel likelihood above
    # (plain log / sum / broadcast, NO logpdf.(Normal...) kernel). This is mathematically the
    # identical anchor — a Normal logpdf summed, minus the parameter-free −½log(2π)·Σw constant —
    # but the logpdf.(Normal.(·,σ_market),·) broadcast, when it shares the tape with the funnel's
    # suff-stat goal term (both differentiate through lp12 = log p₂), makes ReverseDiff's
    # COMPILED tape return a wrong gradient for p2_raw: p₂ collapses to ~0 and σ_market inflates
    # past its prior. The pure funnel (r06) and the none+iso pillar each compile fine alone; only
    # the combination breaks. Expanding to arithmetic sidesteps it (verified: p₂→0.147,
    # σ_market→0.155, matching none+iso) and is what the production trainer's compile=true needs.
    wm = match_weights .* market_mask
    Σwm = sum(wm)
    r_h = log_λ_s_h .+ lp12 .- market_log_h
    r_a = log_λ_s_a .+ lp12 .- market_log_a
    ll_mkt = -Σwm * log(σ_market) - (0.5 / σ_market^2) * (sum(wm .* r_h .^ 2) + sum(wm .* r_a .^ 2))
    Turing.@addlogprob! market_active * config.market_weight * ll_mkt
end

function Features.required_features(model::TeamFunnelIsoDPGoalsModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), LeagueFeature(), Features.TimeIndicesFeature(),
        BBCFunnelFeature(), model.market_feature_config,
    ]
end

function PreGame.build_turing_model(config::TeamFunnelIsoDPGoalsModel, feature_set)
    d = _unpack_funnel(feature_set.data, config)
    mh, ma, mm = _unpack_market(feature_set.data)
    return build_team_funnel_iso_engine(
        d.home_ids, d.away_ids, d.season_idx, d.month_idx, d.league_idx,
        d.suff_h, d.suff_a, d.match_weights,
        mh, ma, mm, _market_active(config),
        config.funnel_weight, config.shot_scale, config.cascade_weight,
        config.sot_on ? 1.0 : 0.0,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        config.league_ha_on ? 1.0 : 0.0, config)
end

# extraction / prediction — identical to l05 flex (the pillar only shaped the posterior;
# pricing is the plain thinned Poisson grid on λ_goals = λ_s · conv)
function PreGame.extract_parameters(model::TeamFunnelIsoDPGoalsModel, df, feature_set, chain)
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

Pred.extract_params(::TeamFunnelIsoDPGoalsModel, row) = (λ_h = row.λ_h, λ_a = row.λ_a)
Pred.compute_score_matrix(::TeamFunnelIsoDPGoalsModel, params; max_goals::Int = 12) =
    _poisson_score(params.λ_h, params.λ_a; max_goals)

println("[l06] funnel+iso ready: TeamFunnelIsoDPGoalsModel — l05 flexible funnel (cw/sot knobs) " *
        "+ the TeamIsoDPGoalsModel market pillar anchoring log(λ_s·conv) to the de-vigged close " *
        "(sampled σ_market, knob market_weight). Defaults = r06 winner core (cw=0, sot off).")
