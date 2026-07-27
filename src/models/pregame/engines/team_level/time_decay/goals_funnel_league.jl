# src/models/pregame/engines/team_level/time_decay/goals_funnel_league.jl
#
# TEAM-LEVEL TWO-LAYER THINNED-POISSON FUNNEL {shots -> goals + league offset}.
#
#     Shots ~ Poisson(λ_s)
#     Goals ~ Poisson(λ_s · p₂)          (p₁ ≡ 1: there is no shots-on-target layer)
#
# Graduated from current_development/bbc_xg_proxy/l05_funnel_flex.jl, cell
# `TeamFunnelFlexDPGoalsModel(cascade_weight = 0, sot_on = false, p2_prior = Normal(logit(0.145), 0.5))`
# — the r06 full-spec winner (60 biweek folds, 3 seasons, ScottishLower [56, 57]): beats the
# structural goals-only baseline on 1X2 by Δ −0.0035 and ties totals. r06/Stage-2 closed the SoT
# layer, per-team conversion and the funnel+iso fusion, so ONLY this clean two-layer form exists
# here — there is deliberately no cascade_weight / sot_on / funnel_weight knob.
#
# WHY IT HELPS: shots arrive at ~10 per side per match against ~1.4 goals, so the observation model
# sees ~7× the count volume for the same 95 parameters. Because `cascade_weight = 0` routes goals
# through the MARGINAL Poisson (not a Binomial conditioned on shots), shots and goals appear in the
# same term as λ_s and JOINTLY pin team strength — the cascade routing, which makes goals
# conditionally independent of λ_s given shots, is what lost totals at r03.
#
# By Poisson thinning the goals marginal stays Poisson(λ_s·p₂), so every downstream pricing object
# is unchanged: the plain Poisson score grid prices 1X2 / totals / BTTS / CS exactly. That is why
# the engine is registered in the Poisson dispatch Union in
# src/predictions/score_computation/poisson.jl — it subtypes AbstractNegBinModel and would
# otherwise take the NegBin path and error on a missing `r` column.
#
# HOME ADVANTAGE rides on log λ_s ONLY: the 2026-07-17 EDA found home advantage is entirely shot
# VOLUME (p₂ 0.323 home vs 0.324 away), i.e. conversion is home-invariant.
#
# DATA: per-side shot counts come from `ds.bbc` via Features.ShotsFunnelFeature. Segments with no
# BBC coverage get an all-zero mask, which drops the shots likelihood entirely and degrades the
# engine to a goals-only Poisson. ⚠ Do not actually run it that way: with no shots, only the
# PRODUCT λ_s·p₂ is identified — λ_s and p₂ sit on a ridge held apart by their priors alone, so
# convergence will be poor and p₂ meaningless. Use a goals-only engine on those segments.

using StatsFuns: logit
using LogExpFunctions: log1pexp

# ==========================================
# 1. THE MODEL CONFIGURATION
# ==========================================
Base.@kwdef struct DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel{
    I<:AbstractInterceptionConfig,
    T<:AbstractDynamicsConfig,
    H<:AbstractHomeAdvantageConfig
    } <: AbstractTimeDecayTeamModel
      # `shot_scale` is a FIXED OFFSET on the log-rate, not part of the prior. That keeps μ_base at
      # O(0) so the sampler's default UniformInit(-2, 2) (src/experiments/presets.jl) starts on the
      # right scale. Putting the shot level in the prior instead (Normal(2.3, 0.3)) makes init start
      # at λ_s ≈ 1 against data of ~10 shots — huge gradients and a crushed step size (ε ~ 4e-4).
      shot_scale::Float64     = log(10.0)
      interception_config::I  = HierarchicalMonthlyInterception(prior_μ_base = Normal(0.0, 0.3))
      dynamics_config::T      = TimeDecayDynamics()
      homeadvantage_config::H = HierarchicalTeamHomeAdvantage()
      # goals per SHOT (not per shot-on-target) — p₁ ≡ 1, so p₂ carries the whole conversion
      p2_prior::Distribution  = Normal(logit(0.145), 0.5)
      league_offset_sd::Float64 = 0.1   # zero-sum δ_league prior sd (gap ≈ 0.047 on 56/57)
      league_ha_sd::Float64     = 0.1
      league_ha_on::Bool        = false # optional per-league HA offset (gated, branch-free)
end

# ==========================================
# 2. THE TURING ENGINE
# ==========================================
@model function build_funnel_double_poisson_goals_league_engine(
    home_team_indices::Vector{Int}, away_team_indices::Vector{Int},
    season_indices::Vector{Int}, month_indices::Vector{Int}, league_indices::Vector{Int},
    suff_h::NamedTuple, suff_a::NamedTuple,
    shot_scale::Float64,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64,
    config::DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel
)
    # --- 1. COMPONENTS ---
    inter ~ to_submodel(build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(build_dynamics(config.dynamics_config, n_teams))

    # --- league offsets (zero-sum; HA offset gated branch-free) ---
    δ_league_raw ~ filldist(Normal(0.0, config.league_offset_sd), n_leagues)
    γ_league_raw ~ filldist(Normal(0.0, config.league_ha_sd), n_leagues)
    δ_league = δ_league_raw .- mean(δ_league_raw)
    γ_league = league_ha_active .* (γ_league_raw .- mean(γ_league_raw))

    # --- 2. SHOT RATES (note the shot_scale offset; HA on the home side only) ---
    int_m = shot_scale .+ view(inter.μ_base, season_indices) .+ view(inter.δ_month, month_indices)
    lg    = view(δ_league, league_indices)
    γ_lg  = view(γ_league, league_indices)

    log_λ_s_h = clamp.(int_m .+ lg .+ view(ha, home_team_indices) .+ γ_lg .+
                       view(dyn.α, home_team_indices) .+ view(dyn.β, away_team_indices), -10.0, 10.0)
    log_λ_s_a = clamp.(int_m .+ lg .+
                       view(dyn.α, away_team_indices) .+ view(dyn.β, home_team_indices), -10.0, 10.0)
    # No 1e-6 floor here (unlike the smile engine): the clamp already bounds λ_s to
    # [4.5e-5, 2.2e4], and log_λ_s is used DIRECTLY as the log-rate below, so rate and log-rate
    # must stay exactly consistent.
    λ_s_h = exp.(log_λ_s_h)
    λ_s_a = exp.(log_λ_s_a)

    # --- 3. CONVERSION (global; logit scale) ---
    p2_raw ~ config.p2_prior
    # stable log p straight off the logit — no logistic round-trip
    log_p2 = -log1pexp(-p2_raw)
    p2     = exp(log_p2)

    # AD-Safe Rejection
    is_bad = any(isnan, λ_s_h) || any(isnan, λ_s_a) || any(isinf, λ_s_h) || any(isinf, λ_s_a) ||
             isnan(log_p2)
    λ_s_h = ifelse.(isnan.(λ_s_h) .| isinf.(λ_s_h), one.(λ_s_h), λ_s_h)
    λ_s_a = ifelse.(isnan.(λ_s_a) .| isinf.(λ_s_a), one.(λ_s_a), λ_s_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    # --- 4. LIKELIHOOD via sufficient statistics (see build_turing_model) ---
    # Counts, masks and decay weights are all DATA, so the weighted log-likelihood collapses onto
    # a handful of constants computed once by the builder:
    #   Σ w·m·logPois(shots | λ_s)   = Σ(w·m·shots)·log λ_s − Σ(w·m)·λ_s              [+ const]
    #   Σ w·logPois(goals | λ_s·p₂)  = Σ(w·goals)·log λ_s + Σ(w·goals)·log p₂ − p₂·Σ(w)·λ_s
    # A leapfrog step is then 4 weighted sums per side instead of broadcast logpdf kernels. The
    # dropped constants (log y!) are parameter-free, so the posterior is EXACTLY unchanged — only
    # the reported `lp` is shifted by a fixed amount (never compare `lp` across engines).
    ll_shots_h = sum(suff_h.c_shots_lin .* log_λ_s_h) - sum(suff_h.c_shots_rate .* λ_s_h)
    ll_shots_a = sum(suff_a.c_shots_lin .* log_λ_s_a) - sum(suff_a.c_shots_rate .* λ_s_a)

    # Goals go through the MARGINAL Poisson, where they share a term with λ_s — this is where
    # goals inform team strength, and it applies to every match including those missing shots.
    ll_goals_h = sum(suff_h.c_goals_lin .* log_λ_s_h) + suff_h.S_goals * log_p2 -
                 p2 * sum(suff_h.c_goals_rate .* λ_s_h)
    ll_goals_a = sum(suff_a.c_goals_lin .* log_λ_s_a) + suff_a.S_goals * log_p2 -
                 p2 * sum(suff_a.c_goals_rate .* λ_s_a)

    Turing.@addlogprob! ll_shots_h + ll_shots_a + ll_goals_h + ll_goals_a
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function Features.required_features(model::DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(),
        Features.GoalsFeature(),
        Features.DatesFeature(),
        Features.MonthFeature(),
        Features.LeagueFeature(),
        Features.ShotsFunnelFeature(),
        Features.TimeIndicesFeature(),
    ]
end

"""
Per-side sufficient statistics. `c_*_lin` weight `log λ_s`, `c_*_rate` weight `λ_s`, and `S_goals`
weights `log p₂`.

The shots terms carry the usability mask, so a match with no BBC shot count drops out of the shots
likelihood cleanly (the count itself is a 0 dummy — see Features.ShotsFunnelFeature — so nothing
invalid is ever evaluated). The goals terms carry NO mask: every match informs the goal rate.
"""
function _funnel_suff(shots::Vector{Int}, mask::Vector{Float64}, goals::Vector{Int},
                      w::Vector{Float64})
    c_goals_lin = w .* goals
    return (
        c_shots_lin  = w .* mask .* shots,
        c_shots_rate = w .* mask,
        c_goals_lin  = c_goals_lin,
        c_goals_rate = w,
        S_goals      = sum(c_goals_lin),
    )
end

function build_turing_model(config::DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel,
                            feature_set::FeatureSet)
    data = feature_set.data

    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    n_months  = 12
    n_leagues = Int(data[:n_leagues])

    date_deltas   = Vector{Int}(data[:dates])
    match_weights = 0.5 .^ (date_deltas ./ config.dynamics_config.days_half_life)

    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    season_ids = Vector{Int}(data[:season_indices])
    month_idx  = Vector{Int}(data[:flat_months])
    league_idx = Vector{Int}(data[:flat_league_ids])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    shots_h = Vector{Int}(data[:flat_home_shots_n])
    shots_a = Vector{Int}(data[:flat_away_shots_n])
    mask_h  = Vector{Float64}(data[:flat_funnel_mask_h])
    mask_a  = Vector{Float64}(data[:flat_funnel_mask_a])

    return build_funnel_double_poisson_goals_league_engine(
        home_ids, away_ids, season_ids, month_idx, league_idx,
        _funnel_suff(shots_h, mask_h, home_goals, match_weights),
        _funnel_suff(shots_a, mask_a, away_goals, match_weights),
        config.shot_scale,
        n_teams, n_seasons, n_months, n_leagues,
        config.league_ha_on ? 1.0 : 0.0,
        config
    )
end

# ==========================================
# 4. THE EXTRACTOR
# ==========================================
function extract_parameters(
    model::DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel,
    df::AbstractDataFrame,
    feature_set::FeatureSet,
    chain::Chains
)
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    n_leagues = Int(data[:n_leagues])
    team_map  = data[:team_map]
    league_lookup = data[:league_lookup]

    inter_nt = extract_interception(chain, model.interception_config, n_seasons)
    ha_mat   = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    dyn_nt   = extract_dynamics(chain, model.dynamics_config, "dyn", n_teams)

    n_samples = size(chain, 1) * size(chain, 3)

    # zero-sum league offsets, reconstructed exactly as in the @model
    δ_raw = zeros(n_samples, n_leagues)
    γ_raw = zeros(n_samples, n_leagues)
    for i in 1:n_leagues
        δ_raw[:, i] = vec(Array(chain[Symbol("δ_league_raw[$i]")]))
        model.league_ha_on && (γ_raw[:, i] = vec(Array(chain[Symbol("γ_league_raw[$i]")])))
    end
    δ_mat = δ_raw .- mean(δ_raw, dims=2)
    γ_mat = model.league_ha_on ? (γ_raw .- mean(γ_raw, dims=2)) : zeros(n_samples, n_leagues)

    # global conversion p₂ = logistic(p2_raw)
    p2 = 1 ./ (1 .+ exp.(-vec(Array(chain[:p2_raw]))))

    results = Dict{Int, NamedTuple}()
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

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        m_idx = month(row.match_date)
        # the shot_scale offset must mirror the @model exactly
        int_v = model.shot_scale .+ inter_nt.μ_base[:, s_idx] .+ inter_nt.δ_month[:, m_idx]

        log_λ_s_h = clamp.(int_v .+ lg .+ γ_h .+ γlg .+ α_h .+ β_a, -10.0, 10.0)
        log_λ_s_a = clamp.(int_v .+ lg .+                α_a .+ β_h, -10.0, 10.0)
        λ_s_h = exp.(log_λ_s_h)
        λ_s_a = exp.(log_λ_s_a)

        # thinning: the goal rate is the shot rate times the conversion
        λ_h = λ_s_h .* p2
        λ_a = λ_s_a .* p2

        results[mid] = (;
            λ_h, λ_a,
            λ_s_h, λ_s_a, p2,
            true_xg_h = λ_h, true_xg_a = λ_a,
        )
    end
    return results
end
