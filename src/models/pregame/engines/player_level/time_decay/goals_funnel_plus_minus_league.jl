# src/models/pregame/engines/player_level/time_decay/goals_funnel_plus_minus_league.jl
#
# TWO-LAYER FUNNEL {shots -> goals} + PLUS-MINUS (RAPM) PLAYER PILLAR + LEAGUE OFFSET.
#
#     Shots ~ Poisson(λ_s),  Goals ~ Poisson(λ_s · p₂)
#     log λ_s(home) = shot_scale + μ_base + δ_month + δ_league + ha_h + dyn.α_h + dyn.β_a
#                                + w_att·R_h − w_def·R_a
#
# THE FUSION THE APM PLAN DEFERRED. `goals_funnel_league.jl` and `goals_plus_minus_league.jl` beat
# the same no-APM baseline by almost exactly the same margin and finish in a near-tie with each
# other. This engine asks whether their gains are the SAME information or DIFFERENT information.
#
# WHY THEY SHOULD BE DIFFERENT — and why the xG plus-minus target is the interesting one:
#   * The funnel's gain is OBSERVATION VOLUME. Shots arrive at ~10 per side against ~1.4 goals, so
#     the same ~95 parameters see ~7x the count volume and team strength is pinned far more
#     precisely. But it reads raw shot COUNTS from `ds.bbc` — a 30-yard speculative effort and a
#     six-yard tap-in are the same datum to it.
#   * `XGPlusMinusFeature` is shot QUALITY (BBC zone / body part / set-piece context, via the
#     commentary xG cell table) attributed to individual players. The funnel structurally cannot
#     see that.
#   ⇒ funnel = volume, xG-PM = quality-per-player. Complementary.
#
# The prediction this engine exists to test: fusion should pay for `XGPlusMinusFeature` and NOT for
# `ShotsPlusMinusFeature`, because a shots-PM rating is the very signal the funnel already
# exploits, merely decomposed to players. Running both arms falsifies or supports that directly.
#
# ⚠ TWO REASONS TO EXPECT A SMALL OR NULL EFFECT, recorded so a modest result is not over-read:
#   1. The `bbc_xg_proxy` stream's r07b found funnel + iso-market fusion null / soft-negative, with
#      the lesson "fusion regresses the 1X2 edge to market, keep modular". That was a MARKET pillar
#      rather than a player pillar, but it is the closest prior.
#   2. The funnel makes α/β sharper, which leaves LESS residual for w_att/w_def to explain — the
#      pillar's effect in the goals-only engine was already measured net of α/β.
#
# Home advantage rides on log λ_s only (the 2026-07-17 EDA found home advantage is entirely shot
# VOLUME: p₂ 0.323 home vs 0.324 away). The pillar rides on log λ_s too, so a stronger lineup lifts
# shots AND goals coherently; Poisson thinning keeps the goals marginal exactly Poisson(λ_s·p₂), so
# every downstream pricing object is unchanged and the plain Poisson score grid still applies.
#
# ⚠ Register in the dispatch Union in src/predictions/score_computation/poisson.jl or PPD generation
# takes the NegBin path and errors on a missing `r` column.

using StatsFuns: logit
using LogExpFunctions: log1pexp

# ==========================================
# 1. THE MODEL CONFIGURATION
# ==========================================
Base.@kwdef struct DynamicFunnelPlusMinusGoalsLeagueTimeDecayModel{
    I<:AbstractInterceptionConfig,
    T<:AbstractDynamicsConfig,
    H<:AbstractHomeAdvantageConfig,
    P<:Features.AbstractPlusMinusFeature
    } <: AbstractTimeDecayPlayerModel
      # Fixed OFFSET on the log-rate, not part of the prior — keeps μ_base at O(0) so the sampler's
      # UniformInit(-2, 2) starts on the right scale. See goals_funnel_league.jl.
      shot_scale::Float64      = log(10.0)
      interception_config::I   = HierarchicalMonthlyInterception(prior_μ_base = Normal(0.0, 0.3))
      dynamics_config::T       = TimeDecayDynamics()
      homeadvantage_config::H  = HierarchicalTeamHomeAdvantage()
      # Defaults to the xG target: the pairing predicted to be complementary (see header).
      player_ratings_feature::P = Features.XGPlusMinusFeature()
      p2_prior::Distribution   = Normal(logit(0.145), 0.5)   # goals per SHOT (p₁ ≡ 1)
      w_att_prior::Distribution = Normal(0.0, 0.3)
      w_def_prior::Distribution = Normal(0.0, 0.3)
      league_offset_sd::Float64 = 0.1
      league_ha_sd::Float64     = 0.1
      league_ha_on::Bool        = false
end

# ==========================================
# 2. THE TURING ENGINE
# ==========================================
@model function build_funnel_plus_minus_goals_league_engine(
    home_team_indices::Vector{Int}, away_team_indices::Vector{Int},
    season_indices::Vector{Int}, month_indices::Vector{Int}, league_indices::Vector{Int},
    rat_h::Vector{Float64}, rat_a::Vector{Float64},
    suff_h::NamedTuple, suff_a::NamedTuple,
    shot_scale::Float64,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64,
    config::DynamicFunnelPlusMinusGoalsLeagueTimeDecayModel
)
    # --- 1. COMPONENTS ---
    inter ~ to_submodel(build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(build_dynamics(config.dynamics_config, n_teams))

    w_att ~ config.w_att_prior
    w_def ~ config.w_def_prior

    δ_league_raw ~ filldist(Normal(0.0, config.league_offset_sd), n_leagues)
    γ_league_raw ~ filldist(Normal(0.0, config.league_ha_sd), n_leagues)
    δ_league = δ_league_raw .- mean(δ_league_raw)
    γ_league = league_ha_active .* (γ_league_raw .- mean(γ_league_raw))

    # --- 2. SHOT RATES (shot_scale offset; HA on the home side only) ---
    int_m = shot_scale .+ view(inter.μ_base, season_indices) .+ view(inter.δ_month, month_indices)
    lg    = view(δ_league, league_indices)
    γ_lg  = view(γ_league, league_indices)

    pillar_h = w_att .* rat_h .- w_def .* rat_a
    pillar_a = w_att .* rat_a .- w_def .* rat_h

    log_λ_s_h = clamp.(int_m .+ lg .+ view(ha, home_team_indices) .+ γ_lg .+
                       view(dyn.α, home_team_indices) .+ view(dyn.β, away_team_indices) .+
                       pillar_h, -10.0, 10.0)
    log_λ_s_a = clamp.(int_m .+ lg .+
                       view(dyn.α, away_team_indices) .+ view(dyn.β, home_team_indices) .+
                       pillar_a, -10.0, 10.0)
    λ_s_h = exp.(log_λ_s_h)
    λ_s_a = exp.(log_λ_s_a)

    # --- 3. CONVERSION (global; logit scale) ---
    p2_raw ~ config.p2_prior
    log_p2 = -log1pexp(-p2_raw)
    p2     = exp(log_p2)

    # AD-Safe Rejection
    is_bad = any(isnan, λ_s_h) || any(isnan, λ_s_a) || any(isinf, λ_s_h) || any(isinf, λ_s_a) ||
             isnan(log_p2)
    λ_s_h = ifelse.(isnan.(λ_s_h) .| isinf.(λ_s_h), one.(λ_s_h), λ_s_h)
    λ_s_a = ifelse.(isnan.(λ_s_a) .| isinf.(λ_s_a), one.(λ_s_a), λ_s_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    # --- 4. LIKELIHOOD via sufficient statistics (identical to goals_funnel_league.jl) ---
    ll_shots_h = sum(suff_h.c_shots_lin .* log_λ_s_h) - sum(suff_h.c_shots_rate .* λ_s_h)
    ll_shots_a = sum(suff_a.c_shots_lin .* log_λ_s_a) - sum(suff_a.c_shots_rate .* λ_s_a)

    ll_goals_h = sum(suff_h.c_goals_lin .* log_λ_s_h) + suff_h.S_goals * log_p2 -
                 p2 * sum(suff_h.c_goals_rate .* λ_s_h)
    ll_goals_a = sum(suff_a.c_goals_lin .* log_λ_s_a) + suff_a.S_goals * log_p2 -
                 p2 * sum(suff_a.c_goals_rate .* λ_s_a)

    Turing.@addlogprob! ll_shots_h + ll_shots_a + ll_goals_h + ll_goals_a
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function Features.required_features(model::DynamicFunnelPlusMinusGoalsLeagueTimeDecayModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(),
        Features.GoalsFeature(),
        Features.DatesFeature(),
        Features.MonthFeature(),
        Features.LeagueFeature(),
        Features.ShotsFunnelFeature(),
        model.player_ratings_feature,
        Features.TimeIndicesFeature(),
    ]
end

function build_turing_model(config::DynamicFunnelPlusMinusGoalsLeagueTimeDecayModel,
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

    base = Features.rating_base(config.player_ratings_feature)
    rat_h = _pm_outfield(Vector{Float64}(data[:flat_home_D_rating]),
                         Vector{Float64}(data[:flat_home_M_rating]),
                         Vector{Float64}(data[:flat_home_F_rating]), base)
    rat_a = _pm_outfield(Vector{Float64}(data[:flat_away_D_rating]),
                         Vector{Float64}(data[:flat_away_M_rating]),
                         Vector{Float64}(data[:flat_away_F_rating]), base)

    return build_funnel_plus_minus_goals_league_engine(
        home_ids, away_ids, season_ids, month_idx, league_idx,
        rat_h, rat_a,
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
    model::DynamicFunnelPlusMinusGoalsLeagueTimeDecayModel,
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
    ratings_map   = data[:player_ratings_map]

    inter_nt = extract_interception(chain, model.interception_config, n_seasons)
    ha_mat   = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    dyn_nt   = extract_dynamics(chain, model.dynamics_config, "dyn", n_teams)

    n_samples = size(chain, 1) * size(chain, 3)

    w_att = vec(Array(chain[:w_att]))
    w_def = vec(Array(chain[:w_def]))

    δ_raw = zeros(n_samples, n_leagues)
    γ_raw = zeros(n_samples, n_leagues)
    for i in 1:n_leagues
        δ_raw[:, i] = vec(Array(chain[Symbol("δ_league_raw[$i]")]))
        model.league_ha_on && (γ_raw[:, i] = vec(Array(chain[Symbol("γ_league_raw[$i]")])))
    end
    δ_mat = δ_raw .- mean(δ_raw, dims=2)
    γ_mat = model.league_ha_on ? (γ_raw .- mean(γ_raw, dims=2)) : zeros(n_samples, n_leagues)

    p2 = 1 ./ (1 .+ exp.(-vec(Array(chain[:p2_raw]))))
    base = Features.rating_base(model.player_ratings_feature)

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

        m_r = get(ratings_map, mid, Dict{Tuple{String, String}, Float64}())
        r_h = (get(m_r, ("home", "D"), 0.0) + get(m_r, ("home", "M"), 0.0) +
               get(m_r, ("home", "F"), 0.0)) - 10.0 * base
        r_a = (get(m_r, ("away", "D"), 0.0) + get(m_r, ("away", "M"), 0.0) +
               get(m_r, ("away", "F"), 0.0)) - 10.0 * base

        pillar_h = w_att .* r_h .- w_def .* r_a
        pillar_a = w_att .* r_a .- w_def .* r_h

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        m_idx = month(row.match_date)
        int_v = model.shot_scale .+ inter_nt.μ_base[:, s_idx] .+ inter_nt.δ_month[:, m_idx]

        log_λ_s_h = clamp.(int_v .+ lg .+ γ_h .+ γlg .+ α_h .+ β_a .+ pillar_h, -10.0, 10.0)
        log_λ_s_a = clamp.(int_v .+ lg .+               α_a .+ β_h .+ pillar_a, -10.0, 10.0)
        λ_s_h = exp.(log_λ_s_h)
        λ_s_a = exp.(log_λ_s_a)

        λ_h = λ_s_h .* p2
        λ_a = λ_s_a .* p2

        results[mid] = (;
            λ_h, λ_a,
            λ_s_h, λ_s_a, p2,
            pillar_h, pillar_a,
            true_xg_h = λ_h, true_xg_a = λ_a,
        )
    end
    return results
end
