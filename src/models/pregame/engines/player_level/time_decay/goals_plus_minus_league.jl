# src/models/pregame/engines/player_level/time_decay/goals_plus_minus_league.jl
#
# GOALS DOUBLE-POISSON + PLUS-MINUS (RAPM) PLAYER PILLAR + LEAGUE OFFSET.
#
#     Goals_h ~ Poisson(λ_h),  log λ_h = μ_base(season) + δ_month + δ_league + ha_h
#                                        + dyn.α_h + dyn.β_a + w_att·R_h − w_def·R_a
#     Goals_a ~ Poisson(λ_a),  log λ_a = μ_base(season) + δ_month + δ_league
#                                        + dyn.α_a + dyn.β_h + w_att·R_a − w_def·R_h
#
# `R_x` is the summed, centred rating of side x's OUTFIELD players (D + M + F), supplied by any
# feature family that emits the `flat_<side>_<pos>_rating` vectors: every
# `Features.AbstractPlusMinusFeature` variant (RAPM on shots — green-lit — shots-on-target, goals,
# xG) AND `Features.PlayerRatingsFeature` (SofaScore ratings through a tracker). Swapping the rating
# source is purely a matter of swapping `player_ratings_feature`; nothing in this file changes,
# EXCEPT that a non-zero-base family needs a tighter `w_*_prior` (see the config below).
#
# WHY THIS ENGINE EXISTS: ScottishLower (tournaments 56/57) has ZERO SofaScore player ratings, so
# the nine `outfield_*` xG engines cannot run there, and it has no player xG either. RAPM is the
# home-grown substitute (see src/features/plus_minus/), and this is the goals-only engine that
# consumes it — structurally the funnel engine with the shots layer removed and the pillar added.
#
# GOALKEEPERS ARE EXCLUDED FROM THE PILLAR, deliberately. A keeper plays nearly every minute, so his
# plus-minus is a comparison against his backup and is barely identified — the research measured GK
# RAPM at rho ~ 0.00 against the SofaScore rating, i.e. pure noise. The `flat_*_G_rating` vectors
# are still emitted by the extractor; this engine simply does not read them.
#
# CENTRING: `Features.rating_base(config.player_ratings_feature)` is 0.0 for the whole plus-minus
# family (a ridge coefficient is zero-centred by construction), so the centring below is a no-op
# there. It is written out anyway so the engine works unchanged if a non-zero-centred rating family
# is ever slotted in.
#
# ⚠ SCORE-GRID REGISTRATION: this engine subtypes `AbstractTimeDecayPlayerModel <: AbstractNegBinModel`
# but returns plain `(λ_h, λ_a)`. It MUST be listed in the dispatch Union in
# src/predictions/score_computation/poisson.jl, or PPD generation takes the NegBin path and errors
# on a missing `r` column.
#
# THE MARGINAL-VALUE BASELINE is `DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel` with its
# market pillars switched off — the same goals likelihood, the same dynamics, no APM.

# ==========================================
# 1. THE MODEL CONFIGURATION
# ==========================================
Base.@kwdef struct DynamicGoalsPlusMinusLeagueTimeDecayModel{
    I<:AbstractInterceptionConfig,
    T<:AbstractDynamicsConfig,
    H<:AbstractHomeAdvantageConfig,
    # Any rating family that emits the `flat_<side>_<pos>_rating` vectors + `:player_ratings_map`,
    # i.e. `AbstractPlusMinusFeature` (RAPM) OR `PlayerRatingsFeature` (SofaScore, via a tracker).
    # The engine body never inspects the family — it reads the flat vectors and centres with
    # `Features.rating_base`, which dispatches per family. Widened 2026-07-29 for the ScottishUpper
    # (54/55) stream, where SofaScore ratings exist and RAPM's live-text source starts only at 23/24.
    # ⚠ A non-zero-base family needs a TIGHTER `w_*_prior` than the RAPM defaults — see below.
    P<:Features.AbstractFeatureConfig
    } <: AbstractTimeDecayPlayerModel
      interception_config::I   = HierarchicalMonthlyInterception()
      dynamics_config::T       = TimeDecayDynamics()
      homeadvantage_config::H  = HierarchicalTeamHomeAdvantage()
      player_ratings_feature::P = Features.ShotsPlusMinusFeature()
      # Scale note: a single player's RAPM has sd ~ 0.10 on `y_shots`, so a 10-man outfield sum sits
      # at O(0.3-0.5). Centring both weights on ZERO is the honest prior — the research's own
      # verdict was "real but small signal", so the engine must be free to conclude the pillar is
      # worth nothing rather than being pushed into using it.
      #
      # ⚠ THESE DEFAULTS ARE CALIBRATED TO RAPM's SCALE. A minute-weighted SofaScore rating sums to
      # ~10 × the mean rating over the outfield, so its CENTRED total has sd of order 1-3 — an order
      # of magnitude larger than RAPM's. At sd 0.3 the pillar would swing log-λ by ±0.6 or more.
      # Pass an explicit tighter prior (~Normal(0, 0.05)) when slotting in `PlayerRatingsFeature`,
      # and check the realised centred-rating sd before trusting the fit.
      w_att_prior::Distribution = Normal(0.0, 0.3)
      w_def_prior::Distribution = Normal(0.0, 0.3)
      league_offset_sd::Float64 = 0.1   # zero-sum δ_league prior sd (gap ≈ 0.047 on 56/57)
      league_ha_sd::Float64     = 0.1
      league_ha_on::Bool        = false # optional per-league HA offset (gated, branch-free)
end

# ==========================================
# 2. THE TURING ENGINE
# ==========================================
@model function build_goals_plus_minus_league_engine(
    home_team_indices::Vector{Int}, away_team_indices::Vector{Int},
    season_indices::Vector{Int}, month_indices::Vector{Int}, league_indices::Vector{Int},
    rat_h::Vector{Float64}, rat_a::Vector{Float64},
    suff_h::NamedTuple, suff_a::NamedTuple,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64,
    config::DynamicGoalsPlusMinusLeagueTimeDecayModel
)
    # --- 1. COMPONENTS ---
    inter ~ to_submodel(build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(build_dynamics(config.dynamics_config, n_teams))

    # --- plus-minus pillar weights (global scalars) ---
    w_att ~ config.w_att_prior
    w_def ~ config.w_def_prior

    # --- league offsets (zero-sum; HA offset gated branch-free) ---
    δ_league_raw ~ filldist(Normal(0.0, config.league_offset_sd), n_leagues)
    γ_league_raw ~ filldist(Normal(0.0, config.league_ha_sd), n_leagues)
    δ_league = δ_league_raw .- mean(δ_league_raw)
    γ_league = league_ha_active .* (γ_league_raw .- mean(γ_league_raw))

    # --- 2. RATES ---
    int_m = view(inter.μ_base, season_indices) .+ view(inter.δ_month, month_indices)
    lg    = view(δ_league, league_indices)
    γ_lg  = view(γ_league, league_indices)

    # A better attack raises your own rate; a better opponent (as a defensive unit) lowers it.
    pillar_h = w_att .* rat_h .- w_def .* rat_a
    pillar_a = w_att .* rat_a .- w_def .* rat_h

    log_λ_h = clamp.(int_m .+ lg .+ view(ha, home_team_indices) .+ γ_lg .+
                     view(dyn.α, home_team_indices) .+ view(dyn.β, away_team_indices) .+
                     pillar_h, -10.0, 10.0)
    log_λ_a = clamp.(int_m .+ lg .+
                     view(dyn.α, away_team_indices) .+ view(dyn.β, home_team_indices) .+
                     pillar_a, -10.0, 10.0)
    # No 1e-6 floor: the clamp already bounds λ to [4.5e-5, 2.2e4], and log_λ is used DIRECTLY as
    # the log-rate below, so rate and log-rate must stay exactly consistent.
    λ_h = exp.(log_λ_h)
    λ_a = exp.(log_λ_a)

    # AD-Safe Rejection
    is_bad = any(isnan, λ_h) || any(isnan, λ_a) || any(isinf, λ_h) || any(isinf, λ_a)
    λ_h = ifelse.(isnan.(λ_h) .| isinf.(λ_h), one.(λ_h), λ_h)
    λ_a = ifelse.(isnan.(λ_a) .| isinf.(λ_a), one.(λ_a), λ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    # --- 3. LIKELIHOOD via sufficient statistics ---
    # Counts and decay weights are DATA, so the weighted Poisson log-likelihood collapses onto two
    # constant vectors per side:
    #   Σ w·logPois(goals | λ) = Σ(w·goals)·log λ − Σ(w)·λ                              [+ const]
    # The dropped constant (log y!) is parameter-free, so the posterior is EXACTLY unchanged — only
    # the reported `lp` shifts by a fixed amount (never compare `lp` across engines).
    ll_h = sum(suff_h.c_lin .* log_λ_h) - sum(suff_h.c_rate .* λ_h)
    ll_a = sum(suff_a.c_lin .* log_λ_a) - sum(suff_a.c_rate .* λ_a)
    Turing.@addlogprob! ll_h + ll_a
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function Features.required_features(model::DynamicGoalsPlusMinusLeagueTimeDecayModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(),
        Features.GoalsFeature(),
        Features.DatesFeature(),
        Features.MonthFeature(),
        Features.LeagueFeature(),
        model.player_ratings_feature,
        Features.TimeIndicesFeature(),
    ]
end

"""Per-side Poisson sufficient statistics: `c_lin` weights `log λ`, `c_rate` weights `λ`."""
_pm_goals_suff(goals::Vector{Int}, w::Vector{Float64}) = (c_lin = w .* goals, c_rate = w)

"""
Outfield-collapsed, centred rating for one side: `(D + M + F) − 10·base`, masked to 0 where the side
has no rated minutes.

⚠ The mask is NOT cosmetic. The extractors emit 0.0 for a side with no usable ratings (no lineups, or
a season the rating provider never covered), and 0.0 is indistinguishable from a genuine zero total.
Without the mask that "missing" side is centred to `−10·base`, i.e. a side rated ten full standard
players BELOW league average — for SofaScore ratings (`base = 6.5`) that is −65 on the pillar input,
which the weight prior then multiplies into a nonsense log-rate. Masking maps "no data" to "league
average", which is the honest imputation.

For the plus-minus family `base = 0`, so masked and unmasked agree exactly and RAPM behaviour is
unchanged.
"""
function _pm_outfield(D, M, F, base::Float64)
    tot = D .+ M .+ F
    # Weighted ratings are sums of non-negative minute-weighted ratings, so an exact 0 total means
    # "no rated minutes on this side", never a real rating.
    return ifelse.(tot .> 0.0, tot .- (10.0 * base), zero(eltype(tot)))
end

function build_turing_model(config::DynamicGoalsPlusMinusLeagueTimeDecayModel,
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

    base = Features.rating_base(config.player_ratings_feature)
    rat_h = _pm_outfield(Vector{Float64}(data[:flat_home_D_rating]),
                         Vector{Float64}(data[:flat_home_M_rating]),
                         Vector{Float64}(data[:flat_home_F_rating]), base)
    rat_a = _pm_outfield(Vector{Float64}(data[:flat_away_D_rating]),
                         Vector{Float64}(data[:flat_away_M_rating]),
                         Vector{Float64}(data[:flat_away_F_rating]), base)

    return build_goals_plus_minus_league_engine(
        home_ids, away_ids, season_ids, month_idx, league_idx,
        rat_h, rat_a,
        _pm_goals_suff(home_goals, match_weights),
        _pm_goals_suff(away_goals, match_weights),
        n_teams, n_seasons, n_months, n_leagues,
        config.league_ha_on ? 1.0 : 0.0,
        config
    )
end

# ==========================================
# 4. THE EXTRACTOR
# ==========================================
function extract_parameters(
    model::DynamicGoalsPlusMinusLeagueTimeDecayModel,
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

    # zero-sum league offsets, reconstructed exactly as in the @model
    δ_raw = zeros(n_samples, n_leagues)
    γ_raw = zeros(n_samples, n_leagues)
    for i in 1:n_leagues
        δ_raw[:, i] = vec(Array(chain[Symbol("δ_league_raw[$i]")]))
        model.league_ha_on && (γ_raw[:, i] = vec(Array(chain[Symbol("γ_league_raw[$i]")])))
    end
    δ_mat = δ_raw .- mean(δ_raw, dims=2)
    γ_mat = model.league_ha_on ? (γ_raw .- mean(γ_raw, dims=2)) : zeros(n_samples, n_leagues)

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

        # Ratings for THIS match — the extractor builds the map over the WHOLE store precisely so
        # out-of-sample rows (which are never in the fold) resolve here exactly as in-sample ones
        # do. A match absent from the map (no lineups) falls back to 0.0, a league-average side.
        # Masked exactly as the builder's `_pm_outfield`: a side with no rated minutes contributes 0
        # (league average), NOT −10·base. See the `_pm_outfield` docstring.
        m_r = get(ratings_map, mid, Dict{Tuple{String, String}, Float64}())
        _side(sd) = get(m_r, (sd, "D"), 0.0) + get(m_r, (sd, "M"), 0.0) + get(m_r, (sd, "F"), 0.0)
        t_h, t_a = _side("home"), _side("away")
        r_h = t_h > 0.0 ? t_h - 10.0 * base : 0.0
        r_a = t_a > 0.0 ? t_a - 10.0 * base : 0.0

        pillar_h = w_att .* r_h .- w_def .* r_a
        pillar_a = w_att .* r_a .- w_def .* r_h

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        m_idx = month(row.match_date)
        int_v = inter_nt.μ_base[:, s_idx] .+ inter_nt.δ_month[:, m_idx]

        log_λ_h = clamp.(int_v .+ lg .+ γ_h .+ γlg .+ α_h .+ β_a .+ pillar_h, -10.0, 10.0)
        log_λ_a = clamp.(int_v .+ lg .+               α_a .+ β_h .+ pillar_a, -10.0, 10.0)
        λ_h = exp.(log_λ_h)
        λ_a = exp.(log_λ_a)

        results[mid] = (;
            λ_h, λ_a,
            pillar_h, pillar_a,
            true_xg_h = λ_h, true_xg_a = λ_a,
        )
    end
    return results
end
