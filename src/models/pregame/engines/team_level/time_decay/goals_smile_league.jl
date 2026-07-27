# src/models/pregame/engines/team_level/time_decay/goals_smile_league.jl
#
# TEAM-LEVEL, GOALS-ONLY smile engine {goals + SUPREMACY + per-strike SMILE + league offset}.
#
# Graduated from current_development/scottish_lower_smile/l01_team_dp_league.jl
# (TeamSmileDPGoalsModel, `smile_pois`) — the data-poor-league sibling of
# DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel: no xG, no player ratings, no stats;
# team attack/defence come from TimeDecayDynamics (static, zero-sum, exponentially decayed
# likelihood weights). Built for pooled multi-division segments (e.g. ScottishLower [56, 57]):
# a zero-sum per-league intercept offset δ_league (and an optional gated per-league home-advantage
# offset) rides on the shared team map, so ratings survive promotion/relegation between divisions.
#
# Market pillars (both σ SAMPLED — the release valve; never fix σ):
#   C1 SUPREMACY: model_sup = log λ_h − log λ_a ~ Normal(m_sup, σ_sup)   (no κ at team level)
#   C2 SMILE:     log Λ^model(K) = log(λ_h+λ_a) + log_φ(K) ~ Normal(log Λ^mkt(K), σ_smile)
# φ is a PRICING object only — it never enters the goals likelihood. Per-line O/U is priced by
# SmileScoreMatrix via cdf(Poisson(λ_tot·φ(K)), K); 1X2/BTTS/CS come from the (λ_h,λ_a) grid.
# Market inversion stays Poisson-referenced (see market_extractors.jl — that gap IS the edge).
#
# ⚠ DEFAULTS: supremacy_weight / smile_weight / days_half_life below are the Ireland keeper
# starting values — UPDATE to the r04/r05 Scottish grid winner before production use.

# ==========================================
# 1. THE MODEL CONFIGURATION
# ==========================================
Base.@kwdef struct DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel{
    I<:AbstractInterceptionConfig,
    T<:AbstractDynamicsConfig,
    H<:AbstractHomeAdvantageConfig,
    M<:Features.AbstractMarketFeatureConfig
    } <: AbstractTimeDecayTeamModel
      interception_config::I  = HierarchicalMonthlyInterception()
      dynamics_config::T      = TimeDecayDynamics()        # ⚠ set days_half_life to grid winner
      homeadvantage_config::H = HierarchicalTeamHomeAdvantage()
      market_feature_config::M = Features.DoublePoissonMarketFeature()
      smile_feature::Features.MarketSmileFeature = Features.MarketSmileFeature(Kmax=4)
      σ_supremacy_prior::Distribution = truncated(Normal(0.10, 0.10), lower=0.02)  # SAMPLED
      σ_smile_prior::Distribution     = truncated(Normal(0.15, 0.10), lower=0.02)  # SAMPLED
      smile_shape_sd::Float64         = 0.5
      market_on::Bool                 = true
      supremacy_weight::Float64       = 1.0   # ⚠ update to r05 winner
      smile_weight::Float64           = 0.5   # ⚠ update to r05 winner
      league_offset_sd::Float64       = 0.1   # zero-sum δ_league prior sd (gap ≈ 0.047 on 56/57)
      league_ha_sd::Float64           = 0.1
      league_ha_on::Bool              = false # optional per-league HA offset (gated, branch-free)
end

# ==========================================
# 2. THE TURING ENGINE
# ==========================================
@model function build_smile_double_poisson_goals_league_engine(
    home_team_indices::Vector{Int}, away_team_indices::Vector{Int},
    season_indices::Vector{Int}, month_indices::Vector{Int}, league_indices::Vector{Int},
    home_goals::Vector{Int}, away_goals::Vector{Int},
    match_weights::Vector{Float64},
    market_log_λ_h::Vector{Float64}, market_log_λ_a::Vector{Float64}, market_mask::Vector{Float64},
    smile_logΛ::Matrix{Float64}, smile_mask::Matrix{Float64}, n_strikes::Int,
    market_active::Float64, supremacy_weight::Float64, smile_weight::Float64,
    smile_shape_sd::Float64,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64,
    config::DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel
)
    # --- 1. COMPONENTS ---
    inter ~ to_submodel(build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(build_dynamics(config.dynamics_config, n_teams))

    σ_sup   ~ config.σ_supremacy_prior
    σ_smile ~ config.σ_smile_prior
    log_φ   ~ filldist(Normal(0.0, smile_shape_sd), n_strikes)   # global smile shape

    # --- league offsets (zero-sum; HA offset gated branch-free) ---
    δ_league_raw ~ filldist(Normal(0.0, config.league_offset_sd), n_leagues)
    γ_league_raw ~ filldist(Normal(0.0, config.league_ha_sd), n_leagues)
    δ_league = δ_league_raw .- mean(δ_league_raw)
    γ_league = league_ha_active .* (γ_league_raw .- mean(γ_league_raw))

    # --- 2. RATES ---
    int_m = view(inter.μ_base, season_indices) .+ view(inter.δ_month, month_indices)
    lg    = view(δ_league, league_indices)
    γ_lg  = view(γ_league, league_indices)

    log_λ_h = clamp.(int_m .+ lg .+ view(ha, home_team_indices) .+ γ_lg .+
                     view(dyn.α, home_team_indices) .+ view(dyn.β, away_team_indices), -10.0, 10.0)
    log_λ_a = clamp.(int_m .+ lg .+
                     view(dyn.α, away_team_indices) .+ view(dyn.β, home_team_indices), -10.0, 10.0)
    λ_h = exp.(log_λ_h) .+ 1e-6
    λ_a = exp.(log_λ_a) .+ 1e-6

    # AD-Safe Rejection
    is_bad = any(isnan, λ_h) || any(isnan, λ_a) || any(isinf, λ_h) || any(isinf, λ_a)
    λ_h = ifelse.(isnan.(λ_h) .| isinf.(λ_h), one.(λ_h), λ_h)
    λ_a = ifelse.(isnan.(λ_a) .| isinf.(λ_a), one.(λ_a), λ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    # --- 3. GOALS (double Poisson, time-decayed) ---
    ll_goals_h = logpdf.(Poisson.(λ_h), home_goals)
    ll_goals_a = logpdf.(Poisson.(λ_a), away_goals)
    Turing.@addlogprob! sum((ll_goals_h .+ ll_goals_a) .* match_weights)

    # --- Pillar C1: SUPREMACY (who wins) ---
    model_sup = log_λ_h .- log_λ_a
    m_sup     = market_log_λ_h .- market_log_λ_a
    ll_sup    = logpdf.(Normal.(model_sup, σ_sup), m_sup)
    Turing.@addlogprob! market_active * supremacy_weight *
        sum(ll_sup .* match_weights .* market_mask)

    # --- Pillar C2: LOCAL-INTENSITY SMILE (per-strike totals; pricing-only φ) ---
    log_λ_tot  = log.(λ_h .+ λ_a)
    model_logΛ = log_λ_tot .+ reshape(log_φ, 1, n_strikes)          # [n_matches × nK]
    ll_smile   = logpdf.(Normal.(model_logΛ, σ_smile), smile_logΛ)
    Turing.@addlogprob! market_active * smile_weight *
        sum(ll_smile .* smile_mask .* match_weights)
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function Features.required_features(model::DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(),
        Features.GoalsFeature(),
        Features.DatesFeature(),
        Features.MonthFeature(),
        Features.LeagueFeature(),
        model.market_feature_config,
        model.smile_feature,
        Features.TimeIndicesFeature(),
    ]
end

function build_turing_model(config::DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel, feature_set::FeatureSet)
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

    # Market pillar: trust only matches where BOTH implied rates are present and in a plausible
    # football range (thin closes can invert to degenerate λ).
    _mok(x) = !ismissing(x) && (xf = Float64(x); !isnan(xf) && 0.02 < xf < 20.0)
    market_mask  = Float64.(_mok.(data[:flat_market_λ_home]) .& _mok.(data[:flat_market_λ_away]))
    market_log_h = [_mok(x) ? log(Float64(x)) : 0.0 for x in data[:flat_market_λ_home]]
    market_log_a = [_mok(x) ? log(Float64(x)) : 0.0 for x in data[:flat_market_λ_away]]

    smile_logΛ = Matrix{Float64}(data[:flat_smile_logΛ])
    smile_mask = Matrix{Float64}(data[:flat_smile_mask])

    return build_smile_double_poisson_goals_league_engine(
        home_ids, away_ids, season_ids, month_idx, league_idx,
        home_goals, away_goals, match_weights,
        market_log_h, market_log_a, market_mask,
        smile_logΛ, smile_mask, size(smile_logΛ, 2),
        config.market_on ? 1.0 : 0.0, config.supremacy_weight, config.smile_weight,
        config.smile_shape_sd,
        n_teams, n_seasons, n_months, n_leagues,
        config.league_ha_on ? 1.0 : 0.0,
        config
    )
end

# ==========================================
# 4. THE EXTRACTOR
# ==========================================
function extract_parameters(
    model::DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel,
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
    nK = Int(data[:smile_Kmax]) + 1

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

    # global smile shape φ(K): [n_samples × nK]
    φ_mat = Matrix{Float64}(undef, n_samples, nK)
    for k in 1:nK
        φ_mat[:, k] = exp.(vec(Array(chain[Symbol("log_φ[$k]")])))
    end

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
        int_v = inter_nt.μ_base[:, s_idx] .+ inter_nt.δ_month[:, m_idx]

        log_λ_h = clamp.(int_v .+ lg .+ γ_h .+ γlg .+ α_h .+ β_a, -10.0, 10.0)
        log_λ_a = clamp.(int_v .+ lg .+         α_a .+ β_h, -10.0, 10.0)
        λ_h = exp.(log_λ_h) .+ 1e-6
        λ_a = exp.(log_λ_a) .+ 1e-6

        results[mid] = (;
            λ_h, λ_a,
            λ_tot = λ_h .+ λ_a,
            φ = φ_mat,
            true_xg_h = λ_h, true_xg_a = λ_a,
        )
    end
    return results
end
