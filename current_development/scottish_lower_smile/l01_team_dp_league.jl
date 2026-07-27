# current_development/scottish_lower_smile/l01_team_dp_league.jl
#
# LOADER (temporary module). Team-level, goals-only, time-decay engines for data-poor leagues
# (ScottishLower: no xG / no ratings / no stats), pooled over two divisions with a zero-sum
# per-league intercept offset. Three engines (canonical names, NOTES.md):
#
#   TeamDPGoalsModel       — `none_pois`  : double-Poisson goals, structural only (fast reference)
#   TeamIsoDPGoalsModel    — `iso_pois`   : + isotropic market pillar (sampled σ_market, knob mw)
#   TeamSmileDPGoalsModel  — `smile_pois` : + SUPREMACY anchor + per-strike SMILE pillar
#                                            (knobs supremacy_weight / smile_weight)
#
# Skeletons: src/models/pregame/engines/team_level/time_decay/goals.jl + goals_market.jl
# (component submodels, exponential match weights, sampled-σ release valve, AD-safe rejection).
# Pillars: src/.../player_level/time_decay/outfield_xg_smile_double_poisson.jl — at team level
# the supremacy is simply log λ_h − log λ_a (no κ), and the smile anchors log(λ_h+λ_a)+log_φ(K)
# to the market-inverted Λ^mkt(K). φ is a PRICING object only (never in the goals likelihood).
#
# LEAGUE OFFSET: δ_league_raw ~ N(0, league_offset_sd) per league, zero-sum centred, added to
# BOTH log-rates. True 56-vs-57 gap ≈ log(2.87/2.71) ≈ 0.057 ⇒ N(0, 0.1) covers it. Optional
# per-league home-advantage offset gated by league_ha_on (0/1 scalar, branch-free/AD-safe).
#
# Dispatch ([[dixoncoles-prediction-dispatch-union]]): these structs subtype
# AbstractTimeDecayTeamModel <: AbstractNegBinModel but have no r columns, so explicit
# extract_params / compute_score_matrix overrides ship below (Poisson grid; smile returns the
# src Pred.SmileScoreMatrix so its per-line O/U pricing is reused for free).
#
# Also fixes here (src fix at graduation): required_features of the src NB market engine
# references the PHANTOM Features.MarketLambdaFeature — overridden to DoublePoissonMarketFeature.

using Turing
using Distributions
using DataFrames
using Dates

const PreGame  = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Pred     = BayesianFootball.Predictions
const Data     = BayesianFootball.Data

# Ireland-validated sampled-σ priors (sweep the WEIGHTS, not these).
const SUP_PRIOR    = truncated(Normal(0.10, 0.10), lower=0.02)
const SMILE_PRIOR  = truncated(Normal(0.15, 0.10), lower=0.02)
const MARKET_PRIOR = truncated(Normal(0.10, 0.20), lower=0.01)

# ==========================================
# 0. LEAGUE FEATURE — GRADUATED to src (Stage 4): Features.LeagueFeature + its extractor in
# src/features/extractors/core_extractors.jl (emits :flat_league_ids, :n_leagues, :league_lookup).
# ==========================================
const LeagueFeature = Features.LeagueFeature

# ==========================================
# 0b. SHARED BUILDER HELPERS
# ==========================================
_market_active(config) = config.market_on ? 1.0 : 0.0

# Plausibility mask: the market inversion can return degenerate λ on thin closes
# ([[outfield-xg-engine-gotchas]] — λ ~357 observed); trust only football-plausible rates.
_mok(x) = !ismissing(x) && (xf = Float64(x); !isnan(xf) && 0.02 < xf < 20.0)

function _unpack_market(data)
    market_mask  = Float64.(_mok.(data[:flat_market_λ_home]) .& _mok.(data[:flat_market_λ_away]))
    market_log_h = [_mok(x) ? log(Float64(x)) : 0.0 for x in data[:flat_market_λ_home]]
    market_log_a = [_mok(x) ? log(Float64(x)) : 0.0 for x in data[:flat_market_λ_away]]
    return market_log_h, market_log_a, market_mask
end

function _unpack_core(data, config)
    date_deltas   = Vector{Int}(data[:dates])
    return (;
        home_ids      = Vector{Int}(data[:flat_home_ids]),
        away_ids      = Vector{Int}(data[:flat_away_ids]),
        season_idx    = Vector{Int}(data[:season_indices]),
        month_idx     = Vector{Int}(data[:flat_months]),
        league_idx    = Vector{Int}(data[:flat_league_ids]),
        home_goals    = Vector{Int}(data[:flat_home_goals]),
        away_goals    = Vector{Int}(data[:flat_away_goals]),
        match_weights = 0.5 .^ (date_deltas ./ config.dynamics_config.days_half_life),
        n_teams       = Int(data[:n_teams]),
        n_seasons     = Int(data[:n_seasons]),
        n_months      = 12,
        n_leagues     = Int(data[:n_leagues]),
    )
end

# ==========================================
# 1. none_pois — TeamDPGoalsModel
# ==========================================
Base.@kwdef struct TeamDPGoalsModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig
    } <: PreGame.AbstractTimeDecayTeamModel
      interception_config::I  = PreGame.HierarchicalMonthlyInterception()
      dynamics_config::T      = PreGame.TimeDecayDynamics()
      homeadvantage_config::H = PreGame.HierarchicalTeamHomeAdvantage()
      league_offset_sd::Float64 = 0.1
      league_ha_sd::Float64     = 0.1
      league_ha_on::Bool        = false
end

@model function build_team_dp_goals_league_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    home_goals::Vector{Int}, away_goals::Vector{Int},
    match_weights::Vector{Float64},
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64,
    config
)
    # --- components (identical to the src time-decay team engines) ---
    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(PreGame.build_dynamics(config.dynamics_config, n_teams))

    # --- league offsets (zero-sum; HA offset gated branch-free) ---
    δ_league_raw ~ filldist(Normal(0.0, config.league_offset_sd), n_leagues)
    γ_league_raw ~ filldist(Normal(0.0, config.league_ha_sd), n_leagues)
    δ_league = δ_league_raw .- mean(δ_league_raw)
    γ_league = league_ha_active .* (γ_league_raw .- mean(γ_league_raw))

    int_m = view(inter.μ_base, season_idx) .+ view(inter.δ_month, month_idx)
    lg    = view(δ_league, league_idx)
    γ_lg  = view(γ_league, league_idx)

    log_λ_h = clamp.(int_m .+ lg .+ view(ha, home_ids) .+ γ_lg .+
                     view(dyn.α, home_ids) .+ view(dyn.β, away_ids), -10.0, 10.0)
    log_λ_a = clamp.(int_m .+ lg .+
                     view(dyn.α, away_ids) .+ view(dyn.β, home_ids), -10.0, 10.0)
    λ_h = exp.(log_λ_h) .+ 1e-6
    λ_a = exp.(log_λ_a) .+ 1e-6

    # AD-safe rejection
    is_bad = any(isnan, λ_h) || any(isnan, λ_a) || any(isinf, λ_h) || any(isinf, λ_a)
    λ_h = ifelse.(isnan.(λ_h) .| isinf.(λ_h), one.(λ_h), λ_h)
    λ_a = ifelse.(isnan.(λ_a) .| isinf.(λ_a), one.(λ_a), λ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    # time-decayed double-Poisson goals likelihood
    ll_h = logpdf.(Poisson.(λ_h), home_goals)
    ll_a = logpdf.(Poisson.(λ_a), away_goals)
    Turing.@addlogprob! sum((ll_h .+ ll_a) .* match_weights)
end

function Features.required_features(model::TeamDPGoalsModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), LeagueFeature(), Features.TimeIndicesFeature(),
    ]
end

function PreGame.build_turing_model(config::TeamDPGoalsModel, feature_set)
    d = _unpack_core(feature_set.data, config)
    return build_team_dp_goals_league_engine(
        d.home_ids, d.away_ids, d.season_idx, d.month_idx, d.league_idx,
        d.home_goals, d.away_goals, d.match_weights,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        config.league_ha_on ? 1.0 : 0.0, config)
end

# ==========================================
# 2. iso_pois — TeamIsoDPGoalsModel
# ==========================================
Base.@kwdef struct TeamIsoDPGoalsModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    M<:Features.AbstractMarketFeatureConfig
    } <: PreGame.AbstractTimeDecayTeamModel
      interception_config::I  = PreGame.HierarchicalMonthlyInterception()
      dynamics_config::T      = PreGame.TimeDecayDynamics()
      homeadvantage_config::H = PreGame.HierarchicalTeamHomeAdvantage()
      market_feature_config::M = Features.DoublePoissonMarketFeature()
      market_σ::Distribution   = MARKET_PRIOR      # SAMPLED (release valve — never fix)
      market_weight::Float64   = 1.0
      market_on::Bool          = true
      league_offset_sd::Float64 = 0.1
      league_ha_sd::Float64     = 0.1
      league_ha_on::Bool        = false
end

@model function build_team_iso_dp_goals_league_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    home_goals::Vector{Int}, away_goals::Vector{Int},
    match_weights::Vector{Float64},
    market_log_h::Vector{Float64}, market_log_a::Vector{Float64}, market_mask::Vector{Float64},
    market_active::Float64,
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

    int_m = view(inter.μ_base, season_idx) .+ view(inter.δ_month, month_idx)
    lg    = view(δ_league, league_idx)
    γ_lg  = view(γ_league, league_idx)

    log_λ_h = clamp.(int_m .+ lg .+ view(ha, home_ids) .+ γ_lg .+
                     view(dyn.α, home_ids) .+ view(dyn.β, away_ids), -10.0, 10.0)
    log_λ_a = clamp.(int_m .+ lg .+
                     view(dyn.α, away_ids) .+ view(dyn.β, home_ids), -10.0, 10.0)
    λ_h = exp.(log_λ_h) .+ 1e-6
    λ_a = exp.(log_λ_a) .+ 1e-6

    is_bad = any(isnan, λ_h) || any(isnan, λ_a) || any(isinf, λ_h) || any(isinf, λ_a)
    λ_h = ifelse.(isnan.(λ_h) .| isinf.(λ_h), one.(λ_h), λ_h)
    λ_a = ifelse.(isnan.(λ_a) .| isinf.(λ_a), one.(λ_a), λ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    ll_h = logpdf.(Poisson.(λ_h), home_goals)
    ll_a = logpdf.(Poisson.(λ_a), away_goals)
    Turing.@addlogprob! sum((ll_h .+ ll_a) .* match_weights)

    # isotropic market pillar (h/a log-rates anchored independently, shared sampled σ)
    ll_mkt = logpdf.(Normal.(log_λ_h, σ_market), market_log_h) .+
             logpdf.(Normal.(log_λ_a, σ_market), market_log_a)
    Turing.@addlogprob! market_active * config.market_weight *
        sum(ll_mkt .* match_weights .* market_mask)
end

function Features.required_features(model::TeamIsoDPGoalsModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), LeagueFeature(), model.market_feature_config,
        Features.TimeIndicesFeature(),
    ]
end

function PreGame.build_turing_model(config::TeamIsoDPGoalsModel, feature_set)
    d = _unpack_core(feature_set.data, config)
    mh, ma, mm = _unpack_market(feature_set.data)
    return build_team_iso_dp_goals_league_engine(
        d.home_ids, d.away_ids, d.season_idx, d.month_idx, d.league_idx,
        d.home_goals, d.away_goals, d.match_weights,
        mh, ma, mm, _market_active(config),
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        config.league_ha_on ? 1.0 : 0.0, config)
end

# ==========================================
# 3. smile_pois — TeamSmileDPGoalsModel
# ==========================================
Base.@kwdef struct TeamSmileDPGoalsModel{
    I<:PreGame.AbstractInterceptionConfig,
    T<:PreGame.AbstractDynamicsConfig,
    H<:PreGame.AbstractHomeAdvantageConfig,
    M<:Features.AbstractMarketFeatureConfig
    } <: PreGame.AbstractTimeDecayTeamModel
      interception_config::I  = PreGame.HierarchicalMonthlyInterception()
      dynamics_config::T      = PreGame.TimeDecayDynamics()
      homeadvantage_config::H = PreGame.HierarchicalTeamHomeAdvantage()
      market_feature_config::M = Features.DoublePoissonMarketFeature()
      smile_feature::Features.MarketSmileFeature = Features.MarketSmileFeature(Kmax=4)
      σ_supremacy_prior::Distribution = SUP_PRIOR      # SAMPLED
      σ_smile_prior::Distribution     = SMILE_PRIOR    # SAMPLED
      smile_shape_sd::Float64         = 0.5
      market_on::Bool                 = true
      supremacy_weight::Float64       = 1.0
      smile_weight::Float64           = 0.5            # Ireland keeper starting point
      league_offset_sd::Float64 = 0.1
      league_ha_sd::Float64     = 0.1
      league_ha_on::Bool        = false
end

@model function build_team_smile_dp_goals_league_engine(
    home_ids::Vector{Int}, away_ids::Vector{Int},
    season_idx::Vector{Int}, month_idx::Vector{Int}, league_idx::Vector{Int},
    home_goals::Vector{Int}, away_goals::Vector{Int},
    match_weights::Vector{Float64},
    market_log_h::Vector{Float64}, market_log_a::Vector{Float64}, market_mask::Vector{Float64},
    smile_logΛ::Matrix{Float64}, smile_mask::Matrix{Float64}, n_strikes::Int,
    market_active::Float64, supremacy_weight::Float64, smile_weight::Float64,
    smile_shape_sd::Float64,
    n_teams::Int, n_seasons::Int, n_months::Int, n_leagues::Int,
    league_ha_active::Float64,
    config
)
    inter ~ to_submodel(PreGame.build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(PreGame.build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(PreGame.build_dynamics(config.dynamics_config, n_teams))

    σ_sup   ~ config.σ_supremacy_prior
    σ_smile ~ config.σ_smile_prior
    log_φ   ~ filldist(Normal(0.0, smile_shape_sd), n_strikes)   # global smile shape

    δ_league_raw ~ filldist(Normal(0.0, config.league_offset_sd), n_leagues)
    γ_league_raw ~ filldist(Normal(0.0, config.league_ha_sd), n_leagues)
    δ_league = δ_league_raw .- mean(δ_league_raw)
    γ_league = league_ha_active .* (γ_league_raw .- mean(γ_league_raw))

    int_m = view(inter.μ_base, season_idx) .+ view(inter.δ_month, month_idx)
    lg    = view(δ_league, league_idx)
    γ_lg  = view(γ_league, league_idx)

    log_λ_h = clamp.(int_m .+ lg .+ view(ha, home_ids) .+ γ_lg .+
                     view(dyn.α, home_ids) .+ view(dyn.β, away_ids), -10.0, 10.0)
    log_λ_a = clamp.(int_m .+ lg .+
                     view(dyn.α, away_ids) .+ view(dyn.β, home_ids), -10.0, 10.0)
    λ_h = exp.(log_λ_h) .+ 1e-6
    λ_a = exp.(log_λ_a) .+ 1e-6

    is_bad = any(isnan, λ_h) || any(isnan, λ_a) || any(isinf, λ_h) || any(isinf, λ_a)
    λ_h = ifelse.(isnan.(λ_h) .| isinf.(λ_h), one.(λ_h), λ_h)
    λ_a = ifelse.(isnan.(λ_a) .| isinf.(λ_a), one.(λ_a), λ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    # --- goals (double Poisson, decayed) ---
    ll_h = logpdf.(Poisson.(λ_h), home_goals)
    ll_a = logpdf.(Poisson.(λ_a), away_goals)
    Turing.@addlogprob! sum((ll_h .+ ll_a) .* match_weights)

    # --- Pillar C1: SUPREMACY (who wins) — team level: log λ_h − log λ_a, no κ ---
    model_sup = log_λ_h .- log_λ_a
    m_sup     = market_log_h .- market_log_a
    ll_sup    = logpdf.(Normal.(model_sup, σ_sup), m_sup)
    Turing.@addlogprob! market_active * supremacy_weight *
        sum(ll_sup .* match_weights .* market_mask)

    # --- Pillar C2: LOCAL-INTENSITY SMILE (per-strike totals; φ = pricing only) ---
    log_λ_tot  = log.(λ_h .+ λ_a)
    model_logΛ = log_λ_tot .+ reshape(log_φ, 1, n_strikes)          # [n_matches × nK]
    ll_smile   = logpdf.(Normal.(model_logΛ, σ_smile), smile_logΛ)
    Turing.@addlogprob! market_active * smile_weight *
        sum(ll_smile .* smile_mask .* match_weights)
end

function Features.required_features(model::TeamSmileDPGoalsModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), Features.GoalsFeature(), Features.DatesFeature(),
        Features.MonthFeature(), LeagueFeature(), model.market_feature_config,
        model.smile_feature, Features.TimeIndicesFeature(),
    ]
end

function PreGame.build_turing_model(config::TeamSmileDPGoalsModel, feature_set)
    data = feature_set.data
    d = _unpack_core(data, config)
    mh, ma, mm = _unpack_market(data)
    smile_logΛ = Matrix{Float64}(data[:flat_smile_logΛ])
    smile_mask = Matrix{Float64}(data[:flat_smile_mask])
    return build_team_smile_dp_goals_league_engine(
        d.home_ids, d.away_ids, d.season_idx, d.month_idx, d.league_idx,
        d.home_goals, d.away_goals, d.match_weights,
        mh, ma, mm,
        smile_logΛ, smile_mask, size(smile_logΛ, 2),
        _market_active(config), config.supremacy_weight, config.smile_weight,
        config.smile_shape_sd,
        d.n_teams, d.n_seasons, d.n_months, d.n_leagues,
        config.league_ha_on ? 1.0 : 0.0, config)
end

# ==========================================
# 4. EXTRACTORS (shared core; mirrors src goals_market.jl fixture loop + league offset)
# ==========================================
const TeamLeagueModels = Union{TeamDPGoalsModel, TeamIsoDPGoalsModel, TeamSmileDPGoalsModel}

# [n_samples × n_leagues], zero-sum centred per draw (exactly the @model math).
function _extract_league_offsets(chain, n_leagues::Int, sym_stem::String)
    n_samples = size(chain, 1) * size(chain, 3)
    raw = zeros(n_samples, n_leagues)
    for i in 1:n_leagues
        raw[:, i] = vec(Array(chain[Symbol("$(sym_stem)[$i]")]))
    end
    return raw .- mean(raw, dims=2)
end

function _extract_team_core(model::TeamLeagueModels, df, feature_set, chain)
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    n_leagues = Int(data[:n_leagues])
    team_map  = data[:team_map]
    league_lookup = data[:league_lookup]

    inter_nt = PreGame.extract_interception(chain, model.interception_config, n_seasons)
    ha_mat   = PreGame.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    dyn_nt   = PreGame.extract_dynamics(chain, model.dynamics_config, "dyn", n_teams)
    δ_mat    = _extract_league_offsets(chain, n_leagues, "δ_league_raw")
    γ_mat    = model.league_ha_on ? _extract_league_offsets(chain, n_leagues, "γ_league_raw") :
                                    zeros(size(δ_mat))

    n_samples = size(chain, 1) * size(chain, 3)
    core = Dict{Int, NamedTuple}()
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

        core[mid] = (; λ_h = exp.(log_λ_h) .+ 1e-6, λ_a = exp.(log_λ_a) .+ 1e-6)
    end
    return core, n_samples
end

function PreGame.extract_parameters(model::Union{TeamDPGoalsModel, TeamIsoDPGoalsModel},
                                    df, feature_set, chain)
    core, _ = _extract_team_core(model, df, feature_set, chain)
    results = Dict{Int, NamedTuple}()
    for (mid, c) in core
        results[mid] = (; λ_h = c.λ_h, λ_a = c.λ_a, true_xg_h = c.λ_h, true_xg_a = c.λ_a)
    end
    return results
end

function PreGame.extract_parameters(model::TeamSmileDPGoalsModel, df, feature_set, chain)
    core, n_samples = _extract_team_core(model, df, feature_set, chain)
    nK = Int(feature_set.data[:smile_Kmax]) + 1
    φ_mat = Matrix{Float64}(undef, n_samples, nK)
    for k in 1:nK
        φ_mat[:, k] = exp.(vec(Array(chain[Symbol("log_φ[$k]")])))
    end
    results = Dict{Int, NamedTuple}()
    for (mid, c) in core
        results[mid] = (; λ_h = c.λ_h, λ_a = c.λ_a,
                          λ_tot = c.λ_h .+ c.λ_a, φ = φ_mat,
                          true_xg_h = c.λ_h, true_xg_a = c.λ_a)
    end
    return results
end

# ==========================================
# 5. PREDICTION OVERRIDES (loader-local; Poisson grid + src SmileScoreMatrix reuse)
# ==========================================
function _poisson_score(λ_h, λ_a; max_goals::Int=12)
    n = length(λ_h)
    S = zeros(Float64, max_goals, max_goals, n)
    p_h = zeros(Float64, max_goals); p_a = zeros(Float64, max_goals)
    goals = 0:(max_goals-1)
    @inbounds for k in 1:n
        @. p_h = pdf(Poisson(λ_h[k]), goals)
        @. p_a = pdf(Poisson(λ_a[k]), goals)
        for j in 1:max_goals, i in 1:max_goals
            S[i, j, k] = p_h[i] * p_a[j]
        end
    end
    return Pred.ScoreMatrix(S)
end

Pred.extract_params(::Union{TeamDPGoalsModel, TeamIsoDPGoalsModel}, row) =
    (λ_h = row.λ_h, λ_a = row.λ_a)
Pred.compute_score_matrix(::Union{TeamDPGoalsModel, TeamIsoDPGoalsModel}, params; max_goals::Int=12) =
    _poisson_score(params.λ_h, params.λ_a; max_goals)

Pred.extract_params(::TeamSmileDPGoalsModel, row) =
    (λ_h = row.λ_h, λ_a = row.λ_a, λ_tot = row.λ_tot, φ = row.φ)
function Pred.compute_score_matrix(::TeamSmileDPGoalsModel, params; max_goals::Int=12)
    grid = _poisson_score(params.λ_h, params.λ_a; max_goals)
    Λ = transpose(params.λ_tot .* params.φ)          # -> [nK × n_samples]
    # src SmileScoreMatrix: its compute_market_probs(::SmileScoreMatrix, ::MarketOverUnder)
    # prices O/U per line via cdf(Poisson(Λ(K)), K) — reused for free.
    return Pred.SmileScoreMatrix(grid, Matrix{Float64}(Λ))
end

# ==========================================
# 6. (RESOLVED) phantom Features.MarketLambdaFeature
# ==========================================
# The phantom was fixed IN SRC at Stage 4 (all *Market* engines now request
# DoublePoissonMarketFeature; the dead export is gone) — no loader override needed anymore.

println("[l01] scottish team loader ready: TeamDPGoalsModel (none_pois) / TeamIsoDPGoalsModel " *
        "(iso_pois, knob market_weight) / TeamSmileDPGoalsModel (smile_pois, knobs supremacy_weight" *
        "/smile_weight) — pooled leagues via zero-sum δ_league (Features.LeagueFeature).")
