# src/models/pregame/engines/player_level/time_decay/outfield_xg_smile_double_poisson.jl
#
# LOCAL-INTENSITY SMILE double-Poisson engine {goals + xG + SUPREMACY + per-strike SMILE + outfield}.
#
# Graduated from current_development/split_market_pillar/l03_local_intensity_poisson.jl
# (validated keeper cell `li_smile50`; see RESULTS_smile_grid.md §5).
#
# It replaces the isotropic level market pillar with a per-strike LOCAL-INTENSITY SMILE pillar
# (Dupire / "local intensity"): a global per-strike shape log_φ ∈ ℝ^{Kmax+1} (φ(K)=exp(log_φ(K));
# φ≡1 ⇒ plain Poisson). Per match m, per present strike K it anchors
#     log Λ^model_m(K) = log(λ_h+λ_a)_m + log_φ(K)   ~ Normal(·, σ_smile) to the market-inverted
# per-strike intensity Λ^mkt_m(K) (built off the AD path by MarketSmileFeature). σ_smile is SAMPLED.
# The supremacy (who-wins, λ_h−λ_a) anchor is KEPT (helps 1X2). φ is a PRICING object — it does NOT
# enter the goals likelihood. Per-line O/U is priced with its own intensity
# P(N≤K)=cdf(Poisson(λ_tot·φ(K)),K) (see SmileScoreMatrix in predictions); 1X2/BTTS/correct-score come
# from the unchanged (λ_h,λ_a) grid.

# ==========================================
# 1. THE MODEL CONFIGURATION
# ==========================================
Base.@kwdef struct DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel{
    I<:AbstractInterceptionConfig,
    P<:OutfieldPlayerDynamicsConfig,
    D<:AbstractDispersionConfig,
    H<:AbstractHomeAdvantageConfig,
    K<:AbstractKappaConfig,
    R<:Features.AbstractFeatureConfig,
    M<:Features.AbstractMarketFeatureConfig
  } <: AbstractTimeDecayPlayerModel
      interception_config::I
      player_dynamics_config::P
      dispersion_config::D            # config-compat; unused by the Poisson likelihood
      homeadvantage_config::H
      kappa_config::K
      player_ratings_feature::R
      market_feature_config::M = Features.DoublePoissonMarketFeature()
      smile_feature::Features.MarketSmileFeature = Features.MarketSmileFeature(Kmax=4)
      ν_xg::Distribution               = truncated(Normal(3.0, 0.5), lower=0.5)
      σ_supremacy_prior::Distribution  = truncated(Normal(0.10, 0.10), lower=0.02)
      σ_smile_prior::Distribution      = truncated(Normal(0.15, 0.10), lower=0.02)
      smile_shape_sd::Float64          = 0.5   # prior sd on the global log_φ(K) shape
      market_on::Bool                  = true  # false => BOTH market pillars OFF (control)
      supremacy_weight::Float64        = 1.0
      smile_weight::Float64            = 0.5   # keeper cell (li_smile50)
end

# ==========================================
# 2. THE TURING ENGINE
# ==========================================
@model function build_double_poisson_smile_xg_market_player_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    month_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    match_weights::Vector{Float64},
    home_G_ratings::Vector{Float64}, home_D_ratings::Vector{Float64},
    home_M_ratings::Vector{Float64}, home_F_ratings::Vector{Float64},
    away_G_ratings::Vector{Float64}, away_D_ratings::Vector{Float64},
    away_M_ratings::Vector{Float64}, away_F_ratings::Vector{Float64},
    home_xg::Vector{Float64},
    away_xg::Vector{Float64},
    xg_mask::Vector{Float64},
    market_log_λ_h::Vector{Float64},
    market_log_λ_a::Vector{Float64},
    market_mask::Vector{Float64},
    smile_logΛ::Matrix{Float64},     # [n_matches × nK]
    smile_mask::Matrix{Float64},     # [n_matches × nK]
    n_strikes::Int,
    market_active::Float64,
    supremacy_weight::Float64,
    smile_weight::Float64,
    smile_shape_sd::Float64,
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel
)
    # --- 1. LOAD COMPONENTS ---
    ν_xg    ~ config.ν_xg
    σ_sup   ~ config.σ_supremacy_prior
    σ_smile ~ config.σ_smile_prior
    log_φ   ~ filldist(Normal(0.0, smile_shape_sd), n_strikes)   # global smile shape

    inter ~ to_submodel(build_interception(config.interception_config, n_seasons, n_months))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    kap   ~ to_submodel(build_kappa(config.kappa_config, n_teams))
    p_dyn ~ to_submodel(build_dynamics(config.player_dynamics_config, n_teams))

    # --- 2. VECTORIZED INDEXING & MATH ---
    base_rating = config.player_ratings_feature.tracker.prior_mean

    h_G_c = home_G_ratings .- base_rating
    h_O_c = (home_D_ratings .+ home_M_ratings .+ home_F_ratings) .- (10.0 * base_rating)
    a_G_c = away_G_ratings .- base_rating
    a_O_c = (away_D_ratings .+ away_M_ratings .+ away_F_ratings) .- (10.0 * base_rating)

    att_h = (p_dyn.w_G_att .* h_G_c) .+ (p_dyn.w_Outfield_att .* h_O_c)
    def_h = (p_dyn.w_G_def .* h_G_c) .+ (p_dyn.w_Outfield_def .* h_O_c)
    att_a = (p_dyn.w_G_att .* a_G_c) .+ (p_dyn.w_Outfield_att .* a_O_c)
    def_a = (p_dyn.w_G_def .* a_G_c) .+ (p_dyn.w_Outfield_def .* a_O_c)

    int_m = view(inter.μ_base, season_indices) .+ view(inter.δ_month, month_indices)
    log_λ_h = clamp.(int_m .+ view(ha, home_team_indices) .+ att_h .+ def_a, -20.0, 20.0)
    log_λ_a = clamp.(int_m                                .+ att_a .+ def_h, -20.0, 20.0)

    kap_h = view(kap, home_team_indices)
    kap_a = view(kap, away_team_indices)
    λ_h = kap_h .* exp.(log_λ_h) .+ 1e-6
    λ_a = kap_a .* exp.(log_λ_a) .+ 1e-6

    # AD-Safe Rejection
    is_bad = any(isnan, λ_h) || any(isnan, λ_a) || any(isinf, λ_h) || any(isinf, λ_a)
    λ_h = ifelse.(isnan.(λ_h) .| isinf.(λ_h), one.(λ_h), λ_h)
    λ_a = ifelse.(isnan.(λ_a) .| isinf.(λ_a), one.(λ_a), λ_a)
    Turing.@addlogprob! ifelse(is_bad, -Inf, 0.0)

    # --- Pillar B: Actual Goals (Poisson) ---
    ll_goals_h = logpdf.(Poisson.(λ_h), home_goals)
    ll_goals_a = logpdf.(Poisson.(λ_a), away_goals)
    Turing.@addlogprob! sum((ll_goals_h .+ ll_goals_a) .* match_weights)

    # --- Pillar A: xG (Gamma) ---
    # NB: xg_rate is recomputed from the *raw* log_λ (not the kappa-scaled λ), so it must be
    # sanitized independently — clamp(NaN,…)=NaN would reach the Gamma scale arg and throw
    # DomainError(θ>0) before the is_bad -Inf rejection can take effect.
    xg_rate_h = exp.(log_λ_h) .+ 1e-6
    xg_rate_a = exp.(log_λ_a) .+ 1e-6
    xg_rate_h = ifelse.(isnan.(xg_rate_h) .| isinf.(xg_rate_h), one.(xg_rate_h), xg_rate_h)
    xg_rate_a = ifelse.(isnan.(xg_rate_a) .| isinf.(xg_rate_a), one.(xg_rate_a), xg_rate_a)
    ll_xg_h = logpdf.(Gamma.(ν_xg, xg_rate_h ./ ν_xg), home_xg)
    ll_xg_a = logpdf.(Gamma.(ν_xg, xg_rate_a ./ ν_xg), away_xg)
    Turing.@addlogprob! sum((ll_xg_h .+ ll_xg_a) .* match_weights .* xg_mask)

    # --- Pillar C1: SUPREMACY (who-wins), σ SAMPLED, gated by market_active ---
    market_rate_h = log_λ_h .+ log.(kap_h)
    market_rate_a = log_λ_a .+ log.(kap_a)
    model_sup = market_rate_h .- market_rate_a
    m_sup     = market_log_λ_h .- market_log_λ_a
    ll_sup    = logpdf.(Normal.(model_sup, σ_sup), m_sup)
    Turing.@addlogprob! market_active * supremacy_weight *
        sum(ll_sup .* match_weights .* market_mask)

    # --- Pillar C2: LOCAL-INTENSITY SMILE (per-strike totals), σ SAMPLED ---
    # log Λ^model_m(K) = log(λ_h+λ_a)_m + log_φ(K)  anchored to market log Λ^mkt_m(K).
    log_λ_tot  = log.(λ_h .+ λ_a)                          # [n_matches]
    model_logΛ = log_λ_tot .+ reshape(log_φ, 1, n_strikes) # [n_matches × nK]
    ll_smile   = logpdf.(Normal.(model_logΛ, σ_smile), smile_logΛ)   # [n_matches × nK]
    Turing.@addlogprob! market_active * smile_weight *
        sum(ll_smile .* smile_mask .* match_weights)
end

# ==========================================
# 3. THE BUILDER
# ==========================================
function Features.required_features(model::DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(),
        Features.GoalsFeature(),
        Features.DatesFeature(),
        Features.MonthFeature(),
        Features.XGFeature(),
        model.market_feature_config,
        model.smile_feature,
        model.player_ratings_feature,
        Features.TimeIndicesFeature()
    ]
end

function build_turing_model(config::DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel, feature_set::FeatureSet)
    data = feature_set.data

    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    n_months  = 12

    date_deltas   = Vector{Int}(data[:dates])
    match_weights = 0.5 .^ (date_deltas ./ config.player_dynamics_config.days_half_life)

    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    season_ids = Vector{Int}(data[:season_indices])
    month_idx  = Vector{Int}(data[:flat_months])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    hG = Vector{Float64}(data[:flat_home_G_rating]); hD = Vector{Float64}(data[:flat_home_D_rating])
    hM = Vector{Float64}(data[:flat_home_M_rating]); hF = Vector{Float64}(data[:flat_home_F_rating])
    aG = Vector{Float64}(data[:flat_away_G_rating]); aD = Vector{Float64}(data[:flat_away_D_rating])
    aM = Vector{Float64}(data[:flat_away_M_rating]); aF = Vector{Float64}(data[:flat_away_F_rating])

    home_xg_raw = coalesce.(data[:flat_home_xg], NaN)
    away_xg_raw = coalesce.(data[:flat_away_xg], NaN)
    xg_mask = Float64.(.!isnan.(home_xg_raw) .& .!isnan.(away_xg_raw))
    home_xg = [isnan(x) ? 1.0 : max(Float64(x), 1e-3) for x in home_xg_raw]
    away_xg = [isnan(x) ? 1.0 : max(Float64(x), 1e-3) for x in away_xg_raw]

    # Market pillar: only trust matches where BOTH implied rates are present and in a plausible
    # football range (the market inversion can return a degenerate λ ~357 on thin closing odds).
    _mok(x) = !ismissing(x) && (xf = Float64(x); !isnan(xf) && 0.02 < xf < 20.0)
    market_mask  = Float64.(_mok.(data[:flat_market_λ_home]) .& _mok.(data[:flat_market_λ_away]))
    market_log_h = [_mok(x) ? log(Float64(x)) : 0.0 for x in data[:flat_market_λ_home]]
    market_log_a = [_mok(x) ? log(Float64(x)) : 0.0 for x in data[:flat_market_λ_away]]

    smile_logΛ = Matrix{Float64}(data[:flat_smile_logΛ])
    smile_mask = Matrix{Float64}(data[:flat_smile_mask])
    n_strikes  = size(smile_logΛ, 2)

    market_active = config.market_on ? 1.0 : 0.0

    return build_double_poisson_smile_xg_market_player_engine(
        home_ids, away_ids, season_ids, month_idx,
        home_goals, away_goals, match_weights,
        hG, hD, hM, hF, aG, aD, aM, aF,
        home_xg, away_xg, xg_mask,
        market_log_h, market_log_a, market_mask,
        smile_logΛ, smile_mask, n_strikes,
        market_active, config.supremacy_weight, config.smile_weight,
        config.smile_shape_sd,
        n_teams, n_seasons, n_months, config
    )
end

# ==========================================
# 4. THE EXTRACTOR
# ==========================================
function extract_parameters(
    model::DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel,
    df::AbstractDataFrame,
    feature_set::FeatureSet,
    chain::Chains
)
    data = feature_set.data
    n_seasons = Int(data[:n_seasons]); n_teams = Int(data[:n_teams])
    team_map    = data[:team_map]
    ratings_map = data[:player_ratings_map]
    nK = Int(data[:smile_Kmax]) + 1

    inter_nt = extract_interception(chain, model.interception_config, n_seasons)
    ha_mat   = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    kap_mat  = extract_kappa(chain, model.kappa_config, n_teams)
    p_dyn_nt = extract_dynamics(chain, model.player_dynamics_config, "p_dyn", n_teams)

    n_samples = size(chain, 1) * size(chain, 3)
    base_r = model.player_ratings_feature.tracker.prior_mean

    # Global smile shape φ(K): [n_samples × nK] (same for every match, per posterior draw).
    φ_mat = Matrix{Float64}(undef, n_samples, nK)
    for k in 1:nK
        φ_mat[:, k] = exp.(vec(Array(chain[Symbol("log_φ[$k]")])))
    end

    results = Dict{Int, NamedTuple}()
    for row in eachrow(df)
        h_id = get(team_map, row.home_team, -1)
        a_id = get(team_map, row.away_team, -1)

        m_ratings = get(ratings_map, Int(row.match_id), Dict())
        h_G = get(m_ratings, ("home","G"), 0.0); h_D = get(m_ratings, ("home","D"), 0.0)
        h_M = get(m_ratings, ("home","M"), 0.0); h_F = get(m_ratings, ("home","F"), 0.0)
        a_G = get(m_ratings, ("away","G"), 0.0); a_D = get(m_ratings, ("away","D"), 0.0)
        a_M = get(m_ratings, ("away","M"), 0.0); a_F = get(m_ratings, ("away","F"), 0.0)

        h_G_c = h_G - base_r; h_O_c = (h_D + h_M + h_F) - (10.0 * base_r)
        a_G_c = a_G - base_r; a_O_c = (a_D + a_M + a_F) - (10.0 * base_r)

        att_h = (p_dyn_nt.w_G_att .* h_G_c) .+ (p_dyn_nt.w_Outfield_att .* h_O_c)
        def_h = (p_dyn_nt.w_G_def .* h_G_c) .+ (p_dyn_nt.w_Outfield_def .* h_O_c)
        att_a = (p_dyn_nt.w_G_att .* a_G_c) .+ (p_dyn_nt.w_Outfield_att .* a_O_c)
        def_a = (p_dyn_nt.w_G_def .* a_G_c) .+ (p_dyn_nt.w_Outfield_def .* a_O_c)

        γ_h = h_id > 0 ? ha_mat[:, h_id] : zeros(n_samples)
        κ_h = h_id > 0 ? kap_mat[:, h_id] : ones(n_samples)
        κ_a = a_id > 0 ? kap_mat[:, a_id] : ones(n_samples)

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        m_idx = hasproperty(row, :month_idx) ? Int(row.month_idx) : 1
        μ_v = inter_nt.μ_base[:, s_idx] .+ inter_nt.δ_month[:, m_idx]

        log_λ_h = clamp.(μ_v .+ γ_h .+ att_h .+ def_a, -20.0, 20.0)
        log_λ_a = clamp.(μ_v .+        att_a .+ def_h, -20.0, 20.0)

        λ_h = κ_h .* exp.(log_λ_h) .+ 1e-6
        λ_a = κ_a .* exp.(log_λ_a) .+ 1e-6

        results[Int(row.match_id)] = (;
            λ_h, λ_a,
            λ_tot = λ_h .+ λ_a,
            φ = φ_mat,                 # [n_samples × nK] global smile
            θ_1 = log.(λ_h), θ_2 = log.(λ_a), θ_3 = zeros(n_samples), ρ = zeros(n_samples),
            true_xg_h = exp.(log_λ_h), true_xg_a = exp.(log_λ_a),
        )
    end
    return results
end
