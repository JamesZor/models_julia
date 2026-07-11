# src/models/pregame/engines/xg_market_engine.jl

using Turing
using Distributions

Base.@kwdef struct DynamicMarketXGModel{
  I<:AbstractInterceptionConfig,
  T<:AbstractDynamicsConfig, 
  D<:AbstractDispersionConfig, 
  H<:AbstractHomeAdvantageConfig,
  K<:AbstractKappaConfig
    } <: AbstractStandardTeamModel
    interception_config::I
    dynamics_config::T
    dispersion_config::D
    homeadvantage_config::H
    kappa_config::K
    ν_xg::Distribution = truncated(Normal(3.0, 0.5), lower=0.5) 
    market_σ::Distribution = truncated(Normal(0.1, 0.2), lower=0.01) 
end

# ==========================================
# 1. THE TURING ENGINE
# ==========================================
@model function build_xg_market_engine(
    # --- Base Data ---
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    time_indices::Vector{Int},
    month_indices::Vector{Int},
    home_goals::Vector{Int},
    away_goals::Vector{Int},
    # --- Expected Goals Data ---
    home_xg::Vector{Float64},
    away_xg::Vector{Float64},
    idx_xg::Vector{Int},
    idx_no_xg::Vector{Int},
    # --- Market Data ---
    market_log_λ_h::Vector{Float64},
    market_log_λ_a::Vector{Float64},
    idx_market::Vector{Int},
    # --- Dimensions ---
    n_teams::Int,
    n_history::Int,
    n_target::Int,
    n_seasons::Int,
    n_months::Int,
    config::DynamicMarketXGModel
)
    # ==========================================
    # 1. LOAD COMPONENTS
    # ==========================================
    ν_xg     ~ config.ν_xg
    σ_market ~ config.market_σ

    inter ~ to_submodel(build_interception(config.interception_config, n_seasons))
    disp  ~ to_submodel(build_dispersion(config.dispersion_config, n_teams, n_months))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    kap   ~ to_submodel(build_kappa(config.kappa_config, n_teams))
    dyn   ~ to_submodel(build_dynamics(config.dynamics_config, n_teams, n_history, n_target))

    # ==========================================
    # 2. VECTORIZED INDEXING
    # ==========================================
    idx_h = CartesianIndex.(home_team_indices, time_indices)
    idx_a = CartesianIndex.(away_team_indices, time_indices)

    att_h = view(dyn.α, idx_h)
    def_h = view(dyn.β, idx_h)
    att_a = view(dyn.α, idx_a)
    def_a = view(dyn.β, idx_a)

    γ_h_flat = view(ha, home_team_indices)
    κ_h_flat = view(kap, home_team_indices)
    κ_a_flat = view(kap, away_team_indices)

    inter_match = view(inter, season_indices)

    # --- Dispersion Construction ---
    if hasproperty(disp, :team_vol) # AdvancedVolatilityDispersion
        vol_h = view(disp.team_vol, home_team_indices)
        vol_a = view(disp.team_vol, away_team_indices)
        vol_m = view(disp.month_vol, month_indices)
        
        log_r_h = disp.base .+ disp.home_offset .+ vol_h .+ vol_a .+ vol_m
        log_r_a = disp.base .+ vol_h .+ vol_a .+ vol_m
        
        r_h_flat = exp.(clamp.(log_r_h, -10.0, 10.0))
        r_a_flat = exp.(clamp.(log_r_a, -10.0, 10.0))
    else # GlobalDispersion or HomeAwayDispersion
        r_h_flat = disp.h
        r_a_flat = disp.a
    end

    # ==========================================
    # 3. STABLE RATE GENERATION (True xG)
    # ==========================================
    log_λₕ = clamp.(inter_match .+ γ_h_flat .+ att_h .+ def_a, -20.0, 20.0) 
    log_λₐ = clamp.(inter_match .+             att_a .+ def_h, -20.0, 20.0)

    λₕ = exp.(log_λₕ) .+ 1e-6
    λₐ = exp.(log_λₐ) .+ 1e-6

    # AD-Safe Rejection
    if any(isnan, λₕ) || any(isnan, λₐ) || any(isinf, λₕ) || any(isinf, λₐ)
        Turing.@addlogprob! -Inf
        return
    end

    # ==========================================
    # 4. LIKELIHOOD PIPELINE
    # ==========================================
    
    # --- Pillar A: xG (Gamma) ---
    if !isempty(idx_xg)
        home_xg[idx_xg] ~ arraydist(Gamma.(ν_xg, λₕ[idx_xg] ./ ν_xg))
        away_xg[idx_xg] ~ arraydist(Gamma.(ν_xg, λₐ[idx_xg] ./ ν_xg))
    end

    # --- Pillar B: Actual Goals (NegBin) ---
    # log(Goals) = log(xG * Kappa)
    home_goals ~ arraydist(RobustNegativeBinomial.(r_h_flat, κ_h_flat .* λₕ))
    away_goals ~ arraydist(RobustNegativeBinomial.(r_a_flat, κ_a_flat .* λₐ))

    # --- Pillar C: The Market (Normal) ---
    if !isempty(idx_market)
        # We model the market's implied log-lambda as a noisy observation of the true model log-lambda
        # Note: Market lambda is priced on GOALS, so we must add the Kappa shift
        market_rate_h = log_λₕ[idx_market] .+ log.(κ_h_flat[idx_market])
        market_rate_a = log_λₐ[idx_market] .+ log.(κ_a_flat[idx_market])

        market_log_λ_h[idx_market] ~ arraydist(Normal.(market_rate_h, σ_market))
        market_log_λ_a[idx_market] ~ arraydist(Normal.(market_rate_a, σ_market))
    end
end

# ==========================================
# 2. THE BUILDER
# ==========================================
function Features.required_features(model::DynamicMarketXGModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), 
        Features.GoalsFeature(), 
        Features.MonthFeature(), 
        Features.XGFeature(), 
        Features.DoublePoissonMarketFeature(),
        Features.TimeIndicesFeature()
    ] 
end

function build_turing_model(config::DynamicMarketXGModel, feature_set::FeatureSet)
    data = feature_set.data
    
    n_teams   = Int(data[:n_teams])
    n_rounds  = Int(data[:n_rounds])
    n_history = Int(data[:n_history_steps])
    n_target  = Int(data[:n_target_steps])
    n_seasons = Int(data[:n_seasons]) 
    n_months  = 12
    
    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    season_ids = Vector{Int}(data[:season_indices])
    time_idxs  = Vector{Int}(data[:time_indices])
    month_indices = Vector{Int}(data[:flat_months])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    home_xg = Vector{Float64}(coalesce.(data[:flat_home_xg], NaN))
    away_xg = Vector{Float64}(coalesce.(data[:flat_away_xg], NaN))

    idx_xg    = findall(x -> !isnan(x), home_xg)
    idx_no_xg = findall(isnan, home_xg)

    market_log_h = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_home]), NaN))
    market_log_a = Vector{Float64}(coalesce.(log.(data[:flat_market_λ_away]), NaN))
    idx_market = findall(x -> !isnan(x), market_log_h)

    return build_xg_market_engine(
        home_ids, away_ids, 
        season_ids,
        time_idxs,
        month_indices,
        home_goals, away_goals, 
        home_xg, away_xg, 
        idx_xg, idx_no_xg,
        market_log_h, market_log_a, idx_market,
        n_teams, n_history, n_target, n_seasons, n_months,
        config
    )
end

# ==========================================
# 3. THE EXTRACTOR
# ==========================================
function extract_parameters(
    model::DynamicMarketXGModel, 
    df::AbstractDataFrame, 
    feature_set::FeatureSet,
    chain::Chains
)
    # 1. Unpack Metadata
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_rounds  = Int(data[:n_rounds])
    n_history = Int(data[:n_history_steps])
    n_target  = Int(data[:n_target_steps])
    n_seasons = Int(data[:n_seasons])
    n_months  = 12
    team_map  = data[:team_map]

    # 2. DELEGATE TO COMPONENTS
    inter_mat = extract_interception(chain, model.interception_config, n_seasons)
    disp_nt = extract_dispersion(chain, model.dispersion_config, n_teams, n_months)
    ha_mat  = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    kap_mat = extract_kappa(chain, model.kappa_config, n_teams)
    dyn_nt  = extract_dynamics(chain, model.dynamics_config, "dyn", n_teams, n_history, n_target)

    n_samples = size(chain, 1) * size(chain, 3) 
    results = Dict{Int, NamedTuple}()

    # 3. FIFIXTURE LOOP
    for row in eachrow(df)
        mid = Int(row.match_id)
        t_idx = hasproperty(row, :time_index) ? Int(row.time_index) : n_rounds

        h_idx = get(team_map, row.home_team, -1)
        a_idx = get(team_map, row.away_team, -1)

        α_h = h_idx > 0 ? dyn_nt.α[h_idx, t_idx, :] : zeros(n_samples)
        β_h = h_idx > 0 ? dyn_nt.β[h_idx, t_idx, :] : zeros(n_samples)
        α_a = a_idx > 0 ? dyn_nt.α[a_idx, t_idx, :] : zeros(n_samples)
        β_a = a_idx > 0 ? dyn_nt.β[a_idx, t_idx, :] : zeros(n_samples)

        γ_h = h_idx > 0 ? ha_mat[:, h_idx] : zeros(n_samples)
        
        κ_h = h_idx > 0 ? kap_mat[:, h_idx] : ones(n_samples)
        κ_a = a_idx > 0 ? kap_mat[:, a_idx] : ones(n_samples)

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons
        μ_v = inter_mat[:, s_idx] 

        # --- Reconstruct Dispersion ---
        m_idx = month(row.match_date)
        match_disp = reconstruct_dispersion(disp_nt, h_idx, a_idx, m_idx)

        # 4. FINAL LIKELIHOOD MATH
        log_λ_h = clamp.(μ_v .+ γ_h .+ α_h .+ β_a, -20.0, 20.0)
        log_λ_a = clamp.(μ_v .+        α_a .+ β_h, -20.0, 20.0)

        λ_goals_h = κ_h .* exp.(log_λ_h) .+ 1e-6
        λ_goals_a = κ_a .* exp.(log_λ_a) .+ 1e-6

        results[mid] = (;
            λ_h = λ_goals_h,
            λ_a = λ_goals_a,
            r_h = match_disp.h,  
            r_a = match_disp.a,
            true_xg_h = exp.(log_λ_h), 
            true_xg_a = exp.(log_λ_a),
        )
    end
    
    return results
end
