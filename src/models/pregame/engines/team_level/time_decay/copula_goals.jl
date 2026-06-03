# src/models/pregame/engines/team_level/time_decay/copula_goals.jl

Base.@kwdef struct DynamicCopulaGoalsTimeDecayModel <: AbstractDynamicFrankCopulaNegBinModel
    interception_config::AbstractInterceptionConfig
    dynamics_config::AbstractDynamicsConfig
    dispersion_config::AbstractDispersionConfig
    homeadvantage_config::AbstractHomeAdvantageConfig
    copula_config::AbstractCopulaConfig
end

@model function build_weighted_copula_goals_engine(
    home_team_indices::Vector{Int},
    away_team_indices::Vector{Int},
    season_indices::Vector{Int},
    time_indices::Vector{Int},
    month_indices::Vector{Int},
    home_goals::Vector{Int}, 
    away_goals::Vector{Int},
    match_weights::Vector{Float64},
    n_teams::Int,
    n_seasons::Int,
    n_months::Int,
    config::DynamicCopulaGoalsTimeDecayModel 
)
    # 1. LOAD COMPONENTS
    inter ~ to_submodel(build_interception(config.interception_config, n_seasons, n_months))
    disp  ~ to_submodel(build_dispersion(config.dispersion_config, n_teams, n_months))
    ha    ~ to_submodel(build_home_advantage(config.homeadvantage_config, n_teams))
    dyn   ~ to_submodel(build_dynamics(config.dynamics_config, n_teams))
    
    # Copula parameters
    copula ~ to_submodel(build_copula(config.copula_config, n_teams))

    # 2. VECTORIZED INDEXING 
    att_h = view(dyn.α, home_team_indices)
    def_h = view(dyn.β, home_team_indices)
    att_a = view(dyn.α, away_team_indices)
    def_a = view(dyn.β, away_team_indices)
    inter_match = inter.μ_base[season_indices] .+ inter.δ_month[month_indices]
    home_adv = view(ha, home_team_indices)
    
    # Copula deltas
    δ_κ_h = view(copula.δ_κ, home_team_indices)
    δ_κ_a = view(copula.δ_κ, away_team_indices)
    
    # Match-specific correlation
    κ_frank = copula.κ_base .+ δ_κ_h .+ δ_κ_a

    # --- Dispersion Construction ---
    if hasproperty(disp, :team_vol) 
        vol_h = view(disp.team_vol, home_team_indices)
        vol_a = view(disp.team_vol, away_team_indices)
        vol_m = view(disp.month_vol, month_indices)
        
        log_r_h = disp.base .+ disp.home_offset .+ vol_h .+ vol_a .+ vol_m
        log_r_a = disp.base .+ vol_h .+ vol_a .+ vol_m
        
        r_h_flat = exp.(clamp.(log_r_h, -10.0, 10.0))
        r_a_flat = exp.(clamp.(log_r_a, -10.0, 10.0))
    else 
        r_h_flat = fill(disp.h, length(home_team_indices))
        r_a_flat = fill(disp.a, length(home_team_indices))
    end

    # 3. VECTORIZED RATES (λ)
    λ_h = exp.(inter_match .+ home_adv .+ att_h .+ def_a)
    λ_a = exp.(inter_match .+             att_a .+ def_h)

    # 4. TIME-DECAYED COPULA LIKELIHOOD
    copula_dists = MyDistributions.FrankCopulaNegBin.(r_h_flat, λ_h, r_a_flat, λ_a, κ_frank)
    
    log_lik_joint = logpdf.(copula_dists, home_goals, away_goals)
    
    Turing.@addlogprob! sum(log_lik_joint .* match_weights)
end

# ==============================================================================
# INTERFACES FOR FEATURES & ENGINE
# ==============================================================================

function Features.required_features(model::DynamicCopulaGoalsTimeDecayModel)
    return Features.AbstractFeatureConfig[
        Features.TeamIDsFeature(), 
        Features.GoalsFeature(), 
        Features.DatesFeature(), 
        Features.MonthFeature(),
        Features.TimeIndicesFeature()
    ] 
end

function build_turing_model(model::DynamicCopulaGoalsTimeDecayModel, feature_set::Features.FeatureSet)
    data = feature_set.data
    
    n_teams    = Int(data[:n_teams])
    n_seasons  = Int(data[:n_seasons])
    n_months   = 12
    
    date_deltas = Vector{Int}(data[:dates])
    match_weights = calculate_match_weights(date_deltas, model.dynamics_config.days_half_life)
    
    home_ids   = Vector{Int}(data[:flat_home_ids])
    away_ids   = Vector{Int}(data[:flat_away_ids])
    season_ids = Vector{Int}(data[:season_indices])
    time_idxs  = Vector{Int}(data[:time_indices])
    month_indices = Vector{Int}(data[:flat_months])
    home_goals = Vector{Int}(data[:flat_home_goals])
    away_goals = Vector{Int}(data[:flat_away_goals])

    return build_weighted_copula_goals_engine(
        home_ids,
        away_ids,
        season_ids,
        time_idxs,
        month_indices,
        home_goals,
        away_goals,
        match_weights,
        n_teams,
        n_seasons,
        n_months,
        model
    )
end

function extract_parameters(
    model::DynamicCopulaGoalsTimeDecayModel, 
    df::AbstractDataFrame, 
    feature_set::Features.FeatureSet,
    chain::Chains
)
    # 1. Unpack Metadata
    data = feature_set.data
    n_teams   = Int(data[:n_teams])
    n_seasons = Int(data[:n_seasons])
    n_months  = 12
    team_map  = data[:team_map]

    # 2. DELEGATE TO COMPONENTS
    inter_nt  = extract_interception(chain, model.interception_config, n_seasons)
    disp_nt   = extract_dispersion(chain, model.dispersion_config, n_teams, n_months)
    ha_mat    = extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    dyn_nt    = extract_dynamics(chain, model.dynamics_config, "dyn", n_teams)
    cop_nt    = extract_copula(chain, model.copula_config, "copula", n_teams)

    n_samples = size(chain, 1) * size(chain, 3) 
    results = Dict{Int, NamedTuple}()

    for row in eachrow(df)
        mid = Int(row.match_id)

        h_idx = get(team_map, row.home_team, -1)
        a_idx = get(team_map, row.away_team, -1)

        # dyn_nt.α is [Samples, Teams]
        α_h = h_idx > 0 ? dyn_nt.α[:, h_idx] : zeros(n_samples)
        β_h = h_idx > 0 ? dyn_nt.β[:, h_idx] : zeros(n_samples)
        α_a = a_idx > 0 ? dyn_nt.α[:, a_idx] : zeros(n_samples)
        β_a = a_idx > 0 ? dyn_nt.β[:, a_idx] : zeros(n_samples)

        γ_h = h_idx > 0 ? ha_mat[:, h_idx] : zeros(n_samples)
        
        δ_κ_h = h_idx > 0 ? cop_nt.δ_κ[:, h_idx] : zeros(n_samples)
        δ_κ_a = a_idx > 0 ? cop_nt.δ_κ[:, a_idx] : zeros(n_samples)
        κ_frank = cop_nt.κ_base .+ δ_κ_h .+ δ_κ_a

        s_idx = hasproperty(row, :season_idx) ? Int(row.season_idx) : n_seasons

        # --- Reconstruct Dispersion and Interception ---
        m_idx = month(row.match_date)
        inter_match = inter_nt.μ_base[:, s_idx] .+ inter_nt.δ_month[:, m_idx]
        match_disp = reconstruct_dispersion(disp_nt, h_idx, a_idx, m_idx)

        λ_goals_h = exp.(inter_match .+ γ_h .+ α_h .+ β_a)
        λ_goals_a = exp.(inter_match .+        α_a .+ β_h)

        results[mid] = (;
            λ_h = λ_goals_h,
            λ_a = λ_goals_a,
            r_h = match_disp.h,
            r_a = match_disp.a,
            κ_frank = κ_frank,
            true_xg_h = λ_goals_h, 
            true_xg_a = λ_goals_a
        )
    end

    return results
end
