module TrustMVP

using Turing
using DataFrames
using Distributions
using Statistics
using StatsFuns: logistic

# 1. The Global CLV Prior Calculator
function calculate_global_clv_prior(clv_matrix::AbstractDataFrame)
    model_deviation = clv_matrix.model_prob .- clv_matrix.prob_fair_open
    market_movement = clv_matrix.prob_fair_close .- clv_matrix.prob_fair_open
    
    variance = var(model_deviation)
    beta_clv = variance > 1e-8 ? cov(model_deviation, market_movement) / variance : 0.0
    
    # Scale and anchor (textbook fast channel heuristic)
    dynamic_prior_mean = -1.0 + (beta_clv * 5.0)
    return dynamic_prior_mean, beta_clv
end

# 2. The Turing Model (Global CLV, Hierarchical Teams)
@model function hierarchical_trust_with_clv(
    is_winner::Vector{Int}, 
    p_model::Vector{Float64}, 
    q_market::Vector{Float64},
    home_idx::Vector{Int},
    away_idx::Vector{Int},
    n_teams::Int,
    clv_prior_mean::Float64
)
    # Global trust anchored by CLV
    w0 ~ Normal(clv_prior_mean, 1.0)
    
    # Hierarchical spread
    σ_team ~ Exponential(0.5)
    z_team ~ filldist(Normal(0, 1), n_teams)
    
    # AD-SAFE extraction
    home_effects = view(z_team, home_idx)
    away_effects = view(z_team, away_idx)
    
    η = w0 .+ σ_team .* (home_effects .+ away_effects)
    w_l_vector = logistic.(η)
    
    # Blend
    p_tilde = clamp.(w_l_vector .* p_model .+ (1.0 .- w_l_vector) .* q_market, 1e-6, 1.0 - 1e-6)
    
    # Vectorized likelihood
    ll = logpdf.(Bernoulli.(p_tilde), is_winner)
    Turing.@addlogprob! sum(ll)
end

# Builder for AD-safe static vector passing
function build_trust_model(df::AbstractDataFrame, total_teams::Int, clv_mean::Float64)
    is_winner = Vector{Int}(df.is_winner)
    p_model   = Vector{Float64}(df.model_prob)
    q_market  = Vector{Float64}(df.market_devig)
    home_idx  = Vector{Int}(df.home_team_index)
    away_idx  = Vector{Int}(df.away_team_index)
    
    return hierarchical_trust_with_clv(is_winner, p_model, q_market, home_idx, away_idx, total_teams, clv_mean)
end

# 3. Distributional Staking Evaluator (L1 + L2 Composed)
function distributional_staking(
    next_matrix::DataFrame, 
    chain_trust, 
    match_to_l1_dist::Dict, 
    team_map::Dict
)
    w0_draws = vec(Array(chain_trust[:w0]))
    σ_draws  = vec(Array(chain_trust[:σ_team]))
    
    composed_distributions = Vector{Float64}[]
    dist_staking_results = Float64[]
    
    for row in eachrow(next_matrix)
        l1_draws = match_to_l1_dist[row.match_id]
        q_mkt = row.market_devig
        odds  = row.odds_close
        
        h_idx = get(team_map, row.home_team, 0)
        a_idx = get(team_map, row.away_team, 0)
        
        n_samples = min(length(w0_draws), length(l1_draws))
        
        match_composed_dist = Float64[]
        draw_stakes = Float64[]
        
        for s in 1:n_samples
            w0_s = w0_draws[s]
            σ_s  = σ_draws[s]
            
            h_z_s = h_idx > 0 ? chain_trust[Symbol("z_team[$h_idx]")][s] : 0.0
            a_z_s = a_idx > 0 ? chain_trust[Symbol("z_team[$a_idx]")][s] : 0.0
            
            # Trust for this draw
            η_s = w0_s + σ_s * (h_z_s + a_z_s)
            w_s = logistic(η_s)
            
            # Composed Probability
            p_tilde_s = w_s * l1_draws[s] + (1.0 - w_s) * q_mkt
            push!(match_composed_dist, p_tilde_s)
            
            # Exact Kelly Stake for this draw
            b = odds - 1.0
            stake_s = (p_tilde_s * (b + 1.0) - 1.0) / b
            push!(draw_stakes, max(0.0, stake_s)) # Floor at 0 for back bets
        end
        
        push!(composed_distributions, match_composed_dist)
        push!(dist_staking_results, mean(draw_stakes))
    end
    
    return composed_distributions, dist_staking_results
end

end # module
