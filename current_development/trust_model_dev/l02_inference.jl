# current_development/trust_model_dev/l02_inference.jl
using DataFrames
using Distributions
using StatsFuns: logistic
using MCMCChains

"""
    compose_l2_ppd(l1_ppd_df::DataFrame, ds::BayesianFootball.Data.DataStore, chain::Chains, config::TrustModelConfig, team_map::Dict)

Takes the L1 predictions and the trained Trust Model MCMC chain, and blends them 
to produce the final L2 Posterior Predictive Distribution (PPD).
"""
function compose_l2_ppd(
    l1_ppd_df::DataFrame, 
    ds::BayesianFootball.Data.DataStore, 
    chain::Chains, 
    config::TrustModelConfig, 
    team_map::Dict
)
    # 1. Map outcomes to Parent Market Indices (Same logic as extractor)
    outcome_to_market_id = Dict{String, Int}()
    target_strs = String[]
    for (i, m) in enumerate(config.market_config.markets)
        for out in values(BayesianFootball.Data.Markets.outcomes(m))
            out_str = String(out)
            push!(target_strs, out_str)
            outcome_to_market_id[out_str] = i
        end
    end
    
    # 2. Extract q_market from Odds DataFrame
    odds_df = subset(ds.odds, :selection => ByRow(s -> String(s) ∈ target_strs))
    
    # 3. Inner join exactly like the extractor
    combined = innerjoin(l1_ppd_df, odds_df, on=[:match_id, :selection], makeunique=true)
    
    # 4. Map match_id to teams
    match_team_map = Dict(row.match_id => (row.home_team, row.away_team) for row in eachrow(ds.matches))
    
    # 5. Extract arrays from Chains
    chain_df = DataFrame(chain)
    n_samples = nrow(chain_df)
    
    # 6. Build the new distribution array!
    new_distributions = Vector{Vector{Float64}}(undef, nrow(combined))
    
    for row_idx in 1:nrow(combined)
        row = combined[row_idx, :]
        
        market_idx = outcome_to_market_id[String(row.selection)]
        
        h_team, a_team = match_team_map[row.match_id]
        h_idx = team_map[h_team]
        a_idx = team_map[a_team]
        
        q_market = row.prob_fair_close
        
        # We assume L1 PPD has at least as many samples as our Trust Chain.
        # If not, we can safely cycle through them.
        l1_dist = row.distribution
        l1_n_samples = length(l1_dist)
        
        p_tilde_dist = Float64[]
        for s in 1:n_samples
            # Grab parameters for this specific sample step
            w0 = chain_df[s, Symbol("w0[$market_idx]")]
            
            if config.team_config !== nothing
                sigma = chain_df[s, Symbol("σ_team")]
                tz_home = chain_df[s, Symbol("team_z[$h_idx]")]
                tz_away = chain_df[s, Symbol("team_z[$a_idx]")]
                team_eff_home = tz_home * sigma
                team_eff_away = tz_away * sigma
            else
                team_eff_home = 0.0
                team_eff_away = 0.0
            end
            
            # Calculate weight
            eta = w0 + team_eff_home + team_eff_away
            w_l = logistic(eta)
            
            # Grab corresponding L1 probability 
            p_model = l1_dist[mod1(s, l1_n_samples)]
            
            # BLEND!
            p_tilde = w_l * p_model + (1.0 - w_l) * q_market
            
            push!(p_tilde_dist, p_tilde)
        end
        
        new_distributions[row_idx] = p_tilde_dist
    end
    
    # 7. Package it back into a PPD DataFrame format
    l2_ppd = DataFrame(
        match_id = combined.match_id,
        selection = combined.selection,
        distribution = new_distributions,
        market_name = combined.market_name,
        q_market = combined.prob_fair_close
    )
    
    return l2_ppd
end
