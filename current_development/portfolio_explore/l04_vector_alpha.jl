# l04_vector_alpha.jl

# Build on top of l02 Base Architecture
include("l02_portfolio_backtest.jl")

# ===================================================================
# 1. Vector Alpha Config
# ===================================================================

struct VectorAlphaConfig
    commission::Float64
    alpha_dict::Dict{String, Float64}
end

# ===================================================================
# 2. Vectorized Portfolio Optimization
# ===================================================================

"""
Evaluates the optimal Kelly stakes, and then applies a vectorized 
fractional shrinkage parameter uniquely tailored to each specific selection.
"""
function optimize_portfolio_vector(score_matrix, match_model_prob, odds_map, fair_prob_map, winner_map, config::VectorAlphaConfig)
    
    # 1. Generate the standard unconstrained pure Kelly stakes (alpha = 1.0)
    dummy_b_config = BacktestConfig(commission=config.commission, alphas=[1.0])
    selections, base_stakes, R_mat = optimize_portfolio(score_matrix, match_model_prob, odds_map, fair_prob_map, winner_map, dummy_b_config; alpha=1.0)
    
    if isempty(selections)
        return selections, base_stakes, R_mat
    end
    
    # 2. Construct the alpha shrinkage vector corresponding exactly to the chosen selections
    alpha_vec = zeros(length(selections))
    for i in 1:length(selections)
        sel = selections[i]
        
        m_str = replace(sel.market, "Market[" => "", "]" => "")
        key = "$(m_str)_$(sel.selection)"
        
        if !haskey(config.alpha_dict, key)
            # Print a warning so we can see exactly what string it expected
            println("WARNING: Unmapped Selection Key: ", key)
        end
        
        # If the key isn't in the dict (shouldn't happen), default to 0.0 safety
        alpha_vec[i] = get(config.alpha_dict, key, 0.0) 
    end
    
    # 3. Apply the vectorized fractional shrinkage
    a_blended_net = base_stakes .* alpha_vec
    
    return selections, a_blended_net, R_mat
end
