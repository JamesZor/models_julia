# l03_risk_manager.jl

# We build directly on top of the l02 base architecture
include("l02_portfolio_backtest.jl")

# ===================================================================
# 1. Risk Manager Config & Core Logic
# ===================================================================

struct RiskConfig
    lambda::Float64
end

# Convenience constructor from the paper's D (drawdown limit) and beta (prob tolerance)
function RiskConfig(D::Float64, beta::Float64)
    return RiskConfig(log(beta) / log(D))
end

"""
Solves for the Kelly shrinkage factor `k` ∈ (0, 1] that strictly bounds the 
expected drawdown penalty function to <= 1.0 using Bisection.
"""
function solve_drawdown_multiplier(p_model_vec, returns_vec, lambda; max_iters=50)
    # The penalty function. We subtract 1.0 so we are searching for the root f(k) = 0
    f(k) = sum(p_model_vec[i] * (1.0 + k * returns_vec[i])^(-lambda) for i in 1:length(p_model_vec)) - 1.0
    
    # Fast path: if k=1.0 (base optimal Kelly) already satisfies the constraint
    if f(1.0) <= 0.0
        return 1.0
    end
    
    # Bisection search in (0, 1)
    low = 0.0
    high = 1.0
    for _ in 1:max_iters
        mid = (low + high) / 2.0
        if f(mid) > 0.0
            high = mid  # Exceeded risk tolerance, must lower k
        else
            low = mid   # Within risk tolerance, can try higher k
        end
    end
    
    # Return the conservative lower bound that guarantees f(low) <= 0
    return low 
end

# ===================================================================
# 2. Risk-Managed Match Evaluation
# ===================================================================

function evaluate_match_risk_managed(latents_row, expr, odds_df, markets_config, b_config::BacktestConfig, risk_config::RiskConfig; optimal_alpha::Float64=0.35)
    m_id = latents_row.match_id
    lambda = risk_config.lambda
    
    res_df = DataFrame(
        match_id     = Int[],
        alpha        = Float64[],
        lambda_val   = Float64[],
        base_stake   = Float64[],
        risk_stake   = Float64[],
        shrink_k     = Float64[],
        base_pl      = Float64[],
        risk_pl      = Float64[],
        bets_placed  = Int[],
        status       = String[],
        placed_bets  = String[]
    )
    
    # 1. Fast-Fail if no odds
    raw_odds_map, _, winner_map = extract_market_data(odds_df, m_id, markets_config)
    if isempty(raw_odds_map)
        push!(res_df, (m_id, optimal_alpha, lambda, 0.0, 0.0, 1.0, 0.0, 0.0, 0, "MISSING_ODDS", ""))
        return res_df
    end
    
    odds_map, fair_prob_map = normalize_market_group_odds(raw_odds_map)
    
    # 2. Compute Score Matrix
    param = Predictions.extract_params(expr.config.model, latents_row)
    score_matrix = nothing
    try
        score_matrix = Predictions.compute_score_matrix(expr.config.model, param)
    catch e
        push!(res_df, (m_id, optimal_alpha, lambda, 0.0, 0.0, 1.0, 0.0, 0.0, 0, "SOLVER_FAIL", ""))
        return res_df
    end
    
    match_model_prob = Dict(
        string(m) => Predictions.compute_market_probs(score_matrix, m)
        for m in markets_config.markets
    )
    
    # 3. Base Optimization (using optimal_alpha derived from our r03 backtest)
    selections, base_stakes, R_mat = optimize_portfolio(score_matrix, match_model_prob, odds_map, fair_prob_map, winner_map, b_config; alpha=optimal_alpha)
    
    if isempty(selections)
        push!(res_df, (m_id, optimal_alpha, lambda, 0.0, 0.0, 1.0, 0.0, 0.0, 0, "NO_SELECTIONS", ""))
        return res_df
    end
    
    # 4. Risk Manager Shrinkage Calculation
    P_model_grid = mean(score_matrix.data, dims=3)[:, :, 1]
    p_model_vec  = vec(P_model_grid)
    
    # R_mat is N_states x N_selections. returns_vec is the net return for each possible 144 scoreline.
    returns_vec = R_mat * base_stakes
    
    k_shrink = solve_drawdown_multiplier(p_model_vec, returns_vec, lambda)
    risk_stakes = base_stakes .* k_shrink
    
    # 5. Calculate Financials
    base_tot_st = sum(base_stakes)
    risk_tot_st = sum(risk_stakes)
    
    base_net_pl = 0.0
    risk_net_pl = 0.0
    bets_placed = 0
    placed_bets_arr = String[]
    
    for i in 1:length(selections)
        sel = selections[i]
        bst = base_stakes[i]
        rst = risk_stakes[i]
        
        if bst > 0
            bets_placed += 1
            if sel.is_winner
                pl_factor = (1.0 - b_config.commission) * (sel.odds - 1.0)
                base_net_pl += bst * pl_factor
                risk_net_pl += rst * pl_factor
            else
                base_net_pl -= bst
                risk_net_pl -= rst
            end
            
            # Record using the risk-managed stakes
            push!(placed_bets_arr, "$(sel.market):$(sel.selection)($(round(rst*100, digits=2))%)")
        end
    end
    
    placed_bets_str = join(placed_bets_arr, ", ")
    
    push!(res_df, (m_id, optimal_alpha, lambda, base_tot_st, risk_tot_st, k_shrink, base_net_pl, risk_net_pl, bets_placed, "SUCCESS", placed_bets_str))
    
    return res_df
end

# ===================================================================
# 3. Multithreaded Execution Wrapper
# ===================================================================

function run_risk_backtest(latents_df::DataFrame, expr, odds_df::DataFrame, markets_config, b_config::BacktestConfig, risk_config::RiskConfig; optimal_alpha=0.35)
    n_matches = nrow(latents_df)
    results = Vector{DataFrame}(undef, n_matches)
    
    Threads.@threads for i in 1:n_matches
        results[i] = evaluate_match_risk_managed(latents_df[i, :], expr, odds_df, markets_config, b_config, risk_config; optimal_alpha=optimal_alpha)
    end
    
    return vcat(results...)
end
