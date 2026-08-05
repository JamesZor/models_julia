# l05_multi_match.jl

# We build directly on top of the vector alpha architecture
include("l04_vector_alpha.jl")
include("l03_risk_manager.jl") # For RiskConfig

# ===================================================================
# 1. Multi-Match Risk Manager Core Logic (Global Drawdown Constraint)
# ===================================================================

"""
Solves for the Kelly shrinkage factor `k` ∈ (0, 1] that strictly bounds the 
expected drawdown penalty function to <= 0.0 across L concurrent matches using Bisection.
Based on the Stochastic Portfolio Theory Multi-Match formulation.
"""
function solve_global_drawdown_multiplier(match_probs_list::Vector{Vector{Float64}}, match_returns_list::Vector{Vector{Float64}}, lambda::Float64; max_iters=50)
    L = length(match_probs_list)
    if L == 0
        return 1.0
    end
    
    # The penalty function for L concurrent independent matches.
    # We sum the log of the expected penalty for each match.
    # The constraint is satisfied when the sum is <= 0.0.
    function f(k)
        total_log_penalty = 0.0
        for t in 1:L
            p_vec = match_probs_list[t]
            r_vec = match_returns_list[t]
            
            # Expected penalty for this single match
            match_penalty = sum(p_vec[i] * (1.0 + k * r_vec[i])^(-lambda) for i in 1:length(p_vec))
            
            # Aggregate via logarithm for multiplicative independence
            total_log_penalty += log(match_penalty)
        end
        return total_log_penalty
    end
    
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
    
    return low 
end
