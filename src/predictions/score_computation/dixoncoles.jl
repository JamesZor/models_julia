# src/predictions/score_computation/dixon_coles.jl

using Distributions
using ..Models.PreGame: DynamicDixonColesXGOutfieldPlayerTimeDecayModel, DynamicDixonColesXGOutfieldPlayerTimeDecayNoMarketModel,
    DynamicDixonColesXGFullPositionPlayerTimeDecayModel
# using LogExpFunctions: logpdf

const AbstractDixonColesPlayerModels = Union{
    AbstractDixonColesModel,
    DynamicDixonColesXGOutfieldPlayerTimeDecayModel,
    DynamicDixonColesXGOutfieldPlayerTimeDecayNoMarketModel,
    DynamicDixonColesXGFullPositionPlayerTimeDecayModel
}

# 1. Adapter: Maps the flat model output to named parameters
# Note: In our model implementation, θ_3 corresponds to the transformed Rho
function extract_params(model::AbstractDixonColesPlayerModels, row)
    return (θ_1 = row.θ_1, θ_2 = row.θ_2, ρ = row.θ_3)
end

# 2. Kernel: Params -> ScoreMatrix
function compute_score_matrix(model::AbstractDixonColesPlayerModels, params; max_goals::Int=12)
    
    T1, T2, Rho = params.θ_1, params.θ_2, params.ρ
    n_samples = length(T1)
    
    S = zeros(Float64, max_goals, max_goals, n_samples)
    
    # Pre-allocate arrays for marginal log-probs to avoid re-calculating inside the grid
    log_h_marg = zeros(Float64, max_goals)
    log_a_marg = zeros(Float64, max_goals)
    
    @inbounds for k in 1:n_samples
        # 1. Unpack Parameters for this sample
        θ_h = T1[k] # Log-Rate Home
        θ_a = T2[k] # Log-Rate Away
        ρ   = Rho[k]
        
        # 2. Derive Rates
        λ = exp(θ_h)
        μ = exp(θ_a)
        
        # 3. Pre-calculate Independent Marginals (Optimization)
        dist_h = Poisson(λ)
        dist_a = Poisson(μ)
        
        for i in 1:max_goals
            log_h_marg[i] = logpdf(dist_h, i - 1)
        end
        for j in 1:max_goals
            log_a_marg[j] = logpdf(dist_a, j - 1)
        end
        
        # 4. Fill Grid with Dixon-Coles Correction
        for j in 1:max_goals
            a_score = j - 1
            lp_a = log_a_marg[j]
            
            for i in 1:max_goals
                h_score = i - 1
                lp_h = log_h_marg[i]
                
                # Base Independent Log-Probability
                log_p_base = lp_h + lp_a
                
                # Calculate Tau (Correction Factor)
                # Derived from Dixon & Coles (1997) Eq 4.2 
                tau = 1.0
                
                if h_score == 0 && a_score == 0
                    tau = 1.0 - (λ * μ * ρ)
                elseif h_score == 1 && a_score == 0
                    tau = 1.0 + (μ * ρ)
                elseif h_score == 0 && a_score == 1
                    tau = 1.0 + (λ * ρ)
                elseif h_score == 1 && a_score == 1
                    tau = 1.0 - ρ
                end
                
                # Safety: If constraint violated (tau <= 0), prob is 0.
                if tau <= 0
                    S[i, j, k] = 0.0
                else
                    # P(x,y) = P_indep(x,y) * tau
                    # log P = log P_base + log(tau)
                    S[i, j, k] = exp(log_p_base + log(tau))
                end
            end
        end
    end
    
    return ScoreMatrix(S)
end
