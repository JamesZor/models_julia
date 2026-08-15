# current_development/manager_wealth/l03_manager_wealth_predict.jl
#
# ==============================================================================
# LOADER: Prediction Dispatch for Manager & Wealth-Enabled Engine
# ==============================================================================
#
# PURPOSE:
#   Implements `Predictions` method overloads for
#   `DynamicSmileDoublePoissonXGWealthManagerPlayerTimeDecayModel`.
#
# ==============================================================================

using Distributions

const Pred = BayesianFootball.Predictions

# 1. Adapter: Row -> Params
Pred.extract_params(::DynamicSmileDoublePoissonXGWealthManagerPlayerTimeDecayModel, row) =
    (λ_h = row.λ_h, λ_a = row.λ_a, λ_tot = row.λ_tot, φ = row.φ)

# Grid kernel: independent Double Poisson
function _smile_poisson_grid_mgr_wealth(λ_h, λ_a; max_goals::Int=12)
    n = length(λ_h)
    S = zeros(Float64, max_goals, max_goals, n)
    p_h = zeros(Float64, max_goals); p_a = zeros(Float64, max_goals)
    goals = 0:(max_goals-1)
    @inbounds for k in 1:n
        @. p_h = pdf(Poisson(λ_h[k]), goals)
        @. p_a = pdf(Poisson(λ_a[k]), goals)
        for j in 1:max_goals
            pj = p_a[j]
            for i in 1:max_goals
                S[i, j, k] = p_h[i] * pj
            end
        end
    end
    return Pred.ScoreMatrix(S)
end

# 2. Kernel: params -> SmileScoreMatrix
function Pred.compute_score_matrix(
    ::DynamicSmileDoublePoissonXGWealthManagerPlayerTimeDecayModel, params; max_goals::Int=12
)
    grid = _smile_poisson_grid_mgr_wealth(params.λ_h, params.λ_a; max_goals)
    Λ = transpose(params.λ_tot .* params.φ) # (n_samples × nK)' -> (nK × n_samples)
    return Pred.SmileScoreMatrix(grid, Matrix{Float64}(Λ))
end
