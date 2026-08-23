# current_development/manager_pace_scalar/l03_manager_pace_predict.jl
#
# ==============================================================================
# LOADER: Prediction Dispatch for Scalar Manager Pace Engine
# ==============================================================================
#
# PURPOSE:
#   Implements `Predictions` method overloads for
#   `DynamicSmileDoublePoissonXGWealthManagerPaceModel`.
#
# ==============================================================================

using Distributions

const Pred = BayesianFootball.Predictions

# 1. Adapter: Row -> Params
Pred.extract_params(::DynamicSmileDoublePoissonXGWealthManagerPaceModel, row) =
    (λ_h = row.λ_h, λ_a = row.λ_a, λ_tot = row.λ_tot, φ = row.φ)

# Grid kernel: independent Double Poisson with safety clamping & normalization
function _smile_poisson_grid_mgr_pace(λ_h, λ_a; max_goals::Int=12)
    n = length(λ_h)
    S = zeros(Float64, max_goals, max_goals, n)
    p_h = zeros(Float64, max_goals); p_a = zeros(Float64, max_goals)
    goals = 0:(max_goals-1)
    @inbounds for k in 1:n
        lh = clamp(coalesce(λ_h[k], 1.2), 1e-4, 15.0)
        la = clamp(coalesce(λ_a[k], 1.0), 1e-4, 15.0)
        @. p_h = pdf(Poisson(lh), goals)
        @. p_a = pdf(Poisson(la), goals)
        sum_p = 0.0
        for j in 1:max_goals
            pj = p_a[j]
            for i in 1:max_goals
                p_val = p_h[i] * pj
                S[i, j, k] = p_val
                sum_p += p_val
            end
        end
        if sum_p > 0.0
            S[:, :, k] ./= sum_p
        else
            S[1, 1, k] = 1.0
        end
    end
    return Pred.ScoreMatrix(S)
end

# 2. Kernel: params -> SmileScoreMatrix
function Pred.compute_score_matrix(
    ::DynamicSmileDoublePoissonXGWealthManagerPaceModel, params; max_goals::Int=12
)
    grid = _smile_poisson_grid_mgr_pace(params.λ_h, params.λ_a; max_goals)
    Λ = transpose(params.λ_tot .* params.φ) # (n_samples × nK)' -> (nK × n_samples)
    return Pred.SmileScoreMatrix(grid, Matrix{Float64}(Λ))
end
