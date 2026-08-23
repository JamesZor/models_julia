# src/predictions/score_computation/recombination.jl
#
# Score computation & discrete Poisson convolution for Recombination Models:
#   - DynamicPxGRecombModel
#   - DynamicRecombinedGoalsModel

using Distributions
using ..Models.PreGame: DynamicPxGRecombModel, DynamicRecombinedGoalsModel

const AbstractRecombinationModels = Union{
    DynamicPxGRecombModel,
    DynamicRecombinedGoalsModel
}

# 1. Adapter: Latent DataFrame Row -> NamedTuple of Sample Vectors
function extract_params(model::AbstractRecombinationModels, row)
    if hasproperty(row, :λ_h) && hasproperty(row, :λ_a)
        return (λ_h = row.λ_h, λ_a = row.λ_a)
    elseif hasproperty(row, :μ_total_h) && hasproperty(row, :μ_total_a)
        return (λ_h = row.μ_total_h, λ_a = row.μ_total_a)
    elseif hasproperty(row, :μ_open_h) && hasproperty(row, :μ_open_a)
        # Reconstruct total rate if individual components are present
        κ_h = hasproperty(row, :κ_h) ? row.κ_h : ones(Float64, length(row.μ_open_h))
        κ_a = hasproperty(row, :κ_a) ? row.κ_a : ones(Float64, length(row.μ_open_a))
        q_pen = hasproperty(row, :q_pen) ? row.q_pen : fill(0.768, length(row.μ_open_h))
        λ_pen_h = hasproperty(row, :λ_pen_h) ? row.λ_pen_h : fill(0.207, length(row.μ_open_h))
        λ_pen_a = hasproperty(row, :λ_pen_a) ? row.λ_pen_a : fill(0.207, length(row.μ_open_h))
        λ_og = hasproperty(row, :λ_og) ? row.λ_og : fill(0.0276, length(row.μ_open_h))

        λ_h = (row.μ_open_h .* κ_h) .+ (q_pen .* λ_pen_h) .+ λ_og
        λ_a = (row.μ_open_a .* κ_a) .+ (q_pen .* λ_pen_a) .+ λ_og
        return (λ_h = λ_h, λ_a = λ_a)
    else
        error("Unable to extract recombination rate parameters from row: $(propertynames(row))")
    end
end

# 2. Kernel: Params -> ScoreMatrix (exact discrete Poisson convolution)
function compute_score_matrix(model::AbstractRecombinationModels, params; max_goals::Real=12)
    λ_h, λ_a = params.λ_h, params.λ_a
    n_samples = length(λ_h)
    S = zeros(Float64, max_goals, max_goals, n_samples)

    p_h = zeros(Float64, max_goals)
    p_a = zeros(Float64, max_goals)
    goals = 0:(max_goals-1)

    @inbounds for k in 1:n_samples
        d_h = Poisson(max(1e-6, λ_h[k]))
        d_a = Poisson(max(1e-6, λ_a[k]))

        @. p_h = pdf(d_h, goals)
        @. p_a = pdf(d_a, goals)

        # Normalize 1D vectors to sum to 1.0 within max_goals truncation
        sum_h = sum(p_h)
        sum_a = sum(p_a)
        if sum_h > 0.0
            p_h ./= sum_h
        end
        if sum_a > 0.0
            p_a ./= sum_a
        end

        for j in 1:max_goals
            pj = p_a[j]
            for i in 1:max_goals
                S[i, j, k] = p_h[i] * pj
            end
        end
    end

    return ScoreMatrix(S)
end

function compute_score_matrix(model::AbstractRecombinationModels, row::DataFrameRow; max_goals::Real=12)
    params = extract_params(model, row)
    return compute_score_matrix(model, params; max_goals=max_goals)
end
