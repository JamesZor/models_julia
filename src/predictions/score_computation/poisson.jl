# src/predictions/score_computation/poisson.jl

using Distributions
using ..Models.PreGame: DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel, DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel, DynamicDoublePoissonBigChanceOutfieldPlayerTimeDecayModel

const AbstractDoublePoissonPlayerModels = Union{
    AbstractPoissonModel,
    DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel,
    DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel,
    DynamicDoublePoissonBigChanceOutfieldPlayerTimeDecayModel
}

# 1. Adapter: DataFrame Row -> NamedTuple
function extract_params(model::AbstractDoublePoissonPlayerModels, row)
    # The latent dataframe columns (λ_h, λ_a) are Vectors of samples
    return (λ_h = row.λ_h, λ_a = row.λ_a)
end

# 2. Kernel: Params -> ScoreMatrix
function compute_score_matrix(model::AbstractDoublePoissonPlayerModels, params; max_goals::Real=12)
    λ_h, λ_a = params.λ_h, params.λ_a
    n_samples = length(λ_h)
    S = zeros(Float64, max_goals, max_goals, n_samples)
    
    # Pre-allocate temporary vectors for the current sample to avoid allocations inside the loop
    p_h = zeros(Float64, max_goals)
    p_a = zeros(Float64, max_goals)
    goals = 0:(max_goals-1)

    @inbounds for k in 1:n_samples
        # 1. Create distributions once per sample
        d_h = Poisson(λ_h[k])
        d_a = Poisson(λ_a[k])

        # 2. Compute PDF vectors once per sample (reduces calls from M*M to 2*M)
        # We use broadcast here to fill the pre-allocated buffers
        @. p_h = pdf(d_h, goals)
        @. p_a = pdf(d_a, goals)
        
        # 3. Outer Product
        # S[i, j, k] = p_h[i] * p_a[j]
        # Julia's column-major layout prefers iterating i (rows) then j (cols)
        for j in 1:max_goals
            pj = p_a[j] # Cache the away prob for this column
            for i in 1:max_goals
                S[i, j, k] = p_h[i] * pj
            end
        end
    end
    
    return ScoreMatrix(S)
end
