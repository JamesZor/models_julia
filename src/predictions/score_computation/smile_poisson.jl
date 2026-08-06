# src/predictions/score_computation/smile_poisson.jl
#
# Prediction path for the LOCAL-INTENSITY SMILE double-Poisson engine
# (DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel).
#
# The smile is a PRICING object: per-line O/U is priced with its OWN per-strike intensity
# Λ^model(K)=λ_tot·φ(K) via P(N≤K)=cdf(Poisson(Λ^model(K)),K); everything else (1X2 / BTTS /
# correct-score) prices from the unchanged (λ_h,λ_a) score grid. A dedicated SmileScoreMatrix
# carries BOTH so the O/U route uses φ and the rest does not — do NOT collapse this into the
# plain Poisson Union route (that would silently price O/U without φ and "de-smile" the model).

using Distributions
using ..Models.PreGame: DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel,
                        DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel
using ..Data: MarketOverUnder, AbstractMarket, outcomes

# All engines that price O/U through the smile (λ_tot·φ(K)) — extend this Union when a new
# smile engine graduates; do NOT let a smile engine fall through to a plain grid route.
const AbstractSmilePoissonEngines = Union{
    DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel,
    DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel,
}

# Carry the (λ_h,λ_a) grid PLUS per-sample per-strike model intensities Λ^model(K)=λ_tot·φ(K).
struct SmileScoreMatrix <: AbstractScoreMatrix
    grid::ScoreMatrix                # [max_goals × max_goals × n_samples] for 1X2/BTTS/correct-score
    Λ::Matrix{Float64}               # [nK × n_samples] per-strike total intensity (K = row-1)
end

score_matrix_data(sm::SmileScoreMatrix) = sm.grid.data


# 1. Adapter: DataFrame Row -> NamedTuple
extract_params(::AbstractSmilePoissonEngines, row) =
    (λ_h = row.λ_h, λ_a = row.λ_a, λ_tot = row.λ_tot, φ = row.φ)

# Grid kernel (independent double-Poisson), identical math to score_computation/poisson.jl.
function _smile_poisson_grid(λ_h, λ_a; max_goals::Int=12)
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
    return ScoreMatrix(S)
end

# 2. Kernel: Params -> SmileScoreMatrix
function compute_score_matrix(::AbstractSmilePoissonEngines, params; max_goals::Int=12)
    grid = _smile_poisson_grid(params.λ_h, params.λ_a; max_goals)
    # Λ^model(K) = λ_tot · φ(K), shape [nK × n_samples]. params.φ is [n_samples × nK].
    Λ = transpose(params.λ_tot .* params.φ)          # (n_samples × nK)' -> (nK × n_samples)
    return SmileScoreMatrix(grid, Matrix{Float64}(Λ))
end

# 3a. Per-line O/U pricing via the smile: P(N≤K)=cdf(Poisson(Λ^model(K)),K). Fall back to the
#     grid for strikes beyond the learned smile range (nK strikes: K=0..Kmax).
function compute_market_probs(S::SmileScoreMatrix, m::MarketOverUnder)
    K = Int(floor(m.line))
    nK = size(S.Λ, 1)
    if K < 0 || K + 1 > nK
        return compute_market_probs(S.grid, m)   # outside learned smile -> grid
    end
    n = size(S.Λ, 2)
    under = Vector{Float64}(undef, n)
    @inbounds for s in 1:n
        under[s] = cdf(Poisson(S.Λ[K + 1, s]), K)
    end
    over = 1.0 .- under
    keys = outcomes(m)
    return Dict(keys.over => over, keys.under => under)
end

# 3b. Everything else (1X2 / BTTS / correct-score) prices from the unchanged grid.
compute_market_probs(S::SmileScoreMatrix, m::AbstractMarket) =
    compute_market_probs(S.grid, m)
