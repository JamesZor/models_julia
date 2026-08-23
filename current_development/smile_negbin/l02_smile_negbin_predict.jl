# current_development/smile_negbin/l02_smile_negbin_predict.jl
#
# LOADER. Prediction-side wiring for DynamicSmileDoubleNegBinXGOutfieldPlayerTimeDecayModel.
#
# Requires l01_smile_negbin_engine.jl to be included first (it defines the model type and the
# PreGame / Pred / Features / Data aliases).
#
# ---------------------------------------------------------------------------------------------
# HOW LITTLE THIS HAS TO DO, AND WHY
# ---------------------------------------------------------------------------------------------
#
# `predict_row` (src/predictions/inference.jl:15) is three lines:
#
#     params = extract_params(model, row)
#     S      = compute_score_matrix(model, params)
#     compute_market_probs(S, market)  for each market
#
# The first two dispatch on the MODEL; the third dispatches on the SCORE-MATRIX CONTAINER. So the
# only things this file must supply are the two model-dispatched methods. As long as
# `compute_score_matrix` returns a `Pred.SmileScoreMatrix`, the O/U-via-smile pricing rule
# (`compute_market_probs(S::SmileScoreMatrix, ::MarketOverUnder)`, smile_poisson.jl:66) and the
# catch-all `(S::SmileScoreMatrix, ::AbstractMarket) -> compute_market_probs(S.grid, m)` fire
# unmodified. 1X2 / BTTS / correct-score in `market_inference/` read only `S.data` and never ask
# what distribution built the grid. Verified by reading those dispatch signatures, not assumed.
#
# ---------------------------------------------------------------------------------------------
# WHY THE O/U ROUTE STAYS A POISSON CDF — ON PURPOSE
# ---------------------------------------------------------------------------------------------
#
# `compute_market_probs(::SmileScoreMatrix, ::MarketOverUnder)` prices under-K as
# `cdf(Poisson(λ_tot·φ(K)), K)`. That is NOT the goals likelihood leaking back in: Λ(K)=λ_tot·φ(K)
# is the MARKET-INVERTED per-strike intensity — the market's own total-goals dispersion is already
# encoded in the SHAPE φ(K), which is exactly what the smile pillar learns. Re-deriving totals as
# a NegBin here would double-count dispersion (once in φ, once in r) and would also break the
# apples-to-apples comparison against the Poisson parent, whose O/U prices come from this identical
# rule. So: the NegBin changes 1X2 / BTTS / correct-score (grid-priced), and leaves O/U alone.
#
# There is also no "sum two NegBins" problem to solve. Totals never come from summing the home and
# away marginals on this engine — that route was replaced by the market-anchored λ_tot·φ(K) object
# — so the fact that NegBin(a)+NegBin(b) is not NegBin unless both share p (never true here, since
# μ_h ≠ μ_a) simply never arises.
#
# ---------------------------------------------------------------------------------------------
# WHY EVERY DEFINITION IS FULLY QUALIFIED
# ---------------------------------------------------------------------------------------------
#
# These EXTEND `BayesianFootball.Predictions` methods. An unqualified `compute_score_matrix(...)`
# here would define a brand-new function in `Main` that the simulator never calls, and the model
# would silently fall through to the generic `AbstractNegBinModel` route instead — a plain
# `ScoreMatrix`, no smile, O/U priced off the grid. Same trap `orderbook_layer2/l07_route2.jl`
# documents for `BayesianFootball.Portfolio.keep`.
#
# The fall-through matters here specifically because `AbstractTimeDecayPlayerModel <:
# AbstractPlayerModel <: AbstractNegBinModel` (src/models/pregame/types.jl:21-30), so a generic
# NegBin route DOES exist for this type and WOULD quietly win if these methods were missing.
# Ours are strictly more specific, so they take precedence.

using Distributions

# 1. Adapter: DataFrame row (or NamedTuple) -> params.
#    The Poisson parent's four fields plus the two dispersion vectors emitted by l01's extractor.
Pred.extract_params(::DynamicSmileDoubleNegBinXGOutfieldPlayerTimeDecayModel, row) =
    (λ_h = row.λ_h, λ_a = row.λ_a, λ_tot = row.λ_tot, φ = row.φ,
     r_h = row.r_h, r_a = row.r_a)

# Grid kernel: independent double-NEGATIVE-BINOMIAL. Structurally identical to
# `_smile_poisson_grid` (smile_poisson.jl:38-56) — same loop order, same [i,j,k] layout, same
# `ScoreMatrix` wrapper — with `Poisson(λ)` replaced by `RobustNegativeBinomial(r, λ)`.
function _smile_negbin_grid(λ_h, λ_a, r_h, r_a; max_goals::Int=12)
    n = length(λ_h)
    S = zeros(Float64, max_goals, max_goals, n)
    p_h = zeros(Float64, max_goals); p_a = zeros(Float64, max_goals)
    goals = 0:(max_goals-1)
    @inbounds for k in 1:n
        @. p_h = pdf(RobustNegativeBinomial(r_h[k], λ_h[k]), goals)
        @. p_a = pdf(RobustNegativeBinomial(r_a[k], λ_a[k]), goals)
        for j in 1:max_goals
            pj = p_a[j]
            for i in 1:max_goals
                S[i, j, k] = p_h[i] * pj
            end
        end
    end
    return Pred.ScoreMatrix(S)
end

# 2. Kernel: params -> SmileScoreMatrix. The Λ construction and the wrapper are byte-for-byte the
#    parent's, which is what keeps the O/U pricing rule applicable without new code.
function Pred.compute_score_matrix(
    ::DynamicSmileDoubleNegBinXGOutfieldPlayerTimeDecayModel, params; max_goals::Int=12
)
    grid = _smile_negbin_grid(params.λ_h, params.λ_a, params.r_h, params.r_a; max_goals)
    # Λ^model(K) = λ_tot · φ(K), shape [nK × n_samples]. params.φ is [n_samples × nK].
    Λ = transpose(params.λ_tot .* params.φ)          # (n_samples × nK)' -> (nK × n_samples)
    return Pred.SmileScoreMatrix(grid, Matrix{Float64}(Λ))
end

# 3. Nothing further. `compute_market_probs` is container-dispatched and already covers this type
#    through `Pred.SmileScoreMatrix`. `get_latent_column_symbols` is likewise already correct: the
#    `AbstractNegBinModel` method (negativebinomial.jl:29) picks up :λ_h, :λ_a, :r_h, :r_a, which
#    is exactly what its two consumers (CRPS, RQR) read — neither touches the smile.
