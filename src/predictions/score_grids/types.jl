"The score-grid truncation used by the legacy prediction kernels."
const TPL_MAX_GOALS = 12

"""Reusable marginal-PMF workspace for typed posterior score-grid kernels."""
struct GridWorkspace
    p_h::Vector{Float64}
    p_a::Vector{Float64}
    max_goals::Int

    function GridWorkspace(max_goals::Integer = TPL_MAX_GOALS)
        max_goals > 0 || error("GridWorkspace: max_goals must be positive, got $max_goals.")
        n = Int(max_goals)
        return new(zeros(Float64, n), zeros(Float64, n), n)
    end
end

"Allocate one `(home goals × away goals × draws)` destination grid."
alloc_score_grid(l::AbstractPosteriorLatents, max_goals::Integer = TPL_MAX_GOALS) =
    Array{Float64, 3}(undef, Int(max_goals), Int(max_goals), n_draws(l))

@inline function _tpl_check_target(S::Array{Float64,3}, ws::GridWorkspace,
                                   l::AbstractPosteriorLatents, i::Int)
    1 <= i <= n_matches(l) ||
        error("fixture index $i is out of range 1:$(n_matches(l)).")
    size(S) == (ws.max_goals, ws.max_goals, n_draws(l)) || error(
        "destination grid is $(size(S)); expected " *
        "$((ws.max_goals, ws.max_goals, n_draws(l))). Use `alloc_score_grid`.")
    return nothing
end

"One fixture's ordinary score grid and smile-specific over/under curve."
struct SmileScoreGrid
    grid::Array{Float64,3}
    λ_tot::Vector{Float64}
    φ::Matrix{Float64}
    strikes::Vector{Float64}
end

"Allocate the per-fixture smile buffers for reuse across fixtures."
alloc_smile_buffers(l::SmileLatents) = (;
    λ_tot = Vector{Float64}(undef, n_draws(l)),
    φ = Matrix{Float64}(undef, n_strikes(l), n_draws(l)),
)

market_keys(m::Market1X2) = (outcomes(m).home, outcomes(m).draw, outcomes(m).away)
market_keys(m::MarketBTTS) = (outcomes(m).yes, outcomes(m).no)
market_keys(m::MarketOverUnder) = (outcomes(m).over, outcomes(m).under)
market_arity(m) = length(market_keys(m))

"Allocate a concretely typed tuple of market-outcome vectors."
alloc_market_book(market, n_draws::Integer) =
    ntuple(_ -> Vector{Float64}(undef, Int(n_draws)), market_arity(market))
