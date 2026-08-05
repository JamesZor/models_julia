# src/Portfolio/payoff.jl
#
# THE payoff morphism. Both the hypothetical payoff matrix `R` (what the allocator optimises
# against) and the realised settlement vector factor through this one function, which in turn
# defers to `Data.grade_selection` -- the same grader that produces `is_winner` in the odds
# pipeline.
#
# That single-sourcing is what makes "the win mask agrees with settlement" a testable identity
# (see P1 in test/portfolio_tests.jl) instead of a convention that drifts. The prototype built
# masks by string-matching market names and graded settlement separately; the two could and did
# disagree in principle, and a `missing` grade was silently booked as a full loss.

export payoff, payoff_matrix, settle_vector, grid_index

"""
    grid_index(h, a, max_h) -> Int

Row of the flattened score grid for scoreline `h`-`a`. The grid is `vec` of a `max_h x max_a`
matrix (home = rows, away = columns), so this must stay column-major to line up with
`vec(mean(score_matrix.data, dims=3))`.
"""
@inline grid_index(h::Integer, a::Integer, max_h::Integer) = a * max_h + h + 1

"""
    payoff(sel, h, a, commission) -> Float64

Per-unit payoff of `sel` if the match ends `h`-`a`:

* win       -> `net_return(commission, sel.odds_used)`
* push/void -> `0.0` (stake returned)
* loss      -> `-1.0`

`Data.grade_selection` returns `missing` for a genuine push (Draw-No-Bet on a draw, an Asian
line landing exactly on the margin) *and* for a market it does not know how to grade. Both are
treated as stake-returned here, which is right for the former; the latter is prevented upstream
by only admitting markets present in the `MarketConfig`.
"""
@inline function payoff(sel::Selection, h::Integer, a::Integer, c::AbstractCommissionModel)
    g = Data.grade_selection(sel.group, sel.line, sel.selection, Int(h), Int(a))
    ismissing(g) && return 0.0
    return g ? net_return(c, sel.odds_used) : -1.0
end

"""
    payoff_matrix(sels, max_h, max_a, commission) -> Matrix{Float64}

The Jacot return matrix `R` (`N x n`, `N = max_h * max_a`): wealth after staking `a` is
`1 .+ R * a`, so `R[w, j]` is the per-unit payoff of selection `j` in state `w`.
"""
function payoff_matrix(sels::Vector{Selection}, max_h::Integer, max_a::Integer,
                       c::AbstractCommissionModel)
    n = length(sels)
    R = zeros(Float64, max_h * max_a, n)
    @inbounds for j in 1:n, col in 1:max_a, row in 1:max_h
        R[grid_index(row - 1, col - 1, max_h), j] = payoff(sels[j], row - 1, col - 1, c)
    end
    return R
end

"""
    settle_vector(sels, h, a, commission) -> Vector{Float64}

Realised per-unit payoff of each selection at the actual scoreline. By construction this equals
`payoff_matrix(...)[grid_index(h, a, max_h), :]` whenever the scoreline is inside the grid.
"""
settle_vector(sels::Vector{Selection}, h::Integer, a::Integer, c::AbstractCommissionModel) =
    Float64[payoff(s, h, a, c) for s in sels]
