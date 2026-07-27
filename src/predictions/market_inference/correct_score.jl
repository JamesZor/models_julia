# src/predictions/market_inference/correct_score.jl

using ..Data: MarketCorrectScore, outcomes

function compute_market_probs(S::ScoreMatrix, market::MarketCorrectScore)
    (max_h, max_a, n_samples) = size(S.data)

    # Exact-scoreline accumulators for the 0..3 x 0..3 grid.
    grid = Dict{Symbol,Vector{Float64}}()
    for h in 0:3, a in 0:3
        grid[Symbol("cs_", h, a)] = zeros(Float64, n_samples)
    end

    # "Any other" buckets: mass on scorelines outside the 4x4 grid, split by result.
    other_home = zeros(Float64, n_samples)  # home win, >=4 goals on a side
    other_draw = zeros(Float64, n_samples)  # draw    4-4, 5-5, ...
    other_away = zeros(Float64, n_samples)  # away win

    @inbounds for k in 1:n_samples
        for c in 1:max_a
            a_goals = c - 1
            for r in 1:max_h
                h_goals = r - 1
                p = S.data[r, c, k]

                if h_goals <= 3 && a_goals <= 3
                    grid[Symbol("cs_", h_goals, a_goals)][k] += p
                elseif h_goals > a_goals
                    other_home[k] += p
                elseif h_goals < a_goals
                    other_away[k] += p
                else
                    other_draw[k] += p
                end
            end
        end
    end

    grid[:cs_any_other_home] = other_home
    grid[:cs_any_other_draw] = other_draw
    grid[:cs_any_other_away] = other_away
    return grid
end
