# ==============================================================================
# Gate 5 — score matrix
# ==============================================================================
# Distribution-specific work belongs to the adapter.  This file checks only the
# public score-grid contract: [home goals, away goals, posterior draw].

using DataFrames
using Statistics
using Printf

const SLMarkets = BayesianFootball.Data.Markets
const SLPredictions = BayesianFootball.Predictions
const SLData = BayesianFootball.Data

"""Return the markets declared by the protocol contract."""
function sl_book_markets(contract::SLContract)
    markets = Any[SLMarkets.Market1X2(), SLMarkets.MarketBTTS()]
    append!(markets, (SLMarkets.MarketOverUnder(line) for line in contract.totals_lines))
    return markets
end

"""Price one latent row through the adapter-owned score-matrix route."""
function sl_score_matrix(adapter::AbstractSLModelAdapter, row; max_goals::Int = 12)
    params = sl_extract_params(adapter, row)
    return sl_compute_score_matrix(adapter, params; max_goals = max_goals)
end

"""Get score-grid storage through the generic Predictions accessor."""
_sl_grid(score) = SLPredictions.score_matrix_data(score)

"""Whether an adapter says its score grid is already probability-normalised."""
function _sl_grid_is_normalized(adapter::AbstractSLModelAdapter)
    caps = sl_capabilities(adapter)
    if hasproperty(caps, :score_matrix_normalized)
        return getproperty(caps, :score_matrix_normalized)
    elseif hasproperty(caps, :normalize_score_matrix)
        return !getproperty(caps, :normalize_score_matrix)
    end
    return false
end

"""Gate 5a: resolved adapter dispatch and public grid dimensions."""
function sl_gate_score_dispatch(adapter::AbstractSLModelAdapter, row; max_goals::Int = 12)
    params = sl_extract_params(adapter, row)
    method = which(sl_compute_score_matrix, (typeof(adapter), typeof(params)))
    caps = sl_capabilities(adapter)
    expected = hasproperty(caps, :expected_score_dispatch) ? caps.expected_score_dispatch : nothing
    grid = _sl_grid(sl_score_matrix(adapter, row; max_goals = max_goals))

    checks = Any[
        sl_result("score-matrix adapter dispatch",
                  expected === nothing || basename(string(method.file)) == expected,
                  "$(basename(string(method.file))):$(method.line)"),
        sl_result("grid shape", ndims(grid) == 3 && size(grid, 1) == max_goals &&
                  size(grid, 2) == max_goals,
                  "$(size(grid)); [home, away, draws]"),
    ]
    return vcat(checks, sl_adapter_check(adapter, :score_dispatch, row, params))
end

"""
    sl_gate_score_grid(adapter, df, contract; n_rows=5, tol=1e-12)

Gate 5b: parity with the adapter's documented reference, non-negativity,
truncation price impact under grid widening, orientation, and the exact truncated
first-moment identity.  The adapter supplies the reference grid, so this framework
never assumes a particular count distribution.
"""
function sl_gate_score_grid(adapter::AbstractSLModelAdapter, df::AbstractDataFrame,
                            contract::SLContract; n_rows::Int = 5, tol::Float64 = 1e-12)
    isempty(df) && return [sl_result("score grid inputs", false, "empty")]
    mg = contract.max_goals
    parity = 0.0
    minimum_cell = Inf
    truncation = 0.0
    checked = 0

    for row in eachrow(first(df, min(n_rows, nrow(df))))
        grid = _sl_grid(sl_score_matrix(adapter, row; max_goals = mg))
        for draw in 1:min(size(grid, 3), 25)
            reference = sl_reference_grid(adapter, row, draw, mg)
            actual = grid[:, :, draw]
            parity = max(parity, maximum(abs.(actual .- reference)))
            minimum_cell = min(minimum_cell, minimum(actual))
            truncation = max(truncation, abs(1.0 - sum(actual)))
            checked += 1
        end
    end

    hot_index = argmax([sum(_sl_grid(sl_score_matrix(adapter, r; max_goals = mg))[:, :, 1]) == 0 ? 0.0 :
                        sum((i + j - 2) * mean(_sl_grid(sl_score_matrix(adapter, r; max_goals = mg))[i, j, :])
                            for i in 1:mg, j in 1:mg)
                        for r in eachrow(df)])
    hot = df[hot_index, :]
    over = SLData.outcomes(SLMarkets.MarketOverUnder(3.5)).over
    price_at(goals) = mean(SLPredictions.compute_market_probs(
        sl_score_matrix(adapter, hot; max_goals = goals), SLMarkets.MarketOverUnder(3.5))[over])
    widening_shift = abs(price_at(mg) - price_at(mg + 8))

    first_row = first(eachrow(df))
    first_grid = _sl_grid(sl_score_matrix(adapter, first_row; max_goals = mg))[:, :, 1]
    goals = collect(0:(mg - 1))
    home_moment = sum(goals .* vec(sum(first_grid, dims = 2)))
    away_moment = sum(goals .* vec(sum(first_grid, dims = 1)))
    reference = sl_reference_grid(adapter, first_row, 1, mg)
    reference_home = sum(goals .* vec(sum(reference, dims = 2)))
    reference_away = sum(goals .* vec(sum(reference, dims = 1)))
    orientation_error = max(abs(home_moment - reference_home), abs(away_moment - reference_away))

    return Any[
        sl_result("grid parity vs documented distribution", parity <= tol,
                  @sprintf("max |ΔP| = %.3e over %d grids", parity, checked)),
        sl_result("all cells non-negative", minimum_cell >= 0.0,
                  @sprintf("min cell = %.3e", minimum_cell)),
        sl_result("truncation costs nothing that matters", widening_shift <= 1e-4,
                  @sprintf("P(over 3.5) moves %.3e widening %d→%d; raw tail %.3e",
                           widening_shift, mg, mg + 8, truncation)),
        sl_result("grid orientation [home, away]", orientation_error <= tol,
                  @sprintf("truncated home/away moment error %.3e", orientation_error)),
        sl_result("moment matches the truncated distribution exactly", orientation_error <= tol,
                  @sprintf("first-moment identity error %.3e", orientation_error)),
    ]
end

"""Gate 5c: every direct-market family must partition precisely the same grid."""
function sl_gate_market_identities(adapter::AbstractSLModelAdapter, df::AbstractDataFrame,
                                   contract::SLContract; n_rows::Int = 5, tol::Float64 = 1e-12)
    (isempty(df) || isempty(contract.totals_lines)) &&
        return [sl_result("market identity inputs", false, "empty")]
    all(line -> !isinteger(line), contract.totals_lines) ||
        return [sl_result("non-push totals", false, "integer line")]

    one_error = 0.0; btts_error = 0.0; totals_error = 0.0
    cross_error = 0.0; direct_error = 0.0; monotonicity_error = 0.0
    for row in eachrow(first(df, min(n_rows, nrow(df))))
        score = sl_score_matrix(adapter, row; max_goals = contract.max_goals)
        grid = _sl_grid(score)
        one = SLPredictions.compute_market_probs(score, SLMarkets.Market1X2())
        btts = SLPredictions.compute_market_probs(score, SLMarkets.MarketBTTS())
        totals = Dict(line => SLPredictions.compute_market_probs(score, SLMarkets.MarketOverUnder(line))
                      for line in contract.totals_lines)
        for draw in axes(grid, 3)
            mass = sum(grid[:, :, draw])
            one_sum = one[:home][draw] + one[:draw][draw] + one[:away][draw]
            btts_sum = btts[:btts_yes][draw] + btts[:btts_no][draw]
            one_error = max(one_error, abs(one_sum - mass))
            btts_error = max(btts_error, abs(btts_sum - mass))
            cross_error = max(cross_error, abs(one_sum - btts_sum))
            unders = Float64[]
            for line in contract.totals_lines
                outcome = SLData.outcomes(SLMarkets.MarketOverUnder(line))
                quoted = totals[line]
                total_sum = quoted[outcome.over][draw] + quoted[outcome.under][draw]
                totals_error = max(totals_error, abs(total_sum - mass))
                cross_error = max(cross_error, abs(total_sum - one_sum))
                direct = sum(grid[h, a, draw] for h in axes(grid, 1), a in axes(grid, 2)
                             if h + a - 2 < line)
                direct_error = max(direct_error, abs(direct - quoted[outcome.under][draw]))
                push!(unders, quoted[outcome.under][draw])
            end
            monotonicity_error = max(monotonicity_error, -min(0.0, minimum(diff(unders))))
        end
    end
    return Any[
        sl_result("1X2 partitions the grid", one_error <= tol, @sprintf("%.3e", one_error)),
        sl_result("BTTS partitions the grid", btts_error <= tol, @sprintf("%.3e", btts_error)),
        sl_result("O/U partitions the grid", totals_error <= tol, @sprintf("%.3e", totals_error)),
        sl_result("families agree with each other", cross_error <= tol, @sprintf("%.3e", cross_error)),
        sl_result("O/U agrees with a direct cell sum", direct_error <= tol, @sprintf("%.3e", direct_error)),
        sl_result("under(line) non-decreasing in line", monotonicity_error <= tol,
                  @sprintf("%.3e", monotonicity_error)),
    ]
end
