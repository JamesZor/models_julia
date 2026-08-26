# ==============================================================================
# Model 00 — GATE 5 : SCORE MATRIX (Pure Poisson)
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# Gate 5 verifies the transformation from latent intensities (λ_h, λ_a) to
# Poisson joint score matrices and market probabilities:
#   5a: Dispatch to src/predictions/score_computation/poisson.jl
#   5b: Score matrix grid parity vs stock Poisson outer product
#   5c: Market identities (1X2, Over/Under, BTTS consistency)
#
# ==============================================================================

using BayesianFootball
using DataFrames
using Distributions
using Statistics
using Printf

const Predictions = BayesianFootball.Predictions
const Markets     = BayesianFootball.Data.Markets
const outcomes    = BayesianFootball.Data.outcomes


# ==============================================================================
# 1. Helpers
# ==============================================================================

function tp00_score_matrix(model::DynamicPoissonGoalsTimeDecayModel, row; max_goals::Int = 12)
    params = Predictions.extract_params(model, row)
    return Predictions.compute_score_matrix(model, params; max_goals = max_goals)
end

"""
    tp00_reference_grid(λ_h, λ_a, max_goals) -> Matrix

Independent Poisson outer-product scoreline grid.
"""
function tp00_reference_grid(λ_h::Real, λ_a::Real, max_goals::Int)
    dh = Poisson(λ_h)
    da = Poisson(λ_a)
    ph = [pdf(dh, h) for h in 0:(max_goals - 1)]
    pa = [pdf(da, a) for a in 0:(max_goals - 1)]
    return ph * pa'          # [home, away]
end


# ==============================================================================
# 2. GATE 5a — Dispatch
# ==============================================================================

function tp00_gate_score_dispatch(model::DynamicPoissonGoalsTimeDecayModel, row; max_goals::Int = 12)
    params = Predictions.extract_params(model, row)
    m = which(Predictions.compute_score_matrix, (typeof(model), typeof(params)))
    file = String(m.file)

    results = []
    push!(results, (
        name   = "compute_score_matrix dispatches to poisson.jl",
        pass   = occursin("poisson.jl", file),
        detail = "dispatches to $(basename(file)):$(m.line)",
    ))

    return results
end


# ==============================================================================
# 3. GATE 5b — The Grid
# ==============================================================================

function tp00_gate_score_grid(model::DynamicPoissonGoalsTimeDecayModel, df::DataFrame, contract::SLContract;
                             n_probe_rows::Int = 5, max_goals::Int = contract.max_goals)
    results = []
    max_grid_diff = 0.0

    for row in eachrow(first(df, n_probe_rows))
        sm = tp00_score_matrix(model, row; max_goals = max_goals)
        n_samples = length(row.λ_h)
        for k in 1:min(n_samples, 25)
            ref = tp00_reference_grid(row.λ_h[k], row.λ_a[k], max_goals)
            grid_slice = sm.matrix[:, :, k]
            max_grid_diff = max(max_grid_diff, maximum(abs.(grid_slice .- ref)))
        end
    end

    push!(results, (
        name   = "grid parity vs stock Poisson",
        pass   = max_grid_diff <= 1e-12,
        detail = @sprintf("max |ΔP| = %.3e", max_grid_diff),
    ))

    # Orientation & moments
    first_row = first(df)
    sm1 = tp00_score_matrix(model, first_row; max_goals = max_goals)
    goals = 0:(max_goals - 1)

    mean_h = mean(sum(sm1.matrix[h+1, a+1, :] .* h for h in goals, a in goals))
    mean_a = mean(sum(sm1.matrix[h+1, a+1, :] .* a for h in goals, a in goals))
    expected_h = mean(first_row.λ_h)
    expected_a = mean(first_row.λ_a)

    push!(results, (
        name   = "orientation [home, away]",
        pass   = (mean_h > mean_a) == (expected_h > expected_a),
        detail = @sprintf("E[home]=%.3f vs λ_h=%.3f, E[away]=%.3f vs λ_a=%.3f", mean_h, expected_h, mean_a, expected_a),
    ))

    trunc_mass = 1.0 - mean(sum(sm1.matrix, dims=(1, 2)))
    push!(results, (
        name   = "truncation mass measured",
        pass   = trunc_mass <= 1e-3,
        detail = @sprintf("truncation mass at max_goals=%d: %.3e", max_goals, trunc_mass),
    ))

    return results
end


# ==============================================================================
# 4. GATE 5c — Market Identities
# ==============================================================================

function tp00_gate_market_identities(model::DynamicPoissonGoalsTimeDecayModel, df::DataFrame, contract::SLContract;
                                     max_goals::Int = contract.max_goals)
    results = []
    first_row = first(df)
    sm = tp00_score_matrix(model, first_row; max_goals = max_goals)
    grid = mean(sm.matrix, dims=3)[:, :, 1]
    grid_sum = sum(grid)

    p_1x2 = Markets.calculate_1x2_probabilities(sm)
    diff_1x2 = abs(sum(p_1x2) - grid_sum)
    push!(results, (
        name   = "1X2 partition sum matches grid",
        pass   = diff_1x2 <= 1e-10,
        detail = @sprintf("|1X2_sum - grid_sum| = %.3e", diff_1x2),
    ))

    p_btts = Markets.calculate_btts_probabilities(sm)
    diff_btts = abs(sum(p_btts) - grid_sum)
    push!(results, (
        name   = "BTTS partition sum matches grid",
        pass   = diff_btts <= 1e-10,
        detail = @sprintf("|btts_sum - grid_sum| = %.3e", diff_btts),
    ))

    ou_ok = true
    for line in contract.totals_lines
        p_ou = Markets.calculate_over_under_probabilities(sm, line)
        if abs(sum(p_ou) - grid_sum) > 1e-10
            ou_ok = false
        end
    end
    push!(results, (
        name   = "Over/Under partition sum matches grid",
        pass   = ou_ok,
        detail = "consistent across lines $(contract.totals_lines)",
    ))

    return results
end


# ==============================================================================
# 5. Market Summary Display
# ==============================================================================

function tp00_market_summary(model::DynamicPoissonGoalsTimeDecayModel, df::DataFrame, contract::SLContract; n_rows::Int = 8)
    println()
    println("-" ^ 74)
    println("SAMPLE PREDICTED PRICES (Model 00 Pure Poisson)")
    println("-" ^ 74)
    println("  λ_h    λ_a    Home    Draw    Away    Over 2.5  BTTS")
    println("-" ^ 74)
    for row in eachrow(first(df, n_rows))
        sm = tp00_score_matrix(model, row; max_goals = contract.max_goals)
        p_1x2 = Markets.calculate_1x2_probabilities(sm)
        p_ou  = Markets.calculate_over_under_probabilities(sm, 2.5)
        p_btts= Markets.calculate_btts_probabilities(sm)

        @printf("  %.3f  %.3f  %.3f   %.3f   %.3f   %.3f     %.3f\n",
                mean(row.λ_h), mean(row.λ_a),
                p_1x2[1], p_1x2[2], p_1x2[3],
                p_ou[1], p_btts[1])
    end
    println("-" ^ 74)
    return nothing
end
