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
    file = basename(String(m.file))

    results = Any[]
    push!(results, (
        name   = "model subtypes AbstractPoissonModel",
        pass   = model isa BayesianFootball.TypesInterfaces.AbstractPoissonModel,
        detail = string(typeof(model).name.name, " <: ", supertype(typeof(model))),
    ))

    push!(results, (
        name   = "compute_score_matrix dispatches to poisson.jl",
        pass   = file == "poisson.jl",
        detail = "dispatches to $(file):$(m.line)",
    ))

    S = Predictions.compute_score_matrix(model, params; max_goals = max_goals)
    push!(results, (
        name   = "grid shape",
        pass   = size(S.data, 1) == max_goals && size(S.data, 2) == max_goals,
        detail = "$(size(S.data)) — [home, away, draws], covering 0..$(max_goals-1) goals",
    ))

    return results
end


# ==============================================================================
# 3. GATE 5b — The Grid
# ==============================================================================

function tp00_gate_score_grid(model::DynamicPoissonGoalsTimeDecayModel, df::AbstractDataFrame, contract::SLContract;
                             n_rows::Int = 5, tol::Float64 = 1e-12)
    mg      = contract.max_goals
    results = Any[]

    worst_parity = 0.0
    worst_trunc  = 0.0
    min_cell     = Inf
    n_checked    = 0

    for row in eachrow(first(df, n_rows))
        S = tp00_score_matrix(model, row; max_goals = mg)
        n_draws = size(S.data, 3)

        for k in 1:min(n_draws, 25)
            grid = S.data[:, :, k]
            ref  = tp00_reference_grid(row.λ_h[k], row.λ_a[k], mg)

            worst_parity = max(worst_parity, maximum(abs.(grid .- ref)))
            worst_trunc  = max(worst_trunc, 1.0 - sum(grid))
            min_cell     = min(min_cell, minimum(grid))
            n_checked   += 1
        end
    end

    push!(results, (
        name   = "grid parity vs documented distribution",
        pass   = worst_parity <= tol,
        detail = @sprintf("max |ΔP| = %.3e over %d grids", worst_parity, n_checked),
    ))

    push!(results, (
        name   = "all cells non-negative",
        pass   = min_cell >= 0.0,
        detail = @sprintf("min cell = %.3e", min_cell),
    ))

    # Truncation check
    hot = df[argmax([mean(r.λ_h) + mean(r.λ_a) for r in eachrow(df)]), :]
    o35 = outcomes(Markets.MarketOverUnder(3.5))
    p_at(g) = mean(Predictions.compute_market_probs(
                       tp00_score_matrix(model, hot; max_goals = g),
                       Markets.MarketOverUnder(3.5))[o35.over])
    price_shift = abs(p_at(mg) - p_at(mg + 8))

    push!(results, (
        name   = "truncation costs nothing that matters",
        pass   = price_shift <= 1e-4,
        detail = @sprintf("P(over 3.5) moves %.3e widening %d→%d goals (worst fixture, λ_tot %.2f); mass %.3e",
                          price_shift, mg, mg + 8,
                          mean(hot.λ_h) + mean(hot.λ_a), worst_trunc),
    ))

    # Orientation & moments
    row1 = first(eachrow(df))
    S1   = tp00_score_matrix(model, row1; max_goals = mg)
    g1   = S1.data[:, :, 1]

    goals      = collect(0:(mg - 1))
    e_home     = sum(goals .* vec(sum(g1, dims = 2)))
    e_away     = sum(goals .* vec(sum(g1, dims = 1)))
    λ_h1, λ_a1 = row1.λ_h[1], row1.λ_a[1]
    separation = abs(λ_h1 - λ_a1)

    push!(results, (
        name   = "grid orientation [home, away]",
        pass   = abs(e_home - λ_h1) < 0.05 * separation &&
                 abs(e_away - λ_a1) < 0.05 * separation,
        detail = @sprintf("E[home] %.4f vs λ_h %.4f | E[away] %.4f vs λ_a %.4f | marginals separated by %.4f",
                          e_home, λ_h1, e_away, λ_a1, separation),
    ))

    dh = Poisson(λ_h1)
    da = Poisson(λ_a1)
    e_home_expected = sum(k * pdf(dh, k) for k in 0:(mg - 1)) *
                      sum(pdf(da, a)     for a in 0:(mg - 1))
    moment_err = abs(e_home - e_home_expected)

    push!(results, (
        name   = "moment matches the truncated distribution exactly",
        pass   = moment_err <= 1e-12,
        detail = @sprintf("E[home] %.6f vs truncated expectation %.6f, |Δ| = %.3e (λ_h %.4f, shortfall %.3e is truncation)",
                          e_home, e_home_expected, moment_err, λ_h1, λ_h1 - e_home),
    ))

    return results
end


# ==============================================================================
# 4. GATE 5c — Market Identities
# ==============================================================================

function tp00_gate_market_identities(model::DynamicPoissonGoalsTimeDecayModel, df::AbstractDataFrame, contract::SLContract;
                                     n_rows::Int = 5, tol::Float64 = 1e-12)
    mg      = contract.max_goals
    lines   = contract.totals_lines
    results = Any[]

    all(l -> !isinteger(l), lines) || error("integer O/U line in contract: pushes break the identity")

    worst_1x2   = 0.0
    worst_btts  = 0.0
    worst_ou    = 0.0
    worst_cross = 0.0
    worst_mono  = 0.0
    worst_ouref = 0.0
    n_checked   = 0

    for row in eachrow(first(df, n_rows))
        S = tp00_score_matrix(model, row; max_goals = mg)

        p_1x2  = Predictions.compute_market_probs(S, Markets.Market1X2())
        p_btts = Predictions.compute_market_probs(S, Markets.MarketBTTS())
        p_ou   = Dict(l => Predictions.compute_market_probs(S, Markets.MarketOverUnder(l)) for l in lines)

        for k in 1:min(size(S.data, 3), 25)
            mass = sum(S.data[:, :, k])

            s_1x2  = p_1x2[:home][k] + p_1x2[:draw][k] + p_1x2[:away][k]
            s_btts = p_btts[:btts_yes][k] + p_btts[:btts_no][k]

            worst_1x2   = max(worst_1x2,  abs(s_1x2  - mass))
            worst_btts  = max(worst_btts, abs(s_btts - mass))
            worst_cross = max(worst_cross, abs(s_1x2 - s_btts))

            unders = Float64[]
            for l in lines
                o = outcomes(Markets.MarketOverUnder(l))
                s_ou = p_ou[l][o.over][k] + p_ou[l][o.under][k]
                worst_ou    = max(worst_ou, abs(s_ou - mass))
                worst_cross = max(worst_cross, abs(s_ou - s_1x2))
                push!(unders, p_ou[l][o.under][k])

                grid = S.data[:, :, k]
                ref_under = sum(grid[r, c] for r in 1:mg, c in 1:mg if (r - 1) + (c - 1) < l)
                worst_ouref = max(worst_ouref, abs(ref_under - p_ou[l][o.under][k]))
            end

            for i in 1:(length(unders) - 1)
                worst_mono = max(worst_mono, unders[i] - unders[i + 1])
            end
            n_checked += 1
        end
    end

    push!(results, (
        name   = "1X2 partition sum matches grid",
        pass   = worst_1x2 <= tol,
        detail = @sprintf("max |Σ - mass| = %.3e over %d draws", worst_1x2, n_checked),
    ))

    push!(results, (
        name   = "BTTS partition sum matches grid",
        pass   = worst_btts <= tol,
        detail = @sprintf("max |Σ - mass| = %.3e", worst_btts),
    ))

    push!(results, (
        name   = "O/U partition sums match grid",
        pass   = worst_ou <= tol,
        detail = @sprintf("max |Σ - mass| = %.3e across lines %s", worst_ou, string(lines)),
    ))

    push!(results, (
        name   = "cross-family partition consistency",
        pass   = worst_cross <= tol,
        detail = @sprintf("max difference between market families = %.3e", worst_cross),
    ))

    push!(results, (
        name   = "O/U probabilities match direct cell sum",
        pass   = worst_ouref <= tol,
        detail = @sprintf("max |P_pricer - P_direct| = %.3e", worst_ouref),
    ))

    push!(results, (
        name   = "under(line) is monotone in line",
        pass   = worst_mono <= 0.0,
        detail = worst_mono <= 0.0 ? "P(under) strictly non-decreasing across $lines" :
                                     @sprintf("MONOTONICITY VIOLATED: decrease of %.3e", worst_mono),
    ))

    return results
end


# ==============================================================================
# 5. Market Summary Display
# ==============================================================================

function tp00_market_summary(model::DynamicPoissonGoalsTimeDecayModel, df::AbstractDataFrame, contract::SLContract; n_rows::Int = 8)
    rows = NamedTuple[]
    for row in eachrow(first(df, n_rows))
        S      = tp00_score_matrix(model, row; max_goals = contract.max_goals)
        p_1x2  = Predictions.compute_market_probs(S, Markets.Market1X2())
        p_btts = Predictions.compute_market_probs(S, Markets.MarketBTTS())
        o25    = outcomes(Markets.MarketOverUnder(2.5))
        p_25   = Predictions.compute_market_probs(S, Markets.MarketOverUnder(2.5))

        push!(rows, (
            match_id = row.match_id,
            λ_h      = round(mean(row.λ_h), digits = 3),
            λ_a      = round(mean(row.λ_a), digits = 3),
            home     = round(mean(p_1x2[:home]),      digits = 3),
            draw     = round(mean(p_1x2[:draw]),      digits = 3),
            away     = round(mean(p_1x2[:away]),      digits = 3),
            over25   = round(mean(p_25[o25.over]),    digits = 3),
            btts     = round(mean(p_btts[:btts_yes]), digits = 3),
        ))
    end
    return DataFrame(rows)
end

function tp00_market_summary(model::DynamicPoissonGoalsTimeDecayModel, df::DataFrame, contract::SLContract; n_rows::Int = 8)
    return invoke(tp00_market_summary, Tuple{DynamicPoissonGoalsTimeDecayModel, AbstractDataFrame, SLContract}, model, df, contract; n_rows = n_rows)
end
