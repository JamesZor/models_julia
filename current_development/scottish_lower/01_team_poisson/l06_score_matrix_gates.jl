# ==============================================================================
# Model 01 — GATE 5 : SCORE MATRIX
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# Gate 4 established that the posterior λ and r reaching the pricer are the ones
# the model fitted. Gate 5 is the last translation step before money: λ and r
# become a joint distribution over scorelines, and that grid becomes market
# probabilities.
#
# Three distinct failure modes live here, and they need different tests:
#
#   5a  dispatch     is the RIGHT compute_score_matrix method being called?
#                    Engines are selected by abstract supertype and by Union
#                    membership. A new engine that nobody added to the relevant
#                    Union silently falls through to a default meant for a
#                    different likelihood.
#
#   5b  the grid     does the grid equal the documented distribution, is it
#                    oriented the right way round, and how much mass falls off
#                    the truncated edge?
#
#   5c  markets      do the market probabilities agree with sums over the same
#                    grid cells?
#
# On orientation: a transposed grid is the nastiest bug in this stage, because it
# produces perfectly well-formed probabilities that are simply the wrong way
# round. Nothing downstream can detect it — it looks like a badly calibrated
# model, not a broken one. It is caught here by the fact that γ > 0 makes the
# home and away marginals genuinely different.
#
# ==============================================================================

using BayesianFootball
using DataFrames
using Distributions
using Statistics
using Printf

const Predictions = BayesianFootball.Predictions
const Markets     = BayesianFootball.Data.Markets

# `outcomes` names a market's selection symbols (:home/:draw/:away, :btts_yes, ...).
# Qualified rather than imported: DataFrames also exports names that clash in Main.
const outcomes = BayesianFootball.Data.outcomes


# ==============================================================================
# 1. Helpers
# ==============================================================================

"""
    tp_score_matrix(model, row; max_goals) -> ScoreMatrix

Price one row of the extracted latents through the package's own pricer.

`max_goals` is read from the contract rather than left at the method default, so
that the truncation measured in 5b is the truncation the protocol actually uses.
"""
function tp_score_matrix(model, row; max_goals::Int = 12)
    params = Predictions.extract_params(model, row)
    return Predictions.compute_score_matrix(model, params; max_goals = max_goals)
end

"""
    tp_reference_grid(λ_h, λ_a, r_h, r_a, max_goals) -> Matrix

The scoreline grid as MODEL.md documents it, built from stock
`Distributions.NegativeBinomial` rather than from anything in `src`.

Two independent NegBin marginals, outer product. `DoubleNegativeBinomial` adds no
dependence term (`MyDistributions/double_negative_binomial.jl:31-39` is a plain
sum of two log-pdfs), and this configuration has no Dixon-Coles or copula
component, so home and away goals are conditionally independent given λ. If a
dependence term is ever added, this reference stops matching — deliberately.

Mean-parameterised NegBin maps to the standard `(r, p)` form as `p = r / (r + λ)`.
"""
function tp_reference_grid(λ_h::Real, λ_a::Real, r_h::Real, r_a::Real, max_goals::Int)
    dh = NegativeBinomial(r_h, r_h / (r_h + λ_h))
    da = NegativeBinomial(r_a, r_a / (r_a + λ_a))
    ph = [pdf(dh, h) for h in 0:(max_goals - 1)]
    pa = [pdf(da, a) for a in 0:(max_goals - 1)]
    return ph * pa'          # [home, away]
end


# ==============================================================================
# 2. GATE 5a — Dispatch
# ==============================================================================

"""
    tp_gate_score_dispatch(model, row) -> Vector

Which method actually runs, and is it the one this model's likelihood requires?

This exists because of a known failure mode in this repository: engines are
routed to a pricer by abstract supertype and by `Union` membership
(`predictions/score_computation/*.jl`). An engine omitted from the relevant Union
does not error at definition time — it falls through to whichever default its
supertype matches, and only fails later with a confusing message about NegBin
shape parameters, or worse, prices silently under the wrong likelihood.

So the gate asserts the resolved method by FILE, not just that a call succeeds.
"""
function tp_gate_score_dispatch(model, row; max_goals::Int = 12)
    params = Predictions.extract_params(model, row)

    m_score  = which(Predictions.compute_score_matrix, (typeof(model), typeof(params)))
    m_params = which(Predictions.extract_params,       (typeof(model), typeof(row)))
    file     = basename(string(m_score.file))

    results = Any[]

    push!(results, (
        name   = "model subtypes AbstractNegBinModel",
        pass   = model isa BayesianFootball.Models.PreGame.AbstractNegBinModel,
        detail = string(typeof(model).name.name, " <: ", supertype(typeof(model))),
    ))

    push!(results, (
        name   = "score matrix pricer",
        pass   = file == "negativebinomial.jl",
        detail = "$(file):$(m_score.line) — expected negativebinomial.jl",
    ))

    push!(results, (
        name   = "params reader",
        pass   = basename(string(m_params.file)) == "negativebinomial.jl",
        detail = "$(basename(string(m_params.file))):$(m_params.line)",
    ))

    # extract_params has two routes (:r versus :r_h/:r_a). Which one fired matters:
    # the wrong one would silently use one side's dispersion for both.
    push!(results, (
        name   = "dispersion route",
        pass   = hasproperty(row, :r_h) && hasproperty(row, :r_a),
        detail = hasproperty(row, :r) ? "shared :r column" : "separate :r_h / :r_a columns",
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
# 3. GATE 5b — The grid itself
# ==============================================================================

"""
    tp_gate_score_grid(model, df, contract; n_rows, tol) -> Vector

Parity against `tp_reference_grid`, orientation, non-negativity, and truncation.

Truncation is REPORTED as well as gated. `max_goals = 12` is a choice, and the
right way to defend it is the mass it discards, not the fact that it looked big
enough. At λ ≈ 1.4 the tail beyond 11 goals is negligible; at λ ≈ 4 it would not
be, and a future engine on a higher-scoring league inherits this same default.
"""
function tp_gate_score_grid(model, df::AbstractDataFrame, contract::SLContract;
                            n_rows::Int   = 5,
                            tol::Float64  = 1e-12)
    mg      = contract.max_goals
    results = Any[]

    worst_parity = 0.0
    worst_trunc  = 0.0
    min_cell     = Inf
    n_checked    = 0

    for row in eachrow(first(df, n_rows))
        S = tp_score_matrix(model, row; max_goals = mg)
        n_draws = size(S.data, 3)

        for k in 1:min(n_draws, 25)          # 25 draws per row is ample
            grid = S.data[:, :, k]
            ref  = tp_reference_grid(row.λ_h[k], row.λ_a[k], row.r_h[k], row.r_a[k], mg)

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

    # Truncation is a CHOICE, so it is judged by what it costs, not by whether the
    # mass "looks small". Raw mass is the wrong quantity: it is dominated by a few
    # extreme draws, while what matters is the shift in a price a bet would be
    # placed at. Measured against a grid 8 goals wider on the highest-scoring
    # fixture, at the line the discarded mass affects most (every truncated
    # scoreline is an "over").
    hot = df[argmax([mean(r.λ_h) + mean(r.λ_a) for r in eachrow(df)]), :]
    o35 = outcomes(Markets.MarketOverUnder(3.5))
    p_at(g) = mean(Predictions.compute_market_probs(
                       tp_score_matrix(model, hot; max_goals = g),
                       Markets.MarketOverUnder(3.5))[o35.over])
    price_shift = abs(p_at(mg) - p_at(mg + 8))

    push!(results, (
        name   = "truncation costs nothing that matters",
        pass   = price_shift <= 1e-4,
        detail = @sprintf("P(over 3.5) moves %.3e widening %d→%d goals (worst fixture, λ_tot %.2f); mass %.3e",
                          price_shift, mg, mg + 8,
                          mean(hot.λ_h) + mean(hot.λ_a), worst_trunc),
    ))

    # ---- Orientation. See the header: a transposed grid is well-formed and wrong.
    row1 = first(eachrow(df))
    S1   = tp_score_matrix(model, row1; max_goals = mg)
    g1   = S1.data[:, :, 1]

    goals      = collect(0:(mg - 1))
    e_home     = sum(goals .* vec(sum(g1, dims = 2)))
    e_away     = sum(goals .* vec(sum(g1, dims = 1)))
    λ_h1, λ_a1 = row1.λ_h[1], row1.λ_a[1]
    separation = abs(λ_h1 - λ_a1)

    # Tested RELATIVE to the separation between the two marginals, not against an
    # absolute tolerance. The question is "which marginal is this", and the answer
    # is only meaningful because γ > 0 makes the two differ. An absolute tolerance
    # would instead be measuring truncation, which is a different question and is
    # gated above.
    push!(results, (
        name   = "grid orientation [home, away]",
        pass   = abs(e_home - λ_h1) < 0.05 * separation &&
                 abs(e_away - λ_a1) < 0.05 * separation,
        detail = @sprintf("E[home] %.4f vs λ_h %.4f | E[away] %.4f vs λ_a %.4f | marginals separated by %.4f",
                          e_home, λ_h1, e_away, λ_a1, separation),
    ))

    # The mean of a mean-parameterised NegBin is λ EXACTLY, so on the full support
    # E[home] would be λ_h. On a truncated grid it must fall short by precisely the
    # first moment of the discarded tail — computed here from the stock
    # distribution. Requiring that identity to hold to 1e-12 is much stronger than
    # requiring E[home] ≈ λ_h: it says the shortfall is truncation and nothing else.
    dh        = NegativeBinomial(row1.r_h[1], row1.r_h[1] / (row1.r_h[1] + λ_h1))
    tail_mean = λ_h1 - sum(k * pdf(dh, k) for k in 0:(mg - 1))
    moment_err = abs((λ_h1 - e_home) - tail_mean)

    push!(results, (
        name   = "moment shortfall is exactly the truncated tail",
        pass   = moment_err <= 1e-12,
        detail = @sprintf("shortfall %.3e, tail first moment %.3e, |Δ| = %.3e",
                          λ_h1 - e_home, tail_mean, moment_err),
    ))

    return results
end


# ==============================================================================
# 4. GATE 5c — Market identities
# ==============================================================================

"""
    tp_gate_market_identities(model, df, contract; n_rows, tol) -> Vector

Every market in the book is a partition of the same grid, so each family must sum
to the same total — and that total is `1 - truncation_mass`, NOT 1.

Nothing in `src` normalises the NegBin grid (only `frank_copula.jl:51` does), so
asserting "1X2 sums to 1" would be asserting something false. What must hold is
INTERNAL CONSISTENCY: 1X2, BTTS and each O/U line all partition the same cells, so
they must all sum to the same number as each other and as `sum(grid)`.

That is the stronger test anyway. A pricer that dropped a cell, double-counted the
diagonal, or used `>=` where it meant `>` would break consistency while still
summing to something near 1.

The O/U lines are all half-lines (0.5/1.5/2.5/3.5) so no push mass exists;
`over_under.jl:31-36` silently discards exact-integer totals, which would break
this identity on an integer line. Asserted below so that a future integer line
cannot be added unnoticed.
"""
function tp_gate_market_identities(model, df::AbstractDataFrame, contract::SLContract;
                                   n_rows::Int  = 5,
                                   tol::Float64 = 1e-12)
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
        S = tp_score_matrix(model, row; max_goals = mg)

        p_1x2  = Predictions.compute_market_probs(S, Markets.Market1X2())
        p_btts = Predictions.compute_market_probs(S, Markets.MarketBTTS())
        p_ou   = Dict(l => Predictions.compute_market_probs(S, Markets.MarketOverUnder(l)) for l in lines)

        for k in 1:min(size(S.data, 3), 25)
            mass = sum(S.data[:, :, k])

            s_1x2  = p_1x2[:home][k] + p_1x2[:draw][k] + p_1x2[:away][k]
            s_btts = p_btts[:btts_yes][k] + p_btts[:btts_no][k]

            worst_1x2  = max(worst_1x2,  abs(s_1x2  - mass))
            worst_btts = max(worst_btts, abs(s_btts - mass))
            worst_cross = max(worst_cross, abs(s_1x2 - s_btts))

            unders = Float64[]
            for l in lines
                o = outcomes(Markets.MarketOverUnder(l))
                s_ou = p_ou[l][o.over][k] + p_ou[l][o.under][k]
                worst_ou = max(worst_ou, abs(s_ou - mass))
                worst_cross = max(worst_cross, abs(s_ou - s_1x2))
                push!(unders, p_ou[l][o.under][k])

                # Recompute under(l) directly from the grid, independently of the
                # pricer's loop, to check its comparison logic rather than trust it.
                grid = S.data[:, :, k]
                ref_under = sum(grid[r, c] for r in 1:mg, c in 1:mg if (r - 1) + (c - 1) < l)
                worst_ouref = max(worst_ouref, abs(ref_under - p_ou[l][o.under][k]))
            end

            # Under(l) must be non-decreasing in the line: a higher bar admits more
            # scorelines. Record the largest DECREASE, zero if monotone.
            worst_mono = max(worst_mono, -min(0.0, minimum(diff(unders))))

            n_checked += 1
        end
    end

    push!(results, (
        name   = "1X2 partitions the grid",
        pass   = worst_1x2 <= tol,
        detail = @sprintf("max |home+draw+away - sum(grid)| = %.3e over %d draws", worst_1x2, n_checked),
    ))

    push!(results, (
        name   = "BTTS partitions the grid",
        pass   = worst_btts <= tol,
        detail = @sprintf("max |yes+no - sum(grid)| = %.3e", worst_btts),
    ))

    push!(results, (
        name   = "O/U partitions the grid",
        pass   = worst_ou <= tol,
        detail = @sprintf("max |over+under - sum(grid)| = %.3e across lines %s", worst_ou, string(lines)),
    ))

    push!(results, (
        name   = "families agree with each other",
        pass   = worst_cross <= tol,
        detail = @sprintf("max disagreement between 1X2 / BTTS / O/U totals = %.3e", worst_cross),
    ))

    push!(results, (
        name   = "O/U agrees with a direct cell sum",
        pass   = worst_ouref <= tol,
        detail = @sprintf("max |pricer under - recomputed under| = %.3e", worst_ouref),
    ))

    push!(results, (
        name   = "under(line) non-decreasing in line",
        pass   = worst_mono <= tol,
        detail = worst_mono <= tol ? "monotone across $(lines)" :
                 @sprintf("VIOLATED by %.3e", worst_mono),
    ))

    return results
end

"""
    tp_market_summary(model, df, contract; n_rows) -> DataFrame

Posterior-mean market prices for the first few fixtures. Not a gate — this is the
first point in the protocol where the model says something a human can sanity
check against intuition, so it is worth looking at before trusting gate 6.
"""
function tp_market_summary(model, df::AbstractDataFrame, contract::SLContract; n_rows::Int = 5)
    rows = NamedTuple[]
    for row in eachrow(first(df, n_rows))
        S      = tp_score_matrix(model, row; max_goals = contract.max_goals)
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
