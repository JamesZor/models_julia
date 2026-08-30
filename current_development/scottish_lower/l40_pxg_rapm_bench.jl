# ==============================================================================
# l40 — LOADER for the pxG / pxG-APM covariate benchmark
# ==============================================================================
#
# Definitions only, no execution. Everything here is machinery the runner should
# be able to CALL without reading: coverage accounting, design-column preflight,
# safe chain lookups, and the leaderboard formatter.
#
# The runner (r40_train_pxg_rapm_models.jl) stays a readable notebook: question,
# data, models, splits, sampler, gates, report.
# ==============================================================================

using DataFrames
using Dates
using Printf
using Statistics
using MCMCChains

const L40_PG = BayesianFootball.Models.PreGame
const L40_FEATURES = BayesianFootball.Features

# ==============================================================================
# 1. COVERAGE ACCOUNTING
# ==============================================================================

"""
    l40_pxg_coverage(ds, feature) -> NamedTuple

How each match in the store was measured, by rung of the pxG ladder. A run in which
`goals` dominates is not a pxG experiment — it is the baseline wearing a different
name — so this is a gate, not a diagnostic.
"""
function l40_pxg_coverage(ds::BayesianFootball.Data.DataStore, feature::PxGFeature)
    observations = L40_FEATURES.pxg_match_observations(ds, feature)
    total = nrow(ds.matches)
    counts = Dict{Symbol,Int}(:commentary => 0, :shot_counts => 0, :goals => 0)
    for v in values(observations)
        counts[v.source] = get(counts, v.source, 0) + 1
    end
    measured = sum(values(counts))
    return (
        total_matches = total,
        measured = measured,
        commentary = counts[:commentary],
        shot_counts = counts[:shot_counts],
        goals = counts[:goals],
        commentary_share = total == 0 ? 0.0 : counts[:commentary] / total,
        unmeasured = total - measured,
    )
end

"""
    l40_print_pxg_coverage(cov)

One block, printed before any sampling starts.
"""
function l40_print_pxg_coverage(cov)
    println("  pxG measurement ladder over $(cov.total_matches) matches:")
    @printf("    live-text commentary : %6d  (%5.1f%%)\n",
            cov.commentary, 100 * cov.commentary_share)
    @printf("    BBC shot counts      : %6d\n", cov.shot_counts)
    @printf("    goals fallback       : %6d\n", cov.goals)
    @printf("    unmeasured           : %6d\n", cov.unmeasured)
    return nothing
end

# ==============================================================================
# 2. DESIGN-COLUMN PREFLIGHT
# ==============================================================================

"""
    l40_covariate_preflight(ds, model, splitter) -> DataFrame

Build the model's fold-1 feature set and describe every covariate design column it
produced: length, spread, and how much of it is the neutral zero.

This is the cheap place to catch the two failures that otherwise surface hours into a
grid run — a covariate that is constant (the feature found no coverage and the weight
is unidentified) and a covariate whose scale is wildly off its prior.
"""
function l40_covariate_preflight(ds::BayesianFootball.Data.DataStore, model, splitter)
    boundaries = BayesianFootball.Data.create_id_boundaries(ds, splitter)
    feature_sets = L40_FEATURES.create_features(boundaries, ds, model, splitter)
    fs = feature_sets[1]

    rows = NamedTuple[]
    for covariate in L40_PG.cb_covariates(model)
        column = L40_PG.covariate_column(covariate, fs)
        neutral = count(iszero, column)
        push!(rows, (
            covariate = String(L40_PG.covariate_name(covariate)),
            role = String(nameof(typeof(L40_PG.covariate_role(covariate)))),
            n = length(column),
            mean = mean(column),
            sd = length(column) > 1 ? std(column) : 0.0,
            min = isempty(column) ? 0.0 : minimum(column),
            max = isempty(column) ? 0.0 : maximum(column),
            neutral_share = isempty(column) ? 1.0 : neutral / length(column),
        ))
    end
    return DataFrame(rows)
end

"""
    l40_print_preflight(table)

The preflight table, plus the two warnings it exists to raise.
"""
function l40_print_preflight(table::DataFrame)
    if nrow(table) == 0
        println("  (baseline: no covariates)")
        return nothing
    end
    @printf("  %-12s | %-14s | %5s | %8s | %8s | %8s | %8s | %8s\n",
            "covariate", "role", "n", "mean", "sd", "min", "max", "neutral")
    println("  " * "-"^92)
    for r in eachrow(table)
        @printf("  %-12s | %-14s | %5d | %+8.4f | %8.4f | %+8.4f | %+8.4f | %7.1f%%\n",
                r.covariate, r.role, r.n, r.mean, r.sd, r.min, r.max,
                100 * r.neutral_share)
    end
    for r in eachrow(table)
        if r.sd < 1e-8
            println("  [WARN] $(r.covariate) is CONSTANT — its weight is unidentified.")
        end
        if r.neutral_share > 0.5
            @printf("  [WARN] %s is neutral on %.1f%% of the fold — thin coverage.\n",
                    r.covariate, 100 * r.neutral_share)
        end
    end
    return nothing
end

# ==============================================================================
# 3. SAFE CHAIN LOOKUPS
# ==============================================================================

"""
    l40_chain_mean(chain, site) -> Float64

The posterior mean of one site, or `NaN` when this model does not carry it. Models in
the grid deliberately have different parameter sets, so a missing site is a fact about
the arm, not an error.
"""
function l40_chain_mean(chain::Chains, site::Symbol)
    site in Set(Symbol.(names(chain))) || return NaN
    return mean(chain[site])
end

"Home advantage lives under three different site names across the component library."
function l40_home_advantage(chain::Chains)
    for site in (Symbol("ha.γ"), Symbol("ha.γ_global"), Symbol("ha.γ_raw"))
        value = l40_chain_mean(chain, site)
        isnan(value) || return value
    end
    return NaN
end

# ==============================================================================
# 4. ONE ROW OF THE LEADERBOARD
# ==============================================================================

"""
    l40_summarise_fit(name, fit, ds, elapsed) -> NamedTuple

Convergence, the covariate weights this arm actually carries, and out-of-sample
proper scores. Everything the runner prints comes from here.
"""
function l40_summarise_fit(name::AbstractString, fit, ds, elapsed::Float64)
    diagnostics = fit.diagnostics
    chain = fit.folds[1].chain
    report = BayesianFootball.evaluate_predictions(fit, ds)

    return (
        name = String(name),
        elapsed = elapsed,
        passed = diagnostics.passed,
        max_rhat = diagnostics.max_rhat,
        min_ess = diagnostics.min_ess_bulk,
        divergences = diagnostics.n_divergent,
        gamma = l40_home_advantage(chain),
        w_prod = l40_chain_mean(chain, Symbol("production_wealth.w")),
        w_pxg = l40_chain_mean(chain, Symbol("pxg.w")),
        w_rapm = l40_chain_mean(chain, Symbol("pxg_rapm.w")),
        logloss = report.model.logloss,
        brier = report.model.brier,
        ece = report.model.ece,
        rps = report.model.rps,
    )
end

# ==============================================================================
# 5. THE LEADERBOARD
# ==============================================================================

l40_fmt(value, spec) = isnan(value) ? "—" : Printf.format(Printf.Format(spec), value)

"""
    l40_print_leaderboard(rows; baseline)

Arms sorted by out-of-sample log loss, with the delta against the named baseline. The
delta is the decision quantity: a covariate that does not beat `m00` on genuine
walk-forward log loss has not earned its parameter.
"""
function l40_print_leaderboard(rows::Vector; baseline::AbstractString = "m00_baseline")
    ranked = sort(collect(rows), by = r -> isnan(r.logloss) ? Inf : r.logloss)
    base_idx = findfirst(r -> r.name == baseline, ranked)
    base_ll = base_idx === nothing ? NaN : ranked[base_idx].logloss

    println("="^134)
    @printf(" %-26s | %6s | %5s | %6s | %4s | %7s | %7s | %7s | %8s | %9s | %7s\n",
            "Model", "Time", "R-hat", "ESS", "Div", "γ (HA)", "w_pxg", "w_rapm",
            "LogLoss", "ΔLogLoss", "Brier")
    println("-"^134)
    for r in ranked
        delta = isnan(base_ll) || isnan(r.logloss) ? NaN : r.logloss - base_ll
        @printf(" %-26s | %5.0fs | %5s | %6s | %4d | %7s | %7s | %7s | %8s | %9s | %7s\n",
                r.name, r.elapsed,
                l40_fmt(r.max_rhat, "%.3f"),
                isnan(r.min_ess) ? "—" : string(Int(round(r.min_ess))),
                r.divergences,
                l40_fmt(r.gamma, "%+.3f"),
                l40_fmt(r.w_pxg, "%+.3f"),
                l40_fmt(r.w_rapm, "%+.3f"),
                l40_fmt(r.logloss, "%.4f"),
                l40_fmt(delta, "%+.4f"),
                l40_fmt(r.brier, "%.4f"))
    end
    println("="^134)

    failed = [r.name for r in ranked if !r.passed]
    if isempty(failed)
        println(" Convergence: all arms passed.")
    else
        println(" Convergence: FAILED for $(join(failed, ", ")) — their scores are not comparable.")
    end
    return DataFrame(ranked)
end
