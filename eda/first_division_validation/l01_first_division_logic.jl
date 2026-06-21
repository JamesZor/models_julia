# eda/first_division_validation/l01_first_division_logic.jl
#
# Loader (math / functions only — NO top-level execution).
#
# Stage-A EDA for the NEW competition Ireland First Division (tournament 718),
# contrasted side-by-side with Ireland Premier (79). Goal: characterise 718 as a
# data-generating process — full data/coverage audit, marginal goal moments, the
# discrete-count ladder (Poisson / NB1 / NB2 / Weibull / ZIP / ZINB / COM-Poisson)
# and the Dixon-Coles bivariate ladder — to settle whether First Division is a
# distinct regime needing its own stratum, or is poolable with the Premier.
#
# Pairs with r01_first_division_runner.jl.
#
# REUSE: the count fitters, summaries, GoF and league diagnostics already exist.
#   - eda/ireland_validation/l01_bigchance_logic.jl  → summarise_count,
#     compare_count_models, compare_nb1_nb2, rootogram_data, chi_square_gof,
#     fit_poisson_entry/fit_negbin_entry/fit_weibull_entry/fit_zip/fit_zinb/
#     fit_compoisson, fit_nb1_entry/fit_nb2_entry.
#   - eda/ireland_validation/l00_validation_logic.jl → test_overdispersion,
#     test_home_advantage_mean/_variance, test_team_volatility, test_temporal_stability.
# Both are include-safe (no top-level execution). eda/basic_goals/l00 is NOT
# include-safe (top-level code referencing a global `ds`), so the few goal/DC
# fitters it holds (analyze_goal_models, the Dixon-Coles bivariate ladder) are
# re-declared below — exactly the pattern l01_bigchance_logic.jl already used.

using DataFrames
using Distributions
using Statistics
using StatsBase
using Printf
using Optim
using BayesianFootball

include("../ireland_validation/l00_validation_logic.jl")
include("../ireland_validation/l01_bigchance_logic.jl")

# ============================================================================
# A. GOAL-FAMILY χ² TABLE (re-declared from eda/basic_goals/l00 — that file has
#    top-level execution and cannot be `include`d). Poisson / RobustNB / Weibull
#    with AIC + Pearson χ² p-value, used to *validate the fitters on 79* against
#    the published basic_goals / ireland_validation numbers before trusting 718.
# ============================================================================

"AIC + binned χ²(0:7) goodness-of-fit for a fitted Distributions.jl model."
function goalfit_metrics(dist, data::AbstractVector{<:Integer})
    isnothing(dist) && return nothing
    ll = loglikelihood(dist, data)
    k_params = length(params(dist))
    aic = 2 * k_params - 2 * ll
    obs_counts = counts(data, 0:7)
    n = length(data)
    expected = [pdf(dist, i) * n for i in 0:6]
    push!(expected, (1 - cdf(dist, 6)) * n)
    expected_safe = max.(expected, 1e-6)
    chi_sq = sum((obs_counts .- expected_safe) .^ 2 ./ expected_safe)
    dfree = length(obs_counts) - 1 - k_params
    p_val = ccdf(Chisq(dfree), chi_sq)
    return (log_likelihood = ll, aic = aic, chi_sq = chi_sq, df = dfree, p_value = p_val)
end

"Fit Poisson / RobustNB / WeibullCount to one goal vector (reuses l01 fit_mle's)."
function fit_goal_distributions(data::AbstractVector{<:Integer})
    p_dist = fit(Poisson, data)
    nb_dist = fit_mle(MyDistributions.RobustNegativeBinomial, data)
    wc_dist = fit_mle(MyDistributions.WeibullCount, data)
    return (poisson = p_dist, nb = nb_dist, wc = wc_dist)
end

"""
    analyze_goal_models(goals_dict)

Poisson vs RobustNB vs WeibullCount on each goal vector (LL / AIC / χ² / df / p).
This is the legacy basic_goals table — used to reproduce the published 79 numbers.
"""
function analyze_goal_models(goals_dict::Dict{String, <:AbstractVector{<:Integer}})
    for (label, data) in goals_dict
        println("\n" * "═"^66)
        println(" GOAL MODEL COMPARISON: $(uppercase(label)) (n=$(length(data)))")
        println("═"^66)
        fits = fit_goal_distributions(data)
        p_stats = goalfit_metrics(fits.poisson, data)
        nb_stats = goalfit_metrics(fits.nb, data)
        wc_stats = goalfit_metrics(fits.wc, data)
        @printf("%-18s | %-12s | %-12s | %-12s\n", "Metric", "Poisson", "Robust NB", "Weibull Cnt")
        println("-"^66)
        prow(name, metric, fmt) = @printf("%-18s | %-12s | %-12s | %-12s\n", name,
            Printf.format(fmt, getproperty(p_stats, metric)),
            Printf.format(fmt, getproperty(nb_stats, metric)),
            Printf.format(fmt, getproperty(wc_stats, metric)))
        prow("Log likelihood", :log_likelihood, Printf.Format("%.2f"))
        prow("AIC", :aic, Printf.Format("%.2f"))
        prow("Chi sq", :chi_sq, Printf.Format("%.2f"))
        prow("Degrees of fr.", :df, Printf.Format("%d"))
        prow("P-value", :p_value, Printf.Format("%.4f"))
    end
end

# ============================================================================
# B. DIXON-COLES BIVARIATE LADDER (re-declared from eda/basic_goals/l00).
#    Indep-Poisson, DC-Poisson, Indep-NB, DC-NB, Indep-Weibull, DC-Weibull.
#    Each fit returns (ll, k[, ρ]); we tabulate AIC and the DC dependence ρ.
# ============================================================================

"Dixon-Coles low-score τ correction (shared by every DC variant)."
function _dc_tau(h, a, λ, μ, ρ)
    if h == 0 && a == 0; return 1.0 - λ * μ * ρ
    elseif h == 1 && a == 0; return 1.0 + μ * ρ
    elseif h == 0 && a == 1; return 1.0 + λ * ρ
    elseif h == 1 && a == 1; return 1.0 - ρ
    else; return 1.0
    end
end

function _dc_pois_ll(λ, μ, ρ, hg, ag)
    ll = 0.0
    for (h, a) in zip(hg, ag)
        τ = _dc_tau(h, a, λ, μ, ρ)
        ll += logpdf(Poisson(λ), h) + logpdf(Poisson(μ), a) + (τ > 0 ? log(τ) : -Inf)
    end
    return ll
end

function _dc_nb_ll(λ, μ, r_h, r_a, ρ, hg, ag)
    ll = 0.0
    for (h, a) in zip(hg, ag)
        τ = _dc_tau(h, a, λ, μ, ρ)
        ll += logpdf(MyDistributions.RobustNegativeBinomial(r_h, λ), h) +
              logpdf(MyDistributions.RobustNegativeBinomial(r_a, μ), a) +
              (τ > 0 ? log(τ) : -Inf)
    end
    return ll
end

function _dc_wb_ll(c_h, λ_h, c_a, λ_a, ρ, hg, ag)
    ll = 0.0
    for (h, a) in zip(hg, ag)
        τ = _dc_tau(h, a, λ_h, λ_a, ρ)  # scale params proxy the rate in the τ term
        ll += logpdf(MyDistributions.WeibullCount(c_h, λ_h), h) +
              logpdf(MyDistributions.WeibullCount(c_a, λ_a), a) +
              (τ > 0 ? log(τ) : -Inf)
    end
    return ll
end

"""
    fit_dc_ladder(home_data, away_data)

Fit the full bivariate ladder: independent & Dixon-Coles for Poisson, NB and
Weibull-count. Returns a NamedTuple of (name, ll, k, ρ) entries.
"""
function fit_dc_ladder(home_data, away_data)
    λ0 = mean(home_data); μ0 = mean(away_data)
    v_h = var(home_data); v_a = var(away_data)
    r_h0 = v_h > λ0 ? λ0^2 / (v_h - λ0) : 10.0
    r_a0 = v_a > μ0 ? μ0^2 / (v_a - μ0) : 10.0

    # Indep Poisson — closed-form MLE at the means.
    ll_ip = sum(logpdf.(Poisson(λ0), home_data)) + sum(logpdf.(Poisson(μ0), away_data))

    # DC Poisson
    r_dp = optimize(p -> -_dc_pois_ll(exp(p[1]), exp(p[2]), tanh(p[3]), home_data, away_data),
                    [log(λ0), log(μ0), 0.0])

    # Indep NB
    r_inb = optimize(p -> -(sum(logpdf.(MyDistributions.RobustNegativeBinomial(exp(p[3]), exp(p[1])), home_data)) +
                            sum(logpdf.(MyDistributions.RobustNegativeBinomial(exp(p[4]), exp(p[2])), away_data))),
                     [log(λ0), log(μ0), log(r_h0), log(r_a0)])

    # DC NB (warm-start from indep NB)
    r_dnb = optimize(p -> -_dc_nb_ll(exp(p[1]), exp(p[2]), exp(p[3]), exp(p[4]), tanh(p[5]), home_data, away_data),
                     [r_inb.minimizer[1], r_inb.minimizer[2], r_inb.minimizer[3], r_inb.minimizer[4], 0.0])

    # Indep Weibull
    r_iwb = optimize(p -> -(sum(logpdf.(MyDistributions.WeibullCount(exp(p[1]), exp(p[2])), home_data)) +
                            sum(logpdf.(MyDistributions.WeibullCount(exp(p[3]), exp(p[4])), away_data))),
                     [log(1.0), log(λ0), log(1.0), log(μ0)], NelderMead(), Optim.Options(iterations = 3000))

    # DC Weibull (warm-start from indep Weibull)
    r_dwb = optimize(p -> -_dc_wb_ll(exp(p[1]), exp(p[2]), exp(p[3]), exp(p[4]), tanh(p[5]), home_data, away_data),
                     [r_iwb.minimizer[1], r_iwb.minimizer[2], r_iwb.minimizer[3], r_iwb.minimizer[4], 0.0],
                     NelderMead(), Optim.Options(iterations = 3000))

    return (
        indep_pois = (name = "Indep Poisson", ll = ll_ip,            k = 2, ρ = 0.0),
        dc_pois    = (name = "DC Poisson",    ll = -r_dp.minimum,    k = 3, ρ = tanh(r_dp.minimizer[3])),
        indep_nb   = (name = "Indep NB",      ll = -r_inb.minimum,   k = 4, ρ = 0.0),
        dc_nb      = (name = "DC NB",         ll = -r_dnb.minimum,   k = 5, ρ = tanh(r_dnb.minimizer[5])),
        indep_wb   = (name = "Indep Weibull", ll = -r_iwb.minimum,   k = 4, ρ = 0.0),
        dc_wb      = (name = "DC Weibull",    ll = -r_dwb.minimum,   k = 5, ρ = tanh(r_dwb.minimizer[5])),
    )
end

"""
    analyze_heavyweight_models(ds_or_df; label)

Run + tabulate the full Dixon-Coles bivariate ladder on a DataStore (or matches
DataFrame). Prints LL / AIC for all six and the DC dependence ρ; returns the
fits NamedTuple with AICs attached.
"""
function analyze_heavyweight_models(ds::Data.DataStore; label::String = "")
    return analyze_heavyweight_models(ds.matches; label = label)
end

function analyze_heavyweight_models(matches::DataFrame; label::String = "")
    home_data = collect(skipmissing(matches.home_score))
    away_data = collect(skipmissing(matches.away_score))
    f = fit_dc_ladder(home_data, away_data)
    order = [f.indep_pois, f.dc_pois, f.indep_nb, f.dc_nb, f.indep_wb, f.dc_wb]
    aic(e) = 2e.k - 2e.ll

    println("\n" * "═"^78)
    println(" DIXON-COLES BIVARIATE LADDER: $(uppercase(label))  (n=$(length(home_data)))")
    println("═"^78)
    @printf("%-16s | %-3s | %-12s | %-12s | %-8s\n", "Model", "k", "LogLik", "AIC", "ρ")
    println("-"^78)
    for e in Base.sort(order, by = aic)
        @printf("%-16s | %-3d | %-12.2f | %-12.2f | %+.4f\n", e.name, e.k, e.ll, aic(e), e.ρ)
    end
    best = order[argmin(aic.(order))]
    println("-"^78)
    @printf("Winner by AIC: %s (AIC %.2f, ρ=%+.4f)\n", best.name, aic(best), best.ρ)
    return (fits = f, best = best.name, aic = Dict(e.name => aic(e) for e in order))
end

# ============================================================================
# C. NEW HELPER — per-season feature-coverage audit (the only genuinely new
#    function; everything else is reused). Produces the Stage-B readiness map.
# ============================================================================

"""
    feature_coverage_by_season(ds) -> DataFrame

Per-season audit of the DataStore. One row per season with:
  matches, played (non-missing scores), goal coverage,
  xG / bigChance / shots non-missing coverage (from ds.statistics, period=="ALL",
  joined to matches' season), and odds / betfair / lineups / incidents row presence.

Coverage is reported as the fraction of PLAYED matches in that season whose stats
row has a non-missing value for that field, since stats are the gating resource.
"""
function feature_coverage_by_season(ds::Data.DataStore)
    matches = ds.matches
    # season → played match_ids
    played = filter(r -> !ismissing(r.home_score), matches)
    seasons = sort(unique(string.(matches.season)))

    stats_all = nrow(ds.statistics) == 0 ? ds.statistics :
                filter(r -> r.period == "ALL", ds.statistics)
    smap = Dict(r.match_id => r for r in eachrow(stats_all))

    # row→season lookups for the auxiliary frames
    season_of = Dict(r.match_id => string(r.season) for r in eachrow(matches))
    function rows_by_season(df)
        d = Dict{String, Int}()
        :match_id in propertynames(df) || return d
        for id in df.match_id
            s = get(season_of, id, missing)
            ismissing(s) && continue
            d[s] = get(d, s, 0) + 1
        end
        return d
    end
    odds_n     = rows_by_season(ds.odds)
    betfair_n  = rows_by_season(ds.betfair_odds)
    lineups_n  = rows_by_season(ds.lineups)
    incid_n    = rows_by_season(ds.incidents)

    rows = NamedTuple[]
    for s in seasons
        ms = filter(r -> string(r.season) == s, matches)
        ps = filter(r -> string(r.season) == s, played)
        nplayed = nrow(ps)
        # stats-derived coverage over PLAYED matches in this season
        has_xg = has_bc = has_shots = has_stats = 0
        for r in eachrow(ps)
            haskey(smap, r.match_id) || continue
            has_stats += 1
            st = smap[r.match_id]
            (!ismissing(st.expectedGoals_home) && !ismissing(st.expectedGoals_away)) && (has_xg += 1)
            (!ismissing(st.bigChanceCreated_home) && !ismissing(st.bigChanceCreated_away)) && (has_bc += 1)
            # shots column name varies; probe common ones
            shot_ok = false
            for c in (:totalShots_home, :shotsTotal_home, :shots_home)
                if c in propertynames(st) && !ismissing(getproperty(st, c)); shot_ok = true; break; end
            end
            shot_ok && (has_shots += 1)
        end
        frac(x) = nplayed == 0 ? 0.0 : round(x / nplayed; digits = 3)
        push!(rows, (
            season = s,
            matches = nrow(ms),
            played = nplayed,
            goals_cov = frac(count(r -> !ismissing(r.home_score), eachrow(ps))),
            stats_cov = frac(has_stats),
            xg_cov = frac(has_xg),
            bigchance_cov = frac(has_bc),
            shots_cov = frac(has_shots),
            odds_rows = get(odds_n, s, 0),
            betfair_rows = get(betfair_n, s, 0),
            lineups_rows = get(lineups_n, s, 0),
            incidents_rows = get(incid_n, s, 0),
        ))
    end
    return DataFrame(rows)
end

"""
    datastore_overview(ds) -> Nothing

Prints a top-level audit of every DataStore field: row count and (where a season
column exists) the season span. Cheap orientation before the per-season table.
"""
function datastore_overview(ds::Data.DataStore)
    fields = [(:matches, ds.matches), (:statistics, ds.statistics), (:odds, ds.odds),
              (:lineups, ds.lineups), (:incidents, ds.incidents), (:betfair_odds, ds.betfair_odds)]
    println("\n" * "═"^60)
    println(" DATASTORE OVERVIEW: $(typeof(ds.segment))")
    println("═"^60)
    @printf("%-14s | %-8s | %-10s\n", "Field", "rows", "cols")
    println("-"^60)
    for (nm, df) in fields
        @printf("%-14s | %-8d | %-10d\n", nm, nrow(df), ncol(df))
    end
    if "season" in names(ds.matches)
        ss = sort(unique(string.(ds.matches.season)))
        println("-"^60)
        println("matches seasons: ", join(ss, ", "))
    end
    return nothing
end

"Goal vectors for a DataStore in the Dict shape analyze_goal_models expects."
function get_goals(ds::Data.DataStore)
    Dict{String, AbstractVector{<:Integer}}(
        "home"  => collect(skipmissing(ds.matches.home_score)),
        "away"  => collect(skipmissing(ds.matches.away_score)),
        "total" => vcat(collect(skipmissing(ds.matches.home_score)),
                        collect(skipmissing(ds.matches.away_score))),
    )
end
