# ==============================================================================
# l93 — LOADER: the statistical toolkit shared by the pxG / RAPM EDA runners
# ==============================================================================
#
# Definitions only, no execution. Shared by r93 (synergy), r94 (RAPM forensics)
# and r95 (pxG forensics), which are three views of the same two features and
# would otherwise triplicate every table, histogram and regression.
#
# EVERYTHING IS TEXT. No Plots dependency, no PNG artefacts: these runners exist
# to produce numbers that can be pasted into PXG_RAPM_EDA_REPORT.md and diffed on
# a re-scrape. An ASCII histogram is worse to look at than a real one and far
# better to review.
#
# ⚠ WHAT THIS STORE CANNOT TELL US. Scottish tiers 56/57 carry NO SofaScore
# statistics, NO player ratings and NO xG (verified: `ds.statistics` is empty,
# `lineups.rating` is missing on all 74,225 rows). Every external-validity check
# the bbc_xg_proxy stream ran on tiers 54/55 is therefore UNAVAILABLE here. The
# only ground truth on this store is the scoreline, so that is what these runners
# calibrate against, and no claim of agreement with a reference xG can be made.
# ==============================================================================

using DataFrames
using Dates
using Printf
using Statistics
using StatsBase: corspearman, corkendall, tiedrank, skewness, kurtosis, sample

const EDA_DATA = BayesianFootball.Data
const EDA_FEATURES = BayesianFootball.Features
const EDA_PG = BayesianFootball.Models.PreGame

# ==============================================================================
# 1. PRINTING
# ==============================================================================

eda_rule(n::Int = 100, ch::Char = '=') = println(repeat(ch, n))

function eda_banner(title::AbstractString)
    println()
    eda_rule(100, '=')
    println(" ", uppercase(title))
    eda_rule(100, '=')
end

function eda_section(number, title::AbstractString)
    println()
    println("[$number] ", title)
    eda_rule(100, '-')
end

"Format a possibly-NaN number, so a missing statistic prints as a dash not as `NaN`."
eda_fmt(x::Real, spec::AbstractString = "%.4f") =
    (isnan(x) || !isfinite(x)) ? "—" : Printf.format(Printf.Format(spec), x)

# ==============================================================================
# 2. NUMERIC HYGIENE
# ==============================================================================

"Coerce anything to Float64, mapping missing/unparseable/non-finite to NaN."
function eda_num(x)::Float64
    (x === nothing || ismissing(x)) && return NaN
    v = try
        Float64(x)
    catch
        return NaN
    end
    return isfinite(v) ? v : NaN
end

"Drop the pairwise-incomplete rows of two vectors."
function eda_pairs(x::AbstractVector, y::AbstractVector)
    n = min(length(x), length(y))
    xs = Float64[]
    ys = Float64[]
    for i in 1:n
        a = eda_num(x[i])
        b = eda_num(y[i])
        (isnan(a) || isnan(b)) && continue
        push!(xs, a)
        push!(ys, b)
    end
    return xs, ys
end

function eda_pearson(x::AbstractVector, y::AbstractVector)
    xs, ys = eda_pairs(x, y)
    length(xs) < 3 && return NaN
    (std(xs) < 1e-12 || std(ys) < 1e-12) && return NaN
    return cor(xs, ys)
end

function eda_spearman(x::AbstractVector, y::AbstractVector)
    xs, ys = eda_pairs(x, y)
    length(xs) < 3 && return NaN
    (std(xs) < 1e-12 || std(ys) < 1e-12) && return NaN
    return corspearman(xs, ys)
end

# ==============================================================================
# 3. UNIVARIATE DESCRIPTION
# ==============================================================================

"""
    eda_describe(name, x) -> NamedTuple

Moments and quantiles, plus the two shares that matter for a design column: how
much of it is exactly zero (the neutral imputation) and how much is missing.
"""
function eda_describe(name::AbstractString, x::AbstractVector)
    raw = [eda_num(v) for v in x]
    v = filter(!isnan, raw)
    n = length(v)
    if n == 0
        return (name = String(name), n = 0, mean = NaN, sd = NaN, skew = NaN,
                kurt = NaN, min = NaN, q05 = NaN, q25 = NaN, med = NaN, q75 = NaN,
                q95 = NaN, max = NaN, zero_share = NaN, nan_share = 1.0)
    end
    return (
        name = String(name),
        n = n,
        mean = mean(v),
        sd = n > 1 ? std(v) : 0.0,
        skew = n > 2 && std(v) > 1e-12 ? skewness(v) : NaN,
        kurt = n > 3 && std(v) > 1e-12 ? kurtosis(v) : NaN,
        min = minimum(v),
        q05 = quantile(v, 0.05),
        q25 = quantile(v, 0.25),
        med = median(v),
        q75 = quantile(v, 0.75),
        q95 = quantile(v, 0.95),
        max = maximum(v),
        zero_share = count(iszero, v) / n,
        nan_share = (length(raw) - n) / length(raw),
    )
end

function eda_print_describe(rows::Vector; title::AbstractString = "")
    isempty(title) || println(title)
    @printf("  %-26s | %6s | %8s | %8s | %7s | %7s | %8s | %8s | %8s | %7s\n",
            "series", "n", "mean", "sd", "skew", "kurt", "min", "median", "max", "zeros")
    println("  " * repeat('-', 118))
    for r in rows
        @printf("  %-26s | %6d | %8s | %8s | %7s | %7s | %8s | %8s | %8s | %6s%%\n",
                r.name, r.n, eda_fmt(r.mean), eda_fmt(r.sd),
                eda_fmt(r.skew, "%.3f"), eda_fmt(r.kurt, "%.3f"),
                eda_fmt(r.min, "%.3f"), eda_fmt(r.med, "%.3f"), eda_fmt(r.max, "%.3f"),
                eda_fmt(100 * r.zero_share, "%.1f"))
    end
    return nothing
end

# ==============================================================================
# 4. ASCII HISTOGRAM
# ==============================================================================

"""
    eda_histogram(x; bins, width, label)

A fixed-width bar chart of `x`. Prints nothing for an empty or constant series
rather than dividing by a zero range.
"""
function eda_histogram(x::AbstractVector; bins::Int = 24, width::Int = 52,
                       label::AbstractString = "")
    v = filter(!isnan, [eda_num(u) for u in x])
    isempty(label) || println("  ", label)
    if length(v) < 2
        println("    (too few observations)")
        return nothing
    end
    lo, hi = minimum(v), maximum(v)
    if hi - lo < 1e-12
        @printf("    (constant at %.4f, n = %d)\n", lo, length(v))
        return nothing
    end
    edges = range(lo, hi; length = bins + 1)
    counts = zeros(Int, bins)
    for u in v
        idx = clamp(1 + floor(Int, (u - lo) / (hi - lo) * bins), 1, bins)
        counts[idx] += 1
    end
    peak = maximum(counts)
    for b in 1:bins
        bar = repeat('█', round(Int, width * counts[b] / peak))
        @printf("    %8.3f .. %8.3f | %-*s %6d\n",
                edges[b], edges[b + 1], width, bar, counts[b])
    end
    @printf("    n = %d, mean = %.4f, sd = %.4f\n", length(v), mean(v), std(v))
    return nothing
end

"A compact frequency table for a categorical vector, most frequent first."
function eda_freq_table(labels::AbstractVector; top::Int = 25,
                        title::AbstractString = "", width::Int = 34)
    isempty(title) || println("  ", title)
    counts = Dict{Any,Int}()
    for l in labels
        counts[l] = get(counts, l, 0) + 1
    end
    total = sum(values(counts))
    total == 0 && (println("    (empty)"); return DataFrame())
    ordered = sort(collect(counts), by = p -> -p[2])
    peak = ordered[1][2]
    for (label, count) in first(ordered, min(top, length(ordered)))
        bar = repeat('▇', round(Int, width * count / peak))
        @printf("    %-30s %7d  %5.1f%%  %s\n",
                string(label), count, 100 * count / total, bar)
    end
    length(ordered) > top && println("    ... and $(length(ordered) - top) more")
    return DataFrame(label = [string(p[1]) for p in ordered],
                     count = [p[2] for p in ordered],
                     share = [p[2] / total for p in ordered])
end

# ==============================================================================
# 5. ORDINARY LEAST SQUARES (hand-rolled: no formula plumbing needed here)
# ==============================================================================

"""
    eda_ols(X, y; intercept = true) -> NamedTuple

`(beta, se, t, r2, adj_r2, resid, n, k)`. Solved by QR (`\\`), which is stable on
the near-collinear designs this suite deliberately constructs.
"""
function eda_ols(X::AbstractMatrix, y::AbstractVector; intercept::Bool = true)
    A = intercept ? hcat(ones(size(X, 1)), Float64.(X)) : Float64.(X)
    yv = Float64.(y)
    n, k = size(A)
    n > k || return (beta = fill(NaN, k), se = fill(NaN, k), t = fill(NaN, k),
                     r2 = NaN, adj_r2 = NaN, resid = fill(NaN, n), n = n, k = k)
    beta = A \ yv
    resid = yv .- A * beta
    ss_res = sum(abs2, resid)
    ss_tot = sum(abs2, yv .- mean(yv))
    r2 = ss_tot < 1e-12 ? NaN : 1 - ss_res / ss_tot
    adj_r2 = isnan(r2) ? NaN : 1 - (1 - r2) * (n - 1) / (n - k)
    sigma2 = ss_res / (n - k)
    se = try
        sqrt.(max.(diag(inv(A' * A)) .* sigma2, 0.0))
    catch
        fill(NaN, k)
    end
    t = beta ./ se
    return (beta = beta, se = se, t = t, r2 = r2, adj_r2 = adj_r2,
            resid = resid, n = n, k = k)
end

"""
    eda_vif(names, columns) -> DataFrame

Variance inflation factor per column: `1 / (1 - R²_j)` where `R²_j` regresses
column `j` on every other column. Above 5 is worth noting; above 10 the two
covariates are measuring the same thing and the model cannot separate them.
"""
function eda_vif(names::Vector{String}, columns::Vector{Vector{Float64}})
    k = length(columns)
    out = NamedTuple[]
    for j in 1:k
        others = [columns[i] for i in 1:k if i != j]
        if isempty(others)
            push!(out, (feature = names[j], r2 = 0.0, vif = 1.0))
            continue
        end
        fit = eda_ols(reduce(hcat, others), columns[j])
        r2 = isnan(fit.r2) ? 0.0 : clamp(fit.r2, 0.0, 1 - 1e-12)
        push!(out, (feature = names[j], r2 = r2, vif = 1 / (1 - r2)))
    end
    return DataFrame(out)
end

# ==============================================================================
# 6. CORRELATION MATRIX
# ==============================================================================

"""
    eda_corr_matrix(names, columns; method) -> Matrix{Float64}

Pairwise complete correlations. `method` is `:pearson` or `:spearman`.
"""
function eda_corr_matrix(names::Vector{String}, columns::Vector{Vector{Float64}};
                         method::Symbol = :pearson)
    k = length(columns)
    M = fill(NaN, k, k)
    f = method === :spearman ? eda_spearman : eda_pearson
    for i in 1:k, j in 1:k
        M[i, j] = i == j ? 1.0 : f(columns[i], columns[j])
    end
    return M
end

function eda_print_corr(names::Vector{String}, M::Matrix{Float64};
                        title::AbstractString = "", flag::Float64 = 0.60)
    isempty(title) || println("  ", title)
    labels = [length(n) > 13 ? n[1:13] : n for n in names]
    @printf("  %-15s", "")
    for l in labels
        @printf(" %9s", length(l) > 9 ? l[1:9] : l)
    end
    println()
    for i in eachindex(labels)
        @printf("  %-15s", labels[i])
        for j in eachindex(labels)
            @printf(" %9s", eda_fmt(M[i, j], "%+.3f"))
        end
        println()
    end
    flagged = [(names[i], names[j], M[i, j])
               for i in 1:length(names) for j in (i + 1):length(names)
               if !isnan(M[i, j]) && abs(M[i, j]) >= flag]
    if isempty(flagged)
        @printf("  (no |ρ| >= %.2f off the diagonal)\n", flag)
    else
        for (a, b, r) in flagged
            @printf("  [NOTE] %s ~ %s : %+.3f — these two overlap substantially.\n", a, b, r)
        end
    end
    return nothing
end

# ==============================================================================
# 7. DECILE / CALIBRATION TABLES AND DISCRIMINATION
# ==============================================================================

"""
    eda_decile_table(x, y; k) -> DataFrame

Bin observations by `x`'s rank into `k` equal groups and report the mean of both
`x` and `y` in each. For a calibration check `x` is the prediction and `y` the
realised value: a well-calibrated feature has the two columns tracking.
"""
function eda_decile_table(x::AbstractVector, y::AbstractVector; k::Int = 10)
    xs, ys = eda_pairs(x, y)
    n = length(xs)
    n < k && return DataFrame()
    order = sortperm(xs)
    rows = NamedTuple[]
    for b in 1:k
        lo = 1 + fld((b - 1) * n, k)
        hi = fld(b * n, k)
        hi >= lo || continue
        idx = order[lo:hi]
        push!(rows, (bin = b, n = length(idx),
                     x_mean = mean(xs[idx]), x_lo = minimum(xs[idx]),
                     x_hi = maximum(xs[idx]), y_mean = mean(ys[idx]),
                     y_sd = length(idx) > 1 ? std(ys[idx]) : 0.0))
    end
    return DataFrame(rows)
end

function eda_print_decile(table::DataFrame; xlab = "x", ylab = "y",
                          title::AbstractString = "")
    nrow(table) == 0 && (println("  (too few observations for a decile table)"); return nothing)
    isempty(title) || println("  ", title)
    @printf("  %4s | %6s | %10s | %10s | %10s | %10s | %s\n",
            "bin", "n", "$(xlab) lo", "$(xlab) hi", "$(xlab) mean", "$(ylab) mean", "")
    println("  " * repeat('-', 96))
    peak = maximum(abs, table.y_mean)
    for r in eachrow(table)
        bar = peak < 1e-12 ? "" : repeat('▇', round(Int, 24 * abs(r.y_mean) / peak))
        @printf("  %4d | %6d | %+10.4f | %+10.4f | %+10.4f | %+10.4f | %s\n",
                r.bin, r.n, r.x_lo, r.x_hi, r.x_mean, r.y_mean, bar)
    end
    # Monotonicity is the property a supremacy covariate must have; report it plainly.
    diffs = diff(table.y_mean)
    up = count(>(0), diffs)
    @printf("  monotone steps: %d of %d increasing (%s)\n",
            up, length(diffs),
            up == length(diffs) ? "strictly monotone" :
            up == 0 ? "strictly decreasing" : "mixed")
    return nothing
end

"""
    eda_auc(scores, labels) -> Float64

Mann-Whitney AUC: the probability a randomly chosen positive outranks a randomly
chosen negative. 0.5 is no information. Ties are handled by mid-ranks.
"""
function eda_auc(scores::AbstractVector, labels::AbstractVector)
    s = Float64[]
    l = Bool[]
    for i in eachindex(scores)
        v = eda_num(scores[i])
        isnan(v) && continue
        push!(s, v)
        push!(l, Bool(labels[i]))
    end
    n_pos = count(l)
    n_neg = length(l) - n_pos
    (n_pos == 0 || n_neg == 0) && return NaN
    ranks = tiedrank(s)
    return (sum(ranks[l]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
end

# ==============================================================================
# 8. FEATURE EXTRACTION WITHOUT A MODEL
# ==============================================================================
#
# `create_features` dispatches on a model's `required_features`. EDA wants to ask
# for an arbitrary bag of feature configs without assembling an engine, so this is
# the thinnest possible model that satisfies that contract.

struct EdaProbe <: BayesianFootball.AbstractFootballModel
    configs::Vector{EDA_FEATURES.AbstractFeatureConfig}
end

EDA_FEATURES.required_features(m::EdaProbe) = m.configs

"""
    eda_boundary(ds; history_frac = 1.0) -> SplitBoundary

A boundary over the whole store. The default puts EVERY match in the history
block, which is what a descriptive pass wants: features that fit something
(the RAPM ridge, the shot-xG cell table) then see all of it.

The point-in-time walk is unaffected — each match's pxG column is still built
only from strictly earlier kickoffs — so the emitted covariate remains a genuine
pre-match quantity even when the fit set is the whole store.
"""
function eda_boundary(ds::EDA_DATA.DataStore; history_frac::Float64 = 1.0)
    frame = sort(ds.matches, [:match_date, :match_id])
    ids = Int.(frame.match_id)
    cut = clamp(round(Int, history_frac * length(ids)), 0, length(ids))
    return EDA_DATA.SplitBoundary(1, 1, ids[1:cut], ids[(cut + 1):end])
end

"""
    eda_features(ds, configs; boundary, dynamics_col) -> FeatureSet
"""
function eda_features(ds::EDA_DATA.DataStore,
                      configs::Vector{<:EDA_FEATURES.AbstractFeatureConfig};
                      boundary = eda_boundary(ds),
                      dynamics_col::Symbol = :match_month)
    probe = EdaProbe(EDA_FEATURES.AbstractFeatureConfig[c for c in configs])
    return EDA_FEATURES.create_features(boundary, ds, probe, dynamics_col)
end

# ==============================================================================
# 9. THE MATCH FRAME
# ==============================================================================

"""
    eda_match_frame(ds) -> DataFrame

One tidy row per match in `ordered_match_ids` order, carrying the outcome
quantities every runner scores against: goal supremacy, total goals, and the
three 1X2 indicators.
"""
function eda_match_frame(ds::EDA_DATA.DataStore; ordered_ids = nothing)
    by_id = Dict(Int(r.match_id) => r for r in eachrow(ds.matches))
    ids = ordered_ids === nothing ?
          Int.(sort(ds.matches, [:match_date, :match_id]).match_id) : Int.(ordered_ids)
    rows = NamedTuple[]
    for id in ids
        haskey(by_id, id) || continue
        r = by_id[id]
        gh = eda_num(r.home_score)
        ga = eda_num(r.away_score)
        push!(rows, (
            match_id = id,
            match_date = r.match_date,
            season = String(coalesce(r.season, "?")),
            tournament_id = Int(r.tournament_id),
            home_team = String(r.home_team),
            away_team = String(r.away_team),
            goals_home = gh,
            goals_away = ga,
            supremacy = gh - ga,
            total = gh + ga,
            home_win = !isnan(gh) && !isnan(ga) && gh > ga,
            draw = !isnan(gh) && !isnan(ga) && gh == ga,
            away_win = !isnan(gh) && !isnan(ga) && gh < ga,
            played = !isnan(gh) && !isnan(ga),
        ))
    end
    return DataFrame(rows)
end

# ==============================================================================
# 10. SIGNAL SUMMARY
# ==============================================================================

"""
    eda_signal(name, x, frame) -> NamedTuple

The standard battery every candidate covariate is put through: association with
goal supremacy and with total goals, home-win discrimination, and the share of
the column that is the neutral zero.

`n_live` counts the non-neutral, played observations — the ones the covariate
actually has an opinion about. A feature with a strong correlation over 40 live
matches has told you far less than the headline number suggests.
"""
function eda_signal(name::AbstractString, x::AbstractVector, frame::DataFrame)
    played = frame.played
    xs = Float64[eda_num(v) for v in x]
    live = played .& .!isnan.(xs) .& (xs .!= 0.0)
    return (
        feature = String(name),
        n = count(played),
        n_live = count(live),
        zero_share = count(iszero, xs[played]) / max(count(played), 1),
        r_supremacy = eda_pearson(xs[played], frame.supremacy[played]),
        rho_supremacy = eda_spearman(xs[played], frame.supremacy[played]),
        r_total = eda_pearson(xs[played], frame.total[played]),
        auc_home_win = eda_auc(xs[played], frame.home_win[played]),
    )
end

function eda_print_signal(rows::Vector; title::AbstractString = "")
    isempty(title) || println("  ", title)
    @printf("  %-22s | %6s | %6s | %7s | %10s | %10s | %9s | %9s\n",
            "feature", "n", "n_live", "zeros", "r(sup)", "rho(sup)", "r(total)", "AUC(H)")
    println("  " * repeat('-', 106))
    for r in rows
        @printf("  %-22s | %6d | %6d | %6.1f%% | %10s | %10s | %9s | %9s\n",
                r.feature, r.n, r.n_live, 100 * r.zero_share,
                eda_fmt(r.r_supremacy, "%+.4f"), eda_fmt(r.rho_supremacy, "%+.4f"),
                eda_fmt(r.r_total, "%+.4f"), eda_fmt(r.auc_home_win, "%.4f"))
    end
    return nothing
end

# ==============================================================================
# 11. AGREEMENT BETWEEN A PROXY AND A REFERENCE MEASUREMENT
# ==============================================================================
#
# Correlation alone cannot validate a proxy. A proxy that is exactly half the
# reference correlates at 1.000 and is wrong by a factor of two on every
# observation. The battery below separates the three ways a proxy can fail:
#
#   ASSOCIATION  does it move with the reference?        r, rho
#   ACCURACY     how far off is it, per observation?     MAE, RMSE
#   AGREEMENT    is it on the same scale?                bias, slope, sd ratio, CCC
#
# Lin's concordance correlation coefficient is the one number that punishes all
# three at once: CCC = 2*cov / (var_x + var_y + (mean_x - mean_y)^2). It equals
# Pearson r only when the proxy is unbiased AND unit-slope, and degrades toward
# zero as either fails.

"""
    eda_agreement(label, pred, actual) -> NamedTuple

The full proxy-validation battery. `pred` is the proxy, `actual` the reference.

`slope` and `intercept` come from regressing ACTUAL on PRED, so `slope > 1` means
the proxy is compressed (it under-spreads and must be scaled up) and `slope < 1`
means it over-spreads. `bias` is `mean(pred) - mean(actual)`: positive means the
proxy systematically over-states.
"""
function eda_agreement(label::AbstractString, pred::AbstractVector, actual::AbstractVector)
    p, a = eda_pairs(pred, actual)
    n = length(p)
    if n < 10
        return (label = String(label), n = n, r = NaN, rho = NaN, mae = NaN, rmse = NaN,
                bias = NaN, mape = NaN, slope = NaN, intercept = NaN, r2 = NaN,
                ccc = NaN, sd_ratio = NaN, mean_pred = NaN, mean_actual = NaN)
    end
    resid = p .- a
    fit = eda_ols(reshape(p, n, 1), a)
    mp, ma = mean(p), mean(a)
    sp, sa = std(p), std(a)
    covariance = n > 1 ? cov(p, a) : 0.0
    denominator = sp^2 + sa^2 + (mp - ma)^2
    nonzero = a .> 1e-9
    return (
        label = String(label),
        n = n,
        r = eda_pearson(p, a),
        rho = eda_spearman(p, a),
        mae = mean(abs, resid),
        rmse = sqrt(mean(abs2, resid)),
        bias = mp - ma,
        mape = count(nonzero) == 0 ? NaN : mean(abs.(resid[nonzero]) ./ a[nonzero]),
        slope = fit.beta[2],
        intercept = fit.beta[1],
        r2 = fit.r2,
        ccc = denominator < 1e-12 ? NaN : 2 * covariance / denominator,
        sd_ratio = sa < 1e-12 ? NaN : sp / sa,
        mean_pred = mp,
        mean_actual = ma,
    )
end

function eda_print_agreement(rows::Vector; title::AbstractString = "")
    isempty(title) || println("  ", title)
    @printf("  %-30s | %6s | %6s | %6s | %6s | %6s | %7s | %6s | %6s | %6s | %6s\n",
            "stratum", "n", "r", "rho", "MAE", "RMSE", "bias", "slope", "CCC", "sd_r", "R²")
    println("  " * repeat('-', 128))
    for r in rows
        @printf("  %-30s | %6d | %6s | %6s | %6s | %6s | %7s | %6s | %6s | %6s | %6s\n",
                r.label, r.n,
                eda_fmt(r.r, "%.3f"), eda_fmt(r.rho, "%.3f"),
                eda_fmt(r.mae, "%.3f"), eda_fmt(r.rmse, "%.3f"),
                eda_fmt(r.bias, "%+.3f"), eda_fmt(r.slope, "%.3f"),
                eda_fmt(r.ccc, "%.3f"), eda_fmt(r.sd_ratio, "%.3f"),
                eda_fmt(r.r2, "%.3f"))
    end
    return nothing
end

"""
    eda_scatter(x, y; width, height, xlab, ylab)

A density-shaded ASCII scatter. Cells are shaded by how many points land in them,
so the bulk of the cloud is legible instead of saturating to a solid block.
"""
function eda_scatter(x::AbstractVector, y::AbstractVector; width::Int = 64,
                     height::Int = 22, xlab::AbstractString = "x",
                     ylab::AbstractString = "y", title::AbstractString = "")
    xs, ys = eda_pairs(x, y)
    isempty(title) || println("  ", title)
    if length(xs) < 5
        println("    (too few observations to plot)")
        return nothing
    end
    xlo, xhi = minimum(xs), maximum(xs)
    ylo, yhi = minimum(ys), maximum(ys)
    (xhi - xlo < 1e-12 || yhi - ylo < 1e-12) && (println("    (degenerate range)"); return nothing)

    grid = zeros(Int, height, width)
    for i in eachindex(xs)
        col = clamp(1 + floor(Int, (xs[i] - xlo) / (xhi - xlo) * (width - 1)), 1, width)
        row = clamp(height - floor(Int, (ys[i] - ylo) / (yhi - ylo) * (height - 1)), 1, height)
        grid[row, col] += 1
    end
    peak = maximum(grid)
    shades = [' ', '·', ':', '+', '*', '#', '@']
    for r in 1:height
        label = r == 1 ? @sprintf("%8.2f", yhi) : r == height ? @sprintf("%8.2f", ylo) : repeat(" ", 8)
        line = IOBuffer()
        for c in 1:width
            v = grid[r, c]
            idx = v == 0 ? 1 : clamp(1 + ceil(Int, (length(shades) - 1) * v / peak), 2, length(shades))
            print(line, shades[idx])
        end
        println("    ", label, " |", String(take!(line)), "|")
    end
    println("    ", repeat(" ", 8), " +", repeat("-", width), "+")
    @printf("    %8s  %-*s%s\n", "", width - 10, @sprintf(" %.2f", xlo), @sprintf("%.2f ", xhi))
    println("    x = ", xlab, "   y = ", ylab,
            "   (shade = density, peak cell = $peak points)")
    return nothing
end

# ==============================================================================
# 12. HELD-OUT SIGNAL
# ==============================================================================
#
# ⚠ WHY THIS EXISTS, AND WHY `eda_boundary()` ALONE IS NOT ENOUGH.
#
# `eda_boundary()` puts every match in the history block, which is right for a
# DESCRIPTIVE pass but wrong for measuring signal against a feature that FITS
# something. The RAPM ridge is the case in point: with history = all matches, the
# rating vector is estimated on the very matches the covariate is then correlated
# against, so a goal-differential target partly memorises the goal differences it
# is being scored on. Measured that way `:y_goals` scores r = 0.42 against
# `:y_xg`'s 0.24 — almost all of which is the leak, not the target.
#
# The pxG form feature does NOT have this problem (its rolling walk is
# point-in-time whatever the boundary says), but it costs nothing to measure both
# features the same honest way.

"""
    eda_holdout(ds, configs; history_frac, dynamics_col) -> NamedTuple

Build `configs` under a chronological split and return everything needed to score
ONLY the held-out block: `(fs, frame, mask, n_history, n_target)`.

Anything that fits — the RAPM ridge, the shot-xG cell table — sees the history
block only, exactly as it would in production, and `mask` selects the target
matches it never saw.
"""
function eda_holdout(ds::EDA_DATA.DataStore,
                     configs::Vector{<:EDA_FEATURES.AbstractFeatureConfig};
                     history_frac::Float64 = 0.8,
                     dynamics_col::Symbol = :match_month)
    boundary = eda_boundary(ds; history_frac = history_frac)
    fs = eda_features(ds, configs; boundary = boundary, dynamics_col = dynamics_col)
    ids = Int.(fs.data[:ordered_match_ids])
    frame = eda_match_frame(ds; ordered_ids = ids)
    target = Set(Int.(boundary.target_match_ids))
    mask = Bool[id in target for id in ids]
    return (fs = fs, frame = frame, mask = mask,
            n_history = length(boundary.history_match_ids),
            n_target = length(boundary.target_match_ids))
end

"""
    eda_holdout_signal(ds, config, key, label; history_frac) -> NamedTuple

`eda_signal` restricted to the held-out block. `key` names the design column in
the feature set (e.g. `:flat_pxg_supremacy`, `:flat_pxg_rapm`).
"""
function eda_holdout_signal(ds::EDA_DATA.DataStore,
                            config::EDA_FEATURES.AbstractFeatureConfig,
                            key::Symbol, label::AbstractString;
                            history_frac::Float64 = 0.8)
    held = eda_holdout(ds, EDA_FEATURES.AbstractFeatureConfig[config];
                       history_frac = history_frac)
    column = Float64.(held.fs.data[key])
    return eda_signal(label, column[held.mask], held.frame[held.mask, :])
end
