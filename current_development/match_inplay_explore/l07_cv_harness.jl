#=
l07_cv_harness.jl  —  Repeated k-fold cross-validation for in-play model comparison.

Why: with ~253 matches a single 75/25 split's calibration is dominated by sampling noise (see
heavytail_diagnosis_report.md). This harness reports each metric as mean ± SE across many folds, and —
because all model specs share the same folds — PAIRED differences (with a t-stat), which is the
statistically correct way to tell a real improvement from noise.

Workhorse = the fast Poisson GLM (l02), which gives the same point estimates as the Bayesian baseline,
so repeated CV is cheap. Metrics per fold:
  - count_ll  : held-out mean Poisson log-likelihood of realized remaining goals (higher better)
  - mean_bias : mean(actual − predicted) remaining goals (≈0 ⇒ unbiased)
  - ECE/Brier/LogLoss : Over/Under calibration (lines 1.5/2.5/3.5), Poisson total.

Reuses: l01 panel, l02 dataset idea, l06 causal momentum (`row_net_momentum`).
=#

using DataFrames, GLM, Distributions, Statistics, Random, LinearAlgebra
include(joinpath(@__DIR__, "l06_momentum_feature.jl"))   # row_net_momentum / causal_momentum_auc

# ---- per-(bin,side) frame: covariates + settlement info (one place to add features) ----
"""
    build_bins(panel, fin, mom_lookup; tmax=80, resid_max=0.08) -> DataFrame

One row per (bin, side) with: match_id, t_m, t_m2, is_home, trailing, leading, man_adv, log_pregame,
momentum (standardised causal net), logrem (offset), rem_goals (target), T (current total), fh, fa.
`fin` = Dict(match_id => (home_score, away_score)).
"""
function build_bins(panel, fin, mom_lookup; tmax = 80.0, resid_max = 0.08)
    usable(r) = !ismissing(r.pg_λ_h) && r.t_m <= tmax && r.residual < resid_max && haskey(fin, r.match_id)
    # global momentum standardisation (scale only; absorbed by the GLM coefficient)
    mvals = Float64[]
    for r in eachrow(panel); usable(r) || continue
        push!(mvals, row_net_momentum(mom_lookup, r.match_id, r.t_m, 1.0))
        push!(mvals, row_net_momentum(mom_lookup, r.match_id, r.t_m, 0.0))
    end
    mc, ms = mean(mvals), std(mvals) + 1e-9
    rows = NamedTuple[]
    for r in eachrow(panel); usable(r) || continue
        fh, fa = fin[r.match_id]; T = r.gh + r.ga; rf = max((90.0 - r.t_m) / 90.0, 0.05)
        sides = ((1, fh - r.gh, r.gh - r.ga, r.away_reds - r.home_reds, log(r.pg_λ_h)),
                 (0, fa - r.ga, r.ga - r.gh, r.home_reds - r.away_reds, log(r.pg_λ_a)))
        for (ih, rem, gds, man, logpg) in sides
            rem < 0 && continue
            mz = (row_net_momentum(mom_lookup, r.match_id, r.t_m, Float64(ih)) - mc) / ms
            push!(rows, (match_id = r.match_id, t_m = r.t_m, t_m2 = r.t_m^2, is_home = Float64(ih),
                trailing = Float64(gds < 0), leading = Float64(gds > 0), man_adv = Float64(man),
                log_pregame = logpg, momentum = mz, logrem = log(rf), rem_goals = rem, T = T, fh = fh, fa = fa))
        end
    end
    return DataFrame(rows)
end

# ---- calibration metrics ----
brier(p, y)   = mean((p .- y).^2)
logloss(p, y) = -mean(y .* log.(clamp.(p, 1e-9, 1)) .+ (1 .- y) .* log.(clamp.(1 .- p, 1e-9, 1)))
function ece(p, y; nb = 10)
    e = 0.0; N = length(p)
    for b in 0:nb-1
        idx = findall(x -> b/nb <= x < (b+1)/nb || (b == nb-1 && x == 1.0), p); isempty(idx) && continue
        e += (length(idx)/N) * abs(mean(p[idx]) - mean(y[idx]))
    end
    return e
end

"Fit a GLM spec on the train rows, score on the test rows (count LL, mean bias, OU calibration)."
function eval_fold(form, Dtr, Dte; lines = (1.5, 2.5, 3.5))
    m = glm(form, Dtr, Poisson(), LogLink(); offset = Dtr.logrem)
    pred = predict(m, Dte; offset = Dte.logrem)
    cll = mean(logpdf.(Poisson.(pred), Dte.rem_goals))
    mb  = mean(Dte.rem_goals) - mean(pred)
    Te = copy(Dte); Te.pred = pred
    g = combine(groupby(Te, [:match_id, :t_m]), :pred => sum => :mtot,
                :T => first => :T, :fh => first => :fh, :fa => first => :fa)
    ps = Float64[]; ys = Float64[]
    for row in eachrow(g), L in lines
        need = Int(round(L - row.T + 0.5))
        push!(ps, need <= 0 ? 1.0 : ccdf(Poisson(row.mtot), need - 1))
        push!(ys, Float64((row.fh + row.fa) > L))
    end
    return (count_ll = cll, mean_bias = mb, ECE = ece(ps, ys), Brier = brier(ps, ys), LogLoss = logloss(ps, ys))
end

"Repeated k-fold splits over match ids → Vector of (train_ids::Set, test_ids::Set)."
function kfold_repeats(ids; k = 5, repeats = 4, seed0 = 1)
    folds = Tuple{Set{Int},Set{Int}}[]
    for rep in 1:repeats
        sh = shuffle(MersenneTwister(seed0 + rep), collect(ids)); n = length(sh)
        for f in 1:k
            te = sh[round(Int, (f-1)*n/k)+1 : round(Int, f*n/k)]
            push!(folds, (Set(setdiff(sh, te)), Set(te)))
        end
    end
    return folds
end

"Run every spec on every fold → long DataFrame (spec, fold, metrics)."
function run_cv(specs, BINS, folds)
    rows = NamedTuple[]
    for (fi, (tr, te)) in enumerate(folds)
        Dtr = subset(BINS, :match_id => ByRow(in(tr))); Dte = subset(BINS, :match_id => ByRow(in(te)))
        for (nm, form) in specs
            push!(rows, (spec = nm, fold = fi, eval_fold(form, Dtr, Dte)...))
        end
    end
    return DataFrame(rows)
end

sem(x) = std(x) / sqrt(length(x))

"Mean ± SE per spec for each metric."
function summarise_cv(R; metrics = (:count_ll, :mean_bias, :ECE, :Brier, :LogLoss))
    f(x) = "$(round(mean(x), digits=4)) ± $(round(sem(x), digits=4))"
    combine(groupby(R, :spec), [m => f => m for m in metrics]...)
end

"Paired difference spec `a` − spec `b` on shared folds (mean, SE, t)."
function paired_diff(R, metric, a, b)
    w = unstack(R, :fold, :spec, metric); d = w[!, a] .- w[!, b]
    (mean = round(mean(d), digits = 4), se = round(sem(d), digits = 4), t = round(mean(d) / sem(d), digits = 2))
end
