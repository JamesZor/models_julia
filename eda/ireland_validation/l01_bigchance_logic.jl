# eda/ireland_validation/l01_bigchance_logic.jl
#
# Loader (math / functions only — NO top-level execution).
#
# Deep EDA on `bigChanceCreated` (SofaScore, per-team-per-match count) for the
# League of Ireland. Goal: characterise it as a discrete count random variable
# (expectation, variance, dispersion regime, zero-mass) and decide which count
# family fits best — Poisson / Negative-Binomial / Weibull-count / ZIP / ZINB /
# COM-Poisson — before it becomes a joint likelihood pillar tied to the latent
# attack rate λ in the outfield Dixon-Coles model.
#
# Pairs with r02_bigchance_runner.jl.

using DataFrames
using Distributions
using Statistics
using StatsBase
using HypothesisTests
using GLM
using Optim
using Printf
using SpecialFunctions: loggamma
using BayesianFootball

# ============================================================================
# 0. SMALL NUMERICAL HELPERS
# ============================================================================

"Numerically stable log(exp(a) + exp(b))."
logaddexp(a::Real, b::Real) = a > b ? a + log1p(exp(b - a)) : b + log1p(exp(a - b))

"Numerically stable log(sum(exp.(xs)))."
function logsumexp(xs::AbstractVector{<:Real})
    m = maximum(xs)
    isfinite(m) || return m
    return m + log(sum(exp.(xs .- m)))
end

logistic(x::Real) = 1.0 / (1.0 + exp(-x))
logit(p::Real) = log(p / (1.0 - p))

# ============================================================================
# 1. REFERENCE FITTERS (re-declared from eda/basic_goals/l00 to stay self-
#    contained — that file has top-level execution and cannot be `include`d
#    safely here).
# ============================================================================

"MLE fit of the project's RobustNegativeBinomial(r, μ). r = dispersion, μ = mean."
function fit_mle(::Type{MyDistributions.RobustNegativeBinomial}, data)
    m = mean(data)
    v = var(data)
    r_guess = v > m ? m^2 / (v - m) : 10.0
    func(p) = -sum(logpdf(MyDistributions.RobustNegativeBinomial(exp(p[1]), exp(p[2])), data))
    res = optimize(func, [log(r_guess), log(max(m, 1e-4))])
    return MyDistributions.RobustNegativeBinomial(exp(res.minimizer[1]), exp(res.minimizer[2]))
end

"MLE fit of the project's WeibullCount(c, λ). c = hazard shape (c<1 over-, c>1 under-dispersed)."
function fit_mle(::Type{MyDistributions.WeibullCount}, data)
    c_guess = 1.0
    λ_guess = max(mean(data), 1e-4)
    function objective(p)
        dist = MyDistributions.WeibullCount(exp(p[1]), exp(p[2]))
        return -sum(max(logpdf(dist, x), -1e6) for x in data)
    end
    res = optimize(objective, [log(c_guess), log(λ_guess)], NelderMead(),
                   Optim.Options(iterations = 2000))
    return MyDistributions.WeibullCount(exp(res.minimizer[1]), exp(res.minimizer[2]))
end

# ============================================================================
# 2. INLINE COUNT-DISTRIBUTION FITTERS (no MyDistributions type yet — only
#    graduate to src/ if one of these wins).
#
# Each fitter returns a NamedTuple:
#   (name, k, loglik, aic, bic, params, pmf)
# where `pmf(j)` is the fitted probability mass at integer j ≥ 0. The pmf
# closure lets the SAME goodness-of-fit machinery (rootogram / χ²) run for
# every family, whether or not it is a Distributions.jl type.
# ============================================================================

_aic(k, ll) = 2k - 2ll
_bic(k, ll, n) = k * log(n) - 2ll

"""
    fit_zip(data)

Zero-Inflated Poisson. A structural-zero mixture:

    P(Y=0) = π + (1-π)·e^{-λ}
    P(Y=k) = (1-π)·Poisson(k; λ),  k ≥ 1

`π` is the probability of the always-zero (structural) state, `λ` the Poisson
rate of the count state. 2 free parameters.
"""
function fit_zip(data)
    n = length(data)
    λ0 = max(mean(data), 1e-4)
    p0_emp = mean(data .== 0)
    π0 = clamp(p0_emp - exp(-λ0), 1e-3, 0.9)   # excess zeros beyond Poisson

    function nll(p)
        π = logistic(p[1]); λ = exp(p[2])
        s = 0.0
        for x in data
            if x == 0
                s += logaddexp(log(π), log1p(-π) - λ)
            else
                s += log1p(-π) + logpdf(Poisson(λ), x)
            end
        end
        return -s
    end

    res = optimize(nll, [logit(π0), log(λ0)], NelderMead(), Optim.Options(iterations = 2000))
    π = logistic(res.minimizer[1]); λ = exp(res.minimizer[2])
    ll = -res.minimum
    pmf = j -> j == 0 ? π + (1 - π) * exp(-λ) : (1 - π) * pdf(Poisson(λ), j)
    return (name = "ZIP", k = 2, loglik = ll, aic = _aic(2, ll), bic = _bic(2, ll, n),
            params = (π = π, λ = λ), pmf = pmf, converged = Optim.converged(res))
end

"""
    fit_zinb(data)

Zero-Inflated Negative Binomial: structural-zero mixture with an over-dispersed
NB count state (reuses RobustNegativeBinomial(r, μ)).

    P(Y=0) = π + (1-π)·NB(0; r, μ)
    P(Y=k) = (1-π)·NB(k; r, μ),  k ≥ 1

3 free parameters (π, r, μ).
"""
function fit_zinb(data)
    n = length(data)
    μ0 = max(mean(data), 1e-4); v0 = var(data)
    r0 = v0 > μ0 ? μ0^2 / (v0 - μ0) : 10.0
    p0_emp = mean(data .== 0)
    nb0 = MyDistributions.RobustNegativeBinomial(r0, μ0)
    π0 = clamp(p0_emp - exp(logpdf(nb0, 0)), 1e-3, 0.9)

    function nll(p)
        π = logistic(p[1]); r = exp(p[2]); μ = exp(p[3])
        nb = MyDistributions.RobustNegativeBinomial(r, μ)
        s = 0.0
        for x in data
            lp = logpdf(nb, x)
            if x == 0
                s += logaddexp(log(π), log1p(-π) + lp)
            else
                s += log1p(-π) + lp
            end
        end
        return -s
    end

    res = optimize(nll, [logit(π0), log(r0), log(μ0)], NelderMead(),
                   Optim.Options(iterations = 3000))
    π = logistic(res.minimizer[1]); r = exp(res.minimizer[2]); μ = exp(res.minimizer[3])
    ll = -res.minimum
    nb = MyDistributions.RobustNegativeBinomial(r, μ)
    pmf = j -> j == 0 ? π + (1 - π) * exp(logpdf(nb, 0)) : (1 - π) * exp(logpdf(nb, j))
    return (name = "ZINB", k = 3, loglik = ll, aic = _aic(3, ll), bic = _bic(3, ll, n),
            params = (π = π, r = r, μ = μ), pmf = pmf, converged = Optim.converged(res))
end

"""
    com_logpmf(j, log_λ, ν, logZ)

Conway-Maxwell-Poisson log-PMF:

    P(Y=j) = (λ^j / (j!)^ν) / Z(λ, ν),   Z(λ,ν) = Σ_{i≥0} λ^i / (i!)^ν

`ν` is the dispersion knob: ν=1 → Poisson, ν<1 → over-dispersed, ν>1 → under-
dispersed. Computed in log-space for stability.
"""
com_logpmf(j, log_λ, ν, logZ) = j * log_λ - ν * loggamma(j + 1) - logZ

"Stable log normalising constant logZ(λ, ν), truncated at upper bound J."
function com_logZ(log_λ, ν, J::Int)
    terms = [i * log_λ - ν * loggamma(i + 1) for i in 0:J]
    return logsumexp(terms)
end

"""
    fit_compoisson(data)

Conway-Maxwell-Poisson MLE. Truncates the series Z at J = max(data) + 50 (well
into the negligible tail for these means). 2 free parameters (λ, ν).
"""
function fit_compoisson(data)
    n = length(data)
    J = maximum(data) + 50
    λ0 = max(mean(data), 1e-4)

    function nll(p)
        log_λ = p[1]; ν = exp(p[2])
        logZ = com_logZ(log_λ, ν, J)
        s = 0.0
        for x in data
            s += com_logpmf(x, log_λ, ν, logZ)
        end
        return -s
    end

    res = optimize(nll, [log(λ0), log(1.0)], NelderMead(), Optim.Options(iterations = 3000))
    log_λ = res.minimizer[1]; ν = exp(res.minimizer[2])
    λ = exp(log_λ)
    ll = -res.minimum
    logZ = com_logZ(log_λ, ν, J)
    pmf = j -> exp(com_logpmf(j, log_λ, ν, logZ))
    return (name = "COM-Poisson", k = 2, loglik = ll, aic = _aic(2, ll), bic = _bic(2, ll, n),
            params = (λ = λ, ν = ν), pmf = pmf, converged = Optim.converged(res))
end

# ---- wrappers for the standard 3 families, in the same NamedTuple shape ----

function fit_poisson_entry(data)
    n = length(data); λ = mean(data)
    d = Poisson(λ)
    ll = sum(logpdf(d, x) for x in data)
    return (name = "Poisson", k = 1, loglik = ll, aic = _aic(1, ll), bic = _bic(1, ll, n),
            params = (λ = λ,), pmf = j -> pdf(d, j), converged = true)
end

function fit_negbin_entry(data)
    n = length(data)
    d = fit_mle(MyDistributions.RobustNegativeBinomial, data)
    ll = sum(logpdf(d, x) for x in data)
    return (name = "NegBin", k = 2, loglik = ll, aic = _aic(2, ll), bic = _bic(2, ll, n),
            params = (r = d.r, μ = d.μ), pmf = j -> exp(logpdf(d, j)), converged = true)
end

function fit_weibull_entry(data)
    n = length(data)
    d = fit_mle(MyDistributions.WeibullCount, data)
    ll = sum(logpdf(d, x) for x in data)
    return (name = "WeibullCount", k = 2, loglik = ll, aic = _aic(2, ll), bic = _bic(2, ll, n),
            params = (c = d.c, λ = d.λ), pmf = j -> exp(logpdf(d, j)), converged = true)
end

# ============================================================================
# 3. MARGINAL SUMMARY (moments / dispersion / zero-mass)
# ============================================================================

"""
    summarise_count(data, label)

Mean, variance, index of dispersion (V/M), empirical vs Poisson-implied zero
mass, max, and skewness. V/M > 1 ⇒ over-dispersed (NB territory); V/M < 1 ⇒
under-dispersed (COM-Poisson ν>1 / Weibull c>1 territory).
"""
function summarise_count(data::AbstractVector{<:Integer}, label::String)
    m = mean(data); v = var(data); n = length(data)
    di = v / m
    p0_emp = mean(data .== 0)
    p0_pois = exp(-m)
    sk = skewness(data)
    mx = maximum(data)

    println("\n" * "═"^60)
    println(" COUNT SUMMARY: $(uppercase(label))  (n=$n)")
    println("═"^60)
    @printf("Mean: %.4f | Variance: %.4f | Index of Dispersion V/M: %.4f\n", m, v, di)
    @printf("Zeros (empirical): %.4f | Zeros (Poisson-implied e^-μ): %.4f | excess: %+.4f\n",
            p0_emp, p0_pois, p0_emp - p0_pois)
    @printf("Max: %d | Skewness: %.4f\n", mx, sk)
    regime = di > 1.05 ? "OVER-dispersed" : (di < 0.95 ? "UNDER-dispersed" : "≈ equidispersed")
    println("Regime: $regime")
    return (mean = m, var = v, di = di, p0_emp = p0_emp, p0_pois = p0_pois,
            excess_zeros = p0_emp - p0_pois, max = mx, skew = sk, n = n)
end

# ============================================================================
# 4. MODEL COMPARISON (the core distribution decision)
# ============================================================================

"""
    compare_count_models(data, label)

Fit Poisson, NegBin, Weibull-count, ZIP, ZINB, COM-Poisson; tabulate
LL / k / AIC / BIC and declare the winner by AIC and by BIC.

AIC = 2k − 2ℓ, BIC = k·ln(n) − 2ℓ. BIC penalises the 3-parameter mixtures
(ZINB) harder, so a split AIC/BIC verdict flags marginal complexity.
Returns the vector of fitted-model NamedTuples (sorted by AIC).
"""
function compare_count_models(data::AbstractVector{<:Integer}, label::String)
    fits = [
        fit_poisson_entry(data),
        fit_negbin_entry(data),
        fit_weibull_entry(data),
        fit_zip(data),
        fit_zinb(data),
        fit_compoisson(data),
    ]

    println("\n" * "═"^72)
    println(" MODEL COMPARISON: $(uppercase(label))  (n=$(length(data)))")
    println("═"^72)
    @printf("%-14s | %-3s | %-12s | %-12s | %-12s | %-5s\n",
            "Model", "k", "LogLik", "AIC", "BIC", "conv")
    println("-"^72)
    for f in Base.sort(fits, by = x -> x.aic)
        @printf("%-14s | %-3d | %-12.2f | %-12.2f | %-12.2f | %-5s\n",
                f.name, f.k, f.loglik, f.aic, f.bic, f.converged)
    end

    best_aic = fits[argmin([f.aic for f in fits])]
    best_bic = fits[argmin([f.bic for f in fits])]
    println("-"^72)
    println("Winner by AIC: $(best_aic.name)  (params = $(best_aic.params))")
    println("Winner by BIC: $(best_bic.name)  (params = $(best_bic.params))")
    return Base.sort(fits, by = x -> x.aic)
end

# ============================================================================
# 5. GOODNESS-OF-FIT (rootogram + Pearson χ²) — works for any fitted entry via
#    its `pmf` closure.
# ============================================================================

"""
    rootogram_data(data, pmf; maxbin)

Hanging-rootogram table on the √-scale. For each count j: observed O_j,
expected E_j = n·pmf(j); the rootogram "hangs" √E from √O, so a well-fit model
leaves `hang = √O − √E` near zero across all bins.
"""
function rootogram_data(data::AbstractVector{<:Integer}, pmf; maxbin = maximum(data))
    n = length(data)
    js = 0:maxbin
    O = [count(==(j), data) for j in js]
    E = [n * pmf(j) for j in js]
    df = DataFrame(count = collect(js), observed = O, expected = round.(E, digits = 2),
                   sqrtO = round.(sqrt.(O), digits = 3), sqrtE = round.(sqrt.(E), digits = 3),
                   hang = round.(sqrt.(O) .- sqrt.(E), digits = 3))
    return df
end

"""
    chi_square_gof(data, pmf, k_params; maxbin)

Pearson discrete goodness-of-fit. Bins 0..(maxbin−1) plus a pooled tail
(≥ maxbin) so probabilities sum to 1.

    χ² = Σ (O − E)² / E,   df = (#bins − 1 − k_params)

Large p ⇒ no evidence against the fitted distribution.
"""
function chi_square_gof(data::AbstractVector{<:Integer}, pmf, k_params::Int; maxbin = maximum(data))
    n = length(data)
    js = 0:(maxbin - 1)
    O = Float64[count(==(j), data) for j in js]
    E = Float64[n * pmf(j) for j in js]
    # pooled tail ≥ maxbin
    tail_p = max(1.0 - sum(pmf(j) for j in 0:(maxbin - 1)), 0.0)
    push!(O, count(>=(maxbin), data)); push!(E, n * tail_p)
    Es = max.(E, 1e-8)
    χ2 = sum((O .- Es).^2 ./ Es)
    nbins = length(O)
    dof = nbins - 1 - k_params
    pval = dof > 0 ? ccdf(Chisq(dof), χ2) : NaN
    @printf("χ² GoF: χ²=%.3f | bins=%d | df=%d | p=%.4f\n", χ2, nbins, dof, pval)
    return (chi2 = χ2, bins = nbins, df = dof, p = pval)
end

# ============================================================================
# 6. LINK ANALYSIS — how does bigChanceCreated relate to the SHARED latent
#    attack rate (goals & xG)? This decides the pillar's link form next session.
# ============================================================================

"""
    build_bigchance_long(ds)

Per-team-per-match long table with one row per (match, side):
columns `match_id, team, is_home, big_chance, goals, xg`.
Drops rows with missing bigChanceCreated. bigChance is rounded to Int.
"""
function build_bigchance_long(ds::Data.DataStore)
    stats = filter(r -> r.period == "ALL", ds.statistics)
    smap = Dict(r.match_id => r for r in eachrow(stats))

    rows = NamedTuple[]
    for mr in eachrow(ds.matches)
        haskey(smap, mr.match_id) || continue
        ismissing(mr.home_score) && continue
        s = smap[mr.match_id]
        # home side
        if !ismissing(s.bigChanceCreated_home)
            push!(rows, (match_id = mr.match_id, team = mr.home_team, is_home = true,
                         big_chance = round(Int, s.bigChanceCreated_home),
                         goals = Int(mr.home_score),
                         xg = ismissing(s.expectedGoals_home) ? NaN : Float64(s.expectedGoals_home)))
        end
        # away side
        if !ismissing(s.bigChanceCreated_away)
            push!(rows, (match_id = mr.match_id, team = mr.away_team, is_home = false,
                         big_chance = round(Int, s.bigChanceCreated_away),
                         goals = Int(mr.away_score),
                         xg = ismissing(s.expectedGoals_away) ? NaN : Float64(s.expectedGoals_away)))
        end
    end
    return DataFrame(rows)
end

"""
    bigchance_vs_outcomes(long_df)

Pearson + Spearman correlation of bigChanceCreated with goals and with xG, plus
a Poisson GLM `goals ~ big_chance` (does chance creation predict goals?) and an
OLS `big_chance ~ xg` (is bigChance a scaled view of xG?). These tell us whether
E[bigChance] is a monotone multiple of the attack rate λ.
"""
function bigchance_vs_outcomes(long_df::DataFrame)
    bc = Float64.(long_df.big_chance)
    g  = Float64.(long_df.goals)

    println("\n" * "═"^60)
    println(" LINK: bigChanceCreated vs GOALS / xG")
    println("═"^60)
    @printf("corr(bigChance, goals): Pearson %.4f | Spearman %.4f\n",
            cor(bc, g), corspearman(bc, g))

    has_xg = .!isnan.(long_df.xg)
    if any(has_xg)
        bcx = bc[has_xg]; xgx = Float64.(long_df.xg[has_xg])
        @printf("corr(bigChance, xG)   : Pearson %.4f | Spearman %.4f  (n=%d)\n",
                cor(bcx, xgx), corspearman(bcx, xgx), length(xgx))
    end

    # Poisson GLM: goals ~ big_chance
    glm_df = DataFrame(goals = long_df.goals, big_chance = long_df.big_chance)
    pglm = glm(@formula(goals ~ big_chance), glm_df, Poisson(), LogLink())
    println("\nPoisson GLM  goals ~ big_chance:")
    println(coeftable(pglm))

    # OLS: big_chance ~ xg (where xG available)
    if any(has_xg)
        ols_df = DataFrame(big_chance = bc[has_xg], xg = Float64.(long_df.xg[has_xg]))
        ols = lm(@formula(big_chance ~ xg), ols_df)
        println("\nOLS  big_chance ~ xg:")
        println(coeftable(ols))
    end
    return pglm
end

"""
    mean_variance_scaling(long_df; min_matches=20)

Per-team mean vs variance of bigChanceCreated. Fits the NB mean-variance law
`Var = Mean + α·Mean²` (α = 1/r) against the Poisson law `Var = Mean`. A
positive, significant α confirms genuine over-dispersion (NB/ZINB) rather than
pure Poisson noise.
"""
function mean_variance_scaling(long_df::DataFrame; min_matches::Int = 20)
    g = combine(groupby(long_df, :team),
                :big_chance => mean => :mean_bc,
                :big_chance => var  => :var_bc,
                nrow => :n)
    filter!(r -> r.n >= min_matches, g)

    println("\n" * "═"^60)
    println(" MEAN–VARIANCE SCALING (per team, n≥$min_matches)")
    println("═"^60)
    @printf("Teams: %d | mean of team V/M: %.4f\n", nrow(g), mean(g.var_bc ./ g.mean_bc))

    # Var - Mean = α·Mean²  → regress (var-mean) on mean² through origin.
    g.y = g.var_bc .- g.mean_bc
    g.m2 = g.mean_bc .^ 2
    fit = lm(@formula(y ~ 0 + m2), g)
    α = coef(fit)[1]
    println("\nNB law  Var − Mean = α·Mean²  (through origin):")
    println(coeftable(fit))
    if α > 0
        @printf("→ α = %.4f  ⇒ implied NB dispersion r ≈ 1/α = %.3f (over-dispersion present)\n", α, 1 / α)
    else
        @printf("→ α = %.4f ≤ 0  ⇒ no NB-style over-dispersion at team level\n", α)
    end
    return g
end

"""
    home_away_bigchance(long_df)

Home vs away mean/variance of bigChanceCreated and a Mann-Whitney U test — does
the home-advantage seen on goals also appear on chance creation?
"""
function home_away_bigchance(long_df::DataFrame)
    h = Float64.(long_df.big_chance[long_df.is_home])
    a = Float64.(long_df.big_chance[.!long_df.is_home])

    println("\n" * "═"^60)
    println(" HOME vs AWAY bigChanceCreated")
    println("═"^60)
    @printf("Home: mean %.4f var %.4f (n=%d)\n", mean(h), var(h), length(h))
    @printf("Away: mean %.4f var %.4f (n=%d)\n", mean(a), var(a), length(a))
    @printf("Difference in means: %+.4f\n", mean(h) - mean(a))
    mwu = MannWhitneyUTest(h, a)
    @printf("Mann-Whitney U p-value: %.4e\n", pvalue(mwu))
    if pvalue(mwu) < 0.05
        println("Result: statistically significant home advantage on chance creation.")
    else
        println("Result: no significant home advantage on chance creation.")
    end
    return (home_mean = mean(h), away_mean = mean(a), p = pvalue(mwu))
end
