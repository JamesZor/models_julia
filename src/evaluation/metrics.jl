# src/evaluation/metrics.jl
#
# The typed scoring kernels, and the calibration metrics.
#
# ------------------------------------------------------------------------------
# THE PARITY CONSTRAINT, AND WHAT IT DICTATES
# ------------------------------------------------------------------------------
#
# The six kernels in §3 must produce the SAME Float64 as the ones in `metrics_methods/`,
# not merely the same real number. Two consequences run through them:
#
#   * ROW ORDER IS PRESERVED. Every aggregate is a `mean` or `sum` over a vector, and
#     floating-point addition is not associative, so the rows must be accumulated in the
#     order `innerjoin(ds.odds, …)` produced them — which is `ds.odds`' own order. Hence
#     `evaluation_rows` walks the odds table, not the fixtures.
#
#   * THE SCALAR FORMULAE ARE REUSED, NOT RE-DERIVED. `calc_logloss`, `calc_lpd_scalar`,
#     `compute_crps`, `get_miq` and `evaluate_group_edge` are the functions the legacy
#     kernels already call, in this same module. Where a body is written afresh here it
#     is because the legacy one allocates per call, and the rewrite is documented term
#     for term at the site.
#
# ------------------------------------------------------------------------------
# MARGINALS BY DISPATCH
# ------------------------------------------------------------------------------
#
# `crps.jl`'s `_crps_get_r` and `rqr.jl`'s `_rqr_get_r` pick the goal distribution with a
# `hasproperty(df, :r)` / `hasproperty(df, :r_h)` cascade and an `Inf` sentinel meaning
# "Poisson": a frame carrying neither column reaches the Poisson branch by falling off
# the end of an `if`. Here the choice is method dispatch on the container type, so a
# container with no dispersion cannot reach the negative-binomial method at all.

export marginals, crps_parameters, posterior_mean
export summarize_stats, calc_lpd_samples!
export brier_score, expected_calibration_error, max_calibration_error,
       calibration_curve, ranked_probability_score, CalibrationCurve
export PredictionScore, PredictionScores, CalibrationComponent, evaluate_predictions


# ==============================================================================
# 1. GOAL-DISTRIBUTION DISPATCHES
# ==============================================================================

"""
    posterior_mean(M, i) -> Float64

Posterior mean of parameter row `i`. Zero allocations, and bit-identical to
`mean(M[i, :])` — a strided `SubArray` reduces with the same pairwise blocking Base uses
on the contiguous `Vector` the legacy PPD frame stored, which is what makes the two
means the same `Float64` rather than merely the same number.
"""
@inline posterior_mean(M::AbstractMatrix{Float64}, i::Integer) = mean(view(M, i, :))

"""
    marginals(latents, i) -> (home::UnivariateDistribution, away::UnivariateDistribution)

The plug-in marginal goal distributions for fixture row `i`.

| container                       | marginal                             |
|---------------------------------|--------------------------------------|
| `CountLatents{T, Nothing}`      | `Poisson(λ̄)`                         |
| `CountLatents{T, <:NamedTuple}` | `NegativeBinomial(r̄, r̄/(r̄+λ̄))`       |
| `RecombLatents{T}`              | `Poisson(λ̄_open + λ̄_pen + λ̄_og)`     |
| `SmileLatents{T, Nothing}`      | `Poisson(λ̄)` on the GRID intensities |
| `SmileLatents{T, <:NamedTuple}` | `NegativeBinomial` on the grid ones   |

PLUG-IN, NOT POSTERIOR-PREDICTIVE: one marginal built from the posterior MEAN λ, rather
than the posterior-averaged CDF. That is inherited from `crps.jl` and `rqr.jl` rather
than chosen — the Bayesian form is strictly better calibrated, and changing it here
would make every CRPS number in this repository's history incomparable with every new
one for a reason unrelated to the container swap.

The smile methods deliberately read `λ_home`/`λ_away` and NOT `λ_tot·φ`: CRPS, RQR and
PIT are per-SIDE goal-count diagnostics and the smile curve prices a TOTAL. Using the
pricing intensity here would test a different distribution from the one being reported.
"""
function marginals end

marginals(l::CountLatents{Float64, Nothing}, i::Integer) =
    (Poisson(posterior_mean(l.λ_home, i)), Poisson(posterior_mean(l.λ_away, i)))

function marginals(l::CountLatents{Float64, <:NamedTuple}, i::Integer)
    λh = posterior_mean(l.λ_home, i)
    λa = posterior_mean(l.λ_away, i)
    rh = posterior_mean(l.observation_params.r_h, i)
    ra = posterior_mean(l.observation_params.r_a, i)
    return (NegativeBinomial(rh, rh / (rh + λh)),
            NegativeBinomial(ra, ra / (ra + λa)))
end

function marginals(l::RecombLatents{Float64}, i::Integer)
    nd = n_draws(l)
    sh = 0.0
    sa = 0.0
    @inbounds for k in 1:nd
        sh += recomb_total_home(l, i, k)
        sa += recomb_total_away(l, i, k)
    end
    return (Poisson(sh / nd), Poisson(sa / nd))
end

marginals(l::SmileLatents{Float64, Nothing}, i::Integer) =
    (Poisson(posterior_mean(l.λ_home, i)), Poisson(posterior_mean(l.λ_away, i)))

function marginals(l::SmileLatents{Float64, <:NamedTuple}, i::Integer)
    λh = posterior_mean(l.λ_home, i)
    λa = posterior_mean(l.λ_away, i)
    rh = posterior_mean(l.observation_params.r_h, i)
    ra = posterior_mean(l.observation_params.r_a, i)
    return (NegativeBinomial(rh, rh / (rh + λh)),
            NegativeBinomial(ra, ra / (ra + λa)))
end

"""
    crps_parameters(latents, i) -> (λ_h, r_h, λ_a, r_a)

The four numbers `compute_crps` takes, with `Inf` for "no dispersion" so the shared
legacy formula can be called verbatim. Separate from `marginals` because `compute_crps`
builds its own distribution, and reproducing that construction rather than passing a
distribution in is what keeps the parity exact.
"""
crps_parameters(l::CountLatents{Float64, Nothing}, i::Integer) =
    (posterior_mean(l.λ_home, i), Inf, posterior_mean(l.λ_away, i), Inf)

crps_parameters(l::CountLatents{Float64, <:NamedTuple}, i::Integer) =
    (posterior_mean(l.λ_home, i), posterior_mean(l.observation_params.r_h, i),
     posterior_mean(l.λ_away, i), posterior_mean(l.observation_params.r_a, i))

crps_parameters(l::SmileLatents{Float64, Nothing}, i::Integer) =
    (posterior_mean(l.λ_home, i), Inf, posterior_mean(l.λ_away, i), Inf)

crps_parameters(l::SmileLatents{Float64, <:NamedTuple}, i::Integer) =
    (posterior_mean(l.λ_home, i), posterior_mean(l.observation_params.r_h, i),
     posterior_mean(l.λ_away, i), posterior_mean(l.observation_params.r_a, i))

function crps_parameters(l::RecombLatents{Float64}, i::Integer)
    dh, da = marginals(l, i)
    return (mean(dh), Inf, mean(da), Inf)
end


# ==============================================================================
# 2. SCALAR HELPERS THE TYPED PATH ADDS
# ==============================================================================

"""
    calc_lpd_samples!(log_liks, expbuf, samples, y) -> Float64

`log( (1/S) Σ_s p(y | θ^s) )` by log-sum-exp, into caller-supplied scratch.

TERM FOR TERM `metrics_methods/lpd.jl`'s `calc_lpd_samples`, which is

    log_liks = log.(clamp.(samples, 1e-15, 1-1e-15))   # or 1 .- samples for y = 0
    lmax     = maximum(log_liks)
    lmax + log(mean(exp.(log_liks .- lmax)))

with the two broadcast temporaries hoisted into reusable buffers. The reductions are
still Base's `maximum` and `mean` over the same length in the same order, so the result
is bit-identical — which is why the buffers are `Vector`s rather than an on-line
accumulation.
"""
function calc_lpd_samples!(log_liks::Vector{Float64}, expbuf::Vector{Float64},
                           samples::AbstractVector{Float64}, y::Float64)
    n = length(samples)
    length(log_liks) >= n || resize!(log_liks, n)
    length(expbuf) >= n || resize!(expbuf, n)
    ll = view(log_liks, 1:n)
    eb = view(expbuf, 1:n)
    if y == 1.0
        @inbounds for k in 1:n
            ll[k] = log(clamp(samples[k], 1e-15, 1.0 - 1e-15))
        end
    else
        @inbounds for k in 1:n
            ll[k] = log(clamp(1.0 - samples[k], 1e-15, 1.0 - 1e-15))
        end
    end
    lmax = maximum(ll)
    @inbounds for k in 1:n
        eb[k] = exp(ll[k] - lmax)
    end
    return lmax + log(mean(eb))
end

"""
    compute_rqr(y, dist, rng) -> Float64

`metrics_methods/rqr.jl`'s `compute_rqr`, with the distribution passed in rather than
reconstructed from an `Inf` sentinel, and with an EXPLICIT rng.

    u ~ Uniform(F(y−1), F(y)),   r = Φ⁻¹(clamp(u, 1e-7, 1−1e-7))

One `rand` per call, so a seeded rng reproduces the legacy sequence exactly provided the
calls happen in the same order — which the typed RQR kernel guarantees (every home
residual in fixture order, then every away one).
"""
function compute_rqr(y::Integer, dist::UnivariateDistribution, rng::AbstractRNG)
    cdf_lower = y > 0 ? cdf(dist, y - 1) : 0.0
    cdf_upper = cdf(dist, y)
    u = rand(rng, Uniform(cdf_lower, cdf_upper))
    u = clamp(u, 1e-7, 1.0 - 1e-7)
    return quantile(Normal(0, 1), u)
end

"""
    summarize_stats(x) -> DistributionStats

`metrics_methods/rqr.jl`'s `_summarize_stats`, made total. Fewer than three points makes
`ShapiroWilkTest` throw, so that case reports `NaN` for `W` and `p` and keeps the
moments rather than losing the whole row.
"""
function summarize_stats(x::Vector{Float64})::DistributionStats
    n = length(x)
    if n < 3
        return DistributionStats(n == 0 ? NaN : mean(x), n < 2 ? NaN : std(x),
                                 NaN, NaN, NaN, NaN)
    end
    w, p = try
        sw = ShapiroWilkTest(x)
        (sw.W, pvalue(sw))
    catch
        (NaN, NaN)
    end
    return DistributionStats(mean(x), std(x), skewness(x), kurtosis(x), w, p)
end

"""
    get_miq(samples::AbstractVector{Float64}, market_prob) -> Float64 or missing

`metrics_methods/miq.jl`'s `get_miq` without the broadcast temporary, for the contiguous
tensor VIEW the typed path hands it. The `Vector{Float64}` method in `miq.jl` is more
specific and still wins for a legacy caller, so nothing that worked changes.
"""
@inline function get_miq(samples::AbstractVector{Float64}, market_prob::Float64)
    isnan(market_prob) && return missing
    c = 0
    @inbounds for k in eachindex(samples)
        samples[k] <= market_prob && (c += 1)
    end
    return c / length(samples)
end


# ==============================================================================
# 3. THE SIX TYPED KERNELS
# ==============================================================================
#
# One pass over the odds rows, in odds-frame order, with two integer lookups per row.
#
# The per-rule OPTIONS the prototype carried as struct fields are keywords here instead:
# `LogLoss`, `LPD`, `CRPS`, `RQR`, `GLMEdge` and `MIQ` are the EXISTING trigger types in
# `metrics_methods/`, and adding fields to them would change `propertynames`, the
# serialised shape of every saved trigger, and the column suffix rule — for no gain that
# a keyword does not give.

"""
    compute_metric(metric, ctx::EvaluationContext) -> AbstractEvaluationResult

The typed kernel. Reads the prebuilt indexes; builds nothing.
"""
function compute_metric end

# --- 3.1 LogLoss ---------------------------------------------------------------

function compute_metric(m::LogLoss, ctx::EvaluationContext)::LogLossResult
    rows = evaluation_rows(ctx; selections = _selection_filter(m))
    n = length(rows)
    n == 0 && return LogLossResult(LogLossComponent(NaN, NaN, NaN, 0))

    model_ll = Vector{Float64}(undef, n)
    market_ll = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        r = rows[i]
        y = row_y(r)
        model_ll[i] = calc_logloss(r.model_prob, y)
        market_ll[i] = calc_logloss(r.market_prob, y)
    end

    mm = mean(model_ll)
    mk = mean(market_ll)
    return LogLossResult(LogLossComponent(mm, mk, mm - mk, n))
end

# --- 3.2 LPD -------------------------------------------------------------------

"""
    compute_metric(::LPD, ctx; target = :market) -> LPDResult

Log posterior predictive density — the log of the POSTERIOR MEAN likelihood, not the
likelihood at the posterior mean, evaluated by log-sum-exp.

TWO TARGETS:

| `target`  | `y_i` is                       | baseline                              |
|-----------|--------------------------------|---------------------------------------|
| `:market` | a binary market outcome        | `log p_fair_close` — the closing line |
| `:score`  | the realised `(g_h, g_a)` pair | none; the market fields are `NaN`     |

`:market` is `metrics_methods/lpd.jl` and is the default, so a caller that switches to
the typed path gets the number it already had. `:score` is the joint density of the
realised scoreline read straight off the score grid — a strictly sharper test, and the
one that can distinguish two models pricing 1X2 identically out of different score
distributions. No market baseline is invented for it: the odds table carries no full
correct-score distribution for these leagues, so `diff_lpd` would be a comparison
against a number nobody quoted.
"""
function compute_metric(m::LPD, ctx::EvaluationContext; target::Symbol = :market)::LPDResult
    target in (:market, :score) ||
        error("LPD: target must be :market or :score, got :$target.")
    return target === :score ? _lpd_score(m, ctx) : _lpd_market(m, ctx)
end

function _lpd_market(m::LPD, ctx::EvaluationContext)::LPDResult
    p = ctx.probs
    rows = evaluation_rows(ctx; selections = _selection_filter(m))

    lpd_model = Float64[]
    lpd_market = Float64[]
    sizehint!(lpd_model, length(rows))
    sizehint!(lpd_market, length(rows))

    log_liks = Vector{Float64}(undef, p.n_draws)
    expbuf = Vector{Float64}(undef, p.n_draws)

    @inbounds for r in rows
        v = view(p.draws, :, r.fixture, r.column)
        y = row_y(r)
        push!(lpd_model, calc_lpd_samples!(log_liks, expbuf, v, y))
        push!(lpd_market, calc_lpd_scalar(r.market_prob, y))
    end

    return _pack_lpd(lpd_model, lpd_market)
end

"""
    _lpd_score(metric, ctx) -> LPDResult

    LPD_i = log( (1/S) Σ_s S_i[g_h+1, g_a+1, s] )

read off the score grid, one fixture at a time, with one grid and one workspace for the
whole sweep.

A scoreline BEYOND the grid's truncation cannot be evaluated — the grid stops at
`max_goals` — so such a fixture is skipped and counted out of `n_obs` rather than scored
as `log(0)`. At the repository's `max_goals = 12` that is a 12-goal side.
"""
function _lpd_score(::LPD, ctx::EvaluationContext)::LPDResult
    l = ctx.latents
    mg = ctx.max_goals
    ws = GridWorkspace(mg)
    S = alloc_score_grid(l, mg)
    ids = latent_match_ids(l)
    nd = n_draws(l)

    out = Float64[]
    sizehint!(out, length(ids))
    probs = Vector{Float64}(undef, nd)

    @inbounds for i in eachindex(ids)
        sc = outcome_of(ctx.outcomes, ids[i])
        sc === nothing && continue
        gh, ga = sc
        (0 <= gh < mg && 0 <= ga < mg) || continue
        compute_score_grid!(S, ws, l, i)
        for k in 1:nd
            probs[k] = S[gh + 1, ga + 1, k]
        end
        # Log-mean-exp is unnecessary: these are probabilities, not log-densities, and
        # the mean of `nd` numbers in [0,1] cannot overflow. The clamp is the same 1e-15
        # floor `calc_lpd_samples!` applies, so a fixture the model gave zero mass scores
        # −34.5 rather than −Inf and does not poison the mean.
        push!(out, log(clamp(mean(probs), 1e-15, Inf)))
    end

    return _pack_lpd(out, fill(NaN, length(out)))
end

function _pack_lpd(model::Vector{Float64}, market::Vector{Float64})::LPDResult
    n = length(model)
    n == 0 && return LPDResult(LPDComponent(NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0))
    mm = mean(model)
    mk = isempty(market) || all(isnan, market) ? NaN : mean(market)
    return LPDResult(LPDComponent(
        mm,
        n < 2 ? NaN : std(model),
        n < 3 ? NaN : skewness(model),
        n < 4 ? NaN : kurtosis(model),
        mk,
        mm - mk,
        sum(model),
        n))
end

# --- 3.3 CRPS ------------------------------------------------------------------

"""
    compute_metric(::CRPS, ctx; max_goals = 30) -> CRPSResults

Continuous Ranked Probability Score of each side's goal count against its realised value:

    CRPS = Σ_{x=0}^{max_goals} ( F(x) − 1{x ≥ y} )²

`all` is the per-match AVERAGE of home and away, matching `crps.jl` — so a match
contributes one number rather than two. (`RQRResult.all` POOLS them instead. Two
different conventions in two adjacent legacy files; both preserved.)

The tail beyond 30 goals contributes `(1 − 1)² = 0` for any realised score, so the
truncation is exact for football and the keyword exists only for other sports.
"""
function compute_metric(::CRPS, ctx::EvaluationContext;
                        max_goals::Integer = 30)::CRPSResults
    l = ctx.latents
    ids = latent_match_ids(l)

    crps_home = Float64[]
    crps_away = Float64[]
    sizehint!(crps_home, length(ids))
    sizehint!(crps_away, length(ids))

    @inbounds for i in eachindex(ids)
        sc = outcome_of(ctx.outcomes, ids[i])
        sc === nothing && continue
        gh, ga = sc
        λh, rh, λa, ra = crps_parameters(l, i)
        push!(crps_home, compute_crps(gh, λh, rh; max_goals = max_goals))
        push!(crps_away, compute_crps(ga, λa, ra; max_goals = max_goals))
    end

    isempty(crps_home) && return CRPSResults(CRPSComponent(NaN), CRPSComponent(NaN),
                                             CRPSComponent(NaN))
    crps_all = (crps_home .+ crps_away) ./ 2.0
    return CRPSResults(CRPSComponent(mean(crps_home)),
                       CRPSComponent(mean(crps_away)),
                       CRPSComponent(mean(crps_all)))
end

# --- 3.4 RQR -------------------------------------------------------------------

"""
    compute_metric(::RQR, ctx; n_sims = 1, seed = 42, rng = Xoshiro(seed)) -> RQRResult

Randomized Quantile Residuals — Dunn & Smyth's device for giving a discrete distribution
a continuous residual: `u ~ Uniform(F(y−1), F(y))`, `r = Φ⁻¹(u)`. If the marginal is
correctly specified, `r` is standard normal, and the reported `DistributionStats` test
that.

SEEDED, WITH A PRIVATE STREAM. `rqr.jl` draws from the unseeded global RNG, so two
consecutive calls on identical inputs disagree — the diagnostic cannot be re-checked and
two models' RQR rows are incomparable unless computed in the same session in the same
order. Here two evaluations of the same fit agree, and evaluating never perturbs the
caller's global RNG.

`n_sims` IS RANDOMISATION REPLICATES, AND IT AVERAGES SUMMARIES, NOT RESIDUALS.
Averaging the residuals across replicates would shrink them toward the mid-quantile
normal score and manufacture normality, so each replicate is summarised in full and the
SUMMARIES are averaged. The default of `1` reproduces `rqr.jl` draw for draw given the
same stream, which is what makes the two comparable at all; raise it for a less noisy
Shapiro-Wilk `p`.

THE CALL ORDER IS LOAD-BEARING: within a replicate every home residual is drawn first,
in fixture order, then every away one — the order `rqr.jl`'s two broadcasts produce.
"""
function compute_metric(::RQR, ctx::EvaluationContext;
                        n_sims::Integer = 1, seed::Integer = 42,
                        rng::AbstractRNG = Random.Xoshiro(seed))::RQRResult
    l = ctx.latents
    ids = latent_match_ids(l)

    dists_h = UnivariateDistribution[]
    dists_a = UnivariateDistribution[]
    goals_h = Int[]
    goals_a = Int[]
    for i in eachindex(ids)
        sc = outcome_of(ctx.outcomes, ids[i])
        sc === nothing && continue
        dh, da = marginals(l, i)
        push!(dists_h, dh); push!(dists_a, da)
        push!(goals_h, sc[1]); push!(goals_a, sc[2])
    end

    nobs = length(goals_h)
    nan6 = DistributionStats(NaN, NaN, NaN, NaN, NaN, NaN)
    nobs == 0 && return RQRResult(nan6, nan6, nan6)

    reps = max(Int(n_sims), 1)
    acc_h = zeros(Float64, 6)
    acc_a = zeros(Float64, 6)
    acc_all = zeros(Float64, 6)
    rh = Vector{Float64}(undef, nobs)
    ra = Vector{Float64}(undef, nobs)

    for _ in 1:reps
        @inbounds for i in 1:nobs
            rh[i] = compute_rqr(goals_h[i], dists_h[i], rng)
        end
        @inbounds for i in 1:nobs
            ra[i] = compute_rqr(goals_a[i], dists_a[i], rng)
        end
        _accumulate_stats!(acc_h, summarize_stats(copy(rh)))
        _accumulate_stats!(acc_a, summarize_stats(copy(ra)))
        _accumulate_stats!(acc_all, summarize_stats(vcat(rh, ra)))
    end

    return RQRResult(_mean_stats(acc_h, reps), _mean_stats(acc_a, reps),
                     _mean_stats(acc_all, reps))
end

function _accumulate_stats!(acc::Vector{Float64}, s::DistributionStats)
    acc[1] += s.mean
    acc[2] += s.std
    acc[3] += s.skewness
    acc[4] += s.kurtosis
    acc[5] += s.shapiro_w
    acc[6] += s.shapiro_p
    return acc
end

# `n == 1` divides by one, which is exact, so a single-replicate run returns the summary
# itself rather than a rounded copy of it.
_mean_stats(acc::Vector{Float64}, n::Int) =
    DistributionStats(acc[1] / n, acc[2] / n, acc[3] / n, acc[4] / n, acc[5] / n, acc[6] / n)

# --- 3.5 GLMEdge ---------------------------------------------------------------

"""
    compute_metric(::GLMEdge, ctx; min_edge = 0.0) -> GLMEdgeResult

Logistic regression of the realised outcome on the market's fair probability and the
model's EDGE over it:

    logit P(Y = 1) = β₀ + β₁·p_fair_close + β₂·(p̂_model − p_fair_close)

`β₂` — reported as `spread_fair` — is the question: does the model's disagreement with
the closing line predict which way the result goes?

`min_edge` drops rows with `|p̂ − p_fair| < min_edge`, a way to ask whether the model's
*confident* disagreements are the informative ones. `0.0` keeps every row, which is the
legacy behaviour.

THE `n_obs < 10` GUARD IS CHECKED BEFORE THE ODDS FILTER, as in `glm_edge.jl`. That
ordering means a metric with 12 rows of which 9 have a missing `odds_close` reaches `glm`
with 3 — preserved rather than fixed, because changing it would change published numbers.
"""
function compute_metric(m::GLMEdge, ctx::EvaluationContext;
                        min_edge::Real = 0.0)::GLMEdgeResult
    o = ctx.odds
    p = ctx.probs
    filt = _selection_filter(m)
    edge = Float64(min_edge)

    prob_fair = Float64[]
    spread_fair = Float64[]
    Y = Float64[]
    n_prefilter = 0

    @inbounds for i in 1:o.n
        o.has_fair[i] || continue
        sel = o.selection[i]
        _passes(filt, sel) || continue
        p̂ = prob_mean(p, o.match_id[i], sel)
        p̂ === nothing && continue
        n_prefilter += 1
        # `dropmissing!(analysis_df, [:odds_close, :is_winner])`, glm_edge.jl
        o.has_odds_close[i] || continue
        o.is_winner[i] < 0 && continue
        sf = p̂ - o.prob_fair_close[i]
        abs(sf) < edge && continue
        push!(prob_fair, o.prob_fair_close[i])
        push!(spread_fair, sf)
        push!(Y, Float64(o.is_winner[i]))
    end

    empty_coef = GLMCoefComponent(NaN, NaN, NaN, NaN)
    if n_prefilter < 10
        @warn "GLMEdge: not enough observations ($n_prefilter) for selections " *
              "$(scored_selections(m)). Returning NaNs."
        return GLMEdgeResult(empty_coef, empty_coef, empty_coef, n_prefilter)
    end

    n = length(Y)
    n < 3 && return GLMEdgeResult(empty_coef, empty_coef, empty_coef, n)

    df = DataFrame(Y = Y, prob_fair_close = prob_fair, spread_fair = spread_fair)
    reg = try
        glm(@formula(Y ~ prob_fair_close + spread_fair), df, Binomial(), LogitLink())
    catch e
        @warn "GLMEdge: the logistic fit failed" exception = e
        return GLMEdgeResult(empty_coef, empty_coef, empty_coef, n)
    end

    ct = coeftable(reg)
    function grab(name::String)
        idx = findfirst(==(name), ct.rownms)
        idx === nothing && return empty_coef
        return GLMCoefComponent(ct.cols[1][idx], ct.cols[2][idx],
                                ct.cols[3][idx], ct.cols[4][idx])
    end

    return GLMEdgeResult(grab("(Intercept)"), grab("prob_fair_close"),
                         grab("spread_fair"), n)
end

# --- 3.6 MIQ -------------------------------------------------------------------

"""
    compute_metric(::MIQ, ctx) -> MIQResult

Market-Implied Quantile. For each priced selection, where does the market's fair
probability sit inside the model's POSTERIOR distribution of that probability?

    q_i = (1/S) · #{ s : p̂ᵢ^(s) ≤ p_fair,i }

The diagnostic is the gap between the `q` distribution of the selections that WON and of
those that LOST: if the model has an edge the market underprices winners, so winners
carry a lower `q` and `mean_gap = mean(q | lose) − mean(q | win)` is positive.

Unlike every other rule here, MIQ reads the FULL posterior of the price rather than its
mean — collapsing to a point probability would destroy the quantity being measured.
"""
function compute_metric(::MIQ, ctx::EvaluationContext)::MIQResult
    o = ctx.odds
    p = ctx.probs

    q_all = Float64[]
    won_all = Bool[]
    by_sel = Dict{Symbol, Tuple{Vector{Float64}, Vector{Bool}}}()
    for s in p.selections
        by_sel[s] = (Float64[], Bool[])
    end

    @inbounds for i in 1:o.n
        sel = o.selection[i]
        r, c = _eval_locate(p, o.match_id[i], sel)
        (r == 0 || c == 0) && continue
        # `get_miq` returns `missing` for an absent market probability; `dropmissing` on
        # :market_quantile (miq.jl) removes exactly those rows.
        o.has_fair[i] || continue
        v = view(p.draws, :, r, c)
        q = get_miq(v, o.prob_fair_close[i])
        q === missing && continue
        # `is_winner` is compared with `.== true` / `.== false` (miq.jl), so a missing
        # outcome falls into NEITHER group. Reproduced by skipping it.
        w = o.is_winner[i]
        w < 0 && continue
        won = w == 1
        push!(q_all, q); push!(won_all, won)
        bucket = by_sel[sel]
        push!(bucket[1], q); push!(bucket[2], won)
    end

    empty_stats = MIQStats(missing, missing, missing, missing, missing, 0, 0)
    stats_for(s::Symbol) = haskey(by_sel, s) ?
        evaluate_group_edge(by_sel[s][1], by_sel[s][2]) : empty_stats

    return MIQResult(
        evaluate_group_edge(q_all, won_all),
        (stats_for(s) for s in MIQ_FIELD_SELECTIONS)...)
end

"The selections `MIQResult` reports, in field order. `:all` is the pool, not a selection."
const MIQ_FIELD_SELECTIONS =
    (:home, :draw, :away, :over_15, :under_15, :over_25, :under_25,
     :over_35, :under_35, :btts_yes, :btts_no)


# ==============================================================================
# 4. CALIBRATION METRICS
# ==============================================================================
#
# The three legacy market metrics (LogLoss, LPD, GLMEdge) all ask "is the model better
# than the closing line". None of them asks "does the model mean what it says" — whether
# the fixtures it priced at 0.30 came in 30% of the time. That is CALIBRATION, and it is
# a different failure: a model can beat the market on log-loss while being systematically
# overconfident, and a staking layer reads the probability, not the ranking.
#
# Brier is the quadratic proper score, ECE/MCE are the binned reliability gaps, and RPS
# is the ORDERED multi-class score for 1X2 — the one that knows a home win predicted as
# a draw is a smaller error than a home win predicted as an away win, which no sum of
# binary scores does.

"""
    brier_score(rows) -> (score, n_obs)
    brier_score(p, y)  -> Float64

Mean squared error of a probability against a binary outcome, `mean((p − y)²)`.

The quadratic proper scoring rule. Bounded in `[0, 1]`, unlike log-loss, so one
catastrophically confident wrong call cannot dominate a fold — which is why it is worth
reporting alongside log-loss rather than instead of it.
"""
@inline brier_score(p::Float64, y::Float64) = (p - y)^2

function brier_score(rows::AbstractVector{EvaluationRow}; source::Symbol = :model)
    n = 0
    acc = 0.0
    @inbounds for r in rows
        p = source === :market ? r.market_prob : r.model_prob
        isfinite(p) || continue
        n += 1
        acc += brier_score(p, row_y(r))
    end
    return (n == 0 ? NaN : acc / n, n)
end

"""
    CalibrationCurve(edges, counts, mean_predicted, observed)

A reliability diagram as a table: for each probability bin, how many rows fell in it,
what the model said on average, and how often the selection actually won.

A perfectly calibrated model has `observed ≈ mean_predicted` in every bin. `edges` has
`length(counts) + 1` entries; empty bins carry `NaN` rather than `0.0`, because a bin
with no observations has no observed frequency and reporting one as zero would drag any
summary of the curve toward the origin.
"""
struct CalibrationCurve
    edges::Vector{Float64}
    counts::Vector{Int}
    mean_predicted::Vector{Float64}
    observed::Vector{Float64}
end

Base.length(c::CalibrationCurve) = length(c.counts)

function Base.show(io::IO, c::CalibrationCurve)
    print(io, "CalibrationCurve(", length(c), " bins, ", sum(c.counts), " observations)")
end

function Base.show(io::IO, ::MIME"text/plain", c::CalibrationCurve)
    println(io, "CalibrationCurve — ", sum(c.counts), " observations in ", length(c), " bins")
    println(io, "    bin              n   predicted   observed        gap")
    println(io, "  ", "-"^52)
    for b in 1:length(c)
        if c.counts[b] == 0
            @printf(io, "  [%.2f, %.2f)  %6d           —          —          —\n",
                    c.edges[b], c.edges[b + 1], 0)
            continue
        end
        @printf(io, "  [%.2f, %.2f)  %6d  %10.4f %10.4f %10.4f\n",
                c.edges[b], c.edges[b + 1], c.counts[b],
                c.mean_predicted[b], c.observed[b],
                c.observed[b] - c.mean_predicted[b])
    end
end

"""
    calibration_curve(rows; n_bins = 10, source = :model) -> CalibrationCurve

Bin the rows by predicted probability and report the observed win rate in each.

EQUAL-WIDTH bins over `[0, 1]`, not equal-count. Equal-count quantile bins make the
diagram look better on a model whose probabilities cluster — the bins narrow exactly
where the mass is — and the question here is about the probability SCALE, which is the
axis equal-width bins preserve. `p == 1.0` goes into the last bin rather than off the
end.
"""
function calibration_curve(rows::AbstractVector{EvaluationRow};
                           n_bins::Integer = 10, source::Symbol = :model)
    nb = max(Int(n_bins), 1)
    edges = collect(range(0.0, 1.0; length = nb + 1))
    counts = zeros(Int, nb)
    sum_pred = zeros(Float64, nb)
    sum_obs = zeros(Float64, nb)

    @inbounds for r in rows
        p = source === :market ? r.market_prob : r.model_prob
        isfinite(p) || continue
        b = clamp(floor(Int, p * nb) + 1, 1, nb)
        counts[b] += 1
        sum_pred[b] += p
        sum_obs[b] += row_y(r)
    end

    pred = Vector{Float64}(undef, nb)
    obs = Vector{Float64}(undef, nb)
    @inbounds for b in 1:nb
        if counts[b] == 0
            pred[b] = NaN
            obs[b] = NaN
        else
            pred[b] = sum_pred[b] / counts[b]
            obs[b] = sum_obs[b] / counts[b]
        end
    end
    return CalibrationCurve(edges, counts, pred, obs)
end

calibration_curve(ctx::EvaluationContext; selections = nothing, kwargs...) =
    calibration_curve(evaluation_rows(ctx; selections = selections); kwargs...)

"""
    expected_calibration_error(curve) -> Float64
    expected_calibration_error(rows; n_bins = 10, source = :model) -> Float64

`ECE = Σ_b (n_b / N) · |observed_b − predicted_b|` — the count-weighted mean reliability
gap. `0` is perfect; `NaN` when nothing was scored.
"""
function expected_calibration_error(c::CalibrationCurve)
    total = sum(c.counts)
    total == 0 && return NaN
    acc = 0.0
    @inbounds for b in 1:length(c)
        c.counts[b] == 0 && continue
        acc += c.counts[b] * abs(c.observed[b] - c.mean_predicted[b])
    end
    return acc / total
end

expected_calibration_error(rows::AbstractVector{EvaluationRow}; kwargs...) =
    expected_calibration_error(calibration_curve(rows; kwargs...))

"""
    max_calibration_error(curve) -> Float64

The LARGEST reliability gap over non-empty bins. ECE can be small while one bin is badly
wrong; MCE is the number that says so.
"""
function max_calibration_error(c::CalibrationCurve)
    worst = NaN
    @inbounds for b in 1:length(c)
        c.counts[b] == 0 && continue
        g = abs(c.observed[b] - c.mean_predicted[b])
        (isnan(worst) || g > worst) && (worst = g)
    end
    return worst
end

max_calibration_error(rows::AbstractVector{EvaluationRow}; kwargs...) =
    max_calibration_error(calibration_curve(rows; kwargs...))

"""
    ranked_probability_score(ctx; source = :model) -> (rps, n_obs)

Epstein's Ranked Probability Score over the ORDERED 1X2 outcomes:

    RPS = (1 / (K − 1)) Σ_{k=1}^{K−1} ( Σ_{j≤k} p_j − Σ_{j≤k} y_j )²,   K = 3

with the categories ordered home, draw, away — the football convention, in which a home
win forecast as a draw is a smaller error than a home win forecast as an away win. No
sum of independent binary scores expresses that, which is the whole reason RPS is
reported next to Brier rather than in place of it.

`0` is perfect and `1` is maximally wrong. Requires `Market1X2()` to have been priced
and the fixture's result to be recorded; returns `(NaN, 0)` when neither holds. With
`source = :market` the same score is computed from `prob_fair_close`, giving the closing
line's RPS as a baseline.
"""
function ranked_probability_score(ctx::EvaluationContext; source::Symbol = :model)
    o = ctx.odds
    p = ctx.probs
    trio = Dict{Int, Vector{Float64}}()

    @inbounds for i in 1:o.n
        sel = o.selection[i]
        k = sel === :home ? 1 : sel === :draw ? 2 : sel === :away ? 3 : 0
        k == 0 && continue
        v = if source === :market
            o.has_fair[i] ? o.prob_fair_close[i] : NaN
        else
            pm = prob_mean(p, o.match_id[i], sel)
            pm === nothing ? NaN : pm
        end
        isfinite(v) || continue
        slot = get!(trio, o.match_id[i], fill(NaN, 3))
        slot[k] = v
    end

    n = 0
    acc = 0.0
    for (mid, probs) in trio
        all(isfinite, probs) || continue
        sc = outcome_of(ctx.outcomes, mid)
        sc === nothing && continue
        gh, ga = sc
        obs = gh > ga ? 1 : gh == ga ? 2 : 3
        cum_p = 0.0
        cum_o = 0.0
        s = 0.0
        @inbounds for k in 1:2
            cum_p += probs[k]
            cum_o += (obs == k ? 1.0 : 0.0)
            d = cum_p - cum_o
            s += d * d
        end
        acc += s / 2.0
        n += 1
    end
    return (n == 0 ? NaN : acc / n, n)
end

"""
    CalibrationComponent(logloss, brier, ece, mce, rps, n_obs)

One side's calibration summary. `rps` is over 1X2 only and is `NaN` when that market was
not priced or no result was recorded.
"""
struct CalibrationComponent <: AbstractMetricComponent
    logloss::Float64
    brier::Float64
    ece::Float64
    mce::Float64
    rps::Float64
    n_obs::Int
end

"""
    PredictionScores(model, market)

What [`evaluate_predictions`](@ref) returns: the model's calibration summary and the
closing line's, side by side.

The market column is not decoration. Every one of these numbers is scale-dependent — a
Brier of 0.21 is excellent on 1X2 and poor on a 0.9-probability favourite market — so
the only reading that transfers between folds is the model's number NEXT TO the price
the market was making at the same moment.
"""
struct PredictionScores <: AbstractEvaluationResult
    model::CalibrationComponent
    market::CalibrationComponent
end

get_metric_method_name(::PredictionScores) = "predictions"

"""
    PredictionScore(; selections = Symbol[], n_bins = 10)
    PredictionScore(selection::Symbol)
    PredictionScore(selections::Vector{Symbol})

The trigger for [`evaluate_predictions`](@ref), so calibration can join a batch alongside
the six legacy rules.

Named `PredictionScore` rather than `Calibration` because `BayesianFootball.Calibration`
is the Layer-2 shift-model MODULE, and a type exported under that name would shadow it.
"""
struct PredictionScore <: AbstractScoringRule
    selections::Vector{Symbol}
    n_bins::Int
end

PredictionScore(; selections = Symbol[], n_bins::Integer = 10) =
    PredictionScore(Symbol[Symbol(s) for s in selections], Int(n_bins))
PredictionScore(selection::Symbol) = PredictionScore(; selections = [selection])
PredictionScore(selections::AbstractVector{Symbol}) =
    PredictionScore(; selections = selections)

get_metric_method_name(m::PredictionScore) =
    isempty(m.selections) ? "predictions_all" :
    "predictions_" * join(String.(m.selections), "_")

scored_markets(m::PredictionScore) = _scope_markets(scored_selections(m))
needs_outcomes(::PredictionScore) = true
needs_draws(::PredictionScore) = false

"""
    evaluate_predictions(ctx; selections = nothing, n_bins = 10) -> PredictionScores

Score the model's probabilities as PROBABILITIES: log-loss, Brier, expected and maximum
calibration error, and the ordered 1X2 ranked probability score — each alongside the
same quantity computed from the closing line.

This is the entry point that does not ask "did the model beat the market" but "does the
model mean what it says". Both matter, and they fail independently: a model can win on
log-loss while being systematically overconfident, and the staking layer reads the
probability rather than the ranking.
"""
function evaluate_predictions(ctx::EvaluationContext;
                              selections = nothing, n_bins::Integer = 10)
    rows = evaluation_rows(ctx; selections = selections)
    return PredictionScores(_calibration_side(ctx, rows, :model, n_bins),
                            _calibration_side(ctx, rows, :market, n_bins))
end

function evaluate_predictions(fit::Fit, ds::DataStore;
                              selections = nothing, n_bins::Integer = 10,
                              max_goals::Integer = Predictions.TPL_MAX_GOALS,
                              threaded::Bool = true)
    lat = fit_latents(fit)
    ctx = build_evaluation_context(lat, ds.odds, ds.matches, [PredictionScore()];
                                   max_goals = max_goals, threaded = threaded)
    return evaluate_predictions(ctx; selections = selections, n_bins = n_bins)
end

function _calibration_side(ctx::EvaluationContext, rows::Vector{EvaluationRow},
                           source::Symbol, n_bins::Integer)
    n = 0
    ll = 0.0
    @inbounds for r in rows
        p = source === :market ? r.market_prob : r.model_prob
        isfinite(p) || continue
        n += 1
        ll += calc_logloss(p, row_y(r))
    end
    brier, _ = brier_score(rows; source = source)
    curve = calibration_curve(rows; n_bins = n_bins, source = source)
    rps, _ = ranked_probability_score(ctx; source = source)
    return CalibrationComponent(n == 0 ? NaN : ll / n, brier,
                                expected_calibration_error(curve),
                                max_calibration_error(curve), rps, n)
end

compute_metric(m::PredictionScore, ctx::EvaluationContext) =
    evaluate_predictions(ctx; selections = _selection_filter(m), n_bins = m.n_bins)
