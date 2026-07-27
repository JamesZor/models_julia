# current_development/bayesian_layer_2/src/l01_bayes_calib.jl
#
# Bayesian Layer-2 calibration — prototype loader.
# Companion to docs/archive/l2_bayesian_calibration_research.md.
#
# Core object: a *Bayesian* logit-shift calibrator fit by Laplace approximation.
#   logit(p_cal) = logit(p_L1)   +   Δ_i,        Δ_i = xᵢᵀ β,   β ~ N(β̂, Σ)
#                  ^^^^^^^^^^^^^      ^^^^^^^^
#                  GLM offset        learned shift (carries its OWN variance)
#
# Because Δ has a posterior variance, the calibrated posterior is the CONVOLUTION
# of the L1 posterior with the shift law (research note §1) — it WIDENS the L1
# posterior, which is the fix for the PIT over-confidence the Betfair study found.
# The current production L2 (BasicLogitShift) is the σ_Δ → 0 limit of this model.
#
# Three nested stages, one fitter:
#   Stage 2  global shift     X = [1]                      (offset form, β-slope ≡ 1)
#   Stage 3  + time decay     w_i = exp(-c·ΔT_i)           (power likelihood)
#   Stage 4  + team effects   X = [1, team dummies], ridge prior 1/τ² = partial pooling
#
# Pure IRLS / Newton — NOT a Turing model, so no AD-safety constraints apply.

using LinearAlgebra
using Statistics
using Distributions
using Dates
using DataFrames
using StatsFuns: logit, logistic
using Random

const EPS = 1e-6
const HALFLIFE_DEFAULT = 60.0                 # days — matches DCMH_HalfLife_60
decay_rate(halflife_days::Real) = log(2) / halflife_days

# ----------------------------------------------------------------------------
# 1. Core: weighted, ridge-penalised logistic regression with offset (Laplace)
# ----------------------------------------------------------------------------
# Maximises   Σ_i w_i [y_i η_i − log(1+e^{η_i})]  −  ½ βᵀ P β,   η_i = offset_i + xᵢᵀβ
# Returns the MAP β̂ and the Laplace posterior covariance Σ = (XᵀW̃X + P)⁻¹.
# The Gaussian prior precision P encodes both the global-shift prior (1/s_α²) and,
# for team columns, the partial-pooling precision (1/τ²).

struct LaplaceFit
    β::Vector{Float64}        # MAP coefficients
    Σ::Matrix{Float64}        # posterior covariance (Laplace)
    loglik::Float64           # weighted Bernoulli log-likelihood at β̂ (no prior)
    evidence::Float64         # Laplace log marginal likelihood  log p(y | hyperparams)
end

function laplace_logistic_ridge(X::Matrix{Float64}, y::Vector{Float64},
                                offset::Vector{Float64}, w::Vector{Float64},
                                P::Matrix{Float64}; maxiter::Int = 200, tol::Float64 = 1e-11)
    n, d = size(X)
    β = zeros(d)
    for _ in 1:maxiter
        η  = clamp.(offset .+ X * β, -30.0, 30.0)
        p  = logistic.(η)
        W̃  = w .* p .* (1 .- p)
        g  = X' * (w .* (y .- p)) .- P * β          # penalised gradient
        H  = Symmetric(X' * (W̃ .* X) .+ P)          # penalised observed information
        Δβ = H \ g
        β .+= Δβ
        maximum(abs, Δβ) < tol && break
    end
    # Recompute curvature at the mode for the covariance / evidence
    η  = clamp.(offset .+ X * β, -30.0, 30.0)
    p  = logistic.(η)
    W̃  = w .* p .* (1 .- p)
    H  = Symmetric(X' * (W̃ .* X) .+ P)
    Σ  = inv(H)
    loglik = sum(w .* (y .* log.(clamp.(p, EPS, 1 - EPS)) .+
                       (1 .- y) .* log.(clamp.(1 .- p, EPS, 1 - EPS))))
    # Laplace evidence: ℓ + log N(β̂;0,P⁻¹) + ½ log det(2πΣ)
    logprior = -0.5 * dot(β, P * β) + 0.5 * logdet(P) - 0.5 * d * log(2π)
    laplace  =  0.5 * (d * log(2π) + logdet(Matrix(Σ)))
    return LaplaceFit(β, Matrix(Σ), loglik, loglik + logprior + laplace)
end

# ----------------------------------------------------------------------------
# 2. Fitted calibrator (carries everything apply/predict needs)
# ----------------------------------------------------------------------------
struct BayesShiftCalibrator
    kind::Symbol                     # :global or :team
    fit::LaplaceFit
    team_index::Dict{String,Int}     # team => column index in β   (empty for :global)
    s_α::Float64                     # global-shift prior sd
    τ::Float64                       # team-effect prior sd (partial pooling)
    c::Float64                       # decay rate (1/days);  0 ⇒ no decay
    ref_date::Date                   # T:  weights = exp(-c (T - tᵢ))
    ess::Float64                     # global effective sample size
    prob_col::Symbol
end

# weights helper ------------------------------------------------------------
function _weights(dates::AbstractVector{Date}, ref_date::Date, c::Float64)
    c <= 0 && return ones(length(dates))
    Δdays = Float64.(Dates.value.(ref_date .- dates))
    return exp.(-c .* Δdays)
end
ess(w::AbstractVector) = sum(w)^2 / sum(abs2, w)

# offset / outcome extraction ----------------------------------------------
function _offset_y(df::AbstractDataFrame, prob_col::Symbol)
    offset = logit.(clamp.(Float64.(df[!, prob_col]), EPS, 1 - EPS))
    y      = Float64.(df.is_winner)
    return offset, y
end

# ----------------------------------------------------------------------------
# 3a. Stage 2/3 — global shift (optionally time-decayed)
# ----------------------------------------------------------------------------
function fit_global_shift(df::AbstractDataFrame; prob_col::Symbol = :prob_mean,
                          s_α::Float64 = 1.0, halflife::Float64 = HALFLIFE_DEFAULT,
                          decay::Bool = false, ref_date::Date = maximum(df.match_date))
    offset, y = _offset_y(df, prob_col)
    c = decay ? decay_rate(halflife) : 0.0
    w = _weights(df.match_date, ref_date, c)
    X = ones(length(y), 1)
    P = reshape([1 / s_α^2], 1, 1)
    fit = laplace_logistic_ridge(X, y, offset, w, P)
    return BayesShiftCalibrator(:global, fit, Dict{String,Int}(), s_α, NaN, c,
                                ref_date, ess(w), prob_col)
end

# ----------------------------------------------------------------------------
# 3b. Stage 4 — global + partially-pooled team residual-bias effects
# ----------------------------------------------------------------------------
# Δ_i = α + u_home(i) + u_away(i),   u_team ~ N(0, τ²)  (ridge = partial pooling).
# The team column for team t is the indicator (t plays in match i), exactly
# reproducing the existing TeamBiasLogitShift design — but now SHRUNK and with
# a posterior covariance.
function _team_design(df::AbstractDataFrame, prob_col::Symbol)
    teams = sort(unique(vcat(String.(df.home_team), String.(df.away_team))))
    tindex = Dict(t => i + 1 for (i, t) in enumerate(teams))   # +1: col 1 = intercept
    n = nrow(df); d = length(teams) + 1
    X = zeros(n, d)
    X[:, 1] .= 1.0
    for (r, row) in enumerate(eachrow(df))
        X[r, tindex[String(row.home_team)]] += 1.0
        X[r, tindex[String(row.away_team)]] += 1.0
    end
    return X, tindex, teams
end

function fit_team_shift(df::AbstractDataFrame; prob_col::Symbol = :prob_mean,
                        s_α::Float64 = 1.0, τ::Float64 = 0.3,
                        halflife::Float64 = HALFLIFE_DEFAULT, decay::Bool = false,
                        ref_date::Date = maximum(df.match_date))
    offset, y = _offset_y(df, prob_col)
    c = decay ? decay_rate(halflife) : 0.0
    w = _weights(df.match_date, ref_date, c)
    X, tindex, _ = _team_design(df, prob_col)
    P = diagm([1 / s_α^2; fill(1 / τ^2, size(X, 2) - 1)])
    fit = laplace_logistic_ridge(X, y, offset, w, P)
    return BayesShiftCalibrator(:team, fit, tindex, s_α, τ, c, ref_date, ess(w), prob_col)
end

# Empirical-Bayes choice of τ: maximise the Laplace evidence over a grid -----
function fit_team_shift_eb(df::AbstractDataFrame; τ_grid = 0.05:0.05:0.8, kwargs...)
    best = nothing; best_ev = -Inf; best_τ = NaN
    for τ in τ_grid
        cal = fit_team_shift(df; τ = float(τ), kwargs...)
        if cal.fit.evidence > best_ev
            best_ev = cal.fit.evidence; best = cal; best_τ = τ
        end
    end
    return best, best_τ, best_ev
end

# team-effect table (û_j with posterior sd) ---------------------------------
function team_effects(cal::BayesShiftCalibrator)
    @assert cal.kind == :team
    rows = NamedTuple[]
    for (t, j) in sort(collect(cal.team_index), by = last)
        push!(rows, (team = t, u = cal.fit.β[j], sd = sqrt(cal.fit.Σ[j, j])))
    end
    df = DataFrame(rows)
    df.z = df.u ./ df.sd                       # |z|≳2 ⇒ L1 has real residual bias for this team
    sort!(df, :u)
    return df
end

# ----------------------------------------------------------------------------
# 4. Predictive shift law per match:  Δ_i ~ N(μ_i, σ_i²)
# ----------------------------------------------------------------------------
# μ_i = xᵢᵀβ̂,  σ_i² = xᵢᵀ Σ xᵢ.  Unseen team ⇒ contributes 0 to μ and +τ² to σ²
# (a draw from the team prior — the Bayesian way to handle a promoted side).
function predict_shift(cal::BayesShiftCalibrator, df::AbstractDataFrame)
    n = nrow(df); μ = zeros(n); σ = zeros(n)
    if cal.kind == :global
        μ .= cal.fit.β[1]
        σ .= sqrt(cal.fit.Σ[1, 1])
        return μ, σ
    end
    d = length(cal.fit.β)
    for (r, row) in enumerate(eachrow(df))
        x = zeros(d); x[1] = 1.0; extra = 0.0
        for tm in (String(row.home_team), String(row.away_team))
            j = get(cal.team_index, tm, 0)
            j == 0 ? (extra += cal.τ^2) : (x[j] += 1.0)
        end
        μ[r] = dot(x, cal.fit.β)
        σ[r] = sqrt(max(0.0, dot(x, cal.fit.Σ * x) + extra))
    end
    return μ, σ
end

# calibrated *point* probability (posterior-mean shift) — for log-score eval
calibrated_point(cal, df) = begin
    offset, _ = _offset_y(df, cal.prob_col)
    μ, _ = predict_shift(cal, df)
    logistic.(offset .+ μ)
end

# ----------------------------------------------------------------------------
# 5. Posterior collapse  (the §1 convolution, three equivalent samplers)
# ----------------------------------------------------------------------------
# Given the L1 draws (probabilities) for one match and the shift law N(μ,σ²):

# (a) paired-N — one shift draw per L1 draw. The recommended production form:
#     keeps N draws, drop-in replacement for the current shifted_dists.
function collapse_paired(l1draws::AbstractVector{<:Real}, μ::Float64, σ::Float64;
                         rng = Random.default_rng())
    z = logit.(clamp.(l1draws, EPS, 1 - EPS))
    return logistic.(z .+ rand(rng, Normal(μ, σ), length(z)))
end

# (b) full grid — every L1 draw × M shift draws, flattened (the "N×M" the user
#     worried was 2-D; it is just a size-NM sample of the 1-D convolution).
function collapse_grid(l1draws::AbstractVector{<:Real}, μ::Float64, σ::Float64;
                       M::Int = 200, rng = Random.default_rng())
    z = logit.(clamp.(l1draws, EPS, 1 - EPS))
    δ = rand(rng, Normal(μ, σ), M)
    out = Vector{Float64}(undef, length(z) * M)
    k = 1
    @inbounds for d in δ, zi in z
        out[k] = logistic(zi + d); k += 1
    end
    return out
end

# (c) analytic moments on the logit scale (research note §1.4):
#     E[z_cal] = E[z_L1] + μ ,  Var[z_cal] = Var[z_L1] + σ²   (independence).
function analytic_logit_moments(l1draws::AbstractVector{<:Real}, μ::Float64, σ::Float64)
    z = logit.(clamp.(l1draws, EPS, 1 - EPS))
    return mean(z) + μ, var(z) + σ^2
end

# Numerical verification that (a)≈(b)≈(c): compares logit-scale mean/var of the
# three constructions and a 2-sample KS distance between grid and paired draws.
function verify_convolution(l1draws::AbstractVector{<:Real}, μ::Float64, σ::Float64;
                            M::Int = 200, seed::Int = 1)
    rng = MersenneTwister(seed)
    paired = collapse_paired(l1draws, μ, σ; rng = rng)
    grid   = collapse_grid(l1draws, μ, σ; M = M, rng = rng)
    am, av = analytic_logit_moments(l1draws, μ, σ)
    lp = logit.(clamp.(paired, EPS, 1 - EPS)); lg = logit.(clamp.(grid, EPS, 1 - EPS))
    ks = _ks_distance(grid, paired)
    return (analytic_mean = am, analytic_var = av,
            paired_mean = mean(lp), paired_var = var(lp),
            grid_mean = mean(lg),  grid_var = var(lg),
            ks_grid_vs_paired = ks,
            l1_mean = mean(logit.(clamp.(l1draws, EPS, 1 - EPS))),
            l1_var  = var(logit.(clamp.(l1draws, EPS, 1 - EPS))))
end

function _ks_distance(a::AbstractVector, b::AbstractVector)
    xs = sort(vcat(a, b)); na = length(a); nb = length(b)
    sa = sort(a); sb = sort(b); D = 0.0
    for x in xs
        Fa = searchsortedlast(sa, x) / na
        Fb = searchsortedlast(sb, x) / nb
        D = max(D, abs(Fa - Fb))
    end
    return D
end

# Apply to a whole market DataFrame → new distribution vectors (paired form),
# matching the existing apply_calibration contract (shifted_scalars, shifted_dists).
function apply_bayes_shift(cal::BayesShiftCalibrator, df::AbstractDataFrame;
                           seed::Int = 1)
    μ, σ = predict_shift(cal, df)
    offset, _ = _offset_y(df, cal.prob_col)
    shifted_scalars = logistic.(offset .+ μ)
    rng = MersenneTwister(seed)
    shifted_dists = map(enumerate(df.distribution)) do (i, dist)
        collapse_paired(dist, μ[i], σ[i]; rng = rng)
    end
    return shifted_scalars, shifted_dists
end

# ----------------------------------------------------------------------------
# 6. Evaluation — OOS log-score & PIT, strict walk-forward over splits
# ----------------------------------------------------------------------------
logscore(p::Real, y::Real) = y * log(clamp(p, EPS, 1 - EPS)) + (1 - y) * log(clamp(1 - p, EPS, 1 - EPS))

# Walk-forward: for each split k ≥ start_k, fit on splits < k, score split k.
# `fitter(train_df, ref_date) -> BayesShiftCalibrator`.  Returns mean OOS log-score
# for the calibrated point prediction, plus the raw-L1 baseline on the same rows.
function walk_forward_logscore(df::AbstractDataFrame, fitter;
                               start_k::Int = 4, prob_col::Symbol = :prob_mean,
                               min_train::Int = 10)
    splits = sort(unique(df.split_id))
    cal_ls = Float64[]; raw_ls = Float64[]; n_rows = 0
    for k in (start_k + 1):length(splits)
        train = filter(:split_id => in(Set(splits[1:k-1])), df)
        test  = filter(:split_id => ==(splits[k]), df)
        (nrow(train) < min_train || nrow(test) == 0) && continue
        ref = minimum(test.match_date)                      # T = start of the held-out fold
        cal = fitter(train, ref)
        p_cal = calibrated_point(cal, test)
        p_raw = Float64.(test[!, prob_col])
        y = Float64.(test.is_winner)
        append!(cal_ls, logscore.(p_cal, y)); append!(raw_ls, logscore.(p_raw, y))
        n_rows += nrow(test)
    end
    return (n = n_rows, raw = mean(raw_ls), cal = mean(cal_ls),
            improvement = mean(cal_ls) - mean(raw_ls))     # >0 ⇒ calibration helps
end

# PIT of the closing/market prob (or outcome) against the calibrated posterior.
# Here: PIT = P(posterior ≤ realised market prob).  Reuses the betfair-study idea.
function pit_values(dists::AbstractVector, ref_probs::AbstractVector)
    [mean(d .<= r) for (d, r) in zip(dists, ref_probs)]
end
function interval_coverage(dists::AbstractVector, y::AbstractVector; levels = (0.5, 0.8, 0.95))
    cov = Dict{Float64,Float64}()
    for lv in levels
        lo = (1 - lv) / 2; hi = 1 - lo; hit = 0
        for (d, yi) in zip(dists, y)
            ql, qh = quantile(d, lo), quantile(d, hi)
            # outcome-consistency: does the realised win-rate fall in the predicted band?
            # use the posterior-mean as the point and check the realised y vs band via PIT-like rule
            hit += (ql <= yi <= qh) ? 1 : 0
        end
        cov[lv] = hit / length(y)
    end
    return cov
end
