# ==============================================================================
# l61 — LOADER: the held-out gauntlet
# ==============================================================================
#
# Definitions only, no execution. Paired with r62.
#
# WHY A POISSON RIDGE AND NOT A CORRELATION. Linear association with the scoreline is a
# weak proxy for what the count model does. The engine already carries team strength in
# dyn.alpha/dyn.beta, so a covariate's real job is to explain what those CANNOT — and a
# raw correlation cannot see that distinction. r93 flagged this limitation; l61 removes
# it by fitting the engine's own linear predictor, without MCMC:
#
#     log lambda_h = mu + gamma + alpha_h + beta_a + SUM_k q_k^h
#     log lambda_a = mu         + alpha_a + beta_h + SUM_k q_k^a
#
# fitted by iteratively reweighted least squares with a ridge penalty on the team
# effects, exactly as the engine shrinks them with a prior. Every match contributes TWO
# rows — one per side — which is what makes a supremacy covariate (+x, -x) and a level
# covariate (+x, +x) fall out of one design.
#
# The baseline model is therefore not "nothing": it is a full team-strength model. A
# candidate that beats it has explained something team strength does not, which is the
# only claim worth making.
#
# METRICS. Out-of-sample multiclass log loss on 1X2 is the primary; Brier, RPS and AUC
# accompany it. Every comparison is reported as a PAIRED BOOTSTRAP over held-out matches
# because point differences of a few thousandths of a nat are routine and meaningless.
# ==============================================================================

using DataFrames
using Dates
using LinearAlgebra
using Printf
using Random
using Statistics

const L61_MAX_GOALS = 12

# ==============================================================================
# 1. DESIGN
# ==============================================================================

"""
    l61_design(frame, covariates, roles; team_index) -> (X, y, n_team_cols)

Two rows per match. Columns are `[intercept, home, attack(1..T), defence(1..T),
covariates...]`; a supremacy covariate enters `(+x, -x)` on the two rows and a level
covariate `(+x, +x)`.
"""
function l61_design(frame::DataFrame, covariates::Vector{Vector{Float64}},
                    roles::Vector{Symbol}, team_index::Dict{String,Int})
    n = nrow(frame)
    T = length(team_index)
    p = 2 + 2T + length(covariates)
    X = zeros(2n, p)
    y = zeros(Int, 2n)
    for i in 1:n
        h = team_index[frame.home_team[i]]
        a = team_index[frame.away_team[i]]
        rh = 2i - 1; ra = 2i
        X[rh, 1] = 1.0; X[ra, 1] = 1.0
        X[rh, 2] = 1.0                      # home advantage on the home row only
        X[rh, 2 + h] = 1.0;      X[rh, 2 + T + a] = 1.0
        X[ra, 2 + a] = 1.0;      X[ra, 2 + T + h] = 1.0
        for (k, col) in enumerate(covariates)
            c = 2 + 2T + k
            X[rh, c] = col[i]
            X[ra, c] = roles[k] === :level ? col[i] : -col[i]
        end
        y[rh] = round(Int, frame.goals_home[i])
        y[ra] = round(Int, frame.goals_away[i])
    end
    return X, y, T
end

# ==============================================================================
# 2. POISSON RIDGE BY IRLS
# ==============================================================================

"""
    l61_poisson_ridge(X, y, penalty; lambda, iterations) -> beta

`penalty` is a 0/1 mask marking the coefficients the ridge may shrink. The intercept,
home advantage and the covariates are left unpenalised: shrinking a covariate would make
the comparison between candidates depend on their arbitrary scales.
"""
function l61_poisson_ridge(X::Matrix{Float64}, y::Vector{Int}, penalty::Vector{Float64};
                           lambda::Float64 = 4.0, iterations::Int = 40, tol::Float64 = 1e-9)
    n, p = size(X)
    beta = zeros(p)
    beta[1] = log(max(mean(y), 0.05))
    P = Diagonal(lambda .* penalty)
    for _ in 1:iterations
        eta = clamp.(X * beta, -8.0, 8.0)
        mu = exp.(eta)
        z = eta .+ (y .- mu) ./ max.(mu, 1e-9)
        XtW = X' * Diagonal(mu)
        A = Symmetric(XtW * X + P)
        b = XtW * z
        new_beta = try
            cholesky(A + 1e-10I) \ b
        catch
            pinv(Matrix(A)) * b
        end
        delta = maximum(abs, new_beta .- beta)
        beta = new_beta
        delta < tol && break
    end
    return beta
end

"Penalty mask: shrink the 2T team effects, leave everything else free."
l61_penalty_mask(p::Int, T::Int) = vcat(0.0, 0.0, ones(2T), zeros(p - 2 - 2T))

# ==============================================================================
# 3. SCORING
# ==============================================================================

"1X2 probabilities from a pair of independent Poisson rates."
function l61_outcome_probs(lambda_h::Float64, lambda_a::Float64)
    ph = zeros(L61_MAX_GOALS + 1); pa = zeros(L61_MAX_GOALS + 1)
    lh = clamp(lambda_h, 1e-6, 15.0); la = clamp(lambda_a, 1e-6, 15.0)
    ph[1] = exp(-lh); pa[1] = exp(-la)
    for g in 1:L61_MAX_GOALS
        ph[g + 1] = ph[g] * lh / g
        pa[g + 1] = pa[g] * la / g
    end
    ph ./= sum(ph); pa ./= sum(pa)
    home = 0.0; draw = 0.0; away = 0.0
    for i in 0:L61_MAX_GOALS, j in 0:L61_MAX_GOALS
        q = ph[i + 1] * pa[j + 1]
        i > j ? (home += q) : i == j ? (draw += q) : (away += q)
    end
    total = home + draw + away
    return (home / total, draw / total, away / total)
end

"""
    l61_metrics(probs, frame) -> NamedTuple

Multiclass log loss (the primary), Brier, RPS, accuracy, and the AUC of the home-win
probability. RPS is included because 1X2 is ordered — a model that puts its mass on the
away win when the home side wins should be punished more than one that says draw.
"""
function l61_metrics(probs::Vector{NTuple{3,Float64}}, frame::DataFrame)
    n = length(probs)
    logloss = 0.0; brier = 0.0; rps = 0.0; correct = 0
    scores = Float64[]; labels = Bool[]
    for i in 1:n
        p = probs[i]
        outcome = frame.home_win[i] ? 1 : frame.draw[i] ? 2 : 3
        logloss -= log(clamp(p[outcome], 1e-12, 1.0))
        target = (outcome == 1 ? 1.0 : 0.0, outcome == 2 ? 1.0 : 0.0, outcome == 3 ? 1.0 : 0.0)
        brier += sum((p[k] - target[k])^2 for k in 1:3)
        # Ranked probability score over the ordered H / D / A scale.
        cp = 0.0; ct = 0.0; acc = 0.0
        for k in 1:2
            cp += p[k]; ct += target[k]; acc += (cp - ct)^2
        end
        rps += acc / 2
        correct += argmax(collect(p)) == outcome ? 1 : 0
        push!(scores, p[1]); push!(labels, frame.home_win[i])
    end
    return (logloss = logloss / n, brier = brier / n, rps = rps / n,
            accuracy = correct / n, auc = eda_auc(scores, labels), n = n)
end

# ==============================================================================
# 4. ONE FIT-AND-SCORE PASS
# ==============================================================================

"""
    l61_run(frame, mask, covariates, roles; lambda) -> (metrics, probs, beta)

Fit on `.!mask`, score on `mask`. Teams are indexed over the whole frame so a side that
appears only in the held-out block still resolves — it simply carries the shrunk-to-zero
effect the ridge gives an unobserved team, which is the honest treatment of a promoted
side rather than an error.
"""
function l61_run(frame::DataFrame, mask::AbstractVector{Bool},
                 covariates::Vector{Vector{Float64}}, roles::Vector{Symbol};
                 lambda::Float64 = 4.0)
    teams = sort(unique(vcat(frame.home_team, frame.away_team)))
    team_index = Dict(t => i for (i, t) in enumerate(teams))
    X, y, T = l61_design(frame, covariates, roles, team_index)

    row_train = falses(2 * nrow(frame))
    for i in 1:nrow(frame)
        mask[i] || (row_train[2i - 1] = true; row_train[2i] = true)
    end
    beta = l61_poisson_ridge(X[row_train, :], y[row_train],
                             l61_penalty_mask(size(X, 2), T); lambda = lambda)

    probs = NTuple{3,Float64}[]
    for i in 1:nrow(frame)
        mask[i] || continue
        lam_h = exp(clamp(dot(X[2i - 1, :], beta), -8.0, 8.0))
        lam_a = exp(clamp(dot(X[2i, :], beta), -8.0, 8.0))
        push!(probs, l61_outcome_probs(lam_h, lam_a))
    end
    return l61_metrics(probs, frame[mask, :]), probs, beta
end

# ==============================================================================
# 5. PAIRED BOOTSTRAP ON LOG LOSS
# ==============================================================================

"""
    l61_bootstrap_logloss(probs_candidate, probs_baseline, frame; draws) -> (mean, lo, hi)

Paired bootstrap over held-out matches of the per-match log-loss DIFFERENCE. Negative is
better. Pairing matters: both models are scored on the same fixtures, so most of the
sampling noise is common and cancels.
"""
function l61_bootstrap_logloss(candidate::Vector{NTuple{3,Float64}},
                               baseline::Vector{NTuple{3,Float64}},
                               frame::DataFrame; draws::Int = 4000)
    n = length(candidate)
    n < 40 && return (NaN, NaN, NaN)
    per = Vector{Float64}(undef, n)
    for i in 1:n
        outcome = frame.home_win[i] ? 1 : frame.draw[i] ? 2 : 3
        per[i] = -log(clamp(candidate[i][outcome], 1e-12, 1.0)) +
                  log(clamp(baseline[i][outcome], 1e-12, 1.0))
    end
    draws_out = Vector{Float64}(undef, draws)
    idx = Vector{Int}(undef, n)
    for b in 1:draws
        for i in 1:n
            idx[i] = rand(1:n)
        end
        draws_out[b] = mean(@view per[idx])
    end
    return (mean(per), quantile(draws_out, 0.05), quantile(draws_out, 0.95))
end
