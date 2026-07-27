# current_development/plus_minus_ratings/l04_ridge_apm.jl
#
# LOADER (temporary module). WP5 — the reference ridge RAPM estimator and its tuning harness.
#
# THE ESTIMATOR
#   minimise  ‖W^{1/2}(y − Xβ)‖²  +  λ‖Rβ‖²        ⇒   β = (XᵀWX + λ RᵀR)⁻¹ XᵀWy
#
# `R` is where the two published variants differ:
#   * PLAIN RIDGE (base paper):     R = diag(penalised)  — shrink every player toward ZERO.
#   * TEAMMATE SIMILARITY (Hvattum, Arntzen & Pantuso 2020 §3.1; RESEARCH_rapm.md §1.1):
#         R = I − w_SIM · S
#     where `S` is the row-normalised "played alongside" matrix. A player is now shrunk toward
#     the ratings of the players he has SHARED THE MOST MINUTES WITH, not toward the global mean.
#     `w_SIM = 0` recovers plain ridge exactly, so the two are one code path and one grid axis.
#
# Why this matters here: WP2 measured 30.7% of our players below the 540-minute floor, and plain
# ridge collapses every one of them onto the global average. Shrinking toward teammates gives
# them a much better prior at zero data cost — and it is the published FOOTBALL method, unlike
# the box-score priors the basketball literature favours (which RESEARCH_rapm.md §2.1 suggests
# would buy little here anyway).
#
# Home advantage and the dismissal dummies are NEVER penalised (`cols.penalised`), per
# Hvattum et al.: "Regularization is not applied to ... the home field advantages and the effects
# of red cards."
#
# TUNING follows the base paper §5.2.1: aggregate the starting XI's ratings into a team-strength
# covariate, fit an ordered logit for home/draw/away, and minimise OUT-OF-SAMPLE multiclass
# Brier. That is also exactly the WP7 "validity" axis (Hvattum & Gelade 2021), so one harness
# serves both.

using DataFrames
using Dates
using LinearAlgebra
using SparseArrays
using Statistics
using ForwardDiff

include(joinpath(@__DIR__, "l03_targets.jl"))

# ==========================================
# 1. THE SIMILARITY MATRIX
# ==========================================
"""
    similarity_matrix(segments, cols; k=10) -> SparseMatrixCSC

Row-normalised `S` over the player columns: `S[p, p']` > 0 for the `k` players who shared the
most on-pitch minutes with `p` on the SAME team, weighted by those shared minutes.

Note this deliberately induces a mild "good team" pull — a player on a strong side is shrunk
toward strong teammates. Hvattum et al. accept that trade and make `w_SIM` the dial; we report
`w_SIM = 0` alongside the tuned value so the effect is always visible rather than assumed.
"""
function similarity_matrix(segments::DataFrame, cols::DesignCols; k::Int = 10)
    shared = Dict{Tuple{Int, Int}, Float64}()
    for seg in eachrow(segments)
        d = seg.duration
        for side in (seg.home_players, seg.away_players)
            n = length(side)
            for i in 1:n, j in (i + 1):n
                a, b = side[i], side[j]
                key = a < b ? (a, b) : (b, a)
                shared[key] = get(shared, key, 0.0) + d
            end
        end
    end

    nbrs = Dict{Int, Vector{Tuple{Int, Float64}}}()
    for ((a, b), m) in shared
        push!(get!(nbrs, a, Tuple{Int,Float64}[]), (b, m))
        push!(get!(nbrs, b, Tuple{Int,Float64}[]), (a, m))
    end

    np = length(cols.player_ids)
    I = Int[]; J = Int[]; V = Float64[]
    for (p, i) in cols.player_index
        lst = get(nbrs, p, Tuple{Int,Float64}[])
        isempty(lst) && continue
        sort!(lst, by = x -> x[2], rev = true)
        top = first(lst, min(k, length(lst)))
        tot = sum(x[2] for x in top)
        tot <= 0 && continue
        for (q, m) in top
            j = get(cols.player_index, q, 0); j == 0 && continue
            push!(I, i); push!(J, j); push!(V, m / tot)
        end
    end
    return sparse(I, J, V, np, cols.n)
end

# ==========================================
# 2. THE SOLVE
# ==========================================
"""
    penalty_matrix(cols, S, w_sim) -> SparseMatrixCSC

`RᵀR` for the objective above. Unpenalised columns contribute nothing.
"""
function penalty_matrix(cols::DesignCols, S::Union{Nothing, SparseMatrixCSC}, w_sim::Float64)
    np = length(cols.player_ids)
    if S === nothing || w_sim == 0.0
        return spdiagm(0 => Float64.(cols.penalised))
    end
    R = spzeros(cols.n, cols.n)
    for i in 1:np; R[i, i] = 1.0; end
    R[1:np, :] .-= w_sim .* S
    for j in (np + 1):cols.n
        cols.penalised[j] && (R[j, j] = 1.0)
    end
    return R' * R
end

"""
    ridge_solve(A, b, RtR, λ) -> Vector{Float64}

`A = XᵀWX`, `b = XᵀWy`. Cholesky with a jitter fallback: the WP2 spectrum found 38
rank-deficient directions, so an unlucky (λ, penalty) combination can still be singular on the
unpenalised block.
"""
function ridge_solve(A::Matrix{Float64}, b::Vector{Float64}, RtR, λ::Float64)
    M = A + λ .* Matrix(RtR)
    for jit in (0.0, 1e-8, 1e-6, 1e-4)
        try
            return cholesky(Symmetric(M + jit * I)) \ b
        catch
            continue
        end
    end
    return pinv(Symmetric(M)) * b
end

# ==========================================
# 3. ORDERED LOGIT  (the tuning / validity criterion)
# ==========================================
# Outcomes ordered by increasing home strength: 1 = away win, 2 = draw, 3 = home win.
# P(Y≤1) = σ(c₁ − βx), P(Y≤2) = σ(c₂ − βx), with c₂ = c₁ + exp(δ) enforcing the ordering.
_sig(z) = 1 / (1 + exp(-z))

function _ol_probs(θ, x)
    β, c1, δ = θ[1], θ[2], θ[3]
    c2 = c1 + exp(δ)
    p1 = _sig(c1 - β * x)
    p2 = _sig(c2 - β * x) - p1
    return (p1, p2, 1 - p1 - p2)
end

function fit_ordered_logit(x::Vector{Float64}, y::Vector{Int}; iters::Int = 60)
    function nll(θ)
        s = zero(eltype(θ))
        for i in eachindex(x)
            p = _ol_probs(θ, x[i])
            s -= log(clamp(p[y[i]], 1e-12, 1.0))
        end
        return s / length(x)
    end
    θ = [0.1, -0.6, log(1.2)]
    for _ in 1:iters
        g = ForwardDiff.gradient(nll, θ)
        H = ForwardDiff.hessian(nll, θ)
        step = try
            cholesky(Symmetric(H + 1e-6I)) \ g
        catch
            0.01 .* g
        end
        θ_new = θ - step
        # backtrack if the Newton step overshoots
        f0 = nll(θ); t = 1.0
        while nll(θ - t .* step) > f0 && t > 1e-4; t /= 2; end
        θ = θ - t .* step
        maximum(abs.(g)) < 1e-8 && break
    end
    return θ
end

"""Multiclass Brier — the base paper's tuning metric (its R = 3 form)."""
function multiclass_brier(θ, x::Vector{Float64}, y::Vector{Int})
    s = 0.0
    for i in eachindex(x)
        p = _ol_probs(θ, x[i])
        for k in 1:3
            s += (p[k] - (y[i] == k ? 1.0 : 0.0))^2
        end
    end
    return s / length(x)
end

function multiclass_logloss(θ, x::Vector{Float64}, y::Vector{Int})
    s = 0.0
    for i in eachindex(x)
        p = _ol_probs(θ, x[i])
        s -= log(clamp(p[y[i]], 1e-12, 1.0))
    end
    return s / length(x)
end

# ==========================================
# 4. TEAM STRENGTH FROM RATINGS
# ==========================================
"""
    match_strength(segments, β, cols) -> DataFrame

Per match: `strength = Σ(home starters' ratings) − Σ(away starters' ratings)`, plus the observed
outcome coded 1 = away win, 2 = draw, 3 = home win.

Uses the STARTING XI (the players on the pitch in the match's first segment), matching the base
paper, which builds its covariate from the starting eleven.
"""
function match_strength(segments::DataFrame, β::Vector{Float64}, cols::DesignCols)
    meta = pm_match_meta()
    sc = Dict(Int(r.match_id) => (Float64(r.home_score), Float64(r.away_score))
              for r in eachrow(meta) if !ismissing(r.home_score) && !ismissing(r.away_score))

    rows = NamedTuple[]
    for g in groupby(segments, :match_id)
        mid = Int(g.match_id[1])
        haskey(sc, mid) || continue
        first_seg = g[argmin(g.t_start), :]
        rate(p) = (i = get(cols.player_index, p, 0); i == 0 ? 0.0 : β[i])
        s = sum(rate.(first_seg.home_players)) - sum(rate.(first_seg.away_players))
        h, a = sc[mid]
        push!(rows, (match_id = mid, tournament_id = Int(g.tournament_id[1]),
                     season = String(g.season[1]),
                     start_timestamp = g.start_timestamp[1],
                     strength = s, y = h > a ? 3 : (h == a ? 2 : 1)))
    end
    return DataFrame(rows)
end

# ==========================================
# 5. ONE FORWARD-CHAINED EVALUATION
# ==========================================
"""
    evaluate_season(segments, eval_season; target, λ, half_life, w_sim, ...) -> NamedTuple

Protocol:
  1. fit RAPM on every segment from matches STRICTLY BEFORE the evaluation season begins, with
     the time decay anchored at that boundary;
  2. build the team-strength covariate for the EVALUATION matches from that rating vector;
  3. fit the ordered-logit link ON THE EVALUATION SEASON, and fit the no-information floor
     (strength pinned to 0) on those same matches;
  4. report both, and their difference.

WHY THE LINK IS FIT IN-SAMPLE — this was a bug in the first WP5 run. Fitting the link on the
TRAINING matches looks more rigorous but is badly wrong: those matches' ratings were fit on
them, so the strength covariate is far more predictive there than it will ever be
out-of-sample. The logit learns a large slope from that inflated signal, applies it to genuinely
out-of-sample eval ratings, and is overconfident — which is why the first run scored the goals
arm 0.064 Brier WORSE than the no-information floor, an impossible result for a model that can
at worst ignore its covariate.

The RATINGS remain strictly out-of-sample, which is what we are actually measuring. The link is
three parameters fit on ~640 matches, so its in-sample optimism is ~1/640 — and the floor is fit
the same way on the same matches, so the *difference* between them is an honest read on whether
the ratings carry information about outcomes.
"""
const SHOT_TARGETS = (:y_shots, :y_sot, :y_xg)

function evaluate_season(segments_all::DataFrame, eval_season::String;
                         target::Symbol = :y_goals, λ::Float64 = 1.0,
                         half_life::Float64 = 365.0, w_sim::Float64 = 0.0,
                         comp_sets = nothing, sim_k::Int = 10,
                         covered_only::Bool = false,
                         weights_kw::NamedTuple = NamedTuple())
    # CRITICAL: on a match with no live_text the shot columns are 0 by construction, which is
    # indistinguishable from a genuine 0-0 shot difference. Fitting a shot-based target over
    # uncovered segments would feed the regression tens of thousands of fake zeros. Always
    # restrict those targets to the covered subset.
    need_cov = covered_only || target in SHOT_TARGETS
    segments = need_cov && hasproperty(segments_all, :covered) ?
               segments_all[segments_all.covered, :] : segments_all

    ev_rows = segments.season .== eval_season
    any(ev_rows) || return nothing
    t_cut = minimum(DateTime.(segments.start_timestamp[ev_rows]))
    tr_rows = DateTime.(segments.start_timestamp) .< t_cut
    (sum(tr_rows) < 500) && return nothing

    tr = segments[tr_rows, :]
    wcfg = SegmentWeights(; half_life_days = half_life, weights_kw...)
    X, y, w, cols = build_design(tr; target = target, weights = wcfg,
                                 T_rating = t_cut, comp_sets = comp_sets)
    S = w_sim == 0.0 ? nothing : similarity_matrix(tr, cols; k = sim_k)
    A = Matrix(Symmetric(Matrix(X' * Diagonal(w) * X)))
    b = Vector(X' * (w .* y))
    β = ridge_solve(A, b, penalty_matrix(cols, S, w_sim), λ)

    ms = match_strength(segments, β, cols)
    ms_ev = ms[ms.season .== eval_season, :]
    nrow(ms_ev) < 50 && return nothing

    θ  = fit_ordered_logit(ms_ev.strength, ms_ev.y)                    # link, in-sample
    θ0 = fit_ordered_logit(zeros(nrow(ms_ev)), ms_ev.y)                # floor, same matches
    return (season = eval_season, n_train = sum(tr_rows), n_eval = nrow(ms_ev),
            brier = multiclass_brier(θ, ms_ev.strength, ms_ev.y),
            logloss = multiclass_logloss(θ, ms_ev.strength, ms_ev.y),
            floor_brier   = multiclass_brier(θ0, zeros(nrow(ms_ev)), ms_ev.y),
            floor_logloss = multiclass_logloss(θ0, zeros(nrow(ms_ev)), ms_ev.y),
            beta_strength = θ[1],
            ha = β[cols.ha], red1 = β[cols.reds[1]],
            red2 = β[cols.reds[2]], red3 = β[cols.reds[3]],
            sd_players = std(β[1:length(cols.player_ids)]))
end

"""
    sweep(segments, seasons; grid...) -> DataFrame

Cartesian sweep, pooling the per-season results. Brier is averaged over seasons weighted by the
number of evaluation matches, so a short season cannot dominate.
"""
function sweep(segments::DataFrame, eval_seasons::Vector{String};
               targets = [:y_goals], lambdas = [1.0], half_lives = [365.0], w_sims = [0.0],
               comp_sets = nothing, covered_only::Bool = false, label::String = "")
    rows = NamedTuple[]
    for tgt in targets, hl in half_lives, ws in w_sims, λ in lambdas
        per = filter(!isnothing,
                     [evaluate_season(segments, s; target = tgt, λ = λ, half_life = hl,
                                      w_sim = ws, comp_sets = comp_sets,
                                      covered_only = covered_only) for s in eval_seasons])
        isempty(per) && continue
        n = sum(p.n_eval for p in per)
        br = sum(p.brier * p.n_eval for p in per) / n
        fl = sum(p.floor_brier * p.n_eval for p in per) / n
        push!(rows, (target = String(tgt) * label, lambda = λ, half_life = hl, w_sim = ws,
                     seasons = length(per), n_eval = n,
                     brier   = br,
                     logloss = sum(p.logloss * p.n_eval for p in per) / n,
                     # Δ vs the floor fit on the SAME matches. Negative = the ratings help.
                     d_brier = br - fl,
                     d_logloss = sum((p.logloss - p.floor_logloss) * p.n_eval for p in per) / n,
                     beta_strength = mean(p.beta_strength for p in per),
                     ha    = mean(p.ha    for p in per),
                     red1  = mean(p.red1  for p in per),
                     sd_players = mean(p.sd_players for p in per)))
    end
    return DataFrame(rows)
end
