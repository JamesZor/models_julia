# src/features/plus_minus/ridge.jl
#
# The reference ridge RAPM estimator.
#
#   minimise  ||W^{1/2}(y - X beta)||^2  +  lambda ||R beta||^2
#     =>      beta = (XᵀWX + lambda RᵀR)^{-1} XᵀWy
#
# `R` is where the two published variants differ:
#   * PLAIN RIDGE (base paper):     R = diag(penalised)  — shrink every player toward ZERO.
#   * TEAMMATE SIMILARITY (Hvattum, Arntzen & Pantuso 2020 §3.1):
#         R = I - w_SIM · S
#     where `S` is the row-normalised "played alongside" matrix, so a player is shrunk toward the
#     ratings of the players he has SHARED THE MOST MINUTES WITH. `w_SIM = 0` recovers plain ridge
#     exactly, so the two are one code path and one config knob.
#
# ⚠ WHY THE DEFAULT IS `w_sim = 0` DESPITE `w_sim = 0.9` SCORING BETTER ON MATCH-OUTCOME BRIER:
# raising w_SIM improves outcome prediction PRECISELY BY MAKING THE RATING MORE TEAM-LIKE (club R²
# 0.389 -> 0.755 on `y_shots`). The pre-registered decision rule was split-half reliability plus
# retrodiction, and the honest cell that passed it is `y_shots, w_SIM = 0`. Anything higher is
# partly re-deriving team strength, which the engine already models with `dyn.α / dyn.β`.
#
# Home advantage and the dismissal dummies are NEVER penalised (`cols.penalised`), per Hvattum et
# al.: "Regularization is not applied to ... the home field advantages and the effects of red cards."
#
# FAITHFUL PORT of the estimator half of current_development/plus_minus_ratings/l04_ridge_apm.jl.
# The ordered-logit tuning harness in that file is NOT ported: it was the research's own validation
# criterion, and here the criterion is the L1 engine's out-of-sample proper scoring.

using DataFrames
using Dates
using LinearAlgebra
using SparseArrays
using Statistics

# ==========================================
# 1. THE SIMILARITY MATRIX
# ==========================================
"""
    similarity_matrix(segments, cols; k=10) -> SparseMatrixCSC

Row-normalised `S` over the player columns: `S[p, p'] > 0` for the `k` players who shared the most
on-pitch minutes with `p` on the SAME team, weighted by those shared minutes.

Note this deliberately induces a "good team" pull — a player on a strong side is shrunk toward
strong teammates. That is exactly why `w_sim` defaults to 0; see the file header.
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

`A = XᵀWX`, `b = XᵀWy`. Cholesky with a jitter fallback: the research found 38 rank-deficient
directions in the segment spectrum, so an unlucky (λ, penalty) combination can still be singular on
the unpenalised block.
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
# 3. ONE FIT
# ==========================================
"""
    fit_ratings(segments; target, λ, w_sim, half_life, T_rating, comp_sets, sim_k)
        -> DataFrame(player_id, rapm) | nothing

Ridge RAPM over an arbitrary segment subset. Mirrors `r08_reliability.jl::fit_ratings`, which is the
function the WP7 verdict was measured with — cross-check any change against it.

Shot-based targets are automatically restricted to `segments.covered` (see `attach_shot_targets!`);
skipping that silently feeds the regression a fake 0 for every match without live text.

Returns `nothing` when the subset is too small to fit (< 500 segments), which the caller must treat
as "emit zero ratings", not as an error.
"""
function fit_ratings(segments::DataFrame;
                     target::Symbol = :y_shots,
                     λ::Float64 = 1000.0,
                     w_sim::Float64 = 0.0,
                     half_life::Float64 = 730.0,
                     T_rating::Union{Nothing, Date} = nothing,
                     comp_sets::Union{Nothing, Dict{Int, Dict{Int, Set{Int}}}} = nothing,
                     sim_k::Int = 10)
    nrow(segments) == 0 && return nothing
    segs = (target in PM_SHOT_TARGETS && hasproperty(segments, :covered)) ?
           segments[segments.covered, :] : segments
    nrow(segs) < 500 && return nothing

    Te = T_rating === nothing ? maximum(segs.match_date) : T_rating
    wcfg = SegmentWeights(; half_life_days = half_life)
    X, y, w, cols = build_design(segs; target = target, weights = wcfg,
                                 T_rating = Te, comp_sets = comp_sets)

    S = w_sim == 0.0 ? nothing : similarity_matrix(segs, cols; k = sim_k)
    A = Matrix(Symmetric(Matrix(X' * Diagonal(w) * X)))
    b = Vector(X' * (w .* y))
    β = ridge_solve(A, b, penalty_matrix(cols, S, w_sim), λ)

    np = length(cols.player_ids)
    return DataFrame(player_id = cols.player_ids, rapm = β[1:np])
end
