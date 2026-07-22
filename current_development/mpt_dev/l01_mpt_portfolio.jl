#=
LOADER l01 — Markowitz / MPT portfolio strategies over the score grid.

Implements the strategies from Uhrin et al., "Optimal Sports Betting Strategies In Practice"
(docs/modern_portfolio_theory/) that the staking_layer does NOT already have. The Kelly side is
already done: `solve_P` (l02_kelly.jl) IS the paper's §4.3 Kelly, and the policy layer's cap +
trust blend cover §5.1 (bet limit) and §5.2 (fractioning). What is missing is the Markowitz side
and the two risk-constrained Kelly variants:

    solve_mpt            §4.4    max μ'f − γ f'Σf
    solve_msharpe        §4.2.1  max μ'f / √(f'Σf)
    solve_quad_kelly     §4.3.1  Taylor-2 Kelly  (≡ MPT on the SECOND MOMENT, γ = ½)
    solve_kelly_drawdown §5.3    Kelly s.t. P(W_min < α) ≤ β
    solve_kelly_dro      §5.4    max min_{q∈Π} Kelly, box ambiguity set
    frac                 §5.2    fractional wrapper, shared by all of them

EVERYTHING here shares the staking_layer contract: `p` is a length-K vector of world
probabilities (K = 144 flattened score-grid states), `R` is the K × M return matrix
(`return_matrix(d)` from l01_book_schema.jl), and the returned `f` is an M-vector of bankroll
fractions with f ≥ 0, Σf ≤ cap. The unallocated 1 − Σf is the paper's cash asset.

Requires `staking_layer/src/loader.jl` to be included first (uses `proj_cap!` and `G_growth`).

A NOTE ON WHICH STRATEGIES CARE ABOUT POSTERIOR UNCERTAINTY
The Kelly objective Σ_ω p(ω) log(1 + R(ω)'f) is LINEAR in p. So averaging the posterior draws
first (`pbar`) and solving once is exactly equivalent to solving the Bayes-expected objective —
there is nothing to gain from per-draw solving. The Markowitz objectives are NOT linear in p
(Σ is quadratic, Sharpe is a ratio), so for those the posterior spread genuinely changes the
answer. That asymmetry is the main reason this file is worth having.
=#

using Statistics
using LinearAlgebra

# ------------------------------------------------------------------
# 0. Moments of the return distribution
# ------------------------------------------------------------------

"""
    moments(p, R) -> (μ, Σ, S2)

First and second moments of the per-unit-stake net return under world distribution `p`.
  `μ`  = E[ρ]        M          expected net return of each selection
  `Σ`  = Cov[ρ]      M × M      covariance (the MPT risk measure, §4.4)
  `S2` = E[ρρ']      M × M      raw second moment (what Quadratic Kelly uses, §4.3.1)

No independence assumption: the covariance comes straight from the joint grid, so the fact that
`home` and `over_25` co-move within a match is captured exactly. This is strictly better than the
paper's diagonal simplification (§4.4), which only holds for mutually exclusive outcomes.
"""
function moments(p::AbstractVector{Float64}, R::AbstractMatrix{Float64})
    μ  = R' * p
    S2 = R' * (p .* R)
    Σ  = S2 - μ * μ'
    return (μ, Symmetric(Σ), Symmetric(S2))
end

"Euclidean projection onto the simplex {a .>= 0, sum(a) == cap} (in place)."
function proj_simplex!(a::AbstractVector{Float64}, cap::Float64)
    u   = Base.sort(a, rev=true)
    css = cumsum(u)
    ρ   = findlast(i -> u[i] + (cap - css[i]) / i > 0, 1:length(a))
    θ   = (css[ρ] - cap) / ρ
    a  .= max.(a .- θ, 0.0)
    return a
end

"""
    projected_ascent(f0, obj, grad!, proj!; iters, tol) -> f

Shared projected-gradient-ascent driver with backtracking line search. `obj(f)` returns the
scalar to maximise (may be -Inf outside the feasible region), `grad!(g, f)` fills `g`, `proj!(f)`
projects onto the feasible set in place. Used by every solver below so they share one convergence
story rather than five slightly different ones.
"""
function projected_ascent(f0::Vector{Float64}, obj, grad!, proj!;
                          iters::Int=4000, tol::Float64=1e-12, η0::Float64=0.5)
    f = proj!(copy(f0))
    g = similar(f)
    fnew = copy(f)
    for _ in 1:iters
        grad!(g, f)
        cur  = obj(f)
        step = η0
        ok   = false
        for _ in 1:60
            fnew = proj!(f .+ step .* g)
            if obj(fnew) > cur - 1e-15
                ok = true
                break
            end
            step /= 2
        end
        ok || break
        Δ = maximum(abs.(fnew .- f))
        f = fnew
        Δ < tol && break
    end
    return f
end

# ------------------------------------------------------------------
# 1. §4.4 — Modern Portfolio Theory, mean minus γ·variance
# ------------------------------------------------------------------

"""
    solve_mpt(p, R; γ=1.0, cap=1.0) -> f

Paper eq. 4.4:  maximise  E[ρ·f] − γ·f'Σf   s.t.  f ≥ 0, Σf ≤ cap.

`γ` is the risk-aversion knob: γ → 0 collapses to "bet everything on the single highest-EV
selection", γ → ∞ collapses to all-cash. Concave (Σ ⪰ 0) over a convex set ⇒ global optimum.

Note the paper writes Σf = 1 (fully invested including cash). We use Σf ≤ cap with cash implicit,
which is the same feasible set expressed in the staking_layer's convention.
"""
function solve_mpt(p::AbstractVector{Float64}, R::AbstractMatrix{Float64};
                   γ::Float64=1.0, cap::Float64=1.0, f0=nothing, kwargs...)
    μ, Σ, _ = moments(p, R)
    M = size(R, 2)
    obj(f)     = dot(μ, f) - γ * dot(f, Σ * f)
    grad!(g,f) = (g .= μ .- 2γ .* (Σ * f))
    return projected_ascent(f0 === nothing ? fill(1e-3, M) : copy(f0),
                            obj, grad!, f -> proj_cap!(f, cap); kwargs...)
end

# ------------------------------------------------------------------
# 2. §4.3.1 — Quadratic (Taylor-2) Kelly
# ------------------------------------------------------------------

"""
    solve_quad_kelly(p, R; cap=1.0) -> f

Paper eq. 4.10/4.11:  maximise  E[ρ·f − (ρ·f)²/2].

This is MPT with γ = ½ applied to the RAW SECOND MOMENT E[ρρ'], not to the covariance — the
distinction matters whenever μ is not small, and it is the reason quad-Kelly and MPT(γ=½) give
different portfolios in practice despite the paper's remark that they coincide. Use this as the
cheap stand-in for `solve_P` when you need thousands of re-solves (e.g. the parallel-games sweep),
and check the gap against exact Kelly on a subsample.
"""
function solve_quad_kelly(p::AbstractVector{Float64}, R::AbstractMatrix{Float64};
                          cap::Float64=1.0, f0=nothing, kwargs...)
    μ, _, S2 = moments(p, R)
    M = size(R, 2)
    obj(f)     = dot(μ, f) - 0.5 * dot(f, S2 * f)
    grad!(g,f) = (g .= μ .- (S2 * f))
    return projected_ascent(f0 === nothing ? fill(1e-3, M) : copy(f0),
                            obj, grad!, f -> proj_cap!(f, cap); kwargs...)
end

# ------------------------------------------------------------------
# 3. §4.2.1 — Maximum Sharpe ratio
# ------------------------------------------------------------------

"""
    solve_msharpe(p, R; cap=1.0) -> f

Paper eq. 4.6:  maximise  E[ρ·f] / √(f'Σf)   s.t.  f ≥ 0, Σf = cap.

Two deliberate choices, both flagged in the paper:

  * The ratio is scale-invariant, so an unconstrained solve is degenerate and the empty portfolio
    has infinite Sharpe (paper footnote 5). We therefore optimise on the EQUALITY simplex
    Σf = cap, which pins the scale, and let the `frac`/cap wrapper handle sizing.
  * If no selection has positive expected return the problem has no sensible answer; we return
    all-cash rather than the least-bad negative-EV portfolio.

MaxSharpe is quasi-concave where μ'f > 0, so gradient ascent finds the global optimum, but it is
notoriously corner-seeking and sensitive to estimation error — expect it to pile onto one or two
selections. That fragility is the finding, not a bug.
"""
function solve_msharpe(p::AbstractVector{Float64}, R::AbstractMatrix{Float64};
                       cap::Float64=1.0, f0=nothing, kwargs...)
    μ, Σ, _ = moments(p, R)
    M = size(R, 2)
    all(μ .<= 0) && return zeros(M)

    function obj(f)
        num = dot(μ, f)
        var = dot(f, Σ * f)
        (var <= 1e-18 || num <= 0) && return -Inf
        return num / sqrt(var)
    end
    function grad!(g, f)
        num = dot(μ, f)
        var = dot(f, Σ * f)
        sd  = sqrt(max(var, 1e-18))
        # d/df (μ'f / √(f'Σf)) = μ/sd − (μ'f)(Σf)/sd³
        g .= μ ./ sd .- (num / sd^3) .* (Σ * f)
    end

    # warm start on the positive-EV selections so the first step is inside {μ'f > 0}
    start = f0 === nothing ? (μ .> 0) .* (cap / max(count(μ .> 0), 1)) : copy(f0)
    return projected_ascent(collect(Float64, start), obj, grad!,
                            f -> proj_simplex!(f, cap); kwargs...)
end

# ------------------------------------------------------------------
# 4. §5.3 — Kelly with a drawdown constraint
# ------------------------------------------------------------------

"Log of the drawdown functional  log Σ_i p_i (1 + R_i·f)^(−λ)  (log-sum-exp stabilised)."
function _drawdown_c(f, p, R, λ)
    W = 1.0 .+ R * f
    any(W .<= 1e-12) && return Inf
    z = log.(p .+ 1e-300) .- λ .* log.(W)
    mx = maximum(z)
    return mx + log(sum(exp.(z .- mx)))
end

"""
    solve_kelly_drawdown(p, R; α=0.3, β=0.1, cap=1.0) -> f

Paper §5.3: Kelly subject to P(W_min < α) ≤ β, imposed via the Busseti et al. sufficient
condition E[(O·f)^(−λ)] ≤ 1 with λ = log(β)/log(α), i.e. the constraint eq. 5.5

    log Σ_i p_i (1 + R_i·f)^(−λ)  ≤  0

Solved by quadratic penalty: maximise G(f) − ν·max(0, c(f))² with ν escalating over outer
iterations. Not an exact interior-point solve — for a dev prototype the penalty gets you within
a few basis points of the constrained optimum, and you can check feasibility on the returned `f`
via `_drawdown_c(f, p, R, λ) <= 0`. Swap in a proper conic solver before graduating to src/.

Defaults α=0.3, β=0.1 read as: "at most a 10% chance of ever dropping below 30% of bankroll".
"""
function solve_kelly_drawdown(p::AbstractVector{Float64}, R::AbstractMatrix{Float64};
                              α::Float64=0.3, β::Float64=0.1, cap::Float64=1.0,
                              ν0::Float64=10.0, outer::Int=6, f0=nothing, kwargs...)
    λ = log(β) / log(α)
    M = size(R, 2)
    f = f0 === nothing ? fill(1e-3, M) : copy(f0)
    ν = ν0

    for _ in 1:outer
        function obj(fv)
            g = G_growth(fv, p, R)
            isfinite(g) || return -Inf
            c = _drawdown_c(fv, p, R, λ)
            isfinite(c) || return -Inf
            return g - ν * max(0.0, c)^2
        end
        function grad!(gv, fv)
            W = 1.0 .+ R * fv
            gv .= R' * (p ./ W)
            c = _drawdown_c(fv, p, R, λ)
            if isfinite(c) && c > 0
                # ∇c = −λ · R' (w ./ W),  w = normalised p_i(1+R_i f)^(−λ)
                z  = log.(p .+ 1e-300) .- λ .* log.(W)
                w  = exp.(z .- maximum(z)); w ./= sum(w)
                gv .-= 2ν * c .* (-λ .* (R' * (w ./ W)))
            end
        end
        f = projected_ascent(f, obj, grad!, fv -> proj_cap!(fv, cap); kwargs...)
        ν *= 10
    end
    return f
end

# ------------------------------------------------------------------
# 5. §5.4 — Distributionally robust Kelly (box ambiguity set)
# ------------------------------------------------------------------

"""
    dro_worst_case(u, p, η) -> q

Exact inner minimiser of Σ q_i u_i over the box ambiguity set (paper eq. 5.7)

    Π = { q : |q_i − p_i| ≤ η·p_i,  Σq_i = 1,  q ≥ 0 }

Closed form by water-filling: start every state at its floor (1−η)p_i (total mass 1−η), then
spend the remaining η of budget on the states with the SMALLEST u (each can absorb at most
2η·p_i). No LP solver needed, and it gives an exact subgradient for the outer ascent via
Danskin's theorem.
"""
function dro_worst_case(u::AbstractVector{Float64}, p::AbstractVector{Float64}, η::Float64)
    q      = (1 - η) .* p
    budget = 1.0 - sum(q)
    for i in sortperm(u)                      # ascending u = worst outcomes first
        budget <= 1e-15 && break
        room = 2η * p[i]
        add  = min(room, budget)
        q[i] += add
        budget -= add
    end
    return q
end

"""
    solve_kelly_dro(p, R; η=0.1, cap=1.0) -> f

Paper eq. 5.6: maximise min_{q ∈ Π} Σ q_i log(1 + R_i·f), box set of radius η around `p`.

The paper needs this because it has no posterior — η is a hand-tuned stand-in for parameter
uncertainty. YOU HAVE THE POSTERIOR, so the honest comparison is this against a Bayesian robust
variant (e.g. solving against a low posterior quantile of the grid, or against the worst grid
among the draws). Keeping the box version in makes that comparison possible; expect it to be the
safest and lowest-growth strategy in the race, exactly as the paper reports.
"""
function solve_kelly_dro(p::AbstractVector{Float64}, R::AbstractMatrix{Float64};
                         η::Float64=0.1, cap::Float64=1.0, f0=nothing, kwargs...)
    M = size(R, 2)
    function obj(f)
        W = 1.0 .+ R * f
        any(W .<= 1e-12) && return -Inf
        u = log.(W)
        return dot(dro_worst_case(u, p, η), u)
    end
    function grad!(g, f)
        W = 1.0 .+ R * f
        q = dro_worst_case(log.(max.(W, 1e-12)), p, η)
        g .= R' * (q ./ W)                    # Danskin: gradient at the worst-case q
    end
    return projected_ascent(f0 === nothing ? fill(1e-3, M) : copy(f0),
                            obj, grad!, f -> proj_cap!(f, cap); kwargs...)
end

# ------------------------------------------------------------------
# 6. §5.2 — fractional wrapper + diagnostics
# ------------------------------------------------------------------

"Paper eq. 5.1: bet only fraction ω of the risky portfolio, hold 1−ω in cash. 'Half-Kelly' = 0.5."
frac(f::AbstractVector{Float64}, ω::Float64) = ω .* f

"Sharpe ratio of a portfolio under `p` (0 for the empty portfolio, not Inf)."
function sharpe(f, p, R)
    μ, Σ, _ = moments(p, R)
    var = dot(f, Σ * f)
    var <= 1e-18 && return 0.0
    return dot(μ, f) / sqrt(var)
end

"""
    compare_strategies(m; cap=0.2, γ=1.0, ω=0.5) -> DataFrame-ready NamedTuple vector

One-match diagnostic: run every strategy on a StakingMatch's (pbar, R) and report allocation,
expected log-growth, Sharpe, and realised return under the settled score. Sanity check before
wiring any of this into the race — if quad-Kelly and `solve_P` disagree by more than a few
percent of stake here, something is wrong with the moments.
"""
function compare_strategies(m; cap::Float64=0.2, γ::Float64=1.0, ω::Float64=0.5)
    p, R = m.pbar, m.R
    realised(f) = dot(f, [m.won[j] ? (m.d[j] - 1.0) : -1.0 for j in 1:length(m.d)])
    cands = [
        "Kelly"          => solve_P(p, R; cap=cap),
        "KellyFrac"      => frac(solve_P(p, R; cap=cap), ω),
        "QuadKelly"      => solve_quad_kelly(p, R; cap=cap),
        "MPT(γ)"         => solve_mpt(p, R; γ=γ, cap=cap),
        "MSharpe"        => solve_msharpe(p, R; cap=cap),
        "MSharpeFrac"    => frac(solve_msharpe(p, R; cap=cap), ω),
        "KellyDrawdown"  => solve_kelly_drawdown(p, R; cap=cap),
        "KellyRobust"    => solve_kelly_dro(p, R; cap=cap),
    ]
    return [(strategy=n, stake=round(sum(f), digits=4),
             G=round(G_growth(f, p, R), digits=6),
             sharpe=round(sharpe(f, p, R), digits=4),
             realised=round(realised(f), digits=4),
             f=round.(f, digits=4)) for (n, f) in cands]
end
