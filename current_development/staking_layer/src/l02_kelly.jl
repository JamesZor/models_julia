#=
LOADER l02 — structural Kelly portfolio solver (P) over the score grid.

Extracted verbatim from the old unified_staking/l01_structural_kelly.jl (the module now absorbs it).
Implements docs/bets_multi/unified_kelly_postgrad_notes.md:
  (P)  a* = argmax_a  Σ_ω p(ω) log(1 + r(ω)ᵀa)   s.t. a ≥ 0, Σa ≤ cap

States ω = the flattened 12×12 score grid; markets = binary masks (the R matrix, 144 × M).
Solver: projected gradient ascent with backtracking (Euclidean projection onto the capped simplex).
Concave objective + convex feasible set ⇒ global optimum. Validated against Long's closed form
(notes Example A → (0.1115, 0.0808), cash 0.8077).

Pure math — no grid constants, no schema, no market knowledge. Works on any (p, R).
=#

"Euclidean projection onto {a .>= 0, sum(a) <= cap} (in place)."
function proj_cap!(a::AbstractVector{Float64}, cap::Float64)
    a .= max.(a, 0.0)
    s = sum(a)
    if s > cap
        u = Base.sort(a, rev=true)   # qualify: DataFrames also exports `sort` under full package load
        css = cumsum(u)
        ρ = findlast(i -> u[i] + (cap - css[i]) / i > 0, 1:length(a))
        θ = (css[ρ] - cap) / ρ
        a .= max.(a .- θ, 0.0)
    end
    return a
end

"Expected log-growth Σ p log(1 + Ra); -Inf outside the wealth-positive region."
G_growth(a, p, R) = begin
    W = 1.0 .+ R * a
    any(W .<= 1e-12) ? -Inf : sum(p .* log.(W))
end

"""
Solve (P): projected gradient ascent with backtracking. `a0` warm start (use a*(p̄) when
re-solving per posterior draw). Concave objective + convex feasible set ⇒ global optimum.
"""
function solve_P(p, R; cap=1.0, a0=nothing, iters=4000, tol=1e-12)
    M = size(R, 2)
    a = a0 === nothing ? fill(1e-3, M) : copy(a0)
    proj_cap!(a, cap)
    g = similar(a)
    η = 0.5
    anew = copy(a)
    for _ in 1:iters
        W = 1.0 .+ R * a
        g .= R' * (p ./ W)
        gold = G_growth(a, p, R)
        step = η
        for _ in 1:60
            anew = proj_cap!(a .+ step .* g, cap)
            G_growth(anew, p, R) > gold - 1e-15 && break
            step /= 2
        end
        maximum(abs.(anew .- a)) < tol && (a = anew; break)
        a = anew
    end
    return a
end
