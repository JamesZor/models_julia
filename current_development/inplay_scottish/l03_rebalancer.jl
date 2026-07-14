#=
l03_rebalancer.jl — WP4 loader: π(ω) payoff-vector state + convex rebalancer
(concept map Concepts 3 + 7; maths grounded in RESEARCH.md §4).

State: π(ω) ∈ R^{G×G} = net payoff per final-score cell from ALL past trades at the
odds actually taken (stakes are NOT the state — you cannot unwind at the entry price).

Program, solved at each state change (P̄ = posterior-predictive mean matrix from l02;
the growth objective is linear in P(ω), so averaging draws inside the log ≡ using P̄ —
the *composed-draw mean matrix*, never the matrix at posterior-mean parameters):

  Δa* = argmax_Δa  Σ_ω P̄(ω) · log(W0 + π(ω) + Σ_k Δa_k r_k(ω; o_k))  −  c·‖Δa‖₁

Concave − ℓ1 ⇒ proximal gradient (ISTA) with backtracking; soft-threshold = the
no-trade region (subgradient of ‖·‖₁ at 0 is [−c,c]).  Δa_k > 0 = back k now,
Δa_k < 0 = lay k now (unit = stake; lay liability handled through r_k).
=#

using LinearAlgebra, Statistics

# ---------------------------------------------------------------------------
# 1. Contracts on the final-score grid
# ---------------------------------------------------------------------------

struct Contract
    sel::Symbol
    price::Float64          # current decimal odds (the odds you'd get NOW)
    win::BitMatrix          # winning cells on the (G×G) final-score grid (0-based goals)
end

"Winning-cell indicator for a selection symbol on a G×G grid (index 1 = 0 goals)."
function cells_for(sel::Symbol, G::Int)::BitMatrix
    W = falses(G, G); s = String(sel)
    for h in 0:(G-1), a in 0:(G-1)
        w = if sel === :home;      h > a
        elseif sel === :draw;      h == a
        elseif sel === :away;      h < a
        elseif sel === :btts_yes;  h > 0 && a > 0
        elseif sel === :btts_no;   !(h > 0 && a > 0)
        elseif startswith(s, "over_")
            h + a > parse(Int, s[6:end-1]) + 0.5 - 0.5   # over_k5 ⇒ total ≥ k+1
        elseif startswith(s, "under_")
            h + a <= parse(Int, s[7:end-1])
        elseif startswith(s, "cs_") && length(s) == 5
            h == parse(Int, s[4:4]) && a == parse(Int, s[5:5])
        else
            error("unknown selection $sel")
        end
        W[h+1, a+1] = w
    end
    return W
end

"Per-unit-stake return of a BACK bet at odds o under each outcome (commission on win)."
back_return(o::Float64, win::BitMatrix; comm = 0.02) =
    ifelse.(win, (o - 1.0) * (1.0 - comm), -1.0)

# ---------------------------------------------------------------------------
# 2. Payoff-vector state
# ---------------------------------------------------------------------------

"""
    Trade(sel, price, stake)  — stake > 0 back, stake < 0 lay (at that price).

`payoff_vector(trades, G; comm)` -> π(ω) grid. A lay of stake s at odds o is encoded
as −s times the back return at o (Betfair lay: win (1−comm)·s when the selection
loses… we fold it through the signed back return, which is exact for matched sizes).
"""
struct Trade
    sel::Symbol
    price::Float64
    stake::Float64
end

function payoff_vector(trades::Vector{Trade}, G::Int; comm = 0.02)
    π = zeros(G, G)
    for t in trades
        π .+= t.stake .* back_return(t.price, cells_for(t.sel, G); comm = comm)
    end
    return π
end

# ---------------------------------------------------------------------------
# 3. The convex rebalancing program (proximal gradient with backtracking)
# ---------------------------------------------------------------------------

soft(x, τ) = sign(x) * max(abs(x) - τ, 0.0)

"""
    rebalance(P̄, π, contracts; W0, c, comm, max_iter, tol)
        -> (Δa, obj, n_trades)

`P̄` :: G×G posterior-predictive mean matrix (sums to 1).  `π` :: G×G payoff state.
`c` :: proportional crossing cost per unit stake (spread haircut). Returns the new
trades Δa (aligned with `contracts`); zeros = inside the no-trade region.
"""
function rebalance(P̄::AbstractMatrix, π::AbstractMatrix, contracts::Vector{Contract};
                   W0 = 1.0, c = 0.01, comm = 0.02, max_iter = 500, tol = 1e-9)
    K = length(contracts)
    R = [back_return(ct.price, ct.win; comm = comm) for ct in contracts]  # K grids
    p = vec(P̄); πv = vec(π); Rv = [vec(r) for r in R]
    live = p .> 0

    wealth(Δa) = W0 .+ πv .+ sum(Δa[k] .* Rv[k] for k in 1:K; init = zeros(length(πv)))
    function obj(Δa)
        w = wealth(Δa)
        any(w[live] .<= 0) && return -Inf
        sum(p[live] .* log.(w[live])) - c * sum(abs, Δa)
    end
    function grad(Δa)
        w = wealth(Δa)
        g = zeros(K)
        for k in 1:K
            g[k] = sum(p[live] .* Rv[k][live] ./ w[live])
        end
        return g
    end

    Δa = zeros(K); f = obj(Δa); η = 0.1
    for _ in 1:max_iter
        g = grad(Δa)
        # backtracking proximal step
        stepped = false
        for _ in 1:40
            cand = soft.(Δa .+ η .* g, η * c)
            fc = obj(cand)
            if fc > f - 1e-15 && isfinite(fc)
                if fc - f < tol && maximum(abs.(cand .- Δa)) < 1e-8
                    return (Δa = cand, obj = fc, n_trades = count(!iszero, cand))
                end
                Δa, f = cand, fc; stepped = true; break
            end
            η *= 0.5
        end
        stepped || break
        η *= 1.3   # gentle step growth
    end
    return (Δa = Δa, obj = f, n_trades = count(!iszero, Δa))
end

# ---------------------------------------------------------------------------
# 4. Convenience: expected growth of the current position (exit-signal building block)
# ---------------------------------------------------------------------------

"Expected log-wealth of holding the current book unchanged under P̄."
hold_growth(P̄, π; W0 = 1.0) = begin
    p = vec(P̄); w = W0 .+ vec(π)
    any((p .> 0) .& (w .<= 0)) ? -Inf : sum(p[p .> 0] .* log.(w[p .> 0]))
end
