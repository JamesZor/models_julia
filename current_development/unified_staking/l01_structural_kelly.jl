#=
LOADER — structural Kelly portfolio (P) + Baker–McHale shrinkage (U-MC) over the score grid.

Implements the corrected `docs/bets_multi/unified_kelly_postgrad_notes.md`:
  (P)    a* = argmax Σ_ω p̄(ω) log(1 + r(ω)ᵀa)   s.t. a ≥ 0, Σa ≤ cap
  (U-MC) k* = argmax_k (1/S) Σ_s Σ_ω p̄(ω) log(1 + k·r(ω)ᵀ a*(p⁽ˢ⁾))
Execute k*·a*(p̄).

States ω = the 12×12 score grid (goals 0..11), flattened. Markets = binary masks over the grid
(1X2, O/U ladder, BTTS) priced at Betfair close (`odds_close` from summarize_betfair_market).
Per-draw state probabilities come from the L1 posterior λ draws (double-Poisson grid,
renormalized for truncation).

Solver: projected gradient ascent with backtracking on {a ≥ 0, Σa ≤ cap} (Euclidean projection
onto the capped simplex). Validated against Long's closed form on the notes' Example A:
returns (0.1115, 0.0808), cash 0.8077 exactly (see r01).

NOT modelled (flag before any live use): Betfair commission (2–5% on net winnings — kills thin
near-arbs); simultaneous executability of TWA close quotes; per-line recentring calibration
(the model's 1X2 deviations are UNCALIBRATED bias per r13 — see NOTES.md).
=#

using DataFrames
using Distributions
using Random
using Statistics

const GG = 12
const HGRID = vec([h for h in 0:GG-1, a in 0:GG-1])
const AGRID = vec([a for h in 0:GG-1, a in 0:GG-1])

# ---------- (P) solver ----------

"Euclidean projection onto {a .>= 0, sum(a) <= cap} (in place)."
function proj_cap!(a::AbstractVector{Float64}, cap::Float64)
    a .= max.(a, 0.0)
    s = sum(a)
    if s > cap
        u = Base.sort(a, rev=true)   # qualify: DataFrames also exports `sort` (ambiguous under full package load)
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

# ---------- market masks over the score grid ----------

"Binary indicator over the 144 grid states for a (market_name, market_line, selection) row."
function mask_for(mname, mline, sel)
    s = String(sel)
    if mname == "1X2"
        s == "home" && return HGRID .> AGRID
        s == "draw" && return HGRID .== AGRID
        return HGRID .< AGRID
    elseif mname == "OverUnder"
        return startswith(s, "over") ? (HGRID .+ AGRID .> mline) : (HGRID .+ AGRID .< mline)
    elseif mname == "BTTS"
        yes = (HGRID .>= 1) .& (AGRID .>= 1)
        return s == "btts_yes" ? yes : .!yes
    end
    error("unknown market $mname")
end

# ---------- per-draw state probabilities ----------

"144 × n_draws state-probability matrix for one match from latents (λ_h, λ_a vectors per row)."
function state_draws(lat_df::DataFrame, mid)
    row = lat_df[findfirst(==(mid), lat_df.match_id), :]
    λh, λa = row.λ_h, row.λ_a
    S = length(λh)
    P = Matrix{Float64}(undef, GG * GG, S)
    for s in 1:S
        ph = pdf.(Poisson(λh[s]), 0:GG-1)
        pa = pdf.(Poisson(λa[s]), 0:GG-1)
        g = vec(ph * pa')
        P[:, s] = g ./ sum(g)                 # renormalize grid truncation → exact partition
    end
    return P
end

# ---------- the full per-match pipeline ----------

"""
Build the book (1X2 + O/U + BTTS at Betfair close), solve (P) at p̄, re-solve per posterior draw,
pick k* by (U-MC), and return the executed book. `odf` = Betfair-summarized odds DataFrame,
`lat_df` = latents df with per-draw λ.
"""
function run_match(odf::DataFrame, lat_df::DataFrame, mid;
                   cap=1.0, S_dec=200, kgrid=0.01:0.01:1.0, seed=11,
                   families=Set(["1X2", "OverUnder", "BTTS"]))
    book = odf[(odf.match_id .== mid) .& in.(odf.market_name, Ref(families)), :]
    sort!(book, [:market_name, :market_line, :selection])
    masks = [mask_for(r.market_name, r.market_line, r.selection) for r in eachrow(book)]
    d = book.odds_close
    R = hcat([d[m] .* masks[m] .- 1.0 for m in eachindex(masks)]...)

    P = state_draws(lat_df, mid)
    pbar = vec(mean(P, dims=2))
    astar = solve_P(pbar, R; cap=cap)

    idx = rand(Xoshiro(seed), 1:size(P, 2), S_dec)
    A = Matrix{Float64}(undef, length(astar), S_dec)
    for (j, s) in enumerate(idx)
        A[:, j] = solve_P(view(P, :, s), R; cap=cap, a0=astar, iters=800)
    end
    Ψ(k) = mean(G_growth(k .* view(A, :, j), pbar, R) for j in 1:S_dec)
    ks = collect(kgrid)
    kstar = ks[argmax(Ψ.(ks))]

    pm = [sum(pbar[m]) for m in masks]
    out = DataFrame(market = string.(book.market_name) .* "_" .* string.(book.market_line),
                    sel = book.selection, odds = round.(d, digits=3),
                    p_model = round.(pm, digits=4),
                    p_fair_mkt = round.(book.prob_fair_close, digits=4),
                    ev = round.(pm .* d .- 1.0, digits=4),
                    a_star = round.(astar, digits=4),
                    exec = round.(kstar .* astar, digits=4))
    return (book = out[out.a_star .> 0, :], kstar = kstar, total = sum(astar),
            cash = 1 - sum(astar), G = G_growth(astar, pbar, R), A = A, R = R, pbar = pbar)
end

"Settle the executed book at the actual final score; returns terminal wealth per unit bankroll."
function settle(odf::DataFrame, matches::DataFrame, mid, res;
                families=Set(["1X2", "OverUnder", "BTTS"]))
    m = matches[findfirst(==(mid), matches.match_id), :]
    h, a = m.home_score, m.away_score
    st = (HGRID .== h) .& (AGRID .== a)
    book = odf[(odf.match_id .== mid) .& in.(odf.market_name, Ref(families)), :]
    sort!(book, [:market_name, :market_line, :selection])
    masks = [mask_for(r.market_name, r.market_line, r.selection) for r in eachrow(book)]
    rows = string.(book.market_name) .* "_" .* string.(book.market_line) .* "_" .* string.(book.selection)
    exec_full = zeros(nrow(book))
    for (i, rn) in enumerate(string.(res.book.market) .* "_" .* string.(res.book.sel))
        exec_full[findfirst(==(rn), rows)] = res.book.exec[i]
    end
    W = 1.0 - sum(exec_full) +
        sum(exec_full[j] * book.odds_close[j] * (any(masks[j] .& st) ? 1.0 : 0.0) for j in 1:nrow(book))
    return (score = "$h-$a", W = round(W, digits=4))
end
