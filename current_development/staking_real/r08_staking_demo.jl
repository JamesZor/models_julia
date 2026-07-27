#=
RUNNER — Deconstructed Textbook Demo (Chapters 12, 13, 16)
Picks a random match, computes the Unified Kelly optimal stakes (P) and Baker-McHale shrinkage (U-MC)
for both Layer 1 and Layer 1 + Layer 2 (Trust Blend).

Completely self-contained so you can modify the solver, IPF tilt, and utility functions inline!
=#

using BayesianFootball
using DataFrames
using Distributions
using Random
using Statistics
using Printf

println("🚀 Booting Deconstructed Staking Demo...")

# ==============================================================================
# 1. STRUCTURAL KELLY SOLVER & U-MC MECHANICS
# ==============================================================================

const GG = 12
const HGRID = vec([h for h in 0:GG-1, a in 0:GG-1])
const AGRID = vec([a for h in 0:GG-1, a in 0:GG-1])

"Euclidean projection onto {a .>= 0, sum(a) <= cap} (in place)."
function proj_cap!(a::AbstractVector{Float64}, cap::Float64)
    a .= max.(a, 0.0)
    s = sum(a)
    if s > cap
        u = sort(a, rev=true)
        css = cumsum(u)
        ρ = findlast(i -> u[i] + (cap - css[i]) / i > 0, 1:length(a))
        θ = (css[ρ] - cap) / ρ
        a .= max.(a .- θ, 0.0)
    end
    return a
end

"Expected log-growth Σ p log(1 + Ra); -Inf outside the wealth-positive region."
function G_growth(a, p, R)
    W = 1.0 .+ R * a
    return any(W .<= 1e-12) ? -Inf : sum(p .* log.(W))
end

"""
Solve (P): projected gradient ascent with backtracking.
Concave objective + convex feasible set => global optimum.
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

"Binary indicator over the 144 grid states for a market selection."
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

"Extract the 144 x S state-probability matrix for one match from latents."
function state_draws(lat_df::DataFrame, mid)
    row = lat_df[findfirst(==(mid), lat_df.match_id), :]
    λh, λa = row.λ_h, row.λ_a
    S = length(λh)
    P = Matrix{Float64}(undef, GG * GG, S)
    for s in 1:S
        ph = pdf.(Poisson(λh[s]), 0:GG-1)
        pa = pdf.(Poisson(λa[s]), 0:GG-1)
        g = vec(ph * pa')
        P[:, s] = g ./ sum(g) # renormalize grid truncation -> exact partition
    end
    return P
end

# ==============================================================================
# 2. I-PROJECTION (COHERENT GRID TILT)
# ==============================================================================

logit(p) = log(p / (1.0 - p))

"""
    tilt_core_grid(pbar, targets; cycles=50, tol=1e-8)

Coherent IPF multiplier that imprints the blended targets onto the plain grid.
Targets: Tuple (home, draw, over25, btts_yes).
"""
function tilt_core_grid(pbar, targets; cycles=50, tol=1e-8)
    masks = [
        mask_for("1X2", 0.0, "home"),
        mask_for("1X2", 0.0, "draw"),
        mask_for("OverUnder", 2.5, "over"),
        mask_for("BTTS", 0.0, "btts_yes")
    ]
    targs = [targets.home, targets.draw, targets.over25, targets.btts_yes]
    
    g = copy(pbar)
    mult = ones(length(g))
    
    for _ in 1:cycles
        moved = 0.0
        for j in 1:4
            m = masks[j]
            cur = sum(g[m])
            t = clamp(targs[j], 1e-9, 1 - 1e-9)
            
            δ = logit(t) - logit(clamp(cur, 1e-9, 1 - 1e-9))
            e = exp(δ)
            
            g[m] .*= e
            mult[m] .*= e
            
            z = sum(g)
            g ./= z
            mult ./= z
            
            moved = max(moved, abs(δ))
        end
        moved < tol && break
    end
    return mult
end


# ==============================================================================
# 3. DEMO EXECUTION
# ==============================================================================

if !@isdefined(l1_latents)
    println("Extracting L1 Latent draws from layered_results...")
    l1_latents = BayesianFootball.Experiments.extract_oos_predictions(ds, layered_results.l1_results)
end

function evaluate_match_staking(mid::Int; cap::Float64=0.2)
    h_team = ds.matches.home_team[findfirst(==(mid), ds.matches.match_id)]
    a_team = ds.matches.away_team[findfirst(==(mid), ds.matches.match_id)]
    println("\n⚽ Selected Match: $h_team vs $a_team (ID: $mid)")

    # 2. Extract the model posterior draws (Layer 1)
    # state_draws outputs the 144 x S matrix of posterior grids
    P_raw = state_draws(l1_latents.df, mid)
    pbar_raw = vec(mean(P_raw, dims=2))

    # 3. Build the book (we use the CORE families as requested)
    families = Set(["1X2", "OverUnder", "BTTS"])
    book = ds1.odds[(ds1.odds.match_id .== mid) .& in.(ds1.odds.market_name, Ref(families)), :]
    sort!(book, [:market_name, :market_line, :selection])

    if nrow(book) == 0
        println("⚠️ No valid core markets found for this match!")
        return
    end

    masks = [mask_for(r.market_name, r.market_line, r.selection) for r in eachrow(book)]
    d = book.odds_close
    R = hcat([d[m] .* masks[m] .- 1.0 for m in eachindex(masks)]...)

    # ------------------------------------------------------------------------------
    # LAYER 1: No Trust (Unified Staking + U-MC)
    # ------------------------------------------------------------------------------
    astar_raw = solve_P(pbar_raw, R; cap=cap)

    # Compute Baker-McHale Shrinkage (U-MC) across the raw posterior draws
    S_dec = min(200, size(P_raw, 2))
    idx = rand(Xoshiro(11), 1:size(P_raw, 2), S_dec)
    A_raw = Matrix{Float64}(undef, length(astar_raw), S_dec)
    for (j, s) in enumerate(idx)
        A_raw[:, j] = solve_P(view(P_raw, :, s), R; cap=cap, a0=astar_raw, iters=800)
    end
    Ψ_raw(k) = mean(G_growth(k .* view(A_raw, :, j), pbar_raw, R) for j in 1:S_dec)
    ks = collect(0.01:0.01:1.0)
    kstar_raw = ks[argmax(Ψ_raw.(ks))]

    # ------------------------------------------------------------------------------
    # LAYER 1 + LAYER 2: Trust-Calibrated Blend
    # ------------------------------------------------------------------------------
    # Extract fully calibrated Trust-Blended targets directly from Layer 2's output!
    l2_book = l1_l2_ppd.df[l1_l2_ppd.df.match_id .== mid, :]
    
    if nrow(l2_book) == 0
        println("⚠️ Match not found in L2 calibrated output!")
        return
    end

    t_home_idx = findfirst(r -> r.market_name == "1X2" && r.selection == :home, eachrow(l2_book))
    t_draw_idx = findfirst(r -> r.market_name == "1X2" && r.selection == :draw, eachrow(l2_book))
    t_o25_idx  = findfirst(r -> r.market_name == "OverUnder" && r.market_line == 2.5 && r.selection == :over_25, eachrow(l2_book))
    t_btts_idx = findfirst(r -> r.market_name == "BTTS" && r.selection == :btts_yes, eachrow(l2_book))

    # Safely pull targets, fallback to L1 marginals if the market was dropped in L2
    targets = (
        home     = t_home_idx !== nothing ? mean(l2_book.distribution[t_home_idx]) : sum(pbar_raw[mask_for("1X2", 0.0, "home")]),
        draw     = t_draw_idx !== nothing ? mean(l2_book.distribution[t_draw_idx]) : sum(pbar_raw[mask_for("1X2", 0.0, "draw")]),
        over25   = t_o25_idx  !== nothing ? mean(l2_book.distribution[t_o25_idx])  : sum(pbar_raw[mask_for("OverUnder", 2.5, "over")]),
        btts_yes = t_btts_idx !== nothing ? mean(l2_book.distribution[t_btts_idx]) : sum(pbar_raw[mask_for("BTTS", 0.0, "btts_yes")])
    )

    # Compute the I-Projection Coherent Multiplier
    mult = tilt_core_grid(pbar_raw, targets)

    # The Textbook Definition of L1+L2 Dispension:
    # We apply the I-projection multiplier to EVERY SINGLE POSTERIOR DRAW, keeping the grid coherent
    # but preserving the exact logit-dispersion of the original posterior for Baker-McHale!
    P_tilt = P_raw .* mult
    P_tilt ./= sum(P_tilt, dims=1) # Renormalize each draw to sum to 1
    pbar_tilt = vec(mean(P_tilt, dims=2))

    astar_tilt = solve_P(pbar_tilt, R; cap=cap)

    # Compute Baker-McHale Shrinkage (U-MC) across the TILTED posterior draws
    A_tilt = Matrix{Float64}(undef, length(astar_tilt), S_dec)
    for (j, s) in enumerate(idx)
        A_tilt[:, j] = solve_P(view(P_tilt, :, s), R; cap=cap, a0=astar_tilt, iters=800)
    end
    Ψ_tilt(k) = mean(G_growth(k .* view(A_tilt, :, j), pbar_tilt, R) for j in 1:S_dec)
    kstar_tilt = ks[argmax(Ψ_tilt.(ks))]

    # ==============================================================================
    # 4. SETTLEMENT & COMPARISON BREAKDOWN
    # ==============================================================================
    println("\n=======================================================================================")
    println(" 📊 FINAL COMPARISON BREAKDOWN & SETTLEMENT")
    println("=======================================================================================")

    h_score = ds.matches.home_score[findfirst(==(mid), ds.matches.match_id)]
    a_score = ds.matches.away_score[findfirst(==(mid), ds.matches.match_id)]
    println("🏁 Actual Match Result: $h_team $h_score - $a_score $a_team\n")
    @printf("Layer 1 U-MC Shrinkage (k*): %.2f\n", kstar_raw)
    @printf("L1 + L2 U-MC Shrinkage (k*): %.2f\n\n", kstar_tilt)

    l1_pnl = 0.0
    l2_pnl = 0.0

    @printf("%-15s %-12s | %-8s | %-16s | %-16s | %-10s | %-10s\n", 
            "MARKET", "SELECTION", "ODDS", "L1 (Prob / Stake)", "L1+L2 (Prob / Stake)", "OUTCOME", "PnL Diff")
    println("-" ^ 92)

    for (i, r) in enumerate(eachrow(book))
        stake1 = kstar_raw * astar_raw[i]
        stake2 = kstar_tilt * astar_tilt[i]
        
        # Calculate probabilities from the grids
        p1 = sum(pbar_raw[masks[i]])
        p2 = sum(pbar_tilt[masks[i]])
        
        payoff = sel_payoff(r.market_name, r.selection, r.market_line, h_score, a_score, r.odds_close)
        res_str = payoff > 0 ? "WIN" : "LOSS"
        
        pnl1 = stake1 * payoff
        pnl2 = stake2 * payoff
        l1_pnl += pnl1
        l2_pnl += pnl2
        
        # Only print rows where at least one model placed a stake
        if stake1 > 1e-4 || stake2 > 1e-4
            l1_str = @sprintf("%5.1f%% / %5.2f%%", p1 * 100, stake1 * 100)
            l2_str = @sprintf("%5.1f%% / %5.2f%%", p2 * 100, stake2 * 100)
            diff_str = (pnl2 - pnl1) > 0 ? @sprintf("+%.3f", pnl2 - pnl1) : @sprintf("%.3f", pnl2 - pnl1)
            
            @printf("%-15s %-12s | %-8.2f | %-16s | %-16s | %-10s | %-10s\n", 
                    r.market_name, r.selection, r.odds_close, l1_str, l2_str, res_str, diff_str)
        end
    end

    println("-" ^ 92)
    @printf("%-40s L1 Total PnL: %6.3f units\n", "", l1_pnl)
    @printf("%-40s L2 Total PnL: %6.3f units\n", "", l2_pnl)
    @printf("%-40s L2 Advantage: %6.3f units\n", "", l2_pnl - l1_pnl)
    println("=======================================================================================")
end

# Evaluate a random match to kick things off!
valid_match_ids = unique(l1_ppd.df.match_id)
random_mid = rand(valid_match_ids)
evaluate_match_staking(random_mid; cap=0.2)
