using BayesianFootball
using DataFrames
using LinearAlgebra
using Statistics
using Printf

# ==============================================================================
# UTILITY MATH FUNCTIONS
# ==============================================================================
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

function G_growth(a, p, R)
    W = 1.0 .+ R * a
    return any(W .<= 1e-12) ? -Inf : sum(p .* log.(W))
end

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

logit(p) = log(p / (1.0 - p))

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

function mask_for(mname, mline, sel)
    s = String(sel)
    HGRID = vec([h for h in 0:11, a in 0:11])
    AGRID = vec([a for h in 0:11, a in 0:11])
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
end

function state_draws(lat_df::DataFrame, mid)
    row = lat_df[findfirst(==(mid), lat_df.match_id), :]
    λh, λa = row.λ_h, row.λ_a
    S = length(λh)
    P = Matrix{Float64}(undef, 144, S)
    for s in 1:S
        ph = pdf.(Poisson(λh[s]), 0:11)
        pa = pdf.(Poisson(λa[s]), 0:11)
        g = vec(ph * pa')
        P[:, s] = g ./ sum(g)
    end
    return P
end

# ==============================================================================
# DEMO SCRIPT
# ==============================================================================
function run_math_demo()
    mid = 13250709 # Let's use the exact Drogheda vs Shamrock Rovers match!
    
    h_team = ds.matches.home_team[findfirst(==(mid), ds.matches.match_id)]
    a_team = ds.matches.away_team[findfirst(==(mid), ds.matches.match_id)]
    println("======================================================")
    println("📐 STRUCTURAL MATH DEMO: $h_team vs $a_team")
    println("======================================================")

    P_raw = state_draws(l1_latents.df, mid)
    p = vec(mean(P_raw, dims=2))

    # We'll pick the first 4 markets available in the book for this match to keep the matrix readable!
    families = Set(["1X2", "OverUnder"])
    book = ds1.odds[(ds1.odds.match_id .== mid) .& in.(ds1.odds.market_name, Ref(families)), :]
    sort!(book, [:market_name, :market_line, :selection])
    
    # Take up to 4 markets
    M = min(4, nrow(book))
    demo_markets = book[1:M, :]
    
    odds = demo_markets.odds_close
    masks = [mask_for(r.market_name, r.market_line, r.selection) for r in eachrow(demo_markets)]
    
    # Construct Return Matrix R (144 states × M markets)
    R = hcat([odds[m] .* masks[m] .- 1.0 for m in 1:M]...)

    println("\n[STEP 1]: THE RETURN MATRIX (R)")
    println("R is a 144x$M matrix. Here is how it looks for just a few specific scorelines:")
    
    # Dynamically build the header string
    header = "State (H, A) |"
    for r in eachrow(demo_markets)
        header *= @sprintf(" %10s (%4.2f) |", r.selection, r.odds_close)
    end
    println(header)
    println("-" ^ length(header))
    
    scores_to_show = [(1,0), (0,1), (2,2), (0,0)]
    for (h, a) in scores_to_show
        idx = (a * 12) + h + 1 # 0-indexed flat index
        row_str = @sprintf("Score (%d, %d) |", h, a)
        for m in 1:M
            row_str *= @sprintf(" %17.2f |", R[idx, m])
        end
        println(row_str)
    end

    # Solve optimal stakes for the gradient/hessian point
    a_dummy = fill(0.01, M) # Evaluate at a flat 1% exposure
    W = 1.0 .+ R * a_dummy
    
    println("\n[STEP 2]: THE GRADIENT AND HESSIAN")
    grad = R' * (p ./ W)
    H = -R' * (Diagonal(p ./ (W.^2))) * R
    
    println("Gradient (Direction of steepest expected log-growth evaluated at 1% stakes):")
    for m in 1:M
        @printf("  %10s: %8.4f\n", demo_markets.selection[m], grad[m])
    end
    
    println("\nHessian Matrix (Covariance of Payoffs):")
    print("          ")
    for m in 1:M
        @printf("%12s ", demo_markets.selection[m])
    end
    println()
    
    for i in 1:M
        @printf("%-10s", demo_markets.selection[i])
        for j in 1:M
            @printf("%12.4f ", H[i, j])
        end
        println()
    end

    # ==============================================================================
    # STEP 3: THE SENSITIVITY (JACOBIAN INTUITION)
    # ==============================================================================
    println("\n[STEP 3]: SENSITIVITY TO PROBABILITY (THE JACOBIAN EFFECT)")
    # Find the true optimal stakes
    astar_true = solve_P(p, R, cap=0.2)
    
    # Artificially "nudge" the probability of Home by exactly 2%
    p_nudged = copy(p)
    home_mask = mask_for("1X2", 0.0, :home)
    p_nudged[home_mask] .*= 1.02  # Nudge home probability up
    p_nudged ./= sum(p_nudged)    # Renormalize
    
    astar_nudged = solve_P(p_nudged, R, cap=0.2)
    
    println("Original Optimal Stakes:")
    for m in 1:M
        @printf("  %10s: %5.2f%%\n", demo_markets.selection[m], astar_true[m]*100)
    end
    
    println("\nStakes after a tiny 2% relative error inserted into Home probability:")
    for m in 1:M
        @printf("  %10s: %5.2f%%\n", demo_markets.selection[m], astar_nudged[m]*100)
    end
    println("Notice how a tiny shift in `p` causes a massive swing in `a*`. This intense sensitivity")
    println("is governed by the Jacobian matrix. This is exactly why Baker-McHale shrinks stakes (k*)!")

    # ==============================================================================
    # STEP 4: LAYER 2 AND THE GRADIENT SHIFT
    # ==============================================================================
    println("\n[STEP 4]: LAYER 2 AND THE GRADIENT SHIFT")
    # Grab the actual Layer 2 targets for this match
    l2_book = l1_l2_ppd.df[l1_l2_ppd.df.match_id .== mid, :]
    
    t_home_idx = findfirst(r -> r.market_name == "1X2" && r.selection == :home, eachrow(l2_book))
    t_draw_idx = findfirst(r -> r.market_name == "1X2" && r.selection == :draw, eachrow(l2_book))
    t_o25_idx  = findfirst(r -> r.market_name == "OverUnder" && r.market_line == 2.5 && r.selection == :over_25, eachrow(l2_book))
    t_btts_idx = findfirst(r -> r.market_name == "BTTS" && r.selection == :btts_yes, eachrow(l2_book))

    # Safely pull targets
    targets = (
        home     = t_home_idx !== nothing ? mean(l2_book.distribution[t_home_idx]) : sum(p[mask_for("1X2", 0.0, "home")]),
        draw     = t_draw_idx !== nothing ? mean(l2_book.distribution[t_draw_idx]) : sum(p[mask_for("1X2", 0.0, "draw")]),
        over25   = t_o25_idx  !== nothing ? mean(l2_book.distribution[t_o25_idx])  : sum(p[mask_for("OverUnder", 2.5, "over")]),
        btts_yes = t_btts_idx !== nothing ? mean(l2_book.distribution[t_btts_idx]) : sum(p[mask_for("BTTS", 0.0, "btts_yes")])
    )

    mult = tilt_core_grid(p, targets)
    p_l2 = p .* mult
    p_l2 ./= sum(p_l2)
    
    # Calculate the New Gradient from Layer 2
    grad_l2 = R' * (p_l2 ./ W)
    
    println("How Layer 2 Calibration shifts the Gradient (Optimizer force-field):")
    for m in 1:M
        @printf("  %10s | L1 Grad: %8.4f  ->  L2 Grad: %8.4f\n", demo_markets.selection[m], grad[m], grad_l2[m])
    end
    println("\nThe L2 probability shift forces the Gradient down on the longshots, actively steering")
    println("the solver away from the Favorite-Longshot trap before stakes are even placed!")
end

run_math_demo()
