#=
r01_portfolio_kelly_hedge.jl  —  Portfolio-level Kelly sizing + continuous partial in-play hedging.

CONTEXT / CORRECTION (2026-06-14)
--------------------------------
An earlier analysis concluded "hedging adds nothing" by comparing HOLD vs FULL-EXIT at 1/4-Kelly. Two
mistakes, both flagged by the user:

  1. The staking baseline was wrong. The DCMH ledger is staked per-bet with `BayesianKelly` (Baker-McHale
     distributional optimiser — already shrinks for posterior variance/skew, src/signals/implementations/kelly.jl).
     That is correct PER BET, but a match fires ~6 SIMULTANEOUS, CORRELATED bets (nested OU lines + BTTS +
     1X2). Summing independent Kelly fractions over-stakes the JOINT position: mean 40% of bankroll/match,
     MAX 217%, and one match returns -143% => literal BANKRUPTCY (growth = -Inf, bank -> 0). The fix is a
     PORTFOLIO cap (Kelly on the joint book), not a flat 1/4 scaling.

  2. The hedge action space was wrong. Exit must be CONTINUOUS (lay off a fraction φ∈[0,1], i.e. cash out
     part / reduce risk), not binary hold-vs-full-exit. Full exit (φ=1) is essentially never optimal.

RESULTS (Ireland DCMH backtest, 258 matches, sequential per-match compounding)
------------------------------------------------------------------------------
  * Raw independent per-bet BayesianKelly: BANKRUPT (worst match -143%).
  * Portfolio cap scan (hold-only): textbook Kelly parabola, growth-optimal joint cap ≈ 0.20
    (log-growth 3.08, ×21.7 bank); >0.45 negative; full-bankroll exposure -6.0.
  * Continuous partial hedge on the sized book: optimum is INTERIOR (φ≈0.5–0.75), never full exit.
        exit@70': hold 3.08, φ=0.5 -> 3.38   (full exit φ=1 -> 2.67, WORSE than hold)
        exit@80': hold 3.08, φ=0.75-> 3.81   (full exit φ=1 -> 3.61)
  * SYNERGY: hedging cuts variance => the optimal book grows. Joint optimum cap×φ:
        70' exit: cap 0.25, φ=0.50 -> 3.60   (vs hold-only optimum 3.08)
        80' exit: cap 0.30, φ=0.75 -> 4.30   (≈3.4× the hold-only-optimum terminal wealth)
    Where an un-hedged book at cap 0.40–0.45 collapses toward ruin, the hedged book stays positive.

CAVEATS: in-sample (258 backtest matches, single league); wall-clock 80' ≈ match-minute ~65 (late, thin
liquidity / possible mild near-settlement lookahead — trust the 70' result more); LTP ≠ tradeable lay
price; this is a RULE-based hedge (fixed φ at a fixed minute). The model-driven version sets φ from the
in-play model's updated posterior vs the live price (Phase 2, see 00_IN_PLAY_RESEARCH_LOG.md).

INPUTS (built upstream): `L` = DCMH BacktestLedger (match_id,date,selection,odds,stake,pnl,is_winner,…);
`bf_ip` = in-play Betfair ticks (match_id,selection,minutes_to_kickoff,traded_price). See r00_explore.jl.
=#

using DataFrames, Statistics, Dates

const COMM = 0.05   # exchange commission on net winnings

# ---- live cash-out price near wall-clock minute T (minutes_to_kickoff) ----
function exit_price(mid, sel, t_wall; window = 6.0)
    sub = bf_ip[(bf_ip.match_id .== mid) .& (bf_ip.selection .== sel) .&
                (abs.(bf_ip.minutes_to_kickoff .- t_wall) .<= window), :]
    isempty(sub) && return missing
    sub.traded_price[argmin(abs.(sub.minutes_to_kickoff .- t_wall))]
end

held_pnl(r)        = r.is_winner ? r.odds - 1.0 : -1.0      # per unit stake, held to settlement
cashout_pnl(r, ox) = (l = r.odds/ox - 1.0; l > 0 ? l*(1-COMM) : l)   # per unit stake, locked by laying

"Per-unit-stake cash-out pnl for every ledger row at wall-clock minute T (missing if no live price)."
exit_col(L, T) = [ (ox = exit_price(r.match_id, r.selection, T); ismissing(ox) ? missing : cashout_pnl(r, ox))
                   for r in eachrow(L) ]

"""
    hedged_growth(L; cap, φ, ccol) -> log-growth

Sequential per-match compounding. Within a match, scale all simultaneous stakes so their sum ≤ `cap`
(portfolio Kelly), then settle each bet as a convex blend of cash-out (φ) and hold (1-φ); bets with no
live price are held. Returns -Inf if any match returns ≤ -100% (ruin).
"""
function hedged_growth(L; cap = 0.2, φ = 0.0, ccol = :c70)
    order = unique(L.match_id)
    bym   = Dict(k.match_id => v for (k, v) in pairs(groupby(L, :match_id)))
    rs = Float64[]
    for mid in order
        sub = bym[mid]; tot = sum(sub.stake); scale = tot > cap ? cap/tot : 1.0
        r = 0.0
        for x in eachrow(sub)
            c  = x[ccol]
            pu = ismissing(c) ? x.h : φ*c + (1-φ)*x.h        # partial cash-out is linear in φ
            r += x.stake * scale * pu
        end
        push!(rs, r)
    end
    any(rs .<= -1.0) ? -Inf : sum(log1p.(rs))
end

# ---------------------------------------------------------------- run ----
L2 = sort(L, :date)
L2.h   = [held_pnl(r) for r in eachrow(L2)]
L2.c70 = exit_col(L2, 70.0)
L2.c80 = exit_col(L2, 80.0)

φs   = 0.0:0.25:1.0
caps = 0.10:0.05:0.45

# 1. portfolio-cap parabola (hold-only) — shows the bankruptcy and the optimum
hold_scan = [(cap = c, growth = round(hedged_growth(L2; cap = c, φ = 0.0), digits = 3)) for c in caps]

# 2. joint cap × φ optimum at each exit minute
function joint_scan(ccol)
    [(cap = c,
      best_φ = φs[argmax([hedged_growth(L2; cap = c, φ = φ, ccol = ccol) for φ in φs])],
      hold   = round(hedged_growth(L2; cap = c, φ = 0.0,  ccol = ccol), digits = 3),
      best   = round(maximum([hedged_growth(L2; cap = c, φ = φ, ccol = ccol) for φ in φs]), digits = 3))
     for c in caps]
end

println("[hold-only portfolio-cap scan]"); show(DataFrame(hold_scan), allrows = true); println()
println("\n[joint cap × φ, exit @70']"); show(DataFrame(joint_scan(:c70)), allrows = true); println()
println("\n[joint cap × φ, exit @80']"); show(DataFrame(joint_scan(:c80)), allrows = true); println()
