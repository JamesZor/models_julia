#=
RUNNER — two follow-up experiments on the core-11 book (src_sup40_sw40, Ireland 2025-26 OOS):

A. LOW-TRUST COLD START (w0=0.3).  Start every unit distrusting the model (w=0.3, "defer to
   market") and watch which units the EB fit RAISES over the season — i.e. which markets earn
   trust. NB the EB estimate is empirical-Bayes (pooled from the data), so the learned w is
   cold-start-invariant: the 0.3 start only reframes the baseline + drives matches 1-25 staking.
   A unit whose w climbs above 0.3 is one the model has real edge on; one that falls has none.

B. CAP SWEEP.  Sweep the unified portfolio cap Σa ≤ cap and tabulate terminal wealth, growth
   G/match and max drawdown per strategy — the portfolio-Kelly risk/growth curve. Expectation
   ([[portfolio-kelly-partial-hedge]]): a low cap (~0.15-0.3) is the growth-optimal, ruin-safe
   region; raising the cap over-bets the model's bad markets and tips into drawdown/ruin.
   (FLAT_1pct is ~cap-invariant: ~7 bets × 1% never hits the cap.)

Reuses run_real_season from r01 (identical books/settlement). Assumes `built` (core c=0.02
ExtMatch-free SimMatch book) is in the session; else rebuild via build_real_books.
=#

using Statistics, Printf

include(joinpath(@__DIR__, "r01_race_src_sup40.jl"))

const CAP_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]

# ---------- A. low-trust cold start ----------

function init_experiment(built; w0=0.3, outdir=joinpath(@__DIR__, "results"))
    mkpath(outdir)
    rs = run_real_season(built; w0_start=w0)
    lines = String["LOW-TRUST COLD START — src_sup40_sw40 · core-11 book · w0=$w0 · n=$(rs.n)",
                   "Start all units at w=$w0 (distrust); EB fit then learns per-unit trust from data.",
                   "", "="^76,
                   @sprintf("%6s  %s", "match", join([@sprintf("%8s", u) for u in UNIT_NAMES], "")), "="^76]
    for (i, w) in rs.w_trace
        push!(lines, @sprintf("%6d  %s", i, join([@sprintf("%8.3f", x) for x in w], "")))
    end
    push!(lines, "")
    push!(lines, @sprintf("%-10s %8s %8s %10s", "unit", "start", "final", "Δ (earned)"))
    order = sortperm([rs.w_final[u]-w0 for u in 1:7], rev=true)
    for u in order
        d = rs.w_final[u] - w0
        push!(lines, @sprintf("%-10s %8.3f %8.3f %+10.3f %s", UNIT_NAMES[u], w0, rs.w_final[u], d,
                              d > 0.03 ? "↑ earns trust" : d < -0.03 ? "↓ loses trust" : "· flat"))
    end
    body = join(lines, "\n")
    write(joinpath(outdir, "e_init_w03.txt"), body)
    # csv for the plot
    io = IOBuffer(); println(io, "match," * join(UNIT_NAMES, ","))
    for (i, w) in rs.w_trace; println(io, string(i) * "," * join(string.(round.(w, digits=5)), ",")); end
    write(joinpath(outdir, "w_trace_init_w03.csv"), String(take!(io)))
    return (rs=rs, body=body)
end

# ---------- B. cap sweep ----------

"""
    cap_sweep(built; caps=CAP_GRID) -> (rows, body)

Run the core race at each cap; collect (cap, strategy, terminal_W, G/match, maxDD, ruined).
"""
function cap_sweep(built; caps=CAP_GRID, outdir=joinpath(@__DIR__, "results"))
    mkpath(outdir)
    rows = NamedTuple[]
    for cap in caps
        rs = run_real_season(built; cap=cap)
        for s in REAL_STRATEGIES
            sm = summarize_logw(rs.logw[s])
            push!(rows, (cap=cap, strat=s, termW=sm.terminal_W, G=sm.G_per_match,
                         maxDD=sm.max_dd, ruined=rs.ruined[s]))
        end
    end
    # formatted per-strategy blocks
    lines = String["CAP SWEEP — src_sup40_sw40 · core-11 book · c=0.02",
                   "portfolio cap Σa ≤ cap.  metrics: terminal W, growth G/match, max drawdown.",
                   "FLAT_1pct is ~cap-invariant (≈7 bets × 1% never hits the cap)."]
    for s in REAL_STRATEGIES
        push!(lines, "", "="^58, s, "="^58)
        push!(lines, @sprintf("%6s %10s %12s %8s %8s", "cap", "term_W", "G/match", "maxDD", "ruined"))
        for r in filter(x -> x.strat == s, rows)
            push!(lines, @sprintf("%6.2f %10.4f %+12.5f %8.3f %8s", r.cap, r.termW, r.G, r.maxDD,
                                  r.ruined ? "YES" : "-"))
        end
    end
    body = join(lines, "\n")
    write(joinpath(outdir, "e_cap_sweep.txt"), body)
    # csv
    io = IOBuffer(); println(io, "cap,strategy,terminal_W,G_per_match,maxDD,ruined")
    for r in rows
        println(io, @sprintf("%.2f,%s,%.6f,%.6f,%.6f,%d", r.cap, r.strat, r.termW, r.G, r.maxDD, r.ruined ? 1 : 0))
    end
    write(joinpath(outdir, "cap_sweep.csv"), String(take!(io)))
    return (rows=rows, body=body)
end
