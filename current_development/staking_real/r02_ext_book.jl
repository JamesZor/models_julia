#=
RUNNER — EXTENDED-book staking race (v2): the unified Kelly layer over EVERY Betfair market we
can price off the score grid, not just the core 11. On Ireland 2025-26 that adds CorrectScore
(19 cells/buckets) and the full O/U ladder (0.5/4.5/5.5) to 1X2 + O/U 1.5/2.5/3.5 + BTTS;
DoubleChance/DrawNoBet/AsianHandicap auto-join on any league that carries them.

Same policy as r01: per-line trust blend → coherent IPF grid tilt (now over the full O/U ladder)
→ capped unified Kelly (P). CorrectScore/DC/DNB/AH inherit trust through the grid. Settlement is
the generalized net-return-per-state `sel_payoff` (push + AH half-win aware). The EB trust fit +
its 7-unit alarm are unchanged (same independent directions), so the w-trajectory matches r01 —
the NEW reads are the extended race W and the per-family P/L that now includes CorrectScore.

Strategies (the b21 per-bet baseline lives in r01; here we race the unified layer on the big book):
  FLAT_1pct · U_cap02 (w=1) · TRUST05_U_cap02 · CURATED05_U_cap02 · TRUST_EB_U_cap02.

Assumes `lat`, `ppd`, `odds_bf`, `matches_df` live in the session (r01 preflight/cache).
=#

using Random, Statistics, Printf
using LinearAlgebra: dot

include(joinpath(@__DIR__, "..", "staking_sim", "l01_sim_market_model.jl"))
include(joinpath(@__DIR__, "..", "staking_sim", "l02_strategies.jl"))
include(joinpath(@__DIR__, "l01_real_books.jl"))
include(joinpath(@__DIR__, "l02_real_ext_book.jl"))

const EXT_STRATEGIES = ["FLAT_1pct", "U_cap02", "TRUST05_U_cap02", "CURATED05_U_cap02", "TRUST_EB_U_cap02"]

"Flat 1% on every selection whose model EV (net return per unit, w=1 grid) clears `min_edge`."
function ext_flat(em::ExtMatch, p_model; f=0.01, min_edge=0.03, cap=0.2)
    M = length(em.d); a = zeros(M)
    M == 0 && return a
    ev = em.R' * p_model
    for m in 1:M; ev[m] >= min_edge && (a[m] = f); end
    return guard!(a; cap=cap)
end

function run_ext_season(matches; cap=0.2, refit_every=25, min_edge=0.03, w0_start=0.5,
                        strategies=EXT_STRATEGIES)
    hist = TrustHist(); w_eb = fill(w0_start, 7)
    logw = Dict(s=>Float64[] for s in strategies); nbets = Dict(s=>0 for s in strategies)
    turn = Dict(s=>0.0 for s in strategies); ruined = Dict(s=>false for s in strategies)
    cumW = Dict(s=>1.0 for s in strategies)
    fam_profit = Dict((s,f)=>0.0 for s in strategies, f in 1:7)
    fam_turn   = Dict((s,f)=>0.0 for s in strategies, f in 1:7)
    fam_nbets  = Dict((s,f)=>0   for s in strategies, f in 1:7)
    w_trace = Tuple{Int,Vector{Float64}}[(1, copy(w_eb))]; w0_trace = Tuple{Int,Float64}[]
    book_sizes = Int[]

    for (i, em) in enumerate(matches)
        push!(book_sizes, length(em.d))
        if i > 1 && (i-1) % refit_every == 0
            w_eb, hp = fit_trust_eb(hist); push!(w_trace, (i, copy(w_eb))); push!(w0_trace, (i, hp.w0))
        end
        p_one = ext_tilted_pbar(em, W_ONE)
        p_05  = ext_tilted_pbar(em, W_HALF)
        p_cur = ext_tilted_pbar(em, W_CUR)
        p_eb  = ext_tilted_pbar(em, w_eb)

        for s in strategies
            if ruined[s]; push!(logw[s], 0.0); continue; end
            a = if s == "FLAT_1pct"          ext_flat(em, p_one; min_edge=min_edge, cap=cap)
                elseif s == "U_cap02"        (length(em.d)==0 ? Float64[] : solve_P(p_one, em.R; cap=cap))
                elseif s == "TRUST05_U_cap02"  (length(em.d)==0 ? Float64[] : solve_P(p_05, em.R; cap=cap))
                elseif s == "CURATED05_U_cap02"(length(em.d)==0 ? Float64[] : solve_P(p_cur, em.R; cap=cap))
                elseif s == "TRUST_EB_U_cap02" (length(em.d)==0 ? Float64[] : solve_P(p_eb, em.R; cap=cap))
                else error("unknown $s") end
            r = isempty(a) ? 1.0 : max(1.0 + dot(a, em.settle), 1e-12)
            push!(logw[s], log(r)); cumW[s] *= r
            nbets[s] += count(>(1e-8), a); turn[s] += isempty(a) ? 0.0 : sum(a)
            for m in eachindex(a)
                a[m] <= 1e-8 && continue
                f = em.fam[m]
                fam_profit[(s,f)] += a[m]*em.settle[m]; fam_turn[(s,f)] += a[m]; fam_nbets[(s,f)] += 1
            end
            cumW[s] < 0.01 && (ruined[s] = true)
        end
        push_hist_ext!(hist, em)
    end
    return (; strategies, logw, nbets, turn, ruined, cumW, fam_profit, fam_turn, fam_nbets,
            w_trace, w0_trace, w_final=w_eb, book_sizes, n=length(matches))
end

# ---------- reporting ----------

function ext_summary_rows(rs)
    rows = [@sprintf("%-18s %10s %14s %8s %7s %9s %8s", "strategy","term_W","G/match±SE","maxDD","n_bets","turnover","ruined")]
    for s in rs.strategies
        lw = rs.logw[s]; sm = summarize_logw(lw); se = std(lw)/sqrt(length(lw))
        push!(rows, @sprintf("%-18s %10.4f  %+8.5f±%.5f %7.3f %7d %9.2f %8s",
              s, sm.terminal_W, sm.G_per_match, se, sm.max_dd, rs.nbets[s], rs.turn[s], rs.ruined[s] ? "YES" : "-"))
    end
    rows
end

function ext_family_rows(rs)
    rows = [@sprintf("%-18s %-13s %9s %9s %8s %7s", "strategy","family","profit","turnover","roi%","n_bets")]
    for s in rs.strategies
        for f in 1:7
            rs.fam_nbets[(s,f)] == 0 && continue
            t = rs.fam_turn[(s,f)]; p = rs.fam_profit[(s,f)]; roi = t>0 ? 100p/t : 0.0
            push!(rows, @sprintf("%-18s %-13s %+9.4f %9.4f %+8.2f %7d", s, FAM_LABEL[f], p, t, roi, rs.fam_nbets[(s,f)]))
        end
    end
    rows
end

function ext_wtrace_rows(rs)
    rows = [@sprintf("%6s  %s", "match", join([@sprintf("%8s", u) for u in UNIT_NAMES], ""))]
    for (i, w) in rs.w_trace; push!(rows, @sprintf("%6d  %s", i, join([@sprintf("%8.3f", x) for x in w], ""))); end
    push!(rows, @sprintf("FINAL per-unit w: %s", join([@sprintf("%s=%.3f", UNIT_NAMES[u], rs.w_final[u]) for u in 1:7], "  ")))
    rows
end

function run_and_report_ext(matches, c::Float64; outdir=joinpath(@__DIR__, "results"))
    mkpath(outdir); tag = @sprintf("c%03d", round(Int, c*1000)); rs = run_ext_season(matches)
    hdr(s) = ["", "="^80, s, "="^80]
    bs = rs.book_sizes
    lines = String["EXTENDED-BOOK STAKING RACE — src_sup40_sw40 · Ireland 2025-26 · n=$(rs.n) matches",
                   "commission c = $c   ·   bettable selections/match: min $(minimum(bs)) / median $(Int(round(median(bs)))) / max $(maximum(bs))",
                   "book = 1X2 + O/U(0.5–5.5) + BTTS + CorrectScore (DC/DNB/AH auto-join where present)"]
    append!(lines, hdr("SUMMARY TABLE")); append!(lines, ext_summary_rows(rs))
    append!(lines, hdr("PER-FAMILY P/L ATTRIBUTION (now incl. CorrectScore)")); append!(lines, ext_family_rows(rs))
    append!(lines, hdr("EB TRUST w-TRAJECTORY (7 core units — unchanged from r01)")); append!(lines, ext_wtrace_rows(rs))
    body = join(lines, "\n"); write(joinpath(outdir, "e_ext_summary_$(tag).txt"), body)
    return (rs=rs, body=body, tag=tag)
end
