#=
RUNNER — REAL-data staking race on the `src_sup40_sw40` smile engine (Ireland 2025-26 OOS).

Backtests the sim-validated staking layer on real Betfair-close books and answers the MVP
question: does the EB trust fit pull w DOWN on the markets the model is bad at (home/away 1X2)
while HOLDING the good ones (unders, BTTS)? 275 matches — this is signature-reading, not
ranking.

STRATEGY REGISTRY (pluggable — add a new staking system = one `elseif` in `stake_for` + one
name in REAL_STRATEGIES; everything else — books, settlement, attribution — is shared):
  FLAT_1pct          flat 1% on every model-edge ≥ 3% selection (edge on the SMILE model prob)
  PB_BK_cap02        per-bet Bayesian–McHale Kelly (0.03) on smile per-sel draws, Σa ≤ 0.2.
                     The b21-comparable baseline: same per-line Kelly the r21 backtest used,
                     just portfolio-capped (memory: uncapped per-bet Kelly bankrupts).
  U_cap02            raw unified (P), cap 0.2, w=1 — grid tilted to the SMILE model probs
                     (so O/U is priced by Λ=λ_tot·φ, exactly as b21 certified).
  TRUST05_U_cap02    unified with a FLAT w=0.5 blend toward the market on every unit.
  CURATED05_U_cap02  unified with the sim-E4 curation: w=0 on 1X2 (abstain — vig moat on the
                     model's bad markets), w=0.5 on totals+BTTS.
  TRUST_EB_U_cap02   unified with the EB-LEARNED per-unit w — cold-start 0.5, refit every 25.
                     This is the system under test: it should DISCOVER the curation.

All strategies bet from match 1, share identical books, compound sequentially; ruin-freeze at
wealth < 0.01. Commission c folds into payout for BOTH decisions and settlement; the race is
run twice (c=0.02 and c=0).

Run after: git pull + include the l01s/l02/this file. Assumes `lat`, `ppd`, `odds_bf`,
`ds1` already live in the session (built by the preflight). If not, rebuild them (see §0).
=#

using Random
using Statistics
using Printf
using LinearAlgebra: dot

include(joinpath(@__DIR__, "..", "staking_sim", "l01_sim_market_model.jl"))
include(joinpath(@__DIR__, "..", "staking_sim", "l02_strategies.jl"))
include(joinpath(@__DIR__, "l01_real_books.jl"))

const REAL_STRATEGIES = ["FLAT_1pct", "PB_BK_cap02", "U_cap02",
                         "TRUST05_U_cap02", "CURATED05_U_cap02", "TRUST_EB_U_cap02"]

# b21 per-selection ROI% (src_sup40_sw40, BayesianKelly, Betfair) — adapter cross-check target.
const B21_ROI = Dict("home"=>-9.4, "draw"=>33.54, "away"=>22.69,
                     "over_15"=>-7.5, "under_15"=>42.02, "over_25"=>1.75, "under_25"=>12.21,
                     "over_35"=>16.03, "under_35"=>9.0, "btts_yes"=>32.6, "btts_no"=>48.22)

const W_HALF = fill(0.5, 7)
const W_CUR  = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]
const W_ONE  = ones(7)

"""
    run_real_season(built; cap=0.2, refit_every=25, min_edge=0.03, w0_start=0.5)

Sequential single-pass race over the real matches (already kickoff-ordered). Returns a
NamedTuple with per-strategy logw/turnover/ruin, per-family & per-selection P/L attribution,
the EB w-trajectory + pooled w0 path, and the max w=1 smile-tilt reproduction error.
"""
function run_real_season(built; cap=0.2, refit_every=25, min_edge=0.03, w0_start=0.5,
                         strategies=REAL_STRATEGIES)
    matches   = built.matches
    smile_sel = built.smile_sel
    smile_dists = built.smile_dists
    n = length(matches)

    hist = TrustHist()
    w_eb = fill(w0_start, 7)

    logw   = Dict(s => Float64[] for s in strategies)
    nbets  = Dict(s => 0   for s in strategies)
    turn   = Dict(s => 0.0 for s in strategies)
    ruined = Dict(s => false for s in strategies)
    cumW   = Dict(s => 1.0 for s in strategies)

    fam_profit = Dict((s, f) => 0.0 for s in strategies, f in 1:3)
    fam_turn   = Dict((s, f) => 0.0 for s in strategies, f in 1:3)
    fam_nbets  = Dict((s, f) => 0   for s in strategies, f in 1:3)

    sel_profit = zeros(11); sel_turn = zeros(11); sel_nbets = zeros(Int, 11); sel_wins = zeros(Int, 11)

    w_trace  = Tuple{Int,Vector{Float64}}[(1, copy(w_eb))]
    w0_trace = Tuple{Int,Float64}[]
    max_tilt_err = 0.0

    for (i, sm) in enumerate(matches)
        pbar_sel = smile_sel[i]
        sdists   = smile_dists[i]

        if i > 1 && (i - 1) % refit_every == 0
            w_eb, hp = fit_trust_eb(hist)
            push!(w_trace, (i, copy(w_eb)))
            push!(w0_trace, (i, hp.w0))
        end

        mult_eb  = coherent_multiplier(sm.pbar, blend_targets(pbar_sel, sm.q_mkt, w_eb))
        mult_05  = coherent_multiplier(sm.pbar, blend_targets(pbar_sel, sm.q_mkt, W_HALF))
        mult_cur = coherent_multiplier(sm.pbar, blend_targets(pbar_sel, sm.q_mkt, W_CUR))
        mult_one = coherent_multiplier(sm.pbar, blend_targets(pbar_sel, sm.q_mkt, W_ONE))

        # verification 3: w=1 grid tilt reproduces the smile over-probs (units 4/6/8)
        gtilt = normalize_mult(sm.pbar, mult_one)
        for m in (4, 6, 8)
            max_tilt_err = max(max_tilt_err, abs(dot(Float64.(SEL_MASKS[m]), gtilt) - pbar_sel[m]))
        end

        for s in strategies
            if ruined[s]
                push!(logw[s], 0.0); continue
            end
            a = if s == "FLAT_1pct"
                    stakes_flat(sm, pbar_sel; f=0.01, min_edge=min_edge)
                elseif s == "PB_BK_cap02"
                    guard!(stakes_signal(sm, sdists, SIG.BayesianKelly(0.03)); cap=cap)
                elseif s == "U_cap02"
                    stakes_unified(sm; cap=cap, p=normalize_mult(sm.pbar, mult_one))
                elseif s == "TRUST05_U_cap02"
                    stakes_unified(sm; cap=cap, p=normalize_mult(sm.pbar, mult_05))
                elseif s == "CURATED05_U_cap02"
                    stakes_unified(sm; cap=cap, p=normalize_mult(sm.pbar, mult_cur))
                elseif s == "TRUST_EB_U_cap02"
                    stakes_unified(sm; cap=cap, p=normalize_mult(sm.pbar, mult_eb))
                else
                    error("unknown strategy $s")
                end
            r = max(match_return(a, sm), 1e-12)
            push!(logw[s], log(r))
            cumW[s] *= r
            nbets[s] += count(>(1e-8), a)
            turn[s]  += sum(a)
            for m in 1:11
                a[m] <= 1e-8 && continue
                f = FAM_OF_SEL[m]
                rr = sm.won[m] ? (sm.d[m] - 1.0) : -1.0
                fam_profit[(s, f)] += a[m] * rr
                fam_turn[(s, f)]   += a[m]
                fam_nbets[(s, f)]  += 1
                if s == "PB_BK_cap02"
                    sel_profit[m] += a[m] * rr; sel_turn[m] += a[m]
                    sel_nbets[m]  += 1;         sel_wins[m] += sm.won[m]
                end
            end
            cumW[s] < 0.01 && (ruined[s] = true)
        end

        push_hist!(hist, sm, pbar_sel)   # settled AFTER betting — no leakage
    end

    return (; strategies, logw, nbets, turn, ruined, cumW,
            fam_profit, fam_turn, fam_nbets,
            sel_profit, sel_turn, sel_nbets, sel_wins,
            w_trace, w0_trace, w_final=w_eb, max_tilt_err, n)
end

# ---------- reporting ----------

"Summary-table rows (one per strategy): terminal W, G/match ± SE, maxDD, n_bets, turnover."
function summary_rows(rs)
    rows = String[]
    push!(rows, @sprintf("%-18s %10s %12s %8s %7s %9s %8s", "strategy", "term_W", "G/match±SE", "maxDD", "n_bets", "turnover", "ruined"))
    for s in rs.strategies
        lw = rs.logw[s]
        sm = summarize_logw(lw)
        se = std(lw) / sqrt(length(lw))
        push!(rows, @sprintf("%-18s %10.4f  %+7.5f±%.5f %7.3f %7d %9.2f %8s",
              s, sm.terminal_W, sm.G_per_match, se, sm.max_dd, rs.nbets[s], rs.turn[s],
              rs.ruined[s] ? "YES" : "-"))
    end
    return rows
end

"Per-family (1X2 / totals / BTTS) net-staked P/L attribution per strategy (linear, non-compounded)."
function family_rows(rs)
    fam_name = ("1X2", "totals", "BTTS")
    rows = String[]
    push!(rows, @sprintf("%-18s %-8s %9s %9s %8s %8s", "strategy", "family", "profit", "turnover", "roi%", "n_bets"))
    for s in rs.strategies
        for f in 1:3
            t = rs.fam_turn[(s, f)]; p = rs.fam_profit[(s, f)]
            roi = t > 0 ? 100p / t : 0.0
            push!(rows, @sprintf("%-18s %-8s %+9.4f %9.4f %+8.2f %8d", s, fam_name[f], p, t, roi, rs.fam_nbets[(s, f)]))
        end
    end
    return rows
end

"b21 adapter cross-check: PB_BK_cap02 per-selection ROI% vs the b21 src_sup40_sw40 rows."
function crosscheck_rows(rs)
    rows = String[]
    push!(rows, @sprintf("%-10s %9s %9s %7s %8s %6s", "selection", "PB_roi%", "b21_roi%", "sign", "PB_n", "wins"))
    agree = 0; tot = 0
    for m in 1:11
        nm = SEL_NAMES[m]
        roi = rs.sel_turn[m] > 0 ? 100 * rs.sel_profit[m] / rs.sel_turn[m] : 0.0
        b = get(B21_ROI, nm, NaN)
        ok = !isnan(b) && sign(roi) == sign(b)
        tot += 1; agree += ok
        push!(rows, @sprintf("%-10s %+9.2f %+9.2f %7s %8d %6d", nm, roi, b, ok ? "OK" : "x", rs.sel_nbets[m], rs.sel_wins[m]))
    end
    push!(rows, @sprintf("sign agreement: %d/%d", agree, tot))
    return rows
end

"EB w-trajectory table: rows = refit points, cols = the 7 units."
function wtrace_rows(rs)
    rows = String[]
    push!(rows, @sprintf("%6s  %s", "match", join([@sprintf("%8s", u) for u in UNIT_NAMES], "")))
    for (i, w) in rs.w_trace
        push!(rows, @sprintf("%6d  %s", i, join([@sprintf("%8.3f", x) for x in w], "")))
    end
    push!(rows, "")
    push!(rows, "pooled w0 (logit → logistic):")
    for (i, w0) in rs.w0_trace
        push!(rows, @sprintf("  match %4d   w0=%+.3f  → w̄=%.3f", i, w0, 1 / (1 + exp(-w0))))
    end
    push!(rows, @sprintf("FINAL per-unit w: %s", join([@sprintf("%s=%.3f", UNIT_NAMES[u], rs.w_final[u]) for u in 1:7], "  ")))
    return rows
end

# ---------- driver ----------

"Full race for one commission level; writes results/*.txt|csv and returns the run struct."
function run_and_report(built, c::Float64; outdir=joinpath(@__DIR__, "results"))
    mkpath(outdir)
    tag = @sprintf("c%03d", round(Int, c * 1000))
    rs = run_real_season(built)

    hdr(s) = ["", "="^78, s, "="^78]
    lines = String[]
    append!(lines, ["REAL-DATA STAKING RACE — src_sup40_sw40 · Ireland 2025-26 · n=$(rs.n) matches",
                    "commission c = $c   (folded into payout for decisions AND settlement)",
                    "smile w=1 tilt reproduction: max|Δ over-prob| = $(rs.max_tilt_err)"])
    append!(lines, hdr("SUMMARY TABLE")); append!(lines, summary_rows(rs))
    append!(lines, hdr("PER-FAMILY P/L ATTRIBUTION")); append!(lines, family_rows(rs))
    append!(lines, hdr("EB TRUST w-TRAJECTORY (the money shot)")); append!(lines, wtrace_rows(rs))
    append!(lines, hdr("b21 ADAPTER CROSS-CHECK (PB_BK_cap02 vs b21 ROI signs)")); append!(lines, crosscheck_rows(rs))

    body = join(lines, "\n")
    write(joinpath(outdir, "e_real_summary_$(tag).txt"), body)

    # machine-readable w-trajectory CSV (for the plot)
    io = IOBuffer()
    println(io, "match," * join(UNIT_NAMES, ","))
    for (i, w) in rs.w_trace
        println(io, string(i) * "," * join(string.(round.(w, digits=5)), ","))
    end
    write(joinpath(outdir, "w_trace_$(tag).csv"), String(take!(io)))

    return (rs=rs, body=body, tag=tag)
end
