#=
RUNNER — Experiment 4: does a favourite–longshot-bias (FLB) world reproduce the REAL
per-line signature, and which trust policy survives it?

Motivation (user, from split_market_pillar/r10 real backtest on Ireland): home/away 1X2
lines show NEGATIVE growth/ROI across nearly all model configs, while btts_yes (and some
totals cells) are positive. Hypothesis: the market shades longshot odds (dominant-team
league ⇒ pronounced FLB) — an ODDS-QUALITY defect the sim's flat multiplicative vig
cannot produce, and one that E1–E3's per-line-homogeneous info edge never stressed.

Two central predictions to test:
  (a) ρ_flb < 1 turns 1X2 negative for the model's bets (adverse selection: model noise
      finds "value" mostly on shaded longshots) while totals/BTTS lines — probs nearer
      0.5, shading ≈ neutral — keep their edge. If so, the real r10 signature is
      reproduced by ONE dial.
  (b) The Bernoulli/calibration EB trust fit is BLIND to FLB: shading changes the odds,
      not (much) the market's probability accuracy, so learned w stays ≈0.5 and the
      trust blend keeps betting the poisoned lines. Only line-level curation (w=0 on
      1X2 ⇒ vig-moat abstention) or a growth-based fit can defend.

Parts:
  flb_diag()  — 40k-match per-selection staked-EV/growth table over a ρ grid, plus
                oracle trust and the EB fit's w on a large FLB-world history (test (b)).
  run_e4()    — 300-season race, worlds "flb" (ρ chosen from diag, devig_quotes=true)
                and "base" (E1/E3 good world), strategies:
                FLAT_1pct · U_cap02 (raw) · TRUST05 (flat w=.5) · TRUST (EB) ·
                CURATED05 (w=0 on 1X2, .5 on totals/BTTS — soft market curation).
                Same seeds as E1/E3 ⇒ base world comparable. Chunked with progress
                printlns (keeps kaimon's 10-min gate alive — E3 lesson).

Server: include l01, l02 (updated: ρ_flb/devig_quotes dials, CURATED05 strategy),
then this file; flb_diag() first, then run_e4(ρ=...).
=#

using Random
using Statistics
using Printf
using Serialization

if !@isdefined(SimConfig)
    include(joinpath(@__DIR__, "l01_sim_market_model.jl"))
end
if !@isdefined(fit_trust_eb)
    include(joinpath(@__DIR__, "l02_strategies.jl"))
end

const R5_RESULTS = joinpath(@__DIR__, "results"); mkpath(R5_RESULTS)

"""
Streamed oracle trust: like l02's `oracle_trust` but over FRESH cfg.n_matches campaigns
instead of one long run — the single-campaign version drifts unboundedly (σ_in·√rounds ≈ 1.3
at 40k matches) and a lucky/unlucky seed flips the whole answer (seed 12 → all-zero w).
"""
function oracle_trust_stream(cfg::SimConfig; n=40_000, seed=1, wgrid=collect(0.0:0.02:1.0))
    rng = Xoshiro(seed)
    p = [Float64[] for _ in 1:7]; q = [Float64[] for _ in 1:7]
    pt = [Float64[] for _ in 1:7]; dd = [Float64[] for _ in 1:7]
    k = 0
    while k < n
        for sm in simulate_campaign(cfg, rng; n_matches=cfg.n_matches, S=16)
            ps = MMASK' * sm.pbar
            for u in 1:7
                m = UNIT_REP_SEL[u]
                push!(p[u], ps[m]); push!(q[u], sm.q_mkt[m])
                push!(pt[u], sm.p_true[m]); push!(dd[u], sm.d[m])
            end
        end
        k += cfg.n_matches
    end
    worac = zeros(7)
    for u in 1:7
        best, bw = -Inf, 0.0
        for w in wgrid
            g = 0.0
            @inbounds for i in eachindex(pt[u])
                b = dd[u][i] - 1.0
                p̃ = w * p[u][i] + (1.0 - w) * q[u][i]
                f = min(max(0.0, (p̃ * dd[u][i] - 1.0) / b), 0.98)
                g += pt[u][i] * log1p(b * f) + (1.0 - pt[u][i]) * log1p(-f)
            end
            g /= length(pt[u])
            g > best && ((best, bw) = (g, w))
        end
        worac[u] = bw
    end
    return worac
end
const E4_STRATS = ["FLAT_1pct", "U_cap02", "TRUST05_U_cap02", "TRUST_U_cap02",
                   "CURATED05_U_cap02"]

"""
Per-selection plug-in-Kelly diagnostics (min_edge 3%, half-cap like the real filter) on
`n` fresh matches per ρ: bets, mean quoted odds, mean per-bet true EV, growth contribution.
Then, for each ρ: oracle trust w (growth) and the EB calibration fit's w on a 5k-match
history — prediction (b) says the latter won't move even when the former collapses on 1X2.
"""
function flb_diag(; ρgrid=[1.0, 0.95, 0.90, 0.85], n=40_000, n_eb=5_000, seed=11)
    out = Dict{Float64,Any}()
    lines = String[]
    for ρ in ρgrid
        cfg = SimConfig(ρ_flb=ρ, devig_quotes=true)
        rng = Xoshiro(seed)
        nb = zeros(Int, 11); ev = zeros(11); gl = zeros(11); od = zeros(11)
        for _ in 1:(n ÷ 1000)
            for sm in simulate_campaign(cfg, rng; n_matches=1000, S=16)
                ps = MMASK' * sm.pbar
                for m in 1:11
                    e = ps[m] - 1.0 / sm.d[m]
                    e < 0.03 && continue
                    b = sm.d[m] - 1.0
                    f = min((ps[m] * sm.d[m] - 1.0) / b, 0.5)
                    nb[m] += 1; od[m] += sm.d[m]
                    ev[m] += sm.p_true[m] * sm.d[m] - 1.0
                    gl[m] += sm.p_true[m] * log1p(b * f) + (1.0 - sm.p_true[m]) * log1p(-f)
                end
            end
        end
        push!(lines, "ρ_flb = $ρ  ($n matches, min_edge 3%)")
        push!(lines, "  sel          bets  avg_d   EV/bet    G/match")
        for m in 1:11
            nb[m] == 0 && continue
            push!(lines, @sprintf("  %-11s %5d  %5.2f  %+8.4f  %+9.6f",
                                  SEL_NAMES[m], nb[m], od[m] / nb[m], ev[m] / nb[m], gl[m] / n))
        end
        # (b) what each fitting objective sees — both streamed over fresh campaigns
        worac = oracle_trust_stream(cfg; n=n, seed=seed + 1)
        rng2 = Xoshiro(seed + 2)
        hist = TrustHist(); k_eb = 0
        while k_eb < n_eb
            for sm in simulate_campaign(cfg, rng2; n_matches=cfg.n_matches, S=16)
                push_hist!(hist, sm, MMASK' * sm.pbar)
            end
            k_eb += cfg.n_matches
        end
        w_eb, hyp = fit_trust_eb(hist)
        push!(lines, "  oracle w (growth):      " * join(round.(worac, digits=2), " "))
        push!(lines, "  EB w (calib, $(n_eb)m):  " * join(round.(w_eb, digits=2), " ") *
                     "   (w0=$(round(hyp.w0, digits=2)), τ=$(hyp.τ))")
        push!(lines, "")
        out[ρ] = (nb=nb, ev=ev, gl=gl, worac=worac, w_eb=w_eb)
    end
    txt = join(lines, "\n")
    write(joinpath(R5_RESULTS, "e4_diag.txt"), txt)
    serialize(joinpath(R5_RESULTS, "e4_diag.jls"), out)
    println(txt)
    return out
end

function run_e4(; N=300, ρ=0.90, base_seed=20260704, chunk=25)
    res = Dict{String,Any}()
    for (wn, cfg) in (("flb", SimConfig(ρ_flb=ρ, devig_quotes=true)), ("base", SimConfig()))
        acc = Vector{Any}(undef, N)
        for lo in 1:chunk:N
            hi = min(lo + chunk - 1, N)
            Threads.@threads :dynamic for i in lo:hi
                r = run_season(cfg, base_seed + i; strategies=E4_STRATS)
                acc[i] = (results=Dict(s => (logw=v.logw, ruined=v.ruined)
                                       for (s, v) in r.results),
                          w_final=r.w_final)
            end
            println("[e4:$wn] $hi/$N seasons done"); flush(stdout)
        end
        res[wn] = acc
        serialize(joinpath(R5_RESULTS, "e4_partial.jls"), res)
    end
    serialize(joinpath(R5_RESULTS, "e4.jls"), res)
    return summarize_e4(res)
end

function summarize_e4(res=deserialize(joinpath(R5_RESULTS, "e4.jls")))
    lines = String[]
    push!(lines, "world,strategy,medW,q05W,q95W,meanG,medDD,ruin_pct")
    for wn in ("flb", "base")
        acc = res[wn]
        for s in E4_STRATS
            sums = [summarize_logw(a.results[s].logw) for a in acc]
            tw = [x.terminal_W for x in sums]
            push!(lines, join([wn, s,
                round(median(tw), digits=3),
                round(quantile(tw, 0.05), digits=3),
                round(quantile(tw, 0.95), digits=3),
                round(mean(x.G_per_match for x in sums), digits=5),
                round(median(x.max_dd for x in sums), digits=3),
                round(100 * mean(a.results[s].ruined for a in acc), digits=1)], ","))
        end
        wf = reduce(hcat, [a.w_final for a in acc])   # 7 × N end-of-season EB w
        push!(lines, "$wn,w_final_median," *
                     join(round.([median(wf[u, :]) for u in 1:7], digits=3), " "))
    end
    txt = join(lines, "\n")
    write(joinpath(R5_RESULTS, "e4_summary.txt"), txt)
    return txt
end
