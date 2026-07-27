#=
RUNNER — Experiment 2: why is the learned trust w flat across lines (p4), and would a
time-decayed trust fit help?

E2a (identifiability ladder): fit the EB trust on growing samples (300 → 20k obs per line,
30 replications) and watch whether/when the lines separate; print the per-line Fisher
information I(w) = δ²/(p̃(1−p̃)) and the sample size it implies for sd(ŵ) = 0.1
(trust_blend_notes §3.1 arithmetic). If the lines separate at large n, p4's flatness is an
information problem, not a fit/sensitivity bug.

E2b (time decay): exponential down-weighting of old observations (half-life H, in
observations) in the trust likelihood, walk-forward refit every 30 matches, two worlds:
  STATIC — planted bias fixed (the main-race world). Decay can only shed sample here.
  DRIFT  — bias regime flips at match 1000 (γ_tot −0.05→+0.05, γ_btts +0.10→−0.10).
Score = Bernoulli log-score of the blended probability on the NEXT 30 matches (walk-forward,
what trust exists for), vs the market-only (w=0) baseline. Same seeds across worlds ⇒
pre-flip halves are identical draws (paired comparison).

Server (kaimon), after git pull:
    include("current_development/staking_sim/l01_sim_market_model.jl")
    include("current_development/staking_sim/l02_strategies.jl")   # picks up halflife kwarg
    include("current_development/staking_sim/r03_trust_experiments.jl")
    run_e2a(); run_e2b(); make_outputs_e2()
=#

using Random
using Statistics
using Serialization
using Plots
using StatsPlots

if !@isdefined(SimConfig)
    include(joinpath(@__DIR__, "l01_sim_market_model.jl"))
end
if !@isdefined(fit_trust_eb)
    include(joinpath(@__DIR__, "l02_strategies.jl"))
end

const R3_RESULTS = joinpath(@__DIR__, "results"); mkpath(R3_RESULTS)
const R3_PLOTS = joinpath(@__DIR__, "plots"); mkpath(R3_PLOTS)

const NLADDER = [300, 600, 1200, 2500, 5000, 10000, 20000]
const NLABELS = ["300", "600", "1.2k", "2.5k", "5k", "10k", "20k"]
const HLGRID = [Inf, 1000.0, 400.0, 150.0]
const HLNAMES = ["static", "H=1000", "H=400", "H=150"]
const CFG_DRIFT = SimConfig(γ_tot=+0.05, γ_btts=-0.10)   # sign-flipped bias regime

# ---------------- E2a: identifiability ----------------

"""
Per-line Fisher information of w at w = 1/2, Î_u = mean(δ²/(p̃(1−p̃))), and the sample size
n = 1/(0.1²·Î) needed for sd(ŵ) = 0.1 — the §3.1 arithmetic evaluated on this sim's books.
"""
function fisher_info_table(cfg::SimConfig=SimConfig(); n=40_000, seed=7)
    rng = Xoshiro(seed)
    I = zeros(7)
    k = 0
    while k < n
        for sm in simulate_campaign(cfg, rng; n_matches=min(cfg.n_matches, n - k), S=16)
            ps = MMASK' * sm.pbar
            for u in 1:7
                m = UNIT_REP_SEL[u]
                δ = ps[m] - sm.q_mkt[m]
                p̃ = 0.5 * (ps[m] + sm.q_mkt[m])
                I[u] += δ^2 / (p̃ * (1.0 - p̃))
            end
            k += 1
        end
    end
    I ./= n
    return (info=I, n_sd01=1.0 ./ (0.1^2 .* I))
end

"E2a: EB trust fit at each ladder point of a growing stream; W[rep, ladder, unit]."
function run_e2a(; R=30, base_seed=20260705)
    cfg = SimConfig()
    W = Array{Float64}(undef, R, length(NLADDER), 7)
    Threads.@threads :dynamic for r in 1:R
        rng = Xoshiro(base_seed + r)
        hist = TrustHist()
        k, li = 0, 1
        while li <= length(NLADDER)
            for sm in simulate_campaign(cfg, rng;
                                        n_matches=min(cfg.n_matches, NLADDER[end] - k), S=16)
                k += 1
                push_hist!(hist, sm, MMASK' * sm.pbar)
                if li <= length(NLADDER) && k == NLADDER[li]
                    W[r, li, :] .= fit_trust_eb(hist)[1]
                    li += 1
                end
            end
        end
    end
    serialize(joinpath(R3_RESULTS, "e2a_W.jls"), W)
    return W
end

# ---------------- E2b: time decay, static vs drifting bias ----------------

"Compact per-unit stream from chained fresh campaigns: n×7 (p_model, q_mkt, y)."
function stream_units(cfg::SimConfig, rng::AbstractRNG, n::Int)
    P = Matrix{Float64}(undef, n, 7); Q = similar(P); Y = similar(P)
    k = 0
    while k < n
        for sm in simulate_campaign(cfg, rng; n_matches=min(cfg.n_matches, n - k), S=16)
            k += 1
            ps = MMASK' * sm.pbar
            for u in 1:7
                m = UNIT_REP_SEL[u]
                P[k, u] = ps[m]; Q[k, u] = sm.q_mkt[m]; Y[k, u] = Float64(sm.won[m])
            end
        end
    end
    return P, Q, Y
end

"Mean Bernoulli log-score of the per-unit blend over a block of matches."
function block_score(w::Vector{Float64}, P, Q, Y, blk)
    s, c = 0.0, 0
    for i in blk, u in 1:7
        p̃ = clamp(w[u] * P[i, u] + (1.0 - w[u]) * Q[i, u], 1e-9, 1 - 1e-9)
        s += Y[i, u] * log(p̃) + (1.0 - Y[i, u]) * log1p(-p̃)
        c += 1
    end
    return s / c
end

function run_e2b(; R=30, n_pre=1000, n_post=1000, warm=300, refit_every=30,
                 base_seed=20260710)
    n_tot = n_pre + n_post
    refits = collect(warm:refit_every:(n_tot - refit_every))
    out = Dict{String,Any}("refits" => refits, "n_pre" => n_pre)
    for (world, cfgB) in (("static", SimConfig()), ("drift", CFG_DRIFT))
        Wtraj = Array{Float64}(undef, R, length(refits), length(HLGRID), 7)
        LS = Array{Float64}(undef, R, length(refits), length(HLGRID))
        LS0 = Array{Float64}(undef, R, length(refits))
        Threads.@threads :dynamic for r in 1:R
            rng = Xoshiro(base_seed + r)             # same seed both worlds ⇒ paired
            Pa, Qa, Ya = stream_units(SimConfig(), rng, n_pre)
            Pb, Qb, Yb = stream_units(cfgB, rng, n_post)
            P = vcat(Pa, Pb); Q = vcat(Qa, Qb); Y = vcat(Ya, Yb)
            hist = TrustHist()
            hk = 0
            for (ri, kf) in enumerate(refits)
                while hk < kf
                    hk += 1
                    for u in 1:7
                        push!(hist.p[u], P[hk, u])
                        push!(hist.q[u], Q[hk, u])
                        push!(hist.y[u], Y[hk, u])
                    end
                end
                blk = (kf + 1):min(kf + refit_every, n_tot)
                LS0[r, ri] = block_score(zeros(7), P, Q, Y, blk)
                for (hi, H) in enumerate(HLGRID)
                    w = fit_trust_eb(hist; halflife=H)[1]
                    Wtraj[r, ri, hi, :] .= w
                    LS[r, ri, hi] = block_score(w, P, Q, Y, blk)
                end
            end
        end
        out["W_$world"] = Wtraj
        out["LS_$world"] = LS
        out["LS0_$world"] = LS0
    end
    serialize(joinpath(R3_RESULTS, "e2b.jls"), out)
    return out
end

# ---------------- outputs ----------------

function make_outputs_e2()
    W = deserialize(joinpath(R3_RESULTS, "e2a_W.jls"))
    e2b = deserialize(joinpath(R3_RESULTS, "e2b.jls"))
    fi = fisher_info_table()
    worac = deserialize(joinpath(R3_RESULTS, "worac.jls"))
    worac_drift_file = joinpath(R3_RESULTS, "worac_drift.jls")
    worac_drift = isfile(worac_drift_file) ? deserialize(worac_drift_file) :
                  oracle_trust(CFG_DRIFT)
    serialize(worac_drift_file, worac_drift)

    # p5: identifiability ladder — median ŵ ± 10–90% band vs n, growth-oracle dashed red
    panels = []
    for u in 1:7
        med = [median(W[:, li, u]) for li in eachindex(NLADDER)]
        lo = [quantile(W[:, li, u], 0.10) for li in eachindex(NLADDER)]
        hi = [quantile(W[:, li, u], 0.90) for li in eachindex(NLADDER)]
        p = plot(NLADDER, med; xscale=:log10, ribbon=(med .- lo, hi .- med),
                 xticks=(NLADDER, NLABELS), title=UNIT_NAMES[u], legend=false,
                 ylim=(0, 1), lw=2)
        hline!(p, [worac[u]]; color=:red, ls=:dash)
        push!(panels, p)
    end
    push!(panels, plot(; framestyle=:none))
    p5 = plot(panels...; layout=(2, 4), size=(1500, 700), left_margin=5Plots.mm,
              bottom_margin=5Plots.mm)
    savefig(p5, joinpath(R3_PLOTS, "p5_ident_ladder.png"))

    # p6: DRIFT world — mean fitted w trajectory per half-life, three units
    refits = e2b["refits"]; n_pre = e2b["n_pre"]
    Wd = e2b["W_drift"]
    panels = []
    for (u, nm) in ((1, "home (bias-free)"), (5, "over_25"), (7, "btts_yes"))
        p = plot(; title=nm, ylim=(0, 1), xlabel="match",
                 legend=(u == 1 ? :bottomleft : false))
        for hi in eachindex(HLGRID)
            tr = [mean(Wd[:, ri, hi, u]) for ri in eachindex(refits)]
            plot!(p, refits, tr; label=HLNAMES[hi], lw=2)
        end
        vline!(p, [n_pre]; color=:black, ls=:dot, label=false)
        hline!(p, [worac[u]]; color=:red, ls=:dash, label=false)
        hline!(p, [worac_drift[u]]; color=:purple, ls=:dash, label=false)
        push!(panels, p)
    end
    p6 = plot(panels...; layout=(1, 3), size=(1400, 420), left_margin=5Plots.mm,
              bottom_margin=5Plots.mm)
    savefig(p6, joinpath(R3_PLOTS, "p6_drift_w.png"))

    # p7: walk-forward blend log-score minus market baseline, ×1000, pre/post flip
    panels = []
    for world in ("static", "drift")
        LS = e2b["LS_$world"]; LS0 = e2b["LS0_$world"]
        pre = findall(<(n_pre), refits); post = findall(>=(n_pre), refits)
        prevals = [1000 * mean(LS[:, pre, hi] .- LS0[:, pre]) for hi in eachindex(HLGRID)]
        postvals = [1000 * mean(LS[:, post, hi] .- LS0[:, post]) for hi in eachindex(HLGRID)]
        p = groupedbar([prevals postvals]; bar_position=:dodge,
                       xticks=(1:length(HLGRID), HLNAMES),
                       label=["m300–1000" "m1000–2000"],
                       title=world == "static" ? "STATIC bias" : "DRIFT (flip at m1000)",
                       ylabel="blend − market log-score ×1000")
        hline!(p, [0.0]; color=:black, ls=:dot, label=false)
        push!(panels, p)
    end
    p7 = plot(panels...; layout=(1, 2), size=(1200, 450), left_margin=5Plots.mm,
              bottom_margin=5Plots.mm)
    savefig(p7, joinpath(R3_PLOTS, "p7_decay_score.png"))

    # summary txt (paste into experiments.md / report)
    open(joinpath(R3_RESULTS, "e2_summary.txt"), "w") do io
        println(io, "unit,fisher_info,n_for_sd01,w_median_n600,w_median_n20k,w_oracle,w_oracle_drift")
        li600 = findfirst(==(600), NLADDER); li20k = findfirst(==(20000), NLADDER)
        for u in 1:7
            println(io, join([UNIT_NAMES[u], round(fi.info[u], sigdigits=3),
                              round(Int, fi.n_sd01[u]),
                              round(median(W[:, li600, u]), digits=3),
                              round(median(W[:, li20k, u]), digits=3),
                              worac[u], worac_drift[u]], ","))
        end
        println(io)
        for world in ("static", "drift")
            LS = e2b["LS_$world"]; LS0 = e2b["LS0_$world"]
            refs = e2b["refits"]
            pre = findall(<(n_pre), refs); post = findall(>=(n_pre), refs)
            for (hi, hn) in enumerate(HLNAMES)
                println(io, "$world,$hn,pre=", round(1000 * mean(LS[:, pre, hi] .- LS0[:, pre]), digits=3),
                        ",post=", round(1000 * mean(LS[:, post, hi] .- LS0[:, post]), digits=3))
            end
        end
    end
    return (fisher=fi, worac_drift=worac_drift)
end
