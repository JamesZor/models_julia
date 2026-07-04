#=
RUNNER — Experiment 3: is the EB trust fit worth anything over hard-coding w = 0.5?

E2 showed the learned per-line w is pinned to the pooled prior (~0.53) at campaign data
rates — so does the machinery earn its keep, or is a constant half-trust blend enough?
The constant can't do one thing the EB fit can: LEARN THE GLOBAL LEVEL. Pooling all 7
lines gives ~4.2k obs/season for the common w0 (7× the per-line rate), enough to detect
"this model deserves much less than half-trust".

Race (identical books, run_season): FLAT_1pct · U_cap02 (w=1, raw model) ·
TRUST05_U_cap02 (w hard-coded 0.5) · TRUST_U_cap02 (EB-learned), in two worlds:
  GOOD — base SimConfig: genuine info edge (σ_mod=0.05 < σ_mkt=0.08) + planted bias.
         Expected: EB ≈ fixed 0.5 (the E2 flatness in action).
  BAD  — σ_mod=0.12 > σ_mkt: model has NO information edge, same planted bias.
         Correct global trust is low; fixed 0.5 over-trusts junk, EB should pull w down.
Same seeds as the main race ⇒ GOOD-world numbers comparable to Experiment 1.

Server: include l01, l02 (updated: TRUST05_U_cap02 strategy), then this file; run_e3().
=#

using Random
using Statistics
using Serialization

if !@isdefined(SimConfig)
    include(joinpath(@__DIR__, "l01_sim_market_model.jl"))
end
if !@isdefined(fit_trust_eb)
    include(joinpath(@__DIR__, "l02_strategies.jl"))
end

const R4_RESULTS = joinpath(@__DIR__, "results"); mkpath(R4_RESULTS)
const E3_STRATS = ["FLAT_1pct", "U_cap02", "TRUST05_U_cap02", "TRUST_U_cap02"]

function run_e3(; N=300, base_seed=20260704)
    res = Dict{String,Any}()
    for (wn, cfg) in (("good", SimConfig()), ("bad", SimConfig(σ_mod=0.12)))
        acc = Vector{Any}(undef, N)
        Threads.@threads :dynamic for i in 1:N
            r = run_season(cfg, base_seed + i; strategies=E3_STRATS)
            acc[i] = (results=Dict(s => (logw=v.logw, ruined=v.ruined)
                                   for (s, v) in r.results),
                      w_final=r.w_final)
        end
        res[wn] = acc
    end
    serialize(joinpath(R4_RESULTS, "e3.jls"), res)
    return summarize_e3(res)
end

function summarize_e3(res=deserialize(joinpath(R4_RESULTS, "e3.jls")))
    lines = String[]
    push!(lines, "world,strategy,medW,q05W,q95W,meanG,medDD,ruin_pct")
    for wn in ("good", "bad")
        acc = res[wn]
        for s in E3_STRATS
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
    write(joinpath(R4_RESULTS, "e3_summary.txt"), txt)
    return txt
end
