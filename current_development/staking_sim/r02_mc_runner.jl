#=
RUNNER — Monte Carlo staking-strategy race on simulated double-Poisson seasons.

Run on the kaimon session (after r01 numbers are baked into SimConfig defaults):
    ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
    using BayesianFootball
    include("current_development/staking_sim/r02_mc_runner.jl")   # defines everything
    smoke()                                                        # timed 1-season check
    run_mc()                                                       # N seasons, chunked+resumable
    make_outputs()                                                 # summary.csv + 4 plots

Chunked + checkpointed: after every chunk, completed seasons are serialized to
results/results_partial.jls and a progress line is printed (keeps the kaimon 10-min gate
alive; on interruption, re-include and run_mc() resumes). Threads: julia --project -t 32;
per-season RNG = Xoshiro(BASE_SEED + i) — no global RNG anywhere.
=#

using Random
using Statistics
using Serialization
using Printf
using Plots
using StatsPlots   # boxplot/violin recipes; project dep (loads GR — a few seconds)

include(joinpath(@__DIR__, "l01_sim_market_model.jl"))
include(joinpath(@__DIR__, "l02_strategies.jl"))

const RESULTS_DIR = joinpath(@__DIR__, "results"); mkpath(RESULTS_DIR)
const PLOT_DIR = joinpath(@__DIR__, "plots"); mkpath(PLOT_DIR)
const PARTIAL = joinpath(RESULTS_DIR, "results_partial.jls")
const FINAL = joinpath(RESULTS_DIR, "results.jls")

const N_SEASONS = 300
const BASE_SEED = 20260704
const CHUNK = 25
const CFG = SimConfig()   # r01-calibrated defaults

# ---------- smoke test (verification item 4) ----------

function smoke()
    println("sanity_checks():"); sanity_checks()
    println("\ntimed 1 season (seed=$(BASE_SEED)):")
    t = @elapsed res = run_season(CFG, BASE_SEED)
    for s in STRATEGY_NAMES
        m = summarize_logw(res.results[s].logw)
        @printf("  %-14s  W=%8.3f  G/m=%+.5f  maxDD=%5.1f%%  bets=%4d  ruined=%s\n",
                s, m.terminal_W, m.G_per_match, 100m.max_dd,
                res.results[s].n_bets, res.results[s].ruined)
    end
    println("  w_final = ", round.(res.w_final, digits=2), "  (units: ", UNIT_NAMES, ")")
    @printf("  elapsed %.1fs → est. full run %.1f min on %d threads\n",
            t, t * N_SEASONS / 60 / Threads.nthreads(), Threads.nthreads())
    return t
end

# ---------- the MC loop ----------

function run_mc(; N=N_SEASONS)
    done = isfile(PARTIAL) ? deserialize(PARTIAL) : Dict{Int,Any}()
    todo = [i for i in 1:N if !haskey(done, i)]
    println("run_mc: $(length(done)) done, $(length(todo)) to go, $(Threads.nthreads()) threads")
    t0 = time()
    for chunk in Iterators.partition(todo, CHUNK)
        buf = Vector{Any}(undef, length(chunk))
        Threads.@threads :dynamic for j in eachindex(chunk)
            i = chunk[j]
            buf[j] = run_season(CFG, BASE_SEED + i)
        end
        for (j, i) in enumerate(chunk)
            done[i] = buf[j]
        end
        serialize(PARTIAL, done)
        el = time() - t0
        nd = length(done)
        @printf("chunk done: %d/%d seasons, %.1f min elapsed, ETA %.1f min\n",
                nd, N, el / 60, el / 60 * (N - nd) / max(nd - (N - length(todo)), 1))
    end
    serialize(FINAL, (cfg=CFG, results=done))
    println("saved $(FINAL)")
    return done
end

# ---------- aggregation + outputs ----------

function aggregate(done::Dict{Int,Any})
    rows = []
    for s in STRATEGY_NAMES
        tw = Float64[]; g = Float64[]; dd = Float64[]; ru = Float64[]; tn = Float64[]; nb = Float64[]
        for (_, r) in done
            m = summarize_logw(r.results[s].logw)
            push!(tw, m.terminal_W); push!(g, m.G_per_match); push!(dd, m.max_dd)
            push!(ru, r.results[s].ruined ? 1.0 : 0.0)
            push!(tn, r.results[s].turnover / length(r.results[s].logw))
            push!(nb, r.results[s].n_bets / length(r.results[s].logw))
        end
        push!(rows, (strategy=s,
                     medW=median(tw), q05W=quantile(tw, 0.05), q95W=quantile(tw, 0.95),
                     meanG=mean(g), medDD=median(dd), q95DD=quantile(dd, 0.95),
                     ruin_pct=100mean(ru), turnover_pm=mean(tn), bets_pm=mean(nb)))
    end
    return rows
end

function write_summary(rows)
    open(joinpath(RESULTS_DIR, "summary.csv"), "w") do io
        println(io, "strategy,medW,q05W,q95W,meanG,medDD,q95DD,ruin_pct,turn_pm,bets_pm")
        for r in rows
            println(io, join([r.strategy, r.medW, r.q05W, r.q95W, r.meanG, r.medDD,
                              r.q95DD, r.ruin_pct, r.turnover_pm, r.bets_pm], ','))
        end
    end
    for r in rows
        @printf("%-14s medW=%8.3f [%7.3f, %8.3f]  G=%+.5f  medDD=%5.1f%%  ruin=%4.1f%%  turn/m=%.3f\n",
                r.strategy, r.medW, r.q05W, r.q95W, r.meanG, 100r.medDD, r.ruin_pct, r.turnover_pm)
    end
end

function make_plots(done::Dict{Int,Any}; worac=nothing)
    seeds = sort(collect(keys(done)))

    # p1: wealth fans for 6 headline strategies
    show6 = ["K_full", "K_half", "BM_num", "U_cap02", "TRUST_U_cap02", "FLAT_1pct"]
    panels = []
    for s in show6
        trajs = [exp.(cumsum(done[i].results[s].logw)) for i in seeds]
        L = minimum(length.(trajs))
        M = hcat([t[1:L] for t in trajs]...)          # L × n_seasons
        med = [median(view(M, i, :)) for i in 1:L]
        lo = [quantile(view(M, i, :), 0.05) for i in 1:L]
        hi = [quantile(view(M, i, :), 0.95) for i in 1:L]
        p = plot(1:L, med, ribbon=(med .- lo, hi .- med), lw=2, legend=false,
                 title=s, yscale=:log10, titlefontsize=9, guidefontsize=7)
        for t in trajs[1:min(20, length(trajs))]
            plot!(p, 1:L, max.(t[1:L], 1e-6), alpha=0.15, lw=0.4, color=:grey)
        end
        push!(panels, p)
    end
    savefig(plot(panels...; layout=(2, 3), size=(1400, 750),
                 left_margin=5Plots.mm, bottom_margin=4Plots.mm),
            joinpath(PLOT_DIR, "p1_wealth_fan.png"))

    # p2: terminal log10-wealth boxplots, ordered by median
    tw = Dict(s => [exp(sum(done[i].results[s].logw)) for i in seeds] for s in STRATEGY_NAMES)
    ord = Base.sort(STRATEGY_NAMES, by=s -> median(tw[s]))
    xs = String[]; ys = Float64[]
    for s in ord, v in tw[s]
        push!(xs, s); push!(ys, log10(max(v, 1e-6)))
    end
    savefig(boxplot(xs, ys, xrotation=30, legend=false, ylabel="log10 terminal W",
                    size=(1100, 550), bottom_margin=8Plots.mm),
            joinpath(PLOT_DIR, "p2_terminal_box.png"))

    # p3: growth vs drawdown scatter (marker size = ruin freq)
    rows = aggregate(done)
    sc = scatter([100r.medDD for r in rows], [r.meanG for r in rows],
                 ms=[4 + 0.4r.ruin_pct for r in rows],
                 series_annotations=[Plots.text(" " * String(r.strategy), 7, :left) for r in rows],
                 xlabel="median max drawdown (%)", ylabel="mean log-growth / match",
                 legend=false, size=(950, 600))
    hline!(sc, [0.0], color=:grey, ls=:dash)
    savefig(sc, joinpath(PLOT_DIR, "p3_dd_growth.png"))

    # p4: learned trust per line vs oracle
    xs4 = String[]; ys4 = Float64[]
    for (u, un) in enumerate(UNIT_NAMES), i in seeds
        push!(xs4, un); push!(ys4, done[i].w_final[u])
    end
    p4 = boxplot(xs4, ys4, legend=false, ylabel="end-of-season trust w (EB)",
                 ylims=(0, 1), size=(950, 550), bottom_margin=6Plots.mm)
    if worac !== nothing
        scatter!(p4, UNIT_NAMES, worac, marker=:x, ms=10, color=:red,
                 label="oracle", legend=:topright)
    end
    savefig(p4, joinpath(PLOT_DIR, "p4_trust_w.png"))
    println("plots → $(PLOT_DIR)")
end

function make_outputs()
    done = isfile(FINAL) ? deserialize(FINAL).results : deserialize(PARTIAL)
    rows = aggregate(done)
    write_summary(rows)
    println("computing oracle trust (40k matches)…")
    worac = oracle_trust(CFG)
    println("  w_oracle = ", round.(worac, digits=2), "  (units: ", UNIT_NAMES, ")")
    serialize(joinpath(RESULTS_DIR, "worac.jls"), worac)
    make_plots(done; worac=worac)
end

println("r02 loaded: smoke() → run_mc() → make_outputs()")
