#=
r04b_pregame_source.jl — WP-B runner: swap the pregame engine, then re-run Gate B.

Gate B is explicitly NOT inherited from r02. The t=0 consistency check is a property of the
**pregame/multiplier PAIR**: the NHPP absorbs the pregame level into α, so composing a
multiplier chain trained against one pregame engine with a different engine's λ is exactly
the uncongeniality failure RESEARCH.md §3 warns about. So this runner refits the NHPP on
the new latents and then checks the pair.

  §1 load all three pregame sources, report coverage and the new shot latents
  §2 assemble + refit the NHPP with funnel_apm_xg as the offset
  §3 GATE B  — composed kickoff prices vs the pregame model's own prices
               (incumbent: kernel 0.988, max abs price gap < 0.01 over 17 selections)
  §4 repeat for funnel_winner, since WP-F races both pregame arms

Run on the kaimon server session (-t 16, pinthreads(:cores)):
    ENV["JULIA_PKG_PRECOMPILE_AUTO"]="0"; using BayesianFootball
    include("current_development/inplay_scottish/r04b_pregame_source.jl")
    GATE_B.verdict
=#

using DataFrames, Statistics, Serialization, Random

const BF = BayesianFootball
!(@isdefined ds) && (ds = BF.Data.load_datastore_cached(BF.Data.ScottishLower()))

include(joinpath(@__DIR__, "l01_nhpp_scottish.jl"))
include(joinpath(@__DIR__, "l02_ppd_compose.jl"))
include(joinpath(@__DIR__, "l04_bbc_timeline.jl"))
include(joinpath(@__DIR__, "l05_pregame_source.jl"))

OUT = joinpath(@__DIR__, "out"); mkpath(OUT)

# Pregame latents exist only for the two target seasons the funnel arms were run on.
# 56's 23/24 + 25/26 have incident holes (r00), so they drop out under require_incidents.
const TRAIN_PAIRS = Set([(56, "24/25"), (56, "25/26"),
                         (57, "23/24"), (57, "24/25"), (57, "25/26")])

# ---------------------------------------------------------------------------
# §1 sources
# ---------------------------------------------------------------------------

sources = Dict{String, AbstractPregameSource}(
    "funnel_apm_xg" => known_source("funnel_apm_xg"),
    "funnel_winner" => known_source("funnel_winner"),
)

# Archival only. The legacy latents were produced on the homelab and never scp'd to
# mcmc-beast, so it is normal for this to be absent here; every race arm composes against
# one of the funnel engines above, so nothing downstream depends on it.
const LEGACY_JLS = joinpath(dirname(dirname(@__DIR__)), "data", "scottish_decay_grid",
                            "latents_hl365_hs2.jls")
isfile(LEGACY_JLS) && (sources["legacy_hl365"] = LatentsFileSource(LEGACY_JLS))

draws = Dict(k => pregame_draws(v, ds) for (k, v) in sources)
qa = Dict(k => draws_qa(v, ds) for (k, v) in draws)

# ---------------------------------------------------------------------------
# §2/§3 refit + gate, per pregame engine
# ---------------------------------------------------------------------------

config = NHPPXConfig()

"""
Compose the book at kickoff (0-0, no reds, t=0) and compare, selection by selection,
against the pregame model's own book on the same draws. The composed price differs only by
the NHPP's net normalisation `E[K] = E[∫exp(α + βz + δ_time) dt]`, so `K ≈ 1` and the price
gap ≈ 0 is the statement that the in-play module has not moved the pregame marginal — the
congeniality condition.
"""
function gate_b(chain, cfg, ms_list, dsrc; n_match = 100, n_pairs = 2000)
    K0_h, K0_a = intensity_kernels(chain, cfg; gh = 0, ga = 0, t_now = 0.0)
    rows = DataFrame(mid = Int[], sel = Symbol[], composed = Float64[], pregame = Float64[])
    rng = Xoshiro(7)
    sel = ms_list[randperm(rng, length(ms_list))[1:min(n_match, length(ms_list))]]
    for ms in sel
        λh = dsrc[ms.mid].λ_h; λa = dsrc[ms.mid].λ_a
        ppd = inplay_ppd(chain, cfg, λh, λa; gh = 0, ga = 0, t_now = 0.0,
                         n_pairs = n_pairs, rng = Xoshiro(ms.mid))
        Spre = compose_score_matrix(λh, λa, ones(1), ones(1); gh = 0, ga = 0,
                                    n_pairs = n_pairs, rng = Xoshiro(ms.mid))
        for m in default_markets(), (s, v) in Pred.compute_market_probs(Spre, m)
            push!(rows, (ms.mid, s, mean(ppd[s]), mean(v)))
        end
    end
    per_sel = combine(groupby(rows, :sel),
        [:composed, :pregame] => ((c, p) -> mean(c .- p)) => :mean_diff,
        [:composed, :pregame] => ((c, p) -> maximum(abs.(c .- p))) => :max_abs_diff)
    (kernel_h = mean(K0_h), kernel_a = mean(K0_a),
     n_selections = nrow(per_sel), per_sel = per_sel,
     max_abs_diff = maximum(per_sel.max_abs_diff),
     pass = maximum(per_sel.max_abs_diff) < 0.01 &&
            abs(mean(K0_h) - 1) < 0.05 && abs(mean(K0_a) - 1) < 0.05)
end

"""
    gate_b_decompose(chain, cfg, ms_list, dsrc; ...)

Attributes whatever gap `gate_b` finds. Re-prices the pregame reference with λ scaled by
the scalar kernel `K`; if the gap collapses, the entire disagreement is a uniform LEVEL
rescaling and the two books agree in SHAPE — which is a different (and much milder) thing
than the composed grid disagreeing structurally with the pregame grid.

Reports `max_raw` (what `gate_b` scores) beside `max_scaled` (what is left once the level
is removed). `max_scaled` at the Monte Carlo floor means the gate reduces to a single
number, `K`, rather than to 17 independent failures.
"""
function gate_b_decompose(chain, cfg, ms_list, dsrc; n_match = 100, n_pairs = 2000)
    K0_h, K0_a = intensity_kernels(chain, cfg; gh = 0, ga = 0, t_now = 0.0)
    kh, ka = mean(K0_h), mean(K0_a)
    rows = DataFrame(sel = Symbol[], d_raw = Float64[], d_scaled = Float64[])
    rng = Xoshiro(7)
    sel = ms_list[randperm(rng, length(ms_list))[1:min(n_match, length(ms_list))]]
    for ms in sel
        λh = dsrc[ms.mid].λ_h; λa = dsrc[ms.mid].λ_a
        ppd = inplay_ppd(chain, cfg, λh, λa; gh = 0, ga = 0, t_now = 0.0,
                         n_pairs = n_pairs, rng = Xoshiro(ms.mid))
        Spre = compose_score_matrix(λh, λa, ones(1), ones(1); gh = 0, ga = 0,
                                    n_pairs = n_pairs, rng = Xoshiro(ms.mid))
        Sscl = compose_score_matrix(λh .* kh, λa .* ka, ones(1), ones(1); gh = 0, ga = 0,
                                    n_pairs = n_pairs, rng = Xoshiro(ms.mid))
        pre = Dict{Symbol, Float64}(); scl = Dict{Symbol, Float64}()
        for m in default_markets()
            for (s, v) in Pred.compute_market_probs(Spre, m); pre[s] = mean(v); end
            for (s, v) in Pred.compute_market_probs(Sscl, m); scl[s] = mean(v); end
        end
        for s in keys(pre)
            push!(rows, (s, mean(ppd[s]) - pre[s], mean(ppd[s]) - scl[s]))
        end
    end
    per_sel = combine(groupby(rows, :sel),
        :d_raw    => (x -> maximum(abs.(x))) => :max_raw,
        :d_scaled => (x -> maximum(abs.(x))) => :max_scaled,
        :d_scaled => mean => :mean_scaled)
    (kernel = kh, per_sel = per_sel,
     max_raw = maximum(per_sel.max_raw), max_scaled = maximum(per_sel.max_scaled),
     level_explains = 1 - maximum(per_sel.max_scaled) / maximum(per_sel.max_raw))
end

function fit_arm(name; samples = 600, warmup = 300, chains = 4)
    d = draws[name]
    ms = assemble_matches(ds, d, TRAIN_PAIRS)          # incidents source, l01 clock
    sl = build_slices(ms; Δt = config.Δt, Tend = config.Tend)
    cv = cv_race(sl)
    race = DataFrame(comparison = ["time − pg_only", "state − time", "full − state"],
                     result = [paired_diff(cv, :time, :pg_only),
                               paired_diff(cv, :state, :time),
                               paired_diff(cv, :full, :state)])
    ch = Samplers.run_sampler(make_nhppx_model(sl, config),
             Samplers.NUTSConfig(n_samples = samples, n_chains = chains,
                                 n_warmup = warmup, max_depth = 8, show_progress = false))
    post = DataFrame(param = Symbol[], mean = Float64[], sd = Float64[], rhat = Float64[])
    ss = MCMCChains.summarystats(ch)
    for p in (:α, :β, :γ_tr, :γ_ld, :γ_man, :σ_time)
        v = _cv(ch, p); push!(post, (p, mean(v), std(v), ss[p, :rhat]))
    end
    serialize(joinpath(OUT, "r04b_chain_$(name).jls"), ch)
    serialize(joinpath(OUT, "r04b_mseqs_$(name).jls"), ms)
    (name = name, mseqs = ms, slices = sl, chain = ch, race = race, post = post,
     gate = gate_b(ch, config, ms, d),
     decomp = gate_b_decompose(ch, config, ms, d),
     n_matches = length(ms), n_rows = nrow(sl), n_goals = sum(sl.y),
     max_rhat = maximum(skipmissing(ss[:, :rhat])))
end

arms = Dict(n => fit_arm(n) for n in ("funnel_apm_xg", "funnel_winner"))

# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

GATE_B = (
    qa = qa, arms = arms,
    summary = DataFrame(
        arm = collect(keys(arms)),
        matches = [a.n_matches for a in values(arms)],
        goals = [a.n_goals for a in values(arms)],
        kernel_h = [a.gate.kernel_h for a in values(arms)],
        kernel_a = [a.gate.kernel_a for a in values(arms)],
        max_price_gap = [a.gate.max_abs_diff for a in values(arms)],
        gap_after_level = [a.decomp.max_scaled for a in values(arms)],
        n_sel = [a.gate.n_selections for a in values(arms)],
        max_rhat = [a.max_rhat for a in values(arms)],
        pass = [a.gate.pass for a in values(arms)]),
    verdict = all(a.gate.pass for a in values(arms)) ? "GATE B PASS" : "GATE B FAIL",
)

@info "Gate B" GATE_B.verdict
