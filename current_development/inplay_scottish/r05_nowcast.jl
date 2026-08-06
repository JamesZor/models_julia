#=
r05_nowcast.jl — WP-D runner: Gate D for MVP-2 (in-play nowcast).

GATE D HAS TWO HALVES AND BOTH MUST MOVE TOGETHER:
  (i)  paired CV t on adding γ_sf, and
  (ii) an OOS remaining-goals calibration check at 60/75/85′.
A significant coefficient with flat calibration means the term is absorbing something else.

PRE-REGISTERED CONFOUND: surplus may proxy a PREGAME MIS-ESTIMATE rather than in-match
information — shots-above-expectation is exactly what a too-low pregame λ looks like. If
γ_sf is significant it must be refit with the pregame rate FREE, and survive. §4 does that
two ways: a GLM race where `log_pg` is folded into the offset (fixed at 1) vs left free,
and a Turing refit with `free_pg = true`.

  §1 assemble nowcast slices, audit the surplus covariate for leaks
  §2 GATE D (i): CV race, incumbent `full` vs `nowcast` vs `nowcast_opp`
  §3 Bayesian fit + GATE D (ii): calibration overall AND by surplus tercile
  §4 the confound test
  §5 leak audit — recompute with surplus shifted one slice INTO the future; if the
     "future" version is not clearly better, the causal version is not really using
     in-match information

Run on the kaimon server session (-t 16, pinthreads(:cores)):
    ENV["JULIA_PKG_PRECOMPILE_AUTO"]="0"; using BayesianFootball
    include("current_development/inplay_scottish/r05_nowcast.jl")
    GATE_D.verdict
=#

using DataFrames, Statistics, Serialization, Random

const BF = BayesianFootball
!(@isdefined ds) && (ds = BF.Data.load_datastore_cached(BF.Data.ScottishLower()))

for f in ("l01_nhpp_scottish.jl", "l02_ppd_compose.jl", "l04_bbc_timeline.jl",
          "l05_pregame_source.jl", "l06_shot_flow.jl", "l07_nowcast.jl")
    include(joinpath(@__DIR__, f))
end

OUT = joinpath(@__DIR__, "out"); mkpath(OUT)

const TRAIN_PAIRS = Set([(56, "24/25"), (56, "25/26"),
                         (57, "23/24"), (57, "24/25"), (57, "25/26")])

# ---------------------------------------------------------------------------
# §1 data + surplus audit
# ---------------------------------------------------------------------------

seqs = deserialize(joinpath(OUT, "r04a_seqs.jls"))
draws = Dict(n => pregame_draws(known_source(n), ds)
             for n in ("funnel_apm_xg", "funnel_winner"))
mseqs = Dict(n => assemble_matches(ds, d, TRAIN_PAIRS; seqs = seqs)
             for (n, d) in draws)
slices = Dict(n => build_nowcast_slices(m) for (n, m) in mseqs)

sl = slices["funnel_apm_xg"]
surplus_audit = (
    rows = nrow(sl),
    zero_surplus_rows = count(==(0.0), sl.surplus),
    frac_zero = mean(sl.surplus .== 0.0),
    mean = mean(sl.surplus), sd = std(sl.surplus),
    q = quantile(sl.surplus, [0.05, 0.5, 0.95]),
    clamped_lo = mean(sl.surplus .<= SURPLUS_CLAMP[1] + 1e-9),
    clamped_hi = mean(sl.surplus .>= SURPLUS_CLAMP[2] - 1e-9),
    # A causal covariate must be uncorrelated with the CURRENT slice's own goal count
    # beyond what the model explains; report the raw correlation for the record.
    cor_with_y = cor(sl.surplus, Float64.(sl.y)),
    # calibration of E[shots]: realised shots-before vs expected-before, late in the match
    late = let s = subset(sl, :time_idx => ByRow(>=(15)))
        (mean_obs = mean(s.shots_before), mean_exp = mean(s.exp_shots),
         ratio = mean(s.shots_before) / mean(s.exp_shots))
    end,
)

# ---------------------------------------------------------------------------
# §2 GATE D (i) — CV race
# ---------------------------------------------------------------------------

cvs = Dict(n => nowcast_cv(s) for (n, s) in slices)

const SPECS_ALL = merge(Dict(k => v for (k, v) in SPEC_FORMULAS), NOWCAST_FORMULAS)
const FIXED_PG  = Set([:nowcast_fixed_pg, :full_fixed_pg])
mcs = Dict(n => match_clustered_cv(s, SPECS_ALL, FIXED_PG) for (n, s) in slices)

# Both t's are reported side by side. The MATCH-CLUSTERED one is the inferential number;
# the fold-paired one is kept only to document the inflation factor.
gate_d_cv = DataFrame(arm = String[], comparison = String[], mean = Float64[],
                      t_fold_paired = Float64[], t_match_clustered = Float64[])
for n in sort(collect(keys(cvs))), (a, b) in (
        (:nowcast, :full), (:nowcast_opp, :full), (:nowcast_opp, :nowcast),
        (:nowcast_fixed_pg, :full_fixed_pg), (:full, :state), (:state, :time))
    r = paired_diff(cvs[n], a, b); m = mc_diff(mcs[n], a, b)
    push!(gate_d_cv, (n, "$a − $b", m.mean, r.t, m.t))
end

# ---------------------------------------------------------------------------
# §3 Bayesian fit + GATE D (ii) — calibration
# ---------------------------------------------------------------------------

nc_config = NowcastConfig()
nc_chain = Samplers.run_sampler(make_nowcast_model(slices["funnel_apm_xg"], nc_config),
    Samplers.NUTSConfig(n_samples = 600, n_chains = 4, n_warmup = 300,
                        max_depth = 8, show_progress = false))
nc_ss = MCMCChains.summarystats(nc_chain)
nc_post = DataFrame(param = Symbol[], mean = Float64[], sd = Float64[],
                    q05 = Float64[], q95 = Float64[], rhat = Float64[])
for p in (:α, :β, :γ_tr, :γ_ld, :γ_man, :γ_sf, :σ_time)
    v = _cv(nc_chain, p)
    push!(nc_post, (p, mean(v), std(v), quantile(v, 0.05), quantile(v, 0.95),
                    nc_ss[p, :rhat]))
end

nc_bias, nc_bias_tercile, nc_rows =
    nowcast_checkpoint_bias(mseqs["funnel_apm_xg"], nc_chain, nc_config)

# ---------------------------------------------------------------------------
# §4 the confound test — is γ_sf a pregame correction?
# ---------------------------------------------------------------------------

free_config = NowcastConfig(free_pg = true)
free_chain = Samplers.run_sampler(make_nowcast_model(slices["funnel_apm_xg"], free_config),
    Samplers.NUTSConfig(n_samples = 600, n_chains = 4, n_warmup = 300,
                        max_depth = 8, show_progress = false))
free_ss = MCMCChains.summarystats(free_chain)
free_post = DataFrame(param = Symbol[], mean = Float64[], sd = Float64[],
                      q05 = Float64[], q95 = Float64[], rhat = Float64[])
for p in (:γ_sf, :ρ_)
    v = _cv(free_chain, p)
    push!(free_post, (p, mean(v), std(v), quantile(v, 0.05), quantile(v, 0.95),
                      free_ss[p, :rhat]))
end

γ_sf_fixed = mean(_cv(nc_chain, :γ_sf))
γ_sf_free  = mean(_cv(free_chain, :γ_sf))
confound = (γ_sf_fixed = γ_sf_fixed, γ_sf_free = γ_sf_free,
            retained = γ_sf_free / γ_sf_fixed,
            ρ = mean(_cv(free_chain, :ρ_)),
            # if γ_sf were a pregame correction it should be LARGER where the pregame
            # rate is pinned at 1 (the *_fixed_pg CV comparison)
            cv_fixed_pg_t = only(subset(gate_d_cv,
                :arm => ByRow(==("funnel_apm_xg")),
                :comparison => ByRow(==("nowcast_fixed_pg − full_fixed_pg"))).t_match_clustered))

# ---------------------------------------------------------------------------
# §5 leak audit — a FUTURE-shifted surplus, as a NEGATIVE control on the causal build
# ---------------------------------------------------------------------------
# ⚠ THIS CHECK ANSWERS LESS THAN IT LOOKS LIKE. It was written to ask "does surplus read
# in-match information?" — but a goal IS a shot (`goals ⊂ shots` by construction), so
# letting the covariate see the slice being predicted is near-tautological and the future
# version wins by three orders of magnitude no matter what. It cannot separate information
# from identity.
#
# What it DOES establish, and the reason it is kept: the causal build is not leaking. If
# `build_nowcast_slices` had an off-by-one, the causal t would sit near the future t rather
# than ~500× below it in mean effect. Read it as a negative control, nothing more.

leak_audit = let s = copy(slices["funnel_apm_xg"])
    fut = similar(s.surplus)
    for g in groupby(s, [:match_id, :is_home])
        idx = parentindices(g)[1]
        fut[idx] = vcat(g.surplus[2:end], g.surplus[end])
    end
    s.surplus = fut
    cv2 = nowcast_cv(s)
    (future = paired_diff(cv2, :nowcast, :full),
     causal = paired_diff(cvs["funnel_apm_xg"], :nowcast, :full))
end

# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

cv_t = only(subset(gate_d_cv, :arm => ByRow(==("funnel_apm_xg")),
                   :comparison => ByRow(==("nowcast − full"))).t_match_clustered)
cv_sig = abs(cv_t) > 2
cal_moved = any(abs.(nc_bias.bias) .< abs.([0.0747, 0.0512, 0.0244]))  # vs MVP-1
γ_credible = !(quantile(_cv(nc_chain, :γ_sf), 0.05) < 0 <
                quantile(_cv(nc_chain, :γ_sf), 0.95))

GATE_D = (
    surplus_audit = surplus_audit, cv = gate_d_cv, post = nc_post,
    bias = nc_bias, bias_by_surplus = nc_bias_tercile,
    confound = confound, free_post = free_post, leak_audit = leak_audit,
    max_rhat = maximum(skipmissing(nc_ss[:, :rhat])),
    cv_t = cv_t, γ_credible = γ_credible,
    verdict = (cv_sig && γ_credible) ?
        "GATE D PASS (i) — confirm calibration + confound before believing it" :
        "GATE D NULL — surplus adds nothing over the incumbent",
)

serialize(joinpath(OUT, "r05_nowcast_chain.jls"), nc_chain)
serialize(joinpath(OUT, "r05_gate_d.jls"),
          (; GATE_D.surplus_audit, GATE_D.cv, GATE_D.post, GATE_D.bias,
             GATE_D.bias_by_surplus, GATE_D.confound, GATE_D.free_post,
             GATE_D.leak_audit, GATE_D.verdict))

@info "Gate D" GATE_D.verdict GATE_D.cv_t
