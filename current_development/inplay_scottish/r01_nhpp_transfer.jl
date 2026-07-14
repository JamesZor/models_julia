#=
r01_nhpp_transfer.jl — WP2 runner: transfer the Ireland NHPP spec to Scottish 56/57.

Homelab-sized (light NUTS). Sequence:
  §1 load ds + pregame OOS latents (decay-grid winner hl365_hs2), assemble training set
  §2 GLM CV race: pg_only vs +time vs +state vs +man_adv (paired t, 5×4 folds by match)
  §3 Bayesian NHPP fit (hier_time, linear state+man) via Samplers.run_sampler
  §4 diagnostics: posterior multipliers vs priors/Ireland, δ_time profile,
     late-checkpoint remaining-goals bias (the l08 headline check)
  §5 serialize chain + slices for WP3

Run on kaimon (homelab):
    ENV["JULIA_PKG_PRECOMPILE_AUTO"]="0"; using BayesianFootball
    include("current_development/inplay_scottish/r01_nhpp_transfer.jl")
=#

using Serialization, Statistics, DataFrames

const BF = BayesianFootball
!(@isdefined ds) && (ds = BF.Data.load_datastore_cached(BF.Data.ScottishLower()))

include(joinpath(@__DIR__, "l01_nhpp_scottish.jl"))

# ---------------------------------------------------------------------------
# §1 data
# ---------------------------------------------------------------------------

const LATENTS_JLS = joinpath(dirname(@__DIR__), "..", "data", "scottish_decay_grid",
                             "latents_hl365_hs2.jls")
# if missing: include scottish_lower_smile/l01_team_dp_league.jl (defines the Main
# struct JLD2 needs), Experiments.load_experiment(<winner dir>), extract_oos_predictions.
latents_df = deserialize(abspath(LATENTS_JLS))

# 56's 23/24 + 25/26 excluded: incident holes (r00). OOS λ only exists for 23/24+.
const TRAIN_PAIRS = Set([(56, "24/25"),
                         (57, "23/24"), (57, "24/25"), (57, "25/26")])

mseqs  = assemble_nhpp_matches(ds, latents_df, TRAIN_PAIRS)
config = NHPPXConfig()
slices = build_slices(mseqs; Δt = config.Δt, Tend = config.Tend)

data_stats = (matches = length(mseqs), rows = nrow(slices),
              goals = sum(slices.y), red_rows = sum(slices.man_adv .!= 0),
              by_pair = combine(groupby(
                  DataFrame(t = [m.tournament_id for m in mseqs],
                            s = [m.season for m in mseqs]), [:t, :s]), nrow))

# ---------------------------------------------------------------------------
# §2 GLM CV race
# ---------------------------------------------------------------------------

cv = cv_race(slices)
race = DataFrame(
    comparison = ["time − pg_only", "state − time", "full − state"],
    result = [paired_diff(cv, :time, :pg_only),
              paired_diff(cv, :state, :time),
              paired_diff(cv, :full, :state)])

glm_full = glm(SPEC_FORMULAS[:full], slices, Poisson(), LogLink(); offset = slices.off)

# ---------------------------------------------------------------------------
# §3 Bayesian NHPP
# ---------------------------------------------------------------------------

model = make_nhppx_model(slices, config)
chain = Samplers.run_sampler(model,
    Samplers.NUTSConfig(n_samples = 600, n_chains = 4, n_warmup = 300,
                        max_depth = 8, show_progress = false))

post = DataFrame(param = Symbol[], mean = Float64[], sd = Float64[],
                 q05 = Float64[], q95 = Float64[], rhat = Float64[])
sumstats = MCMCChains.summarystats(chain)
for p in (:α, :β, :γ_tr, :γ_ld, :γ_man, :σ_time)
    v = _cv(chain, p)
    rh = sumstats[p, :rhat]
    push!(post, (p, mean(v), std(v), quantile(v, 0.05), quantile(v, 0.95), rh))
end

nb = Int(cld(config.Tend, config.Δt))
δ_time_profile = DataFrame(
    bin = ["$(Int((b-1)*config.Δt))–$(Int(b*config.Δt))" for b in 1:nb],
    δ = vec(mean(_cm(chain, :z_time, nb) .* _cv(chain, :σ_time); dims = 1)))

# ---------------------------------------------------------------------------
# §4 late-checkpoint remaining-goals bias (posterior-mean pregame λ; per-draw Λ)
# ---------------------------------------------------------------------------

function checkpoint_bias(mseqs, chain, config; checkpoints = (60.0, 75.0, 85.0))
    rows = DataFrame(t0 = Float64[], pred = Float64[], real = Float64[])
    for ms in mseqs, t0 in checkpoints
        gh = count(g ->  g.home && g.t < t0, ms.goals)
        ga = count(g -> !g.home && g.t < t0, ms.goals)
        rh = count(c ->  c.home && c.t < t0, ms.reds)
        ra = count(c -> !c.home && c.t < t0, ms.reds)
        Λh, Λa = remaining_intensity(chain, config; pg_h = ms.pgh, pg_a = ms.pga,
                                     gh = gh, ga = ga, reds_h = rh, reds_a = ra, t_now = t0)
        push!(rows, (t0, mean(Λh) + mean(Λa),
                     Float64(count(g -> g.t >= t0, ms.goals))))
    end
    combine(groupby(rows, :t0), nrow => :n,
            [:pred, :real] => ((p, r) -> mean(p .- r)) => :bias,
            [:pred, :real] => ((p, r) -> std(p .- r) / sqrt(length(p))) => :se)
end

bias_tbl = checkpoint_bias(mseqs, chain, config)

# ---------------------------------------------------------------------------
# §5 persist + report
# ---------------------------------------------------------------------------

OUT = joinpath(@__DIR__, "out"); mkpath(OUT)
serialize(joinpath(OUT, "r01_chain.jls"), chain)
serialize(joinpath(OUT, "r01_slices.jls"), slices)
serialize(joinpath(OUT, "r01_mseqs.jls"), mseqs)

R01 = (data_stats = data_stats, cv = cv, race = race, glm_full = coeftable(glm_full),
       post = post, δ_time = δ_time_profile, bias = bias_tbl)

@info "r01 done" data_stats.matches data_stats.rows
