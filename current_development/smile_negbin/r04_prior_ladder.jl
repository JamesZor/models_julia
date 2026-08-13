# current_development/smile_negbin/r04_prior_ladder.jl
#
# RUNNER. Does the data carry ANY information about the dispersion `r`?
#
# ---------------------------------------------------------------------------------------------
# THE QUESTION
# ---------------------------------------------------------------------------------------------
#
# r03 measured posterior sd / prior sd = 0.96-0.98 for `disp.log_r` on every fold: the posterior
# IS the prior. Two explanations are consistent with that, and they imply opposite decisions:
#
#   (i)  the prior `Normal(3.1, 0.4)` is simply tight, and a wider one would let the data speak;
#   (ii) the likelihood carries no information about `r` at this effective sample size, and the
#        engine's over-dispersion is a prior in disguise no matter how it is parameterised.
#
# A prior LADDER separates them. Run the identical fold under progressively less informative
# priors and watch the ratio. Under (i) the ratio falls as the prior widens — the posterior stops
# tracking it. Under (ii) the ratio stays pinned near 1.0 at every rung.
#
# ---------------------------------------------------------------------------------------------
# WHY THE TOP RUNG IS BOUNDED-FLAT, NOT UNBOUNDED-FLAT
# ---------------------------------------------------------------------------------------------
#
# `r -> ∞` IS the Poisson limit: `Var[NegBin(r,μ)] = μ + μ²/r -> μ`. So if the observed counts are
# not over-dispersed, the likelihood is MONOTONE INCREASING in `log r` with no interior maximum,
# and a genuinely improper flat prior gives an IMPROPER POSTERIOR. The chain then drifts upward
# forever, R-hat explodes, and the run looks like a sampler failure when it is in fact the answer.
#
# So the top rung is `Uniform(-2, 8)`: flat across `r ∈ [0.14, 2981]`, which spans everything from
# violent over-dispersion to numerically-Poisson (`Var/E = 1.0005` at r=2981, λ=1.35). It is
# non-informative over the entire range that could matter while remaining proper, so a drift to the
# Poisson end shows up as POSTERIOR MASS PILING AT THE UPPER EDGE — a readable measurement —
# rather than as divergence.
#
# The second rung `Normal(3.1, 2.0)` is deliberately kept below `Normal(3.1, 3.0)`: `build_dispersion`
# applies `exp(clamp(log_r, -10, 10))`, so any prior with meaningful mass past log_r = 10 creates a
# flat LIKELIHOOD PLATEAU at the clamp that chains wander across, contaminating R-hat for reasons
# that have nothing to do with the data. At sd 2.0 the clamp is 3.45 sd out. The clamp-hit fraction
# is reported anyway, so if this reasoning is wrong it is visible rather than silent.
#
# ---------------------------------------------------------------------------------------------
# SAMPLER: more warmup, more chains, as asked
# ---------------------------------------------------------------------------------------------
#
#   warmup 800 (vs r03's 250, production's 300)   samples 500   max_depth 10   accept_rate 0.65
#
# `accept_rate` is deliberately NOT raised for the harder rungs. Raising it would reduce
# divergences but would also mean the three rungs are no longer sampled identically, and the whole
# point is a controlled comparison ACROSS PRIORS.
#
# Chain allocation: 4 / 6 / 6 = 16 tasks, exactly one wave on a 16-thread session. The baseline
# rung gets fewer because r03 already showed it converges; the wide and flat rungs — where the
# geometry is genuinely harder — get more.
#
# ---------------------------------------------------------------------------------------------
# ONE FOLD, AND WHY THE PIPELINE IS ENTERED ONE LEVEL DOWN
# ---------------------------------------------------------------------------------------------
#
# 2026 biweek 12: the largest fold, inside the order-book corpus window. Features are built ONCE
# with the shipped `Features.create_features` and shared across all three rungs — `required_features`
# does not depend on `dispersion_config`, so the three models need byte-identical inputs, and
# rebuilding them per rung would burn the player-ratings tracker three times for nothing.
#
# Each chain then goes through `Training.train(model, config, feature_set; chain_id=c)` — the exact
# single-chain entry point `_train_queued` uses in production (strategies/independent.jl:150),
# including `UniformInit` and `AutoReverseDiff(compile=true)`. The only thing not reused is the
# split-level loop, because here the queue runs over (PRIOR, chain) rather than (split, chain).
# This is a diagnostic; it produces no artifact that anything downstream consumes.
#
# ---------------------------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------------------------
#
#   include("current_development/smile_negbin/r04_prior_ladder.jl")
#
# Expect ~60-90 min. Silent while sampling — a watchdog "failed" on no-output is a false alarm;
# check `isdefined(Main, :sn4_out)`.

using BayesianFootball
using DataFrames, Dates, Distributions, Statistics, Printf, Serialization

include(joinpath(@__DIR__, "l01_smile_negbin_engine.jl"))
include(joinpath(@__DIR__, "l02_smile_negbin_predict.jl"))

using Turing: MCMCChains

const Experiments = BayesianFootball.Experiments
const Samplers    = BayesianFootball.Samplers
const Training    = BayesianFootball.Training

const SN4_OUT_DIR = "./data/l2_ireland_engines"
const SN4_LOG     = joinpath(@__DIR__, "r04_out.txt")

const SN4_STEP    = 12          # 2026 biweek 12 — largest fold, inside the corpus window
const SN4_WARMUP  = 800
const SN4_SAMPLES = 500
const SN4_DEPTH   = 10
const SN4_ACCEPT  = 0.65

const SN4_LAMBDA_REF = 1.35     # typical league λ, for turning r into Var/E
const SN4_VE_BAR     = 1.05     # "materially over-dispersed" threshold

# ===================================================================
# 1. The ladder
# ===================================================================

sn4_rungs = [
    (name = "A tight   N(3.1, 0.4)",
     disp = PreGame.HomeAwayDispersion(log_r = Normal(3.1, 0.4),
                                       δ_r_home = Normal(0.0, 0.5)),
     chains = 4),
    (name = "B wide    N(3.1, 2.0)",
     disp = PreGame.HomeAwayDispersion(log_r = Normal(3.1, 2.0),
                                       δ_r_home = Normal(0.0, 2.0)),
     chains = 6),
    (name = "C flat    U(-2, 8)",
     disp = PreGame.HomeAwayDispersion(log_r = Uniform(-2.0, 8.0),
                                       δ_r_home = Uniform(-3.0, 3.0)),
     chains = 6),
]

sn4_model(disp) = DynamicSmileDoubleNegBinXGOutfieldPlayerTimeDecayModel(
    interception_config    = PreGame.HierarchicalMonthlyInterception(),
    player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(days_half_life = 60.0),
    dispersion_config      = disp,
    homeadvantage_config   = PreGame.HierarchicalTeamHomeAdvantage(),
    kappa_config           = PreGame.HierarchicalTeamKappa(),
    player_ratings_feature = Features.PlayerRatingsFeature(
                                 Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)),
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    smile_feature          = Features.MarketSmileFeature(Kmax = 4),
    market_on              = true,
    supremacy_weight       = 0.4,
    smile_weight           = 0.4,
)

# ===================================================================
# 2. The fold, built once
# ===================================================================

sn4_pin = joinpath(SN4_OUT_DIR, "ds_ire79.jls")
isfile(sn4_pin) || error("r04: no pin at $sn4_pin")
sn4_ds = deserialize(sn4_pin)

sn4_splitter = Data.GroupedCVConfig(
    tournament_groups = [Data.tournament_ids(sn4_ds.segment)],
    target_seasons    = ["2026"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = SN4_STEP,
    end_dynamics      = SN4_STEP,
    stop_early        = false,
)

sn4_bounds = Data.create_id_boundaries(sn4_ds, sn4_splitter)
println("\nboundaries: ", [(string(md.target_season), md.time_step) for (_, md) in sn4_bounds])

sn4_all_fs = Features.create_features(sn4_bounds, sn4_ds, sn4_model(sn4_rungs[1].disp),
                                      sn4_splitter.dynamics_col)
sn4_idx = findfirst(i -> sn4_all_fs[i][2].time_step == SN4_STEP, eachindex(sn4_all_fs))
isnothing(sn4_idx) && error("r04: no fold at time_step $SN4_STEP")
sn4_fs = sn4_all_fs[sn4_idx][1]

# The effective sample size of the GOALS pillar — the quantity that decides whether `r` is
# learnable at all. `match_weights = 0.5 ^ (Δdays / half_life)`, exactly as build_turing_model
# computes it.
sn4_eff_N = sum(0.5 .^ (Vector{Int}(sn4_fs.data[:dates]) ./ 60.0))
@printf("\nfold biweek %d: %d history matches, goals-pillar effective N %.1f (%d goal counts)\n",
        SN4_STEP, length(sn4_fs.data[:dates]), sn4_eff_N, round(Int, 2 * sn4_eff_N))

# ===================================================================
# 3. Queue over (rung, chain)
# ===================================================================

sn4_cfg(nc) = Training.TrainingConfig(
    Samplers.QueuedNUTSConfig(n_samples = SN4_SAMPLES, n_chains = nc, n_warmup = SN4_WARMUP,
                              accept_rate = SN4_ACCEPT, max_depth = SN4_DEPTH,
                              initialisation = Samplers.UniformInit(-2.0, 2.0),
                              show_progress = false),
    Training.Independent(parallel = true, max_concurrent_tasks = Threads.nthreads()),
    nothing, false)

sn4_tasks = [(ri, c) for ri in eachindex(sn4_rungs) for c in 1:sn4_rungs[ri].chains]
sn4_slots = Dict(ri => Vector{Any}(undef, sn4_rungs[ri].chains) for ri in eachindex(sn4_rungs))

println("\n", "="^90)
@printf("PRIOR LADDER  %d rungs, %d chain-tasks, %d+%d draws, max_depth %d, %d threads\n",
        length(sn4_rungs), length(sn4_tasks), SN4_WARMUP, SN4_SAMPLES, SN4_DEPTH,
        Threads.nthreads())
for r in sn4_rungs
    @printf("  %-24s  chains %d   prior sd(log_r) %.3f\n", r.name, r.chains, std(r.disp.log_r))
end
println("="^90)

sn4_sem  = Base.Semaphore(min(length(sn4_tasks), Threads.nthreads()))
sn4_lock = ReentrantLock()
sn4_t0   = time()

@sync for (ri, c) in sn4_tasks
    Threads.@spawn begin
        Base.acquire(sn4_sem)
        try
            ch = Training.train(sn4_model(sn4_rungs[ri].disp), sn4_cfg(sn4_rungs[ri].chains),
                                sn4_fs; chain_id = c)
            lock(sn4_lock) do; sn4_slots[ri][c] = ch; end
        catch e
            @error "rung $ri chain $c failed" exception = (e, catch_backtrace())
            lock(sn4_lock) do; sn4_slots[ri][c] = nothing; end
        finally
            Base.release(sn4_sem)
        end
    end
end

@printf("\nsampled in %.1f min\n", (time() - sn4_t0) / 60)

# ===================================================================
# 4. Read the ladder
# ===================================================================

sn4_rows = NamedTuple[]
for ri in eachindex(sn4_rungs)
    rung = sn4_rungs[ri]
    good = [c for c in sn4_slots[ri] if !isnothing(c)]
    if isempty(good)
        @warn "rung $(rung.name): every chain failed"
        continue
    end
    ch = cat(good...; dims = 3)

    summ = DataFrame(MCMCChains.summarize(ch))
    getf(r, syms) = (for s in syms; hasproperty(r, s) && return Float64(getproperty(r, s)); end; NaN)
    sampler_names = ("lp","n_steps","acceptance_rate","tree_depth","numerical_error","step_size",
                     "nom_step_size","is_accept","hamiltonian_energy","hamiltonian_energy_error",
                     "max_hamiltonian_energy_error")
    par = filter(r -> !(string(r.parameters) in sampler_names) && !isnan(getf(r, (:rhat,))), summ)

    grab(sym) = (row = findfirst(r -> string(r.parameters) == sym, eachrow(par));
                 isnothing(row) ? (rhat = NaN, ess = NaN) :
                 (rhat = getf(par[row, :], (:rhat,)), ess = getf(par[row, :], (:ess,:ess_bulk))))

    log_r = vec(Array(ch[Symbol("disp.log_r")]))
    δ     = vec(Array(ch[Symbol("disp.δ_r_home")]))
    r_a   = exp.(clamp.(log_r, -10, 10))
    r_h   = exp.(clamp.(log_r .+ δ, -10, 10))

    prior_sd = std(rung.disp.log_r)
    post_sd  = std(log_r)

    # Var/E = 1 + λ/r  =>  "materially over-dispersed" is r < λ / (bar - 1)
    r_bar = SN4_LAMBDA_REF / (SN4_VE_BAR - 1.0)

    push!(sn4_rows, (
        rung      = rung.name,
        n_chains  = length(good),
        max_rhat  = maximum(getf.(eachrow(par), Ref((:rhat,)))),
        rhat_logr = grab("disp.log_r").rhat,
        ess_logr  = grab("disp.log_r").ess,
        rhat_δ    = grab("disp.δ_r_home").rhat,
        div_pct   = :numerical_error in Symbol.(names(ch)) ?
                        100 * mean(vec(Array(ch[:numerical_error]))) : NaN,
        prior_sd  = prior_sd,
        post_sd   = post_sd,
        ratio     = post_sd / prior_sd,
        r_h_med   = median(r_h),
        r_a_med   = median(r_a),
        r_a_lo    = quantile(r_a, 0.05),
        r_a_hi    = quantile(r_a, 0.95),
        p_overdis = mean(r_a .< r_bar),          # P(Var/E > 1.05)
        p_poisson = mean(log_r .> 6.0),          # r > 403 => Var/E < 1.004, numerically Poisson
        # Same tail under the PRIOR. Without it, "20% of draws say Poisson" is unreadable — under
        # U(-2,8) that IS the prior, under N(3.1,0.4) it would be a 10-sigma event.
        p_pois_pri = ccdf(rung.disp.log_r, 6.0),
        p_clamped = mean(abs.(log_r) .>= 10.0),  # likelihood-plateau contamination check
    ))
end

sn4_out = DataFrame(sn4_rows)

println("\n", "="^90)
println("LADDER — does the posterior stop tracking the prior as the prior widens?")
println("="^90)
show(stdout, MIME"text/plain"(),
     select(sn4_out, :rung, :n_chains, :prior_sd, :post_sd, :ratio, :max_rhat, :rhat_logr,
            :ess_logr, :div_pct))
println()
println("\n  dispersion posterior (λ_ref = $SN4_LAMBDA_REF):")
show(stdout, MIME"text/plain"(),
     select(sn4_out, :rung, :r_h_med, :r_a_med, :r_a_lo, :r_a_hi, :p_overdis, :p_poisson,
            :p_pois_pri, :p_clamped))
println()

# ===================================================================
# 5. Verdict
# ===================================================================
#
# Order matters. "posterior fills the prior" and "posterior piles at the Poisson end" can BOTH be
# true of the same run, and they mean different things — so the Poisson test is asked first, using
# the prior's own tail mass as the reference. Judging the ratio first would file a decisive
# "the data says Poisson" result under "no information".

sn4_pick(pre) = (i = findfirst(startswith(pre), sn4_out.rung); isnothing(i) ? nothing : sn4_out[i, :])
sn4_flat  = something(sn4_pick("C"), sn4_out[end, :])
sn4_tight = something(sn4_pick("A"), sn4_out[1, :])
sn4_lift  = sn4_flat.p_poisson / max(sn4_flat.p_pois_pri, 1e-9)

println("\n", "="^90)
@printf("goals-pillar effective N on this fold: %.1f matches (%d goal counts)\n",
        sn4_eff_N, round(Int, 2 * sn4_eff_N))
@printf("ratio  tight %.2f -> flat %.2f     Poisson-tail lift under the flat prior %.2fx\n",
        sn4_tight.ratio, sn4_flat.ratio, sn4_lift)

if sn4_lift > 1.5
    println("""
VERDICT: the data speaks, and it says POISSON.
  Under the flat prior $(round(100*sn4_flat.p_poisson, digits=1))% of the posterior sits at
  r > 403 (Var/E < 1.004) against $(round(100*sn4_flat.p_pois_pri, digits=1))% under the prior —
  a $(round(sn4_lift, digits=1))x lift. The likelihood is pushing toward the Poisson limit, i.e.
  the parent engine was not mis-specified on this league and the NegBin buys nothing here.""")
elseif sn4_flat.ratio > 0.85
    println("""
VERDICT: the data carries essentially NO information about r.
  Even under a flat prior the posterior fills it (ratio $(round(sn4_flat.ratio, digits=2))).
  Any over-dispersion this engine prices is the PRIOR's, not the data's. Widening the prior does
  not fix that — it just makes the arbitrariness wider. The options that remain are structural:
  lengthen the goals half-life so r sees more data, pool r across 79 and 718, or accept a
  deliberately-chosen informative prior as a modelling decision and say so.""")
else
    println("""
VERDICT: the data DOES inform r — the tight prior was the binding constraint.
  Ratio falls to $(round(sn4_flat.ratio, digits=2)) under the flat prior, with the posterior
  concentrating at r_a ≈ $(round(sn4_flat.r_a_med, digits=1))
  (P(Var/E > $SN4_VE_BAR) = $(round(sn4_flat.p_overdis, digits=2))).
  Re-run r03/r02 with the wide prior; the tight one was suppressing a real signal.""")
end
println("="^90)

open(SN4_LOG, "w") do io
    println(io, "smile_negbin r04 — dispersion prior ladder, Ireland 79, 2026 biweek ", SN4_STEP)
    println(io, "run at ", now())
    @printf(io, "warmup %d, samples %d, max_depth %d, %d chain-tasks\n",
            SN4_WARMUP, SN4_SAMPLES, SN4_DEPTH, length(sn4_tasks))
    @printf(io, "goals-pillar effective N %.1f matches (%d goal counts)\n\n",
            sn4_eff_N, round(Int, 2 * sn4_eff_N))
    show(io, MIME"text/plain"(), sn4_out)
    println(io)
end
println("\nwrote $SN4_LOG")

nothing
