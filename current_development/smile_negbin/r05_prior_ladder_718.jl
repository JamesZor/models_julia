# current_development/smile_negbin/r05_prior_ladder_718.jl
#
# RUNNER. r04's dispersion prior ladder, re-run on Ireland 718 First Division.
#
# ---------------------------------------------------------------------------------------------
# WHY 718 IS THE LEAGUE THAT DESERVES THE SECOND LOOK
# ---------------------------------------------------------------------------------------------
#
# r04 settled 79: the ratio falls 0.92 -> 0.74 -> 0.59 as the prior widens (so `r` IS identified),
# and what the data then says is POISSON — 42% of the flat-prior posterior at r > 403, median
# Var/E = 1.005. Raw marginal V/M on 79 is 0.966, mildly UNDER-dispersed, which a NegBin cannot
# represent at any `r`.
#
# 718 is a different case on two counts, and neither is decorative:
#
#   1. It is the league `outfield_xg_double_negbin.jl` was originally motivated by — its header
#      cites `V/M ≈ 1.14`. That figure does not reproduce on the pinned store over 2024-2026
#      (measured 1.031), but 718 is still the only one of the pair that is above Poisson at all.
#   2. Lower division, higher mean goals (1.351 vs 1.225), noisier team quality — the a-priori
#      case for genuine over-dispersion is stronger here than anywhere else in the corpus.
#
# So this is the fair test of the NegBin engine, not a repeat. If 718 also comes back Poisson, the
# goals likelihood is closed as a lever across the whole Ireland corpus rather than on one league.
#
# ---------------------------------------------------------------------------------------------
# WHAT THIS ADDS OVER r04: THE CONDITIONAL DISPERSION INDEX (§2, runs BEFORE the MCMC)
# ---------------------------------------------------------------------------------------------
#
# r04's model-free corroboration was the MARGINAL variance-to-mean, pooled over matches. That
# quantity is inflated by λ heterogeneity — matches differ in expected goals, and that between-match
# spread lands in the marginal variance even under a perfectly Poisson likelihood. So a marginal
# V/M of 1.03 is an UPPER bound on the within-match over-dispersion, and it left a gap in the
# argument: the conditional number could be anywhere below it.
#
# This closes that gap directly. The feature set already carries the market's own per-match
# `λ_home` / `λ_away` (`:flat_market_λ_home`, from the vig-free double-Poisson inversion), so the
# conditioning mean is available without fitting anything. The Pearson dispersion index
#
#     D = (1/n) Σ (y_i - λ_i)² / λ_i
#
# has E[D] = 1 under Poisson with KNOWN λ, and `n·D ~ χ²_n`, giving an actual p-value. For a
# NegBin, E[(y-λ)²/λ] = 1 + λ/r, so `r̂ = λ̄ / (D - 1)` is a method-of-moments estimate that can be
# read straight against the posterior median the ladder produces.
#
# Two honesty checks are reported alongside it, because D is only as good as the λ it conditions on:
#   - `calib = mean(y) / mean(λ)`. If the market λ is biased, D inflates for a reason that has
#     nothing to do with dispersion. A calib far from 1.0 invalidates the test rather than
#     supporting it.
#   - market coverage. Folds where the market feature is absent are excluded, not zero-filled.
#
# This section costs seconds and writes its OWN log before the sampler starts, so the answer
# survives even if the ladder is interrupted.
#
# ---------------------------------------------------------------------------------------------
# HELD FIXED FROM r04 SO THE TWO LADDERS ARE COMPARABLE
# ---------------------------------------------------------------------------------------------
#
# Same three rungs, same 4/6/6 chain allocation, same warmup 800 / samples 500 / max_depth 10 /
# accept_rate 0.65, same `UniformInit`, same single fold (2026 biweek 12 — the largest, 360 history
# matches on both leagues, inside the order-book corpus window). Only the DataStore changes.
#
# `λ_ref` stays 1.35, which happens to be 718's own mean goals (1.351) — on 79 it was slightly
# generous at 1.225, so if anything the r04 thresholds were easier to clear than these.
#
# ---------------------------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------------------------
#
#   include("current_development/smile_negbin/r05_prior_ladder_718.jl")
#
# Expect ~60-90 min. Silent while sampling — a watchdog "failed" on no-output is a false alarm;
# check `isdefined(Main, :sn5_out)`.

using BayesianFootball
using DataFrames, Dates, Distributions, Statistics, Printf, Serialization

include(joinpath(@__DIR__, "l01_smile_negbin_engine.jl"))
include(joinpath(@__DIR__, "l02_smile_negbin_predict.jl"))

using Turing: MCMCChains

const Samplers5 = BayesianFootball.Samplers
const Training5 = BayesianFootball.Training

const SN5_OUT_DIR  = "./data/l2_ireland_engines"
const SN5_LOG      = joinpath(@__DIR__, "r05_out.txt")
const SN5_FREE_LOG = joinpath(@__DIR__, "r05_modelfree.txt")

const SN5_STEP    = 12          # 2026 biweek 12 — r04's fold, largest, inside the corpus window
const SN5_WARMUP  = 800
const SN5_SAMPLES = 500
const SN5_DEPTH   = 10
const SN5_ACCEPT  = 0.65

const SN5_LAMBDA_REF = 1.35     # 718's own mean goals is 1.351
const SN5_VE_BAR     = 1.05     # "materially over-dispersed" threshold

# ===================================================================
# 1. The ladder — identical rungs to r04
# ===================================================================

sn5_rungs = [
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

sn5_model(disp) = DynamicSmileDoubleNegBinXGOutfieldPlayerTimeDecayModel(
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

sn5_pin = joinpath(SN5_OUT_DIR, "ds_ire718_bfpillar.jls")
isfile(sn5_pin) || error("r05: no pin at $sn5_pin")
sn5_ds = deserialize(sn5_pin)

sn5_splitter = Data.GroupedCVConfig(
    tournament_groups = [Data.tournament_ids(sn5_ds.segment)],
    target_seasons    = ["2026"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = SN5_STEP,
    end_dynamics      = SN5_STEP,
    stop_early        = false,
)

sn5_bounds = Data.create_id_boundaries(sn5_ds, sn5_splitter)

sn5_all_fs = Features.create_features(sn5_bounds, sn5_ds, sn5_model(sn5_rungs[1].disp),
                                      sn5_splitter.dynamics_col)
sn5_idx = findfirst(i -> sn5_all_fs[i][2].time_step == SN5_STEP, eachindex(sn5_all_fs))
isnothing(sn5_idx) && error("r05: no fold at time_step $SN5_STEP")
sn5_fs = sn5_all_fs[sn5_idx][1]

sn5_w     = 0.5 .^ (Vector{Int}(sn5_fs.data[:dates]) ./ 60.0)
sn5_eff_N = sum(sn5_w)

# ===================================================================
# 2b. MODEL-FREE: is there any within-match over-dispersion to find?
# ===================================================================
#
# Runs before the sampler and writes its own log. If D ≈ 1 with a calibrated λ, the ladder's
# answer is already knowable and the MCMC is confirmation rather than discovery.

_sn5_ok(x) = !ismissing(x) && !isnan(Float64(x)) && Float64(x) > 0

# A SANITY GATE ON THE CONDITIONING MEAN, added after the first run.
#
# `:flat_market_λ_home` on the 718 pin carries a small number of impossible values — the first run
# found λ_home = 357.15 and λ_home = 0.001 against a median of 1.496. Three rows out of the 276
# with market coverage. That is 1.1% of the data, and it is enough to destroy the test: a single
# observation at λ = 357 contributes (y-λ)²/λ ≈ 355 to a sum of n = 276, adding ~1.3 to D on its
# own. The first run reported D_home = 9.14, which is an artefact of those rows and nothing else.
#
# The `calib` guard caught it — home calib came back 0.387 against away's 0.993 — which is exactly
# what it is for. The gate below makes the test robust rather than merely self-invalidating.
# Both the raw and gated statistics are reported, so the contamination stays visible.
#
# 0.2 .. 5.0 goals is deliberately loose: it admits every plausible football scoreline mean and
# excludes only values that cannot be a per-team goal expectation.
_sn5_sane(x) = _sn5_ok(x) && 0.2 <= Float64(x) <= 5.0

function sn5_dispersion_index(y::Vector{Int}, λ::Vector{Float64}, w::Vector{Float64})
    n  = length(y)
    n == 0 && return (n = 0, D = NaN, p = NaN, calib = NaN, D_w = NaN, r_mom = NaN, λ̄ = NaN)
    z  = (y .- λ) .^ 2 ./ λ
    D  = mean(z)
    # λ is taken as KNOWN (the market's, not fitted here), so the reference is χ²_n, not χ²_{n-k}.
    p  = ccdf(Chisq(n), n * D)
    Dw = sum(w .* z) / sum(w)                      # time-decay weighted, as the likelihood sees it
    λ̄  = mean(λ)
    # NegBin moment identity: E[(y-λ)²/λ] = 1 + λ/r  =>  r̂ = λ̄ / (D-1). Undefined at or below
    # Poisson, which is itself the informative outcome.
    r_mom = D > 1.0 ? λ̄ / (D - 1.0) : Inf
    return (n = n, D = D, p = p, calib = mean(y) / λ̄, D_w = Dw, r_mom = r_mom, λ̄ = λ̄)
end

function sn5_free_stats(d, pick)
    yh, ya = Vector{Int}(d[:flat_home_goals]), Vector{Int}(d[:flat_away_goals])
    mh, ma = d[:flat_market_λ_home], d[:flat_market_λ_away]
    keep = findall(i -> pick(mh[i]) && pick(ma[i]), eachindex(yh))
    y = vcat(yh[keep], ya[keep])
    λ = vcat(Float64.(mh[keep]), Float64.(ma[keep]))
    w = vcat(sn5_w[keep], sn5_w[keep])
    (n_rows = length(keep), coverage = length(keep) / length(yh),
     home = sn5_dispersion_index(yh[keep], Float64.(mh[keep]), sn5_w[keep]),
     away = sn5_dispersion_index(ya[keep], Float64.(ma[keep]), sn5_w[keep]),
     both = sn5_dispersion_index(y, λ, w),
     # the marginal V/M r04 reported, on the SAME rows, so the two are directly comparable
     marg_VM = var(y) / mean(y))
end

sn5_free_raw = sn5_free_stats(sn5_fs.data, _sn5_ok)     # every row with a market λ present
sn5_free     = sn5_free_stats(sn5_fs.data, _sn5_sane)   # ... minus the impossible ones
sn5_n_bad    = sn5_free_raw.n_rows - sn5_free.n_rows

open(SN5_FREE_LOG, "w") do io
    println(io, "smile_negbin r05 — model-free dispersion, Ireland 718, 2026 biweek $SN5_STEP")
    println(io, "run at ", now())
    @printf(io, "history %d matches, market λ present on %.1f%%, %d row(s) dropped as impossible, " *
                "goals-pillar effective N %.1f\n\n",
            length(sn5_fs.data[:dates]), 100 * sn5_free_raw.coverage, sn5_n_bad, sn5_eff_N)
    println(io, "Pearson dispersion index D = mean((y-λ)²/λ) against the MARKET's λ.")
    println(io, "Poisson => D = 1.  NegBin => D = 1 + λ/r.  p = P(χ²_n ≥ nD).\n")
    for (tag, blk) in (("GATED (0.2 <= λ <= 5)", sn5_free), ("RAW (all present)", sn5_free_raw))
        println(io, "  ", tag)
        for (nm, s) in (("home", blk.home), ("away", blk.away), ("both", blk.both))
            @printf(io, "    %-5s n %4d   D %.4f   D_weighted %.4f   p %.4f   calib %.4f   r̂_mom %s\n",
                    nm, s.n, s.D, s.D_w, s.p, s.calib,
                    isinf(s.r_mom) ? "Inf (at/below Poisson)" : @sprintf("%.1f", s.r_mom))
        end
    end
    @printf(io, "\n  marginal V/M on the gated rows: %.4f  (upper bound on the conditional D)\n",
            sn5_free.marg_VM)
    println(io, "\n  calib = mean(y)/mean(λ); far from 1.00 means the market λ is biased and D is " *
                "not a clean dispersion test. Compare RAW vs GATED calib to see the contamination.")
end

# ===================================================================
# 3. Queue over (rung, chain)
# ===================================================================

sn5_cfg(nc) = Training5.TrainingConfig(
    Samplers5.QueuedNUTSConfig(n_samples = SN5_SAMPLES, n_chains = nc, n_warmup = SN5_WARMUP,
                               accept_rate = SN5_ACCEPT, max_depth = SN5_DEPTH,
                               initialisation = Samplers5.UniformInit(-2.0, 2.0),
                               show_progress = false),
    Training5.Independent(parallel = true, max_concurrent_tasks = Threads.nthreads()),
    nothing, false)

sn5_tasks = [(ri, c) for ri in eachindex(sn5_rungs) for c in 1:sn5_rungs[ri].chains]
sn5_slots = Dict(ri => Vector{Any}(undef, sn5_rungs[ri].chains) for ri in eachindex(sn5_rungs))

sn5_sem  = Base.Semaphore(min(length(sn5_tasks), Threads.nthreads()))
sn5_lock = ReentrantLock()
sn5_t0   = time()

@sync for (ri, c) in sn5_tasks
    Threads.@spawn begin
        Base.acquire(sn5_sem)
        try
            ch = Training5.train(sn5_model(sn5_rungs[ri].disp), sn5_cfg(sn5_rungs[ri].chains),
                                 sn5_fs; chain_id = c)
            lock(sn5_lock) do; sn5_slots[ri][c] = ch; end
        catch e
            @error "rung $ri chain $c failed" exception = (e, catch_backtrace())
            lock(sn5_lock) do; sn5_slots[ri][c] = nothing; end
        finally
            Base.release(sn5_sem)
        end
    end
end

sn5_minutes = (time() - sn5_t0) / 60

# ===================================================================
# 4. Read the ladder
# ===================================================================

sn5_rows = NamedTuple[]
for ri in eachindex(sn5_rungs)
    rung = sn5_rungs[ri]
    good = [c for c in sn5_slots[ri] if !isnothing(c)]
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
    r_bar    = SN5_LAMBDA_REF / (SN5_VE_BAR - 1.0)   # Var/E = 1 + λ/r > 1.05  <=>  r < r_bar

    push!(sn5_rows, (
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
        ve_med    = 1 + SN5_LAMBDA_REF / median(r_a),
        p_overdis = mean(r_a .< r_bar),          # P(Var/E > 1.05)
        p_poisson = mean(log_r .> 6.0),          # r > 403 => Var/E < 1.004, numerically Poisson
        p_pois_pri = ccdf(rung.disp.log_r, 6.0), # the same tail under the PRIOR — the reference
        p_clamped = mean(abs.(log_r) .>= 10.0),  # likelihood-plateau contamination check
    ))
end

sn5_out = DataFrame(sn5_rows)

# ===================================================================
# 5. Verdict — r04's ordering: ask the Poisson question before the ratio question
# ===================================================================

sn5_pick(pre) = (i = findfirst(startswith(pre), sn5_out.rung); isnothing(i) ? nothing : sn5_out[i, :])
sn5_flat  = something(sn5_pick("C"), sn5_out[end, :])
sn5_tight = something(sn5_pick("A"), sn5_out[1, :])
sn5_lift  = sn5_flat.p_poisson / max(sn5_flat.p_pois_pri, 1e-9)

sn5_verdict =
    sn5_lift > 1.5 ?
        "POISSON — the data speaks and it says the parent was right on 718 too." :
    sn5_flat.ratio > 0.85 ?
        "NO INFORMATION — even the flat prior is unmoved; any over-dispersion priced is the prior's." :
        "OVER-DISPERSION IS REAL on 718 — the tight prior was binding and r is genuinely finite."

open(SN5_LOG, "w") do io
    println(io, "smile_negbin r05 — dispersion prior ladder, Ireland 718, 2026 biweek ", SN5_STEP)
    println(io, "run at ", now())
    @printf(io, "warmup %d, samples %d, max_depth %d, %d chain-tasks, %.1f min\n",
            SN5_WARMUP, SN5_SAMPLES, SN5_DEPTH, length(sn5_tasks), sn5_minutes)
    @printf(io, "goals-pillar effective N %.1f matches (%d goal counts)\n",
            sn5_eff_N, round(Int, 2 * sn5_eff_N))
    @printf(io, "model-free: conditional D %.4f (p %.4f, calib %.4f), marginal V/M %.4f\n\n",
            sn5_free.both.D, sn5_free.both.p, sn5_free.both.calib, sn5_free.marg_VM)
    show(io, MIME"text/plain"(), sn5_out)
    println(io)
    @printf(io, "\nratio  tight %.2f -> flat %.2f     Poisson-tail lift %.2fx\nVERDICT: %s\n",
            sn5_tight.ratio, sn5_flat.ratio, sn5_lift, sn5_verdict)
end

nothing
