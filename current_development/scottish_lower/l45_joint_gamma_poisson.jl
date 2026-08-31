# ==============================================================================
# l45 — LOADER for the two-arm (Gamma proxy xG + Poisson goals) joint model
# ==============================================================================
#
# Definitions only, no execution. Everything here is machinery the runners
# (r45 smoke, r46 grid) should be able to CALL without reading: arm assembly,
# proxy-xG coverage accounting, per-fold arm preflight, the smoke gates, and the
# leaderboard formatter.
#
# WHAT THE JOINT MODEL IS, IN ONE PARAGRAPH. One shared latent team intensity
# μ = exp(η) is read by two likelihoods at once: `pxg ~ Gamma(ν, μ/ν)` wherever a
# BBC proxy-xG measurement exists (23/24 onward), and `goals ~ Poisson(κ·μ)` on
# every match in the fold. The proxy arm sharpens μ on the seasons that have live
# text; the goals arm carries that sharpened μ back across the whole history. κ is
# the league finishing factor and is the headline diagnostic: κ > 1 says the league
# converts more than the pxG cell table expects, κ < 1 less.
#
# THE ONE GATE THAT MATTERS. If the proxy arm's mask is empty (or nearly so) on a
# fold, the joint model IS the baseline Poisson model carrying two extra parameters,
# and ν is sampling its prior. `l45_arm_preflight` measures that before any sampling
# starts, because discovering it from a leaderboard costs a grid run.
# ==============================================================================

using DataFrames
using Dates
using Printf
using Statistics
using LinearAlgebra
using MCMCChains
using DynamicPPL
using LogDensityProblems
using ReverseDiff
using ForwardDiff

const L45_PG       = BayesianFootball.Models.PreGame
const L45_FEATURES = BayesianFootball.Features

# ==============================================================================
# 1. ARM ASSEMBLY
# ==============================================================================
#
# The structural spine is written out in full on every arm rather than factored
# into a helper, because the comparability claim of both runners IS that these
# lines are identical across the five models. Only the covariate list changes.

"""
    l45_joint_arms(; half_life_days, observation, wealth_prior, distance_prior, curve)
        -> Vector{Tuple{String, Any}}

The five joint arms, in leaderboard order:

    m00_joint_baseline          spine only
    m02_joint_squad_wealth      + raw starting-XI wealth differential
    m03_joint_distance          + away-ground travel distance
    m04_joint_wealth_distance   + both
    m05_joint_production_wealth + age-adjusted production wealth (Richards sigmoid)

Every arm carries the SAME `observation`, so a difference in κ or ν between arms is
a statement about the covariate, not about the observation.
"""
function l45_joint_arms(;
    half_life_days::Float64 = 180.0,
    observation = JointGammaPoissonObservation(),
    wealth_prior = truncated(Normal(0.10, 0.05), lower = 0.0),
    distance_prior = truncated(Normal(0.04, 0.03), lower = 0.0),
    curve = RichardsSigmoid(23.0, 0.80, 2.0),
)
    m00 = CountModelBuilder(:m00_joint_baseline) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = half_life_days)) |>
        add(GlobalHomeAdvantage()) |>
        add(observation) |>
        build

    m02 = CountModelBuilder(:m02_joint_squad_wealth) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = half_life_days)) |>
        add(GlobalHomeAdvantage()) |>
        add(WealthCovariate(prior = wealth_prior)) |>
        add(observation) |>
        build

    m03 = CountModelBuilder(:m03_joint_distance) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = half_life_days)) |>
        add(GlobalHomeAdvantage()) |>
        add(DistanceCovariate(prior = distance_prior)) |>
        add(observation) |>
        build

    m04 = CountModelBuilder(:m04_joint_wealth_distance) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = half_life_days)) |>
        add(GlobalHomeAdvantage()) |>
        add(WealthCovariate(prior = wealth_prior)) |>
        add(DistanceCovariate(prior = distance_prior)) |>
        add(observation) |>
        build

    m05 = CountModelBuilder(:m05_joint_production_wealth) |>
        add(GlobalInterception()) |>
        add(TimeDecayDynamics(days_half_life = half_life_days)) |>
        add(GlobalHomeAdvantage()) |>
        add(ProductionWealthCovariate(
            feature = ProductionWealthFeature(curve = curve),
            prior   = wealth_prior)) |>
        add(observation) |>
        build

    # `Tuple{String,Any}` is deliberate. Each arm is a DIFFERENT concrete
    # `PoissonCountModel{...}` — its covariate tuple is baked into the type — so a
    # promoted eltype would depend on `typejoin`, and the runners `push!` a control arm
    # onto this vector afterwards.
    return Tuple{String,Any}[
        ("m00_joint_baseline",          m00),
        ("m02_joint_squad_wealth",      m02),
        ("m03_joint_distance",          m03),
        ("m04_joint_wealth_distance",   m04),
        ("m05_joint_production_wealth", m05),
    ]
end

"""
    l45_poisson_control(; half_life_days) -> PoissonCountModel

The single-arm control: the same spine and no proxy arm at all. Fitting it beside
the joint baseline is what turns "the joint model converged" into "the joint model
was worth fitting" — the two differ ONLY in the Gamma likelihood.
"""
l45_poisson_control(; half_life_days::Float64 = 180.0) =
    CountModelBuilder(:m00_poisson_control) |>
    add(GlobalInterception()) |>
    add(TimeDecayDynamics(days_half_life = half_life_days)) |>
    add(GlobalHomeAdvantage()) |>
    add(PoissonObservation()) |>
    build

# ==============================================================================
# 2. PROXY-xG COVERAGE ACCOUNTING
# ==============================================================================

"""
    l45_observation_coverage(ds, feature) -> DataFrame

One row per season: how many matches the Gamma arm can actually observe, and by
which rung of the measurement ladder. The seasonal breakdown is the point — BBC
live text starts in 23/24, so a store spanning 22/23 is EXPECTED to show a hard
coverage cliff, and a run that does not show one has a data problem, not a lucky
break.
"""
function l45_observation_coverage(ds::BayesianFootball.Data.DataStore,
                                  feature::MatchProxyXGFeature)
    matches = ds.matches
    nrow(matches) == 0 && return DataFrame()

    # Taken from the ladder directly, so the RUNG is visible per match. Reading only the
    # feature's emitted mask would report "100% covered" for a store whose coverage is
    # entirely shot-count pseudo-xG — which is a different measurement wearing the same name.
    observations = L45_FEATURES.pxg_match_observations(
        ds, L45_FEATURES.PxGFeature(k = feature.k, fallback = feature.fallback))

    by_season = Dict{String, Vector{Int}}()
    for row in eachrow(matches)
        push!(get!(by_season, String(row.season), Int[]), Int(row.match_id))
    end

    rows = NamedTuple[]
    for season in sort(collect(keys(by_season)))
        ids = by_season[season]
        commentary = 0
        shot_counts = 0
        values_h = Float64[]
        for id in ids
            obs = get(observations, id, nothing)
            obs === nothing && continue
            obs.source === :commentary && (commentary += 1)
            obs.source === :shot_counts && (shot_counts += 1)
            push!(values_h, obs.h)
        end
        observed = commentary + shot_counts
        push!(rows, (
            season = season,
            matches = length(ids),
            observed = observed,
            commentary = commentary,
            shot_counts = shot_counts,
            share = isempty(ids) ? 0.0 : observed / length(ids),
            commentary_share = isempty(ids) ? 0.0 : commentary / length(ids),
            mean_pxg_home = isempty(values_h) ? NaN : mean(values_h),
        ))
    end
    return DataFrame(rows)
end

"""
    l45_print_observation_coverage(table, counts)

The seasonal coverage table plus the ladder mix, printed before any sampling.
"""
function l45_print_observation_coverage(table::DataFrame)
    if nrow(table) == 0
        println("  (no matches in the store)")
        return nothing
    end
    @printf("  %-8s | %8s | %8s | %10s | %11s | %10s | %9s\n",
            "season", "matches", "observed", "commentary", "shot counts", "commentary",
            "mean pxG")
    println("  " * "-"^80)
    for r in eachrow(table)
        @printf("  %-8s | %8d | %8d | %10d | %11d | %9.1f%% | %9s\n",
                r.season, r.matches, r.observed, r.commentary, r.shot_counts,
                100 * r.commentary_share,
                isnan(r.mean_pxg_home) ? "—" : @sprintf("%.3f", r.mean_pxg_home))
    end
    total = sum(table.matches)
    observed = sum(table.observed)
    commentary = sum(table.commentary)
    @printf("  TOTAL: %d of %d matches observed (%.1f%%), of which %d are live-text commentary (%.1f%% of all matches)\n",
            observed, total, total == 0 ? 0.0 : 100 * observed / total,
            commentary, total == 0 ? 0.0 : 100 * commentary / total)

    # The reading that matters. `shots x league-average xG per shot` is a VOLUME measure with
    # no chance-quality content; a Gamma arm fed mostly that is not testing the two-arm
    # premise, it is regressing goals on shot counts through an extra parameter.
    if observed > 0 && commentary / observed < 0.5
        @printf("  [WARN] only %.1f%% of the observed matches are live-text commentary; the rest is\n",
                100 * commentary / observed)
        println("         shot-count pseudo-xG (shots x a league constant), which carries volume")
        println("         but no chance quality. Consider fallback = :none.")
    end
    return nothing
end

# ==============================================================================
# 3. PER-FOLD ARM PREFLIGHT
# ==============================================================================

"""
    _l45_extremum(f, od) -> Float64

`f` over the proxy values that actually reach the likelihood, or `NaN` when none do.

Written as its own function rather than `minimum(...; init = NaN)`: `min(NaN, x)` is
`NaN` in Julia, so an `init` sentinel would poison every non-empty answer as well as
the empty one.
"""
function _l45_extremum(f, od)
    od === nothing && return NaN
    covered = od.pxg_h[od.mask_weights .> 0.0]
    return isempty(covered) ? NaN : f(covered)
end

"""
    l45_arm_preflight(ds, model, splitter) -> DataFrame

For every fold: the size of each arm's evidence, taken from the design the engine
will actually receive rather than from the store.

`decayed_mask_share` is the honest number. The Gamma arm's contribution is weighted
by the same time-decay kernel as the goals arm, so a fold whose covered matches are
all old contributes far less than its raw count suggests. `mask_share` counts
matches; `decayed_mask_share` counts evidence.
"""
function l45_arm_preflight(ds::BayesianFootball.Data.DataStore, model, splitter)
    boundaries = BayesianFootball.Data.create_id_boundaries(ds, splitter)
    feature_sets = L45_FEATURES.create_features(boundaries, ds, model, splitter)

    rows = NamedTuple[]
    for (fold, (boundary, _)) in enumerate(boundaries)
        fs = first(feature_sets[fold])
        design = L45_PG.Builder.cb_design(model, fs)
        od = design.observation_data

        n = design.n_matches
        weight_total = sum(design.match_weights)
        push!(rows, (
            fold = fold,
            n_history = length(boundary.history_match_ids),
            n_target = length(boundary.target_match_ids),
            n_matches = n,
            n_observed = od === nothing ? 0 : od.n_observed,
            mask_share = (od === nothing || n == 0) ? 0.0 : od.n_observed / n,
            decayed_mask_share = (od === nothing || weight_total == 0.0) ? 0.0 :
                                 sum(od.mask_weights) / weight_total,
            min_pxg = _l45_extremum(minimum, od),
            max_pxg = _l45_extremum(maximum, od),
            covariates = join(string.(L45_PG.cb_covariate_names(model)), "+"),
        ))
    end
    return DataFrame(rows)
end

"""
    l45_print_arm_preflight(table; min_decayed_share)

The preflight table and the two warnings it exists to raise:

  * an EMPTY proxy arm — the model is the baseline plus two unidentified parameters;
  * a THIN proxy arm — ν is mostly prior and κ is doing all the work.
"""
function l45_print_arm_preflight(table::DataFrame; min_decayed_share::Float64 = 0.10)
    if nrow(table) == 0
        println("  (no folds)")
        return nothing
    end
    @printf("  %4s | %5s | %5s | %7s | %8s | %7s | %9s | %8s | %8s\n",
            "fold", "hist", "targ", "matches", "observed", "mask", "decayed", "min pxG", "max pxG")
    println("  " * "-"^90)
    for r in eachrow(table)
        @printf("  %4d | %5d | %5d | %7d | %8d | %6.1f%% | %8.1f%% | %8s | %8s\n",
                r.fold, r.n_history, r.n_target, r.n_matches, r.n_observed,
                100 * r.mask_share, 100 * r.decayed_mask_share,
                isnan(r.min_pxg) ? "—" : @sprintf("%.2f", r.min_pxg),
                isnan(r.max_pxg) ? "—" : @sprintf("%.2f", r.max_pxg))
    end

    scored = filter(r -> r.n_target > 0, table)
    empty_folds = [r.fold for r in eachrow(scored) if r.n_observed == 0]
    thin_folds = [r.fold for r in eachrow(scored)
                  if r.n_observed > 0 && r.decayed_mask_share < min_decayed_share]

    if !isempty(empty_folds)
        println("  [STOP] fold(s) $(join(empty_folds, ", ")) have an EMPTY proxy arm — " *
                "the joint model there is the baseline Poisson carrying ν and log κ as " *
                "unidentified parameters. Do not read κ from those folds.")
    elseif !isempty(thin_folds)
        @printf("  [WARN] fold(s) %s carry under %.0f%% decayed proxy evidence — ν is mostly prior.\n",
                join(thin_folds, ", "), 100 * min_decayed_share)
    else
        println("  [OK]   every scored fold carries a non-trivial proxy arm.")
    end
    return nothing
end

# ==============================================================================
# 3b. FITTING THE ARMS, TWO AT A TIME
# ==============================================================================
#
# WHY THIS EXISTS. `QueuedExecution` flattens FOLDS x CHAINS into one queue. The smoke
# splitter produces 2 folds and 4 chains, so a single arm offers the queue only 8 tasks —
# and on a 16-core box that measured a load average of 8.73, i.e. half the machine idle for
# the whole run. The grid runner has no such problem (40 folds x 4 chains = 160 tasks
# saturates 16 threads from the first wave); this is a smoke-test-only concern.
#
# THE CAP IS NOT OPTIONAL. Each `fit_model` builds its own `QueuedExecution` defaulting to
# `max_concurrent_tasks = nthreads()`. Spawning two arms without capping gives 2 x 16 = 32
# tasks on 16 PINNED cores, and oversubscribing pinned threads is worse than the idle it
# was meant to fix. `tasks_per_model` is therefore derived, not guessed.
#
# REPRODUCIBILITY CAVEAT. `run_sampler` does not seed explicitly, so each chain's stream
# comes from its task's RNG, which Julia derives from spawn order. Adding a model-level
# spawn layer changes that tree, so a parallel run is NOT bit-comparable with a sequential
# one. Statistically equivalent, not diffable — do not compare chains across the two modes.

"""
    l45_fit_arms(models, ds, splitter, sampler, save_root; concurrent_models, tasks_per_model)
        -> (fits::Dict, elapsed::Dict)

Fit every arm, `concurrent_models` at a time, each arm's queue capped so the total tasks in
flight is at most `Threads.nthreads()`.

`quiet = true` is forced: two arms sharing one terminal produce interleaved progress bars
that are worse than no progress bars. The runner prints a line per arm as it lands.
"""
function l45_fit_arms(models, ds, splitter, sampler, save_root;
                      concurrent_models::Int = 2,
                      tasks_per_model::Int = max(1, Threads.nthreads() ÷ max(1, concurrent_models)))
    fits = Dict{String, Any}()
    elapsed = Dict{String, Float64}()
    guard = ReentrantLock()
    slots = Base.Semaphore(max(1, concurrent_models))

    @printf("  %d arm(s), %d at a time, %d queue task(s) each (%d threads)\n",
            length(models), concurrent_models, tasks_per_model, Threads.nthreads())

    @sync for (name, model) in models
        Threads.@spawn begin
            Base.acquire(slots)
            try
                config = BayesianFootball.FitConfig(
                    name      = name,
                    model     = model,
                    splitter  = splitter,
                    sampler   = sampler,
                    execution = BayesianFootball.QueuedExecution(
                        max_concurrent_tasks = tasks_per_model),
                    save_dir  = joinpath(save_root, name),
                )
                started = time()
                fit = BayesianFootball.fit_model(config, ds; quiet = true)
                took = time() - started
                lock(guard) do
                    fits[name] = fit
                    elapsed[name] = took
                    @printf("    landed %-30s %6.0fs  R̂ %.4f  div %d\n",
                            name, took, fit.diagnostics.max_rhat, fit.diagnostics.n_divergent)
                end
            catch err
                lock(guard) do
                    @printf("    FAILED %-30s %s\n", name, sprint(showerror, err))
                end
                rethrow()
            finally
                Base.release(slots)
            end
        end
    end
    return fits, elapsed
end

# ==============================================================================
# 4. SAFE CHAIN LOOKUPS
# ==============================================================================

"The posterior mean of one site, or `NaN` when this arm does not carry it."
function l45_chain_mean(chain::Chains, site::Symbol)
    site in Set(Symbol.(names(chain))) || return NaN
    return mean(chain[site])
end

"Home advantage lives under three different site names across the component library."
function l45_home_advantage(chain::Chains)
    for site in (Symbol("ha.γ"), Symbol("ha.γ_global"), Symbol("ha.γ_raw"))
        value = l45_chain_mean(chain, site)
        isnan(value) || return value
    end
    return NaN
end

"""
    l45_finishing_factor(chain) -> Float64

κ = exp(mean(log κ)). Reported on the log scale's own mean rather than as
`mean(exp(log κ))`, so it is the posterior median of a log-normal-ish quantity and
is not pulled up by the tail.
"""
l45_finishing_factor(chain::Chains) = exp(l45_chain_mean(chain, Symbol("obs.log_κ")))

"ν, the Gamma precision. Higher means the proxy measurement pins μ more tightly."
l45_proxy_precision(chain::Chains) = l45_chain_mean(chain, Symbol("obs.ν"))

# ==============================================================================
# 5. THE SMOKE GATES
# ==============================================================================

"Uniform gate row, shaped so `l45_print_gates` can print any of them the same way."
l45_gate(name, pass::Bool, detail) = (; name = String(name), pass, detail = String(detail))

"""
    l45_smoke_gates(name, fit; max_rhat, kappa_band, nu_band) -> Vector

What a single-fold smoke fit must show before a 40-fold grid is worth starting.

  * CONVERGENCE — R̂ below `max_rhat` and no divergences. Anything else and the rest
    of the table is describing a chain that did not mix.
  * κ IN BAND — pxG is already calibrated in goal units, so κ should sit near 1. A κ
    far from 1 is either a genuine finishing effect or a units bug in the cell table,
    and the two are indistinguishable from the posterior alone; the band flags it for
    a human rather than deciding.
  * ν IN BAND — ν is the proxy arm's precision. At the prior mean the Gamma arm says
    a 1.5 xG performance has an sd of about 0.75 goals, which is the right order.
  * LATENT SEPARATION — λ and μ must actually differ by κ. If they come back equal,
    `_cb_rates` fell through to the plain-Poisson method and the joint arm's
    diagnostics are silently mirroring the wrong quantity.
"""
function l45_smoke_gates(name::AbstractString, fit;
                         fold::Int = 1,
                         max_rhat::Float64 = 1.05,
                         kappa_band::Tuple{Float64,Float64} = (0.60, 1.60),
                         nu_band::Tuple{Float64,Float64} = (1.0, 12.0))
    diagnostics = fit.diagnostics
    chain = fit.folds[fold].chain

    rhat = diagnostics.max_rhat
    divergences = diagnostics.n_divergent
    κ = l45_finishing_factor(chain)
    ν = l45_proxy_precision(chain)

    gates = [
        # The FRAMEWORK'S OWN verdict, first. R̂ and divergences are two of six gates
        # `audit_convergence` applies (ESS, BFMI and tree-depth saturation are the others),
        # so checking only the two this file happens to know about produced a run that
        # printed "ALL GATES PASSED" directly above "Convergence: FAILED for ...". A smoke
        # test that can contradict itself is worse than one that reports nothing.
        l45_gate("$name · convergence audit",
                 diagnostics.passed,
                 diagnostics.passed ? "all $(length(diagnostics.folds)) fold(s) passed every gate" :
                 "FAILED: " * (isempty(diagnostics.failed_gates) ? "(no gate names recorded)" :
                               join(diagnostics.failed_gates, ", ")) *
                 (isempty(diagnostics.failures) ? "" : "  |  " * join(diagnostics.failures, "; "))),
        l45_gate("$name · R̂ < $max_rhat",
                 !isnan(rhat) && rhat < max_rhat,
                 isnan(rhat) ? "R̂ unavailable" : @sprintf("max R̂ = %.4f", rhat)),
        l45_gate("$name · no divergences",
                 divergences == 0,
                 "$(divergences) divergent transition(s)"),
        l45_gate("$name · BFMI ≥ $(diagnostics.thresholds.min_bfmi)",
                 !isnan(diagnostics.min_bfmi) && diagnostics.min_bfmi >= diagnostics.thresholds.min_bfmi,
                 @sprintf("min BFMI = %.4f (worst fold %d)",
                          diagnostics.min_bfmi, diagnostics.worst_bfmi_fold)),
        l45_gate("$name · tree-depth saturation ≤ $(100 * diagnostics.thresholds.max_treedepth_rate)%",
                 diagnostics.treedepth_rate <= diagnostics.thresholds.max_treedepth_rate,
                 @sprintf("%d transitions capped at depth %d (%.2f%%)",
                          diagnostics.n_depth_capped, diagnostics.max_tree_depth,
                          100 * diagnostics.treedepth_rate)),
        l45_gate("$name · min ESS ≥ $(diagnostics.thresholds.min_ess)",
                 !isnan(diagnostics.min_ess_bulk) && diagnostics.min_ess_bulk >= diagnostics.thresholds.min_ess,
                 @sprintf("bulk %.0f (fold %d), tail %.0f",
                          diagnostics.min_ess_bulk, diagnostics.worst_ess_bulk_fold,
                          diagnostics.min_ess_tail)),
        l45_gate("$name · κ in $(kappa_band)",
                 !isnan(κ) && kappa_band[1] <= κ <= kappa_band[2],
                 isnan(κ) ? "obs.log_κ absent from the chain" :
                 @sprintf("κ = %.4f (log κ = %+.4f)", κ, log(κ))),
        l45_gate("$name · ν in $(nu_band)",
                 !isnan(ν) && nu_band[1] <= ν <= nu_band[2],
                 isnan(ν) ? "obs.ν absent from the chain" : @sprintf("ν = %.3f", ν)),
    ]
    return gates
end

"""
    l45_first_scored_fold(boundaries) -> Int

The index of the first boundary that actually carries target matches.

NOT always 1. With `end_dynamics = 1, stop_early = true` the leading boundary is a
pure-history warm-up with an empty target block, so a gate that reads fold 1 blindly
reports a failure about the SPLITTER while looking like a failure about the model.
"""
function l45_first_scored_fold(boundaries)
    for (i, (boundary, _)) in enumerate(boundaries)
        isempty(boundary.target_match_ids) || return i
    end
    error("no boundary carries target matches; the splitter produced only warm-up folds")
end

"""
    l45_fold_fixtures(ds, boundary) -> DataFrame

The fold's held-out fixtures, in match-id order. Rebuilt from the boundary rather than
read off the `Fit`, because `FoldFit` carries only `(fold, chain, meta)` — the frame the
latents were extracted against is not kept on it.
"""
function l45_fold_fixtures(ds::BayesianFootball.Data.DataStore, boundary)
    target = Set(Int.(boundary.target_match_ids))
    frame = filter(row -> Int(row.match_id) in target, ds.matches)
    return sort(frame, :match_id)
end

"""
    l45_latent_gate(name, model, chain, feature_set, oos) -> gate row

Extract the fold's rates and check the joint arm's own invariant: λ = κ·μ, with κ the
same number the chain reports. A pass means the score grid is being fed GOAL intensities
while the diagnostics are fed xG intensities, which is the whole reason the two are kept
apart. A silent fall-through to the plain-Poisson `_cb_rates` method would make them
equal, and nothing downstream would notice.
"""
function l45_latent_gate(name::AbstractString, model, chain::Chains, feature_set,
                         oos::AbstractDataFrame)
    nrow(oos) == 0 && return l45_gate("$name · λ = κ·μ", false, "fold has no OOS fixtures")

    κ = l45_finishing_factor(chain)
    rates = L45_PG.extract_parameters(model, oos, feature_set, chain)
    probe = rates[Int(first(oos.match_id))]

    haskey(probe, :μ_h) || return l45_gate("$name · λ = κ·μ", false,
        "extract_parameters returned $(keys(probe)) — the joint `_cb_rates` method did not fire")

    ratio = mean(probe.λ_h ./ probe.μ_h)
    ok = isapprox(ratio, mean(probe.κ); rtol = 1e-8) &&
         all(isfinite, probe.λ_h) && all(>(0.0), probe.λ_h)
    return l45_gate("$name · λ = κ·μ", ok,
        @sprintf("mean λ_h/μ_h = %.6f, chain κ = %.6f", ratio, κ))
end

"""
    l45_identification_gate(name, chain, observation; min_shrinkage) -> (gates, row)

Is the proxy arm actually informing ν and κ, or are they sampling their priors?

THE QUESTION THIS EXISTS TO ANSWER. The first smoke run returned ν = 4.00, 4.00, 4.01,
4.02, 4.02 across five independent fits — against a prior whose mean is exactly 4.0. A
posterior MEAN cannot distinguish "the prior was well chosen" from "the likelihood said
nothing", and the two have opposite consequences: in the second case the joint model is a
Poisson model carrying two spare parameters, converging beautifully and meaning nothing.

The discriminator is the posterior SD against the prior SD. A parameter the data has
identified contracts; one sampling its prior does not. `shrinkage = 1 - sd_post/sd_prior`
is that contraction: ~0 means the arm contributed nothing, ~1 means it dominated.

ν is a GATE because an unidentified ν invalidates the whole two-arm premise. κ is reported
but not gated — κ is identified by the GOALS arm, which is always present, so it tells you
about the model rather than about the proxy feed.
"""
function l45_identification_gate(name::AbstractString, chain::Chains, observation;
                                 min_shrinkage::Float64 = 0.5)
    observation isa JointGammaPoissonObservation ||
        return (NamedTuple[], nothing)

    nu_draws = Symbol("obs.ν") in Set(Symbol.(names(chain))) ?
               vec(Array(chain[Symbol("obs.ν")])) : Float64[]
    lk_draws = Symbol("obs.log_κ") in Set(Symbol.(names(chain))) ?
               vec(Array(chain[Symbol("obs.log_κ")])) : Float64[]

    isempty(nu_draws) && return ([l45_gate("$name · ν identified", false,
                                           "obs.ν absent from the chain")], nothing)

    nu_prior_sd = std(observation.shape_prior)
    lk_prior_sd = std(observation.log_kappa_prior)
    nu_post_sd = std(nu_draws)
    lk_post_sd = isempty(lk_draws) ? NaN : std(lk_draws)

    nu_shrink = 1.0 - nu_post_sd / nu_prior_sd
    lk_shrink = isnan(lk_post_sd) ? NaN : 1.0 - lk_post_sd / lk_prior_sd

    row = (name = String(name),
           nu_mean = mean(nu_draws), nu_post_sd = nu_post_sd,
           nu_prior_sd = nu_prior_sd, nu_shrinkage = nu_shrink,
           kappa_log_mean = isempty(lk_draws) ? NaN : mean(lk_draws),
           kappa_post_sd = lk_post_sd, kappa_prior_sd = lk_prior_sd,
           kappa_shrinkage = lk_shrink)

    gates = [
        l45_gate("$name · ν identified by the proxy arm", nu_shrink >= min_shrinkage,
                 @sprintf("ν = %.3f ± %.3f against prior sd %.3f — shrinkage %.1f%%%s",
                          row.nu_mean, nu_post_sd, nu_prior_sd, 100 * nu_shrink,
                          nu_shrink < min_shrinkage ?
                          "  (ν IS SAMPLING ITS PRIOR — the Gamma arm is contributing nothing)" : "")),
    ]
    return (gates, row)
end

"""
    l45_print_identification(rows)

The ν / κ identification table. Printed whether or not the gate passed, because the
NUMBERS are the finding — a borderline shrinkage is a different conversation from a zero.
"""
function l45_print_identification(rows)
    isempty(rows) && return nothing
    @printf("  %-28s | %16s | %10s | %16s | %10s\n",
            "Model", "ν (post sd)", "ν shrink", "log κ (post sd)", "κ shrink")
    println("  " * "-"^92)
    for r in rows
        r === nothing && continue
        @printf("  %-28s | %7.3f (%6.3f) | %9.1f%% | %7.4f (%6.4f) | %9.1f%%\n",
                r.name, r.nu_mean, r.nu_post_sd, 100 * r.nu_shrinkage,
                r.kappa_log_mean, r.kappa_post_sd, 100 * r.kappa_shrinkage)
    end
    return nothing
end

"""
    l45_clamp_gate(name, model, chain, feature_set, oos) -> gate row

Does the rate guard actually bind at the posterior draws this model produces?

docs/turing_ad_performance_guide.md §9 asks this of every engine, and the joint arm is the
one that most needs it asked. `clamp` is a value-dependent branch, so a compiled tape
recorded outside the clamp is only valid while sampling stays outside it. The joint arm
also carries `exp(-η)`: the guard's LOWER bound is the one under pressure, and a trajectory
pressed against it is both a numerically suspect tape and a sign the Gamma arm is fighting
the goals arm over the same intensity.

η is read back as `log μ` from the fitted rates, so these are genuine posterior draws over
the fixtures the model actually prices — not prior draws standing in for them.
"""
function l45_clamp_gate(name::AbstractString, model, chain::Chains, feature_set,
                        oos::AbstractDataFrame; margin::Float64 = 0.5)
    guard = model.guard
    guard isa L45_PG.Builder.ClampGuard ||
        return l45_gate("$name · guard headroom", true, "NoGuard — nothing to bind")
    nrow(oos) == 0 && return l45_gate("$name · guard headroom", false, "no OOS fixtures")

    rates = L45_PG.extract_parameters(model, oos, feature_set, chain)
    lo_worst = Inf      # smallest η seen, distance above guard.lo is the headroom
    hi_worst = -Inf
    for id in Int.(oos.match_id)
        r = rates[id]
        μ = haskey(r, :μ_h) ? (r.μ_h, r.μ_a) : (r.λ_h, r.λ_a)
        for side in μ
            η = log.(side)
            lo_worst = min(lo_worst, minimum(η))
            hi_worst = max(hi_worst, maximum(η))
        end
    end

    low_head = lo_worst - guard.lo
    high_head = guard.hi - hi_worst
    ok = low_head > margin && high_head > margin
    return l45_gate("$name · guard headroom", ok,
        @sprintf("η ∈ [%+.3f, %+.3f] against clamp [%.1f, %.1f] — headroom %.3f below, %.3f above",
                 lo_worst, hi_worst, guard.lo, guard.hi, low_head, high_head))
end

"""
    l45_ad_audit(name, model, feature_set; max_gradient_ms) -> (gates, row)

The AD guide's §10.1 and §10.3 blocks, run on a REAL fold rather than a toy one.

Reports tape instruction count and the warmed MINIMUM gradient time (§10.1 is explicit that
a cold median is noise), and checks the compiled tape against a fresh ReverseDiff gradient
and against ForwardDiff — including at perturbed points, which is the check that catches a
branch smuggled into the tape.

Returns the gate rows and a NamedTuple of the raw numbers, so the runner can print a table
without re-running anything.
"""
function l45_ad_audit(name::AbstractString, model, feature_set;
                      max_gradient_ms::Float64 = 0.10,
                      reps::Int = 200, warmup::Int = 50)
    turing_model = L45_PG.build_turing_model(model, feature_set)
    vi = DynamicPPL.VarInfo(turing_model)
    turing_model(vi)
    θ = copy(vi[:])
    lf = DynamicPPL.LogDensityFunction(turing_model)
    f = x -> LogDensityProblems.logdensity(lf, x)

    raw = ReverseDiff.GradientTape(f, θ)
    tape = ReverseDiff.compile(raw)
    g = similar(θ)
    for _ in 1:warmup
        ReverseDiff.gradient!(g, tape, θ)
    end
    gradient_ms = 1e3 * minimum(@elapsed(ReverseDiff.gradient!(g, tape, θ)) for _ in 1:reps)

    g_tape = (h = similar(θ); ReverseDiff.gradient!(h, tape, θ); h)
    g_fresh = ReverseDiff.gradient(f, θ)
    g_forward = ForwardDiff.gradient(f, θ)
    relerr(a, b) = norm(a - b) / max(norm(a), norm(b), 1.0)

    err_fresh = relerr(g_fresh, g_tape)
    err_forward = relerr(g_tape, g_forward)

    # The same COMPILED tape at other points. A value branch baked into the tape shows up
    # here and nowhere else.
    err_perturbed = 0.0
    for δ in (0.001, -0.002, 0.003)
        θp = θ .+ δ .* sin.(collect(eachindex(θ)))
        gp = similar(θp)
        ReverseDiff.gradient!(gp, tape, θp)
        err_perturbed = max(err_perturbed, relerr(gp, ReverseDiff.gradient(f, θp)))
    end

    row = (name = String(name), instructions = length(raw.tape), n_params = length(θ),
           gradient_ms = gradient_ms, err_fresh = err_fresh,
           err_forward = err_forward, err_perturbed = err_perturbed,
           logdensity = f(θ))

    gates = [
        l45_gate("$name · logdensity finite", isfinite(row.logdensity),
                 @sprintf("log p = %.4f at %d parameters", row.logdensity, row.n_params)),
        l45_gate("$name · compiled == fresh ReverseDiff", err_fresh <= 1e-8,
                 @sprintf("relerr %.3e", err_fresh)),
        l45_gate("$name · ReverseDiff == ForwardDiff", err_forward <= 1e-6,
                 @sprintf("relerr %.3e", err_forward)),
        l45_gate("$name · compiled tape valid off-point", err_perturbed <= 1e-8,
                 @sprintf("worst relerr %.3e over 3 perturbations", err_perturbed)),
        l45_gate("$name · gradient < $(max_gradient_ms) ms", gradient_ms < max_gradient_ms,
                 @sprintf("%.4f ms over %d tape instructions", gradient_ms, row.instructions)),
    ]
    return gates, row
end

"""
    l45_print_gates(gates) -> Bool

Print every gate and return whether all of them passed. Printing the whole table
rather than throwing on the first failure is deliberate: a smoke run should tell you
everything that is wrong in one pass.
"""
function l45_print_gates(gates)
    width = maximum(length(g.name) for g in gates)
    for g in gates
        @printf("  [%s] %-*s  %s\n", g.pass ? "PASS" : "FAIL", width, g.name, g.detail)
    end
    passed = all(g.pass for g in gates)
    println(passed ? "\n  ALL GATES PASSED." :
            "\n  $(count(g -> !g.pass, gates)) GATE(S) FAILED — do not start the grid run.")
    return passed
end

# ==============================================================================
# 6. LEADERBOARD
# ==============================================================================

"""
    l45_summarise_fit(name, fit, ds, elapsed) -> NamedTuple

Convergence, the two joint-arm parameters, the covariate weights this arm carries,
and out-of-sample proper scores.
"""
function l45_summarise_fit(name::AbstractString, fit, ds, elapsed::Float64; fold::Int = 1)
    diagnostics = fit.diagnostics
    chain = fit.folds[fold].chain
    report = BayesianFootball.evaluate_predictions(fit, ds)

    return (
        name = String(name),
        elapsed = elapsed,
        passed = diagnostics.passed,
        max_rhat = diagnostics.max_rhat,
        min_ess = diagnostics.min_ess_bulk,
        divergences = diagnostics.n_divergent,
        gamma = l45_home_advantage(chain),
        kappa = l45_finishing_factor(chain),
        nu = l45_proxy_precision(chain),
        w_wealth = l45_chain_mean(chain, Symbol("wealth.w")),
        w_dist = l45_chain_mean(chain, Symbol("distance.w")),
        w_prod = l45_chain_mean(chain, Symbol("production_wealth.w")),
        logloss = report.model.logloss,
        brier = report.model.brier,
        ece = report.model.ece,
        rps = report.model.rps,
        market_logloss = report.market.logloss,
    )
end

l45_fmt(value, spec) = isnan(value) ? "—" : Printf.format(Printf.Format(spec), value)

"""
    l45_print_leaderboard(rows; baseline) -> DataFrame

Arms sorted by out-of-sample log loss, with the delta against the named baseline.
The delta is the decision quantity: an arm that does not beat the baseline on
genuine walk-forward log loss has not earned its parameter.
"""
function l45_print_leaderboard(rows; baseline::AbstractString = "m00_joint_baseline")
    ranked = sort(collect(rows), by = r -> isnan(r.logloss) ? Inf : r.logloss)
    base_idx = findfirst(r -> r.name == baseline, ranked)
    base_ll = base_idx === nothing ? NaN : ranked[base_idx].logloss

    println("="^146)
    @printf(" %-28s | %6s | %5s | %6s | %4s | %7s | %6s | %6s | %7s | %7s | %8s | %9s\n",
            "Model", "Time", "R-hat", "ESS", "Div", "γ (HA)", "κ", "ν",
            "w_wealth", "w_dist", "LogLoss", "ΔLogLoss")
    println("-"^146)
    for r in ranked
        delta = isnan(base_ll) || isnan(r.logloss) ? NaN : r.logloss - base_ll
        @printf(" %-28s | %5.0fs | %5s | %6s | %4d | %7s | %6s | %6s | %7s | %7s | %8s | %9s\n",
                r.name, r.elapsed,
                l45_fmt(r.max_rhat, "%.3f"),
                isnan(r.min_ess) ? "—" : string(Int(round(r.min_ess))),
                r.divergences,
                l45_fmt(r.gamma, "%+.3f"),
                l45_fmt(r.kappa, "%.3f"),
                l45_fmt(r.nu, "%.2f"),
                l45_fmt(r.w_wealth, "%+.3f"),
                l45_fmt(r.w_dist, "%+.3f"),
                l45_fmt(r.logloss, "%.4f"),
                l45_fmt(delta, "%+.4f"))
    end
    println("="^146)

    failed = [r.name for r in ranked if !r.passed]
    if isempty(failed)
        println(" Convergence: all arms passed.")
    else
        println(" Convergence: FAILED for $(join(failed, ", ")) — their scores are not comparable.")
    end
    return DataFrame(ranked)
end
