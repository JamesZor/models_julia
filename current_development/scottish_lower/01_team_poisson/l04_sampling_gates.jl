# ==============================================================================
# Model 01 — GATE 3 : SAMPLING
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# Gate 3 asks three separate questions, in increasing cost:
#
#   (a) equation parity   is the Turing model the model MODEL.md documents?
#   (b) gradient health   is the AD path correct and fast enough to sample?
#   (c) smoke run         does it actually sample, and converge?
#
# (a) is the load-bearing one. It compares DynamicPPL's log density against
# l02_equations.jl — an independent implementation written from the documentation
# rather than from the engine. If those agree, the fitted model and the documented
# model are the same object, and gate 4 can then check the priced model against
# the same reference.
#
# The smoke run persists a real chain through src/experiments. That artifact is
# the input to gate 4 — nothing here is throwaway.
#
# ==============================================================================

using BayesianFootball
using DynamicPPL
using LogDensityProblems
using ForwardDiff
using ReverseDiff
using LinearAlgebra
using Random
using Statistics
using Printf

const PGm = BayesianFootball.Models.PreGame


# ==============================================================================
# 1. Prior draws
# ==============================================================================

"""
    tp_params_from_varinfo(vi) -> TPParams

Read a draw out of a `VarInfo` BY NAME.

Deliberately not by position. A positional mapping would silently reinterpret the
vector if DynamicPPL ever reordered sites, which is precisely the class of error
this gate exists to detect.
"""
function tp_params_from_varinfo(vi)
    v = Dict(string(k) => vi[k] for k in keys(vi))
    return TPParams(
        μ     = v["inter.μ"],
        log_r = v["disp.log_r"],
        γ     = v["ha.γ_global"],
        σ_a   = v["dyn.σ_a"],
        σ_d   = v["dyn.σ_d"],
        raw_a = Vector{Float64}(v["dyn.raw_a"]),
        raw_d = Vector{Float64}(v["dyn.raw_d"]),
    )
end

"""
    tp_prior_draw(model, fs; seed) -> NamedTuple

One seeded prior draw, returned in both representations: the flat vector
DynamicPPL scores, and the named `TPParams` l02 scores.
"""
function tp_prior_draw(model, fs; seed::Int = 20260825)
    turing_model = PGm.build_turing_model(model, fs)
    Random.seed!(seed)
    vi = DynamicPPL.VarInfo(turing_model)
    turing_model(vi)

    return (
        turing_model = turing_model,
        vi           = vi,
        θ            = copy(vi[:]),
        params       = tp_params_from_varinfo(vi),
    )
end

"""
    tp_logdensity_fn(turing_model) -> Function

`θ -> log density`, in the model's ORIGINAL space.

`DynamicPPL.VarInfo(model)` is unlinked, so no Jacobian corrections appear and the
Gamma-distributed scales are scored directly. Verified 2026-08-25: parity with
`tp_logjoint` is exact, which would not hold if the space were linked.
"""
function tp_logdensity_fn(turing_model)
    lf = DynamicPPL.LogDensityFunction(turing_model)
    return x -> LogDensityProblems.logdensity(lf, x)
end


# ==============================================================================
# 2. GATE 3a — Equation parity
# ==============================================================================

"""
    tp_gate_equation_parity(model, fs; seeds)

Compare DynamicPPL's log density against `l02_equations.jl` at several independent
prior draws.

Several draws, not one: a single point can agree by coincidence — for instance if
a term that should scale a parameter happens to be near 1. Independent draws are
used rather than perturbations of one draw so that no assumption is made about the
ordering of the flat parameter vector.
"""
function tp_gate_equation_parity(model, fs; seeds = [20260825, 7, 991])
    tp_assert_default(model)
    data = tp_equation_data(fs)
    hl   = model.dynamics_config.days_half_life

    diffs = Float64[]
    for s in seeds
        draw = tp_prior_draw(model, fs; seed = s)
        f    = tp_logdensity_fn(draw.turing_model)
        push!(diffs, f(draw.θ) - tp_logjoint(draw.params, data, model))
    end

    worst = maximum(abs, diffs)
    results = [(
        name   = "log density parity (Turing vs l02)",
        pass   = worst <= 1e-8,
        detail = @sprintf("max |Δ| = %.3e over %d prior draws", worst, length(seeds)),
    )]

    # The sampled-site manifest must match what MODEL.md claims and what gate 4
    # will later read out of the chain.
    draw = tp_prior_draw(model, fs; seed = first(seeds))
    observed = Set(string.(keys(draw.vi)))
    expected = Set(["inter.μ", "disp.log_r", "ha.γ_global",
                    "dyn.σ_a", "dyn.σ_d", "dyn.raw_a", "dyn.raw_d"])
    push!(results, (
        name   = "sampled-site manifest",
        pass   = observed == expected,
        detail = observed == expected ? "$(length(observed)) sites, as documented" :
                 "unexpected: $(sort(collect(symdiff(observed, expected))))",
    ))

    n_teams = Int(fs.data[:n_teams])
    push!(results, (
        name   = "parameter count",
        pass   = length(draw.θ) == 5 + 2 * n_teams,
        detail = "$(length(draw.θ)) = 5 scalars + 2 x $n_teams team effects",
    ))

    return results
end


# ==============================================================================
# 3. GATE 3b — Gradient health
# ==============================================================================

_tp_relerr(a, b) = norm(a - b) / max(norm(a), norm(b), 1.0)

"""
    tp_gate_gradients(model, fs; seed, probes) -> (results, artifacts)

Four independent routes to the same gradient:

  fresh ReverseDiff     re-records control flow every call
  compiled ReverseDiff  the tape NUTS will actually replay
  ForwardDiff           a structurally different AD mode
  finite differences    no AD at all

A compiled tape is a STATIC recording. If the model contains a data-dependent
branch, the tape silently keeps the branch taken at record time and stays wrong at
other parameter values — which is why agreement is also checked at perturbed
points, not only where the tape was recorded.

Thresholds follow the repository's AD guide: fresh/compiled 1e-8, AD-mode 1e-6,
finite differences 1e-4.
"""
function tp_gate_gradients(model, fs; seed::Int = 20260825, probes = [0.001, -0.002, 0.003])
    draw = tp_prior_draw(model, fs; seed = seed)
    f    = tp_logdensity_fn(draw.turing_model)
    θ    = draw.θ

    results = []

    push!(results, (
        name   = "log density finite at prior draw",
        pass   = isfinite(f(θ)),
        detail = @sprintf("logdensity = %.4f", f(θ)),
    ))

    compile_s = @elapsed tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, θ))

    g_compiled = similar(θ)
    ReverseDiff.gradient!(g_compiled, tape, θ)
    g_fresh   = ReverseDiff.gradient(f, θ)
    g_forward = ForwardDiff.gradient(f, θ)

    push!(results, (
        name   = "gradients finite",
        pass   = all(isfinite, g_compiled) && all(isfinite, g_fresh) && all(isfinite, g_forward),
        detail = "compiled / fresh / ForwardDiff all finite",
    ))

    e_tape = _tp_relerr(g_fresh, g_compiled)
    push!(results, (
        name   = "compiled tape == fresh ReverseDiff",
        pass   = e_tape <= 1e-8,
        detail = @sprintf("relerr = %.3e", e_tape),
    ))

    e_mode = _tp_relerr(g_compiled, g_forward)
    push!(results, (
        name   = "ReverseDiff == ForwardDiff",
        pass   = e_mode <= 1e-6,
        detail = @sprintf("relerr = %.3e", e_mode),
    ))

    # Finite differences at indices spread across the parameter vector.
    n_teams = Int(fs.data[:n_teams])
    idx = unique([1, 4, 6, 6 + n_teams ÷ 2, length(θ)])
    worst_fd = 0.0
    for k in idx
        ε = 1e-6
        xp = copy(θ); xp[k] += ε
        xm = copy(θ); xm[k] -= ε
        central = (f(xp) - f(xm)) / (2ε)
        worst_fd = max(worst_fd, abs(g_compiled[k] - central))
    end
    push!(results, (
        name   = "finite differences agree",
        pass   = worst_fd <= 1e-4,
        detail = @sprintf("max |Δ| = %.3e at indices %s", worst_fd, string(idx)),
    ))

    # Static-tape safety away from the recording point.
    worst_probe = 0.0
    for δ in probes
        θp = θ .+ δ .* sin.(collect(eachindex(θ)))
        gp = similar(θp)
        ReverseDiff.gradient!(gp, tape, θp)
        worst_probe = max(worst_probe, _tp_relerr(ReverseDiff.gradient(f, θp), gp))
    end
    push!(results, (
        name   = "static tape safe at perturbed points",
        pass   = worst_probe <= 1e-8,
        detail = @sprintf("max relerr = %.3e over %d probes", worst_probe, length(probes)),
    ))

    # Timing is reported, never gating: a slow model is a cost, a wrong one is a bug.
    g_bench = similar(θ)
    times = Float64[]
    for _ in 1:50
        push!(times, @elapsed ReverseDiff.gradient!(g_bench, tape, θ))
    end
    med_ms = median(times) * 1e3
    push!(results, (
        name   = "compiled gradient latency",
        pass   = true,
        detail = @sprintf("median %.3f ms (compile %.2f s) — target < 1 ms", med_ms, compile_s),
    ))

    return (results, (; θ, tape, f, median_ms = med_ms))
end


# ==============================================================================
# 4. GATE 3c — Smoke run
# ==============================================================================
#
# One fold, sampled and PERSISTED through src/experiments. The saved artifact is
# the input to gate 4, so this is not a throwaway check.
#
# This is the only MCMC in the walkthrough. James runs it.

using MCMCChains
using DataFrames
using Statistics

const Experiments = BayesianFootball.Experiments
const Training    = BayesianFootball.Training
const Samplers    = BayesianFootball.Samplers

"""
    tp_smoke_splitter(contract) -> GroupedCVConfig

The contract's splitter bounded to a SINGLE fold via `end_dynamics = 0`.

This yields the season-opening fold: fitted on prior seasons only, predicting the
first observed block of the target season. It is the smallest and least
representative fold in the set, and it is chosen because it is the only way to get
exactly one fold through the real `run_experiment` path rather than assembling a
one-element FeatureCollection by hand.

Consequence for reading the result: a passing smoke here is NECESSARY, not
sufficient. Later folds carry ~1060 rows against this one's 720, so both gradient
cost and sampling time scale up from whatever this reports.
"""
function tp_smoke_splitter(contract::SLContract)
    return Data.GroupedCVConfig(
        tournament_groups = [contract.tournaments],
        target_seasons    = contract.dev_seasons,
        history_seasons   = contract.history_seasons,
        dynamics_col      = contract.dynamics_col,
        warmup_period     = contract.warmup_period,
        stop_early        = contract.stop_early,
        end_dynamics      = 0,
    )
end

"""
    tp_smoke_config(model, contract; save_dir) -> ExperimentConfig

`ExperimentConfig` is built directly rather than through `create_experiment_task`,
because that helper does not expose `end_dynamics` and would therefore run every
fold.
"""
function tp_smoke_config(model, contract::SLContract; save_dir::AbstractString)
    sampler = Samplers.QueuedNUTSConfig(
        n_samples      = contract.smoke_samples,
        n_chains       = contract.smoke_chains,
        n_warmup       = contract.smoke_warmup,
        accept_rate    = contract.accept_rate,
        max_depth      = contract.max_depth,
        initialisation = Samplers.UniformInit(-contract.init_range, contract.init_range),
        show_progress  = false,
    )

    execution = Training.Independent(
        parallel             = true,
        max_concurrent_tasks = contract.smoke_chains,
    )

    return Experiments.ExperimentConfig(
        name            = "tp01_smoke_$(sl_hash(model))",
        model           = model,
        splitter        = tp_smoke_splitter(contract),
        training_config = Training.TrainingConfig(sampler, execution, nothing, false),
        save_dir        = save_dir,
        description     = "Model 01 gate-3 smoke: one fold, persisted for gate 4.",
    )
end

"""
    tp_run_smoke(ds, model, contract) -> (results, path)

RUNS MCMC. Roughly `smoke_chains` chains x `smoke_warmup + smoke_samples`
iterations on one fold.

Saves through `Experiments.save_experiment`, so gate 4 can reload the exact
artifact rather than a hand-built object.
"""
function tp_run_smoke(ds, model, contract::SLContract)
    save_dir = sl_artifact_dir(contract, "01_team_poisson", sl_hash(model))
    config   = tp_smoke_config(model, contract; save_dir = save_dir)

    results = Experiments.run_experiment(ds, config)
    path    = Experiments.save_experiment(results)

    return (results, path)
end


# ==============================================================================
# 5. Convergence diagnostics
# ==============================================================================

"""
    tp_bfmi(chain) -> Vector{Float64}

Bayesian fraction of missing information, per chain, from the sampler's
`hamiltonian_energy` internal:

    BFMI = Σ (E_t − E_{t−1})² / (N · Var(E))

Computed here because `MCMCDiagnosticTools` is not a direct dependency. Values
below ~0.3 indicate the sampler is exploring the energy distribution poorly,
usually a sign of a badly scaled posterior.
"""
function tp_bfmi(chain)
    :hamiltonian_energy in names(chain, :internals) || return Float64[]
    E = Array(chain[:hamiltonian_energy])
    out = Float64[]
    for c in axes(E, 2)
        e = vec(E[:, c])
        v = var(e)
        push!(out, v > 0 ? sum(diff(e) .^ 2) / (length(e) * v) : NaN)
    end
    return out
end

"""
    tp_gate_convergence(results, contract; rhat_max, ess_min)

Did it converge, and did the sampler behave?

Rhat and ESS are the "did the chains agree" questions; divergences and depth-cap
hits are the "did the geometry fight back" questions. A run can pass the first
pair and still be untrustworthy if it fails the second, so all four are reported.
"""
function tp_gate_convergence(results, contract::SLContract;
                             rhat_max::Float64 = 1.01, ess_min::Float64 = 400.0)
    chains = [c for (c, _) in results.training_results]
    out = []

    push!(out, (
        name   = "fold sampled",
        pass   = length(chains) == 1,
        detail = "$(length(chains)) chain object(s) returned",
    ))
    isempty(chains) && return out

    chain = first(chains)
    stats = DataFrame(MCMCChains.summarystats(chain))

    max_rhat = maximum(skipmissing(stats.rhat))
    push!(out, (
        name   = "Rhat",
        pass   = max_rhat <= rhat_max,
        detail = @sprintf("max %.5f (threshold %.2f)", max_rhat, rhat_max),
    ))

    min_bulk = minimum(skipmissing(stats.ess_bulk))
    min_tail = minimum(skipmissing(stats.ess_tail))
    push!(out, (
        name   = "effective sample size",
        pass   = min_bulk >= ess_min && min_tail >= ess_min,
        detail = @sprintf("min bulk %.0f, min tail %.0f (threshold %.0f)", min_bulk, min_tail, ess_min),
    ))

    internals = names(chain, :internals)

    n_div = :numerical_error in internals ? Int(sum(Array(chain[:numerical_error]))) : -1
    push!(out, (
        name   = "divergences",
        pass   = n_div == 0,
        detail = n_div < 0 ? "numerical_error not recorded" : "$n_div divergent transitions",
    ))

    if :tree_depth in internals
        depths = Array(chain[:tree_depth])
        n_cap  = count(>=(contract.max_depth), depths)
        push!(out, (
            name   = "tree depth",
            pass   = n_cap == 0,
            detail = "max $(Int(maximum(depths))), $n_cap hits at cap $(contract.max_depth)",
        ))
    end

    bfmi = tp_bfmi(chain)
    if !isempty(bfmi)
        push!(out, (
            name   = "BFMI",
            pass   = minimum(bfmi) >= 0.3,
            detail = @sprintf("min %.3f across %d chains (threshold 0.30)", minimum(bfmi), length(bfmi)),
        ))
    end

    return out
end
