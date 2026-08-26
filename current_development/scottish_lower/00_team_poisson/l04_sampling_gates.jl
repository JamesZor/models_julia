# ==============================================================================
# Model 00 — GATE 3 : SAMPLING (Pure Poisson)
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# Covers Gate 3 for Model 00:
#   3a: Equation parity (Turing @model vs independent l02_equations.jl)
#   3b: Gradient health (compiled ReverseDiff vs fresh vs ForwardDiff vs FiniteDiff)
#   3c: Smoke run & convergence (1 fold, 4 chains x 500/500, persisted)
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
using Dates
using MCMCChains

const PGm = BayesianFootball.Models.PreGame


# ==============================================================================
# 1. Prior Draws
# ==============================================================================

function tp00_params_from_varinfo(vi)
    v = Dict(string(k) => vi[k] for k in keys(vi))
    return TP00Params(
        μ     = v["inter.μ"],
        γ     = v["ha.γ_global"],
        σ_a   = v["dyn.σ_a"],
        σ_d   = v["dyn.σ_d"],
        raw_a = Vector{Float64}(v["dyn.raw_a"]),
        raw_d = Vector{Float64}(v["dyn.raw_d"]),
    )
end

function tp00_prior_draw(model::DynamicPoissonGoalsTimeDecayModel, fs; seed::Int = 20260825)
    turing_model = PGm.build_turing_model(model, fs)
    Random.seed!(seed)
    vi = DynamicPPL.VarInfo(turing_model)
    turing_model(vi)

    return (
        turing_model = turing_model,
        vi           = vi,
        θ            = copy(vi[:]),
        params       = tp00_params_from_varinfo(vi),
    )
end

function tp00_logdensity_fn(turing_model)
    lf = DynamicPPL.LogDensityFunction(turing_model)
    return x -> LogDensityProblems.logdensity(lf, x)
end


# ==============================================================================
# 2. GATE 3a — Equation Parity
# ==============================================================================

function tp00_gate_equation_parity(model::DynamicPoissonGoalsTimeDecayModel, fs; seeds = [20260825, 7, 991])
    tp00_assert_default(model)
    data = tp00_equation_data(fs)

    diffs = Float64[]
    for s in seeds
        draw = tp00_prior_draw(model, fs; seed = s)
        f    = tp00_logdensity_fn(draw.turing_model)
        push!(diffs, f(draw.θ) - tp00_logjoint(draw.params, data, model))
    end

    worst = maximum(abs, diffs)
    results = [(
        name   = "log density parity (Turing vs l02)",
        pass   = worst <= 1e-8,
        detail = @sprintf("max |Δ| = %.3e over %d prior draws", worst, length(seeds)),
    )]

    draw = tp00_prior_draw(model, fs; seed = first(seeds))
    observed = Set(string.(keys(draw.vi)))
    expected = Set(["inter.μ", "ha.γ_global", "dyn.σ_a", "dyn.σ_d", "dyn.raw_a", "dyn.raw_d"])
    push!(results, (
        name   = "sampled-site manifest",
        pass   = observed == expected,
        detail = observed == expected ? "$(length(observed)) sites, as documented" :
                 "unexpected: $(sort(collect(symdiff(observed, expected))))",
    ))

    n_teams = Int(fs.data[:n_teams])
    push!(results, (
        name   = "parameter count",
        pass   = length(draw.θ) == 4 + 2 * n_teams,
        detail = "$(length(draw.θ)) = 4 scalars + 2 x $n_teams team effects",
    ))

    return results
end


# ==============================================================================
# 3. GATE 3b — Gradient Health
# ==============================================================================

_tp00_relerr(a, b) = norm(a - b) / max(norm(a), norm(b), 1.0)

function tp00_gate_gradients(model::DynamicPoissonGoalsTimeDecayModel, fs; seed::Int = 20260825, probes = [0.001, -0.002, 0.003])
    draw = tp00_prior_draw(model, fs; seed = seed)
    f    = tp00_logdensity_fn(draw.turing_model)
    θ    = draw.θ

    results = []

    push!(results, (
        name   = "log density finite at prior draw",
        pass   = isfinite(f(θ)),
        detail = @sprintf("logdensity = %.4f", f(θ)),
    ))

    tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, θ))

    g_compiled = similar(θ)
    ReverseDiff.gradient!(g_compiled, tape, θ)
    g_fresh   = ReverseDiff.gradient(f, θ)
    g_forward = ForwardDiff.gradient(f, θ)

    push!(results, (
        name   = "gradients finite",
        pass   = all(isfinite, g_compiled) && all(isfinite, g_fresh) && all(isfinite, g_forward),
        detail = "compiled / fresh / ForwardDiff all finite",
    ))

    e_tape = _tp00_relerr(g_fresh, g_compiled)
    push!(results, (
        name   = "compiled tape == fresh ReverseDiff",
        pass   = e_tape <= 1e-8,
        detail = @sprintf("relerr = %.3e", e_tape),
    ))

    e_mode = _tp00_relerr(g_compiled, g_forward)
    push!(results, (
        name   = "ReverseDiff == ForwardDiff",
        pass   = e_mode <= 1e-6,
        detail = @sprintf("relerr = %.3e", e_mode),
    ))

    n_teams = Int(fs.data[:n_teams])
    idx = unique([1, 2, 4, 4 + n_teams ÷ 2, length(θ)])
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

    worst_probe = 0.0
    for δ in probes
        θp = θ .+ δ .* sin.(collect(eachindex(θ)))
        gp = similar(θp)
        ReverseDiff.gradient!(gp, tape, θp)
        worst_probe = max(worst_probe, _tp00_relerr(ReverseDiff.gradient(f, θp), gp))
    end
    push!(results, (
        name   = "static tape safe at perturbed points",
        pass   = worst_probe <= 1e-8,
        detail = @sprintf("max relerr = %.3e over %d probes", worst_probe, length(probes)),
    ))

    return (results, (tape = tape, θ = θ, f = f))
end


# ==============================================================================
# 4. GATE 3c — Smoke & Grid Runners
# ==============================================================================

function tp00_smoke_config(model::DynamicPoissonGoalsTimeDecayModel, contract::SLContract;
                           save_dir = sl_artifact_dir(contract, "00_team_poisson", sl_hash(model)))
    name = "tp00_smoke_$(sl_hash(model))_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"

    splitter = Data.GroupedCVConfig(
        tournament_groups = [contract.tournaments],
        target_seasons    = contract.dev_seasons,
        history_seasons   = contract.history_seasons,
        dynamics_col      = contract.dynamics_col,
        warmup_period     = contract.warmup_period,
        stop_early        = contract.stop_early,
        end_dynamics      = 0,
    )

    sampler = Samplers.QueuedNUTSConfig(
        n_chains              = contract.smoke_chains,
        target_accept_rate    = contract.accept_rate,
        max_depth             = contract.max_depth,
        seed                  = contract.seed,
        max_concurrent_tasks  = 4,
        init_type             = :uniform,
        init_range            = contract.init_range,
    )

    return Experiments.ExperimentTask(
        name            = name,
        model           = model,
        save_dir        = save_dir,
        target_seasons  = contract.dev_seasons,
        history_seasons = contract.history_seasons,
        samples         = contract.smoke_samples,
        warmup          = contract.smoke_warmup,
        save_to_disk    = true,
        save_diagnostics= true,
        splitter        = splitter,
        sampler         = sampler,
    )
end

function tp00_run_smoke(ds, model::DynamicPoissonGoalsTimeDecayModel, contract::SLContract; kwargs...)
    task = tp00_smoke_config(model, contract; kwargs...)
    results = Experiments.run_experiment(ds, task)
    path = joinpath(task.save_dir, task.name)
    return (results, path)
end

function tp00_load_smoke(path::AbstractString)
    return Experiments.load_experiment(path)
end

function tp00_grid_config(model::DynamicPoissonGoalsTimeDecayModel, contract::SLContract;
                          save_dir = sl_artifact_dir(contract, "00_team_poisson", sl_hash(model)))
    name = "tp00_grid_$(sl_hash(model))_$(Dates.format(now(), "yyyymmdd_HHMMSS"))"

    splitter = sl_splitter(contract)

    sampler = Samplers.QueuedNUTSConfig(
        n_chains              = contract.grid_chains,
        target_accept_rate    = contract.accept_rate,
        max_depth             = contract.max_depth,
        seed                  = contract.seed,
        max_concurrent_tasks  = contract.queue_tasks,
        init_type             = :uniform,
        init_range            = contract.init_range,
    )

    return Experiments.ExperimentTask(
        name            = name,
        model           = model,
        save_dir        = save_dir,
        target_seasons  = contract.dev_seasons,
        history_seasons = contract.history_seasons,
        samples         = contract.grid_samples,
        warmup          = contract.grid_warmup,
        save_to_disk    = true,
        save_diagnostics= true,
        splitter        = splitter,
        sampler         = sampler,
    )
end

function tp00_run_grid(ds, model::DynamicPoissonGoalsTimeDecayModel, contract::SLContract; kwargs...)
    task = tp00_grid_config(model, contract; kwargs...)
    results = Experiments.run_experiment(ds, task)
    path = joinpath(task.save_dir, task.name)
    return (results, path)
end

function tp00_gate_convergence(results, contract::SLContract; expected_folds::Int = 1)
    chains = [c for (c, _) in results.training_results]
    results_list = []

    push!(results_list, (
        name   = "expected fold count present",
        pass   = length(chains) == expected_folds,
        detail = "$(length(chains)) / $expected_folds folds present",
    ))

    # Evaluate summary statistics across all folds
    max_rhat = 0.0
    min_ess_bulk = Inf
    min_ess_tail = Inf
    total_divs = 0
    total_draws = 0

    for ch in chains
        s = DataFrame(MCMCChains.summarystats(ch))
        rhat_col = :rhat in propertynames(s) ? s.rhat : s.r_hat
        max_rhat = max(max_rhat, maximum(filter(isfinite, rhat_col)))

        if :ess_bulk in propertynames(s)
            min_ess_bulk = min(min_ess_bulk, minimum(filter(isfinite, s.ess_bulk)))
        end
        if :ess_tail in propertynames(s)
            min_ess_tail = min(min_ess_tail, minimum(filter(isfinite, s.ess_tail)))
        end

        if :numerical_error in names(ch.value)
            divs = sum(ch.value[:numerical_error])
            total_divs += divs
        end
        total_draws += size(ch, 1) * size(ch, 3)
    end

    push!(results_list, (
        name   = "Rhat convergence (<= 1.01)",
        pass   = max_rhat <= 1.01,
        detail = @sprintf("max Rhat = %.5f across folds", max_rhat),
    ))

    push!(results_list, (
        name   = "Effective sample size (ESS >= 400)",
        pass   = min_ess_bulk >= 400 && min_ess_tail >= 400,
        detail = @sprintf("min bulk = %.0f, min tail = %.0f", min_ess_bulk, min_ess_tail),
    ))

    div_rate = total_draws > 0 ? total_divs / total_draws : 0.0
    push!(results_list, (
        name   = "Divergences rate (<= 0.1%)",
        pass   = div_rate <= 0.001,
        detail = @sprintf("%d / %d draws (%.4f%%)", total_divs, total_draws, div_rate * 100),
    ))

    return results_list
end
