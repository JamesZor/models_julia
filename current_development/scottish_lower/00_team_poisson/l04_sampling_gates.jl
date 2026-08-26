# ==============================================================================
# Model 00 — GATE 3 : SAMPLING (Pure Poisson)
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# Gate 3 asks three separate questions:
#   (a) equation parity   is the Turing model the model MODEL.md documents?
#   (b) gradient health   is the AD path correct and fast enough to sample?
#   (c) smoke run         does it actually sample, and converge?
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
using Profile
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

    # Latency benchmark
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

    return (results, (tape = tape, θ = θ, f = f, median_ms = med_ms))
end


# ==============================================================================
# 4. GATE 3c — Smoke & Grid Configs and Runners
# ==============================================================================

function tp00_smoke_splitter(contract::SLContract)
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

function tp00_smoke_config(model::DynamicPoissonGoalsTimeDecayModel, contract::SLContract;
                           save_dir::AbstractString)
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
        name            = "tp00_smoke_$(sl_hash(model))",
        model           = model,
        splitter        = tp00_smoke_splitter(contract),
        training_config = Training.TrainingConfig(sampler, execution, nothing, false),
        save_dir        = save_dir,
        description     = "Model 00 gate-3 smoke: one fold, persisted for gate 4.",
    )
end

function tp00_run_smoke(ds, model::DynamicPoissonGoalsTimeDecayModel, contract::SLContract)
    save_dir = sl_artifact_dir(contract, "00_team_poisson", sl_hash(model))
    config   = tp00_smoke_config(model, contract; save_dir = save_dir)

    results = Experiments.run_experiment(ds, config)
    Experiments.save_experiment(results)

    return (results, results.save_path)
end

function tp00_load_smoke(path::AbstractString)
    return Experiments.load_experiment(path)
end

function tp00_grid_config(model::DynamicPoissonGoalsTimeDecayModel, contract::SLContract;
                          save_dir::AbstractString)
    sampler = Samplers.QueuedNUTSConfig(
        n_samples      = contract.grid_samples,
        n_chains       = contract.grid_chains,
        n_warmup       = contract.grid_warmup,
        accept_rate    = contract.accept_rate,
        max_depth      = contract.max_depth,
        initialisation = Samplers.UniformInit(-contract.init_range, contract.init_range),
        show_progress  = true,
    )

    execution = Training.Independent(
        parallel             = true,
        max_concurrent_tasks = contract.queue_tasks,
    )

    return Experiments.ExperimentConfig(
        name            = "tp00_grid_$(sl_hash(model))",
        model           = model,
        splitter        = sl_splitter(contract),
        training_config = Training.TrainingConfig(sampler, execution, nothing, false),
        save_dir        = save_dir,
        description     = "Model 00 full 24/25 grid: all folds, input to gates 6-7.",
    )
end

function tp00_run_grid(ds, model::DynamicPoissonGoalsTimeDecayModel, contract::SLContract)
    save_dir = sl_artifact_dir(contract, "00_team_poisson", sl_hash(model))
    config   = tp00_grid_config(model, contract; save_dir = save_dir)

    results = Experiments.run_experiment(ds, config)
    Experiments.save_experiment(results)
    return (results, results.save_path)
end


# ==============================================================================
# 5. Convergence Diagnostics
# ==============================================================================

function tp00_bfmi(chain)
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

function tp00_gate_convergence(results, contract::SLContract;
                               rhat_max::Float64            = 1.01,
                               ess_min::Float64             = 400.0,
                               divergence_rate_max::Float64 = 0.001,
                               expected_folds               = nothing)
    chains = [c for (c, _) in results.training_results]
    out = []

    n = length(chains)
    push!(out, (
        name   = "folds sampled",
        pass   = n > 0 && (expected_folds === nothing || n == expected_folds),
        detail = expected_folds === nothing ? "$n fold(s) returned" :
                 "$n of $expected_folds folds returned",
    ))
    isempty(chains) && return out

    rhats  = Float64[]; bulks = Float64[]; tails = Float64[]
    divs   = Int[];     depth_max = Int[]; depth_cap = Int[]; bfmis = Float64[]

    for ch in chains
        st = DataFrame(MCMCChains.summarystats(ch))
        push!(rhats, maximum(skipmissing(st.rhat)))
        push!(bulks, minimum(skipmissing(st.ess_bulk)))
        push!(tails, minimum(skipmissing(st.ess_tail)))

        internals = names(ch, :internals)
        push!(divs, :numerical_error in internals ? Int(sum(Array(ch[:numerical_error]))) : -1)

        if :tree_depth in internals
            d = Array(ch[:tree_depth])
            push!(depth_max, Int(maximum(d)))
            push!(depth_cap, count(>=(contract.max_depth), d))
        end

        b = tp00_bfmi(ch)
        isempty(b) || push!(bfmis, minimum(b))
    end

    worst_rhat, i_rhat = findmax(rhats)
    push!(out, (
        name   = "Rhat",
        pass   = worst_rhat <= rhat_max,
        detail = @sprintf("max %.5f (fold %d) — %d/%d folds under %.2f",
                          worst_rhat, i_rhat, count(<=(rhat_max), rhats), n, rhat_max),
    ))

    min_bulk, i_bulk = findmin(bulks)
    min_tail = minimum(tails)
    push!(out, (
        name   = "effective sample size",
        pass   = min_bulk >= ess_min && min_tail >= ess_min,
        detail = @sprintf("min bulk %.0f (fold %d), min tail %.0f — %d/%d folds above %.0f",
                          min_bulk, i_bulk, min_tail,
                          count(b -> b >= ess_min, bulks), n, ess_min),
    ))

    bad_div   = [i for (i, d) in enumerate(divs) if d > 0]
    n_draws   = sum(size(ch, 1) * size(ch, 3) for ch in chains)
    div_rate  = sum(max.(divs, 0)) / n_draws

    push!(out, (
        name   = "divergences rare",
        pass   = div_rate <= divergence_rate_max,
        detail = any(<(0), divs) ? "numerical_error not recorded" :
                 isempty(bad_div) ? "0 across all $n folds ($n_draws draws)" :
                 @sprintf("%d total = %.4f%% of %d draws (threshold %.2f%%), in folds %s",
                          sum(divs), 100 * div_rate, n_draws,
                          100 * divergence_rate_max, string(bad_div)),
    ))

    if !isempty(bad_div)
        ratios = Float64[]
        for i in bad_div
            ch  = chains[i]
            d   = vec(Array(ch[:numerical_error])) .> 0
            any(d) && all(d) && continue
            for site in ("dyn.σ_a", "dyn.σ_d")
                v = vec(Array(ch[Symbol(site)]))
                push!(ratios, mean(v[d]) / mean(v[.!d]))
            end
        end
        worst = isempty(ratios) ? 1.0 : minimum(ratios)
        push!(out, (
            name   = "divergences not a funnel",
            pass   = worst >= 0.5,
            detail = @sprintf("σ at divergent draws is %.2f-%.2fx the bulk mean%s",
                              minimum(ratios), maximum(ratios),
                              worst < 0.5 ? "  ⚠ CLUSTERED AT SMALL σ — funnel" :
                                            "  (no clustering ⇒ integrator noise)"),
        ))
    end

    if !isempty(depth_max)
        push!(out, (
            name   = "tree depth",
            pass   = sum(depth_cap) == 0,
            detail = "max $(maximum(depth_max)), $(sum(depth_cap)) hits at cap $(contract.max_depth)",
        ))
    end

    if !isempty(bfmis)
        worst_bfmi, i_bfmi = findmin(bfmis)
        push!(out, (
            name   = "BFMI",
            pass   = worst_bfmi >= 0.3,
            detail = @sprintf("min %.3f (fold %d) across %d folds (threshold 0.30)",
                              worst_bfmi, i_bfmi, n),
        ))
    end

    return out
end
