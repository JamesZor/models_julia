# ==============================================================================
# Shared protocol — GATE 3 : SAMPLING
# ==============================================================================
# Loader only.  Model-specific equations, sites, and parameter translations are
# supplied by AbstractSLModelAdapter hooks; this file owns the checks.

using DynamicPPL
using LogDensityProblems
using ForwardDiff
using ReverseDiff
using LinearAlgebra
using Random
using Statistics
using Printf
using Profile
using MCMCChains
using DataFrames

# ------------------------------------------------------------------------------
# 1. Prior draws
# ------------------------------------------------------------------------------

"""One seeded prior draw in DynamicPPL's vector and adapter parameter spaces."""
function sl_prior_draw(adapter::AbstractSLModelAdapter, fs; seed::Int = 20260825)
    turing_model = sl_build_turing_model(adapter, fs)
    Random.seed!(seed)
    vi = DynamicPPL.VarInfo(turing_model)
    turing_model(vi)
    return (; turing_model, vi, θ = copy(vi[:]), params = sl_params_from_varinfo(adapter, vi))
end

"""Return `θ -> logdensity` in the model's original (unlinked) space."""
function sl_logdensity_fn(turing_model)
    lf = DynamicPPL.LogDensityFunction(turing_model)
    return x -> LogDensityProblems.logdensity(lf, x)
end

# ------------------------------------------------------------------------------
# 2. GATE 3a — equation parity
# ------------------------------------------------------------------------------

function sl_gate_equation_parity(adapter::AbstractSLModelAdapter, fs;
                                 seeds = [20260825, 7, 991])
    isempty(seeds) && return [sl_result("equation inputs", false, "no seeds")]
    sl_assert_model_contract(adapter)

    data  = sl_equation_data(adapter, fs)
    diffs = Float64[]
    for seed in seeds
        draw = sl_prior_draw(adapter, fs; seed)
        f = sl_logdensity_fn(draw.turing_model)
        push!(diffs, f(draw.θ) - sl_equation_logjoint(adapter, draw.params, data))
    end

    worst = maximum(abs, diffs)
    results = [sl_result("log density parity (Turing vs reference)", worst <= 1e-8,
                         @sprintf("max |Δ| = %.3e over %d prior draws", worst, length(seeds)))]

    # VarInfo records grouped sites; Chains expands vector sites into columns.
    # They are deliberately checked separately: comparing either representation to
    # the other would conceal exactly the schema error this gate is meant to catch.
    draw     = sl_prior_draw(adapter, fs; seed = first(seeds))
    schema   = sl_posterior_schema(adapter)
    observed = Set(string.(keys(draw.vi)))
    grouped  = Set(string.(schema.varinfo_sites))
    push!(results, sl_result(
        "grouped VarInfo site manifest", observed == grouped,
        observed == grouped ? "$(length(observed)) grouped sites, as documented" :
        "unexpected: $(sort(collect(symdiff(observed, grouped))))",
    ))

    n_teams  = Int(fs.data[:n_teams])
    expanded = Set(string.(sl_sampled_sites(adapter, n_teams)))
    expected = Set(string.(schema.chain_columns(n_teams)))
    push!(results, sl_result(
        "expanded synthetic-chain columns", expanded == expected,
        expanded == expected ? "$(length(expanded)) expanded columns, as documented" :
        "unexpected: $(sort(collect(symdiff(expanded, expected))))",
    ))

    expected_n = schema.parameter_count(n_teams)
    push!(results, sl_result("parameter count", length(draw.θ) == expected_n,
                             "$(length(draw.θ)) parameters; expected $expected_n"))
    return vcat(results, sl_adapter_check(adapter, :equation, fs, draw))
end

# ------------------------------------------------------------------------------
# 3. GATE 3b — gradient health
# ------------------------------------------------------------------------------

_sl_relerr(a, b) = norm(a - b) / max(norm(a), norm(b), 1.0)

function sl_gate_gradients(adapter::AbstractSLModelAdapter, fs;
                           seed::Int = 20260825, probes = [0.001, -0.002, 0.003])
    draw = sl_prior_draw(adapter, fs; seed)
    f, θ = sl_logdensity_fn(draw.turing_model), draw.θ
    isempty(θ) && return ([sl_result("gradient inputs", false, "empty parameter vector")], nothing)

    compile_s = @elapsed tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, θ))
    g_compiled = similar(θ)
    ReverseDiff.gradient!(g_compiled, tape, θ)
    g_fresh   = ReverseDiff.gradient(f, θ)
    g_forward = ForwardDiff.gradient(f, θ)

    results = Any[
        sl_result("log density finite at prior draw", isfinite(f(θ)),
                  @sprintf("logdensity = %.4f", f(θ))),
        sl_result("gradients finite",
                  all(isfinite, g_compiled) && all(isfinite, g_fresh) && all(isfinite, g_forward),
                  "compiled / fresh / ForwardDiff all finite"),
    ]
    e_tape = _sl_relerr(g_fresh, g_compiled)
    push!(results, sl_result("compiled tape == fresh ReverseDiff", e_tape <= 1e-8,
                             @sprintf("relerr = %.3e", e_tape)))
    e_mode = _sl_relerr(g_compiled, g_forward)
    push!(results, sl_result("ReverseDiff == ForwardDiff", e_mode <= 1e-6,
                             @sprintf("relerr = %.3e", e_mode)))

    idx = unique(round.(Int, range(1, length(θ); length = min(5, length(θ)))))
    worst_fd = 0.0
    for k in idx
        ε = 1e-6
        xp = copy(θ); xp[k] += ε
        xm = copy(θ); xm[k] -= ε
        worst_fd = max(worst_fd, abs(g_compiled[k] - (f(xp) - f(xm)) / (2ε)))
    end
    push!(results, sl_result("finite differences agree", worst_fd <= 1e-4,
                             @sprintf("max |Δ| = %.3e at indices %s", worst_fd, string(idx))))

    worst_probe = isempty(probes) ? Inf : 0.0
    for δ in probes
        θp = θ .+ δ .* sin.(collect(eachindex(θ)))
        gp = similar(θp)
        ReverseDiff.gradient!(gp, tape, θp)
        worst_probe = max(worst_probe, _sl_relerr(ReverseDiff.gradient(f, θp), gp))
    end
    push!(results, sl_result("static tape safe at perturbed points", worst_probe <= 1e-8,
                             @sprintf("max relerr = %.3e over %d probes", worst_probe, length(probes))))

    g_bench = similar(θ)
    times = [@elapsed ReverseDiff.gradient!(g_bench, tape, θ) for _ in 1:50]
    median_ms = median(times) * 1e3
    push!(results, sl_result("compiled gradient latency", true,
                             @sprintf("median %.3f ms (compile %.2f s) — target < 1 ms", median_ms, compile_s)))
    return (vcat(results, sl_adapter_check(adapter, :gradients, fs, draw)), (; θ, tape, f, median_ms))
end

# ------------------------------------------------------------------------------
# 4. Experiments — smoke artifact and full grid
# ------------------------------------------------------------------------------

function _sl_smoke_splitter(contract::SLContract)
    return SLData.GroupedCVConfig(tournament_groups = [contract.tournaments],
        target_seasons = contract.dev_seasons, history_seasons = contract.history_seasons,
        dynamics_col = contract.dynamics_col, warmup_period = contract.warmup_period,
        stop_early = contract.stop_early, end_dynamics = 0)
end

function sl_experiment_config(adapter::AbstractSLModelAdapter, contract::SLContract;
                              smoke::Bool, save_dir::AbstractString)
    sampler = SLSamplers.QueuedNUTSConfig(
        n_samples = smoke ? contract.smoke_samples : contract.grid_samples,
        n_chains = smoke ? contract.smoke_chains : contract.grid_chains,
        n_warmup = smoke ? contract.smoke_warmup : contract.grid_warmup,
        accept_rate = contract.accept_rate, max_depth = contract.max_depth,
        initialisation = SLSamplers.UniformInit(-contract.init_range, contract.init_range),
        show_progress = !smoke,
    )
    execution = SLTraining.Independent(parallel = true,
        max_concurrent_tasks = smoke ? contract.smoke_chains : contract.queue_tasks)
    return SLExperiments.ExperimentConfig(
        name = "$(sl_model_name(adapter))_$(smoke ? "smoke" : "grid")_$(sl_artifact_hash(adapter, contract))",
        model = sl_model(adapter), splitter = smoke ? _sl_smoke_splitter(contract) : sl_splitter(contract),
        training_config = SLTraining.TrainingConfig(sampler, execution, nothing, false),
        save_dir = save_dir,
        description = "Scottish Lower protocol $(smoke ? "smoke" : "grid") artifact $(sl_artifact_hash(adapter, contract)).",
    )
end

function sl_run_experiment(ds, adapter::AbstractSLModelAdapter, contract::SLContract; smoke::Bool)
    path = sl_artifact_dir(adapter, contract)
    results = SLExperiments.run_experiment(ds, sl_experiment_config(adapter, contract; smoke, save_dir = path))
    SLExperiments.save_experiment(results)
    return (results, results.save_path)
end

"""Reload the persisted artifact; never substitute an in-memory experiment result."""
sl_load_experiment(path::AbstractString) = SLExperiments.load_experiment(path)

# ------------------------------------------------------------------------------
# 5. Convergence diagnostics
# ------------------------------------------------------------------------------

"""BFMI per physical chain, not per parameter column."""
function sl_bfmi(chain)
    :hamiltonian_energy in names(chain, :internals) || return Float64[]
    E = Array(chain[:hamiltonian_energy])
    # MCMCChains selectors are normally iteration × 1 × chain.  Retain support
    # for older iteration × chain selectors without treating parameter axis as chains.
    energies = ndims(E) == 3 ? (vec(E[:, :, c]) for c in axes(E, 3)) :
               ndims(E) == 2 ? (vec(E[:, c]) for c in axes(E, 2)) : (vec(E),)
    out = Float64[]
    for e in energies
        v = var(e)
        push!(out, v > 0 ? sum(diff(e) .^ 2) / (length(e) * v) : NaN)
    end
    return out
end

function sl_gate_convergence(results, adapter::AbstractSLModelAdapter, contract::SLContract;
                             rhat_max::Float64 = 1.01, ess_min::Float64 = 400.0,
                             divergence_rate_max::Float64 = 0.001, expected_folds = nothing)
    chains = [c for (c, _) in results.training_results]
    n = length(chains)
    out = Any[sl_result("folds sampled", n > 0 && (expected_folds === nothing || n == expected_folds),
                        expected_folds === nothing ? "$n fold(s) returned" : "$n of $expected_folds folds returned")]
    isempty(chains) && return out

    stats = [DataFrame(MCMCChains.summarystats(ch)) for ch in chains]
    rhats = [maximum(skipmissing(st.rhat)) for st in stats]
    bulks = [minimum(skipmissing(st.ess_bulk)) for st in stats]
    tails = [minimum(skipmissing(st.ess_tail)) for st in stats]
    worst_rhat, i_rhat = findmax(rhats)
    min_bulk, i_bulk = findmin(bulks)
    push!(out, sl_result("Rhat", worst_rhat <= rhat_max,
        @sprintf("max %.5f (fold %d) — %d/%d folds under %.2f", worst_rhat, i_rhat, count(<=(rhat_max), rhats), n, rhat_max)))
    push!(out, sl_result("effective sample size", min_bulk >= ess_min && minimum(tails) >= ess_min,
        @sprintf("min bulk %.0f (fold %d), min tail %.0f — %d/%d folds above %.0f", min_bulk, i_bulk, minimum(tails), count(>=(ess_min), bulks), n, ess_min)))

    internals = [names(ch, :internals) for ch in chains]
    has_divergence = all(:numerical_error in x for x in internals)
    has_depth = all(:tree_depth in x for x in internals)
    n_draws = sum(size(ch, 1) * size(ch, 3) for ch in chains)
    divs = has_divergence ? [Int(sum(Array(ch[:numerical_error]))) for ch in chains] : Int[]
    div_rate = has_divergence ? sum(divs) / n_draws : Inf
    bad_div = has_divergence ? findall(>(0), divs) : Int[]
    push!(out, sl_result("divergence telemetry", has_divergence && div_rate <= divergence_rate_max,
        has_divergence ? (isempty(bad_div) ? "0 across all $n folds ($n_draws draws)" : @sprintf("%d total = %.4f%% of %d draws (threshold %.2f%%), in folds %s", sum(divs), 100div_rate, n_draws, 100divergence_rate_max, string(bad_div))) : "numerical_error not recorded"))

    # The tolerated divergence rate is valid only when divergences are not
    # concentrated at adapter-declared hierarchical funnel sites. Missing sites or
    # missing divergence telemetry are failures, never silently skipped.
    funnel_sites = sl_capabilities(adapter).funnel_sites
    site_present = has_divergence && !isempty(funnel_sites) && all(all(Symbol(s) in names(ch) for s in funnel_sites) for ch in chains)
    ratios = Float64[]
    if site_present
        for i in bad_div
            ch = chains[i]
            divergent = vec(Array(ch[:numerical_error])) .> 0
            any(divergent) && !all(divergent) || continue
            for site in funnel_sites
                values = vec(Array(ch[Symbol(site)]))
                push!(ratios, mean(values[divergent]) / mean(values[.!divergent]))
            end
        end
    end
    funnel_ok = site_present && (isempty(bad_div) || (!isempty(ratios) && minimum(ratios) >= 0.5))
    funnel_detail = !has_divergence ? "numerical_error not recorded" : isempty(funnel_sites) ? "adapter declares no funnel sites" : !site_present ? "declared funnel site missing from chain" : isempty(bad_div) ? "no divergences to cluster" : @sprintf("scale at divergent draws is %.2f-%.2fx the bulk mean%s", minimum(ratios), maximum(ratios), minimum(ratios) < .5 ? " — CLUSTERED AT SMALL SCALE" : " (no clustering ⇒ integrator noise)")
    push!(out, sl_result("divergences not a funnel", funnel_ok, funnel_detail))

    cap_hits = has_depth ? sum(count(>=(contract.max_depth), Array(ch[:tree_depth])) for ch in chains) : -1
    max_depth = has_depth ? maximum(maximum(Array(ch[:tree_depth])) for ch in chains) : -1
    push!(out, sl_result("tree-depth telemetry", has_depth && cap_hits == 0,
                         has_depth ? "max $max_depth, $cap_hits hits at cap $(contract.max_depth)" : "tree_depth not recorded"))

    bfmis = vcat(sl_bfmi.(chains)...)
    push!(out, sl_result("BFMI", !isempty(bfmis) && all(x -> isfinite(x) && x >= .3, bfmis),
                         isempty(bfmis) ? "hamiltonian_energy not recorded" : @sprintf("min %.3f across %d physical chains (threshold 0.30)", minimum(bfmis), length(bfmis))))
    return vcat(out, sl_adapter_check(adapter, :convergence, results, contract))
end

# ------------------------------------------------------------------------------
# 6. Gradient profiling (diagnostic, never a gate)
# ------------------------------------------------------------------------------

function sl_ad_backend_matches_src()
    src = joinpath(dirname(dirname(pathof(BayesianFootball))), "src", "samplers", "engines", "nuts.jl")
    isfile(src) || return (; ok = false, detail = "nuts.jl not found at $src")
    text = read(src, String)
    n = length(collect(eachmatch(r"AutoReverseDiff\(compile\s*=\s*true\)", text)))
    methods = length(collect(eachmatch(r"function run_sampler", text)))
    return (; ok = n >= methods && methods > 0, detail = "$n of $methods run_sampler methods hardcode AutoReverseDiff(compile=true)")
end

function sl_hot_frames(; top::Int = 12)
    data = Profile.fetch(include_meta = false)
    lidict = Profile.getdict(data)
    counts = Dict{Tuple{Symbol,String},Int}(); kept = 0; dropped = 0; at_leaf = true
    for ip in data
        if ip == 0; at_leaf = true; continue; end
        at_leaf || continue
        at_leaf = false
        frames = get(lidict, ip, nothing)
        if frames === nothing || isempty(frames); dropped += 1; continue; end
        fr, file = first(frames), string(first(frames).file)
        if fr.func === Symbol("") || occursin("libc", file) || occursin("libpthread", file); dropped += 1; continue; end
        key = (fr.func, string(basename(file), ":", fr.line)); counts[key] = get(counts, key, 0) + 1; kept += 1
    end
    kept == 0 && return (; frames = NamedTuple{(:pct,:func,:loc)}[], kept, dropped)
    ranked = sort(collect(counts); by = last, rev = true)
    return (; frames = [(; pct = 100c / kept, func = k[1], loc = k[2]) for (k,c) in first(ranked, min(top,length(ranked)))], kept, dropped)
end

function sl_grad_profile(adapter::AbstractSLModelAdapter, fs; seed::Int = 20260825, n_bench::Int = 200, n_profile::Int = 1500, top::Int = 12)
    draw = sl_prior_draw(adapter, fs; seed); θ = draw.θ; f = sl_logdensity_fn(draw.turing_model)
    raw_tape = ReverseDiff.GradientTape(f, θ); n_inst = length(raw_tape.tape); n_scalar = count(i -> typeof(i).name.name === :ScalarInstruction, raw_tape.tape)
    compile_s = @elapsed tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, θ)); g = similar(θ); ReverseDiff.gradient!(g, tape, θ)
    for _ in 1:5; f(θ); ReverseDiff.gradient!(g, tape, θ); end
    med(xs) = median(xs) * 1e3; primal_ms = med([@elapsed f(θ) for _ in 1:n_bench]); grad_all = [@elapsed ReverseDiff.gradient!(g, tape, θ) for _ in 1:n_bench]; bytes = @allocated ReverseDiff.gradient!(g, tape, θ)
    Profile.clear(); Profile.init(n = 10^7, delay = .00002); Profile.@profile for _ in 1:n_profile; ReverseDiff.gradient!(g, tape, θ); end
    hot = sl_hot_frames(top = top); Profile.clear()
    n_rows = length(fs.data[:flat_home_goals])
    return (; n_params=length(θ), n_rows, n_obs=2n_rows, n_inst, n_scalar, inst_per_obs=n_inst/(2n_rows), ad_matches_src=sl_ad_backend_matches_src(), compile_s, primal_ms, grad_ms=median(grad_all)*1e3, grad_min_ms=minimum(grad_all)*1e3, grad_max_ms=maximum(grad_all)*1e3, ratio=median(grad_all)*1e3/primal_ms, bytes, us_per_obs=median(grad_all)*1e6/(2n_rows), hot)
end

function sl_profile_table(rep; label::AbstractString = "")
    line = "-"^76; io = IOBuffer(); w(fmt,args...) = Printf.format(io, Printf.Format(fmt), args...)
    w("\n%s\nGRADIENT PROFILE   %s\n%s\n", line,label,line); w("  matches / observations      %d / %d\n  sampled parameters          %d\n", rep.n_rows,rep.n_obs,rep.n_params)
    w("  AD backend matches src      %s\n", rep.ad_matches_src.ok ? "yes - "*rep.ad_matches_src.detail : "NO - "*rep.ad_matches_src.detail)
    w("  tape instructions           %d  (%d scalar, %.1f per observation)\n  tape compile                %.2f s\n", rep.n_inst,rep.n_scalar,rep.inst_per_obs,rep.compile_s)
    w("  primal   (log density)      %.3f ms\n  gradient (compiled tape)    %.3f ms median   [min %.3f, max %.3f]\n  gradient / primal           %.1fx\n  per observation             %.2f us\n  allocations per gradient    %d bytes\n", rep.primal_ms,rep.grad_ms,rep.grad_min_ms,rep.grad_max_ms,rep.ratio,rep.us_per_obs,rep.bytes)
    w("%s\nSELF TIME  (%d resolved samples; %d unresolved C frames dropped)\n", line,rep.hot.kept,rep.hot.dropped); for h in rep.hot.frames; w("  %5.1f%%  %-26s %s\n",h.pct,string(h.func),h.loc); end; w("%s\n",line)
    String(take!(io))
end
