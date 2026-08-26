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
using Profile

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
    Experiments.save_experiment(results)

    # save_experiment returns nothing; the artifact location lives on the results
    # object. Returned explicitly because gate 4 loads the chain back FROM DISK
    # rather than reusing this in-memory object.
    return (results, results.save_path)
end


# ==============================================================================
# 4b. The full grid  (gate 6 input)
# ==============================================================================
#
# Gate 3c samples ONE fold to prove the machinery works. Gates 6 and 7 need every
# fold of the development season, at the contract's grid settings rather than the
# smoke's cheaper ones.
#
# Same splitter as gate 0 built the fold inventory from, so the boundaries gate 2
# checked for leakage are the boundaries this trains on.

"""
    tp_grid_config(model, contract; save_dir) -> ExperimentConfig

The full 24/25 walk-forward run: 20 folds x `grid_chains`, queued.

Differs from `tp_smoke_config` only in the sampler budget and in using the real
splitter (all target steps, not just step 0). Everything else — init range, accept
rate, tree depth, seed — is the same contract, so a difference between smoke and
grid can only be sample size.
"""
function tp_grid_config(model, contract::SLContract; save_dir::AbstractString)
    sampler = Samplers.QueuedNUTSConfig(
        n_samples      = contract.grid_samples,
        n_chains       = contract.grid_chains,
        n_warmup       = contract.grid_warmup,
        accept_rate    = contract.accept_rate,
        max_depth      = contract.max_depth,
        initialisation = Samplers.UniformInit(-contract.init_range, contract.init_range),
        show_progress  = true,
    )

    # The queue flattens folds x chains into one global pool, so concurrency is set
    # by the box (physical cores), not by the chain count.
    execution = Training.Independent(
        parallel             = true,
        max_concurrent_tasks = contract.queue_tasks,
    )

    return Experiments.ExperimentConfig(
        name            = "tp01_grid_$(sl_hash(model))",
        model           = model,
        splitter        = sl_splitter(contract),
        training_config = Training.TrainingConfig(sampler, execution, nothing, false),
        save_dir        = save_dir,
        description     = "Model 01 full 24/25 grid: all folds, input to gates 6-7.",
    )
end

"""
    tp_run_grid(ds, model, contract) -> (results, path)

***THIS SAMPLES ALL 20 FOLDS.*** Requires the threading setup — see
docs/SERVER_AND_KAIMON.md.
"""
function tp_run_grid(ds, model, contract::SLContract)
    save_dir = sl_artifact_dir(contract, "01_team_poisson", sl_hash(model))
    config   = tp_grid_config(model, contract; save_dir = save_dir)

    results = Experiments.run_experiment(ds, config)
    Experiments.save_experiment(results)
    return (results, results.save_path)
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
                             rhat_max::Float64  = 1.01,
                             ess_min::Float64   = 400.0,
                             expected_folds     = nothing)
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

    # Every fold is examined and the WORST is reported, with the fold named.
    # Summarising only the first fold would let 19 divergent folds through — the
    # failure mode this gate exists to prevent.
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

        b = tp_bfmi(ch)
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

    bad_div = [i for (i, d) in enumerate(divs) if d > 0]
    push!(out, (
        name   = "divergences",
        pass   = all(==(0), divs),
        detail = any(<(0), divs) ? "numerical_error not recorded" :
                 isempty(bad_div) ? "0 across all $n folds" :
                 "$(sum(divs)) total, in folds $(bad_div)",
    ))

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


# ==============================================================================
# 6. Gradient profile report  (diagnostic — NOT a gate)
# ==============================================================================
#
# Gate 3b answers "is the gradient correct, and roughly how fast". This section
# answers "where does the time actually go", which is the question you ask once
# the answer to the second half of 3b is disappointing.
#
# It is deliberately not a gate. A slow model is a cost to be traded off; only a
# WRONG model is a defect. Nothing here can fail the walkthrough.
#
# The AD backend measured here is the one the sampler really uses:
# `AutoReverseDiff(compile=true)`, hardcoded at src/samplers/engines/nuts.jl:101
# and defaulted at src/samplers/types.jl:31. If that ever changes, this report
# silently stops being representative — hence `tp_ad_backend_matches_src`.
#
# ==============================================================================

"""
    tp_ad_backend_matches_src() -> (; ok, detail)

The profile below compiles its own ReverseDiff tape. That is only informative if
NUTS does the same thing, so assert it rather than assume it.

Checked by reading src, NOT by inspecting a config object: `QueuedNUTSConfig` has
**no `adtype` field**. Both `run_sampler` methods hardcode
`adtype = AutoReverseDiff(compile=true)` (nuts.jl:101 and :120), so the backend is
not configurable from the contract at all. Recorded here because gate 1 claims
nothing about the fitted model is hidden, and an un-overridable sampler default is
exactly the kind of thing that claim has to cover.
"""
function tp_ad_backend_matches_src()
    src = joinpath(dirname(dirname(pathof(BayesianFootball))),
                   "src", "samplers", "engines", "nuts.jl")
    isfile(src) || return (ok = false, detail = "nuts.jl not found at $src")
    txt = read(src, String)
    n   = length(collect(eachmatch(r"AutoReverseDiff\(compile\s*=\s*true\)", txt)))
    n_methods = length(collect(eachmatch(r"function run_sampler", txt)))
    ok = n >= n_methods && n_methods > 0
    return (ok = ok,
            detail = "$n of $n_methods run_sampler methods hardcode AutoReverseDiff(compile=true)")
end

"""
    tp_hot_frames(; top) -> Vector{(; pct, func, loc)}

Self-time leaders from the last `Profile.@profile` run.

Reads the sample buffer directly instead of calling `Profile.print`, for two
reasons: the kaimon bridge strips `print` calls out of agent-submitted code, and
a returned Vector can be asserted on and stored in FINDINGS.md, whereas printed
text cannot.

Profile buffer layout: samples are separated by `0`, and within a sample index 1
is the INNERMOST frame. Counting first-frames therefore gives self time, which is
what identifies the expensive kernel; cumulative time would just re-report the
top of the call stack.
"""
function tp_hot_frames(; top::Int = 12)
    data   = Profile.fetch(include_meta = false)
    lidict = Profile.getdict(data)

    counts  = Dict{Tuple{Symbol,String},Int}()
    kept    = 0
    dropped = 0
    at_leaf = true
    for ip in data
        if ip == 0
            at_leaf = true
            continue
        end
        if at_leaf
            at_leaf = false
            frames  = get(lidict, ip, nothing)
            if frames === nothing || isempty(frames)
                dropped += 1
                continue
            end
            fr   = first(frames)
            file = string(fr.file)
            # Julia samples ALL threads. With `-t 16` the 15 idle workers park in
            # libc and resolve to no function, which otherwise swamps the ranking
            # at ~97% and hides the actual kernel. They are not our gradient.
            if fr.func === Symbol("") || occursin("libc", file) || occursin("libpthread", file)
                dropped += 1
                continue
            end
            counts[(fr.func, string(basename(file), ":", fr.line))] =
                get(counts, (fr.func, string(basename(file), ":", fr.line)), 0) + 1
            kept += 1
        end
    end

    kept == 0 && return (frames = NamedTuple{(:pct, :func, :loc)}[], kept = 0, dropped = dropped)
    ranked = sort(collect(counts); by = last, rev = true)
    return (frames = [(pct = 100 * c / kept, func = k[1], loc = k[2])
                      for (k, c) in first(ranked, min(top, length(ranked)))],
            kept = kept, dropped = dropped)
end

"""
    tp_grad_profile(model, fs; ...) -> NamedTuple

Full AD cost report for one FeatureSet: parameter count, tape compile time,
primal and gradient latency, allocations, and the self-time leaders.

`primal_ms` is reported alongside `grad_ms` because the RATIO is the diagnostic
number. Reverse-mode AD should cost roughly 3–5x the primal. A much larger ratio
means the reverse pass is not running a hand-written adjoint — it has fallen back
to ReverseDiff's generic broadcast rule, which re-evaluates the kernel per element
under ForwardDiff duals. That is a fixable cost; a slow primal is not.
"""
function tp_grad_profile(model, fs;
                         seed::Int      = 20260825,
                         n_bench::Int   = 200,
                         n_profile::Int = 1500,
                         top::Int       = 12)

    draw = tp_prior_draw(model, fs; seed = seed)
    θ    = draw.θ
    f    = tp_logdensity_fn(draw.turing_model)

    raw_tape  = ReverseDiff.GradientTape(f, θ)
    n_inst    = length(raw_tape.tape)
    n_scalar  = count(i -> typeof(i).name.name === :ScalarInstruction, raw_tape.tape)
    compile_s = @elapsed tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, θ))
    g = similar(θ)
    ReverseDiff.gradient!(g, tape, θ)          # warm

    med(xs) = median(xs) * 1e3
    for _ in 1:5; f(θ); ReverseDiff.gradient!(g, tape, θ); end
    primal_ms = med([@elapsed f(θ)                          for _ in 1:n_bench])
    grad_all  = [@elapsed ReverseDiff.gradient!(g, tape, θ) for _ in 1:n_bench]
    bytes     = @allocated ReverseDiff.gradient!(g, tape, θ)

    Profile.clear()
    Profile.init(n = 10^7, delay = 0.00002)
    Profile.@profile for _ in 1:n_profile
        ReverseDiff.gradient!(g, tape, θ)
    end
    hot = tp_hot_frames(top = top)
    Profile.clear()

    n_rows = length(fs.data[:flat_home_goals])

    return (
        n_params      = length(θ),
        n_rows        = n_rows,
        n_inst        = n_inst,
        n_scalar      = n_scalar,
        inst_per_obs  = n_inst / (2 * n_rows),
        n_obs         = 2 * n_rows,
        ad_matches_src = tp_ad_backend_matches_src(),
        compile_s     = compile_s,
        primal_ms     = primal_ms,
        grad_ms       = median(grad_all) * 1e3,
        grad_min_ms   = minimum(grad_all) * 1e3,
        grad_max_ms   = maximum(grad_all) * 1e3,
        ratio         = (median(grad_all) * 1e3) / primal_ms,
        bytes         = bytes,
        us_per_obs    = median(grad_all) * 1e6 / (2 * n_rows),
        hot           = hot,
    )
end

"""
    tp_profile_table(rep; label)

Render `tp_grad_profile` output. Same headline fields as the archived benchmark
table (parameters / tape compile / gradient eval) so the numbers are directly
comparable, plus the ratio and self-time breakdown that table lacked — which is
what tells you WHY a number is what it is.
"""
function tp_profile_table(rep; label::AbstractString = "")
    line = "-"^76
    io   = IOBuffer()
    w(fmt, args...) = Printf.format(io, Printf.Format(fmt), args...)

    w("\n%s\nGRADIENT PROFILE   %s\n%s\n", line, label, line)
    w("  matches / observations      %d / %d\n", rep.n_rows, rep.n_obs)
    w("  sampled parameters          %d\n", rep.n_params)
    w("  AD backend matches src      %s\n",
      rep.ad_matches_src.ok ? "yes - " * rep.ad_matches_src.detail
                            : "NO - "  * rep.ad_matches_src.detail)
    w("  tape instructions           %d  (%d scalar, %.1f per observation)\n",
      rep.n_inst, rep.n_scalar, rep.inst_per_obs)
    w("  tape compile                %.2f s\n", rep.compile_s)
    w("  primal   (log density)      %.3f ms\n", rep.primal_ms)
    w("  gradient (compiled tape)    %.3f ms median   [min %.3f, max %.3f]\n",
      rep.grad_ms, rep.grad_min_ms, rep.grad_max_ms)
    w("  gradient / primal           %.1fx   (reverse-mode should be ~3-5x)\n", rep.ratio)
    w("  per observation             %.2f us\n", rep.us_per_obs)
    w("  allocations per gradient    %d bytes\n", rep.bytes)
    w("  guide target                < 1 ms  (docs/turing_ad_performance_guide.md,\n")
    w("                              measured there with @belapsed = MIN, not median)\n")
    w("%s\nSELF TIME  (%d resolved samples; %d unresolved C frames dropped -- idle worker threads)\n",
      line, rep.hot.kept, rep.hot.dropped)
    for h in rep.hot.frames
        w("  %5.1f%%  %-26s %s\n", h.pct, string(h.func), h.loc)
    end
    w("%s\n", line)
    return String(take!(io))
end
