using Test
using BayesianFootball
using BayesianFootball: Data, Training, Experiments
using MCMCChains
using Random
using DataFrames
using Dates
using Logging
using Serialization

const INF = BayesianFootball.Training.Inference

"A model the framework can hold but no latent family is registered for."
struct InferenceMockModel <: BayesianFootball.AbstractFootballModel end

"Run `f()` with logging suppressed — for the cases that are DELIBERATELY made to fail."
inference_quiet(f) = Logging.with_logger(f, Logging.NullLogger())

"""
A synthetic NUTS-shaped chain: two parameters plus the three internals the audit reads.

`rhat_break` shifts each chain's location so the between-chain variance dominates;
`n_divergent` marks that many transitions as divergent; `depth_capped` saturates that
many trajectories at `max_depth`; `energy` overrides the Hamiltonian energy series.
"""
function inference_chain(; n = 200, n_chains = 4, seed = 1, rhat_break = false,
                           n_divergent = 0, depth_capped = 0, max_depth = 10,
                           energy = nothing)
    rng = Xoshiro(seed)
    vals = Array{Float64, 3}(undef, n, 5, n_chains)
    for c in 1:n_chains
        shift = rhat_break ? 10.0 * c : 0.0
        vals[:, 1, c] .= randn(rng, n) .+ shift
        vals[:, 2, c] .= 0.5 .* randn(rng, n)
        vals[:, 3, c] .= 0.0                       # numerical_error
        vals[:, 4, c] .= 4.0                       # tree_depth
        vals[:, 5, c] .= energy === nothing ? 100.0 .+ 3.0 .* randn(rng, n) : energy[:, c]
    end
    for k in 1:n_divergent
        vals[k, 3, 1] = 1.0
    end
    for k in 1:depth_capped
        vals[k, 4, 1] = Float64(max_depth)
    end
    return Chains(vals, [:a, :b, :numerical_error, :tree_depth, :hamiltonian_energy],
                  (parameters = [:a, :b],
                   internals = [:numerical_error, :tree_depth, :hamiltonian_energy]))
end

"Parameters-only chain: nothing for the divergence, depth or BFMI gates to measure."
inference_bare_chain(; n = 200, n_chains = 4, seed = 1) =
    Chains(randn(Xoshiro(seed), n, 2, n_chains), [:a, :b])

"One repricing tick per state, written into the caller's buffers. Must allocate 0 bytes."
function inference_sweep(rates, ws, pre, states)
    for s in states
        solve_ingame_rates!(rates, ws, pre, 1, s)
    end
    return nothing
end

"A `(FeatureSet, SplitMetaData)` collection of `n` folds, with no real features in it."
inference_folds(n::Int) =
    [(BayesianFootball.FeatureSet(:n_teams => 4),
      Data.SplitMetaData(1, "23/24", "24/25", 1, i, 0)) for i in 1:n]

"A `FitConfig` over `chains`, replayed fold by fold, saving under `save_dir`."
inference_config(chains, save_dir; name = "inference_test") =
    FitConfig(name = name, model = InferenceMockModel(),
              splitter = Data.CVConfig(target_seasons = ["24/25"]),
              sampler = ReplaySampler(chains),
              execution = SequentialExecution(),
              save_dir = save_dir)


@testset "Unified inference framework" begin

# ==============================================================================
# 1. FIT CONTAINERS AND INDEXING
# ==============================================================================

@testset "Fit containers and indexing" begin
    chs = [inference_bare_chain(seed = i) for i in 1:3]
    dir = mktempdir()
    fit = fit_model(inference_config(chs, dir); feature_sets = inference_folds(3),
                    quiet = true)

    @test fit isa Fit
    @test length(fit) == 3
    @test size(fit) == (3,)
    @test firstindex(fit) == 1 && lastindex(fit) == 3
    @test fit[1] isa FoldFit
    @test fit[1].chain === chs[1]
    @test fit[3].chain === chs[3]
    @test fit[2].fold == 2
    @test [f.fold for f in fit] == [1, 2, 3]
    @test eltype(fit) <: FoldFit
    @test fold_chains(fit) == chs
    @test length(fold_metas(fit)) == 3
    @test total_draws(fit) == 3 * 200 * 4
    @test fit_name(fit) == "inference_test"

    # The concrete `folds` parameter is what makes `fit[i].chain` a direct load.
    @test isconcretetype(eltype(fit))

    # Provenance is captured, not invented.
    @test fit.metadata isa FitMetadata
    @test fit.metadata.n_threads == Threads.nthreads()
    @test fit.metadata.julia_version == VERSION
    @test fit.metadata.elapsed_seconds >= 0.0
    @test any(startswith("time:"), fit.config.tags)

    # `save_path` follows the legacy `<name>_<yyyymmdd_HHMMSS>` shape.
    @test occursin(r"inference_test_\d{8}_\d{6}$", fit.save_path)
    @test dirname(fit.save_path) == dir

    # A fold index must be positive, and a chain/meta transposition is a MethodError
    # rather than a silently mis-typed container.
    @test_throws ErrorException FoldFit(0, chs[1], Data.SplitMetaData(1, "a", "b", 1, 1, 0))
    @test_throws MethodError FoldFit(1, Data.SplitMetaData(1, "a", "b", 1, 1, 0), chs[1])
end

@testset "FitConfig recipe" begin
    model = InferenceMockModel()
    splitter = Data.CVConfig(target_seasons = ["24/25"])
    sampler = BayesianFootball.Samplers.MAPConfig()

    cfg = FitConfig(name = "flat", model = model, splitter = splitter, sampler = sampler)
    @test cfg.name == "flat"
    @test cfg.sampler === sampler
    @test cfg.execution isa AutoExecution
    @test cfg.save_dir == "./data/fits"

    # The legacy nested shape constructs the same flat config.
    tc = Training.TrainingConfig(sampler = sampler,
                                 strategy = Training.Independent(parallel = true,
                                                                 max_concurrent_splits = 3))
    legacy = FitConfig(name = "nested", model = model, splitter = splitter,
                       training_config = tc)
    @test legacy.sampler === sampler
    @test legacy.execution isa AutoExecution
    @test legacy.execution.max_concurrent_splits == 3
    @test INF.legacy_checkpointing(tc) == (nothing, false)

    # …and the synthesised `training_config` view reads back what `save_experiment` reads.
    @test legacy.training_config isa INF.LegacyTrainingConfig
    @test legacy.training_config.sampler === sampler
    @test legacy.training_config.checkpoint_dir === nothing
    @test :training_config in propertynames(legacy)

    @test_throws ErrorException FitConfig(name = "x", model = model, splitter = splitter)
    @test_throws ErrorException FitConfig(name = "x", model = model, splitter = splitter,
                                          training_config = (; strategy = nothing))
end

@testset "Execution strategy resolution" begin
    queued = BayesianFootball.Samplers.QueuedNUTSConfig(n_samples = 10, n_chains = 2)
    map_cfg = BayesianFootball.Samplers.MAPConfig()

    @test INF.resolve_execution(AutoExecution(), queued) isa QueuedExecution
    @test INF.resolve_execution(AutoExecution(max_concurrent_tasks = 7), queued
                                ).max_concurrent_tasks == 7
    @test INF.resolve_execution(SequentialExecution(), queued) isa SequentialExecution
    @test INF.resolve_execution(ThreadedExecution(max_concurrent_splits = 2), map_cfg
                                ).max_concurrent_splits == 2

    resolved = INF.resolve_execution(AutoExecution(), map_cfg)
    @test resolved isa (Threads.nthreads() > 1 ? ThreadedExecution : SequentialExecution)

    # The legacy strategy maps across, caps and all, and defers the queued/threaded
    # choice exactly as `train_independent` does — to the sampler.
    @test INF.execution_from_strategy(Training.Independent(parallel = false)) isa
          SequentialExecution
    auto = INF.execution_from_strategy(
        Training.Independent(parallel = true, max_concurrent_tasks = 11))
    @test auto isa AutoExecution
    @test auto.max_concurrent_tasks == 11
    @test INF.resolve_execution(auto, queued).max_concurrent_tasks == 11
end


# ==============================================================================
# 2. CONVERGENCE AUDIT
# ==============================================================================

@testset "Convergence audit" begin
    healthy = INF.audit_fold(1, inference_chain(seed = 11))
    @test healthy.applicable
    @test healthy.n_params == 2
    @test healthy.n_draws == 200 && healthy.n_chains == 4
    @test healthy.n_transitions == 800
    @test healthy.max_rhat < 1.01
    @test healthy.min_ess_bulk > 400
    @test healthy.n_divergent == 0
    @test healthy.divergence_rate == 0.0

    summary = audit_convergence([FoldFit(1, inference_chain(seed = 11),
                                         Data.SplitMetaData(1, "a", "b", 1, 1, 0))])
    @test summary isa ConvergenceSummary
    @test summary.passed
    @test isempty(summary.failures)
    @test isempty(summary.failed_gates)
    @test isempty(summary.abstained)
    @test summary.n_folds == 1 && summary.n_applicable == 1
    @test occursin("PASS", INF.diagnostics_line(summary))

    # R-hat and rank-normalised ESS both read between-chain variance, so a chain built
    # to break one breaks the other by construction. Assert what is true, not more.
    broken = summarise_convergence([INF.audit_fold(1, inference_chain(seed = 12,
                                                                     rhat_break = true))])
    @test !broken.passed
    @test "R-hat" in broken.failed_gates
    @test broken.max_rhat > 1.01
    @test !isempty(broken.failures)
    @test any(f -> occursin("R-hat", f), broken.failures)
    @test broken.folds[1].worst_rhat_param === :a

    # Divergences trip ONLY their own gate.
    diverged = summarise_convergence([INF.audit_fold(1, inference_chain(seed = 13,
                                                                       n_divergent = 40))])
    @test !diverged.passed
    @test diverged.failed_gates == ["divergences"]
    @test diverged.n_divergent == 40
    @test diverged.divergence_rate ≈ 40 / 800

    # Tree-depth saturation is a performance gate, and separate from the four
    # correctness ones.
    capped = summarise_convergence([INF.audit_fold(1, inference_chain(seed = 14,
                                                                     depth_capped = 100);
                                                   max_depth = 10)])
    @test capped.failed_gates == ["tree depth"]
    @test capped.max_tree_depth == 10
    @test capped.n_depth_capped == 100

    # A sampler that records nothing must not earn a clean bill of health by silence.
    bare = summarise_convergence([INF.audit_fold(1, inference_bare_chain(seed = 15))])
    @test sort(bare.abstained) == ["BFMI", "divergences", "tree depth"]
    @test isempty(bare.failed_gates)
    @test isnan(bare.divergence_rate)
    @test isnan(bare.min_bfmi)

    # A point estimate has no R-hat: excluded from every reduction, and counted.
    point = summarise_convergence([INF.audit_fold(1, "a MAP point estimate"),
                                   INF.audit_fold(2, inference_chain(seed = 16))])
    @test point.n_folds == 2
    @test point.n_applicable == 1
    @test !point.folds[1].applicable
    @test point.worst_rhat_fold == 2

    # Re-gating an existing summary needs no chains.
    strict = summarise_convergence(summary.folds;
                                   thresholds = ConvergenceThresholds(min_ess = 1e6))
    @test !strict.passed
    @test "bulk ESS" in strict.failed_gates
    @test "tail ESS" in strict.failed_gates
    @test summary.passed                       # the original verdict is untouched

    # The table renders for both applicable and point-estimate folds.
    io = IOBuffer()
    convergence_table(point; io = io)
    out = String(take!(io))
    @test occursin("point estimate", out)
    @test occursin("ok", out)
end

@testset "BFMI against its analytic value" begin
    # For an AR(1) energy series E-BFMI → 2(1 − φ). A long series is needed: at 300
    # draws and φ = 0.97 the ratio holds ~5 effectively independent points and its
    # expectation is not the ratio of expectations.
    for φ in (0.2, 0.5, 0.8)
        n = 20_000
        e = zeros(n, 1)
        rng = Xoshiro(round(Int, 100 * φ))
        for i in 2:n
            e[i, 1] = φ * e[i - 1, 1] + randn(rng)
        end
        measured = only(INF.bfmi(e))
        target = 2 * (1 - φ)
        @test abs(measured - target) / target < 0.10
    end

    @test isempty(INF.bfmi(zeros(1, 2)))
    @test all(isnan, INF.bfmi(zeros(10, 2)))       # zero variance → undefined, not 0/0

    low = summarise_convergence([INF.audit_fold(
        1, inference_chain(seed = 17, energy = repeat(cumsum(ones(200)), 1, 4)))])
    @test "BFMI" in low.failed_gates
end


# ==============================================================================
# 3. PERSISTENCE
# ==============================================================================

@testset "Atomic writes" begin
    dir = mktempdir()
    target = joinpath(dir, "payload.txt")

    @test INF.atomic_write(tmp -> write(tmp, "first"), target) == target
    @test read(target, String) == "first"

    # A failed write leaves the PREVIOUS version intact and no scratch behind.
    @test_throws ErrorException INF.atomic_write(target) do tmp
        write(tmp, "half")
        error("interrupted")
    end
    @test read(target, String) == "first"
    @test readdir(dir) == ["payload.txt"]

    INF.atomic_write(tmp -> write(tmp, "second"), target)
    @test read(target, String) == "second"
end

@testset "Fit serialization roundtrip" begin
    chs = [inference_chain(seed = 20 + i) for i in 1:3]
    dir = mktempdir()
    fit = fit_model(inference_config(chs, dir; name = "roundtrip");
                    feature_sets = inference_folds(3), quiet = true)

    path = save_fit(fit; quiet = true)
    @test isdir(path)
    @test isfile(joinpath(path, "results.jld2"))
    @test isfile(joinpath(path, "meta.json"))
    @test isfile(joinpath(path, "config.json"))
    @test !any(f -> occursin(".tmp.", f), readdir(path))

    back = load_fit(path; quiet = true)
    @test back isa Fit
    @test length(back) == length(fit)
    @test fit_name(back) == "roundtrip"
    @test back.config.tags == fit.config.tags
    @test back.diagnostics.passed == fit.diagnostics.passed
    @test back.diagnostics.max_rhat == fit.diagnostics.max_rhat
    @test [f.fold for f in back] == [1, 2, 3]
    @test size(back[2].chain) == size(fit[2].chain)
    @test Array(back[2].chain) == Array(fit[2].chain)
    @test back[2].meta == fit[2].meta

    # The sidecar answers "is this run usable" without opening the binary.
    meta = read_fit_meta(path)
    @test meta.name == "roundtrip"
    @test meta.kind == "Fit"
    @test meta.model == "InferenceMockModel"
    @test meta.n_folds == 3
    @test meta.converged === fit.diagnostics.passed
    @test meta.max_rhat ≈ fit.diagnostics.max_rhat
    @test meta.n_divergent == fit.diagnostics.n_divergent
    @test !meta.has_oos_latents

    rows = list_fits(dir; quiet = true)
    @test length(rows) == 1
    @test rows[1].path == path
    @test length(load_fits(dir)) == 1

    # A directory with no sidecar is listed with an UNKNOWN verdict, never a pass.
    rm(joinpath(path, "meta.json"))
    bare = read_fit_meta(path)
    @test bare.converged === missing
    @test isnan(bare.max_rhat)

    # …and a directory with nothing in it at all is not a fit.
    @test_throws ErrorException load_fit(mktempdir(); quiet = true)
    @test isempty(list_fits(joinpath(dir, "does_not_exist"); quiet = true))
end

@testset "Latent cache" begin
    dir = mktempdir()
    latents = CountLatents([1, 2], [1.1 1.2; 0.9 1.0], [0.8 0.7; 1.3 1.4])

    @test save_latents(dir, latents) == joinpath(dir, "oos_latents.jls")
    reloaded = load_latents(dir)
    @test reloaded isa CountLatents
    @test latent_match_ids(reloaded) == [1, 2]
    @test reloaded.λ_home == latents.λ_home

    @test load_latents(mktempdir()) === nothing        # a miss is `nothing`

    # A cache that is not a latents container is a MISS, not a garbage return value.
    # `deserialize` does not reliably throw on a foreign file, so the guard is on the
    # type that comes back rather than only on the read.
    corrupt = mktempdir()
    write(joinpath(corrupt, "oos_latents.jls"), "not a serialized object")
    @test inference_quiet(() -> load_latents(corrupt)) === nothing
    junk = mktempdir()
    Serialization.serialize(joinpath(junk, "oos_latents.jls"), [1, 2, 3])
    @test inference_quiet(() -> load_latents(junk)) === nothing
end


# ==============================================================================
# 4. LEGACY COMPATIBILITY
# ==============================================================================

@testset "Legacy property bridge" begin
    chs = [inference_bare_chain(seed = 30 + i) for i in 1:2]
    fit = fit_model(inference_config(chs, mktempdir());
                    feature_sets = inference_folds(2), quiet = true)

    tr = fit.training_results
    @test tr isa INF.LegacyTrainingResults
    @test length(tr) == 2
    @test length(tr.items) == 2
    @test tr.items[1][1] === chs[1]
    @test tr.items[2][2] === fit[2].meta
    @test tr[1] == (chs[1], fit[1].meta)
    @test [c for (c, _) in tr] == chs
    @test fit.vocabulary === nothing
    @test :training_results in propertynames(fit)
    @test :vocabulary in propertynames(fit)

    # `.items` materialises rather than caching, so the two views cannot drift.
    @test tr.items !== tr.items

    hand_made = INF.legacy_training_results([(chs[1], fit[1].meta), nothing,
                                             (chs[2], fit[2].meta)])
    @test length(hand_made) == 2
    @test hand_made.items[2][1] === chs[2]
end

@testset "ExperimentResults <-> Fit bridge" begin
    chs = [inference_chain(seed = 40 + i) for i in 1:3]
    dir = mktempdir()
    fit = fit_model(inference_config(chs, dir; name = "bridge");
                    feature_sets = inference_folds(3), quiet = true)

    # Fit -> legacy
    legacy = experiment_from_fit(fit)
    @test legacy isa Experiments.ExperimentResults
    @test legacy.config isa Experiments.ExperimentConfig
    @test legacy.config.name == "bridge"
    @test legacy.config.training_config isa Training.TrainingConfig
    @test legacy.config.training_config.sampler === fit.config.sampler
    @test legacy.config.training_config.strategy isa Training.Independent
    @test legacy.vocabulary === nothing
    @test legacy.save_path == fit.save_path
    @test legacy.training_results isa Training.TrainingResults
    @test length(legacy.training_results.items) == 3
    @test legacy.training_results.items[2][1] === chs[2]

    # legacy -> Fit, audited on the way in because the legacy container has no such field
    round = fit_from_experiment(legacy)
    @test round isa Fit
    @test length(round) == 3
    @test round[1].chain === chs[1]
    @test round.config.name == "bridge"
    @test round.config.sampler === fit.config.sampler
    @test round.diagnostics isa ConvergenceSummary
    @test round.diagnostics.passed == fit.diagnostics.passed
    @test round.diagnostics.max_rhat ≈ fit.diagnostics.max_rhat

    # `upgrade_to_fit` on something that is already a Fit is the identity.
    @test upgrade_to_fit(fit) === fit

    # The strategy mapping is not a guess: it reports what the run would have done.
    queued = BayesianFootball.Samplers.QueuedNUTSConfig(n_samples = 10, n_chains = 2)
    strat = INF.legacy_strategy(AutoExecution(max_concurrent_tasks = 5), queued)
    @test strat isa Training.Independent
    @test strat.parallel
    @test strat.max_concurrent_tasks == 5
    @test !INF.legacy_strategy(SequentialExecution(), queued).parallel

    # A container with no chains, and one with no `config`, both fail loudly.
    @test_throws ErrorException upgrade_to_fit((; config = fit.config))
    @test_throws ErrorException upgrade_to_fit("not a result")
end

@testset "Legacy run on disk upgrades" begin
    chs = [inference_chain(seed = 50 + i) for i in 1:2]
    fit = fit_model(inference_config(chs, mktempdir(); name = "legacy_disk");
                    feature_sets = inference_folds(2), quiet = true)

    # Saved by the LEGACY writer, into the legacy layout.
    path = joinpath(mktempdir(), "legacy_disk_20250104_113000")
    Experiments.save_experiment(experiment_from_fit(fit); path = path, quiet = true)
    @test isfile(joinpath(path, "results.jld2"))

    upgraded = load_fit(path; quiet = true)
    @test upgraded isa Fit
    @test length(upgraded) == 2
    @test Array(upgraded[1].chain) == Array(chs[1])
    @test upgraded.config.name == "legacy_disk"
    @test upgraded.config.sampler isa ReplaySampler              # flattened out of the nest
    @test length(upgraded.config.sampler.chains) == 2
    @test upgraded.diagnostics.passed                            # computed on load
    @test upgraded.save_path == path

    # The timestamp comes back out of the directory name, and the elapsed time out of
    # the `time:` tag the legacy runner wrote — approximate by construction.
    @test Dates.year(upgraded.metadata.timestamp) == 2025
    @test Dates.month(upgraded.metadata.timestamp) == 1
    @test Dates.day(upgraded.metadata.timestamp) == 4
    @test INF._inf_legacy_elapsed((; tags = ["time:3m 20s"])) == 200.0
    @test INF._inf_legacy_elapsed((; tags = ["time:2h 15m"])) == 8100.0
    @test INF._inf_legacy_elapsed((; tags = ["time:12.4s"])) == 12.4
    @test INF._inf_legacy_elapsed((; tags = String[])) == 0.0
end


# ==============================================================================
# 5. THE ENGINE
# ==============================================================================

@testset "Fold dispatch and failure recording" begin
    chs = [inference_bare_chain(seed = 60 + i) for i in 1:3]
    fss = inference_folds(3)
    model = InferenceMockModel()

    for exec in (SequentialExecution(), ThreadedExecution(max_concurrent_splits = 2),
                 QueuedExecution(max_concurrent_tasks = 2))
        out = run_folds(model, ReplaySampler(chs), exec, fss)
        @test length(out) == 3
        @test out == chs
    end

    seen = Int[]
    lk = ReentrantLock()
    run_folds(model, ReplaySampler(chs), SequentialExecution(), fss;
              on_progress = (done, total) -> (lock(lk) do; push!(seen, done); end))
    @test seen == [1, 2, 3]

    # A fold that throws is logged and left `nothing`; its siblings survive.
    short = ReplaySampler(chs[1:2])
    out = inference_quiet(() -> run_folds(model, short, SequentialExecution(), fss))
    @test out[1] === chs[1] && out[2] === chs[2] && out[3] === nothing

    # …and the loss survives into the Fit as a tag, rather than as a short vector with
    # nothing that says why.
    fit = inference_quiet() do
        fit_model(inference_config(chs[1:2], mktempdir()); feature_sets = fss, quiet = true)
    end
    @test length(fit) == 2
    @test "folds_failed:1" in fit.config.tags

    # Every fold failing is an error, not an empty Fit.
    @test_throws ErrorException inference_quiet() do
        fit_model(inference_config(Chains[], mktempdir()); feature_sets = fss, quiet = true)
    end
    @test_throws ErrorException fit_model(inference_config(chs, mktempdir());
                                          feature_sets = [], quiet = true)
end

@testset "Checkpoints interoperate with Training" begin
    chs = [inference_bare_chain(seed = 70 + i) for i in 1:3]
    fss = inference_folds(3)
    ckpt = mktempdir()

    # The engine's path is `Training.get_checkpoint_path`'s, zero-padded to three.
    @test INF.checkpoint_path(ckpt, 7) == joinpath(ckpt, "split_007.jls")
    @test INF.checkpoint_path(ckpt, 7) == Training.get_checkpoint_path(ckpt, 7)

    fit = fit_model(inference_config(chs, mktempdir()); feature_sets = fss,
                    checkpoint_dir = ckpt, quiet = true)
    @test length(fit) == 3
    @test isfile(INF.checkpoint_path(ckpt, 1))

    # A resume reads them back and samples nothing: the replay sampler is empty, so any
    # fold that was NOT restored would fail.
    resumed = fit_model(inference_config(Chains[], mktempdir()); feature_sets = fss,
                        checkpoint_dir = ckpt, quiet = true)
    @test length(resumed) == 3
    @test Array(resumed[2].chain) == Array(chs[2])

    # A checkpoint written by the LEGACY writer restores too — same filename, same
    # `(result, metadata)` payload, unwrapped back to the sampler result.
    legacy_ckpt = mktempdir()
    for i in 1:3
        Training.save_split_checkpoint(legacy_ckpt, i, (chs[i], fss[i][2]))
    end
    from_legacy = fit_model(inference_config(Chains[], mktempdir()); feature_sets = fss,
                            checkpoint_dir = legacy_ckpt, quiet = true)
    @test length(from_legacy) == 3
    @test from_legacy[3].chain isa Chains
    @test Array(from_legacy[3].chain) == Array(chs[3])

    # A corrupt checkpoint is a cache miss, not a fatal resume.
    write(INF.checkpoint_path(legacy_ckpt, 2), "garbage")
    recovered = inference_quiet() do
        fit_model(inference_config(chs, mktempdir()); feature_sets = fss,
                  checkpoint_dir = legacy_ckpt, quiet = true)
    end
    @test length(recovered) == 3

    # Cleanup removes them only once every fold has landed.
    fit_model(inference_config(chs, mktempdir()); feature_sets = fss,
              checkpoint_dir = ckpt, cleanup_checkpoints = true, quiet = true)
    @test !isfile(INF.checkpoint_path(ckpt, 1))
end

@testset "Latent extraction and merge" begin
    ids_a, ids_b = [1, 2], [3, 4, 5]
    λha = [1.1 1.2; 0.9 1.0]
    λaa = [0.8 0.7; 1.3 1.4]
    λhb = [1.5 1.6; 1.7 1.8; 1.9 2.0]
    λab = [0.5 0.6; 0.7 0.8; 0.9 1.0]

    a = CountLatents(ids_a, λha, λaa)
    b = CountLatents(ids_b, λhb, λab)
    merged = merge_latents([a, b])

    @test merged isa CountLatents
    @test n_matches(merged) == 5
    @test n_draws(merged) == 2
    # Fold-then-fixture order: a merge that sorted would price fixture i with fixture j's
    # posterior and every downstream number would still look reasonable.
    @test latent_match_ids(merged) == [1, 2, 3, 4, 5]
    @test merged.λ_home[1:2, :] == λha
    @test merged.λ_home[3:5, :] == λhb
    @test merged.λ_away[3:5, :] == λab

    @test merge_latents([a]) === a
    @test merge_latents([]) === nothing

    # Differing draw counts and differing families are errors naming both folds, not a
    # length mismatch inside a pricing kernel hours later.
    @test_throws ErrorException merge_latents([a, CountLatents([9], [1.0 1.0 1.0],
                                                               [1.0 1.0 1.0])])
    @test_throws ErrorException merge_latents(
        [a, CountLatents(ids_b, λhb, λab, (; r_h = λhb, r_a = λab))])

    # NegBin observation params ride along with the merge.
    na = CountLatents(ids_a, λha, λaa, (; r_h = fill(4.0, 2, 2), r_a = fill(5.0, 2, 2)))
    nb = CountLatents(ids_b, λhb, λab, (; r_h = fill(6.0, 3, 2), r_a = fill(7.0, 3, 2)))
    nm = merge_latents([na, nb])
    @test observation_family(nm) === :negbin
    @test nm.observation_params.r_h[1, 1] == 4.0
    @test nm.observation_params.r_h[5, 1] == 6.0

    # The whole reason for a `Fit` to carry latents at all: an unregistered family is
    # recorded as a REASON on the fit, not lost to a log line.
    chs = [inference_bare_chain(seed = 80 + i) for i in 1:2]
    fss = inference_folds(2)
    oos = [DataFrame(match_id = [1, 2]), DataFrame(match_id = [3])]
    fit = fit_model(inference_config(chs, mktempdir()); feature_sets = fss,
                    oos_fixtures = oos, quiet = true)
    @test fit.latents === nothing
    @test any(t -> startswith(t, "latents:failed"), fit.config.tags)

    empty_oos = fit_model(inference_config(chs, mktempdir()); feature_sets = fss,
                          oos_fixtures = [DataFrame(match_id = Int[]),
                                          DataFrame(match_id = Int[])],
                          quiet = true)
    @test empty_oos.latents === nothing
    @test "latents:none(no out-of-sample fixtures)" in empty_oos.config.tags

    skipped = fit_model(inference_config(chs, mktempdir()); feature_sets = fss,
                        oos_fixtures = oos, with_latents = false, quiet = true)
    @test skipped.latents === nothing
    @test !any(t -> startswith(t, "latents:"), skipped.config.tags)
end


# ==============================================================================
# 6. IN-GAME LIVE RATE SOLVER
# ==============================================================================

@testset "In-game live rate solver" begin
    n_draws_pre = 8
    pre = CountLatents([11, 22],
                       0.5 .+ rand(Xoshiro(101), 2, n_draws_pre),
                       0.5 .+ rand(Xoshiro(102), 2, n_draws_pre))

    # --- the identity kernel: one bin [0,1], every coefficient zero, so the integral is
    #     exactly 1.0 with no rounding and Λ must equal the pre-game rate BIT FOR BIT.
    identity_ws = IngameRatesWorkspace(zeros(n_draws_pre), zeros(n_draws_pre),
                                       zeros(n_draws_pre), zeros(n_draws_pre),
                                       zeros(n_draws_pre), zeros(1, n_draws_pre),
                                       [0.0, 1.0], 1.0)
    rates = alloc_live_rates(identity_ws)
    @test rates isa LiveMatchRates
    @test length(rates) == n_draws_pre
    @test solve_ingame_rates!(rates, identity_ws, pre, 1, kickoff_state()) === rates
    @test rates.Λ_home == pre.λ_home[1, :]
    @test rates.Λ_away == pre.λ_away[1, :]
    solve_ingame_rates!(rates, identity_ws, pre, 2, kickoff_state())
    @test rates.Λ_home == pre.λ_home[2, :]

    # --- zero allocations, warmed, against an empty-closure baseline.
    solve_ingame_rates!(rates, identity_ws, pre, 1, kickoff_state())
    @test @allocated(solve_ingame_rates!(rates, identity_ws, pre, 1, kickoff_state())) == 0

    λh1 = pre.λ_home[1, :]
    λa1 = pre.λ_away[1, :]
    state = MatchState(t = 63.0, g_h = 1, g_a = 1, r_a = 1)
    solve_ingame_rates!(rates, identity_ws, λh1, λa1, state)
    @test @allocated(solve_ingame_rates!(rates, identity_ws, λh1, λa1, state)) == 0

    # A whole repricing tick — every state a live match walks through — allocates nothing.
    ticks = [MatchState(t = Float64(t), g_h = t ÷ 40, g_a = t ÷ 60, r_a = t ÷ 80)
             for t in 0:5:90]
    inference_sweep(rates, identity_ws, pre, ticks)
    @test @allocated(inference_sweep(rates, identity_ws, pre, ticks)) == 0

    # --- the model's own bins, and what the state terms do.
    model = NHPPIntensityModel(Δt = 5.0, Tend = 95.0, time_bins = false)
    n_bins = INF.n_time_bins(model)
    @test n_bins == 19
    edges = collect(range(0.0, 95.0; length = n_bins + 1))
    ws = IngameRatesWorkspace(fill(-4.0, n_draws_pre), zeros(n_draws_pre),
                              fill(0.3, n_draws_pre), fill(-0.2, n_draws_pre),
                              fill(0.25, n_draws_pre), zeros(n_bins, n_draws_pre),
                              edges, 95.0)
    @test INF.workspace_n_draws(ws) == n_draws_pre
    @test INF.workspace_n_bins(ws) == n_bins

    at_0  = solve_ingame_rates(ws, pre, 1, MatchState(t = 0.0))
    at_60 = solve_ingame_rates(ws, pre, 1, MatchState(t = 60.0))
    @test all(at_60.Λ_home .< at_0.Λ_home)                    # less match left
    @test all(iszero, solve_ingame_rates(ws, pre, 1, MatchState(t = 95.0)).Λ_home)

    trailing = solve_ingame_rates(ws, pre, 1, MatchState(t = 60.0, g_h = 0, g_a = 1))
    @test all(trailing.Λ_home .> at_60.Λ_home)                # a trailing side attacks
    @test all(trailing.Λ_away .< at_60.Λ_away)                # a leading side sits back

    a_man_up = solve_ingame_rates(ws, pre, 1, MatchState(t = 60.0, r_a = 1))
    @test all(a_man_up.Λ_home .> at_60.Λ_home)
    @test all(a_man_up.Λ_away .< at_60.Λ_away)

    # `t` in the middle of a bin integrates that bin's REMAINDER, not the whole of it.
    mid = solve_ingame_rates(ws, pre, 1, MatchState(t = 62.5))
    at_65 = solve_ingame_rates(ws, pre, 1, MatchState(t = 65.0))
    @test all(at_60.Λ_home .> mid.Λ_home)
    @test all(mid.Λ_home .> at_65.Λ_home)

    # --- the state helpers.
    @test kickoff_state() == MatchState(0.0, 0, 0, 0, 0)
    @test INF.goal_diff(MatchState(t = 10.0, g_h = 2, g_a = 1)) == 1
    @test INF.man_advantage(MatchState(t = 10.0, r_a = 2, r_h = 1)) == 1.0

    # --- the construction contract: a mismatch is caught here, not recycled silently.
    @test_throws ErrorException IngameRatesWorkspace(zeros(3), zeros(2), zeros(3),
                                                    zeros(3), zeros(3), zeros(1, 3),
                                                    [0.0, 1.0], 1.0)
    @test_throws ErrorException IngameRatesWorkspace(zeros(3), zeros(3), zeros(3),
                                                    zeros(3), zeros(3), zeros(2, 3),
                                                    [0.0, 1.0], 1.0)
    @test_throws ErrorException IngameRatesWorkspace([NaN, 0.0], zeros(2), zeros(2),
                                                     zeros(2), zeros(2), zeros(1, 2),
                                                     [0.0, 1.0], 1.0)
    @test_throws ErrorException LiveMatchRates(zeros(3), zeros(2))
    @test_throws ErrorException solve_ingame_rates!(LiveMatchRates(zeros(3), zeros(3)),
                                                   identity_ws, pre, 1, kickoff_state())
    @test_throws ErrorException solve_ingame_rates!(rates, identity_ws, pre, 9,
                                                   kickoff_state())
end

@testset "In-game workspace from a chain" begin
    model = NHPPIntensityModel(Δt = 5.0, Tend = 95.0, time_bins = true)
    n_bins = INF.n_time_bins(model)
    n, n_chains = 50, 2
    scalars = [:α, :β, :γ_tr, :γ_ld, :γ_man, :σ_time]
    vals = Array{Float64, 3}(undef, n, length(scalars) + n_bins, n_chains)
    rng = Xoshiro(9)
    for c in 1:n_chains
        vals[:, 1, c] .= -4.0 .+ 0.01 .* randn(rng, n)
        vals[:, 2, c] .= 0.05 .* randn(rng, n)
        vals[:, 3, c] .= 0.3
        vals[:, 4, c] .= -0.2
        vals[:, 5, c] .= 0.25
        vals[:, 6, c] .= 0.1
        for b in 1:n_bins
            vals[:, 6 + b, c] .= 0.01 * b
        end
    end
    chain = Chains(vals, vcat(scalars, [Symbol("z_time[$b]") for b in 1:n_bins]))

    ws = build_ingame_workspace(chain, model, 12)
    @test INF.workspace_n_draws(ws) == 12
    @test INF.workspace_n_bins(ws) == n_bins
    @test ws.Tend == 95.0
    @test ws.edges[1] == 0.0 && ws.edges[end] == 95.0
    @test all(ws.γ_trail .== 0.3)
    @test all(ws.γ_red .== 0.25)
    @test any(!iszero, ws.δ_time)                       # z_time × σ_time landed

    # The pairing is resolved ONCE, with a seeded RNG: two repricings of an unchanged
    # match state must return the same number, so the same seed must give the same draws.
    @test build_ingame_workspace(chain, model, 12; seed = 7).α ==
          build_ingame_workspace(chain, model, 12; seed = 7).α

    # An exact draw count pairs 1:1 in order rather than resampling.
    exact = build_ingame_workspace(chain, model, n * n_chains)
    @test exact.α == vec(Array(chain[:α]))

    # `time_bins = false` disables the offset without needing the sites to be absent.
    flat = build_ingame_workspace(chain, NHPPIntensityModel(time_bins = false), 12)
    @test all(iszero, flat.δ_time)

    # A missing optional site is exp(0) = 1, not an error; a missing level term IS one.
    partial = Chains(vals[:, 1:2, :], [:α, :β])
    lean = build_ingame_workspace(partial, model, 12)
    @test all(iszero, lean.γ_red)
    @test all(iszero, lean.δ_time)
    @test_throws ErrorException build_ingame_workspace(Chains(vals[:, 2:2, :], [:β]),
                                                       model, 12)
    @test_throws ErrorException build_ingame_workspace(chain, model, 0)

    # And the paired workspace drives the solver against a container of the same width.
    pre = CountLatents([11], fill(1.4, 1, 12), fill(1.1, 1, 12))
    Λ = solve_ingame_rates(ws, pre, 1, kickoff_state())
    @test all(isfinite, Λ.Λ_home)
    @test all(>(0.0), Λ.Λ_home)
    @test_throws ErrorException solve_ingame_rates(
        build_ingame_workspace(chain, model, 5), pre, 1, kickoff_state())
end


# ==============================================================================
# 7. DISPLAY
# ==============================================================================

@testset "Display" begin
    chs = [inference_chain(seed = 90 + i) for i in 1:2]
    fit = fit_model(inference_config(chs, mktempdir(); name = "shown");
                    feature_sets = inference_folds(2), quiet = true)

    @test occursin("Fit(shown, 2 folds)", sprint(show, fit))
    long = sprint(show, MIME"text/plain"(), fit)
    @test occursin("InferenceMockModel", long)
    @test occursin("diagnostics", long)
    @test occursin("save_path", long)

    @test occursin("FoldFit(1", sprint(show, fit[1]))
    @test occursin("FitConfig(shown", sprint(show, fit.config))
    @test occursin("shown", sprint(show, MIME"text/plain"(), fit.config))
    @test occursin("ConvergenceSummary", sprint(show, fit.diagnostics))
    verdict = fit.diagnostics.passed ? "PASS" : "FAIL"
    @test occursin(verdict, sprint(show, MIME"text/plain"(), fit.diagnostics))
    @test occursin(verdict, sprint(show, fit.diagnostics))
    @test occursin("TrainingConfig(strategy=", sprint(show, fit.config.training_config))
    @test occursin("FitMetadata", sprint(show, fit.metadata))
    @test occursin("ReplaySampler(2 folds)", sprint(show, fit.config.sampler))
    @test occursin("63.0'", sprint(show, MatchState(t = 63.0, g_h = 1, g_a = 1)))
    @test occursin("NHPPIntensityModel", sprint(show, NHPPIntensityModel()))
    @test occursin("IngameRatesWorkspace",
                   sprint(show, IngameRatesWorkspace(zeros(2), zeros(2), zeros(2),
                                                     zeros(2), zeros(2), zeros(1, 2),
                                                     [0.0, 1.0], 1.0)))
    @test occursin("LiveMatchRates(2 draws)", sprint(show, LiveMatchRates(zeros(2),
                                                                          zeros(2))))

    @test INF.format_elapsed(12.44) == "12.4s"
    @test INF.format_elapsed(200.0) == "3m 20s"
    @test INF.format_elapsed(8100.0) == "2h 15m"
    # Provenance capture must not be able to kill a six-hour run: a non-repository
    # directory is `"unknown"`, never a throw. (git writes its own complaint to stderr.)
    @test redirect_stderr(() -> INF.git_commit_id(mktempdir()), devnull) == "unknown"
end

end # testset "Unified inference framework"
