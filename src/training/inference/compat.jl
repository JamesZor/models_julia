# src/training/inference/compat.jl
#
# The bidirectional bridge between `Experiments.ExperimentResults` and `Fit`.
#
#     upgrade_to_fit / fit_from_experiment    legacy container  →  Fit
#     experiment_from_fit                     Fit               →  legacy container
#
# NOTHING IN `src/experiments/` OR `src/training/` IS SHADOWED OR REPLACED.
# `Experiments.run_experiment`, `Experiments.save_experiment`, `Training.train` and every
# other legacy entry point behave exactly as before; this file adds a conversion in each
# direction so a caller can move a result between the two worlds without re-running
# anything. That is a deliberate narrowing of the prototype, which aliased the legacy
# NAMES onto the new types inside its own module — a thing that cannot be done inside
# `BayesianFootball` itself, because `Experiments` and `Training` are already bound here.
#
# THE FORWARD DIRECTION IS DUCK-TYPED, ON PURPOSE.
#
# `upgrade_to_fit` reads its argument through `hasproperty`/`getproperty` rather than
# dispatching on `ExperimentResults`. That is not laziness: JLD2 hands back a
# `ReconstructedMutable`/`ReconstructedStatic` — a property bag with no stable type — for
# any struct whose definition has drifted since the file was written, and
# `runner.jl:265` already carries a hand-written `Base.convert` shim for exactly one such
# drift (`Training.Independent` gaining a field). That approach needs a new shim per
# struct per change; reading through `hasproperty` needs none. It also means this file
# has no load-time dependency on `Experiments`, which is loaded AFTER `Training`.
#
# THE REVERSE DIRECTION resolves `Experiments` and `Training` at CALL time through the
# root module, for the same load-order reason.
#
# A legacy container carries no diagnostics, so `upgrade_to_fit` AUDITS the chains it
# recovered. That is a few seconds on a 40-fold run and it means an old result answers
# `fit.diagnostics.passed` like any other.

"The package root, resolved from the module tree rather than by name lookup, so that
`Experiments` — loaded after `Training` — can be reached at call time."
const INFERENCE_ROOT = parentmodule(parentmodule(@__MODULE__))

"A sibling module of `Training`, fetched when it is needed rather than when this loads."
function _inf_root_module(name::Symbol)
    isdefined(INFERENCE_ROOT, name) || error(
        "Inference: `$(INFERENCE_ROOT).$name` is not loaded yet. This conversion is only " *
        "available after the whole package has finished loading.")
    return getfield(INFERENCE_ROOT, name)
end


# ==============================================================================
# 1. LEGACY  →  FIT
# ==============================================================================

"""
    upgrade_to_fit(obj; save_path = "", thresholds = ConvergenceThresholds()) -> Fit

Turn whatever came out of a `.jld2` into a `Fit`.

Dispatches on the one case it can name — an object that already IS a `Fit` — and falls
through to a duck-typed reader for the ones it cannot, because a JLD2-reconstructed
struct has no stable type to dispatch on, only properties. See the file header.

The chains it recovers are audited, so the returned `Fit` has real diagnostics where the
legacy container had no such field. `save_path` is where the OOS latent cache is looked
for; an object carrying its own `save_path` wins over the argument.
"""
upgrade_to_fit(f::Fit; kwargs...) = f

function upgrade_to_fit(obj; save_path::AbstractString = "",
                        thresholds::ConvergenceThresholds = ConvergenceThresholds())
    hasproperty(obj, :config) || error(
        "upgrade_to_fit: cannot read a $(typeof(obj)) — it has no `config` property. " *
        "Expected a `Fit`, an `ExperimentResults`, or a JLD2-reconstructed one of those.")

    legacy_cfg = obj.config
    tr = hasproperty(obj, :training_results) ? obj.training_results : nothing
    tr === nothing && error(
        "upgrade_to_fit: a $(typeof(obj)) with a `config` but no `training_results`. " *
        "Nothing here knows how to find its chains.")

    items = hasproperty(tr, :items) ? tr.items : collect(tr)

    folds = _inf_narrow(FoldFit[FoldFit(i, it[1], it[2])
                                for (i, it) in enumerate(items) if it !== nothing])
    isempty(folds) && error("upgrade_to_fit: the legacy container holds no usable folds.")

    sampler = _inf_legacy_sampler(legacy_cfg)
    config = FitConfig(
        name = String(get_or(legacy_cfg, :name, "recovered")),
        model = legacy_cfg.model,
        splitter = legacy_cfg.splitter,
        sampler = sampler,
        execution = AutoExecution(),
        tags = Vector{String}(get_or(legacy_cfg, :tags, String[])),
        description = String(get_or(legacy_cfg, :description, "")),
        save_dir = String(get_or(legacy_cfg, :save_dir, "./data/fits")),
    )

    diagnostics = audit_convergence(folds; thresholds = thresholds,
                                    max_depth = sampler_max_depth(sampler))

    # Where the file ACTUALLY is wins over where the object once said it would go.
    # `Experiments.save_experiment(res; path = …)` does not update `res.save_path`, so a
    # run saved anywhere but its default carries a stale field — and that field is what
    # the OOS cache lookup and the timestamp recovery would otherwise both key on.
    sp = isempty(save_path) ? String(get_or(obj, :save_path, "")) : String(save_path)
    latents = isempty(sp) ? nothing : load_latents(sp)
    latents isa AbstractPosteriorLatents || (latents = nothing)

    meta = FitMetadata(_inf_legacy_timestamp(sp), _inf_legacy_elapsed(legacy_cfg),
                       VERSION, 1, "unknown")

    return Fit(config, folds, latents, diagnostics, meta, sp)
end

"""
    fit_from_experiment(results; kwargs...) -> Fit

`upgrade_to_fit` under the name that says what it is for. Takes an
`Experiments.ExperimentResults` (or anything shaped like one) and returns the equivalent
`Fit`, audited.
"""
fit_from_experiment(results; kwargs...) = upgrade_to_fit(results; kwargs...)

"The sampler out of a legacy config's nested `training_config`, or `nothing`."
function _inf_legacy_sampler(cfg)
    hasproperty(cfg, :sampler) && return cfg.sampler
    if hasproperty(cfg, :training_config)
        tc = cfg.training_config
        hasproperty(tc, :sampler) && return tc.sampler
    end
    return nothing
end

"""
Recover the run time from the legacy `time:` tag `run_experiment` pushes (`runner.jl:73`).

Approximate BY CONSTRUCTION — the tag is a formatted string, so `"3m 20s"` comes back as
200.0 and `"2h 15m"` loses the seconds. Better than zero, and any run saved by this
framework carries the exact value in its sidecar instead.
"""
function _inf_legacy_elapsed(cfg)
    for t in get_or(cfg, :tags, String[])
        startswith(t, "time:") || continue
        s = replace(t, "time:" => "")
        total = 0.0
        m = match(r"(\d+(?:\.\d+)?)h", s); m === nothing || (total += 3600 * parse(Float64, m[1]))
        m = match(r"(\d+(?:\.\d+)?)m", s); m === nothing || (total += 60 * parse(Float64, m[1]))
        m = match(r"(\d+(?:\.\d+)?)s", s); m === nothing || (total += parse(Float64, m[1]))
        return total
    end
    return 0.0
end

"The run time from the directory name `<name>_yyyymmdd_HHMMSS`, or now."
function _inf_legacy_timestamp(path::AbstractString)
    isempty(path) && return now()
    m = match(r"(\d{8})_(\d{6})$", basename(path))
    m === nothing && return now()
    try
        return DateTime(m[1] * m[2], dateformat"yyyymmddHHMMSS")
    catch
        return now()
    end
end


# ==============================================================================
# 2. FIT  →  LEGACY
# ==============================================================================

"""
    legacy_strategy(execution, sampler) -> Training.Independent

Map an `AbstractExecution` back onto the legacy `Training.Independent` strategy, which is
the only shape `TrainingConfig` accepts. `AutoExecution` is resolved against the sampler
first, so what comes back is what the run would actually have done rather than a deferred
decision the legacy struct cannot express.
"""
function legacy_strategy(execution::AbstractExecution, sampler)
    Tr = _inf_root_module(:Training)
    e = resolve_execution(execution, sampler)
    e isa SequentialExecution && return Tr.Independent(parallel = false)
    e isa ThreadedExecution &&
        return Tr.Independent(parallel = true,
                              max_concurrent_splits = e.max_concurrent_splits)
    e isa QueuedExecution &&
        return Tr.Independent(parallel = true,
                              max_concurrent_tasks = e.max_concurrent_tasks)
    return Tr.Independent(parallel = false)
end

"""
    experiment_from_fit(fit; checkpoint_dir = nothing, cleanup_checkpoints = false)
        -> Experiments.ExperimentResults

The reverse bridge: a `Fit` presented as the legacy container, for a call site that is
still typed on `ExperimentResults` — `Experiments.save_experiment`,
`Experiments.Diagnostics.extract_chains`, a runner not yet migrated.

LOSSY IN ONE DIRECTION ONLY, and worth naming: `ExperimentResults` has nowhere to put the
convergence summary, the fit metadata, or the typed OOS latents. Those survive on the
`Fit` you already hold; they do not survive a round trip through this function. Going the
other way (`upgrade_to_fit`) loses nothing, because the legacy container never had them.

`checkpoint_dir` and `cleanup_checkpoints` are the two `TrainingConfig` fields that moved
to `fit_model`'s keywords; pass the values the run actually used if the receiving code
reads them.
"""
function experiment_from_fit(f::Fit; checkpoint_dir::Union{Nothing, String} = nothing,
                             cleanup_checkpoints::Bool = false)
    Exp = _inf_root_module(:Experiments)
    Tr  = _inf_root_module(:Training)
    cfg = getfield(f, :config)

    training_config = Tr.TrainingConfig(
        sampler = cfg.sampler,
        strategy = legacy_strategy(cfg.execution, cfg.sampler),
        checkpoint_dir = checkpoint_dir,
        cleanup_checkpoints = cleanup_checkpoints)

    legacy_config = Exp.ExperimentConfig(
        name = cfg.name,
        model = cfg.model,
        splitter = cfg.splitter,
        training_config = training_config,
        tags = copy(cfg.tags),
        description = cfg.description,
        save_dir = cfg.save_dir)

    items = [as_legacy_tuple(fd) for fd in getfield(f, :folds)]

    return Exp.ExperimentResults(legacy_config,
                                 Tr.TrainingResults(items),
                                 nothing,
                                 getfield(f, :save_path))
end

"""
    legacy_training_results(items) -> LegacyTrainingResults

A `Training.TrainingResults`-shaped view built from a plain vector of `(chain, meta)`
tuples, for code that stitched one together by hand — a resumed run assembling
checkpoints, a test fixture. `nothing` entries are dropped, as they are everywhere else
in this framework.
"""
legacy_training_results(items::AbstractVector) =
    LegacyTrainingResults(_inf_narrow(FoldFit[FoldFit(i, it[1], it[2])
                                              for (i, it) in enumerate(items)
                                              if it !== nothing]))
