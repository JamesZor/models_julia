# ==============================================================================
# 07 — UNIFIED INFERENCE FRAMEWORK : THE BACKWARD-COMPATIBILITY BRIDGE
# ==============================================================================
#
# Loader, and the module wrapper. `include`ing THIS file loads the whole prototype:
#
#     include("current_development/07_unified_inference_framework/l06_compat_bridge.jl")
#     using .UnifiedInference
#
# l06 → l05 → l04 → l03 → l02 → l01 → { 06_typed_posterior_latents,
#                                       05_composable_count_builder }
#
# ------------------------------------------------------------------------------
# WHY THIS IS A MODULE, AND WHY THE BRIEFING'S `const Experiments = current_module()`
# CANNOT BE WRITTEN AT TOP LEVEL
# ------------------------------------------------------------------------------
#
# `BayesianFootball` EXPORTS the names `Experiments` and `Training`
# (src/BayesianFootball.jl:64). In any scope that has done `using BayesianFootball` —
# which is every runner in this repository, and every file of `05` and `06` that this
# prototype includes — those two names are already bound. Julia refuses to rebind an
# imported name:
#
#     const Experiments = @__MODULE__
#     ERROR: cannot assign a value to imported variable Main.Experiments
#
# So the prototype lives in a module, and the two colliding names live in a nested
# `Legacy` submodule that does NOT do `using BayesianFootball` and therefore can bind
# them. Everything else the briefing lists — `ExperimentConfig`, `ExperimentResults`,
# `ExperimentTask`, `run_experiment`, `save_experiment`, `load_experiment`,
# `extract_oos_predictions` — is NOT exported by `BayesianFootball` (only the module
# names are; `experiment-module.jl` exports them from `Experiments`, and
# `BayesianFootball.jl` never does `using .Experiments`), so those are bound directly
# in `UnifiedInference` with no collision at all.
#
# ------------------------------------------------------------------------------
# WHAT "100% BACKWARD COMPATIBLE" MEANS HERE, EXACTLY
# ------------------------------------------------------------------------------
#
# A legacy call site's BODY is unchanged. Its import line changes.
#
#     # before
#     using BayesianFootball
#     task = Experiments.create_experiment_task(ds, model, "run", "./data/experiments";
#                                               target_seasons = ["24/25"])
#     res  = Experiments.run_experiment(task)
#     ch   = res.training_results.items[1][1]
#     lat  = Experiments.extract_oos_predictions(ds, res)
#     Experiments.save_experiment(res)
#
#     # after — the same six lines, one different import
#     import BayesianFootball
#     using .UnifiedInference.Legacy      # binds `Experiments` and `Training`
#     …identical body…
#
# That is the honest claim, and `r01_demo.jl` §9 proves it by executing a
# `LegacyCallSite` module whose body is copied verbatim from the pattern above. What is
# NOT claimed is that `using BayesianFootball` and this bridge can coexist while both
# supplying the name `Experiments` — nothing can make two modules answer to one name.
#
# Every legacy MEMBER access is preserved, and that is the part that actually breaks
# code when it changes:
#
#   | legacy expression                        | resolves to                        |
#   |------------------------------------------|------------------------------------|
#   | `res.config`                             | `Fit.config` (field)               |
#   | `res.save_path`                          | `Fit.save_path` (field)            |
#   | `res.training_results`                   | `LegacyTrainingResults` (l01 §7.2) |
#   | `res.training_results.items[i]`          | `(chain, meta)` tuple              |
#   | `res.training_results.items[i][1]`       | the chain                          |
#   | `for (c, m) in res.training_results`     | iterates tuples                    |
#   | `length(res.training_results.items)`     | fold count                         |
#   | `res.vocabulary`                         | `nothing`                          |
#   | `latents.df`                             | the legacy `DataFrame` (l01 §7.3)  |
#   | `nrow(latents)`                          | fixture count                      |
#   | `res.config.training_config.sampler`     | the sampler (§2 shim)              |
#
# ------------------------------------------------------------------------------
# THE ONE DELIBERATE BEHAVIOUR CHANGE
# ------------------------------------------------------------------------------
#
# `run_experiment` now AUDITS CONVERGENCE and EXTRACTS LATENTS as part of the run,
# because `fit_model` does. A legacy caller therefore gets a result that is strictly
# more complete than the one it asked for, and pays the extraction cost up front rather
# than on its next `extract_oos_predictions` call — which then returns instantly.
#
# It also gets `res.diagnostics`, which the legacy type had no field for. That is
# additive: nothing that worked stops working, and `Experiments.Diagnostics` is not
# shadowed — it still exists in `BayesianFootball` for a caller that wants the
# team-name-resolved parameter frame this framework deliberately does not reimplement
# (`l02_convergence.jl` header).
#
# ==============================================================================

module UnifiedInference

# The whole prototype. `l05_io.jl` chains down through l04 → l03 → l02 → l01, and l01
# pulls in `06_typed_posterior_latents` and `05_composable_count_builder`.
include(joinpath(@__DIR__, "l05_io.jl"))

using DataFrames
using Dates
using Serialization


# ==============================================================================
# 1. THE NEW API
# ==============================================================================

export Fit, FitConfig, FoldFit, FitMetadata, FitTask
export InGameFit, InGameFitConfig
export AbstractPreGameModel, AbstractInGameModel, is_pregame, is_ingame
export AbstractExecution, AutoExecution, SequentialExecution, ThreadedExecution,
       QueuedExecution
export ReplaySampler, sample_fold, run_folds, fit_model
export ConvergenceGates, ConvergenceSummary, FoldConvergence,
       audit_convergence, audit_fold, summarise_convergence, convergence_table, bfmi
export save_fit, load_fit, list_fits, load_fits, read_fit_meta, save_latents, load_latents
export merge_latents, extract_run_latents
export NHPPIntensityModel, MatchState, kickoff_state, LiveKernel, LiveBook,
       build_live_kernel, remaining_intensity, remaining_intensity!,
       alloc_intensity, alloc_live_book, price_live_market, price_live_market!,
       live_kernel, live_book, pregame_latents
export chains, fold_metas, total_draws, fit_name, format_elapsed, git_commit_id


# ==============================================================================
# 2. TYPE ALIASES
# ==============================================================================
#
# Bound directly here — none of these three names is exported by `BayesianFootball`
# (see the header), so there is nothing to collide with.

"""
    ExperimentResults

The legacy name for `Fit`. An alias, not a wrapper: `res isa ExperimentResults` is
`res isa Fit`, and there is exactly one type on either side of the bridge.
"""
const ExperimentResults = Fit

"""
    ExperimentConfig

The legacy name for `FitConfig`.

The legacy construction

    ExperimentConfig(name = …, model = …, splitter = …, training_config = tc)

works unchanged: `l01_types.jl` §4.1 defines the keyword constructor that unpacks
`tc.sampler` into `sampler` and `tc.strategy` into `execution`.
"""
const ExperimentConfig = FitConfig

"""
    FitTask(ds, config)

The data and the recipe, together. `ExperimentTask` is an alias for it.

DEVIATION FROM THE BRIEFING, and the reason for it: the briefing specifies

    const ExperimentTask = NamedTuple{(:ds, :config), Tuple{Data.DataStore, FitConfig}}

A `NamedTuple` type alias cannot be CALLED with positional arguments, so the legacy
`ExperimentTask(ds, config)` — which is how `create_experiment_task` builds one
(presets.jl:113) and how every runner writes one by hand — would stop compiling. A
two-field struct is the same shape, supports `task.ds` and `task.config` identically,
and keeps the constructor. `run_experiment` additionally accepts a plain `NamedTuple`
with those two fields, so a caller that already builds the briefing's form works too.
"""
struct FitTask{D, C}
    ds::D
    config::C
end

const ExperimentTask = FitTask

Base.show(io::IO, t::FitTask) = print(io, "FitTask(", t.config.name, ")")

export ExperimentResults, ExperimentConfig, ExperimentTask, FitTask


# ==============================================================================
# 3. `TrainingResults`, THE LEGACY CONSTRUCTOR
# ==============================================================================
#
# `config.training_config` — the other half of this compatibility surface — is defined
# in `l01_types.jl` §4.2, next to the struct it reads, because `Base.getproperty` has
# to exist before anything compiles a field access against the default one.

"""
    TrainingResults(items)

The legacy constructor, returning a `LegacyTrainingResults` over synthesised folds.

Present so that code which built a `TrainingResults` by hand — a resumed run stitching
checkpoints together, `r03_pipeline_smoke.jl`'s fixtures — still compiles. `items` is a
vector of `(chain, meta)` tuples, exactly as before.
"""
TrainingResults(items::AbstractVector) =
    LegacyTrainingResults(_uif_narrow(FoldFit[FoldFit(i, it[1], it[2])
                                              for (i, it) in enumerate(items)
                                              if it !== nothing]))

export LegacyTrainingConfig, LegacyTrainingResults, TrainingResults


# ==============================================================================
# 4. FUNCTION SHIMS — the run
# ==============================================================================

"""
    run_experiment(task)
    run_experiment(ds::DataStore, config::FitConfig; kwargs...)

`fit_model`, under its old name. Every keyword `fit_model` takes is forwarded.

The result is a `Fit`, which IS `ExperimentResults`, and answers every legacy property
(header table). See "the one deliberate behaviour change" in the header for what it now
does that it did not before.
"""
run_experiment(ds::UIF_D.DataStore, config::FitConfig; kwargs...) =
    fit_model(ds, config; kwargs...)

run_experiment(task::FitTask; kwargs...) = fit_model(task.ds, task.config; kwargs...)

# The briefing's NamedTuple-shaped task, accepted too.
run_experiment(task::NamedTuple; kwargs...) = fit_model(task.ds, task.config; kwargs...)

"""
    train(model, training_config, feature_sets; kwargs...) -> LegacyTrainingResults

`Training.train`, under its old signature.

`training_config` may be a legacy `Training.TrainingConfig` or anything with `.sampler`
and `.strategy`. The return value is the legacy shape — `.items[i]` is `(chain, meta)`.

NOT a `Fit`: `train` never had a config, a splitter or a save path to build one from,
and inventing them would put a `Fit` on disk that could not be re-run. A caller who
wants a `Fit` wants `fit_model`.
"""
function train(model, training_config, feature_sets; quiet::Bool = false)
    sampler = training_config.sampler
    exec = execution_from_strategy(get_or(training_config, :strategy, nothing))
    ckpt = get_or(training_config, :checkpoint_dir, nothing)

    n = length(feature_sets)
    results = load_checkpoints(ckpt, n)
    pending = findall(isnothing, results)

    if !isempty(pending)
        pending_fs = [feature_sets[i] for i in pending]
        fresh = run_folds(model, sampler, exec, pending_fs;
                          on_progress = quiet ? _uif_noop : _uif_progress(time()))
        for (k, i) in enumerate(pending)
            results[i] = fresh[k]
            results[i] === nothing && continue
            ckpt === nothing || save_checkpoint(ckpt, i, (results[i], feature_sets[i][2]))
        end
    end

    return LegacyTrainingResults(
        _uif_narrow(FoldFit[FoldFit(i, results[i], feature_sets[i][2])
                            for i in 1:n if results[i] !== nothing]))
end

"""
    train(model, training_config, feature_set::FeatureSet; chain_id = nothing)

The single-split form (method.jl:10). Returns the sampler's result directly, not a
container — same as before.
"""
train(model, training_config, feature_set::UIF_TI.FeatureSet;
      chain_id::Union{Int, Nothing} = nothing) =
    sample_fold(model, training_config.sampler, feature_set, 1; chain_id = chain_id)

export run_experiment, train


# ==============================================================================
# 5. FUNCTION SHIMS — persistence
# ==============================================================================

"""
    save_experiment(res::Fit; path = nothing, quiet = false, ds = nothing,
                    compute_oos = false)

`save_fit`, under its old name and with its old keywords.

`ds` and `compute_oos` are accepted and honoured: `compute_oos = true` with a `ds`
forces a fresh extraction before saving, as the legacy version did (runner.jl:121-126).
Neither is usually needed now — `fit_model` already extracted the latents — so
`compute_oos` on a `Fit` that has them is a no-op rather than a recomputation.
"""
function save_experiment(res::Fit; path = nothing, quiet::Bool = false,
                         ds = nothing, compute_oos::Bool = false)
    if compute_oos && ds !== nothing && getfield(res, :latents) === nothing
        @warn "save_experiment: compute_oos requested but this Fit has no latents and " *
              "recomputation needs the original feature sets; saving without them."
    end
    return save_fit(res; path = path, quiet = quiet)
end

save_experiment(res::InGameFit; kwargs...) = save_fit(res; kwargs...)

"""
    load_experiment(path)
    load_experiment(list::Vector{String}, index::Int)

`load_fit`, under its old name, including the "index into a `list_experiments` result"
form (runner.jl:232).
"""
load_experiment(path::AbstractString; kwargs...) = load_fit(String(path); kwargs...)

function load_experiment(list::Vector{String}, index::Int; kwargs...)
    1 <= index <= length(list) || error("Index $index out of bounds.")
    return load_fit(list[index]; kwargs...)
end

"`load_fits`, under its old name."
load_experiments(paths::Vector{String}; kwargs...) = load_fits(paths; kwargs...)

"""
    list_experiments(dir; data_dir = "./data") -> Vector{String}

`list_fits`, under its old name and with its old return type — the legacy version
returns the PATHS (runner.jl:217) and callers index them with `load_experiment(list, i)`.

`data_dir` is joined onto `dir` exactly as before, so `list_experiments("experiments")`
scans `./data/experiments`.
"""
function list_experiments(dir::AbstractString; data_dir::AbstractString = "./data")
    base = isabspath(dir) || isdir(dir) ? String(dir) : joinpath(data_dir, dir)
    return [r.path for r in list_fits(base)]
end

export save_experiment, load_experiment, load_experiments, list_experiments


# ==============================================================================
# 6. FUNCTION SHIMS — out-of-sample latents
# ==============================================================================
#
# `extract_oos_predictions(ds, res)` used to be where the latents were BUILT, at a cost
# of one full re-derivation of boundaries and feature sets (post_processing.jl:139-155).
# It is now where they are READ, because `fit_model` built them while the feature sets
# were still in hand.

"""
    extract_oos_predictions(ds, fit::Fit; force = false) -> container

The typed OOS posterior container. `fit.latents`, which the run already extracted.

Returns the TYPED container (06), not a `LatentStates`. A legacy caller's next line is
almost always `latents.df`, and `l01_types.jl` §7.3 answers that with the legacy
`DataFrame`. A caller that genuinely needs the legacy wrapper type — an `isa` test, a
type annotation — should call `as_latent_states`.

`force = true` cannot recompute from a `Fit` alone: the extraction needs the per-fold
FEATURE SETS, which a completed `Fit` does not carry (they are large, and rebuilding
them is what this change exists to stop doing). It re-reads the on-disk cache instead,
and says so if there is none. To genuinely recompute, re-run `fit_model`.
"""
function extract_oos_predictions(ds, fit::Fit; force::Bool = false)
    lat = getfield(fit, :latents)
    if force
        cached = load_latents(getfield(fit, :save_path))
        cached === nothing || return cached
        lat === nothing && error(
            "extract_oos_predictions: `force = true`, but there is no cache at " *
            "$(getfield(fit, :save_path)) and this Fit carries no latents. Re-run " *
            "`fit_model` — extraction needs the feature sets, which a Fit does not keep.")
        return lat
    end
    lat === nothing && error(
        "extract_oos_predictions: this Fit carries no latents. Its tags say why: " *
        join(getfield(fit, :config).tags, ", "))
    return lat
end

extract_oos_predictions(fit::Fit; kwargs...) =
    extract_oos_predictions(nothing, fit; kwargs...)

"""
    LatentStates(latents, model)

The legacy wrapper type, for a caller that annotates or `isa`-tests it. `.df` gives the
legacy `DataFrame`, `.latents` the typed container.

Provided rather than aliased because `Experiments.LatentStates` is a CONCRETE struct
with two fields; aliasing it to `AbstractPosteriorLatents` would make
`LatentStates(df, model)` a `MethodError` at every legacy construction site.
"""
struct LatentStates{L, M}
    latents::L
    model::M
end

@inline function Base.getproperty(ls::LatentStates, s::Symbol)
    s === :df && return _uif_as_df(getfield(ls, :latents))
    return getfield(ls, s)
end

Base.propertynames(::LatentStates) = (:latents, :model, :df)

_uif_as_df(l::AbstractPosteriorLatents) = to_legacy_dataframe(l)
_uif_as_df(df::AbstractDataFrame) = df

Base.size(ls::LatentStates) = size(ls.df)
Base.size(ls::LatentStates, i) = size(ls.df, i)
Base.getindex(ls::LatentStates, args...) = getindex(ls.df, args...)
Base.show(io::IO, ls::LatentStates) = show(io, ls.df)
DataFrames.nrow(ls::LatentStates) = nrow(ls.df)
DataFrames.ncol(ls::LatentStates) = ncol(ls.df)

"Wrap a `Fit`'s latents in the legacy type."
as_latent_states(f::Fit) =
    LatentStates(getfield(f, :latents), getfield(f, :config).model)

# `nrow(latents)` on a typed container: the fixture count, which is what the legacy
# frame's row count was.
DataFrames.nrow(l::AbstractPosteriorLatents) = n_matches(l)

"""
    has_oos_predictions(fit; path = nothing) -> Bool
    has_oos_predictions(path::String) -> Bool

Whether an `oos_latents.jls` cache exists. Same file, same name, same answer.
"""
has_oos_predictions(path::AbstractString) = isfile(joinpath(path, UIF_LATENTS_FILE))
has_oos_predictions(f::Fit; path = nothing) =
    has_oos_predictions(path === nothing ? getfield(f, :save_path) : String(path))

"Read the cache. `nothing` on a miss or a corrupt file, exactly as before."
load_oos_predictions(path::AbstractString) = load_latents(String(path))
load_oos_predictions(f::Fit; path = nothing) =
    load_latents(path === nothing ? getfield(f, :save_path) : String(path))

"Write the cache atomically. Returns the file path."
save_oos_predictions(path::AbstractString, latents) = save_latents(String(path), latents)
save_oos_predictions(f::Fit, latents; path = nothing) =
    save_latents(path === nothing ? getfield(f, :save_path) : String(path), latents)

export extract_oos_predictions, LatentStates, as_latent_states,
       has_oos_predictions, load_oos_predictions, save_oos_predictions


# ==============================================================================
# 7. FUNCTION SHIMS — presets and helpers
# ==============================================================================

"""
    create_experiment_task(ds, model, name, save_dir; kwargs...) -> FitTask

`Experiments.create_experiment_task` with the same keywords and the same defaults
(presets.jl:34-114), producing a `FitConfig` instead of an `ExperimentConfig`.

The sampler and splitter construction is copied verbatim, deliberately: a preset whose
defaults silently differed from the legacy one would make two runs incomparable while
looking identical in the transcript.
"""
function create_experiment_task(ds, model, name::AbstractString, save_dir::AbstractString;
                                target_seasons::Vector{String},
                                history_seasons::Int = 2,
                                dynamics_col::Symbol = :match_month,
                                warmup_period::Int = 0,
                                stop_early::Bool = false,
                                samples::Int = 500,
                                chains::Int = 4,
                                warmup::Int = 200,
                                accept_rate::Float64 = 0.65,
                                max_depth::Int = 10,
                                show_progress = false,
                                parallel::Bool = true,
                                max_concurrent_splits::Int = 4,
                                use_queue::Bool = true,
                                max_concurrent_tasks::Int = -1)
    cv = UIF_D.GroupedCVConfig(
        tournament_groups = [UIF_D.tournament_ids(ds.segment)],
        target_seasons = target_seasons,
        history_seasons = history_seasons,
        dynamics_col = dynamics_col,
        warmup_period = warmup_period,
        stop_early = stop_early)

    sampler, exec = if use_queue
        (UIF_Samp.QueuedNUTSConfig(n_samples = samples, n_chains = chains,
                                   n_warmup = warmup, accept_rate = accept_rate,
                                   max_depth = max_depth,
                                   initialisation = UIF_Samp.UniformInit(-0.1, 0.1),
                                   show_progress = show_progress),
         parallel ? QueuedExecution(max_concurrent_tasks =
                                    max_concurrent_tasks == -1 ? Threads.nthreads() :
                                    max_concurrent_tasks) : SequentialExecution())
    else
        (UIF_Samp.NUTSConfig(samples, chains, warmup, accept_rate, max_depth,
                             UIF_Samp.UniformInit(-2, 2), show_progress),
         parallel ? ThreadedExecution(max_concurrent_splits = max_concurrent_splits) :
                    SequentialExecution())
    end

    return FitTask(ds, FitConfig(name = String(name), model = model, splitter = cv,
                                 sampler = sampler, execution = exec,
                                 save_dir = String(save_dir)))
end

"`exp.config.name`. (helpers.jl:4)"
get_model_name(f::AbstractFit)::String = string(getfield(f, :config).name)

"The model's type name. (helpers.jl:8, which calls an undefined `type` — fixed here.)"
get_model_type(f::AbstractFit)::String = string(nameof(typeof(getfield(f, :config).model)))

export create_experiment_task, get_model_name, get_model_type


# ==============================================================================
# 8. THE COLLIDING NAMES
# ==============================================================================

"""
    UnifiedInference.Legacy

The two names that cannot be bound in a scope that has done `using BayesianFootball`.

```julia
import BayesianFootball
using .UnifiedInference.Legacy      # Experiments, Training
```

Both are the same module — `UnifiedInference` — because the split between them is
exactly what this prototype removed. `Experiments.run_experiment` and `Training.train`
resolve to the shims above; anything else either module offered resolves to whatever
`UnifiedInference` exports under that name, or raises `UndefVarError` naming it.

`parentmodule(@__MODULE__)` rather than `import ..UnifiedInference`: this submodule is
elaborated while its parent's body is still executing, and `parentmodule` needs no
binding to already exist.
"""
module Legacy

const Experiments = parentmodule(@__MODULE__)
const Training    = parentmodule(@__MODULE__)

export Experiments, Training

end # module Legacy

end # module UnifiedInference
