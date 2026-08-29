# src/training/inference/types.jl
#
# The containers of the unified inference lifecycle:
#
#     split → sample → AUDIT → EXTRACT → Fit
#
# `Fit` replaces the four-hop `ExperimentResults.training_results.items[i][1]` read with
# `fit[i].chain`, and — the substantive change — makes the convergence audit a FIELD
# rather than an optional function call. A `Fit` that was never audited cannot be
# constructed, so any fit, in memory or six months old on disk, answers
# `fit.diagnostics.passed` without a `DataStore`, a splitter, or a re-run.
#
# `src/experiments/` and the legacy `Training.train` path are untouched; see `compat.jl`
# for the bidirectional bridge between the two worlds.

# ==============================================================================
# 1. THE MODEL HIERARCHY
# ==============================================================================

"""
    AbstractPreGameModel

Alias for the repository's root model type. Every engine in `src/models/pregame/` is one.

An alias rather than a new abstract type: there is no way to retroactively re-parent the
40-odd existing engines, and a fresh root would make `FitConfig{M<:AbstractPreGameModel}`
reject `DynamicXGModel`. Use [`is_pregame`](@ref) for the predicate that discriminates.
"""
const AbstractPreGameModel = TypesInterfaces.AbstractFootballModel

"""
    AbstractInGameModel <: AbstractFootballModel

A model that prices a match ALREADY IN PROGRESS, given a pre-game posterior as its
baseline.

The distinction is about what `λ` means, not about when the code runs:

  * a pre-game model's `λ` is a whole-match expected count, and pricing integrates it
    over `[0, 90]` implicitly;
  * an in-game model's `λ(t)` is an INSTANTANEOUS rate, and pricing must integrate it
    over `[t_now, Tend]` explicitly, conditioned on the score and man-count at `t_now`.

Feeding one into the other's pricer produces a plausible, systematically wrong number.
"""
abstract type AbstractInGameModel <: TypesInterfaces.AbstractFootballModel end

"""
    is_ingame(model) -> Bool
    is_pregame(model) -> Bool

Which side of the split a model is on. `is_pregame` is the complement of `is_ingame`
because `AbstractPreGameModel` is the shared root, so `model isa AbstractPreGameModel`
is true for both and is not the test you want.
"""
is_ingame(model) = model isa AbstractInGameModel
is_pregame(model) = !is_ingame(model)

"`obj.name` if it exists, `default` otherwise. Used only on legacy/reconstructed objects."
get_or(obj, name::Symbol, default) = hasproperty(obj, name) ? getproperty(obj, name) : default


# ==============================================================================
# 2. FOLD FIT — one split's outcome
# ==============================================================================

"""
    FoldFit(fold, chain, meta)

A single fold's inference outcome: what the sampler returned, and which slice of the
calendar it was fitted on.

| field   | is                                                                  |
|---------|---------------------------------------------------------------------|
| `fold`  | 1-based split index, in `Data.create_id_boundaries` order            |
| `chain` | the sampler's return value — `MCMCChains.Chains` for every NUTS path |
| `meta`  | `SplitMetaData` / `GroupedSplitMetaData` for that split              |

This is the legacy `Tuple{Chains, SplitMetaData}` with the two things a tuple cannot
carry: a name for each slot, and the fold index. `items[i][1]` becomes `folds[i].chain`,
and a transposed destructure becomes a `MethodError` rather than a `Chains` being
treated as metadata.

`chain` is deliberately UNCONSTRAINED rather than `C<:Chains`: `run_sampler(::MAPConfig)`
returns a point estimate, and the tighter bound would make this framework unable to hold
a run that `Training.train` holds today. Everything that needs a real chain — the audit,
the extractor — dispatches on `::Chains` at its own entry point instead.
"""
struct FoldFit{C, M<:Data.AbstractSplitMetaData}
    fold::Int
    chain::C
    meta::M

    function FoldFit(fold::Integer, chain::C, meta::M) where {C, M<:Data.AbstractSplitMetaData}
        fold > 0 || error("FoldFit: fold index must be positive, got $fold.")
        return new{C, M}(Int(fold), chain, meta)
    end
end

"The legacy `(chain, meta)` tuple this fold would have been. For the compatibility bridge."
as_legacy_tuple(f::FoldFit) = (f.chain, f.meta)

"Draws in this fold's chain, flattened over chains. `0` for a non-`Chains` result."
fold_n_draws(f::FoldFit{<:Chains}) = size(f.chain, 1) * size(f.chain, 3)
fold_n_draws(::FoldFit) = 0

function Base.show(io::IO, f::FoldFit)
    print(io, "FoldFit(", f.fold, ", ", nameof(typeof(f.chain)))
    f.chain isa Chains && print(io, " ", size(f.chain, 1), "×", size(f.chain, 2),
                                "×", size(f.chain, 3))
    print(io, ", ", f.meta, ")")
end


# ==============================================================================
# 3. EXECUTION STRATEGY
# ==============================================================================

"""
    AbstractExecution

How the folds are dispatched onto threads. Orthogonal to WHAT is sampled, which is the
sampler's business. This is the one thing `Training.TrainingConfig` carried that a flat
`FitConfig` would otherwise drop.
"""
abstract type AbstractExecution end

"""
    AutoExecution(; max_concurrent_splits = 0, max_concurrent_tasks = 0)

Read the strategy off the sampler at RUN time: `QueuedNUTSConfig` → `QueuedExecution`,
anything else → `ThreadedExecution` when there is more than one thread,
`SequentialExecution` otherwise.

This is exactly the test `Training.train_independent` performs
(`strategies/independent.jl`), moved from inside the training loop to the point where it
is a decision rather than a branch.

Both caps default to `0`, meaning "decide from `Threads.nthreads()` when the run starts",
so a config built on a laptop and run on a 32-core box uses the 32 cores. A non-zero cap
is honoured verbatim.
"""
Base.@kwdef struct AutoExecution <: AbstractExecution
    max_concurrent_splits::Int = 0
    max_concurrent_tasks::Int = 0
end

"One fold at a time, in order. Reproducible, and the only sane mode under a debugger."
struct SequentialExecution <: AbstractExecution end

"""
    ThreadedExecution(; max_concurrent_splits = nthreads() ÷ 2)

One task per fold, capped by a semaphore. The legacy `Independent(parallel = true)`.
"""
Base.@kwdef struct ThreadedExecution <: AbstractExecution
    max_concurrent_splits::Int = max(1, Threads.nthreads() ÷ 2)
end

"""
    QueuedExecution(; max_concurrent_tasks = nthreads())

The `K splits × N chains` flattening from `Training._train_queued`: every (split, chain)
pair is one queue entry, so a fold whose chains finish early does not leave cores idle
waiting for its slowest sibling. Requires a sampler that accepts a `chain_id`, i.e.
`QueuedNUTSConfig`.
"""
Base.@kwdef struct QueuedExecution <: AbstractExecution
    max_concurrent_tasks::Int = Threads.nthreads()
end

"""
    resolve_execution(execution, sampler) -> AbstractExecution

`AutoExecution` resolved against a concrete sampler; everything else passed through.
Kept out of the `FitConfig` constructor so that a config built on a 1-thread machine and
run on a 32-thread one uses the 32 threads.
"""
resolve_execution(e::AbstractExecution, sampler) = e

function resolve_execution(a::AutoExecution, sampler)
    if nameof(typeof(sampler)) === :QueuedNUTSConfig
        n = a.max_concurrent_tasks > 0 ? a.max_concurrent_tasks : Threads.nthreads()
        return QueuedExecution(max_concurrent_tasks = n)
    elseif Threads.nthreads() > 1
        n = a.max_concurrent_splits > 0 ? a.max_concurrent_splits :
            max(1, Threads.nthreads() ÷ 2)
        return ThreadedExecution(max_concurrent_splits = n)
    else
        return SequentialExecution()
    end
end

"""
    execution_from_strategy(strategy) -> AbstractExecution

Map a legacy `Training.Independent` onto an `AbstractExecution`, carrying its two
concurrency caps across.

`parallel = true` maps to `AutoExecution`, NOT to `QueuedExecution`, because the legacy
loop does not decide queued-vs-threaded from the strategy either: it decides from the
SAMPLER (`typeof(config.sampler).name.name == :QueuedNUTSConfig`,
`strategies/independent.jl`). Reproducing that means deferring, which is what
`AutoExecution` is.
"""
function execution_from_strategy(strategy)
    nameof(typeof(strategy)) === :Independent || return AutoExecution()
    get_or(strategy, :parallel, false) || return SequentialExecution()
    return AutoExecution(
        max_concurrent_splits = get_or(strategy, :max_concurrent_splits, 0),
        max_concurrent_tasks  = get_or(strategy, :max_concurrent_tasks, 0),
    )
end


# ==============================================================================
# 4. CONVERGENCE TELEMETRY CONTAINERS
# ==============================================================================
#
# The logic that fills these lives in `convergence.jl`; they are declared here because
# `Fit` has a `ConvergenceSummary` as a concrete field.

"""
    ConvergenceThresholds(; max_rhat = 1.01, min_ess = 400.0,
                            max_divergence_rate = 0.001, min_bfmi = 0.30,
                            max_treedepth_rate = 0.05)

The thresholds `ConvergenceSummary.passed` is a conjunction of. Every default is a
modelling decision, and none is arbitrary:

  * `max_rhat = 1.01` — the 2019 Vehtari et al. revision of the older 1.1, which was
    shown to pass chains that had visibly not mixed.
  * `min_ess = 400` — 100 effective draws per chain at 4 chains; below this the Monte
    Carlo standard error on a posterior mean is a material fraction of the posterior SD.
  * `max_divergence_rate = 0.001` — 0.10% of transitions. The raw count is reported next
    to it so a reader can apply their own.
  * `min_bfmi = 0.30` — Betancourt's threshold for "reparameterise this model".
  * `max_treedepth_rate = 0.05` — a PERFORMANCE gate, and the one gate whose failure
    does not invalidate the posterior: saturating trajectories were truncated, so each
    draw cost the maximum and bought less than it should have.
"""
Base.@kwdef struct ConvergenceThresholds
    max_rhat::Float64 = 1.01
    min_ess::Float64 = 400.0
    max_divergence_rate::Float64 = 0.001
    min_bfmi::Float64 = 0.30
    max_treedepth_rate::Float64 = 0.05
end

"""
    FoldConvergence

One fold's six metrics, plus the parameter name that produced each worst case.

Carrying `worst_rhat_param` and friends is what makes the summary actionable: "max R-hat
1.34" sends a reader back to the chains; "max R-hat 1.34 at `dyn.σ_a`" names a
non-centred scale parameter and, usually, the fix.

`applicable = false` marks a point-estimate fold (MAP/MLE, or any chain with one draw),
whose metrics are all `NaN`/`0` and which is excluded from every run-level reduction and
COUNTED, so a summary can say "3 of 12 folds were point estimates" rather than quietly
averaging over nine.
"""
struct FoldConvergence
    fold::Int
    applicable::Bool
    n_params::Int
    n_draws::Int
    n_chains::Int
    max_rhat::Float64
    worst_rhat_param::Symbol
    min_ess_bulk::Float64
    worst_ess_bulk_param::Symbol
    min_ess_tail::Float64
    worst_ess_tail_param::Symbol
    n_divergent::Int
    n_transitions::Int
    divergence_rate::Float64
    max_tree_depth::Int
    n_depth_capped::Int
    treedepth_rate::Float64
    min_bfmi::Float64
end

"""
    ConvergenceSummary

The reduction over folds, and the verdict.

`passed` is a CONJUNCTION over the thresholds, evaluated only on applicable folds.
`failures` gives one human-readable line per fallen gate, with the number that did it;
`failed_gates` gives the same set as bare gate NAMES, so a caller can write
`"BFMI" in summary.failed_gates` instead of matching against prose that may be reworded.

ABSTENTION. A gate whose metric is `NaN` for every fold — no divergence record because
the sampler does not emit one, no energy record because it is not Hamiltonian — is
neither passed nor failed: it is listed in `abstained`. Treating an unmeasured gate as
passed would let a sampler earn a clean bill of health by recording nothing.
"""
struct ConvergenceSummary
    folds::Vector{FoldConvergence}
    thresholds::ConvergenceThresholds
    n_folds::Int
    n_applicable::Int
    max_rhat::Float64
    worst_rhat_fold::Int
    min_ess_bulk::Float64
    worst_ess_bulk_fold::Int
    min_ess_tail::Float64
    worst_ess_tail_fold::Int
    n_divergent::Int
    n_transitions::Int
    divergence_rate::Float64
    max_tree_depth::Int
    n_depth_capped::Int
    treedepth_rate::Float64
    min_bfmi::Float64
    worst_bfmi_fold::Int
    passed::Bool
    failures::Vector{String}
    failed_gates::Vector{String}
    abstained::Vector{String}
end


# ==============================================================================
# 5. FIT CONFIG — the recipe
# ==============================================================================

"""
    FitConfig(; name, model, splitter, sampler, execution = AutoExecution(),
                tags = String[], description = "", save_dir = "./data/fits")
    FitConfig(; name, model, splitter, training_config, kwargs...)

The immutable specification for one inference run. FLAT, where `ExperimentConfig` was
nested: reading the sampler back out of a saved legacy run took
`config.training_config.sampler`, and here it is `config.sampler`.

| field         | meaning                                                         |
|---------------|-----------------------------------------------------------------|
| `name`        | run name; the save directory is `save_dir/name_<timestamp>`      |
| `model`       | the engine — any `AbstractFootballModel`                         |
| `splitter`    | `CVConfig` / `GroupedCVConfig` / `StaticSplit` / …               |
| `sampler`     | `NUTSConfig`, `QueuedNUTSConfig`, `MAPConfig`, `ReplaySampler`   |
| `execution`   | fold dispatch strategy; `AutoExecution()` reads it off `sampler` |
| `tags`        | free-form labels; `fit_model` appends `time:<elapsed>`           |
| `description` | free-form prose                                                  |
| `save_dir`    | root directory for `save_fit`                                    |

Checkpointing is NOT a field: it moved to `fit_model`'s keywords, where it is a property
of the run rather than of the recipe — the same recipe re-run without checkpoints is the
same recipe. `legacy_checkpointing(tc)` recovers the two fields off a legacy
`TrainingConfig` for forwarding.

The legacy-shaped construction

    FitConfig(name = "x", model = m, splitter = s,
              training_config = Training.TrainingConfig(sampler, Independent(), nothing, false))

works: `training_config.sampler` becomes `sampler` and `training_config.strategy` is
mapped by [`execution_from_strategy`](@ref). Both shapes share ONE method, and that is
forced rather than chosen — keyword methods dispatch on their positional signature, so a
second zero-positional-argument constructor would silently REPLACE the first rather than
overload it.
"""
struct FitConfig{M<:TypesInterfaces.AbstractFootballModel,
                 S<:Data.AbstractSplitter,
                 Sam,
                 E<:AbstractExecution}
    name::String
    model::M
    splitter::S
    sampler::Sam
    execution::E
    tags::Vector{String}
    description::String
    save_dir::String
end

function FitConfig(; name::AbstractString,
                     model,
                     splitter,
                     sampler = nothing,
                     training_config = nothing,
                     execution::AbstractExecution = AutoExecution(),
                     tags::Vector{String} = String[],
                     description::AbstractString = "",
                     save_dir::AbstractString = "./data/fits")
    sam = sampler
    exe = execution
    if sam === nothing
        training_config === nothing && error(
            "FitConfig: pass either `sampler = …` or the legacy `training_config = …`.")
        hasproperty(training_config, :sampler) || error(
            "FitConfig: `training_config` must have a `.sampler` field; got a " *
            "$(typeof(training_config)). Pass `sampler = …` directly instead.")
        sam = training_config.sampler
        exe = execution_from_strategy(get_or(training_config, :strategy, nothing))
    end
    return FitConfig(String(name), model, splitter, sam, exe,
                     tags, String(description), String(save_dir))
end

"""
    LegacyTrainingConfig

What `config.training_config` returns: the four fields the legacy `TrainingConfig` had,
computed from the flat `FitConfig`. A VIEW — storing one would put the nesting back.

`checkpoint_dir` is always `nothing` and `cleanup_checkpoints` always `false`, because
checkpointing moved from the recipe to `fit_model`'s keywords.
"""
struct LegacyTrainingConfig{S, E}
    sampler::S
    strategy::E
    checkpoint_dir::Nothing
    cleanup_checkpoints::Bool
end

LegacyTrainingConfig(c::FitConfig) =
    LegacyTrainingConfig(getfield(c, :sampler),
                         resolve_execution(getfield(c, :execution), getfield(c, :sampler)),
                         nothing, false)

Base.show(io::IO, t::LegacyTrainingConfig) =
    print(io, "TrainingConfig(strategy=", nameof(typeof(t.strategy)),
          ", checkpointing=false)")

# `save_experiment` reads `res.config.training_config.sampler` (runner.jl). `FitConfig`
# has no such field, so the read is SYNTHESISED — and defined here, next to the struct,
# because `Base.getproperty` must exist before anything compiles a field access against
# the default one.
@inline function Base.getproperty(c::FitConfig, s::Symbol)
    s === :training_config && return LegacyTrainingConfig(c)
    return getfield(c, s)
end

Base.propertynames(::FitConfig) = (fieldnames(FitConfig)..., :training_config)

"""
    legacy_checkpointing(training_config) -> (dir, cleanup)

The two checkpoint fields a legacy `TrainingConfig` carried, for a caller that wants to
forward them to `fit_model(...; checkpoint_dir = …, cleanup_checkpoints = …)`.
"""
legacy_checkpointing(tc) = (get_or(tc, :checkpoint_dir, nothing),
                            get_or(tc, :cleanup_checkpoints, false))

function Base.show(io::IO, c::FitConfig)
    print(io, "FitConfig(", c.name, ", ", nameof(typeof(c.model)),
          ", ", nameof(typeof(c.splitter)), ", ", nameof(typeof(c.sampler)), ")")
end

function Base.show(io::IO, ::MIME"text/plain", c::FitConfig)
    println(io, "FitConfig")
    println(io, "  name        : ", c.name)
    println(io, "  model       : ", c.model)
    println(io, "  splitter    : ", c.splitter)
    println(io, "  sampler     : ", c.sampler)
    println(io, "  execution   : ", c.execution)
    println(io, "  tags        : ", isempty(c.tags) ? "—" : join(c.tags, ", "))
    isempty(c.description) || println(io, "  description : ", c.description)
    print(io,   "  save_dir    : ", c.save_dir)
end


# ==============================================================================
# 6. FIT METADATA — provenance
# ==============================================================================

"""
    FitMetadata(timestamp, elapsed_seconds, julia_version, n_threads, git_commit)

What was true about the machine when the run happened.

`git_commit` is the short SHA of the working tree at run time, `"unknown"` when `git` is
unavailable, or `"<sha>-dirty"` when the tree had uncommitted changes. The `-dirty`
suffix is not decoration: a fit produced from an uncommitted working tree cannot be
reproduced from the repository, and a run that will be compared against another six
months later should say so on its own face.
"""
struct FitMetadata
    timestamp::DateTime
    elapsed_seconds::Float64
    julia_version::VersionNumber
    n_threads::Int
    git_commit::String
end

"""
    git_commit_id(dir = pwd()) -> String

Short SHA plus a `-dirty` suffix when the tree has uncommitted changes. Never throws: a
missing `git`, a non-repository directory, or a detached worktree all return `"unknown"`,
because provenance capture must not be able to kill a six-hour run.
"""
function git_commit_id(dir::AbstractString = pwd())
    try
        sha = readchomp(Cmd(`git rev-parse --short HEAD`; dir = String(dir)))
        dirty = !isempty(readchomp(Cmd(`git status --porcelain`; dir = String(dir))))
        return dirty ? sha * "-dirty" : sha
    catch
        return "unknown"
    end
end

"""
    capture_metadata(start_time; dir = pwd()) -> FitMetadata

Close the stopwatch opened at `start_time` (a `time()` value) and record the machine.
"""
capture_metadata(start_time::Float64; dir::AbstractString = pwd()) =
    FitMetadata(now(), time() - start_time, VERSION, Threads.nthreads(), git_commit_id(dir))

"""
    format_elapsed(seconds) -> String

`"12.4s"`, `"3m 20s"`, `"2h 15m"`. Same shape as `Experiments._format_time`, so `time:`
tags written by this framework and by the legacy runner sort and read the same way.
"""
function format_elapsed(seconds::Real)
    s = Float64(seconds)
    if s < 60
        return string(round(s, digits = 1), "s")
    elseif s < 3600
        return string(floor(Int, s / 60), "m ", round(Int, s % 60), "s")
    else
        return string(floor(Int, s / 3600), "h ", floor(Int, (s % 3600) / 60), "m")
    end
end

function Base.show(io::IO, m::FitMetadata)
    print(io, "FitMetadata(", Dates.format(m.timestamp, "yyyy-mm-dd HH:MM:SS"),
          ", ", format_elapsed(m.elapsed_seconds),
          ", julia ", m.julia_version, ", ", m.n_threads, " threads, ", m.git_commit, ")")
end


# ==============================================================================
# 7. FIT — the run outcome
# ==============================================================================

"""
    Fit(config, folds, latents, diagnostics, metadata, save_path)

Everything one inference run produced.

| field         | is                                                    |
|---------------|-------------------------------------------------------|
| `config`      | the `FitConfig` that produced it                      |
| `folds`       | `Vector{<:FoldFit}`, one per split, in splitter order  |
| `latents`     | typed OOS posterior container, or `nothing`           |
| `diagnostics` | `ConvergenceSummary` — never optional, see below      |
| `metadata`    | `FitMetadata`                                         |
| `save_path`   | the default directory `save_fit` will write to        |

INDEXING. `fit[i] === fit.folds[i]`, `length(fit) == n_folds`, and `fit` iterates its
folds. The legacy `exp.training_results.items[i][1]` is `fit[i].chain`.

DIAGNOSTICS ARE A FIELD, NOT A FUNCTION. `Experiments.Diagnostics.check_convergence` is
something a user may run, and mostly does not. Making the audit part of construction
means every `Fit` in existence can answer "did this converge" with one field read,
without a `DataStore`, without the splitter, and without re-running anything.

LATENTS MAY BE `nothing`, and that is not the same as "extraction failed silently".
`fit_model` extracts them when the model's family is registered with
`Models.latent_family` and records the reason in `fit.config.tags` when it is not.
"""
struct Fit{C<:FitConfig,
           F<:Vector{<:FoldFit},
           L<:Union{Nothing, AbstractPosteriorLatents}}
    config::C
    folds::F
    latents::L
    diagnostics::ConvergenceSummary
    metadata::FitMetadata
    save_path::String
end

# --- 7.1 the vector interface -------------------------------------------------
#
# `getfield`, not `f.folds`: §7.2 overloads `getproperty` on `Fit`, and a plain field
# access here would route back through it on every index.

Base.length(f::Fit)   = length(getfield(f, :folds))
Base.size(f::Fit)     = (length(f),)
Base.getindex(f::Fit, i::Int) = getfield(f, :folds)[i]
Base.IndexStyle(::Type{<:Fit}) = IndexLinear()
Base.firstindex(::Fit) = 1
Base.lastindex(f::Fit) = length(f)
Base.iterate(f::Fit, s::Int = 1) = s > length(f) ? nothing : (f[s], s + 1)
Base.eltype(::Type{<:Fit{C,F}}) where {C,F} = eltype(F)

# `fold_chains`, not `chains`: `MCMCChains` exports a `chains` of its own, and defining a
# second one in a module that has `using MCMCChains` is a binding conflict, not an
# overload.
"The chains, in fold order. The thing most downstream code actually wants."
fold_chains(f::Fit) = [fd.chain for fd in getfield(f, :folds)]

"The split metadata, in fold order."
fold_metas(f::Fit) = [fd.meta for fd in getfield(f, :folds)]

"Total posterior draws across every fold. `0` for optimisation results."
total_draws(f::Fit) = sum(fold_n_draws, getfield(f, :folds); init = 0)

"`fit.config.name`, without routing through the property bridge."
fit_name(f::Fit) = getfield(f, :config).name

# --- 7.2 legacy property bridge -----------------------------------------------

"""
    LegacyTrainingResults(folds)

What `fit.training_results` returns: an `AbstractVector{Tuple{chain, meta}}` whose
`.items` is the `Vector` legacy code indexes.

`Training.TrainingResults` is `<:AbstractVector{Tuple{C,M}}` with an `items` field, and
legacy code uses both faces of it — `for (chain, meta) in res.training_results.items`
and `length(res.training_results.items)`. Both work here.

`.items` MATERIALISES a fresh vector on each access. That is a deliberate `O(n_folds)`
per read rather than a stored duplicate: a `Fit` that cached the legacy tuples would
carry two views of the same chains, and the two would drift the moment anything mutated
either. `n_folds` is tens, not millions.
"""
struct LegacyTrainingResults{C, M} <: AbstractVector{Tuple{C, M}}
    folds::Vector{<:FoldFit}

    function LegacyTrainingResults(folds::Vector{<:FoldFit})
        C = isempty(folds) ? Any : typeof(folds[1].chain)
        M = isempty(folds) ? Data.SplitMetaData : typeof(folds[1].meta)
        return new{C, M}(folds)
    end
end

Base.size(tr::LegacyTrainingResults) = size(getfield(tr, :folds))
Base.getindex(tr::LegacyTrainingResults, i::Int) = as_legacy_tuple(getfield(tr, :folds)[i])
Base.IndexStyle(::Type{<:LegacyTrainingResults}) = IndexLinear()

@inline function Base.getproperty(tr::LegacyTrainingResults, s::Symbol)
    s === :items && return [as_legacy_tuple(f) for f in getfield(tr, :folds)]
    return getfield(tr, s)
end

Base.propertynames(::LegacyTrainingResults) = (:folds, :items)

Base.show(io::IO, ::MIME"text/plain", tr::LegacyTrainingResults) =
    print(io, "TrainingResults: ", length(tr), " ")

"""
    Base.getproperty(f::Fit, s::Symbol)

Real fields pass straight through to `getfield`. Two legacy names are synthesised:

  * `:training_results` → a `LegacyTrainingResults` over `f.folds`, so
    `res.training_results.items[i][1]` resolves to `f.folds[i].chain`.
  * `:vocabulary` → `nothing`. `ExperimentResults` declared this field (`types.jl:42`,
    commented "HACK: not needed - using nothing"), every construction site set it to
    `nothing`, and no reader in the repository reads it. It is answered, not stored.

`@inline` plus a literal-`Symbol` dispatch chain, so a hot loop reading `fit.folds`
compiles to the same instruction a plain field access would.
"""
@inline function Base.getproperty(f::Fit, s::Symbol)
    s === :training_results && return LegacyTrainingResults(getfield(f, :folds))
    s === :vocabulary       && return nothing
    return getfield(f, s)
end

Base.propertynames(::Fit) = (fieldnames(Fit)..., :training_results, :vocabulary)


# ==============================================================================
# 8. IN-GAME RATE CONTAINERS
# ==============================================================================
#
# The solver that fills these lives in `ingame.jl`. Both are declared here for the same
# reason the convergence containers are: they are the framework's named data shapes, and
# `ingame.jl` is the arithmetic over them.

"""
    IngameRatesWorkspace

An in-play chain's posterior, materialised as dense vectors ALREADY PAIRED to a pre-game
container's draw count. Everything `solve_ingame_rates!` reads, allocated once.

| field     | shape              | meaning                                   |
|-----------|--------------------|-------------------------------------------|
| `α`       | `n_draws`          | log multiplier at `t = 45`, level term     |
| `β`       | `n_draws`          | slope on the centred clock `z(t)`          |
| `γ_trail` | `n_draws`          | log multiplier while trailing              |
| `γ_lead`  | `n_draws`          | log multiplier while leading               |
| `γ_red`   | `n_draws`          | log multiplier per man of advantage        |
| `δ_time`  | `n_bins × n_draws` | per-bin offset; a zero matrix disables it  |
| `edges`   | `n_bins + 1`       | bin boundaries over `[0, Tend]`            |

`δ_time` is stored BIN-MAJOR because the hot loop walks bins for a fixed draw, which is
the contiguous direction in a column-major array. That is the opposite convention to the
typed latent matrices, for the opposite reason: there the sweep is over fixtures at a
fixed draw.

Every field carries exactly `n_draws` columns and the constructor enforces it, so a
mismatch between the in-play posterior and the pre-game one is caught here rather than
becoming a recycled index inside the kernel.
"""
struct IngameRatesWorkspace
    α::Vector{Float64}
    β::Vector{Float64}
    γ_trail::Vector{Float64}
    γ_lead::Vector{Float64}
    γ_red::Vector{Float64}
    δ_time::Matrix{Float64}
    edges::Vector{Float64}
    Tend::Float64

    function IngameRatesWorkspace(α, β, γ_trail, γ_lead, γ_red, δ_time, edges, Tend)
        nd = length(α)
        nd > 0 || error("IngameRatesWorkspace: α is empty; there is no posterior to pair.")
        for (nm, v) in ((:β, β), (:γ_trail, γ_trail), (:γ_lead, γ_lead), (:γ_red, γ_red))
            length(v) == nd || error(
                "IngameRatesWorkspace: $nm has $(length(v)) draws but α has $nd. Every " *
                "site must come from the same paired posterior sweep.")
        end
        size(δ_time, 2) == nd || error(
            "IngameRatesWorkspace: δ_time has $(size(δ_time, 2)) draw columns but α has $nd.")
        size(δ_time, 1) == length(edges) - 1 || error(
            "IngameRatesWorkspace: δ_time has $(size(δ_time, 1)) bin rows but `edges` " *
            "describes $(length(edges) - 1) bins.")
        issorted(edges) || error("IngameRatesWorkspace: `edges` must be non-decreasing.")
        all(isfinite, α) && all(isfinite, β) || error(
            "IngameRatesWorkspace: non-finite α or β — exp() of it is Inf and every " *
            "resulting rate NaN.")
        return new(Vector{Float64}(α), Vector{Float64}(β),
                   Vector{Float64}(γ_trail), Vector{Float64}(γ_lead),
                   Vector{Float64}(γ_red), Matrix{Float64}(δ_time),
                   Vector{Float64}(edges), Float64(Tend))
    end
end

"Posterior draws this workspace is paired to."
workspace_n_draws(ws::IngameRatesWorkspace) = length(ws.α)

"Time bins this workspace integrates over."
workspace_n_bins(ws::IngameRatesWorkspace) = size(ws.δ_time, 1)

Base.show(io::IO, ws::IngameRatesWorkspace) =
    print(io, "IngameRatesWorkspace(", workspace_n_draws(ws), " draws, ",
          workspace_n_bins(ws), " bins to ", ws.Tend, "')")

"""
    LiveMatchRates(Λ_home, Λ_away)

The destination `solve_ingame_rates!` writes into: the integrated remaining goal
intensity per side, one entry per posterior draw.

`Λ_side = ∫_{t_now}^{Tend} λ_side(u) du`, so the remaining-goals count for that side is
`Poisson(Λ_side)` and the final score is `goals_so_far + that`. Allocate one per worker
with [`alloc_live_rates`](@ref) and reuse it across ticks — the whole point of the solver
is that a repricing tick allocates nothing.
"""
struct LiveMatchRates
    Λ_home::Vector{Float64}
    Λ_away::Vector{Float64}

    function LiveMatchRates(Λ_home::Vector{Float64}, Λ_away::Vector{Float64})
        length(Λ_home) == length(Λ_away) || error(
            "LiveMatchRates: Λ_home has $(length(Λ_home)) draws but Λ_away has " *
            "$(length(Λ_away)). Both sides come from one paired sweep.")
        return new(Λ_home, Λ_away)
    end
end

Base.length(r::LiveMatchRates) = length(r.Λ_home)

Base.show(io::IO, r::LiveMatchRates) =
    print(io, "LiveMatchRates(", length(r), " draws)")


# ==============================================================================
# 9. DISPLAY
# ==============================================================================

Base.show(io::IO, f::Fit) = print(io, "Fit(", fit_name(f), ", ", length(f), " folds)")

function Base.show(io::IO, ::MIME"text/plain", f::Fit)
    cfg  = getfield(f, :config)
    meta = getfield(f, :metadata)
    lat  = getfield(f, :latents)
    println(io, "Fit: ", cfg.name)
    println(io, "  model       : ", nameof(typeof(cfg.model)))
    println(io, "  splitter    : ", nameof(typeof(cfg.splitter)))
    println(io, "  sampler     : ", nameof(typeof(cfg.sampler)))
    println(io, "  folds       : ", length(f), "  (", total_draws(f), " draws total)")
    println(io, "  latents     : ", lat === nothing ? "—  (family not registered)" :
                string(nameof(typeof(lat)), "  ", n_matches(lat), " fixtures × ",
                       n_draws(lat), " draws"))
    println(io, "  diagnostics : ", diagnostics_line(getfield(f, :diagnostics)))
    println(io, "  ran         : ", Dates.format(meta.timestamp, "yyyy-mm-dd HH:MM"),
                " in ", format_elapsed(meta.elapsed_seconds),
                " on ", meta.n_threads, " thread(s)")
    println(io, "  commit      : ", meta.git_commit)
    print(io,   "  save_path   : ", getfield(f, :save_path))
end
