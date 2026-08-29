# ==============================================================================
# 07 — UNIFIED INFERENCE FRAMEWORK : THE TYPE HIERARCHY
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# ------------------------------------------------------------------------------
# WHAT THIS FILE REPLACES
# ------------------------------------------------------------------------------
#
# Two modules that are one module wearing a coat:
#
#   src/training/     `Training.train(model, config, feature_sets)` loops over splits
#                     and calls `Samplers.run_sampler`. It adds a progress line and a
#                     checkpoint file. It owns `TrainingConfig` and `TrainingResults`.
#
#   src/experiments/  `Experiments.run_experiment(ds, config)` calls
#                     `Data.create_id_boundaries`, `Features.create_features`,
#                     `Training.train`, and a stopwatch. It owns `ExperimentConfig`
#                     (which WRAPS `TrainingConfig`) and `ExperimentResults` (which
#                     WRAPS `TrainingResults`).
#
# The nesting is not free. Reading one fold's chain out of a completed run is
#
#     exp_results.training_results.items[i][1]
#
# — four hops, two of which exist only because the split happened. `ExperimentResults`
# also carries `vocabulary::Any`, an NLP leftover that every construction site sets to
# `nothing` (runner.jl:84) and no reader ever reads.
#
# And the split is in the WRONG PLACE. What actually belongs together is:
#
#     sample the chains  →  audit whether they converged  →  extract the latents
#
# Today those are three modules (`Training`, `Experiments.Diagnostics`,
# `Experiments.post_processing`), and nothing forces the second to happen at all. A
# run whose R-hat is 1.4 saves, loads, prices, and stakes exactly like one that
# converged. `Fit` makes the audit a FIELD, so a fit that was never audited cannot be
# constructed.
#
# ------------------------------------------------------------------------------
# THE FOUR CONTAINERS
# ------------------------------------------------------------------------------
#
#   FoldFit     one split: its chain and its split metadata.
#   FitConfig   the immutable recipe. Flat: name, model, splitter, sampler.
#   Fit         the outcome: folds + latents + diagnostics + provenance.
#   InGameFit   the same, for a model conditioned on a pre-game `Fit`.
#
# `Fit` is `IndexLinear`, so `fit[i]` is fold `i` and `length(fit)` is the fold count.
# The four-hop read above becomes `fit[i].chain`.
#
# ------------------------------------------------------------------------------
# WHERE THIS DEVIATES FROM THE BRIEFING, AND WHY
# ------------------------------------------------------------------------------
#
# 1. `AbstractFootballModel` IS NOT REDECLARED.
#
#    The briefing opens with `abstract type AbstractFootballModel end`. Declaring that
#    here would create a SECOND, unrelated abstract type: every engine in
#    `src/models/pregame/` subtypes `BayesianFootball.TypesInterfaces.AbstractFootballModel`,
#    and none of them would satisfy the new one. `FitConfig{M<:AbstractFootballModel}`
#    would then reject `DynamicXGModel` — the exact opposite of the compatibility this
#    prototype exists to guarantee.
#
#    So the repository's root type is reused verbatim, and the in-game branch is carved
#    out BENEATH it:
#
#        AbstractFootballModel                    (BayesianFootball.TypesInterfaces)
#        ├── every engine in src/models/pregame/  → pre-game
#        └── AbstractInGameModel                  (new, here)
#
#    `AbstractPreGameModel` is an alias for the root rather than a sibling, because
#    there is no way to retroactively re-parent 40-odd existing engine types. The
#    predicate that actually discriminates is `is_pregame(m)`, defined below, and it is
#    what the code uses.
#
# 2. `FoldFit`'s chain slot is UNCONSTRAINED, not `C<:Chains`.
#
#    `Samplers.run_sampler(::MAPConfig)` and `(::MLEConfig)` do not return a `Chains`
#    (src/samplers/engines/optimization.jl). `C<:Chains` would make this framework
#    unable to hold a MAP run that `Training.train` holds today. Backward compatibility
#    is the requirement; the tighter bound loses to it. Everything that needs a real
#    chain — the convergence audit, latent extraction — dispatches on `::Chains` at its
#    own entry point instead, and says so when handed something else.
#
# 3. `FitConfig` gains ONE field beyond the briefing's seven: `execution`.
#
#    The briefing's `FitConfig` drops `TrainingConfig`, which carried BOTH the sampler
#    and the `Independent(parallel, max_concurrent_tasks)` execution strategy. The
#    sampler is kept as its own field; the strategy has nowhere else to live, and
#    `QueuedNUTSConfig`'s whole point is the flattened task queue. It defaults to
#    `AutoExecution()`, which reads the strategy off the sampler type, so every
#    construction the briefing writes still compiles and behaves.
#
# ==============================================================================

using Dates
using DataFrames
using MCMCChains
using Printf
using Statistics

# The typed posterior containers (06) — `l04_parity.jl` transitively includes
# l03 → l02 → l01, so this one line loads that whole prototype, containers, kernels,
# parity harness and deterministic synthetic posteriors alike.
include(joinpath(@__DIR__, "..", "06_typed_posterior_latents", "l04_parity.jl"))

# The composable count builder (05). Loaded here rather than in the runner so that a
# `Fit` over a `ComposableCountModel` is constructible from the framework alone.
include(joinpath(@__DIR__, "..", "scottish_lower", "05_composable_count_builder",
                 "l03_engine.jl"))

const UIF_BF   = BayesianFootball
const UIF_D    = BayesianFootball.Data
const UIF_TI   = BayesianFootball.TypesInterfaces
const UIF_PG   = BayesianFootball.Models.PreGame
const UIF_Samp = BayesianFootball.Samplers
const UIF_Feat = BayesianFootball.Features


# ==============================================================================
# 1. THE MODEL HIERARCHY
# ==============================================================================

"""
    AbstractPreGameModel

Alias for the repository's root model type. Every engine in `src/models/pregame/` and
every `ComposableCountModel` from `05_composable_count_builder` is one.

An alias rather than a new abstract type: see the header, deviation 1. Use
`is_pregame(model)` for the predicate that actually discriminates.
"""
const AbstractPreGameModel = UIF_TI.AbstractFootballModel

"""
    AbstractInGameModel <: AbstractFootballModel

A model that prices a match ALREADY IN PROGRESS, given a pre-game posterior as its
baseline.

The distinction from a pre-game model is not cosmetic and is not about when the code
runs. It is about what the model's `λ` MEANS:

  * a pre-game model's `λ` is a whole-match expected count, and pricing integrates it
    over `[0, 90]` implicitly;
  * an in-game model's `λ(t)` is an INSTANTANEOUS rate, and pricing must integrate it
    over `[t_now, 90]` explicitly, conditioned on the score and man-count that hold at
    `t_now`.

Feeding one into the other's pricer produces a plausible, systematically wrong number,
which is exactly the failure the type split makes impossible.
"""
abstract type AbstractInGameModel <: UIF_TI.AbstractFootballModel end

"""
    is_pregame(model) -> Bool
    is_ingame(model)  -> Bool

Which side of the split a model is on. `is_pregame` is the complement of
`is_ingame` because `AbstractPreGameModel` is the shared root (header, deviation 1),
so `model isa AbstractPreGameModel` is true for BOTH and is not the test you want.
"""
is_ingame(model) = model isa AbstractInGameModel
is_pregame(model) = !is_ingame(model)


# ==============================================================================
# 2. FOLD FIT — one split's outcome
# ==============================================================================

"""
    FoldFit(fold, chain, meta)

A single fold's inference outcome: what the sampler returned, and which slice of the
calendar it was fitted on.

| field   | is                                                                  |
|---------|---------------------------------------------------------------------|
| `fold`  | 1-based split index, in `create_id_boundaries` order                 |
| `chain` | the sampler's return value — `MCMCChains.Chains` for every NUTS path |
| `meta`  | `SplitMetaData` / `GroupedSplitMetaData` for that split              |

This is the legacy `Tuple{Chains, SplitMetaData}` with the two things a tuple cannot
carry: a name for each slot, and the fold index. `results_array[i][1]` becomes
`folds[i].chain`, and a transposed destructure becomes a `MethodError` instead of a
`Chains` object being treated as metadata.

`chain` is deliberately unconstrained — see the header, deviation 2.
"""
struct FoldFit{C, M<:UIF_D.AbstractSplitMetaData}
    fold::Int
    chain::C
    meta::M

    function FoldFit(fold::Integer, chain::C, meta::M) where {C, M<:UIF_D.AbstractSplitMetaData}
        fold > 0 || error("FoldFit: fold index must be positive, got $fold.")
        return new{C, M}(Int(fold), chain, meta)
    end
end

"The legacy tuple this fold would have been. For the compatibility bridge only."
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
#
# The one thing `TrainingConfig` carried that `FitConfig` would otherwise drop.
# See the header, deviation 3.

"""
    AbstractExecution

How the folds are dispatched onto threads. Orthogonal to WHAT is sampled, which is the
sampler's business.
"""
abstract type AbstractExecution end

"""
    AutoExecution(; max_concurrent_splits = 0, max_concurrent_tasks = 0)

Read the strategy off the sampler at RUN time: `QueuedNUTSConfig` → `QueuedExecution`,
anything else → `ThreadedExecution` when there is more than one thread,
`SequentialExecution` otherwise. The default, and the reason every `FitConfig` in the
briefing still works unmodified.

This is exactly the test `train_independent` performs (independent.jl:32-38), moved
from inside the training loop to the point where it is a decision rather than a branch.

Both caps default to `0`, meaning "decide from `Threads.nthreads()` when the run
starts". A config built on a laptop and run on a 32-core box therefore uses the 32
cores; a non-zero cap is honoured verbatim.
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

The `K splits × N chains` flattening from `Training._train_queued`
(src/training/strategies/independent.jl:111): every (split, chain) pair is one queue
entry, so a fold whose chains finish early does not leave cores idle waiting for its
slowest sibling. Requires a sampler that accepts a `chain_id`, i.e. `QueuedNUTSConfig`.
"""
Base.@kwdef struct QueuedExecution <: AbstractExecution
    max_concurrent_tasks::Int = Threads.nthreads()
end

"""
    resolve_execution(exec, sampler) -> AbstractExecution

`AutoExecution` resolved against a concrete sampler; everything else passed through.
Kept out of the `FitConfig` constructor so that a config built on a 1-thread machine
and run on a 32-thread one uses the 32 threads.
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


# ==============================================================================
# 4. FIT CONFIG — the pre-game recipe
# ==============================================================================

"""
    FitConfig(; name, model, splitter, sampler, execution = AutoExecution(),
                tags = String[], description = "", save_dir = "./data/fits")

The immutable specification for one pre-game inference run.

FLAT, where `ExperimentConfig` was nested. The legacy form was

    ExperimentConfig(name = …, model = …, splitter = …,
                     training_config = TrainingConfig(sampler, Independent(…), nothing, false))

and reading the sampler back out of a saved run took `config.training_config.sampler`.
`TrainingConfig` existed to bundle a sampler with an execution strategy and two
checkpoint fields; the sampler and the strategy are now their own fields, and
checkpointing moved to `fit_model`'s keyword (`l03_engine.jl` §5) where it is a
property of the RUN, not of the recipe — the same recipe re-run without checkpoints is
the same recipe.

A legacy-shaped construction still works:

    FitConfig(name = "x", model = m, splitter = s,
              training_config = Training.TrainingConfig(sampler, Independent(), nothing, false))

unpacks `training_config` into `sampler` and `execution`. See §4.1.

| field         | meaning                                                        |
|---------------|----------------------------------------------------------------|
| `name`        | run name; the save directory is `save_dir/name_<timestamp>`     |
| `model`       | the engine — any `AbstractFootballModel`                        |
| `splitter`    | `CVConfig` / `GroupedCVConfig` / `StaticSplit` / …              |
| `sampler`     | `NUTSConfig`, `QueuedNUTSConfig`, `MAPConfig`, `ReplaySampler`  |
| `execution`   | fold dispatch strategy; `AutoExecution()` reads it off `sampler`|
| `tags`        | free-form labels; `fit_model` appends `time:<elapsed>`          |
| `description` | free-form prose                                                 |
| `save_dir`    | root directory for `save_fit`                                   |
"""
struct FitConfig{M<:UIF_TI.AbstractFootballModel,
                 S<:UIF_D.AbstractSplitter,
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

# --- 4.1 the one keyword constructor, accepting BOTH shapes -------------------
#
# `ExperimentConfig(name = …, training_config = TrainingConfig(sampler, strategy, …))`
# is what every runner in `current_development/` currently writes. Accepting it here is
# what makes `const ExperimentConfig = FitConfig` (l06) a genuine drop-in rather than a
# rename that breaks at the first call site.
#
# ONE constructor, not two, and that is forced rather than chosen: keyword methods
# dispatch on their POSITIONAL signature, and two zero-positional-argument constructors
# have the same one. A second `FitConfig(; …, training_config, …)` definition would not
# overload the first — it would silently REPLACE it, and every `sampler = …`
# construction in this file would start raising `UndefKeywordError`. So both shapes are
# handled in one body, with `sampler` and `training_config` each defaulting to
# `nothing` and exactly one of them required.

"""
    execution_from_strategy(strategy) -> AbstractExecution

Map a legacy `Training.Independent` onto an `AbstractExecution`, carrying its two
concurrency caps across.

`parallel = true` maps to `AutoExecution`, NOT to `QueuedExecution`, because the legacy
loop does not decide queued-vs-threaded from the strategy either: it decides from the
SAMPLER (`is_queued = typeof(config.sampler).name.name == :QueuedNUTSConfig`,
independent.jl:32). Reproducing that means deferring, which is what `AutoExecution` is.
"""
function execution_from_strategy(strategy)
    nameof(typeof(strategy)) === :Independent || return AutoExecution()
    get_or(strategy, :parallel, false) || return SequentialExecution()
    return AutoExecution(
        max_concurrent_splits = get_or(strategy, :max_concurrent_splits, 0),
        max_concurrent_tasks  = get_or(strategy, :max_concurrent_tasks, 0),
    )
end

"""
    FitConfig(; name, model, splitter, sampler, execution = AutoExecution(),
                tags = String[], description = "", save_dir = "./data/fits")
    FitConfig(; name, model, splitter, training_config, kwargs...)

Both shapes, one method.

The second is the legacy `ExperimentConfig` call: `training_config.sampler` becomes
`sampler`, and `training_config.strategy` is mapped by `execution_from_strategy`.

`checkpoint_dir` and `cleanup_checkpoints` on a legacy `TrainingConfig` are NOT
silently dropped — `legacy_checkpointing(tc)` returns them, for forwarding to
`fit_model`'s keywords of the same names.
"""
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

# --- 4.2 the `training_config` property view ----------------------------------
#
# Legacy code reads the sampler back out of a config as
# `res.config.training_config.sampler` — `save_experiment` does exactly that
# (runner.jl:142). `FitConfig` has no such field, so the read is SYNTHESISED.
#
# Defined here, next to the struct, for the same reason `Fit`'s property bridge is
# (§7.2): `Base.getproperty` must exist before anything compiles a field access against
# the default one.

"""
    LegacyTrainingConfig

What `config.training_config` returns: the four fields the legacy `TrainingConfig` had,
computed from the flat `FitConfig`. A VIEW — storing one would put the nesting back.

`checkpoint_dir` is always `nothing` and `cleanup_checkpoints` always `false`, because
checkpointing moved from the recipe to `fit_model`'s keywords (`l03_engine.jl` §4).
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

"`obj.name` if it exists, `default` otherwise. Used only on legacy/reconstructed objects."
get_or(obj, name::Symbol, default) = hasproperty(obj, name) ? getproperty(obj, name) : default

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
# 5. IN-GAME FIT CONFIG — the conditional recipe
# ==============================================================================

"""
    InGameFitConfig(; name, model, pregame, splitter, sampler, execution = AutoExecution(),
                      tags = String[], description = "", save_dir = "./data/inplay_fits")

The specification for an in-game run. Identical to `FitConfig` but for one field, and
that field is the whole point.

`pregame` is the BASELINE SOURCE: either a completed pre-game `Fit`, or the
`AbstractPosteriorLatents` container extracted from one. An in-game intensity model
does not learn a team's scoring rate — it learns a MULTIPLIER on a rate the pre-game
model already estimated:

    log λ_side(t) = log λ_side^pre + α + β·z(t) + γ_state·(lead/trail)
                                   + γ_red·man_adv + δ_time[bin]

Holding the source in the config, rather than passing it to the pricer, is what makes
the pairing auditable after the fact. `l04_ingame_bridge.jl`'s header sets out the
failure this prevents: an in-game chain fitted against pre-game posterior A, priced
six weeks later against pre-game posterior B, is off by exactly the ratio of the two
baselines and looks completely normal on a chart.
"""
Base.@kwdef struct InGameFitConfig{M<:UIF_TI.AbstractFootballModel,
                                   P,
                                   S<:UIF_D.AbstractSplitter,
                                   Sam,
                                   E<:AbstractExecution}
    name::String
    model::M
    pregame::P
    splitter::S
    sampler::Sam
    execution::E = AutoExecution()
    tags::Vector{String} = String[]
    description::String = ""
    save_dir::String = "./data/inplay_fits"
end

function Base.show(io::IO, c::InGameFitConfig)
    print(io, "InGameFitConfig(", c.name, ", ", nameof(typeof(c.model)),
          ", pregame=", nameof(typeof(c.pregame)), ")")
end


# ==============================================================================
# 6. FIT METADATA — provenance
# ==============================================================================

"""
    FitMetadata(timestamp, elapsed_seconds, julia_version, n_threads, git_commit)

What was true about the machine when the run happened.

`git_commit` is the short SHA of the working tree at run time, or `"unknown"` when
`git` is unavailable, or `"<sha>-dirty"` when the tree had uncommitted changes. The
`-dirty` suffix is not decoration: a fit produced from an uncommitted working tree
cannot be reproduced from the repository, and a run that is going to be compared
against another six months later should say so on its own face.
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

Short SHA plus a `-dirty` suffix when the tree has uncommitted changes. Never throws:
a missing `git`, a non-repository directory, or a detached worktree all return
`"unknown"`, because provenance capture must not be able to kill a six-hour run.
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

`"12.4s"`, `"3m 20s"`, `"2h 15m"`. Same shape as `Experiments._format_time`
(runner.jl:18), so `time:` tags written by this framework and by the legacy runner sort
and read the same way.
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
# 7. FIT — the pre-game outcome
# ==============================================================================

"""
    Fit(config, folds, latents, diagnostics, metadata, save_path)

Everything one pre-game inference run produced.

| field         | is                                                                  |
|---------------|---------------------------------------------------------------------|
| `config`      | the `FitConfig` that produced it                                     |
| `folds`       | `Vector{<:FoldFit}`, one per split, in splitter order                 |
| `latents`     | typed OOS posterior container (06), or `nothing`                     |
| `diagnostics` | `ConvergenceSummary` — never optional, see below                     |
| `metadata`    | `FitMetadata`                                                        |
| `save_path`   | the default directory `save_fit` will write to                       |

INDEXING. `fit[i] === fit.folds[i]`, `length(fit) == n_folds`, and `fit` iterates its
folds. The legacy `exp.training_results.items[i][1]` is `fit[i].chain`.

DIAGNOSTICS ARE A FIELD, NOT A FUNCTION. `Experiments.Diagnostics.check_convergence`
is something a user may run, and mostly does not. Making the audit part of
construction means every `Fit` in existence — in memory, on disk, six months old —
can answer "did this converge" without a `DataStore`, without the splitter, and
without re-running anything. `fit.diagnostics.passed` is one field read.

LATENTS MAY BE `nothing`, and that is not the same as "extraction failed silently".
`fit_model` extracts them when the model's family is registered with
`latent_family` (06, `l02_extract.jl` §1) and records the reason in
`fit.config.tags` when it is not.
"""
struct Fit{C<:FitConfig,
           F<:Vector{<:FoldFit},
           L<:Union{Nothing, AbstractPosteriorLatents},
           D}
    config::C
    folds::F
    latents::L
    diagnostics::D
    metadata::FitMetadata
    save_path::String
end

"""
    InGameFit(config, folds, pregame_latents, diagnostics, metadata, save_path)

Everything one in-game inference run produced.

`pregame_latents` is the RESOLVED baseline — the container itself, not the `Fit` it
came from. Resolving at construction rather than at pricing time is deliberate: the
alternative keeps a whole pre-game `Fit` (every chain of every fold) alive for the
lifetime of the in-game object, and makes "which baseline priced this" a question you
answer by chasing a reference.
"""
struct InGameFit{C<:InGameFitConfig,
                 F<:Vector{<:FoldFit},
                 P<:AbstractPosteriorLatents,
                 D}
    config::C
    folds::F
    pregame_latents::P
    diagnostics::D
    metadata::FitMetadata
    save_path::String
end

const AbstractFit = Union{Fit, InGameFit}

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

Base.length(f::InGameFit)   = length(getfield(f, :folds))
Base.size(f::InGameFit)     = (length(f),)
Base.getindex(f::InGameFit, i::Int) = getfield(f, :folds)[i]
Base.IndexStyle(::Type{<:InGameFit}) = IndexLinear()
Base.firstindex(::InGameFit) = 1
Base.lastindex(f::InGameFit) = length(f)
Base.iterate(f::InGameFit, s::Int = 1) = s > length(f) ? nothing : (f[s], s + 1)

"The chains, in fold order. The thing 90% of downstream code actually wants."
chains(f::AbstractFit) = [fd.chain for fd in getfield(f, :folds)]

"The split metadata, in fold order."
fold_metas(f::AbstractFit) = [fd.meta for fd in getfield(f, :folds)]

"Total posterior draws across every fold. `0` for optimisation results."
total_draws(f::AbstractFit) = sum(fold_n_draws, getfield(f, :folds); init = 0)

"`fit.config.name`, without routing through the property bridge."
fit_name(f::AbstractFit) = getfield(f, :config).name


# --- 7.2 legacy property bridge -----------------------------------------------
#
# Lives in l01 rather than l06 because `Base.getproperty` must be defined ONCE for
# `Fit`, at the point the struct is defined. Splitting the definition across two files
# would mean the first `fit.config` executed before l06 loaded would compile against
# the default `getproperty` and stay compiled that way.
#
# The full compatibility story — what a legacy call site may and may not do — is in
# `l06_compat_bridge.jl`. This is just the mechanism.

"""
    LegacyTrainingResults(folds)

What `res.training_results` returns: an `AbstractVector{Tuple{chain, meta}}` whose
`.items` is the `Vector` legacy code indexes.

`Training.TrainingResults` is `<:AbstractVector{Tuple{C,M}}` with an `items` field, and
legacy code uses both faces of it — `for (chain, meta) in res.training_results.items`
(smile_negbin/r03_pipeline_smoke.jl:214) and `length(res.training_results.items)`
(r02_train_ireland.jl:156). Both work here.

`.items` MATERIALISES a fresh vector on each access. That is a deliberate `O(n_folds)`
per read rather than a stored duplicate of the folds: a `Fit` that cached the legacy
tuples would carry two views of the same chains, and the two would drift the moment
anything mutated either. `n_folds` is tens, not millions.
"""
struct LegacyTrainingResults{C, M} <: AbstractVector{Tuple{C, M}}
    folds::Vector{<:FoldFit}

    function LegacyTrainingResults(folds::Vector{<:FoldFit})
        C = isempty(folds) ? Any : typeof(folds[1].chain)
        M = isempty(folds) ? UIF_D.SplitMetaData : typeof(folds[1].meta)
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
  * `:vocabulary` → `nothing`. `ExperimentResults` declared this field
    (`types.jl:42`, commented "HACK: not needed - using nothing"), every construction
    site set it to `nothing` (runner.jl:84), and no reader in the repository reads it.
    It is answered rather than stored.

`@inline` and a literal-`Symbol` dispatch chain, so a hot loop that reads `fit.folds`
compiles to the same instruction a plain field access would.
"""
@inline function Base.getproperty(f::Fit, s::Symbol)
    s === :training_results && return LegacyTrainingResults(getfield(f, :folds))
    s === :vocabulary       && return nothing
    return getfield(f, s)
end

Base.propertynames(::Fit) = (fieldnames(Fit)..., :training_results, :vocabulary)

@inline function Base.getproperty(f::InGameFit, s::Symbol)
    s === :training_results && return LegacyTrainingResults(getfield(f, :folds))
    s === :vocabulary       && return nothing
    s === :latents          && return getfield(f, :pregame_latents)
    return getfield(f, s)
end

Base.propertynames(::InGameFit) =
    (fieldnames(InGameFit)..., :training_results, :vocabulary, :latents)


# --- 7.3 the `.df` bridge on typed latents ------------------------------------
#
# `extract_oos_predictions` used to return a `LatentStates` whose `.df` was the untyped
# DataFrame. It now returns the typed container (06), and a legacy caller's next line is
# `latents.df`. `to_legacy_dataframe` (06, l02 §6) rebuilds exactly that frame.
#
# Defined per CONCRETE type, not on `AbstractPosteriorLatents`. An abstract-typed
# `getproperty` would intercept `l.λ_home` inside the score-grid kernels for any future
# container as well, and the zero-allocation claim in `r01_demo.jl` would then depend on
# a method nobody looked at. Three explicit methods are three things a reader can check.

@inline function Base.getproperty(l::CountLatents, s::Symbol)
    s === :df && return to_legacy_dataframe(l)
    return getfield(l, s)
end

@inline function Base.getproperty(l::SmileLatents, s::Symbol)
    s === :df && return to_legacy_dataframe(l)
    return getfield(l, s)
end

@inline function Base.getproperty(l::RecombLatents, s::Symbol)
    s === :df && return to_legacy_dataframe(l)
    return getfield(l, s)
end

Base.propertynames(l::CountLatents)  = (fieldnames(CountLatents)..., :df)
Base.propertynames(l::SmileLatents)  = (fieldnames(SmileLatents)..., :df)
Base.propertynames(l::RecombLatents) = (fieldnames(RecombLatents)..., :df)


# ==============================================================================
# 8. DISPLAY
# ==============================================================================

function Base.show(io::IO, f::Fit)
    print(io, "Fit(", fit_name(f), ", ", length(f), " folds)")
end

function _uif_show_fit(io::IO, f::AbstractFit, kind::String, extra)
    cfg  = getfield(f, :config)
    meta = getfield(f, :metadata)
    diag = getfield(f, :diagnostics)
    println(io, kind, ": ", cfg.name)
    println(io, "  model       : ", nameof(typeof(cfg.model)))
    println(io, "  splitter    : ", nameof(typeof(cfg.splitter)))
    println(io, "  sampler     : ", nameof(typeof(cfg.sampler)))
    println(io, "  folds       : ", length(f), "  (", total_draws(f), " draws total)")
    for (k, v) in extra
        println(io, "  ", rpad(k, 11), " : ", v)
    end
    println(io, "  diagnostics : ", _uif_diag_line(diag))
    println(io, "  ran         : ", Dates.format(meta.timestamp, "yyyy-mm-dd HH:MM"),
                " in ", format_elapsed(meta.elapsed_seconds),
                " on ", meta.n_threads, " thread(s)")
    println(io, "  commit      : ", meta.git_commit)
    print(io,   "  save_path   : ", getfield(f, :save_path))
end

# Overridden in l02 once `ConvergenceSummary` exists; this is the fallback for a `Fit`
# constructed with something else in the slot.
_uif_diag_line(d) = string(d)

function Base.show(io::IO, ::MIME"text/plain", f::Fit)
    lat = getfield(f, :latents)
    extra = (("latents", lat === nothing ? "—  (family not registered)" :
                          string(nameof(typeof(lat)), "  ", n_matches(lat),
                                 " fixtures × ", n_draws(lat), " draws")),)
    _uif_show_fit(io, f, "Fit", extra)
end

function Base.show(io::IO, f::InGameFit)
    print(io, "InGameFit(", fit_name(f), ", ", length(f), " folds)")
end

function Base.show(io::IO, ::MIME"text/plain", f::InGameFit)
    pre = getfield(f, :pregame_latents)
    extra = (("baseline", string(nameof(typeof(pre)), "  ", n_matches(pre),
                                 " fixtures × ", n_draws(pre), " draws")),)
    _uif_show_fit(io, f, "InGameFit", extra)
end
