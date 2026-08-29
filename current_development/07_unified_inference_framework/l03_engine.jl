# ==============================================================================
# 07 — UNIFIED INFERENCE FRAMEWORK : THE EXECUTION ENGINE
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# ------------------------------------------------------------------------------
# WHAT THIS FILE REPLACES
# ------------------------------------------------------------------------------
#
#   Experiments.run_experiment   (runner.jl:41-87)   — splits, features, stopwatch
#   Training.train               (method.jl:27-43)   — a 3-branch strategy dispatch
#   Training.train_independent   (independent.jl:16) — checkpoints + two loops
#   Training._train_standard     (independent.jl:48)
#   Training._train_queued       (independent.jl:111)
#
# Five functions across two modules, of which exactly two do work: `_train_standard`
# and `_train_queued`. The other three forward.
#
# ------------------------------------------------------------------------------
# THE LIFECYCLE, MADE ATOMIC
# ------------------------------------------------------------------------------
#
#     boundaries → features → SAMPLE → AUDIT → EXTRACT LATENTS → Fit
#
# The middle three are one transaction here. Today they are three separate user
# actions, and the second and third are optional:
#
#     res     = Experiments.run_experiment(task)                  # sample
#     chains  = Diagnostics.extract_chains(ds, res)               # audit — if you remember
#     latents = Experiments.extract_oos_predictions(ds, res)      # extract — hours later
#
# Both of the later two RE-DERIVE the boundaries and RE-BUILD the feature sets from the
# `DataStore` (post_processing.jl:139-155, extraction.jl:13-21), because the run threw
# them away. `extract_oos_predictions` even carries a `DataStore` drift guard
# (post_processing.jl:147) whose sole purpose is to notice that the re-derivation
# produced a different number of folds than the run did — a check that exists only
# because the work is done twice.
#
# Doing it once, while the feature sets are still in scope, deletes the guard, the
# second derivation, and the class of bug the guard was watching for.
#
# ------------------------------------------------------------------------------
# THE ONE STRUCTURAL DECISION IN THIS FILE
# ------------------------------------------------------------------------------
#
# `fit_model` has TWO entry points, and the `DataStore` one is a thin wrapper over the
# other:
#
#     fit_model(ds::DataStore, config)                      # derives folds from `ds`
#     fit_model(config; feature_sets, metas, oos_fixtures)  # takes them
#
# This is not a testing hook bolted on afterwards. It is the seam the legacy design
# lacks, and its absence is why `r01_demo.jl` of this prototype can verify the whole
# lifecycle with no database: everything downstream of "here are the folds" is
# exercised by the second entry point, and the first adds only the two `Data`/`Features`
# calls it forwards to.
#
# It is also the seam a `MatchDay` caller wants, which already has its fixtures in hand
# and no interest in re-deriving a walk-forward split to price this Saturday.
#
# ==============================================================================

using Base.Threads
using DataFrames
using Dates
using MCMCChains
using Printf
using Serialization

include(joinpath(@__DIR__, "l02_convergence.jl"))


# ==============================================================================
# 1. THE SAMPLING SEAM
# ==============================================================================

"""
    sample_fold(model, sampler, feature_set, fold; chain_id = nothing)

Produce one fold's inference result. THE single point at which this framework touches a
sampler.

The default method is `Training.train`'s body verbatim (method.jl:10-18): build the
Turing model from the feature set, hand it to `Samplers.run_sampler`, forward
`chain_id` when the sampler wants one. `fold` is passed but unused — it exists so a
sampler CAN be fold-aware, which is what `ReplaySampler` is.

Extending this is how a new sampler joins the framework, and it is one method.
"""
function sample_fold(model, sampler, feature_set, fold::Int;
                     chain_id::Union{Int, Nothing} = nothing)
    turing_model = UIF_PG.build_turing_model(model, feature_set)
    return chain_id === nothing ?
        UIF_Samp.run_sampler(turing_model, sampler) :
        UIF_Samp.run_sampler(turing_model, sampler, chain_id)
end

"""
    ReplaySampler(chains; n_chains = 1)

A sampler that returns chains it was given, one per fold, in fold order.

Not a mock. It is the mechanism behind three things this framework needs for real:

  * re-auditing or re-extracting latents from an OLD run without re-sampling it —
    `fit_model(config_with_replay; …)` rebuilds a complete `Fit`, diagnostics and
    latents included, from chains already on disk;
  * `r01_demo.jl` verifying the entire lifecycle with no database and no MCMC;
  * bisecting a pipeline change against a FIXED posterior, so that a price difference
    is attributable to the change and not to a different random seed.

`n_chains = 1` because a replayed chain is already whatever shape it was saved as;
the queued executor's per-chain fan-out would replay the same object N times.
"""
struct ReplaySampler{C} <: UIF_Samp.AbstractSamplerConfig
    chains::Vector{C}
    n_chains::Int
end

ReplaySampler(chains::AbstractVector; n_chains::Integer = 1) =
    ReplaySampler{eltype(chains)}(collect(chains), Int(n_chains))

function sample_fold(model, s::ReplaySampler, feature_set, fold::Int;
                     chain_id::Union{Int, Nothing} = nothing)
    1 <= fold <= length(s.chains) || error(
        "ReplaySampler: asked for fold $fold but only $(length(s.chains)) chain(s) " *
        "were supplied. The replay must cover every fold the splitter produces.")
    return s.chains[fold]
end

Base.show(io::IO, s::ReplaySampler) =
    print(io, "ReplaySampler(", length(s.chains), " folds)")

"The `max_depth` a sampler was configured with, for the tree-depth gate. 10 if it has none."
sampler_max_depth(sampler) = get_or(sampler, :max_depth, 10)

"How many chains this sampler runs per fold. 1 when it does not say."
sampler_n_chains(sampler) = get_or(sampler, :n_chains, 1)


# ==============================================================================
# 2. FOLD DISPATCH
# ==============================================================================
#
# Three executors, one signature. Each returns `Vector{Any}` of length `n_folds`, with
# `nothing` in any slot whose sampling threw.
#
# FAILURES ARE RECORDED, NOT RAISED. That is the legacy behaviour
# (independent.jl:95-96 logs `@error` and moves on) and it is right for a 6-hour
# walk-forward run: losing fold 9 to a `PosDefException` should not throw away folds
# 1-8 and 10-40. What is NEW is that the failure survives into the `Fit` — `fit_model`
# counts the empty slots and writes a `folds_failed:N` tag — where the legacy path
# leaves a short `training_results.items` and nothing that says why.

"""
    run_folds(model, sampler, exec, feature_sets; on_progress) -> Vector{Any}

Sample every fold under `exec`. `feature_sets` is anything indexable yielding
`(FeatureSet, meta)` tuples — a `FeatureCollection`, or a plain `Vector`.
"""
function run_folds(model, sampler, exec::AbstractExecution, feature_sets;
                   on_progress = _uif_noop)
    return run_folds(model, sampler, resolve_execution(exec, sampler), feature_sets,
                     Val(:resolved); on_progress = on_progress)
end

_uif_noop(args...) = nothing

# --- sequential ---------------------------------------------------------------

function run_folds(model, sampler, ::SequentialExecution, feature_sets, ::Val{:resolved};
                   on_progress = _uif_noop)
    n = length(feature_sets)
    out = Vector{Any}(nothing, n)
    for i in 1:n
        fs, _ = feature_sets[i]
        try
            out[i] = sample_fold(model, sampler, fs, i)
        catch e
            @error "Fold $i failed" exception = (e, catch_backtrace())
        end
        on_progress(i, n)
    end
    return out
end

# --- threaded, one task per fold ----------------------------------------------

function run_folds(model, sampler, exec::ThreadedExecution, feature_sets, ::Val{:resolved};
                   on_progress = _uif_noop)
    n = length(feature_sets)
    out = Vector{Any}(nothing, n)
    conc = clamp(exec.max_concurrent_splits, 1, max(1, n))
    sem = Base.Semaphore(conc)
    lk = ReentrantLock()
    done = Atomic{Int}(0)
    @sync for i in 1:n
        Threads.@spawn begin
            Base.acquire(sem)
            try
                fs, _ = feature_sets[i]
                r = sample_fold(model, sampler, fs, i)
                lock(lk) do
                    out[i] = r
                end
            catch e
                @error "Fold $i failed" exception = (e, catch_backtrace())
            finally
                on_progress(atomic_add!(done, 1) + 1, n)
                Base.release(sem)
            end
        end
    end
    return out
end

# --- queued, one task per (fold, chain) ---------------------------------------

"""
Flatten `n_folds × n_chains` into ONE queue, exactly as `_train_queued` does
(independent.jl:111-178).

The reason this exists at all: under the per-fold executor, a machine with 32 threads
and 4 chains runs 8 folds at a time and then WAITS for the slowest of the 32 chains
before starting fold 9. Walk-forward folds differ in size by a factor of several
(fold 1 has one month of history, fold 40 has three seasons), so the tail is long. One
flat queue keeps every core busy until the queue is empty.

Chains are combined with `cat(…; dims = 3)` the moment a fold's last chain lands, and
the per-fold buffer is dropped, so peak memory is the live chains and not all of them.
"""
function run_folds(model, sampler, exec::QueuedExecution, feature_sets, ::Val{:resolved};
                   on_progress = _uif_noop)
    n = length(feature_sets)
    nc = sampler_n_chains(sampler)
    nc > 1 || return run_folds(model, sampler,
                               ThreadedExecution(max_concurrent_splits =
                                                 exec.max_concurrent_tasks),
                               feature_sets, Val(:resolved); on_progress = on_progress)

    out = Vector{Any}(nothing, n)
    buf = Dict{Int, Vector{Any}}(i => Vector{Any}(nothing, nc) for i in 1:n)
    landed = Dict{Int, Int}(i => 0 for i in 1:n)

    tasks = [(i, c) for i in 1:n for c in 1:nc]
    total = length(tasks)
    conc = clamp(exec.max_concurrent_tasks, 1, max(1, total))
    sem = Base.Semaphore(conc)
    lk = ReentrantLock()
    done = Atomic{Int}(0)

    @sync for (i, c) in tasks
        Threads.@spawn begin
            Base.acquire(sem)
            try
                fs, _ = feature_sets[i]
                r = sample_fold(model, sampler, fs, i; chain_id = c)
                lock(lk) do
                    buf[i][c] = r
                    landed[i] += 1
                    if landed[i] == nc
                        parts = filter(!isnothing, buf[i])
                        out[i] = length(parts) == 1 ? parts[1] : cat(parts...; dims = 3)
                        delete!(buf, i)
                    end
                end
            catch e
                @error "Fold $i chain $c failed" exception = (e, catch_backtrace())
                lock(lk) do
                    landed[i] += 1
                    if landed[i] == nc && haskey(buf, i)
                        parts = filter(!isnothing, buf[i])
                        out[i] = isempty(parts) ? nothing :
                                 (length(parts) == 1 ? parts[1] : cat(parts...; dims = 3))
                        delete!(buf, i)
                    end
                end
            finally
                on_progress(atomic_add!(done, 1) + 1, total)
                Base.release(sem)
            end
        end
    end
    return out
end


# ==============================================================================
# 3. LATENT EXTRACTION AND MERGE
# ==============================================================================
#
# `extract_latents` (06, l02) works one fold at a time and returns one container per
# fold. Legacy `extract_oos_predictions` returns ONE frame for the whole run
# (`vcat(split_dfs...)`, post_processing.jl:169). Downstream — calibration, backtesting,
# CLV — reads that single object.
#
# So the folds' containers are concatenated ALONG FIXTURES. The draw dimension is
# shared, not concatenated: fold 3's draw 17 and fold 9's draw 17 come from different
# chains and are not the same posterior sample. Nothing here treats them as paired, and
# nothing downstream may either — the merged container is a stack of independent
# per-fixture posteriors, which is precisely what the legacy frame was.

"""
    merge_latents(containers) -> AbstractPosteriorLatents

Concatenate per-fold containers along the FIXTURE axis.

Requires an identical draw count across folds. That is not a limitation of the
implementation — it is the same requirement the legacy `vcat` had, silently: a frame
whose `λ_h` cells are 3200-long for folds 1-8 and 800-long for fold 9 is accepted by
`vcat` and then produces a length mismatch inside a pricing kernel, hours later. Here
it is an error naming both folds.
"""
function merge_latents(cs::AbstractVector)
    isempty(cs) && return nothing
    length(cs) == 1 && return first(cs)
    nd = n_draws(first(cs))
    for (i, c) in enumerate(cs)
        typeof(c) == typeof(first(cs)) || error(
            "merge_latents: fold $i is a $(typeof(c)) but fold 1 is a " *
            "$(typeof(first(cs))). Every fold of one run must produce one family.")
        n_draws(c) == nd || error(
            "merge_latents: fold $i has $(n_draws(c)) draws but fold 1 has $nd. " *
            "Folds must be sampled with the same sampler configuration.")
    end
    return _merge_latents(cs)
end

_merge_ids(cs) = reduce(vcat, latent_match_ids(c) for c in cs)
_merge_mat(cs, f) = reduce(vcat, f(c) for c in cs)

_merge_latents(cs::AbstractVector{<:CountLatents{T, Nothing}}) where {T} =
    CountLatents(_merge_ids(cs),
                 _merge_mat(cs, c -> getfield(c, :λ_home)),
                 _merge_mat(cs, c -> getfield(c, :λ_away)))

_merge_latents(cs::AbstractVector{<:CountLatents{T, <:NamedTuple}}) where {T} =
    CountLatents(_merge_ids(cs),
                 _merge_mat(cs, c -> getfield(c, :λ_home)),
                 _merge_mat(cs, c -> getfield(c, :λ_away)),
                 (r_h = _merge_mat(cs, c -> getfield(c, :observation_params).r_h),
                  r_a = _merge_mat(cs, c -> getfield(c, :observation_params).r_a)))

_merge_latents(cs::AbstractVector{<:RecombLatents}) =
    RecombLatents(_merge_ids(cs),
                  _merge_mat(cs, c -> getfield(c, :λ_open_h)),
                  _merge_mat(cs, c -> getfield(c, :λ_open_a)),
                  _merge_mat(cs, c -> getfield(c, :λ_pen_h)),
                  _merge_mat(cs, c -> getfield(c, :λ_pen_a)),
                  _merge_mat(cs, c -> getfield(c, :λ_og_h)),
                  _merge_mat(cs, c -> getfield(c, :λ_og_a)),
                  _merge_mat(cs, c -> getfield(c, :pxg_h)),
                  _merge_mat(cs, c -> getfield(c, :pxg_a)))

function _merge_latents(cs::AbstractVector{<:SmileLatents})
    first(cs).strikes == last(cs).strikes || error(
        "merge_latents: folds carry different strike ladders; a merged φ would index " *
        "different market lines per row.")
    obs = getfield(first(cs), :observation_params) === nothing ? nothing :
          (r_h = _merge_mat(cs, c -> getfield(c, :observation_params).r_h),
           r_a = _merge_mat(cs, c -> getfield(c, :observation_params).r_a))
    return SmileLatents(_merge_ids(cs),
                        _merge_mat(cs, c -> getfield(c, :λ_home)),
                        _merge_mat(cs, c -> getfield(c, :λ_away)),
                        obs,
                        _merge_mat(cs, c -> getfield(c, :λ_tot)),
                        cat((getfield(c, :φ) for c in cs)...; dims = 1),
                        copy(first(cs).strikes))
end

# A heterogeneous `Vector{Any}` from `map` over folds: re-narrow, then dispatch.
_merge_latents(cs::AbstractVector) = _merge_latents([c for c in cs])

"""
    extract_run_latents(model, folds, oos_fixtures, feature_sets) -> container or nothing

Typed OOS latents for the whole run, or `nothing` with a reason.

`nothing` — never a throw — for three situations, all of which are legitimate:

  * the model's family is not registered with `latent_family` (06, l02 §1);
  * every fold has an empty out-of-sample fixture set (a terminal walk-forward step);
  * the sampler returned point estimates, which have no posterior to extract.

Returns `(latents, note)`. `fit_model` records `note` as a tag so the reason is
attached to the `Fit` rather than lost to a log line.
"""
function extract_run_latents(model, folds::Vector{<:FoldFit}, oos_fixtures, feature_sets)
    per = Any[]
    for (k, f) in enumerate(folds)
        f.chain isa Chains || return (nothing, "latents:skipped(point-estimate)")
        fx = oos_fixtures[f.fold]
        (fx === nothing || nrow(fx) == 0) && continue
        fs = feature_sets[f.fold][1]
        try
            push!(per, extract_latents(model, f.chain, fx, fs))
        catch e
            return (nothing, "latents:failed(fold $(f.fold): $(_uif_brief(e)))")
        end
    end
    isempty(per) && return (nothing, "latents:none(no out-of-sample fixtures)")
    try
        return (merge_latents(per), "")
    catch e
        return (nothing, "latents:failed(merge: $(_uif_brief(e)))")
    end
end

"""
    _uif_narrow(v) -> Vector

Re-type a `Vector{Abstract}` to its concrete element type when every element shares
one. A comprehension over a `Vector{Any}` of sampler results infers `FoldFit` and stops
there; narrowing to `Vector{FoldFit{Chains, SplitMetaData}}` is what lets `Fit`'s `F`
parameter be concrete, which is what makes `fit[i].chain` a direct load rather than a
dynamic dispatch.
"""
function _uif_narrow(v::Vector)
    isempty(v) && return v
    T = typeof(v[1])
    return all(x -> typeof(x) === T, v) ? Vector{T}(v) : v
end

"First line of an exception's message, for a tag that has to fit on one line."
function _uif_brief(e)
    s = sprint(showerror, e)
    i = findfirst('\n', s)
    s = i === nothing ? s : s[1:(i - 1)]
    return length(s) > 120 ? s[1:117] * "..." : s
end


# ==============================================================================
# 4. CHECKPOINTS
# ==============================================================================
#
# A property of the RUN, not of the recipe — hence a `fit_model` keyword rather than a
# `FitConfig` field. Same on-disk layout as `Training.save_split_checkpoint`
# (src/training/checkpointing.jl) so an interrupted legacy run can be resumed by this
# engine and vice versa.

"Path of fold `i`'s checkpoint inside `dir`."
checkpoint_path(dir::AbstractString, i::Integer) = joinpath(dir, "split_$(i).jls")

"""
    load_checkpoints(dir, n) -> Vector{Any}

Fold results already on disk, `nothing` for the ones still to run. A corrupt file is
treated as absent and re-run, not raised: the whole point of a checkpoint is to make a
run resumable, and dying on the resume defeats it.
"""
function load_checkpoints(dir::Union{Nothing, AbstractString}, n::Integer)
    out = Vector{Any}(nothing, n)
    dir === nothing && return out
    isdir(dir) || return out
    for i in 1:n
        p = checkpoint_path(dir, i)
        isfile(p) || continue
        try
            out[i] = Serialization.deserialize(p)
        catch e
            @warn "Checkpoint $p unreadable; fold $i will be re-run" exception = e
        end
    end
    return out
end

"Atomically write one fold's result. `.tmp` → `mv`, so an interrupt never leaves half a file."
function save_checkpoint(dir::AbstractString, i::Integer, value)
    mkpath(dir)
    target = checkpoint_path(dir, i)
    tmp = target * ".tmp." * string(rand(UInt64), base = 16)
    try
        Serialization.serialize(tmp, value)
        mv(tmp, target; force = true)
    catch e
        isfile(tmp) && rm(tmp; force = true)
        rethrow(e)
    end
    return target
end


# ==============================================================================
# 5. FIT MODEL — the pre-game entry points
# ==============================================================================

"""
    fit_model(ds::DataStore, config::FitConfig; kwargs...) -> Fit

The whole lifecycle: split, build features, sample, audit, extract, package.

Replaces `Experiments.run_experiment(ds, config)` one-for-one, and additionally
performs the convergence audit and the latent extraction that the legacy caller had to
remember to run afterwards against a re-derived `DataStore`.

# Keywords
- `gates` — `ConvergenceGates` for the audit.
- `checkpoint_dir` — resume/persist per-fold results. `nothing` for in-memory only.
- `cleanup_checkpoints` — delete them once every fold has landed.
- `with_latents` — set `false` to skip OOS extraction (a training-only run).
- `quiet` — suppress the progress log.
"""
function fit_model(ds::UIF_D.DataStore, config::FitConfig; quiet::Bool = false, kwargs...)
    quiet || _uif_header(config.name)

    quiet || _uif_step(1, "Generating data splits")
    boundaries = UIF_D.create_id_boundaries(ds, config.splitter)
    quiet || _uif_info("$(length(boundaries)) splits")

    quiet || _uif_step(2, "Building feature sets")
    feature_sets = UIF_Feat.create_features(boundaries, ds, config.model, config.splitter)

    quiet || _uif_step(3, "Resolving out-of-sample fixtures")
    oos = Any[UIF_D.get_next_matches(ds, feature_sets[i], config.splitter)
              for i in 1:length(feature_sets)]

    return fit_model(config; feature_sets = feature_sets, oos_fixtures = oos,
                     quiet = quiet, kwargs...)
end

"""
    fit_model(config::FitConfig; feature_sets, oos_fixtures = nothing, kwargs...) -> Fit

The `DataStore`-free entry point: everything from "here are the folds" onwards.

`feature_sets` is a `FeatureCollection` or any indexable of `(FeatureSet, meta)` tuples.
`oos_fixtures[i]` is fold `i`'s held-out fixture frame, or `nothing` to skip latent
extraction for that fold.

See the file header for why this seam exists and who else wants it.
"""
function fit_model(config::FitConfig;
                   feature_sets,
                   oos_fixtures = nothing,
                   gates::ConvergenceGates = ConvergenceGates(),
                   checkpoint_dir::Union{Nothing, String} = nothing,
                   cleanup_checkpoints::Bool = false,
                   with_latents::Bool = true,
                   quiet::Bool = false)
    start = time()
    n = length(feature_sets)
    n > 0 || error("fit_model: `feature_sets` is empty — the splitter produced no folds.")

    # --- sample ---------------------------------------------------------------
    results = load_checkpoints(checkpoint_dir, n)
    pending = findall(isnothing, results)

    if isempty(pending)
        quiet || _uif_info("all $n folds restored from checkpoints")
    else
        quiet || _uif_step(4, "Sampling $(length(pending)) of $n folds " *
                              "($(nameof(typeof(config.sampler))), " *
                              "$(nameof(typeof(resolve_execution(config.execution, config.sampler)))))")
        pending_fs = [feature_sets[i] for i in pending]
        prog = quiet ? _uif_noop : _uif_progress(start)
        fresh = run_folds(config.model, config.sampler, config.execution, pending_fs;
                          on_progress = prog)
        for (k, i) in enumerate(pending)
            results[i] = fresh[k]
            results[i] === nothing && continue
            checkpoint_dir === nothing ||
                save_checkpoint(checkpoint_dir, i, (results[i], feature_sets[i][2]))
        end
    end

    n_failed = count(isnothing, results)
    folds = _uif_narrow(FoldFit[FoldFit(i, results[i], feature_sets[i][2])
                                for i in 1:n if results[i] !== nothing])
    isempty(folds) && error(
        "fit_model: every one of $n folds failed to sample. The per-fold errors were " *
        "logged above; nothing was saved.")

    # --- audit ----------------------------------------------------------------
    quiet || _uif_step(5, "Auditing convergence")
    diagnostics = audit_convergence(folds; gates = gates,
                                    max_depth = sampler_max_depth(config.sampler))
    quiet || _uif_info(_uif_diag_line(diagnostics))

    # --- latents --------------------------------------------------------------
    latents = nothing
    note = ""
    if with_latents && oos_fixtures !== nothing
        quiet || _uif_step(6, "Extracting out-of-sample latents")
        latents, note = extract_run_latents(config.model, folds, oos_fixtures, feature_sets)
        quiet || _uif_info(latents === nothing ? note :
                           "$(nameof(typeof(latents)))  $(n_matches(latents)) fixtures " *
                           "× $(n_draws(latents)) draws")
    elseif with_latents
        note = "latents:skipped(no oos_fixtures supplied)"
    end

    # --- package --------------------------------------------------------------
    meta = capture_metadata(start)
    tags = copy(config.tags)
    push!(tags, "time:" * format_elapsed(meta.elapsed_seconds))
    n_failed > 0 && push!(tags, "folds_failed:$n_failed")
    isempty(note) || push!(tags, note)
    diagnostics.passed || push!(tags, "convergence:FAIL")

    stamped = FitConfig(name = config.name, model = config.model,
                        splitter = config.splitter, sampler = config.sampler,
                        execution = config.execution, tags = tags,
                        description = config.description, save_dir = config.save_dir)

    save_path = default_save_path(stamped, meta)

    if checkpoint_dir !== nothing && cleanup_checkpoints && n_failed == 0
        for i in 1:n
            p = checkpoint_path(checkpoint_dir, i)
            isfile(p) && rm(p; force = true)
        end
    end

    quiet || _uif_footer(meta, n_failed)
    return Fit(stamped, folds, latents, diagnostics, meta, save_path)
end

"""
    default_save_path(config, metadata) -> String

`save_dir/name_yyyymmdd_HHMMSS`. Same shape as `run_experiment`'s (runner.jl:78-79), so
a `list_fits` scan and a `list_experiments` scan sort a mixed directory identically.
"""
default_save_path(config, meta::FitMetadata) =
    joinpath(config.save_dir,
             string(config.name, "_", Dates.format(meta.timestamp, "yyyymmdd_HHMMSS")))


# ==============================================================================
# 6. PROGRESS LOG
# ==============================================================================

function _uif_header(name)
    printstyled("\n>> FIT: ", color = :magenta, bold = true)
    printstyled(name, "\n", color = :white, bold = true)
    println("-"^66)
end

function _uif_step(n, msg)
    printstyled(" [$n] ", color = :cyan, bold = true)
    println(msg, "...")
end

_uif_info(msg) = (printstyled("     > ", color = :light_black); println(msg))

function _uif_footer(meta::FitMetadata, n_failed::Int)
    println("-"^66)
    if n_failed == 0
        printstyled("DONE", color = :green, bold = true)
    else
        printstyled("DONE ($n_failed fold(s) failed)", color = :yellow, bold = true)
    end
    println(" in ", format_elapsed(meta.elapsed_seconds))
end

"A progress callback that reports percentage and ETA, throttled to one line per fold."
function _uif_progress(start::Float64)
    return function (done::Int, total::Int)
        el = time() - start
        eta = done == 0 ? 0.0 : (el / done) * (total - done)
        printstyled(@sprintf("     %4d / %-4d (%5.1f%%)  elapsed %-9s eta %-9s\n",
                             done, total, 100 * done / total,
                             format_elapsed(el), format_elapsed(eta)),
                    color = :green)
    end
end
