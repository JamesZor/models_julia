# src/training/inference/engine.jl
#
# The execution engine.
#
#     boundaries → features → SAMPLE → AUDIT → EXTRACT LATENTS → Fit
#
# The middle three are ONE transaction here. In the legacy path they are three separate
# user actions and the second and third are optional:
#
#     res     = Experiments.run_experiment(task)                  # sample
#     chains  = Diagnostics.extract_chains(ds, res)               # audit — if remembered
#     latents = Experiments.extract_oos_predictions(ds, res)      # extract — hours later
#
# Both of the later two RE-DERIVE the boundaries and REBUILD the feature sets from the
# `DataStore`, because the run threw them away. `extract_oos_predictions` even carries a
# drift guard (`post_processing.jl:147`) whose sole purpose is to notice that the
# re-derivation produced a different number of folds than the run did — a check that
# exists only because the work is done twice. Doing it once, while the feature sets are
# still in scope, deletes the guard, the second derivation, and the class of bug the
# guard watched for.
#
# `fit_model` has TWO entry points and the `DataStore` one is a thin wrapper:
#
#     fit_model(ds::DataStore, config)             # derives the folds from `ds`
#     fit_model(config; feature_sets, oos_fixtures) # takes them
#
# The second is not a testing hook. It is the seam `MatchDay` wants — it has its
# fixtures in hand and no interest in re-deriving a walk-forward split to price
# Saturday — and it is what lets the whole lifecycle be exercised without a database.

# ==============================================================================
# 1. THE SAMPLING SEAM
# ==============================================================================

"""
    sample_fold(model, sampler, feature_set, fold; chain_id = nothing)

Produce one fold's inference result. THE single point at which this framework touches a
sampler.

The default method is `Training.train`'s body verbatim (`method.jl`): build the Turing
model from the feature set, hand it to `Samplers.run_sampler`, forward `chain_id` when
the sampler wants one. `fold` is passed but unused by the default method — it exists so
a sampler CAN be fold-aware, which is what [`ReplaySampler`](@ref) is.

Extending this is how a new sampler joins the framework, and it is one method.
"""
function sample_fold(model, sampler, feature_set, fold::Int;
                     chain_id::Union{Int, Nothing} = nothing)
    turing_model = PreGame.build_turing_model(model, feature_set)
    return chain_id === nothing ?
        Samplers.run_sampler(turing_model, sampler) :
        Samplers.run_sampler(turing_model, sampler, chain_id)
end

"""
    ReplaySampler(chains; n_chains = 1)

A sampler that returns chains it was given, one per fold, in fold order.

Not a mock. It is the mechanism behind three things this framework needs for real:

  * re-auditing or re-extracting latents from an OLD run without re-sampling it —
    `fit_model` with a replay sampler rebuilds a complete `Fit`, diagnostics and latents
    included, from chains already on disk;
  * verifying the lifecycle end to end with no database and no MCMC;
  * bisecting a pipeline change against a FIXED posterior, so a price difference is
    attributable to the change and not to a different random seed.

`n_chains = 1` because a replayed chain is already whatever shape it was saved as; the
queued executor's per-chain fan-out would replay the same object N times.
"""
struct ReplaySampler{C} <: Samplers.AbstractSamplerConfig
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
# (`independent.jl:95` logs `@error` and moves on) and it is right for a six-hour
# walk-forward run: losing fold 9 to a `PosDefException` should not throw away folds 1-8
# and 10-40. What is NEW is that the failure survives into the `Fit` — `fit_model` counts
# the empty slots and writes a `folds_failed:N` tag — where the legacy path leaves a
# short `training_results.items` and nothing that says why.

_inf_noop(args...) = nothing

"""
    run_folds(model, sampler, execution, feature_sets; on_progress) -> Vector{Any}

Sample every fold under `execution`. `feature_sets` is anything indexable yielding
`(FeatureSet, meta)` tuples — a `FeatureCollection`, or a plain `Vector`.
"""
function run_folds(model, sampler, exec::AbstractExecution, feature_sets;
                   on_progress = _inf_noop)
    return run_folds(model, sampler, resolve_execution(exec, sampler), feature_sets,
                     Val(:resolved); on_progress = on_progress)
end

# --- sequential ---------------------------------------------------------------

function run_folds(model, sampler, ::SequentialExecution, feature_sets, ::Val{:resolved};
                   on_progress = _inf_noop)
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
                   on_progress = _inf_noop)
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
Flatten `n_folds × n_chains` into ONE queue, exactly as `Training._train_queued` does.

The reason this exists: under the per-fold executor a machine with 32 threads and 4
chains runs 8 folds at a time and then WAITS for the slowest of the 32 chains before
starting fold 9. Walk-forward folds differ in size by a factor of several (fold 1 has
one month of history, fold 40 has three seasons), so the tail is long. One flat queue
keeps every core busy until the queue is empty.

Chains are combined with `cat(…; dims = 3)` the moment a fold's last chain lands and the
per-fold buffer is dropped, so peak memory is the live chains rather than all of them.
"""
function run_folds(model, sampler, exec::QueuedExecution, feature_sets, ::Val{:resolved};
                   on_progress = _inf_noop)
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
# `Models.extract_latents` works one fold at a time and returns one container per fold.
# Legacy `extract_oos_predictions` returns ONE frame for the whole run
# (`vcat(split_dfs...)`), and downstream — calibration, backtesting, CLV — reads that
# single object.
#
# So the folds' containers are concatenated ALONG FIXTURES. The draw dimension is
# SHARED, not concatenated: fold 3's draw 17 and fold 9's draw 17 come from different
# chains and are not the same posterior sample. Nothing here treats them as paired, and
# nothing downstream may either — the merged container is a stack of independent
# per-fixture posteriors, which is precisely what the legacy frame was.

"""
    merge_latents(containers) -> AbstractPosteriorLatents

Concatenate per-fold containers along the FIXTURE axis.

Requires an identical draw count across folds. That is not a limitation of the
implementation — it is the same requirement the legacy `vcat` had, silently: a frame
whose `λ_h` cells are 3200-long for folds 1-8 and 800-long for fold 9 is accepted by
`vcat` and then produces a length mismatch inside a pricing kernel hours later. Here it
is an error naming both folds.
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

# A heterogeneous `Vector{Any}` from a comprehension over folds: re-narrow, then dispatch.
_merge_latents(cs::AbstractVector) = _merge_latents([c for c in cs])

"""
    extract_run_latents(model, folds, oos_fixtures, feature_sets) -> (container, note)

Typed OOS latents for the whole run, or `nothing` with a reason.

`nothing` — never a throw — for three situations, all legitimate:

  * the model's family is not registered with `Models.latent_family`;
  * every fold has an empty out-of-sample fixture set (a terminal walk-forward step);
  * the sampler returned point estimates, which have no posterior to extract.

`fit_model` records `note` as a tag, so the reason is attached to the `Fit` rather than
lost to a log line.
"""
function extract_run_latents(model, folds::Vector{<:FoldFit}, oos_fixtures, feature_sets)
    per = Any[]
    for f in folds
        f.chain isa Chains || return (nothing, "latents:skipped(point-estimate)")
        fx = oos_fixtures[f.fold]
        (fx === nothing || nrow(fx) == 0) && continue
        fs = feature_sets[f.fold][1]
        try
            push!(per, extract_latents(model, f.chain, fx, fs))
        catch e
            return (nothing, "latents:failed(fold $(f.fold): $(_inf_brief(e)))")
        end
    end
    isempty(per) && return (nothing, "latents:none(no out-of-sample fixtures)")
    try
        return (merge_latents(per), "")
    catch e
        return (nothing, "latents:failed(merge: $(_inf_brief(e)))")
    end
end

"""
    _inf_narrow(v) -> Vector

Re-type a `Vector{Abstract}` to its concrete element type when every element shares one.
A comprehension over a `Vector{Any}` of sampler results infers `FoldFit` and stops there;
narrowing to `Vector{FoldFit{Chains, SplitMetaData}}` is what lets `Fit`'s `F` parameter
be concrete, which is what makes `fit[i].chain` a direct load rather than a dynamic
dispatch.
"""
function _inf_narrow(v::Vector)
    isempty(v) && return v
    T = typeof(v[1])
    return all(x -> typeof(x) === T, v) ? Vector{T}(v) : v
end

"First line of an exception's message, for a tag that has to fit on one line."
function _inf_brief(e)
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
# `FitConfig` field.
#
# The filename is `Training.get_checkpoint_path`'s (`split_007.jls`, zero-padded to
# three) and the payload is the same `(result, metadata)` tuple `save_split_checkpoint`
# writes, so an interrupted legacy run resumes under this engine and vice versa.
# `load_checkpoints` UNWRAPS that tuple back to the sampler result, because the fold's
# metadata comes from the feature sets of the resumed run, not from the checkpoint.

"Path of fold `i`'s checkpoint inside `dir`. Matches `Training.get_checkpoint_path`."
checkpoint_path(dir::AbstractString, i::Integer) =
    joinpath(dir, "split_$(lpad(Int(i), 3, '0')).jls")

"The sampler result out of a checkpoint payload, which may be a `(result, meta)` tuple."
_inf_checkpoint_result(x::Tuple{Any, <:Data.AbstractSplitMetaData}) = x[1]
_inf_checkpoint_result(x) = x

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
            out[i] = _inf_checkpoint_result(Serialization.deserialize(p))
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
# 5. FIT MODEL — the entry points
# ==============================================================================

"""
    fit_model(ds::DataStore, config::FitConfig; kwargs...) -> Fit

The whole lifecycle: split, build features, sample, audit, extract, package.

Replaces `Experiments.run_experiment(ds, config)` one-for-one, and additionally performs
the convergence audit and the latent extraction that the legacy caller had to remember
to run afterwards against a re-derived `DataStore`.

# Keywords
- `thresholds` — `ConvergenceThresholds` for the audit.
- `checkpoint_dir` — resume/persist per-fold results. `nothing` for in-memory only.
- `cleanup_checkpoints` — delete them once every fold has landed.
- `with_latents` — set `false` to skip OOS extraction (a training-only run).
- `quiet` — suppress the progress log.
"""
function fit_model(ds::Data.DataStore, config::FitConfig; quiet::Bool = false, kwargs...)
    quiet || _inf_header(config.name)

    quiet || _inf_step(1, "Generating data splits")
    boundaries = Data.create_id_boundaries(ds, config.splitter)
    quiet || _inf_info("$(length(boundaries)) splits")

    quiet || _inf_step(2, "Building feature sets")
    feature_sets = Features.create_features(boundaries, ds, config.model, config.splitter)

    quiet || _inf_step(3, "Resolving out-of-sample fixtures")
    oos = Any[Data.get_next_matches(ds, feature_sets[i], config.splitter)
              for i in 1:length(feature_sets)]

    return fit_model(config; feature_sets = feature_sets, oos_fixtures = oos,
                     quiet = quiet, kwargs...)
end

fit_model(config::FitConfig, ds::Data.DataStore; kwargs...) = fit_model(ds, config; kwargs...)

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
                   thresholds::ConvergenceThresholds = ConvergenceThresholds(),
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
        quiet || _inf_info("all $n folds restored from checkpoints")
    else
        quiet || _inf_step(4, "Sampling $(length(pending)) of $n folds " *
                              "($(nameof(typeof(config.sampler))), " *
                              "$(nameof(typeof(resolve_execution(config.execution, config.sampler)))))")
        pending_fs = [feature_sets[i] for i in pending]
        prog = quiet ? _inf_noop : _inf_progress(start)
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
    folds = _inf_narrow(FoldFit[FoldFit(i, results[i], feature_sets[i][2])
                                for i in 1:n if results[i] !== nothing])
    isempty(folds) && error(
        "fit_model: every one of $n folds failed to sample. The per-fold errors were " *
        "logged above; nothing was saved.")

    # --- audit ----------------------------------------------------------------
    quiet || _inf_step(5, "Auditing convergence")
    diagnostics = audit_convergence(folds; thresholds = thresholds,
                                    max_depth = sampler_max_depth(config.sampler))
    quiet || _inf_info(diagnostics_line(diagnostics))

    # --- latents --------------------------------------------------------------
    latents = nothing
    note = ""
    if with_latents && oos_fixtures !== nothing
        quiet || _inf_step(6, "Extracting out-of-sample latents")
        latents, note = extract_run_latents(config.model, folds, oos_fixtures, feature_sets)
        quiet || _inf_info(latents === nothing ? note :
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

    quiet || _inf_footer(meta, n_failed)
    return Fit(stamped, folds, latents, diagnostics, meta, save_path)
end

"""
    default_save_path(config, metadata) -> String

`save_dir/name_yyyymmdd_HHMMSS`. Same shape as `run_experiment`'s, so a `list_fits` scan
and a `list_experiments` scan sort a mixed directory identically.
"""
default_save_path(config, meta::FitMetadata) =
    joinpath(config.save_dir,
             string(config.name, "_", Dates.format(meta.timestamp, "yyyymmdd_HHMMSS")))


# ==============================================================================
# 6. PROGRESS LOG
# ==============================================================================

function _inf_header(name)
    printstyled("\n>> FIT: ", color = :magenta, bold = true)
    printstyled(name, "\n", color = :white, bold = true)
    println("-"^66)
end

function _inf_step(n, msg)
    printstyled(" [$n] ", color = :cyan, bold = true)
    println(msg, "...")
end

_inf_info(msg) = (printstyled("     > ", color = :light_black); println(msg))

function _inf_footer(meta::FitMetadata, n_failed::Int)
    println("-"^66)
    if n_failed == 0
        printstyled("DONE", color = :green, bold = true)
    else
        printstyled("DONE ($n_failed fold(s) failed)", color = :yellow, bold = true)
    end
    println(" in ", format_elapsed(meta.elapsed_seconds))
end

"A progress callback updating an interactive ProgressMeter in-place, thread-safe, with start timestamp and ETA/speed."
function _inf_progress(start_time::Float64 = time())
    p = nothing
    lk = ReentrantLock()
    start_dt = Dates.format(Dates.now(), "HH:MM:SS")
    return function (done::Int, total::Int)
        lock(lk) do
            if p === nothing
                desc = "     > [$start_dt] Sampling: "
                p = ProgressMeter.Progress(total; desc = desc, color = :green, showspeed = true, output = stderr)
            end
            ProgressMeter.update!(p, done)
        end
    end
end
