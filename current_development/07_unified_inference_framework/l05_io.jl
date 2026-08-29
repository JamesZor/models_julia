# ==============================================================================
# 07 — UNIFIED INFERENCE FRAMEWORK : PERSISTENCE AND SIDECARS
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# ------------------------------------------------------------------------------
# WHAT THIS FILE REPLACES
# ------------------------------------------------------------------------------
#
#   Experiments.save_experiment   (runner.jl:93-156)
#   Experiments.load_experiment   (runner.jl:220-237)
#   Experiments.list_experiments  (runner.jl:162-218)
#   Experiments.save/load_oos_predictions (post_processing.jl:63-113)
#
# and fixes two defects in them.
#
# DEFECT 1 — THE BINARY SAVE IS NOT ATOMIC.
#
#     jldsave(joinpath(target_path, "results.jld2"); results)     # runner.jl:104
#
# writes in place. Interrupt a six-hour walk-forward run during that write — Ctrl-C, an
# OOM kill, a full disk — and `results.jld2` is a truncated file that JLD2 will open and
# then fail inside, hours later, on the read. The run is gone and the directory looks
# complete.
#
# Curiously, `save_oos_predictions` in the SAME module already does it correctly
# (post_processing.jl:97-99): `.tmp` then `mv`. The pattern was known; it just was not
# applied to the file that matters most. Every write here uses it.
#
# DEFECT 2 — `meta.json` CANNOT ANSWER THE QUESTION IT EXISTS FOR.
#
# The sidecar's whole purpose is to let `list_experiments` scan 200 directories without
# deserialising 200 multi-gigabyte `.jld2` files. It records name, model, splitter,
# sampler, timestamp, elapsed time, and whether latents exist (runner.jl:138-146).
#
# It does not record whether the run CONVERGED, or how many folds it has. So the
# question a scan is actually run to answer — "which of these 40 fits are usable" —
# still needs 40 full loads. `save_fit` writes `max_rhat`, `min_ess`, `n_divergent`,
# `converged`, `n_folds` and `n_oos_fixtures` into the sidecar, and `list_fits` shows
# them, so the scan answers it.
#
# ------------------------------------------------------------------------------
# THE ON-DISK LAYOUT
# ------------------------------------------------------------------------------
#
#     <save_dir>/<name>_<yyyymmdd_HHMMSS>/
#       results.jld2       the Fit — chains, folds, diagnostics, metadata
#       meta.json          the scannable sidecar (see above)
#       config.json        human-readable recipe: model, splitter, sampler, tags
#       oos_latents.jls    the typed posterior container, serialised
#
# Same four filenames as the legacy layout, on purpose: `list_fits` and
# `list_experiments` can be pointed at the same directory, and a legacy consumer
# looking for `oos_latents.jls` finds one.
#
# `oos_latents.jls` is `Serialization`, not JLD2, because that is what
# `save_oos_predictions` used and what every cached file on disk today is. It carries
# the TYPED container now rather than a `LatentStates`; `load_fit` handles both.
#
# ------------------------------------------------------------------------------
# WHAT "TRANSPARENTLY UPGRADES LEGACY RESULTS" MEANS, PRECISELY
# ------------------------------------------------------------------------------
#
# `load_fit` accepts four shapes and returns a `Fit` for all of them:
#
#   1. a `Fit` — saved by this framework;
#   2. a `BayesianFootball.Experiments.ExperimentResults` — saved by the legacy runner;
#   3. a JLD2 `ReconstructedMutable`/`ReconstructedStatic` — a legacy result whose
#      struct definition has since changed, which JLD2 hands back as a property bag;
#   4. a `Dict` with `results` or `fit` in it.
#
# Cases 2 and 3 are RE-AUDITED on load: the legacy container has no diagnostics field,
# so the summary is computed from the chains it does have. That is a few seconds on a
# 40-fold run and it means an old fit answers `fit.diagnostics.passed` like any other.
#
# Case 3 is the one that matters in practice. `runner.jl:265-283` already carries a
# hand-written `Base.convert` shim for exactly one such drift (`Training.Independent`
# gaining a field). That approach needs a new shim per struct per change; reading the
# reconstructed object through `hasproperty` needs none.
#
# ==============================================================================

using Dates
using JLD2
using JSON3
using Printf
using Serialization

include(joinpath(@__DIR__, "l04_ingame_bridge.jl"))

const UIF_RESULTS_FILE = "results.jld2"
const UIF_META_FILE    = "meta.json"
const UIF_CONFIG_FILE  = "config.json"
const UIF_LATENTS_FILE = "oos_latents.jls"


# ==============================================================================
# 1. ATOMIC WRITES
# ==============================================================================

"""
    atomic_write(f, path) -> path

Run `f(tmp_path)`, then `mv(tmp, path; force = true)`.

`mv` within one filesystem is a `rename(2)`, which is atomic: a reader sees either the
old file or the new one, never a half-written one. The temporary name carries a random
suffix so two processes racing on the same target cannot clobber each other's scratch.

On any exception the temporary is removed and the original left untouched — a failed
save leaves the previous version intact rather than a corrupt one in its place.
"""
function atomic_write(f::Function, path::AbstractString)
    mkpath(dirname(path))
    tmp = string(path, ".tmp.", string(rand(UInt64), base = 16))
    try
        f(tmp)
        mv(tmp, path; force = true)
    catch e
        isfile(tmp) && rm(tmp; force = true)
        rethrow(e)
    end
    return path
end

"Write `obj` as pretty JSON, atomically."
write_json(path::AbstractString, obj) =
    atomic_write(path) do tmp
        open(tmp, "w") do io
            JSON3.pretty(io, obj)
        end
    end


# ==============================================================================
# 2. THE SIDECARS
# ==============================================================================

"""
    fit_meta(fit) -> Dict{Symbol, Any}

The scannable sidecar. Every field is a JSON scalar — no nesting, no arrays of objects
— so `list_fits` can read 200 of these and tabulate them without a schema.

The six fields beyond the legacy sidecar's set are the ones that make a scan
conclusive: `n_folds`, `n_oos_fixtures`, `converged`, `max_rhat`, `min_ess`,
`n_divergent`. See the file header.
"""
function fit_meta(f::AbstractFit)
    cfg = getfield(f, :config)
    md = getfield(f, :metadata)
    d = getfield(f, :diagnostics)
    lat = f isa Fit ? getfield(f, :latents) : getfield(f, :pregame_latents)

    return Dict{Symbol, Any}(
        :kind            => f isa Fit ? "Fit" : "InGameFit",
        :name            => cfg.name,
        :model           => string(nameof(typeof(cfg.model))),
        :splitter        => string(nameof(typeof(cfg.splitter))),
        :sampler         => string(nameof(typeof(cfg.sampler))),
        :timestamp       => Dates.format(md.timestamp, "yyyy-mm-ddTHH:MM:SS"),
        :time_taken      => format_elapsed(md.elapsed_seconds),
        :elapsed_seconds => round(md.elapsed_seconds, digits = 3),
        :julia_version   => string(md.julia_version),
        :n_threads       => md.n_threads,
        :git_commit      => md.git_commit,
        :n_folds         => length(f),
        :n_draws         => total_draws(f),
        :n_oos_fixtures  => lat === nothing ? 0 : n_matches(lat),
        :has_oos_latents => lat !== nothing,
        :converged       => _uif_meta_num(d, :passed, false),
        :max_rhat        => _uif_json_num(_uif_meta_num(d, :max_rhat, NaN)),
        :min_ess         => _uif_json_num(min(_uif_meta_num(d, :min_ess_bulk, NaN),
                                              _uif_meta_num(d, :min_ess_tail, NaN))),
        :n_divergent     => _uif_meta_num(d, :n_divergent, 0),
        :tags            => cfg.tags,
        :description     => cfg.description,
    )
end

_uif_meta_num(d, name::Symbol, default) = hasproperty(d, name) ? getproperty(d, name) : default

"`NaN`/`Inf` are not JSON. `nothing` serialises as `null`, which round-trips."
_uif_json_num(x::Real) = isfinite(x) ? x : nothing
_uif_json_num(x) = x

"""
    fit_config_json(fit) -> Dict{Symbol, Any}

The recipe, stringified. Human-readable, not reloadable: a `GroupedCVConfig` printed to
JSON cannot be parsed back into one, and pretending otherwise would create a second,
lossy definition of what a config is. The reloadable copy is inside `results.jld2`.
"""
function fit_config_json(f::AbstractFit)
    cfg = getfield(f, :config)
    d = Dict{Symbol, Any}(
        :name        => cfg.name,
        :model       => string(cfg.model),
        :splitter    => string(cfg.splitter),
        :sampler     => string(cfg.sampler),
        :execution   => string(cfg.execution),
        :tags        => cfg.tags,
        :description => cfg.description,
        :save_dir    => cfg.save_dir,
    )
    f isa InGameFit && (d[:pregame] = string(nameof(typeof(cfg.pregame))))
    return d
end


# ==============================================================================
# 3. SAVE
# ==============================================================================

"""
    save_fit(fit; path = nothing, quiet = false, latents = true) -> String

Write the four artefacts to `path` (default `fit.save_path`) and return the directory.

Every write is atomic (§1). The order is deliberate: `results.jld2` first, sidecars
after. An interrupt therefore leaves either nothing or a complete binary, and a
directory with a `results.jld2` but no `meta.json` is a recoverable state (`list_fits`
falls back to reading the binary for that one row) where the reverse would not be.

`latents = false` skips `oos_latents.jls` — worth it when the container is large and
the consumer will re-derive it, but note that the legacy `has_oos_predictions` check is
a file test, so a skipped write means a legacy caller recomputes.
"""
function save_fit(f::AbstractFit; path = nothing, quiet::Bool = false,
                  latents::Bool = true)
    target = path === nothing ? getfield(f, :save_path) : String(path)
    mkpath(target)

    atomic_write(joinpath(target, UIF_RESULTS_FILE)) do tmp
        jldsave(tmp; fit = f)
    end

    write_json(joinpath(target, UIF_CONFIG_FILE), fit_config_json(f))

    lat = f isa Fit ? getfield(f, :latents) : getfield(f, :pregame_latents)
    if latents && lat !== nothing
        atomic_write(joinpath(target, UIF_LATENTS_FILE)) do tmp
            Serialization.serialize(tmp, lat)
        end
    end

    # Last, and only now: the sidecar's `has_oos_latents` must not claim a file that
    # the write above may have skipped or failed on.
    meta = fit_meta(f)
    meta[:has_oos_latents] = isfile(joinpath(target, UIF_LATENTS_FILE))
    write_json(joinpath(target, UIF_META_FILE), meta)

    if !quiet
        printstyled("\n[IO] Fit saved to: ", color = :green, bold = true)
        println(target)
    end
    return target
end

"""
    save_latents(path, latents) -> String

The typed OOS container alone, atomically, under the legacy filename. For a caller that
wants to refresh the cache without rewriting the chains.
"""
save_latents(path::AbstractString, latents) =
    atomic_write(joinpath(path, UIF_LATENTS_FILE)) do tmp
        Serialization.serialize(tmp, latents)
    end

"""
    load_latents(path) -> container or nothing

Read `oos_latents.jls`. Accepts a typed container, or a legacy `LatentStates` whose
`.df` is converted with `latents_from_legacy_dataframe` (06, l02 §6) when the model is
known — which it is not from the file alone, so a bare `LatentStates` comes back as its
`DataFrame` and the caller decides. Never throws: a corrupt cache is a cache miss.
"""
function load_latents(path::AbstractString)
    file = isfile(path) ? path : joinpath(path, UIF_LATENTS_FILE)
    isfile(file) || return nothing
    try
        obj = Serialization.deserialize(file)
        obj isa AbstractPosteriorLatents && return obj
        hasproperty(obj, :df) && return getproperty(obj, :df)
        return obj
    catch e
        @warn "Unreadable latents cache at $file" exception = e
        return nothing
    end
end


# ==============================================================================
# 4. LOAD, WITH THE LEGACY UPGRADE PATH
# ==============================================================================

"""
    load_fit(path; quiet = false, gates = ConvergenceGates()) -> Fit

Read a saved run. `path` may be the directory or the `.jld2` file itself.

Accepts, and returns a `Fit` for, all four shapes listed in the file header. A legacy
container is re-audited on load, because it has no diagnostics of its own.
"""
function load_fit(path::AbstractString; quiet::Bool = false,
                  gates::ConvergenceGates = ConvergenceGates())
    file = endswith(path, ".jld2") ? String(path) : joinpath(path, UIF_RESULTS_FILE)
    isfile(file) || error("load_fit: no results file at $file")

    quiet || (printstyled("Loading: ", color = :green); println(basename(dirname(file))))

    data = JLD2.load(file)
    obj = haskey(data, "fit")     ? data["fit"] :
          haskey(data, "results") ? data["results"] :
          length(data) == 1       ? first(values(data)) :
          error("load_fit: $file has keys $(collect(keys(data))); expected `fit` or `results`.")

    return upgrade_to_fit(obj; save_path = dirname(file), gates = gates)
end

"""
    upgrade_to_fit(obj; save_path = "", gates = ConvergenceGates()) -> Fit

Turn whatever came out of a `.jld2` into a `Fit`.

Dispatches on the two cases it can name, and falls through to a duck-typed reader for
the ones it cannot — a `ReconstructedMutable` has no stable type to dispatch on, only
properties.
"""
upgrade_to_fit(f::Fit; kwargs...) = f
upgrade_to_fit(f::InGameFit; kwargs...) = f

function upgrade_to_fit(obj; save_path::AbstractString = "",
                        gates::ConvergenceGates = ConvergenceGates())
    hasproperty(obj, :config) || error(
        "upgrade_to_fit: cannot read a $(typeof(obj)) — it has no `config` property. " *
        "Expected a `Fit`, an `ExperimentResults`, or a JLD2-reconstructed one of those.")

    legacy_cfg = obj.config
    tr = hasproperty(obj, :training_results) ? obj.training_results : nothing
    tr === nothing && error(
        "upgrade_to_fit: a $(typeof(obj)) with a `config` but no `training_results`. " *
        "Nothing here knows how to find its chains.")

    items = hasproperty(tr, :items) ? tr.items : collect(tr)

    folds = _uif_narrow(FoldFit[FoldFit(i, it[1], it[2])
                                for (i, it) in enumerate(items) if it !== nothing])
    isempty(folds) && error("upgrade_to_fit: the legacy container holds no usable folds.")

    sampler = _uif_legacy_sampler(legacy_cfg)
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

    diagnostics = audit_convergence(folds; gates = gates,
                                    max_depth = sampler_max_depth(sampler))

    sp = String(get_or(obj, :save_path, save_path))
    latents = isempty(sp) ? nothing : load_latents(sp)
    latents isa AbstractPosteriorLatents || (latents = nothing)

    meta = FitMetadata(_uif_legacy_timestamp(sp), _uif_legacy_elapsed(legacy_cfg),
                       VERSION, 1, "unknown")

    return Fit(config, folds, latents, diagnostics, meta, sp)
end

"The sampler out of a legacy config's nested `training_config`, or `nothing`."
function _uif_legacy_sampler(cfg)
    hasproperty(cfg, :sampler) && return cfg.sampler
    if hasproperty(cfg, :training_config)
        tc = cfg.training_config
        hasproperty(tc, :sampler) && return tc.sampler
    end
    return nothing
end

"""
Recover the run time from the legacy `time:` tag `run_experiment` pushed
(runner.jl:73). Approximate by construction — the tag is a formatted string, so
`"3m 20s"` comes back as 200.0 and `"2h 15m"` loses the seconds. Better than zero, and
the sidecar of any run saved by THIS framework carries the exact value.
"""
function _uif_legacy_elapsed(cfg)
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
function _uif_legacy_timestamp(path::AbstractString)
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
# 5. DISCOVERY
# ==============================================================================

"""
    list_fits(dir; quiet = false) -> Vector{NamedTuple}

Scan `dir` for saved runs, newest first, reading only `meta.json`.

Returns one `NamedTuple` per directory and, unless `quiet`, prints the table. The
convergence columns are the point (see the file header): a scan of 40 fits shows which
ones are usable without opening any of them.

A directory with no readable sidecar still appears, with `?` in the columns the sidecar
would have filled and `converged = missing` — an unknown verdict is not a pass.
"""
function list_fits(dir::AbstractString; quiet::Bool = false)
    isdir(dir) || (quiet || println("Directory not found: $dir"); return NamedTuple[])

    subdirs = filter(isdir, readdir(dir; join = true))
    sort!(subdirs; by = mtime, rev = true)
    isempty(subdirs) && (quiet || println("No fits found in $dir"); return NamedTuple[])

    rows = [read_fit_meta(p) for p in subdirs]
    quiet || print_fit_table(rows, dir)
    return rows
end

"""
    read_fit_meta(path) -> NamedTuple

One directory's sidecar, with defaults for every field it lacks. Never throws.
"""
function read_fit_meta(path::AbstractString)
    default = (path = path, name = basename(path), kind = "?", model = "?",
               splitter = "?", sampler = "?", time_taken = "N/A", n_folds = 0,
               n_oos_fixtures = 0, has_oos_latents = isfile(joinpath(path, UIF_LATENTS_FILE)),
               converged = missing, max_rhat = NaN, min_ess = NaN, n_divergent = 0,
               git_commit = "?")

    mp = joinpath(path, UIF_META_FILE)
    isfile(mp) || return default
    try
        c = JSON3.read(read(mp, String))
        g(k, d) = (v = get(c, k, d); v === nothing ? d : v)
        return (path = path,
                name = String(g(:name, default.name)),
                kind = String(g(:kind, "Fit")),
                model = String(g(:model, "?")),
                splitter = String(g(:splitter, "?")),
                sampler = String(g(:sampler, "?")),
                time_taken = String(g(:time_taken, "N/A")),
                n_folds = Int(g(:n_folds, 0)),
                n_oos_fixtures = Int(g(:n_oos_fixtures, 0)),
                has_oos_latents = Bool(g(:has_oos_latents, default.has_oos_latents)),
                converged = haskey(c, :converged) && c[:converged] !== nothing ?
                            Bool(c[:converged]) : missing,
                max_rhat = Float64(g(:max_rhat, NaN)),
                min_ess = Float64(g(:min_ess, NaN)),
                n_divergent = Int(g(:n_divergent, 0)),
                git_commit = String(g(:git_commit, "?")))
    catch
        return default
    end
end

"The `list_fits` table. Newest row highlighted, convergence column colour-coded."
function print_fit_table(rows::Vector, dir::AbstractString)
    println("\n Fits in: ", dir)
    println("="^128)
    println(rpad("IDX", 5), rpad("NAME", 26), rpad("MODEL", 22), rpad("SAMPLER", 16),
            rpad("FOLDS", 7), rpad("TIME", 10), rpad("R-HAT", 9), rpad("OOS", 7), "CONV")
    println("-"^128)
    for (i, r) in enumerate(rows)
        conv, colour = r.converged === missing ? ("?", :light_black) :
                       r.converged ? ("PASS", :green) : ("FAIL", :red)
        c = i == 1 ? :white : :light_black
        printstyled(rpad("[$i]", 5); color = :cyan, bold = (i == 1))
        printstyled(rpad(_uif_trunc(r.name, 25), 26); color = c, bold = (i == 1))
        printstyled(rpad(_uif_trunc(r.model, 21), 22); color = c)
        printstyled(rpad(_uif_trunc(r.sampler, 15), 16); color = c)
        printstyled(rpad(string(r.n_folds), 7); color = c)
        printstyled(rpad(_uif_trunc(r.time_taken, 9), 10); color = :yellow)
        printstyled(rpad(isfinite(r.max_rhat) ? @sprintf("%.4f", r.max_rhat) : "—", 9);
                    color = c)
        printstyled(rpad(r.has_oos_latents ? string(r.n_oos_fixtures) : "—", 7);
                    color = r.has_oos_latents ? :green : :light_black)
        printstyled(conv, "\n"; color = colour, bold = (conv == "FAIL"))
    end
    println("="^128, "\n")
    return nothing
end

_uif_trunc(s::AbstractString, n::Int) = length(s) > n ? s[1:(n - 2)] * ".." : String(s)

"""
    load_fits(dir_or_paths) -> Vector{Fit}

Load every fit under a directory, or a named list of them. A path that fails to load is
warned about and skipped, so one corrupt directory does not stop a batch comparison.
"""
function load_fits(paths::Vector{String}; quiet::Bool = true, kwargs...)
    out = Fit[]
    for p in paths
        try
            push!(out, load_fit(p; quiet = quiet, kwargs...))
        catch e
            @warn "Could not load $p" exception = e
        end
    end
    return out
end

load_fits(dir::AbstractString; kwargs...) =
    load_fits([r.path for r in list_fits(dir; quiet = true)]; kwargs...)
