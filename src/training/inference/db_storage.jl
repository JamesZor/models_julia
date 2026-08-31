# PostgreSQL storage for the unified inference lifecycle.
#
# Queryable run/fold/latent summaries are stored relationally. An exact, Zstd-compressed
# Serialization artefact is stored alongside them because arbitrary model, splitter and chain
# types cannot be reconstructed from human-readable JSON without a second lossy type registry.

const _DB_COUNT_MAGIC = UInt8[0x42, 0x46, 0x43, 0x4c] # "BFCL"
const _DB_COUNT_VERSION = UInt8(1)

"Open a PostgreSQL connection. Kept behind one function so tests can isolate connectivity."
_db_connect(storage::PostgresStorage) = LibPQ.Connection(storage.conn_str)

function _db_exec(conn::LibPQ.Connection, sql::AbstractString, params = ())
    result = LibPQ.execute(conn, sql, params)
    try
        return try
            LibPQ.num_affected_rows(result)
        catch e
            e isa ArgumentError || rethrow()
            0
        end
    finally
        close(result)
    end
end

function _db_rows(conn::LibPQ.Connection, sql::AbstractString, params = ())
    result = LibPQ.execute(conn, sql, params)
    try
        return DataFrame(result)
    finally
        close(result)
    end
end

"PostgreSQL text representation for a `bytea` query parameter."
_db_bytea(bytes::AbstractVector{UInt8}) = "\\x" * bytes2hex(bytes)

"Apply the idempotent experiment schema to `storage`."
function ensure_schema!(storage::PostgresStorage)
    schema_path = joinpath(@__DIR__, "db", "schema.sql")
    conn = _db_connect(storage)
    try
        for statement in split(read(schema_path, String), ';')
            isempty(strip(statement)) || _db_exec(conn, statement)
        end
    finally
        close(conn)
    end
    return storage
end

"A stable JSON-safe description; the exact reloadable object lives in `fit_artifacts`."
function _db_config_description(value)
    fields = Dict{String, Any}()
    for name in fieldnames(typeof(value))
        v = getfield(value, name)
        fields[string(name)] = v isa Union{Nothing, Bool, Number, AbstractString, Symbol} ?
                               (v isa Symbol ? string(v) : v) : string(v)
    end
    return Dict("type" => string(typeof(value)), "display" => string(value), "fields" => fields)
end

function _db_config_payload(fit::Fit, experiment_name::AbstractString)
    cfg = getfield(fit, :config)
    return Dict(
        "experiment_name" => String(experiment_name),
        "name" => cfg.name,
        "model" => _db_config_description(cfg.model),
        "splitter" => _db_config_description(cfg.splitter),
        "sampler" => _db_config_description(cfg.sampler),
        "execution" => _db_config_description(cfg.execution),
        "tags" => cfg.tags,
        "description" => cfg.description,
    )
end

_db_recipe_tags(tags) = filter(tags) do tag
    !any(prefix -> startswith(tag, prefix), ("time:", "folds_failed:", "latents:"))
end

"SHA-256 identity of an experiment name and its inference recipe (not run telemetry)."
function config_hash(fit::Fit, storage::PostgresStorage)
    canonical = join((storage.experiment_name, fit.config.name,
                      string(fit.config.model), string(fit.config.splitter),
                      string(fit.config.sampler), string(fit.config.execution),
                      join(_db_recipe_tags(fit.config.tags), "\u001f"),
                      fit.config.description), "\u001e")
    return bytes2hex(SHA.sha256(canonical))
end

function _db_git_branch()
    try
        return readchomp(`git rev-parse --abbrev-ref HEAD`)
    catch
        return "unknown"
    end
end

"Zstd-compress one match's home/away count draws, including optional NegBin dispersions."
function compress_draws(lambda_home::AbstractVector{<:Real},
                        lambda_away::AbstractVector{<:Real}, observation_params = nothing)
    n = length(lambda_home)
    length(lambda_away) == n || error(
        "compress_draws: home has $n draws but away has $(length(lambda_away)).")
    has_obs = observation_params !== nothing
    if has_obs
        keys(observation_params) == (:r_h, :r_a) || error(
            "compress_draws: observation parameters must be `(; r_h, r_a)`.")
        length(observation_params.r_h) == n && length(observation_params.r_a) == n || error(
            "compress_draws: observation draw counts must both equal $n.")
    end

    io = IOBuffer()
    write(io, _DB_COUNT_MAGIC)
    write(io, _DB_COUNT_VERSION)
    write(io, has_obs ? UInt8(1) : UInt8(0))
    write(io, UInt32(n))
    write(io, Float64.(lambda_home))
    write(io, Float64.(lambda_away))
    if has_obs
        write(io, Float64.(observation_params.r_h))
        write(io, Float64.(observation_params.r_a))
    end
    return transcode(ZstdCompressor, take!(io))
end

"Inverse of [`compress_draws`](@ref)."
function decompress_draws(blob::AbstractVector{UInt8})
    raw = transcode(ZstdDecompressor, Vector{UInt8}(blob))
    io = IOBuffer(raw)
    read(io, 4) == _DB_COUNT_MAGIC || error("decompress_draws: invalid count-draw magic bytes.")
    read(io, UInt8) == _DB_COUNT_VERSION || error(
        "decompress_draws: unsupported count-draw format version.")
    has_obs = read(io, UInt8) == UInt8(1)
    n = Int(read(io, UInt32))
    lambda_home = read!(io, Vector{Float64}(undef, n))
    lambda_away = read!(io, Vector{Float64}(undef, n))
    observation_params = if has_obs
        (; r_h = read!(io, Vector{Float64}(undef, n)),
           r_a = read!(io, Vector{Float64}(undef, n)))
    else
        nothing
    end
    eof(io) || error("decompress_draws: trailing bytes after $n draws.")
    return (; lambda_home, lambda_away, observation_params)
end

function _db_artifact_blob(value)
    io = IOBuffer()
    Serialization.serialize(io, value)
    return transcode(ZstdCompressor, take!(io))
end

_db_artifact_value(blob::AbstractVector{UInt8}) =
    Serialization.deserialize(IOBuffer(transcode(ZstdDecompressor, Vector{UInt8}(blob))))


# ==============================================================================
# CONFIGURATION SINGLE SOURCE OF TRUTH
# ==============================================================================

function _truth_config_value(config)
    if nameof(typeof(config)) === :PortfolioSystem &&
       hasproperty(config, :book) && hasproperty(config, :policy)
        return (getproperty(config, :book), getproperty(config, :policy))
    elseif config isa NamedTuple && keys(config) == (:book, :policy)
        return (config.book, config.policy)
    end
    return config
end

function _truth_config_type(config)
    config isa FitConfig && return "fit"
    if config isa Tuple && length(config) == 2 &&
       nameof(typeof(config[1])) === :BookSpec && nameof(typeof(config[2])) === :PolicySpec
        return "portfolio"
    end
    return string(nameof(typeof(config)))
end

function _truth_config_canonical(config::FitConfig)
    return join((config.name, string(config.model), string(config.splitter),
                 string(config.sampler), string(config.execution),
                 join(_db_recipe_tags(config.tags), "\u001f"),
                 config.description, config.save_dir), "\u001e")
end

_truth_config_canonical(config::Tuple) = join(string.(config), "\u001e")
_truth_config_canonical(config) = string(config)

function _truth_config_hash(config)
    kind = _truth_config_type(config)
    return bytes2hex(SHA.sha256(kind * "\u001d" * _truth_config_canonical(config)))
end

function _truth_config_json(config::FitConfig)
    return Dict(
        "type" => "fit",
        "name" => config.name,
        "model" => _db_config_description(config.model),
        "splitter" => _db_config_description(config.splitter),
        "sampler" => _db_config_description(config.sampler),
        "execution" => _db_config_description(config.execution),
        "tags" => config.tags,
        "description" => config.description,
        "save_dir" => config.save_dir,
    )
end

function _truth_config_json(config::Tuple)
    return Dict("type" => "portfolio",
                "book_spec" => _db_config_description(config[1]),
                "policy_spec" => _db_config_description(config[2]))
end

_truth_config_json(config) = Dict("type" => _truth_config_type(config),
                                  "value" => _db_config_description(config))

"""
    save_config(db::PostgresStorage, name::String, config; description = "", tags = [])

Save or replace one named recipe in the PostgreSQL configuration registry and return its
SHA-256 hash. `config` may be a `FitConfig`, `(BookSpec, PolicySpec)`, or `PortfolioSystem`.
Names are unique within `db.experiment_name`; hashes identify equivalent recipe values.
"""
function save_config(db::PostgresStorage, name::String, config;
                     description::AbstractString = "", tags = [])
    isempty(strip(name)) && error("save_config: name must not be empty.")
    value = _truth_config_value(config)
    kind = _truth_config_type(value)
    tag_strings = String[string(tag) for tag in tags]
    hash = _truth_config_hash(value)
    summary_json = JSON3.write(_truth_config_json(value))
    tags_json = JSON3.write(tag_strings)
    blob = _db_artifact_blob(value)
    timestamp = now()

    conn = _db_connect(db)
    try
        _db_exec(conn, """
            INSERT INTO config_registry (
                registry_id, experiment_name, name, config_hash, config_type, config_json,
                config_blob, description, tags, created_at, updated_at
            ) VALUES (\$1::uuid, \$2, \$3, \$4, \$5, \$6::jsonb, \$7::bytea,
                      \$8, \$9::jsonb, \$10, \$10)
            ON CONFLICT (experiment_name, name) DO UPDATE SET
                config_hash = EXCLUDED.config_hash,
                config_type = EXCLUDED.config_type,
                config_json = EXCLUDED.config_json,
                config_blob = EXCLUDED.config_blob,
                description = EXCLUDED.description,
                tags = EXCLUDED.tags,
                updated_at = EXCLUDED.updated_at;
        """, (string(uuid4()), db.experiment_name, name, hash, kind, summary_json,
              _db_bytea(blob), String(description), tags_json, timestamp))
    finally
        close(conn)
    end
    return hash
end

save_config(db::PostgresStorage, name::AbstractString, config; kwargs...) =
    save_config(db, String(name), config; kwargs...)

function _load_truth_config(db::PostgresStorage, name_or_hash::String)
    conn = _db_connect(db)
    try
        rows = _db_rows(conn, """
            SELECT config_blob FROM config_registry
            WHERE experiment_name = \$1 AND (name = \$2 OR config_hash = \$2)
            ORDER BY CASE WHEN name = \$2 THEN 0 ELSE 1 END, updated_at DESC
            LIMIT 1;
        """, (db.experiment_name, name_or_hash))
        nrow(rows) == 1 || error(
            "No saved config named or hashed '$name_or_hash' in experiment '$(db.experiment_name)'.")
        return _db_artifact_value(rows.config_blob[1])
    finally
        close(conn)
    end
end

"Load a named/hashed exact `FitConfig` from the configuration registry."
function load_fit_config(db::PostgresStorage, name_or_hash::String)
    config = _load_truth_config(db, name_or_hash)
    config isa FitConfig || error(
        "load_fit_config: '$name_or_hash' holds $(typeof(config)), not a FitConfig.")
    return config
end

load_fit_config(db::PostgresStorage, name_or_hash::AbstractString) =
    load_fit_config(db, String(name_or_hash))

"Load a named/hashed `(BookSpec, PolicySpec)` tuple from the configuration registry."
function load_portfolio_spec(db::PostgresStorage, name_or_hash::String)
    config = _load_truth_config(db, name_or_hash)
    valid = config isa Tuple && length(config) == 2 &&
            nameof(typeof(config[1])) === :BookSpec && nameof(typeof(config[2])) === :PolicySpec
    valid || error(
        "load_portfolio_spec: '$name_or_hash' holds $(typeof(config)), not " *
        "(BookSpec, PolicySpec).")
    return config
end

load_portfolio_spec(db::PostgresStorage, name_or_hash::AbstractString) =
    load_portfolio_spec(db, String(name_or_hash))

"""
    list_configs(db::PostgresStorage; tag = nothing, config_type = nothing) -> DataFrame

List saved recipes newest first. `tag` requires exact membership in the JSON tag array;
`config_type` accepts `"fit"`, `"portfolio"`, or another registry type name.
"""
function list_configs(db::PostgresStorage; tag = nothing, config_type = nothing)
    tag_value = tag === nothing ? missing : string(tag)
    type_value = config_type === nothing ? missing : string(config_type)
    conn = _db_connect(db)
    try
        rows = _db_rows(conn, """
            SELECT name, config_hash, config_type, description, tags, created_at, updated_at
            FROM config_registry
            WHERE experiment_name = \$1
              AND (\$2::varchar IS NULL OR tags ? \$2)
              AND (\$3::varchar IS NULL OR config_type = \$3)
            ORDER BY updated_at DESC, name;
        """, (db.experiment_name, tag_value, type_value))
        if :tags in propertynames(rows)
            rows.tags = [String[string(x) for x in JSON3.read(String(value))]
                         for value in rows.tags]
        end
        return rows
    finally
        close(conn)
    end
end

_db_nullable(x::Real) = isfinite(x) ? x : missing
_db_nullable(x) = x
_db_nullable_int(x::Real) = isfinite(x) ? round(Int, x) : missing

function _db_fold_diagnostic(fit::Fit, fold_index::Int)
    diagnostics = getfield(fit, :diagnostics).folds
    i = findfirst(d -> d.fold == fold_index, diagnostics)
    return i === nothing ? nothing : diagnostics[i]
end

function _db_insert_latents!(conn::LibPQ.Connection, fold_id::UUID, latents::CountLatents)
    for i in eachindex(latents.match_ids)
        home = Vector{Float64}(@view latents.λ_home[i, :])
        away = Vector{Float64}(@view latents.λ_away[i, :])
        obs = latents.observation_params === nothing ? nothing :
              (; r_h = Vector{Float64}(@view(latents.observation_params.r_h[i, :])),
                 r_a = Vector{Float64}(@view(latents.observation_params.r_a[i, :])))
        blob = compress_draws(home, away, obs)
        _db_exec(conn, """
            INSERT INTO match_latents (
                fold_id, match_id,
                mean_lambda_h, std_lambda_h, p10_h, p50_h, p90_h,
                mean_lambda_a, std_lambda_a, p10_a, p50_a, p90_a, draws_blob
            ) VALUES (
                \$1::uuid, \$2, \$3, \$4, \$5, \$6, \$7,
                \$8, \$9, \$10, \$11, \$12, \$13::bytea
            );
        """, (string(fold_id), latents.match_ids[i],
              mean(home), std(home), quantile(home, 0.10), quantile(home, 0.50),
              quantile(home, 0.90), mean(away), std(away), quantile(away, 0.10),
              quantile(away, 0.50), quantile(away, 0.90), _db_bytea(blob)))
    end
    return nothing
end

"Persist a fit and return its run UUID. Identical config hashes are idempotent."
function save_fit(fit::Fit, storage::PostgresStorage)
    latents = getfield(fit, :latents)
    latents === nothing || latents isa CountLatents || error(
        "PostgresStorage currently stores CountLatents; got $(typeof(latents)). " *
        "Use FileStorage for this latent family.")

    hash = config_hash(fit, storage)
    conn = _db_connect(storage)
    try
        existing = _db_rows(conn,
            "SELECT config_id FROM configs WHERE config_hash = \$1 LIMIT 1;", (hash,))
        if nrow(existing) == 1
            return UUID(string(existing.config_id[1]))
        end

        run_id = uuid4()
        fold_ids = UUID[uuid4() for _ in fit.folds]
        metadata = getfield(fit, :metadata)
        created_at = metadata.timestamp - Millisecond(round(Int, 1000 * metadata.elapsed_seconds))
        payload = _db_config_payload(fit, storage.experiment_name)
        model_json = JSON3.write(payload["model"])
        split_json = JSON3.write(payload["splitter"])
        sampler_json = JSON3.write(Dict(
            "sampler" => payload["sampler"], "execution" => payload["execution"],
            "name" => payload["name"], "tags" => payload["tags"],
            "description" => payload["description"]))

        _db_exec(conn, "BEGIN;")
        try
            _db_exec(conn, """
                INSERT INTO runs (run_id, experiment_name, status, git_commit, git_branch,
                                  created_at, finished_at, duration_seconds)
                VALUES (\$1::uuid, \$2, \$3, \$4, \$5, \$6, \$7, \$8);
            """, (string(run_id), storage.experiment_name, "completed", metadata.git_commit,
                  _db_git_branch(), created_at, metadata.timestamp, metadata.elapsed_seconds))
            _db_exec(conn, """
                INSERT INTO configs (config_id, config_hash, model_config, split_config,
                                     sampler_config)
                VALUES (\$1::uuid, \$2, \$3::jsonb, \$4::jsonb, \$5::jsonb);
            """, (string(run_id), hash, model_json, split_json, sampler_json))

            for (i, fold) in enumerate(fit.folds)
                diagnostic = _db_fold_diagnostic(fit, fold.fold)
                rhat = diagnostic === nothing ? missing : _db_nullable(diagnostic.max_rhat)
                ess_bulk = diagnostic === nothing ? missing :
                           _db_nullable_int(diagnostic.min_ess_bulk)
                ess_tail = diagnostic === nothing ? missing :
                           _db_nullable_int(diagnostic.min_ess_tail)
                divergences = diagnostic === nothing ? 0 : diagnostic.n_divergent
                converged = diagnostic === nothing ? false :
                    summarise_convergence([diagnostic];
                        thresholds = fit.diagnostics.thresholds).passed
                runtime = metadata.elapsed_seconds / max(length(fit), 1)
                _db_exec(conn, """
                    INSERT INTO fold_results (
                        fold_id, run_id, fold_idx, r_hat_max, ess_bulk_min, ess_tail_min,
                        divergences, converged, logloss, brier, rps, runtime_seconds
                    ) VALUES (\$1::uuid, \$2::uuid, \$3, \$4, \$5, \$6, \$7, \$8,
                              NULL, NULL, NULL, \$9);
                """, (string(fold_ids[i]), string(run_id), fold.fold, rhat, ess_bulk,
                      ess_tail, divergences, converged, runtime))
            end

            # Fit.latents is a run-level merged panel and no longer carries its source fold IDs.
            # Associate that panel with the first fold for FK ownership; row order remains exact.
            if latents isa CountLatents
                isempty(fold_ids) && error("PostgresStorage: CountLatents require at least one fold.")
                _db_insert_latents!(conn, first(fold_ids), latents)
            end

            artifact = _db_artifact_blob(fit)
            _db_exec(conn,
                "INSERT INTO fit_artifacts (run_id, fit_blob) VALUES (\$1::uuid, \$2::bytea);",
                (string(run_id), _db_bytea(artifact)))
            _db_exec(conn, "COMMIT;")
        catch
            try
                _db_exec(conn, "ROLLBACK;")
            catch
            end
            rethrow()
        end
        return run_id
    finally
        close(conn)
    end
end

"Write both backends. The return names both independently addressable artefacts."
function save_fit(fit::Fit, storage::DualStorage; kwargs...)
    path = save_fit(fit, storage.file; kwargs...)
    run_id = save_fit(fit, storage.db)
    return (; path, run_id)
end

function _db_load_count_latents(conn::LibPQ.Connection, run_id::UUID)
    rows = _db_rows(conn, """
        SELECT ml.match_id, ml.draws_blob
        FROM match_latents ml
        JOIN fold_results fr ON fr.fold_id = ml.fold_id
        WHERE fr.run_id = \$1::uuid
        ORDER BY ml.latent_id;
    """, (string(run_id),))
    nrow(rows) == 0 && return nothing

    decoded = [decompress_draws(rows.draws_blob[i]) for i in 1:nrow(rows)]
    n = length(decoded[1].lambda_home)
    all(d -> length(d.lambda_home) == n && length(d.lambda_away) == n, decoded) || error(
        "load_fit: run $run_id has inconsistent latent draw counts.")
    lambda_home = reduce(vcat, [permutedims(d.lambda_home) for d in decoded])
    lambda_away = reduce(vcat, [permutedims(d.lambda_away) for d in decoded])
    any_obs = any(d -> d.observation_params !== nothing, decoded)
    all_obs = all(d -> d.observation_params !== nothing, decoded)
    any_obs == all_obs || error("load_fit: run $run_id mixes Poisson and NegBin latent rows.")
    if all_obs
        r_h = reduce(vcat, [permutedims(d.observation_params.r_h) for d in decoded])
        r_a = reduce(vcat, [permutedims(d.observation_params.r_a) for d in decoded])
        return CountLatents(Int.(rows.match_id), lambda_home, lambda_away, (; r_h, r_a))
    end
    return CountLatents(Int.(rows.match_id), lambda_home, lambda_away)
end

"Load an exact `Fit`, replacing its latent panel with the relationally reconstructed copy."
function load_fit(run_id::UUID, storage::PostgresStorage)
    conn = _db_connect(storage)
    try
        rows = _db_rows(conn,
            "SELECT fit_blob FROM fit_artifacts WHERE run_id = \$1::uuid;", (string(run_id),))
        nrow(rows) == 1 || error("load_fit: no PostgreSQL fit artefact for run $run_id.")
        fit = _db_artifact_value(rows.fit_blob[1])
        fit isa Fit || error("load_fit: PostgreSQL artefact for $run_id holds $(typeof(fit)).")
        latents = _db_load_count_latents(conn, run_id)
        latents === nothing && return fit
        return Fit(fit.config, fit.folds, latents, fit.diagnostics, fit.metadata, fit.save_path)
    finally
        close(conn)
    end
end

load_fit(run_id::AbstractString, storage::PostgresStorage) = load_fit(UUID(run_id), storage)
