# ==============================================================================
# src/Calibration/db_storage.jl — the calibrator tier in `mcmc_experiments`
# ==============================================================================
#
# WHICH DATABASE. `mcmc_experiments` on `mcmc-beast:5432`, reached only as
# `Training.PostgresStorage(experiment_name)`, which resolves `BF_EXPERIMENTS_DB_URL` or
# lets libpq read `~/.pgpass`. Nothing here opens `betdb`, and nothing here writes a
# paper-trading row. See AGENTS.md §3 for why those are two different databases.
#
# WHAT IT STORES, AND WHY IN TWO TABLES.
#
#   calibration_runs       one queryable row per (model run x calibrator): headline proper
#                          scores, CLV, coverage, and the recipe as JSONB
#   calibration_artifacts  the exact serialized calibrator, the calibrated container, and
#                          the per-fixture diagnostic frame
#
# The split is the one `runs` / `fit_artifacts` already makes and for the same reason: a
# SQL row answers "which calibrator scored best across the grid" without deserialising
# anything, and a blob answers "give me back exactly what ran" without a schema migration
# every time a field is added.
#
# `diagnostics_blob` is not decoration. The per-fixture delta / w / kappa frame is what a
# post-mortem reads, and it is the ONE thing that does not reconstruct from the calibrator
# plus the book — it depends on the posterior, which lives in another table under another
# run's key.
#
# THE LINK TO A PORTFOLIO RUN is through `portfolio_runs.metadata`, not a foreign key.
# `Portfolio.save_portfolio_db` already merges a caller's metadata into that JSONB column,
# so `link_portfolio_run` needs no change to `src/Portfolio/` — and `portfolio_runs` must
# stay insertable without a calibration tier, which a NOT NULL foreign key would prevent.
# ==============================================================================

const _CAL_LINK_KEY = "calibration_run_id"

"""
    save_calibration_db(cf, model_run_id, storage; scores, clv, metadata,
                        store_latents = true) -> UUID

Persist a calibration run and return its `calibration_run_id`.

`model_run_id` is the UUID `save_fit` returned for the RAW inference run — the posterior
this calibration was applied to. It is a real foreign key: a calibration of a run that was
deleted is not a thing.

| keyword | is |
|---|---|
| `scores` | a NamedTuple from [`calibration_scores`](@ref), or `nothing`. Headline scope only |
| `clv` | a NamedTuple from [`clv_summary`](@ref), or `nothing` |
| `metadata` | anything else — the wide-book scores, the fixture set, the runner that produced it |
| `store_latents` | write the calibrated container to `calibration_artifacts`. `false` when it is large and reproducible from the calibrator plus the book |

# What goes in which column

`log_loss` / `ece` / `brier` are **headline scope** (1X2 + O/U 2.5 + BTTS), because that
is the only scope in which the published Gate-1 thresholds mean anything. A wide-book
score under the same column name would make two rows of a leaderboard incomparable, so
put it in `metadata` under its own key.

The whole write is ONE transaction. A `calibration_runs` row without its artefact is a
run you cannot reproduce, and the failure mode of finding one three weeks later is worse
than the failure mode of the insert rolling back now.
"""
function save_calibration_db(cf::CalibratedFit, model_run_id::UUID,
                             storage::Training.PostgresStorage;
                             scores = nothing,
                             clv = nothing,
                             metadata = NamedTuple(),
                             store_latents::Bool = true)
    cal = cf.calibrator
    calibration_run_id = uuid4()
    cov = cf.coverage

    base = Dict{String, Any}(
        "calibrator_label" => calibrator_label(cal),
        "fit_name" => Training.fit_name(cf.fit),
        "converged" => cf.fit.diagnostics.passed,
        "coverage" => cov.coverage,
        "coverage_quoted" => cov.coverage_quoted,
        "n_quoted" => cov.n_quoted,
        "n_refused" => cov.n_refused,
        "n_absent" => cov.n_absent,
        "refusals" => Dict{String, Any}(r => n for (r, n) in inversion_refusals(cf.market_rates)),
        "weights" => _nt_to_dict(weight_summary(cf.rate_diagnostics)),
        "dispersion" => _nt_to_dict(dispersion_summary(cf.rate_diagnostics)),
        "created_at" => string(cf.created_at),
    )
    scores === nothing || (base["scores"] = _nt_to_dict(scores))
    clv === nothing || (base["clv"] = _nt_to_dict(clv))
    for (k, v) in pairs(metadata)
        base[string(k)] = v
    end

    conn = Training.Inference._db_connect(storage)
    try
        Training.Inference._db_exec(conn, "BEGIN;")
        try
            Training.Inference._db_exec(conn, """
                INSERT INTO calibration_runs (
                    calibration_run_id, model_run_id, experiment_name, calibrator_name,
                    calibrator_hash, config_json, book_as_of_minutes, n_fixtures,
                    n_inverted, log_loss, ece, brier, clv_mean_pct, clv_weighted_pct,
                    created_at, metadata
                ) VALUES (\$1::uuid, \$2::uuid, \$3, \$4, \$5, \$6::jsonb, \$7, \$8, \$9,
                          \$10, \$11, \$12, \$13, \$14, \$15, \$16::jsonb);
            """, (string(calibration_run_id), string(model_run_id),
                  storage.experiment_name, cal.name, calibrator_hash(cal),
                  JSON3.write(calibrator_json(cal)),
                  cf.book_as_of_minutes, cov.n_fixtures, cov.n_accepted,
                  _cal_metric(scores, :logloss), _cal_metric(scores, :ece),
                  _cal_metric(scores, :brier),
                  _cal_metric(clv, :mean_clv_pct), _cal_metric(clv, :stake_weighted_clv_pct),
                  now(), JSON3.write(base)))

            cal_blob = Training.Inference._db_artifact_blob(cal)
            lat_blob = store_latents ?
                Training.Inference._db_bytea(
                    Training.Inference._db_artifact_blob(cf.latents)) : missing
            diag_blob = Training.Inference._db_bytea(
                Training.Inference._db_artifact_blob(cf.rate_diagnostics))
            Training.Inference._db_exec(conn, """
                INSERT INTO calibration_artifacts (
                    calibration_run_id, calibrator_blob, calibrated_latents_blob,
                    diagnostics_blob
                ) VALUES (\$1::uuid, \$2::bytea, \$3::bytea, \$4::bytea);
            """, (string(calibration_run_id),
                  Training.Inference._db_bytea(cal_blob), lat_blob, diag_blob))

            Training.Inference._db_exec(conn, "COMMIT;")
        catch
            try
                Training.Inference._db_exec(conn, "ROLLBACK;")
            catch
            end
            rethrow()
        end
        return calibration_run_id
    finally
        close(conn)
    end
end

save_calibration_db(cf::CalibratedFit, model_run_id::AbstractString,
                    storage::Training.PostgresStorage; kwargs...) =
    save_calibration_db(cf, UUID(model_run_id), storage; kwargs...)

_nt_to_dict(nt) = Dict{String, Any}(string(k) => _json_scalar(v) for (k, v) in pairs(nt))
_json_scalar(v::Real) = isfinite(v) ? v : nothing
_json_scalar(v::Symbol) = string(v)
_json_scalar(v) = v

"A finite metric from a summary NamedTuple, or `missing` — NaN is not a DOUBLE PRECISION."
function _cal_metric(nt, key::Symbol)
    nt === nothing && return missing
    haskey(nt, key) || return missing
    v = getfield(nt, key)
    return (v isa Real && isfinite(v)) ? Float64(v) : missing
end

"""
    load_calibration_db(calibration_run_id, storage) -> NamedTuple

Recover a calibration run: the exact `calibrator`, the `latents` container it produced
(or `nothing` when it was not stored), the per-fixture `diagnostics` frame, and the
queryable `row`.

The container is returned rather than a `CalibratedFit`, because rebuilding one would
need the raw `Fit` — which lives under its own key and may be gigabytes. Compose them
when you want one:

```julia
c   = load_calibration_db(cal_run_id, db)
fit = load_fit(db, c.row.model_run_id)
cf  = calibrate_fit(c.calibrator, fit, book)     # re-derives, and must match c.latents
```
"""
function load_calibration_db(calibration_run_id::UUID, storage::Training.PostgresStorage)
    conn = Training.Inference._db_connect(storage)
    try
        rows = Training.Inference._db_rows(conn, """
            SELECT r.calibration_run_id, r.model_run_id, r.experiment_name,
                   r.calibrator_name, r.calibrator_hash, r.config_json,
                   r.book_as_of_minutes, r.n_fixtures, r.n_inverted,
                   r.log_loss, r.ece, r.brier, r.clv_mean_pct, r.clv_weighted_pct,
                   r.created_at, r.metadata,
                   a.calibrator_blob, a.calibrated_latents_blob, a.diagnostics_blob
            FROM calibration_runs r
            LEFT JOIN calibration_artifacts a USING (calibration_run_id)
            WHERE r.calibration_run_id = \$1::uuid;
        """, (string(calibration_run_id),))
        nrow(rows) == 1 || error(
            "load_calibration_db: no calibration run $calibration_run_id in " *
            "experiment '$(storage.experiment_name)'.")

        blob = rows.calibrator_blob[1]
        ismissing(blob) && error(
            "load_calibration_db: run $calibration_run_id has a row but no artefact. " *
            "`save_calibration_db` writes both in one transaction, so this row was not " *
            "written by it.")
        cal = Training.Inference._db_artifact_value(blob)
        cal isa AbstractCalibrator || error(
            "load_calibration_db: artefact for $calibration_run_id holds $(typeof(cal)).")

        lat = ismissing(rows.calibrated_latents_blob[1]) ? nothing :
              Training.Inference._db_artifact_value(rows.calibrated_latents_blob[1])
        diag = ismissing(rows.diagnostics_blob[1]) ? nothing :
               Training.Inference._db_artifact_value(rows.diagnostics_blob[1])

        return (; calibrator = cal, latents = lat, diagnostics = diag,
                row = only(eachrow(select(rows, Not([:calibrator_blob,
                                                     :calibrated_latents_blob,
                                                     :diagnostics_blob])))))
    finally
        close(conn)
    end
end

load_calibration_db(calibration_run_id::AbstractString, storage::Training.PostgresStorage) =
    load_calibration_db(UUID(calibration_run_id), storage)

"""
    list_calibration_runs(storage; model_run_id = nothing, calibrator = nothing) -> DataFrame

The calibration leaderboard for this experiment, newest first. Filter by the model run,
by calibrator name, or by neither.

`book_as_of_minutes` is a first-class column rather than a JSON path deliberately: two
calibration runs at different price instants are not comparable (README §7.3), and the
most important filter should not need `->>`.
"""
function list_calibration_runs(storage::Training.PostgresStorage;
                               model_run_id = nothing, calibrator = nothing)
    conn = Training.Inference._db_connect(storage)
    try
        return Training.Inference._db_rows(conn, """
            SELECT calibration_run_id, model_run_id, calibrator_name, calibrator_hash,
                   book_as_of_minutes, n_fixtures, n_inverted,
                   log_loss, ece, brier, clv_mean_pct, clv_weighted_pct, created_at
            FROM calibration_runs
            WHERE experiment_name = \$1
              AND (\$2::uuid IS NULL OR model_run_id = \$2::uuid)
              AND (\$3::text IS NULL OR calibrator_name = \$3)
            ORDER BY created_at DESC;
        """, (storage.experiment_name,
              model_run_id === nothing ? missing : string(model_run_id),
              calibrator === nothing ? missing : String(calibrator)))
    finally
        close(conn)
    end
end


# ==============================================================================
# 2. LINKING A PORTFOLIO RUN
# ==============================================================================

"""
    link_portfolio_run(result, model_run_id, calibration_run_id, storage;
                       book_spec, policy_spec, metadata) -> UUID

`Portfolio.save_portfolio_db` with the calibration lineage stamped into
`portfolio_runs.metadata`, and the key spelling in ONE place.

There is no foreign key and there should not be: `portfolio_runs` has to stay insertable
for a portfolio built on a raw fit, which a NOT NULL reference would prevent, and a
nullable one buys nothing a JSONB key does not.

`model_run_id` is the RAW run's UUID, not the calibration's — `portfolio_runs.model_run_id`
means "which posterior was this priced from", and the calibrated posterior's lineage is
the calibration row, which itself points back to the same raw run.
"""
function link_portfolio_run(result, model_run_id::UUID, calibration_run_id::UUID,
                            storage::Training.PostgresStorage;
                            book_spec = nothing, policy_spec = nothing,
                            calibrator = nothing, metadata = NamedTuple())
    extra = Dict{String, Any}(_CAL_LINK_KEY => string(calibration_run_id))
    calibrator === nothing || begin
        extra["calibrator"] = calibrator.name
        extra["calibrator_label"] = calibrator_label(calibrator)
        extra["book_as_of_minutes"] = calibrator.book_as_of_minutes
    end
    for (k, v) in pairs(metadata)
        extra[string(k)] = v
    end
    return Portfolio.save_portfolio_db(result, model_run_id, storage;
                                       book_spec = book_spec, policy_spec = policy_spec,
                                       metadata = NamedTuple(Symbol(k) => v for (k, v) in extra))
end

link_portfolio_run(result, model_run_id::AbstractString, calibration_run_id, storage; kwargs...) =
    link_portfolio_run(result, UUID(model_run_id), UUID(string(calibration_run_id)), storage;
                       kwargs...)

"""
    portfolio_runs_for_calibration(calibration_run_id, storage) -> DataFrame

Every portfolio run stamped with this calibration, newest first. The read-back half of
[`link_portfolio_run`](@ref).
"""
function portfolio_runs_for_calibration(calibration_run_id, storage::Training.PostgresStorage)
    conn = Training.Inference._db_connect(storage)
    try
        return Training.Inference._db_rows(conn, """
            SELECT portfolio_run_id, model_run_id, book_spec_hash, policy_spec_hash,
                   total_return_pct, flat_roi_pct, max_drawdown_pct, sharpe_ann,
                   win_rate, n_bets, created_at
            FROM portfolio_runs
            WHERE metadata ->> '$(_CAL_LINK_KEY)' = \$1
            ORDER BY created_at DESC;
        """, (string(calibration_run_id),))
    finally
        close(conn)
    end
end
