# Incremental walk-forward extension for PostgreSQL-backed Fits.
#
# Splitters remain declarative.  Every call derives boundaries from the supplied current
# DataStore, compares their stable 1-based positions with fold_results, and samples only missing
# positions.  The relational rows and exact Fit artefact are replaced in one transaction.

"The database identity and display metadata for a model run."
function _extension_run(db::PostgresStorage, key)
    run_id = key isa UUID ? key : _run_uuid(db, key)
    conn = _db_connect(db)
    try
        rows = _db_rows(conn, """
            SELECT id, run_id, name, duration_seconds
            FROM runs
            WHERE experiment_name = \$1 AND run_id = \$2::uuid
            LIMIT 1;
        """, (db.experiment_name, string(run_id)))
        nrow(rows) == 1 || error(
            "No run $(repr(key)) in experiment '$(db.experiment_name)'.")
        duration = ismissing(rows.duration_seconds[1]) ? 0.0 : Float64(rows.duration_seconds[1])
        return (; id = Int(rows.id[1]), run_id, name = String(rows.name[1]), duration)
    finally
        close(conn)
    end
end

_extension_run(db::PostgresStorage, key::Symbol) = _extension_run(db, String(key))

"Completed fold positions and their persisted runtime summary."
function _extension_completed(db::PostgresStorage, run_id::UUID)
    conn = _db_connect(db)
    try
        rows = _db_rows(conn, """
            SELECT fold_idx, runtime_seconds, n_matches, first_match_date, last_match_date
            FROM fold_results
            WHERE run_id = \$1::uuid
            ORDER BY fold_idx;
        """, (string(run_id),))
        return rows
    finally
        close(conn)
    end
end

function _extension_date_column(df::AbstractDataFrame)
    :match_date in propertynames(df) && return :match_date
    :date in propertynames(df) && return :date
    return nothing
end

function _extension_fixture_stats(fixtures)
    fixtures === nothing && return (; n_matches = 0, first_date = nothing, last_date = nothing)
    n = nrow(fixtures)
    n == 0 && return (; n_matches = 0, first_date = nothing, last_date = nothing)
    column = _extension_date_column(fixtures)
    column === nothing && return (; n_matches = n, first_date = nothing, last_date = nothing)
    dates = Date[]
    for value in fixtures[!, column]
        ismissing(value) && continue
        push!(dates, Date(value))
    end
    return isempty(dates) ?
        (; n_matches = n, first_date = nothing, last_date = nothing) :
        (; n_matches = n, first_date = minimum(dates), last_date = maximum(dates))
end

function _extension_oos(ds::Data.DataStore, boundaries, splitter)
    return Any[Data.get_next_matches(ds, boundary, splitter) for boundary in boundaries]
end

function _extension_delta(db::PostgresStorage, run, ds::Data.DataStore, splitter)
    boundaries = Data.create_id_boundaries(ds, splitter)
    completed = _extension_completed(db, run.run_id)
    completed_indices = Set(Int.(completed.fold_idx))
    delta = Int[i for i in eachindex(boundaries) if i ∉ completed_indices]
    oos = _extension_oos(ds, boundaries, splitter)
    return (; boundaries, completed, delta, oos)
end

function _extension_range(stats)
    dates = Date[]
    for stat in stats
        stat.first_date === nothing || push!(dates, stat.first_date)
        stat.last_date === nothing || push!(dates, stat.last_date)
    end
    isempty(dates) && return "—"
    return minimum(dates) == maximum(dates) ? string(first(dates)) :
           string(minimum(dates), " to ", maximum(dates))
end

function _extension_estimate(completed::DataFrame, n_delta::Int)
    n_delta == 0 && return "0s"
    runtimes = Float64[Float64(x) for x in completed.runtime_seconds
                       if !ismissing(x) && isfinite(x)]
    isempty(runtimes) && return "unknown"
    return format_elapsed(mean(runtimes) * n_delta)
end

"""
    preview_extension(db, run_id_or_name, ds; splitter=nothing, io=stdout)

Derive the current walk-forward boundaries without sampling and report which database fold
positions are missing.  `splitter` may replace the run's stored splitter (for example when a new
target season is opened); omission uses the exact splitter in the persisted Fit.
"""
function preview_extension(db::PostgresStorage, key, ds::Data.DataStore;
                           splitter = nothing, io::IO = stdout)
    run = _extension_run(db, key)
    fit = load_fit(db, run.run_id)
    active_splitter = splitter === nothing ? fit.config.splitter : splitter
    active_splitter isa Union{Data.CVConfig,Data.GroupedCVConfig} || error(
        "preview_extension supports CVConfig and GroupedCVConfig; got $(typeof(active_splitter)).")
    plan = _extension_delta(db, run, ds, active_splitter)
    existing_count = nrow(plan.completed)
    new_count = length(plan.delta)

    if new_count == 0
        println(io, "Run #$(run.id) '$(run.name)' is up-to-date ($existing_count folds completed). 0 new folds needed.")
    else
        stats = [_extension_fixture_stats(plan.oos[i]) for i in plan.delta]
        new_matches = sum(s -> s.n_matches, stats; init = 0)
        table = [[string(run.id), run.name, string(existing_count), string(new_count),
                  string(new_matches), _extension_range(stats),
                  _extension_estimate(plan.completed, new_count)]]
        _print_db_table(io,
            ["Run ID", "Run Name", "Existing Folds", "New Folds to Fit", "New Matches",
             "New Date Range", "Estimated Compute"], table;
            max_widths = [8, 30, 16, 18, 12, 24, 18])
    end

    return (; delta_folds = plan.delta, existing_count, new_count,
            is_uptodate = isempty(plan.delta))
end

preview_extension(db::PostgresStorage, key::Symbol, ds::Data.DataStore; kwargs...) =
    preview_extension(db, String(key), ds; kwargs...)

# Preserve global splitter positions while reusing all three existing fold executors.  run_folds
# sees a dense delta collection; this wrapper translates its local position at the one sampling
# seam and otherwise leaves the sampler untouched.
struct _ExtensionSampler{S}
    sampler::S
    fold_indices::Vector{Int}
end

sampler_n_chains(s::_ExtensionSampler) = sampler_n_chains(s.sampler)
sampler_max_depth(s::_ExtensionSampler) = sampler_max_depth(s.sampler)

function sample_fold(model, sampler::_ExtensionSampler, feature_set, local_fold::Int;
                     chain_id::Union{Int,Nothing} = nothing)
    global_fold = sampler.fold_indices[local_fold]
    return sample_fold(model, sampler.sampler, feature_set, global_fold; chain_id = chain_id)
end

function _extension_extract_latents(model, folds, feature_sets, oos)
    per_fold = Dict{Int,Any}()
    for (local_index, fold) in enumerate(folds)
        fixtures = oos[local_index]
        (fixtures === nothing || nrow(fixtures) == 0) && continue
        fold.chain isa Chains || error(
            "extend_fit: fold $(fold.fold) returned a point estimate; CountLatents require posterior draws.")
        latent = extract_latents(model, fold.chain, fixtures, feature_sets[local_index][1])
        latent isa CountLatents || error(
            "extend_fit: PostgreSQL extension requires CountLatents; fold $(fold.fold) produced $(typeof(latent)).")
        per_fold[fold.fold] = latent
    end
    merged = isempty(per_fold) ? nothing :
             merge_latents(Any[per_fold[fold.fold] for fold in folds if haskey(per_fold, fold.fold)])
    return per_fold, merged
end

@inline function _extension_count_pmf!(dest::Vector{Float64}, mean_rate::Float64,
                                       dispersion::Union{Nothing,Float64})
    if dispersion === nothing
        dest[1] = exp(-mean_rate)
        @inbounds for k in 1:(length(dest) - 1)
            dest[k + 1] = dest[k] * mean_rate / k
        end
    else
        probability = dispersion / (dispersion + mean_rate)
        dest[1] = probability^dispersion
        @inbounds for k in 1:(length(dest) - 1)
            dest[k + 1] = dest[k] * ((k - 1 + dispersion) / k) * (1.0 - probability)
        end
    end
    return dest
end

"Out-of-sample LogLoss, multiclass Brier and 1X2 RPS for one CountLatents panel."
function _extension_scores(latents::CountLatents, fixtures::AbstractDataFrame)
    (:home_score in propertynames(fixtures) && :away_score in propertynames(fixtures)) ||
        return (; logloss = missing, brier = missing, rps = missing)
    fixture_by_id = Dict(Int(row.match_id) => row for row in eachrow(fixtures))
    max_goals = 24
    home_pmf = Vector{Float64}(undef, max_goals + 1)
    away_pmf = similar(home_pmf)
    probabilities = zeros(3) # home, draw, away; reused for every fixture
    total_logloss = 0.0
    total_brier = 0.0
    total_rps = 0.0
    n_scored = 0

    for i in eachindex(latents.match_ids)
        row = get(fixture_by_id, latents.match_ids[i], nothing)
        row === nothing && continue
        (ismissing(row.home_score) || ismissing(row.away_score)) && continue
        fill!(probabilities, 0.0)
        for draw in axes(latents.λ_home, 2)
            obs = latents.observation_params
            r_h = obs === nothing ? nothing : Float64(obs.r_h[i, draw])
            r_a = obs === nothing ? nothing : Float64(obs.r_a[i, draw])
            _extension_count_pmf!(home_pmf, Float64(latents.λ_home[i, draw]), r_h)
            _extension_count_pmf!(away_pmf, Float64(latents.λ_away[i, draw]), r_a)
            p_home = 0.0
            p_draw = 0.0
            p_away = 0.0
            @inbounds for h in 0:max_goals, a in 0:max_goals
                mass = home_pmf[h + 1] * away_pmf[a + 1]
                if h > a
                    p_home += mass
                elseif h == a
                    p_draw += mass
                else
                    p_away += mass
                end
            end
            normalizer = p_home + p_draw + p_away
            probabilities[1] += p_home / normalizer
            probabilities[2] += p_draw / normalizer
            probabilities[3] += p_away / normalizer
        end
        probabilities ./= size(latents.λ_home, 2)
        outcome = row.home_score > row.away_score ? 1 : row.home_score == row.away_score ? 2 : 3
        total_logloss -= log(max(probabilities[outcome], eps(Float64)))
        for k in 1:3
            target = k == outcome ? 1.0 : 0.0
            total_brier += (probabilities[k] - target)^2
        end
        target_cumulative_1 = outcome <= 1 ? 1.0 : 0.0
        target_cumulative_2 = outcome <= 2 ? 1.0 : 0.0
        total_rps += (probabilities[1] - target_cumulative_1)^2 +
                     (probabilities[1] + probabilities[2] - target_cumulative_2)^2
        n_scored += 1
    end
    n_scored == 0 && return (; logloss = missing, brier = missing, rps = missing)
    return (; logloss = total_logloss / n_scored,
            brier = total_brier / n_scored,
            rps = total_rps / n_scored)
end

function _extension_config(config::FitConfig, splitter, execution, elapsed::Float64)
    tags = filter(config.tags) do tag
        !startswith(tag, "time:") && !startswith(tag, "extension:")
    end
    push!(tags, "time:" * format_elapsed(elapsed))
    push!(tags, "extension:" * Dates.format(now(), "yyyymmdd_HHMMSS"))
    return FitConfig(name = config.name, model = config.model, splitter = splitter,
                     sampler = config.sampler, execution = execution, tags = tags,
                     description = config.description, save_dir = config.save_dir)
end

function _extension_split_json(splitter, n_total::Int, latest_fold::Int, latest_date)
    payload = _db_config_description(splitter)
    payload["n_folds_total"] = n_total
    payload["latest_fold_idx"] = latest_fold
    payload["latest_fold_date"] = latest_date === nothing ? nothing : string(latest_date)
    return JSON3.write(payload)
end

function _extension_insert_fold!(conn, run_id::UUID, fold::FoldFit, diagnostic,
                                 thresholds::ConvergenceThresholds,
                                 scores, runtime::Float64, stats, latents)
    fold_id = uuid4()
    rhat = _db_nullable(diagnostic.max_rhat)
    ess_bulk = _db_nullable_int(diagnostic.min_ess_bulk)
    ess_tail = _db_nullable_int(diagnostic.min_ess_tail)
    converged = summarise_convergence([diagnostic]; thresholds).passed
    _db_exec(conn, """
        INSERT INTO fold_results (
            fold_id, run_id, fold_idx, r_hat_max, ess_bulk_min, ess_tail_min,
            divergences, converged, logloss, brier, rps, runtime_seconds,
            n_matches, first_match_date, last_match_date
        ) VALUES (\$1::uuid, \$2::uuid, \$3, \$4, \$5, \$6, \$7, \$8,
                  \$9, \$10, \$11, \$12, \$13, \$14, \$15);
    """, (string(fold_id), string(run_id), fold.fold, rhat, ess_bulk, ess_tail,
          diagnostic.n_divergent, converged, scores.logloss, scores.brier, scores.rps,
          runtime, stats.n_matches, stats.first_date, stats.last_date))
    latents === nothing || _db_insert_latents!(conn, fold_id, latents)
    return fold_id
end

"""
    extend_fit(db, run_id_or_name, ds; execution=nothing, splitter=nothing, quiet=false)

Sample only walk-forward positions absent from `fold_results`, audit and extract their OOS
posterior, then atomically append relational rows and replace the exact Fit artefact.
"""
function extend_fit(db::PostgresStorage, key, ds::Data.DataStore;
                    execution = nothing, splitter = nothing, quiet::Bool = false)
    started = time()
    run = _extension_run(db, key)
    existing = load_fit(db, run.run_id)
    active_splitter = splitter === nothing ? existing.config.splitter : splitter
    active_splitter isa Union{Data.CVConfig,Data.GroupedCVConfig} || error(
        "extend_fit supports CVConfig and GroupedCVConfig; got $(typeof(active_splitter)).")
    active_execution = execution === nothing ? existing.config.execution : execution
    active_execution isa AbstractExecution || error(
        "extend_fit: execution must be an AbstractExecution; got $(typeof(active_execution)).")
    plan = _extension_delta(db, run, ds, active_splitter)

    if isempty(plan.delta)
        quiet || println("Run #$(run.id) is already up-to-date with 0 new folds. Returning loaded Fit.")
        return existing
    end

    quiet || _inf_info("sampling missing folds $(join(plan.delta, ", "))")
    selected_boundaries = [plan.boundaries[i] for i in plan.delta]
    feature_sets = Features.create_features(selected_boundaries, ds, existing.config.model,
                                            active_splitter)
    selected_oos = Any[plan.oos[i] for i in plan.delta]
    wrapped_sampler = _ExtensionSampler(existing.config.sampler, plan.delta)
    resolved = resolve_execution(active_execution, existing.config.sampler)
    results = run_folds(existing.config.model, wrapped_sampler, resolved, feature_sets;
                        on_progress = quiet ? _inf_noop : _inf_progress(started))
    successful = findall(!isnothing, results)
    isempty(successful) && error(
        "extend_fit: every missing fold failed to sample; the database was not changed.")

    new_folds = _inf_narrow(FoldFit[FoldFit(plan.delta[i], results[i], feature_sets[i][2])
                                    for i in successful])
    successful_features = [feature_sets[i] for i in successful]
    successful_oos = Any[selected_oos[i] for i in successful]
    delta_diagnostics = audit_convergence(new_folds;
        thresholds = existing.diagnostics.thresholds,
        max_depth = sampler_max_depth(existing.config.sampler))
    per_fold_latents, new_latents = _extension_extract_latents(
        existing.config.model, new_folds, successful_features, successful_oos)

    combined_folds = _inf_narrow(vcat(existing.folds, new_folds))
    sort!(combined_folds, by = fold -> fold.fold)
    combined_diagnostics = audit_convergence(combined_folds;
        thresholds = existing.diagnostics.thresholds,
        max_depth = sampler_max_depth(existing.config.sampler))
    combined_latents = existing.latents === nothing ? new_latents :
                       new_latents === nothing ? existing.latents :
                       merge_latents(Any[existing.latents, new_latents])

    extension_elapsed = time() - started
    total_elapsed = run.duration + extension_elapsed
    config = _extension_config(existing.config, active_splitter, active_execution, total_elapsed)
    metadata = FitMetadata(now(), total_elapsed, VERSION, Threads.nthreads(), git_commit_id())
    extended = Fit(config, combined_folds, combined_latents, combined_diagnostics,
                   metadata, default_save_path(config, metadata))

    runtime_per_fold = extension_elapsed / length(new_folds)
    stats_by_fold = Dict(new_folds[i].fold => _extension_fixture_stats(successful_oos[i])
                         for i in eachindex(new_folds))
    scores_by_fold = Dict(fold.fold =>
        (haskey(per_fold_latents, fold.fold) ?
            _extension_scores(per_fold_latents[fold.fold], successful_oos[i]) :
            (; logloss = missing, brier = missing, rps = missing))
        for (i, fold) in enumerate(new_folds))
    diagnostics_by_fold = Dict(d.fold => d for d in delta_diagnostics.folds)
    latest_dates = Date[s.last_date for s in values(stats_by_fold) if s.last_date !== nothing]
    latest_date = isempty(latest_dates) ? nothing : maximum(latest_dates)

    conn = _db_connect(db)
    try
        _db_exec(conn, "BEGIN;")
        try
            locked = _db_rows(conn,
                "SELECT run_id FROM runs WHERE run_id = \$1::uuid FOR UPDATE;",
                (string(run.run_id),))
            nrow(locked) == 1 || error("extend_fit: run $(run.run_id) disappeared.")
            current = _db_rows(conn,
                "SELECT fold_idx FROM fold_results WHERE run_id = \$1::uuid;",
                (string(run.run_id),))
            occupied = Set(Int.(current.fold_idx))
            conflict = Int[fold.fold for fold in new_folds if fold.fold in occupied]
            isempty(conflict) || error(
                "extend_fit: folds $(join(conflict, ", ")) were inserted concurrently; retry preview_extension.")

            for fold in new_folds
                _extension_insert_fold!(conn, run.run_id, fold,
                    diagnostics_by_fold[fold.fold], existing.diagnostics.thresholds,
                    scores_by_fold[fold.fold], runtime_per_fold,
                    stats_by_fold[fold.fold], get(per_fold_latents, fold.fold, nothing))
            end
            _db_exec(conn, """
                UPDATE fit_artifacts SET fit_blob = \$2::bytea WHERE run_id = \$1::uuid;
            """, (string(run.run_id), _db_bytea(_db_artifact_blob(extended))))
            _db_exec(conn, """
                UPDATE configs SET split_config = \$2::jsonb WHERE config_id = \$1::uuid;
            """, (string(run.run_id), _extension_split_json(
                active_splitter, length(plan.boundaries), maximum(f.fold for f in combined_folds),
                latest_date)))
            _db_exec(conn, """
                UPDATE runs SET duration_seconds = \$2, finished_at = \$3, status = 'completed'
                WHERE run_id = \$1::uuid;
            """, (string(run.run_id), total_elapsed, metadata.timestamp))
            _db_exec(conn, "COMMIT;")
        catch
            try
                _db_exec(conn, "ROLLBACK;")
            catch
            end
            rethrow()
        end
    finally
        close(conn)
    end

    n_new_latents = new_latents === nothing ? 0 : n_matches(new_latents)
    quiet || _inf_info("extended run #$(run.id) by $(length(new_folds)) fold(s); " *
                       "$n_new_latents new latent fixture(s)")
    return extended
end

extend_fit(db::PostgresStorage, key::Symbol, ds::Data.DataStore; kwargs...) =
    extend_fit(db, String(key), ds; kwargs...)
