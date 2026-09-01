# ==============================================================================
# Scottish Lower Poisson 24/26 Grid — PostgreSQL End-to-End Verification
# ==============================================================================
#
# Audits the database created by r21_sync_to_postgres.jl. This runner performs no MCMC and no
# writes except deliberate idempotency checks against already registered recipes/Fits.
#
# Usage:
#
#   julia --project -t 8 experiments/scottish_lower/01_poisson_2426_grid/r22_verify_postgres_sync.jl

include(joinpath(@__DIR__, "r21_sync_to_postgres.jl"))

# ==============================================================================
# 1. Small query helpers
# ==============================================================================

function pg22_run_row(db::PostgresStorage, name::String)
    conn = LibPQ.Connection(db.conn_str)
    try
        result = LibPQ.execute(conn, """
            SELECT r.id, r.run_id, c.config_hash
            FROM runs r
            JOIN configs c ON c.config_id = r.run_id
            WHERE r.experiment_name = \$1 AND r.name = \$2
            ORDER BY r.id DESC
            LIMIT 1;
        """, (db.experiment_name, name))
        try
            rows = DataFrame(result)
            nrow(rows) == 1 || error("Expected one persisted run named $name, got $(nrow(rows)).")
            return rows[1, :]
        finally
            close(result)
        end
    finally
        close(conn)
    end
end

function pg22_portfolio_rows(db::PostgresStorage)
    conn = LibPQ.Connection(db.conn_str)
    try
        result = LibPQ.execute(conn, """
            SELECT r.name, pr.n_bets, pr.flat_roi_pct, pr.sharpe_ann
            FROM portfolio_runs pr
            JOIN runs r ON r.run_id = pr.model_run_id
            WHERE r.experiment_name = \$1
            ORDER BY r.name, pr.id DESC;
        """, (db.experiment_name,))
        try
            return DataFrame(result)
        finally
            close(result)
        end
    finally
        close(conn)
    end
end

function pg22_config_id(configs::DataFrame, name::String, kind::String)
    selected = configs[(configs.name .== name) .& (configs.config_type .== kind), :]
    nrow(selected) == 1 || error(
        "Expected one $kind config named $name, got $(nrow(selected)).")
    return Int(selected.id[1])
end

# ==============================================================================
# 2. Verification workflow
# ==============================================================================

function pg22_verify_postgres_sync()
    println("\n", "="^92)
    println(" SCOTTISH LOWER POISSON 24/26 — POSTGRESQL VERIFICATION (NO MCMC)")
    println("="^92)

    db = PostgresStorage(PG21_EXPERIMENT)
    ensure_schema!(db)
    ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)
    models = pg21_models()
    splitter = pg21_splitter()
    sampler = pg21_sampler()
    configs_by_name = pg21_fit_configs(models, splitter, sampler)
    boundaries = Data.create_id_boundaries(ds, splitter)
    length(boundaries) == PG21_EXPECTED_FOLDS || error(
        "Canonical splitter produced $(length(boundaries)) folds; expected $PG21_EXPECTED_FOLDS.")

    println("\n[1/5] Config and run deduplication...")
    run_ids = Dict{String,UUID}()
    source_fits = Dict{String,Fit}()
    for (model_index, name) in enumerate(PG21_MODEL_NAMES)
        source = pg21_source_fit(name, configs_by_name[name], model_index, ds, boundaries)
        row = pg22_run_row(db, name)
        persisted_id = UUID(string(row.run_id))
        expected_hash = config_hash(source.fit, db)
        String(row.config_hash) == expected_hash || error(
            "$name config hash differs: database=$(row.config_hash), expected=$expected_hash.")
        save_fit(source.fit, db) == persisted_id || error(
            "$name save_fit was not idempotent.")
        run_ids[name] = persisted_id
        source_fits[name] = source.fit
    end

    println("\n[2/5] Lossless chain and CountLatents reconstruction...")
    original = source_fits["m00_baseline"]
    loaded = load_fit(db, "m00_baseline")
    length(loaded) == PG21_EXPECTED_FOLDS || error(
        "m00 loaded $(length(loaded)) folds; expected $PG21_EXPECTED_FOLDS.")
    for fold in 1:PG21_EXPECTED_FOLDS
        Array(loaded[fold].chain) == Array(original[fold].chain) || error(
            "m00 fold $fold chain changed during PostgreSQL round-trip.")
    end
    loaded.latents isa CountLatents || error(
        "m00 reconstructed $(typeof(loaded.latents)); expected CountLatents.")
    loaded.latents.match_ids == original.latents.match_ids || error(
        "m00 latent match ordering changed during PostgreSQL round-trip.")
    loaded.latents.λ_home == original.latents.λ_home || error(
        "m00 λ_home changed during PostgreSQL round-trip.")
    loaded.latents.λ_away == original.latents.λ_away || error(
        "m00 λ_away changed during PostgreSQL round-trip.")

    println("\n[3/5] Modular component reconstruction by name and integer ID...")
    registry = list_configs(db)
    m05_id = pg22_config_id(registry, "m05_production_wealth", "model")
    splitter_id = pg22_config_id(registry, "scottish_lower_2426_40fold", "splitter")
    sampler_id = pg22_config_id(registry, "queued_nuts_4x800", "sampler")
    fit_config_id = pg22_config_id(registry, "m05_production_wealth_fit", "fit")

    m05_by_name = load_model(db, "m05_production_wealth")
    m05_by_id = load_model(db, m05_id)
    m05_by_name isa BayesianFootball.Models.ComposableCountModel || error(
        "m05 name lookup returned $(typeof(m05_by_name)).")
    string(m05_by_name) == string(m05_by_id) || error(
        "m05 name and integer-ID reconstruction differ.")
    string(load_splitter(db, splitter_id)) == string(splitter) || error(
        "Canonical splitter did not reconstruct losslessly.")
    string(load_sampler(db, sampler_id)) == string(sampler) || error(
        "Canonical sampler did not reconstruct losslessly.")
    load_fit_config(db, fit_config_id).name == "m05_production_wealth" || error(
        "Canonical m05 FitConfig did not reconstruct by integer ID.")

    println("\n[4/5] REPL discovery and architecture rendering...")
    experiments = explore_experiments(db)
    row_index = findfirst(==(PG21_EXPERIMENT), String.(experiments.experiment_name))
    row_index === nothing && error("explore_experiments did not return $PG21_EXPERIMENT.")
    experiment = experiments[row_index, :]
    Int(experiment.n_runs) == 5 || error(
        "Expected 5 runs in $PG21_EXPERIMENT, got $(experiment.n_runs).")
    ismissing(experiment.best_logloss) && error("Experiment LogLoss summary is missing.")
    ismissing(experiment.best_brier) && error("Experiment Brier summary is missing.")
    isfinite(Float64(experiment.best_logloss)) || error("Experiment LogLoss is not finite.")
    isfinite(Float64(experiment.best_brier)) || error("Experiment Brier is not finite.")

    wealth = search_configs(db, "wealth")
    wealth_names = Set(String.(wealth.name))
    "m02_wealth" in wealth_names || error("search_configs omitted m02_wealth.")
    "m05_production_wealth" in wealth_names || error(
        "search_configs omitted m05_production_wealth.")

    architecture_io = IOBuffer()
    shown = show_config(db, m05_id; io = architecture_io)
    architecture = String(take!(architecture_io))
    shown isa BayesianFootball.Models.ComposableCountModel || error(
        "show_config returned $(typeof(shown)); expected ComposableCountModel.")
    occursin("Architecture", architecture) || error(
        "show_config output omitted the architecture tree.")
    occursin("production_wealth", architecture) || error(
        "show_config output omitted the production-wealth component.")
    print(architecture)

    println("\n[5/5] Portfolio summary audit...")
    portfolios = pg22_portfolio_rows(db)
    latest = unique(portfolios, :name)
    nrow(latest) == 5 || error("Expected portfolio rows for 5 models, got $(nrow(latest)).")
    all(>(0), Int.(latest.n_bets)) || error("At least one model has no persisted bets.")
    all(x -> !ismissing(x) && isfinite(Float64(x)), latest.flat_roi_pct) || error(
        "At least one persisted portfolio ROI is missing or non-finite.")
    all(x -> !ismissing(x) && isfinite(Float64(x)), latest.sharpe_ann) || error(
        "At least one persisted portfolio Sharpe is missing or non-finite.")

    println("\nVerification PASS")
    println("  runs               : 5")
    println("  chains per run     : $PG21_EXPECTED_FOLDS folds")
    println("  OOS latent matches : ", n_matches(loaded.latents))
    println("  model integer ID   : ", m05_id)
    println("  splitter/sampler   : ", (splitter_id, sampler_id))
    println("  m05 FitConfig ID   : ", fit_config_id)
    println("  run IDs            : ", run_ids)
    return (; db, run_ids, loaded, registry, experiments, wealth, portfolios)
end

if abspath(PROGRAM_FILE) == @__FILE__
    pg22_verify_postgres_sync()
end
