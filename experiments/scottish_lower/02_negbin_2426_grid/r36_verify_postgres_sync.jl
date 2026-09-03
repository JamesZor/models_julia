# ==============================================================================
# Scottish Lower Negative Binomial 24/26 Grid — PostgreSQL End-to-End Verification
# ==============================================================================
#
# Audits the database created by r35_sync_to_postgres.jl. This runner performs no MCMC and no
# writes except deliberate idempotency checks against already registered recipes/Fits.
#
# Sibling of `01_poisson_2426_grid/r22_verify_postgres_sync.jl`, with one extra obligation:
# the negative-binomial dispersion must survive the round trip. A `CountLatents` whose
# `observation_params` came back `nothing` still loads, still prices, and still returns a
# plausible ROI — it has simply become a Poisson model wearing the NegBin name. That is the
# failure this file exists to catch.
#
# Usage:
#
#   julia --project -t 16 experiments/scottish_lower/02_negbin_2426_grid/r36_verify_postgres_sync.jl

include(joinpath(@__DIR__, "r35_sync_to_postgres.jl"))

using Printf

# ==============================================================================
# 1. Small query helpers
# ==============================================================================

function pg36_run_row(db::PostgresStorage, name::String)
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

function pg36_portfolio_rows(db::PostgresStorage)
    conn = LibPQ.Connection(db.conn_str)
    try
        result = LibPQ.execute(conn, """
            SELECT r.name,
                   COALESCE(pr.metadata->>'odds_source', 'unknown') AS odds_source,
                   pr.portfolio_run_id, pr.n_bets, pr.total_return_pct, pr.flat_roi_pct,
                   pr.roi_1x2_pct, pr.max_drawdown_pct, pr.sharpe_ann, pr.win_rate
            FROM portfolio_runs pr
            JOIN runs r ON r.run_id = pr.model_run_id
            WHERE r.experiment_name = \$1
            ORDER BY r.name, odds_source, pr.id DESC;
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

function pg36_config_id(configs::DataFrame, name::String, kind::String)
    selected = configs[(configs.name .== name) .& (configs.config_type .== kind), :]
    nrow(selected) == 1 || error(
        "Expected one $kind config named $name, got $(nrow(selected)).")
    return Int(selected.id[1])
end

# ==============================================================================
# 2. Verification workflow
# ==============================================================================

function pg36_verify_postgres_sync()
    println("\n", "="^92)
    println(" SCOTTISH LOWER NEGATIVE BINOMIAL 24/26 — POSTGRESQL VERIFICATION (NO MCMC)")
    println("="^92)

    db = PostgresStorage(PG35_EXPERIMENT)
    ensure_schema!(db)
    ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)
    models = pg35_models()
    splitter = pg35_splitter()
    sampler = pg35_sampler()
    configs_by_name = pg35_fit_configs(models, splitter, sampler)
    boundaries = Data.create_id_boundaries(ds, splitter)
    length(boundaries) == PG35_EXPECTED_FOLDS || error(
        "Canonical splitter produced $(length(boundaries)) folds; expected $PG35_EXPECTED_FOLDS.")

    println("\n[1/6] Config and run deduplication...")
    run_ids = Dict{String,UUID}()
    source_fits = Dict{String,Fit}()
    for (model_index, name) in enumerate(PG35_MODEL_NAMES)
        source = pg35_source_fit(name, configs_by_name[name], model_index, ds, boundaries)
        row = pg36_run_row(db, name)
        persisted_id = UUID(string(row.run_id))
        expected_hash = config_hash(source.fit, db)
        String(row.config_hash) == expected_hash || error(
            "$name config hash differs: database=$(row.config_hash), expected=$expected_hash.")
        save_fit(source.fit, db) == persisted_id || error(
            "$name save_fit was not idempotent.")
        run_ids[name] = persisted_id
        source_fits[name] = source.fit
        println("  $name: run=$persisted_id (hash stable, save_fit idempotent)")
    end

    println("\n[2/6] Lossless chain and CountLatents reconstruction...")
    original = source_fits["m00_negbin_baseline"]
    loaded = load_fit(db, "m00_negbin_baseline")
    length(loaded) == PG35_EXPECTED_FOLDS || error(
        "m00 loaded $(length(loaded)) folds; expected $PG35_EXPECTED_FOLDS.")
    for fold in 1:PG35_EXPECTED_FOLDS
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

    println("\n[3/6] Negative-binomial dispersion survived the round trip...")
    for name in PG35_MODEL_NAMES
        reloaded = load_fit(db, name)
        observation_family(reloaded.latents) == :negbin || error(
            "$name came back as $(observation_family(reloaded.latents)) latents. The " *
            "dispersion was dropped and this run is a Poisson model under a NegBin name.")
        obs = reloaded.latents.observation_params
        expected = source_fits[name].latents.observation_params
        obs.r_h == expected.r_h || error("$name r_h changed during PostgreSQL round-trip.")
        obs.r_a == expected.r_a || error("$name r_a changed during PostgreSQL round-trip.")
        all(>(0), obs.r_h) && all(>(0), obs.r_a) || error(
            "$name reconstructed a non-positive dispersion.")
        @printf("  %-30s r_h ∈ [%.3f, %.3f], r_a ∈ [%.3f, %.3f]\n",
                name, minimum(obs.r_h), maximum(obs.r_h),
                minimum(obs.r_a), maximum(obs.r_a))
    end

    println("\n[4/6] Modular component reconstruction by name and integer ID...")
    registry = list_configs(db)
    m05_id = pg36_config_id(registry, "m05_negbin_production_wealth", "model")
    splitter_id = pg36_config_id(registry, "scottish_lower_negbin_2426_40fold", "splitter")
    sampler_id = pg36_config_id(registry, "queued_nuts_4x800_negbin", "sampler")
    fit_config_id = pg36_config_id(registry, "m05_negbin_production_wealth_fit", "fit")

    m05_by_name = load_model(db, "m05_negbin_production_wealth")
    m05_by_id = load_model(db, m05_id)
    m05_by_name isa BayesianFootball.Models.ComposableCountModel || error(
        "m05 name lookup returned $(typeof(m05_by_name)).")
    string(m05_by_name) == string(m05_by_id) || error(
        "m05 name and integer-ID reconstruction differ.")
    string(load_splitter(db, splitter_id)) == string(splitter) || error(
        "Canonical splitter did not reconstruct losslessly.")
    string(load_sampler(db, sampler_id)) == string(sampler) || error(
        "Canonical sampler did not reconstruct losslessly.")
    load_fit_config(db, fit_config_id).name == "m05_negbin_production_wealth" || error(
        "Canonical m05 FitConfig did not reconstruct by integer ID.")

    println("\n[5/6] REPL discovery and architecture rendering...")
    experiments = explore_experiments(db)
    row_index = findfirst(==(PG35_EXPERIMENT), String.(experiments.experiment_name))
    row_index === nothing && error("explore_experiments did not return $PG35_EXPERIMENT.")
    experiment = experiments[row_index, :]
    Int(experiment.n_runs) == length(PG35_MODEL_NAMES) || error(
        "Expected $(length(PG35_MODEL_NAMES)) runs in $PG35_EXPERIMENT, got $(experiment.n_runs).")
    ismissing(experiment.best_logloss) && error("Experiment LogLoss summary is missing.")
    ismissing(experiment.best_brier) && error("Experiment Brier summary is missing.")
    isfinite(Float64(experiment.best_logloss)) || error("Experiment LogLoss is not finite.")
    isfinite(Float64(experiment.best_brier)) || error("Experiment Brier is not finite.")

    wealth = search_configs(db, "wealth")
    wealth_names = Set(String.(wealth.name))
    "m02_negbin_wealth" in wealth_names || error("search_configs omitted m02_negbin_wealth.")
    "m05_negbin_production_wealth" in wealth_names || error(
        "search_configs omitted m05_negbin_production_wealth.")

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

    println("\n[6/6] Portfolio summary audit (both price sources)...")
    portfolios = pg36_portfolio_rows(db)
    latest = unique(portfolios, [:name, :odds_source])
    expected_rows = length(PG35_MODEL_NAMES) * length(PG35_PRICE_SOURCES)
    nrow(latest) == expected_rows || error(
        "Expected $expected_rows portfolio rows (5 models × 2 price sources), got $(nrow(latest)).")
    "unknown" in Set(String.(latest.odds_source)) && error(
        "A portfolio row carries no metadata->>'odds_source'; its price basis is unrecoverable.")
    all(>(0), Int.(latest.n_bets)) || error("At least one model has no persisted bets.")
    all(x -> !ismissing(x) && isfinite(Float64(x)), latest.flat_roi_pct) || error(
        "At least one persisted portfolio ROI is missing or non-finite.")
    all(x -> !ismissing(x) && isfinite(Float64(x)), latest.sharpe_ann) || error(
        "At least one persisted portfolio Sharpe is missing or non-finite.")

    for price in PG35_PRICE_SOURCES
        rows = latest[String.(latest.odds_source) .== price.tag, :]
        reloaded = load_portfolio_db(UUID(string(rows.portfolio_run_id[1])), db)
        isfinite(reloaded.summary.total_return_pct) || error(
            "load_portfolio_db returned a non-finite return for $(price.tag).")
    end

    println()
    @printf(" %-30s | %-34s | %6s | %9s | %9s | %8s\n",
            "Model", "Price source", "Bets", "Return %", "Flat ROI", "Sharpe")
    println("-"^108)
    for row in eachrow(sort(latest, [:name, :odds_source]))
        @printf(" %-30s | %-34s | %6d | %9.2f | %9.2f | %8.3f\n",
                row.name, row.odds_source, Int(row.n_bets),
                Float64(row.total_return_pct), Float64(row.flat_roi_pct),
                Float64(row.sharpe_ann))
    end

    println("\nVerification PASS")
    println("  runs               : ", length(PG35_MODEL_NAMES))
    println("  folds per run      : $PG35_EXPECTED_FOLDS")
    println("  OOS latent matches : ", n_matches(loaded.latents))
    println("  observation family : ", observation_family(loaded.latents))
    println("  model integer ID   : ", m05_id)
    println("  splitter/sampler   : ", (splitter_id, sampler_id))
    println("  m05 FitConfig ID   : ", fit_config_id)
    println("  run IDs            : ", run_ids)
    return (; db, run_ids, loaded, registry, experiments, wealth, portfolios)
end

if abspath(PROGRAM_FILE) == @__FILE__
    pg36_verify_postgres_sync()
end
