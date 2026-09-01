# ==============================================================================
# Scottish Lower Poisson 24/26 Grid — PostgreSQL Ingestion
# ==============================================================================
#
# This is a persistence and portfolio-reconstruction runner. It does NOT launch MCMC.
# It registers the five canonical r20 recipes, imports completed local Fits when present,
# and otherwise creates deterministic synthetic Fits so every PostgreSQL persistence path can
# be exercised locally without competing with production sampling.
#
# Usage (from the repository root):
#
#   julia --project -t 8 experiments/scottish_lower/01_poisson_2426_grid/r21_sync_to_postgres.jl
#
# Connection credentials are resolved by PostgresStorage from BF_EXPERIMENTS_DB_URL or libpq's
# ~/.pgpass. No connection string is printed or embedded here.

using BayesianFootball
using DataFrames
using Dates
using Distributions
using JSON3
using LibPQ
using MCMCChains
using Random
using Statistics
using UUIDs

const PG21_EXPERIMENT = "scottish_lower_poisson_2426"
const PG21_SAVE_ROOT = "./data/scottish_lower_2426_grid"
const PG21_TAGS = ["production", "poisson", "scottish_lower", "2426"]
const PG21_EXPECTED_FOLDS = 40
const PG21_SYNTHETIC_DRAWS = 160
const PG21_SYNTHETIC_CHAINS = 4

const PG21_MODEL_NAMES = [
    "m00_baseline",
    "m02_wealth",
    "m03_distance",
    "m04_joint",
    "m05_production_wealth",
]

const PG21_MODEL_DESCRIPTIONS = Dict(
    "m00_baseline" =>
        "Canonical Scottish Lower Poisson baseline: global intercept, 180-day time decay, " *
        "global home advantage, and Poisson observation.",
    "m02_wealth" =>
        "Canonical baseline plus point-in-time raw starting-XI squad wealth from Transfermarkt.",
    "m03_distance" =>
        "Canonical baseline plus away-ground travel-distance fatigue.",
    "m04_joint" =>
        "Canonical joint Poisson model with raw starting-XI wealth and travel distance.",
    "m05_production_wealth" =>
        "Canonical baseline plus age-adjusted production wealth using RichardsSigmoid(23, 0.80, 2).",
)

# ==============================================================================
# 1. Canonical recipes — kept visibly aligned with r20
# ==============================================================================

function pg21_models()
    m00 = CountModelBuilder(:m00_baseline_poisson) |>
        BayesianFootball.add(GlobalInterception()) |>
        BayesianFootball.add(TimeDecayDynamics(days_half_life = 180.0)) |>
        BayesianFootball.add(GlobalHomeAdvantage()) |>
        BayesianFootball.add(PoissonObservation()) |>
        BayesianFootball.build

    m02 = CountModelBuilder(:m02_poisson_wealth) |>
        BayesianFootball.add(GlobalInterception()) |>
        BayesianFootball.add(TimeDecayDynamics(days_half_life = 180.0)) |>
        BayesianFootball.add(GlobalHomeAdvantage()) |>
        BayesianFootball.add(WealthCovariate(
            prior = truncated(Normal(0.10, 0.05), lower = 0.0))) |>
        BayesianFootball.add(PoissonObservation()) |>
        BayesianFootball.build

    m03 = CountModelBuilder(:m03_poisson_distance) |>
        BayesianFootball.add(GlobalInterception()) |>
        BayesianFootball.add(TimeDecayDynamics(days_half_life = 180.0)) |>
        BayesianFootball.add(GlobalHomeAdvantage()) |>
        BayesianFootball.add(DistanceCovariate(
            prior = truncated(Normal(0.04, 0.03), lower = 0.0))) |>
        BayesianFootball.add(PoissonObservation()) |>
        BayesianFootball.build

    m04 = CountModelBuilder(:m04_poisson_joint) |>
        BayesianFootball.add(GlobalInterception()) |>
        BayesianFootball.add(TimeDecayDynamics(days_half_life = 180.0)) |>
        BayesianFootball.add(GlobalHomeAdvantage()) |>
        BayesianFootball.add(WealthCovariate(
            prior = truncated(Normal(0.10, 0.05), lower = 0.0))) |>
        BayesianFootball.add(DistanceCovariate(
            prior = truncated(Normal(0.04, 0.03), lower = 0.0))) |>
        BayesianFootball.add(PoissonObservation()) |>
        BayesianFootball.build

    m05 = CountModelBuilder(:m05_poisson_production_wealth) |>
        BayesianFootball.add(GlobalInterception()) |>
        BayesianFootball.add(TimeDecayDynamics(days_half_life = 180.0)) |>
        BayesianFootball.add(GlobalHomeAdvantage()) |>
        BayesianFootball.add(ProductionWealthCovariate(
            feature = ProductionWealthFeature(curve = RichardsSigmoid(23.0, 0.80, 2.0)),
            prior = truncated(Normal(0.10, 0.05), lower = 0.0),
        )) |>
        BayesianFootball.add(PoissonObservation()) |>
        BayesianFootball.build

    return [
        "m00_baseline" => m00,
        "m02_wealth" => m02,
        "m03_distance" => m03,
        "m04_joint" => m04,
        "m05_production_wealth" => m05,
    ]
end

pg21_splitter() = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons = ["24/25", "25/26"],
    history_seasons = 2,
    dynamics_col = :match_biweek,
    warmup_period = 0,
    stop_early = true,
)

pg21_sampler() = QueuedNUTSConfig(
    n_samples = 800,
    n_warmup = 800,
    n_chains = 4,
    accept_rate = 0.65,
)

pg21_book_spec() = BookSpec(
    markets = Data.MarketConfig([
        Data.Market1X2(),
        Data.MarketOverUnder(2.5),
        Data.MarketBTTS(),
    ]),
    price = DeArb(),
    allocator = KellyLogUtility(),
    shrink = BakerMcHale(),
    exec = ExecutionConfig(
        commission = PerBetCommission(0.02),
        budget = 0.99,
        min_selection_stake = 0.001,
    ),
)

pg21_policy_spec() = PolicySpec(
    trust = FlatTrust(0.30),
    risk = SlateDrawdown(23.0),
    cap = FixedCap(0.20),
    grouping = DailySlate(),
)

function pg21_fit_configs(models, splitter, sampler)
    configs = Dict{String,FitConfig}()
    for (name, model) in models
        configs[name] = FitConfig(
            name = name,
            model = model,
            splitter = splitter,
            sampler = sampler,
            execution = QueuedExecution(max_concurrent_tasks = Threads.nthreads()),
            tags = copy(PG21_TAGS),
            description = PG21_MODEL_DESCRIPTIONS[name],
            save_dir = joinpath(PG21_SAVE_ROOT, name),
        )
    end
    return configs
end

# ==============================================================================
# 2. Existing-fit discovery and deterministic local fallback
# ==============================================================================

function pg21_latest_completed_fit(name::String)
    directory = joinpath(PG21_SAVE_ROOT, name)
    isdir(directory) || return nothing
    candidates = list_fits(directory; quiet = true)
    for candidate in Iterators.reverse(candidates)
        fit = load_fit(candidate.path; quiet = true)
        if length(fit) == PG21_EXPECTED_FOLDS && fit.latents isa CountLatents
            return fit
        end
    end
    return nothing
end

function pg21_synthetic_chain(seed::Int)
    rng = Xoshiro(seed)
    n = PG21_SYNTHETIC_DRAWS
    n_chains = PG21_SYNTHETIC_CHAINS
    values = Array{Float64,3}(undef, n, 5, n_chains)
    for chain in 1:n_chains
        values[:, 1, chain] .= randn(rng, n)
        values[:, 2, chain] .= 0.5 .* randn(rng, n)
        values[:, 3, chain] .= 0.0
        values[:, 4, chain] .= 4.0
        values[:, 5, chain] .= 100.0 .+ 3.0 .* randn(rng, n)
    end
    return Chains(
        values,
        [:synthetic_intercept, :synthetic_home_advantage, :numerical_error,
         :tree_depth, :hamiltonian_energy],
        (parameters = [:synthetic_intercept, :synthetic_home_advantage],
         internals = [:numerical_error, :tree_depth, :hamiltonian_energy]),
    )
end

function pg21_oos_match_ids(ds::Data.DataStore, boundaries, splitter)
    ids = Int[]
    for boundary in boundaries
        fixtures = Data.get_next_matches(ds, boundary, splitter)
        append!(ids, Int.(fixtures.match_id))
    end
    allunique(ids) || error("Synthetic fallback: walk-forward OOS match IDs are not unique.")
    return ids
end

function pg21_synthetic_latents(ids::Vector{Int}, model_index::Int, n_draws::Int)
    lambda_home = Matrix{Float64}(undef, length(ids), n_draws)
    lambda_away = Matrix{Float64}(undef, length(ids), n_draws)
    home_shift = 0.04 * (model_index - 1)
    away_shift = 0.015 * (model_index - 1)
    for draw in 1:n_draws
        draw_phase = 0.017 * draw
        for row in eachindex(ids)
            fixture_phase = 0.013 * ids[row]
            lambda_home[row, draw] = 1.62 + home_shift + 0.08 * sin(fixture_phase + draw_phase)
            lambda_away[row, draw] = 1.04 + away_shift + 0.07 * cos(fixture_phase - draw_phase)
        end
    end
    return CountLatents(ids, lambda_home, lambda_away)
end

function pg21_synthetic_fit(name::String, config::FitConfig, model_index::Int,
                            ds::Data.DataStore, boundaries)
    length(boundaries) == PG21_EXPECTED_FOLDS || error(
        "Synthetic fallback expected $PG21_EXPECTED_FOLDS folds, got $(length(boundaries)).")
    chains = [pg21_synthetic_chain(10_000 * model_index + fold)
              for fold in eachindex(boundaries)]
    folds = [FoldFit(fold, chains[fold], boundaries[fold][2])
             for fold in eachindex(boundaries)]
    diagnostics = audit_convergence(folds; max_depth = 10)
    ids = pg21_oos_match_ids(ds, boundaries, config.splitter)
    n_draws = size(first(chains), 1) * size(first(chains), 3)
    latents = pg21_synthetic_latents(ids, model_index, n_draws)
    metadata = FitMetadata(now(), 0.0, VERSION, Threads.nthreads(), "synthetic-no-mcmc")
    save_path = joinpath(config.save_dir, name * "_synthetic")
    return Fit(config, folds, latents, diagnostics, metadata, save_path)
end

function pg21_source_fit(name::String, config::FitConfig, model_index::Int,
                         ds::Data.DataStore, boundaries)
    disk_fit = pg21_latest_completed_fit(name)
    disk_fit === nothing || return (fit = disk_fit, source = :disk)
    return (fit = pg21_synthetic_fit(name, config, model_index, ds, boundaries),
            source = :synthetic)
end

# ==============================================================================
# 3. Relational score and portfolio helpers
# ==============================================================================

function pg21_persist_scores!(db::PostgresStorage, run_id::UUID, scores::PredictionScores)
    values = (scores.model.logloss, scores.model.brier, scores.model.rps)
    all(isfinite, values) || error("Evaluation produced non-finite model scores: $values")
    conn = LibPQ.Connection(db.conn_str)
    try
        result = LibPQ.execute(conn, """
            UPDATE fold_results
            SET logloss = \$2, brier = \$3, rps = \$4
            WHERE run_id = \$1::uuid;
        """, (string(run_id), values...))
        close(result)
    finally
        close(conn)
    end
    return nothing
end

function pg21_existing_portfolio(db::PostgresStorage, run_id::UUID, book_spec, policy_spec)
    book_hash = portfolio_spec_hash(book_spec)
    policy_hash = portfolio_spec_hash(policy_spec)
    conn = LibPQ.Connection(db.conn_str)
    try
        result = LibPQ.execute(conn, """
            SELECT portfolio_run_id
            FROM portfolio_runs
            WHERE model_run_id = \$1::uuid
              AND book_spec_hash = \$2
              AND policy_spec_hash = \$3
            ORDER BY id DESC
            LIMIT 1;
        """, (string(run_id), book_hash, policy_hash))
        try
            rows = DataFrame(result)
            return nrow(rows) == 0 ? nothing : UUID(string(rows.portfolio_run_id[1]))
        finally
            close(result)
        end
    finally
        close(conn)
    end
end

function pg21_save_portfolio!(db::PostgresStorage, run_id::UUID, fit::Fit,
                              ds::Data.DataStore, book_spec, policy_spec)
    existing = pg21_existing_portfolio(db, run_id, book_spec, policy_spec)
    existing === nothing || return (portfolio_id = existing,
                                    result = load_portfolio_db(existing, db),
                                    reused = true)
    result, _, _ = run_portfolio_simulation(
        book_spec,
        policy_spec,
        fit,
        ds.odds,
        ds;
        bootstrap = false,
        require_converged = false,
        quiet = true,
    )
    portfolio_id = save_portfolio_db(
        result,
        run_id,
        db;
        book_spec,
        policy_spec,
        metadata = (; ingestion = "r21_sync_to_postgres"),
    )
    return (; portfolio_id, result, reused = false)
end

# ==============================================================================
# 4. Ingestion workflow
# ==============================================================================

function pg21_sync_to_postgres()
    println("\n", "="^92)
    println(" SCOTTISH LOWER POISSON 24/26 — POSTGRESQL SYNC (NO MCMC)")
    println("="^92)

    db = PostgresStorage(PG21_EXPERIMENT)
    ensure_schema!(db)
    ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)

    models = pg21_models()
    splitter = pg21_splitter()
    sampler = pg21_sampler()
    book_spec = pg21_book_spec()
    policy_spec = pg21_policy_spec()
    fit_configs = pg21_fit_configs(models, splitter, sampler)
    boundaries = Data.create_id_boundaries(ds, splitter)
    length(boundaries) == PG21_EXPECTED_FOLDS || error(
        "Canonical splitter produced $(length(boundaries)) folds; expected $PG21_EXPECTED_FOLDS.")

    println("\n[1/4] Registering canonical components...")
    model_ids = Dict{String,Int}()
    for (name, model) in models
        model_ids[name] = save_model(
            db,
            name,
            model;
            description = PG21_MODEL_DESCRIPTIONS[name],
            tags = PG21_TAGS,
        )
    end
    splitter_id = save_splitter(
        db,
        "scottish_lower_2426_40fold",
        splitter;
        description = "Pooled League One/Two, seasons 24/25 and 25/26, 40-fold match-biweek walk-forward.",
        tags = PG21_TAGS,
    )
    sampler_id = save_sampler(
        db,
        "queued_nuts_4x800",
        sampler;
        description = "Four queued NUTS chains, 800 warmup plus 800 retained draws per fold.",
        tags = PG21_TAGS,
    )
    book_id = save_book_spec(
        db,
        "scottish_lower_closing_main",
        book_spec;
        description = "Closing 1X2, Over/Under 2.5 and BTTS book with 2% commission and Baker-McHale shrinkage.",
        tags = PG21_TAGS,
    )
    policy_id = save_policy_spec(
        db,
        "scottish_lower_quarter_kelly",
        policy_spec;
        description = "30% flat trust, slate drawdown 23, 20% cap, daily settlement grouping.",
        tags = PG21_TAGS,
    )

    println("\n[2/4] Registering assembled FitConfig recipes...")
    fit_hashes = Dict{String,String}()
    for name in PG21_MODEL_NAMES
        fit_hashes[name] = save_config(
            db,
            name * "_fit",
            fit_configs[name];
            description = PG21_MODEL_DESCRIPTIONS[name] * " Canonical 40-fold FitConfig.",
            tags = PG21_TAGS,
        )
    end

    println("\n[3/4] Importing completed Fits (deterministic synthetic fallback when absent)...")
    fits = Dict{String,Fit}()
    run_ids = Dict{String,UUID}()
    sources = Dict{String,Symbol}()
    portfolio_ids = Dict{String,UUID}()
    for (model_index, name) in enumerate(PG21_MODEL_NAMES)
        source = pg21_source_fit(name, fit_configs[name], model_index, ds, boundaries)
        fit = source.fit
        length(fit) == PG21_EXPECTED_FOLDS || error(
            "$name has $(length(fit)) folds; expected $PG21_EXPECTED_FOLDS.")
        run_id = save_fit(fit, db)
        scores = evaluate_predictions(fit, ds; threaded = true)
        pg21_persist_scores!(db, run_id, scores)

        portfolio = pg21_save_portfolio!(db, run_id, fit, ds, book_spec, policy_spec)
        isfinite(portfolio.result.summary.sharpe_ann) || error(
            "$name portfolio Sharpe is not finite; ingestion cannot claim a valid portfolio summary.")

        fits[name] = fit
        run_ids[name] = run_id
        sources[name] = source.source
        portfolio_ids[name] = portfolio.portfolio_id
        println("  $name: source=$(source.source), run=$run_id, " *
                "portfolio=$(portfolio.portfolio_id), bets=$(portfolio.result.summary.n_bets)")
    end

    println("\n[4/4] Sync summary")
    println("  model IDs       : ", model_ids)
    println("  splitter ID     : ", splitter_id)
    println("  sampler ID      : ", sampler_id)
    println("  book/policy IDs : ", (book_id, policy_id))
    println("  fit hashes      : ", fit_hashes)
    println("  sources         : ", sources)
    println("  PostgreSQL sync complete; no MCMC was launched.")

    return (; db, ds, models, splitter, sampler, book_spec, policy_spec, fit_configs,
            fits, run_ids, portfolio_ids, model_ids, splitter_id, sampler_id,
            book_id, policy_id, fit_hashes, sources)
end

if abspath(PROGRAM_FILE) == @__FILE__
    pg21_sync_to_postgres()
end
