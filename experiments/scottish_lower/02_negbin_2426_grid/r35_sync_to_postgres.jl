# ==============================================================================
# Scottish Lower Negative Binomial 24/26 Grid — PostgreSQL Ingestion
# ==============================================================================
#
# Persistence and portfolio-reconstruction runner for the `scottish_lower_negbin_2426`
# namespace. It does NOT launch MCMC. It registers the five canonical r31 recipes, imports
# the completed local Fits when present, and otherwise creates deterministic synthetic Fits
# so every PostgreSQL persistence path can be exercised offline without competing with
# production sampling.
#
# Sibling of `01_poisson_2426_grid/r21_sync_to_postgres.jl`; read that first.
#
# TWO PRICE SOURCES ARE PERSISTED PER MODEL, and this is deliberate. Experiments 01 and 03
# priced their portfolios off `ds.odds` (bookmaker close, overround intact); experiments 05
# and 06 priced off the Betfair exchange close. Those two sources reach opposite conclusions
# from identical posteriors — see `03_joint_gamma_poisson/NOTES.md` §2. Ingesting only one
# would make experiment 02 comparable to half the database, so both are written and tagged
# with `metadata->>'odds_source'`.
#
# Usage (from the repository root):
#
#   julia --project -t 16 experiments/scottish_lower/02_negbin_2426_grid/r35_sync_to_postgres.jl
#
# Connection credentials are resolved by PostgresStorage from BF_EXPERIMENTS_DB_URL or
# libpq's ~/.pgpass. No connection string is printed or embedded here.

using BayesianFootball
using DataFrames
using Dates
using Distributions
using JSON3
using LibPQ
using LinearAlgebra
using MCMCChains
using Random
using Statistics
using ThreadPinning
using UUIDs

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

const PG35_PORTFOLIO = BayesianFootball.Portfolio

const PG35_EXPERIMENT = "scottish_lower_negbin_2426"
const PG35_TAGS = ["production", "negbin", "scottish_lower", "2426"]
const PG35_EXPECTED_FOLDS = 40
const PG35_SYNTHETIC_DRAWS = 160
const PG35_SYNTHETIC_CHAINS = 4

# The r31 production save root on mcmc-beast, then the fallbacks a local checkout may hold.
# `BF_NEGBIN_FIT_ROOT` overrides all of them for an ad-hoc relocation.
const PG35_FIT_ROOTS = String[
    get(ENV, "BF_NEGBIN_FIT_ROOT", ""),
    "/root/BayesianFootball/experiments/scottish_lower_2426_negbin",
    joinpath(@__DIR__, "..", "..", "scottish_lower_2426_negbin"),
    "./data/scottish_lower_2426_negbin",
]

const PG35_MODEL_NAMES = [
    "m00_negbin_baseline",
    "m02_negbin_wealth",
    "m03_negbin_distance",
    "m04_negbin_joint",
    "m05_negbin_production_wealth",
]

const PG35_MODEL_DESCRIPTIONS = Dict(
    "m00_negbin_baseline" =>
        "Canonical Scottish Lower Negative Binomial baseline: global intercept, 180-day time " *
        "decay, global home advantage, and a NegBin observation with global dispersion.",
    "m02_negbin_wealth" =>
        "Canonical NegBin baseline plus point-in-time raw starting-XI squad wealth from Transfermarkt.",
    "m03_negbin_distance" =>
        "Canonical NegBin baseline plus away-ground travel-distance fatigue.",
    "m04_negbin_joint" =>
        "Canonical NegBin model with raw starting-XI wealth and travel distance.",
    "m05_negbin_production_wealth" =>
        "Canonical NegBin baseline plus age-adjusted production wealth using RichardsSigmoid(23, 0.80, 2).",
)

# ==============================================================================
# 1. Canonical recipes — kept visibly aligned with r31
# ==============================================================================

function pg35_models()
    m00 = CountModelBuilder(:m00_negbin_baseline) |>
        BayesianFootball.add(GlobalInterception()) |>
        BayesianFootball.add(TimeDecayDynamics(days_half_life = 180.0)) |>
        BayesianFootball.add(GlobalHomeAdvantage()) |>
        BayesianFootball.add(NegativeBinomialObservation(GlobalDispersion())) |>
        BayesianFootball.build

    m02 = CountModelBuilder(:m02_negbin_wealth) |>
        BayesianFootball.add(GlobalInterception()) |>
        BayesianFootball.add(TimeDecayDynamics(days_half_life = 180.0)) |>
        BayesianFootball.add(GlobalHomeAdvantage()) |>
        BayesianFootball.add(WealthCovariate(
            prior = truncated(Normal(0.10, 0.05), lower = 0.0))) |>
        BayesianFootball.add(NegativeBinomialObservation(GlobalDispersion())) |>
        BayesianFootball.build

    m03 = CountModelBuilder(:m03_negbin_distance) |>
        BayesianFootball.add(GlobalInterception()) |>
        BayesianFootball.add(TimeDecayDynamics(days_half_life = 180.0)) |>
        BayesianFootball.add(GlobalHomeAdvantage()) |>
        BayesianFootball.add(DistanceCovariate(
            prior = truncated(Normal(0.04, 0.03), lower = 0.0))) |>
        BayesianFootball.add(NegativeBinomialObservation(GlobalDispersion())) |>
        BayesianFootball.build

    m04 = CountModelBuilder(:m04_negbin_joint) |>
        BayesianFootball.add(GlobalInterception()) |>
        BayesianFootball.add(TimeDecayDynamics(days_half_life = 180.0)) |>
        BayesianFootball.add(GlobalHomeAdvantage()) |>
        BayesianFootball.add(WealthCovariate(
            prior = truncated(Normal(0.10, 0.05), lower = 0.0))) |>
        BayesianFootball.add(DistanceCovariate(
            prior = truncated(Normal(0.04, 0.03), lower = 0.0))) |>
        BayesianFootball.add(NegativeBinomialObservation(GlobalDispersion())) |>
        BayesianFootball.build

    m05 = CountModelBuilder(:m05_negbin_production_wealth) |>
        BayesianFootball.add(GlobalInterception()) |>
        BayesianFootball.add(TimeDecayDynamics(days_half_life = 180.0)) |>
        BayesianFootball.add(GlobalHomeAdvantage()) |>
        BayesianFootball.add(ProductionWealthCovariate(
            feature = ProductionWealthFeature(curve = RichardsSigmoid(23.0, 0.80, 2.0)),
            prior = truncated(Normal(0.10, 0.05), lower = 0.0),
        )) |>
        BayesianFootball.add(NegativeBinomialObservation(GlobalDispersion())) |>
        BayesianFootball.build

    return [
        "m00_negbin_baseline" => m00,
        "m02_negbin_wealth" => m02,
        "m03_negbin_distance" => m03,
        "m04_negbin_joint" => m04,
        "m05_negbin_production_wealth" => m05,
    ]
end

pg35_splitter() = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons = ["24/25", "25/26"],
    history_seasons = 2,
    dynamics_col = :match_biweek,
    warmup_period = 0,
    stop_early = true,
)

pg35_sampler() = QueuedNUTSConfig(
    n_samples = 800,
    n_warmup = 800,
    n_chains = 4,
    accept_rate = 0.65,
)

# 30% fractional Kelly on the three main markets. `FlatTrust(1.0)` is correct here and not a
# missing setting: the 0.30 scaling lives in the shrinkage stage, so a fractional trust on
# top would compound to 9% Kelly.
pg35_book_spec() = BookSpec(
    markets = Data.MarketConfig(Data.AbstractMarket[
        Data.Market1X2(),
        Data.MarketOverUnder(2.5),
        Data.MarketBTTS(),
    ]),
    price = DeArb(),
    allocator = KellyLogUtility(),
    shrink = PG35_PORTFOLIO.FractionalKelly(0.30),
    exec = ExecutionConfig(
        commission = PerBetCommission(0.02),
        budget = 0.99,
        min_selection_stake = 0.001,
    ),
)

pg35_policy_spec() = PolicySpec(
    trust = FlatTrust(1.0),
    risk = SlateDrawdown(20.0),
    cap = FixedCap(0.25),
    grouping = DailySlate(),
)

function pg35_fit_configs(models, splitter, sampler)
    configs = Dict{String,FitConfig}()
    for (name, model) in models
        configs[name] = FitConfig(
            name = name,
            model = model,
            splitter = splitter,
            sampler = sampler,
            execution = QueuedExecution(max_concurrent_tasks = Threads.nthreads()),
            tags = copy(PG35_TAGS),
            description = PG35_MODEL_DESCRIPTIONS[name],
            save_dir = joinpath(pg35_fit_root(), name),
        )
    end
    return configs
end

# ==============================================================================
# 2. Existing-fit discovery and deterministic local fallback
# ==============================================================================

"First `PG35_FIT_ROOTS` entry that exists, or the beast path as the nominal default."
function pg35_fit_root()
    for root in PG35_FIT_ROOTS
        isempty(strip(root)) && continue
        isdir(root) && return abspath(root)
    end
    return PG35_FIT_ROOTS[2]
end

"""
    pg35_latest_completed_fit(name) -> Fit or nothing

r31 wrote each model's artefacts directly into `<root>/<name>/`, so the fit is loaded from
that directory rather than scanned for with `list_fits` (which enumerates SUBdirectories and
would return the `checkpoints/` folder). The nested layout r21 uses is still accepted as a
fallback, because a re-run under `FileStorage` would produce it.
"""
function pg35_latest_completed_fit(name::String)
    directory = joinpath(pg35_fit_root(), name)
    isdir(directory) || return nothing

    if isfile(joinpath(directory, "results.jld2"))
        fit = load_fit(directory; quiet = true)
        if length(fit) == PG35_EXPECTED_FOLDS && fit.latents isa CountLatents
            return fit
        end
    end

    for candidate in Iterators.reverse(list_fits(directory; quiet = true))
        isfile(joinpath(candidate.path, "results.jld2")) || continue
        fit = load_fit(candidate.path; quiet = true)
        if length(fit) == PG35_EXPECTED_FOLDS && fit.latents isa CountLatents
            return fit
        end
    end
    return nothing
end

"A NegBin chain carries `disp.log_r`; the synthetic stand-in must too, or the fallback would
exercise a Poisson-shaped persistence path under a negative-binomial name."
function pg35_synthetic_chain(seed::Int)
    rng = Xoshiro(seed)
    n = PG35_SYNTHETIC_DRAWS
    n_chains = PG35_SYNTHETIC_CHAINS
    values = Array{Float64,3}(undef, n, 6, n_chains)
    for chain in 1:n_chains
        values[:, 1, chain] .= randn(rng, n)
        values[:, 2, chain] .= 0.5 .* randn(rng, n)
        values[:, 3, chain] .= 1.6 .+ 0.15 .* randn(rng, n)   # disp.log_r ⇒ r ≈ 5
        values[:, 4, chain] .= 0.0
        values[:, 5, chain] .= 4.0
        values[:, 6, chain] .= 100.0 .+ 3.0 .* randn(rng, n)
    end
    return Chains(
        values,
        [:synthetic_intercept, :synthetic_home_advantage, Symbol("disp.log_r"),
         :numerical_error, :tree_depth, :hamiltonian_energy],
        (parameters = [:synthetic_intercept, :synthetic_home_advantage, Symbol("disp.log_r")],
         internals = [:numerical_error, :tree_depth, :hamiltonian_energy]),
    )
end

function pg35_oos_match_ids(ds::Data.DataStore, boundaries, splitter)
    ids = Int[]
    for boundary in boundaries
        fixtures = Data.get_next_matches(ds, boundary, splitter)
        append!(ids, Int.(fixtures.match_id))
    end
    allunique(ids) || error("Synthetic fallback: walk-forward OOS match IDs are not unique.")
    return ids
end

function pg35_synthetic_latents(ids::Vector{Int}, model_index::Int, n_draws::Int)
    lambda_home = Matrix{Float64}(undef, length(ids), n_draws)
    lambda_away = Matrix{Float64}(undef, length(ids), n_draws)
    r_home = Matrix{Float64}(undef, length(ids), n_draws)
    r_away = Matrix{Float64}(undef, length(ids), n_draws)
    home_shift = 0.04 * (model_index - 1)
    away_shift = 0.015 * (model_index - 1)
    for draw in 1:n_draws
        draw_phase = 0.017 * draw
        for row in eachindex(ids)
            fixture_phase = 0.013 * ids[row]
            lambda_home[row, draw] = 1.62 + home_shift + 0.08 * sin(fixture_phase + draw_phase)
            lambda_away[row, draw] = 1.04 + away_shift + 0.07 * cos(fixture_phase - draw_phase)
            r_home[row, draw] = 5.0 + 0.35 * sin(draw_phase)
            r_away[row, draw] = 5.0 + 0.35 * cos(draw_phase)
        end
    end
    return CountLatents(ids, lambda_home, lambda_away, (; r_h = r_home, r_a = r_away))
end

function pg35_synthetic_fit(name::String, config::FitConfig, model_index::Int,
                            ds::Data.DataStore, boundaries)
    length(boundaries) == PG35_EXPECTED_FOLDS || error(
        "Synthetic fallback expected $PG35_EXPECTED_FOLDS folds, got $(length(boundaries)).")
    chains = [pg35_synthetic_chain(20_000 * model_index + fold)
              for fold in eachindex(boundaries)]
    folds = [FoldFit(fold, chains[fold], boundaries[fold][2])
             for fold in eachindex(boundaries)]
    diagnostics = audit_convergence(folds; max_depth = 10)
    ids = pg35_oos_match_ids(ds, boundaries, config.splitter)
    n_draws = size(first(chains), 1) * size(first(chains), 3)
    latents = pg35_synthetic_latents(ids, model_index, n_draws)
    metadata = FitMetadata(now(), 0.0, VERSION, Threads.nthreads(), "synthetic-no-mcmc")
    save_path = joinpath(config.save_dir, name * "_synthetic")
    return Fit(config, folds, latents, diagnostics, metadata, save_path)
end

function pg35_source_fit(name::String, config::FitConfig, model_index::Int,
                         ds::Data.DataStore, boundaries)
    disk_fit = pg35_latest_completed_fit(name)
    disk_fit === nothing || return (fit = disk_fit, source = :disk)
    return (fit = pg35_synthetic_fit(name, config, model_index, ds, boundaries),
            source = :synthetic)
end

# ==============================================================================
# 3. Price sources
# ==============================================================================

"Betfair exchange close, time-weighted over the last 20 minutes before kickoff."
function pg35_betfair_closing_odds(ds::Data.DataStore)
    raw = Data.summarize_odds(ds.betfair_odds, Data.TWAEstimator(); window = (-20.0, 0.0))
    odds = DataFrame(
        match_id = Int.(raw.match_id),
        market_name = String.(raw.market_name),
        market_line = Float64.(raw.market_line),
        selection = Symbol.(raw.selection),
        odds_close = Float64.(raw.odds),
    )
    filter!(row -> isfinite(row.odds_close) && row.odds_close > 1.0, odds)
    sort!(odds, [:match_id, :market_name, :market_line, :selection])
    return odds
end

# `odds_source` is part of the identity of a portfolio row, not decoration: the same fit under
# the same BookSpec and PolicySpec produces a different `PortfolioResult` for each source, so
# deduplication has to key on it too or the second write would be discarded as a duplicate.
const PG35_PRICE_SOURCES = [
    (tag = "betfair_twa_minus20_to_close", label = "Betfair exchange close (TWA −20m→0)"),
    (tag = "bookmaker_close", label = "Bookmaker close from ds.odds"),
]

# ==============================================================================
# 4. Relational score and portfolio helpers
# ==============================================================================

function pg35_persist_scores!(db::PostgresStorage, run_id::UUID, scores::PredictionScores)
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

function pg35_existing_portfolio(db::PostgresStorage, run_id::UUID, book_spec, policy_spec,
                                 odds_source::String)
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
              AND metadata->>'odds_source' = \$4
            ORDER BY id DESC
            LIMIT 1;
        """, (string(run_id), book_hash, policy_hash, odds_source))
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

function pg35_save_portfolio!(db::PostgresStorage, run_id::UUID, fit::Fit, name::String,
                              odds::AbstractDataFrame, odds_source::String,
                              ds::Data.DataStore, book_spec, policy_spec)
    existing = pg35_existing_portfolio(db, run_id, book_spec, policy_spec, odds_source)
    existing === nothing || return (portfolio_id = existing,
                                    result = load_portfolio_db(existing, db),
                                    reused = true)
    result, _, _ = run_portfolio_simulation(
        book_spec,
        policy_spec,
        fit,
        odds,
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
        metadata = (; candidate = name, odds_source = odds_source,
                      ingestion = "r35_sync_to_postgres"),
    )
    return (; portfolio_id, result, reused = false)
end

# ==============================================================================
# 5. Ingestion workflow
# ==============================================================================

function pg35_sync_to_postgres()
    println("\n", "="^92)
    println(" SCOTTISH LOWER NEGATIVE BINOMIAL 24/26 — POSTGRESQL SYNC (NO MCMC)")
    println("="^92)

    db = PostgresStorage(PG35_EXPERIMENT)
    ensure_schema!(db)
    println("  storage : ", db)
    println("  fit root: ", pg35_fit_root())
    ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)

    models = pg35_models()
    splitter = pg35_splitter()
    sampler = pg35_sampler()
    book_spec = pg35_book_spec()
    policy_spec = pg35_policy_spec()
    fit_configs = pg35_fit_configs(models, splitter, sampler)
    boundaries = Data.create_id_boundaries(ds, splitter)
    length(boundaries) == PG35_EXPECTED_FOLDS || error(
        "Canonical splitter produced $(length(boundaries)) folds; expected $PG35_EXPECTED_FOLDS.")

    bf_odds = pg35_betfair_closing_odds(ds)
    price_frames = Dict(
        "betfair_twa_minus20_to_close" => bf_odds,
        "bookmaker_close" => ds.odds,
    )
    println("  Betfair rows: $(nrow(bf_odds)) across " *
            "$(length(unique(bf_odds.match_id))) matches")

    println("\n[1/4] Registering canonical components...")
    model_ids = Dict{String,Int}()
    for (name, model) in models
        model_ids[name] = save_model(
            db,
            name,
            model;
            description = PG35_MODEL_DESCRIPTIONS[name],
            tags = PG35_TAGS,
        )
    end
    splitter_id = save_splitter(
        db,
        "scottish_lower_negbin_2426_40fold",
        splitter;
        description = "Pooled League One/Two, seasons 24/25 and 25/26, 40-fold match-biweek walk-forward.",
        tags = PG35_TAGS,
    )
    sampler_id = save_sampler(
        db,
        "queued_nuts_4x800_negbin",
        sampler;
        description = "Four queued NUTS chains, 800 warmup plus 800 retained draws per fold.",
        tags = PG35_TAGS,
    )
    book_id = save_book_spec(
        db,
        "negbin_fractional_kelly_main_markets",
        book_spec;
        description = "30% fractional Kelly on 1X2, Over/Under 2.5 and BTTS with 2% exchange commission.",
        tags = [PG35_TAGS; "portfolio"],
    )
    policy_id = save_policy_spec(
        db,
        "negbin_slate_drawdown_20_cap_25",
        policy_spec;
        description = "Unit flat trust, slate drawdown 20, 25% exposure cap, daily settlement grouping.",
        tags = [PG35_TAGS; "portfolio"],
    )

    println("\n[2/4] Registering assembled FitConfig recipes...")
    fit_hashes = Dict{String,String}()
    for name in PG35_MODEL_NAMES
        fit_hashes[name] = save_config(
            db,
            name * "_fit",
            fit_configs[name];
            description = PG35_MODEL_DESCRIPTIONS[name] * " Canonical 40-fold FitConfig.",
            tags = PG35_TAGS,
        )
    end

    println("\n[3/4] Importing completed Fits (deterministic synthetic fallback when absent)...")
    fits = Dict{String,Fit}()
    run_ids = Dict{String,UUID}()
    sources = Dict{String,Symbol}()
    portfolio_ids = Dict{String,Dict{String,UUID}}()
    for (model_index, name) in enumerate(PG35_MODEL_NAMES)
        source = pg35_source_fit(name, fit_configs[name], model_index, ds, boundaries)
        fit = source.fit
        length(fit) == PG35_EXPECTED_FOLDS || error(
            "$name has $(length(fit)) folds; expected $PG35_EXPECTED_FOLDS.")
        observation_family(fit.latents) == :negbin || error(
            "$name reconstructed $(observation_family(fit.latents)) latents; expected :negbin. " *
            "A NegBin grid whose latents carry no dispersion would be silently ingested as Poisson.")
        run_id = save_fit(fit, db)
        scores = evaluate_predictions(fit, ds; threaded = true)
        pg35_persist_scores!(db, run_id, scores)

        per_source = Dict{String,UUID}()
        for price in PG35_PRICE_SOURCES
            portfolio = pg35_save_portfolio!(
                db, run_id, fit, name, price_frames[price.tag], price.tag,
                ds, book_spec, policy_spec)
            isfinite(portfolio.result.summary.sharpe_ann) || error(
                "$name portfolio Sharpe on $(price.tag) is not finite; ingestion cannot " *
                "claim a valid portfolio summary.")
            per_source[price.tag] = portfolio.portfolio_id
            println("  $name [$(price.tag)]: portfolio=$(portfolio.portfolio_id), " *
                    "bets=$(portfolio.result.summary.n_bets), " *
                    "return=$(round(portfolio.result.summary.total_return_pct, digits = 2))%, " *
                    "reused=$(portfolio.reused)")
        end

        fits[name] = fit
        run_ids[name] = run_id
        sources[name] = source.source
        portfolio_ids[name] = per_source
        println("  $name: source=$(source.source), run=$run_id, " *
                "logloss=$(round(scores.model.logloss, digits = 4))")
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
            book_id, policy_id, fit_hashes, sources, bf_odds)
end

if abspath(PROGRAM_FILE) == @__FILE__
    pg35_sync_to_postgres()
end
