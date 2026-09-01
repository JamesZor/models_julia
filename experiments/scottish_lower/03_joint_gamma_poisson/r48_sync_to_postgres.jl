# ==============================================================================
# Scottish Lower Two-Arm Joint 24/26 Grid — PostgreSQL Ingestion
# ==============================================================================
#
# A persistence runner. It does NOT launch MCMC. It imports the eight completed r46 Fits,
# registers their canonical components, and persists posteriors and betting portfolios into
# the `scottish_lower_joint_2426` experiment namespace.
#
# WHERE THE CONFIG TRUTH COMES FROM. Unlike r21, this script does NOT re-declare the model
# recipes. Every component is registered from `fit.config` on the LOADED Fit — the exact
# object r46 sampled with. A re-declared recipe can drift from what actually ran and the
# database would then attest to a model nobody fitted; reading it off the artifact makes
# that impossible by construction.
#
# NO SYNTHETIC FALLBACK. All eight arms completed on mcmc-beast, so a missing Fit is a real
# problem (wrong path, failed run, incomplete copy) and this script stops rather than
# inventing deterministic stand-ins. Synthetic rows in a production namespace are read as
# real by everything downstream.
#
# Usage (from the repository root):
#
#   julia --project -t 16 experiments/scottish_lower/03_joint_gamma_poisson/r48_sync_to_postgres.jl
#
# Credentials are resolved by PostgresStorage from BF_EXPERIMENTS_DB_URL or libpq's
# ~/.pgpass. No connection string is printed or embedded here.
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Distributions
using LibPQ
using MCMCChains
using Printf
using Statistics
using UUIDs

const PG48_EXPERIMENT     = "scottish_lower_joint_2426"
const PG48_SAVE_ROOT      = get(ENV, "PG48_SAVE_ROOT", "./data/scottish_lower_2426_joint")
const PG48_TAGS           = ["production", "joint", "gamma_poisson", "scottish_lower", "2426"]
const PG48_EXPECTED_FOLDS = 40
const PG48_CONTROL        = "m00_poisson_control"

const PG48_ARMS = [
    "m00_joint_baseline",
    "m02_joint_squad_wealth",
    "m03_joint_distance",
    "m04_joint_wealth_distance",
    "m05_joint_production_wealth",
    "m07_joint_bench_depth",
    "m08_joint_composite",
    PG48_CONTROL,
]

const PG48_DESCRIPTIONS = Dict(
    "m00_joint_baseline" =>
        "Two-arm joint baseline: pxG ~ Gamma(ν, μ/ν) on BBC live-text matches (23/24+) and " *
        "goals ~ Poisson(κ·μ) on all matches, over a global intercept, 180-day time decay " *
        "and global home advantage.",
    "m02_joint_squad_wealth" =>
        "Joint baseline plus point-in-time raw starting-XI squad wealth from Transfermarkt.",
    "m03_joint_distance" =>
        "Joint baseline plus away-ground travel-distance fatigue.",
    "m04_joint_wealth_distance" =>
        "Joint baseline plus raw starting-XI wealth and travel distance.",
    "m05_joint_production_wealth" =>
        "Joint baseline plus age-adjusted production wealth using RichardsSigmoid(23, 0.80, 2).",
    "m07_joint_bench_depth" =>
        "Joint baseline plus log bench-depth differential from substitute lineups.",
    "m08_joint_composite" =>
        "Joint baseline plus age-adjusted production wealth and bench depth.",
    PG48_CONTROL =>
        "SINGLE-ARM CONTROL. The identical spine with NO Gamma arm — goals ~ Poisson(μ) " *
        "only. This is the decision comparison for the whole experiment: it isolates the " *
        "second likelihood from every other difference.",
)

# ==============================================================================
# 1. Book and policy — identical to r46/r47 and to experiments 01 and 02
# ==============================================================================

pg48_book_spec() = BookSpec(
    markets   = Data.MarketConfig([
        Data.Market1X2(),
        Data.MarketOverUnder(2.5),
        Data.MarketBTTS(),
    ]),
    price     = DeArb(),
    allocator = KellyLogUtility(),
    shrink    = BakerMcHale(),
    exec      = ExecutionConfig(
        commission          = PerBetCommission(0.02),
        budget              = 0.99,
        min_selection_stake = 0.001,
    ),
)

pg48_policy_spec() = PolicySpec(
    trust    = FlatTrust(0.30),
    risk     = SlateDrawdown(23.0),
    cap      = FixedCap(0.20),
    grouping = DailySlate(),
)

# ==============================================================================
# 2. Loading the completed Fits
# ==============================================================================

"Resolve a timestamped fit directory to the one holding `results.jld2`."
function pg48_resolve_fit_dir(path::AbstractString)
    isfile(joinpath(path, "results.jld2")) && return path
    isdir(path) || error("No fit directory at $path — check PG48_SAVE_ROOT.")
    stamped = filter(d -> isfile(joinpath(path, d, "results.jld2")), readdir(path))
    isempty(stamped) && error("No results.jld2 under $path — the r46 run may be incomplete.")
    sort!(stamped; rev = true)
    return joinpath(path, first(stamped))
end

function pg48_load_fit(name::AbstractString)
    dir = pg48_resolve_fit_dir(joinpath(PG48_SAVE_ROOT, name))
    fit = load_fit(dir; quiet = true)
    length(fit) == PG48_EXPECTED_FOLDS || error(
        "$name has $(length(fit)) folds; expected $PG48_EXPECTED_FOLDS. Refusing to ingest " *
        "a partial grid into a production namespace.")
    return fit
end

# ==============================================================================
# 3. Relational score and portfolio helpers
# ==============================================================================

function pg48_persist_scores!(db::PostgresStorage, run_id::UUID, scores)
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

"Return an existing portfolio run for this (fit, book, policy) triple, or `nothing`."
function pg48_existing_portfolio(db::PostgresStorage, run_id::UUID, book_spec, policy_spec)
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

function pg48_save_portfolio!(db::PostgresStorage, run_id::UUID, fit, ds, book_spec, policy_spec)
    existing = pg48_existing_portfolio(db, run_id, book_spec, policy_spec)
    existing === nothing || return (portfolio_id = existing,
                                    result = load_portfolio_db(existing, db),
                                    reused = true)
    result, _, _ = run_portfolio_simulation(
        book_spec, policy_spec, fit, ds.odds, ds;
        bootstrap = false, require_converged = false, quiet = true,
    )
    portfolio_id = save_portfolio_db(
        result, run_id, db;
        book_spec, policy_spec,
        metadata = (; ingestion = "r48_sync_to_postgres"),
    )
    return (; portfolio_id, result, reused = false)
end

# ==============================================================================
# 4. Ingestion workflow
# ==============================================================================

function pg48_sync_to_postgres()
    println("\n", "="^100)
    println(" SCOTTISH LOWER TWO-ARM JOINT 24/26 — POSTGRESQL SYNC (NO MCMC)")
    println("="^100)

    db = PostgresStorage(PG48_EXPERIMENT)
    ensure_schema!(db)
    ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)

    book_spec = pg48_book_spec()
    policy_spec = pg48_policy_spec()

    println("\n[1/5] Loading completed r46 Fits from $(PG48_SAVE_ROOT) ...")
    fits = Dict{String,Any}()
    for name in PG48_ARMS
        fits[name] = pg48_load_fit(name)
        @printf("  %-30s %d folds\n", name, length(fits[name]))
    end

    # Config truth: read off the artifacts, never re-declared. The splitter and sampler are
    # shared by construction in r46, so registering them once from the control arm and
    # asserting equality across the rest is a real check, not a formality.
    reference = fits[PG48_CONTROL].config
    for name in PG48_ARMS
        cfg = fits[name].config
        cfg.splitter == reference.splitter || error(
            "$name was fitted with a different splitter than $PG48_CONTROL; the arms are " *
            "not comparable and must not be ingested as one experiment.")
        cfg.sampler == reference.sampler || error(
            "$name was fitted with a different sampler than $PG48_CONTROL.")
    end
    println("  splitter and sampler verified identical across all $(length(PG48_ARMS)) arms")

    println("\n[2/5] Registering canonical components (from fit.config) ...")
    model_ids = Dict{String,Int}()
    for name in PG48_ARMS
        model_ids[name] = save_model(
            db, name, fits[name].config.model;
            description = PG48_DESCRIPTIONS[name], tags = PG48_TAGS)
    end
    splitter_id = save_splitter(
        db, "scottish_lower_2426_40fold", reference.splitter;
        description = "Pooled League One/Two, seasons 24/25 and 25/26, 40-fold match-biweek " *
                      "walk-forward, 38 scored folds, 710 out-of-sample fixtures.",
        tags = PG48_TAGS)
    sampler_id = save_sampler(
        db, "queued_nuts_4x800", reference.sampler;
        description = "Four queued NUTS chains, 800 warmup plus 800 retained draws per fold.",
        tags = PG48_TAGS)
    book_id = save_book_spec(
        db, "scottish_lower_closing_main", book_spec;
        description = "1X2, Over/Under 2.5 and BTTS book with 2% commission and Baker-McHale shrinkage.",
        tags = PG48_TAGS)
    policy_id = save_policy_spec(
        db, "scottish_lower_quarter_kelly", policy_spec;
        description = "30% flat trust, slate drawdown 23, 20% cap, daily settlement grouping.",
        tags = PG48_TAGS)

    println("\n[3/5] Registering assembled FitConfig recipes ...")
    fit_hashes = Dict{String,String}()
    for name in PG48_ARMS
        fit_hashes[name] = save_config(
            db, name * "_fit", fits[name].config;
            description = PG48_DESCRIPTIONS[name] * " Canonical 40-fold FitConfig.",
            tags = PG48_TAGS)
    end

    println("\n[4/5] Persisting Fits, scores and portfolios ...")
    run_ids = Dict{String,UUID}()
    portfolio_ids = Dict{String,UUID}()
    rows = NamedTuple[]
    for name in PG48_ARMS
        fit = fits[name]
        run_id = save_fit(fit, db)
        scores = evaluate_predictions(fit, ds; threaded = true)
        pg48_persist_scores!(db, run_id, scores)

        portfolio = pg48_save_portfolio!(db, run_id, fit, ds, book_spec, policy_spec)
        isfinite(portfolio.result.summary.sharpe_ann) || error(
            "$name portfolio Sharpe is not finite; refusing to claim a valid portfolio summary.")

        run_ids[name] = run_id
        portfolio_ids[name] = portfolio.portfolio_id
        s = portfolio.result.summary
        push!(rows, (name = name, run_id = run_id, portfolio_id = portfolio.portfolio_id,
                     logloss = scores.model.logloss, brier = scores.model.brier,
                     rps = scores.model.rps, n_bets = s.n_bets,
                     total_return_pct = s.total_return_pct, roi = s.roi,
                     mdd = s.mdd, sharpe = s.sharpe_ann, reused = portfolio.reused))
        @printf("  %-30s run=%s portfolio=%s bets=%d%s\n", name, string(run_id),
                string(portfolio.portfolio_id), s.n_bets, portfolio.reused ? " (reused)" : "")
    end

    println("\n[5/5] Verifying lossless retrieval ...")
    for name in PG48_ARMS
        restored = load_fit(db, run_ids[name])
        length(restored) == PG48_EXPECTED_FOLDS || error(
            "$name round-tripped with $(length(restored)) folds; expected $PG48_EXPECTED_FOLDS.")
    end
    println("  all $(length(PG48_ARMS)) fits round-tripped at $PG48_EXPECTED_FOLDS folds")

    println("\n", "="^130)
    println(" INGESTION SUMMARY — experiment namespace: $PG48_EXPERIMENT")
    println("="^130)
    @printf(" %-30s | %8s | %7s | %6s | %9s | %8s | %7s | %-36s\n",
            "Model", "LogLoss", "Brier", "Bets", "Return %", "Sharpe", "MDD %", "run_id")
    println("-"^130)
    for r in rows
        @printf(" %-30s | %8.4f | %7.4f | %6d | %+8.2f%% | %8.3f | %6.2f%% | %-36s\n",
                r.name, r.logloss, r.brier, r.n_bets, r.total_return_pct, r.sharpe, r.mdd,
                string(r.run_id))
    end
    println("="^130)
    println("  component IDs : models=", model_ids)
    println("  splitter=", splitter_id, "  sampler=", sampler_id,
            "  book=", book_id, "  policy=", policy_id)
    println("  portfolio IDs : ", portfolio_ids)
    println("  PostgreSQL sync complete; no MCMC was launched.")

    return (; db, ds, fits, run_ids, portfolio_ids, model_ids, splitter_id, sampler_id,
            book_id, policy_id, fit_hashes, rows)
end

if abspath(PROGRAM_FILE) == @__FILE__
    pg48_sync_to_postgres()
end
