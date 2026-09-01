using Test
using BayesianFootball
using BayesianFootball: Data, Training, Portfolio
using DataFrames
using Dates
using LibPQ
using MCMCChains
using Random
using UUIDs

# Loads canonical Poisson-grid constructors without executing the runner's guarded main.
include(joinpath(@__DIR__, "..", "experiments", "scottish_lower",
                 "01_poisson_2426_grid", "r21_sync_to_postgres.jl"))

struct DBStorageMockModel <: BayesianFootball.AbstractFootballModel end

function db_storage_fit(root::AbstractString)
    chain = Chains(randn(Xoshiro(91), 40, 2, 2), [:a, :b])
    folds = [(BayesianFootball.FeatureSet(:n_teams => 2),
              Data.SplitMetaData(1, "23/24", "24/25", 1, 1, 0))]
    config = FitConfig(name = "db_storage_test", model = DBStorageMockModel(),
                       splitter = Data.CVConfig(target_seasons = ["24/25"]),
                       sampler = ReplaySampler([chain]), execution = SequentialExecution(),
                       save_dir = String(root))
    base = fit_model(config; feature_sets = folds, quiet = true)
    latents = CountLatents([101, 102],
                           reshape(collect(range(0.8, 1.8; length = 6400)), 2, 3200),
                           reshape(collect(range(0.6, 1.6; length = 6400)), 2, 3200))
    return Fit(base.config, base.folds, latents, base.diagnostics, base.metadata, base.save_path)
end

db_storage_count_model() = build_count_model(
    :m00_joint_baseline, GlobalInterception(), StaticZeroDynamics(), GlobalHomeAdvantage())

function db_storage_portfolio_result()
    bets = DataFrame(match_id = [101, 102], date = [Date(2025, 1, 1), Date(2025, 1, 1)],
                     family = ["1X2_home", "O/U 2.5_over_25"],
                     selection = [:home, :over_25], odds = [2.0, 1.9],
                     stake = [0.05, 0.04], pnl = [0.05, -0.04], payoff = [1.0, -1.0],
                     p_model = [0.55, 0.56], p_market = [0.50, 0.53])
    trajectory = Portfolio.Trajectory([1.0, 1.01], [Date(2025, 1, 1)], [0.01], [0.8],
                                      [0.09], 0, 0.09, 0.01, bets)
    states = [DailyState(1, Date(2025, 1, 1), 2, 2, 1000.0, 1010.0,
                         0.09, 0.01, 0.09, 0.8, false)]
    summary = Portfolio.portfolio_summary(states, trajectory, 1000.0;
                                          stake_1x2 = 0.05, pl_1x2 = 0.05, n_wins = 1)
    return PortfolioResult(states, summary, NamedTuple(), nothing, trajectory,
                           DataFrame(), true, String[])
end

@testset "Experiment database storage" begin
    @testset "FileStorage" begin
        fit = db_storage_fit(mktempdir())
        root = mktempdir()
        path = save_fit(fit, FileStorage(root); quiet = true)
        @test dirname(path) == root
        @test isfile(joinpath(path, "results.jld2"))
        @test isfile(joinpath(path, "config.json"))
        @test load_fit(path; quiet = true).latents.λ_home == fit.latents.λ_home
    end

    @testset "Zstd count-draw round trip" begin
        home = collect(range(0.5, 2.0; length = 3200))
        away = reverse(home)
        obs = (; r_h = fill(4.0, 3200), r_a = fill(5.0, 3200))
        blob = compress_draws(home, away, obs)
        @test blob[1:4] != UInt8[0x42, 0x46, 0x43, 0x4c] # compressed, not raw payload
        back = decompress_draws(blob)
        @test back.lambda_home == home
        @test back.lambda_away == away
        @test back.observation_params == obs
        @test_throws ErrorException compress_draws(home, away[1:end-1])
    end

    @testset "Schema and backend construction" begin
        schema = read(joinpath(@__DIR__, "..", "src", "training", "inference", "db",
                               "schema.sql"), String)
        for table in ("runs", "configs", "fold_results", "match_latents",
                      "portfolio_runs", "portfolio_bets", "config_registry")
            @test occursin("CREATE TABLE IF NOT EXISTS $table", schema)
        end
        @test occursin("idx_configs_config_hash", schema)
        @test occursin("idx_match_latents_match_id", schema)
        @test occursin(r"CREATE TABLE IF NOT EXISTS runs \(\s*id BIGSERIAL PRIMARY KEY"s,
                       schema)
        @test occursin(r"CREATE TABLE IF NOT EXISTS portfolio_runs \(\s*id BIGSERIAL PRIMARY KEY"s,
                       schema)
        @test occursin(r"CREATE TABLE IF NOT EXISTS config_registry \(\s*id BIGSERIAL PRIMARY KEY"s,
                       schema)
        @test occursin("idx_config_registry_name", schema)
        @test PostgresStorage("postgresql://localhost/x", "unit").experiment_name == "unit"

        secret_url = "postgresql://postgres:super_secret@db.internal:5544/experiments"
        from_env = withenv("BF_EXPERIMENTS_DB_URL" => secret_url) do
            PostgresStorage("masked")
        end
        shown = sprint(show, from_env)
        @test from_env.host == "db.internal"
        @test from_env.port == 5544
        @test from_env.dbname == "experiments"
        @test occursin("experiment=\"masked\"", shown)
        @test !occursin("super_secret", shown)
        @test !occursin("postgresql://", shown)

        fallback = withenv("BF_EXPERIMENTS_DB_URL" => "") do
            PostgresStorage("fallback")
        end
        @test fallback.conn_str ==
              "postgresql://postgres@mcmc-beast:5432/mcmc_experiments"

        from_keywords = PostgresStorage(host = "localhost", port = 5439,
                                        dbname = "local_experiments",
                                        experiment_name = "scottish")
        @test from_keywords.conn_str ==
              "postgresql://postgres@localhost:5439/local_experiments"
        @test !occursin("password", lowercase(sprint(show, from_keywords)))
    end

    # Integration is opt-in so the package suite remains hermetic. CI or a developer can point
    # this at a disposable local PostgreSQL container; this work package also runs it against the
    # provisioned mcmc-beast service.
    test_url = get(ENV, "BF_EXPERIMENTS_TEST_DB_URL", "")
    if isempty(test_url)
        @testset "PostgresStorage integration (set BF_EXPERIMENTS_TEST_DB_URL)" begin
            @test true
        end
    else
        @testset "PostgresStorage and DualStorage integration" begin
            experiment = "db_storage_test_$(uuid4())"
            storage = PostgresStorage(test_url, experiment)
            ensure_schema!(storage)
            fit = db_storage_fit(mktempdir())

            fit_hash = save_config(storage, "production-fit", fit.config;
                                   description = "single source of truth",
                                   tags = ["production", "poisson"])
            @test length(fit_hash) == 64
            truth_fit = load_fit_config(storage, "production-fit")
            @test truth_fit isa FitConfig
            @test string(truth_fit.model) == string(fit.config.model)
            @test load_fit_config(storage, fit_hash).name == fit.config.name

            book = BookSpec(markets = Data.MarketConfig(Data.AbstractMarket[
                                Data.Market1X2(), Data.MarketOverUnder(2.5)]),
                            shrink = Portfolio.NoShrinkage())
            policy = PolicySpec()
            portfolio_hash = save_config(storage, "production-portfolio", (book, policy);
                                         tags = ["production", "portfolio"])
            loaded_book, loaded_policy = load_portfolio_spec(storage, portfolio_hash)
            @test string(loaded_book) == string(book)
            @test string(loaded_policy) == string(policy)
            @test_throws ErrorException load_portfolio_spec(storage, "production-fit")
            @test_throws ErrorException load_fit_config(storage, "production-portfolio")

            listed = list_configs(storage)
            @test nrow(listed) == 2
            @test Set(listed.config_type) == Set(["fit", "portfolio"])
            @test nrow(list_configs(storage; tag = "poisson")) == 1
            @test nrow(list_configs(storage; config_type = "portfolio")) == 1
            @test list_configs(storage; tag = "production").tags isa Vector
            fit_config_id = only(listed.id[listed.config_type .== "fit"])
            @test load_fit_config(storage, fit_config_id).name == fit.config.name

            # Updating a name replaces its metadata without creating a second recipe row.
            @test save_config(storage, "production-fit", fit.config;
                              description = "promoted", tags = ["production"]) == fit_hash
            @test nrow(list_configs(storage; config_type = "fit")) == 1

            model = db_storage_count_model()
            splitter = Data.CVConfig(target_seasons = ["24/25"])
            sampler = BayesianFootball.Samplers.MAPConfig()
            model_id = save_model(storage, "m00_joint_baseline", model;
                                  description = "baseline count model", tags = ["baseline"])
            splitter_id = save_splitter(storage, "split_2426", splitter; tags = ["walkforward"])
            sampler_id = save_sampler(storage, "map_smoke", sampler; tags = ["smoke"])
            book_id = save_book_spec(storage, "main_book", book; tags = ["production"])
            policy_id = save_policy_spec(storage, "flat_policy", policy; tags = ["production"])
            @test all(id -> id isa Integer,
                      (model_id, splitter_id, sampler_id, book_id, policy_id))

            @test string(load_model(storage, model_id)) == string(model)
            @test string(load_model(storage, "m00_joint_baseline")) == string(model)
            @test string(load_model(storage, :m00_joint_baseline)) == string(model)
            @test string(load_splitter(storage, splitter_id)) == string(splitter)
            @test string(load_splitter(storage, "split_2426")) == string(splitter)
            @test string(load_sampler(storage, sampler_id)) == string(sampler)
            @test string(load_sampler(storage, "map_smoke")) == string(sampler)
            @test string(load_book_spec(storage, book_id)) == string(book)
            @test string(load_book_spec(storage, "main_book")) == string(book)
            @test string(load_policy_spec(storage, policy_id)) == string(policy)
            @test string(load_policy_spec(storage, "flat_policy")) == string(policy)
            @test_throws ErrorException load_model(storage, sampler_id)

            search_io = IOBuffer()
            search_result = search_configs(storage, "baseline"; io = search_io)
            @test nrow(search_result) == 1
            @test occursin("m00_joint_baseline", String(take!(search_io)))
            @test nrow(search_configs(storage, "tag=\"production\""; io = devnull)) == 4
            @test nrow(search_configs(storage, "config_type=:model"; io = devnull)) == 1

            show_io = IOBuffer()
            shown_model = show_config(storage, model_id; io = show_io)
            show_text = String(take!(show_io))
            @test shown_model isa BayesianFootball.Models.ComposableCountModel
            @test occursin("Architecture", show_text)
            @test occursin("interception", show_text)

            run_id = save_fit(fit, storage)
            @test run_id isa UUID
            @test save_fit(fit, storage) == run_id # config-hash deduplication
            @test length(config_hash(fit, storage)) == 64

            loaded = load_fit(run_id, storage)
            @test loaded isa Fit
            @test loaded.latents isa CountLatents
            @test loaded.latents.match_ids == fit.latents.match_ids
            @test loaded.latents.λ_home == fit.latents.λ_home
            @test Array(loaded[1].chain) == Array(fit[1].chain)

            @testset "Canonical Poisson 2426 registry and five-run round trip" begin
                grid_models = pg21_models()
                grid_splitter = pg21_splitter()
                grid_sampler = pg21_sampler()
                grid_configs = pg21_fit_configs(grid_models, grid_splitter, grid_sampler)
                grid_model_ids = Dict{String,Int}()
                grid_run_ids = Dict{String,UUID}()

                for (name, model) in grid_models
                    grid_model_ids[name] = save_model(
                        storage, name, model;
                        description = PG21_MODEL_DESCRIPTIONS[name], tags = PG21_TAGS)
                end
                canonical_splitter_id = save_splitter(
                    storage, "scottish_lower_2426_40fold", grid_splitter;
                    tags = PG21_TAGS)
                canonical_sampler_id = save_sampler(
                    storage, "queued_nuts_4x800", grid_sampler; tags = PG21_TAGS)
                for name in PG21_MODEL_NAMES
                    hash = save_config(storage, name * "_fit", grid_configs[name];
                                       tags = PG21_TAGS)
                    @test length(hash) == 64
                    canonical_fit = Fit(grid_configs[name], fit.folds, fit.latents,
                                        fit.diagnostics, fit.metadata, fit.save_path)
                    grid_run_ids[name] = save_fit(canonical_fit, storage)
                    round_trip = load_fit(storage, name)
                    @test Array(round_trip[1].chain) == Array(fit[1].chain)
                    @test round_trip.latents.λ_home == fit.latents.λ_home
                    @test config_hash(canonical_fit, storage) ==
                          config_hash(round_trip, storage)
                end

                @test length(unique(values(grid_run_ids))) == 5
                @test load_model(storage, "m05_production_wealth") isa
                      BayesianFootball.Models.ComposableCountModel
                @test string(load_model(storage, grid_model_ids["m05_production_wealth"])) ==
                      string(load_model(storage, "m05_production_wealth"))
                @test string(load_splitter(storage, canonical_splitter_id)) ==
                      string(grid_splitter)
                @test string(load_sampler(storage, canonical_sampler_id)) ==
                      string(grid_sampler)
                wealth_rows = search_configs(storage, "wealth"; io = devnull)
                wealth_names = Set(String.(wealth_rows.name))
                @test "m02_wealth" in wealth_names
                @test "m05_production_wealth" in wealth_names
            end

            conn_for_id = LibPQ.Connection(test_url)
            run_rows = try
                DataFrame(LibPQ.execute(conn_for_id,
                    "SELECT id FROM runs WHERE run_id = \$1::uuid;", (string(run_id),)))
            finally
                close(conn_for_id)
            end
            run_integer_id = Int(run_rows.id[1])
            @test load_fit(storage, run_integer_id).latents.λ_home == fit.latents.λ_home
            @test load_fit(storage, fit.config.name).latents.λ_home == fit.latents.λ_home
            @test load_fit(storage, Symbol(fit.config.name)).latents.λ_home == fit.latents.λ_home

            explore_io = IOBuffer()
            experiments = explore_experiments(storage; io = explore_io)
            @test storage.experiment_name in experiments.experiment_name
            @test occursin("Top LogLoss", String(take!(explore_io)))

            dual = DualStorage(FileStorage(mktempdir()), storage)
            both = save_fit(fit, dual; quiet = true)
            @test both.run_id == run_id
            @test isfile(joinpath(both.path, "results.jld2"))

            result = db_storage_portfolio_result()
            portfolio_id = save_portfolio_db(result, run_id, storage;
                                              book_spec_hash = "book-test",
                                              policy_spec_hash = "policy-test")
            reloaded = load_portfolio_db(portfolio_id, storage)
            @test reloaded isa PortfolioResult
            @test reloaded.summary.total_return_pct == result.summary.total_return_pct
            @test isequal(reloaded.trajectory.bets, result.trajectory.bets)

            conn = LibPQ.Connection(test_url)
            try
                rows = DataFrame(LibPQ.execute(conn,
                    "SELECT COUNT(*) AS n FROM configs WHERE config_id = \$1::uuid;",
                    (string(run_id),)))
                @test rows.n[1] == 1
                bets = DataFrame(LibPQ.execute(conn,
                    "SELECT COUNT(*) AS n FROM portfolio_bets WHERE portfolio_run_id = \$1::uuid;",
                    (string(portfolio_id),)))
                @test bets.n[1] == 2
                result_set = LibPQ.execute(conn, "DELETE FROM runs WHERE run_id = \$1::uuid;",
                                           (string(run_id),))
                close(result_set)
                config_set = LibPQ.execute(conn,
                    "DELETE FROM config_registry WHERE experiment_name = \$1;", (experiment,))
                close(config_set)
            finally
                close(conn)
            end
        end
    end
end
