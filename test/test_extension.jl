using Test
using BayesianFootball
using BayesianFootball: Data, Models, Training, Portfolio
using DataFrames
using Dates
using LibPQ
using MCMCChains
using Random
using UUIDs

struct ExtensionMockModel <: BayesianFootball.TypesInterfaces.AbstractPoissonModel end

mutable struct ExtensionMockSampler <: BayesianFootball.Samplers.AbstractSamplerConfig
    chains::Vector{Chains}
    sampled_folds::Vector{Int}
    n_chains::Int
end

function Training.Inference.sample_fold(::ExtensionMockModel, sampler::ExtensionMockSampler,
                                        feature_set, fold::Int;
                                        chain_id::Union{Int,Nothing} = nothing)
    push!(sampler.sampled_folds, fold)
    return sampler.chains[fold]
end

function Models.extract_latents(::ExtensionMockModel, chain::Chains,
                                fixtures::AbstractDataFrame, feature_set)
    ids = Int.(fixtures.match_id)
    draws = size(chain, 1) * size(chain, 3)
    lambda_home = Matrix{Float64}(undef, length(ids), draws)
    lambda_away = Matrix{Float64}(undef, length(ids), draws)
    for i in eachindex(ids)
        lambda_home[i, :] .= 1.8 + 0.01 * i
        lambda_away[i, :] .= 0.8 + 0.01 * i
    end
    return CountLatents(ids, lambda_home, lambda_away)
end

function extension_datastore(n_target_weeks::Int)
    history = DataFrame(
        match_id = [101, 102], tournament_id = fill(1, 2), season = fill("23/24", 2),
        match_week = [1, 2], match_date = Date.(2024, 1, [1, 8]),
        home_team = ["A", "B"], away_team = ["B", "A"],
        home_score = [1, 0], away_score = [0, 1],
    )
    target = DataFrame(
        match_id = collect(201:(200 + n_target_weeks)),
        tournament_id = fill(1, n_target_weeks), season = fill("24/25", n_target_weeks),
        match_week = collect(1:n_target_weeks),
        match_date = [Date(2025, 1, 1) + Week(i - 1) for i in 1:n_target_weeks],
        home_team = [isodd(i) ? "A" : "B" for i in 1:n_target_weeks],
        away_team = [isodd(i) ? "B" : "A" for i in 1:n_target_weeks],
        home_score = [isodd(i) ? 2 : 1 for i in 1:n_target_weeks],
        away_score = [isodd(i) ? 0 : 1 for i in 1:n_target_weeks],
    )
    matches = vcat(history, target)
    empty = DataFrame()
    return Data.DataStore(Data.ScottishLower(), matches, empty, empty, empty, empty, empty)
end

function extension_chain(seed)
    return Chains(randn(Xoshiro(seed), 200, 2, 4), [:a, :b])
end

function extension_odds(ids)
    rows = NamedTuple[]
    for id in ids
        for (selection, probability) in zip((:home, :draw, :away), (0.20, 0.20, 0.60))
            push!(rows, (match_id = id, market_name = "1X2", market_line = 0.0,
                         selection, odds_close = 1.0 / probability))
        end
    end
    return DataFrame(rows)
end

@testset "Incremental live extension" begin
    @testset "zero-allocation count scoring kernel" begin
        probability = zeros(25)
        Training.Inference._extension_count_pmf!(probability, 1.4, nothing)
        @test @allocated(Training.Inference._extension_count_pmf!(probability, 1.4, nothing)) == 0
        Training.Inference._extension_count_pmf!(probability, 1.4, 4.0)
        @test @allocated(Training.Inference._extension_count_pmf!(probability, 1.4, 4.0)) == 0
        @test all(isfinite, probability)
    end

    test_url = get(ENV, "BF_EXPERIMENTS_TEST_DB_URL", "")
    if isempty(test_url)
        @testset "PostgreSQL integration (set BF_EXPERIMENTS_TEST_DB_URL)" begin
            @test true
        end
    else
        experiment = "extension_test_$(uuid4())"
        db = PostgresStorage(test_url, experiment)
        ensure_schema!(db)
        ds_two = extension_datastore(2)
        ds_three = extension_datastore(3)
        splitter = Data.CVConfig(tournament_ids = [1], target_seasons = ["24/25"],
                                 history_seasons = 1, dynamics_col = :match_week,
                                 warmup_period = 1, stop_early = true)
        sampler = ExtensionMockSampler([extension_chain(i) for i in 1:3], Int[], 1)
        config = FitConfig(name = "incremental_mock", model = ExtensionMockModel(),
                           splitter = splitter, sampler = sampler,
                           execution = SequentialExecution(), save_dir = mktempdir())
        initial = fit_model(ds_two, config; quiet = true)
        @test length(initial) == 2
        empty!(initial.config.sampler.sampled_folds)
        run_id = save_fit(initial, db)

        @testset "preview and selective Fit extension" begin
            io = IOBuffer()
            preview = preview_extension(db, run_id, ds_three; io)
            output = String(take!(io))
            @test preview.delta_folds == [3]
            @test preview.existing_count == 2
            @test preview.new_count == 1
            @test !preview.is_uptodate
            @test occursin("New Folds to Fit", output)
            @test occursin("2025-01-15", output)

            extended = extend_fit(db, run_id, ds_three; quiet = true)
            @test length(extended) == 3
            @test [fold.fold for fold in extended.folds] == [1, 2, 3]
            @test extended.config.sampler.sampled_folds == [3]
            @test latent_match_ids(extended.latents) == [201, 202, 203]

            conn = LibPQ.Connection(test_url)
            try
                folds = DataFrame(LibPQ.execute(conn, """
                    SELECT fold_idx, logloss, brier, rps, n_matches
                    FROM fold_results WHERE run_id = \$1::uuid ORDER BY fold_idx;
                """, (string(run_id),)))
                @test folds.fold_idx == [1, 2, 3]
                @test folds.n_matches[3] == 1
                @test !ismissing(folds.logloss[3])
                @test !ismissing(folds.brier[3])
                @test !ismissing(folds.rps[3])
                latents = DataFrame(LibPQ.execute(conn, """
                    SELECT COUNT(*) AS n FROM match_latents ml
                    JOIN fold_results fr ON fr.fold_id = ml.fold_id
                    WHERE fr.run_id = \$1::uuid;
                """, (string(run_id),)))
                @test latents.n[1] == 3
                artifact = DataFrame(LibPQ.execute(conn, """
                    SELECT octet_length(fit_blob) AS bytes FROM fit_artifacts
                    WHERE run_id = \$1::uuid;
                """, (string(run_id),)))
                @test artifact.bytes[1] > 0
                split_config = DataFrame(LibPQ.execute(conn, """
                    SELECT split_config->>'n_folds_total' AS n_folds_total,
                           split_config->>'latest_fold_idx' AS latest_fold_idx,
                           split_config->>'latest_fold_date' AS latest_fold_date
                    FROM configs WHERE config_id = \$1::uuid;
                """, (string(run_id),)))
                @test split_config.n_folds_total[1] == "3"
                @test split_config.latest_fold_idx[1] == "3"
                @test split_config.latest_fold_date[1] == "2025-01-15"
            finally
                close(conn)
            end

            loaded = load_fit(db, run_id)
            @test length(loaded) == 3
            @test latent_match_ids(loaded.latents) == [201, 202, 203]

            up_to_date_io = IOBuffer()
            current = preview_extension(db, run_id, ds_three; io = up_to_date_io)
            @test current.is_uptodate
            @test current.new_count == 0
            @test occursin("is up-to-date (3 folds completed). 0 new folds needed.",
                           String(take!(up_to_date_io)))
            no_op = extend_fit(db, run_id, ds_three; quiet = true)
            @test length(no_op) == 3
            @test no_op.config.sampler.sampled_folds == [3]
        end

        @testset "portfolio roll-forward" begin
            initial_fit = Fit(initial.config, initial.folds, initial.latents,
                              initial.diagnostics, initial.metadata, initial.save_path)
            odds = extension_odds([201, 202, 203])
            book = BookSpec(markets = Data.MarketConfig(Data.AbstractMarket[Data.Market1X2()]),
                            shrink = Portfolio.NoShrinkage())
            policy = PolicySpec(trust = FlatTrust(1.0), risk = SlateDrawdown(50.0),
                                cap = FixedCap(0.50))
            first_result, _, _ = run_portfolio_simulation(
                book, policy, initial_fit, odds, ds_two;
                require_converged = false, bootstrap = false, quiet = true)
            @test first_result.summary.n_bets > 0
            portfolio_id = save_portfolio_db(first_result, run_id, db;
                                              book_spec = book, policy_spec = policy)
            extended_fit = load_fit(db, run_id)
            # Specs are recovered losslessly from portfolio_artifacts when omitted.
            updated = extend_portfolio(db, portfolio_id, extended_fit, odds, ds_three)
            @test updated.summary.n_bets > first_result.summary.n_bets
            @test 203 in updated.trajectory.bets.match_id
            @test length(updated.daily_states) >= length(first_result.daily_states)

            conn = LibPQ.Connection(test_url)
            try
                summary = DataFrame(LibPQ.execute(conn, """
                    SELECT n_bets, total_return_pct, flat_roi_pct
                    FROM portfolio_runs WHERE portfolio_run_id = \$1::uuid;
                """, (string(portfolio_id),)))
                @test summary.n_bets[1] == updated.summary.n_bets
                @test summary.total_return_pct[1] == updated.summary.total_return_pct
                @test summary.flat_roi_pct[1] == updated.summary.roi
                new_bets = DataFrame(LibPQ.execute(conn, """
                    SELECT COUNT(*) AS n FROM portfolio_bets
                    WHERE portfolio_run_id = \$1::uuid AND match_id = 203;
                """, (string(portfolio_id),)))
                @test new_bets.n[1] > 0
            finally
                close(conn)
            end
            @test load_portfolio_db(portfolio_id, db).summary.n_bets == updated.summary.n_bets
        end

        conn = LibPQ.Connection(test_url)
        try
            result = LibPQ.execute(conn,
                "DELETE FROM runs WHERE experiment_name = \$1;", (experiment,))
            close(result)
        finally
            close(conn)
        end
    end
end
