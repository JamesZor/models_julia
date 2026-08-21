# test/caching_tests.jl

using Test
using BayesianFootball
using BayesianFootball.Experiments
using BayesianFootball.Training
using BayesianFootball.Samplers
using DataFrames
using Dates
using Serialization

struct MockModel <: BayesianFootball.AbstractFootballModel end

@testset "OOS Latents Caching" begin
    # 1. Setup mock model and LatentStates
    model = MockModel()
    
    test_df = DataFrame(
        match_id = [101, 102, 103],
        home_score = [2, 1, 0],
        away_score = [1, 1, 3]
    )
    
    latents = LatentStates(test_df, model)

    @testset "Direct Path Caching Functions" begin
        mktempdir() do tmp_dir
            # Initial state: no cache
            @test !has_oos_predictions(tmp_dir)
            @test load_oos_predictions(tmp_dir) === nothing

            # Save
            saved_path = save_oos_predictions(tmp_dir, latents)
            @test isfile(saved_path)
            @test basename(saved_path) == Experiments.OOS_LATENTS_FILENAME
            @test has_oos_predictions(tmp_dir)

            # Load
            loaded = load_oos_predictions(tmp_dir)
            @test loaded isa LatentStates
            @test loaded.df == test_df
            @test loaded.model == model
            @test nrow(loaded) == 3

            # Corrupted file test: graceful fallback to nothing
            write(saved_path, "corrupted non-serialized binary payload")
            @test load_oos_predictions(tmp_dir) === nothing
        end
    end

    @testset "ExperimentResults Caching Functions" begin
        mktempdir() do tmp_dir
            exp_dir = joinpath(tmp_dir, "exp_1")
            
            # Construct minimal mock ExperimentConfig and ExperimentResults
            config = ExperimentConfig(
                name = "test_experiment",
                model = model,
                splitter = BayesianFootball.Data.CVConfig(
                    target_seasons = ["2022/2023"],
                    dynamics_col = :match_week
                ),
                training_config = TrainingConfig(
                    sampler = Samplers.MAPConfig(),
                    strategy = Training.Independent()
                ),
                save_dir = tmp_dir
            )
            
            training_results = TrainingResults{Any, Any}(Tuple{Any, Any}[])
            exp_results = ExperimentResults(config, training_results, nothing, exp_dir)

            @test !has_oos_predictions(exp_results)
            @test load_oos_predictions(exp_results) === nothing

            # Save via exp_results
            saved_path = save_oos_predictions(exp_results, latents)
            @test isfile(saved_path)
            @test has_oos_predictions(exp_results)

            # Load via exp_results
            loaded = load_oos_predictions(exp_results)
            @test loaded isa LatentStates
            @test loaded.df == test_df

            # Test extract_oos_predictions cache-hit
            # When cached, it returns cached latents directly (without touching DataStore splits)
            dummy_ds = BayesianFootball.Data.DataStore(
                BayesianFootball.Data.ScottishLower(),
                DataFrame(
                    match_id = [1, 2, 3],
                    tournament_id = [1, 1, 1],
                    season = ["2022/2023", "2022/2023", "2022/2023"],
                    match_week = [1, 2, 3],
                    match_date = [Date("2022-08-01"), Date("2022-08-08"), Date("2022-08-15")],
                    home_team = ["A", "B", "C"],
                    away_team = ["B", "C", "A"],
                    home_score = [1, 2, 0],
                    away_score = [0, 1, 1]
                ),
                DataFrame(), # statistics
                DataFrame(), # odds
                DataFrame(), # lineups
                DataFrame(), # incidents
                DataFrame(), # betfair_odds
                DataFrame(), # bbc
                DataFrame()  # bbc_events
            )
            extracted = extract_oos_predictions(dummy_ds, exp_results; force = false)
            @test extracted isa LatentStates
            @test extracted.df == test_df

            # Test safety guard: forced recompute on misaligned DataStore vs training_results
            # dummy_ds has mock matches with 0 valid CV folds, while training_results has 1 mock item -> drift error
            exp_results_mismatch = ExperimentResults(
                config, 
                TrainingResults{Any, Any}(Tuple{Any, Any}[(nothing, nothing)]), 
                nothing, 
                exp_dir
            )
            @test_throws ErrorException extract_oos_predictions(dummy_ds, exp_results_mismatch; force = true)
        end
    end

    @testset "Runner Integration & Metadata" begin
        mktempdir() do tmp_dir
            exp_dir = joinpath(tmp_dir, "exp_meta_test")
            config = ExperimentConfig(
                name = "meta_test",
                model = model,
                splitter = BayesianFootball.Data.CVConfig(
                    target_seasons = ["2022/2023"],
                    dynamics_col = :match_week
                ),
                training_config = TrainingConfig(
                    sampler = Samplers.MAPConfig(),
                    strategy = Training.Independent()
                ),
                save_dir = tmp_dir
            )
            
            training_results = TrainingResults{Any, Any}(Tuple{Any, Any}[])
            exp_results = ExperimentResults(config, training_results, nothing, exp_dir)

            # Save without latents
            save_experiment(exp_results)
            meta_path = joinpath(exp_dir, "meta.json")
            @test isfile(meta_path)
            meta_content = read(meta_path, String)
            @test occursin("\"has_oos_latents\": false", meta_content)

            # Save latents and resave experiment
            save_oos_predictions(exp_results, latents)
            save_experiment(exp_results)
            meta_content2 = read(meta_path, String)
            @test occursin("\"has_oos_latents\": true", meta_content2)

            # Test list_experiments table rendering with LATENTS column
            subdirs = list_experiments(basename(tmp_dir); data_dir = dirname(tmp_dir))
            @test length(subdirs) >= 1
        end
    end
end
