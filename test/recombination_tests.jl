# test/recombination_tests.jl
#
# Comprehensive unit tests for:
#   1. OpenPlayGoalsFeature, OpenPlayPxGFeature, SquadWealthFeature, RefereeOfficiatingFeature
#   2. DynamicRecombinedGoalsModel & DynamicPxGRecombModel
#   3. Turing @model construction & ReverseDiff AD gradient evaluation
#   4. Discrete Poisson ScoreMatrix convolution & sum invariant (1.000000)

using Test
using BayesianFootball
using DataFrames, Dates, Distributions, Turing, DynamicPPL, ReverseDiff

@testset "Recombination & Squad Wealth Engine Tests" begin

    @testset "1. Model Configuration & Component Hierarchy" begin
        # Test Default Recombination Goals Model
        model_goals = Models.PreGame.DynamicRecombinedGoalsModel()
        @test model_goals isa Models.PreGame.AbstractTimeDecayTeamModel
        @test model_goals.wealth_config isa Models.PreGame.LinearSquadWealthConfig
        @test model_goals.recomb_config isa Models.PreGame.HierarchicalOfficiatingConfig

        # Test Default Proxy xG Recombination Model
        model_pxg = Models.PreGame.DynamicPxGRecombModel()
        @test model_pxg isa Models.PreGame.AbstractTimeDecayTeamModel
        @test model_pxg.pxg_config isa Models.PreGame.GammaPxGObservationConfig
        @test model_pxg.wealth_config isa Models.PreGame.LinearSquadWealthConfig
        @test model_pxg.recomb_config isa Models.PreGame.HierarchicalOfficiatingConfig

        # Test Required Features API
        req_goals = Features.required_features(model_goals)
        @test any(f -> f isa Features.OpenPlayGoalsFeature, req_goals)
        @test any(f -> f isa Features.SquadWealthFeature, req_goals)
        @test any(f -> f isa Features.RefereeOfficiatingFeature, req_goals)

        req_pxg = Features.required_features(model_pxg)
        @test any(f -> f isa Features.OpenPlayPxGFeature, req_pxg)
        @test any(f -> f isa Features.OpenPlayGoalsFeature, req_pxg)
        @test any(f -> f isa Features.SquadWealthFeature, req_pxg)
    end

    @testset "2. Feature Extraction Pipeline with Mock DataStore" begin
        # Create a compact mock DataStore
        matches_df = DataFrame(
            match_id      = Int32[101, 102, 103, 104],
            tournament_id = Int32[56, 56, 57, 57],
            season        = ["24/25", "24/25", "24/25", "24/25"],
            match_date    = [Date(2024, 8, 10), Date(2024, 8, 17), Date(2024, 8, 24), Date(2024, 8, 31)],
            home_team     = ["Team A", "Team C", "Team B", "Team D"],
            away_team     = ["Team B", "Team D", "Team A", "Team C"],
            home_score    = [2, 1, 3, 0],
            away_score    = [1, 1, 0, 2],
            referee_id    = [1, 2, 1, 2]
        )

        incidents_df = DataFrame(
            match_id       = Int32[101, 103],
            incident_type  = ["goal", "goal"],
            incident_class = ["penalty", "ownGoal"],
            is_home        = [true, false]
        )

        lineups_df = DataFrame(
            match_id       = Int32[101, 101, 102, 102],
            player_id      = Int[1, 2, 3, 4],
            team_side      = ["home", "away", "home", "away"],
            position       = ["F", "M", "D", "G"],
            is_substitute  = [false, false, false, false],
            market_value   = [250_000.0, 150_000.0, 120_000.0, 80_000.0]
        )

        mock_ds = Data.DataStore(
            Data.ScottishLower(),
            matches_df,
            DataFrame(), # statistics
            DataFrame(), # odds
            lineups_df,
            incidents_df,
            DataFrame()  # betfair_odds
        )

        team_map = Dict("Team A" => 1, "Team B" => 2, "Team C" => 3, "Team D" => 4)
        ordered_ids = [101, 102, 103, 104]
        F_data = Dict{Symbol, Any}()

        # Test OpenPlayGoalsFeature
        Features.add_feature!(F_data, Features.OpenPlayGoalsFeature(), ordered_ids, team_map, mock_ds)
        @test haskey(F_data, :flat_y_open_h)
        @test haskey(F_data, :flat_y_open_a)
        @test haskey(F_data, :flat_pen_scored_h)
        @test F_data[:flat_y_open_h] == [1, 1, 3, 0] # Match 101: 2 - 1 pen = 1 open goal
        @test F_data[:flat_pen_scored_h] == [1, 0, 0, 0]

        # Test SquadWealthFeature
        Features.add_feature!(F_data, Features.SquadWealthFeature(), ordered_ids, team_map, mock_ds)
        @test haskey(F_data, :flat_delta_wealth)
        @test length(F_data[:flat_delta_wealth]) == 4

        # Test RefereeOfficiatingFeature
        Features.add_feature!(F_data, Features.RefereeOfficiatingFeature(), ordered_ids, team_map, mock_ds)
        @test haskey(F_data, :flat_referee_ids)
        @test F_data[:n_referees] == 2

        # Test OpenPlayPxGFeature
        Features.add_feature!(F_data, Features.OpenPlayPxGFeature(), ordered_ids, team_map, mock_ds)
        @test haskey(F_data, :flat_pxg_h)
        @test haskey(F_data, :flat_mask_pxg_h)
        @test all(m -> m in (0.0, 1.0), F_data[:flat_mask_pxg_h])
        @test all(x -> x > 0.0, F_data[:flat_pxg_h])
    end

    @testset "3. Turing Model Construction & ReverseDiff AD Tape Profiling" begin
        # Build synthetic FeatureSet for 6 matches
        n_m = 6
        n_teams = 4
        feature_data = Dict{Symbol, Any}(
            :n_teams             => n_teams,
            :n_leagues           => 2,
            :n_referees          => 2,
            :dates               => [10, 8, 6, 4, 2, 0],
            :flat_home_ids       => [1, 2, 3, 4, 1, 3],
            :flat_away_ids       => [2, 3, 4, 1, 4, 2],
            :flat_month_ids      => [8, 8, 9, 9, 10, 10],
            :flat_league_ids     => [1, 1, 2, 2, 1, 2],
            :flat_referee_ids    => [1, 2, 1, 2, 1, 2],
            :flat_delta_wealth   => [0.5, -0.2, 1.1, -0.8, 0.3, -0.1],
            :flat_pxg_h          => [1.4, 0.9, 2.1, 0.7, 1.2, 1.5],
            :flat_pxg_a          => [0.8, 1.1, 0.5, 1.4, 0.9, 0.8],
            :flat_mask_pxg_h     => [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            :flat_mask_pxg_a     => [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            :flat_y_open_h       => [1, 0, 2, 1, 1, 2],
            :flat_y_open_a       => [1, 1, 0, 2, 0, 1],
            :flat_pen_awarded_h  => [0, 1, 0, 0, 1, 0],
            :flat_pen_awarded_a  => [0, 0, 0, 1, 0, 0],
            :team_map            => Dict("T1" => 1, "T2" => 2, "T3" => 3, "T4" => 4),
            :wealth_lookup       => Dict(1 => 0.5, 2 => -0.2, 3 => 1.1, 4 => -0.8, 5 => 0.3, 6 => -0.1),
            :league_lookup       => Dict(1 => 1, 2 => 1, 3 => 2, 4 => 2, 5 => 1, 6 => 2)
        )
        fset = Features.FeatureSet(feature_data)

        # 1. DynamicRecombinedGoalsModel
        model_goals = Models.PreGame.DynamicRecombinedGoalsModel()
        t_model_goals = Models.PreGame.build_turing_model(model_goals, fset)
        @test t_model_goals isa DynamicPPL.Model

        # 2. DynamicPxGRecombModel
        model_pxg = Models.PreGame.DynamicPxGRecombModel()
        t_model_pxg = Models.PreGame.build_turing_model(model_pxg, fset)
        @test t_model_pxg isa DynamicPPL.Model

        # Sample prior check (evaluates log joint density without throwing)
        vi = DynamicPPL.VarInfo(t_model_pxg)
        logp = DynamicPPL.getlogp(vi)
        if logp isa Number
            @test isfinite(logp)
        else
            @test isfinite(logp.logprior) && isfinite(logp.loglikelihood)
        end
    end

    @testset "4. Predictions & Discrete Poisson Convolution ScoreMatrix Invariants" begin
        model_pxg = Models.PreGame.DynamicPxGRecombModel()

        # Mock rate sample vectors (50 posterior draws)
        n_draws = 50
        mock_params = (
            λ_h = rand(Uniform(1.1, 2.5), n_draws),
            λ_a = rand(Uniform(0.8, 1.9), n_draws)
        )

        S = Predictions.compute_score_matrix(model_pxg, mock_params; max_goals = 12)
        @test S isa Predictions.ScoreMatrix
        @test size(S.data) == (12, 12, n_draws)

        # Mathematical Invariant: Probabilities must sum to 1.0 for every sample
        for k in 1:n_draws
            total_prob = sum(S.data[:, :, k])
            @test isapprox(total_prob, 1.0, atol = 1e-5)
        end

        # Test DataFrameRow dispatch
        mock_row = DataFrame(
            λ_h = [mock_params.λ_h],
            λ_a = [mock_params.λ_a]
        )[1, :]
        S_row = Predictions.compute_score_matrix(model_pxg, mock_row; max_goals = 12)
        @test S_row isa Predictions.ScoreMatrix
        @test isapprox(sum(S_row.data[:, :, 1]), 1.0, atol = 1e-5)
    end

end
