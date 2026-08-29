using Test
using BayesianFootball
using DataFrames
using Dates

@testset "Features Module" begin
    matches = DataFrame(
        match_id=Int[1, 2],
        tournament_id=Int[79, 79],
        season=["24/25", "24/25"],
        match_date=Date.(String["2024-08-03", "2024-08-10"]),
        match_hour=Int[14, 14],
        match_week=Int[1, 2],
        match_biweek=Int[1, 1],
        match_month=Int[1, 1],
        home_team=["alpha", "beta"],
        away_team=["beta", "alpha"],
    )
    empty = DataFrame()
    ds = BayesianFootball.Data.DataStore(
        BayesianFootball.Data.Ireland(), matches,
        empty, empty, empty, empty, empty, empty, empty)
    boundary = BayesianFootball.Data.SplitBoundary(1, 1, Int[], [1, 2])

    feature_set = BayesianFootball.Features.create_features(
        boundary, ds, SplitClockProbe(), :match_week)

    @test feature_set isa BayesianFootball.FeatureSet
    @test feature_set.data[:n_teams] == 2
    @test feature_set.data[:team_map] == Dict("alpha" => 1, "beta" => 2)
    @test feature_set.data[:ordered_match_ids] == [1, 2]
    @test feature_set.data[:time_indices] == [1, 2]
    @test feature_set.data[:n_rounds] == 2

    @test_throws ErrorException BayesianFootball.Features.create_features(
        BayesianFootball.Data.SplitBoundary(1, 1, Int[], [1, 1]),
        ds, SplitClockProbe(), :match_week)
end

function squad_wealth_store(matches, lineups)
    empty = DataFrame()
    return BayesianFootball.Data.DataStore(
        BayesianFootball.Data.Ireland(), matches,
        empty, empty, lineups, empty, empty, empty, empty)
end

@testset "SquadWealthFeature" begin
    config = BayesianFootball.Features.SquadWealthFeature()
    @test config.log_scale == 0.50
    @test config.decay_half_life_days == 30.0
    @test config.min_valid_players_per_side == 1

    @testset "full lineups and partial valid values" begin
        matches = DataFrame(
            match_id=Int32[11, 12],
            start_timestamp=DateTime[DateTime(2024, 1, 1), DateTime(2024, 1, 8)],
            home_team=["alpha", "alpha"], away_team=["beta", "beta"],
        )
        lineups = DataFrame(
            match_id=Int32[11, 11, 11, 11, 12, 12, 12, 12, 12],
            team_side=["home", "home", "away", "away", "home", "home", "home", "away", "away"],
            is_substitute=Union{Missing, Bool}[false, false, false, false, false, false, missing, false, false],
            proposed_market_value=Union{Missing, Float64}[
                100.0, 400.0, 100.0, 100.0,
                900.0, 0.0, missing, 100.0, missing,
            ],
        )
        F_data = Dict{Symbol, Any}(:history_match_ids => Set([11, 12]))
        BayesianFootball.Features.add_feature!(
            F_data, config, [11, 12], Dict("alpha" => 1, "beta" => 2),
            squad_wealth_store(matches, lineups))

        @test F_data[:flat_delta_wealth][1] ≈ log(2.0) / config.log_scale
        @test F_data[:flat_delta_wealth][2] ≈ log(9.0) / config.log_scale
        @test F_data[:flat_wealth_available] == [1.0, 1.0]
        @test F_data[:flat_wealth_home_count] == [2, 1]
        @test F_data[:flat_wealth_away_count] == [2, 1]
        @test F_data[:wealth_by_match_id][Int32(12)] ≈ log(9.0) / config.log_scale
    end

    @testset "missing lineups use decayed rolling values" begin
        matches = DataFrame(
            match_id=Int32[21, 22],
            start_timestamp=DateTime[DateTime(2024, 1, 1), DateTime(2024, 1, 31)],
            home_team=["alpha", "alpha"], away_team=["beta", "beta"],
        )
        lineups = DataFrame(
            match_id=Int32[21, 21], team_side=["home", "away"],
            is_substitute=[false, false], proposed_market_value=[400.0, 100.0],
        )
        F_data = Dict{Symbol, Any}(:history_match_ids => Set([21]))
        BayesianFootball.Features.add_feature!(
            F_data, config, [22, 21], Dict("alpha" => 1, "beta" => 2),
            squad_wealth_store(matches, lineups))

        # Extraction is chronological even when requested output order is not.
        @test F_data[:flat_delta_wealth][1] ≈ 0.5 * log(4.0) / config.log_scale
        @test F_data[:flat_delta_wealth][2] ≈ log(4.0) / config.log_scale
        @test F_data[:flat_wealth_available] == [0.5, 1.0]
        @test F_data[:flat_wealth_home_count] == [0, 1]
        @test F_data[:flat_wealth_away_count] == [0, 1]
    end

    @testset "cold start matches the population baseline" begin
        matches = DataFrame(
            match_id=Int32[31], start_timestamp=DateTime[DateTime(2020, 10, 1)],
            home_team=["promoted"], away_team=["newcomer"],
        )
        F_data = Dict{Symbol, Any}(:history_match_ids => Set{Int}())
        BayesianFootball.Features.add_feature!(
            F_data, config, [31], Dict("promoted" => 1, "newcomer" => 2),
            squad_wealth_store(matches, DataFrame()))

        @test F_data[:flat_delta_wealth] == [0.0]
        @test F_data[:flat_wealth_available] == [0.0]
        @test F_data[:flat_wealth_home_count] == [0]
        @test F_data[:flat_wealth_away_count] == [0]
    end

    @testset "malformed sides and duplicate players are ignored" begin
        matches = DataFrame(
            match_id=Int32[41], start_timestamp=DateTime[DateTime(2024, 2, 1)],
            home_team=["alpha"], away_team=["beta"],
        )
        lineups = DataFrame(
            match_id=fill(Int32(41), 5),
            player_id=Int32[1, 1, 2, 3, 4],
            team_side=Union{Missing, String}["home", "home", "away", "bench", missing],
            is_substitute=[false, false, false, false, false],
            proposed_market_value=[400.0, 40_000.0, 100.0, 900.0, 900.0],
        )
        F_data = Dict{Symbol, Any}(:history_match_ids => Set([41]))
        BayesianFootball.Features.add_feature!(
            F_data, config, [41], Dict("alpha" => 1, "beta" => 2),
            squad_wealth_store(matches, lineups))

        @test F_data[:flat_delta_wealth] ≈ [log(4.0) / config.log_scale]
        @test F_data[:flat_wealth_home_count] == [1]
        @test F_data[:flat_wealth_away_count] == [1]

        no_starter_flag = select(lineups, Not(:is_substitute))
        no_flag_data = Dict{Symbol, Any}(:history_match_ids => Set([41]))
        BayesianFootball.Features.add_feature!(
            no_flag_data, config, [41], Dict("alpha" => 1, "beta" => 2),
            squad_wealth_store(matches, no_starter_flag))
        @test no_flag_data[:flat_delta_wealth] == [0.0]
        @test no_flag_data[:flat_wealth_available] == [0.0]
    end

    @testset "dynamic scale fitted on history matches" begin
        dyn_config = BayesianFootball.Features.SquadWealthFeature(log_scale = nothing)
        @test dyn_config.log_scale === nothing
        matches = DataFrame(
            match_id=Int32[51, 52],
            start_timestamp=DateTime[DateTime(2024, 1, 1), DateTime(2024, 1, 8)],
            home_team=["alpha", "alpha"], away_team=["beta", "beta"],
        )
        lineups = DataFrame(
            match_id=Int32[51, 51, 52, 52], team_side=["home", "away", "home", "away"],
            is_substitute=[false, false, false, false],
            proposed_market_value=[400.0, 100.0, 400.0, 100.0],
        )
        F_data = Dict{Symbol, Any}(:history_match_ids => Set([51, 52]))
        BayesianFootball.Features.add_feature!(
            F_data, dyn_config, [51, 52], Dict("alpha" => 1, "beta" => 2),
            squad_wealth_store(matches, lineups))

        # Default fallback to 0.50 when n < 10 history diffs
        @test F_data[:flat_delta_wealth][1] ≈ log(4.0) / 0.50
        @test F_data[:flat_delta_wealth][2] ≈ log(4.0) / 0.50
    end
end

@testset "DistanceFeature" begin
    Features = BayesianFootball.Features
    config = Features.DistanceFeature(metric=:hav_miles)
    @test isfile(config.geocodes_csv)

    # Cliftonhill (Albion Rovers) to the Excelsior (Airdrieonians).
    miles = Features.haversine_distance(55.8611, -4.0167, 55.8664, -3.9558)
    @test miles ≈ 2.39 atol=0.02
    @test Features.haversine_distance(55.8611, -4.0167, 55.8664, -3.9558; unit=:km) ≈
          miles * (6371.0088 / 3958.7613)
    @test_throws ArgumentError Features.haversine_distance(
        55.0, -4.0, 56.0, -3.0; unit=:metres)

    @test Features.distance_tier_category(24.99) == 1
    @test Features.distance_tier_category(25.0) == 2
    @test Features.distance_tier_category(75.0) == 3
    @test Features.distance_tier_category(140.0) == 3
    @test Features.distance_tier_category(140.01) == 4
    @test Features.estimate_scottish_road_metrics(25.0).road_miles == 31.25
    @test Features.estimate_scottish_road_metrics(80.0).road_miles == 100.0
    @test Features.estimate_scottish_road_metrics(80.01).road_miles ≈ 104.013

    mktempdir() do directory
        invalid_catalog = joinpath(directory, "invalid.csv")
        write(invalid_catalog, "team_slug,latitude,longitude\na,91.0,-4.0\n")
        @test_throws ErrorException Features.load_stadium_catalog(invalid_catalog)
    end

    matches = DataFrame(
        match_id=Int32[101, 102, 103],
        home_team=["albion-rovers", "airdrieonians", "unmapped-club"],
        away_team=["airdrieonians", "alloa-athletic", "airdrieonians"],
        match_date=Date[Date(2024, 8, 3), Date(2024, 8, 6), Date(2024, 8, 7)],
    )
    empty = DataFrame()
    ds = BayesianFootball.Data.DataStore(
        BayesianFootball.Data.ScottishLower(), matches,
        empty, empty, empty, empty, empty, empty, empty)
    F_data = Dict{Symbol, Any}()
    Features.add_feature!(F_data, config, [103, 101, 102], Dict{String, Int}(), ds)

    @test F_data[:flat_distance] === F_data[:flat_distance_miles]
    @test F_data[:flat_distance_miles][2] ≈ miles atol=1e-10
    @test F_data[:flat_distance_fallback] == [1, 0, 0]
    @test F_data[:flat_distance_miles][1] == 45.0
    @test F_data[:flat_distance_tier][1] == 2
    @test F_data[:flat_is_midweek] == [1.0, 0.0, 1.0]
    @test F_data[:distance_by_match_id] isa Dict{Int32, Float64}
    @test Set(keys(F_data[:distance_by_match_id])) == Set(Int32[101, 102, 103])

    # A row-level missing match_date uses start_timestamp; with neither it stays zero.
    timestamp_matches = DataFrame(
        match_id=Int32[201, 202],
        home_team=["albion-rovers", "albion-rovers"],
        away_team=["airdrieonians", "airdrieonians"],
        match_date=Union{Missing, Date}[missing, missing],
        start_timestamp=Union{Missing, DateTime}[DateTime(2024, 8, 6, 19, 45), missing],
    )
    timestamp_table = Features.build_match_distance_table(
        timestamp_matches; geocodes_df=Features.load_stadium_catalog(config.geocodes_csv))
    @test timestamp_table.is_midweek == [1.0, 0.0]

    # Catalog moments, rather than fixture moments, make prior rows invariant to future rows.
    catalog = Features.load_stadium_catalog(config.geocodes_csv)
    first_table = Features.build_match_distance_table(matches[1:2, :]; geocodes_df=catalog)
    extended_table = Features.build_match_distance_table(matches; geocodes_df=catalog)
    @test first_table.dist_z == extended_table.dist_z[1:2]
    @test first_table.log_dist_z == extended_table.log_dist_z[1:2]

    no_midweek = Dict{Symbol, Any}()
    Features.add_feature!(
        no_midweek, Features.DistanceFeature(metric=:drive_minutes, include_midweek=false),
        [101, 102, 103], Dict{String, Int}(), ds)
    @test no_midweek[:flat_is_midweek] == zeros(3)
    @test no_midweek[:flat_distance] === no_midweek[:flat_drive_minutes]
end
