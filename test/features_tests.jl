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
