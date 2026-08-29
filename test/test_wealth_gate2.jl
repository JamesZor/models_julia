using Test
using BayesianFootball
using DataFrames
using Dates
using Statistics

const WEALTH_PROTOCOL_ROOT = joinpath(
    @__DIR__, "..", "current_development", "scottish_lower", "_protocol")
include(joinpath(WEALTH_PROTOCOL_ROOT, "ScottishLowerProtocol.jl"))
using .ScottishLowerProtocol

struct WealthGateProbe <: BayesianFootball.AbstractFootballModel
    wealth_feature::BayesianFootball.Features.SquadWealthFeature
end

function BayesianFootball.Features.required_features(model::WealthGateProbe)
    return BayesianFootball.Features.AbstractFeatureConfig[model.wealth_feature]
end

struct WealthGateAdapter{M} <: AbstractSLModelAdapter
    model::M
end

ScottishLowerProtocol.sl_model(adapter::WealthGateAdapter) = adapter.model

function synthetic_wealth_store()
    matches = DataFrame(
        tournament_id = Int32[56, 56, 56],
        season_id = Int32[1, 1, 2],
        season = ["23/24", "23/24", "24/25"],
        match_id = Int32[1, 2, 3],
        match_date = Date[Date(2024, 1, 6), Date(2024, 1, 13), Date(2025, 1, 4)],
        match_hour = Int[15, 15, 15],
        match_week = Int[1, 2, 1],
        match_biweek = Int[1, 1, 1],
        match_month = Int[1, 1, 1],
        home_team = ["alpha", "promoted-club", "alpha"],
        away_team = ["beta", "beta", "beta"],
    )

    lineups = DataFrame(
        match_id = Int32[1, 1, 1, 1, 2, 2, 3, 3],
        team_side = ["home", "home", "away", "away", "home", "away", "home", "away"],
        player_id = Int32[11, 12, 21, 22, 31, 41, 11, 21],
        is_substitute = falses(8),
        market_value = Union{Missing, Float64}[
            200_000.0, missing, 100_000.0, missing,
            missing, 100_000.0, 9_000_000.0, 50_000.0,
        ],
        valuation_timestamp = DateTime[
            DateTime(2024, 1, 5), DateTime(2024, 1, 5),
            DateTime(2024, 1, 5), DateTime(2024, 1, 5),
            DateTime(2024, 1, 12), DateTime(2024, 1, 12),
            DateTime(2025, 1, 3), DateTime(2025, 1, 3),
        ],
    )

    empty = DataFrame()
    return BayesianFootball.Data.DataStore(
        BayesianFootball.Data.ScottishLower(), matches,
        empty, empty, lineups, empty, empty, empty, empty)
end

@testset "Squad wealth feature — Gate 2" begin
    @testset "observed values and rolling fallback" begin
        ds = synthetic_wealth_store()
        config = BayesianFootball.Features.SquadWealthFeature()
        records = BayesianFootball.Features._build_match_wealth_records(
            ds.lineups, ds.matches, Int[1, 2], Int[1], config)

        @test records[Int32(1)].delta ≈ log(2.0) / config.log_scale
        @test records[Int32(1)].available == 1.0
        @test records[Int32(1)].home_count == 1
        @test records[Int32(1)].away_count == 1
        @test records[Int32(2)].available == 0.0
        @test records[Int32(2)].home_count == 0
        @test records[Int32(2)].away_count == 1

        # Valuations are match-scoped, so legacy valuation timestamps are ignored.
        no_timestamp = select(ds.lineups, Not(:valuation_timestamp))
        lookup = BayesianFootball.Features._build_match_wealth_lookup(
            no_timestamp, ds.matches, Int[1], config)
        @test lookup[Int32(1)] ≈ log(2.0) / config.log_scale
    end

    @testset "future rows cannot alter historical FeatureSet" begin
        ds = synthetic_wealth_store()
        boundary = BayesianFootball.Data.SplitBoundary(1, 1, Int[1], Int[2])
        model = WealthGateProbe(BayesianFootball.Features.SquadWealthFeature())

        full = BayesianFootball.Features.create_features(boundary, ds, model, :match_biweek)
        truncated_ds = sl_truncate_datastore(ds, Int[1, 2])
        truncated = BayesianFootball.Features.create_features(
            boundary, truncated_ds, model, :match_biweek)
        same, differing = sl_featureset_equal(full, truncated)

        @test same
        @test isempty(differing)
        @test full.data[:flat_delta_wealth][1] ≈ log(2) / model.wealth_feature.log_scale
        @test full.data[:flat_wealth_available] == Float64[1.0, 0.0]
        @test full.data[:flat_wealth_home_count] == Int[1, 0]
        @test full.data[:flat_wealth_away_count] == Int[1, 1]
        @test eltype(full.data[:flat_delta_wealth]) === Float64
        @test eltype(full.data[:flat_wealth_available]) === Float64
        @test !any(isnan, full.data[:flat_delta_wealth])
    end

    @testset "all 20 Scottish Lower folds satisfy shared Gate 2" begin
        ds = BayesianFootball.Data.load_datastore_cached(
            BayesianFootball.Data.ScottishLower(); max_age_hours=10_000)
        contract = sl_contract()
        folds = sl_build_folds(ds, contract)
        @test length(folds) == 20

        model = WealthGateProbe(BayesianFootball.Features.SquadWealthFeature())
        results, feature_sets = sl_gate_features(
            ds, folds, WealthGateAdapter(model), contract)

        @test length(results) == 7
        @test all(result.pass for result in results)
        for fs in feature_sets
            wealth = fs.data[:flat_delta_wealth]
            available = fs.data[:flat_wealth_available]
            @test eltype(wealth) === Float64
            @test eltype(available) === Float64
            @test !any(isnan, wealth)
            @test !any(ismissing, wealth)
            @test all(value -> value in (0.0, 0.5, 1.0), available)
        end
    end
end
