using Test
using BayesianFootball
using DataFrames
using Dates

const DISTANCE_ROOT = joinpath(
    @__DIR__, "..", "current_development", "scottish_lower", "archive", "distance")
const PROTOCOL_ROOT = joinpath(
    @__DIR__, "..", "current_development", "scottish_lower", "_protocol")

include(joinpath(DISTANCE_ROOT, "l01_distance_features.jl"))
include(joinpath(PROTOCOL_ROOT, "ScottishLowerProtocol.jl"))
using .ScottishLowerProtocol

struct DistanceGateProbe <: BayesianFootball.AbstractFootballModel
    distance_feature::ScottishDistanceFeature
end

function BayesianFootball.Features.required_features(model::DistanceGateProbe)
    return BayesianFootball.Features.AbstractFeatureConfig[model.distance_feature]
end

struct DistanceGateAdapter{M} <: AbstractSLModelAdapter
    model::M
end

ScottishLowerProtocol.sl_model(adapter::DistanceGateAdapter) = adapter.model

@testset "Distance feature — Gate 2" begin
    @testset "geodesic and catalog contract" begin
        catalog = load_scottish_stadium_catalog()
        @test nrow(catalog) == length(unique(catalog.team_slug))
        @test all(isfinite, catalog.latitude)
        @test all(isfinite, catalog.longitude)

        @test haversine_distance(55.0, -4.0, 55.0, -4.0) == 0.0
        miles = haversine_distance(55.9533, -3.1883, 55.8642, -4.2518)
        km = haversine_distance(55.9533, -3.1883, 55.8642, -4.2518; unit=:km)
        @test km ≈ miles * (EARTH_RADIUS_KM / EARTH_RADIUS_MILES) rtol=1e-12
        @test_throws ArgumentError haversine_distance(55.0, -4.0, 56.0, -3.0; unit=:metres)
    end

    @testset "future perturbation and promoted-team fallback" begin
        past = DataFrame(
            match_id = Int[1, 2],
            home_team = ["airdrieonians", "elgin-city"],
            away_team = ["stranraer", "peterhead"],
            match_date = Date[Date(2024, 8, 3), Date(2024, 8, 10)],
        )
        future = DataFrame(
            match_id = Int[3],
            home_team = ["promoted-unmapped-club"],
            away_team = ["airdrieonians"],
            match_date = Date[Date(2025, 8, 2)],
        )

        past_features = build_match_distance_table(past)
        full_features = build_match_distance_table(vcat(past, future))

        for key in (:hav_miles, :dist_z, :log_dist_z)
            @test isequal(past_features[!, key], full_features[1:2, key])
        end
        @test full_features.distance_fallback == Int[0, 0, 1]
        @test full_features.hav_miles[3] == 45.0
        @test isfinite(full_features.log_dist_z[3])
    end

    @testset "shared Scottish Lower Gate 2 protocol" begin
        ds = BayesianFootball.Data.load_datastore_cached(
            BayesianFootball.Data.ScottishLower(); max_age_hours=10_000)
        contract = sl_contract()
        folds = sl_build_folds(ds, contract)
        adapter = DistanceGateAdapter(DistanceGateProbe(ScottishDistanceFeature()))

        results, feature_sets = sl_gate_features(ds, folds, adapter, contract)
        @test length(results) == 7
        @test all(result.pass for result in results)

        distance_keys = (
            :flat_distance,
            :flat_distance_z,
            :flat_log_distance_z,
            :flat_distance_miles,
            :flat_road_miles,
            :flat_drive_minutes,
            :flat_distance_tier,
            :flat_is_midweek,
            :flat_distance_fallback,
        )
        for fs in feature_sets
            @test !haskey(fs.data, :distance_df)
            @test keytype(fs.data[:team_map]) <: AbstractString
            @test sort(unique(Int.(fs.data[:time_indices]))) ==
                  collect(1:length(unique(fs.data[:time_indices])))
            for key in distance_keys
                values = fs.data[key]
                @test eltype(values) === Float64 || eltype(values) === Int
                @test !any(ismissing, values)
                @test !any(value -> value isa AbstractFloat && isnan(value), values)
            end
        end
    end
end
