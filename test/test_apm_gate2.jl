using Test
using BayesianFootball
using DataFrames

const APM_PROTOCOL_ROOT = joinpath(
    @__DIR__, "..", "current_development", "scottish_lower", "_protocol")
include(joinpath(APM_PROTOCOL_ROOT, "ScottishLowerProtocol.jl"))
using .ScottishLowerProtocol

struct APMGateProbe{F<:BayesianFootball.Features.AbstractPlusMinusFeature} <:
       BayesianFootball.AbstractFootballModel
    feature::F
end

BayesianFootball.Features.required_features(model::APMGateProbe) =
    BayesianFootball.Features.AbstractFeatureConfig[model.feature]

struct APMGateAdapter{M} <: AbstractSLModelAdapter
    model::M
end

ScottishLowerProtocol.sl_model(adapter::APMGateAdapter) = adapter.model

@testset "Shots RAPM feature — Gate 2" begin
    ds = BayesianFootball.Data.load_datastore_cached(
        BayesianFootball.Data.ScottishLower(); max_age_hours=10_000)
    config = BayesianFootball.Features.ShotsPlusMinusFeature()

    @testset "all four defaults are history-only" begin
        @test BayesianFootball.Features.ShotsPlusMinusFeature().fit_on === :history
        @test BayesianFootball.Features.ShotsOnTargetPlusMinusFeature().fit_on === :history
        @test BayesianFootball.Features.GoalsPlusMinusFeature().fit_on === :history
        @test BayesianFootball.Features.XGPlusMinusFeature().fit_on === :history
    end

    @testset "empty BBC/incident coverage emits neutral typed fallback" begin
        # Preserve schemas while removing the data RAPM needs for personnel events and shots.
        empty_coverage_ds = BayesianFootball.Data.DataStore(
            ds.segment, ds.matches, ds.statistics, ds.odds, ds.lineups,
            first(ds.incidents, 0), ds.betfair_odds, ds.bbc, first(ds.bbc_events, 0))
        ids = Int[first(ds.matches.match_id)]
        data = Dict{Symbol, Any}(:history_match_ids => Set(ids))
        BayesianFootball.Features.add_feature!(data, config, ids, Dict(), empty_coverage_ds)

        @test data[:flat_plus_minus_fallback] == Int[1]
        @test data[:plus_minus_fit_match_ids] == ids
        @test eltype(data[:flat_plus_minus_fallback]) === Int
        @test eltype(data[:plus_minus_fit_match_ids]) === Int
        for side in ("home", "away"), pos in ("G", "D", "M", "F")
            values = data[Symbol("flat_$(side)_$(pos)_rating")]
            @test eltype(values) === Float64
            @test values == Float64[0.0]
            @test !any(isnan, values)
        end
    end

    @testset "XG RAPM first-fold perturbation is history-safe" begin
        contract = sl_contract()
        fold = only(sl_build_folds(ds, contract)[1:1])
        xg_model = APMGateProbe(BayesianFootball.Features.XGPlusMinusFeature())
        full = BayesianFootball.Features.create_features(
            fold.boundary, ds, xg_model, sl_splitter(contract).dynamics_col)
        truncated_ds = sl_truncate_datastore(
            ds, vcat(fold.fitted_ids, Int.(fold.oos_df.match_id)))
        truncated = BayesianFootball.Features.create_features(
            fold.boundary, truncated_ds, xg_model, sl_splitter(contract).dynamics_col)
        same, differing = sl_featureset_equal(full, truncated)

        @test same
        @test isempty(differing)
    end

    @testset "all 20 Scottish Lower folds satisfy shared Gate 2" begin
        contract = sl_contract()
        folds = sl_build_folds(ds, contract)
        @test length(folds) == 20

        adapter = APMGateAdapter(APMGateProbe(config))
        results, feature_sets = sl_gate_features(ds, folds, adapter, contract)
        @test length(results) == 7
        @test all(result.pass for result in results)

        for fs in feature_sets
            @test fs.data[:plus_minus_fit_match_ids] ==
                  sort!(collect(fs.data[:history_match_ids]))
            @test eltype(fs.data[:plus_minus_fit_match_ids]) === Int
            @test eltype(fs.data[:flat_plus_minus_fallback]) === Int
            @test all(value -> value in (0, 1), fs.data[:flat_plus_minus_fallback])
            for side in ("home", "away"), pos in ("G", "D", "M", "F")
                values = fs.data[Symbol("flat_$(side)_$(pos)_rating")]
                @test eltype(values) === Float64
                @test !any(ismissing, values)
                @test !any(isnan, values)
            end
        end
    end
end
