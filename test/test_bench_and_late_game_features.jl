using Test
using BayesianFootball
using DataFrames
using Dates
using Distributions
import DynamicPPL
import LogDensityProblems
using ReverseDiff
using Statistics

const BDL_PG = BayesianFootball.Models.PreGame
const BDL_FEATURES = BayesianFootball.Features

function bdl_store(; matches=DataFrame(), lineups=DataFrame(), bbc_events=DataFrame())
    empty = DataFrame()
    return BayesianFootball.Data.DataStore(
        BayesianFootball.Data.ScottishLower(), matches,
        empty, empty, lineups, empty, empty, empty, bbc_events)
end

function bdl_matches()
    return DataFrame(
        match_id=Int[1, 2, 3, 4, 5],
        start_timestamp=DateTime[
            DateTime(2024, 1, 1, 15), DateTime(2024, 1, 8, 15),
            DateTime(2024, 1, 15, 15), DateTime(2024, 1, 15, 18),
            DateTime(2024, 1, 22, 15),
        ],
        home_team=["alpha", "gamma", "alpha", "alpha", "alpha"],
        away_team=["beta", "delta", "gamma", "delta", "gamma"],
    )
end

function bdl_extract(config, ds; ordered_ids=Int.(ds.matches.match_id), history=Int[1, 2, 3, 4])
    data = Dict{Symbol,Any}(:history_match_ids => Set(history))
    BDL_FEATURES.add_feature!(data, config, ordered_ids, Dict{String,Int}(), ds)
    return data
end

function bdl_bench_lineups(; future_multiplier=1.0)
    rows = NamedTuple[]
    # Three valid substitutes on every side. The total pairs produce differentials
    # with a non-zero history spread, so auto-scaling is testable.
    totals = Dict(
        1 => (600.0, 300.0), 2 => (300.0, 600.0),
        3 => (900.0, 300.0), 4 => (450.0, 450.0),
        5 => (1200.0 * future_multiplier, 300.0),
    )
    for match_id in 1:5, (side, total) in (("home", totals[match_id][1]),
                                           ("away", totals[match_id][2]))
        for player in 1:3
            push!(rows, (match_id=match_id, team_side=side, is_substitute=true,
                         proposed_market_value=total / 3,
                         valuation_timestamp=DateTime(2023, 12, 1), player_id=player))
        end
        # Starting-XI wealth must be invisible to this feature.
        push!(rows, (match_id=match_id, team_side=side, is_substitute=false,
                     proposed_market_value=1.0e12,
                     valuation_timestamp=DateTime(2023, 12, 1), player_id=99))
    end
    return DataFrame(rows)
end

@testset "BenchDepthFeature extraction, scaling, and PIT" begin
    matches = bdl_matches()
    ds = bdl_store(matches=matches, lineups=bdl_bench_lineups())
    config = BenchDepthFeature(log_transform=true, min_bench_count=3)
    data = bdl_extract(config, ds)

    raw = [log1p(600 / 1000) - log1p(300 / 1000),
           log1p(300 / 1000) - log1p(600 / 1000),
           log1p(900 / 1000) - log1p(300 / 1000), 0.0]
    sigma = std(raw)
    @test data[:flat_delta_bench_depth][1:4] ≈ raw ./ sigma
    @test data[:bench_depth_scale] ≈ sigma
    @test data[:flat_delta_bench_depth] isa Vector{Float64}
    @test all(isfinite, data[:flat_delta_bench_depth])
    @test data[:flat_bench_depth_fallback] == zeros(Int, 5)

    # A future target lineup cannot alter the frozen-history scale or any earlier row.
    rewritten = bdl_extract(
        config,
        bdl_store(matches=matches, lineups=bdl_bench_lineups(future_multiplier=1000.0)))
    @test rewritten[:flat_delta_bench_depth][1:4] ==
          data[:flat_delta_bench_depth][1:4]
    @test rewritten[:bench_depth_scale] == data[:bench_depth_scale]

    # A valuation at kickoff fails the strict stamp < kickoff guard.
    unsafe = bdl_bench_lineups()
    unsafe.valuation_timestamp[unsafe.match_id .== 3] .= matches.start_timestamp[3]
    guarded = bdl_extract(config, bdl_store(matches=matches, lineups=unsafe))
    @test guarded[:flat_delta_bench_depth][3] == 0.0
    @test guarded[:flat_bench_depth_fallback][3] == 1

    sparse = bdl_bench_lineups()[.!((bdl_bench_lineups().match_id .== 2) .&
                                      (bdl_bench_lineups().team_side .== "away") .&
                                      (bdl_bench_lineups().player_id .== 3)), :]
    sparse_data = bdl_extract(
        BenchDepthFeature(scale=1.0), bdl_store(matches=matches, lineups=sparse))
    @test sparse_data[:flat_delta_bench_depth][2] == 0.0
end

function bdl_shot_events(; future_late=false)
    rows = NamedTuple[]
    # Match-level late shares: m1=(0,1), m2=(1/2,0), m3=(1,0), m4=(0,1).
    minutes = Dict(
        1 => ([10, 20], [75, 80]),
        2 => ([10, 75], [20, 30]),
        3 => ([75, 80], [10, 20]),
        4 => ([10, 20], [75, 80]),
        5 => (future_late ? [75, 80] : [10, 20], [10, 20]),
    )
    for match_id in 1:5, (is_home, side_minutes) in
        ((true, minutes[match_id][1]), (false, minutes[match_id][2]))
        for minute in side_minutes
            push!(rows, (match_id=match_id, time=minute,
                         added_time=missing, event_type="attempt_saved",
                         is_home_event=is_home,
                         text="Right footed shot from the centre of the box."))
        end
    end
    events = DataFrame(rows)
    # Keep the fitted shot-cell rate strictly positive; every event shares the
    # same cell, so this does not alter any late/total proportion.
    events.event_type[1] = "goal"
    return events
end

@testset "LateGameChanceFeature rolling form and destructive PIT" begin
    matches = bdl_matches()
    config = LateGameChanceFeature(minute_threshold=70, half_life_matches=16.0)
    ds = bdl_store(matches=matches, bbc_events=bdl_shot_events())
    data = bdl_extract(config, ds)

    # m3 and m4 have different kickoffs on one day. Both must read only m1/m2 state.
    raw = [-0.5, 0.0]
    sigma = std(raw)
    @test data[:flat_delta_late_game_chance][1:2] == [0.0, 0.0]
    @test data[:flat_delta_late_game_chance][3] ≈ raw[1] / sigma
    @test data[:flat_delta_late_game_chance][4] ≈ raw[2] / sigma
    @test data[:late_game_chance_scale] ≈ sigma
    @test data[:flat_delta_late_game_chance] isa Vector{Float64}
    @test all(isfinite, data[:flat_delta_late_game_chance])

    # Rewriting only future match 5's minute distribution cannot move history.
    future = bdl_extract(config, bdl_store(
        matches=matches, bbc_events=bdl_shot_events(future_late=true)))
    @test future[:flat_delta_late_game_chance][1:4] ==
          data[:flat_delta_late_game_chance][1:4]
    @test future[:late_game_chance_scale] == data[:late_game_chance_scale]

    blank = bdl_extract(config, bdl_store(matches=matches, bbc_events=DataFrame()))
    @test blank[:flat_delta_late_game_chance] == zeros(5)
    @test blank[:flat_late_game_chance_fallback] == ones(Int, 5)
end

function bdl_feature_set(n=8)
    return BayesianFootball.FeatureSet(Dict{Symbol,Any}(
        :flat_home_ids => Int[isodd(i) ? 1 : 2 for i in 1:n],
        :flat_away_ids => Int[isodd(i) ? 2 : 1 for i in 1:n],
        :season_indices => ones(Int, n),
        :time_indices => ones(Int, n),
        :flat_months => Int[mod1(i, 12) for i in 1:n],
        :flat_home_goals => Int[mod(i, 3) for i in 1:n],
        :flat_away_goals => Int[mod(i + 1, 3) for i in 1:n],
        :dates => collect(0:(n - 1)),
        :flat_delta_bench_depth => collect(range(-1.0, 1.0; length=n)),
        :flat_delta_late_game_chance => collect(range(0.5, -0.5; length=n)),
        :n_teams => 2, :n_seasons => 1, :n_rounds => 1,
        :team_map => Dict("alpha" => 1, "beta" => 2),
    ))
end

@testset "Bench and late-game covariates compose into a static ReverseDiff tape" begin
    builder = CountModelBuilder(:bench_late_ad)
    add!(builder,
         BDL_PG.GlobalInterception(μ=Normal(0.2, 0.1)),
         BDL_PG.TimeDecayDynamics(days_half_life=180.0),
         BDL_PG.GlobalHomeAdvantage(),
         BenchDepthCovariate(log_transform=true),
         LateGameChanceCovariate(),
         NoGuard())
    model = build_count_model(builder)

    @test BDL_PG.cb_covariate_names(model) == (:bench_depth, :late_game_chance)
    required = BDL_FEATURES.required_features(model)
    @test any(feature -> feature isa BenchDepthFeature, required)
    @test any(feature -> feature isa LateGameChanceFeature, required)
    @test BenchDepthCovariate === BayesianFootball.Models.BenchDepthCovariate
    @test LateGameChanceCovariate === BDL_PG.LateGameChanceCovariate

    function tape_for(feature_set)
        turing_model = BDL_PG.build_turing_model(model, feature_set)
        varinfo = DynamicPPL.VarInfo(turing_model)
        turing_model(varinfo)
        theta = copy(varinfo[:])
        density = DynamicPPL.LogDensityFunction(turing_model)
        objective = x -> LogDensityProblems.logdensity(density, x)
        return ReverseDiff.GradientTape(objective, theta), objective, theta
    end

    tape, objective, theta = tape_for(bdl_feature_set(8))
    compiled = ReverseDiff.compile(tape)
    gradient = similar(theta)
    ReverseDiff.gradient!(gradient, compiled, theta)
    @test isfinite(objective(theta))
    @test all(isfinite, gradient)
    @test gradient ≈ ReverseDiff.gradient(objective, theta) rtol=1e-8 atol=1e-8

    long_tape, _, _ = tape_for(bdl_feature_set(32))
    @test length(long_tape.tape) == length(tape.tape)
end
