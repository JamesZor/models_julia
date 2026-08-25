using Test
using BayesianFootball
using DataFrames
using Dates
using Random

const SplitData = BayesianFootball.Data
const SplitFeatures = BayesianFootball.Features

struct SplitClockProbe <: BayesianFootball.AbstractFootballModel end
SplitFeatures.required_features(::SplitClockProbe) = SplitFeatures.AbstractFeatureConfig[
    SplitFeatures.TimeIndicesFeature(),
]

function split_store(segment, matches)
    empty = DataFrame()
    return SplitData.DataStore(
        segment, matches, empty, empty, empty, empty, empty, empty, empty)
end

kickoff(row) = DateTime(row.match_date) + Hour(row.match_hour)

function pooled_clock_matches()
    return DataFrame(
        match_id=Int[
            9001, 9002,                         # history
            1, 2,                               # target anchor
            3, 4,                               # last safe fitted calendar bin
            5,                                  # League Two extra week
            6, 7, 8,                            # 2024-10-19 shared slate
            9, 10,                              # next calendar bin
        ],
        tournament_id=Int[
            56, 57,
            56, 57,
            56, 57,
            57,
            56, 56, 57,
            56, 57,
        ],
        season=[
            "23/24", "23/24",
            "24/25", "24/25",
            "24/25", "24/25",
            "24/25",
            "24/25", "24/25", "24/25",
            "24/25", "24/25",
        ],
        match_date=Date.(String[
            "2024-05-04", "2024-05-04",
            "2024-08-03", "2024-08-03",
            "2024-10-05", "2024-10-05",
            "2024-10-12",
            "2024-10-19", "2024-10-19", "2024-10-19",
            "2024-10-26", "2024-10-26",
        ]),
        match_hour=Int[14, 14, 14, 14, 14, 14, 12, 14, 16, 14, 14, 14],
        # Deliberately reproduce the local-clock disagreement on 2024-10-19.
        match_week=Int[1, 1, 1, 1, 9, 9, 10, 10, 10, 11, 11, 12],
        match_biweek=Int[1, 1, 1, 1, 5, 5, 5, 5, 5, 6, 6, 6],
        match_month=Int[1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3],
        home_team=["h$id" for id in [9001, 9002, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        away_team=["a$id" for id in [9001, 9002, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
    )
end

@testset "Grouped calendar splitting" begin
    matches = pooled_clock_matches()
    ds = split_store(SplitData.ScottishLower(), matches)
    config = SplitData.GroupedCVConfig(
        tournament_groups=[[56, 57]],
        target_seasons=["24/25"],
        history_seasons=1,
        dynamics_col=:match_biweek,
        warmup_period=0,
        stop_early=false,
    )

    # Demonstrate the incumbent local-column failure encoded by this fixture.
    old_fitted = matches[
        (matches.season .== "24/25") .& (matches.match_biweek .<= 5), :]
    old_heldout = matches[
        (matches.season .== "24/25") .& (matches.match_biweek .== 6), :]
    @test maximum(kickoff.(eachrow(old_fitted))) >= minimum(kickoff.(eachrow(old_heldout)))
    @test maximum(kickoff.(eachrow(old_fitted))) == DateTime(2024, 10, 19, 16)
    @test minimum(kickoff.(eachrow(old_heldout))) == DateTime(2024, 10, 19, 14)

    boundaries = SplitData.create_id_boundaries(ds, config)
    @test !isempty(boundaries)

    # History-only baseline predicts the first target-season calendar bin.
    baseline, baseline_meta = first(boundaries)
    @test baseline_meta.time_step == 0
    @test Set(baseline.history_match_ids) == Set([9001, 9002])
    @test isempty(baseline.target_match_ids)
    @test Set(SplitData.get_next_matches(ds, (baseline, baseline_meta), config).match_id) ==
          Set([1, 2])

    # No emitted fold is empty or temporally contaminated.
    kickoff_by_id = Dict(Int(row.match_id) => kickoff(row) for row in eachrow(matches))
    for (boundary, meta) in boundaries
        heldout = DataFrame(SplitData.get_next_matches(ds, (boundary, meta), config))
        @test !isempty(heldout)
        fitted_ids = vcat(boundary.history_match_ids, boundary.target_match_ids)
        @test maximum(kickoff_by_id[id] for id in fitted_ids) <
              minimum(kickoff.(eachrow(heldout)))
        heldout_kickoffs = kickoff.(eachrow(heldout))
        @test maximum(heldout_kickoffs) - minimum(heldout_kickoffs) < Day(14)
    end

    # The shared bin containing 2024-10-19 holds both tournaments, including 14:00 and 16:00.
    focus_index = findfirst(boundaries) do pair
        heldout = SplitData.get_next_matches(ds, pair, config)
        Date("2024-10-19") in heldout.match_date
    end
    @test !isnothing(focus_index)
    focus_boundary, focus_meta = boundaries[focus_index]
    focus_heldout = DataFrame(SplitData.get_next_matches(
        ds, (focus_boundary, focus_meta), config))
    @test Set(focus_heldout.tournament_id) == Set([56, 57])
    @test 7 in focus_heldout.match_id # tournament 56 at 16:00 is held out, never fitted
    @test !(7 in focus_boundary.target_match_ids)

    # Calendar biweek 2 is wholly blank. Step 1 predicts the next observed step (3), not empty 2.
    gap_pair = only(filter(pair -> last(pair).time_step == 1, boundaries))
    gap_next = SplitData.get_next_matches(ds, gap_pair, config)
    @test Set(gap_next.match_id) == Set([3, 4])
    @test !(2 in [meta.time_step for (_, meta) in boundaries])

    # The legacy SubDataFrame API represents the same dynamic fitted ID sets.
    legacy = SplitData.create_data_splits(ds, config)
    relational_dynamic = filter(pair -> last(pair).time_step != 0, boundaries)
    legacy_ids = Dict(meta.time_step => Set(Int.(view_df.match_id)) for (view_df, meta) in legacy)
    relational_ids = Dict(
        meta.time_step => Set(vcat(boundary.history_match_ids, boundary.target_match_ids))
        for (boundary, meta) in relational_dynamic
    )
    @test legacy_ids == relational_ids

    # Feature time is assigned by ID, uses one state per shared observed bin, and is row-order safe.
    fitted_focus_index = findfirst(boundaries) do pair
        boundary = first(pair)
        8 in boundary.target_match_ids && 7 in boundary.target_match_ids
    end
    fitted_boundary, fitted_meta = boundaries[fitted_focus_index]
    feature_set = SplitFeatures.create_features(
        [(fitted_boundary, fitted_meta)], ds, SplitClockProbe(), config)[1][1]
    id_to_time = Dict(zip(
        feature_set.data[:ordered_match_ids], feature_set.data[:time_indices]))
    raw_steps = feature_set.data[:effective_target_steps]
    for raw_step in unique(values(raw_steps))
        ids = [id for (id, step) in raw_steps if step == raw_step]
        @test length(unique(id_to_time[id] for id in ids)) == 1
    end
    @test sort(unique(feature_set.data[:time_indices])) ==
          collect(1:feature_set.data[:n_rounds])

    shuffled_matches = matches[randperm(MersenneTwister(1001), nrow(matches)), :]
    shuffled_ds = split_store(SplitData.ScottishLower(), shuffled_matches)
    shuffled_boundaries = SplitData.create_id_boundaries(shuffled_ds, config)
    shuffled_pair = only(filter(pair -> last(pair).time_step == fitted_meta.time_step,
                                shuffled_boundaries))
    shuffled_feature_set = SplitFeatures.create_features(
        [shuffled_pair], shuffled_ds, SplitClockProbe(), config)[1][1]
    shuffled_id_to_time = Dict(zip(
        shuffled_feature_set.data[:ordered_match_ids],
        shuffled_feature_set.data[:time_indices]))
    @test id_to_time == shuffled_id_to_time

    # All supported pooled clocks have a fixed documented maximum held-out width.
    for (column, width_days) in [(:match_week, 7), (:match_biweek, 14), (:match_month, 28)]
        width_config = SplitData.GroupedCVConfig(
            tournament_groups=[[56, 57]],
            target_seasons=["24/25"],
            history_seasons=1,
            dynamics_col=column,
            warmup_period=0,
        )
        for pair in SplitData.create_id_boundaries(ds, width_config)
            heldout = SplitData.get_next_matches(ds, pair, width_config)
            heldout_kickoffs = kickoff.(eachrow(heldout))
            @test maximum(heldout_kickoffs) - minimum(heldout_kickoffs) < Day(width_days)
        end
    end

    # Pooled boundaries are prediction-bearing only. stop_early cannot reintroduce a terminal
    # empty fold; warmup applies after the independent history-only baseline.
    stop_config = SplitData.GroupedCVConfig(
        tournament_groups=[[56, 57]], target_seasons=["24/25"], history_seasons=1,
        dynamics_col=:match_biweek, warmup_period=0, stop_early=true)
    @test [(b.target_step, m.time_step) for (b, m) in
           SplitData.create_id_boundaries(ds, stop_config)] ==
          [(b.target_step, m.time_step) for (b, m) in boundaries]

    warm_config = SplitData.GroupedCVConfig(
        tournament_groups=[[56, 57]], target_seasons=["24/25"], history_seasons=1,
        dynamics_col=:match_biweek, warmup_period=5, end_dynamics=5)
    warm_steps = [meta.time_step for (_, meta) in
                  SplitData.create_id_boundaries(ds, warm_config)]
    @test warm_steps == [0, 5]

    @test_throws ErrorException SplitData.create_id_boundaries(
        ds,
        SplitData.GroupedCVConfig(
            tournament_groups=[[56, 56]], target_seasons=["24/25"],
            dynamics_col=:match_biweek),
    )

    # Missing kickoff data outside the fitted/held-out IDs must not invalidate a safe fold.
    unrelated = DataFrame(
        match_id=Int[1, 2, 999],
        match_date=Union{Missing,Date}[Date("2024-01-01"), Date("2024-01-08"), missing],
        match_hour=Union{Missing,Int}[14, 14, missing],
    )
    @test isnothing(SplitData._assert_temporal_safety(
        unrelated, [1], [2]; group_ids=[56, 57], season="24/25",
        train_step=1, predict_step=2))
    @test_throws ErrorException SplitData._assert_temporal_safety(
        unrelated, [2], [1]; group_ids=[56, 57], season="24/25",
        train_step=1, predict_step=2)
end

@testset "Singleton grouped folds remain on the stored clock" begin
    matches = DataFrame(
        match_id=Int[90, 1, 2, 3, 4, 5],
        tournament_id=fill(79, 6),
        season=["23/24", "24/25", "24/25", "24/25", "24/25", "24/25"],
        match_date=Date.(String[
            "2024-05-01", "2024-08-03", "2024-08-31",
            "2024-09-14", "2024-10-12", "2024-12-07",
        ]),
        match_hour=fill(14, 6),
        match_week=Int[1, 1, 2, 3, 4, 5],
        match_biweek=Int[1, 1, 1, 2, 2, 3],
        match_month=Int[1, 1, 1, 1, 1, 2],
        home_team=["h$id" for id in [90, 1, 2, 3, 4, 5]],
        away_team=["a$id" for id in [90, 1, 2, 3, 4, 5]],
    )
    snapshot(pair) = let boundary = first(pair), meta = last(pair)
        (boundary.fold_id, boundary.target_step, boundary.history_match_ids,
         boundary.target_match_ids, meta.time_step)
    end
    expected = [
        (1, 0, [90], Int[], 0),
        (2, 1, [90], [1, 2], 1),
        (3, 2, [90], [1, 2, 3, 4], 2),
        (4, 3, [90], [1, 2, 3, 4, 5], 3),
    ]

    for (segment, tournament_id) in [
        (SplitData.Ireland(), 79),
        (SplitData.IrelandFirstDivision(), 718),
        (SplitData.Veikkausliiga(), 31),
    ]
        singleton_matches = copy(matches)
        singleton_matches.tournament_id .= tournament_id
        ds = split_store(segment, singleton_matches)
        grouped = SplitData.GroupedCVConfig(
            tournament_groups=[[tournament_id]],
            target_seasons=["24/25"],
            history_seasons=1,
            dynamics_col=:match_biweek,
            warmup_period=1,
            stop_early=false,
        )
        single = SplitData.CVConfig(
            tournament_ids=[tournament_id],
            target_seasons=["24/25"],
            history_seasons=1,
            dynamics_col=:match_biweek,
            warmup_period=1,
            stop_early=false,
        )

        grouped_boundaries = SplitData.create_id_boundaries(ds, grouped)
        single_boundaries = SplitData.create_id_boundaries(ds, single)
        @test snapshot.(grouped_boundaries) == snapshot.(single_boundaries)
        @test snapshot.(grouped_boundaries) == expected
        @test [Int.(SplitData.get_next_matches(ds, pair, grouped).match_id)
               for pair in grouped_boundaries] == [[1, 2], [3, 4], [5], Int[]]
    end
end
