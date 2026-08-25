module T001DevPatch

using BayesianFootball
using DataFrames
using Dates

const BF = BayesianFootball
const Data = BF.Data
const Features = BF.Features
const TI_WIDTH_WEEKS = Dict(:match_week => 1, :match_biweek => 2, :match_month => 4)

week_ending_sunday(d::Date) = d + Day(7 - dayofweek(d))
kickoff(row) = DateTime(row.match_date) + Hour(row.match_hour)

"Return the effective raw split step for each match in one group-season."
function effective_step_map(matches::AbstractDataFrame, group_ids::Vector{Int}, season,
                            dynamics_col::Symbol)
    mask = in.(matches.tournament_id, Ref(group_ids)) .& (matches.season .== season)
    season_df = matches[mask, :]
    isempty(season_df) && return Dict{Int,Int}()

    # T001 acceptance criterion 4: singleton groups retain the stored clock exactly.
    if length(group_ids) == 1
        return Dict(Int(row.match_id) => Int(row[dynamics_col]) for row in eachrow(season_df))
    end

    width = get(TI_WIDTH_WEEKS, dynamics_col, nothing)
    isnothing(width) && error(
        "T001 dev patch: pooled dynamics_col=$dynamics_col is not calendar-defined; " *
        "supported columns are $(sort!(collect(keys(TI_WIDTH_WEEKS))))")

    any(ismissing, season_df.match_date) && error(
        "T001 dev patch: missing match_date in pooled group $group_ids season $season")
    anchor = minimum(week_ending_sunday.(Date.(season_df.match_date)))

    result = Dict{Int,Int}()
    for row in eachrow(season_df)
        elapsed_weeks = div(Dates.value(week_ending_sunday(Date(row.match_date)) - anchor), 7)
        result[Int(row.match_id)] = cld(1 + elapsed_weeks, width)
    end
    return result
end

function observed_steps(matches, group_ids, season, dynamics_col)
    return sort!(unique!(collect(values(
        effective_step_map(matches, group_ids, season, dynamics_col)))))
end

function next_observed_step(matches, group_ids, season, dynamics_col, current_step)
    steps = observed_steps(matches, group_ids, season, dynamics_col)
    idx = findfirst(>(current_step), steps)
    return isnothing(idx) ? nothing : steps[idx]
end

function kickoff_map(matches::AbstractDataFrame)
    required = (:match_id, :match_date, :match_hour)
    missing_cols = setdiff(collect(required), propertynames(matches))
    isempty(missing_cols) || error("T001 dev patch: kickoff columns missing: $missing_cols")
    return Dict(Int(row.match_id) => kickoff(row) for row in eachrow(matches))
end

function assert_safe_fold(matches, fitted_ids, heldout_ids; group_ids, season,
                          train_step, predict_step)
    isempty(fitted_ids) && return nothing
    isempty(heldout_ids) && error(
        "T001 dev patch: attempted to emit an empty held-out fold for group $group_ids, " *
        "season $season, step $predict_step")

    ko = kickoff_map(matches)
    unresolved = setdiff(unique(vcat(Int.(fitted_ids), Int.(heldout_ids))), collect(keys(ko)))
    isempty(unresolved) || error("T001 dev patch: unresolved match IDs $unresolved")

    fitted_last_id = argmax(id -> ko[Int(id)], fitted_ids)
    heldout_first_id = argmin(id -> ko[Int(id)], heldout_ids)
    fitted_last = ko[Int(fitted_last_id)]
    heldout_first = ko[Int(heldout_first_id)]

    fitted_last < heldout_first || error(
        "T001 dev patch temporal safety failure: group=$group_ids season=$season " *
        "train_step=$train_step predict_step=$predict_step; fitted match " *
        "$(Int(fitted_last_id)) at $fitted_last is not before held-out match " *
        "$(Int(heldout_first_id)) at $heldout_first")
    return nothing
end

function pooled_group_boundaries(matches::DataFrame, group_ids::Vector{Int}, config,
                                 meta_type::Type)
    splits = Vector{Tuple{Data.SplitBoundary,Data.AbstractSplitMetaData}}()
    group_mask = in.(matches.tournament_id, Ref(group_ids))
    any(group_mask) || return splits
    all_seasons = sort(unique(matches.season))

    for target_season in config.target_seasons
        target_idx = findfirst(==(target_season), all_seasons)
        if isnothing(target_idx)
            @warn "T001 dev patch: target season $target_season absent for group $group_ids"
            continue
        end

        start_idx = max(1, target_idx - config.history_seasons)
        history_seasons = all_seasons[start_idx:(target_idx - 1)]
        history_ids = Int.(matches[
            group_mask .& in.(matches.season, Ref(history_seasons)), :match_id])

        step_of = effective_step_map(matches, group_ids, target_season, config.dynamics_col)
        isempty(step_of) && continue
        steps = sort!(unique!(collect(values(step_of))))
        ids_at(step) = Int[id for (id, value) in step_of if value == step]
        ids_through(step) = Int[id for (id, value) in step_of if value <= step]

        fold_counter = 1
        if !isempty(history_ids)
            predict_step = first(steps)
            heldout_ids = ids_at(predict_step)
            assert_safe_fold(matches, history_ids, heldout_ids;
                             group_ids, season=target_season,
                             train_step=0, predict_step)
            boundary = Data.SplitBoundary(fold_counter, 0, copy(history_ids), Int[])
            meta = meta_type(group_ids, target_season, target_season,
                             config.history_seasons, 0, config.warmup_period)
            push!(splits, (boundary, meta))
            fold_counter += 1
        end

        # Every emitted dynamic fold predicts the next observed bin. The terminal training
        # state has no OOS card and therefore does not become an empty fold.
        train_steps = steps[1:(end - 1)]
        filter!(>=(config.warmup_period), train_steps)
        if !isnothing(config.end_dynamics)
            filter!(<=(config.end_dynamics), train_steps)
        end

        for (position, train_step) in enumerate(steps[1:(end - 1)])
            train_step in train_steps || continue
            predict_step = steps[position + 1]
            target_ids = ids_through(train_step)
            heldout_ids = ids_at(predict_step)
            fitted_ids = vcat(history_ids, target_ids)
            assert_safe_fold(matches, fitted_ids, heldout_ids;
                             group_ids, season=target_season,
                             train_step, predict_step)

            boundary = Data.SplitBoundary(
                fold_counter,
                train_step,
                copy(history_ids),
                copy(target_ids),
            )
            meta = meta_type(group_ids, target_season, target_season,
                             config.history_seasons, train_step, config.warmup_period)
            push!(splits, (boundary, meta))
            fold_counter += 1
        end
    end
    return splits
end

function patched_create_id_boundaries(data_store, config::Data.GroupedCVConfig)
    splits = Vector{Tuple{Data.SplitBoundary,Data.GroupedSplitMetaData}}()
    for group in config.tournament_groups
        group_splits = if length(group) == 1
            # Call the incumbent internal helper so singleton boundaries are byte-for-byte
            # compatible rather than merely calendar-equivalent.
            Data._process_tournament_group_ids(
                data_store.matches, group, config, Data.GroupedSplitMetaData)
        else
            pooled_group_boundaries(
                data_store.matches, group, config, Data.GroupedSplitMetaData)
        end
        for (boundary, meta) in group_splits
            push!(splits, (boundary, meta::Data.GroupedSplitMetaData))
        end
    end
    return splits
end

function patched_create_data_splits(data_store, config::Data.GroupedCVConfig)
    splits = Vector{Tuple{SubDataFrame,Data.GroupedSplitMetaData}}()
    for group in config.tournament_groups
        if length(group) == 1
            group_splits = Data._process_tournament_group(
                data_store.matches, group, config, Data.GroupedSplitMetaData)
            for (view_df, meta) in group_splits
                push!(splits, (view_df, meta::Data.GroupedSplitMetaData))
            end
            continue
        end

        group_splits = pooled_group_boundaries(
            data_store.matches, group, config, Data.GroupedSplitMetaData)
        for (boundary, meta) in group_splits
            meta.time_step == 0 && continue # legacy API never emitted the history-only baseline
            ids = Set(vcat(boundary.history_match_ids, boundary.target_match_ids))
            rows = findall(in(ids), Int.(data_store.matches.match_id))
            push!(splits, (view(data_store.matches, rows, :), meta::Data.GroupedSplitMetaData))
        end
    end
    return splits
end

function patched_get_next_matches(ds::Data.DataStore, meta::Data.GroupedSplitMetaData,
                                  config::Data.GroupedCVConfig)
    if length(meta.tournament_ids) == 1
        return subset(
            ds.matches,
            :tournament_id => ByRow(in(meta.tournament_ids)),
            :season => ByRow(isequal(meta.target_season)),
            config.dynamics_col => ByRow(isequal(meta.time_step + 1)),
        )
    end

    predict_step = next_observed_step(
        ds.matches, meta.tournament_ids, meta.target_season,
        config.dynamics_col, meta.time_step)
    isnothing(predict_step) && return ds.matches[Int[], :]
    step_of = effective_step_map(
        ds.matches, meta.tournament_ids, meta.target_season, config.dynamics_col)
    heldout_ids = Set(id for (id, step) in step_of if step == predict_step)
    return subset(ds.matches, :match_id => ByRow(id -> Int(id) in heldout_ids))
end

"Replace the builder's positional group-count vector with a row-wise ID lookup."
function align_feature_time!(feature_set, boundary, meta, ds, dynamics_col)
    all_ids = Set(vcat(boundary.history_match_ids, boundary.target_match_ids))
    matches_df = subset(ds.matches, :match_id => ByRow(id -> Int(id) in all_ids))
    history_ids = Set(boundary.history_match_ids)
    target_ids = Set(boundary.target_match_ids)
    history_df = subset(matches_df, :match_id => ByRow(id -> Int(id) in history_ids))
    target_df = subset(matches_df, :match_id => ByRow(id -> Int(id) in target_ids))

    history_seasons = sort(unique(history_df.season))
    history_state = Dict(season => i for (i, season) in enumerate(history_seasons))
    history_times = Int[history_state[row.season] for row in eachrow(history_df)]

    group_ids = meta isa Data.GroupedSplitMetaData ? meta.tournament_ids : [meta.tournament_id]
    raw_step_of = effective_step_map(ds.matches, group_ids, meta.target_season, dynamics_col)
    observed_target_steps = sort(unique(raw_step_of[Int(id)] for id in target_df.match_id))
    target_state = Dict(step => i for (i, step) in enumerate(observed_target_steps))
    target_times = Int[
        length(history_seasons) + target_state[raw_step_of[Int(id)]]
        for id in target_df.match_id
    ]

    feature_set.data[:time_indices] = vcat(history_times, target_times)
    feature_set.data[:n_history_steps] = length(history_seasons)
    feature_set.data[:n_target_steps] = length(observed_target_steps)
    feature_set.data[:n_rounds] = length(history_seasons) + length(observed_target_steps)
    feature_set.data[:ordered_match_ids] = Int.(vcat(history_df.match_id, target_df.match_id))
    feature_set.data[:effective_target_steps] = Dict(
        Int(id) => raw_step_of[Int(id)] for id in target_df.match_id)
    return feature_set
end

function patched_create_features(splits, ds, model, dynamics_col)
    raw = [
        let feature_set = Features.create_features(boundary, ds, model, dynamics_col)
            align_feature_time!(feature_set, boundary, meta, ds, dynamics_col)
            (feature_set, meta)
        end
        for (boundary, meta) in splits
    ]
    return BF.TypesInterfaces.FeatureCollection(raw)
end

const PATCHED = Ref(false)

"Install the prototype as more-specific/replacement methods in the package modules."
function apply!()
    PATCHED[] && return nothing

    @eval Data begin
        function create_id_boundaries(data_store, config::GroupedCVConfig)
            return $T001DevPatch.patched_create_id_boundaries(data_store, config)
        end

        function create_data_splits(data_store, config::GroupedCVConfig)
            return $T001DevPatch.patched_create_data_splits(data_store, config)
        end

        function get_next_matches(ds::DataStore, meta::GroupedSplitMetaData,
                                  config::GroupedCVConfig)
            return $T001DevPatch.patched_get_next_matches(ds, meta, config)
        end
    end

    @eval Features begin
        function create_features(
            splits::Vector{<:Tuple{Data.SplitBoundary,<:Any}},
            ds::Data.DataStore,
            model::AbstractFootballModel,
            dynamics_col::Symbol=:match_month,
        )
            return $T001DevPatch.patched_create_features(
                splits, ds, model, dynamics_col)
        end
    end

    PATCHED[] = true
    return nothing
end

end # module T001DevPatch
