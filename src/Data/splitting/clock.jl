# Shared temporal-clock helpers for grouped walk-forward splitting.
#
# Stored match_week/match_biweek/match_month remain tournament-local for compatibility.
# Multi-tournament GroupedCVConfig folds instead derive a calendar-anchored clock here.

const CALENDAR_DYNAMICS_WIDTH_WEEKS = Dict(
    :match_week => 1,
    :match_biweek => 2,
    :match_month => 4,
)

_week_ending_sunday(date::Date) = date + Day(7 - dayofweek(date))
_match_kickoff(row) = DateTime(row.match_date) + Hour(row.match_hour)

function _uses_shared_calendar(group_ids::AbstractVector{<:Integer})
    length(unique(group_ids)) == length(group_ids) || error(
        "Tournament groups must not contain duplicate IDs: $(collect(group_ids))")
    return length(group_ids) > 1
end

"""
    _effective_step_map(matches, group_ids, season, dynamics_col)

Return `match_id => effective raw step` for one tournament group and season.
Singleton groups retain the stored dynamics column exactly. Multi-tournament groups use fixed,
calendar-anchored 7/14/28-day periods for `match_week`/`match_biweek`/`match_month`.
"""
function _effective_step_map(
    matches::AbstractDataFrame,
    group_ids::AbstractVector{<:Integer},
    season,
    dynamics_col::Symbol,
)::Dict{Int,Int}
    season_mask = coalesce.(matches.season .== season, false)
    group_mask = in.(matches.tournament_id, Ref(group_ids))
    season_df = matches[group_mask .& season_mask, :]
    isempty(season_df) && return Dict{Int,Int}()

    if !_uses_shared_calendar(group_ids)
        dynamics_col in propertynames(season_df) || error(
            "Dynamics column :$dynamics_col is absent from matches")
        return Dict(
            Int(row.match_id) => Int(row[dynamics_col]) for row in eachrow(season_df)
        )
    end

    width_weeks = get(CALENDAR_DYNAMICS_WIDTH_WEEKS, dynamics_col, nothing)
    isnothing(width_weeks) && error(
        "Pooled GroupedCVConfig requires a calendar-defined dynamics column; got " *
        ":$dynamics_col. Supported columns are " *
        "$(sort!(collect(keys(CALENDAR_DYNAMICS_WIDTH_WEEKS)))).")

    :match_date in propertynames(season_df) || error(
        "Pooled calendar splitting requires :match_date")
    any(ismissing, season_df.match_date) && error(
        "Pooled calendar splitting found missing match_date for group $(collect(group_ids)), " *
        "season $season")

    anchor = minimum(_week_ending_sunday.(Date.(season_df.match_date)))
    result = Dict{Int,Int}()
    for row in eachrow(season_df)
        week_end = _week_ending_sunday(Date(row.match_date))
        elapsed_weeks = div(Dates.value(week_end - anchor), 7)
        result[Int(row.match_id)] = cld(1 + elapsed_weeks, width_weeks)
    end
    return result
end

function _observed_effective_steps(matches, group_ids, season, dynamics_col)
    step_map = _effective_step_map(matches, group_ids, season, dynamics_col)
    return sort!(unique!(collect(values(step_map))))
end

function _next_observed_effective_step(matches, group_ids, season, dynamics_col, current_step)
    steps = _observed_effective_steps(matches, group_ids, season, dynamics_col)
    index = findfirst(>(current_step), steps)
    return isnothing(index) ? nothing : steps[index]
end

function _kickoff_map(
    matches::AbstractDataFrame,
    requested_ids::AbstractVector{<:Integer},
)::Dict{Int,DateTime}
    required = [:match_id, :match_date, :match_hour]
    missing_columns = setdiff(required, propertynames(matches))
    isempty(missing_columns) || error(
        "Temporal split safety requires columns $required; missing $missing_columns")

    requested = Set(Int.(requested_ids))
    relevant = subset(matches, :match_id => ByRow(id -> Int(id) in requested))
    any(ismissing, relevant.match_date) && error(
        "Temporal split safety found missing match_date in requested matches")
    any(ismissing, relevant.match_hour) && error(
        "Temporal split safety found missing match_hour in requested matches")
    return Dict(Int(row.match_id) => _match_kickoff(row) for row in eachrow(relevant))
end

"Assert that every fitted match kicks off strictly before every held-out match."
function _assert_temporal_safety(
    matches::AbstractDataFrame,
    fitted_ids::AbstractVector{<:Integer},
    heldout_ids::AbstractVector{<:Integer};
    group_ids,
    season,
    train_step,
    predict_step,
)
    isempty(fitted_ids) && return nothing
    isempty(heldout_ids) && error(
        "Cannot validate an empty held-out fold: group=$(collect(group_ids)), season=$season, " *
        "predict_step=$predict_step")

    requested_ids = unique(Int.(vcat(fitted_ids, heldout_ids)))
    kickoff_by_id = _kickoff_map(matches, requested_ids)
    unresolved = setdiff(requested_ids, collect(keys(kickoff_by_id)))
    isempty(unresolved) || error("Temporal split contains unresolved match IDs $unresolved")

    fitted_last_id = argmax(id -> kickoff_by_id[Int(id)], fitted_ids)
    heldout_first_id = argmin(id -> kickoff_by_id[Int(id)], heldout_ids)
    fitted_last = kickoff_by_id[Int(fitted_last_id)]
    heldout_first = kickoff_by_id[Int(heldout_first_id)]

    fitted_last < heldout_first || error(
        "Temporal split contamination: group=$(collect(group_ids)), season=$season, " *
        "train_step=$train_step, predict_step=$predict_step; fitted match " *
        "$(Int(fitted_last_id)) at $fitted_last is not before held-out match " *
        "$(Int(heldout_first_id)) at $heldout_first")
    return nothing
end
