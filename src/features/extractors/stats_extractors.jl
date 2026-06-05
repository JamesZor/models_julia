# src/features/extractors/stats_extractors.jl

# 1. Shots (Aggregate shots for the whole match)
function add_feature!(F_data::Dict, ::ShotsFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    # Filter statistics for "ALL" period and aggregate
    # match_id -> (home_shots, away_shots)
    stats_map = Dict(
        row.match_id => (
            coalesce(row.shotsOnGoal_home, 0.0) + coalesce(row.shotsOffGoal_home, 0.0) + coalesce(row.blockedScoringAttempt_home, 0.0),
            coalesce(row.shotsOnGoal_away, 0.0) + coalesce(row.shotsOffGoal_away, 0.0) + coalesce(row.blockedScoringAttempt_away, 0.0)
        ) 
        for row in eachrow(ds.statistics) if row.period == "ALL"
    )
    
    F_data[:flat_home_shots] = [get(stats_map, id, (NaN, NaN))[1] for id in ordered_ids]
    F_data[:flat_away_shots] = [get(stats_map, id, (NaN, NaN))[2] for id in ordered_ids]
end

# 2. Expected Goals (xG)
function add_feature!(F_data::Dict, ::XGFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    stats_map = Dict(
        row.match_id => (row.expectedGoals_home, row.expectedGoals_away) 
        for row in eachrow(ds.statistics) if row.period == "ALL"
    )
    
    F_data[:flat_home_xg] = [get(stats_map, id, (NaN, NaN))[1] for id in ordered_ids]
    F_data[:flat_away_xg] = [get(stats_map, id, (NaN, NaN))[2] for id in ordered_ids]
end

# 3. Big Chances Created (match-level, period == "ALL")
function add_feature!(F_data::Dict, ::BigChanceFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    stats_map = Dict(
        row.match_id => (coalesce(row.bigChanceCreated_home, NaN), coalesce(row.bigChanceCreated_away, NaN))
        for row in eachrow(ds.statistics) if row.period == "ALL"
    )

    F_data[:flat_home_big_chances] = [get(stats_map, id, (NaN, NaN))[1] for id in ordered_ids]
    F_data[:flat_away_big_chances] = [get(stats_map, id, (NaN, NaN))[2] for id in ordered_ids]
end

# 4. Total Shots Inside Box (match-level, period == "ALL")
function add_feature!(F_data::Dict, ::ShotsInsideBoxFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    stats_map = Dict(
        row.match_id => (coalesce(row.totalShotsInsideBox_home, NaN), coalesce(row.totalShotsInsideBox_away, NaN))
        for row in eachrow(ds.statistics) if row.period == "ALL"
    )

    F_data[:flat_home_shots_inside_box] = [get(stats_map, id, (NaN, NaN))[1] for id in ordered_ids]
    F_data[:flat_away_shots_inside_box] = [get(stats_map, id, (NaN, NaN))[2] for id in ordered_ids]
end

# 5. Final Third Entries (match-level, period == "ALL")
function add_feature!(F_data::Dict, ::FinalThirdEntriesFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    stats_map = Dict(
        row.match_id => (coalesce(row.finalThirdEntries_home, NaN), coalesce(row.finalThirdEntries_away, NaN))
        for row in eachrow(ds.statistics) if row.period == "ALL"
    )

    F_data[:flat_home_final_third_entries] = [get(stats_map, id, (NaN, NaN))[1] for id in ordered_ids]
    F_data[:flat_away_final_third_entries] = [get(stats_map, id, (NaN, NaN))[2] for id in ordered_ids]
end

# 6. Touches In Opposition Box (match-level, period == "ALL")
function add_feature!(F_data::Dict, ::TouchesInOppBoxFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    stats_map = Dict(
        row.match_id => (coalesce(row.touchesInOppBox_home, NaN), coalesce(row.touchesInOppBox_away, NaN))
        for row in eachrow(ds.statistics) if row.period == "ALL"
    )

    F_data[:flat_home_touches_opp_box] = [get(stats_map, id, (NaN, NaN))[1] for id in ordered_ids]
    F_data[:flat_away_touches_opp_box] = [get(stats_map, id, (NaN, NaN))[2] for id in ordered_ids]
end
