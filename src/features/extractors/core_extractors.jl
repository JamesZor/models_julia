# src/features/extractors/core_extractors.jl

# Fallback error for missing features
function add_feature!(F_data::Dict, config::AbstractFeatureConfig, ordered_ids, team_map::Dict, ds::Data.DataStore)
    error("No feature extractor defined for config type: $(typeof(config))")
end

# 1. Team IDs (Mapping match_id to vocabulary indices)
function add_feature!(F_data::Dict, ::TeamIDsFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    # Match ID -> (HomeTeamName, AwayTeamName)
    match_team_map = Dict(row.match_id => (row.home_team, row.away_team) for row in eachrow(ds.matches))
    
    F_data[:flat_home_ids] = [team_map[match_team_map[id][1]] for id in ordered_ids]
    F_data[:flat_away_ids] = [team_map[match_team_map[id][2]] for id in ordered_ids]
end

# 2. Goals (Actual scores)
function add_feature!(F_data::Dict, ::GoalsFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    # Match ID -> (HomeScore, AwayScore)
    score_map = Dict(row.match_id => (row.home_score, row.away_score) for row in eachrow(ds.matches))
    
    F_data[:flat_home_goals] = [Int(score_map[id][1]) for id in ordered_ids]
    F_data[:flat_away_goals] = [Int(score_map[id][2]) for id in ordered_ids]
end

# 3. League index (pooled multi-division segments — e.g. ScottishLower [56, 57]).
# League indices are keyed off the FULL DataStore (not the split) so they are stable across
# folds; :league_lookup (match_id -> league_idx) is stashed for prediction-time reconstruction.
function add_feature!(F_data::Dict, ::LeagueFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    league_ids = sort(unique(Int.(ds.matches.tournament_id)))
    league_map = Dict(t => i for (i, t) in enumerate(league_ids))
    league_lookup = Dict(Int(r.match_id) => league_map[Int(r.tournament_id)]
                         for r in eachrow(ds.matches))
    F_data[:flat_league_ids] = [league_lookup[Int(id)] for id in ordered_ids]
    F_data[:n_leagues]       = length(league_ids)
    F_data[:league_lookup]   = league_lookup
end
