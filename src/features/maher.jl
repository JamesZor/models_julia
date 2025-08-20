# Maher model feature extraction
function feature_map_basic_maher_model(data::SubDataFrame, mapping::MappedData) 
    home_teams_id = [mapping.team[team_name] for team_name in data.home_team]
    away_teams_id = [mapping.team[team_name] for team_name in data.away_team]
    leauge_ids = [mapping.league[string(league)] for league in data.tournament_id] 
    home_goals_st = data.home_score - data.home_score_ht
    away_goals_st = data.away_score - data.away_score_ht
    n_teams = length(mapping.team)
    n_leagues = length(unique(leauge_ids)) 
    
    return (
        home_team_ids=home_teams_id,
        away_team_ids=away_teams_id,
        leauge_ids=leauge_ids,
        goals_home_ft=data.home_score, 
        goals_away_ft=data.away_score,
        goals_home_ht=data.home_score_ht,
        goals_away_ht=data.away_score_ht,
        goals_home_st=home_goals_st,
        goals_away_st=away_goals_st,
        round=data.round,
        n_teams=n_teams,
        n_leauges=n_leagues,
    )
end
