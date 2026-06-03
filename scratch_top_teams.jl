using DataFrames

df = ds.matches

# Combine home and away performance
home = select(df, :home_team => :team, :home_score => :goals_scored, :away_score => :goals_conceded, :winner_code => (w -> w .== 1) => :win, :winner_code => (w -> w .== 3) => :draw)
away = select(df, :away_team => :team, :away_score => :goals_scored, :home_score => :goals_conceded, :winner_code => (w -> w .== 2) => :win, :winner_code => (w -> w .== 3) => :draw)

all_matches = vcat(home, away)
team_stats = combine(groupby(all_matches, :team), 
    nrow => :matches, 
    :goals_scored => sum => :goals_scored, 
    :goals_conceded => sum => :goals_conceded,
    :win => sum => :wins,
    :draw => sum => :draws)

team_stats.points = team_stats.wins .* 3 .+ team_stats.draws .* 1
sort!(team_stats, [:points, :goals_scored], rev=true)

first(team_stats, 10)
