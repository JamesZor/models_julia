# current_development/scottish_lower/corners/l01_corner_data.jl
#
# Data Ingestion & Enrichment Pipeline for Scottish League Corner Analysis
# Covers Scottish Premiership (54), Championship (55), League One (56), League Two (57)

using BayesianFootball
using DataFrames
using Dates
using Statistics
using LibPQ
using InlineStrings
using DotEnv

const env_path = joinpath(pkgdir(BayesianFootball), ".env")
if isfile(env_path)
    DotEnv.load!(ENV, env_path)
end

"""
    fetch_scottish_corner_dataset(; db_url::String = get(ENV, "BF_DB_URL", "")) -> DataFrame

Extracts match-level corner statistics, corner-derived goals, penalty goals, own goals,
and open-play goals across all Scottish tiers (54, 55, 56, 57).
"""
function fetch_scottish_corner_dataset(; db_url::String = get(ENV, "BF_DB_URL", "postgresql://admin:123456@100.124.38.117:5433/betdb"))::DataFrame
    conn = LibPQ.Connection(db_url)
    
    query = """
    WITH bbc_corners AS (
        SELECT 
            match_id,
            MAX(home_value) FILTER (WHERE stat_type = 'cornersWon') as corners_h,
            MAX(away_value) FILTER (WHERE stat_type = 'cornersWon') as corners_a
        FROM bbc.match_stats
        WHERE stat_type = 'cornersWon'
        GROUP BY match_id
    ),
    bbc_corner_events AS (
        SELECT 
            lt.match_id,
            COUNT(CASE WHEN lt.event_type = 'goal' AND lt.team = m.home_team AND lt.text ILIKE '%following a corner%' THEN 1 END) as corner_goals_h,
            COUNT(CASE WHEN lt.event_type = 'goal' AND lt.team = m.away_team AND lt.text ILIKE '%following a corner%' THEN 1 END) as corner_goals_a,
            COUNT(CASE WHEN lt.team = m.home_team AND lt.text ILIKE '%following a corner%' THEN 1 END) as corner_shots_h,
            COUNT(CASE WHEN lt.team = m.away_team AND lt.text ILIKE '%following a corner%' THEN 1 END) as corner_shots_a
        FROM bbc.live_text lt
        JOIN sofascore.matches m ON lt.match_id = m.match_id
        GROUP BY lt.match_id
    ),
    pen_incidents AS (
        SELECT 
            match_id,
            COUNT(CASE WHEN data->>'incidentClass' = 'penalty' AND is_home = true THEN 1 END) as pen_goals_h,
            COUNT(CASE WHEN data->>'incidentClass' = 'penalty' AND is_home = false THEN 1 END) as pen_goals_a,
            COUNT(CASE WHEN data->>'incidentClass' = 'ownGoal' AND is_home = true THEN 1 END) as og_goals_h,
            COUNT(CASE WHEN data->>'incidentClass' = 'ownGoal' AND is_home = false THEN 1 END) as og_goals_a
        FROM sofascore.match_incidents
        WHERE incident_type = 'goal'
        GROUP BY match_id
    )
    SELECT 
        m.match_id,
        m.tournament_id,
        m.season_id,
        s.name as season,
        m.start_timestamp as match_datetime,
        m.start_timestamp::date as match_date,
        m.home_team,
        m.away_team,
        m.home_score as goals_total_h,
        m.away_score as goals_total_a,
        COALESCE(c.corners_h, 0)::int as corners_h,
        COALESCE(c.corners_a, 0)::int as corners_a,
        COALESCE(bce.corner_shots_h, 0)::int as corner_shots_h,
        COALESCE(bce.corner_shots_a, 0)::int as corner_shots_a,
        COALESCE(bce.corner_goals_h, 0)::int as corner_goals_h,
        COALESCE(bce.corner_goals_a, 0)::int as corner_goals_a,
        COALESCE(p.pen_goals_h, 0)::int as pen_goals_h,
        COALESCE(p.pen_goals_a, 0)::int as pen_goals_a,
        COALESCE(p.og_goals_h, 0)::int as og_goals_h,
        COALESCE(p.og_goals_a, 0)::int as og_goals_a
    FROM sofascore.matches m
    JOIN sofascore.seasons s ON s.season_id = m.season_id AND s.tournament_id = m.tournament_id
    JOIN bbc.match_meta b ON m.match_id = b.match_id
    LEFT JOIN bbc_corners c ON m.match_id = c.match_id
    LEFT JOIN bbc_corner_events bce ON m.match_id = bce.match_id
    LEFT JOIN pen_incidents p ON m.match_id = p.match_id
    WHERE m.tournament_id IN (54, 55, 56, 57) 
      AND b.scores_match 
      AND m.status_type = 'finished'
    ORDER BY m.start_timestamp;
    """
    
    df = DataFrame(LibPQ.execute(conn, query))
    close(conn)
    
    # Calculate pure open-play goals: Total - Penalties - OwnGoals - CornerGoals
    df[!, :open_goals_h] = max.(0, df.goals_total_h .- df.pen_goals_h .- df.og_goals_h .- df.corner_goals_h)
    df[!, :open_goals_a] = max.(0, df.goals_total_a .- df.pen_goals_a .- df.og_goals_a .- df.corner_goals_a)
    
    # Total corners in match
    df[!, :corners_total] = df.corners_h .+ df.corners_a
    df[!, :corner_goals_total] = df.corner_goals_h .+ df.corner_goals_a
    df[!, :goals_total] = df.goals_total_h .+ df.goals_total_a

    return df
end
