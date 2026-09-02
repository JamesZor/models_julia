# current_development/match_day_inference/src/fixtures.jl

using LibPQ
using DataFrames
using Dates

"""
    fetch_todays_matches(segment::Data.DataTournemantSegment)

Fetches today's unstarted matches for the given tournament segment from the PostgreSQL database.
"""
function fetch_todays_matches(segment::Data.DataTournemantSegment)::AbstractDataFrame
    # Use the same instance as the DataStore: betdb (:5433) via BF_DB_URL.
    # Upcoming fixtures live in the raw scraper feed `sofascore.events`
    # (the curated `sofascore.matches` table holds finished matches only).
    db_url = get(ENV, "BF_DB_URL") do
        error("BF_DB_URL is not set. Export it before fetching fixtures, e.g.\n" *
              "  export BF_DB_URL=\"postgresql://admin:<password>@100.124.38.117:5433/betdb\"")
    end
    db_conn = Data.connect_to_db(Data.DBConfig(db_url))

    try
        return fetch_todays_matches(db_conn, segment)
    finally
        close(db_conn)
    end
end

"""
    fetch_todays_matches(db_conn::LibPQ.Connection, segment::Data.DataTournemantSegment)

Fetches today's unstarted matches for the given tournament segment using an active LibPQ database connection.
"""
function fetch_todays_matches(db_conn::LibPQ.Connection, segment::Data.DataTournemantSegment)::AbstractDataFrame
    query = """
    SELECT 
        match_id,
        home_team,
        away_team,
        round,
        tournament_id,
        season_id
    FROM
        sofascore.events
    WHERE
        status_type = 'notstarted'
        AND start_timestamp >= EXTRACT(EPOCH FROM CURRENT_DATE)
        AND start_timestamp < EXTRACT(EPOCH FROM CURRENT_DATE + INTERVAL '1 day')
        AND tournament_id = ANY(\$1);
    """

    t_ids = Data.tournament_ids(segment)
    df = DataFrame(LibPQ.execute(db_conn, query, (t_ids,)))

    df.match_week .= 999
    df.match_date .= today()

    return df
end

"""
    fetch_todays_matches(ds::Data.DataStore)

Wrapper to fetch today's matches using the DataStore segment.
"""
function fetch_todays_matches(ds::Data.DataStore)::AbstractDataFrame
    return fetch_todays_matches(ds.segment)
end
