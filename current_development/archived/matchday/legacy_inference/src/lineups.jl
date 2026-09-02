# current_development/match_day_inference/src/lineups.jl

using JSON3
using DataFrames
using Dates
using CurlHTTP
using LibPQ

"""
    load_lineup_from_json(filepath::String)

Loads SofaScore-format JSON lineups file and returns structured player entries.
"""
function load_lineup_from_json(filepath::String)
    if !isfile(filepath)
        return nothing
    end
    try
        content = read(filepath, String)
        data = JSON3.read(content)
        
        extract_players(side_data) = map(side_data.players) do p
            (
                player_id = Int(p.player.id),
                player_name = String(p.player.name),
                position = String(p.position),
                substitute = Bool(p.substitute),
                sofascore_rating = haskey(p, :avgRating) && !isnothing(p.avgRating) ? Float64(p.avgRating) : 0.0
            )
        end
        
        return (
            home = extract_players(data.home),
            away = extract_players(data.away)
        )
    catch e
        @warn "Failed to parse lineup JSON file $filepath: $e"
        return nothing
    end
end

"""
    get_most_recent_lineup(ds::Data.DataStore, team_name::String)

Finds the team's most recent starting/squad lineup from historical database records.
"""
function get_most_recent_lineup(ds::Data.DataStore, team_name::String)
    # Find matches involving this team
    team_matches = subset(ds.matches, 
        [:home_team, :away_team] => ByRow((h, a) -> h == team_name || a == team_name)
    )
    if isempty(team_matches)
        @warn "No historical matches found for team: $team_name"
        return []
    end
    
    # Sort chronologically to get the latest match
    sort!(team_matches, :match_date, rev=true)
    latest_match = Base.first(team_matches)  # qualify: a Main-scope `first` global would shadow this
    latest_mid = latest_match.match_id
    
    # Identify which side the team was playing on in that latest match
    side = latest_match.home_team == team_name ? "home" : "away"
    
    # Filter lineups for this match and side
    m_lineups = subset(ds.lineups, 
        :match_id => ByRow(==(latest_mid)),
        :team_side => ByRow(s -> String(s) == side)
    )
    
    if isempty(m_lineups)
        @warn "No lineups found in database for team $team_name in match $latest_mid"
        return []
    end
    
    return map(eachrow(m_lineups)) do row
        (
            player_id = Int(row.player_id),
            player_name = ismissing(row.player_name) ? "Unknown" : String(row.player_name),
            position = ismissing(row.position) ? "M" : String(row.position),
            substitute = coalesce(row.is_substitute, false),
            sofascore_rating = hasproperty(row, :rating) && !ismissing(row.rating) ? Float64(row.rating) : 0.0
        )
    end
end

"""
    fetch_provisional_lineup(match_id::Int)

Fetches the provisional (pre-match announced/predicted) lineup for `match_id` from the
`sofascore.lineup_provisional` table in betdb (:5433, via `BF_DB_URL`).

This is the scraper's tmp lineup feed: it holds the actual confirmed/predicted XI for
upcoming fixtures, unlike `sofascore.match_player_lineups` (which only fills in after a
match finishes). Player `rating` is not present pre-match, so `sofascore_rating` is set
to 0.0 — the model sources each player's strength from the historical rating tracker
keyed by `player_id`, not from this field.

Returns `(confirmed::Bool, home::Vector, away::Vector)`, or `nothing` if no rows exist.
"""
function fetch_provisional_lineup(match_id::Int)
    db_url = get(ENV, "BF_DB_URL") do
        error("BF_DB_URL is not set. Export it before fetching provisional lineups, e.g.\n" *
              "  export BF_DB_URL=\"postgresql://admin:<password>@100.124.38.117:5433/betdb\"")
    end
    db_conn = Data.connect_to_db(Data.DBConfig(db_url))
    try
        return fetch_provisional_lineup(db_conn, match_id)
    finally
        close(db_conn)
    end
end

"""
    fetch_provisional_lineup(db_conn::LibPQ.Connection, match_id::Int)

Provisional lineup fetch using an active LibPQ connection (see `fetch_provisional_lineup(::Int)`).
"""
function fetch_provisional_lineup(db_conn::LibPQ.Connection, match_id::Int)
    query = """
    SELECT player_id, player_name, position, substitute, is_home_team, confirmed
    FROM sofascore.lineup_provisional
    WHERE match_id = \$1;
    """
    df = DataFrame(LibPQ.execute(db_conn, query, (match_id,)))
    if isempty(df)
        return nothing
    end

    to_players(side_df) = map(eachrow(side_df)) do row
        (
            player_id = Int(row.player_id),
            player_name = ismissing(row.player_name) ? "Unknown" : String(row.player_name),
            position = ismissing(row.position) ? "M" : String(row.position),
            substitute = coalesce(row.substitute, false),
            sofascore_rating = 0.0  # not available pre-match; model uses historical tracker
        )
    end

    home_df = filter(:is_home_team => ==(true), df)
    away_df = filter(:is_home_team => ==(false), df)
    confirmed = any(coalesce.(df.confirmed, false))

    return (
        confirmed = confirmed,
        home = to_players(home_df),
        away = to_players(away_df)
    )
end

"""
    get_matchday_lineup(ds::Data.DataStore, match_id::Int, home_team::String, away_team::String, json_dir::String)

Retrieves the lineup for today's match. Priority:
  1. Local JSON override file `<match_id>.json` (manual pin).
  2. `sofascore.lineup_provisional` — the scraper's announced/predicted XI.
  3. Fallback: each team's most recent historical XI from `ds.lineups`.
"""
function get_matchday_lineup(ds::Data.DataStore, match_id::Int, home_team::String, away_team::String, json_dir::String)
    filepath = joinpath(json_dir, "$(match_id).json")
    if isfile(filepath)
        println("└─ [Lineup] Loaded from JSON file: $filepath")
        return load_lineup_from_json(filepath)
    end

    prov = fetch_provisional_lineup(match_id)
    if !isnothing(prov) && !isempty(prov.home) && !isempty(prov.away)
        tag = prov.confirmed ? "confirmed" : "provisional"
        println("└─ [Lineup] Loaded $tag XI from sofascore.lineup_provisional " *
                "(home N=$(length(prov.home)), away N=$(length(prov.away)))")
        return (home = prov.home, away = prov.away)
    end

    println("└─ [Lineup] No JSON or provisional lineup for match $match_id. " *
            "Falling back to most recent database lineup...")
    home_players = get_most_recent_lineup(ds, home_team)
    away_players = get_most_recent_lineup(ds, away_team)
    return (home = home_players, away = away_players)
end

"""
    fetch_lineup_from_sofascore(match_id::Int)

Fetches the live lineup directly from Sofascore API for a given match ID.
Returns a NamedTuple: `(confirmed = Bool, home = Vector, away = Vector)` or `nothing` if failed/empty.
"""
function fetch_lineup_from_sofascore(match_id::Int)
    
    url = "https://api.sofascore.com/api/v1/event/$(match_id)/lineups"
    headers = [
        "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ]
    
    # Initialize CurlEasy
    curl = CurlEasy(
        url = url,
        method = CurlHTTP.GET,
        verbose = false
    )
    
    # Execute request
    res, http_status, errormessage = curl_execute(curl, "", headers)
    
    if http_status != 200
        return nothing
    end
    
    body = String(curl.userdata[:databuffer])
    if isempty(body) || body == "null"
        return nothing
    end
    
    try
        data = JSON3.read(body)
        
        # If the response doesn't have home/away players, it might be empty or not released
        if !haskey(data, :home) || !haskey(data.home, :players)
            return nothing
        end
        
        confirmed = haskey(data, :confirmed) ? Bool(data.confirmed) : false
        
        extract_players(side_data) = map(side_data.players) do p
            (
                player_id = Int(p.player.id),
                player_name = String(p.player.name),
                position = String(p.position),
                substitute = Bool(p.substitute),
                sofascore_rating = haskey(p, :avgRating) && !isnothing(p.avgRating) ? Float64(p.avgRating) : 0.0
            )
        end
        
        return (
            confirmed = confirmed,
            home = extract_players(data.home),
            away = extract_players(data.away)
        )
    catch e
        @warn "Failed to parse Sofascore lineup response for match $match_id: $e"
        return nothing
    end
end

