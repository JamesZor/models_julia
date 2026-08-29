function fetch_data(conn::LibPQ.Connection, t_ids::Vector{Int}, ::LineUpsData)
    # 1. Base Details (SofaScore primary with BBC fallback for missing match sheets)
    base_query = """
        WITH sofa_lineups AS (
            SELECT
                m.tournament_id, m.season_id, l.match_id,
                CASE WHEN l.is_home_team THEN 'home' ELSE 'away' END AS team_side,
                l.player_id, l.player_name, l.position, l.shirt_number,
                l.substitute AS is_substitute, l.captain AS is_captain,
                l.minutes_played, l.rating, l.goals, l.expected_goals, l.expected_assists,
                l.proposed_market_value, l.proposed_market_value_currency
            FROM sofascore.match_player_lineups l
            JOIN sofascore.matches m ON l.match_id = m.match_id
            WHERE m.tournament_id = ANY(\$1)
        ),
        sofa_matches_with_lineups AS (
            SELECT DISTINCT match_id FROM sofa_lineups
        ),
        player_market_values AS (
            SELECT DISTINCT ON (player_id)
                player_id,
                proposed_market_value,
                proposed_market_value_currency
            FROM sofascore.match_player_lineups
            WHERE proposed_market_value IS NOT NULL AND proposed_market_value > 0
            ORDER BY player_id, match_id DESC
        ),
        bbc_fallback_lineups AS (
            SELECT
                m.tournament_id, m.season_id, bl.match_id,
                CASE WHEN bl.is_home_team THEN 'home' ELSE 'away' END AS team_side,
                COALESCE(bl.sofascore_player_id, 0)::integer AS player_id,
                bl.bbc_name AS player_name,
                bl.position,
                bl.shirt_number,
                bl.is_substitute,
                bl.is_captain,
                NULL::integer AS minutes_played,
                NULL::double precision AS rating,
                NULL::integer AS goals,
                NULL::double precision AS expected_goals,
                NULL::double precision AS expected_assists,
                pv.proposed_market_value,
                COALESCE(pv.proposed_market_value_currency, 'EUR')::character varying AS proposed_market_value_currency
            FROM bbc.match_lineup bl
            JOIN sofascore.matches m ON bl.match_id = m.match_id
            LEFT JOIN player_market_values pv ON bl.sofascore_player_id = pv.player_id
            WHERE m.tournament_id = ANY(\$1)
            AND bl.match_id NOT IN (SELECT match_id FROM sofa_matches_with_lineups)
        )
        SELECT * FROM sofa_lineups
        UNION ALL
        SELECT * FROM bbc_fallback_lineups
    """
    local base_df
    try
        base_df = DataFrame(LibPQ.execute(conn, base_query, [t_ids]))
    catch e
        @warn "Failed to fetch LineUpsData (base_query): $(e)"
        return DataFrame()
    end
    if nrow(base_df) == 0; return base_df; end

    # 2. JSON Stats
    json_query = """
        SELECT 
            l.match_id, l.player_id, stats.key AS stat_key,
            (stats.value)::text AS stat_value
        FROM sofascore.match_player_lineups l
        JOIN sofascore.matches m ON l.match_id = m.match_id,
        jsonb_each(l.statistics) AS stats
        WHERE m.tournament_id = ANY(\$1) AND stats.key != 'ratingVersions'
    """
    local stats_long_df
    try
        stats_long_df = DataFrame(LibPQ.execute(conn, json_query, [t_ids]))
    catch e
        @warn "Failed to fetch LineUpsData (json_query): $(e)"
        base_df.assists .= missing
        return base_df
    end
    
    if nrow(stats_long_df) == 0
        base_df.assists .= missing 
        return base_df
    end

    # Parse and pivot stats
    stats_long_df.stat_value = passmissing(parse).(Float64, stats_long_df.stat_value)
    stats_wide_df = unstack(
        stats_long_df, [:match_id, :player_id], :stat_key, :stat_value, combine = first
    )

    # Remove overlapping columns before join
    overlapping_cols = setdiff(intersect(names(base_df), names(stats_wide_df)), ["match_id", "player_id"])
    if !isempty(overlapping_cols)
        select!(stats_wide_df, Not(overlapping_cols))
    end

    return leftjoin(base_df, stats_wide_df, on = [:match_id, :player_id])
end

function process_data(df::DataFrame, ::LineUpsData)
    desired_renames = Dict(
        "totalPass"    => "total_passes",
        "accuratePass" => "accurate_passes",
        "goalAssist"   => "assists",
        "duelWon"      => "duels_won",
        "duelLost"     => "duels_lost",
        "aerialWon"    => "aerials_won",
        "aerialLost"   => "aerials_lost"
    )
    
    valid_renames = [old => new for (old, new) in desired_renames if old in names(df)]
    if !isempty(valid_renames)
        rename!(df, valid_renames)
    end

    # Apply Schema
    schema = Dict{Symbol, Type}(
        :tournament_id => Int32,
        :season_id => Int32,
        :match_id => Int32,
        :team_side => InlineStrings.String31,
        :player_id => Int32,
        :player_name => Union{Missing, String},
        :position => Union{Missing, InlineStrings.String31},
        :shirt_number => Union{Missing, Int32},
        :is_substitute => Union{Missing, Bool},
        :is_captain => Union{Missing, Bool},
        :minutes_played => Union{Missing, Int32},
        :goals => Union{Missing, Int32},
        :proposed_market_value => Union{Missing, Int64},
        :proposed_market_value_currency => Union{Missing, InlineStrings.String3}
    )
    
    for col in names(df)
        sym = Symbol(col)
        if !haskey(schema, sym)
            schema[sym] = Union{Missing, Float64}
        end
    end
    apply_schema!(df, schema)

    return df
end
