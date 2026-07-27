# src/Data/fetchers/betfair_odds.jl

using DataFrames
using JSON3
using Dates
using TimeZones
using LibPQ

"""
Helper to map Betfair market_type to ds.odds taxonomy
"""
function map_market_info(m_type::String)
    if startswith(m_type, "OVER_UNDER_")
        # Extract "25" from "OVER_UNDER_25" and convert to 2.5
        line_val = parse(Float64, replace(m_type, "OVER_UNDER_" => "")) / 10.0
        return "OverUnder", line_val
    elseif m_type == "MATCH_ODDS"
        return "1X2", 0.0
    elseif m_type == "BOTH_TEAMS_TO_SCORE"
        return "BTTS", 0.0
    elseif m_type == "CORRECT_SCORE"
        return "CorrectScore", 0.0
    elseif m_type == "DOUBLE_CHANCE"
        return "DoubleChance", 0.0
    elseif m_type == "DRAW_NO_BET"
        return "DrawNoBet", 0.0
    elseif m_type == "ASIAN_HANDICAP"
        # Per-selection line is derived later (each key carries its own handicap).
        return "AsianHandicap", 0.0
    end
    return m_type, 0.0
end

"""
Helper to map selection symbols to ds.odds taxonomy
"""
function map_selection_symbol(sel::Symbol)
    s = string(sel)
    
    # 1. Handle Over/Under: over_2_5 -> :over_25
    if startswith(s, "over_") || startswith(s, "under_")
        parts = split(s, "_")
        return Symbol(parts[1], "_", join(parts[2:end]))
    
    # 2. Handle Correct Scores: "0_0" -> :cs_00
    elseif occursin(r"^\d+_\d+$", s) 
        return Symbol("cs_", replace(s, "_" => ""))
    
    # 3. Handle Any Other: "any_other_home" -> :cs_any_other_home
    elseif startswith(s, "any_other")
        return Symbol("cs_", s)

    # 4. Double Chance: align Betfair keys with the MarketDC taxonomy
    elseif s == "dc_home_draw"
        return :DC_1X
    elseif s == "dc_away_draw"
        return :DC_X2
    elseif s == "dc_home_away"
        return :DC_12

    # 5. Standard selections (dnb_home, ah_home_0_5, ...) remain as is
    else
        return sel
    end
end

function unpack_betfair_odds(raw_df::DataFrame)
    long_data = NamedTuple{
        (:match_id, :market_name, :market_line, :selection, :timestamp, :minutes_to_kickoff, :traded_price), 
        Tuple{Int32, String, Float64, Symbol, DateTime, Float64, Float64}
    }[]

    for row in eachrow(raw_df)
        ismissing(row.odds_data) && continue
        odds_json = JSON3.read(row.odds_data)
        !haskey(odds_json, :timestamps) && continue
        
        # Get standard taxonomy info
        market_name, market_line = map_market_info(row.market_type)
        
        kickoff_dt = DateTime(row.start_timestamp, Dates.UTC)
        kickoff_unix_ms = datetime2unix(kickoff_dt) * 1000.0
        raw_timestamps = odds_json[:timestamps]

        for (selection_key, odds_array) in pairs(odds_json)
            selection_key == :timestamps && continue
            
            # Map the selection symbol
            clean_selection = map_selection_symbol(selection_key)

            # Asian Handicap encodes a distinct line per selection key; derive the
            # canonical (home-perspective) line so paired sides share a de-vig group.
            sel_line = market_name == "AsianHandicap" ?
                Markets.ah_home_line(clean_selection) : market_line

            for (i, price) in enumerate(odds_array)
                price === nothing && continue
                
                ts_ms = raw_timestamps[i]
                mins_to_ko = (ts_ms - kickoff_unix_ms) / 60000.0
                
                push!(long_data, (
                    match_id = row.match_id,
                    market_name = market_name,
                    market_line = sel_line,
                    selection = clean_selection,
                    timestamp = unix2datetime(ts_ms / 1000.0),
                    minutes_to_kickoff = mins_to_ko,
                    traded_price = Float64(price)
                ))
            end
        end
    end
    return DataFrame(long_data)
end

function fetch_data(conn::LibPQ.Connection, t_ids::Vector{Int}, ::BetfairData)
    query = """
        SELECT 
            m.match_id,
            m.start_timestamp,
            mk.market_type,
            o.odds_data
        FROM sofascore.matches m
        INNER JOIN betfair.match_meta mm ON m.match_id = mm.match_id
        INNER JOIN betfair.odds_history o ON m.match_id = o.match_id
        INNER JOIN betfair.markets mk ON o.market_id = mk.market_id
        WHERE m.tournament_id = ANY(\$1)
        AND mm.status = 'SUCCESS'
        ORDER BY m.match_id ASC
    """
    
    try
        return DataFrame(LibPQ.execute(conn, query, [t_ids]))
    catch e
        @warn "Failed to fetch Betfair data: \$e. Returning empty DataFrame."
        return DataFrame(match_id=Int32[], start_timestamp=DateTime[], market_type=String[], odds_data=String[])
    end
end

function process_data(df::DataFrame, ::BetfairData; kwargs...)
    if nrow(df) == 0
        # Return empty dataframe with correct schema
        return DataFrame(match_id=Int32[], market_name=String[], market_line=Float64[], selection=Symbol[], timestamp=DateTime[], minutes_to_kickoff=Float64[], traded_price=Float64[])
    end
    return unpack_betfair_odds(df)
end
