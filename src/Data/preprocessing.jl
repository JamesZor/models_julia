# src/data/preprocessing.jl


using DataFrames
using Dates

# using ..Data: AbstractDataFrame, DataStore # Use AbstractDataFrame from Data module if defined there, else use DataFrames.AbstractDataFrame

export add_match_week_column, add_global_round_column, add_inital_odds_from_fractions! # Export user-facing names
export add_split_col_match_week


"""
    sunday_of_week(dt::Date)

Returns the date of the Sunday following the given date (or the date itself if it is Sunday).
Used to group matches occurring in the same week (Mon-Sun).
"""
function sunday_of_week(dt::Date)::Date
    day_num = dayofweek(dt)
    return dt + Day(7 - day_num)
end

"""
    add_match_week_column(matches_df::AbstractDataFrame)

Adds a ':match_week' column (Int) that resets for every season and tournament.
The first week of matches in a season becomes Week 1, the next Week 2, etc.

Groups by: [:tournament_id, :season]
"""
function add_match_week_column(matches_df::AbstractDataFrame)::DataFrame
    df = copy(matches_df)
    add_match_week_column!(df)
    return df
end

function add_match_week_column!(matches_df::AbstractDataFrame)
    df = matches_df # Work on a copy to avoid mutating the original
    
    # 1. Ensure global sort order first (Tournament -> Season -> Date)
    # This ensures that when we group, the data is relatively ordered, 
    # though the transform logic below explicitly handles date sorting too.
    sort!(df, [:tournament_id, :season, :match_date])

    # 2. Define the per-season logic
    # We take the vector of dates for a specific season, map them to Week Ending Sundays,
    # and then index those Sundays 1..N
    transform!(groupby(df, [:tournament_id, :season]), :match_date => (dates -> begin
        # A. Map distinct dates to their "Week Ending Sunday"
        #    (Matches Mon-Sun will share the same sunday_date)
        week_dates = sunday_of_week.(dates)
        
        # B. Find the unique weeks and sort them chronologically
        unique_weeks = sort(unique(week_dates))
        
        # C. Create a map: SundayDate -> Index (1, 2, 3...)
        week_map = Dict(w => i for (i, w) in enumerate(unique_weeks))
        
        # D. Map the original dates row-by-row to their Week Index
        return [week_map[w] for w in week_dates]
    end) => :match_week)

end


"""
Adds a ':global_round' column based on date sorting and team participation.
Returns a *new* DataFrame copy.
"""
function add_global_round_column(matches_df::AbstractDataFrame)::DataFrame
    df = copy(matches_df) # Work on a copy
    sort!(df, :match_date)
    num_matches = nrow(df)
    global_rounds = Vector{Int}(undef, num_matches)
    global_round_counter = 1
    teams_in_current_round = Set{String}()

    for (i, row) in enumerate(eachrow(df))
        home_team, away_team = row.home_team, row.away_team
        if home_team in teams_in_current_round || away_team in teams_in_current_round #
            global_round_counter += 1
            empty!(teams_in_current_round)
        end
        global_rounds[i] = global_round_counter
        push!(teams_in_current_round, home_team, away_team)
    end

    df.global_round = global_rounds
    return df
end



"""
Section for the odds to be converted, since we have then in fraction for the opening and closing lines.
Hence need to create them in a decimal form - standardise them.
"""


"""
Parses a fractional odds string (e.g., "19/10") into a decimal value (e.g., 2.9).
Returns 0.0 if parsing fails (e.g., for "SP", missing, or "1").
"""
function parse_fractional_to_decimal(s::Union{Missing, AbstractString})
    if ismissing(s)
        return 0.0
    end
    
    parts = split(s, '/')
    
    # Must be exactly two parts (numerator and denominator)
    if length(parts) != 2
        return 0.0
    end
    
    try
        n = parse(Float64, parts[1])
        d = parse(Float64, parts[2])
        
        # Avoid division by zero
        if d == 0.0
            return 0.0
        end
        
        # Convert from fractional (e.g., 1.9) to decimal (e.g., 2.9)
        return (n / d) + 1.0
    catch e
        # This will catch errors if parts[1] or parts[2] are not valid numbers (e.g., "SP")
        return 0.0
    end
end

function add_inital_odds_from_fractions!(df_odds::DataFrame)
  df_odds.initial_decimal = parse_fractional_to_decimal.(df_odds.initial_fractional_value);
  df_odds.initial_decimal = round.(df_odds.initial_decimal, digits=2);
end

function add_inital_odds_from_fractions!(ds::DataStore)
  add_inital_odds_from_fractions!(ds.odds);
end



"""
basic method to split the data from a certain week of the seaosn.
"""

function add_split_col_match_week(data_store::DataStore, week_number::Int64 )::DataStore
    all_matches_transformed = combine(groupby(data_store.matches, :season)) do sub_df
        df = DataFrame(sub_df)
        df = add_match_week_column(df)
        df.split_col = max.(0, df.match_week .- week_number)
        return df
    end
    
    return DataStore(
        data_store.segment,
        all_matches_transformed,
        data_store.statistics,
        data_store.odds,
        data_store.lineups,
        data_store.incidents,
        data_store.betfair_odds,
        data_store.bbc
    )
end
