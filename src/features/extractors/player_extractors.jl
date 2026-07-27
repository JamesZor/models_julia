# src/features/extractors/player_extractors.jl

using DataFrames
using Statistics
using Dates

"""
    add_feature!(F_data::Dict, config::PlayerRatingsFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)

Extracts player positional ratings using a specific tracking algorithm defined in `config.tracker`.
"""
function add_feature!(F_data::Dict, config::PlayerRatingsFeature, ordered_ids, team_map::Dict, ds::Data.DataStore)
    # println("[INFO] Extracting Player Ratings Feature using $(typeof(config.tracker))...")

    # 1. Prepare base data with dates for sorting
    # Use select for efficiency
    lineups = select(ds.lineups, :match_id, :player_id, :team_side, :position, :rating, :minutes_played)
    matches_dates = select(ds.matches, :match_id, :match_date)
    
    # Join to get dates and sort chronologically
    df_lineups = innerjoin(lineups, matches_dates, on = :match_id)
    sort!(df_lineups, :match_date)

    # 2. Calculate Pre-Match Ratings using the tracker
    gdf = groupby(df_lineups, :player_id)
    transform!(gdf, :rating => (r -> calculate_player_ratings(config.tracker, r)) => :pre_match_rating)

    # 3. Handle debuts / cold starts (NaNs/Missing)
    # Filter valid ratings to find global mean
    valid_ratings = filter(x -> !ismissing(x) && !isnan(x), df_lineups.rating)
    global_avg = isempty(valid_ratings) ? 0.0 : mean(valid_ratings)
    
    df_lineups.pre_match_rating = coalesce.(replace(df_lineups.pre_match_rating, NaN => global_avg), global_avg)

    # 4. Minute Weighting
    mins = coalesce.(df_lineups.minutes_played, 0.0)
    df_lineups.weighted_rating = df_lineups.pre_match_rating .* (clamp.(mins, 0.0, 90.0) ./ 90.0)

    # 5. Positional Aggregation
    function clean_pos(p)
        if ismissing(p) || p == ""
            return "M" # Default to Midfield
        end
        return p
    end
    df_lineups.clean_position = clean_pos.(df_lineups.position)

    # Aggregate by match, side, and position (G, D, M, F)
    agg_df = combine(groupby(df_lineups, [:match_id, :team_side, :clean_position]), 
                     :weighted_rating => sum => :total_rating)

    # 6. Pivot to match level lookup map
    # match_id -> (side, pos) -> rating
    ratings_map = Dict{Int, Dict{Tuple{String, String}, Float64}}()
    
    for row in eachrow(agg_df)
        m_id = Int(row.match_id)
        if !haskey(ratings_map, m_id)
            ratings_map[m_id] = Dict{Tuple{String, String}, Float64}()
        end
        ratings_map[m_id][(row.team_side, row.clean_position)] = row.total_rating
    end

    # Helper to get rating or 0.0 if not found
    get_r(m_id, side, pos) = get(get(ratings_map, m_id, Dict()), (side, pos), 0.0)

    # 7. Fill F_data with the 8 flat vectors for the ordered_ids
    positions = ["G", "D", "M", "F"]
    sides = ["home", "away"]

    for side in sides
        for pos in positions
            key = Symbol("flat_$(side)_$(pos)_rating")
            F_data[key] = [get_r(id, side, pos) for id in ordered_ids]
        end
    end
    
    # Store the lookup map for ALL matches (supports OOS prediction)
    F_data[:player_ratings_map] = ratings_map
    
end
