# current_development/team_wealth/l01_wealth_data.jl
#
# ==============================================================================
# LOADER: Team Wealth & Squad Valuation Data Extraction Pipeline
# ==============================================================================
#
# PURPOSE:
#   Extracts player market valuations from `sofascore.match_incidents`, joins them
#   onto match lineups, aggregates starting-XI market values, and implements
#   `TeamWealthFeature` as a direct extension of `BayesianFootball.Features`.
#
# REPL / PAIR-PROGRAMMING WORKFLOW:
#   Send code block-by-block from Neovim to Kitty/Julia REPL.
# ==============================================================================

using LibPQ
using DataFrames
using Statistics
using StatsBase
using Printf
using Serialization

using BayesianFootball
const Features = BayesianFootball.Features
const Data     = BayesianFootball.Data

import BayesianFootball.Features: AbstractFeatureConfig, add_feature!, required_features

# ==============================================================================
# SECTION 0: DATABASE CONNECTION
# ==============================================================================

"""
    wealth_db_connect() -> LibPQ.Connection

Opens a PostgreSQL connection from `BF_DB_URL` with fallback to local betdb.
"""
function wealth_db_connect()
    url = get(ENV, "BF_DB_URL", "postgresql://admin:CpPhGzIZ2qHtAh6cJT%2FHHFovs0CqfTx6@192.168.1.88:5433/betdb?sslmode=disable")
    return LibPQ.Connection(url)
end


# ==============================================================================
# SECTION 1: EXTRACT PLAYER VALUATION CATALOG
# ==============================================================================

"""
    fetch_player_valuations(conn::LibPQ.Connection; tournament_ids=[79, 718]) -> DataFrame

Scans `sofascore.match_incidents.data` and `lineup_provisional` for player valuations.
Returns a deduplicated catalog of `player_id`, `player_name`, and `market_value` (EUR).
"""
function fetch_player_valuations(conn::LibPQ.Connection; tournament_ids=[79, 718])
    sql = """
    SELECT DISTINCT ON (player_id)
        player_id,
        player_name,
        player_position,
        market_value
    FROM (
        SELECT 
            (data->'player'->>'id')::int AS player_id,
            data->'player'->>'name' AS player_name,
            data->'player'->>'position' AS player_position,
            (data->'player'->'proposedMarketValueRaw'->>'value')::numeric AS market_value
        FROM sofascore.match_incidents
        WHERE data->'player'->'proposedMarketValueRaw'->>'value' IS NOT NULL
        
        UNION ALL
        
        SELECT 
            (data->'playerIn'->>'id')::int AS player_id,
            data->'playerIn'->>'name' AS player_name,
            data->'playerIn'->>'position' AS player_position,
            (data->'playerIn'->'proposedMarketValueRaw'->>'value')::numeric AS market_value
        FROM sofascore.match_incidents
        WHERE data->'playerIn'->'proposedMarketValueRaw'->>'value' IS NOT NULL
        
        UNION ALL
        
        SELECT 
            (data->'playerOut'->>'id')::int AS player_id,
            data->'playerOut'->>'name' AS player_name,
            data->'playerOut'->>'position' AS player_position,
            (data->'playerOut'->'proposedMarketValueRaw'->>'value')::numeric AS market_value
        FROM sofascore.match_incidents
        WHERE data->'playerOut'->'proposedMarketValueRaw'->>'value' IS NOT NULL
        
        UNION ALL
        
        SELECT 
            player_id,
            player_name,
            position AS player_position,
            (raw_data->'player'->'proposedMarketValueRaw'->>'value')::numeric AS market_value
        FROM sofascore.lineup_provisional
        WHERE raw_data->'player'->'proposedMarketValueRaw'->>'value' IS NOT NULL
    ) raw_vals
    ORDER BY player_id, market_value DESC;
    """
    
    df = DataFrame(LibPQ.execute(conn, sql))
    df.player_id = Int.(df.player_id)
    df.market_value = Float64.(df.market_value)
    return df
end


# ==============================================================================
# SECTION 2: FETCH MATCH LINEUPS & MAP VALUATIONS
# ==============================================================================

"""
    fetch_match_lineup_values(conn::LibPQ.Connection, val_catalog::DataFrame; tournament_id=79) -> DataFrame

Fetches starting XI for each match and joins player valuations.
"""
function fetch_match_lineup_values(conn::LibPQ.Connection, val_catalog::DataFrame; tournament_id=79)
    sql_lineups = """
    SELECT 
        m.match_id,
        m.season_id,
        s.name AS season_name,
        m.start_timestamp,
        m.home_team,
        m.away_team,
        l.player_id,
        l.player_name,
        l.position,
        l.is_home_team
    FROM sofascore.match_player_lineups l
    JOIN sofascore.matches m ON l.match_id = m.match_id
    JOIN sofascore.seasons s ON m.season_id = s.season_id
    WHERE m.tournament_id = \$1
      AND l.substitute = false
    ORDER BY m.start_timestamp, m.match_id, l.is_home_team DESC;
    """
    
    df_lineups = DataFrame(LibPQ.execute(conn, sql_lineups, [tournament_id]))
    df_lineups.player_id = Int.(df_lineups.player_id)
    
    df_merged = leftjoin(df_lineups, select(val_catalog, :player_id, :market_value), on=:player_id)
    
    pos_medians = Dict("G" => 80_000.0, "D" => 120_000.0, "M" => 130_000.0, "F" => 140_000.0)
    default_median = 100_000.0
    
    df_merged.clean_value = [
        ismissing(row.market_value) ? get(pos_medians, String(coalesce(row.position, "")), default_median) : Float64(row.market_value)
        for row in eachrow(df_merged)
    ]
    
    return df_merged
end

"""
    fetch_match_lineup_values(ds::Data.DataStore, val_catalog::DataFrame; fallback=100_000.0) -> DataFrame

Overload working directly from a cached `DataStore`.
"""
function fetch_match_lineup_values(ds::Data.DataStore, val_catalog::DataFrame; fallback=100_000.0)
    lineups = filter(r -> !coalesce(r.is_substitute, false), ds.lineups)
    df_merged = leftjoin(lineups, select(val_catalog, :player_id, :market_value), on=:player_id)
    
    df_merged.clean_value = [
        ismissing(row.market_value) ? fallback : Float64(row.market_value)
        for row in eachrow(df_merged)
    ]
    return df_merged
end


# ==============================================================================
# SECTION 3: MATCH-LEVEL WEALTH AGGREGATION & STANDARDIZATION
# ==============================================================================

"""
    compute_starting_xi_log_wealth(player_values::AbstractVector, fallback_default::Float64=100_000.0) -> Float64

Computes robust geometric log-mean with team-context mean scaling for missing values.
"""
function compute_starting_xi_log_wealth(player_values::AbstractVector, fallback_default::Float64=100_000.0)
    known = [Float64(v) for v in player_values if !ismissing(v) && Float64(v) > 0]
    team_mean = isempty(known) ? fallback_default : mean(known)
    imputed = [ismissing(v) || Float64(v) <= 0 ? team_mean : Float64(v) for v in player_values]
    return mean(log.(imputed))
end

"""
    build_match_wealth_table(df_lineup_vals::DataFrame) -> DataFrame

Aggregates starting-XI values, standardizes seasonally, and calculates `delta_w = w_home - w_away`.
"""
function build_match_wealth_table(df_lineup_vals::DataFrame)
    team_side_col = :team_side in propertynames(df_lineup_vals) ? :team_side : (:is_home_team in propertynames(df_lineup_vals) ? :is_home_team : :is_home)
    is_home_expr(r) = (team_side_col == :team_side) ? (r.team_side == "home") : Bool(r[team_side_col])
    
    df_lineup_vals.is_home_bool = is_home_expr.(eachrow(df_lineup_vals))

    group_cols = intersect([:match_id, :season_id, :season, :is_home_bool], propertynames(df_lineup_vals))
    
    team_wealth_rows = combine(groupby(df_lineup_vals, group_cols),
        :clean_value => compute_starting_xi_log_wealth => :log_xi_wealth,
        :clean_value => sum => :raw_xi_sum
    )
    
    home_df = filter(r -> r.is_home_bool, team_wealth_rows)
    away_df = filter(r -> !r.is_home_bool, team_wealth_rows)
    
    matches_wealth = innerjoin(
        select(home_df, :match_id, :log_xi_wealth => :log_w_h, :raw_xi_sum => :home_xi_val),
        select(away_df, :match_id, :log_xi_wealth => :log_w_a, :raw_xi_sum => :away_xi_val),
        on = :match_id
    )
    
    # Standardize across all matches in the set
    all_vals = vcat(matches_wealth.log_w_h, matches_wealth.log_w_a)
    mu_w = mean(all_vals)
    sigma_w = std(all_vals)
    sigma_w = (sigma_w == 0.0 || isnan(sigma_w)) ? 1.0 : sigma_w
    
    matches_wealth.w_h_z = (matches_wealth.log_w_h .- mu_w) ./ sigma_w
    matches_wealth.w_a_z = (matches_wealth.log_w_a .- mu_w) ./ sigma_w
    matches_wealth.delta_w = matches_wealth.w_h_z .- matches_wealth.w_a_z
    
    return matches_wealth
end


# ==============================================================================
# SECTION 4: MONKEY-PATCH / EXTENSION OF BayesianFootball.Features
# ==============================================================================

"""
    TeamWealthFeature <: AbstractFeatureConfig

Plugs into the `Features.required_features(model)` pipeline to extract `flat_wealth_diff`.
"""
Base.@kwdef struct TeamWealthFeature <: Features.AbstractFeatureConfig
    impute_default::Float64 = 100_000.0
end

"""
    Features.add_feature!(F_data::Dict, config::TeamWealthFeature, ordered_ids, team_map, ds)

The core hook that `src/features/builder.jl` calls during feature collection.
"""
function Features.add_feature!(
    F_data::Dict, 
    config::TeamWealthFeature, 
    ordered_ids::Vector{Int}, 
    team_map::Dict, 
    ds::Data.DataStore
)
    # 1. Fetch valuation catalog (with local cache support for server/offline execution)
    cache_file = joinpath(dirname(dirname(@__DIR__)), "data", "l2_ireland_engines", "player_val_catalog.jls")
    val_cat = nothing
    if isfile(cache_file)
        try
            val_cat = deserialize(cache_file)
        catch e
            @warn "Failed to read valuation cache: $e"
        end
    end
    
    if val_cat === nothing
        try
            conn = wealth_db_connect()
            t_ids = unique(Int.(ds.matches.tournament_id))
            val_cat = fetch_player_valuations(conn; tournament_ids=t_ids)
            close(conn)
            try
                mkpath(dirname(cache_file))
                serialize(cache_file, val_cat)
            catch; end
        catch e
            @warn "Database connection failed ($e). Using default median valuation catalog fallback."
            val_cat = DataFrame(player_id=Int[], player_name=String[], player_position=String[], market_value=Float64[])
        end
    end
    
    # 2. Join with lineups
    lineup_vals = fetch_match_lineup_values(ds, val_cat; fallback=config.impute_default)
    
    # 3. Build match wealth table
    match_wealth = build_match_wealth_table(lineup_vals)
    
    # 4. Generate aligned vector
    delta_w_map = Dict(r.match_id => r.delta_w for r in eachrow(match_wealth))
    w_h_map     = Dict(r.match_id => r.w_h_z   for r in eachrow(match_wealth))
    w_a_map     = Dict(r.match_id => r.w_a_z   for r in eachrow(match_wealth))
    
    F_data[:flat_wealth_diff] = Float64[get(delta_w_map, id, 0.0) for id in ordered_ids]
    F_data[:flat_home_wealth] = Float64[get(w_h_map, id, 0.0)     for id in ordered_ids]
    F_data[:flat_away_wealth] = Float64[get(w_a_map, id, 0.0)     for id in ordered_ids]
    
    # Stash lookup map for prediction time
    F_data[:wealth_lookup_map] = delta_w_map
end
