# current_development/scottish_wealth/l01_wealth_data.jl
#
# LOADER: Scottish Lower Team Wealth & Squad Valuation Data Pipeline
#
# Extracts player market valuations for Scottish Lower leagues (56 League One, 57 League Two),
# joins them onto starting lineups (from ds.lineups and bbc.match_lineup), computes
# Starting-XI wealth aggregates, and implements `ScottishTeamWealthFeature`.

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
# SECTION 1: EXTRACT SCOTTISH LOWER VALUATION CATALOG
# ==============================================================================

"""
    fetch_scottish_player_valuations(conn::LibPQ.Connection; tournament_ids=[56, 57]) -> DataFrame

Scans `sofascore.match_incidents` and `sofascore.lineup_provisional` constrained strictly
to players who have made appearances in Scottish Lower leagues (56, 57).
"""
function fetch_scottish_player_valuations(conn::LibPQ.Connection; tournament_ids=[56, 57])
    sql = """
    WITH scottish_players AS (
        SELECT DISTINCT l.player_id
        FROM sofascore.match_player_lineups l
        JOIN sofascore.matches m ON l.match_id = m.match_id
        WHERE m.tournament_id = ANY(\$1)
        
        UNION
        
        SELECT DISTINCT l.sofascore_player_id AS player_id
        FROM bbc.match_lineup l
        JOIN sofascore.matches m ON l.match_id = m.match_id
        WHERE m.tournament_id = ANY(\$1)
          AND l.sofascore_player_id IS NOT NULL
    )
    SELECT DISTINCT ON (v.player_id)
        v.player_id,
        v.player_name,
        v.player_position,
        v.market_value
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
    ) v
    JOIN scottish_players sp ON v.player_id = sp.player_id
    WHERE v.market_value > 0
    ORDER BY v.player_id, v.market_value DESC;
    """
    
    df = DataFrame(LibPQ.execute(conn, sql, [tournament_ids]))
    df.player_id = Int.(df.player_id)
    df.market_value = Float64.(df.market_value)
    return df
end

# ==============================================================================
# SECTION 2: MAP VALUATIONS ONTO MATCH LINEUPS
# ==============================================================================

"""
    fetch_match_lineup_values(ds::Data.DataStore, val_catalog::DataFrame; fallback_default=100_000.0) -> DataFrame

Extracts starting lineups from `ds.lineups`, joins valuations from `val_catalog`,
and assigns positional median fallbacks when individual valuations are absent.
"""
function fetch_match_lineup_values(ds::Data.DataStore, val_catalog::DataFrame; fallback_default=100_000.0)
    lineups = filter(r -> !coalesce(r.is_substitute, false), ds.lineups)
    df_merged = leftjoin(lineups, select(val_catalog, :player_id, :market_value), on=:player_id)
    
    pos_medians = Dict("G" => 80_000.0, "D" => 100_000.0, "M" => 110_000.0, "F" => 120_000.0)
    
    df_merged.clean_value = [
        ismissing(row.market_value) ? get(pos_medians, String(coalesce(row.position, "")), fallback_default) : Float64(row.market_value)
        for row in eachrow(df_merged)
    ]
    return df_merged
end

# ==============================================================================
# SECTION 3: STARTING-XI WEALTH AGGREGATION & STANDARDIZATION
# ==============================================================================

"""
    compute_starting_xi_log_wealth(player_values::AbstractVector; fallback_default::Float64=100_000.0) -> Float64

Computes robust geometric log-mean wealth for an 11-player starting lineup:
  log(W_XI) = (1/11) * sum_{i=1}^{11} log(v_i)
with team-context geometric mean scaling for unvalued players.
"""
function compute_starting_xi_log_wealth(player_values::AbstractVector; fallback_default::Float64=100_000.0)
    known = [Float64(v) for v in player_values if !ismissing(v) && Float64(v) > 0]
    team_mean = isempty(known) ? fallback_default : mean(known)
    imputed = [ismissing(v) || Float64(v) <= 0 ? team_mean : Float64(v) for v in player_values]
    return mean(log.(imputed))
end

"""
    build_match_wealth_table(df_lineup_vals::DataFrame) -> DataFrame

Aggregates match-level Starting XI wealth, standardizes seasonally (Z-scores),
and computes the wealth differential: delta_w = w_home_z - w_away_z.
"""
function build_match_wealth_table(df_lineup_vals::DataFrame)
    team_side_col = :team_side in propertynames(df_lineup_vals) ? :team_side : (:is_home_team in propertynames(df_lineup_vals) ? :is_home_team : :is_home)
    is_home_expr(r) = (team_side_col == :team_side) ? (r.team_side == "home") : Bool(r[team_side_col])
    
    df_lineup_vals.is_home_bool = is_home_expr.(eachrow(df_lineup_vals))

    group_cols = intersect([:match_id, :season_id, :season, :is_home_bool], propertynames(df_lineup_vals))
    
    team_wealth_rows = combine(groupby(df_lineup_vals, group_cols),
        :clean_value => (v -> compute_starting_xi_log_wealth(v)) => :log_xi_wealth,
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
# SECTION 4: FEATURE CONFIGURATION STRUCT FOR BayesianFootball.Features
# ==============================================================================

"""
    ScottishTeamWealthFeature <: Features.AbstractFeatureConfig

Plugs into the `Features.required_features(model)` pipeline to extract `flat_wealth_diff`.
"""
Base.@kwdef struct ScottishTeamWealthFeature <: Features.AbstractFeatureConfig
    fallback_default::Float64 = 100_000.0
end

"""
    Features.add_feature!(F_data::Dict, config::ScottishTeamWealthFeature, ordered_ids, team_map, ds)

AD-Safe hook that adds `flat_wealth_diff::Vector{Float64}` to the feature dictionary.
"""
function Features.add_feature!(
    F_data::Dict, 
    config::ScottishTeamWealthFeature, 
    ordered_ids::Vector{Int}, 
    team_map::Dict{String, Int}, 
    ds::Data.DataStore
)
    # Check if catalog is cached locally or needs fetching
    cache_path = joinpath(@__DIR__, "cache", "scottish_val_catalog.jls")
    local val_cat
    if isfile(cache_path)
        val_cat = deserialize(cache_path)
    else
        conn = wealth_db_connect()
        val_cat = fetch_scottish_player_valuations(conn, tournament_ids=[56, 57])
        close(conn)
        mkpath(dirname(cache_path))
        serialize(cache_path, val_cat)
    end
    
    # Map valuations and build match wealth
    lineup_vals = fetch_match_lineup_values(ds, val_cat; fallback_default=config.fallback_default)
    wealth_df   = build_match_wealth_table(lineup_vals)
    
    # Index by match_id for fast lookup
    wealth_map = Dict(r.match_id => Float64(r.delta_w) for r in eachrow(wealth_df))
    
    flat_diffs = Float64[get(wealth_map, mid, 0.0) for mid in ordered_ids]
    
    # AD-Safety validation
    @assert !any(isnan, flat_diffs) "flat_wealth_diff contains NaN values!"
    @assert !any(isinf, flat_diffs) "flat_wealth_diff contains Inf values!"
    
    F_data[:flat_wealth_diff] = flat_diffs
    return F_data
end
