# current_development/manager_wealth/l01_manager_wealth_data.jl
#
# ==============================================================================
# LOADER: Manager Identity & Team Wealth Data Pipeline
# ==============================================================================
#
# PURPOSE:
#   Extracts and encodes manager identities (manager_home, manager_away) and
#   starting-XI market valuations into model-ready features for hierarchical
#   manager effect modeling and squad wealth weighting.
#
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

# Re-use wealth extraction utilities from team_wealth
include(joinpath(dirname(@__DIR__), "team_wealth", "l01_wealth_data.jl"))

# ==============================================================================
# SECTION 1: MANAGER EXTRACTION & CATALOG
# ==============================================================================

"""
    fetch_match_managers(conn::LibPQ.Connection; tournament_ids=[79, 718]) -> DataFrame

Queries `sofascore.matches` for manager assignments across target tournaments.
"""
function fetch_match_managers(conn::LibPQ.Connection; tournament_ids=[79, 718])
    t_list = join(tournament_ids, ", ")
    sql = """
    SELECT 
        match_id,
        tournament_id,
        season_id,
        home_team,
        away_team,
        manager_home,
        manager_away
    FROM sofascore.matches
    WHERE tournament_id IN ($t_list)
    ORDER BY start_timestamp, match_id;
    """
    df = DataFrame(LibPQ.execute(conn, sql))
    df.match_id = Int.(df.match_id)
    return df
end

"""
    build_manager_catalog(df_matches::DataFrame; fallback_name::String="Unknown Manager") -> Tuple{Dict{String, Int}, Dict{Int, Tuple{Int, Int}}}

Builds:
1. `manager_map`: mapping unique manager names to contiguous IDs 1..N_managers (with fallback_name at 1).
2. `match_manager_ids`: mapping match_id => (home_mgr_id, away_mgr_id).
"""
function build_manager_catalog(df_matches::DataFrame; fallback_name::String="Unknown Manager")
    all_names = String[]
    for row in eachrow(df_matches)
        if !ismissing(row.manager_home) && !isempty(strip(String(row.manager_home)))
            push!(all_names, strip(String(row.manager_home)))
        end
        if !ismissing(row.manager_away) && !isempty(strip(String(row.manager_away)))
            push!(all_names, strip(String(row.manager_away)))
        end
    end
    unique_names = sort(unique(all_names))
    
    # Manager 1 is reserved for fallback/unknown
    manager_map = Dict{String, Int}(fallback_name => 1)
    current_id = 2
    for name in unique_names
        if name != fallback_name
            manager_map[name] = current_id
            current_id += 1
        end
    end
    
    match_manager_ids = Dict{Int, Tuple{Int, Int}}()
    for row in eachrow(df_matches)
        m_id = Int(row.match_id)
        h_name = ismissing(row.manager_home) ? fallback_name : strip(String(row.manager_home))
        a_name = ismissing(row.manager_away) ? fallback_name : strip(String(row.manager_away))
        
        h_id = get(manager_map, h_name, 1)
        a_id = get(manager_map, a_name, 1)
        match_manager_ids[m_id] = (h_id, a_id)
    end
    
    return manager_map, match_manager_ids
end


# ==============================================================================
# SECTION 2: MANAGER FEATURE CONFIGURATION & HOOK
# ==============================================================================

"""
    ManagerFeature <: Features.AbstractFeatureConfig

Extracts manager IDs per match into contiguous integer vectors `flat_home_manager_id`
and `flat_away_manager_id`.
"""
Base.@kwdef struct ManagerFeature <: Features.AbstractFeatureConfig
    fallback_manager::String = "Unknown Manager"
end

"""
    Features.add_feature!(F_data::Dict, config::ManagerFeature, ordered_ids, team_map, ds)

The hook called by `Features.build_feature_set`.
"""
function Features.add_feature!(
    F_data::Dict,
    config::ManagerFeature,
    ordered_ids::Vector{Int},
    team_map::Dict,
    ds::Data.DataStore
)
    # 1. Fetch manager matches (with cache support for offline execution)
    cache_file = joinpath(dirname(dirname(@__DIR__)), "data", "l2_ireland_engines", "manager_catalog.jls")
    df_mgrs = nothing
    if isfile(cache_file)
        try
            df_mgrs = deserialize(cache_file)
        catch e
            @warn "Failed to read manager catalog cache: $e"
        end
    end
    
    if df_mgrs === nothing
        try
            conn = wealth_db_connect()
            t_ids = unique(Int.(ds.matches.tournament_id))
            df_mgrs = fetch_match_managers(conn; tournament_ids=t_ids)
            close(conn)
            try
                mkpath(dirname(cache_file))
                serialize(cache_file, df_mgrs)
            catch; end
        catch e
            @warn "Database connection failed ($e). Using ds.matches fallback for manager catalog."
            df_mgrs = DataFrame(
                match_id = Int.(ds.matches.match_id),
                tournament_id = Int.(ds.matches.tournament_id),
                season_id = Int.(ds.matches.season_id),
                home_team = String.(ds.matches.home_team),
                away_team = String.(ds.matches.away_team),
                manager_home = hasproperty(ds.matches, :manager_home) ? ds.matches.manager_home : fill(config.fallback_manager, nrow(ds.matches)),
                manager_away = hasproperty(ds.matches, :manager_away) ? ds.matches.manager_away : fill(config.fallback_manager, nrow(ds.matches))
            )
        end
    end
    
    # 2. Build catalog
    manager_map, match_manager_ids = build_manager_catalog(df_mgrs; fallback_name=config.fallback_manager)
    n_managers = length(manager_map)
    
    # 3. Populate flat arrays aligned with ordered_ids
    home_mgr_vec = Int[get(match_manager_ids, id, (1, 1))[1] for id in ordered_ids]
    away_mgr_vec = Int[get(match_manager_ids, id, (1, 1))[2] for id in ordered_ids]
    
    F_data[:flat_home_manager_id] = home_mgr_vec
    F_data[:flat_away_manager_id] = away_mgr_vec
    F_data[:n_managers]           = n_managers
    F_data[:manager_map]          = manager_map
    F_data[:match_manager_map]    = match_manager_ids
end
