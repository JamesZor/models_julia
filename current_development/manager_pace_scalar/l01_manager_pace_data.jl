# current_development/manager_pace_scalar/l01_manager_pace_data.jl
#
# ==============================================================================
# LOADER: Scalar Manager Tactical Pace & Team Wealth Feature Pipeline
# ==============================================================================
#
# PURPOSE:
#   Computes pre-standardized empirical Bayes tactical pace Z-scores per manager
#   with shrinkage weight n0 = 15 pseudo-matches, generating a match-level
#   scalar pace_sum = Z_home_mgr + Z_away_mgr for single-parameter MCMC modeling.
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

function get_manager_matches_df(ds::Data.DataStore; fallback_manager::String="Unknown Manager")
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
                manager_home = hasproperty(ds.matches, :manager_home) ? ds.matches.manager_home : fill(fallback_manager, nrow(ds.matches)),
                manager_away = hasproperty(ds.matches, :manager_away) ? ds.matches.manager_away : fill(fallback_manager, nrow(ds.matches))
            )
        end
    end
    return df_mgrs
end

# ==============================================================================
# SECTION 2: SCALAR MANAGER PACE FEATURE CONFIG & HOOK
# ==============================================================================

"""
    ManagerPaceFeature <: Features.AbstractFeatureConfig

Computes empirical Bayes shrunk manager pace Z-scores:
1. Calculates historical total match goals per manager.
2. Shrinks toward league mean using prior weight `pseudo_matches = 15.0`.
3. Standardizes to Z-scores across managers.
4. Generates match-level scalar `pace_sum = Z_home_mgr + Z_away_mgr`.
"""
Base.@kwdef struct ManagerPaceFeature <: Features.AbstractFeatureConfig
    pseudo_matches::Float64 = 15.0
    fallback_manager::String = "Unknown Manager"
end

"""
    Features.add_feature!(F_data::Dict, config::ManagerPaceFeature, ordered_ids, team_map, ds)
"""
function Features.add_feature!(
    F_data::Dict,
    config::ManagerPaceFeature,
    ordered_ids::Vector{Int},
    team_map::Dict,
    ds::Data.DataStore
)
    # 1. Fetch manager matches mapping
    df_mgrs = get_manager_matches_df(ds; fallback_manager=config.fallback_manager)
    
    match_managers = Dict{Int, Tuple{String, String}}()
    for row in eachrow(df_mgrs)
        m_id = Int(row.match_id)
        h_name = ismissing(row.manager_home) || isempty(strip(String(row.manager_home))) ? config.fallback_manager : strip(String(row.manager_home))
        a_name = ismissing(row.manager_away) || isempty(strip(String(row.manager_away))) ? config.fallback_manager : strip(String(row.manager_away))
        match_managers[m_id] = (h_name, a_name)
    end
    
    # 2. Extract historical training slice goal totals
    matches_sub = filter(r -> Int(r.match_id) in ordered_ids, ds.matches)
    hist_goals = Float64[Float64(row.home_score + row.away_score) for row in eachrow(matches_sub) if !ismissing(row.home_score) && !ismissing(row.away_score)]
    
    μ_league = isempty(hist_goals) ? 2.65 : mean(hist_goals)
    σ_league = isempty(hist_goals) || length(hist_goals) < 2 ? 1.50 : max(std(hist_goals), 0.20)
    
    # 3. Calculate manager goal sums and counts in historical slice
    mgr_goal_sums = Dict{String, Float64}()
    mgr_match_counts = Dict{String, Int}()
    
    for row in eachrow(matches_sub)
        if ismissing(row.home_score) || ismissing(row.away_score)
            continue
        end
        m_id = Int(row.match_id)
        tot_g = Float64(row.home_score + row.away_score)
        h_mgr, a_mgr = get(match_managers, m_id, (config.fallback_manager, config.fallback_manager))
        
        if h_mgr != config.fallback_manager
            mgr_goal_sums[h_mgr] = get(mgr_goal_sums, h_mgr, 0.0) + tot_g
            mgr_match_counts[h_mgr] = get(mgr_match_counts, h_mgr, 0) + 1
        end
        if a_mgr != config.fallback_manager
            mgr_goal_sums[a_mgr] = get(mgr_goal_sums, a_mgr, 0.0) + tot_g
            mgr_match_counts[a_mgr] = get(mgr_match_counts, a_mgr, 0) + 1
        end
    end
    
    # 4. Apply Empirical Bayes Shrinkage and Compute Z-scores
    n0 = config.pseudo_matches
    manager_z_map = Dict{String, Float64}(config.fallback_manager => 0.0)
    
    for (mgr, n_m) in mgr_match_counts
        g_bar = mgr_goal_sums[mgr] / n_m
        g_hat = (n_m * g_bar + n0 * μ_league) / (n_m + n0)
        z_mgr = (g_hat - μ_league) / σ_league
        manager_z_map[mgr] = z_mgr
    end
    
    # 5. Populate match-level pace_sum aligned with ordered_ids
    pace_sum_vec = zeros(Float64, length(ordered_ids))
    for (i, m_id) in enumerate(ordered_ids)
        h_mgr, a_mgr = get(match_managers, m_id, (config.fallback_manager, config.fallback_manager))
        zh = get(manager_z_map, h_mgr, 0.0)
        za = get(manager_z_map, a_mgr, 0.0)
        pace_sum_vec[i] = zh + za
    end
    
    F_data[:pace_sum]           = pace_sum_vec
    F_data[:manager_z_map]      = manager_z_map
    F_data[:match_managers]     = match_managers
    F_data[:league_pace_mean]   = μ_league
    F_data[:league_pace_std]    = σ_league
    F_data[:fallback_manager]   = config.fallback_manager
end
