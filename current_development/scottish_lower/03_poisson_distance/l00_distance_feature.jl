# current_development/scottish_lower/distance/l01_distance_features.jl
#
# LOADER: Scottish Lower Leagues Travel Distance & Geographic Fatigue Feature Pipeline
#
# Computes pairwise geographic Haversine distances, road distance estimates,
# log-distance transformations, and distance tiers for Scottish League One (#56)
# and Scottish League Two (#57) matches.
# Implements `ScottishDistanceFeature <: Features.AbstractFeatureConfig`.

using CSV
using DataFrames
using Dates
using Statistics
using StatsBase
using Printf

using BayesianFootball
const Features = BayesianFootball.Features
const Data     = BayesianFootball.Data

import BayesianFootball.Features: AbstractFeatureConfig, add_feature!, required_features

# ==============================================================================
# SECTION 1: GEODESIC & DISTANCE MATHEMATICS
# ==============================================================================

const EARTH_RADIUS_MILES = 3958.7613
const EARTH_RADIUS_KM    = 6371.0088

"""
    haversine_distance(lat1::Real, lon1::Real, lat2::Real, lon2::Real; unit=:miles) -> Float64

Computes the Great-Circle Haversine distance between two WGS84 GPS coordinates.
"""
function haversine_distance(lat1::Real, lon1::Real, lat2::Real, lon2::Real; unit::Symbol=:miles)
    unit in (:miles, :km) || throw(ArgumentError("unit must be :miles or :km, got $unit"))
    r = unit == :km ? EARTH_RADIUS_KM : EARTH_RADIUS_MILES
    
    phi1 = deg2rad(Float64(lat1))
    phi2 = deg2rad(Float64(lat2))
    dphi = deg2rad(Float64(lat2 - lat1))
    dlambda = deg2rad(Float64(lon2 - lon1))
    
    a = sin(dphi / 2.0)^2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0)^2
    a = clamp(a, 0.0, 1.0)
    c = 2.0 * atan(sqrt(a), sqrt(1.0 - a))
    
    return r * c
end

"""
    estimate_scottish_road_metrics(hav_miles::Real) -> (road_miles::Float64, drive_minutes::Float64)

Empirical road detour multiplier and speed model for Scottish topography.
Accounts for Firth of Forth/Tay crossings and Highland single-carriageway corridors:
- Short trips (< 25 mi): 1.20x detour, ~30 mph avg speed.
- Medium trips (25 - 80 mi): 1.25x detour, ~42 mph avg speed.
- Long / Highland trips (> 80 mi): 1.30x detour, ~48 mph avg speed.
"""
function estimate_scottish_road_metrics(hav_miles::Real)
    d = Float64(hav_miles)
    if d < 25.0
        detour = 1.20
        speed_mph = 30.0
    elseif d < 80.0
        detour = 1.25
        speed_mph = 42.0
    else
        detour = 1.30
        speed_mph = 48.0
    end
    road_miles = d * detour
    drive_minutes = (road_miles / speed_mph) * 60.0
    return (road_miles = road_miles, drive_minutes = drive_minutes)
end

"""
    distance_tier_category(miles::Real) -> Int

Classifies match distance into 4 operational tiers:
- 1: Local Derby (< 25 miles)
- 2: Moderate Travel (25 - 75 miles)
- 3: Long Haul (75 - 140 miles)
- 4: Extreme Highland / Border Travel (> 140 miles)
"""
function distance_tier_category(miles::Real)
    m = Float64(miles)
    m < 25.0 && return 1
    m < 75.0 && return 2
    m < 140.0 && return 3
    return 4
end

# ==============================================================================
# SECTION 2: GEOCODING CATALOG LOADER & DISTANCE TABLE BUILDER
# ==============================================================================

const DEFAULT_GEOCODES_CSV = joinpath(@__DIR__, "scottish_stadium_geocodes.csv")

"""
    load_scottish_stadium_catalog(csv_path::String = DEFAULT_GEOCODES_CSV) -> DataFrame

Loads the 31-ground geocoded coordinates registry.
"""
function load_scottish_stadium_catalog(csv_path::String = DEFAULT_GEOCODES_CSV)
    if !isfile(csv_path)
        error("Stadium geocodes file not found at: $(csv_path)")
    end
    df = CSV.read(csv_path, DataFrame)
    required = (:team_slug, :latitude, :longitude)
    all(name -> name in propertynames(df), required) ||
        error("Stadium catalog must contain columns $(collect(required))")

    df.team_slug = String.(df.team_slug)
    df.latitude  = Float64.(df.latitude)
    df.longitude = Float64.(df.longitude)

    length(unique(df.team_slug)) == nrow(df) || error("Stadium catalog contains duplicate team_slug values")
    all(isfinite, df.latitude) && all(isfinite, df.longitude) ||
        error("Stadium catalog coordinates must be finite")
    all(x -> -90.0 <= x <= 90.0, df.latitude) || error("Stadium latitude outside [-90, 90]")
    all(x -> -180.0 <= x <= 180.0, df.longitude) || error("Stadium longitude outside [-180, 180]")
    return df
end

"""
    catalog_distance_standardization(geocodes_df) -> NamedTuple

Returns fixed raw- and log-distance centering constants computed from every
ordered pair in the versioned stadium catalog. Unlike match-sample moments,
these constants cannot change when future fixtures are added or removed.
"""
function catalog_distance_standardization(geocodes_df::DataFrame)
    n = nrow(geocodes_df)
    n >= 2 || throw(ArgumentError("at least two stadiums are required for standardization"))

    distances = Float64[]
    sizehint!(distances, n * (n - 1))
    for i in 1:n, j in 1:n
        i == j && continue
        push!(distances, haversine_distance(
            geocodes_df.latitude[i], geocodes_df.longitude[i],
            geocodes_df.latitude[j], geocodes_df.longitude[j]))
    end

    log_distances = log1p.(distances)
    raw_scale = std(distances; corrected=false)
    log_scale = std(log_distances; corrected=false)
    return (
        raw_center = mean(distances),
        raw_scale = raw_scale > 0.0 ? raw_scale : 1.0,
        log_center = mean(log_distances),
        log_scale = log_scale > 0.0 ? log_scale : 1.0,
    )
end

"""
    build_match_distance_table(matches_df; geocodes_df, standardization) -> DataFrame

Computes pairwise distances for every match in `matches_df`. Standardization is
catalog-fixed by default, so deleting future matches cannot alter past values.
"""
function build_match_distance_table(
    matches_df::DataFrame;
    geocodes_df::DataFrame = load_scottish_stadium_catalog(),
    standardization = catalog_distance_standardization(geocodes_df),
)
    geo_map = Dict{String, Tuple{Float64, Float64}}()
    for row in eachrow(geocodes_df)
        geo_map[row.team_slug] = (row.latitude, row.longitude)
    end
    
    n = nrow(matches_df)
    hav_miles = zeros(Float64, n)
    hav_km    = zeros(Float64, n)
    road_mi   = zeros(Float64, n)
    drive_min = zeros(Float64, n)
    log_mi    = zeros(Float64, n)
    tiers     = zeros(Int, n)
    is_midwk  = zeros(Float64, n)
    fallback  = zeros(Int, n)
    
    for i in 1:n
        h_team = String(matches_df.home_team[i])
        a_team = String(matches_df.away_team[i])
        
        h_coord = get(geo_map, h_team, nothing)
        a_coord = get(geo_map, a_team, nothing)
        
        if isnothing(h_coord) || isnothing(a_coord)
            # Deterministic population fallback for promoted or unmapped grounds.
            dist_m = 45.0
            fallback[i] = 1
        else
            dist_m = haversine_distance(h_coord[1], h_coord[2], a_coord[1], a_coord[2]; unit=:miles)
        end
        
        road_est = estimate_scottish_road_metrics(dist_m)
        
        hav_miles[i] = dist_m
        hav_km[i]    = dist_m * 1.60934
        road_mi[i]   = road_est.road_miles
        drive_min[i] = road_est.drive_minutes
        log_mi[i]    = log(1.0 + dist_m)
        tiers[i]     = distance_tier_category(dist_m)
        
        # Midweek match flag (Tuesday = 2, Wednesday = 3, Thursday = 4)
        if :match_date in propertynames(matches_df)
            dow = dayofweek(matches_df.match_date[i])
            is_midwk[i] = (dow in (2, 3, 4)) ? 1.0 : 0.0
        elseif :start_timestamp in propertynames(matches_df)
            dow = dayofweek(Date(matches_df.start_timestamp[i]))
            is_midwk[i] = (dow in (2, 3, 4)) ? 1.0 : 0.0
        else
            is_midwk[i] = 0.0
        end
    end
    
    # Catalog-fixed standardization: no moments are fitted on fixture rows.
    dist_z = (hav_miles .- standardization.raw_center) ./ standardization.raw_scale
    log_dist_z = (log_mi .- standardization.log_center) ./ standardization.log_scale
    
    result = DataFrame(
        match_id        = Int.(matches_df.match_id),
        home_team       = matches_df.home_team,
        away_team       = matches_df.away_team,
        hav_miles       = hav_miles,
        hav_km          = hav_km,
        road_miles      = road_mi,
        drive_minutes   = drive_min,
        log_miles       = log_mi,
        dist_z          = dist_z,
        log_dist_z      = log_dist_z,
        distance_tier   = tiers,
        is_midweek      = is_midwk,
        distance_fallback = fallback,
    )
    
    return result
end

# ==============================================================================
# SECTION 3: FEATURE CONFIGURATION STRUCT FOR BayesianFootball.Features
# ==============================================================================

"""
    ScottishDistanceFeature <: Features.AbstractFeatureConfig

Plugs into the `Features.required_features(model)` pipeline to extract
distance travelled features (`flat_distance`, `flat_distance_z`, `flat_log_distance_z`, etc.).

# Fields:
- `metric::Symbol = :log_dist_z` (options: `:log_dist_z`, `:dist_z`, `:hav_miles`, `:road_miles`, `:drive_minutes`)
- `include_midweek::Bool = true`
- `geocodes_csv::String = DEFAULT_GEOCODES_CSV`
"""
Base.@kwdef struct ScottishDistanceFeature <: Features.AbstractFeatureConfig
    metric::Symbol = :log_dist_z
    include_midweek::Bool = true
    geocodes_csv::String = DEFAULT_GEOCODES_CSV
end

"""
    Features.add_feature!(F_data::Dict, config::ScottishDistanceFeature, ordered_ids::Vector{Int}, team_map::Dict, ds::Data.DataStore)

AD-Safe hook that adds travel distance vectors to `F_data`.
"""
function Features.add_feature!(
    F_data::Dict, 
    config::ScottishDistanceFeature, 
    ordered_ids::Vector{Int}, 
    team_map::Dict, 
    ds::Data.DataStore
)
    geocodes_df = load_scottish_stadium_catalog(config.geocodes_csv)
    selected_ids = Set(ordered_ids)
    selected_matches = subset(ds.matches, :match_id => ByRow(id -> Int(id) in selected_ids))
    dist_df = build_match_distance_table(selected_matches; geocodes_df=geocodes_df)

    # Fast match_id lookup
    dist_row_map = Dict(r.match_id => r for r in eachrow(dist_df))
    
    n = length(ordered_ids)
    flat_dist_z     = zeros(Float64, n)
    flat_log_dist_z = zeros(Float64, n)
    flat_hav_miles  = zeros(Float64, n)
    flat_road_miles = zeros(Float64, n)
    flat_drive_mins = zeros(Float64, n)
    flat_tiers      = zeros(Int, n)
    flat_is_midweek = zeros(Float64, n)
    flat_fallback   = zeros(Int, n)
    
    for (i, mid) in enumerate(ordered_ids)
        row = get(dist_row_map, mid, nothing)
        if isnothing(row)
            flat_dist_z[i]     = 0.0
            flat_log_dist_z[i] = 0.0
            flat_hav_miles[i]  = 45.0
            flat_road_miles[i] = 56.0
            flat_drive_mins[i] = 75.0
            flat_tiers[i]      = 2
            flat_is_midweek[i] = 0.0
            flat_fallback[i]   = 1
        else
            flat_dist_z[i]     = Float64(row.dist_z)
            flat_log_dist_z[i] = Float64(row.log_dist_z)
            flat_hav_miles[i]  = Float64(row.hav_miles)
            flat_road_miles[i] = Float64(row.road_miles)
            flat_drive_mins[i] = Float64(row.drive_minutes)
            flat_tiers[i]      = Int(row.distance_tier)
            flat_is_midweek[i] = config.include_midweek ? Float64(row.is_midweek) : 0.0
            flat_fallback[i]   = Int(row.distance_fallback)
        end
    end
    
    # Primary selected feature vector
    primary_vec = if config.metric == :log_dist_z
        flat_log_dist_z
    elseif config.metric == :dist_z
        flat_dist_z
    elseif config.metric == :hav_miles
        flat_hav_miles
    elseif config.metric == :road_miles
        flat_road_miles
    elseif config.metric == :drive_minutes
        flat_drive_mins
    else
        flat_log_dist_z
    end
    
    F_data[:flat_distance]         = primary_vec
    F_data[:flat_distance_z]       = flat_dist_z
    F_data[:flat_log_distance_z]   = flat_log_dist_z
    F_data[:flat_distance_miles]   = flat_hav_miles
    F_data[:flat_road_miles]       = flat_road_miles
    F_data[:flat_drive_minutes]    = flat_drive_mins
    F_data[:flat_distance_tier]    = flat_tiers
    F_data[:flat_is_midweek]       = flat_is_midweek
    F_data[:flat_distance_fallback] = flat_fallback
    # Only selected fold IDs are retained. Storing the full `dist_df` leaks future
    # fixture membership and makes perturbation comparisons fail structurally.
    F_data[:distance_by_match_id] = Dict(
        Int(mid) => Float64(primary_vec[i]) for (i, mid) in enumerate(ordered_ids))
end

function Features.add_feature!(
    F_data::Dict, 
    config::ScottishDistanceFeature, 
    ordered_ids, 
    team_map::Dict, 
    ds::Data.DataStore
)
    Features.add_feature!(F_data, config, collect(Int.(ordered_ids)), team_map, ds)
end
