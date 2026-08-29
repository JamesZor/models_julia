# Travel distance and geographic-fatigue features.

const EARTH_RADIUS_MILES = 3958.7613
const EARTH_RADIUS_KM = 6371.0088
const DISTANCE_FALLBACK_MILES = 45.0
const DISTANCE_METRICS = (:log_dist_z, :dist_z, :hav_miles, :road_miles, :drive_minutes)

"""
    haversine_distance(lat1, lon1, lat2, lon2; unit=:miles)

Great-circle distance between two WGS84 coordinates.
"""
function haversine_distance(lat1::Real, lon1::Real, lat2::Real, lon2::Real; unit::Symbol=:miles)
    unit in (:miles, :km) || throw(ArgumentError("unit must be :miles or :km, got $unit"))
    radius = unit === :miles ? EARTH_RADIUS_MILES : EARTH_RADIUS_KM

    φ1 = deg2rad(Float64(lat1))
    φ2 = deg2rad(Float64(lat2))
    Δφ = deg2rad(Float64(lat2) - Float64(lat1))
    Δλ = deg2rad(Float64(lon2) - Float64(lon1))
    a = sin(Δφ / 2)^2 + cos(φ1) * cos(φ2) * sin(Δλ / 2)^2
    a = clamp(a, 0.0, 1.0)
    return radius * 2 * atan(sqrt(a), sqrt(1 - a))
end

"Estimate road mileage and driving time from straight-line mileage."
function estimate_scottish_road_metrics(hav_miles::Real)
    distance = Float64(hav_miles)
    if distance < 25.0
        detour, speed = 1.20, 30.0
    elseif distance <= 80.0
        detour, speed = 1.25, 42.0
    else
        detour, speed = 1.30, 48.0
    end
    road_miles = distance * detour
    return (road_miles=road_miles, drive_minutes=road_miles / speed * 60.0)
end

"Classify mileage as derby (1), moderate (2), long haul (3), or extreme (4)."
function distance_tier_category(miles::Real)
    distance = Float64(miles)
    distance < 25.0 && return 1
    distance < 75.0 && return 2
    distance <= 140.0 && return 3
    return 4
end

"Load and validate the stadium coordinate catalog."
function load_stadium_catalog(csv_path::AbstractString)
    isfile(csv_path) || error("Stadium geocodes file not found at: $csv_path")
    catalog = CSV.read(csv_path, DataFrame)
    required = (:team_slug, :latitude, :longitude)
    all(column -> column in propertynames(catalog), required) ||
        error("Stadium catalog must contain columns $(collect(required))")

    catalog.team_slug = String.(catalog.team_slug)
    catalog.latitude = Float64.(catalog.latitude)
    catalog.longitude = Float64.(catalog.longitude)
    allunique(catalog.team_slug) || error("Stadium catalog contains duplicate team_slug values")
    all(isfinite, catalog.latitude) && all(isfinite, catalog.longitude) ||
        error("Stadium catalog coordinates must be finite")
    all(latitude -> -90.0 <= latitude <= 90.0, catalog.latitude) ||
        error("Stadium latitude outside [-90, 90]")
    all(longitude -> -180.0 <= longitude <= 180.0, catalog.longitude) ||
        error("Stadium longitude outside [-180, 180]")
    return catalog
end

"Compute catalog-fixed raw and log distance standardization moments."
function catalog_distance_standardization(geocodes_df::AbstractDataFrame)
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
        raw_center=mean(distances),
        raw_scale=raw_scale > 0.0 ? raw_scale : 1.0,
        log_center=mean(log_distances),
        log_scale=log_scale > 0.0 ? log_scale : 1.0,
    )
end

"Build distance metrics for match rows, using a deterministic fallback for unknown grounds."
function build_match_distance_table(
    matches_df::AbstractDataFrame;
    geocodes_df::AbstractDataFrame,
    standardization=catalog_distance_standardization(geocodes_df),
)
    required = (:match_id, :home_team, :away_team)
    all(column -> column in propertynames(matches_df), required) ||
        throw(ArgumentError("matches must contain columns $(collect(required))"))

    coordinates = Dict(
        String(row.team_slug) => (Float64(row.latitude), Float64(row.longitude))
        for row in eachrow(geocodes_df)
    )
    n = nrow(matches_df)
    hav_miles = zeros(n)
    road_miles = zeros(n)
    drive_minutes = zeros(n)
    tiers = zeros(Int, n)
    is_midweek = zeros(n)
    fallback = zeros(Int, n)

    has_match_date = :match_date in propertynames(matches_df)
    has_timestamp = :start_timestamp in propertynames(matches_df)
    for i in 1:n
        home = get(coordinates, String(matches_df.home_team[i]), nothing)
        away = get(coordinates, String(matches_df.away_team[i]), nothing)
        if home === nothing || away === nothing
            distance = DISTANCE_FALLBACK_MILES
            fallback[i] = 1
        else
            distance = haversine_distance(home[1], home[2], away[1], away[2])
        end
        road = estimate_scottish_road_metrics(distance)
        hav_miles[i] = distance
        road_miles[i] = road.road_miles
        drive_minutes[i] = road.drive_minutes
        tiers[i] = distance_tier_category(distance)

        # Prefer the processed match date, but permit raw/synthetic rows to fall back
        # to their timestamp when the date column exists but this particular value is missing.
        date_value = has_match_date && !ismissing(matches_df.match_date[i]) ?
                     matches_df.match_date[i] :
                     has_timestamp && !ismissing(matches_df.start_timestamp[i]) ?
                     Date(matches_df.start_timestamp[i]) : nothing
        if date_value !== nothing
            is_midweek[i] = dayofweek(date_value) in (2, 3, 4) ? 1.0 : 0.0
        end
    end

    log_miles = log1p.(hav_miles)
    return DataFrame(
        match_id=Int32.(matches_df.match_id),
        home_team=matches_df.home_team,
        away_team=matches_df.away_team,
        hav_miles=hav_miles,
        hav_km=hav_miles .* 1.60934,
        road_miles=road_miles,
        drive_minutes=drive_minutes,
        log_miles=log_miles,
        dist_z=(hav_miles .- standardization.raw_center) ./ standardization.raw_scale,
        log_dist_z=(log_miles .- standardization.log_center) ./ standardization.log_scale,
        distance_tier=tiers,
        is_midweek=is_midweek,
        distance_fallback=fallback,
    )
end

"Add distance vectors in requested match order."
function add_feature!(
    F_data::Dict,
    config::DistanceFeature,
    ordered_ids,
    team_map::Dict,
    ds::Data.DataStore,
)
    config.metric in DISTANCE_METRICS || throw(ArgumentError(
        "unsupported distance metric $(config.metric); expected one of $(DISTANCE_METRICS)"))
    catalog = load_stadium_catalog(config.geocodes_csv)
    standardization = catalog_distance_standardization(catalog)

    wanted_ids = Set(Int.(ordered_ids))
    selected_matches = subset(ds.matches, :match_id => ByRow(id -> Int(id) in wanted_ids))
    distances = build_match_distance_table(
        selected_matches; geocodes_df=catalog, standardization=standardization)
    rows = Dict(Int(row.match_id) => row for row in eachrow(distances))

    # A missing DataStore row should not normally occur, but deterministic values keep this
    # extractor total when called directly outside create_features.
    fallback_road = estimate_scottish_road_metrics(DISTANCE_FALLBACK_MILES)
    fallback_z = (DISTANCE_FALLBACK_MILES - standardization.raw_center) / standardization.raw_scale
    fallback_log_z = (log1p(DISTANCE_FALLBACK_MILES) - standardization.log_center) /
                     standardization.log_scale

    n = length(ordered_ids)
    flat_dist_z = zeros(n)
    flat_log_dist_z = zeros(n)
    flat_miles = zeros(n)
    flat_road_miles = zeros(n)
    flat_drive_minutes = zeros(n)
    flat_tier = zeros(Int, n)
    flat_midweek = zeros(n)
    flat_fallback = zeros(Int, n)
    for (i, match_id) in enumerate(ordered_ids)
        row = get(rows, Int(match_id), nothing)
        if row === nothing
            flat_dist_z[i] = fallback_z
            flat_log_dist_z[i] = fallback_log_z
            flat_miles[i] = DISTANCE_FALLBACK_MILES
            flat_road_miles[i] = fallback_road.road_miles
            flat_drive_minutes[i] = fallback_road.drive_minutes
            flat_tier[i] = 2
            flat_fallback[i] = 1
        else
            flat_dist_z[i] = row.dist_z
            flat_log_dist_z[i] = row.log_dist_z
            flat_miles[i] = row.hav_miles
            flat_road_miles[i] = row.road_miles
            flat_drive_minutes[i] = row.drive_minutes
            flat_tier[i] = row.distance_tier
            flat_midweek[i] = config.include_midweek ? row.is_midweek : 0.0
            flat_fallback[i] = row.distance_fallback
        end
    end

    primary = config.metric === :log_dist_z ? flat_log_dist_z :
              config.metric === :dist_z ? flat_dist_z :
              config.metric === :hav_miles ? flat_miles :
              config.metric === :road_miles ? flat_road_miles : flat_drive_minutes
    F_data[:flat_distance] = primary
    F_data[:flat_distance_z] = flat_dist_z
    F_data[:flat_log_distance_z] = flat_log_dist_z
    F_data[:flat_distance_miles] = flat_miles
    F_data[:flat_road_miles] = flat_road_miles
    F_data[:flat_drive_minutes] = flat_drive_minutes
    F_data[:flat_distance_tier] = flat_tier
    F_data[:flat_is_midweek] = flat_midweek
    F_data[:flat_distance_fallback] = flat_fallback
    F_data[:distance_by_match_id] = Dict{Int32, Float64}(
        Int32(match_id) => primary[i] for (i, match_id) in enumerate(ordered_ids))
    return nothing
end
