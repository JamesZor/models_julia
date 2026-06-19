# src/data/fetchers/datastore.jl

"""
    connect_to_db(db_config::DBConfig)::LibPQ.Connection
Establish a connection to the PostgreSQL database.
"""
function connect_to_db(db_config::DBConfig)::LibPQ.Connection
    return LibPQ.Connection(db_config.url)
end

"""
    get_datastore(conn, segment) -> DataStore
Executes all data pipelines concurrently and aggregates them into the DataStore struct.
"""
function get_datastore(conn::LibPQ.Connection, segment::DataTournemantSegment; custom_config = Markets.DEFAULT_MARKET_CONFIG)::DataStore
    local matches, statistics, incidents, lineups, odds, betfair_odds

    @info "Building DataStore for $(typeof(segment))..."

    # Concurrently fire the load_data pipeline for all domains
    @sync begin
        @async matches      = load_data(conn, segment, MatchesData())
        @async statistics   = load_data(conn, segment, StatisticsData())
        @async lineups      = load_data(conn, segment, LineUpsData())
        @async incidents    = load_data(conn, segment, IncidentsData())
        @async odds         = load_data(conn, segment, OddsData(); config = custom_config)
        @async betfair_odds = load_data(conn, segment, BetfairData())
    end

    return DataStore(segment, matches, statistics, odds, lineups, incidents, betfair_odds)
end


"""
    load_datastore_sql(segment) -> DataStore
Main entry point for external use. Manages the DB connection lifecycle and returns the data.
"""
function load_datastore_sql(segment::DataTournemantSegment)::DataStore
    # The connection string MUST come from the environment — no credentials are
    # committed here. Point BF_DB_URL at the new schema-split instance
    # (`:5433/betdb`); the old `:5432/sofascrape_db` is the untouched rollback.
    # The password was rotated in the Phase 2 credential rotation and lives only
    # in the environment.
    db_url = get(ENV, "BF_DB_URL") do
        error("BF_DB_URL is not set. Export it before loading data, e.g.\n" *
              "  export BF_DB_URL=\"postgresql://admin:<password>@100.124.38.117:5433/betdb\"")
    end
    db_config = DBConfig(db_url)
    
    db_conn = connect_to_db(db_config)
    
    try
        data_store = get_datastore(db_conn, segment)
        return data_store
    finally
        # Always close the connection, even if an error occurs during fetching
        close(db_conn) 
    end
end

using Serialization

"""
    load_datastore_cached(segment; force=false, max_age_hours=24) -> DataStore
Loads the DataStore from a local disk cache to dramatically speed up data loading.
- If `force=true`, ignores the cache and fetches fresh from SQL.
- If the cache file is older than `max_age_hours`, automatically fetches fresh from SQL.
"""
function load_datastore_cached(segment::DataTournemantSegment; force::Bool=false, max_age_hours::Int=24)::DataStore
    # Create the cache directory in the project root
    cache_dir = joinpath(pkgdir(@__MODULE__), ".cache")
    if !isdir(cache_dir)
        mkdir(cache_dir)
    end

    # Define the cache file path based on the segment type
    seg_name = string(typeof(segment))
    seg_name_clean = last(split(seg_name, ".")) 
    cache_file = joinpath(cache_dir, "datastore_$(seg_name_clean).jls")

    # Check if cache exists and is valid
    if !force && isfile(cache_file)
        # Check file age
        mtime_unix = mtime(cache_file)
        file_age_hours = (time() - mtime_unix) / 3600.0

        if file_age_hours <= max_age_hours
            @info "Loading DataStore for $(seg_name_clean) from local cache (Age: $(round(file_age_hours, digits=1)) hours)..."
            return deserialize(cache_file)
        else
            @info "Cache for $(seg_name_clean) is expired ($(round(file_age_hours, digits=1)) hours old). Fetching fresh data..."
        end
    elseif force
        @info "Force refresh triggered. Fetching fresh data..."
    end

    # Fetch fresh data from SQL
    ds = load_datastore_sql(segment)

    # Save to cache for next time
    @info "Saving $(seg_name_clean) DataStore to local cache..."
    serialize(cache_file, ds)

    return ds
end
