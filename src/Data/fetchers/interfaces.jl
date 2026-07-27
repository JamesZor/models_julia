# src/data/fetchers/interfaces.jl

# ---------------------------------------------------------
# Data Domain Types
# ---------------------------------------------------------

struct MatchesData    <: FootballDataType end 
struct StatisticsData <: FootballDataType end 
struct OddsData       <: FootballDataType end 
struct LineUpsData    <: FootballDataType end
struct IncidentsData  <: FootballDataType end
# BBC match-page shot counts (bbc.match_stats / bbc.match_meta). Only the Scottish tiers
# (tournaments 54-57) have coverage; every other segment resolves to an empty DataFrame.
struct BBCData        <: FootballDataType end

# ---------------------------------------------------------
# The Pipeline Contracts (Fallbacks)
# ---------------------------------------------------------

"""
    fetch_data(conn, t_ids, data_type)
Executes the raw SQL query. Must be implemented for every FootballDataType.
"""
function fetch_data(conn::LibPQ.Connection, t_ids::Vector{Int}, data_type::FootballDataType)
    error("fetch_data (SQL query) not implemented for $(typeof(data_type))")
end

"""
    process_data(df, data_type)
Performs Julia-side ETL (cleaning, pivoting, padding). 
Defaults to returning the raw DataFrame if no specific processing is defined.
"""
function process_data(df::DataFrame, ::FootballDataType; kwargs...)
    return df
end

"""
    validate_data(df, data_type) -> Bool
Executes QA checks. Returns true if healthy. 
Defaults to true if no specific QA rules are defined.
"""
function validate_data(df::DataFrame, ::FootballDataType)
    return true
end

# ---------------------------------------------------------
# Schema Enforcement
# ---------------------------------------------------------

"""
    apply_schema!(df, schema)
Enforces strict column types on a DataFrame to strip `Union{Missing, T}` from guaranteed columns.
"""
function apply_schema!(df::DataFrame, schema::Dict{Symbol, Type})
    for (col, T) in schema
        if string(col) in names(df)
            df[!, col] = convert.(T, df[!, col])
        end
    end
    return df
end

# ---------------------------------------------------------
# The Master Orchestrator
# ---------------------------------------------------------

"""
    load_data(conn, segment, data_type)
The one function to rule them all. Handles ID extraction and the full Fetch->Process->QA pipeline.
"""
function load_data(conn::LibPQ.Connection, segment::DataTournemantSegment, data_type::FootballDataType; kwargs...)::DataFrame
    t_ids = tournament_ids(segment)
    
    local raw_df
    try
        raw_df = fetch_data(conn, t_ids, data_type)
    catch e
        @warn "SQL Fetch failed for $(typeof(data_type)): $(e)"
        return DataFrame()
    end
    
    if nrow(raw_df) == 0; return raw_df; end
    
    local clean_df
    try
        # Pass kwargs (like config=...) down to the processor
        clean_df = process_data(raw_df, data_type; kwargs...)
    catch e
        @warn "Processing/Schema Enforcement failed for $(typeof(data_type)): $(e)"
        return DataFrame()
    end
    
    # if !validate_data(clean_df, data_type)
    #     @warn "QA Verifier failed for $(typeof(data_type))."
    # end
    
    return clean_df
end
