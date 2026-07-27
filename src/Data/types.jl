# src/data/types.jl

"""
    DBConfig
Configuration struct for establishing a connection to the PostgreSQL database.
"""
struct DBConfig 
    url::String
end 

abstract type DataTournemantSegment end


abstract type FootballDataType end
struct BetfairData <: FootballDataType end

"""
    DataStore
The central data structure holding all processed DataFrames for a specific segment.
Once the fetchers complete their asynchronous tasks, they populate this struct.
"""
struct DataStore
    segment::DataTournemantSegment
    matches::DataFrame
    statistics::DataFrame
    odds::DataFrame
    lineups::DataFrame
    incidents::DataFrame
    betfair_odds::DataFrame
    bbc::DataFrame
end

"""
Backwards-compatible constructor: `bbc` defaults to an empty DataFrame.

Plenty of code (research runners in `current_development/`, tests) rebuilds a store positionally
to swap one domain out — e.g. `DataStore(ds.segment, ds.matches, ds.statistics, odds_bf, ds.lineups,
ds.incidents, ds.betfair_odds)`. Those call sites keep working and simply carry no BBC data, which
every downstream extractor must already tolerate (most segments have no BBC coverage at all).
"""
DataStore(segment::DataTournemantSegment, matches::DataFrame, statistics::DataFrame,
          odds::DataFrame, lineups::DataFrame, incidents::DataFrame, betfair_odds::DataFrame) =
    DataStore(segment, matches, statistics, odds, lineups, incidents, betfair_odds, DataFrame())
