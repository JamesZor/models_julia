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
    # Raw BBC shot-bearing commentary events (one row per attempt), as opposed to `bbc`, which is
    # per-MATCH totals. Feeds the plus-minus (RAPM) rating family — see
    # src/features/plus_minus/. Scottish tiers only, live-text era only (23/24+); every other
    # segment gets an empty DataFrame and the PM extractors emit zero ratings.
    bbc_events::DataFrame
end

"""
Backwards-compatible constructors: `bbc` and `bbc_events` default to empty DataFrames.

Plenty of code (research runners in `current_development/`, tests) rebuilds a store positionally
to swap one domain out — e.g. `DataStore(ds.segment, ds.matches, ds.statistics, odds_bf, ds.lineups,
ds.incidents, ds.betfair_odds)`. Those call sites keep working and simply carry no BBC data, which
every downstream extractor must already tolerate (most segments have no BBC coverage at all).

⚠ A 7-arg rebuild silently DROPS both BBC domains, so the funnel engine degrades to goals-only and
the plus-minus features emit zeros. Pass all 9 fields through when the store is only being rebuilt
to swap `odds`.
"""
DataStore(segment::DataTournemantSegment, matches::DataFrame, statistics::DataFrame,
          odds::DataFrame, lineups::DataFrame, incidents::DataFrame, betfair_odds::DataFrame) =
    DataStore(segment, matches, statistics, odds, lineups, incidents, betfair_odds,
              DataFrame(), DataFrame())

DataStore(segment::DataTournemantSegment, matches::DataFrame, statistics::DataFrame,
          odds::DataFrame, lineups::DataFrame, incidents::DataFrame, betfair_odds::DataFrame,
          bbc::DataFrame) =
    DataStore(segment, matches, statistics, odds, lineups, incidents, betfair_odds,
              bbc, DataFrame())
