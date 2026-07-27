# src/data/fetchers/sql/bbc.jl
#
# BBC match-page shot counts. One row per match: shots + shots-on-target per side, plus the
# sofascore meta needed to key/QA them. Graduated from
# current_development/bbc_xg_proxy/l03_funnel_cascade.jl::fetch_funnel_counts, which side-loaded
# this table into a .jls cache; it now rides the standard Fetch -> Process -> QA contract so it is
# cached with the rest of the DataStore.
#
# Coverage note: only the Scottish tiers (tournaments 54, 55, 56, 57) have BBC pages. Every other
# segment falls through `load_data`'s empty-DataFrame path, which is the intended behaviour —
# downstream features must degrade, not error, when `ds.bbc` is empty.
#
# We do NOT rebuild shots as SoT + off-target + blocked: `shotsBlocked` is 33-65% backfilled in the
# lower tiers, so `shotsTotal` is pulled directly.

const FUNNEL_STATS = ["shotsTotal", "shotsOnTarget"]

function fetch_data(conn::LibPQ.Connection, t_ids::Vector{Int}, ::BBCData)
    # FUNNEL_STATS is a compile-time constant, so interpolating it is safe; the tournament ids are
    # bound as a parameter like every other fetcher.
    stats_in = join(("'" * s * "'" for s in FUNNEL_STATS), ", ")
    query = """
    WITH st AS (
        SELECT DISTINCT ON (match_id, stat_type)
               match_id, stat_type, home_value, away_value
        FROM bbc.match_stats
        WHERE stat_type IN ($stats_in)
        ORDER BY match_id, stat_type, filled   -- genuine (filled = false) wins the de-dup
    ),
    piv AS (
        SELECT match_id,
               max(home_value) FILTER (WHERE stat_type = 'shotsTotal')    AS shots_h,
               max(away_value) FILTER (WHERE stat_type = 'shotsTotal')    AS shots_a,
               max(home_value) FILTER (WHERE stat_type = 'shotsOnTarget') AS sot_h,
               max(away_value) FILTER (WHERE stat_type = 'shotsOnTarget') AS sot_a
        FROM st
        GROUP BY match_id
    )
    SELECT m.match_id, m.tournament_id, s.name AS season, m.start_timestamp,
           m.home_score, m.away_score,
           piv.shots_h, piv.shots_a, piv.sot_h, piv.sot_a
    FROM bbc.match_meta b
    JOIN sofascore.matches m ON m.match_id = b.match_id
    JOIN sofascore.seasons s ON s.season_id = m.season_id AND s.tournament_id = m.tournament_id
    JOIN piv ON piv.match_id = b.match_id
    WHERE m.tournament_id = ANY(\$1) AND b.scores_match
    ORDER BY m.start_timestamp
    """
    try
        return DataFrame(LibPQ.execute(conn, query, [t_ids]))
    catch e
        @warn "Failed to fetch BBCData: $(e)"
        return DataFrame()
    end
end

# Counts stay `Union{Missing, Float64}` — whether a given match is usable is decided by the
# feature extractor (which needs a per-side mask), not here.
function process_data(df::DataFrame, ::BBCData)
    for c in ("shots_h", "shots_a", "sot_h", "sot_a")
        c in names(df) || continue
        df[!, c] = passmissing(Float64).(df[!, c])
    end
    apply_schema!(df, BBC_COLS_TYPES)
    return df
end

# QA (documentation-grade: `load_data` currently has the validate step commented out).
function validate_data(df::DataFrame, ::BBCData)
    nrow(df) == 0 && return true

    n_shots = count(!ismissing, df.shots_h) + count(!ismissing, df.shots_a)
    coverage = n_shots / (2 * nrow(df))
    if coverage < 0.95
        @warn "BBCData: shot coverage $(round(100 * coverage, digits=1))% < 95%"
        return false
    end

    bad = count(eachrow(df)) do r
        (!ismissing(r.sot_h) && !ismissing(r.shots_h) && r.sot_h > r.shots_h) ||
        (!ismissing(r.sot_a) && !ismissing(r.shots_a) && r.sot_a > r.shots_a)
    end
    if bad > 0
        @warn "BBCData: $bad match sides have sot > shots"
        return false
    end

    return true
end
