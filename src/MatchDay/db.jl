# src/MatchDay/db.jl
#
# Direct SQL against betdb. A deliberate decision: the datastore package is an actively edited
# proof of concept, and interposing a stable view would cost a migration on every change. The
# coupling is accepted because the 1-minute intervals mean timing is not critical, so a schema
# break shows up as a loud query error rather than as bad prices.
#
# Every query lives here so that a schema change has exactly one blast radius.

export betfair_to_key, key_to_betfair

"""
    _conn() -> LibPQ.Connection

Opens a connection from `BF_DB_URL`. Callers are responsible for closing; `_query` does.
"""
function _conn()
    url = get(ENV, "BF_DB_URL") do
        error("BF_DB_URL is not set. Export it before using MatchDay, e.g.\n" *
              "  export BF_DB_URL=\"postgresql://user:pass@host:5433/betdb\"")
    end
    return Data.connect_to_db(Data.DBConfig(url))
end

"Run a parameterised read and return a DataFrame. Opens and closes its own connection."
function _query(sql::AbstractString, params::Tuple = ())
    c = _conn()
    try
        return DataFrame(LibPQ.execute(c, sql, params))
    finally
        close(c)
    end
end

# ===================================================================
# Market-name mapping
# ===================================================================
#
# Moved from the prototype's live_betting.jl. The exchange names markets and runners as strings;
# the model names them (group, line, selection). This is the only place that translation lives.

"""
    key_to_betfair(group, line) -> String | nothing

`("OverUnder", 2.5) -> "OVER_UNDER_25"`, `("1X2", 0.0) -> "MATCH_ODDS"`.
Returns `nothing` for market groups the exchange does not carry under these names.
"""
function key_to_betfair(group::AbstractString, line::Real)
    group == "1X2"  && return "MATCH_ODDS"
    group == "BTTS" && return "BOTH_TEAMS_TO_SCORE"
    if group == "OverUnder"
        return "OVER_UNDER_" * replace(@sprintf("%.1f", line), "." => "")
    end
    return nothing
end

"""
    betfair_to_key(market_type, runner_name) -> SelectionKey | nothing

Inverse mapping, from a `market_metadata.market_type` plus an `order_book_1m.symbol`.

Runner names are matched on structure, not on team identity: `"Over 2.5 Goals"`, `"Under 2.5
Goals"`, `"Yes"`, `"No"`, `"Draw"`. The home/away runners of MATCH_ODDS carry team names, which
cannot be resolved without the fixture, so 1X2 home/away are handled by
[`betfair_to_key_1x2`](@ref) instead.
"""
function betfair_to_key(market_type::AbstractString, runner::AbstractString)
    if startswith(market_type, "OVER_UNDER_")
        digits = market_type[12:end]
        length(digits) < 2 && return nothing
        line = tryparse(Float64, digits[1:end-1] * "." * digits[end:end])
        line === nothing && return nothing
        sel = startswith(runner, "Over")  ? Symbol("over_",  replace(@sprintf("%.1f", line), "." => "")) :
              startswith(runner, "Under") ? Symbol("under_", replace(@sprintf("%.1f", line), "." => "")) :
              nothing
        sel === nothing && return nothing
        return (group = "OverUnder", line = line, selection = sel)
    elseif market_type == "BOTH_TEAMS_TO_SCORE"
        sel = runner == "Yes" ? :btts_yes : runner == "No" ? :btts_no : nothing
        sel === nothing && return nothing
        return (group = "BTTS", line = 0.0, selection = sel)
    end
    return nothing
end

"""
    betfair_to_key_1x2(runner, home, away) -> SelectionKey | nothing

MATCH_ODDS runners are team names, so resolving them needs the fixture. Matching is on
normalised names because the exchange and SofaScore spell teams differently -- the same reason
`betfair.match_meta` exists.
"""
function betfair_to_key_1x2(runner::AbstractString, home::AbstractString, away::AbstractString)
    r = _norm(runner)
    r == "draw" && return (group = "1X2", line = 0.0, selection = :draw)
    r == _norm(home) && return (group = "1X2", line = 0.0, selection = :home)
    r == _norm(away) && return (group = "1X2", line = 0.0, selection = :away)
    return nothing
end

_norm(s::AbstractString) = lowercase(replace(strip(s), r"[^A-Za-z0-9]" => ""))

# ===================================================================
# Queries
# ===================================================================

"""
Upcoming fixtures for a set of tournaments, within a kick-off horizon.

`start_timestamp` is an epoch integer, and the window is expressed in epoch seconds rather than
against `CURRENT_DATE`: a 19:45 kick-off and a UTC midnight boundary put a calendar-date filter
on the wrong side of the fixture.
"""
const FIXTURES_SQL = """
SELECT match_id, home_team, away_team, start_timestamp, tournament_id
FROM sofascore.events
WHERE status_type = 'notstarted'
  AND tournament_id = ANY(\$1)
  AND start_timestamp >= \$2
  AND start_timestamp <  \$3
ORDER BY start_timestamp;
"""

"""
The identity crosswalk plus every live market for the event.

`betfair.match_meta` is populated by a separate resolution job. When it runs it resolves 100%;
it produces no rows at all for fixtures it has not seen, which is why an absent row is reported
as `:absent_from_crosswalk` rather than as a matching failure.
"""
const IDENTITY_SQL = """
SELECT mm.betfair_event_id, mm.is_verified, md.market_id, md.market_type
FROM betfair.match_meta mm
LEFT JOIN betfair_live.market_metadata md ON md.event_id = mm.betfair_event_id
WHERE mm.match_id = \$1 AND mm.match_id IS NOT NULL;
"""

"""
Most recent order-book snapshot at or before `as_of`, per (market, runner).

`DISTINCT ON` with a descending `ts` is the point-in-time read that makes replay honest: it can
never see a tick from after the instant being replayed.
"""
const ORDER_BOOK_SQL = """
SELECT DISTINCT ON (o.market_id, o.symbol)
       o.market_id, o.symbol, o.ts, o.bid_prices, o.bid_volumes,
       o.ask_prices, o.ask_volumes, o.market_matched, md.market_type
FROM betfair_live.order_book_1m o
JOIN betfair_live.market_metadata md USING (market_id)
WHERE o.market_id = ANY(\$1) AND o.ts <= \$2 AND o.ts >= \$3
ORDER BY o.market_id, o.symbol, o.ts DESC;
"""

"""
Provisional / announced XI.

`scraped_at <= as_of` is not optional: without it a replay of a T-2h decision would see the
lineup that was scraped at T-30min.
"""
const LINEUP_SQL = """
SELECT player_id, player_name, position, substitute, is_home_team, confirmed, scraped_at
FROM sofascore.lineup_provisional
WHERE match_id = \$1 AND scraped_at <= \$2;
"""
