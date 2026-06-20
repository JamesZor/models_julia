# current_development/order_book/l01_order_book.jl
# Loader: fetch and process Betfair order book snapshots from order_book_1m + market_metadata

using DataFrames
using Dates
using TimeZones
using LibPQ

# ---------------------------------------------------------------------------
# PostgreSQL array parsing
# LibPQ returns array columns as raw strings like "{25600,25400,25200}"
# ---------------------------------------------------------------------------

function parse_pg_int_array(s::String)::Vector{Int}
    stripped = strip(s, ['{', '}'])
    isempty(stripped) && return Int[]
    return parse.(Int, split(stripped, ","))
end

# ---------------------------------------------------------------------------
# Market taxonomy mapping
# Adapted from src/Data/fetchers/betfair_odds.jl — kept local since those
# helpers are private to the Data module
# ---------------------------------------------------------------------------

function _map_market_info(m_type::String)
    if startswith(m_type, "OVER_UNDER_")
        line_val = parse(Float64, replace(m_type, "OVER_UNDER_" => "")) / 10.0
        return "OverUnder", line_val
    elseif m_type == "MATCH_ODDS"
        return "1X2", 0.0
    elseif m_type == "BOTH_TEAMS_TO_SCORE"
        return "BTTS", 0.0
    elseif m_type == "CORRECT_SCORE"
        return "CorrectScore", 0.0
    end
    return m_type, 0.0
end

# order_book_1m uses display strings ("0 - 1", "home") rather than the
# snake_case Symbols used in odds_history — handle both formats
function _map_ob_selection(sym::String)::Symbol
    s = lowercase(strip(sym))

    # Correct score: "0 - 1" or "0-1" → :cs_01
    if occursin(r"^\d+\s*-\s*\d+$", s)
        digits_only = replace(s, r"\s*-\s*" => "")
        return Symbol("cs_", digits_only)

    # Any other score: "any other home" → :cs_any_other_home
    elseif startswith(s, "any other")
        return Symbol("cs_", replace(s, " " => "_"))

    # Over/Under display: "over 2.5" → :over_25
    elseif startswith(s, "over ") || startswith(s, "under ")
        parts = split(s)                       # ["over", "2.5", "goals"] or ["over", "2.5"]
        direction = parts[1]                   # "over" / "under"
        line = replace(parts[2], "." => "")   # "25"  (ignore trailing words like "goals")
        return Symbol(direction, "_", line)

    # Standard selections: home, away, draw, yes, no
    else
        return Symbol(replace(s, " " => "_"))
    end
end

# ---------------------------------------------------------------------------
# Price and volume scale factors
# Prices: raw int / 10000 → decimal odds   e.g. 25600 → 2.56
# Volumes: raw int / 100  → GBP            e.g. 33900 → £339.00
# VERIFY on first run by checking 1X2 home bid_price_1 is in range [1.5, 5.0]
# ---------------------------------------------------------------------------

const _PRICE_SCALE  = 10000.0
const _VOLUME_SCALE = 100.0
const _N_LEVELS     = 3

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

function fetch_order_book_raw(
    conn::LibPQ.Connection;
    date_from::Union{DateTime, Nothing} = nothing,
    date_to::Union{DateTime, Nothing}   = nothing,
)::DataFrame

    has_from = !isnothing(date_from)
    has_to   = !isnothing(date_to)

    where_clause = if has_from && has_to
        "WHERE ob.ts BETWEEN \$1 AND \$2"
    elseif has_from
        "WHERE ob.ts >= \$1"
    elseif has_to
        "WHERE ob.ts <= \$1"
    else
        ""
    end

    query = """
        SELECT ob.market_id,
               ob.symbol,
               ob.ts AT TIME ZONE 'UTC'     AS ts,
               ob.bid_prices::text,
               ob.bid_volumes::text,
               ob.ask_prices::text,
               ob.ask_volumes::text,
               mm.event_id,
               mm.home_team,
               mm.away_team,
               mm.market_type,
               mm.open_date AT TIME ZONE 'UTC' AS open_date
        FROM order_book_1m ob
        JOIN market_metadata mm USING (market_id)
        $where_clause
        ORDER BY ob.market_id, ob.symbol, ob.ts
    """

    params = filter(!isnothing, Any[date_from, date_to])

    try
        return isempty(params) ?
            DataFrame(LibPQ.execute(conn, query)) :
            DataFrame(LibPQ.execute(conn, query, params))
    catch e
        @warn "fetch_order_book_raw failed: $e"
        return DataFrame()
    end
end

# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------

function _extract_levels(arr::Vector{Int}, scale::Float64, n::Int)::NTuple{3, Float64}
    v = ntuple(i -> i <= length(arr) ? arr[i] / scale : NaN, n)
    return v
end

function _to_datetime(x)::DateTime
    x isa ZonedDateTime && return DateTime(x, TimeZones.UTC)
    return DateTime(x)
end

function process_order_book(raw::DataFrame)::DataFrame
    isempty(raw) && return DataFrame()

    rows = NamedTuple{(
        :market_id, :event_id, :home_team, :away_team,
        :market_name, :market_line, :selection,
        :ts, :minutes_to_kickoff,
        :bid_price_1, :bid_price_2, :bid_price_3,
        :bid_vol_1,   :bid_vol_2,   :bid_vol_3,
        :ask_price_1, :ask_price_2, :ask_price_3,
        :ask_vol_1,   :ask_vol_2,   :ask_vol_3,
    ), Tuple{
        String, Int64, String, String,
        String, Float64, Symbol,
        DateTime, Float64,
        Float64, Float64, Float64,
        Float64, Float64, Float64,
        Float64, Float64, Float64,
        Float64, Float64, Float64,
    }}[]

    for row in eachrow(raw)
        any(ismissing, (row.bid_prices, row.bid_volumes, row.ask_prices, row.ask_volumes)) && continue

        bid_p = parse_pg_int_array(row.bid_prices)
        bid_v = parse_pg_int_array(row.bid_volumes)
        ask_p = parse_pg_int_array(row.ask_prices)
        ask_v = parse_pg_int_array(row.ask_volumes)

        bp = _extract_levels(bid_p, _PRICE_SCALE,  _N_LEVELS)
        bv = _extract_levels(bid_v, _VOLUME_SCALE, _N_LEVELS)
        ap = _extract_levels(ask_p, _PRICE_SCALE,  _N_LEVELS)
        av = _extract_levels(ask_v, _VOLUME_SCALE, _N_LEVELS)

        market_name, market_line = _map_market_info(string(row.market_type))
        selection = _map_ob_selection(string(row.symbol))

        ts      = _to_datetime(row.ts)
        open_dt = _to_datetime(row.open_date)
        mins_to_ko = Dates.value(ts - open_dt) / 60000.0

        push!(rows, (
            market_id          = string(row.market_id),
            event_id           = parse(Int64, string(row.event_id)),
            home_team          = string(row.home_team),
            away_team          = string(row.away_team),
            market_name        = market_name,
            market_line        = market_line,
            selection          = selection,
            ts                 = ts,
            minutes_to_kickoff = mins_to_ko,
            bid_price_1 = bp[1], bid_price_2 = bp[2], bid_price_3 = bp[3],
            bid_vol_1   = bv[1], bid_vol_2   = bv[2], bid_vol_3   = bv[3],
            ask_price_1 = ap[1], ask_price_2 = ap[2], ask_price_3 = ap[3],
            ask_vol_1   = av[1], ask_vol_2   = av[2], ask_vol_3   = av[3],
        ))
    end

    return DataFrame(rows)
end
