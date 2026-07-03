# src/Data/markets/implementations/asian_handicap.jl
#
# Asian Handicap market (Betfair `ASIAN_HANDICAP`).
#
# Sign convention (verified empirically against Betfair traded prices):
#   `ah_home_X`  = home with handicap +X  (X goals ADDED to the home side)
#   `ah_home_mX` = home with handicap -X
#   `ah_away_X`  = away with handicap +X  (== home handicap -X)
# A back on `side` with side-perspective signed line L wins iff
#   (side_goals - opp_goals) + L > 0 ;  a value of 0 is a push (stake refunded).
#
# v1 supports WHOLE and HALF lines only (win / loss / push). Quarter lines
# (x.25 / x.75) split the stake across two adjacent lines and need fractional
# settlement in the backtest PnL - they are ingested as ticks but graded
# `missing` (excluded from backtests) until that is added. See grade_selection.

struct MarketAsianHandicap <: AbstractMarket
    side::Symbol      # :home or :away
    line::Float64     # side-perspective signed handicap (e.g. -0.5, +1.0)
end

# --- Line <-> Betfair-key encoding ---------------------------------------

# Encode the magnitude of a line into the Betfair body form: 0.5 -> "0_5", 1.0 -> "1".
function _ah_encode_line(L::Float64)
    mag = abs(L)
    body = isinteger(mag) ? string(Int(mag)) : replace(string(mag), "." => "_")
    return (L < 0 ? "m" : "") * body
end

# Full selection symbol, e.g. (:home, -0.5) -> :ah_home_m0_5
ah_selection_symbol(side::Symbol, L::Float64) = Symbol("ah_", side, "_", _ah_encode_line(L))

# Parse a selection symbol back to (side, side-perspective signed line).
function parse_ah_selection(sel::Symbol)
    s = string(sel)
    side = startswith(s, "ah_away_") ? :away : :home
    body = replace(s, "ah_home_" => "", "ah_away_" => "")
    neg = startswith(body, "m")
    body = replace(body, r"^m" => "")
    L = parse(Float64, replace(body, "_" => "."))
    return (side, neg ? -L : L)
end

# Canonical HOME-perspective line for a selection. Used as `market_line` so the
# two sides of the same handicap (home +L and away -L) share a de-vig group.
function ah_home_line(sel::Symbol)
    side, L = parse_ah_selection(sel)
    return side === :home ? L : -L
end

# Quarter line => stake splits across two adjacent lines (unsupported in v1).
ah_is_quarter(L::Float64) = !isapprox(mod(abs(L), 0.5), 0.0; atol = 1e-9)

# --- Interface ---
Base.show(io::IO, m::MarketAsianHandicap) = print(io, "Market[AH $(m.side) $(m.line)]")
market_group(::MarketAsianHandicap) = "AsianHandicap"
# Canonical home-perspective line so paired selections group together for de-vig.
market_line(m::MarketAsianHandicap) = m.side === :home ? m.line : -m.line
outcomes(m::MarketAsianHandicap) = (bet = ah_selection_symbol(m.side, m.line),)

# --- Logic (Betfair-only: no SofaScore extraction) ---
_process_market_type(raw_odds::DataFrame, m::MarketAsianHandicap) =
    _build_long_rows(DataFrame(), Dict{String,Symbol}(), market_group(m), market_line(m))

"""
    standard_asian_handicaps(; lines=[0.5, 1.0, 1.5, 2.0]) -> Vector{MarketAsianHandicap}

Convenience constructor: both sides of the given (whole/half) handicap magnitudes,
signed both ways, for use in a `MarketConfig`.
"""
function standard_asian_handicaps(; lines = [0.5, 1.0, 1.5, 2.0])
    mkts = MarketAsianHandicap[]
    for l in lines, side in (:home, :away), s in (1.0, -1.0)
        push!(mkts, MarketAsianHandicap(side, s * l))
    end
    return mkts
end
