# src/Data/markets/implementations/correct_score.jl
#
# Correct Score market (Betfair `CORRECT_SCORE`).
# Betfair enumerates the 4x4 grid 0-0 .. 3-3 explicitly, then buckets everything
# outside that grid into three "any other" selections keyed by the result:
#   any_other_home / any_other_draw / any_other_away
# These map directly onto cells (and cell-aggregates) of the ScoreMatrix, so no
# new physics is required - see src/predictions/market_inference/correct_score.jl.

struct MarketCorrectScore <: AbstractMarket end

# The explicit scoreline grid Betfair prices (home, away), 0..3 each.
const CS_GRID = [(h, a) for h in 0:3 for a in 0:3]

# Selection symbol for an exact scoreline cell, e.g. (2,1) -> :cs_21
_cs_cell_symbol(h::Int, a::Int) = Symbol("cs_", h, a)

# --- Interface ---
Base.show(io::IO, ::MarketCorrectScore) = print(io, "Market[CorrectScore]")
market_group(::MarketCorrectScore) = "CorrectScore"

function outcomes(::MarketCorrectScore)
    cells = (_cs_cell_symbol(h, a) for (h, a) in CS_GRID)
    others = (:cs_any_other_home, :cs_any_other_draw, :cs_any_other_away)
    return Tuple(Iterators.flatten((cells, others)))
end

# --- Logic (Betfair-only: no SofaScore extraction) ---
# The concrete odds arrive via the Betfair pipeline (unpack_betfair_odds +
# summarize_betfair_market), so the SofaScore extractor is a no-op empty frame.
_process_market_type(raw_odds::DataFrame, m::MarketCorrectScore) =
    _build_long_rows(DataFrame(), Dict{String,Symbol}(), market_group(m), market_line(m))
