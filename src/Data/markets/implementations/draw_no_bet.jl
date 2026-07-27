# src/Data/markets/implementations/draw_no_bet.jl
#
# Draw No Bet market (Betfair `DRAW_NO_BET`).
# Two selections (home / away); on a draw the stake is refunded (a push).
# The push is handled in grade_selection (returns `missing` -> zero PnL) and on
# the model side by renormalising over the non-draw outcomes.

struct MarketDrawNoBet <: AbstractMarket end

# --- Interface ---
Base.show(io::IO, ::MarketDrawNoBet) = print(io, "Market[DrawNoBet]")
market_group(::MarketDrawNoBet) = "DrawNoBet"
outcomes(::MarketDrawNoBet) = (home = :dnb_home, away = :dnb_away)

# --- Logic (Betfair-only: no SofaScore extraction) ---
_process_market_type(raw_odds::DataFrame, m::MarketDrawNoBet) =
    _build_long_rows(DataFrame(), Dict{String,Symbol}(), market_group(m), market_line(m))
