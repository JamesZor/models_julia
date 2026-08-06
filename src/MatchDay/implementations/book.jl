# src/MatchDay/implementations/book.jl
#
# Book sources, quote rules, instrument rules, and stake rounding.

export ArchivedOrderBook, RedisLive, BestAvailable, MidPrice,
       DirectBackOnly, BestOfBackLay, NoMinimum, FloorOrDrop, FloorOrRoundUp

# ===================================================================
# Book sources
# ===================================================================

"""
    ArchivedOrderBook(; conn = nothing)

Reads `betfair_live.order_book_1m` -- the persisted Redis feed. `whatstheodds`'
`stream_worker.py` drains Redis into this table (`ON CONFLICT (market_id, symbol, ts) DO
UPDATE`, minute-downsampled), so this one adapter serves both replay and, once Redis is back,
the same data path production uses.

Serves the most recent snapshot at or before `as_of`, which is what makes replay honest.

Note `last_price_traded` is NULL in 100% of rows -- the drain writes prices and volumes only.
There is no traded price here, so this source cannot reproduce the backtest's `odds_close`
(which comes from `betfair.odds_history`). It prices off the book, which is the executable
number but a different quantity.
"""
Base.@kwdef struct ArchivedOrderBook <: AbstractBookSource
    max_age::Period = Hour(6)
end

"""
    RedisLive(; host, port)

Declared seam, not implemented. The homelab Redis is down and `Redis.jl` is not a dependency;
`ArchivedOrderBook` reads the same data out of Postgres. Selecting this errors rather than
silently returning an empty book.
"""
Base.@kwdef struct RedisLive <: AbstractBookSource
    host::String = "100.124.38.117"
    port::Int    = 6379
end

quotes(s::RedisLive, ::Resolved, ::DateTime) = error(
    "RedisLive is a declared seam and is not implemented. The live Redis feed is drained into " *
    "betfair_live.order_book_1m; use ArchivedOrderBook, which reads the same data.")

function quotes(s::ArchivedOrderBook, r::Resolved, as_of::DateTime)
    out = Dict{SelectionKey,BookLevels}()
    isempty(r.market_ids) && return out

    ids = collect(values(r.market_ids))
    rows = _query(ORDER_BOOK_SQL, (ids, as_of, as_of - s.max_age))
    isempty(rows) && return out

    for row in eachrow(rows)
        key = betfair_to_key(String(row.market_type), String(row.symbol))
        key === nothing && continue
        # Prices and volumes are integers scaled x10000 on the exchange feed.
        lv = BookLevels(_unscale(row.bid_prices), _unscale(row.bid_volumes),
                        _unscale(row.ask_prices), _unscale(row.ask_volumes),
                        ismissing(row.market_matched) ? NaN : Float64(row.market_matched) / 10_000,
                        DateTime(row.ts))
        out[key] = lv
    end
    return out
end

_unscale(x) = ismissing(x) || x === nothing ? Float64[] : Float64[v / 10_000 for v in x]

# ===================================================================
# Quote rules
# ===================================================================

"""
    BestAvailable()

Best price at the top of the book: highest bid to back, lowest ask to lay. Both are the prices
you would actually be filled at for a small stake, which is the whole point of pricing off the
book rather than off the last trade.
"""
struct BestAvailable <: AbstractQuoteRule end

quote_price(::BestAvailable, b::BookLevels, side::Symbol) =
    side === :back ? best_back(b) :
    side === :lay  ? best_lay(b)  :
    error("side must be :back or :lay, got $side")

"""
    MidPrice()

Midpoint of the spread. Not executable -- included for measuring how much the spread costs,
never for staking.
"""
struct MidPrice <: AbstractQuoteRule end

function quote_price(::MidPrice, b::BookLevels, ::Symbol)
    bb, bl = best_back(b), best_lay(b)
    (isnan(bb) || isnan(bl)) && return NaN
    return (bb + bl) / 2
end

# ===================================================================
# Instrument rules -- the finding-F morphism
# ===================================================================

"""
    DirectBackOnly()

Back every position at its own price. The conservative baseline: it is what the prototype did
and what the backtest assumed.
"""
struct DirectBackOnly <: AbstractInstrumentRule end

instrument(::DirectBackOnly, key::SelectionKey, ::Union{Nothing,SelectionKey},
           book::Dict{SelectionKey,BookLevels}, q::AbstractQuoteRule) =
    direct_back(key, book, q)

"""
    BestOfBackLay(; max_leverage = 20.0)

Take whichever instrument prices the position better: back it directly, or lay its complement.
A strict improvement -- it never chooses a worse price than `DirectBackOnly` would.

`max_leverage` bounds the backer stake needed per unit of risk, rejecting synthetics off a laid
price near 1. See `synthetic_back`.

Only applies to two-outcome groups; 1X2 has no complement and falls through to a direct back.
"""
Base.@kwdef struct BestOfBackLay <: AbstractInstrumentRule
    max_leverage::Float64 = 20.0
end

function instrument(rule::BestOfBackLay, key::SelectionKey, comp::Union{Nothing,SelectionKey},
                    book::Dict{SelectionKey,BookLevels}, q::AbstractQuoteRule)
    direct = direct_back(key, book, q)
    comp === nothing && return direct
    synth = synthetic_back(key, comp, book, q; max_leverage = rule.max_leverage)
    synth === nothing && return direct
    direct === nothing && return synth
    return synth.odds > direct.odds ? synth : direct
end

# ===================================================================
# Stake rounding -- the exchange minimum
# ===================================================================

"""
    NoMinimum()

Ignore the exchange minimum entirely. Correct for research and replay, where the question is
what the strategy is worth rather than what it is possible to place.
"""
struct NoMinimum <: AbstractStakeRounding end
round_stake(::NoMinimum, stake::Real, ::Instrument) = Float64(stake)

"""
    FloorOrDrop(minimum = 1.0)

Drop any leg whose **venue stake** would fall below the exchange minimum.

The venue stake, not the risk: a lay at 1.26 places `risk/0.26` with the backer, so it clears a
£1 minimum with only £0.26 at risk. The morphism therefore buys smaller minimum positions on
short prices, and this is where that shows up.

Dropping loses the diversification the joint solve was buying, but never over-stakes -- which
is the safer half of the trade-off when the cap is what keeps the bankroll positive.
"""
Base.@kwdef struct FloorOrDrop <: AbstractStakeRounding
    minimum::Float64 = 1.0
end

function round_stake(r::FloorOrDrop, stake::Real, inst::Instrument)
    stake <= 0 && return 0.0
    return venue_stake(inst, stake) < r.minimum ? 0.0 : Float64(stake)
end

"""
    FloorOrRoundUp(minimum = 1.0; max_inflation = 3.0)

Round a sub-minimum leg **up** to the exchange minimum rather than dropping it, but refuse when
that would inflate the intended stake by more than `max_inflation`.

Rounding up breaks the allocation: it over-stakes a leg the optimiser deliberately sized small,
and can push a slate past its exposure cap. `max_inflation` bounds how badly. Prefer
`FloorOrDrop` unless the bankroll is small enough that dropping empties the book.
"""
Base.@kwdef struct FloorOrRoundUp <: AbstractStakeRounding
    minimum::Float64       = 1.0
    max_inflation::Float64 = 3.0
end

function round_stake(r::FloorOrRoundUp, stake::Real, inst::Instrument)
    stake <= 0 && return 0.0
    v = venue_stake(inst, stake)
    v >= r.minimum && return Float64(stake)
    inflated = r.minimum / inst.leverage           # risk implied by staking the minimum
    return inflated / stake > r.max_inflation ? 0.0 : inflated
end
