# src/MatchDay/ledger/fills.jl
#
# What the book would have done with an order, without placing one.
#
# Paper trading is only worth anything if this is pessimistic in the right places, so the three
# models are explicit and recorded per fill rather than being a hidden constant. The default is
# the pessimistic one, and the optimistic one is named `Optimistic` so nobody can select it
# without reading the word.

export AbstractFillModel, TouchOnly, LadderSweep, Optimistic, simulate_fill, fill_model_name

"""
    AbstractFillModel

Contract: `simulate_fill(model, levels, side, venue_stake, leverage, at) -> Vector{Fill}`.

Returns an EMPTY vector for no fill, never a zero-size `Fill`. A zero fill is not an event and
recording one would make `count(fills)` mean two different things.
"""
abstract type AbstractFillModel end

"""
    TouchOnly()

Fill at the best price only, up to the size resting there; the remainder expires.

**The default, and the honest one.** `LadderSweep` assumes we cross three price levels
instantly, which is what a market order does and not what this system does -- it rests at the
touch and waits. On the Scottish 26/27 book the difference between the two is precisely the
£25-to-£100 column of the capacity table, which is the entire capacity question, so it must be a
decision rather than a default.
"""
struct TouchOnly <: AbstractFillModel end

"""
    LadderSweep(; max_slippage = 0.02)

Sweep down the archived ladder, stopping at the first level that would take the volume-weighted
price more than `max_slippage` from the touch.

`betfair_live.order_book_1m` archives at most **three** levels (verified over 635,765 rows), so
this is a lower bound on live capacity -- conservative in the safe direction. `max_slippage`
defaults to 2%: on Scottish League One/Two the median edge that clears staking is 2-5%, so a
sweep giving up more than 2% has spent most of what it was chasing.
"""
Base.@kwdef struct LadderSweep <: AbstractFillModel
    max_slippage::Float64 = 0.02
end

"""
    Optimistic()

Fill in full at the touch, whatever the resting size.

**Research only.** A paper track built on this cannot be compared with a live one; the gap
between it and `TouchOnly` is the measurement of how much the capacity assumption is worth, and
that is the only legitimate use.
"""
struct Optimistic <: AbstractFillModel end

fill_model_name(::TouchOnly)    = :touch_only
fill_model_name(::LadderSweep)  = :ladder_sweep_v1
fill_model_name(::Optimistic)   = :optimistic

"""
    simulate_fill(model, levels, side, venue_stake, leverage, at) -> Vector{Fill}

The ladder consumed is chosen by `side`, and the choice is not symmetric: a `:back` order eats
the bid side (`levels.back`, the prices available to back) and a `:lay` order eats the ask side.
Betfair quotes both sides' sizes as backer stake, which is the denomination of `venue_stake`.

`order_id` is filled in by the caller via [`attach_order`](@ref); this function is pure in the
book and the size, so the same call replays identically.
"""
function simulate_fill(model::AbstractFillModel, levels::BookLevels, side::Symbol,
                       venue_stake::Real, leverage::Real, at::DateTime;
                       order_id::UUID = UUID(0))
    venue_stake > 0 || return Fill[]
    prices, sizes = side === :back ? (levels.back, levels.back_size) :
                    side === :lay  ? (levels.lay,  levels.lay_size)  :
                    error("side must be :back or :lay, got $side")
    isempty(prices) && return Fill[]
    return _fill(model, prices, sizes, Float64(venue_stake), Float64(leverage), at, order_id)
end

function _fill(::TouchOnly, prices, sizes, venue_stake, leverage, at, order_id)
    size_here = isempty(sizes) ? 0.0 : Float64(sizes[1])
    taken = min(venue_stake, size_here)
    taken > 1e-9 || return Fill[]
    return [Fill(order_id = order_id, filled_at = at, price = Float64(prices[1]),
                 size = taken, risk_filled = taken / leverage,
                 model = :touch_only, levels_used = 1)]
end

function _fill(m::LadderSweep, prices, sizes, venue_stake, leverage, at, order_id)
    touch = Float64(prices[1])
    out   = Fill[]
    remaining = venue_stake
    n = min(length(prices), length(sizes))
    for i in 1:n
        remaining > 1e-9 || break
        p = Float64(prices[i])
        p > 1.0 || continue
        (touch - p) / touch > m.max_slippage && break     # this level costs more than we will pay
        taken = min(remaining, Float64(sizes[i]))
        taken > 1e-9 || continue
        push!(out, Fill(order_id = order_id, filled_at = at, price = p, size = taken,
                        risk_filled = taken / leverage, model = :ladder_sweep_v1,
                        levels_used = i))
        remaining -= taken
    end
    return out
end

_fill(::Optimistic, prices, _sizes, venue_stake, leverage, at, order_id) =
    [Fill(order_id = order_id, filled_at = at, price = Float64(prices[1]),
          size = venue_stake, risk_filled = venue_stake / leverage,
          model = :optimistic, levels_used = 1)]

"Restamp a vector of fills onto an order. Returns new `Fill`s; never mutates."
attach_order(fills::Vector{Fill}, order_id::UUID) =
    [Fill(order_id = order_id, filled_at = f.filled_at, price = f.price, size = f.size,
          risk_filled = f.risk_filled, model = f.model, levels_used = f.levels_used)
     for f in fills]

"Total backer stake filled."
filled_size(fills::AbstractVector{Fill}) = isempty(fills) ? 0.0 : sum(f -> f.size, fills)

"Total LIABILITY filled -- what the account releases against, not `filled_size`."
filled_risk(fills::AbstractVector{Fill}) = isempty(fills) ? 0.0 : sum(f -> f.risk_filled, fills)

"""
    fill_vwap(fills) -> Float64

Volume-weighted price, averaged in **probability space** (`Σ size / Σ (size/price)`).

The arithmetic mean of two decimal prices is not the price at which the combined stake breaks
even, so averaging them directly overstates a book that filled deep.
"""
function fill_vwap(fills::AbstractVector{Fill})
    isempty(fills) && return NaN
    s = sum(f -> f.size, fills)
    s > 0 || return NaN
    return s / sum(f -> f.size / f.price, fills)
end
