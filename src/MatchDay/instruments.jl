# src/MatchDay/instruments.jl
#
# THE MORPHISM. A lay is a back on the complement, once the position is measured in units of
# RISK rather than units of stake.
#
#   lay `Under` at d, backer stake b   ->   risk = b(d-1),  win = b
#   set risk = s, so b = s/(d-1)       ->   win/risk = 1/(d-1)
#   a back at D has win/risk = D-1     ->   D = 1 + 1/(d-1) = d/(d-1)
#
# Everything downstream of this file -- the payoff matrix, KellyLogUtility, BakerMcHale,
# SlateDrawdown, FixedCap -- is therefore denominated in risk and needs NO knowledge of lays.
# FixedCap sums liability by construction. Only the order ticket differs, and that is
# `venue_stake`.
#
# Measured value of taking the better of the two instruments, E[max(0, gain)], on
# betfair_live.order_book_1m (43,796 uncrossed two-sided snapshots):
#
#   competition             O/U1.5  O/U2.5  O/U3.5 | back overround @3.5
#   Scottish League Two      0.13%   1.09%   6.43% |  7.94%
#   Scottish League One      0.20%   0.97%   3.48% |  6.63%
#   Irish Premier Division   0.28%   0.37%   1.13% |  1.39%
#
# The gain tracks book width almost monotonically: it is worth most where the book is worst.
# The median gain is ~0 -- the book is arbitrage-free, so usually the two instruments agree and
# this is a free option rather than an edge.

export lay_to_back, back_to_lay, venue_stake, complement_of

"""
    lay_to_back(d) -> Float64

Effective decimal odds of laying at `d`, expressed as a back price. `d/(d-1)`.
Diverges as `d -> 1`: laying at 1.02 is a back at 51, and needs 50x leverage to fund.
"""
@inline lay_to_back(d::Real) = d <= 1 ? Inf : d / (d - 1)

"""
    back_to_lay(D) -> Float64

Inverse of [`lay_to_back`](@ref). Self-inverse: `lay_to_back(back_to_lay(D)) == D`.
"""
@inline back_to_lay(D::Real) = D <= 1 ? Inf : D / (D - 1)

"""
    complement_of(key, keys) -> SelectionKey | nothing

The opposing selection in a two-outcome market group, or `nothing` when the group does not have
exactly two outcomes. 1X2 has three, so it has no complement and can only be backed directly.
"""
function complement_of(key::SelectionKey, all_keys)
    same = [k for k in all_keys if k.group == key.group && k.line == key.line]
    length(same) == 2 || return nothing
    idx = findfirst(!=(key), same)
    return idx === nothing ? nothing : same[idx]
end

"""
    direct_back(key, book, qrule) -> Instrument | nothing

Back `key` at its own best available price. Leverage 1.0.
"""
function direct_back(key::SelectionKey, book::Dict{SelectionKey,BookLevels},
                     qrule::AbstractQuoteRule)
    haskey(book, key) || return nothing
    d = quote_price(qrule, book[key], :back)
    (isnan(d) || d <= 1) && return nothing
    return Instrument(key, Float64(d), :back, Float64(d), 1.0)
end

"""
    synthetic_back(key, complement, book, qrule; max_leverage) -> Instrument | nothing

Take the position `key` by **laying its complement**. The effective back price is
`lay_to_back(d)` where `d` is the complement's best lay price.

`max_leverage` rejects synthetics that need more backer stake per unit of risk than we are
willing to fund. This is the price-only substitute for a depth check: at 20x it removes every
laid price below 1.05, which is exactly where the measured "gain" was implausible (O/U 5.5 at
46%, O/U 0.5 at 23% -- an empty back book, not an edge).
"""
function synthetic_back(key::SelectionKey, complement::SelectionKey,
                        book::Dict{SelectionKey,BookLevels}, qrule::AbstractQuoteRule;
                        max_leverage::Float64 = 20.0)
    haskey(book, complement) || return nothing
    d = quote_price(qrule, book[complement], :lay)
    (isnan(d) || d <= 1) && return nothing
    lev = 1.0 / (d - 1.0)
    lev > max_leverage && return nothing
    D = lay_to_back(d)
    isfinite(D) || return nothing
    return Instrument(key, Float64(D), :lay, Float64(d), lev)
end
