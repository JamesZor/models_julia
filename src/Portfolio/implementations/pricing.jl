# src/Portfolio/implementations/pricing.jl

export DeArb, Normalise, RawPrice

"""
    DeArb()

One-sided de-arbitrage: `d * min(overround, 1)`. Shrink a book that prices at an impossible
overround; leave a book with genuine vig alone. **Never settles above the traded price.**

This is not cosmetic. The closing "price" is a time-weighted average of trades that happened at
different moments -- on ScottishLower the median O/U and BTTS market has *one* trade in the
20-minute window -- so ~45% of O/U groups come out at overround < 1. Left alone, the Kelly
solver reads that as a risk-free arbitrage and levers into it: measured 97 full-cover positions
and mean stake 29.2% of bankroll versus 18.6% with this policy.

A second consequence: after de-arbing, with commission > 0 we have `sum 1/(1+c_i) > 1` strictly,
so covering every outcome of a market group is dominated in every state. The optimum therefore
never takes a multi-sided position, which is why this module needs no netting pass.
"""
struct DeArb <: AbstractPricePolicy end
settlement_odds(::DeArb, d::Real, overround::Real) = d * min(overround, 1.0)

"""
    Normalise()

`d * overround` -- divide out the book in both directions. Reproduces the prototype's behaviour
and exists for ablation. Note this settles *above* the traded price whenever there is real vig,
which manufactures edge; do not use it for anything you intend to believe.
"""
struct Normalise <: AbstractPricePolicy end
settlement_odds(::Normalise, d::Real, overround::Real) = d * overround

"Leave the traded price untouched, arbitrages included. Ablation only."
struct RawPrice <: AbstractPricePolicy end
settlement_odds(::RawPrice, d::Real, ::Real) = d
