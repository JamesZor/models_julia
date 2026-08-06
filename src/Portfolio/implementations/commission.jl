# src/Portfolio/implementations/commission.jl

export PerBetCommission, NetMarketCommission, NoCommission

"""
    PerBetCommission(rate)

Charge commission on the gross profit of every winning bet independently.

This is *conservative* relative to a real exchange, which nets winners against losers within a
market before charging. Use it as the default: it never flatters a backtest.
"""
struct PerBetCommission <: AbstractCommissionModel
    rate::Float64
    function PerBetCommission(r::Real)
        0.0 <= r < 1.0 || throw(ArgumentError("commission rate must be in [0,1): $r"))
        new(Float64(r))
    end
end
net_return(c::PerBetCommission, d::Real) = (1.0 - c.rate) * (d - 1.0)

"""
    NetMarketCommission(rate)

What Betfair actually does: commission on net market winnings.

Not yet wired into settlement, which is per-selection -- charging it correctly requires netting
across a market group after the result is known. Declared here so the seam exists; `net_return`
currently behaves as gross (commission deferred to settlement) and `payoff` will over-state
returns if used. Do not select it until settlement-side netting lands.
"""
struct NetMarketCommission <: AbstractCommissionModel
    rate::Float64
end
net_return(::NetMarketCommission, d::Real) =
    error("NetMarketCommission requires market-level netting at settlement; not implemented")

"Zero commission -- for tests and for books priced somewhere other than an exchange."
struct NoCommission <: AbstractCommissionModel end
net_return(::NoCommission, d::Real) = d - 1.0
