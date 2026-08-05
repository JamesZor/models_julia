# src/Portfolio/implementations/filters.jl
#
# Curation: veto a stake the allocator wanted to take. Applied last, so a filter can only ever
# remove exposure -- it cannot resize what survives.

export KeepAll, MinEdge, MarketWhitelist, MinOdds, FilterChain

"Take everything the allocator wants."
struct KeepAll <: AbstractSelectionFilter end
keep(::KeepAll, ::Selection, ::Real, ::SlateContext) = true

"""
    MinEdge(e)

Require the model's edge over the vig-removed market price to clear `e` before backing.

Edge is measured as `p_model - p_market`, both on the same de-vigged scale, so `e` is in
probability points (0.03 ~ three points).
"""
struct MinEdge <: AbstractSelectionFilter
    e::Float64
end
keep(f::MinEdge, s::Selection, ::Real, ::SlateContext) = (s.p_model - s.p_market) >= f.e

"""
    MarketWhitelist(keys)

Only back selections in `keys`, given as `(group, line, selection)` tuples. The blunt form of
curation, and worth having: on the one out-of-sample test available, per-family curation was a
larger lever than anything in the allocator.
"""
struct MarketWhitelist <: AbstractSelectionFilter
    keys::Set{Tuple{String,Float64,Symbol}}
end
keep(f::MarketWhitelist, s::Selection, ::Real, ::SlateContext) =
    (s.group, s.line, s.selection) in f.keys

"""
    MinOdds(o)

Refuse prices below `o`. A crude liquidity and slippage proxy: short prices need the largest
liability for the smallest return and are the least forgiving of a stale quote.
"""
struct MinOdds <: AbstractSelectionFilter
    o::Float64
end
keep(f::MinOdds, s::Selection, ::Real, ::SlateContext) = s.odds_used >= f.o

"""
    FilterChain(filters...)

Conjunction: a selection survives only if every filter keeps it.
"""
struct FilterChain{T<:Tuple} <: AbstractSelectionFilter
    filters::T
end
FilterChain(fs::AbstractSelectionFilter...) = FilterChain(fs)
keep(f::FilterChain, s::Selection, a::Real, ctx::SlateContext) =
    all(g -> keep(g, s, a, ctx), f.filters)
