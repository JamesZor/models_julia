# src/Portfolio/implementations/trust.jl
#
# How much we believe the model over the market.
#
# Standing result, and the reason FlatTrust is the default: on the only out-of-sample test we
# have (ScottishLower, 628 matches, split at the season break), every attempt to *learn* a
# per-selection trust weight lost money. A vector fitted in-sample to a path metric returned
# 0.089x; the same fitted causally to decayed log-loss returned 0.718x. Flat quarter-trust
# returned 1.256x. The value of this seam is being able to keep testing and rejecting such
# ideas cheaply -- not an expectation that one of them works.

export FlatTrust, SelectionTrust, ScheduledTrust

"""
    FlatTrust(w)

One weight for every selection. `FlatTrust(1.0)` is full trust in the model, `FlatTrust(0.0)`
concedes the market is efficient and stakes nothing.

Note that flat trust cannot change the *ordering* of a book, only its scale -- and once a
drawdown constraint binds it cannot even do that (see `risk_factor`). Its effect is therefore
confined to the regime where the risk budget is slack.
"""
struct FlatTrust <: AbstractTrustModel
    w::Float64
    function FlatTrust(w::Real)
        0.0 <= w <= 1.0 || throw(ArgumentError("trust weight must be in [0,1]: $w"))
        new(Float64(w))
    end
end
trust_for(t::FlatTrust, ::Selection, ::SlateContext) = t.w

"""
    SelectionTrust(table; default = 0.25, strict = true)

Per-selection weights, keyed on `(group, line, selection)`.

`strict = true` raises on a selection the table does not cover. That is deliberate: the
prototype fell back to `0.0` on a missing key, which silently stopped betting a whole market and
looked like a modelling result rather than a typo.
"""
struct SelectionTrust <: AbstractTrustModel
    table::Dict{Tuple{String,Float64,Symbol},Float64}
    default::Float64
    strict::Bool
end
SelectionTrust(t::Dict{Tuple{String,Float64,Symbol},Float64}; default = 0.25, strict = true) =
    SelectionTrust(t, default, strict)

function trust_for(t::SelectionTrust, s::Selection, ::SlateContext)
    k = (s.group, s.line, s.selection)
    t.strict && return t.table[k]        # KeyError on a miss, by design
    return get(t.table, k, t.default)
end

"""
    ScheduledTrust(per_slate)

A trust model per slate index, where `per_slate[t]` must have been fitted using only slates
`1..t-1`.

Causality is carried by the *construction* (a single forward pass that fits, then uses, then
appends), not by a date filter at lookup time. A date-filtered lookup into a table fitted on the
whole sample looks safe and is not.
"""
struct ScheduledTrust{M<:AbstractTrustModel} <: AbstractTrustModel
    per_slate::Vector{M}
end
trust_for(t::ScheduledTrust, s::Selection, ctx::SlateContext) =
    trust_for(t.per_slate[ctx.idx], s, ctx)
