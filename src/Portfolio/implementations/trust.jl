# src/Portfolio/implementations/trust.jl
#
# How much we believe the model over the market.
#
# `FlatTrust` remains the league-agnostic default. Scottish Lower's directional tiers are an
# audited production policy, exposed explicitly through `CanonicalScottishLowerTrust()` rather
# than silently applied to every league that uses Portfolio.

export FlatTrust, SelectionTrust, TieredTrust, CanonicalScottishLowerTrust, ScheduledTrust

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

const TrustSelectionKey = Tuple{String,Float64,Symbol}

function _tiered_group_key(group::AbstractString)
    compact = replace(lowercase(strip(group)), r"[^a-z0-9]" => "")
    compact == "1x2" && return "1x2"
    compact == "overunder" && return "over_under"
    compact == "btts" && return "btts"
    return lowercase(strip(group))
end

function _tiered_selection_key(selection::Symbol)
    name = String(selection)
    startswith(name, "over_") && return :over
    startswith(name, "under_") && return :under
    return selection
end

_tiered_key(group::AbstractString, line::Real, selection::Symbol) =
    (_tiered_group_key(group), Float64(line), _tiered_selection_key(selection))

"""
    TieredTrust(table; default = 0.0)

Audited per-selection trust tiers keyed by `(market_group, market_line, selection)`. Unmatched
selections receive `default`, which is zero by default so adding a new market cannot silently
make it stakeable.

Market-group spelling is normalised (`"1X2"` and `"1x2"`; `"OverUnder"` and
`"over_under"`), and line-specific totals symbols such as `:under_25` are normalised to the
readable directional key `:under`. This lets policy tables describe directions without leaking
the odds-feed encoding into configuration.
"""
struct TieredTrust <: AbstractTrustModel
    table::Dict{TrustSelectionKey,Float64}
    default::Float64
end

function TieredTrust(table::AbstractDict; default::Real = 0.0)
    0.0 <= default <= 1.0 ||
        throw(ArgumentError("default trust weight must be in [0,1]: $default"))
    normalised = Dict{TrustSelectionKey,Float64}()
    for (key, weight) in table
        key isa Tuple && length(key) == 3 || throw(ArgumentError(
            "tiered trust keys must be (market_group, market_line, selection), got $key"))
        group, line, selection = key
        group isa AbstractString || throw(ArgumentError(
            "tiered trust market_group must be a string, got $(typeof(group)) in $key"))
        line isa Real || throw(ArgumentError(
            "tiered trust market_line must be real, got $(typeof(line)) in $key"))
        selection isa Symbol || throw(ArgumentError(
            "tiered trust selection must be a Symbol, got $(typeof(selection)) in $key"))
        weight isa Real || throw(ArgumentError(
            "tiered trust weight must be real, got $(typeof(weight)) for $key"))
        0.0 <= weight <= 1.0 ||
            throw(ArgumentError("trust weight must be in [0,1] for $key: $weight"))
        normalised[_tiered_key(group, line, selection)] = Float64(weight)
    end
    return TieredTrust(normalised, Float64(default))
end

trust_for(t::TieredTrust, s::Selection, ::SlateContext) =
    get(t.table, _tiered_key(s.group, s.line, s.selection), t.default)

"""
    CanonicalScottishLowerTrust() -> TieredTrust

The production `P1_conservative_tilt` policy audited over the Scottish Lower 40-fold
walk-forward study: Home and Under 2.5 at 0.35, Draw and Away at 0.25, every other selection at
zero.
"""
CanonicalScottishLowerTrust() = TieredTrust(Dict(
    ("1x2", 0.0, :home)          => 0.35,
    ("over_under", 2.5, :under) => 0.35,
    ("1x2", 0.0, :draw)          => 0.25,
    ("1x2", 0.0, :away)          => 0.25,
); default = 0.0)

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
