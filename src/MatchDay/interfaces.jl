# src/MatchDay/interfaces.jl
#
# One required method per seam, each with an error fallback so a half-implemented component
# fails at the call site instead of silently returning nothing.

export fixtures, resolve, lineup, quotes, quote_price, instrument, round_stake,
       materialise!, ready

"""
    fixtures(src::AbstractFixtureSource, segment, as_of::DateTime) -> Vector{Fixture}

Fixtures that have not kicked off as of `as_of`, for `segment`.

Implementations must filter on a **kick-off horizon**, not on a calendar date. `CURRENT_DATE`
and a 19:45 kick-off put the boundary in the wrong place the moment a timezone is involved.
"""
fixtures(s::AbstractFixtureSource, segment, ::DateTime) =
    error("fixtures not implemented for $(typeof(s))")

"""
    resolve(r::AbstractIdentityResolver, f::Fixture) -> Resolved | Unresolved

Never throws for an unmatched fixture and never returns `nothing`: an unresolved fixture must
remain visible downstream.
"""
resolve(r::AbstractIdentityResolver, ::Fixture) =
    error("resolve not implemented for $(typeof(r))")

"""
    lineup(src::AbstractLineupSource, f::Fixture, as_of::DateTime) -> Lineup | nothing

`nothing` means "no answer from this source"; `SourceChain` then tries the next one.

Implementations must not return data timestamped after `as_of` -- that is how a replay leaks
the future into the past.
"""
lineup(s::AbstractLineupSource, ::Fixture, ::DateTime) =
    error("lineup not implemented for $(typeof(s))")

"""
    quotes(src::AbstractBookSource, r::Resolved, as_of::DateTime) -> Dict{SelectionKey,BookLevels}

The most recent book at or before `as_of` for every selection the source can see.
"""
quotes(s::AbstractBookSource, ::Resolved, ::DateTime) =
    error("quotes not implemented for $(typeof(s))")

"""
    quote_price(rule::AbstractQuoteRule, b::BookLevels, side::Symbol) -> Float64

Collapse depth to one decimal price for `side ∈ (:back, :lay)`. Return `NaN` when the side is
empty; callers treat `NaN` as "not priceable" rather than as an error.
"""
quote_price(r::AbstractQuoteRule, ::BookLevels, ::Symbol) =
    error("quote_price not implemented for $(typeof(r))")

"""
    instrument(rule, key, complement, book, qrule) -> Instrument | nothing

Choose how to express the position `key`. `complement` is the opposing selection in the same
market group, or `nothing` when the group is not two-outcome (1X2), in which case only a direct
back is available.

Returns `nothing` when the position cannot be expressed at all.
"""
instrument(r::AbstractInstrumentRule, ::SelectionKey, ::Union{Nothing,SelectionKey},
           ::Dict{SelectionKey,BookLevels}, ::AbstractQuoteRule) =
    error("instrument not implemented for $(typeof(r))")

"""
    round_stake(r::AbstractStakeRounding, stake::Real, inst::Instrument) -> Float64

Apply the exchange minimum. `stake` is the **risk** in currency; the venue minimum applies to
the backer stake, which for a lay is `risk * leverage` -- so a lay at a short price satisfies a
£1 minimum with far less than £1 at risk. Return `0.0` to drop the leg.
"""
round_stake(r::AbstractStakeRounding, ::Real, ::Instrument) =
    error("round_stake not implemented for $(typeof(r))")

"""
    materialise!(m, ::Val{feature}, fs, fixtures, ctx) -> Bool

Fill `feature` into the `FeatureSet` `fs` for `fixtures`, which are in no training fold.
Return `true` if this materialiser handled the feature, `false` to defer to the next one.

`ctx` is a `NamedTuple` carrying at least `(ds, model, as_of, odds)`.
"""
materialise!(m::AbstractFeatureMaterialiser, ::Val, fs, ::Vector{Fixture}, ctx) =
    error("materialise! not implemented for $(typeof(m))")

"""
    ready(gate::AbstractReadinessGate, card::FixtureCard) -> Ready | Blocked

Gates are conjunctive. `GateChain` runs every member and concatenates reasons rather than
short-circuiting, because the second reason is usually the informative one.
"""
ready(g::AbstractReadinessGate, ::FixtureCard) =
    error("ready not implemented for $(typeof(g))")
