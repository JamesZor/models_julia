# src/MatchDay/types.jl
#
# Abstract seams, then the domain objects that flow through the pipeline, then configuration.
# Mirrors src/Portfolio/types.jl deliberately -- the two modules are meant to read as one system.
#
# All `Base.show` methods live in display.jl, not here.

# ===================================================================
# 1. Abstract types -- the swappable seams
# ===================================================================

"""
    AbstractFixtureSource

Where "what is on" comes from.
Contract: `fixtures(src, segment, as_of) -> Vector{Fixture}`.
"""
abstract type AbstractFixtureSource end

"""
    AbstractIdentityResolver

Maps a SofaScore fixture onto the exchange's event and markets.
Contract: `resolve(r, fixture) -> Union{Resolved,Unresolved}`.

Note this is a *lookup*, not a matcher. `betfair.match_meta` is a purpose-built crosswalk
populated by a separate resolution job; when that job runs it resolves 100%. A fuzzy matcher
here would be a second source of truth papering over an operational gap.
"""
abstract type AbstractIdentityResolver end

"""
    AbstractLineupSource

Contract: `lineup(src, fixture, as_of) -> Union{Nothing,Lineup}`.
Returning `nothing` means "this source has no answer", which is what makes `SourceChain` work.
"""
abstract type AbstractLineupSource end

"""
    AbstractBookSource

Where prices come from. Contract:
`quotes(src, resolved, as_of) -> Dict{SelectionKey,BookLevels}`.

Returns *depth*, not a scalar -- collapsing to one number is `AbstractQuoteRule`'s job, and the
gates need to see the book before it is collapsed.
"""
abstract type AbstractBookSource end

"""
    AbstractQuoteRule

Which number in the book we treat as the price.
Contract: `quote_price(rule, levels, side) -> Float64` where `side ∈ (:back, :lay)`.
"""
abstract type AbstractQuoteRule end

"""
    AbstractInstrumentRule

How a canonical selection is *expressed* as an exchange order. The canonical selection space is
always the model's (`:over`, `:under`, `:home`, ...); this decides whether that position is taken
by backing it or by laying its complement.

Contract: `instrument(rule, key, complement_key, book, qrule) -> Union{Nothing,Instrument}`.
"""
abstract type AbstractInstrumentRule end

"""
    AbstractStakeRounding

The exchange minimum. Contract: `round_stake(r, stake, inst) -> Float64`, returning `0.0` to
drop the leg. `stake` is in currency, not bankroll fractions.
"""
abstract type AbstractStakeRounding end

"""
    AbstractFeatureMaterialiser

Produces one model feature for fixtures that are in no training fold.
Contract: `materialise!(m, ::Val{feature}, fs, fixtures, ctx) -> Bool` (true if it handled it).
"""
abstract type AbstractFeatureMaterialiser end

"""
    AbstractReadinessGate

Refuse-to-bet conditions. Contract: `ready(gate, card) -> Union{Ready,Blocked}`.
Gates are conjunctive and collect every reason -- a silent gate is worse than no gate.
"""
abstract type AbstractReadinessGate end

# ===================================================================
# 2. Domain objects
# ===================================================================

"""
    Fixture

One upcoming match. `kickoff` is a `DateTime`, never a `Date`: lineups firm up ~1h out,
liquidity builds toward the off, and every gate in the system is a function of time-to-kickoff.
"""
struct Fixture
    m_id::Int
    home::String
    away::String
    kickoff::DateTime
    tournament_id::Int
end

"Identifies one priced leg: market group, line, and the model's selection symbol."
const SelectionKey = @NamedTuple{group::String, line::Float64, selection::Symbol}

"""
    BookLevels

Depth at one selection, unscaled from the exchange's x10000 integers.

`back` is the bid side (prices available to back) and `lay` the ask side, both best-first.
Verified by overround sign on `betfair_live.order_book_1m`: the back side sums above 1 and the
lay side below 1, which is the only assignment consistent with an arbitrage-free book.
"""
struct BookLevels
    back::Vector{Float64}
    back_size::Vector{Float64}
    lay::Vector{Float64}
    lay_size::Vector{Float64}
    matched::Float64
    ts::DateTime
end

best_back(b::BookLevels) = isempty(b.back) ? NaN : b.back[1]
best_lay(b::BookLevels)  = isempty(b.lay)  ? NaN : b.lay[1]

"""
    Instrument

How one canonical selection will actually be taken.

* `odds` -- the **effective** decimal odds of the position, whichever side it is expressed on.
  This is what Portfolio sees, and it is denominated so that a unit of stake is a unit of risk.
* `side` -- `:back` or `:lay`.
* `venue_odds` -- the price actually shown on the exchange for `side`.
* `leverage` -- backer stake required per unit of risk. `1.0` for a back; `1/(d-1)` for a lay,
  which is what blows up as a laid price approaches 1.
* `venue_key` -- **the runner the order actually touches.** Equal to `key` for a direct back;
  the COMPLEMENT for a synthetic.

For a lay, `odds = d/(d-1)` and the order placed is `stake_at_venue = risk * leverage`.

`venue_key` exists because `key` and `venue_odds` describe DIFFERENT RUNNERS on a synthetic, and
without it the two get printed together as though they belonged to each other. Backing Over 2.5
by laying Under 2.5 at `d` is `Instrument(over_25, d/(d-1), :lay, d, lev, under_25)`; an order
ticket naming `over_25` at `d` on the lay side is the opposite position at a price that belongs
to the other runner. Measured on the 2026-08-08 ScottishLower slate: 14 of 48 legs were
synthetics, so ~29% of tickets were mis-specified.

The five-argument constructor defaults `venue_key = key`, which is correct for a direct back and
is what keeps every existing call site right by construction rather than by inspection.
"""
struct Instrument
    key::SelectionKey
    odds::Float64
    side::Symbol
    venue_odds::Float64
    leverage::Float64
    venue_key::SelectionKey
end

Instrument(key::SelectionKey, odds::Real, side::Symbol, venue_odds::Real, leverage::Real) =
    Instrument(key, Float64(odds), side, Float64(venue_odds), Float64(leverage), key)

"Backer stake to place at the venue to carry `risk` units of risk."
venue_stake(inst::Instrument, risk::Real) = risk * inst.leverage

struct Resolved
    fixture::Fixture
    bf_event_id::String
    market_ids::Dict{String,String}      # "OVER_UNDER_25" => "1.260457203"
    verified::Bool
end

struct Unresolved
    fixture::Fixture
    reason::Symbol                       # :absent_from_crosswalk | :no_markets | :not_verified
end

struct Player
    player_id::Int
    name::String
    position::Symbol                     # :G :D :M :F -- already normalised
    substitute::Bool
end

"""
    Lineup

`source` and `scraped_at` are load-bearing, not decoration: `confirmed` has never once been
true in `sofascore.lineup_provisional` because every scrape has run 4.4-5.8h before kick-off
and the XI lands ~1h out. Staleness is therefore the usable signal, not the flag.
"""
struct Lineup
    home::Vector{Player}
    away::Vector{Player}
    confirmed::Bool
    source::Symbol                       # :json_pin | :provisional | :last_historical
    scraped_at::DateTime
end

struct Ready end
struct Blocked
    reasons::Vector{Pair{Symbol,String}}
end
is_ready(::Ready) = true
is_ready(::Blocked) = false

"""
    FixtureCard

Everything known about one fixture at one instant. An `Unresolved` fixture still gets a card --
filtering it out at stage 2 is exactly how a fixture becomes invisible rather than reported.
"""
mutable struct FixtureCard{I<:Union{Resolved,Unresolved}}
    fixture::Fixture
    identity::I
    lineup::Union{Nothing,Lineup}
    as_of::DateTime
    readiness::Union{Nothing,Ready,Blocked}
end

FixtureCard(f::Fixture, id, as_of::DateTime) = FixtureCard(f, id, nothing, as_of, nothing)

resolved(c::FixtureCard{Resolved})   = true
resolved(c::FixtureCard{Unresolved}) = false

# ===================================================================
# 3. Configuration
# ===================================================================

"""
    MatchDaySpec

Every stage is a swappable component; `as_of` is **not** a field here because it is a property
of a *run*, not of a configuration -- see `match_day`. No stage reads the clock internally.
"""
Base.@kwdef struct MatchDaySpec{F<:AbstractFixtureSource, I<:AbstractIdentityResolver,
                                L<:AbstractLineupSource,  B<:AbstractBookSource,
                                Q<:AbstractQuoteRule,     N<:AbstractInstrumentRule,
                                R<:AbstractStakeRounding, M<:AbstractFeatureMaterialiser,
                                G<:AbstractReadinessGate}
    fixtures::F   = SofaScoreEvents()
    identity::I   = MatchMetaCrosswalk()
    lineups::L    = SourceChain(ProvisionalDB(), LastHistorical())
    book::B       = ArchivedOrderBook()
    quote_rule::Q = BestAvailable()
    instrument::N = BestOfBackLay()
    rounding::R   = NoMinimum()
    features::M   = MaterialiserChain(RatingsFromTracker(), LineupAggregateFromRAPM(),
                                      LeagueFromFixture())
    gate::G       = GateChain(IdentityResolved(), MaxBookAge(Minute(30)))
    markets::Data.MarketConfig = Data.MarketConfig(
        reduce(vcat, (Data.AbstractMarket[Data.Market1X2(), Data.MarketBTTS()],
                      [Data.MarketOverUnder(i + 0.5) for i in 0:4])))
end

"""
    MatchDayResult

What a run produced. A refusal is a **value**, never an absent row: `blocked` carries every
card the gate stopped and why, so "no bets today" and "the pipeline is broken" are
distinguishable.
"""
struct MatchDayResult
    sheet::DataFrame
    cards::Vector{FixtureCard}
    blocked::Vector{FixtureCard}
    odds::DataFrame
    instruments::Dict{Tuple{Int,SelectionKey},Instrument}
    as_of::DateTime
end

