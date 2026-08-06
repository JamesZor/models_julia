# src/MatchDay/matchday-module.jl

"""
    MatchDay

Turns a fixture list into an executable stake sheet.

`MatchDay` manufactures the two inputs `Portfolio` needs -- a `latents_df` of posterior draws
and an `odds_df` of prices -- for fixtures that have not been played, and **refuses loudly**
when it cannot. It does no staking maths: allocation, shrinkage, drawdown and the exposure cap
all live in `Portfolio` and are reached through `Portfolio.stake_sheet`.

# Pipeline

    fixtures -> identity -> lineups -> BOOK -> features -> inference -> gate -> stake_sheet

The book is built **before** features because market-pillar engines consume odds as a model
feature, so inference depends on the same prices staking does. Any drawing of this as a straight
line is wrong.

# The morphism

A lay is a back on the complement once the position is measured in **risk**: laying at `d` is
backing at `d/(d-1)`, with backer stake `risk/(d-1)`. `AbstractInstrumentRule` picks whichever
instrument prices a position better and hands `Portfolio` the effective odds, so the payoff
matrix, the allocator and `FixedCap` never learn that lays exist -- and `FixedCap` sums
liability by construction. Only `order_ticket` sees the difference.

Measured on `betfair_live.order_book_1m`, taking the better instrument is worth ~0.3% on
Ireland Premier's central lines and 3.5-6.4% on Scottish League One/Two's O/U 3.5 -- it tracks
book width, so it is worth most exactly where the book is worst.

# as_of

Every stage takes an explicit `as_of::DateTime` and no stage reads the clock. That is what makes
a past match day replayable from `order_book_1m`, and replay is the only route to validating any
of this.

# Health warnings, all measured 2026-08-06

* The identity resolver, the order-book drain and the lineup scraper are all **dead** (last
  output 2026-06-22, 2026-08-02, 2026-06-26). `MatchMetaCrosswalk` resolves 100% when the job
  runs and 0% after it stopped.
* `sofascore.lineup_provisional.confirmed` has **never** been true, because every scrape has run
  4.4-5.8h before kick-off and the XI lands ~1h out. `ConfirmedXI` is therefore non-blocking by
  default and `MaxLineupAge` is the usable gate.
* `last_price_traded` is NULL in 100% of `order_book_1m`, so this module prices off the book and
  cannot reproduce the backtest's `odds_close`. Those are different quantities; do not compare
  them without saying so.
"""
module MatchDay

using DataFrames
using Dates
using Statistics
using Printf
using LibPQ
using JSON3

using ..Data
using ..Features
using ..Models
using ..Experiments
using ..Predictions
using ..Portfolio

# --- order matters -----------------------------------------------------------
# types.jl names concrete types in its @kwdef defaults; those are evaluated when a constructor
# is CALLED, not when the struct is defined, so implementations/ may come after.

include("types.jl")
include("interfaces.jl")
include("db.jl")
include("instruments.jl")

include("implementations/sources.jl")
include("implementations/book.jl")
include("implementations/gates.jl")

include("inference.jl")
include("pipeline.jl")

export
    # seams
    AbstractFixtureSource, AbstractIdentityResolver, AbstractLineupSource, AbstractBookSource,
    AbstractQuoteRule, AbstractInstrumentRule, AbstractStakeRounding,
    AbstractFeatureMaterialiser, AbstractReadinessGate,

    # domain
    Fixture, SelectionKey, BookLevels, Instrument, Resolved, Unresolved, Player, Lineup,
    Ready, Blocked, FixtureCard, MatchDaySpec, MatchDayResult,

    # entry points
    match_day, build_cards, price_cards, fixture_info, order_ticket, blocked_report

end
