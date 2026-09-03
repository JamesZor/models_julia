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

# Health warnings, re-measured 2026-09-01

Three of the four warnings this docstring carried on 2026-08-06 are now WRONG, and a stale
measured claim is worse than none -- it is trusted. What replaces them:

* **The exchange collector has been decided-but-not-executed since 2026-08-28.**
  `core.matchday_action` holds 569 `arm / not_armed / executed = false` rows for 2026-08-29
  alone, reason "nothing is capturing and kickoff in 89 min". The supervisor is in dry-run: it
  decides correctly every 60s and carries nothing out. The last `order_book_1m` row is
  `2026-08-28 20:59`, so a full 16-fixture Scottish round on 08-29 has no book at all. Nothing in
  this module can price a live card until that is flipped to `execute`.
* **The identity crosswalk is dead again.** `betfair.match_meta` resolved 100% of Scottish
  fixtures 2026-08-01..2026-08-28 and **0 of 159 fixture-days from 2026-08-29**. Reach for
  `ResolverChain(MatchMetaCrosswalk(), LiveNameMatch())`; the fallback is opt-in rather than a
  default precisely so a dead crosswalk stays visible.
* **RETIRED: "`confirmed` has never been true".** It is true in **1,071 of 1,533** rows of
  `sofascore.lineup_provisional`, and the 2026-08-08/09 round scraped at **T-13 to T-42 min**
  (median ~T-29), not the 4.4-5.8h this docstring used to claim. `ConfirmedXI(blocking = true)`
  and `MaxLineupAge(Hour(2), blocking = true)` are both now usable gates. The feed is itself dead
  since 2026-08-09, which is a different problem with a different fix.
* **RETIRED: "`last_price_traded` is NULL in 100% of rows".** It began populating on
  **2026-08-07** and is present in 56-88% of rows since. Pre- and post-August CLV baselines are
  therefore different quantities and must not be pooled; record which one was used.
* The exchange collector still only carries the **current day**, so a weekend card has to be
  priced one match day at a time rather than in a single Friday run.
* Volumes and prices are both scaled **x10000** on the feed. Verified two ways: per-runner
  `total_matched` sums exactly to `market_matched`, and Kilmarnock v Celtic's MATCH_ODDS peaked at
  `2,499,888,300` = **£249,989**, which is the right order of magnitude for a Scottish Premiership
  fixture where a x100 reading gives £25M. A top-of-book size of `20000` is **£2.00**, the
  Betfair minimum -- not £200.

# The slate, the ledger and the console

`price_slate` is the entry point for live use and `match_day` is its predecessor: the difference
is that `PricedSlate` retains the BOOK it priced against and the SLATE-WIDE allocation
diagnostics (`k_risk`, exposure, `capped`), neither of which is recoverable after the fact.
`ledger/` is the paper-trading substrate -- an atomic `SELECT ... FOR UPDATE` reservation of the
whole stake vector, then a per-order state machine -- and `console/` serves it.
"""
module MatchDay

using DataFrames
using Dates
using Statistics
using Printf
using LibPQ
using JSON3
using UUIDs
using HTTP

using ..Data
using ..Features
using ..Models
using ..Experiments
using ..Predictions
using ..Portfolio

# Qualified rather than `using`, so every call site says where the name comes from and none of
# these modules' exports can shadow one of ours. `Training` and `Evaluation` are reached only
# from `fits.jl`; `BackTesting` is not reached at all.
import ..Training
import ..Evaluation

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
include("fits.jl")
include("slate.jl")

# --- Phase 2: the paper ledger ----------------------------------------------
# Pure logic before persistence: `state_machine.jl` and `fills.jl` are DB-free by construction,
# which is what makes a Saturday replayable as a table of numbers rather than only on a Saturday.
include("ledger/types.jl")
include("ledger/fills.jl")
include("ledger/state_machine.jl")
include("ledger/schema.jl")
include("ledger/db.jl")
include("ledger/reservation.jl")
include("ledger/settle.jl")

# --- Phase 3: the operator console ------------------------------------------
include("console/snapshot.jl")
include("console/server.jl")

# last: it dispatches on every type and component defined above
include("display.jl")

export
    # seams
    AbstractFixtureSource, AbstractIdentityResolver, AbstractLineupSource, AbstractBookSource,
    AbstractQuoteRule, AbstractInstrumentRule, AbstractStakeRounding,
    AbstractFeatureMaterialiser, AbstractReadinessGate,

    # domain
    Fixture, SelectionKey, BookLevels, Instrument, Resolved, Unresolved, Player, Lineup,
    Ready, Blocked, FixtureCard, MatchDaySpec, MatchDayResult,

    # materialisers
    RatingsFromTracker, LeagueFromFixture, MaterialiserChain,

    # identity
    MatchMetaCrosswalk, LiveNameMatch, ResolverChain, team_name_score, match_event_scores,

    # entry points
    match_day, build_cards, price_cards, fixture_info, order_ticket, blocked_report,

    # the slate
    PricedSlate, price_slate, slate_batch_summary, canonical_markets,
    canonical_scottish_lower_policy,
    leg_capacity, annotate_capacity!, sweep_ladder, fill_confidence,
    CanonicalFit, canonical_fit,

    # the ledger
    PaperAccount, PaperOrder, LedgerDelta, OrderState, BatchState,
    TouchOnly, LadderSweep, Optimistic, simulate_fill,
    decide_order, apply_transition, reserve_plan,
    migrate_paper_schema!, drop_paper_schema!, paper_connection,
    insert_slate!, insert_orders!, execute_slate_batch!, record_fills!,
    settle_slate!, clv_for_order, account_row, slate_row, order_rows,

    # the console
    slate_snapshot, ConsoleState, serve_console, stop_console!

end
