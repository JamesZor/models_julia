# current_development/match_day_inference/replay_state.jl
#
# THE REPLAY ENGINE. Loader half of the r08 pair: types, point-in-time sources, the model
# registry, the clock, and the pricing/execution/settlement transitions. `r08_replay_console.jl`
# is the runner and `replay_server.jl` is the HTTP/WebSocket surface.
#
# WHAT THIS IS. `MatchDay`'s own docstring says it: "Every stage takes an explicit `as_of` and no
# stage reads the clock. That is what makes a past match day replayable from `order_book_1m`, and
# replay is the only route to validating any of this." This file is that route. It drives the
# SAME `MatchDay` pipeline the live console drives, with `as_of` supplied by a scrubber instead of
# by `now()`, so anything it proves is a claim about the live path rather than about a parallel
# one.
#
# THREE THINGS IT DOES NOT DO, DELIBERATELY.
#
# 1. IT DOES NOT RE-IMPLEMENT PRICING. `build_cards`, `quote_slate`, `Portfolio.stake_sheet`,
#    `_attach_instruments!` and `annotate_capacity!` are called, in the pipeline's own order.
#    What is new here is only the CACHING and the point-in-time SOURCES.
# 2. IT DOES NOT TOUCH `paper_runbook`. Every ledger call is parameterised by `schema` and
#    `assert_replay_schema` refuses anything but `paper_replay`. The live console on 8085 holds
#    its own connection to its own schema and neither process can reach the other's rows.
# 3. IT DOES NOT SAMPLE. Latents come from `MD.canonical_fit`, i.e. a chain that already exists in
#    `mcmc_experiments`. A replay that had to fit something would be a training run wearing a
#    scrubber.
#
# WHY EVERYTHING IS PRELOADED. A tick at 60x is one simulated minute per wall second. The live
# path's `_query` opens and closes a connection PER FIXTURE per read, so a 10-fixture card costs
# ~20 round trips a tick and the scrubber stutters. `PreloadedBook`, `PreloadedLineups` and
# `FrozenIdentity` read the whole match day ONCE, then serve `as_of` slices out of memory with a
# binary search and no allocation of a connection. The point-in-time contract is unchanged and is
# in fact easier to check here: the ladder vectors are sorted by `ts` and the search is
# `searchsortedlast`, so a tick from after the instant being replayed is unreachable rather than
# merely unqueried.
#
# WHY THE LATENTS ARE CACHED ON THE LINEUP. `Features.create_features` rebuilds every fold and
# costs ~60 s for the hybrid player pillar; `extract_parameters` costs milliseconds. Between them
# sits the only thing that moves during a replay: the XI. So the fold FeatureSet is built once per
# MODEL and the posterior extraction is memoised on a hash of the point-in-time lineups. m00 and
# m05 read no lineup, so they extract exactly once; m12 extracts twice -- before the XI drops and
# after -- and that second extraction IS the lineup shock the console draws.

import BayesianFootball
import DataFrames
import Dates
import LibPQ
import JSON3
import Statistics
import UUIDs

using DataFrames: DataFrame, nrow, eachrow
using Dates: DateTime, Date, Day, Hour, Minute, Second, datetime2unix, unix2datetime
using UUIDs: UUID, uuid4

const MD = BayesianFootball.MatchDay
const PF = BayesianFootball.Portfolio
const DD = BayesianFootball.Data
const TT = BayesianFootball.Training
const MO = BayesianFootball.Models
const FE = BayesianFootball.Features

# ===================================================================
# 1. Isolation constants -- the only defence that is not a convention
# ===================================================================

"The one schema a replay may write. Anything else is refused by `assert_replay_schema`."
const REPLAY_SCHEMA = "paper_replay"

"""
Schemas a replay must never reach. `paper_runbook` is the LIVE paper ledger the 8085 console
writes; `paper` is `MD.PAPER_SCHEMA`, the production default.

This is a list rather than a comment because the schema name is a string that flows through
eleven `MatchDay` entry points, and a string that is only ever checked by eye is not checked.
"""
const FORBIDDEN_SCHEMAS = ("paper_runbook", MD.PAPER_SCHEMA)

"The port the replay console binds. 8085 is the live console and is never touched."
const REPLAY_PORT = 8086

"""
    assert_replay_schema(schema) -> String

Refuse any schema but the replay one, naming what was attempted.

Called at the top of every function in this file that takes a `schema`, including the ones that
only read. A read against `paper_runbook` is not dangerous, but a code path that can read it is
one edit away from being one that writes it.
"""
function assert_replay_schema(schema::AbstractString)
    s = String(schema)
    s in FORBIDDEN_SCHEMAS && error(
        "replay: refusing to touch schema '$s'. The replay console writes ONLY to " *
        "'$REPLAY_SCHEMA'; '$s' is the live paper ledger served by the 8085 console and a " *
        "replayed fill in it would be indistinguishable from a real one.")
    s == REPLAY_SCHEMA || error(
        "replay: schema '$s' is not '$REPLAY_SCHEMA'. Replay execution is confined to one " *
        "schema so that dropping it can never cost a live record.")
    return s
end

# ===================================================================
# 2. The clock
# ===================================================================

"Replay window, in minutes relative to the slate's kick-off."
const T_START = -60      # pre-lineup: the book is thin and no XI has been published
const T_END   = 105      # full time plus stoppage, i.e. the settlement instant
const T_LINEUP  = -30    # the marker the XI is expected to land at (measured median T-29)
const T_EXEC    = -15    # the recommended entry instant
const T_KICKOFF = 0

"Speed multipliers the console offers. 60x is one simulated minute per wall second."
const SPEEDS = (1.0, 5.0, 30.0, 60.0)

"""
    ReplayClock

Where the replay is, and whether it is moving.

`t` is MINUTES RELATIVE TO KICK-OFF rather than an absolute `DateTime`, and that is the whole
point: every gate, marker and jump target in this system is a function of time-to-kickoff, so a
clock denominated in it needs no arithmetic to answer "are we past the XI drop?". The absolute
instant handed to `MatchDay` is derived (`kickoff + Minute(t)`), never stored, so the two can
never disagree.
"""
mutable struct ReplayClock
    t::Int
    playing::Bool
    speed::Float64
end

ReplayClock() = ReplayClock(T_START, false, 60.0)

clamp_t(t::Integer) = Int(clamp(t, T_START, T_END))

# ===================================================================
# 3. Point-in-time sources
# ===================================================================
#
# Three seams, each replacing a live source with a preloaded one. Every one of them keeps the
# `as_of` contract of the source it replaces; none of them can see past it.

"""
    ReplayFixtures(list)

The card, held fixed for the whole replay.

Distinct from `MD.ExplicitFixtures`, which filters `f.kickoff >= as_of`. That filter is correct
live -- a kicked-off fixture is not biddable -- and WRONG here: the replay clock runs to T+105 so
that settlement can be watched, and `ExplicitFixtures` would empty the card at the first whistle
and take the settlement view's inputs with it.
"""
struct ReplayFixtures <: MD.AbstractFixtureSource
    list::Vector{MD.Fixture}
end

MD.fixtures(s::ReplayFixtures, _segment, ::DateTime) = s.list

"""
    FrozenIdentity(by_match)

Identity resolved ONCE, at load, and then held.

The crosswalk (`betfair.match_meta`) and the name-match fallback are both stable facts about a
past match day: nothing about which Betfair event a 2026-08-08 fixture was can change while the
scrubber moves. Re-resolving per tick would be 10 queries a second to learn the same answer, and
would make the replay's identity depend on when it was run.
"""
struct FrozenIdentity <: MD.AbstractIdentityResolver
    by_match::Dict{Int,Union{MD.Resolved,MD.Unresolved}}
end

MD.resolve(r::FrozenIdentity, f::MD.Fixture) =
    get(r.by_match, f.m_id, MD.Unresolved(f, :absent_from_crosswalk))

"""
    PreloadedBook(ladders; max_age)

`betfair_live.order_book_1m` for one match day, held in memory and sliced by `as_of`.

`ladders[match_id][selection]` is the whole day's snapshots for that runner, **ascending in
`ts`**, so the point-in-time read is `searchsortedlast` on the timestamps and cannot return a
tick from after the instant being replayed. That is the same guarantee `ORDER_BOOK_SQL`'s
`DISTINCT ON ... ts <= as_of` gives, made structural rather than textual.

`max_age` reproduces `ArchivedOrderBook`'s staleness bound: a selection whose most recent
snapshot is older than this is omitted entirely, which is what lets `MaxBookAge` see a fixture
with no book rather than one with a stale price.
"""
struct PreloadedBook <: MD.AbstractBookSource
    ladders::Dict{Int,Dict{MD.SelectionKey,Vector{MD.BookLevels}}}
    stamps::Dict{Int,Dict{MD.SelectionKey,Vector{DateTime}}}
    max_age::Dates.Period
end

function MD.quotes(s::PreloadedBook, r::MD.Resolved, as_of::DateTime)
    out = Dict{MD.SelectionKey,MD.BookLevels}()
    mid = r.fixture.m_id
    per_sel = get(s.ladders, mid, nothing)
    per_sel === nothing && return out
    per_ts = s.stamps[mid]
    floor_ts = as_of - s.max_age
    for (key, levels) in per_sel
        stamps = per_ts[key]
        i = searchsortedlast(stamps, as_of)
        i == 0 && continue
        stamps[i] < floor_ts && continue
        out[key] = levels[i]
    end
    return out
end

"""
    PreloadedLineups(rows)

`sofascore.lineup_provisional` for one match day, sliced by `scraped_at <= as_of`.

**There is no historical fallback behind this on purpose.** The live spec chains
`SourceChain(ProvisionalDB(), LastHistorical(ds))` so that a player engine always has *some* XI
to price off. In a replay that fallback would hide the event the replay exists to show: before
the scrape lands, a player model must price with NO lineup and contribute exactly zero, and the
step from zero to the confirmed XI is the shock. Chaining last-season's XI in front of it would
replace a visible discontinuity with an invisible bias.
"""
struct PreloadedLineups <: MD.AbstractLineupSource
    rows::Dict{Int,DataFrame}
end

function MD.lineup(s::PreloadedLineups, f::MD.Fixture, as_of::DateTime)
    df = get(s.rows, f.m_id, nothing)
    df === nothing && return nothing
    sel = findall(<=(as_of), df.scraped_at)
    isempty(sel) && return nothing

    home = MD.Player[]; away = MD.Player[]
    for i in sel
        p = MD.Player(Int(df.player_id[i]),
                      ismissing(df.player_name[i]) ? "Unknown" : String(df.player_name[i]),
                      MD.clean_position(ismissing(df.position[i]) ? "M" : String(df.position[i])),
                      coalesce(df.substitute[i], false))
        push!(coalesce(df.is_home_team[i], true) ? home : away, p)
    end
    (isempty(home) || isempty(away)) && return nothing
    return MD.Lineup(home, away, any(coalesce.(df.confirmed[sel], false)), :provisional,
                     maximum(df.scraped_at[sel]))
end

# ===================================================================
# 4. Feature materialisers for the plus-minus player pillar
# ===================================================================
#
# `MD.INJECTABLE_KEYS` covers the two per-match maps the live engines read. The hybrid RAPM
# pillar reads a THIRD, `:player_lineup_ratings_map`, and that one is the reason this section
# exists -- twice over.
#
# 1. `RatingsFromTracker` reaches for `model.player_ratings_feature.tracker`, which
#    `PlayerLineupPillar` models do not have. It throws a `FieldError` rather than deferring, so
#    the hybrid model cannot be priced through the live materialiser chain at all.
# 2. Worse, and quietly: `:player_lineup_ratings_map` is built by the extractor over EVERY match
#    in `ds.lineups`, including finished ones. For a replayed fixture that map already holds the
#    XI that actually took the field. Leaving it alone would price a T-60m decision off the
#    teamsheet, which is precisely the leak `as_of` exists to make impossible.
#
# Both are fixed by materialising the key from the POINT-IN-TIME lineup, writing a neutral entry
# when there is none.

"""
    REPLAY_INJECTABLE_KEYS

The per-match lookup maps a replay must own. `MD.INJECTABLE_KEYS` plus
`:player_lineup_ratings_map`, which the hybrid pillar reads as
`get(lineup_map, match_id, neutral)` and which otherwise carries the played XI forward.
"""
const REPLAY_INJECTABLE_KEYS = (:player_ratings_map, :league_lookup, :player_lineup_ratings_map)

"""
    PointInTimeLineupRatings()

Materialises the plus-minus player maps from the XI visible at `as_of`.

Claims a key only when the fold FeatureSet carries `:plus_minus_ratings` -- the leak-safe
per-player RAPM vector the ridge was fitted on. That test is what makes this materialiser
composable with `RatingsFromTracker`: on a tracker-rating engine it returns `false` and defers,
on a plus-minus engine it claims and the tracker path is never reached.

The rating vector is NOT recomputed. Only the teamsheet it is applied to moves, which is the
distinction `plus_minus_extractors.jl` is explicit about: "Only the RATING VECTOR is
leak-controlled; applying it to a future teamsheet is precisely the pre-match rating being
tested."
"""
struct PointInTimeLineupRatings <: MD.AbstractFeatureMaterialiser end

"Goalkeepers are excluded from every plus-minus aggregate; see `pm_lineup_aggregates`."
_pm_pos_index(p::Symbol) = p === :D ? 0 : p === :M ? 1 : 2

"""
    pm_aggregate_from_lineup(lu, ratings) -> FE.PMLineupAggregate

One `PMLineupAggregate` from one point-in-time `Lineup`, in the field order
`pm_lineup_aggregates` writes.

The minute-weighted slots (17, 18) take starters at weight 1.0 and substitutes at 0.0. That is
what `pm_lineup_aggregates` itself does for a player with no recorded minute history, and a
pre-match teamsheet has none by construction -- the history it would use is the current match's
own minutes, which have not been played. Any other weighting here would be inventing information
the live path does not have either.
"""
function pm_aggregate_from_lineup(lu::MD.Lineup, ratings::Dict{Int,Float64})
    v = zeros(Float64, 18)
    for (is_home, players) in ((true, lu.home), (false, lu.away))
        for p in players
            p.position === :G && continue
            r = get(ratings, p.player_id, 0.0)
            isfinite(r) || continue
            pos = _pm_pos_index(p.position)
            if p.substitute
                v[is_home ? 3 : 4] += r
                v[(is_home ? 11 : 14) + pos] += r
            else
                v[is_home ? 1 : 2] += r
                v[(is_home ? 5 : 8) + pos] += r
                v[is_home ? 17 : 18] += r
            end
        end
    end
    return FE.PMLineupAggregate(Tuple(v))
end

function MD.materialise!(::PointInTimeLineupRatings, ::Val{:player_lineup_ratings_map}, fs,
                         fx::Vector{MD.Fixture}, ctx)
    haskey(fs.data, :plus_minus_ratings) || return false
    ratings = Dict{Int,Float64}(fs.data[:plus_minus_ratings])
    map_ = fs.data[:player_lineup_ratings_map]
    neutral = FE.PMLineupAggregate(ntuple(_ -> 0.0, 18))
    for f in fx
        lu = get(ctx.lineups, f.m_id, nothing)
        # An entry is written even with no lineup, and it is the NEUTRAL one. Deleting the key
        # would fall through to the same zeros via `get(..., neutral)`, but writing it makes the
        # pre-drop state an explicit fact of the FeatureSet rather than an absence -- which is
        # what `check_coverage` can then assert on.
        map_[f.m_id] = lu === nothing ? neutral : pm_aggregate_from_lineup(lu, ratings)
    end
    return true
end

function MD.materialise!(::PointInTimeLineupRatings, ::Val{:player_ratings_map}, fs,
                         fx::Vector{MD.Fixture}, ctx)
    haskey(fs.data, :plus_minus_ratings) || return false
    ratings = Dict{Int,Float64}(fs.data[:plus_minus_ratings])
    map_ = fs.data[:player_ratings_map]
    for f in fx
        lu = get(ctx.lineups, f.m_id, nothing)
        entry = Dict{Tuple{String,String},Float64}()
        if lu !== nothing
            for (side, players) in (("home", lu.home), ("away", lu.away))
                for p in players
                    p.substitute && continue
                    r = get(ratings, p.player_id, 0.0)
                    (isfinite(r) && r != 0.0) || continue
                    key = (side, String(p.position))
                    entry[key] = get(entry, key, 0.0) + r
                end
            end
        end
        map_[f.m_id] = entry
    end
    return true
end

MD.materialise!(::PointInTimeLineupRatings, ::Val, _fs, ::Vector{MD.Fixture}, _ctx) = false

"""
    replay_materialisers() -> MD.MaterialiserChain

The chain a replay prices through: point-in-time plus-minus first, then the live chain unchanged.

Order matters and is the whole design. `PointInTimeLineupRatings` defers on any engine without
`:plus_minus_ratings`, so a tracker-rating model still reaches `RatingsFromTracker` and is priced
exactly as the 8085 console prices it.
"""
replay_materialisers() = MD.MaterialiserChain(PointInTimeLineupRatings(),
                                              MD.RatingsFromTracker(), MD.LeagueFromFixture())

# ===================================================================
# 5. Loading one match day
# ===================================================================

"""
    ReplayCard

Everything about one historical match day that does not change as the scrubber moves.

`results` is the FULL-TIME score, and it is loaded here rather than at settlement on purpose: a
replay that had to reach the network to settle would be unrunnable offline, and a settlement that
could fail after execution would leave the ledger holding reserved liability with no way to
release it.
"""
struct ReplayCard
    day::Date
    fixtures::Vector{MD.Fixture}
    kickoff::DateTime
    identities::Dict{Int,Union{MD.Resolved,MD.Unresolved}}
    book::PreloadedBook
    lineups::PreloadedLineups
    results::Dict{Int,Tuple{Int,Int}}
    lineup_drop::Dict{Int,DateTime}
    book_span::Tuple{Union{Nothing,DateTime},Union{Nothing,DateTime}}
end

"Absolute instant for a clock reading, and its inverse. The clock stores only `t`."
as_of_at(card::ReplayCard, t::Integer) = card.kickoff + Minute(Int(t))

"""
    available_matchdays(conn; tournament_ids, from, to) -> DataFrame

Which historical days can actually be replayed, and how well.

A day is replayable when it has fixtures AND an order book; the two are reported separately
because their absence means different things. No fixtures is "nothing was on"; fixtures with no
book is "the collector was down", which on this database is the common failure and is the reason
`MatchDay`'s own docstring carries a health warning about it.
"""
function available_matchdays(conn; tournament_ids::Vector{Int} = [56, 57],
                             from::Date = Date(2026, 7, 25), to::Date = Date(2026, 9, 30))
    sql = """
    WITH fx AS (
        SELECT to_timestamp(start_timestamp)::date AS day, match_id, start_timestamp
        FROM sofascore.events
        WHERE tournament_id = ANY(\$1)
          AND start_timestamp >= EXTRACT(EPOCH FROM \$2::date)
          AND start_timestamp <  EXTRACT(EPOCH FROM \$3::date)
    ),
    ob AS (
        SELECT ts::date AS day, count(*) AS n_rows, min(ts) AS first_ts, max(ts) AS last_ts
        FROM betfair_live.order_book_1m GROUP BY 1
    ),
    lu AS (
        -- Filtered to the SAME tournaments as `fx`. Without it a Premiership scrape on the same
        -- Saturday is counted against a Scottish Lower card, and the console's "15 XI on a
        -- 10-fixture day" reads as a bug in the console rather than in the count.
        SELECT to_timestamp(e.start_timestamp)::date AS day,
               count(DISTINCT l.match_id) AS n_lineups
        FROM sofascore.lineup_provisional l
        JOIN sofascore.events e ON e.match_id = l.match_id
        WHERE e.tournament_id = ANY(\$1)
        GROUP BY 1
    )
    SELECT fx.day,
           count(*)                       AS n_fixtures,
           min(to_timestamp(fx.start_timestamp)) AS first_kickoff,
           COALESCE(max(ob.n_rows), 0)    AS book_rows,
           max(ob.first_ts)               AS book_from,
           max(ob.last_ts)                AS book_to,
           COALESCE(max(lu.n_lineups), 0) AS n_lineups
    FROM fx
    LEFT JOIN ob ON ob.day = fx.day
    LEFT JOIN lu ON lu.day = fx.day
    GROUP BY fx.day
    ORDER BY fx.day;
    """
    df = DataFrame(LibPQ.execute(conn, sql, (tournament_ids, from, to)))
    df.replayable = [r.book_rows > 0 && r.n_fixtures > 0 for r in eachrow(df)]
    return df
end

"""
    load_replay_card(conn, day; tournament_ids, identity, max_age) -> ReplayCard

Read one match day into memory: fixtures, identity, the whole order book, every lineup scrape,
and the final scores.

Refuses a day whose fixtures do not share a kick-off instant. `Portfolio` solves the drawdown
budget PER SETTLEMENT WINDOW, so a card spanning two kick-offs has one `k_risk` belonging to
neither -- the same refusal `price_slate` makes, moved forward to load time where it costs a
message instead of a stack trace at T-15.
"""
function load_replay_card(conn, day::Date;
                          tournament_ids::Vector{Int} = [56, 57],
                          identity::MD.AbstractIdentityResolver =
                              MD.ResolverChain(MD.MatchMetaCrosswalk(), MD.LiveNameMatch()),
                          max_age::Dates.Period = Hour(2))
    lo = Int(round(datetime2unix(DateTime(day))))
    hi = Int(round(datetime2unix(DateTime(day) + Day(1))))
    ev = DataFrame(LibPQ.execute(conn, """
        SELECT match_id, home_team, away_team, start_timestamp, tournament_id,
               raw_data->'homeScore'->>'current' AS home_goals,
               raw_data->'awayScore'->>'current' AS away_goals
        FROM sofascore.events
        WHERE tournament_id = ANY(\$1) AND start_timestamp >= \$2 AND start_timestamp < \$3
        ORDER BY start_timestamp;""", (tournament_ids, lo, hi)))
    nrow(ev) == 0 && error("load_replay_card: no fixtures on $day for tournaments " *
                           "$(tournament_ids).")

    fixtures = MD.Fixture[MD.Fixture(Int(r.match_id), String(r.home_team), String(r.away_team),
                                     unix2datetime(r.start_timestamp), Int(r.tournament_id))
                          for r in eachrow(ev)]
    kickoffs = unique(f.kickoff for f in fixtures)
    length(kickoffs) == 1 || error(
        "load_replay_card: $day has $(length(kickoffs)) distinct kick-off instants " *
        "($(join(string.(sort(kickoffs)), ", "))). One PricedSlate carries one `k_risk`, solved " *
        "for one settlement window, so a two-window card would carry a budget belonging to " *
        "neither. Replay each window separately.")

    results = Dict{Int,Tuple{Int,Int}}()
    for r in eachrow(ev)
        (ismissing(r.home_goals) || ismissing(r.away_goals)) && continue
        h = tryparse(Int, String(r.home_goals)); a = tryparse(Int, String(r.away_goals))
        (h === nothing || a === nothing) && continue
        results[Int(r.match_id)] = (h, a)
    end

    identities = Dict{Int,Union{MD.Resolved,MD.Unresolved}}()
    for f in fixtures
        identities[f.m_id] = MD.resolve(identity, f)
    end

    book = _preload_book(conn, fixtures, identities; max_age = max_age)
    lineups, drops = _preload_lineups(conn, fixtures)

    span = _book_span(book)
    return ReplayCard(day, fixtures, first(kickoffs), identities, book, lineups, results,
                      drops, span)
end

"""
    _preload_book(conn, fixtures, identities; max_age) -> PreloadedBook

One query for the whole match day's ladders, bucketed by fixture and selection.

The x10000 unscaling and the `betfair_to_key` structural mapping are the ones `ArchivedOrderBook`
performs, reached through the same functions rather than copied -- a second copy of the scaling
constant is how a replay ends up measuring a book in pence while the live path measures it in
pounds.
"""
function _preload_book(conn, fixtures::Vector{MD.Fixture},
                       identities::Dict{Int,Union{MD.Resolved,MD.Unresolved}};
                       max_age::Dates.Period = Hour(2))
    market_to_match = Dict{String,Int}()
    fixture_of = Dict{Int,MD.Fixture}(f.m_id => f for f in fixtures)
    for (mid, id) in identities
        id isa MD.Resolved || continue
        for market_id in values(id.market_ids)
            market_to_match[market_id] = mid
        end
    end

    ladders = Dict{Int,Dict{MD.SelectionKey,Vector{MD.BookLevels}}}()
    stamps  = Dict{Int,Dict{MD.SelectionKey,Vector{DateTime}}}()
    isempty(market_to_match) && return PreloadedBook(ladders, stamps, max_age)

    rows = DataFrame(LibPQ.execute(conn, """
        SELECT o.market_id, o.symbol, o.ts, o.bid_prices, o.bid_volumes,
               o.ask_prices, o.ask_volumes, o.market_matched, md.market_type
        FROM betfair_live.order_book_1m o
        JOIN betfair_live.market_metadata md USING (market_id)
        WHERE o.market_id = ANY(\$1)
        ORDER BY o.market_id, o.symbol, o.ts;""", (collect(keys(market_to_match)),)))

    for r in eachrow(rows)
        mid = market_to_match[String(r.market_id)]
        f = fixture_of[mid]
        mt, sym = String(r.market_type), String(r.symbol)
        key = something(MD.betfair_to_key(mt, sym),
                        mt == "MATCH_ODDS" ? MD.betfair_to_key_1x2(sym, f.home, f.away) : nothing,
                        Some(nothing))
        key === nothing && continue
        lv = MD.BookLevels(MD._unscale(r.bid_prices), MD._unscale(r.bid_volumes),
                           MD._unscale(r.ask_prices), MD._unscale(r.ask_volumes),
                           ismissing(r.market_matched) ? NaN :
                               Float64(r.market_matched) / 10_000,
                           DateTime(r.ts))
        per_sel = get!(ladders, mid, Dict{MD.SelectionKey,Vector{MD.BookLevels}}())
        per_ts  = get!(stamps,  mid, Dict{MD.SelectionKey,Vector{DateTime}}())
        push!(get!(per_sel, key, MD.BookLevels[]), lv)
        push!(get!(per_ts,  key, DateTime[]), lv.ts)
    end

    # The query is ordered by ts, but a market that changed `market_id` mid-day would interleave.
    # Sorting is cheap and it is what `searchsortedlast` is entitled to assume.
    for (mid, per_sel) in ladders
        per_ts = stamps[mid]
        for (key, levels) in per_sel
            perm = sortperm(per_ts[key])
            per_sel[key] = levels[perm]
            per_ts[key]  = per_ts[key][perm]
        end
    end
    return PreloadedBook(ladders, stamps, max_age)
end

"""
    _preload_lineups(conn, fixtures) -> (PreloadedLineups, Dict{Int,DateTime})

Every lineup scrape for the card, plus the instant each fixture's XI first became visible.

The drop instants are what the console draws as the lineup marker on the scrubber, and they are
read rather than assumed: `MatchDay`'s docstring records that the 2026-08-08/09 round scraped at
T-13 to T-42 with a median around T-29, which is close enough to the nominal T-30 marker to make
an assumed one look right while being wrong per fixture.
"""
function _preload_lineups(conn, fixtures::Vector{MD.Fixture})
    ids = [f.m_id for f in fixtures]
    df = DataFrame(LibPQ.execute(conn, """
        SELECT match_id, player_id, player_name, position, substitute, is_home_team,
               confirmed, scraped_at
        FROM sofascore.lineup_provisional
        WHERE match_id = ANY(\$1)
        ORDER BY match_id, scraped_at;""", (ids,)))

    rows = Dict{Int,DataFrame}()
    drops = Dict{Int,DateTime}()
    nrow(df) == 0 && return (PreloadedLineups(rows), drops)

    df.scraped_at = DateTime.(df.scraped_at)
    for g in DataFrames.groupby(df, :match_id)
        mid = Int(first(g.match_id))
        sub = DataFrame(g)
        sort!(sub, :scraped_at)
        rows[mid] = sub
        drops[mid] = minimum(sub.scraped_at)
    end
    return (PreloadedLineups(rows), drops)
end

function _book_span(b::PreloadedBook)
    lo = nothing; hi = nothing
    for (_, per_sel) in b.stamps, (_, ts) in per_sel
        isempty(ts) && continue
        lo = lo === nothing ? first(ts) : min(lo, first(ts))
        hi = hi === nothing ? last(ts)  : max(hi, last(ts))
    end
    return (lo, hi)
end

# ===================================================================
# 6. The model registry
# ===================================================================

"""
    ModelSlot

One canonical fit, its fold selection, its fold FeatureSet, and its memoised latents.

The three heavy objects are held because rebuilding them is what makes hot-swapping impossible
otherwise. `base_fs` in particular costs a full `Features.create_features` over every fold --
about a minute for the hybrid player pillar -- and it is INVARIANT under the replay clock: the
only thing a tick changes is the lineup that gets materialised INTO a copy of it.

`latents` is keyed on a hash of the point-in-time lineups. m00 and m05 read no lineup, so the
cache holds exactly one entry for them and the model bars are flat across the whole replay, which
is the correct and visible answer for a model with no lineup pillar. m12 holds two: pre-drop and
post-drop.
"""
mutable struct ModelSlot
    key::String
    label::String
    experiment::String
    run_name::String
    status::Symbol                      # :unloaded | :loading | :ready | :failed
    error::String
    fit::Any
    boundaries::Any
    fcol::Any
    fold_idx::Int
    fold_warning::String
    base_fs::Any
    chain::Any
    covered::Vector{Int}
    refused::Vector{Pair{Int,String}}
    latents::Dict{UInt64,DataFrame}
    load_seconds::Float64
end

ModelSlot(key, label, experiment, run_name) =
    ModelSlot(String(key), String(label), String(experiment), String(run_name),
              :unloaded, "", nothing, nothing, nothing, 0, "", nothing, nothing, Int[],
              Pair{Int,String}[], Dict{UInt64,DataFrame}(), 0.0)

"""
    default_model_registry() -> Vector{ModelSlot}

The three canonical pillars this console switches between, in increasing structural richness.

They are registered by `(experiment, run_name)` rather than by run id, because the id is a
sequence number in `mcmc_experiments.runs` and the name is the stable address the registry
protocol asks for. All three are `status = completed` runs; none of them is fitted here.
"""
default_model_registry() = ModelSlot[
    ModelSlot("m00", "m00 Poisson control", "scottish_lower_joint_2426",
              "m00_poisson_control"),
    ModelSlot("m05", "m05 Joint production-wealth", "scottish_lower_joint_2426",
              "m05_joint_production_wealth"),
    ModelSlot("m12", "m12 Hybrid wealth + player RAPM", "scottish_lower_player_grid_2426",
              "m12_hybrid_production_wealth_player_rapm"),
]

"""
    coverage_split(fs, fixtures) -> (covered, refused)

Partition the card into what this fold can actually represent and what it cannot.

`MD.check_coverage` throws on the first problem, which is right for a live slate -- a fixture
priced as league-average is worse than no price -- and wrong for a console whose job is to show
what each model can do. The three checks it makes are not equivalent, though, and only ONE of
them is a genuine refusal here:

* `team_map` -- **not materialisable.** A team the fold never saw has no `α`/`β` to condition on
  and would be priced at the league mean. That is the refusal.
* `player_ratings_map`, `league_lookup` -- materialised per fixture a few lines later, so a
  fixture absent from them at this point is not yet a problem.

`check_coverage` is still called after materialisation, as the assertion. This function only
decides which fixtures to hand it.
"""
function coverage_split(fs, fixtures::Vector{MD.Fixture})
    covered = Int[]; refused = Pair{Int,String}[]
    tm = get(fs.data, :team_map, nothing)
    for f in fixtures
        if tm === nothing
            push!(covered, f.m_id); continue
        end
        missing_teams = String[]
        haskey(tm, f.home) || push!(missing_teams, f.home)
        haskey(tm, f.away) || push!(missing_teams, f.away)
        if isempty(missing_teams)
            push!(covered, f.m_id)
        else
            push!(refused, f.m_id => "absent from this fold's team_map (" *
                                     join(missing_teams, ", ") *
                                     ") -- would be priced at the league mean")
        end
    end
    return (covered, refused)
end

"""
    load_slot!(slot, ds, card) -> ModelSlot

Bring one model to `:ready`: load the chain, choose the fold, build the fold FeatureSet, and
record which fixtures it can price.

Never throws. A model that cannot be loaded goes to `:failed` with the reason attached, because a
console whose model dropdown silently does nothing is the failure mode the live console's
`_no_executor` exists to prevent, one layer up.
"""
function load_slot!(slot::ModelSlot, ds, card::ReplayCard)
    slot.status === :ready && return slot
    slot.status = :loading
    t0 = time()
    try
        fit = MD.canonical_fit(TT.PostgresStorage(slot.experiment), slot.run_name)
        slot.fit = fit
        slot.boundaries = DD.create_id_boundaries(ds, fit.config.splitter)
        slot.fcol = FE.create_features(slot.boundaries, ds, fit.config.model,
                                       fit.config.splitter)
        rebind_slot!(slot, ds, card)
        slot.status = :ready
        slot.error = ""
    catch e
        slot.status = :failed
        slot.error = sprint(showerror, e)
    finally
        slot.load_seconds = time() - t0
    end
    return slot
end

"""
    rebind_slot!(slot, ds, card) -> ModelSlot

Re-choose the fold and re-measure coverage for a DIFFERENT match day, reusing the loaded chain
and the built feature collection.

Switching the match day changes which fold is correct -- `select_split` identifies it by asking
which fold's next observed round IS this card -- but it changes neither the boundaries nor the
features, both of which are functions of `(ds, model, splitter)` alone. Rebuilding them would
cost the hybrid pillar another minute to arrive at an identical object, so the expensive half is
kept and only the selection is redone.

The latents cache is emptied, and must be: it is keyed on the lineup signature, and two match
days can produce the same signature (notably the all-`nothing` one before any XI drops) while
meaning entirely different fixtures.
"""
function rebind_slot!(slot::ModelSlot, ds, card::ReplayCard)
    ids = [f.m_id for f in card.fixtures]
    sel = MD.select_split(slot.fit, slot.boundaries; exclude = ids, ds = ds,
                          config = slot.fit.config.splitter, fixture_ids = ids)
    slot.fold_idx = sel.idx
    slot.fold_warning = sel.warning
    slot.base_fs = slot.fcol[sel.idx][1]
    slot.chain = sel.chain
    covered, refused = coverage_split(slot.base_fs, card.fixtures)
    slot.covered = covered
    slot.refused = refused
    empty!(slot.latents)
    return slot
end

"""
    lineup_signature(cards) -> UInt64

A hash of every fixture's point-in-time XI, and the memoisation key for the latents.

Includes `substitute` per player rather than only the id set, because the bench-weighted
aggregation multiplies substitutes by `w_bench`: moving a player from the XI to the bench changes
the pillar's value without changing which players are named, and a signature blind to that would
serve a stale posterior across exactly the transition this console exists to render.
"""
function lineup_signature(cards::Vector{<:MD.FixtureCard})
    h = UInt64(0x9e3779b97f4a7c15)
    for c in sort(cards; by = c -> c.fixture.m_id)
        h = hash(c.fixture.m_id, h)
        lu = c.lineup
        if lu === nothing
            h = hash(:no_lineup, h)
            continue
        end
        h = hash((lu.source, lu.confirmed), h)
        for players in (lu.home, lu.away)
            h = hash(sort([(p.player_id, p.substitute, p.position) for p in players]), h)
        end
    end
    return h
end

"""
    slot_latents(slot, spec, ds, cards, odds, as_of) -> DataFrame

Posterior latents for the card, memoised on the lineup.

Mirrors `MD.matchday_latents` stage for stage -- materialise, assert coverage, build the fixture
frame, extract -- and differs in exactly two places, both of which are the reason it exists:

1. the fold FeatureSet and the fold choice come from the slot, so `Features.create_features` runs
   once per MODEL rather than once per tick;
2. `REPLAY_INJECTABLE_KEYS` adds `:player_lineup_ratings_map`, without which the hybrid pillar
   would read the played teamsheet at every instant of the replay.

`deepcopy` of the fold FeatureSet is not optional and is the same guard `matchday_latents` uses:
materialising into `slot.base_fs` would make the SECOND tick see the first tick's lineup.
"""
function slot_latents(slot::ModelSlot, spec::MD.MatchDaySpec, ds,
                      cards::Vector{<:MD.FixtureCard}, odds::DataFrame, as_of::DateTime)
    isempty(cards) && return DataFrame()
    sig = lineup_signature(cards)
    cached = get(slot.latents, sig, nothing)
    cached === nothing || return cached

    model = slot.fit.config.model
    fx = MD.Fixture[c.fixture for c in cards]
    fs = deepcopy(slot.base_fs)
    lineups = Dict(c.fixture.m_id => c.lineup for c in cards if c.lineup !== nothing)
    ctx = (ds = ds, model = model, as_of = as_of, odds = odds, lineups = lineups)

    for key in REPLAY_INJECTABLE_KEYS
        haskey(fs.data, key) || continue
        MD.materialise!(spec.features, Val(key), fs, fx, ctx) || error(
            "replay: no materialiser in $(typeof(spec.features)) handles :$key, which " *
            "$(typeof(model)) reads per match_id. Carrying the trained map forward would " *
            "price every fixture off that feature's fallback -- or, for " *
            ":player_lineup_ratings_map, off the teamsheet that actually took the field.")
    end
    MD.check_coverage(fs, fx, model)

    frame = DataFrame(match_id = [f.m_id for f in fx],
                      home_team = [f.home for f in fx],
                      away_team = [f.away for f in fx],
                      match_date = [Date(f.kickoff) for f in fx],
                      month_idx = [Dates.month(f.kickoff) for f in fx],
                      match_week = fill(999, length(fx)))
    raw = MO.PreGame.extract_parameters(model, frame, fs, slot.chain)
    df = MD._raw_to_df(raw)
    slot.latents[sig] = df
    return df
end

# ===================================================================
# 7. The replay state
# ===================================================================

"""
    ReplayState

The whole console, behind one lock.

Everything mutable a tick touches lives here, and every reader takes `state.lock`. That is
heavier than the live console needs -- it holds one immutable `PricedSlate` and re-reads the
account -- and it is what a replay needs, because the auto-play task, the HTTP handlers and the
WebSocket pusher all mutate and read the same clock.
"""
mutable struct ReplayState
    ds::Any
    conn::Any
    card::ReplayCard
    clock::ReplayClock
    system::PF.PortfolioSystem
    models::Vector{ModelSlot}
    active::String
    bankroll::Float64
    account_id::String
    schema::String
    slate::Union{Nothing,MD.PricedSlate}
    slate_t::Int
    tick_note::String
    tick_error::String
    executed::Vector{UUID}
    settlement::Any
    equity_before::Float64
    lock::ReentrantLock
    player::Any
    running::Bool
    reprice_seconds::Float64
    tick_seq::Int
end

function ReplayState(ds, conn, card::ReplayCard; system::PF.PortfolioSystem,
                     models::Vector{ModelSlot} = default_model_registry(),
                     active::AbstractString = "m00", bankroll::Real = 2_400.0,
                     account_id::AbstractString = "replay_scottish",
                     schema::AbstractString = REPLAY_SCHEMA)
    assert_replay_schema(schema)
    return ReplayState(ds, conn, card, ReplayClock(), system, models, String(active),
                       Float64(bankroll), String(account_id), String(schema), nothing,
                       T_START, "", "", UUID[], nothing, Float64(bankroll),
                       ReentrantLock(), nothing, false, 0.0, 0)
end

active_slot(st::ReplayState) =
    st.models[something(findfirst(m -> m.key == st.active, st.models), 1)]

function find_slot(st::ReplayState, key::AbstractString)
    # `something(findfirst(...), error(...))` would be wrong here and silently so: `something`
    # evaluates every argument before deciding, so the error fires on the successful lookup too.
    i = findfirst(m -> m.key == String(key), st.models)
    i === nothing && error("replay: no model '$key' is registered. Registered: " *
                           join([m.key for m in st.models], ", "))
    return st.models[i]
end

"""
    replay_spec(st, fixtures) -> MD.MatchDaySpec

The pipeline configuration for one tick.

Every seam that reads a clock or a network is the preloaded one; every seam that makes a DECISION
-- instrument choice, rounding, gates, market set -- is the live console's, unchanged. That split
is what makes the replay evidence about the live path: if the gates differed, a replay that
passed would say nothing about a Saturday that did not.

`MaxBookAge(Minute(10))` is kept as the live console runs it. It is the gate that goes on to
refuse the card once the collector stops, which on some of these match days happens before T+105
-- and seeing that happen is more useful than a replay that quietly widens the bound.
"""
replay_spec(st::ReplayState, fixtures::Vector{MD.Fixture}) = MD.MatchDaySpec(
    fixtures = ReplayFixtures(fixtures),
    identity = FrozenIdentity(st.card.identities),
    lineups = st.card.lineups,
    book = st.card.book,
    instrument = MD.BestOfBackLay(),
    rounding = MD.FloorOrDrop(minimum = 1.0),
    features = replay_materialisers(),
    gate = MD.GateChain(MD.IdentityResolved(), MD.MaxBookAge(Minute(10)),
                        MD.MaxSpread(0.08), MD.MinMatched(minimum = 20.0)),
    markets = MD.canonical_markets())

# ===================================================================
# 8. Pricing one tick
# ===================================================================

"""
    reprice!(st) -> Union{Nothing,MD.PricedSlate}

Price the card at the current clock instant with the active model.

The stage order is the pipeline's, not a convenience order:

    fixtures -> identity -> lineups -> BOOK -> features -> inference -> gate -> stake_sheet

The book is built before features because market-pillar engines consume odds as a feature, so
inference depends on the same prices staking does. This is stated in `MD.pipeline` and repeated
here because a replay is exactly where someone would be tempted to hoist the book read out of the
loop for speed.

A tick that prices nothing does NOT clear the last slate. Post-kickoff the gates legitimately
refuse the card -- the book goes stale when the collector stops -- and blanking the grid at that
moment would destroy the state the operator is trying to settle. `slate_t` records which minute
the visible slate was priced at, so a frozen grid is labelled rather than mistaken for a live one.
"""
function reprice!(st::ReplayState)
    t0 = time()
    st.tick_error = ""
    slot = active_slot(st)
    if slot.status !== :ready
        st.tick_note = "model $(slot.key) is $(slot.status)" *
                       (isempty(slot.error) ? "" : ": " * slot.error)
        return st.slate
    end

    as_of = as_of_at(st.card, st.clock.t)
    covered = Set(slot.covered)
    fx = MD.Fixture[f for f in st.card.fixtures if f.m_id in covered]
    if isempty(fx)
        st.tick_note = "model $(slot.key) covers none of the $(length(st.card.fixtures)) fixtures"
        return st.slate
    end

    spec = replay_spec(st, fx)
    try
        cards = MD.build_cards(spec, DD.ScottishLower(), as_of)
        q = MD.quote_slate(spec, cards, as_of)
        for c in cards
            c.readiness = MD.ready(spec.gate, c)
        end
        passed  = MD.FixtureCard[c for c in cards if MD.is_ready(c.readiness)]
        blocked = MD.FixtureCard[c for c in cards if !MD.is_ready(c.readiness)]

        if isempty(passed)
            st.tick_note = "every fixture is gated at T$(_signed(st.clock.t))m " *
                           "($(length(blocked)) blocked)"
            st.reprice_seconds = time() - t0
            return st.slate
        end

        latents = slot_latents(slot, spec, st.ds, passed, q.odds, as_of)
        if isempty(latents)
            st.tick_note = "no latents extracted at T$(_signed(st.clock.t))m"
            st.reprice_seconds = time() - t0
            return st.slate
        end

        sheet = PF.stake_sheet(st.system, latents, slot.fit, q.odds, MD.fixture_info(passed);
                               bankroll = st.bankroll)
        if isempty(sheet)
            st.tick_note = "staking produced no leg at T$(_signed(st.clock.t))m"
            st.reprice_seconds = time() - t0
            return st.slate
        end
        MD._attach_instruments!(sheet, q.instruments, spec.rounding)
        if isempty(sheet)
            st.tick_note = "every leg fell below the exchange minimum at " *
                           "T$(_signed(st.clock.t))m"
            st.reprice_seconds = time() - t0
            return st.slate
        end
        MD.annotate_capacity!(sheet, q.books)

        window = minimum(Date(c.fixture.kickoff) for c in cards)
        slate = MD.PricedSlate(uuid4(), st.account_id, window, as_of, st.bankroll, sheet,
                               q.odds, cards, blocked, q.instruments, q.books,
                               Float64(first(sheet.k_risk)),
                               Float64(first(sheet.slate_exposure)),
                               Bool(first(sheet.capped)), _policy_lambda(st.system),
                               _policy_cap(st.system), sum(Float64, sheet.risk),
                               slot.fold_idx, slot.fold_warning)
        st.slate = slate
        st.slate_t = st.clock.t
        st.tick_note = ""
        st.tick_seq += 1
    catch e
        st.tick_error = sprint(showerror, e)
    finally
        st.reprice_seconds = time() - t0
    end
    return st.slate
end

_policy_cap(sys::PF.PortfolioSystem) =
    hasproperty(sys.policy.cap, :cap) ? Float64(sys.policy.cap.cap) : NaN
_policy_lambda(sys::PF.PortfolioSystem) =
    hasproperty(sys.policy.risk, :lambda) ? Float64(sys.policy.risk.lambda) : NaN

_signed(t::Integer) = t < 0 ? string(t) : "+" * string(t)

# ===================================================================
# 9. Clock transitions
# ===================================================================

"""
    seek!(st, t) / step!(st, dt) / jump!(st, marker)

Move the clock and reprice. Every one of them clamps to `[T_START, T_END]` rather than erroring:
a scrubber dragged past the end of the window is an ordinary gesture, not a fault.
"""
function seek!(st::ReplayState, t::Integer)
    lock(st.lock) do
        st.clock.t = clamp_t(t)
        reprice!(st)
    end
    return st.clock.t
end

step!(st::ReplayState, dt::Integer = 1) = seek!(st, st.clock.t + Int(dt))

"Named scrubber targets. The four instants an operator actually navigates between."
const JUMP_TARGETS = Dict("lineups" => T_LINEUP, "exec" => T_EXEC,
                          "kickoff" => T_KICKOFF, "settlement" => T_END,
                          "start" => T_START)

function jump!(st::ReplayState, marker::AbstractString)
    t = get(JUMP_TARGETS, String(marker), nothing)
    t === nothing && error("replay: unknown jump target '$marker'. Known: " *
                           join(sort(collect(keys(JUMP_TARGETS))), ", "))
    return seek!(st, t)
end

function set_speed!(st::ReplayState, speed::Real)
    s = Float64(speed)
    s > 0 || error("replay: speed must be positive, got $s")
    lock(st.lock) do; st.clock.speed = s; end
    return s
end

"""
    play!(st) / pause!(st)

Start and stop the auto-advance task.

One simulated minute takes `60 / speed` WALL seconds INCLUDING the re-pricing that minute costs,
so 60x is genuinely one minute per second rather than one-plus-however-long-the-model-took.
Re-pricing runs 0.4-0.5 s per tick on the team-level pillars, so sleeping a flat `60/speed`
after it would make 60x mean about 0.7 simulated minutes per second and quietly mislabel every
speed on the dial. The elapsed time is measured and subtracted; when a tick costs more than its
budget the loop simply does not sleep, and the console's `reprice_ms` says why.

`speed` is re-read every iteration, so changing speed mid-play takes effect on the next tick
rather than requiring a restart.

It stops itself at `T_END`. A replay that ran off the end of its own window would be advancing a
clock with no book behind it.
"""
function play!(st::ReplayState)
    lock(st.lock) do
        st.clock.playing && return nothing
        st.clock.playing = true
        st.player = Threads.@spawn _play_loop(st)
        return nothing
    end
    return st.clock.playing
end

function _play_loop(st::ReplayState)
    while true
        playing, speed, t = lock(st.lock) do
            (st.clock.playing, st.clock.speed, st.clock.t)
        end
        (playing && st.running) || break
        if t >= T_END
            lock(st.lock) do; st.clock.playing = false; end
            break
        end
        started = time()
        lock(st.lock) do
            st.clock.playing || return nothing
            st.clock.t = clamp_t(st.clock.t + 1)
            reprice!(st)
            return nothing
        end
        sleep(max(0.02, 60.0 / speed - (time() - started)))
    end
    return nothing
end

function pause!(st::ReplayState)
    lock(st.lock) do; st.clock.playing = false; end
    st.player = nothing
    return false
end

"""
    set_model!(st, key) -> ModelSlot

Hot-swap the active model and re-price the card at the CURRENT instant, in the running process.

Loading is lazy and happens here, so the first switch to the hybrid pillar costs its
`Features.create_features` once and every switch after that is a dictionary lookup plus an
`extract_parameters`. The clock does not move: the operator is asking "what would THIS model have
said at this minute", and moving the instant would answer a different question.
"""
function set_model!(st::ReplayState, key::AbstractString)
    slot = find_slot(st, key)
    slot.status === :ready || load_slot!(slot, st.ds, st.card)
    lock(st.lock) do
        st.active = slot.key
        reprice!(st)
    end
    return slot
end

"""
    set_matchday!(st, day) -> ReplayCard

Load a different historical Saturday into the running process and re-price at T-60m.

Every already-loaded model is rebound rather than reloaded (see `rebind_slot!`), so the second
match day costs a fold re-selection instead of another `Features.create_features` per model.

The clock is reset to `T_START` and NOT left where it was. Minutes are relative to kick-off, so a
clock at T+40 would be meaningful on the new card -- and that is exactly the trap: the operator
would be looking at a different match day's second half without having asked to.

Executed slates are forgotten by the SESSION but not deleted from the ledger. They belong to a
different `slate_window`, so settling the new day cannot touch them; `reset_replay_ledger!` is
the deliberate way to remove them.
"""
function set_matchday!(st::ReplayState, day::Date; tournament_ids::Vector{Int} = [56, 57])
    card = load_replay_card(st.conn, day; tournament_ids = tournament_ids)
    lock(st.lock) do
        pause!(st)
        st.card = card
        st.clock.t = T_START
        st.slate = nothing
        st.slate_t = T_START
        st.settlement = nothing
        empty!(st.executed)
        for slot in st.models
            slot.status === :ready || continue
            try
                rebind_slot!(slot, st.ds, card)
            catch e
                slot.status = :failed
                slot.error = sprint(showerror, e)
            end
        end
        reprice!(st)
        return nothing
    end
    return card
end

# ===================================================================
# 10. Execution and settlement, in `paper_replay` only
# ===================================================================

"""
    ensure_replay_account!(st) -> MD.PaperAccount

Build the schema if absent and open the replay account.

`migrate_paper_schema!` is idempotent by construction and is called on every start, which is the
intended way to run it: a migration that has to be remembered is a migration that will be
forgotten on the one Saturday it mattered.
"""
function ensure_replay_account!(st::ReplayState)
    assert_replay_schema(st.schema)
    MD.migrate_paper_schema!(st.conn; schema = st.schema)
    return MD.ensure_account!(st.conn,
        MD.PaperAccount(account_id = st.account_id, opening_balance = st.bankroll,
                        balance = st.bankroll, max_slate_exposure = 0.25);
        schema = st.schema)
end

"""
    execute!(st) -> NamedTuple

Reserve the visible stake vector and simulate its fills against the book AT THIS MINUTE.

Three transactions, in the order the ledger requires and for the reasons it gives:

1. `insert_slate!` + `insert_orders!` -- the batch header and its legs, both idempotent. The
   header is unique on `(account, window, as_of)`, so pressing Execute twice at the same minute
   is a no-op rather than a second position, while executing again a minute later is a genuinely
   different slate.
2. `execute_slate_batch!` -- the atom. One `SELECT ... FOR UPDATE` on the account row, the whole
   vector or nothing. A stake vector solved for 8 fixtures is not valid for 6 of them.
3. `submit_slate!` with `LadderSweep()` -- fills against `slate.books`, which is the depth the
   prices were collapsed from at this instant, so the fill and the price cannot come from
   different reads.

`LadderSweep` rather than the `TouchOnly` default is deliberate and is the work package's ask.
It is the OPTIMISTIC of the two realistic models -- it assumes we cross up to three archived
levels instantly, which is what a market order does and not what the live system does -- so a
replay P&L built on it is an upper bound on the resting-order path. `fill_model` is recorded per
fill row, so the two are never pooled by accident.

Refuses after kick-off unless `allow_in_play = true`. Past T-0 the posterior is pre-game and the
book is in-play -- the book has seen goals the model has not -- so the sheet's edges are a
measurement of that gap rather than a signal, and on a 1-0 they reach four figures. Executing
them would book a P&L no live path could have taken. The override exists because "what would
betting the in-play divergence have done" is a real research question; it just must be asked out
loud.
"""
function execute!(st::ReplayState; allow_in_play::Bool = false)
    assert_replay_schema(st.schema)
    return lock(st.lock) do
        slate = st.slate
        slate === nothing && return (ok = false, error = "nothing is priced yet")
        MD.n_legs(slate) > 0 || return (ok = false, error = "the priced slate has no legs")
        (st.slate_t < T_KICKOFF || allow_in_play) || return (ok = false, error =
            "refusing to execute at T$(_signed(st.slate_t))m: the slate was priced by a " *
            "PRE-GAME posterior against an IN-PLAY book, so its edges measure the gap between " *
            "the two rather than a tradeable price. Pass allow_in_play = true (or " *
            "`{\"allow_in_play\": true}`) if that divergence is what you are studying.")

        ensure_replay_account!(st)
        slate_id = MD.insert_slate!(st.conn, slate; schema = st.schema,
                                    run_name = active_slot(st).run_name)
        orders = MD.orders_to_paper(slate; slate_id = slate_id)
        MD.insert_orders!(st.conn, orders; schema = st.schema)

        at = slate.as_of
        res = MD.execute_slate_batch!(st.conn, st.account_id, slate_id;
                                      schema = st.schema, at = at)
        if res.status !== MD.RESERVED
            return (ok = false, error = "batch $(res.status): $(res.reason)")
        end
        fills = MD.submit_slate!(st.conn, slate_id, slate.books, MD.LadderSweep();
                                 schema = st.schema, at = at)
        slate_id in st.executed || push!(st.executed, slate_id)
        st.equity_before = MD.equity(MD.account_row(st.conn, st.account_id; schema = st.schema))

        return (ok = true, slate_id = string(slate_id), reserved = res.reserved,
                n_admitted = res.n_admitted, n_refused = res.n_refused,
                n_matched = fills.n_matched, n_partial = fills.n_partial,
                n_unfilled = fills.n_unfilled, risk_filled = fills.risk_filled,
                note = "reserved $(res.n_admitted) legs, matched $(fills.n_matched), " *
                       "partial $(fills.n_partial), £$(round(fills.risk_filled, digits = 2)) " *
                       "filled at T$(_signed(st.slate_t))m")
    end
end

"""
    closing_probabilities(st) -> Dict{Tuple{Int,MD.SelectionKey},Float64}

The DE-VIGGED closing probability of every runner, read off the book at T-0.

De-vigging is not optional and it is not cosmetic. `clv_for_order`'s docstring says why: the
book's raw `1/best_back` sums above one, so an un-normalised close would make EVERY leg look like
it beat the market, uniformly. Normalising within `(match_id, group, line)` removes the
overround, which is the same proportional method `Portfolio`'s `DeArb` applies to the entry side
-- entry and close must be measured on the same scale or their difference is not a value.

The mid is used rather than the back price because the close is a MEASUREMENT, not a trade: there
is nothing to execute at T-0 and taking the bid would charge the position half a spread it never
paid.
"""
function closing_probabilities(st::ReplayState)
    close_ts = as_of_at(st.card, T_KICKOFF)
    raw = Dict{Tuple{Int,MD.SelectionKey},Float64}()
    groups = Dict{Tuple{Int,String,Float64},Vector{MD.SelectionKey}}()
    for f in st.card.fixtures
        id = st.card.identities[f.m_id]
        id isa MD.Resolved || continue
        book = MD.quotes(st.card.book, id, close_ts)
        for (key, lv) in book
            bb, bl = MD.best_back(lv), MD.best_lay(lv)
            p = if !isnan(bb) && !isnan(bl)
                2.0 / (bb + bl)
            elseif !isnan(bb)
                1.0 / bb
            elseif !isnan(bl)
                1.0 / bl
            else
                continue
            end
            raw[(f.m_id, key)] = p
            push!(get!(groups, (f.m_id, key.group, key.line), MD.SelectionKey[]), key)
        end
    end

    out = Dict{Tuple{Int,MD.SelectionKey},Float64}()
    for ((mid, _, _), keys) in groups
        total = sum(raw[(mid, k)] for k in keys)
        total > 0 || continue
        # A market missing a runner cannot be de-vigged: its remaining probabilities do not sum
        # to a book. Reported as-is rather than normalised to 1, which would inflate them.
        n_expected = keys[1].group == "1X2" ? 3 : 2
        length(keys) == n_expected || continue
        for k in keys
            out[(mid, k)] = raw[(mid, k)] / total
        end
    end
    return out
end

"""
    settle!(st) -> NamedTuple

Grade every filled leg against the actual full-time score, book the P&L, and measure CLV.

Settlement is the only transition that moves money in a direction the reservation did not
authorise, and it runs once per executed slate. `settle_slate!` is idempotent on
`paper_settlements (order_id)`, so a second press re-reads rather than double-books.

CLV is written afterwards and separately, because the closing price does not exist when the bet
is placed -- which is the whole reason it is the metric with power. A leg whose runner has no
complete closing book is left out of the CLV table rather than given a made-up close; the
settlement row still exists, so P&L and CLV coverage are allowed to differ and the payload
reports both counts.
"""
function settle!(st::ReplayState)
    assert_replay_schema(st.schema)
    return lock(st.lock) do
        isempty(st.executed) && return (ok = false, error = "nothing has been executed yet")
        isempty(st.card.results) &&
            return (ok = false, error = "no full-time scores for $(st.card.day)")

        account_before = MD.account_row(st.conn, st.account_id; schema = st.schema)
        close_probs = closing_probabilities(st)
        close_ts = as_of_at(st.card, T_KICKOFF)
        at = as_of_at(st.card, T_END)

        legs = NamedTuple[]
        n_settled = 0; total_pnl = 0.0; n_clv = 0
        for slate_id in st.executed
            out = MD.settle_slate!(st.conn, slate_id, st.card.results;
                                   schema = st.schema, at = at)
            n_settled += out.n_settled
            total_pnl += out.total_pnl
            append!(legs, _settled_legs(st, slate_id, close_probs, close_ts))
        end
        n_clv = count(l -> l.clv_pp !== nothing, legs)

        account_after = MD.account_row(st.conn, st.account_id; schema = st.schema)
        matched_risk = sum(Float64[l.risk_filled for l in legs]; init = 0.0)
        gross = sum(Float64[l.gross_return for l in legs]; init = 0.0)
        net = sum(Float64[l.net_pnl for l in legs]; init = 0.0)
        beat = count(l -> l.beat_close === true, legs)

        st.settlement = (
            at = string(at),
            day = string(st.card.day),
            model = active_slot(st).run_name,
            n_slates = length(st.executed),
            n_settled = n_settled,
            n_legs = length(legs),
            matched_risk = round(matched_risk, digits = 2),
            gross_return = round(gross, digits = 2),
            net_pnl = round(net, digits = 2),
            roi_pct = matched_risk > 0 ? round(100 * net / matched_risk, digits = 2) : 0.0,
            n_clv = n_clv,
            beat_close = beat,
            beat_close_pct = n_clv > 0 ? round(100 * beat / n_clv, digits = 1) : 0.0,
            equity_before = round(MD.equity(account_before), digits = 2),
            equity_after = round(MD.equity(account_after), digits = 2),
            balance_before = round(account_before.balance, digits = 2),
            balance_after = round(account_after.balance, digits = 2),
            scores = [(match_id = f.m_id, fixture = f.home * " v " * f.away,
                       home_goals = get(st.card.results, f.m_id, (nothing, nothing))[1],
                       away_goals = get(st.card.results, f.m_id, (nothing, nothing))[2])
                      for f in st.card.fixtures],
            legs = legs,
            reconciled = MD.reconcile_account(st.conn, st.account_id; schema = st.schema).ok,
        )
        return (ok = true, note = "settled $(n_settled) legs, net " *
                                  "£$(round(net, digits = 2))", settlement = st.settlement)
    end
end

"""
    _settled_legs(st, slate_id, close_probs, close_ts) -> Vector{NamedTuple}

One row per settled leg, joining what the order was, what filled, how it graded and what the
close said. Also writes the `clv_audit` row, which is the durable half of the same join.
"""
function _settled_legs(st::ReplayState, slate_id::UUID,
                       close_probs::Dict{Tuple{Int,MD.SelectionKey},Float64},
                       close_ts::DateTime)
    orders = MD.slate_orders(st.conn, slate_id; schema = st.schema)
    fills_df = MD.fill_rows(st.conn, slate_id; schema = st.schema)
    settle_df = DataFrame(LibPQ.execute(st.conn,
        """SELECT s.* FROM $(st.schema).paper_settlements s
           JOIN $(st.schema).paper_orders o USING (order_id)
           WHERE o.slate_id = \$1;""", (string(slate_id),)))
    by_order = Dict(String(r.order_id) => r for r in eachrow(settle_df))

    fixture_of = Dict(f.m_id => f for f in st.card.fixtures)
    out = NamedTuple[]
    for o in orders
        row = get(by_order, string(o.order_id), nothing)
        row === nothing && continue
        sub = filter(r -> String(r.order_id) == string(o.order_id), fills_df)
        fills = MD.Fill[MD.Fill(order_id = o.order_id, filled_at = DateTime(r.filled_at),
                                price = Float64(r.price), size = Float64(r.size),
                                risk_filled = Float64(r.risk_filled),
                                model = Symbol(r.fill_model), levels_used = Int(r.level_depth))
                        for r in eachrow(sub)]

        key = (group = o.market_group, line = o.market_line, selection = o.selection)
        cp = get(close_probs, (o.match_id, key), nothing)
        clv_pp = nothing; beat = nothing; close_prob = nothing
        if cp !== nothing && !isempty(fills)
            c = MD.clv_for_order(o, fills, cp, close_ts)
            clv_pp = round(100 * c.clv, digits = 3)
            beat = c.beat_close
            close_prob = round(c.close_prob, digits = 4)
            LibPQ.execute(st.conn, """
                INSERT INTO $(st.schema).clv_audit
                    (order_id, entry_prob, close_prob, close_ts, close_source, clv, clv_pct,
                     beat_close, entry_lead_min)
                VALUES (\$1,\$2,\$3,\$4,\$5,\$6,\$7,\$8,\$9)
                ON CONFLICT (order_id) DO NOTHING;""",
                (string(o.order_id), c.entry_prob, c.close_prob, c.close_ts, c.close_source,
                 c.clv, c.clv_pct, c.beat_close, c.entry_lead_min))
        end

        f = get(fixture_of, o.match_id, nothing)
        score = get(st.card.results, o.match_id, nothing)
        push!(out, (
            match_id = o.match_id,
            fixture = f === nothing ? string(o.match_id) : f.home * " v " * f.away,
            home_goals = score === nothing ? nothing : score[1],
            away_goals = score === nothing ? nothing : score[2],
            market = o.market_group, line = o.market_line,
            selection = String(o.selection), venue_selection = String(o.venue_selection),
            side = String(o.side),
            venue_odds = round(o.venue_odds, digits = 3),
            effective_odds = round(o.effective_odds, digits = 3),
            fill_vwap = isempty(fills) ? nothing : round(MD.fill_vwap(fills), digits = 3),
            venue_stake = round(MD.filled_size(fills), digits = 2),
            risk_filled = round(MD.filled_risk(fills), digits = 2),
            outcome = String(row.outcome),
            gross_return = round(Float64(row.gross_return), digits = 2),
            commission = round(Float64(row.commission), digits = 2),
            net_pnl = round(Float64(row.net_pnl), digits = 2),
            close_prob = close_prob, clv_pp = clv_pp, beat_close = beat,
        ))
    end
    return out
end

"""
    reset_replay_ledger!(st)

Delete this replay account's rows and start the bankroll again.

Confined to `paper_replay` by `assert_replay_schema` and to ONE account by every `WHERE` below,
because a replay is meant to be re-run and an operator who cannot reset without dropping a schema
will eventually reset by dropping the wrong one.
"""
function reset_replay_ledger!(st::ReplayState)
    assert_replay_schema(st.schema)
    lock(st.lock) do
        s, a = st.schema, st.account_id
        LibPQ.execute(st.conn, "DELETE FROM $s.account_ledger WHERE account_id = \$1;", (a,))
        LibPQ.execute(st.conn, "DELETE FROM $s.paper_orders WHERE account_id = \$1;", (a,))
        LibPQ.execute(st.conn, "DELETE FROM $s.paper_slates WHERE account_id = \$1;", (a,))
        LibPQ.execute(st.conn, """
            UPDATE $s.paper_accounts SET balance = opening_balance, reserved = 0,
                   updated_at = now() WHERE account_id = \$1;""", (a,))
        empty!(st.executed)
        st.settlement = nothing
        return nothing
    end
    return MD.account_row(st.conn, st.account_id; schema = st.schema)
end
