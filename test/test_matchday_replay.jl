# test/test_matchday_replay.jl
#
# The match-day REPLAY engine: the clock, the point-in-time sources, model hot-swapping,
# execution into `paper_replay`, and settlement against the historical score.
#
# Structure follows `test/test_matchday_live_pipeline.jl`: every testset pins a CLAIM the design
# makes or a DEFECT it exists to prevent, so a regression surfaces as a named property rather
# than as a number that moved.
#
# THREE TIERS, and they fail for three different reasons.
#
#   R1-R14,  PURE. A synthetic match day built in memory: hand-made ladders, a hand-made XI, a
#   R24-R29  hand-made stake sheet. No database, no DataStore, no trained fit. These are the
#            tests that must never be skipped, because they cover the filtration contract -- if
#            `PreloadedBook` can see past `as_of`, every number the console shows is fiction.
#            R24-R29 are the ladder desk: ticks, weight of money, the three-level book, the
#            order marker and one runner's history. They are numbered AFTER the console-surface
#            and ledger tiers rather than inserted among them so that no existing test changes
#            name -- a report saying "R17 failed" must go on meaning the ledger test it always
#            meant.
#
#   R15-R16, CONSOLE SURFACE. HTTP and WebSocket against a real socket on a random high port,
#   R30      with no database behind it. R30 covers the two desk endpoints.
#
#   R17-R20  LEDGER. PostgreSQL, `paper_replay` ONLY. Skipped with a message when `BF_DB_URL` is
#            unset or unreachable. R18 asserts the isolation claim directly by counting
#            `paper_runbook` rows either side of a full execute-and-settle.
#
#   R21-R23, MODELS. A real Saturday, real canonical fits from `mcmc_experiments`. Slow (the
#   R31-R32  hybrid player pillar costs about a minute to build its features) and skipped with a
#            message when the DataStore cache or the experiment database is out of reach. This is
#            the tier that proves hot-swapping, the lineup shock, and that the fair-odds line on
#            the trajectory chart steps exactly where the teamsheet landed.
#
# RUN:
#   julia --project -t 8 test/test_matchday_replay.jl

using Test
using BayesianFootball
using DataFrames, Dates, Statistics, UUIDs
import JSON3, HTTP, LibPQ

const REPLAY_DIR = joinpath(@__DIR__, "..", "current_development", "match_day_inference")
include(joinpath(REPLAY_DIR, "replay_state.jl"))
include(joinpath(REPLAY_DIR, "replay_server.jl"))

# ===================================================================
# Fixtures for the pure tier
# ===================================================================

const KO = DateTime(2026, 9, 5, 15)

"The one Saturday in the archive carrying BOTH a 1-minute order book and a scraped XI."
const REPLAY_TEST_DAY = Date(2026, 8, 8)

_key(g, l, s) = (group = g, line = l, selection = s)

_levels(back, back_sz, lay, lay_sz; matched = 5_000.0, ts = KO) =
    MD.BookLevels(Float64.(back), Float64.(back_sz), Float64.(lay), Float64.(lay_sz), matched, ts)

_fixture(mid, home, away; tid = 56) = MD.Fixture(mid, home, away, KO, tid)

"A `Resolved` identity with one market id per canonical group, so `_preload_book`'s mapping and
`quotes` both have something to key on."
_resolved(f) = MD.Resolved(f, "e$(f.m_id)",
                           Dict("MATCH_ODDS" => "1.$(f.m_id)", "OVER_UNDER_25" => "2.$(f.m_id)"),
                           true)

"""
Two snapshots per runner: one 15 minutes before kick-off (the execution book) and one AT kick-off
(the closing book), with deliberately different prices. Every point-in-time assertion below rests
on those two being distinguishable.
"""
function _synthetic_ladders()
    exec_ts, close_ts = KO - Minute(15), KO
    L = Dict{Int,Dict{MD.SelectionKey,Vector{MD.BookLevels}}}()
    S = Dict{Int,Dict{MD.SelectionKey,Vector{DateTime}}}()

    function put!(mid, key, snaps)
        per_sel = get!(L, mid, Dict{MD.SelectionKey,Vector{MD.BookLevels}}())
        per_ts  = get!(S, mid, Dict{MD.SelectionKey,Vector{DateTime}}())
        per_sel[key] = snaps
        per_ts[key]  = [s.ts for s in snaps]
    end

    # 901: 1X2, three runners, so the close is a complete book and can be de-vigged.
    put!(901, _key("1X2", 0.0, :home),
         [_levels([2.50], [500.0], [2.55], [500.0]; ts = exec_ts),
          _levels([2.20], [400.0], [2.20], [400.0]; ts = close_ts)])
    put!(901, _key("1X2", 0.0, :draw),
         [_levels([3.60], [300.0], [3.70], [300.0]; ts = exec_ts),
          _levels([3.40], [300.0], [3.40], [300.0]; ts = close_ts)])
    put!(901, _key("1X2", 0.0, :away),
         [_levels([3.80], [300.0], [3.90], [300.0]; ts = exec_ts),
          _levels([3.60], [300.0], [3.60], [300.0]; ts = close_ts)])

    # 902: Over/Under 2.5, both runners. The stake sheet holds `over_25` and trades it by LAYING
    # `under_25`, which is what makes R19's CLV assertion a test of the morphism.
    put!(902, _key("OverUnder", 2.5, :over_25),
         [_levels([2.30], [200.0], [2.40], [200.0]; ts = exec_ts),
          _levels([2.20], [200.0], [2.20], [200.0]; ts = close_ts)])
    put!(902, _key("OverUnder", 2.5, :under_25),
         [_levels([1.75], [400.0], [1.80], [400.0]; ts = exec_ts),
          _levels([1.75], [400.0], [1.75], [400.0]; ts = close_ts)])
    return (L, S)
end

"A two-scrape lineup table: 11 starters and 2 subs per side, published at T-30."
function _synthetic_lineup_rows(mid::Int; scraped_at::DateTime = KO - Minute(30))
    rows = NamedTuple[]
    pid = 1000 * mid
    for home in (true, false), (i, pos) in enumerate(["G", "D", "D", "D", "D", "M", "M", "M",
                                                      "F", "F", "F", "M", "F"])
        pid += 1
        push!(rows, (match_id = mid, player_id = pid, player_name = "p$pid", position = pos,
                     substitute = i > 11, is_home_team = home, confirmed = true,
                     scraped_at = scraped_at))
    end
    return DataFrame(rows)
end

function _synthetic_card(; lineup_at::DateTime = KO - Minute(30))
    fixtures = [_fixture(901, "alpha", "beta"), _fixture(902, "gamma", "delta"; tid = 57)]
    ids = Dict{Int,Union{MD.Resolved,MD.Unresolved}}(f.m_id => _resolved(f) for f in fixtures)
    L, S = _synthetic_ladders()
    book = PreloadedBook(L, S, Hour(2))
    lus = PreloadedLineups(Dict(901 => _synthetic_lineup_rows(901; scraped_at = lineup_at),
                                902 => _synthetic_lineup_rows(902; scraped_at = lineup_at)))
    # 2-1 and 2-1: 901's home wins, and 902's three goals put Over 2.5 in.
    results = Dict(901 => (2, 1), 902 => (2, 1))
    return ReplayCard(Date(KO), fixtures, KO, ids, book, lus, results,
                      Dict(901 => lineup_at, 902 => lineup_at),
                      (KO - Minute(15), KO))
end

_system() = PF.PortfolioSystem(
    PF.BookSpec(markets = MD.canonical_markets(), price = PF.DeArb()),
    PF.PolicySpec(risk = PF.SlateDrawdown(20.0), cap = PF.FixedCap(0.25),
                  trust = PF.FlatTrust(1.0)))

"""
A `ReplayState` over the synthetic card with NO connection and NO loaded model.

`conn = nothing` is deliberate: it proves the clock, the sources and the payload need neither a
database nor a fit, which is the property that makes R1-R14 unskippable.
"""
_pure_state(; card = _synthetic_card(), account = "replay_unit") =
    ReplayState(nothing, nothing, card; system = _system(), bankroll = 1_000.0,
                account_id = account, schema = REPLAY_SCHEMA)

# ===================================================================
@testset "MatchDay replay engine" begin

# ===================================================================
# 1. ISOLATION -- the constraints that keep the live console alive
# ===================================================================

@testset "R1 the replay refuses every schema but its own" begin
    # This is the single most important assertion in the file. `paper_runbook` is the ledger the
    # 8085 console writes on a Saturday; a replayed fill in it would be indistinguishable from a
    # real one, and there is no column that would tell them apart afterwards.
    @test assert_replay_schema(REPLAY_SCHEMA) == "paper_replay"
    @test_throws ErrorException assert_replay_schema("paper_runbook")
    @test_throws ErrorException assert_replay_schema(MD.PAPER_SCHEMA)   # the production default
    @test_throws ErrorException assert_replay_schema("paper_replay_v2") # near-miss, still refused
    @test "paper_runbook" in FORBIDDEN_SCHEMAS && MD.PAPER_SCHEMA in FORBIDDEN_SCHEMAS

    # and a ReplayState cannot be constructed pointing anywhere else
    card = _synthetic_card()
    @test_throws ErrorException ReplayState(nothing, nothing, card; system = _system(),
                                            schema = "paper_runbook")
end

@testset "R2 the replay console cannot bind the live console's port" begin
    # 8085 is r07. Binding it would either fail noisily or -- worse, if the live process had just
    # died -- succeed, and put a replay's Execute button where an operator expects the real one.
    srv = ReplayServer(_pure_state())
    @test_throws ErrorException serve_replay(srv; port = 8085)
    @test REPLAY_PORT == 8086
    @test !srv.running                      # a refused bind leaves nothing running
end

# ===================================================================
# 2. THE TIME STEPPER
# ===================================================================

@testset "R3 the clock advances a minute at a time and clamps at both ends" begin
    st = _pure_state()
    @test st.clock.t == T_START == -60
    @test T_END == 105

    # minute-by-minute, in both directions, with no drift
    for expected in (-59, -58, -57)
        @test step!(st, 1) == expected
    end
    @test step!(st, -1) == -58
    @test step!(st, 10) == -48
    @test seek!(st, T_EXEC) == -15

    # every minute of the window is reachable and lands exactly where it was asked to
    for t in T_START:T_END
        @test seek!(st, t) == t
        @test st.clock.t == t
    end

    # a scrubber dragged past either end clamps rather than throwing: it is an ordinary gesture
    @test seek!(st, -10_000) == T_START
    @test seek!(st, 10_000) == T_END
    @test step!(st, 500) == T_END
    @test clamp_t(-61) == T_START && clamp_t(106) == T_END
end

@testset "R4 the clock is minutes-to-kickoff and `as_of` is derived, never stored" begin
    # A clock storing an absolute instant and a `t` would have two representations of the same
    # fact and they would drift. Deriving one from the other makes the disagreement unwritable.
    st = _pure_state()
    for t in (-60, -30, -15, 0, 45, 105)
        seek!(st, t)
        @test as_of_at(st.card, st.clock.t) == KO + Minute(t)
    end
    @test as_of_at(st.card, 0) == st.card.kickoff
end

@testset "R5 the named jump targets are the four instants an operator navigates between" begin
    st = _pure_state()
    @test jump!(st, "lineups")    == T_LINEUP  == -30
    @test jump!(st, "exec")       == T_EXEC    == -15
    @test jump!(st, "kickoff")    == T_KICKOFF == 0
    @test jump!(st, "settlement") == T_END     == 105
    @test jump!(st, "start")      == T_START
    @test_throws ErrorException jump!(st, "halftime")
end

@testset "R6 speed is validated, and 60x means one simulated minute per second" begin
    st = _pure_state()
    @test set_speed!(st, 60.0) == 60.0
    @test 60.0 in SPEEDS && 1.0 in SPEEDS && 5.0 in SPEEDS && 30.0 in SPEEDS
    @test_throws ErrorException set_speed!(st, 0.0)
    @test_throws ErrorException set_speed!(st, -1.0)

    # play/pause is a state transition, not a timer: pausing an unplayed clock is a no-op rather
    # than an error, because the console posts an intent and does not track what it last sent.
    st.running = true
    seek!(st, T_START)
    play!(st)
    @test st.clock.playing
    sleep(2.5)                       # 60x with no model loaded: the tick costs ~0
    advanced = st.clock.t
    pause!(st)
    @test !st.clock.playing
    @test advanced > T_START         # it moved
    held = st.clock.t
    sleep(0.6)
    @test st.clock.t == held         # and it stopped
    st.running = false
end

# ===================================================================
# 3. THE FILTRATION CONTRACT -- three ways a replay leaks the future
# ===================================================================

@testset "R7 the order book cannot be read past `as_of`" begin
    # THE leak that would make every number on the console fiction. `PreloadedBook` sorts each
    # runner's ladder by `ts` and reads it with `searchsortedlast`, so a tick from after the
    # replayed instant is UNREACHABLE rather than merely unqueried.
    card = _synthetic_card()
    id901 = card.identities[901]
    home = _key("1X2", 0.0, :home)

    # before the first snapshot there is no book at all -- not a stale one, and not an empty
    # BookLevels, which `MaxBookAge` would then have to distinguish from a real one
    @test isempty(MD.quotes(card.book, id901, KO - Minute(60)))

    # at and after the execution snapshot, the execution price
    for t in (-15, -14, -1)
        b = MD.quotes(card.book, id901, KO + Minute(t))
        @test MD.best_back(b[home]) == 2.50
        @test b[home].ts == KO - Minute(15)
    end

    # at and after the close, the closing price -- the update the console draws every tick
    for t in (0, 1, 30)
        b = MD.quotes(card.book, id901, KO + Minute(t))
        @test MD.best_back(b[home]) == 2.20
        @test b[home].ts == KO
    end

    # the transition happens at exactly one minute, and it is the right one
    @test MD.best_back(MD.quotes(card.book, id901, KO - Minute(1))[home]) == 2.50
    @test MD.best_back(MD.quotes(card.book, id901, KO)[home]) == 2.20
end

@testset "R8 a book older than `max_age` is absent, not stale" begin
    # `MaxBookAge` needs to see a fixture with NO book rather than one with an old price. Serving
    # the old price would make the gate's own staleness check the only thing standing between a
    # six-hour-old quote and a stake.
    L, S = _synthetic_ladders()
    tight = PreloadedBook(L, S, Minute(10))
    card = _synthetic_card()
    id901 = card.identities[901]
    @test !isempty(MD.quotes(tight, id901, KO - Minute(10)))    # 10 minutes old: inside
    @test isempty(MD.quotes(tight, id901, KO - Minute(4)))      # 11 minutes old: gone
    @test !isempty(MD.quotes(tight, id901, KO + Minute(5)))     # the close is fresh again
end

@testset "R9 the XI is invisible until it is scraped, with no historical fallback behind it" begin
    # The live spec chains `LastHistorical` behind `ProvisionalDB` so a player engine always has
    # SOME teamsheet. In a replay that fallback would hide the event this console exists to show:
    # the step from no lineup to the confirmed XI.
    card = _synthetic_card(lineup_at = KO - Minute(30))
    f = card.fixtures[1]
    @test MD.lineup(card.lineups, f, KO - Minute(31)) === nothing
    @test MD.lineup(card.lineups, f, KO - Minute(60)) === nothing

    lu = MD.lineup(card.lineups, f, KO - Minute(30))
    @test lu !== nothing
    @test length(lu.home) == 13 && length(lu.away) == 13
    @test count(p -> !p.substitute, lu.home) == 11
    @test count(p -> p.substitute, lu.home) == 2
    @test lu.confirmed
    @test lu.source === :provisional
    @test lu.scraped_at == KO - Minute(30)
    @test count(p -> p.position === :G, lu.home) == 1

    # and it stays visible afterwards
    @test MD.lineup(card.lineups, f, KO) !== nothing
end

@testset "R10 the plus-minus lineup map is rebuilt from the VISIBLE XI, not the played one" begin
    # The quiet leak. `:player_lineup_ratings_map` is emitted by the feature extractor over EVERY
    # match in `ds.lineups`, so for a finished fixture it already holds the teamsheet that took
    # the field. Left alone, a T-60m decision would be priced off it.
    card = _synthetic_card()
    f = card.fixtures[1]
    lu = MD.lineup(card.lineups, f, KO - Minute(30))
    ratings = Dict{Int,Float64}(p.player_id => 1.0 for p in vcat(lu.home, lu.away))

    agg = pm_aggregate_from_lineup(lu, ratings)
    # 11 starters minus the keeper = 10 outfield at 1.0 each; goalkeepers are excluded from every
    # plus-minus aggregate, matching `pm_lineup_aggregates`.
    @test agg.home_outfield ≈ 10.0
    @test agg.away_outfield ≈ 10.0
    @test agg.home_bench ≈ 2.0
    @test agg.home_D ≈ 4.0 && agg.home_M ≈ 3.0 && agg.home_F ≈ 3.0
    @test agg.home_D + agg.home_M + agg.home_F ≈ agg.home_outfield
    # minute-weighted takes starters at 1.0 and substitutes at 0.0, which is what
    # `pm_lineup_aggregates` does for a player with no recorded minute history -- and a pre-match
    # teamsheet has none by construction.
    @test agg.home_minute ≈ 10.0

    # A player the ridge never rated contributes zero rather than a fallback: an unknown player
    # is not an average player, and imputing one would move the pillar on a name.
    sparse = Dict{Int,Float64}(lu.home[2].player_id => 3.0)
    @test pm_aggregate_from_lineup(lu, sparse).home_outfield ≈ 3.0
    @test pm_aggregate_from_lineup(lu, sparse).away_outfield ≈ 0.0
end

@testset "R11 the materialiser writes a NEUTRAL entry when no XI is visible" begin
    # Writing the neutral aggregate rather than deleting the key makes the pre-drop state an
    # explicit fact of the FeatureSet, which `check_coverage` can then assert on. Deleting it
    # would reach the same zeros through a fallback, and a fallback is exactly what this replay
    # cannot tell apart from a bug.
    card = _synthetic_card()
    fx = card.fixtures
    lu = MD.lineup(card.lineups, fx[1], KO)
    ratings = Dict{Int,Float64}(p.player_id => 2.0 for p in vcat(lu.home, lu.away))

    fs = (data = Dict{Symbol,Any}(
              :plus_minus_ratings => ratings,
              :player_lineup_ratings_map => Dict{Int,FE.PMLineupAggregate}(),
              :player_ratings_map => Dict{Int,Dict{Tuple{String,String},Float64}}()),)
    m = PointInTimeLineupRatings()

    # pre-drop: no lineup in the context at all
    ctx0 = (ds = nothing, model = nothing, as_of = KO - Minute(60), odds = DataFrame(),
            lineups = Dict{Int,Any}())
    @test MD.materialise!(m, Val(:player_lineup_ratings_map), fs, fx, ctx0)
    @test haskey(fs.data[:player_lineup_ratings_map], 901)
    @test fs.data[:player_lineup_ratings_map][901].home_outfield == 0.0
    @test MD.materialise!(m, Val(:player_ratings_map), fs, fx, ctx0)
    @test isempty(fs.data[:player_ratings_map][901])

    # post-drop: the same key now carries the visible XI
    ctx1 = (ds = nothing, model = nothing, as_of = KO, odds = DataFrame(),
            lineups = Dict{Int,Any}(901 => lu))
    @test MD.materialise!(m, Val(:player_lineup_ratings_map), fs, fx, ctx1)
    @test fs.data[:player_lineup_ratings_map][901].home_outfield ≈ 20.0
    @test fs.data[:player_lineup_ratings_map][902].home_outfield == 0.0   # still no XI for 902
    @test MD.materialise!(m, Val(:player_ratings_map), fs, fx, ctx1)
    @test fs.data[:player_ratings_map][901][("home", "D")] ≈ 8.0

    # It DEFERS on an engine with no plus-minus vector, which is what lets it sit in front of
    # `RatingsFromTracker` without capturing a tracker-rating model.
    bare = (data = Dict{Symbol,Any}(:player_ratings_map => Dict{Int,Any}()),)
    @test !MD.materialise!(m, Val(:player_ratings_map), bare, fx, ctx1)
    @test !MD.materialise!(m, Val(:some_other_key), fs, fx, ctx1)

    # and the chain keeps the live materialisers behind it, in order
    chain = replay_materialisers()
    @test chain.members[1] isa PointInTimeLineupRatings
    @test any(x -> x isa MD.RatingsFromTracker, chain.members)
    @test any(x -> x isa MD.LeagueFromFixture, chain.members)
    @test :player_lineup_ratings_map in REPLAY_INJECTABLE_KEYS
    @test all(k -> k in REPLAY_INJECTABLE_KEYS, MD.INJECTABLE_KEYS)
end

@testset "R12 the latents cache key moves when the bench does" begin
    # `BenchWeightedPlayerAggregation` multiplies substitutes by `w_bench`, so promoting a
    # substitute into the XI changes the pillar's value without changing which players are named.
    # A signature blind to that would serve a stale posterior across exactly the transition this
    # console renders.
    card = _synthetic_card()
    f = card.fixtures[1]
    lu = MD.lineup(card.lineups, f, KO)

    c_none = MD.FixtureCard(f, card.identities[901], KO)
    c_xi   = MD.FixtureCard(f, card.identities[901], KO); c_xi.lineup = lu
    @test lineup_signature([c_none]) != lineup_signature([c_xi])
    @test lineup_signature([c_xi]) == lineup_signature([c_xi])       # stable

    promoted = MD.Lineup([MD.Player(p.player_id, p.name, p.position, false) for p in lu.home],
                         lu.away, lu.confirmed, lu.source, lu.scraped_at)
    c_prom = MD.FixtureCard(f, card.identities[901], KO); c_prom.lineup = promoted
    @test lineup_signature([c_xi]) != lineup_signature([c_prom])     # same names, different XI

    # and it is order-independent across fixtures, so a card reshuffled by the gate does not
    # invalidate a cache entry that is still correct
    c2 = MD.FixtureCard(card.fixtures[2], card.identities[902], KO)
    @test lineup_signature([c_xi, c2]) == lineup_signature([c2, c_xi])
end

@testset "R13 coverage refuses only what cannot be materialised" begin
    # `check_coverage` throws on the first of three problems. Only one of them is a genuine
    # refusal for a console: a team absent from `team_map` has no α/β to condition on. The other
    # two are per-match maps that are materialised a few lines later.
    fx = _synthetic_card().fixtures
    fs_ok = (data = Dict{Symbol,Any}(:team_map => Dict("alpha" => 1, "beta" => 2,
                                                       "gamma" => 3, "delta" => 4)),)
    covered, refused = coverage_split(fs_ok, fx)
    @test covered == [901, 902] && isempty(refused)

    fs_gap = (data = Dict{Symbol,Any}(:team_map => Dict("alpha" => 1, "beta" => 2)),)
    covered2, refused2 = coverage_split(fs_gap, fx)
    @test covered2 == [901]
    @test length(refused2) == 1 && refused2[1].first == 902
    @test occursin("team_map", refused2[1].second)
    @test occursin("gamma", refused2[1].second) && occursin("delta", refused2[1].second)

    # a FeatureSet with no team_map at all refuses nothing: absence of the check is not a failed
    # check, and treating it as one would blank the card for every engine that does not use one
    covered3, refused3 = coverage_split((data = Dict{Symbol,Any}(),), fx)
    @test covered3 == [901, 902] && isempty(refused3)
end

@testset "R14 the closing book is de-vigged, and keyed on the MODEL selection" begin
    # `clv_for_order` requires a DE-VIGGED close. The book's raw 1/best_back sums above one, so an
    # un-normalised close would make every leg look like it beat the market, uniformly.
    st = _pure_state()
    probs = closing_probabilities(st)

    p1x2 = [probs[(901, _key("1X2", 0.0, s))] for s in (:home, :draw, :away)]
    @test sum(p1x2) ≈ 1.0                       # the overround is gone
    @test all(0 .< p1x2 .< 1)
    # 2.20 / 3.40 / 3.60 at the close: the raw book sums to 1.0264, so the de-vigged home price
    # is strictly SHORTER than its raw implied probability, not longer.
    @test p1x2[1] < 1 / 2.20
    @test p1x2[1] ≈ (1 / 2.20) / (1 / 2.20 + 1 / 3.40 + 1 / 3.60)

    pou = [probs[(902, _key("OverUnder", 2.5, s))] for s in (:over_25, :under_25)]
    @test sum(pou) ≈ 1.0
    @test pou[2] > pou[1]                       # under is the favourite at 1.75 against 2.20

    # An incomplete market is left OUT rather than normalised: two of three 1X2 runners summing
    # to one would inflate both of them.
    L, S = _synthetic_ladders()
    delete!(L[901], _key("1X2", 0.0, :away)); delete!(S[901], _key("1X2", 0.0, :away))
    holed = _synthetic_card()
    partial = closing_probabilities(ReplayState(nothing, nothing,
        ReplayCard(holed.day, holed.fixtures, holed.kickoff, holed.identities,
                   PreloadedBook(L, S, Hour(2)), holed.lineups, holed.results,
                   holed.lineup_drop, holed.book_span); system = _system()))
    @test !haskey(partial, (901, _key("1X2", 0.0, :home)))
    @test haskey(partial, (902, _key("OverUnder", 2.5, :over_25)))    # 902 is still complete
end

# ===================================================================
# 3b. THE LADDER DESK  (pure -- no database, no model, no fit)
# ===================================================================
#
# Numbered from R24 rather than inserted as R15 so that every existing test keeps the name it
# had: a regression report that says "R17 failed" must go on meaning the ledger test it has
# always meant.
#
# These cover the desk's four claims:
#
#   R24  a spread quoted in ticks means the same thing at 1.50 and at 6.00, and one quoted in
#        currency does not.
#   R25  weight of money is a THREE-LEVEL reading of RESTING size, and says so.
#   R26  a market's runners come from the pipeline's own `betfair_to_key`, not from a second
#        table of symbols that could drift from it.
#   R27  the ladder is the archived book at `as_of` and nothing else -- three levels of both
#        sides, de-vigged where that is legitimate, and NO model number when no model is loaded.
#   R28  our order is marked on the runner it would actually touch, consuming the levels the
#        fill simulation consumes.
#   R29  one runner's history walks forward, and a model that cannot price leaves a GAP.

"""
A 903 fixture whose MATCH_ODDS book is genuinely three levels deep on both sides, with sizes
chosen so every aggregate below is checkable by hand:

    home  back 500/300/200 (Σ 1000)   lay 250/150/100 (Σ 500)   -> WOM 1000/1500 = 66.7%
    draw  back 100/50/25   (Σ 175)    lay 300/200/100 (Σ 600)   -> WOM 175/775   = 22.6%
    away  back 200/200/200 (Σ 600)    lay 200/200/200 (Σ 600)   -> WOM 50.0%

Two snapshots per runner again -- T-15 and T-0 -- because the ladder must be shown to move with
the clock and not merely to exist.
"""
function _deep_card()
    exec_ts, close_ts = KO - Minute(15), KO
    f = _fixture(903, "epsilon", "zeta")
    L = Dict{Int,Dict{MD.SelectionKey,Vector{MD.BookLevels}}}()
    S = Dict{Int,Dict{MD.SelectionKey,Vector{DateTime}}}()
    per_sel = Dict{MD.SelectionKey,Vector{MD.BookLevels}}()
    per_ts  = Dict{MD.SelectionKey,Vector{DateTime}}()

    function put!(sel, back, back_sz, lay, lay_sz, matched)
        snaps = [_levels(back, back_sz, lay, lay_sz; matched = matched, ts = exec_ts),
                 # the close: one tick shorter on the back side, and twice the matched volume
                 _levels(back .- 0.02, back_sz, lay .- 0.02, lay_sz;
                         matched = 2 * matched, ts = close_ts)]
        key = _key("1X2", 0.0, sel)
        per_sel[key] = snaps
        per_ts[key]  = [s.ts for s in snaps]
    end

    put!(:home, [2.50, 2.48, 2.46], [500.0, 300.0, 200.0],
                [2.54, 2.56, 2.58], [250.0, 150.0, 100.0], 4_000.0)
    put!(:draw, [3.60, 3.55, 3.50], [100.0,  50.0,  25.0],
                [3.70, 3.75, 3.80], [300.0, 200.0, 100.0], 1_500.0)
    put!(:away, [3.80, 3.75, 3.70], [200.0, 200.0, 200.0],
                [3.90, 3.95, 4.00], [200.0, 200.0, 200.0], 2_000.0)
    L[903] = per_sel; S[903] = per_ts

    ids = Dict{Int,Union{MD.Resolved,MD.Unresolved}}(903 => _resolved(f))
    return ReplayCard(Date(KO), [f], KO, ids, PreloadedBook(L, S, Hour(2)),
                      PreloadedLineups(Dict(903 => _synthetic_lineup_rows(903))),
                      Dict(903 => (2, 1)), Dict(903 => KO - Minute(30)),
                      (exec_ts, close_ts))
end

_deep_state() = ReplayState(nothing, nothing, _deep_card(); system = _system(),
                            bankroll = 1_000.0, account_id = "replay_ladder",
                            schema = REPLAY_SCHEMA)

"""
A hand-built slate over the deep card carrying exactly two orders, one of each side:

* `home`, BACKED for £900 at the venue. The bid ladder is 500/300/200, so the sweep takes the
  whole touch, the whole second level and £100 of the third -- three levels, fully filled.
* `draw`, expressed by LAYING `draw` for £700. The ask ladder is 300/200/100 = £600, so £100
  cannot fill in the archived three levels. That partial is the point of the leg.
"""
function _deep_slate(card::ReplayCard; as_of::DateTime = KO - Minute(15))
    sheet = DataFrame(
        slate = fill(Date(KO), 2), match_id = [903, 903], family = ["1X2_home", "1X2_draw"],
        group = ["1X2", "1X2"], line = [0.0, 0.0], selection = [:home, :draw],
        venue_selection = [:home, :draw], side = [:back, :lay],
        odds_quoted = [2.50, 1.3846], odds = [2.50, 1.3846],
        p_model = [0.45, 0.30], p_market = [0.40, 0.28],
        edge = [0.05, 0.02], frac = [0.90, 0.70],
        stake = [900.0, 700.0], k_risk = fill(0.04, 2), slate_exposure = fill(0.20, 2),
        capped = fill(false, 2), settled = fill(false, 2),
        venue_odds = [2.50, 3.70], risk = [900.0, 1_890.0], venue_stake = [900.0, 700.0],
        depth_touch = [500.0, 300.0], depth_book = [1000.0, 600.0],
        expected_fill = [900.0, 600.0], expected_vwap = [2.487, 3.74],
        expected_slippage = [0.005, -0.01], fillable = [true, false],
        fill_confidence = [:medium, :low])

    cards = MD.FixtureCard[MD.FixtureCard(f, card.identities[f.m_id], as_of)
                           for f in card.fixtures]
    for c in cards
        c.readiness = MD.Ready()
    end
    books = Dict{Tuple{Int,MD.SelectionKey},MD.BookLevels}()
    for (sel, lv) in MD.quotes(card.book, card.identities[903], as_of)
        books[(903, sel)] = lv
    end
    k_home = _key("1X2", 0.0, :home); k_draw = _key("1X2", 0.0, :draw)
    insts = Dict{Tuple{Int,MD.SelectionKey},MD.Instrument}(
        (903, k_home) => MD.Instrument(k_home, 2.50, :back, 2.50, 1.0, k_home),
        (903, k_draw) => MD.Instrument(k_draw, 3.70 / 2.70, :lay, 3.70, 1 / 2.70, k_draw))
    return MD.PricedSlate(uuid4(), "replay_ladder", Date(KO), as_of, 1_000.0, sheet,
                          DataFrame(), cards, MD.FixtureCard[], insts, books,
                          0.04, 0.20, false, 20.0, 0.25, 2_790.0, 7, "")
end

@testset "R24 a spread is quoted in ticks because currency is not comparable across prices" begin
    # THE reason the tick count is carried at all. 0.05 is a wide spread on a 1.50 shot and is
    # not even one increment on a 6.00 one; a desk that only showed the currency difference
    # would rank those two the same way.
    @test spread_ticks(1.50, 1.55) == 5
    @test spread_ticks(6.00, 6.05) == 0

    # the ladder itself, at every band boundary
    @test tick_index(1.01) == 0
    @test tick_index(2.00) == 99
    @test tick_index(3.00) == 149
    @test tick_index(4.00) == 169
    @test tick_index(6.00) == 189
    @test tick_index(10.0) == 209
    @test tick_index(20.0) == 229
    @test tick_index(30.0) == 239
    @test tick_index(50.0) == 249
    @test tick_index(100.0) == 259
    @test tick_index(1000.0) == 349

    # a price BETWEEN two ticks floors to the one at or below it rather than erroring: the
    # archive carries prices the current ladder does not (2.55 is not on it), and a spread that
    # threw on one would blank a whole column.
    @test tick_index(2.55) == tick_index(2.54)
    @test spread_ticks(2.50, 2.54) == 2

    # an absent side is UNMEASURED, not zero. A zero-tick spread and no book at all are
    # different states and `MaxSpread` treats them differently.
    @test spread_ticks(NaN, 2.0) === nothing
    @test spread_ticks(2.0, NaN) === nothing
    @test tick_index(1.0) === nothing            # below the ladder entirely
    @test tick_index(NaN) === nothing
end

@testset "R25 weight of money is three levels of RESTING size, and no more" begin
    @test wom_pct([100.0, 50.0, 25.0], [75.0, 25.0, 0.0]) ≈ 100 * 175 / 275

    # the archive carries three levels; a fourth that a richer feed might supply is IGNORED
    # rather than silently changing what the gauge means between sources.
    @test wom_pct([100.0, 50.0, 25.0, 10_000.0], [75.0, 25.0, 0.0]) ≈ 100 * 175 / 275
    @test top_depth([100.0, 50.0, 25.0, 10_000.0]) == 175.0

    # a one-sided book still reads, and a two-sided-empty one is `nothing` rather than 50%:
    # an absent gauge and a balanced one must stay distinguishable.
    @test wom_pct([100.0], Float64[]) == 100.0
    @test wom_pct(Float64[], [100.0]) == 0.0
    @test wom_pct(Float64[], Float64[]) === nothing
    @test top_depth(Float64[]) == 0.0

    # the book VWAP is in PROBABILITY space, which is not the arithmetic mean of the prices
    @test vwap_book([2.0, 2.5], [100.0, 100.0]) ≈ 200 / (100 / 2.0 + 100 / 2.5)
    @test !isapprox(vwap_book([2.0, 2.5], [100.0, 100.0]), 2.25; atol = 1e-6)
    @test vwap_book(Float64[], Float64[]) === nothing
end

@testset "R26 a market's runners come from the pipeline's own key mapping" begin
    f = _fixture(903, "epsilon", "zeta")

    mo = market_runners("MATCH_ODDS", f)
    @test [r.key for r in mo] == [_key("1X2", 0.0, s) for s in (:home, :draw, :away)]
    # the columns are headed with TEAM NAMES: a desk column an operator can check against the
    # exchange screen next to it
    @test [r.label for r in mo] == ["epsilon", "Draw", "zeta"]

    ou = market_runners("OVER_UNDER_25", f)
    @test [r.key for r in ou] == [_key("OverUnder", 2.5, :over_25),
                                  _key("OverUnder", 2.5, :under_25)]
    @test all(r -> r.key.line == 2.5, ou)

    btts = market_runners("BOTH_TEAMS_TO_SCORE", f)
    @test [r.key.selection for r in btts] == [:btts_yes, :btts_no]

    @test "MATCH_ODDS" in LADDER_MARKETS && "BOTH_TEAMS_TO_SCORE" in LADDER_MARKETS
    @test_throws ErrorException market_runners("CORRECT_SCORE", f)

    # one runner is addressable by anything an operator or a URL would write
    @test runner_of("MATCH_ODDS", "home", f).key == _key("1X2", 0.0, :home)
    @test runner_of("MATCH_ODDS", "epsilon", f).key == _key("1X2", 0.0, :home)
    @test runner_of("OVER_UNDER_25", "over_25", f).key == _key("OverUnder", 2.5, :over_25)
    @test runner_of("OVER_UNDER_25", "Over", f).key == _key("OverUnder", 2.5, :over_25)
    @test_throws ErrorException runner_of("MATCH_ODDS", "under_25", f)
end

@testset "R27 the ladder is three levels of the archived book at `as_of`, and nothing else" begin
    st = _deep_state()
    seek!(st, T_EXEC)
    lad = fixture_ladder(st, 903, "MATCH_ODDS")

    @test lad.ok && lad.resolved
    @test lad.match_id == 903 && lad.market == "MATCH_ODDS"
    @test lad.t == T_EXEC
    @test length(lad.runners) == 3
    @test [r.symbol for r in lad.runners] == ["home", "draw", "away"]

    home = lad.runners[1]
    # THREE levels on both sides, always, so a short book renders as a visibly short one
    @test length(home.back) == 3 && length(home.lay) == 3
    @test [l.price for l in home.back] == [2.50, 2.48, 2.46]
    @test [l.size  for l in home.back] == [500.0, 300.0, 200.0]
    @test [l.price for l in home.lay]  == [2.54, 2.56, 2.58]
    @test home.best_back == 2.50 && home.best_lay == 2.54
    @test home.spread ≈ 0.04
    @test home.spread_ticks == 2                      # 2.50 -> 2.54 is two 0.02 increments
    @test home.mid ≈ 2.52

    # the WOM arithmetic, by hand: 1000 resting to back against 500 resting to lay
    @test home.wom ≈ round(100 * 1000 / 1500, digits = 1)
    @test home.wom_lay ≈ round(100 - 100 * 1000 / 1500, digits = 1)
    @test home.wom + home.wom_lay ≈ 100.0
    @test home.depth_back == 1_000.0 && home.depth_lay == 500.0
    @test home.depth_touch_back == 500.0 && home.depth_touch_lay == 250.0
    @test home.matched == 4_000.0
    @test home.vwap_book ≈ 1000 / (500 / 2.50 + 300 / 2.48 + 200 / 2.46) atol = 5e-4

    draw, away = lad.runners[2], lad.runners[3]
    @test draw.wom ≈ round(100 * 175 / 775, digits = 1)   # laid into, i.e. drifting
    @test away.wom == 50.0                                # balanced
    @test draw.wom < 40 && home.wom > 60                  # the two ends of the colour scale

    # the market is de-vigged within itself and the three runners sum to one
    @test lad.complete
    # `atol` because the payload rounds to four places; the de-vig itself is exact.
    @test sum(r.p_market for r in lad.runners) ≈ 1.0 atol = 5e-4
    # `book_sum` is the constant that normalisation divided by -- the sum of the RAW mid-implied
    # probabilities, which on a mid-priced book sits near 1 from either side and is therefore
    # not called an overround. The raw and the de-vigged number are different, which is the
    # whole reason both are carried.
    @test lad.book_sum ≈ round(2 / (2.50 + 2.54) + 2 / (3.60 + 3.70) + 2 / (3.80 + 3.90),
                               digits = 4)
    @test lad.runners[1].p_market_raw ≈ round(2 / (2.50 + 2.54), digits = 4)
    @test lad.runners[1].p_market != lad.runners[1].p_market_raw
    @test lad.runners[1].p_market ≈ (2 / (2.50 + 2.54)) / lad.book_sum atol = 5e-4

    # NO model is loaded, so there is no model column. Not zero, not the market's own number:
    # a fabricated fair price is exactly the number an operator would then trade against.
    @test all(r -> r.p_model === nothing, lad.runners)
    @test all(r -> r.fair_odds === nothing, lad.runners)
    @test all(r -> r.ev_pct === nothing, lad.runners)
    @test lad.model_status == "unloaded"

    # and nothing rests on the ladder until a slate does
    @test all(r -> r.order === nothing, lad.runners)
    @test all(r -> r.kelly_stake === nothing, lad.runners)

    # THE FILTRATION CONTRACT, on the desk this time. The ladder at T-0 is the CLOSING book
    # and the one at T-16 does not exist yet -- the desk cannot show a level the card grid
    # could not.
    seek!(st, T_KICKOFF)
    close_lad = fixture_ladder(st, 903, "MATCH_ODDS")
    @test close_lad.runners[1].best_back ≈ 2.48          # one tick shorter at the close
    @test close_lad.runners[1].matched == 8_000.0        # twice the volume
    seek!(st, -16)
    early = fixture_ladder(st, 903, "MATCH_ODDS")
    @test all(r -> r.best_back === nothing, early.runners)
    @test all(r -> r.book_ts === nothing, early.runners)
    @test all(r -> r.spread_ticks === nothing, early.runners)

    # a fixture that is not on the card is refused BY NAME rather than answered emptily
    @test_throws ErrorException fixture_ladder(st, 999, "MATCH_ODDS")
    @test_throws ErrorException fixture_ladder(st, 903, "CORRECT_SCORE")
end

@testset "R27b an incomplete market is reported RAW rather than normalised to one" begin
    # Scaling two of three 1X2 runners to sum to one inflates both -- by up to 20% on a real
    # book. The same refusal `closing_probabilities` makes, on the desk.
    card = _deep_card()
    delete!(card.book.ladders[903], _key("1X2", 0.0, :away))
    delete!(card.book.stamps[903],  _key("1X2", 0.0, :away))
    st = ReplayState(nothing, nothing, card; system = _system(), schema = REPLAY_SCHEMA)
    seek!(st, T_EXEC)
    lad = fixture_ladder(st, 903, "MATCH_ODDS")

    @test !lad.complete
    @test lad.book_sum === nothing
    @test lad.runners[3].best_back === nothing            # the away column is empty, not gone
    @test length(lad.runners) == 3
    ps = [r.p_market for r in lad.runners if r.p_market !== nothing]
    @test length(ps) == 2
    @test sum(ps) < 1.0                                   # NOT normalised
    @test lad.runners[1].p_market ≈ round(2 / (2.50 + 2.54), digits = 4)   # the raw mid
end

@testset "R28 the order marker lands on the runner it would actually touch" begin
    st = _deep_state()
    seek!(st, T_EXEC)
    st.slate = _deep_slate(st.card)
    st.slate_t = T_EXEC
    lad = fixture_ladder(st, 903, "MATCH_ODDS")
    home, draw, away = lad.runners

    # £900 backed into a 500/300/200 bid ladder: the whole touch, the whole second level, and
    # £100 of the third. The same best-first walk `sweep_ladder` performs, so the amber
    # highlight and the fill simulation cannot disagree.
    @test home.order !== nothing
    @test home.order.side == "back"
    @test home.order.venue_stake == 900.0
    @test home.order.level_fills == [500.0, 300.0, 100.0]
    @test home.order.levels_used == 3
    @test home.order.fillable
    @test home.order.unfilled == 0.0
    @test home.kelly_stake == 900.0
    @test home.order.selection == "home"

    # £700 laid into a 300/200/100 ask ladder: £600 fills and £100 does not. A partial that the
    # desk reports rather than rounds away -- it is the difference between the stake the
    # portfolio solved and the position that would exist.
    @test draw.order !== nothing
    @test draw.order.side == "lay"
    @test draw.order.level_fills == [300.0, 200.0, 100.0]
    @test !draw.order.fillable
    @test draw.order.unfilled ≈ 100.0
    @test draw.order.risk == 1_890.0

    # nothing was placed on away, and the marker does not bleed across columns
    @test away.order === nothing
    @test away.kelly_stake === nothing

    # the overview cards carry the same two figures, so a WOM pill and a depth pill need no
    # second API call -- and they are read off the VENUE runner's ladder
    cards = enriched_cards(st, st.slate)
    @test length(cards) == 1
    legs = Dict(l.selection => l for l in cards[1].legs)
    @test haskey(legs, "home") && haskey(legs, "draw")
    @test legs["home"].wom ≈ round(100 * 1000 / 1500, digits = 1)
    @test legs["home"].depth_3lvl == 1_000.0            # a back order eats the BID side
    @test legs["draw"].depth_3lvl == 600.0              # a lay order eats the ASK side
    @test legs["draw"].wom ≈ round(100 * 175 / 775, digits = 1)
    # and nothing the live console renders was removed to make room for them
    @test all(haskey(l, :p_model) && haskey(l, :ev_pct) && haskey(l, :depth_touch)
              for l in cards[1].legs)
end

@testset "R29 one runner's history walks forward, and an absent model leaves a GAP" begin
    st = _deep_state()
    seek!(st, T_KICKOFF)
    h = selection_history(st, 903, "home", "MATCH_ODDS")

    @test h.ok && h.match_id == 903 && h.symbol == "home"
    @test h.from_t == T_START && h.to_t == T_KICKOFF
    @test h.n_points == T_KICKOFF - T_START + 1
    @test h.minutes_to_ko == collect(T_START:T_KICKOFF)
    @test issorted(h.minutes_to_ko)              # chronological, which is the whole axis
    @test length(h.best_back) == h.n_points
    @test length(h.best_lay) == length(h.wom) == length(h.market_matched) == h.n_points
    @test length(h.fair_odds) == h.n_points

    # before the first snapshot there is NO price -- not a stale one, and not a zero
    @test all(x -> x === nothing, h.best_back[1:45])            # T-60 .. T-16
    @test h.best_back[46] == 2.50                              # the T-15 snapshot, exactly
    @test h.best_back[findfirst(==(T_EXEC), h.minutes_to_ko)] == 2.50
    @test h.best_lay[findfirst(==(T_EXEC), h.minutes_to_ko)] == 2.54
    @test h.wom[findfirst(==(T_EXEC), h.minutes_to_ko)] ≈ round(100 * 1000 / 1500, digits = 1)

    # the liquidity curve only ever accumulates
    matched = [m for m in h.market_matched if m !== nothing]
    @test !isempty(matched) && issorted(matched)
    @test last(matched) == 8_000.0

    # NO model is loaded, so `fair_odds` is a gap for the whole window. A back-filled line here
    # would be a model opinion at a minute at which the model had none.
    @test all(x -> x === nothing, h.fair_odds)
    @test all(x -> x === nothing, h.p_model)

    # the chart's own coordinates: the drop is SIGNED against kick-off so it can be dropped
    # straight onto the `minutes_to_ko` axis
    @test h.lineup_drop_min == -30
    @test h.exec_window == (from = T_WINDOW_OPEN, to = T_WINDOW_CLOSE)
    @test h.markers.kickoff == T_KICKOFF

    # `to` defaults to the clock: the chart cannot draw a price the console has not scrubbed to
    seek!(st, -30)
    @test selection_history(st, 903, "home").to_t == -30
    @test selection_history(st, 903, "home").n_points == 31
    # ... and the full horizon is asked for explicitly
    @test selection_history(st, 903, "home"; to = T_END).to_t == T_END

    # every runner of the market is addressable, and an unknown one is refused by name
    @test selection_history(st, 903, "away").symbol == "away"
    @test selection_history(st, 903, "epsilon").symbol == "home"
    @test_throws ErrorException selection_history(st, 903, "btts_yes")
    @test_throws ErrorException selection_history(st, 999, "home")

    # the model-evaluation grid pins the drop minute and its neighbours, so a step lands on the
    # minute it happened rather than on the nearest multiple of five
    g = _history_grid(-60, 0, -29)
    @test issorted(g) && allunique(g)
    @test -30 in g && -29 in g && -28 in g
    @test first(g) == -60 && last(g) == 0
    @test _history_grid(-60, 0, nothing) == collect(-60:5:0)
end

# ===================================================================
# 3b. THE EXECUTION TICKET -- overrides and the constrained re-solve
# ===================================================================
#
# The re-solver is the one piece of NEW mathematics in this console, so it is tested against a
# hand-built portfolio problem whose Kelly answer is produced by `Portfolio.stake_slate` itself.
# That is what makes R33 meaningful: the re-solve of an untouched slate must reproduce the
# allocator's own vector, and it can only be shown to do so if the vector it is compared with
# came from the allocator rather than from a literal in this file.

"""
Two one-selection books whose payoff columns are exact, plus the `PricedSlate` the allocator
sizes them into.

Fixture 901 backs `1X2 home` at 2.50 (win pays 1.50 per unit risk); fixture 902 expresses
`OverUnder 2.5 over_25` by LAYING `under_25` at 1.80, i.e. effective odds 2.25, leverage 1.25 and
a win paying 1.25 per unit risk. Two outcomes per book is all the drawdown solve reads -- it
consumes `p_grid` and `R` and nothing else -- so a two-row grid is a complete book for it and not
a simplification of one.
"""
function _resolver_rig(; bankroll::Float64 = 1_000.0, sys = _system(),
                       a_kelly = [0.0833, 0.1000])
    s901 = PF.Selection("1X2_home", "1X2", 0.0, :home, 2.50, 2.50, 0.45, 0.40)
    s902 = PF.Selection("OverUnder_over_25", "OverUnder", 2.5, :over_25, 2.25, 2.25, 0.50, 0.4444)
    b901 = PF.MatchBook(901, Date(KO), [s901], [0.45, 0.55], reshape([1.5, -1.0], 2, 1),
                        nothing, [a_kelly[1]], 1.0, 0.0, true)
    b902 = PF.MatchBook(902, Date(KO), [s902], [0.50, 0.50], reshape([1.25, -1.0], 2, 1),
                        nothing, [a_kelly[2]], 1.0, 0.0, true)
    books = PF.MatchBook[b901, b902]

    alloc = PF.stake_slate(sys.policy, PF.Slate(Date(KO), books),
                           PF.SlateContext(1, Date(KO), bankroll))
    f = [alloc.stakes[1][1], alloc.stakes[2][1]]

    sheet = DataFrame(
        slate = fill(Date(KO), 2), match_id = [901, 902], family = ["1X2", "OverUnder"],
        group = ["1X2", "OverUnder"], line = [0.0, 2.5], selection = [:home, :over_25],
        venue_selection = [:home, :under_25], side = [:back, :lay],
        odds_quoted = [2.50, 2.25], odds = [2.50, 2.25],
        p_model = [0.45, 0.50], p_market = [0.40, 0.4444],
        edge = [0.05, 0.0556], frac = f, stake = f .* bankroll,
        k_risk = fill(alloc.k_risk, 2), slate_exposure = fill(alloc.exposure, 2),
        capped = fill(alloc.capped, 2), settled = fill(false, 2),
        venue_odds = [2.50, 1.80], risk = f .* bankroll,
        venue_stake = [f[1] * bankroll, f[2] * bankroll * 1.25],
        depth_touch = [500.0, 400.0], depth_book = [500.0, 400.0],
        expected_fill = [500.0, 400.0], expected_vwap = [2.50, 1.80],
        expected_slippage = [0.0, 0.0], fillable = [true, true],
        fill_confidence = [:high, :high])

    card = _synthetic_card()
    as_of = KO - Minute(15)
    cards = MD.FixtureCard[MD.FixtureCard(fx, card.identities[fx.m_id], as_of)
                           for fx in card.fixtures]
    for c in cards
        c.readiness = MD.Ready()
    end
    k_home  = _key("1X2", 0.0, :home)
    k_over  = _key("OverUnder", 2.5, :over_25)
    k_under = _key("OverUnder", 2.5, :under_25)
    insts = Dict{Tuple{Int,MD.SelectionKey},MD.Instrument}(
        (901, k_home) => MD.Instrument(k_home, 2.50, :back, 2.50, 1.0, k_home),
        (902, k_over) => MD.Instrument(k_over, 1.80 / 0.80, :lay, 1.80, 1 / 0.80, k_under))
    bookl = Dict{Tuple{Int,MD.SelectionKey},MD.BookLevels}(
        (901, k_home)  => _levels([2.50], [500.0], [2.55], [500.0]; ts = as_of),
        (902, k_under) => _levels([1.75], [400.0], [1.80], [400.0]; ts = as_of))

    slate = MD.PricedSlate(uuid4(), "replay_unit", Date(KO), as_of, bankroll, sheet, DataFrame(),
                           cards, MD.FixtureCard[], insts, bookl, alloc.k_risk, alloc.exposure,
                           alloc.capped, 20.0, 0.25, sum(sheet.risk), 7, "")

    st = ReplayState(nothing, nothing, card; system = sys, bankroll = bankroll,
                     account_id = "replay_unit", schema = REPLAY_SCHEMA)
    st.clock.t = T_EXEC
    st.slate = slate
    st.slate_t = T_EXEC
    return (st = st, books = books, slate = slate, base = Dict(901 => sheet.risk[1],
                                                               902 => sheet.risk[2]))
end

"The sequential drawdown penalty at a set of risk fractions, exactly as `SlateDrawdown` writes it."
function _penalty(books, fracs::Dict{Int,Float64}; lambda::Float64 = 20.0)
    tot = 0.0
    for b in books
        a = [get(fracs, b.m_id, 0.0)]
        rets = b.R * a
        tot += log(sum(b.p_grid[i] * (1 + rets[i])^(-lambda) for i in eachindex(b.p_grid)))
    end
    return tot
end

_risk_of(slate, mid::Int) =
    let i = findfirst(==(mid), slate.sheet.match_id)
        i === nothing ? 0.0 : Float64(slate.sheet.risk[i])
    end

@testset "R33 an untouched slate re-solves to the allocator's own vector" begin
    # The property that makes every other assertion in this section readable: with nothing
    # overridden, the constrained solve must return k = 1 and reproduce the priced stakes. It
    # can only do so because the priced vector sits exactly on one of the two binding
    # constraints -- `SlateDrawdown` with equality, or `FixedCap` -- which is what
    # `Portfolio.stake_slate` guarantees and what a re-solver that merely rescaled could not use.
    rig = _resolver_rig()
    out = resolve_slate_with_overrides(rig.st; books = rig.books)

    @test nrow(out.sheet) == 2
    @test out.k_risk ≈ 1.0
    @test _risk_of(out, 901) ≈ rig.base[901]
    @test _risk_of(out, 902) ≈ rig.base[902]
    @test out.total_risk ≈ rig.slate.total_risk
    @test occursin("no overrides", rig.st.resolve_note)
end

@testset "R34 a placed leg is frozen at the stake and price it actually got" begin
    rig = _resolver_rig()
    st = rig.st
    # £120 of BACKER stake at 2.40 -- a worse price than the 2.50 the sheet quoted, which is the
    # ordinary outcome of pressing the button a few seconds late.
    @test set_override!(st, 901, "1X2", "home", :placed; line = 0.0, stake = 120.0,
                        odds = 2.40).ok
    out = resolve_slate_with_overrides(st; books = rig.books)

    i = findfirst(==(901), out.sheet.match_id)
    @test i !== nothing
    @test out.sheet.risk[i] ≈ 120.0             # a back's risk IS its stake
    @test out.sheet.venue_odds[i] ≈ 2.40
    @test out.sheet.venue_stake[i] ≈ 120.0
    @test out.sheet.odds[i] ≈ 2.40              # risk-denominated odds, unchanged for a back

    # The frozen leg is far larger than the allocator wanted, so the free one must shrink.
    @test 120.0 > rig.base[901]
    @test _risk_of(out, 902) < rig.base[902]

    # And the budget claim, stated so that it is true in BOTH regimes. A placed bet cannot be
    # un-placed, so a commitment that already breaches the drawdown constraint leaves the vector
    # in breach whatever the re-solver does; what it must never do is make things worse by
    # adding uncommitted risk on top. The exact property is therefore: the re-solved vector's
    # penalty is no greater than the penalty of the frozen commitment ALONE, once that is
    # floored at the feasible zero.
    frozen_only = Dict(901 => 120.0 / 1_000.0)
    fr = Dict(901 => 120.0 / 1_000.0, 902 => _risk_of(out, 902) / 1_000.0)
    @test _penalty(rig.books, fr) <= max(0.0, _penalty(rig.books, frozen_only)) + 1e-8
    @test occursin("froze 1", st.resolve_note)
end

@testset "R35 a skipped leg receives zero stake and leaves the executable vector" begin
    rig = _resolver_rig()
    st = rig.st
    @test set_override!(st, 902, "OverUnder", "over_25", :skipped; line = 2.5).ok
    out = resolve_slate_with_overrides(st; books = rig.books)

    @test nrow(out.sheet) == 1                       # a zero-stake leg is not an order
    @test _risk_of(out, 902) == 0.0
    @test first(out.sheet.match_id) == 901
    # It is still on the TICKET, though -- otherwise there is no row to press [Auto] on.
    tk = ticket_payload(st)
    @test tk.n_legs == 2
    skipped = only(filter(l -> l.match_id == 902, tk.legs))
    @test skipped.status == "skipped"
    @test skipped.resolved_stake == 0.0
    @test skipped.in_resolved == false
end

@testset "R36 skipping re-derives the budget for the surviving legs, in both regimes" begin
    # THE CLAIM A RESCALE COULD NOT MAKE, and the one place a plausible intuition about it is
    # wrong. Dropping a leg removes its column from the return matrix AND its exposure, so the
    # survivors are re-budgeted rather than merely left alone. Which DIRECTION they move in
    # depends on which constraint was binding, and the two cases are asserted separately because
    # a test that only knew one of them would encode a superstition.

    # --- (a) the drawdown budget binds ------------------------------------------------------
    # `SlateDrawdown` solves ONE joint log-utility constraint, `Σ_t log E[(1 + kR_t)^-λ] <= 0`.
    # At the joint optimum the terms sum to zero with some POSITIVE and some negative, so a leg
    # whose term is negative is subsidising the rest -- and removing it makes the constraint
    # TIGHTER for the survivor, not looser. Here 902's term is the negative one, so skipping it
    # moves 901 down to its own solo boundary. That is the correct answer and a rescale-based
    # re-solver would have got it wrong in the other direction.
    rig = _resolver_rig()
    st = rig.st
    set_override!(st, 902, "OverUnder", "over_25", :skipped; line = 2.5)
    out = resolve_slate_with_overrides(st; books = rig.books)

    solo = _risk_of(out, 901)
    @test out.k_risk != 1.0                              # the budget was genuinely re-derived
    @test _penalty(rig.books, Dict(901 => solo / 1_000.0)) <= 1e-8
    # ...and it sits ON the boundary rather than anywhere below it: a hair more breaches it.
    @test _penalty(rig.books, Dict(901 => solo / 1_000.0 * 1.01)) > 0
    @test out.slate_exposure <= 0.25 + 1e-12
    @test out.total_risk <= 0.25 * 1_000.0 + 1e-9

    # Putting it back restores the priced vector exactly, so the panel's [Auto] is a real undo.
    set_override!(st, 902, "OverUnder", "over_25", :auto; line = 2.5)
    back = resolve_slate_with_overrides(st; books = rig.books)
    @test _risk_of(back, 901) ≈ rig.base[901]
    @test _risk_of(back, 902) ≈ rig.base[902]

    # --- (b) the exposure cap binds ---------------------------------------------------------
    # The regime the console's batch header flags as `capped`, and the one where releasing a leg
    # unambiguously frees room. With no drawdown budget the only bound is `FixedCap(0.25)`, so
    # the priced vector sits exactly on it and skipping one leg hands the whole 25% to the other.
    capped_sys = PF.PortfolioSystem(
        PF.BookSpec(markets = MD.canonical_markets(), price = PF.DeArb()),
        PF.PolicySpec(risk = PF.NoRisk(), cap = PF.FixedCap(0.25), trust = PF.FlatTrust(1.0)))
    rig2 = _resolver_rig(; sys = capped_sys, a_kelly = [0.20, 0.15])
    @test rig2.slate.capped == true
    @test rig2.slate.total_risk ≈ 250.0                  # the cap, exactly

    set_override!(rig2.st, 902, "OverUnder", "over_25", :skipped; line = 2.5)
    grown = resolve_slate_with_overrides(rig2.st; books = rig2.books)
    @test nrow(grown.sheet) == 1
    @test _risk_of(grown, 901) > rig2.base[901]          # it grew into the released room
    @test _risk_of(grown, 901) ≈ 250.0                   # ...up to the cap and not past it
    @test grown.k_risk ≈ 250.0 / rig2.base[901]
    @test grown.capped == true

    # A commitment spends that same room: £100 placed leaves £150 for the rest.
    clear_overrides!(rig2.st)
    set_override!(rig2.st, 901, "1X2", "home", :placed; line = 0.0, stake = 100.0, odds = 2.50)
    split = resolve_slate_with_overrides(rig2.st; books = rig2.books)
    @test _risk_of(split, 901) ≈ 100.0
    @test _risk_of(split, 902) ≈ 150.0
    @test split.total_risk ≈ 250.0
end

@testset "R37 commitments that fill the cap leave the uncommitted legs at zero" begin
    # The refusal that matters: an allocator which treated a negative residual as room would
    # keep staking after the bankroll's simultaneous exposure was already committed.
    rig = _resolver_rig()
    st = rig.st
    set_override!(st, 901, "1X2", "home", :placed; line = 0.0, stake = 250.0, odds = 2.50)
    out = resolve_slate_with_overrides(st; books = rig.books)

    @test nrow(out.sheet) == 1
    @test _risk_of(out, 901) ≈ 250.0            # 25% of a £1,000 bankroll: the whole cap
    @test _risk_of(out, 902) == 0.0
    @test occursin("already fills", st.resolve_note)

    # A lay's committed risk is its LIABILITY, not its backer stake -- laying £100 at 1.80 risks
    # £80 -- and the ticket must spend the budget in the same denomination the cap is written in.
    clear_overrides!(st)
    set_override!(st, 902, "OverUnder", "over_25", :placed; line = 2.5, stake = 100.0,
                  odds = 1.80)
    out2 = resolve_slate_with_overrides(st; books = rig.books)
    j = findfirst(==(902), out2.sheet.match_id)
    @test out2.sheet.risk[j] ≈ 80.0
    @test out2.sheet.venue_stake[j] ≈ 100.0
    @test ticket_payload(st).committed_risk ≈ 80.0
end

@testset "R38 an override addresses a leg the model recommended, and says so when it cannot" begin
    rig = _resolver_rig()
    st = rig.st
    bad = set_override!(st, 901, "BTTS", "yes", :skipped)
    @test bad.ok == false && occursin("BTTS", bad.error)

    @test_throws ErrorException StakingOverride(:placed; placed_stake = 0.0, placed_odds = 2.0)
    @test_throws ErrorException StakingOverride(:placed; placed_stake = 10.0, placed_odds = 1.0)
    @test_throws ErrorException StakingOverride(:nonsense)

    back = StakingOverride(:placed; placed_stake = 100.0, placed_odds = 2.50)
    lay  = StakingOverride(:placed; placed_stake = 100.0, placed_odds = 1.80)
    @test committed_risk(back, :back) ≈ 100.0
    @test committed_risk(lay, :lay) ≈ 80.0
    @test effective_odds(back, :back) ≈ 2.50
    @test effective_odds(lay, :lay) ≈ 2.25
    @test committed_risk(StakingOverride(:skipped), :back) == 0.0

    # A `[✓ Placed]` press with no numbers means "I took the recommendation".
    clear_overrides!(st)
    r = set_override!(st, 901, "1X2", "home", :placed; line = 0.0)
    @test r.ok && r.stake ≈ round(rig.slate.sheet.venue_stake[1], digits = 2)
    @test r.odds ≈ 2.50

    # The re-solve is retired by a reprice; the OVERRIDE is not. A bet on the exchange stays on
    # it when the clock moves, and a stake vector solved against one book does not.
    resolve!(st)
    @test st.resolved !== nothing && active_slate(st) === st.resolved
    st.slate_t = T_EXEC + 1
    @test active_slate(st) === st.slate
    @test length(st.overrides) == 1
    @test clear_overrides!(st).n_cleared == 1
    @test isempty(st.overrides)
end

# ===================================================================
# 3c. THE INTELLIGENCE WIDGETS
# ===================================================================

@testset "R39 the form panel cannot see a teamsheet the model could not see" begin
    # The filtration contract, applied to the panel a human reads rather than to the pillar the
    # model prices with. Before the scrape lands there is no XI and the delta says so; it does
    # NOT say "full strength", which would be an assertion no source supports.
    st = _pure_state()
    seek!(st, T_START)
    early = fixture_stats(st, 901)
    @test early.ok && early.match_id == 901
    @test early.lineup.available == false
    @test early.lineup.home.status == "no_xi"
    @test early.lineup.away.status == "no_xi"
    @test isempty(early.lineup.xi_home)

    seek!(st, T_EXEC)
    late = fixture_stats(st, 901)
    @test late.lineup.available == true
    @test late.lineup.drop_min == 30                   # the synthetic card scrapes at T-30
    @test late.lineup.home.n_announced == 13
    @test late.lineup.home.n_starters == 11
    @test length(late.lineup.xi_home) == 13
    @test count(p -> !p.substitute, late.lineup.xi_home) == 11

    # No `DataStore` in the pure tier, so there is no history to measure a regular XI against --
    # and the panel reports that rather than declaring a full-strength side off no evidence.
    @test late.lineup.home.status == "no_history"
    @test late.lineup.home.basis_matches == 0
    @test late.form.home.n == 0
    @test isempty(late.form.home.matches)
    # Scottish League One and Two are not statted for xG, and the panel says which number it is
    # showing instead rather than leaving an empty column.
    @test !isempty(late.xg_note)
    @test occursin("shots", late.xg_note)

    bad = _pure_state()
    @test_throws ErrorException fixture_stats(bad, 999)
end

@testset "R40 form is computed from matches strictly before the replayed day" begin
    # `team_form` and `start_rates` both take the boundary as an argument, so the no-lookahead
    # property is testable without a database: a store containing the match day itself must
    # produce a form line that does not include it.
    ds = (matches = DataFrame(
              match_id = [1, 2, 3, 4],
              tournament_id = [56, 56, 56, 56],
              season_id = [77, 77, 77, 77],
              home_team = ["alpha", "beta", "alpha", "alpha"],
              away_team = ["gamma", "alpha", "delta", "beta"],
              home_score = [2, 0, 1, 3],
              away_score = [1, 0, 3, 0],
              match_date = [Date(2026, 8, 1), Date(2026, 8, 8), Date(2026, 8, 15),
                            Date(2026, 9, 5)]),
          # alpha was home in 1, 3 and 4 and away in 2. Match 4 IS the fixture being replayed,
          # and its teamsheet is in the store -- which is exactly why the date filter has to be
          # the thing that excludes it.
          lineups = DataFrame(
              match_id  = [1,1,1, 2,2,2, 3,3,3, 4,4,4],
              team_side = ["home","home","home", "away","away","away",
                           "home","home","home", "home","home","home"],
              player_id = [10,11,12, 10,11,12, 10,11,12, 10,11,12],
              player_name = repeat(["ever-present", "rotated", "benchwarmer"], 4),
              position  = repeat(["G","M","F"], 4),
              is_substitute = [false,false,true, false,false,true,
                               false,true,true,  false,true,true]),
          bbc = DataFrame(), statistics = DataFrame(), odds = DataFrame())

    f = team_form(ds, "alpha", Date(2026, 9, 5); n = 5)
    @test f.n == 3                                  # the 09-05 fixture is the one being replayed
    @test [m.match_id for m in f.matches] == [3, 2, 1]      # newest first, as the badges read
    @test [m.result for m in f.matches] == ["L", "D", "W"]
    @test f.w == 1 && f.d == 1 && f.l == 1 && f.points == 4
    @test f.gf == 3 && f.ga == 4                    # 1+0+2 for, 3+0+1 against
    @test f.xg_available == false && f.xg_for === nothing

    # ...and pulling the boundary back drops the matches after it, which is the whole property.
    @test team_form(ds, "alpha", Date(2026, 8, 8); n = 5).n == 1
    @test start_rates(ds, "alpha", Date(2026, 8, 8)).n_matches == 1

    # The start rate is measured over the three matches BEFORE the replayed day, never over the
    # four the store holds -- the fourth is the fixture being priced, and its teamsheet is the
    # single most tempting piece of lookahead in this panel.
    sr = start_rates(ds, "alpha", Date(2026, 9, 5))
    @test sr.n_matches == 3
    @test sr.match_ids == [1, 2, 3]
    @test sr.players[10].start_rate == 1.0                # started all three
    @test sr.players[11].start_rate ≈ round(2 / 3, digits = 3)
    @test !haskey(sr.players, 12)                         # never started: not a regular at all
    # One season with three matches is below `min_season_matches`, so the trailing basis is used
    # and the payload says so rather than quietly reporting a one-match season as a season.
    @test sr.basis == "trailing"

    # An empty lineup table is "no history", not "everyone is a regular": the denominator is
    # matches that actually carry a teamsheet.
    bare = (matches = ds.matches, lineups = DataFrame(match_id = Int[], team_side = String[],
                player_id = Int[], player_name = String[], position = String[],
                is_substitute = Bool[]), bbc = DataFrame(), statistics = DataFrame(),
            odds = DataFrame())
    @test start_rates(bare, "alpha", Date(2026, 9, 5)).n_matches == 0
    @test start_rates(bare, "alpha", Date(2026, 9, 5)).basis == "none"
end

@testset "R41 the lineup delta names the regulars who are not in the XI" begin
    starters = Dict(:keeper => 1, :captain => 2, :fringe => 3)
    rates = (players = Dict{Int,NamedTuple}(
                 1 => (player_id = 1, name = "regular keeper", position = "G", starts = 9,
                       appearances = 9, start_rate = 0.9),
                 2 => (player_id = 2, name = "regular captain", position = "M", starts = 8,
                       appearances = 10, start_rate = 0.8),
                 3 => (player_id = 3, name = "fringe", position = "F", starts = 2,
                       appearances = 8, start_rate = 0.2)),
             n_matches = 10, basis = "season", match_ids = Int[])

    full = MD.Player[MD.Player(1, "regular keeper", :G, false),
                     MD.Player(2, "regular captain", :M, false),
                     MD.Player(9, "squad", :D, true)]
    d = lineup_delta(full, rates)
    @test d.status == "full_strength"
    @test isempty(d.missing_starters)
    @test d.n_starters == 2 && d.n_announced == 3

    # The captain drops out and the fringe player takes the shirt: BOTH ends of the same event
    # are reported, because on a lower-division card seeing only one of them is ambiguous.
    rotated = MD.Player[MD.Player(1, "regular keeper", :G, false),
                        MD.Player(3, "fringe", :F, false)]
    d2 = lineup_delta(rotated, rates)
    @test d2.status == "missing_starters"
    @test only(d2.missing_starters).player_id == 2
    @test only(d2.missing_starters).start_rate_pct == 80.0
    @test only(d2.unexpected).player_id == 3

    # A substitute is not a starter: naming a regular on the bench is an absence from the XI.
    benched = MD.Player[MD.Player(1, "regular keeper", :G, false),
                        MD.Player(2, "regular captain", :M, true)]
    @test only(lineup_delta(benched, rates).missing_starters).player_id == 2

    @test lineup_delta(nothing, rates).status == "no_xi"
    # The threshold is a parameter and it moves the finding, so it is on the payload.
    @test isempty(lineup_delta(rotated, rates; threshold = 0.95).missing_starters)
end

@testset "R42 the scorecard's proper-score kernels are the ones they claim to be" begin
    # `_poisson_1x2` and `_score_1x2` are the only arithmetic the scorecard does itself; every
    # other number on it is read from a database. They are pinned here so a change to either
    # shows up as a named failure rather than as a tile that moved.
    p = _poisson_1x2(1.5, 1.1)
    @test sum(p) ≈ 1.0
    @test p[1] > p[3]                                    # the stronger side wins more often
    @test all(x -> 0 < x < 1, p)
    @test _poisson_1x2(1.3, 1.3)[1] ≈ _poisson_1x2(1.3, 1.3)[3]   # symmetric λ, symmetric 1X2

    # A perfect forecast scores zero on all three; a confident wrong one is punished by LogLoss
    # far harder than by Brier, which is the reason both are shown.
    perfect = _score_1x2((1.0, 0.0, 0.0), 1)
    @test perfect.logloss ≈ 0.0 && perfect.brier ≈ 0.0 && perfect.rps ≈ 0.0
    flat = _score_1x2((1/3, 1/3, 1/3), 2)
    @test flat.logloss ≈ -log(1/3)
    wrong = _score_1x2((0.98, 0.01, 0.01), 3)
    @test wrong.logloss > flat.logloss && wrong.brier > flat.brier
    # RPS is ordinal: a home forecast that misses to a draw is penalised less than one that
    # misses to an away win, which Brier alone cannot express.
    @test _score_1x2((0.8, 0.15, 0.05), 2).rps < _score_1x2((0.8, 0.15, 0.05), 3).rps
end

# ===================================================================
# 4. THE CONSOLE SURFACE
# ===================================================================

@testset "R15 the replay API answers every documented route and refuses the rest" begin
    st = _pure_state()
    srv = ReplayServer(st)
    port = 18_500 + (Int(rand(UInt16)) % 1_000)
    serve_replay(srv; host = "127.0.0.1", port = port, push = false)
    try
        base = "http://127.0.0.1:$port"

        page = String(HTTP.get("$base/"; retry = false).body)
        @test occursin("EXECUTE SLATE BATCH", page)
        @test occursin("tabular-nums", page)          # or every card jitters on each tick
        @test occursin("alpinejs", page)
        @test occursin("jump('settlement')", page)

        health = JSON3.read(String(HTTP.get("$base/api/health").body))
        @test health.ok == true && health.port == 8086
        @test health.schema == "paper_replay"

        snap = JSON3.read(String(HTTP.get("$base/api/snapshot").body))
        @test snap.replay.t == st.clock.t
        @test snap.replay.t_start == T_START && snap.replay.t_end == T_END
        @test snap.replay.markers.lineups == T_LINEUP
        @test snap.replay.schema == "paper_replay"
        @test snap.replay.in_play == false                # T-60 is pre-game
        @test length(snap.replay.models) == 3
        @test snap.batch.n_legs == 0                  # nothing priced without a model
        @test isempty(snap.cards)

        # every VCR control is an INTENT the server validates, not a state the page sets
        @test JSON3.read(String(HTTP.post("$base/api/replay/jump";
                  body = JSON3.write((target = "exec",))).body)).t == T_EXEC
        @test JSON3.read(String(HTTP.post("$base/api/replay/step";
                  body = JSON3.write((minutes = 1,))).body)).t == T_EXEC + 1
        @test JSON3.read(String(HTTP.post("$base/api/replay/seek";
                  body = JSON3.write((t = -42,))).body)).t == -42
        @test JSON3.read(String(HTTP.post("$base/api/replay/speed";
                  body = JSON3.write((speed = 30,))).body)).speed == 30.0
        # the query string works too, so every control is reachable from curl
        @test JSON3.read(String(HTTP.post("$base/api/replay/seek?t=-7").body)).t == -7

        # `in_play` flips at the whistle and nowhere else; it is what the page uses to say that
        # a four-figure post-kickoff edge is a measurement, not a signal.
        HTTP.post("$base/api/replay/seek"; body = JSON3.write((t = -1,)))
        @test JSON3.read(String(HTTP.get("$base/api/snapshot").body)).replay.in_play == false
        HTTP.post("$base/api/replay/seek"; body = JSON3.write((t = 0,)))
        @test JSON3.read(String(HTTP.get("$base/api/snapshot").body)).replay.in_play == true

        # a refusal is a value with a reason, never a dropped connection
        bad = JSON3.read(String(HTTP.post("$base/api/replay/jump";
                  body = JSON3.write((target = "halftime",)), status_exception = false).body))
        @test bad.ok == false && occursin("halftime", bad.error)
        nomodel = JSON3.read(String(HTTP.post("$base/api/replay/set_model";
                  body = JSON3.write((model = "m99",)), status_exception = false).body))
        @test nomodel.ok == false && occursin("m99", nomodel.error)

        # an unknown route lists what IS served rather than returning an empty body
        miss = HTTP.get("$base/api/nope"; status_exception = false)
        @test miss.status == 404
        @test occursin("/api/replay/set_model", String(miss.body))
    finally
        stop_replay!(srv)
    end
    @test srv.server === nothing
    stop_replay!(srv)                                  # idempotent
end

@testset "R16 a WebSocket client receives the payload without asking" begin
    st = _pure_state()
    srv = ReplayServer(st; interval = 0.2)
    port = 18_500 + (Int(rand(UInt16)) % 1_000)
    serve_replay(srv; host = "127.0.0.1", port = port, push = true)
    try
        got = Ref{Any}(nothing); n = Ref(0)
        HTTP.WebSockets.open("ws://127.0.0.1:$port/ws") do ws
            for msg in ws
                got[] = JSON3.read(String(msg)); n[] += 1
                n[] >= 2 && break        # first frame on connect, then a pushed one
            end
        end
        @test n[] >= 2
        @test got[].replay.day == string(Date(KO))
        @test haskey(got[], :settlement)
    finally
        stop_replay!(srv)
    end
end

@testset "R30 the desk endpoints answer over HTTP, and refuse by name" begin
    # Both are GETs. A ladder read changes nothing, and making it a GET is what lets an operator
    # paste one fixture's URL into a second tab and watch its book while the console scrubs.
    st = _deep_state()
    seek!(st, T_EXEC)
    st.slate = _deep_slate(st.card)
    st.slate_t = T_EXEC
    srv = ReplayServer(st)
    port = 18_500 + (Int(rand(UInt16)) % 1_000)
    serve_replay(srv; host = "127.0.0.1", port = port, push = false)
    try
        base = "http://127.0.0.1:$port"

        # the page carries both views and the library the chart window needs
        page = String(HTTP.get("$base/"; retry = false).body)
        @test occursin("Multi-Ladder Desk", page)
        @test occursin("Slate Radar", page)
        @test occursin("chart.js", page)
        @test occursin("WOM", page)

        res = HTTP.get("$base/api/replay/ladder?match_id=903&market=MATCH_ODDS")
        @test res.status == 200
        @test occursin("application/json", Dict(res.headers)["Content-Type"])
        lad = JSON3.read(String(res.body))
        @test lad.ok == true
        @test lad.match_id == 903
        @test lad.market == "MATCH_ODDS"
        @test length(lad.runners) == 3
        @test all(r -> length(r.back) == 3 && length(r.lay) == 3, lad.runners)
        @test lad.runners[1].best_back == 2.50
        @test lad.runners[1].wom ≈ round(100 * 1000 / 1500, digits = 1)
        @test lad.runners[1].spread_ticks == 2
        @test lad.runners[1].order.level_fills == [500.0, 300.0, 100.0]
        @test lad.t == T_EXEC

        # naming no fixture means the first on the card, because the page asks for a ladder
        # before the operator has chosen one -- on first paint
        @test JSON3.read(String(HTTP.get("$base/api/replay/ladder").body)).match_id == 903

        hist = HTTP.get("$base/api/replay/history?match_id=903&symbol=home&market=MATCH_ODDS")
        @test hist.status == 200
        h = JSON3.read(String(hist.body))
        @test h.ok == true
        @test h.symbol == "home"
        @test h.n_points == length(h.minutes_to_ko) == length(h.best_back)
        @test length(h.fair_odds) == h.n_points && length(h.wom) == h.n_points
        @test h.minutes_to_ko[1] == T_START
        @test h.to_t == T_EXEC                    # the clock, not the horizon
        @test h.lineup_drop_min == -30
        @test issorted(h.minutes_to_ko)
        # the full horizon is reachable, explicitly
        @test JSON3.read(String(HTTP.get(
            "$base/api/replay/history?match_id=903&symbol=home&to=105").body)).to_t == T_END

        # a refusal is a value with a reason, never a dropped connection or an empty ladder
        bad = JSON3.read(String(HTTP.get("$base/api/replay/ladder?match_id=999";
                                         status_exception = false).body))
        @test bad.ok == false && occursin("999", bad.error)
        badmkt = JSON3.read(String(HTTP.get("$base/api/replay/ladder?match_id=903&market=NOPE";
                                            status_exception = false).body))
        @test badmkt.ok == false && occursin("NOPE", badmkt.error)
        badsel = JSON3.read(String(HTTP.get(
            "$base/api/replay/history?match_id=903&symbol=nonsense";
            status_exception = false).body))
        @test badsel.ok == false && occursin("nonsense", badsel.error)

        # both routes are advertised on a 404, like every other one
        miss = String(HTTP.get("$base/api/nope"; status_exception = false).body)
        @test occursin("/api/replay/ladder", miss)
        @test occursin("/api/replay/history", miss)

        # the snapshot's own cards now carry the two depth figures the pills read
        snap = JSON3.read(String(HTTP.get("$base/api/snapshot").body))
        @test length(snap.cards) == 1
        @test all(l -> haskey(l, :wom) && haskey(l, :depth_3lvl), snap.cards[1].legs)
        @test snap.replay.ladder_markets == collect(LADDER_MARKETS)
        @test snap.replay.window_open == T_WINDOW_OPEN
        @test snap.replay.window_close == T_WINDOW_CLOSE
    finally
        stop_replay!(srv)
    end
end

@testset "R43 the ticket and intelligence endpoints answer over HTTP, and refuse by name" begin
    rig = _resolver_rig()
    st = rig.st
    srv = ReplayServer(st)
    port = 18_500 + (Int(rand(UInt16)) % 1_000)
    serve_replay(srv; host = "127.0.0.1", port = port, push = false)
    try
        base = "http://127.0.0.1:$port"

        # the page carries all six workspace panels and the window manager that arranges them
        page = String(HTTP.get("$base/"; retry = false).body)
        @test occursin("Staking Ticket", page)
        @test occursin("RE-SOLVE REMAINING STAKES", page)
        @test occursin("Team Form", page)
        @test occursin("Model Scorecard", page)
        @test occursin("Multi-Ladder Desk", page)
        @test occursin("Trajectory Chart", page)
        @test occursin("winStyle", page)                # the window manager, not a tab strip

        # the snapshot now carries the ticket alongside the cards it is built from
        snap = JSON3.read(String(HTTP.get("$base/api/snapshot").body))
        @test snap.ticket.available == true
        @test snap.ticket.n_legs == 2
        @test snap.ticket.n_auto == 2 && snap.ticket.n_placed == 0
        @test snap.ticket.resolved == false
        @test all(l -> haskey(l, :recommended_stake) && haskey(l, :resolved_stake) &&
                       haskey(l, :status), snap.ticket.legs)

        # ONE override, then a re-solve, and the header follows the vector the button commits
        ov = JSON3.read(String(HTTP.post("$base/api/replay/stake/override";
                 body = JSON3.write((match_id = 902, market = "OverUnder", line = 2.5,
                                     selection = "over_25", status = "skipped"))).body))
        @test ov.ok == true && ov.status == "skipped"

        # No model is loaded behind this socket, so the payoff matrices cannot be rebuilt and
        # the re-solve takes its documented degraded path: the skipped leg still goes, and the
        # survivor is bounded by residual capacity but never levered above its priced size. The
        # reason is on the note, which is the difference between a degradation and a bug.
        res = JSON3.read(String(HTTP.post("$base/api/replay/stake/resolve").body))
        @test res.ok == true
        @test res.n_legs == 1                            # the skipped leg is not an order
        @test res.k_risk ≈ 1.0
        @test occursin("drawdown budget was NOT re-solved", res.note)

        after = JSON3.read(String(HTTP.get("$base/api/snapshot").body))
        @test after.ticket.resolved == true
        @test after.ticket.n_skipped == 1
        @test after.ticket.n_legs == 2                   # still on the ticket, at zero
        @test after.batch.n_legs == 1                    # but not in the batch
        @test after.batch.total_risk ≈ res.total_risk

        rst = JSON3.read(String(HTTP.post("$base/api/replay/stake/reset").body))
        @test rst.ok == true && rst.n_cleared == 1
        @test JSON3.read(String(HTTP.get("$base/api/snapshot").body)).ticket.n_auto == 2

        # the query string works too, so the whole ticket is reachable from curl
        q = JSON3.read(String(HTTP.post(
            "$base/api/replay/stake/override?match_id=901&market=1X2&selection=home" *
            "&status=placed&stake=120&odds=2.4").body))
        @test q.ok == true && q.stake ≈ 120.0

        # a refusal is a value with a reason, never a dropped connection
        bad = JSON3.read(String(HTTP.post("$base/api/replay/stake/override";
                 body = JSON3.write((match_id = 901, market = "BTTS", selection = "yes",
                                     status = "skipped")), status_exception = false).body))
        @test bad.ok == false && occursin("BTTS", bad.error)

        # the form panel is a GET, like the ladder, and refuses an unknown fixture by number
        stats = JSON3.read(String(HTTP.get("$base/api/replay/stats?match_id=901").body))
        @test stats.ok == true && stats.match_id == 901
        @test haskey(stats, :form) && haskey(stats, :lineup)
        @test stats.lineup.home.status in ("no_xi", "no_history", "full_strength",
                                           "missing_starters")
        nofx = JSON3.read(String(HTTP.get("$base/api/replay/stats?match_id=999";
                                          status_exception = false).body))
        @test nofx.ok == false && occursin("999", nofx.error)

        # the scorecard answers for a registered model and refuses an unregistered one by name.
        # Its `oos` / `paired` blocks may be unavailable here -- there is no experiment database
        # behind this socket -- and that is the point: it degrades to a reported reason.
        card = JSON3.read(String(HTTP.get("$base/api/replay/model_scorecard?model=m00").body))
        @test card.ok == true
        @test card.model.key == "m00"
        @test card.model.run_name == "m00_poisson_control"
        @test haskey(card, :oos) && haskey(card, :paired) && haskey(card, :clv)
        @test card.clv.ok == false                       # no ledger connection in this process
        nomodel = JSON3.read(String(HTTP.get("$base/api/replay/model_scorecard?model=m99";
                                             status_exception = false).body))
        @test nomodel.ok == false && occursin("m99", nomodel.error)

        # all five are advertised on a 404, like every other route
        miss = String(HTTP.get("$base/api/nope"; status_exception = false).body)
        for route in ("/api/replay/stake/override", "/api/replay/stake/resolve",
                      "/api/replay/stake/reset", "/api/replay/stats",
                      "/api/replay/model_scorecard")
            @test occursin(route, miss)
        end
    finally
        stop_replay!(srv)
    end
end

# ===================================================================
# 5. THE LEDGER  (PostgreSQL, `paper_replay` only)
# ===================================================================

function _try_conn()
    haskey(ENV, "BF_DB_URL") || return nothing
    try
        return MD.paper_connection()
    catch e
        @info "replay ledger tests skipped: cannot reach BF_DB_URL" error = sprint(showerror, e)
        return nothing
    end
end

"""
A hand-built `PricedSlate` with exact prices, so execution and settlement are asserted against
arithmetic rather than against whatever the model happened to say.

Two legs, deliberately one of each instrument:

* 901 `1X2 home`, backed directly at 2.50, £100 risk.
* 902 `OverUnder 2.5 over_25`, expressed by LAYING `under_25` at 1.80. Effective odds
  `1.80/0.80 = 2.25`, leverage `1/0.80 = 1.25`, so £80 of risk places £100 at the venue.

Both win on the synthetic card's 2-1 scorelines, which is what makes the settlement arithmetic
checkable to the penny.
"""
function _synthetic_slate(card::ReplayCard; account::AbstractString, as_of::DateTime)
    sheet = DataFrame(
        slate = fill(Date(KO), 2), match_id = [901, 902], family = ["1X2", "OverUnder"],
        group = ["1X2", "OverUnder"], line = [0.0, 2.5], selection = [:home, :over_25],
        venue_selection = [:home, :under_25], side = [:back, :lay],
        odds_quoted = [2.50, 2.25], odds = [2.50, 2.25],
        p_model = [0.45, 0.50], p_market = [0.40, 0.4444],
        edge = [0.05, 0.0556], frac = [0.10, 0.08],
        stake = [100.0, 80.0], k_risk = fill(0.04, 2), slate_exposure = fill(0.18, 2),
        capped = fill(false, 2), settled = fill(false, 2),
        venue_odds = [2.50, 1.80], risk = [100.0, 80.0], venue_stake = [100.0, 100.0],
        depth_touch = [500.0, 400.0], depth_book = [500.0, 400.0],
        expected_fill = [100.0, 100.0], expected_vwap = [2.50, 1.80],
        expected_slippage = [0.0, 0.0], fillable = [true, true],
        fill_confidence = [:high, :high])

    cards = MD.FixtureCard[MD.FixtureCard(f, card.identities[f.m_id], as_of)
                           for f in card.fixtures]
    for c in cards
        c.readiness = MD.Ready()
    end

    k_home = _key("1X2", 0.0, :home)
    k_over = _key("OverUnder", 2.5, :over_25)
    k_under = _key("OverUnder", 2.5, :under_25)
    insts = Dict{Tuple{Int,MD.SelectionKey},MD.Instrument}(
        (901, k_home) => MD.Instrument(k_home, 2.50, :back, 2.50, 1.0, k_home),
        (902, k_over) => MD.Instrument(k_over, 1.80 / 0.80, :lay, 1.80, 1 / 0.80, k_under))
    books = Dict{Tuple{Int,MD.SelectionKey},MD.BookLevels}(
        (901, k_home)  => _levels([2.50], [500.0], [2.55], [500.0]; ts = as_of),
        (902, k_under) => _levels([1.75], [400.0], [1.80], [400.0]; ts = as_of))

    return MD.PricedSlate(uuid4(), String(account), Date(KO), as_of, 1_000.0, sheet,
                          DataFrame(), cards, MD.FixtureCard[], insts, books,
                          0.04, 0.18, false, 20.0, 0.25, 180.0, 7, "")
end

"Count every `paper_runbook` table the replay could conceivably touch."
_runbook_counts(conn) = NamedTuple(first(DataFrame(LibPQ.execute(conn, """
    SELECT (SELECT count(*) FROM paper_runbook.paper_accounts)   AS accounts,
           (SELECT count(*) FROM paper_runbook.paper_slates)     AS slates,
           (SELECT count(*) FROM paper_runbook.paper_orders)     AS orders,
           (SELECT count(*) FROM paper_runbook.paper_fills)      AS fills,
           (SELECT count(*) FROM paper_runbook.paper_settlements) AS settlements,
           (SELECT count(*) FROM paper_runbook.clv_audit)        AS clv,
           (SELECT count(*) FROM paper_runbook.account_ledger)   AS ledger;"""))))

let conn = _try_conn()
if conn === nothing
    @info "SKIPPING the PostgreSQL replay testsets -- BF_DB_URL unset or unreachable. The pure " *
          "tier above covers the clock, the filtration contract and the console surface; these " *
          "cover execution and settlement in paper_replay."
    @testset "R17-R19 replay ledger (skipped, no database)" begin
        @test_skip false
    end
else
try
    acct = "replay_test_" * string(uuid4())[1:8]
    card = _synthetic_card()
    st = ReplayState(nothing, conn, card; system = _system(), bankroll = 1_000.0,
                     account_id = acct, schema = REPLAY_SCHEMA)
    exec_at = KO - Minute(15)

@testset "R17 execution reserves the whole vector and fills it in `paper_replay`" begin
    account = ensure_replay_account!(st)
    @test account.account_id == acct
    @test account.balance ≈ 1_000.0 && account.reserved ≈ 0.0

    st.clock.t = T_EXEC
    st.slate = _synthetic_slate(card; account = acct, as_of = exec_at)
    st.slate_t = T_EXEC

    out = execute!(st)
    @test out.ok
    @test out.n_admitted == 2 && out.n_refused == 0
    @test out.reserved ≈ 180.0                     # 100 + 80 of RISK, not 200 of venue stake
    @test out.n_matched == 2 && out.n_partial == 0 # both books cover their order at the touch
    @test out.risk_filled ≈ 180.0

    # the account moved once, for the whole vector
    a = MD.account_row(conn, acct; schema = REPLAY_SCHEMA)
    @test a.balance ≈ 820.0 && a.reserved ≈ 180.0
    @test MD.equity(a) ≈ 1_000.0                   # a reservation moves, never destroys
    @test MD.reconcile_account(conn, acct; schema = REPLAY_SCHEMA).ok

    slate_id = only(st.executed)
    orders = MD.slate_orders(conn, slate_id; schema = REPLAY_SCHEMA)
    @test length(orders) == 2
    @test all(o -> o.state === MD.MATCHED, orders)

    # the synthetic kept BOTH identities: the model holds over_25, the order touched under_25
    lay = only(filter(o -> o.match_id == 902, orders))
    @test lay.selection === :over_25 && lay.venue_selection === :under_25
    @test lay.side === :lay
    @test lay.effective_odds ≈ 1.80 / 0.80
    @test lay.venue_stake ≈ 100.0 && lay.risk ≈ 80.0

    # fills exist, are attributed to the sweep model, and are denominated in LIABILITY
    fills = MD.fill_rows(conn, slate_id; schema = REPLAY_SCHEMA)
    @test nrow(fills) == 2
    @test all(String.(fills.fill_model) .== "ladder_sweep_v1")
    lay_fill = only(filter(r -> Float64(r.price) == 1.80, fills))
    @test Float64(lay_fill.size) ≈ 100.0
    @test Float64(lay_fill.risk_filled) ≈ 80.0     # size / leverage, not size

    # re-pressing Execute at the SAME minute is a no-op, not a second position
    again = execute!(st)
    @test !again.ok
    a2 = MD.account_row(conn, acct; schema = REPLAY_SCHEMA)
    @test a2.balance ≈ a.balance && a2.reserved ≈ a.reserved
    @test length(st.executed) == 1

    # and executing a slate priced AFTER kick-off is refused by default: the posterior is
    # pre-game and the book is in-play, so the sheet's edges measure the gap between them.
    st.slate_t = 30
    st.slate = _synthetic_slate(card; account = acct, as_of = KO + Minute(30))
    inplay = execute!(st)
    @test !inplay.ok
    @test occursin("PRE-GAME", inplay.error) && occursin("allow_in_play", inplay.error)
    @test length(st.executed) == 1                 # nothing was written
    @test MD.account_row(conn, acct; schema = REPLAY_SCHEMA).reserved ≈ a.reserved

    # The override is not exercised here on purpose: it would book a second slate against this
    # account and R19 asserts the settled balance to the penny. What is asserted is that the
    # refusal NAMES the override, so an operator who means it can find it.
    st.slate_t = T_EXEC
    st.slate = _synthetic_slate(card; account = acct, as_of = exec_at)
end

@testset "R18 the replay cannot touch `paper_runbook`" begin
    # The isolation claim, asserted rather than argued: a full execute-and-settle moves no row in
    # the live paper schema. R1 makes the wrong schema unreachable; this makes the right one
    # demonstrably sufficient.
    before = _runbook_counts(conn)

    st.clock.t = T_END
    out = settle!(st)
    @test out.ok

    after = _runbook_counts(conn)
    @test before == after

    # and the replay's own rows are all in `paper_replay`
    n = first(DataFrame(LibPQ.execute(conn, """
        SELECT count(*) AS n FROM paper_replay.paper_orders WHERE account_id = \$1;""",
        (acct,)))).n
    @test n == 2
end

@testset "R19 settlement grades on the MODEL selection and measures CLV against the close" begin
    s = st.settlement
    @test s !== nothing
    @test s.n_settled == 2 && s.n_legs == 2
    @test s.reconciled

    # --- the money -------------------------------------------------------------------------
    # 901: £100 risk at effective 2.50, home wins 2-1  -> win 150, commission 3, net 147
    # 902: £80  risk at effective 2.25, 3 goals is OVER 2.5 -> win 100, commission 2, net 98
    home = only(filter(l -> l.match_id == 901, s.legs))
    over = only(filter(l -> l.match_id == 902, s.legs))
    @test home.outcome == "WIN" && over.outcome == "WIN"
    @test home.net_pnl ≈ 147.0
    @test over.net_pnl ≈ 98.0
    @test s.net_pnl ≈ 245.0
    @test s.matched_risk ≈ 180.0
    @test s.gross_return ≈ 100 + 150 + 80 + 100          # stake back plus winnings, both legs
    @test s.roi_pct ≈ round(100 * 245 / 180, digits = 2)
    @test s.equity_after ≈ s.equity_before + 245.0
    @test s.balance_after ≈ 1_245.0                       # reservation released, winnings booked

    # THE grading defect this pins down: `over_25` is the model's position and `under_25` is the
    # runner the order touched. Grading the runner would invert every synthetic -- here it would
    # turn a £98 win into an £80 loss.
    @test MD.grade_selection("OverUnder", 2.5, :over_25, 2, 1) === :win
    @test MD.grade_selection("OverUnder", 2.5, :under_25, 2, 1) === :lose
    @test over.selection == "over_25" && over.venue_selection == "under_25"

    # --- the closing line ------------------------------------------------------------------
    close = closing_probabilities(st)
    @test home.close_prob ≈ round(close[(901, _key("1X2", 0.0, :home))], digits = 4)
    # filled at 2.50 (p = 0.400) into a de-vigged close of ~0.443: the price shortened, so the
    # bet beat the close and CLV is positive.
    @test home.beat_close === true
    @test home.clv_pp > 0
    @test home.clv_pp ≈ round(100 * (close[(901, _key("1X2", 0.0, :home))] - 1 / 2.50),
                              digits = 3)

    # CLV is keyed on the MODEL selection for the synthetic too. `over_25` closes near 0.443 and
    # the position was entered at 1/2.25 = 0.444, so this is a near-zero CLV; keying it on
    # `under_25` would have read ~+11pp, which is the size of the mistake being excluded.
    @test over.close_prob ≈ round(close[(902, _key("OverUnder", 2.5, :over_25))], digits = 4)
    @test abs(over.clv_pp) < 5.0
    @test !isapprox(over.close_prob, close[(902, _key("OverUnder", 2.5, :under_25))]; atol = 1e-6)

    # entry is measured at the FILL, not at the quote
    @test home.fill_vwap ≈ 2.50
    @test over.fill_vwap ≈ 1.80

    # --- and it is durable -----------------------------------------------------------------
    audit = DataFrame(LibPQ.execute(conn, """
        SELECT c.* FROM paper_replay.clv_audit c
        JOIN paper_replay.paper_orders o USING (order_id)
        WHERE o.account_id = \$1;""", (acct,)))
    @test nrow(audit) == 2
    @test all(Float64.(audit.entry_lead_min) .== 15.0)   # entered at T-15
    @test s.n_clv == 2 && s.beat_close == 1              # the back beat the close, the lay did not

    # the full-time scores travelled with the payload, so the results view can show them
    @test length(s.scores) == 2
    @test only(filter(x -> x.match_id == 901, s.scores)).home_goals == 2

    # settling twice does not double-book: paper_settlements is unique on order_id
    balance_before = MD.account_row(conn, acct; schema = REPLAY_SCHEMA).balance
    settle!(st)
    @test MD.account_row(conn, acct; schema = REPLAY_SCHEMA).balance ≈ balance_before
end

@testset "R44 the scorecard's CLV half reads the rows settlement just wrote" begin
    # The one number on the scorecard that is about BETTING rather than about forecasting, and
    # the only one that moves when the operator presses Execute. It is read live for exactly that
    # reason, so it is asserted against the settlement this account just booked.
    card = model_scorecard(st, "m00")
    @test card.ok == true
    @test card.model.run_name == "m00_poisson_control"
    @test card.clv.ok == true
    @test card.clv.schema == "paper_replay"
    @test card.clv.account_id == acct

    # `n_settled` on the payload counts what the LAST `settle_slate!` call graded, and R19
    # deliberately settles twice to prove idempotency -- so the durable per-leg facts are what
    # this is checked against, not that transient counter.
    settled = st.settlement
    @test card.clv.n_bets == settled.n_clv
    @test card.clv.n_beat == settled.beat_close
    @test card.clv.beat_pct == settled.beat_close_pct
    @test card.clv.n_settled == settled.n_legs           # two rows in paper_settlements
    @test card.clv.net_pnl ≈ settled.net_pnl
    @test card.clv.matched_risk ≈ settled.matched_risk
    @test card.clv.roi_pct ≈ settled.roi_pct

    # The account and the run name are BOTH in the filter: a scorecard keyed on the model alone
    # would pool this replay account's bets with every other replay account's.
    @test _clv_scorecard(st, "some_other_run").n_bets == 0

    # Whatever the experiment database says, the payload is well-formed and every unavailable
    # figure carries a reason rather than a zero.
    @test haskey(card.oos, :ok) && haskey(card.paired, :ok)
    if card.oos.ok
        @test card.oos.n_folds > 0
        if card.oos.n_scored == 0
            @test !isempty(card.oos.note)
        end
    else
        @test !isempty(card.oos.error)
    end
end

@testset "R20 the ledger can be reset without dropping anything" begin
    reset_replay_ledger!(st)
    a = MD.account_row(conn, acct; schema = REPLAY_SCHEMA)
    @test a.balance ≈ 1_000.0 && a.reserved ≈ 0.0
    @test isempty(st.executed) && st.settlement === nothing
    n = first(DataFrame(LibPQ.execute(conn,
        "SELECT count(*) AS n FROM paper_replay.paper_orders WHERE account_id = \$1;",
        (acct,)))).n
    @test n == 0
    @test _runbook_counts(conn) == _runbook_counts(conn)
end

finally
    try
        LibPQ.execute(conn, "DELETE FROM paper_replay.account_ledger WHERE account_id LIKE 'replay_test_%';")
        LibPQ.execute(conn, "DELETE FROM paper_replay.paper_orders  WHERE account_id LIKE 'replay_test_%';")
        LibPQ.execute(conn, "DELETE FROM paper_replay.paper_slates  WHERE account_id LIKE 'replay_test_%';")
        LibPQ.execute(conn, "DELETE FROM paper_replay.paper_accounts WHERE account_id LIKE 'replay_test_%';")
    catch e
        @warn "replay test cleanup failed" error = sprint(showerror, e)
    end
    try close(conn) catch end
end
end
end

# ===================================================================
# 6. THE MODELS  (a real Saturday, real canonical fits)
# ===================================================================
#
# Slow and network-bound: a DataStore, three chains out of `mcmc_experiments`, and one
# `Features.create_features` per model. Skipped with a message rather than silently when any of
# that is out of reach -- the claims here (hot-swap re-prices in memory; the XI moves a player
# model and does not move a team model) cannot be checked any other way.

function _try_model_rig()
    haskey(ENV, "BF_DB_URL") || return nothing
    try
        ds = DD.load_datastore_cached(DD.ScottishLower())
        conn = MD.paper_connection()
        card = load_replay_card(conn, REPLAY_TEST_DAY)
        st = ReplayState(ds, conn, card; system = _system(), bankroll = 2_400.0,
                         account_id = "replay_model_test", schema = REPLAY_SCHEMA,
                         active = "m00")
        return st
    catch e
        @info "replay model testsets skipped: cannot build the rig" error = sprint(showerror, e)
        return nothing
    end
end

let st = _try_model_rig()
if st === nothing
    @info "SKIPPING the model testsets -- no DataStore cache or no mcmc_experiments reach."
    @testset "R21-R23 replay models (skipped)" begin
        @test_skip false
    end
else
try

@testset "R21 the real order book updates minute by minute at different `as_of`" begin
    # The same point-in-time property as R7, on the archive rather than on a fixture. 2026-08-08
    # carries 43,298 rows spanning 12:00-16:03 around a 14:00 kick-off, so the whole replay
    # window has a book behind it.
    card = st.card
    @test card.kickoff == DateTime(2026, 8, 8, 14)
    @test length(card.fixtures) == 10
    @test count(v -> v isa MD.Resolved, values(card.identities)) == 10
    @test card.book_span[1] <= card.kickoff + Minute(T_START)
    @test card.book_span[2] >= card.kickoff + Minute(T_KICKOFF)
    @test length(card.results) == 10

    id = first(v for v in values(card.identities) if v isa MD.Resolved)
    stamps = DateTime[]
    for t in (-60, -45, -30, -15, 0)
        b = MD.quotes(card.book, id, as_of_at(card, t))
        @test !isempty(b)
        push!(stamps, maximum(lv.ts for lv in values(b)))
        # nothing served is ever from the future
        @test all(lv.ts <= as_of_at(card, t) for lv in values(b))
    end
    @test issorted(stamps)                       # the book walks forward with the clock
    @test length(unique(stamps)) > 1             # ...and it actually moves

    # the prices move too, not just the timestamps
    key = first(keys(MD.quotes(card.book, id, as_of_at(card, 0))))
    prices = [MD.best_back(MD.quotes(card.book, id, as_of_at(card, t))[key])
              for t in (-60, -30, -15, 0)]
    @test any(p -> p != prices[1], prices)
end

@testset "R22 switching the model re-prices the slate in memory" begin
    # The claim the console rests on: `POST /api/replay/set_model` swaps the posterior and
    # re-prices at the CURRENT instant, in the running process, with no restart and no re-fit.
    seek!(st, T_EXEC)
    m00 = set_model!(st, "m00")
    @test m00.status === :ready
    @test m00.run_name == "m00_poisson_control"
    slate00 = st.slate
    @test slate00 !== nothing && MD.n_legs(slate00) > 0
    @test st.slate_t == T_EXEC

    m12 = set_model!(st, "m12")
    @test m12.status === :ready
    @test m12.run_name == "m12_hybrid_production_wealth_player_rapm"
    @test m12.experiment == "scottish_lower_player_grid_2426"
    slate12 = st.slate
    @test slate12 !== nothing && MD.n_legs(slate12) > 0

    # the clock did not move: the operator asked what THIS model says at THIS minute
    @test st.clock.t == T_EXEC
    @test slate12.as_of == slate00.as_of
    @test slate12 !== slate00                      # a genuinely new PricedSlate

    # and the posterior is different, on legs both models priced
    p00 = Dict((r.match_id, r.group, r.line, r.selection) => r.p_model
               for r in eachrow(slate00.sheet))
    p12 = Dict((r.match_id, r.group, r.line, r.selection) => r.p_model
               for r in eachrow(slate12.sheet))
    common = intersect(keys(p00), keys(p12))
    @test !isempty(common)
    @test any(k -> !isapprox(p00[k], p12[k]; atol = 1e-6), common)
    # the MARKET price is the same book, so the difference is the model and nothing else
    q00 = Dict((r.match_id, r.group, r.line, r.selection) => r.p_market
               for r in eachrow(slate00.sheet))
    q12 = Dict((r.match_id, r.group, r.line, r.selection) => r.p_market
               for r in eachrow(slate12.sheet))
    @test all(k -> isapprox(q00[k], q12[k]; atol = 1e-9), common)

    # both models load once and are then held: switching back is a cache hit, not a rebuild
    n_before = length(m00.latents)
    t0 = time()
    set_model!(st, "m00")
    @test time() - t0 < 30.0                       # no Features.create_features on the swap back
    @test length(m00.latents) == n_before          # and the same lineup state was reused
    @test active_slot(st).key == "m00"

    # a fold that cannot represent a fixture refuses it by name rather than pricing it at the
    # league mean -- 2026-08-08 carries two such teams for these folds
    @test length(m00.covered) + length(m00.refused) == length(st.card.fixtures)
    isempty(m00.refused) || @test all(p -> occursin("team_map", p.second), m00.refused)
end

@testset "R23 the XI moves a player model and leaves a team model alone" begin
    # THE lineup shock, as a measurement. 2026-08-08 is the only Saturday in the archive with a
    # scraped XI: nine fixtures, published T-13 to T-40 with a median near T-29.
    card = st.card
    @test length(card.lineup_drop) >= 9
    leads = [Int(round(Dates.value(card.kickoff - t) / 60_000)) for t in values(card.lineup_drop)]
    @test all(0 .< leads .< 60)

    pre, post = -50, -10        # before every scrape, and after all of them
    function priced(model, t)
        set_model!(st, model)
        seek!(st, t)
        s = st.slate
        s === nothing && return Dict{Any,Float64}()
        return Dict((r.match_id, r.group, r.line, r.selection) => r.p_model
                    for r in eachrow(s.sheet))
    end

    # A team-level pillar reads no lineup, so its posterior is IDENTICAL either side of the drop.
    # That is the control: without it, a moving hybrid could be a moving book.
    a00, b00 = priced("m00", pre), priced("m00", post)
    shared00 = intersect(keys(a00), keys(b00))
    @test !isempty(shared00)
    @test all(k -> isapprox(a00[k], b00[k]; atol = 1e-9), shared00)

    # The hybrid player pillar reads the XI, so its posterior MOVES -- and only because of the XI,
    # since `pre` and `post` share the fold, the chain and the feature set.
    a12, b12 = priced("m12", pre), priced("m12", post)
    shared12 = intersect(keys(a12), keys(b12))
    @test !isempty(shared12)
    @test any(k -> !isapprox(a12[k], b12[k]; atol = 1e-6), shared12)

    # and the move is recorded as a second cached latent state, not as a re-fit
    m12 = find_slot(st, "m12")
    @test length(m12.latents) >= 2

    # the card reports the XI as visible after the drop and absent before it
    seek!(st, pre)
    @test all(c -> c.lineup === nothing, st.slate.cards)
    seek!(st, post)
    @test any(c -> c.lineup !== nothing && c.lineup.confirmed, st.slate.cards)
    @test any(c -> c.lineup !== nothing && c.lineup.source === :provisional, st.slate.cards)
end

@testset "R31 the fair-odds line steps at the XI drop, and only for a model that reads it" begin
    # THE chart's reason to exist, as a measurement. The trajectory window puts `1/p_model` on
    # the same axis as the market's two prices; the claim it makes is that the green line MOVES
    # when the teamsheet lands and the blue and pink ones do not have to.
    card = st.card
    set_model!(st, "m12")
    seek!(st, -10)                      # after every scrape on this card
    @test st.slate !== nothing

    withxi = intersect(Set(st.slate.sheet.match_id), Set(keys(card.lineup_drop)))
    @test !isempty(withxi)
    mid = first(sort(collect(withxi)))
    fx = card.fixtures[findfirst(f -> f.m_id == mid, card.fixtures)]
    drop_t = lineup_drop_minute(card, fx)
    @test -60 < drop_t < 0

    # The marker is the first REPLAY MINUTE the XI is visible at, not the rounded scrape time.
    # These two assertions are what make the step and the vertical line the same instant: the
    # teamsheet is unreadable one minute earlier and readable at the marker itself.
    @test MD.lineup(card.lineups, fx, as_of_at(card, drop_t - 1)) === nothing
    @test MD.lineup(card.lineups, fx, as_of_at(card, drop_t)) !== nothing

    h12 = selection_history(st, mid, "home", "MATCH_ODDS"; to = -10)
    @test h12.ok && h12.lineup_drop_min == drop_t
    @test h12.model == "m12"
    @test length(h12.fair_odds) == h12.n_points

    priced = [(m, f) for (m, f) in zip(h12.minutes_to_ko, h12.fair_odds) if f !== nothing]
    @test !isempty(priced)
    vals = unique([f for (_, f) in priced])
    @test length(vals) >= 2                       # it STEPS

    # and the step is at the drop minute, not at the nearest multiple of the evaluation grid
    changes = [m for i in 2:length(priced)
                 for m in (priced[i][1],) if priced[i][2] != priced[i-1][2]]
    @test drop_t in changes
    @test length(changes) == 1                    # one XI, one step

    # the whole 50-minute series cost TWO posterior extractions, not fifty: the memo is keyed on
    # the lineup signature and the signature moves only when an XI lands.
    m12 = find_slot(st, "m12")
    @test length(m12.latents) >= 2
    # ...and far fewer than one per minute. The bound is the number of GRID points rather than a
    # constant: the signature also moves when the gate admits or refuses a fixture, so the memo
    # can legitimately miss more than twice on a card whose book is still filling in.
    @test count(k -> k[1] == "m12", keys(st.model_probs)) <=
          length(_history_grid(T_START, -10, drop_t))
    @test count(k -> k[1] == "m12", keys(st.model_probs)) < (-10 - T_START + 1)

    # THE CONTROL. A team-level pillar reads no lineup, so its fair-odds line is FLAT across the
    # same drop. Without it, a moving hybrid could be a moving book.
    set_model!(st, "m00")
    seek!(st, -10)
    h00 = selection_history(st, mid, "home", "MATCH_ODDS"; to = -10)
    @test h00.model == "m00"
    p00 = [f for f in h00.fair_odds if f !== nothing]
    @test !isempty(p00)
    @test length(unique(p00)) == 1

    # the market series moved underneath both of them, which is what makes the comparison a
    # comparison rather than two readings of the same constant
    bb = [b for b in h12.best_back if b !== nothing]
    @test !isempty(bb)
    @test length(unique(bb)) > 1
    @test h12.best_back == h00.best_back           # the same book, priced by two models
end

@testset "R32 the ladder desk prices a real market with a real model" begin
    seek!(st, T_EXEC)
    set_model!(st, "m00")
    mid = first(sort([f.m_id for f in st.card.fixtures
                      if st.card.identities[f.m_id] isa MD.Resolved]))
    lad = fixture_ladder(st, mid, "MATCH_ODDS")

    @test lad.ok && lad.resolved
    @test length(lad.runners) == 3
    priced = [r for r in lad.runners if r.best_back !== nothing]
    @test !isempty(priced)

    # every runner with a book has a measurable spread and a WOM reading
    @test all(r -> r.spread !== nothing && r.spread >= 0, priced)
    @test all(r -> r.spread_ticks !== nothing && r.spread_ticks >= 0, priced)
    @test all(r -> r.wom === nothing || (0 <= r.wom <= 100), priced)
    @test all(r -> r.depth_back >= 0 && r.depth_lay >= 0, priced)
    @test all(r -> length(r.back) == 3 && length(r.lay) == 3, lad.runners)

    # the model column is present for a covered, gated-through fixture, and `fair_odds` is
    # exactly `1/p_model` -- the same identity the card grid prints
    modelled = [r for r in lad.runners if r.p_model !== nothing]
    if isempty(modelled)
        @info "R32: $(mid) is gated at T-15 under m00; the model column is correctly absent"
    else
        @test all(r -> 0 < r.p_model < 1, modelled)
        @test all(r -> isapprox(r.fair_odds, 1 / r.p_model; atol = 5e-3), modelled)
        @test all(r -> r.ev_pct === nothing ||
                       isapprox(r.ev_pct, 100 * (r.p_model - r.p_market) / r.p_market;
                                atol = 0.05), modelled)
    end

    # the other two markets key without error on the same fixture
    for m in ("OVER_UNDER_25", "BOTH_TEAMS_TO_SCORE")
        l = fixture_ladder(st, mid, m)
        @test l.ok && length(l.runners) == 2
    end
end

@testset "R45 the scorecard reports what each run actually persisted, and nothing else" begin
    # Three registered models, three different states of the same table. `m00_poisson_control`
    # wrote proper scores fold by fold; `m12_hybrid_production_wealth_player_rapm`'s runner wrote
    # chains and convergence diagnostics but no evaluation. A scorecard that averaged the folds
    # which DO carry a number would be describing a different model, so the second case must come
    # back as `nothing` with a reason rather than as a figure.
    for key in ("m00", "m05", "m12")
        card = model_scorecard(st, key)
        @test card.ok == true
        @test card.model.key == key
        @test card.oos.ok == true                       # the run exists in `mcmc_experiments`
        @test card.oos.n_folds > 0
        @test card.oos.n_converged <= card.oos.n_folds
        if card.oos.n_scored == 0
            @test card.oos.logloss === nothing
            @test card.oos.brier === nothing
            @test !isempty(card.oos.note)
        else
            @test card.oos.logloss > 0
            @test 0 < card.oos.brier < 2
        end
        # A weighted figure only exists where `n_matches` was migrated in; where it was not, the
        # payload says so rather than silently reporting the plain mean twice.
        if card.oos.n_matches === nothing
            @test card.oos.logloss_weighted === nothing
        end
        # The paired vs-market block is scored on ONE match set or it is refused with a reason.
        if card.paired.ok
            @test card.paired.n > 0
            @test card.paired.crps > 0
            @test card.paired.logloss > 0
            if card.paired.market_logloss !== nothing
                @test card.paired.logloss_vs_market ≈
                      round(card.paired.logloss - card.paired.market_logloss, digits = 4) atol = 1e-4
            end
        else
            @test !isempty(card.paired.error)
        end
    end

    # The control comparison is a DELTA against m00 and is absent on m00 itself.
    @test model_scorecard(st, "m00").control === nothing
    c = model_scorecard(st, "m05").control
    @test c !== nothing && c.key == "m00"
    if c.comparable
        @test c.d_logloss ≈ round(model_scorecard(st, "m05").oos.logloss - c.logloss,
                                  digits = 4) atol = 1e-4
    end

    @test_throws ErrorException model_scorecard(st, "m99")
end

finally
    try close(st.conn) catch end
end
end
end

end
