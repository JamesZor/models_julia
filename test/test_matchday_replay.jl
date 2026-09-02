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
#   R1-R14   PURE. A synthetic match day built in memory: hand-made ladders, a hand-made XI, a
#            hand-made stake sheet. No database, no DataStore, no trained fit. These are the
#            tests that must never be skipped, because they cover the filtration contract -- if
#            `PreloadedBook` can see past `as_of`, every number the console shows is fiction.
#
#   R15-R19  LEDGER. PostgreSQL, `paper_replay` ONLY. Skipped with a message when `BF_DB_URL` is
#            unset or unreachable. R18 asserts the isolation claim directly by counting
#            `paper_runbook` rows either side of a full execute-and-settle.
#
#   R20-R22  MODELS. A real Saturday, real canonical fits from `mcmc_experiments`. Slow (the
#            hybrid player pillar costs about a minute to build its features) and skipped with a
#            message when the DataStore cache or the experiment database is out of reach. This is
#            the tier that proves hot-swapping and the lineup shock.
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
        @test occursin("EXECUTE AT T", page)
        @test occursin("tabular-nums", page)          # or every card jitters on each tick
        @test occursin("alpinejs", page)
        @test occursin("Fast-Forward Settlement", page)

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

finally
    try close(st.conn) catch end
end
end
end

end
