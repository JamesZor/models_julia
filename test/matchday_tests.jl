# test/matchday_tests.jl
#
# Property tests for src/MatchDay. Every one is DB-free: fixtures are hand-built so the suite
# runs without betdb, matching test/portfolio_tests.jl practice.
#
# Each property corresponds to a claim the design makes, or a defect the prototype had.

using Test
using DataFrames, Dates
using BayesianFootball
const MD = BayesianFootball.MatchDay
const PF = BayesianFootball.Portfolio

_key(g, l, s) = (group = g, line = l, selection = s)
_levels(back, lay; matched = 5000.0, ts = DateTime(2026, 8, 2, 14, 57)) =
    MD.BookLevels(back, fill(100.0, length(back)), lay, fill(100.0, length(lay)), matched, ts)

# The real Waterford v Shelbourne O/U 2.5 snapshot, 2026-08-02.
_ou25() = Dict(_key("OverUnder", 2.5, :over_25)  => _levels([4.80], [5.00]),
               _key("OverUnder", 2.5, :under_25) => _levels([1.25], [1.26]))

@testset "MatchDay" begin

@testset "M1 back/lay conversion is an involution" begin
    for d in (1.05, 1.26, 2.0, 4.8, 50.0)
        @test MD.lay_to_back(MD.back_to_lay(d)) ≈ d
        @test MD.back_to_lay(MD.lay_to_back(d)) ≈ d
    end
    @test MD.lay_to_back(1.0) == Inf          # degenerate price, never a division by zero
    @test MD.lay_to_back(0.5) == Inf
end

@testset "M2 a lay in risk units is payoff-identical to a back" begin
    # THE load-bearing claim: it is what lets Portfolio stay unaware that lays exist, and what
    # makes FixedCap sum liability without knowing it.
    for d in (1.1, 1.26, 2.5, 6.0)
        s = 1.0                                   # one unit of RISK
        D = MD.lay_to_back(d)
        inst = MD.Instrument(_key("OverUnder", 2.5, :over_25), D, :lay, d, 1 / (d - 1))
        b = MD.venue_stake(inst, s)               # backer stake we must post
        @test b ≈ s / (d - 1)
        @test b            ≈ s * (D - 1)          # lay win  == back win
        @test -s           ≈ -s                   # lay lose == back lose, trivially
    end
end

@testset "M3 BestOfBackLay is never worse than DirectBackOnly" begin
    book, q = _ou25(), MD.BestAvailable()
    ks = collect(keys(book))
    for k in ks
        c = MD.complement_of(k, ks)
        direct = MD.instrument(MD.DirectBackOnly(), k, c, book, q)
        best   = MD.instrument(MD.BestOfBackLay(),  k, c, book, q)
        @test best !== nothing && direct !== nothing
        @test best.odds >= direct.odds - 1e-12
    end
    # and on this snapshot it is strictly better on the Over, by the documented amount
    k = _key("OverUnder", 2.5, :over_25)
    best = MD.instrument(MD.BestOfBackLay(), k, MD.complement_of(k, ks), book, q)
    @test best.side == :lay
    @test best.odds ≈ 1.26 / 0.26
    @test isapprox(100 * (best.odds / 4.80 - 1), 0.962; atol = 0.01)
end

@testset "M4 leverage cap rejects synthetics off a price near 1" begin
    # The O/U 0.5 "23% gain" artifact: an empty back book, not an edge. Killed on price alone,
    # with no depth query -- which is what makes decision 8 (no size checking) safe.
    book = Dict(_key("OverUnder", 0.5, :over_05)  => _levels([12.0], [13.0]),
                _key("OverUnder", 0.5, :under_05) => _levels([1.02], [1.03]))
    ks, q = collect(keys(book)), MD.BestAvailable()
    k = _key("OverUnder", 0.5, :over_05)
    capped   = MD.instrument(MD.BestOfBackLay(max_leverage = 20.0), k, MD.complement_of(k, ks), book, q)
    uncapped = MD.instrument(MD.BestOfBackLay(max_leverage = 1e6), k, MD.complement_of(k, ks), book, q)
    @test capped.side == :back && capped.odds == 12.0
    @test uncapped.side == :lay && uncapped.leverage > 30
end

@testset "M5 1X2 has no complement and can only be backed" begin
    book = Dict(_key("1X2", 0.0, :home) => _levels([2.0], [2.02]),
                _key("1X2", 0.0, :draw) => _levels([3.4], [3.5]),
                _key("1X2", 0.0, :away) => _levels([4.0], [4.1]))
    ks, q = collect(keys(book)), MD.BestAvailable()
    for k in ks
        @test MD.complement_of(k, ks) === nothing      # three outcomes, not two
        @test MD.instrument(MD.BestOfBackLay(), k, nothing, book, q).side == :back
    end
end

@testset "M6 the exchange minimum applies to venue stake, not to risk" begin
    # A lay at a short price clears a £1 minimum with far less than £1 at risk. This is the
    # interaction that makes the morphism useful at a small bankroll, not just cheaper.
    lay  = MD.Instrument(_key("OverUnder", 2.5, :over_25), 4.846, :lay, 1.26, 1 / 0.26)
    back = MD.Instrument(_key("OverUnder", 2.5, :over_25), 4.80,  :back, 4.80, 1.0)

    @test MD.round_stake(MD.NoMinimum(), 0.05, back) == 0.05      # research mode ignores it
    @test MD.round_stake(MD.FloorOrDrop(minimum = 1.0), 0.50, back) == 0.0   # £0.50 < £1 -> drop
    @test MD.round_stake(MD.FloorOrDrop(minimum = 1.0), 0.50, lay)  == 0.50  # posts £1.92 -> keep
    @test MD.venue_stake(lay, 0.50) > 1.0

    up = MD.FloorOrRoundUp(minimum = 1.0, max_inflation = 3.0)
    @test MD.round_stake(up, 0.50, back) == 1.0    # 2x inflation, allowed
    @test MD.round_stake(up, 0.10, back) == 0.0    # 10x inflation, refused
end

@testset "M6b an order ticket names the runner the order touches" begin
    # THE DEFECT THIS PINS DOWN. `Instrument.key` is the position we want; on a synthetic the
    # venue action is on its COMPLEMENT. `order_ticket` used to emit (selection = key,
    # side = :lay, price = the complement's lay price), which instructs the opposite position at
    # a price belonging to the other runner. Measured on the 2026-08-08 ScottishLower slate,
    # 14 of 48 legs were synthetics, so ~29% of tickets were wrong.
    #
    # M2-M6 all passed throughout, because the MATHS was never wrong -- only the naming. This is
    # the one place the "downstream never needs to know a lay exists" abstraction must leak.
    book, q = _ou25(), MD.BestAvailable()
    ks   = collect(keys(book))
    want = _key("OverUnder", 2.5, :over_25)
    inst = MD.instrument(MD.BestOfBackLay(), want, MD.complement_of(want, ks), book, q)

    @test inst.side === :lay                       # the synthetic wins on this snapshot
    @test inst.key       == want                                       # position held
    @test inst.venue_key == _key("OverUnder", 2.5, :under_25)          # runner traded
    @test inst.venue_odds == 1.26                  # ...and that price is the COMPLEMENT's lay
    @test inst.venue_key != inst.key               # the whole point

    # A direct back trades the runner it names, so the two coincide -- including via the
    # five-argument constructor, which is what keeps older call sites correct.
    direct = MD.instrument(MD.DirectBackOnly(), want, MD.complement_of(want, ks), book, q)
    @test direct.venue_key == direct.key == want
    @test MD.Instrument(want, 4.8, :back, 4.8, 1.0).venue_key == want

    # And the ticket carries the venue runner, with the model's selection still recoverable.
    row = (match_id = 1, group = "OverUnder", line = 2.5, selection = inst.key.selection,
           side = inst.side, venue_odds = inst.venue_odds, venue_selection = inst.venue_key.selection,
           risk = 1.0, venue_stake = MD.venue_stake(inst, 1.0))
    t = MD.order_ticket(row)
    @test t.selection === :under_25        # what you place
    @test t.model_selection === :over_25   # what it expresses
    @test t.side === :lay
    @test t.price == 1.26
    @test t.liability == 1.0               # a lay's liability is the risk, not the venue stake
    @test t.market == "OverUnder" && t.line == 2.5   # morphism never changes the market
end

@testset "M6c select_split refuses a fold containing the fixtures being priced" begin
    # THE DEFECT THIS PINS DOWN, measured on ScottishUpper 2026-08-09:
    #
    #   fold   targets   last target date   fixtures being priced, inside
    #     2        10        2026-08-02       0    <- correct
    #     3        22        2026-08-09       6    <- what min(n_trained, n_bounds) chose
    #
    # The DataStore cache had been force-rebuilt, so the splitter recomputed fold 3's window and
    # it grew to swallow the card. Both counts were 3, so the POSITIONAL rule picked it and the
    # count-mismatch warning never fired. Entirely silent, and the FeatureSet would have been
    # built over a window containing the results.
    #
    # This testset covers rules 2 and 3 (the `exclude` fallback and the positional default).
    # Rule 1 -- POSITIVE identification via `Data.get_next_matches` -- needs a real DataStore and
    # so cannot run in a DB-free suite. It was verified against the same ScottishUpper case:
    #
    #   fold   meta                        get_next_matches   slate fixtures in it
    #     1    Season 26/27, Week 0          10 (week 1)        0
    #     2    Season 26/27, Week 1          12 (week 2)        6   <- chosen, correctly
    #     3    Season 26/27, Week 2           0 (unplayed)       0
    #
    # Rule 1 keys on (target_season, time_step) rather than on list position, so a rebuilt
    # boundary list cannot move the answer. The two rules agree here; rule 1 gets there from the
    # fold's own semantics rather than by elimination.
    bnd(target) = [(; target_match_ids = target, history_match_ids = Int[])]
    boundaries  = [bnd(Int[]), bnd([1, 2, 3]), bnd([1, 2, 3, 40, 41])]
    expr        = (training_results = [(:chain1,), (:chain2,), (:chain3,)],)

    slate = [40, 41]                                   # the fixtures we are pricing

    # positional: takes fold 3, which contains the slate
    @test select_split(expr, boundaries; strict = false).idx == 3
    # by content: steps back to the most recent fold clear of it
    sel = select_split(expr, boundaries; strict = false, exclude = slate)
    @test sel.idx == 2
    @test sel.chain == :chain2
    @test occursin("stepping back", sel.warning)

    # a clean card is unaffected, and reports no warning
    @test select_split(expr, boundaries; strict = false, exclude = [99]).idx == 3
    @test isempty(select_split(expr, boundaries; strict = false, exclude = [99]).warning)

    # nothing to exclude behaves exactly as before
    @test select_split(expr, boundaries; strict = false, exclude = Int[]).idx == 3

    # and when EVERY fold has seen the card, refuse rather than pick one
    @test_throws ErrorException select_split(expr, boundaries; strict = false, exclude = [1])

    # Rule 1 degrades safely. These boundaries carry no split metadata, so `get_next_matches`
    # cannot be called on them; the implementation must fall through to rule 2 rather than
    # throw. That matters because a splitter whose metadata shape this does not understand
    # should cost accuracy of selection, never an outage.
    sel2 = select_split(expr, boundaries; strict = false, exclude = slate,
                        ds = nothing, config = nothing, fixture_ids = slate)
    @test sel2.idx == 2
end

@testset "M7 gates are conjunctive and collect every reason" begin
    # "unresolved" alone is a dead resolver; "unresolved" AND "no quotes" is a dead collector
    # too. Short-circuiting would hide the second, which is usually the informative one.
    f = MD.Fixture(1, "a", "b", DateTime(2026, 8, 2, 18, 45), 79)
    card = MD.FixtureCard(f, MD.Unresolved(f, :absent_from_crosswalk), DateTime(2026, 8, 2, 17))
    out = MD.ready(MD.GateChain(MD.IdentityResolved(), MD.MaxBookAge(Minute(30))), card)
    @test out isa MD.Blocked
    @test length(out.reasons) == 2
    @test :identity in first.(out.reasons) && :book in first.(out.reasons)
    @test !MD.is_ready(out)
end

@testset "M8 ConfirmedXI is non-blocking by default" begin
    # Measured, not chosen: `confirmed` has never been true in sofascore.lineup_provisional,
    # so a blocking default would block 100% of fixtures.
    f = MD.Fixture(1, "a", "b", DateTime(2026, 8, 2, 18, 45), 79)
    card = MD.FixtureCard(f, MD.Resolved(f, "e", Dict("MATCH_ODDS" => "1.1"), true),
                          DateTime(2026, 8, 2, 17))
    card.lineup = MD.Lineup(MD.Player[], MD.Player[], false, :provisional,
                            DateTime(2026, 8, 2, 12))
    @test MD.ready(MD.ConfirmedXI(), card) isa MD.Ready
    @test MD.ready(MD.ConfirmedXI(blocking = true), card) isa MD.Blocked
    # MaxLineupAge is the usable version while `confirmed` is never set
    @test MD.ready(MD.MaxLineupAge(max_age = Hour(2), blocking = true), card) isa MD.Blocked
    @test MD.ready(MD.MaxLineupAge(max_age = Hour(12), blocking = true), card) isa MD.Ready
end

@testset "M9 SourceChain is first-success, GateChain is conjunctive" begin
    # Two different combinators. Collapsing them would be the obvious refactor and would be wrong.
    struct _NeverSource <: MD.AbstractLineupSource end
    MD.lineup(::_NeverSource, ::MD.Fixture, ::DateTime) = nothing
    struct _AlwaysSource <: MD.AbstractLineupSource end
    MD.lineup(::_AlwaysSource, ::MD.Fixture, t::DateTime) =
        MD.Lineup(MD.Player[], MD.Player[], true, :test, t)

    f = MD.Fixture(1, "a", "b", DateTime(2026, 8, 2, 18, 45), 79)
    chain = MD.SourceChain(_NeverSource(), _AlwaysSource(), _NeverSource())
    @test MD.lineup(chain, f, DateTime(2026, 8, 2)).source == :test
    @test MD.lineup(MD.SourceChain(_NeverSource()), f, DateTime(2026, 8, 2)) === nothing
end

@testset "M10 an unplayed fixture produces a stakeable book" begin
    # The defect that made Portfolio.stake_sheet unreachable: fixture_table(ds) is built from
    # FINISHED matches, so it holds no entry with a nothing score and an upcoming fixture is
    # absent entirely. fixture_info builds the live table instead.
    f = MD.Fixture(42, "a", "b", DateTime(2026, 8, 2, 18, 45), 79)
    card = MD.FixtureCard(f, MD.Resolved(f, "e", Dict{String,String}(), true),
                          DateTime(2026, 8, 2, 17))
    fi = MD.fixture_info([card])
    @test haskey(fi, 42)
    @test fi[42].score === nothing          # <- the whole point
    @test fi[42].date == Date(2026, 8, 2)
end

@testset "M11 declared-but-unimplemented seams error, never return empty" begin
    f = MD.Fixture(1, "a", "b", DateTime(2026, 8, 2, 18, 45), 79)
    r = MD.Resolved(f, "e", Dict("MATCH_ODDS" => "1.1"), true)
    @test_throws ErrorException MD.quotes(MD.RedisLive(), r, DateTime(2026, 8, 2))
end

@testset "M12 market-name mapping round-trips" begin
    @test MD.key_to_betfair("OverUnder", 2.5) == "OVER_UNDER_25"
    @test MD.key_to_betfair("1X2", 0.0) == "MATCH_ODDS"
    @test MD.key_to_betfair("BTTS", 0.0) == "BOTH_TEAMS_TO_SCORE"
    @test MD.key_to_betfair("CorrectScore", 0.0) === nothing

    @test MD.betfair_to_key("OVER_UNDER_25", "Over 2.5 Goals") == _key("OverUnder", 2.5, :over_25)
    @test MD.betfair_to_key("BOTH_TEAMS_TO_SCORE", "Yes") == _key("BTTS", 0.0, :btts_yes)
    # the collector normalises MATCH_ODDS runners; verified against order_book_1m
    @test MD.betfair_to_key("MATCH_ODDS", "home") == _key("1X2", 0.0, :home)
    @test MD.betfair_to_key("MATCH_ODDS", "draw") == _key("1X2", 0.0, :draw)
    # CORRECT_SCORE is deliberately unmapped -- a -20% ROI drag in the staking work
    @test MD.betfair_to_key("CORRECT_SCORE", "1 - 1") === nothing
end

@testset "M13 league_lookup is materialised, and its index matches Features'" begin
    # The convention is set by Features.add_feature!(::LeagueFeature, ...):
    # sort(unique(ds.matches.tournament_id)) enumerated from 1, keyed off the FULL DataStore.
    ds = (matches = DataFrame(tournament_id = [57, 56, 57, 56], match_id = [1, 2, 3, 4]),)
    fs = (data = Dict{Symbol,Any}(:league_lookup => Dict{Int,Int}()),)

    fx = [MD.Fixture(101, "ross-county", "montrose",  DateTime(2026, 8, 8, 13), 56),
          MD.Fixture(102, "elgin-city",  "forfar",    DateTime(2026, 8, 8, 13), 57),
          MD.Fixture(103, "a",           "b",         DateTime(2026, 8, 8, 13), 79)]

    @test MD.materialise!(MD.LeagueFromFixture(), Val(:league_lookup), fs, fx, (ds = ds,))
    @test fs.data[:league_lookup][101] == 1        # 56 sorts first
    @test fs.data[:league_lookup][102] == 2        # 57 second
    # a tournament the store has never seen is left UNMAPPED, so check_coverage refuses it
    # rather than silently assigning it a neighbouring league
    @test !haskey(fs.data[:league_lookup], 103)
end

@testset "M14 check_coverage catches an unmaterialised league_lookup" begin
    # Regression: :league_lookup was absent from INJECTABLE_KEYS, so pooled engines hit
    # `get(league_lookup, mid, 0)` and priced the fixture with the δ_league offset ZEROED --
    # i.e. at the mean of League One and League Two rather than in its own division.
    fx = [MD.Fixture(101, "ross-county", "montrose", DateTime(2026, 8, 8, 13), 56)]
    model = nothing

    covered = (data = Dict{Symbol,Any}(:league_lookup => Dict(101 => 1)),)
    @test MD.check_coverage(covered, fx, model)

    bare = (data = Dict{Symbol,Any}(:league_lookup => Dict{Int,Int}()),)
    @test_throws ErrorException MD.check_coverage(bare, fx, model)
end

@testset "M15 a materialiser chain refuses a key no member handles" begin
    # The chain's docstring always claimed this; the caller used to discard the return value.
    fs = (data = Dict{Symbol,Any}(:league_lookup => Dict{Int,Int}()),)
    fx = [MD.Fixture(101, "a", "b", DateTime(2026, 8, 8, 13), 56)]
    ds = (matches = DataFrame(tournament_id = [56], match_id = [1]),)

    @test !MD.materialise!(MD.MaterialiserChain(MD.RatingsFromTracker()),
                           Val(:league_lookup), fs, fx, (ds = ds,))
    @test MD.materialise!(MD.MaterialiserChain(MD.RatingsFromTracker(), MD.LeagueFromFixture()),
                          Val(:league_lookup), fs, fx, (ds = ds,))
    # and the default spec carries a member for every injectable key, so none can go unhandled
    # by accident when a new pooled engine is served
    members = MD.MatchDaySpec().features.members
    @test any(m -> m isa MD.RatingsFromTracker, members)
    @test any(m -> m isa MD.LeagueFromFixture, members)
    @test length(MD.INJECTABLE_KEYS) == 2
end

@testset "M16 team_name_score tiers, on the real 2026-08-07 spellings" begin
    # Every pair below is a fixture that was actually on the card. The scorer is only ever asked
    # "which of the 3-5 events in this kick-off window is this?", so the bar is separation, not
    # absolute similarity -- but each tier still has to fire for the pair that motivated it.
    @test MD.team_name_score("Galway Utd", "galway-united")   == 1.0    # alias utd -> united
    @test MD.team_name_score("Kerry FC", "kerry-fc")          == 1.0    # dropped token
    @test MD.team_name_score("Treaty United", "treaty-united-fc") == 1.0
    @test MD.team_name_score("Partick", "partick-thistle")    == 0.9    # prefix
    @test MD.team_name_score("Dundalk", "dundalk-fc")         == 1.0
    # UCD is unmatchable by any substring rule and is a real fixture -- hence the initialism tier
    @test MD.team_name_score("UCD", "university-college-dublin") == 0.85
    # and wrong pairings have to score near zero, or the margin test means nothing
    @test MD.team_name_score("Cork City", "cobh-ramblers")    == 0.0
    @test MD.team_name_score("UCD", "wexford-fc")             == 0.0
    @test MD.team_name_score("", "livingston")                == 0.0
end

@testset "M17 LiveNameMatch separates the real card by a wide margin" begin
    # Reproduces the assignment offline: the 9 exchange events of 2026-08-07 against the 9
    # SofaScore fixtures. This is the evidence behind min_margin = 0.25 -- the observed worst
    # margin is 0.633, so the threshold sits an order of magnitude clear of the worst real case.
    bf = [("Athlone Town", "Longford"), ("Bray Wanderers", "Finn Harps"),
          ("Cobh Ramblers", "Treaty United"), ("Cork City", "Kerry FC"),
          ("UCD", "Wexford F.C"), ("Derry City", "Sligo Rovers"),
          ("Galway Utd", "Drogheda"), ("Shamrock Rovers", "Dundalk"),
          ("Partick", "Livingston")]
    sofa = [("athlone-town", "longford-town"), ("bray-wanderers", "finn-harps"),
            ("cobh-ramblers", "treaty-united-fc"), ("cork-city", "kerry-fc"),
            ("university-college-dublin", "wexford-fc"), ("derry-city", "sligo-rovers"),
            ("galway-united", "drogheda-united"), ("shamrock-rovers", "dundalk-fc"),
            ("partick-thistle", "livingston")]

    pair(b, s) = (MD.team_name_score(b[1], s[1]) + MD.team_name_score(b[2], s[2])) / 2
    worst_score, worst_margin = 1.0, 1.0
    for (i, s) in enumerate(sofa)
        scores = [pair(b, s) for b in bf]
        order  = sortperm(scores, rev = true)
        @test order[1] == i                                   # the right event wins
        worst_score  = min(worst_score, scores[order[1]])
        worst_margin = min(worst_margin, scores[order[1]] - scores[order[2]])
    end
    @test worst_score  >= 0.875
    @test worst_margin >= 0.633
    # the defaults must actually admit that card
    r = MD.LiveNameMatch()
    @test r.min_score  <= worst_score
    @test r.min_margin <= worst_margin
end

end
