# test/portfolio_tests.jl
#
# Property tests for the Portfolio module. Each one pins a defect the prototype in
# current_development/portfolio_explore actually shipped with, so a regression is caught as a
# named property rather than as a number that moved.
#
# Fixtures are synthetic -- a hand-built score grid and quotes frame -- so these run with no DB,
# no .cache/ and no trained experiment.

using Test
using BayesianFootball
using DataFrames, Dates, Statistics, LinearAlgebra, Random

const PF = BayesianFootball.Portfolio
const PD = BayesianFootball.Data
const PP = BayesianFootball.Predictions

# ===================================================================
# Fixtures
# ===================================================================

"Independent-Poisson score grid, `n_draws` posterior draws with jittered rates."
function fixture_score_matrix(; max_h = 8, max_a = 8, n_draws = 24, lh = 1.4, la = 1.1, seed = 7)
    rng = MersenneTwister(seed)
    data = zeros(Float64, max_h, max_a, n_draws)
    for k in 1:n_draws
        h = lh * exp(0.25 * randn(rng))
        a = la * exp(0.25 * randn(rng))
        ph = [exp(-h) * h^i / factorial(big(i)) for i in 0:(max_h - 1)]
        pa = [exp(-a) * a^j / factorial(big(j)) for j in 0:(max_a - 1)]
        g = Float64.(ph * pa')
        data[:, :, k] = g ./ sum(g)
    end
    return PP.ScoreMatrix(data)
end

"""
Quotes frame for one match: a complete 1X2 book and a complete O/U 2.5 book.

`ou_overround` is the book's actual overround, i.e. `sum(1/d)`. Note the inversion: to get a
book summing to `ov` with shape `w`, the price is `1/(ov*w)`, NOT `ov/w` -- the latter produces
a book summing to `1/ov` and silently flips an intended arbitrage into vig.
"""
function fixture_quotes(; match_id = 1, ou_overround = 1.04)
    p_over  = 0.52                       # relative shape of the pair
    d_over  = 1 / (ou_overround * p_over)
    d_under = 1 / (ou_overround * (1 - p_over))
    DataFrame(
        match_id    = fill(match_id, 5),
        market_name = ["1X2", "1X2", "1X2", "OverUnder", "OverUnder"],
        market_line = [0.0, 0.0, 0.0, 2.5, 2.5],
        selection   = [:home, :draw, :away, :over_25, :under_25],
        odds_close  = [2.40, 3.50, 3.20, d_over, d_under],
    )
end

function fixture_model_probs(sm)
    mk = [PD.Market1X2(), PD.MarketOverUnder(2.5)]
    Dict(string(m) => PP.compute_market_probs(sm, m) for m in mk)
end

fixture_spec(; price = PF.DeArb(), kw...) = PF.BookSpec(
    markets = PD.MarketConfig(PD.AbstractMarket[PD.Market1X2(), PD.MarketOverUnder(2.5)]),
    price = price,
    shrink = PF.NoShrinkage();
    kw...)

"A MatchBook assembled directly, bypassing build_book so no ExperimentResults is needed."
function fixture_book(; match_id = 1, date = Date(2025, 1, 1), h = 2, a = 1,
                        price = PF.DeArb(), ou_overround = 1.04)
    sm   = fixture_score_matrix()
    spec = fixture_spec(price = price)
    sels = PF.extract_selections(fixture_quotes(match_id = match_id,
                                                ou_overround = ou_overround),
                                 match_id, spec, fixture_model_probs(sm))
    max_h, max_a, _ = size(sm.data)
    p = vec(mean(sm.data, dims = 3)[:, :, 1]); p ./= sum(p)
    R = PF.payoff_matrix(sels, max_h, max_a, spec.exec.commission)
    r = PF.allocate(spec.allocator, p, R, spec.exec)
    return (book = PF.MatchBook(match_id, date, sels, p, R,
                                PF.settle_vector(sels, h, a, spec.exec.commission),
                                r.a, 1.0, r.kkt, r.converged),
            spec = spec, sm = sm, score = (h, a), max_h = max_h)
end

ctx0() = PF.SlateContext(1, Date(2025, 1, 1), 1.0)

# ===================================================================
@testset "Portfolio" begin

# -------------------------------------------------------------------
@testset "P1 payoff matrix agrees with settlement" begin
    f = fixture_book()
    h, a = f.score
    row = PF.grid_index(h, a, f.max_h)
    @test f.book.settle ≈ f.book.R[row, :]

    # and at every other scoreline in the grid, not just the realised one
    for hh in 0:3, aa in 0:3
        s = PF.settle_vector(f.book.sels, hh, aa, f.spec.exec.commission)
        @test s ≈ f.book.R[PF.grid_index(hh, aa, f.max_h), :]
    end
end

# -------------------------------------------------------------------
@testset "P10 a push settles at 0, not -1" begin
    # DrawNoBet on a draw is graded `missing` by Data.grade_selection -> stake returned
    s = PF.Selection("DrawNoBet_dnb_home", "DrawNoBet", 0.0, :dnb_home, 2.0, 2.0, 0.5, 0.5)
    c = PF.PerBetCommission(0.02)
    @test PF.payoff(s, 1, 1, c) == 0.0      # draw  -> push
    @test PF.payoff(s, 2, 1, c) > 0.0       # home win
    @test PF.payoff(s, 0, 1, c) == -1.0     # away win
end

# -------------------------------------------------------------------
@testset "P2 trust scales the marginal Kelly stake exactly" begin
    # single selection at exactly fair odds: blending the probability by w and scaling the
    # stake by w must agree, which is what licenses trust-as-a-multiplier downstream.
    p_fair, w = 0.40, 0.35
    d = 1 / p_fair
    p_model = 0.50
    exec = PF.ExecutionConfig(commission = PF.NoCommission())

    grid(pm) = [pm, 1 - pm]
    R = reshape([d - 1.0, -1.0], 2, 1)

    a_full  = PF.allocate(PF.KellyLogUtility(), grid(p_model), R, exec).a
    p_blend = w * p_model + (1 - w) * p_fair
    a_blend = PF.allocate(PF.KellyLogUtility(), grid(p_blend), R, exec).a

    @test a_blend[1] ≈ w * a_full[1] rtol = 1e-3
end

# -------------------------------------------------------------------
@testset "P9 allocator reaches a KKT point" begin
    f = fixture_book()
    @test f.book.kkt < 1e-4
    @test all(f.book.a_kelly .>= 0)
    @test sum(f.book.a_kelly) <= f.spec.exec.budget + 1e-8
    @test all(f.book.a_kelly .<= f.spec.exec.max_selection_stake + 1e-8)
end

# -------------------------------------------------------------------
@testset "P8 de-arb stops the optimum covering a market group" begin
    # a sub-1 book is a window artifact, not an arbitrage: with RawPrice the solver takes the
    # free money and covers the group; with DeArb it must not.
    covered(price, ov) = begin
        f = fixture_book(price = price, ou_overround = ov)
        idx = findall(s -> s.group == "OverUnder", f.book.sels)
        length(idx) == 2 && all(j -> f.book.a_kelly[j] > 0, idx)
    end
    @test covered(PF.RawPrice(), 0.96)      # the artifact is real and the solver bites
    @test !covered(PF.DeArb(), 0.96)        # ... and de-arbing removes it
    @test !covered(PF.DeArb(), 1.04)        # a genuine book is untouched and still not covered

    # DeArb never settles above the traded price
    f = fixture_book(price = PF.DeArb(), ou_overround = 1.04)
    @test all(s -> s.odds_used <= s.odds_quoted + 1e-12, f.book.sels)
end

# -------------------------------------------------------------------
@testset "incomplete market groups are rejected" begin
    sm   = fixture_score_matrix()
    spec = fixture_spec()
    q    = fixture_quotes()
    partial = q[q.selection .!= :under_25, :]        # drop one leg of the O/U pair
    sels = PF.extract_selections(partial, 1, spec, fixture_model_probs(sm))
    @test all(s -> s.group != "OverUnder", sels)
    @test count(s -> s.group == "1X2", sels) == 3

    # match ids arrive as Int32 from ds.matches and Int64 from a latents frame
    @test length(PF.extract_selections(q, Int32(1), spec, fixture_model_probs(sm))) ==
          length(PF.extract_selections(q, Int64(1), spec, fixture_model_probs(sm)))
end

# -------------------------------------------------------------------
@testset "P3 the risk map is homogeneous of degree 0" begin
    f = fixture_book()
    b = f.book
    risk = PF.SlateDrawdown(20.0)
    for c in (0.25, 1.0, 4.0)
        a1 = b.a_kelly; a2 = c .* b.a_kelly
        k1 = PF.risk_factor(risk, [b.p_grid], [b.R * a1])
        k2 = PF.risk_factor(risk, [b.p_grid], [b.R * a2])
        # k(c*a) * (c*a) == k(a) * a, provided the constraint actually binds
        if k1 < 1.0 - 1e-9
            @test k1 .* a1 ≈ k2 .* a2 rtol = 1e-3
        end
    end
end

# -------------------------------------------------------------------
@testset "P4 a ruinous cap is unconstructible" begin
    @test_throws ArgumentError PF.FixedCap(1.0)
    @test_throws ArgumentError PF.FixedCap(1.5)
    @test_throws ArgumentError PF.FixedCap(0.0)
    @test PF.FixedCap(0.25).cap == 0.25

    # and the cap binds: eight full-Kelly books on one slate cannot risk the bankroll
    books = [fixture_book(match_id = i, date = Date(2025, 1, 1)).book for i in 1:8]
    slate = PF.Slate(Date(2025, 1, 1), books)
    pol   = PF.PolicySpec(trust = PF.FlatTrust(1.0), risk = PF.NoRisk(),
                          cap = PF.FixedCap(0.25))
    alloc = PF.stake_slate(pol, slate, ctx0())
    @test alloc.exposure <= 0.25 + 1e-9
    @test alloc.capped

    t = PF.simulate(pol, [slate])
    @test minimum(t.bankroll) > 0
    @test all(>(-1.0), t.slate_pl)
end

# -------------------------------------------------------------------
@testset "P5 policy never mutates a built book" begin
    f = fixture_book()
    slate = PF.Slate(f.book.date, [f.book])
    before = copy(f.book.a_kelly)

    PF.simulate(PF.PolicySpec(trust = PF.FlatTrust(0.25)), [slate])
    PF.simulate(PF.PolicySpec(trust = PF.FlatTrust(1.0), risk = PF.NoRisk()), [slate])

    @test f.book.a_kelly == before

    # and the cache key tracks BookSpec, not PolicySpec
    @test PF.book_cache_key(fixture_spec()) == PF.book_cache_key(fixture_spec())
    @test PF.book_cache_key(fixture_spec()) !=
          PF.book_cache_key(fixture_spec(price = PF.RawPrice()))

    # The case that actually breaks: a component holding a non-isbits field. Julia's default
    # hash for such a struct is identity-based, so a naive key differs for equal specs and the
    # cache never hits. NoShrinkage is field-less and hides this -- BakerMcHale does not.
    bm(; kw...) = PF.BookSpec(markets = PD.MarketConfig(PD.AbstractMarket[PD.Market1X2()]),
                              shrink = PF.BakerMcHale(); kw...)
    @test PF.book_cache_key(bm()) == PF.book_cache_key(bm())
    @test PF.book_cache_key(bm()) != PF.book_cache_key(bm(shrink = PF.NoShrinkage()))
    @test PF.book_cache_key(bm()) !=
          PF.book_cache_key(bm(shrink = PF.BakerMcHale(n_draws = 64)))
    @test PF.component_hash(PF.BakerMcHale()) == PF.component_hash(PF.BakerMcHale())
end

# -------------------------------------------------------------------
@testset "P6 ScheduledTrust resolves by slate index only" begin
    sched = PF.ScheduledTrust([PF.FlatTrust(0.1), PF.FlatTrust(0.9)])
    sel = fixture_book().book.sels[1]
    d = Date(2025, 1, 1)
    # the weight depends on the slate index, never on the date or bankroll it is queried with
    @test PF.trust_for(sched, sel, PF.SlateContext(1, d, 1.0)) == 0.1
    @test PF.trust_for(sched, sel, PF.SlateContext(2, d, 1.0)) == 0.9
    @test PF.trust_for(sched, sel, PF.SlateContext(1, Date(2030, 1, 1), 99.0)) == 0.1
end

# -------------------------------------------------------------------
@testset "P7 unsorted slates are rejected" begin
    b1 = fixture_book(match_id = 1, date = Date(2025, 1, 1)).book
    b2 = fixture_book(match_id = 2, date = Date(2025, 2, 1)).book
    good = [PF.Slate(b1.date, [b1]), PF.Slate(b2.date, [b2])]
    bad  = [PF.Slate(b2.date, [b2]), PF.Slate(b1.date, [b1])]
    @test PF.simulate(PF.PolicySpec(), good) isa PF.Trajectory
    @test_throws AssertionError PF.simulate(PF.PolicySpec(), bad)
    @test_throws AssertionError PF.group(PF.DailySlate(), [b2, b1])
end

# -------------------------------------------------------------------
@testset "strict trust surfaces an unmapped selection" begin
    sel = fixture_book().book.sels[1]
    strict = PF.SelectionTrust(Dict{Tuple{String,Float64,Symbol},Float64}())
    @test_throws KeyError PF.trust_for(strict, sel, ctx0())
    lenient = PF.SelectionTrust(Dict{Tuple{String,Float64,Symbol},Float64}();
                                default = 0.3, strict = false)
    @test PF.trust_for(lenient, sel, ctx0()) == 0.3
end

# -------------------------------------------------------------------
@testset "end to end: slates, metrics, filters" begin
    books = [fixture_book(match_id = i, date = Date(2025, 1, 1) + Day(7 * ((i - 1) ÷ 3))).book
             for i in 1:9]
    slates = PF.group(PF.DailySlate(), books)
    @test length(slates) == 3
    @test sum(length(s.books) for s in slates) == 9

    t = PF.simulate(PF.PolicySpec(), slates)
    m = PF.path_metrics(t)
    @test m.n_slates == 3
    @test m.final > 0
    @test m.mdd <= 0
    @test m.mean_exposure <= 0.25 + 1e-9
    @test nrow(PF.attribution(t)) >= 1

    # a filter can only remove exposure
    unfiltered = PF.simulate(PF.PolicySpec(), slates).total_stake
    filtered   = PF.simulate(PF.PolicySpec(filter = PF.MinEdge(0.05)), slates).total_stake
    @test filtered <= unfiltered + 1e-12

    # SingleMatchSlate degenerates to one match per window
    @test length(PF.group(PF.SingleMatchSlate(), books)) == 9
end

# -------------------------------------------------------------------
@testset "match-day: unsettled books stake but do not simulate" begin
    f = fixture_book()
    played   = f.book
    unplayed = PF.MatchBook(2, played.date, played.sels, played.p_grid, played.R,
                            nothing,                      # <- no result yet
                            played.a_kelly, 1.0, played.kkt, played.converged)

    @test PF.is_settled(played)
    @test !PF.is_settled(unplayed)

    # an unsettled book can be STAKED
    pol   = PF.PolicySpec(trust = PF.FlatTrust(0.25), risk = PF.NoRisk())
    alloc = PF.stake_slate(pol, PF.Slate(played.date, [unplayed]), ctx0())
    @test sum(sum(a) for a in alloc.stakes) > 0
    @test alloc.exposure <= pol.cap.cap + 1e-9

    # ... but NOT simulated
    @test_throws AssertionError PF.simulate(pol, [PF.Slate(played.date, [unplayed])])
    @test PF.simulate(pol, [PF.Slate(played.date, [played])]) isa PF.Trajectory

    # a mixed slate is refused too -- half a result is not a backtest
    @test_throws AssertionError PF.simulate(pol, [PF.Slate(played.date, [played, unplayed])])
end

end # @testset "Portfolio"
