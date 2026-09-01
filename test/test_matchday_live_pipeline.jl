# test/test_matchday_live_pipeline.jl
#
# The live match-day pipeline: slate pricing, the paper ledger, the execution state machine, the
# fill model, settlement, and the operator console.
#
# Structure follows test/matchday_tests.jl and test/portfolio_tests.jl: every testset pins a CLAIM
# the design makes or a DEFECT the design exists to prevent, so a regression surfaces as a named
# property rather than as a number that moved.
#
# DB-FREE BY DEFAULT. The pure layers -- `sweep_ladder`, `simulate_fill`, `decide_order`,
# `reserve_plan`, `settle_order` -- take no connection by construction, which is the whole point
# of splitting them out: a Saturday is replayable as a table of numbers rather than only on a
# Saturday. The ledger testsets need PostgreSQL and are SKIPPED WITH A MESSAGE when `BF_DB_URL`
# is unset or unreachable, never silently passed.

using Test
using BayesianFootball
using DataFrames, Dates, Statistics, LinearAlgebra, Random, UUIDs
import JSON3, HTTP, LibPQ

const MD = BayesianFootball.MatchDay
const PF = BayesianFootball.Portfolio
const PD = BayesianFootball.Data
const PP = BayesianFootball.Predictions

# ===================================================================
# Fixtures
# ===================================================================

_key(g, l, s) = (group = g, line = l, selection = s)

"A two-sided book with explicit ladders, so capacity assertions are exact rather than incidental."
_levels(back, back_sz, lay, lay_sz; matched = 5_000.0, ts = DateTime(2026, 9, 5, 14, 35)) =
    MD.BookLevels(Float64.(back), Float64.(back_sz), Float64.(lay), Float64.(lay_sz), matched, ts)

"The Scottish League Two 1X2 shape: 5 ticks wide, £42 at the touch. Measured 2026-08-15."
_l2_home() = _levels([2.34, 2.32, 2.30], [42.0, 90.0, 150.0],
                     [2.46, 2.48, 2.50], [30.0, 80.0, 120.0])

function _order(; risk = 26.0, venue_stake = 26.0, side = :back, odds = 2.34, leverage = 1.0,
                  edge = 0.024, p_model = 0.451, slate = UUID(1), acct = "t",
                  state = MD.TRIGGERED, selection = :home, venue_selection = :home,
                  group = "1X2", line = 0.0, match_id = 101, oid = uuid4())
    MD.PaperOrder(order_id = oid, slate_id = slate, account_id = acct, match_id = match_id,
                  kickoff = DateTime(2026, 9, 5, 15), market_group = group, market_line = line,
                  selection = selection, venue_selection = venue_selection, side = side,
                  venue_odds = odds, leverage = leverage,
                  effective_odds = side === :lay ? MD.lay_to_back(odds) : odds,
                  p_model = p_model, p_market = p_model - edge, edge = edge,
                  stake_fraction = risk / 1000, risk = risk, venue_stake = venue_stake,
                  quote_ts = DateTime(2026, 9, 5, 14, 35), state = state)
end

_account(; balance = 2400.0, reserved = 0.0, cap = 0.25) =
    MD.PaperAccount(account_id = "t", opening_balance = 2400.0, balance = balance,
                    reserved = reserved, max_slate_exposure = cap, commission_rate = 0.02)

# --- a synthetic slate, built through Portfolio's own primitives -------------------
# Reused from portfolio_tests.jl practice: a hand-built score grid and quotes frame, so no DB,
# no .cache/ and no trained fit are needed to exercise the joint allocator.

function _score_matrix(; lh = 1.35, la = 1.05, n_draws = 16, seed = 11, max_g = 8)
    rng = MersenneTwister(seed)
    data = zeros(Float64, max_g, max_g, n_draws)
    for k in 1:n_draws
        h = lh * exp(0.2 * randn(rng)); a = la * exp(0.2 * randn(rng))
        ph = [exp(-h) * h^i / factorial(big(i)) for i in 0:(max_g - 1)]
        pa = [exp(-a) * a^j / factorial(big(j)) for j in 0:(max_g - 1)]
        g = Float64.(ph * pa'); data[:, :, k] = g ./ sum(g)
    end
    return PP.ScoreMatrix(data)
end

_slate_spec() = PF.BookSpec(
    markets = PD.MarketConfig(PD.AbstractMarket[PD.Market1X2(), PD.MarketOverUnder(2.5)]),
    price = PF.DeArb(), shrink = PF.NoShrinkage())

"Quotes deliberately generous to the model, so every fixture carries a positive-edge leg."
function _quotes(match_id; boost = 1.10)
    DataFrame(match_id = fill(match_id, 5),
              market_name = ["1X2", "1X2", "1X2", "OverUnder", "OverUnder"],
              market_line = [0.0, 0.0, 0.0, 2.5, 2.5],
              selection = [:home, :draw, :away, :over_25, :under_25],
              odds_close = [2.40 * boost, 3.50, 3.20, 2.05 * boost, 1.95])
end

function _match_book(match_id; seed = 11, boost = 1.10)
    sm   = _score_matrix(seed = seed)
    spec = _slate_spec()
    probs = Dict(string(m) => PP.compute_market_probs(sm, m)
                 for m in [PD.Market1X2(), PD.MarketOverUnder(2.5)])
    sels = PF.extract_selections(_quotes(match_id; boost = boost), match_id, spec, probs)
    max_h, max_a, _ = size(sm.data)
    p = vec(mean(sm.data, dims = 3)[:, :, 1]); p ./= sum(p)
    R = PF.payoff_matrix(sels, max_h, max_a, spec.exec.commission)
    r = PF.allocate(spec.allocator, p, R, spec.exec)
    return PF.MatchBook(match_id, Date(2026, 9, 5), sels, p, R, nothing, r.a, 1.0,
                        r.kkt, r.converged)
end

_slate(n; boost = 1.10) =
    PF.Slate(Date(2026, 9, 5), [_match_book(100 + i; seed = 10 + i, boost = boost) for i in 1:n])

# ===================================================================
@testset "MatchDay live pipeline" begin

# ===================================================================
# 1. SLATE PRICING AND THE JOINT KELLY VECTOR
# ===================================================================

@testset "L1 the drawdown budget is ONE factor for the whole slate" begin
    # THE claim the whole architecture rests on. `SlateDrawdown` solves
    # Σ_t log E[(1 + k R_t)^-λ] <= 0 across every match at once and returns a SCALAR, so the
    # stake vector is only valid as a vector. `IsolatedDrawdown` returns one factor per match and
    # is the control: if these two agreed, the slate would not be the unit of anything.
    policy_slate = PF.PolicySpec(risk = PF.SlateDrawdown(20.0), cap = PF.FixedCap(0.25),
                                 trust = PF.FlatTrust(1.0))
    policy_iso   = PF.PolicySpec(risk = PF.IsolatedDrawdown(20.0), cap = PF.FixedCap(0.25),
                                 trust = PF.FlatTrust(1.0))
    sl  = _slate(5)
    ctx = PF.SlateContext(1, Date(2026, 9, 5), 1000.0)

    a_slate = PF.stake_slate(policy_slate, sl, ctx)
    a_iso   = PF.stake_slate(policy_iso,   sl, ctx)

    @test a_slate.k_risk isa Float64          # one number for five matches
    @test a_slate.exposure > 0
    @test a_slate.exposure <= 0.25 + 1e-9     # FixedCap is mandatory and it holds
    # the slate-wide budget is TIGHTER than five isolated ones, because the five settle together
    @test a_slate.exposure < a_iso.exposure
end

@testset "L2 a stake vector solved for 5 fixtures is not valid for 4 of them" begin
    # The correctness argument for the batch, as a measurement. Both coupling stages are checked
    # because they bite at different magnitudes.
    ctx = PF.SlateContext(1, Date(2026, 9, 5), 1000.0)

    # (a) THE CAP, when it binds. `apply_cap` rescales the WHOLE vector by cap/exposure, so
    #     dropping a fixture makes every remaining leg strictly LARGER. Exact and unmissable.
    capped = PF.PolicySpec(risk = PF.NoRisk(), cap = PF.FixedCap(0.05), trust = PF.FlatTrust(1.0))
    c4 = PF.stake_slate(capped, _slate(4), ctx)
    c5 = PF.stake_slate(capped, _slate(5), ctx)
    @test c4.capped && c5.capped
    @test c4.exposure ≈ 0.05 && c5.exposure ≈ 0.05
    live = findall(>(0), c5.stakes[1])
    @test !isempty(live)
    @test all(c4.stakes[1][live] .> c5.stakes[1][live])          # same legs, bigger stakes
    @test isapprox(c4.stakes[1][live] ./ c5.stakes[1][live],
                   fill(c4.stakes[1][live[1]] / c5.stakes[1][live[1]], length(live));
                   rtol = 1e-8)                                  # one common rescale, not a re-solve

    # (b) THE DRAWDOWN BUDGET, when the cap is slack. `k` is solved over every match at once, so
    #     it is recomputed when the slate changes and the shared fixtures' stakes move with it.
    #     Measured here at ~0.1%: these synthetic fixtures are near-identical, and the sequential
    #     penalty scales almost linearly in both L and stake magnitude, so k barely moves.
    #     The direction is NOT monotone in L -- k over n = 2..6 is
    #     0.09632, 0.09552, 0.09544, 0.09556, 0.09534 -- and asserting that it fell would be
    #     asserting a property of these fixtures rather than of the allocator.
    slack = PF.PolicySpec(risk = PF.SlateDrawdown(20.0), cap = PF.FixedCap(0.25),
                          trust = PF.FlatTrust(1.0))
    a4 = PF.stake_slate(slack, _slate(4), ctx)
    a5 = PF.stake_slate(slack, _slate(5), ctx)
    @test !isapprox(a4.k_risk, a5.k_risk; rtol = 1e-9)           # recomputed, not per-match
    @test !isapprox(a4.stakes[1], a5.stakes[1]; rtol = 1e-6)     # and fixture 1 moved
    @test a5.exposure > a4.exposure                             # more simultaneous risk on
end

@testset "L3 FixedCap binds the total, and refuses a cap that permits ruin" begin
    # There is deliberately no `NoCap`: the prototype without one lost 129.5% of bankroll on its
    # worst slate and flipped the sign of every subsequent compounding step. `0 < c < 1` enforced
    # in the constructor is what makes `slate_pl > -1` a theorem.
    @test_throws ArgumentError PF.FixedCap(1.0)
    @test_throws ArgumentError PF.FixedCap(0.0)

    tight = PF.PolicySpec(risk = PF.NoRisk(), cap = PF.FixedCap(0.02), trust = PF.FlatTrust(1.0))
    a = PF.stake_slate(tight, _slate(5), PF.SlateContext(1, Date(2026, 9, 5), 1000.0))
    @test a.capped
    @test a.exposure <= 0.02 + 1e-9
end

@testset "L4 risk_factor is homogeneous of degree 0 -- trust cannot resize a slate" begin
    # Counter-intuitive and load-bearing: once the drawdown constraint binds, trust and shrinkage
    # can only RESHAPE the book. A console control that "scales up the slate" is a no-op, which
    # is why none exists.
    policy(w) = PF.PolicySpec(risk = PF.SlateDrawdown(20.0), cap = PF.FixedCap(0.9),
                              trust = PF.FlatTrust(w))
    sl, ctx = _slate(4), PF.SlateContext(1, Date(2026, 9, 5), 1000.0)
    e = [PF.stake_slate(policy(w), sl, ctx).exposure for w in (0.25, 0.5, 1.0)]
    @test all(isapprox.(e, e[1]; rtol = 1e-6))
end

@testset "L5 sweep_ladder averages in probability space, not in odds" begin
    # The arithmetic mean of two decimal prices is NOT the price at which the combined stake
    # breaks even. Averaging them directly would overstate every book that filled deep.
    prices, sizes = [2.00, 1.50], [10.0, 10.0]
    s = MD.sweep_ladder(prices, sizes, 20.0)
    @test s.filled == 20.0
    @test s.vwap ≈ 20.0 / (10 / 2.0 + 10 / 1.5)          # = 1.714..., not 1.75
    @test s.vwap < 1.75
    @test s.slippage ≈ (2.0 - s.vwap) / 2.0
    @test s.levels == 2

    # partial fill: what the book had, and no more
    p = MD.sweep_ladder([2.0], [5.0], 20.0)
    @test p.filled == 5.0 && p.levels == 1
    # empty ladder is not an error and is not a zero-price fill
    e = MD.sweep_ladder(Float64[], Float64[], 20.0)
    @test e.filled == 0.0 && isnan(e.vwap)
end

@testset "L6 capacity is measured on the VENUE runner, not the model selection" begin
    # THE defect this pins down. On a synthetic the order touches the COMPLEMENT
    # (`Instrument.venue_key`); sweeping the model selection's ladder would measure depth on a
    # book the order never reaches. Measured on the 2026-08-08 ScottishLower slate, 14 of 48 legs
    # were synthetics.
    books = Dict((101, _key("OverUnder", 2.5, :over_25))  => _levels([2.05], [3.0], [2.10], [4.0]),
                 (101, _key("OverUnder", 2.5, :under_25)) => _levels([1.90], [500.0], [1.95], [900.0]))
    sheet = DataFrame(match_id = [101], group = ["OverUnder"], line = [2.5],
                      selection = [:over_25], venue_selection = [:under_25],
                      side = [:lay], venue_stake = [200.0])
    MD.annotate_capacity!(sheet, books)
    @test sheet.depth_touch[1] == 900.0      # the UNDER's lay side, which is what a lay eats
    @test sheet.fillable[1]
    @test sheet.fill_confidence[1] === :high

    # and pointing it at the model selection instead would have found £4
    direct = DataFrame(match_id = [101], group = ["OverUnder"], line = [2.5],
                       selection = [:over_25], venue_selection = [:over_25],
                       side = [:back], venue_stake = [200.0])
    MD.annotate_capacity!(direct, books)
    @test direct.depth_touch[1] == 3.0
    @test !direct.fillable[1]
    @test direct.fill_confidence[1] === :low
end

@testset "L7 fill confidence tiers, on the measured League Two 1X2 book" begin
    lv = _l2_home()                                    # £42 at 2.34, then £90, then £150
    @test MD.leg_capacity(lv, :back, 25.0).confidence  === :high    # touch covers it
    @test MD.leg_capacity(lv, :back, 25.0).slippage    |> isnan |> !
    big = MD.leg_capacity(lv, :back, 100.0)
    @test big.fillable                                              # 42 + 90 covers 100
    @test big.confidence === :medium                                # ...but not at the touch
    @test 0 < big.slippage < 0.01
    huge = MD.leg_capacity(lv, :back, 5_000.0)
    @test !huge.fillable && huge.confidence === :low
    # a missing book is `low`, never `missing` -- an unknown capacity is the same decision as none
    @test MD.leg_capacity(_levels(Float64[], Float64[], Float64[], Float64[]), :back, 10.0).confidence === :low
end

@testset "L8 relative_spread refuses to call a one-sided book tight" begin
    # A one-sided book is the WIDEST book there is, not the tightest. Returning 0.0 here would
    # make MaxSpread wave through exactly the fixtures it exists to refuse.
    @test MD.relative_spread(_levels([2.0], [10.0], [2.1], [10.0])) ≈ 0.1 / 2.05
    @test isnan(MD.relative_spread(_levels([2.0], [10.0], Float64[], Float64[])))
    @test isnan(MD.relative_spread(_levels(Float64[], Float64[], [2.1], [10.0])))
    @test isnan(MD.relative_spread(_levels([2.2], [10.0], [2.1], [10.0])))   # crossed
end

@testset "L9 MaxSpread catches the book MinMatched waves through" begin
    # Scottish League Two BTTS: a 9-tick book with ~£170 resting and ~£25 ever matched. Depth and
    # width are different failures; a depth-only floor passes the worse one.
    f = MD.Fixture(101, "clyde", "annan", DateTime(2026, 9, 5, 15), 57)
    card = MD.FixtureCard(f, MD.Resolved(f, "e", Dict("MATCH_ODDS" => "1.1"), true),
                          DateTime(2026, 9, 5, 14, 35))
    MD._set_card_meta!(card, :spread_median, 0.065)
    MD._set_card_meta!(card, :max_matched, 25.0)

    @test MD.ready(MD.MaxSpread(0.06; blocking = true), card) isa MD.Blocked
    @test MD.ready(MD.MaxSpread(0.08; blocking = true), card) isa MD.Ready
    @test MD.ready(MD.MaxSpread(0.06), card) isa MD.Ready          # non-blocking by default
    # unmeasured spread does not refuse when non-blocking, and does when asked to
    bare = MD.FixtureCard(f, MD.Resolved(f, "e", Dict{String,String}(), true), f.kickoff)
    @test MD.ready(MD.MaxSpread(0.06), bare) isa MD.Ready
    @test MD.ready(MD.MaxSpread(0.06; blocking = true), bare) isa MD.Blocked
end

@testset "L10 canonical_markets prices what actually trades" begin
    ms = MD.canonical_markets().markets
    names = Set(string.(ms))
    @test length(ms) == 6                                  # 1X2 + BTTS + O/U 0.5..3.5
    @test any(m -> m isa PD.Market1X2, ms)
    @test any(m -> m isa PD.MarketBTTS, ms)
    lines = sort([m.line for m in ms if m isa PD.MarketOverUnder])
    @test lines == [0.5, 1.5, 2.5, 3.5]
    # O/U 4.5 is deliberately absent: it prices but does not trade in the lower divisions
    @test !any(m -> m isa PD.MarketOverUnder && m.line == 4.5, ms)
end

# ===================================================================
# 2. THE ORDER STATE MACHINE  (pure, no database)
# ===================================================================

@testset "L11 TRIGGERED moves no money; PENDING_SUBMISSION is where liability appears" begin
    # Pricing writes rows and moves nothing. That is what lets a slate be priced, reviewed and
    # abandoned with the bankroll untouched -- and it is why `has_exposure` is a function of the
    # state rather than of whether a row exists.
    @test !MD.has_exposure(MD.TRIGGERED)
    @test  MD.has_exposure(MD.PENDING_SUBMISSION)
    @test  MD.has_exposure(MD.MATCHED)
    @test !MD.has_exposure(MD.CANCELLED)
    @test  MD.is_terminal(MD.SETTLED) && MD.is_terminal(MD.CANCELLED)
    @test !MD.is_terminal(MD.SUBMITTED)

    o = _order()
    t = MD.decide_order(o, MD.EntryGates(min_edge = 0.0, min_venue_stake = 1.0))
    @test t.to === MD.PENDING_SUBMISSION
    @test t.delta !== nothing
    @test t.delta.kind === :RESERVE
    @test t.delta.delta_balance ≈ -26.0 && t.delta.delta_reserved ≈ 26.0
end

@testset "L12 every refusal is a value, and collects EVERY reason" begin
    # `MatchDayResult.blocked` exists so "no bets today" and "the pipeline is broken" are
    # distinguishable. A ledger that dropped the reason would put that back out of reach.
    o = _order(risk = 0.4, venue_stake = 0.4, edge = -0.01)
    t = MD.decide_order(o, MD.EntryGates(min_edge = 0.0, min_venue_stake = 1.0); spread = 0.2)
    @test t.to === MD.CANCELLED
    @test t.delta === nothing                              # a refusal moves nothing
    @test occursin("edge", t.reason)
    @test occursin("minimum", t.reason)
    @test occursin("spread", t.reason)                     # all three, not the first
end

@testset "L13 a partial fill releases the remainder, and only the remainder" begin
    # Failing to release does not lose money but understates free equity for the rest of the
    # slate, which silently shrinks the next reservation.
    o = _order(state = MD.SUBMITTED, risk = 26.0)
    half = [MD.Fill(order_id = o.order_id, filled_at = o.quote_ts, price = 2.34, size = 10.0,
                    risk_filled = 10.0, model = :touch_only)]
    t = MD.decide_order(o; fills = half)
    @test t.to === MD.PARTIALLY_MATCHED
    @test t.delta.kind === :RELEASE
    @test t.delta.delta_balance ≈ 16.0 && t.delta.delta_reserved ≈ -16.0

    full = [MD.Fill(order_id = o.order_id, filled_at = o.quote_ts, price = 2.34, size = 26.0,
                    risk_filled = 26.0, model = :touch_only)]
    tf = MD.decide_order(o; fills = full)
    @test tf.to === MD.MATCHED && tf.delta === nothing     # nothing to give back

    none = MD.decide_order(o; fills = MD.Fill[])
    @test none.to === MD.CANCELLED
    @test none.delta.delta_balance ≈ 26.0                  # the WHOLE reservation comes back
end

@testset "L14 a transition computed against a stale read is refused" begin
    # This is the race the reservation lock exists to prevent. Applying it silently would defeat
    # the lock entirely.
    o = _order()
    t = MD.decide_order(o)
    o2 = MD.apply_transition(o, t)
    @test o2.state === MD.PENDING_SUBMISSION
    @test_throws ErrorException MD.apply_transition(o2, t)      # `from` no longer matches
    @test_throws ErrorException MD.apply_transition(_order(), t) # different order_id
end

@testset "L15 the account invariant is enforced, in both directions" begin
    a = _account(balance = 100.0, reserved = 0.0)
    a1 = MD.apply_delta(a, MD.LedgerDelta(kind = :RESERVE, account_id = "t",
                                          delta_balance = -30.0, delta_reserved = 30.0))
    @test a1.balance ≈ 70.0 && a1.reserved ≈ 30.0
    @test MD.equity(a1) ≈ MD.equity(a)                     # a reservation moves, never destroys

    # releasing more than was ever held is a bug, not an overdraft
    @test_throws ErrorException MD.apply_delta(a1,
        MD.LedgerDelta(kind = :RELEASE, account_id = "t",
                       delta_balance = 60.0, delta_reserved = -60.0))
    # and a delta for another account is not a transfer
    @test_throws ErrorException MD.apply_delta(a1,
        MD.LedgerDelta(kind = :RESERVE, account_id = "other",
                       delta_balance = -1.0, delta_reserved = 1.0))
end

@testset "L16 reserve_plan commits the vector WHOLE or not at all" begin
    # It deliberately does not scale down to fit. FixedCap already had the rescaling job at
    # pricing time with the whole book in hand; a second, blinder rescale here would produce
    # stakes no allocator authorised and a k_risk that no longer describes them.
    a = _account(balance = 2400.0, cap = 0.25)
    ok = [_order(risk = 100.0, venue_stake = 100.0) for _ in 1:5]
    p = MD.reserve_plan(a, ok)
    @test p.ok && p.total_risk ≈ 500.0
    @test length(p.admitted) == 5 && isempty(p.refused)

    over = [_order(risk = 200.0, venue_stake = 200.0) for _ in 1:5]   # 1000 > 0.25 * 2400
    q = MD.reserve_plan(a, over)
    @test !q.ok
    @test occursin("cap", q.reason)
    @test q.total_risk ≈ 1000.0                    # reported, but NOT scaled to fit
    @test length(q.admitted) == 5                  # the legs passed their own gates...
                                                   # ...and the SLATE still refused

    # refused legs contribute no risk: reserving for them would overstate exposure
    mixed = vcat([_order(risk = 100.0, venue_stake = 100.0) for _ in 1:3],
                 [_order(risk = 100.0, venue_stake = 0.2)   for _ in 1:2])
    m = MD.reserve_plan(a, mixed)
    @test m.ok && m.total_risk ≈ 300.0
    @test length(m.refused) == 2

    # and it refuses to mix two slates under one exposure assert
    @test_throws ErrorException MD.reserve_plan(a, [_order(slate = UUID(1)),
                                                    _order(slate = UUID(2))])
end

@testset "L17 free balance, not just the cap, bounds the slate" begin
    a = _account(balance = 50.0, reserved = 2000.0, cap = 0.9)
    p = MD.reserve_plan(a, [_order(risk = 100.0, venue_stake = 100.0)])
    @test !p.ok
    @test occursin("free balance", p.reason)
end

# ===================================================================
# 3. FILL SIMULATION AND SETTLEMENT
# ===================================================================

@testset "L18 TouchOnly is the default and it is the pessimistic one" begin
    # LadderSweep assumes we cross three levels instantly, which is what a MARKET order does and
    # not what this system does. The gap between them is the whole capacity question, so it must
    # be a decision rather than a default.
    lv, at = _l2_home(), DateTime(2026, 9, 5, 14, 48)
    touch = MD.simulate_fill(MD.TouchOnly(), lv, :back, 100.0, 1.0, at)
    sweep = MD.simulate_fill(MD.LadderSweep(), lv, :back, 100.0, 1.0, at)
    opt   = MD.simulate_fill(MD.Optimistic(), lv, :back, 100.0, 1.0, at)

    @test MD.filled_size(touch) == 42.0        # the touch, and nothing more
    @test MD.filled_size(sweep) == 100.0       # 42 + 58 of the 90 behind it
    @test MD.filled_size(opt)   == 100.0
    @test MD.fill_vwap(opt) == 2.34            # ...but at the touch price, which is the fiction
    @test MD.fill_vwap(sweep) < 2.34
    @test length(sweep) == 2                   # two price levels, two rows
    @test all(f -> f.model === :ladder_sweep_v1, sweep)
end

@testset "L19 LadderSweep stops at max_slippage rather than chasing" begin
    lv = _levels([2.00, 1.50], [10.0, 10_000.0], [2.10], [10.0])
    fills = MD.simulate_fill(MD.LadderSweep(max_slippage = 0.05), lv, :back, 500.0, 1.0,
                             DateTime(2026, 9, 5, 14, 48))
    @test MD.filled_size(fills) == 10.0        # level 2 is 25% away; it is not taken
    @test length(fills) == 1
end

@testset "L20 a lay eats the ask side, and risk_filled is not size" begin
    # Betfair quotes BOTH sides' sizes as backer stake, which is the denomination of venue_stake.
    # `risk_filled = size / leverage` is the liability, and it is what the account releases
    # against -- releasing `size` on a lay would release 3.8x too much at 1.26.
    lv = _levels([1.24], [50.0], [1.26], [77.0])
    lev = 1 / (1.26 - 1)                                     # 3.846...
    fills = MD.simulate_fill(MD.TouchOnly(), lv, :lay, 77.0, lev, DateTime(2026, 9, 5, 14, 48))
    @test length(fills) == 1
    @test fills[1].price == 1.26                             # the ASK, not the bid
    @test fills[1].size == 77.0
    @test fills[1].risk_filled ≈ 77.0 / lev ≈ 20.02
    @test MD.filled_risk(fills) < MD.filled_size(fills)
end

@testset "L21 settlement is denominated in risk, so it never branches on side" begin
    # The morphism denominated a lay's effective odds as d/(d-1) precisely so this arithmetic is
    # the same for both instruments. A back at 4.846 and a lay at 1.26 are the SAME position.
    back = _order(side = :back, odds = 4.846, leverage = 1.0, risk = 10.0, venue_stake = 10.0,
                  group = "OverUnder", line = 2.5, selection = :over_25,
                  venue_selection = :over_25)
    lay  = _order(side = :lay, odds = 1.26, leverage = 1 / 0.26, risk = 10.0,
                  venue_stake = 10.0 / 0.26, group = "OverUnder", line = 2.5,
                  selection = :over_25, venue_selection = :under_25)
    @test lay.effective_odds ≈ 1.26 / 0.26

    f(o) = [MD.Fill(order_id = o.order_id, filled_at = o.quote_ts, price = o.venue_odds,
                    size = o.venue_stake, risk_filled = 10.0, model = :touch_only)]
    # 3 goals -> Over 2.5 wins, for both instruments
    sb = MD.settle_order(back, f(back), 2, 1, 0.0)
    sl = MD.settle_order(lay,  f(lay),  2, 1, 0.0)
    @test sb.outcome === :win && sl.outcome === :win
    @test isapprox(sb.net_pnl, sl.net_pnl; rtol = 1e-3)      # 4.846 vs 1.26/0.26 = 4.846
    # 1 goal -> both lose exactly the risk, never the venue stake
    @test MD.settle_order(back, f(back), 1, 0, 0.0).net_pnl == -10.0
    @test MD.settle_order(lay,  f(lay),  1, 0, 0.0).net_pnl == -10.0
end

@testset "L22 grading is on the MODEL selection, never the venue runner" begin
    # Grading the runner the order touched would invert every synthetic -- the same distinction
    # `Instrument.venue_key` exists for, appearing one last time at the end of the lifecycle.
    @test MD.grade_selection("OverUnder", 2.5, :over_25,  2, 1) === :win
    @test MD.grade_selection("OverUnder", 2.5, :under_25, 2, 1) === :lose
    @test MD.grade_selection("1X2", 0.0, :home, 2, 1) === :win
    @test MD.grade_selection("1X2", 0.0, :draw, 1, 1) === :win
    @test MD.grade_selection("1X2", 0.0, :away, 1, 1) === :lose
    @test MD.grade_selection("BTTS", 0.0, :btts_yes, 1, 1) === :win
    @test MD.grade_selection("BTTS", 0.0, :btts_no,  1, 0) === :win
    @test MD.grade_selection("CorrectScore", 0.0, :cs_1_1, 1, 1) === :void

    # a synthetic: the model holds over_25 and the order touched under_25
    lay = _order(side = :lay, odds = 1.26, leverage = 1 / 0.26, group = "OverUnder", line = 2.5,
                 selection = :over_25, venue_selection = :under_25)
    @test MD.grade_selection(lay.market_group, lay.market_line, lay.selection, 2, 1) === :win
    @test MD.grade_selection(lay.market_group, lay.market_line, lay.venue_selection, 2, 1) === :lose
end

@testset "L23 a partially matched leg settles only what filled" begin
    # The remainder was released at execution; settling it here would create money.
    o = _order(risk = 26.0, odds = 2.34)
    part = [MD.Fill(order_id = o.order_id, filled_at = o.quote_ts, price = 2.34, size = 10.0,
                    risk_filled = 10.0, model = :touch_only)]
    s = MD.settle_order(o, part, 2, 1, 0.0)
    @test s.risk_settled == 10.0
    @test s.net_pnl ≈ 10.0 * (2.34 - 1)
    @test MD.settle_order(o, part, 0, 1, 0.0).net_pnl == -10.0     # not -26
    @test MD.settle_order(o, MD.Fill[], 2, 1, 0.0).net_pnl == 0.0  # nothing filled, nothing owed
end

@testset "L24 commission is charged on winnings only" begin
    o = _order(risk = 100.0, odds = 3.0)
    f = [MD.Fill(order_id = o.order_id, filled_at = o.quote_ts, price = 3.0, size = 100.0,
                 risk_filled = 100.0, model = :touch_only)]
    w = MD.settle_order(o, f, 2, 1, 0.02)
    @test w.commission ≈ 0.02 * 200.0
    @test w.net_pnl ≈ 200.0 - 4.0
    l = MD.settle_order(o, f, 0, 1, 0.02)
    @test l.commission == 0.0 && l.net_pnl == -100.0
end

@testset "L25 mark-to-market is against the price the position could be CLOSED at" begin
    # Marking against the side we entered on flatters every position by the full spread, which on
    # Scottish League Two's 5-tick 1X2 book is 4-5% of the position -- larger than the edge.
    o = _order(risk = 100.0, odds = 2.34, side = :back)
    f = [MD.Fill(order_id = o.order_id, filled_at = o.quote_ts, price = 2.34, size = 100.0,
                 risk_filled = 100.0, model = :touch_only)]
    @test MD.mark_to_market(o, f, 2.34, 2.34) ≈ 0.0                # unchanged book, no PnL
    @test MD.mark_to_market(o, f, 2.00, 2.05) > 0                  # price shortened: we gained
    @test MD.mark_to_market(o, f, 2.80, 2.90) < 0                  # drifted: we lost
    # the conservative mark is never kinder than the mid
    @test MD.mark_to_market(o, f, 2.00, 2.05) <=
          MD.mark_to_market(o, f, 2.00, 2.05; conservative = false)
    @test MD.mark_to_market(o, MD.Fill[], 2.0, 2.05) == 0.0        # nothing filled, nothing marked
end

@testset "L26 CLV is measured at the FILL, not at the quote" begin
    # A leg that filled two ticks worse has spent that difference. Measuring against the price we
    # wanted would credit execution with a fill it did not achieve.
    o = _order(risk = 100.0, odds = 2.34)
    good = [MD.Fill(order_id = o.order_id, filled_at = DateTime(2026, 9, 5, 14, 48),
                    price = 2.34, size = 100.0, risk_filled = 100.0, model = :touch_only)]
    worse = [MD.Fill(order_id = o.order_id, filled_at = DateTime(2026, 9, 5, 14, 48),
                     price = 2.20, size = 100.0, risk_filled = 100.0, model = :touch_only)]
    close_prob = 1 / 2.20
    cg = MD.clv_for_order(o, good,  close_prob, DateTime(2026, 9, 5, 15))
    cw = MD.clv_for_order(o, worse, close_prob, DateTime(2026, 9, 5, 15))
    @test cg.entry_prob ≈ 1 / 2.34
    @test cw.entry_prob ≈ 1 / 2.20
    @test cg.clv > 0 && cg.beat_close                # filled at 2.34, closed at 2.20
    @test isapprox(cw.clv, 0.0; atol = 1e-12)        # filled AT the close: no value either way
    @test cg.entry_lead_min == 12                    # T-12, the recommended entry instant
end

# ===================================================================
# 4. THE PAPER LEDGER  (PostgreSQL)
# ===================================================================

_PAPER_TEST_SCHEMA = "paper_test_" * string(uuid4())[1:8]

function _try_conn()
    haskey(ENV, "BF_DB_URL") || return nothing
    try
        return MD.paper_connection()
    catch e
        @info "ledger tests skipped: cannot reach BF_DB_URL" error = sprint(showerror, e)
        return nothing
    end
end

let conn = _try_conn()
if conn === nothing
    @info "SKIPPING the PostgreSQL ledger testsets -- BF_DB_URL unset or unreachable. " *
          "The pure layers above cover the state machine, the fill model and settlement; " *
          "these cover persistence and the FOR UPDATE reservation."
    @testset "L27-L31 ledger (skipped, no database)" begin
        @test_skip false
    end
else
try
    S = _PAPER_TEST_SCHEMA
    MD.migrate_paper_schema!(conn; schema = S)

    acct_id = "test_" * string(uuid4())[1:8]
    MD.ensure_account!(conn, MD.PaperAccount(account_id = acct_id, opening_balance = 2400.0,
                                             balance = 2400.0, max_slate_exposure = 0.25);
                       schema = S)

    "Insert a slate header directly -- `price_slate` needs a DataStore and a trained fit."
    function _mk_slate(sid::UUID; as_of = DateTime(2026, 9, 5, 14, 35), total = 0.0)
        LibPQ.execute(conn, """
            INSERT INTO $S.paper_slates
                (slate_id, account_id, slate_window, as_of, bankroll, batch_status,
                 k_risk, slate_exposure, exposure_cap, risk_lambda, capped, total_risk)
            VALUES (\$1,\$2,\$3,\$4,\$5,'PRICED',0.0412,0.077,0.25,20.0,false,\$6)
            ON CONFLICT (account_id, slate_window, as_of) DO NOTHING;""",
            (string(sid), acct_id, Date(2026, 9, 5), as_of, 2400.0, total))
        # Mirrors `insert_slate!`: return what the database HOLDS for this instant, not the id we
        # offered. On conflict those differ, and returning the offered one would hand every
        # caller a slate_id with no row behind it.
        df = DataFrame(LibPQ.execute(conn,
            """SELECT slate_id FROM $S.paper_slates
               WHERE account_id = \$1 AND slate_window = \$2 AND as_of = \$3;""",
            (acct_id, Date(2026, 9, 5), as_of)))
        return UUID(String(first(df).slate_id))
    end

    _legs(sid, n; risk = 20.0) =
        [_order(slate = sid, acct = acct_id, risk = risk, venue_stake = risk,
                match_id = 100 + i, oid = uuid4()) for i in 1:n]

@testset "L27 the ledger is the only writer of the account, and it reconciles" begin
    sid = _mk_slate(uuid4())
    a0 = MD.account_row(conn, acct_id; schema = S)
    @test a0.balance ≈ 2400.0 && a0.reserved ≈ 0.0

    d = MD.LedgerDelta(kind = :RESERVE, account_id = acct_id, delta_balance = -100.0,
                       delta_reserved = 100.0, slate_id = sid, note = "unit")
    a1 = MD.post_ledger!(conn, d; schema = S)
    @test a1.balance ≈ 2300.0 && a1.reserved ≈ 100.0

    r = MD.reconcile_account(conn, acct_id; schema = S)
    @test r.ok
    @test r.ledger_balance_delta ≈ r.account_balance_delta
    @test r.ledger_reserved ≈ r.account_reserved

    MD.post_ledger!(conn, MD.LedgerDelta(kind = :RELEASE, account_id = acct_id,
                                         delta_balance = 100.0, delta_reserved = -100.0,
                                         slate_id = sid, note = "undo"); schema = S)
    @test MD.account_row(conn, acct_id; schema = S).balance ≈ 2400.0
    @test MD.reconcile_account(conn, acct_id; schema = S).ok
end

@testset "L28 idempotency: re-pricing and re-inserting cannot double-stake" begin
    # Three mechanisms, and they are what let a crashed match-day process be restarted.
    as_of = DateTime(2026, 9, 5, 14, 30)
    sid  = _mk_slate(uuid4(); as_of = as_of)
    # A SECOND pricing run of the same instant must not make a second slate. `insert_slate!`
    # returns the EXISTING id, which is what lets a crashed match-day process be restarted --
    # so the retry's own uuid is discarded, exactly as here.
    sid2 = _mk_slate(uuid4(); as_of = as_of)
    @test sid2 == sid
    @test nrow(MD.slate_row(conn, sid; schema = S)) == 1

    orders = _legs(sid, 4)
    MD.insert_orders!(conn, orders; schema = S)
    MD.insert_orders!(conn, orders; schema = S)          # the retry
    @test nrow(MD.order_rows(conn, sid; schema = S)) == 4

    # a DIFFERENT order_id for the same (slate, match, market, line, selection) is still one leg
    dup = MD.PaperOrder(order_id = uuid4(), slate_id = sid, account_id = acct_id,
                        match_id = orders[1].match_id, kickoff = orders[1].kickoff,
                        market_group = orders[1].market_group,
                        market_line = orders[1].market_line, selection = orders[1].selection,
                        venue_selection = orders[1].venue_selection, side = :back,
                        venue_odds = 9.99, leverage = 1.0, effective_odds = 9.99,
                        p_model = 0.5, p_market = 0.4, edge = 0.1, stake_fraction = 0.1,
                        risk = 999.0, venue_stake = 999.0, quote_ts = orders[1].quote_ts)
    MD.insert_orders!(conn, [dup]; schema = S)
    @test nrow(MD.order_rows(conn, sid; schema = S)) == 4
end

@testset "L29 execute_slate_batch! is atomic: the whole vector or nothing" begin
    sid = uuid4(); _mk_slate(sid; as_of = DateTime(2026, 9, 5, 14, 36), total = 80.0)
    MD.insert_orders!(conn, _legs(sid, 4; risk = 20.0); schema = S)

    before = MD.account_row(conn, acct_id; schema = S)
    res = MD.execute_slate_batch!(conn, acct_id, sid; schema = S)
    @test res.status === MD.RESERVED
    @test res.n_admitted == 4 && res.n_refused == 0
    @test res.reserved ≈ 80.0
    after = MD.account_row(conn, acct_id; schema = S)
    @test after.balance ≈ before.balance - 80.0
    @test after.reserved ≈ before.reserved + 80.0
    @test MD.equity(after) ≈ MD.equity(before)          # a reservation moves, never destroys

    # every leg moved together
    st = MD.slate_orders(conn, sid; schema = S)
    @test all(o -> o.state === MD.PENDING_SUBMISSION, st)
    @test String(first(MD.slate_row(conn, sid; schema = S)).batch_status) == "RESERVED"

    # ONE ledger row for the whole slate, not one per leg
    led = MD.ledger_rows(conn, acct_id; schema = S)
    reserves = filter(r -> String(r.kind) == "RESERVE" && String(r.slate_id) == string(sid), led)
    @test nrow(reserves) == 1
    @test MD.reconcile_account(conn, acct_id; schema = S).ok
end

@testset "L30 a retry is a no-op, and double-reservation is unrepresentable" begin
    # The partial unique index on account_ledger(slate_id) WHERE kind='RESERVE' is what makes
    # this a property of the schema rather than of the caller's discipline.
    sid = uuid4(); _mk_slate(sid; as_of = DateTime(2026, 9, 5, 14, 37), total = 40.0)
    MD.insert_orders!(conn, _legs(sid, 2; risk = 20.0); schema = S)
    MD.execute_slate_batch!(conn, acct_id, sid; schema = S)
    a1 = MD.account_row(conn, acct_id; schema = S)

    again = MD.execute_slate_batch!(conn, acct_id, sid; schema = S)   # the retry storm
    @test occursin("no-op", again.reason)
    a2 = MD.account_row(conn, acct_id; schema = S)
    @test a2.balance ≈ a1.balance && a2.reserved ≈ a1.reserved

    # and the index itself refuses a second RESERVE row for this slate
    @test_throws Exception LibPQ.execute(conn, """
        INSERT INTO $S.account_ledger
          (account_id, kind, slate_id, delta_balance, delta_reserved, balance_after,
           reserved_after)
        VALUES ('$acct_id','RESERVE','$(string(sid))',-1,1,0,0);""")
    @test MD.reconcile_account(conn, acct_id; schema = S).ok
end

@testset "L31 an over-cap slate is ABANDONED with nothing reserved" begin
    sid = uuid4(); _mk_slate(sid; as_of = DateTime(2026, 9, 5, 14, 38), total = 5_000.0)
    MD.insert_orders!(conn, _legs(sid, 5; risk = 1_000.0); schema = S)

    before = MD.account_row(conn, acct_id; schema = S)
    res = MD.execute_slate_batch!(conn, acct_id, sid; schema = S)
    @test res.status === MD.ABANDONED
    @test res.reserved == 0.0
    @test occursin("cap", res.reason) || occursin("free balance", res.reason)

    after = MD.account_row(conn, acct_id; schema = S)
    @test after.balance ≈ before.balance && after.reserved ≈ before.reserved
    @test all(o -> o.state === MD.CANCELLED, MD.slate_orders(conn, sid; schema = S))
    @test MD.reconcile_account(conn, acct_id; schema = S).ok
end

@testset "L32 submit -> fill -> settle carries PnL to the account" begin
    sid = uuid4(); _mk_slate(sid; as_of = DateTime(2026, 9, 5, 14, 39), total = 40.0)
    orders = [_order(slate = sid, acct = acct_id, match_id = 201, risk = 20.0,
                     venue_stake = 20.0, odds = 2.34, selection = :home,
                     venue_selection = :home, oid = uuid4()),
              _order(slate = sid, acct = acct_id, match_id = 202, risk = 20.0,
                     venue_stake = 20.0, odds = 3.00, selection = :home,
                     venue_selection = :home, oid = uuid4())]
    MD.insert_orders!(conn, orders; schema = S)
    MD.execute_slate_batch!(conn, acct_id, sid; schema = S)

    books = Dict((201, _key("1X2", 0.0, :home)) => _levels([2.34], [42.0], [2.46], [30.0]),
                 (202, _key("1X2", 0.0, :home)) => _levels([3.00], [5.0],  [3.10], [30.0]))
    out = MD.submit_slate!(conn, sid, books, MD.TouchOnly(); schema = S)
    @test out.n_matched == 1        # 201 had £42 at the touch
    @test out.n_partial == 1        # 202 had only £5
    @test out.risk_filled ≈ 25.0
    # the unfilled £15 came back
    @test MD.account_row(conn, acct_id; schema = S).reserved ≈
          MD.reconcile_account(conn, acct_id; schema = S).ledger_reserved
    @test MD.reconcile_account(conn, acct_id; schema = S).ok

    a_before = MD.account_row(conn, acct_id; schema = S)
    st = MD.settle_slate!(conn, sid, Dict(201 => (2, 1), 202 => (0, 1)); schema = S)
    @test st.n_settled == 2
    # 201 wins 20 * 1.34 gross - 2% commission ; 202 loses its 5 filled
    @test st.total_pnl ≈ 20 * 1.34 * 0.98 - 5.0
    a_after = MD.account_row(conn, acct_id; schema = S)
    # `numeric(14,2)`: the ledger is denominated to the penny, so 20*1.34*0.98 = 26.264 is
    # stored as 26.26. Asserting to float precision here would be asserting that money has more
    # than two decimal places.
    @test isapprox(a_after.balance, a_before.balance + 20 + 20 * 1.34 * 0.98; atol = 0.01)
    @test isapprox(a_after.reserved, a_before.reserved - 25.0; atol = 1e-6)
    @test MD.reconcile_account(conn, acct_id; schema = S).ok
    @test all(o -> o.state === MD.SETTLED, MD.slate_orders(conn, sid; schema = S))
end

@testset "L33 kill_slate! releases unfilled liability and leaves fills alone" begin
    sid = uuid4(); _mk_slate(sid; as_of = DateTime(2026, 9, 5, 14, 40), total = 60.0)
    MD.insert_orders!(conn, _legs(sid, 3; risk = 20.0); schema = S)
    MD.execute_slate_batch!(conn, acct_id, sid; schema = S)
    before = MD.account_row(conn, acct_id; schema = S)

    res = MD.kill_slate!(conn, sid; schema = S)
    @test res.status === MD.KILLED
    after = MD.account_row(conn, acct_id; schema = S)
    @test after.balance ≈ before.balance + 60.0
    @test after.reserved ≈ before.reserved - 60.0
    @test all(o -> o.state === MD.CANCELLED, MD.slate_orders(conn, sid; schema = S))
    @test MD.reconcile_account(conn, acct_id; schema = S).ok
end

@testset "L34 recovery finds exactly the orders a crash would have stranded" begin
    sid = uuid4(); _mk_slate(sid; as_of = DateTime(2026, 9, 5, 14, 41), total = 40.0)
    MD.insert_orders!(conn, _legs(sid, 2; risk = 20.0); schema = S)
    MD.execute_slate_batch!(conn, acct_id, sid; schema = S)

    open = MD.recover_open_orders(conn, acct_id; schema = S, at = DateTime(2026, 9, 5, 14, 45))
    @test nrow(open) >= 2
    @test all(s -> s in ("PENDING_SUBMISSION", "SUBMITTED", "PARTIALLY_MATCHED"),
              String.(open.state))
    # after kick-off nothing is recoverable -- the window has gone
    @test nrow(MD.recover_open_orders(conn, acct_id; schema = S,
                                      at = DateTime(2026, 9, 5, 16))) == 0
    MD.kill_slate!(conn, sid; schema = S)
end

@testset "L35 drop_paper_schema! refuses to touch production" begin
    @test_throws ErrorException MD.drop_paper_schema!(conn; schema = MD.PAPER_SCHEMA)
end

finally
    try MD.drop_paper_schema!(conn; schema = _PAPER_TEST_SCHEMA) catch end
    try close(conn) catch end
end
end
end

# ===================================================================
# 5. THE OPERATOR CONSOLE
# ===================================================================

"A snapshot payload with no PricedSlate behind it, so the server is tested on its own."
_payload() = (
    at = string(now()),
    account = (account_id = "t", currency = "GBP", balance = 2224.55, reserved = 186.0,
               equity = 2410.55, free = 2224.55, max_slate_exposure = 0.25, is_live = false),
    batch = (slate_id = string(uuid4()), window = "2026-09-05", as_of = "2026-09-05T14:35:00",
             status = "PRICED", bankroll = 2410.55, total_risk = 186.0,
             slate_exposure = 0.0772, exposure_cap = 0.25, exposure_pct = 7.72,
             cap_pct = 25.0, k_risk = 0.0412, risk_lambda = 20.0, capped = false,
             n_fixtures = 21, n_legs = 34, n_blocked = 0, fold_idx = 38, warning = "",
             n_low_confidence = 3),
    cards = [(match_id = 101, home = "alloa", away = "montrose", kickoff = "2026-09-05T15:00:00",
              tournament_id = 56, minutes_to_kickoff = 25, lineup_source = "provisional",
              lineup_confirmed = true, lineup_lead_min = 29, risk = 26.0, n_legs = 1,
              ev_pct = 4.81,
              legs = [(selection = "over_25", market = "OverUnder", line = 2.5,
                       venue_selection = "under_25", side = "lay", venue_odds = 1.26,
                       effective_odds = 4.846, fair_odds = 1.96, p_model = 0.509,
                       p_market = 0.485, edge = 0.024, edge_pp = 2.4, ev_pct = 4.95,
                       risk = 26.0, venue_stake = 100.0, depth_touch = 900.0,
                       depth_book = 1500.0, slippage_pct = 0.0, fillable = true,
                       confidence = "high")])],
    blocked = [],
)

@testset "L36 the console API answers every documented route" begin
    st = MD.ConsoleState(_payload)
    port = 18_000 + (Int(rand(UInt16)) % 2_000)
    MD.serve_console(st; port = port, push = false)
    try
        base = "http://127.0.0.1:$port"

        page = HTTP.get("$base/"; retry = false)
        @test page.status == 200
        html = String(page.body)
        @test occursin("EXECUTE SLATE BATCH", html)
        @test occursin("tabular-nums", html)            # or every card jitters on each tick
        @test occursin("alpinejs", html)

        snap = JSON3.read(String(HTTP.get("$base/api/snapshot").body))
        @test snap.batch.n_legs == 34
        @test snap.batch.k_risk == 0.0412
        @test snap.account.equity == 2410.55
        @test length(snap.cards) == 1

        health = JSON3.read(String(HTTP.get("$base/api/health").body))
        @test health.ok == true

        # an unknown route lists what IS served rather than returning an empty body
        miss = HTTP.get("$base/api/nope"; status_exception = false)
        @test miss.status == 404
        @test occursin("routes", String(miss.body))
    finally
        MD.stop_console!(st)
    end
end

@testset "L37 the browser is not in the trust path" begin
    # The page POSTs an INTENT. Without an executor wired in, the console says so loudly rather
    # than appearing to work -- a button that silently does nothing is worse than no button.
    st = MD.ConsoleState(_payload)
    port = 18_000 + (Int(rand(UInt16)) % 2_000)
    MD.serve_console(st; port = port, push = false)
    try
        r = JSON3.read(String(HTTP.post("http://127.0.0.1:$port/api/execute";
                                        status_exception = false).body))
        @test r.ok == false
        @test occursin("no executor", r.error)
    finally
        MD.stop_console!(st)
    end

    # and a wired executor is reached, once, with the server validating rather than the page
    calls = Ref(0)
    st2 = MD.ConsoleState(_payload;
                          on_execute = () -> (calls[] += 1; (ok = true, note = "reserved 34 legs")))
    port2 = 18_000 + (Int(rand(UInt16)) % 2_000)
    MD.serve_console(st2; port = port2, push = false)
    try
        r = JSON3.read(String(HTTP.post("http://127.0.0.1:$port2/api/execute").body))
        @test r.ok == true && r.note == "reserved 34 legs"
        @test calls[] == 1
    finally
        MD.stop_console!(st2)
    end
end

@testset "L38 an intent that throws is REPORTED, never a dropped connection" begin
    # At T-12 a spinner with no reason is indistinguishable from the process having died.
    st = MD.ConsoleState(_payload; on_execute = () -> error("book is 41 minutes stale"))
    port = 18_000 + (Int(rand(UInt16)) % 2_000)
    MD.serve_console(st; port = port, push = false)
    try
        r = JSON3.read(String(HTTP.post("http://127.0.0.1:$port/api/execute";
                                        status_exception = false).body))
        @test r.ok == false
        @test occursin("41 minutes stale", r.error)
    finally
        MD.stop_console!(st)
    end
end

@testset "L39 a WebSocket client receives the snapshot without asking" begin
    st = MD.ConsoleState(_payload; interval = 0.2)
    port = 18_000 + (Int(rand(UInt16)) % 2_000)
    MD.serve_console(st; port = port, push = true)
    try
        got = Ref{Any}(nothing); n = Ref(0)
        HTTP.WebSockets.open("ws://127.0.0.1:$port/ws") do ws
            for msg in ws
                got[] = JSON3.read(String(msg)); n[] += 1
                n[] >= 2 && break            # first frame on connect, then a pushed one
            end
        end
        @test n[] >= 2
        @test got[].batch.n_legs == 34
    finally
        MD.stop_console!(st)
    end
    @test st.server === nothing              # stop_console! is idempotent and releases the port
    MD.stop_console!(st)
end

@testset "L40 the snapshot sorts cards by EV and keeps the blocked report" begin
    # Sorting is risk-weighted so a £2 leg at +9% cannot outrank a £26 one at +5%; and `blocked`
    # is not optional, or "no bets today" and "the pipeline is broken" look the same again.
    sheet = DataFrame(
        slate = fill(Date(2026, 9, 5), 3), match_id = [101, 102, 102],
        family = fill("f", 3), group = ["1X2", "1X2", "OverUnder"], line = [0.0, 0.0, 2.5],
        selection = [:home, :home, :over_25], venue_selection = [:home, :home, :over_25],
        odds_quoted = [2.34, 2.10, 2.05], odds = [2.34, 2.10, 2.05],
        p_model = [0.451, 0.500, 0.510], p_market = [0.427, 0.476, 0.488],
        edge = [0.024, 0.024, 0.022], frac = [0.01, 0.001, 0.02],
        stake = [26.0, 2.0, 30.0], k_risk = fill(0.04, 3), slate_exposure = fill(0.077, 3),
        capped = fill(false, 3), settled = fill(false, 3), side = fill(:back, 3),
        venue_odds = [2.34, 2.10, 2.05], risk = [26.0, 2.0, 30.0],
        venue_stake = [26.0, 2.0, 30.0],
        depth_touch = fill(100.0, 3), depth_book = fill(500.0, 3),
        expected_fill = [26.0, 2.0, 30.0], expected_vwap = [2.34, 2.10, 2.05],
        expected_slippage = fill(0.0, 3), fillable = fill(true, 3),
        fill_confidence = [:high, :high, :medium])

    f1 = MD.Fixture(101, "alloa", "montrose", DateTime(2026, 9, 5, 15), 56)
    f2 = MD.Fixture(102, "forfar", "stirling", DateTime(2026, 9, 5, 15), 57)
    f3 = MD.Fixture(103, "clyde", "annan", DateTime(2026, 9, 5, 15), 57)
    c1 = MD.FixtureCard(f1, MD.Resolved(f1, "e1", Dict{String,String}(), true), DateTime(2026, 9, 5, 14, 35))
    c2 = MD.FixtureCard(f2, MD.Resolved(f2, "e2", Dict{String,String}(), true), DateTime(2026, 9, 5, 14, 35))
    c3 = MD.FixtureCard(f3, MD.Unresolved(f3, :absent_from_crosswalk), DateTime(2026, 9, 5, 14, 35))
    c3.readiness = MD.Blocked([:identity => "unresolved (absent_from_crosswalk)"])

    s = MD.PricedSlate(uuid4(), "t", Date(2026, 9, 5), DateTime(2026, 9, 5, 14, 35), 2400.0,
                       sheet, DataFrame(), [c1, c2, c3], [c3],
                       Dict{Tuple{Int,MD.SelectionKey},MD.Instrument}(),
                       Dict{Tuple{Int,MD.SelectionKey},MD.BookLevels}(),
                       0.0412, 0.0242, false, 20.0, 0.25, 58.0, 38, "")

    snap = MD.slate_snapshot(s, _account(balance = 2400.0))
    @test snap.batch.n_legs == 3 && snap.batch.n_fixtures == 2
    @test snap.batch.k_risk == 0.0412 && snap.batch.risk_lambda == 20.0
    @test snap.batch.exposure_pct == 2.42 && snap.batch.cap_pct == 25.0
    @test length(snap.blocked) == 1
    @test occursin("absent_from_crosswalk", snap.blocked[1].reasons[1])

    ev = [c.ev_pct for c in snap.cards]
    @test issorted(ev, rev = true)
    # fixture 102 mixes a £2 leg at +5.04% with a £30 leg at +4.51%; risk-weighting must not let
    # the small one dominate
    c102 = only(filter(c -> c.match_id == 102, snap.cards))
    @test c102.ev_pct < maximum(l.ev_pct for l in c102.legs)
    @test all(c -> haskey(c.legs[1], :p_model) && haskey(c.legs[1], :p_market), snap.cards)
    @test all(c -> c.legs[1].fair_odds ≈ round(1 / c.legs[1].p_model, digits = 3), snap.cards)
end

@testset "L41 slate_batch_summary reports exposure before bets" begin
    sheet = DataFrame(match_id = [101], group = ["1X2"], line = [0.0], selection = [:home],
                      risk = [26.0], fill_confidence = [:low])
    s = MD.PricedSlate(uuid4(), "t", Date(2026, 9, 5), DateTime(2026, 9, 5, 14, 35), 2400.0,
                       sheet, DataFrame(), MD.FixtureCard[], MD.FixtureCard[],
                       Dict{Tuple{Int,MD.SelectionKey},MD.Instrument}(),
                       Dict{Tuple{Int,MD.SelectionKey},MD.BookLevels}(),
                       0.0412, 0.0108, true, 20.0, 0.25, 26.0, 38, "boundary count changed")
    b = MD.slate_batch_summary(s)
    @test b.slate_exposure == 0.0108 && b.exposure_cap == 0.25
    @test b.capped
    @test b.k_risk == 0.0412 && b.risk_lambda == 20.0
    @test b.n_low_confidence == 1
    @test b.fold_idx == 38
end

end
