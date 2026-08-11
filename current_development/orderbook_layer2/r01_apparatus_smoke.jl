# current_development/orderbook_layer2/r01_apparatus_smoke.jl
#
# WP1 acceptance. Runs with no DB, no cache and no trained experiment.
#
# ---------------------------------------------------------------------------------------------
# WHY THIS EXISTS AND WHY IT RUNS BEFORE WP2/WP3
# ---------------------------------------------------------------------------------------------
#
# The Layer-2 apparatus makes exactly one claim that can be wrong quietly: that a tearsheet row
# means the same thing as a `Portfolio.report`. If `l2_curve` compounds differently from
# `Portfolio.simulate` -- by a commission, by a sort order, by using `cumsum` where the system
# uses `cumprod` -- every downstream wealth number is wrong and NOTHING complains.
#
# So the gate is not "does it run" but "does it AGREE", and it is checked against the real
# `Portfolio.simulate` rather than against a reimplementation of it. That is possible before the
# replay exists because `test/portfolio_tests.jl` already builds `MatchBook`s from a hand-rolled
# score grid, so a settled slate sequence is available with no data at all.
#
# A2 is the load-bearing test. Everything else is arithmetic.
#
# ---------------------------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------------------------
#
#   julia --project -e 'using BayesianFootball; include("current_development/orderbook_layer2/r01_apparatus_smoke.jl")'

using Test
using BayesianFootball
using DataFrames, Dates, Statistics, LinearAlgebra, Random, Printf

const PF = BayesianFootball.Portfolio
const PD = BayesianFootball.Data
const PP = BayesianFootball.Predictions
const BT = BayesianFootball.BackTesting

include(joinpath(@__DIR__, "l01_l2_experiment.jl"))
include(joinpath(@__DIR__, "l02_l2_ledger.jl"))
include(joinpath(@__DIR__, "l03_l2_metrics.jl"))

# ===================================================================
# Fixtures — lifted from test/portfolio_tests.jl so the two agree by construction
# ===================================================================

function fixture_score_matrix(; max_h = 8, max_a = 8, n_draws = 24, lh = 1.4, la = 1.1, seed = 7)
    rng = MersenneTwister(seed)
    data = zeros(Float64, max_h, max_a, n_draws)
    for k in 1:n_draws
        h = lh * exp(0.25 * randn(rng)); a = la * exp(0.25 * randn(rng))
        ph = [exp(-h) * h^i / factorial(big(i)) for i in 0:(max_h - 1)]
        pa = [exp(-a) * a^j / factorial(big(j)) for j in 0:(max_a - 1)]
        g = Float64.(ph * pa'); data[:, :, k] = g ./ sum(g)
    end
    return PP.ScoreMatrix(data)
end

function fixture_quotes(; match_id = 1, ou_overround = 1.04)
    p_over = 0.52
    DataFrame(match_id = fill(match_id, 5),
              market_name = ["1X2", "1X2", "1X2", "OverUnder", "OverUnder"],
              market_line = [0.0, 0.0, 0.0, 2.5, 2.5],
              selection = [:home, :draw, :away, :over_25, :under_25],
              odds_close = [2.40, 3.50, 3.20,
                            1 / (ou_overround * p_over), 1 / (ou_overround * (1 - p_over))])
end

fixture_spec() = PF.BookSpec(
    markets = PD.MarketConfig(PD.AbstractMarket[PD.Market1X2(), PD.MarketOverUnder(2.5)]),
    price = PF.DeArb(), shrink = PF.NoShrinkage())

function fixture_book(; match_id = 1, date = Date(2025, 1, 1), h = 2, a = 1)
    sm   = fixture_score_matrix()
    spec = fixture_spec()
    mk   = [PD.Market1X2(), PD.MarketOverUnder(2.5)]
    mp   = Dict(string(m) => PP.compute_market_probs(sm, m) for m in mk)
    sels = PF.extract_selections(fixture_quotes(match_id = match_id), match_id, spec, mp)
    max_h, max_a, _ = size(sm.data)
    p = vec(mean(sm.data, dims = 3)[:, :, 1]); p ./= sum(p)
    R = PF.payoff_matrix(sels, max_h, max_a, spec.exec.commission)
    r = PF.allocate(spec.allocator, p, R, spec.exec)
    return PF.MatchBook(match_id, date, sels, p, R,
                        PF.settle_vector(sels, h, a, spec.exec.commission),
                        r.a, 1.0, r.kkt, r.converged)
end

"""
A settled multi-slate book sequence: 6 fixtures over 3 settlement windows, with scores chosen
so the slates do not all win — a monotone bankroll would hide a compounding bug.
"""
function fixture_trajectory()
    books = [fixture_book(match_id = 1, date = Date(2025, 1, 1), h = 2, a = 1),
             fixture_book(match_id = 2, date = Date(2025, 1, 1), h = 0, a = 0),
             fixture_book(match_id = 3, date = Date(2025, 1, 8), h = 1, a = 3),
             fixture_book(match_id = 4, date = Date(2025, 1, 8), h = 3, a = 0),
             fixture_book(match_id = 5, date = Date(2025, 1, 15), h = 1, a = 1),
             fixture_book(match_id = 6, date = Date(2025, 1, 15), h = 0, a = 2)]
    policy = PF.PolicySpec(trust = PF.FlatTrust(0.5), risk = PF.SlateDrawdown(23.0),
                           cap = PF.FixedCap(0.25))
    slates = PF.group(policy.grouping, books)
    return policy, slates, PF.simulate(policy, slates)
end

"Turn a `Trajectory.bets` frame into a `Layer2Ledger` with the Layer-2 columns stubbed."
function ledger_from_trajectory(traj; drift_factor = 1.0)
    b = copy(traj.bets)
    rename!(b, :date => :slate)
    b.is_winner        = b.payoff .> 0
    b.group            = [split(f, "_")[1] for f in b.family]
    b.line             = zeros(nrow(b))
    b.tournament_id    = fill(79, nrow(b))
    b.as_of            = [DateTime(d) - Hour(1) for d in b.slate]
    b.mins_to_ko       = fill(60.0, nrow(b))
    b.policy_name      = fill("test", nrow(b))
    b.arm              = fill(:frozen, nrow(b))
    b.entry_name       = fill("AtClose()", nrow(b))
    b.odds_close_final = b.odds .* drift_factor
    b.fair_close       = 1.0 ./ (b.odds .* 1.03)
    b.back             = [[o, o * 0.98, o * 0.96] for o in b.odds]
    b.back_size        = [[50.0, 500.0, 5000.0] for _ in 1:nrow(b)]
    b.lay              = [[o * 1.02] for o in b.odds]
    b.lay_size         = [[100.0] for _ in 1:nrow(b)]
    b.rel_spread       = fill(0.02, nrow(b))
    b.matched          = fill(NaN, nrow(b))
    b.stake_cash       = b.stake .* 1000
    b.pnl_cash         = b.pnl .* 1000
    add_entry_buckets!(b)
    return Layer2Ledger(b)
end

# ===================================================================
@testset "WP1 Layer-2 apparatus" begin

policy, slates, traj = fixture_trajectory()
led = ledger_from_trajectory(traj)

@test !isempty(led)
@test length(traj.bankroll) == length(slates) + 1

# -------------------------------------------------------------------
@testset "A1 ledger preserves the trajectory's legs and units" begin
    @test nrow(led.df) == nrow(traj.bets)
    @test sum(led.df.stake) ≈ traj.total_stake
    @test sum(led.df.pnl)   ≈ traj.total_pl
    # stake is a FRACTION of bankroll, not currency -- the whole compounding chain depends on it
    @test all(0 .<= led.df.stake .<= 1)
end

# -------------------------------------------------------------------
@testset "A2 l2_curve reproduces Portfolio.simulate exactly" begin
    # THE load-bearing test. `cumsum` instead of `cumprod`, a missed sort, or a commission
    # applied twice all show up here and nowhere else.
    curve = l2_curve(led.df)
    @test length(curve) == length(traj.bankroll)
    @test curve ≈ traj.bankroll

    pm  = l2_path_metrics(led.df)
    ref = PF.path_metrics(traj)
    @test pm.final            ≈ ref.final
    @test pm.roi              ≈ ref.roi
    @test pm.growth_per_slate ≈ ref.growth_per_slate
    @test pm.mdd              ≈ ref.mdd
    @test pm.ulcer            ≈ ref.ulcer
    @test pm.calmar           ≈ ref.calmar
    @test pm.martin           ≈ ref.martin
    @test pm.n_slates         == length(slates)

    # and the wealth metrics agree with Portfolio.report, which is the actual bridge
    ms  = [BT.CumulativeWealth(), BT.SharpeRatio()]
    rep = PF.report(traj, ms)
    for m in ms
        @test BT.compute_metric(m, curve) ≈ rep[Symbol(BT.metric_name(m))]
    end
end

# -------------------------------------------------------------------
@testset "A3 slate ordering is enforced, not assumed" begin
    # Final wealth is order-invariant; every drawdown statistic is not. A shuffled ledger must
    # still produce the chronological curve.
    shuffled = led.df[shuffle(MersenneTwister(3), 1:nrow(led.df)), :]
    @test l2_curve(shuffled) ≈ traj.bankroll
end

# -------------------------------------------------------------------
@testset "A4 tearsheet agrees with Portfolio.attribution" begin
    att = l2_attribution(led)
    ref = PF.attribution(traj)
    @test nrow(att) == nrow(ref)
    for r in eachrow(ref)
        row = att[att.family .== r.family, :]
        @test nrow(row) == 1
        @test row.pnl[1]   ≈ r.pnl
        @test row.stake[1] ≈ r.stake
        @test row.roi[1]   ≈ r.roi
    end
end

# -------------------------------------------------------------------
@testset "A5 entry buckets label the measured boundaries" begin
    @test entry_bucket(0.0)   == "1_0-5m"
    @test entry_bucket(5.0)   == "1_0-5m"       # inclusive upper edge
    @test entry_bucket(5.1)   == "2_5-15m"
    @test entry_bucket(59.0)  == "4_30-60m"
    @test entry_bucket(60.0)  == "4_30-60m"
    @test entry_bucket(61.0)  == "5_60-120m"
    @test entry_bucket(400.0) == "7_180m+"
    # ordered lexicographically so `sort` gives clock order
    @test issorted([entry_bucket(x) for x in [1, 10, 20, 45, 90, 150, 300]])
end

# -------------------------------------------------------------------
@testset "A6 PriceDrift is the hand-computable value" begin
    # every close 10% longer than entry => drift = log(1.1) and waiting always paid
    l10 = ledger_from_trajectory(traj; drift_factor = 1.1)
    d = BT.compute_distributional_metric(PriceDrift(), l10.df)
    @test d.drift_n > 0
    @test d.drift_mean ≈ log(1.1) atol = 1e-6
    @test d.drift_wait_paid ≈ 100.0
    @test d.drift_ci_lo <= d.drift_mean <= d.drift_ci_hi

    # no movement => exactly zero drift and a degenerate interval
    d0 = BT.compute_distributional_metric(PriceDrift(), led.df)
    @test d0.drift_mean ≈ 0.0 atol = 1e-12
    @test d0.drift_wait_paid ≈ 0.0
end

# -------------------------------------------------------------------
@testset "A7 ClosingLineValue is the hand-computable value" begin
    # fair_close was stubbed as 1/(odds*1.03), so log(odds*fair) = -log(1.03) on every leg
    c = BT.compute_distributional_metric(ClosingLineValue(), led.df)
    @test c.clv_n > 0
    @test c.clv_mean ≈ -log(1.03) atol = 1e-6
    @test c.clv_pos ≈ 0.0                         # uniformly negative
    @test c.beat_close ≈ 0.0                      # no drift => never beat the close
end

# -------------------------------------------------------------------
@testset "A8 FillCost walks the ladder and detects shortfall" begin
    f = BT.compute_distributional_metric(FillCost(stakes = [10.0, 100.0, 1000.0]), led.df)
    @test f.fill_n > 0
    @test f.half_spread > 0                       # lay is 2% above back by construction
    # top level holds 50; 10 fills at the top, 100 and 1000 must walk down
    @test f.short_10   ≈ 0.0
    @test f.short_100  ≈ 100.0
    @test f.short_1000 ≈ 100.0
    @test f.slip_10 ≈ 0.0 atol = 1e-12            # no walking => no slippage
    @test f.slip_100 > f.slip_10                  # cost is monotone in size
    @test f.slip_1000 > f.slip_100
end

# -------------------------------------------------------------------
@testset "A9 _vwap is exact on a hand-computed ladder" begin
    prices = [2.0, 1.9, 1.8]; sizes = [50.0, 50.0, 1e9]
    @test _vwap(prices, sizes, 50.0, 3)  ≈ 2.0
    @test _vwap(prices, sizes, 100.0, 3) ≈ (50 * 2.0 + 50 * 1.9) / 100
    @test _vwap(prices, sizes, 150.0, 3) ≈ (50 * 2.0 + 50 * 1.9 + 50 * 1.8) / 150
    # an order the book cannot fill is charged at the WORST level, never dropped
    @test _vwap([2.0], [10.0], 100.0, 1) ≈ 2.0
end

# -------------------------------------------------------------------
@testset "A10 metrics return their documented keys when starved" begin
    # a groupby cell can legitimately be empty; a metric that errors there takes the tearsheet
    # with it, and a metric that returns different keys breaks the `combine` row shape.
    empty_df = similar(led.df, 0)
    for m in (PriceDrift(), ClosingLineValue(), FillCost())
        full  = BT.compute_distributional_metric(m, led.df)
        blank = BT.compute_distributional_metric(m, empty_df)
        @test keys(full) == keys(blank)
    end
end

# -------------------------------------------------------------------
@testset "A11 tearsheet runs end to end and flags the path warning" begin
    ts = l2_tearsheet(led; groupby_cols = [:policy_name],
                      dist_metrics = vcat(l2_metrics(), BT.BernoulliGammaHurdle()))
    @test nrow(ts) == 1
    @test ts.n_slates[1] == length(slates)
    @test ts.final[1] ≈ traj.bankroll[end]
    @test ts.matches[1] == length(unique(led.df.match_id))
    for k in (:drift_mean, :clv_mean, :half_spread, :roi_ci_lo, :hurdle_p)
        @test hasproperty(ts, k)
    end
    # 3 slates is far below MIN_SLATES_FOR_PATH, so the warning must fire
    @test !ts.path_reliable[1]
    @test !isempty(path_warning(ts))
    @test occursin("PATH METRICS UNRELIABLE", path_warning(ts))
end

# -------------------------------------------------------------------
@testset "A12 entry rules select one row per leg" begin
    # a 3-snapshot ledger for one leg, price improving toward kickoff
    base = led.df[1:1, :]
    rows = DataFrame[]
    for (mk, od, eg) in zip([180.0, 90.0, 5.0], [3.0, 2.5, 2.0], [0.10, 0.05, 0.01])
        r = copy(base); r.mins_to_ko = [mk]; r.odds = [od]; r.edge = [eg]
        push!(rows, r)
    end
    trace = add_entry_buckets!(reduce(vcat, rows))

    @test nrow(apply_entry(AtClose(), trace)) == 1
    @test apply_entry(AtClose(), trace).mins_to_ko[1] == 5.0

    @test apply_entry(FixedLead(Minute(90)), trace).mins_to_ko[1] == 90.0
    @test apply_entry(FixedLead(Minute(100)), trace).mins_to_ko[1] == 90.0   # snaps to nearest

    # oracle takes the best price anywhere in the window
    @test apply_entry(BestPrice(), trace).odds[1] == 3.0

    # first instant whose edge clears the bar, scanning from the earliest
    @test apply_entry(FirstQualifying(0.04), trace).mins_to_ko[1] == 180.0
    @test apply_entry(FirstQualifying(0.08), trace).mins_to_ko[1] == 180.0
    # a bar nothing clears drops the leg rather than falling through to a default row
    @test nrow(apply_entry(FirstQualifying(0.5), trace)) == 0
end

# -------------------------------------------------------------------
@testset "A13 recap_slates! binds only when the assembled book breaches" begin
    # single-instant rules must NEVER trip it -- that is the check that the repair is not
    # silently reshaping the baselines it is meant to leave alone.
    single = copy(led.df)
    recap_slates!(single, 1.0)
    @test !any(single.recapped)
    @test all(single.recap_factor .== 1.0)

    # an assembled book over the cap is scaled back, preserving relative Kelly weights.
    # Stakes are deliberately UNEQUAL so "ratios preserved" is a real claim rather than a
    # tautology about a constant vector.
    over = copy(led.df)
    over.stake = [0.05 * (1 + (i % 4)) for i in 1:nrow(over)]
    before = Dict(k.slate => copy(sub.stake) for (k, sub) in pairs(groupby(over, :slate)))
    recap_slates!(over, 0.25)

    @test any(over.recapped)
    for (k, sub) in pairs(groupby(over, :slate))
        @test sum(sub.stake) <= 0.25 + 1e-9
        b = before[k.slate]
        if sum(b) > 0.25                                  # this slate actually bound
            @test all(sub.recapped)
            # every leg scaled by the SAME factor => all pairwise ratios survive
            f = sub.stake ./ b
            @test all(≈(f[1], atol = 1e-12), f)
            @test f[1] ≈ 0.25 / sum(b)
        else
            @test !any(sub.recapped)
            @test sub.stake ≈ b
        end
    end
end

end # testset

println("\n", "="^70)
println("WP1 apparatus: all acceptance tests passed")
println("="^70)
