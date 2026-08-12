# current_development/orderbook_layer2/r03_replay_gate.jl
#
# WP3 acceptance. Does the decomposed replay still measure the pipeline?
#
# ---------------------------------------------------------------------------------------------
# THE CLAIM UNDER TEST
# ---------------------------------------------------------------------------------------------
#
# `l04_corpus_replay.jl` splits `match_day` into a cached Tier 1 and a cheap Tier 2 so that WP5
# and WP6 can re-stake without re-reading the database. That is only legitimate if the split is
# EXACT. `l02_slate_replay.jl:280-284` makes the point sharply for its own design: it calls the
# real `match_day` rather than a hand-rolled loop precisely because "an optimised replica that
# drifts from it measures the replica".
#
# So G1 is the gate. Everything else is supporting evidence.
#
# ---------------------------------------------------------------------------------------------
# GATES
# ---------------------------------------------------------------------------------------------
#
#   G1  stake_snapshots reproduces match_day's sheet ROW FOR ROW at the same instant
#   G2  the latent arms, measured with a test that CAN fail
#   G3  the adaptive grid is honoured by the fixtures it is built for
#   G3b the grid does not reach past the tradeable book
#   G4  the stamped close is coherent — fair probabilities sum to ~1 per market group
#   G5  entry rules behave on real data as they did on synthetic
#
# G2 previously asserted that `src_sup40_sw40`, being player-level, MUST have latents that move
# with `as_of` — and treated invariance as proof the ratings materialiser was inert. The first
# run disproved that: latents are bit-identical across the window, because serving latents are a
# pure function of `(fixture, split)`. The gate now asserts the measured behaviour, and asserts
# `n_compared > 0` first, because the helper it used to rely on could not fail.
#
# G3b exists because the failure it catches is invisible: an entry rule aimed at a lead the
# readiness gate refuses to price does not error, it snaps to a shallower instant and reports
# that instant's numbers under the deeper label.
#
# ---------------------------------------------------------------------------------------------
# USAGE (server; needs WP2's pinned stores and trained experiments)
# ---------------------------------------------------------------------------------------------
#
#   include("current_development/orderbook_layer2/r03_replay_gate.jl")

using Test
using BayesianFootball
using DataFrames, Dates, Statistics, Printf, Serialization

const PF = BayesianFootball.Portfolio
const MD = BayesianFootball.MatchDay
const DD = BayesianFootball.Data
const EE = BayesianFootball.Experiments

include(joinpath(@__DIR__, "l00_corpus.jl"))
include(joinpath(@__DIR__, "l01_l2_experiment.jl"))
include(joinpath(@__DIR__, "l02_l2_ledger.jl"))
include(joinpath(@__DIR__, "l03_l2_metrics.jl"))
# `replay_spec` and `grade!` are reused from the single-slate harness rather than reimplemented.
include(joinpath(@__DIR__, "..", "matchday_2026_08_08", "l02_slate_replay.jl"))
include(joinpath(@__DIR__, "l04_corpus_replay.jl"))

const ENGINE_DIR = "./data/l2_ireland_engines"

# ===================================================================
# Setup — pinned stores, trained experiments, one league
# ===================================================================

"Load the pin WP2 wrote. Never `load_datastore_cached` here — see trap T2."
function pinned(tag::String)
    p = joinpath(ENGINE_DIR, "ds_$(tag).jls")
    isfile(p) || error("r03: missing pin $p — run r02_train_ireland.jl first")
    return deserialize(p)
end

function newest_experiment(prefix::String)
    dirs = filter(d -> startswith(basename(d), prefix),
                  [joinpath(ENGINE_DIR, d) for d in readdir(ENGINE_DIR)
                   if isdir(joinpath(ENGINE_DIR, d))])
    isempty(dirs) && error("r03: no experiment under $ENGINE_DIR matching $prefix")
    return EE.load_experiment(sort(dirs, by = mtime, rev = true)[1])
end

println("\n", "="^90, "\nWP3 REPLAY GATE\n", "="^90)

ds79   = pinned("ire79")
expr79 = newest_experiment("l2_ire79_sup40_sw40")

corpus = build_corpus("ireland", [79, 718]; from = Date(2026, 5, 20), to = Date(2026, 8, 10))
c79    = subset_corpus(corpus, 79)
@printf("corpus 79: %d fixtures, %d slates\n", length(c79.fixtures), length(corpus_slates(c79)))

grid_rec = recommend_grid(c79; coverage = 0.80)
@printf("recommended grid: lookback %s, fine %s from %s, coarse %s (honoured %.0f%%)\n",
        grid_rec.lookback, grid_rec.fine_step, grid_rec.fine_from, grid_rec.coarse_step,
        100 * grid_rec.honoured)
@printf("  (first-tick lookback would have been %s — see live_lead_min in l00)\n",
        grid_rec.first_tick_lookback)

# One representative slate — the busiest, so the comparison has the most rows to disagree on.
# NB `big_slate`, not `big`: a top-level binding named `big` in Main shadows `Base.big` for
# everything included afterwards, and `r01_apparatus_smoke.jl` uses `factorial(big(i))`.
slates    = corpus_slates(c79)
big_slate = slates[argmax(length(s.fixtures) for s in slates)]
test_spec = replay_spec(big_slate.fixtures)
test_grid = adaptive_grid(big_slate.fixtures; lookback = grid_rec.lookback)
t_mid     = test_grid[max(1, length(test_grid) ÷ 2)]
@printf("gate slate: %s, %d fixtures, %d instants, comparing at %s\n",
        big_slate.day, length(big_slate.fixtures), length(test_grid), t_mid)

sys = PF.PortfolioSystem(
    PF.BookSpec(markets = DD.MarketConfig(DD.AbstractMarket[
                    DD.Market1X2(), DD.MarketBTTS(),
                    (DD.MarketOverUnder(l) for l in (0.5, 1.5, 2.5, 3.5, 4.5))...])),
    PF.PolicySpec(trust = PF.FlatTrust(0.25), risk = PF.SlateDrawdown(23.0),
                  cap = PF.FixedCap(0.25)))

# ===================================================================
@testset "WP3 corpus replay" begin

# -------------------------------------------------------------------
@testset "G1 decomposition reproduces match_day row for row" begin
    ref = MD.match_day(test_spec, sys, nothing, expr79, ds79; as_of = t_mid, bankroll = 1.0)

    # the decomposed path, at the same instant, with the same spec
    cards = MD.build_cards(test_spec, nothing, t_mid)
    odds, insts = MD.price_cards(test_spec, cards, t_mid)
    for c in cards
        c.readiness = MD.ready(test_spec.gate, c)
    end
    passed = [c for c in cards if MD.is_ready(c.readiness)]
    lat, _ = MD.matchday_latents(test_spec, expr79, ds79, passed, odds, t_mid)
    got = PF.stake_sheet(sys, lat, expr79, odds, MD.fixture_info(passed); bankroll = 1.0)
    isempty(got) || MD._attach_instruments!(got, insts, test_spec.rounding)

    @test nrow(ref.sheet) == nrow(got)
    if nrow(ref.sheet) > 0
        key = [:match_id, :group, :line, :selection]
        a = sort(ref.sheet, key); b = sort(got, key)
        for c in key
            @test a[!, c] == b[!, c]
        end
        for c in (:odds, :frac, :stake, :p_model, :p_market, :edge, :risk, :venue_odds)
            hasproperty(a, c) && hasproperty(b, c) && @test a[!, c] ≈ b[!, c]
        end
        @test a.side == b.side
        @test a.venue_selection == b.venue_selection
    end
    @printf("    matched %d rows\n", nrow(got))
end

# -------------------------------------------------------------------
@testset "G2 the latent arms, measured rather than assumed" begin
    # This gate originally asserted the OPPOSITE: that a player-level engine must have latents
    # moving with as_of, and that `ok = true` would mean the materialiser was inert. Measurement
    # says otherwise — see l04's header. Serving latents here are a pure function of
    # (fixture, split), so :live and :frozen are the same arm.
    #
    # It is asserted with `latent_delta`, not `latents_invariant`. The latter filters columns on
    # `eltype <: Number` and latent cells are Vector{Float64} inside Any-eltype columns, so it
    # compares nothing and returns "invariant" unconditionally.
    t1, t2 = test_grid[max(1, length(test_grid) ÷ 3)], test_grid[end]
    d = latent_delta(test_spec, expr79, ds79, t1, t2)
    @printf("    latent_delta(%s .. %s): moved=%s worst=%.3e on %s (%d cells compared)\n",
            t1, t2, d.moved, d.worst, d.col, d.n_compared)

    # the test that makes this non-vacuous: cells must actually have been compared
    @test d.n_compared > 0
    @test !d.moved                       # measured 2026-08-12: bit-identical across 3,200 draws
    @test d.worst == 0.0

    # and the vacuity of the inherited helper is pinned, so a future reader does not trust it
    ok, worst, col = latents_invariant(test_spec, expr79, ds79, big_slate.fixtures, t1, t2)
    @test ok && col === :none            # it "passes" without comparing anything
    @printf("    latents_invariant reports %s on %s — vacuous, see l04 header\n", ok, col)
end

# -------------------------------------------------------------------
@testset "G3 the grid is honoured by the fixtures it serves" begin
    @test !isempty(test_grid)
    @test issorted(test_grid)
    ko = minimum(f.kickoff for f in big_slate.fixtures)
    @test maximum(test_grid) <= ko                 # never sample in-play
    @test minimum(test_grid) >= ko - Minute(180)

    # coarse far out, fine near the off
    gaps = Dates.value.(diff(test_grid)) .÷ 60_000
    @test maximum(gaps) == 15
    @test minimum(gaps) == 3

    # A book reaching the grid's start is NOT the same as a book the gate will price. On this
    # slate the first tick is T-334 but the feed then goes silent for over two hours, so
    # MaxBookAge(10m) blocks every card until ~T-120. `live_lead_min` is the honest number and
    # is what the grid must be judged against.
    cov = filter(:match_id => in(Set(f.m_id for f in big_slate.fixtures)), c79.coverage)
    @printf("    first-tick leads: %s\n", string(round.(Int, cov.first_lead_min)))
    @printf("    LIVE leads      : %s   (max gap %s min)\n",
            string(round.(Int, cov.live_lead_min)), string(round.(Int, cov.max_gap_min)))
    @test all(cov.live_lead_min .>= 60)
    @test all(cov.live_lead_min .<= cov.first_lead_min)
end

# -------------------------------------------------------------------
@testset "G3b the grid does not reach past the tradeable book" begin
    # The failure this catches is silent and reads like a model result: an entry rule aimed at a
    # lead the gate refuses to price returns ZERO legs, which in a tearsheet looks like "no edge
    # early" rather than "no book". Every instant the driver visits must be priceable for at
    # least one fixture in its slate.
    cov = filter(:match_id => in(Set(f.m_id for f in big_slate.fixtures)), c79.coverage)
    deepest_live = maximum(cov.live_lead_min)
    lead_of(t) = Dates.value(minimum(f.kickoff for f in big_slate.fixtures) - t) / 60_000
    @printf("    grid starts at T-%.0f, deepest live lead T-%.0f\n",
            lead_of(minimum(test_grid)), deepest_live)
    @test lead_of(minimum(test_grid)) <= deepest_live

    # and the corpus-level recommendation must be derived from the live lead, not the first tick
    g = recommend_grid(c79; coverage = 0.80)
    @printf("    recommend_grid lookback %s (first-tick equivalent would be %s)\n",
            g.lookback, g.first_tick_lookback)
    @test g.lookback <= g.first_tick_lookback
end

# -------------------------------------------------------------------
@testset "G4 the stamped close is coherent" begin
    snaps = build_snapshots(c79, expr79, ds79; arm = :frozen,
                            lookback = Minute(180), verbose = false)
    @test !isempty(snaps)
    led = stake_snapshots(snaps, sys, expr79; bankroll = 1.0, policy_name = "gate")
    @test !isempty(led)

    df = led.df
    @test hasproperty(df, :odds_close_final)
    @test hasproperty(df, :fair_close)
    fin = filter(r -> isfinite(r.fair_close), df)
    @printf("    %d/%d legs carry a finite fair_close\n", nrow(fin), nrow(df))
    @test nrow(fin) > 0
    @test all(0 .< fin.fair_close .< 1)

    # de-vigged probabilities within one market group must sum to ~1 at the close
    for sub in groupby(fin, [:match_id, :group, :line])
        nrow(sub) >= 2 || continue
        s = sum(unique(sub, :selection).fair_close)
        @test 0.5 < s <= 1.0 + 1e-6
    end

    # units: :stake must be the bankroll FRACTION, not currency
    @test all(0 .<= df.stake .<= 1)
    @test df.pnl ≈ df.stake .* df.payoff

    global _snaps, _led = snaps, led
end

# -------------------------------------------------------------------
@testset "G5 entry rules on real data" begin
    df = _led.df
    for rule in (AtClose(), FixedLead(Minute(90)), BestPrice(), FirstQualifying(0.02))
        picked = apply_entry(rule, df)
        legs = unique(picked, [:match_id, :group, :line, :selection])
        @test nrow(picked) == nrow(legs)          # at most one row per leg
        @test nrow(picked) <= nrow(df)
        @printf("    %-24s %4d legs, median lead %.0f min\n", entry_name(rule), nrow(picked),
                isempty(picked) ? NaN : median(picked.mins_to_ko))
    end

    # the oracle can never be beaten on price by a real rule
    orc = apply_entry(BestPrice(), df)
    cls = apply_entry(AtClose(), df)
    j = innerjoin(select(orc, [:match_id, :group, :line, :selection, :odds]),
                  select(cls, [:match_id, :group, :line, :selection, :odds]),
                  on = [:match_id, :group, :line, :selection], makeunique = true)
    @test all(j.odds .>= j.odds_1 .- 1e-9)
    @printf("    oracle beats close on %d/%d legs, mean gain %.3f%%\n",
            count(j.odds .> j.odds_1 .+ 1e-9), nrow(j),
            100 * mean((j.odds .- j.odds_1) ./ j.odds_1))
end

end # testset

println("\n", "="^90)
println("WP3 replay gate: complete")
println("="^90)
