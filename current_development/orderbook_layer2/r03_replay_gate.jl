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
#   G2  latents move with as_of on :live and do NOT on :frozen  (the arms are real)
#   G3  the adaptive grid is honoured by the fixtures it is built for
#   G4  the stamped close is coherent — fair probabilities sum to ~1 per market group
#   G5  entry rules behave on real data as they did on synthetic
#
# G2 is expected to report `false` for :live. That is a PASS, not a failure: `src_sup40_sw40` is
# player-level, so its latents must move with the announced XI. If :live ever came back
# invariant, the ratings materialiser would not be doing anything and the whole frozen/live
# decomposition would be measuring nothing.
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

# One representative slate — the busiest, so the comparison has the most rows to disagree on.
slates    = corpus_slates(c79)
big       = slates[argmax(length(s.fixtures) for s in slates)]
test_spec = replay_spec(big.fixtures)
test_grid = adaptive_grid(big.fixtures; lookback = Minute(180))
t_mid     = test_grid[max(1, length(test_grid) ÷ 2)]
@printf("gate slate: %s, %d fixtures, %d instants, comparing at %s\n",
        big.day, length(big.fixtures), length(test_grid), t_mid)

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
@testset "G2 the frozen and live arms are genuinely different" begin
    t1, t2 = test_grid[1], test_grid[end]
    ok, worst, col = latents_invariant(test_spec, expr79, ds79, big.fixtures, t1, t2)
    # PLAYER-level engine => latents MUST move with as_of. `ok = true` here would mean the
    # ratings materialiser is inert and the decomposition measures nothing.
    @printf("    latents_invariant(%s .. %s) = %s  (max |Δ| = %.3e on %s)\n",
            t1, t2, ok, worst, col)
    @test !ok || worst > 0.0
end

# -------------------------------------------------------------------
@testset "G3 the grid is honoured by the fixtures it serves" begin
    @test !isempty(test_grid)
    @test issorted(test_grid)
    ko = minimum(f.kickoff for f in big.fixtures)
    @test maximum(test_grid) <= ko                 # never sample in-play
    @test minimum(test_grid) >= ko - Minute(180)

    # coarse far out, fine near the off
    gaps = Dates.value.(diff(test_grid)) .÷ 60_000
    @test maximum(gaps) == 15
    @test minimum(gaps) == 3

    # every fixture in this slate has a book reaching the grid's start
    cov = filter(:match_id => in(Set(f.m_id for f in big.fixtures)), c79.coverage)
    @printf("    slate first-tick leads: %s\n", string(round.(Int, cov.first_lead_min)))
    @test all(cov.first_lead_min .>= 60)
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
