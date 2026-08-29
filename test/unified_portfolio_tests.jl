# test/unified_portfolio_tests.jl
#
# The zero-allocation portfolio & staking path graduated from
# `current_development/09_unified_portfolio_framework/`.
#
# Every testset here pins one of two things:
#
#   * a PROPERTY the new path must have that the old one did not -- one workspace per fold rather
#     than one tensor per fixture, a named skip cause rather than a bare `nothing`, a convergence
#     gate in front of the bankroll;
#   * a PARITY the new path must not break -- the books, the trajectory, the metrics and the
#     bootstrap interval must be BIT-IDENTICAL to the ones `src/Portfolio/book.jl` and
#     `simulate.jl` already produced.
#
# The parity half is the important half, and it is deliberately not a transcription: the legacy
# side runs the real `extract_params` -> `compute_score_matrix` -> `compute_market_probs` ->
# `extract_selections` -> `allocate` path, fed a legacy `DataFrame` built from the same typed
# container the new side reads. `==` on `Float64`, not `isapprox`: a one-ULP perturbation of a
# single lambda propagates as ~1e-19 absolute, which is invisible to any tolerance worth writing
# and unmistakable to `==`.
#
# Fixtures are synthetic and seeded -- no DB, no `.cache/`, no MCMC.

using Test
using BayesianFootball
using BayesianFootball: Data, Models, Predictions, Experiments, Training, Evaluation
using DataFrames, Dates, Statistics, Random, Distributions, MCMCChains

const UPF = BayesianFootball.Portfolio

# ===================================================================
# Fixtures
# ===================================================================

"""
A Poisson engine the LEGACY prediction path can price.

Subtyping `AbstractPoissonModel` rather than the bare root is what makes
`Predictions.compute_score_matrix` work on it, which is what makes the parity comparisons real:
both sides are driven end to end, not transcribed.
"""
struct UPFMockPoisson <: BayesianFootball.TypesInterfaces.AbstractPoissonModel end

"An `ExperimentResults`-shaped stand-in: the legacy builder only ever reads `.config.model`."
struct UPFMockExpr
    config::NamedTuple{(:model,), Tuple{UPFMockPoisson}}
end
UPFMockExpr() = UPFMockExpr((model = UPFMockPoisson(),))

"A typed Poisson posterior over `n` fixtures x `n_draws` draws."
function upf_latents(; n = 12, n_draws = 60, seed = 4)
    rng = Xoshiro(seed)
    ids = collect(7001:(7000 + n))
    λh = 0.8 .+ 1.2 .* rand(rng, n, n_draws)
    λa = 0.6 .+ 1.0 .* rand(rng, n, n_draws)
    return CountLatents(ids, λh, λa)
end

"""
A complete quotes frame: 1X2, O/U 2.5 and BTTS for every fixture, vigged at 5%.

Deliberately NOT the model's own prices -- a market identical to the model gives zero edge on
every selection, no stakes, an empty bet frame, and a parity check that compares two empty things.
"""
function upf_odds(ids; seed = 21, vig = 1.05)
    rng = Xoshiro(seed)
    rows = NamedTuple[]
    for m in ids
        for (grp, ln, sels, ps) in (("1X2", 0.0, [:home, :draw, :away], [0.44, 0.26, 0.30]),
                                    ("OverUnder", 2.5, [:over_25, :under_25], [0.51, 0.49]),
                                    ("BTTS", 0.0, [:btts_yes, :btts_no], [0.53, 0.47]))
            q = ps .* (1 .+ 0.08 .* randn(rng, length(ps)))
            q ./= sum(q); q .*= vig
            for (s, p) in zip(sels, q)
                push!(rows, (match_id = m, market_name = grp, market_line = ln,
                             selection = s, odds_close = 1.0 / p))
            end
        end
    end
    return DataFrame(rows)
end

"Three fixtures per settlement window, so a slate is a slate and not a single match."
function upf_fixtures(ids; seed = 8, per_day = 3, played = true)
    rng = Xoshiro(seed)
    d0 = Date(2025, 4, 1)
    return Dict{Int, UPF.FixtureInfo}(
        Int(m) => (date = d0 + Day(div(i - 1, per_day)),
                   score = played ? (rand(rng, 0:3), rand(rng, 0:3)) : nothing)
        for (i, m) in enumerate(ids))
end

upf_matches_frame(fx) = DataFrame(
    match_id  = sort!(collect(keys(fx))),
    match_date = [fx[k].date for k in sort!(collect(keys(fx)))],
    home_score = [fx[k].score === nothing ? missing : fx[k].score[1]
                  for k in sort!(collect(keys(fx)))],
    away_score = [fx[k].score === nothing ? missing : fx[k].score[2]
                  for k in sort!(collect(keys(fx)))])

upf_spec(; shrink = UPF.NoShrinkage(), kw...) = UPF.BookSpec(
    markets = Data.MarketConfig(Data.AbstractMarket[
        Data.Market1X2(), Data.MarketOverUnder(2.5), Data.MarketBTTS()]),
    shrink = shrink; kw...)

"A `Fit` carrying `l` as its latents, with a chain healthy enough to pass the gates."
function upf_fit(l; name = "upf_mock", seed = 3, n = 400, n_chains = 4)
    ch = Chains(randn(Xoshiro(seed), n, 2, n_chains), [:a, :b])
    fss = [(BayesianFootball.FeatureSet(:n_teams => 4),
            Data.SplitMetaData(1, "23/24", "24/25", 1, 1, 0))]
    fit = fit_model(FitConfig(name = name, model = UPFMockPoisson(),
                              splitter = Data.CVConfig(target_seasons = ["24/25"]),
                              sampler = ReplaySampler([ch]),
                              execution = SequentialExecution(),
                              save_dir = mktempdir());
                    feature_sets = fss, quiet = true)
    return Fit(fit.config, fit.folds, l, fit.diagnostics, fit.metadata, fit.save_path)
end

"The same `Fit` with its audit flipped to a failure -- field-order-independent, so a new gate
added to `ConvergenceSummary` cannot silently turn this back into a passing fit."
function upf_unconverged(fit)
    fs   = collect(fieldnames(Training.ConvergenceSummary))
    vals = Any[getfield(fit.diagnostics, f) for f in fs]
    vals[findfirst(==(:passed), fs)]       = false
    vals[findfirst(==(:failed_gates), fs)] = ["rhat"]
    vals[findfirst(==(:failures), fs)]     = ["max R-hat 3.00 > 1.01 (fold 1)"]
    bad = Training.ConvergenceSummary(vals...)
    return Fit(fit.config, fit.folds, fit.latents, bad, fit.metadata, fit.save_path)
end

# Built once -- every testset reads the same fixture, so a number that moves between testsets is a
# real difference and not a different draw.
const UPF_L    = upf_latents()
const UPF_IDS  = latent_match_ids(UPF_L)
const UPF_ODDS = upf_odds(UPF_IDS)
const UPF_FX   = upf_fixtures(UPF_IDS)
const UPF_SPEC = upf_spec()
const UPF_POL  = UPF.PolicySpec()
const UPF_EXPR = UPFMockExpr()
const UPF_LEGACY_DF = to_legacy_dataframe(UPF_L)

@testset "Unified portfolio framework" begin

# ===================================================================
# 1. OddsIndex
# ===================================================================

@testset "OddsIndex" begin
    oi = build_odds_index(UPF_ODDS)

    @test oi isa OddsIndex
    @test length(oi) == nrow(UPF_ODDS)
    @test oi.n_source_rows == nrow(UPF_ODDS)
    @test length(oi.rows) == length(UPF_IDS)
    @test all(haskey(oi, m) for m in UPF_IDS)
    @test !haskey(oi, -1)
    @test occursin("OddsIndex", sprint(show, oi))

    # The ranges partition 1:n exactly -- no row lost, none counted twice.
    covered = sort!(reduce(vcat, [collect(r) for r in values(oi.rows)]))
    @test covered == collect(1:nrow(UPF_ODDS))

    # Every column is concretely typed. `Vector{Any}` here would put the whole selection loop back
    # on the dynamic-dispatch path the index exists to leave.
    @test oi.match_id isa Vector{Int}
    @test oi.market_name isa Vector{String}
    @test oi.market_line isa Vector{Float64}
    @test oi.selection isa Vector{Symbol}
    @test oi.odds_close isa Vector{Float64}

    # ROW ORDER IS PART OF THE ANSWER: the index must yield each match's rows in the same order
    # `view(df, df.match_id .== m, :)` does, because the resulting selection order fixes the column
    # order of R and hence the Kelly solver's starting point.
    for m in UPF_IDS
        legacy = UPF_ODDS[UPF_ODDS.match_id .== m, :]
        rng = oi.rows[m]
        @test oi.selection[rng] == Symbol.(legacy.selection)
        @test oi.market_name[rng] == String.(legacy.market_name)
        @test oi.odds_close[rng] == Float64.(legacy.odds_close)
    end

    # `missing` odds become NaN, and NaN must be REJECTED at admission. `NaN <= 1.0` is `false`,
    # so a naive port admits every missing quote as a valid price.
    holed = copy(UPF_ODDS)
    holed.odds_close = Vector{Union{Missing, Float64}}(holed.odds_close)
    holed[1, :odds_close] = missing
    hi = build_odds_index(holed)
    @test isnan(hi.odds_close[hi.rows[UPF_IDS[1]][1]])

    # A `missing` market line is refused by name, not three frames down inside `view`.
    bad_line = copy(UPF_ODDS)
    bad_line.market_line = Vector{Union{Missing, Float64}}(bad_line.market_line)
    bad_line[1, :market_line] = missing
    @test_throws ErrorException build_odds_index(bad_line)

    # A missing column is named, with the schema the pipeline actually reads.
    @test_throws ErrorException build_odds_index(select(UPF_ODDS, Not(:odds_close)))

    # Idempotent, so a caller may pass either an index or a frame.
    @test build_odds_index(oi) === oi
end

# ===================================================================
# 2. Fixture table
# ===================================================================

@testset "fixture_table methods" begin
    mf = upf_matches_frame(UPF_FX)
    from_frame = fixture_table(mf)
    @test from_frame isa Dict{Int, UPF.FixtureInfo}
    @test from_frame == UPF_FX

    # A comprehension-built dict of unplayed fixtures infers `score::Nothing`; converting turns a
    # MethodError three frames down into nothing at all. This is the match-day case.
    loose = Dict(m => (date = Date(2025, 5, 1), score = nothing) for m in UPF_IDS)
    conv = fixture_table(loose)
    @test conv isa Dict{Int, UPF.FixtureInfo}
    @test all(v.score === nothing for v in values(conv))

    @test fixture_table(UPF_FX) === UPF_FX
end

# ===================================================================
# 3. BookWorkspace and the zero-allocation pricing loop
# ===================================================================

@testset "BookWorkspace" begin
    w = BookWorkspace(UPF_SPEC, UPF_L)

    @test w isa BookWorkspace
    @test size(w.S) == (12, 12, n_draws(UPF_L))
    @test w.n_draws == n_draws(UPF_L)
    @test length(w.slots_1x2) == 1
    @test length(w.slots_btts) == 1
    @test length(w.slots_ou) == 1
    @test isempty(w.slots_fb)
    @test fallback_market_names(w) == String[]
    @test workspace_bytes(w) > 0
    @test occursin("BookWorkspace", sprint(show, w))

    # `order` records where each of `spec.markets.markets` went, in the caller's sequence -- that
    # sequence is the order selections land in the book.
    @test length(w.order) == length(UPF_SPEC.markets.markets)
    @test first.(w.order) == [:x, :o, :b]

    # A non-kernel market takes the fallback slot and is WARNED about, not silently made slow.
    fb_spec = UPF.BookSpec(markets = Data.MarketConfig(Data.AbstractMarket[
                               Data.Market1X2(), Data.MarketDC()]),
                           shrink = UPF.NoShrinkage())
    fw = @test_logs (:warn,) BookWorkspace(fb_spec, UPF_L)
    @test length(fw.slots_fb) == 1
    @test length(fallback_market_names(fw)) == 1
    @test !isempty(fallback_probs(fw))
    # ... and can be silenced when the caller has already been told.
    @test length(BookWorkspace(fb_spec, UPF_L; quiet = true).slots_fb) == 1

    # No fallback slot => no fallback dictionary to allocate at all.
    @test isempty(fallback_probs(w))
end

@testset "price_fixture! allocates nothing" begin
    w = BookWorkspace(UPF_SPEC, UPF_L)

    # Warm the kernels before checking their steady-state allocation contract.
    price_fixture!(w, UPF_L, 1)
    price_fixture!(w, UPF_L, 2)

    # THE HEADLINE PROPERTY. The legacy builder allocated a fresh (12 x 12 x n_draws) tensor and
    # one Dict per market per fixture; this is 0 bytes for the grid AND every market book.
    @test @allocated(price_fixture!(w, UPF_L, 3)) == 0
    for i in 1:n_matches(UPF_L)
        @test @allocated(price_fixture!(w, UPF_L, i)) == 0
    end

    # The workspace is reused, so the second pricing of the SAME fixture must give the same
    # numbers -- a stale buffer would show up here and nowhere else.
    price_fixture!(w, UPF_L, 5)
    S5 = copy(w.S); b5 = deepcopy(w.slots_1x2[1].book)
    price_fixture!(w, UPF_L, 7)
    price_fixture!(w, UPF_L, 5)
    @test w.S == S5
    @test w.slots_1x2[1].book == b5

    # And it is the same grid the standalone kernel produces.
    ws = Predictions.GridWorkspace()
    S = Predictions.alloc_score_grid(UPF_L)
    Predictions.compute_score_grid!(S, ws, UPF_L, 5)
    price_fixture!(w, UPF_L, 5)
    @test w.S == S
end

# ===================================================================
# 4. Selections, off the index
# ===================================================================

@testset "extract_selections parity" begin
    w  = BookWorkspace(UPF_SPEC, UPF_L)
    oi = build_odds_index(UPF_ODDS)

    for i in 1:n_matches(UPF_L)
        m_id = UPF_IDS[i]
        price_fixture!(w, UPF_L, i)
        fast = extract_selections(w, oi, m_id, UPF_SPEC)

        sm = Predictions.ScoreMatrix(copy(w.S))
        mp = Dict(string(m) => Predictions.compute_market_probs(sm, m)
                  for m in UPF_SPEC.markets.markets)
        slow = UPF.extract_selections(UPF_ODDS, m_id, UPF_SPEC, mp)

        @test !isempty(fast)
        @test fast == slow          # bit-identical, and in the same ORDER
    end

    # An incomplete market group is dropped whole: vig removal divides by the sum over whatever
    # legs are present, so a 1X2 group missing one way manufactures up to 20% of edge.
    thin = UPF_ODDS[.!((UPF_ODDS.match_id .== UPF_IDS[1]) .&
                       (UPF_ODDS.selection .== :draw)), :]
    ti = build_odds_index(thin)
    price_fixture!(w, UPF_L, 1)
    kept = extract_selections(w, ti, UPF_IDS[1], UPF_SPEC)
    @test all(s -> s.group != "1X2", kept)
    @test any(s -> s.group == "OverUnder", kept)

    # An unknown match is empty, not an error.
    @test isempty(extract_selections(w, oi, -1, UPF_SPEC))

    # `selection_family` is the trust key and is unchanged.
    @test selection_family("1X2", 0.0, :home) == "1X2_home"
    @test selection_family("OverUnder", 2.5, :over_25) == "O/U 2.5_over_25"
end

# ===================================================================
# 5. Book parity against the legacy builder
# ===================================================================

@testset "build_books parity with the legacy builder" begin
    for shrink in (UPF.NoShrinkage(), UPF.FractionalKelly(0.5), UPF.BakerMcHale(n_draws = 16))
        spec = upf_spec(shrink = shrink)
        legacy = UPF.build_books(spec, UPF_LEGACY_DF, UPF_EXPR, UPF_ODDS, UPF_FX)
        new    = build_books(spec, UPF_L, UPF_ODDS, UPF_FX)

        @test length(new) == length(legacy) == length(UPF_IDS)
        for (a, b) in zip(legacy, new)
            @test a.m_id     == b.m_id
            @test a.date     == b.date
            @test a.sels     == b.sels
            @test a.p_grid   == b.p_grid        # `==`, not isapprox: 0 ULP or it is a defect
            @test a.R        == b.R
            @test a.settle   == b.settle
            @test a.a_kelly  == b.a_kelly       # the Kelly solve too -- LBFGS is deterministic
            @test a.k_shrink == b.k_shrink      # including BakerMcHale's seeded resample
            @test a.kkt      == b.kkt
            @test a.converged == b.converged
        end
    end

    # Chronological order is established in the builder, once, so nothing downstream has to
    # remember to sort. Path metrics on an unsorted series are meaningless.
    books = build_books(UPF_SPEC, UPF_L, UPF_ODDS, UPF_FX)
    @test issorted(books, by = b -> (b.date, b.m_id))

    # A `DataStore`-shaped `matches` frame and a prebuilt dict are the same fixture table.
    @test build_books(UPF_SPEC, UPF_L, UPF_ODDS, upf_matches_frame(UPF_FX)) == books

    # An `OddsIndex` may be passed in place of the frame, and must not change a number.
    @test build_books(UPF_SPEC, UPF_L, build_odds_index(UPF_ODDS), UPF_FX) == books

    # The single-fixture primitive agrees with the fold builder.
    w  = BookWorkspace(UPF_SPEC, UPF_L)
    oi = build_odds_index(UPF_ODDS)
    one = UPF.build_book(UPF_SPEC, w, UPF_L, 1, oi, UPF_FX)
    @test one !== nothing
    @test one == books[findfirst(b -> b.m_id == UPF_IDS[1], books)]
end

# ===================================================================
# 6. BuildReport -- the four skip causes, separated
# ===================================================================

@testset "BuildReport" begin
    books, br = build_books_reported(UPF_SPEC, UPF_L, UPF_ODDS, UPF_FX)
    @test br isa BuildReport
    @test br.n_fixtures == n_matches(UPF_L)
    @test br.n_books == length(books)
    @test n_skipped(br) == 0
    @test br.converged === nothing        # no Fit was involved, so there is nothing to claim
    @test br.gated == false
    @test br.elapsed >= 0
    @test occursin("BuildReport", sprint(show, br))
    @test occursin("built", sprint(show, MIME"text/plain"(), br))

    # Each cause is named and counted, where `src` returned a bare `nothing` for all four.
    dropped_fixture = filter(p -> p.first != UPF_IDS[1], UPF_FX)
    unplayed = Dict{Int, UPF.FixtureInfo}(
        k => (k == UPF_IDS[2] ? (date = v.date, score = nothing) : v)
        for (k, v) in dropped_fixture)
    no_quotes = UPF_ODDS[UPF_ODDS.match_id .!= UPF_IDS[3], :]
    # every leg of every group stripped from one fixture but one -> quoted, no complete group
    partial = no_quotes[.!((no_quotes.match_id .== UPF_IDS[4]) .&
                           (no_quotes.selection .!= :home)), :]

    b2, r2 = build_books_reported(UPF_SPEC, UPF_L, partial, unplayed)
    @test r2.skipped_no_fixture == [UPF_IDS[1]]
    @test r2.skipped_unplayed == [UPF_IDS[2]]
    @test r2.skipped_no_quotes == [UPF_IDS[3]]
    @test r2.skipped_no_selections == [UPF_IDS[4]]
    @test n_skipped(r2) == 4
    @test r2.n_books == length(UPF_IDS) - 4
    @test occursin("skipped", sprint(show, MIME"text/plain"(), r2))

    # The books that DID survive are bit-identical to the ones the undamaged build produced -- a
    # builder that dropped a DIFFERENT fixture would also produce a shorter list.
    kept = [b for b in books if b.m_id ∉ UPF_IDS[1:4]]
    @test b2 == kept

    # `build_books` is `build_books_reported` without the second value. Same books.
    @test build_books(UPF_SPEC, UPF_L, partial, unplayed) == b2

    @test n_skipped(BuildReport(10)) == 0
end

# ===================================================================
# 7. Kelly staking simulation and daily slate bankroll tracking
# ===================================================================

@testset "simulate_portfolio" begin
    books = build_books(UPF_SPEC, UPF_L, UPF_ODDS, UPF_FX)
    res = simulate_portfolio(UPF_POL, books; initial_bankroll = 1000.0, B = 200)

    @test res isa PortfolioResult
    @test length(res) == length(res.daily_states) == 4     # 12 fixtures, 3 per window
    @test res[1] isa DailyState
    @test [d for d in res] == res.daily_states
    @test occursin("PortfolioResult", sprint(show, res))

    # --- daily slate bankroll tracking -------------------------------------------------
    @test issorted([d.date for d in res.daily_states])
    @test res.daily_states[1].bankroll_open == 1000.0
    for d in res.daily_states
        @test d.bankroll_close ≈ d.bankroll_open * (1.0 + d.pnl_frac)
        @test d.n_fixtures == 3
        @test 0.0 <= d.exposure <= 1.0        # FixedCap(0.25) makes ruin unrepresentable
        @test d.stake_frac >= 0.0
        @test log_growth(d) ≈ log(1.0 + d.pnl_frac)
    end
    # The windows chain: each opens where the last one closed.
    for i in 2:length(res.daily_states)
        @test res.daily_states[i].bankroll_open == res.daily_states[i - 1].bankroll_close
    end
    @test res.summary.final_bankroll ≈ res.daily_states[end].bankroll_close
    @test sum(d.n_bets for d in res.daily_states) == res.summary.n_bets
    @test res.summary.n_bets > 0          # a simulation with no bets tests nothing

    # `initial_bankroll` is a REPORTING SCALE. The trajectory at 1.0 is the trajectory at 1000.0,
    # so a bankroll-dependent trust or filter sees the same number under either.
    res1 = simulate_portfolio(UPF_POL, books; initial_bankroll = 1.0, bootstrap = false)
    @test res1.trajectory.bankroll == res.trajectory.bankroll
    @test res1.trajectory.slate_pl == res.trajectory.slate_pl
    @test isequal(res1.trajectory.bets, res.trajectory.bets)
    @test res1.summary.roi == res.summary.roi

    # Reproducible: same inputs, same trajectory and same interval, twice.
    again = simulate_portfolio(UPF_POL, books; initial_bankroll = 1000.0, B = 200)
    @test again.trajectory.bankroll == res.trajectory.bankroll
    @test again.bootstrap_ci.roi_lo == res.bootstrap_ci.roi_lo

    # Simulating must not perturb the caller's global RNG -- BakerMcHale and the bootstrap sample.
    Random.seed!(1234); before = rand()
    Random.seed!(1234); simulate_portfolio(UPF_POL, books; B = 100); after = rand()
    @test before == after

    # The two refusals, both inherited from `simulate` and both hard.
    unsettled = build_books(UPF_SPEC, UPF_L, UPF_ODDS, upf_fixtures(UPF_IDS, played = false);
                            require_result = false)
    @test length(unsettled) == length(UPF_IDS)
    @test length(unsettled_books(unsettled)) == length(UPF_IDS)
    @test UnsettledBooks(unsettled) == unsettled_books(unsettled)
    @test isempty(unsettled_books(books))
    @test_throws AssertionError simulate_portfolio(UPF_POL, unsettled)

    # A `PortfolioSystem` stakes with its own policy.
    sys = UPF.PortfolioSystem(UPF_SPEC, UPF_POL)
    @test simulate_portfolio(sys, books; bootstrap = false).trajectory.bankroll ==
          res.trajectory.bankroll

    # Wealth metrics ride along on the slate-level bankroll curve.
    with_m = simulate_portfolio(UPF_POL, books; bootstrap = false,
                                metrics = [BayesianFootball.BackTesting.SharpeRatio()])
    @test length(keys(with_m.metrics)) == 1

    # An empty book is a degenerate run, not a crash.
    empty_res = simulate_portfolio(UPF_POL, UPF.MatchBook[]; bootstrap = false)
    @test length(empty_res) == 0
    @test empty_res.summary.n_bets == 0
    @test isnan(empty_res.summary.cagr)
    @test isempty(states_frame(empty_res))
end

@testset "simulate_portfolio is `simulate`, to the last bit" begin
    books = build_books(UPF_SPEC, UPF_L, UPF_ODDS, UPF_FX)
    res   = simulate_portfolio(UPF_POL, books; initial_bankroll = 1000.0, B = 400, seed = 1)
    traj  = UPF.simulate(UPF_POL, UPF.group(UPF_POL.grouping, books))

    # `simulate_portfolio` runs its OWN forward walk rather than decorating `simulate`'s result --
    # if it called it, this testset would be checking that a function agrees with itself.
    @test res.trajectory.bankroll    == traj.bankroll
    @test res.trajectory.dates       == traj.dates
    @test res.trajectory.slate_pl    == traj.slate_pl
    @test res.trajectory.k_risk      == traj.k_risk
    @test res.trajectory.exposure    == traj.exposure
    @test res.trajectory.n_capped    == traj.n_capped
    @test res.trajectory.total_stake == traj.total_stake
    @test res.trajectory.total_pl    == traj.total_pl
    @test isequal(res.trajectory.bets, traj.bets)

    # Every field `path_metrics` has, computed the same way.
    pm = UPF.path_metrics(traj)
    s  = res.summary
    @test pm.final            == res.trajectory.bankroll[end]
    @test pm.roi              == s.roi
    @test pm.growth_per_slate == s.growth_per_slate
    @test pm.mdd              == s.mdd
    @test pm.ulcer            == s.ulcer
    @test pm.calmar           == s.calmar
    @test pm.martin           == s.martin
    @test pm.n_slates         == s.n_slates
    @test pm.n_bets           == s.n_bets
    @test pm.mean_exposure    == s.mean_exposure
    @test pm.max_exposure     == s.max_exposure
    @test pm.worst_slate      == s.worst_slate
    @test pm.mean_k_risk      == s.mean_k_risk
    @test pm.n_capped         == s.n_capped

    # And the match-clustered ROI interval is the same draw sequence.
    ci = UPF.bootstrap_roi(traj.bets; B = 400, seed = 1)
    @test ci.lo == res.bootstrap_ci.roi_lo
    @test ci.hi == res.bootstrap_ci.roi_hi
    @test ci.sd == res.bootstrap_ci.roi_sd
    @test 0.0 <= res.bootstrap_ci.p_roi_positive <= 1.0
    @test occursin("BootstrapCI", sprint(show, res.bootstrap_ci))

    # `attribution` is the `src` function, not a second one.
    @test isequal(res.attribution, UPF.attribution(traj))

    @test bootstrap_portfolio(res; B = 50, seed = 2) ==
          bootstrap_portfolio(res.trajectory; B = 50, seed = 2)
end

# ===================================================================
# 8. Drawdown, and the rest of the summary
# ===================================================================

@testset "drawdown and summary conventions" begin
    books = build_books(UPF_SPEC, UPF_L, UPF_ODDS, UPF_FX)
    res = simulate_portfolio(UPF_POL, books; initial_bankroll = 1000.0, bootstrap = false)
    s = res.summary

    # `mdd` is a NEGATIVE PERCENT, matching `path_metrics`, and is bounded by the worst window.
    @test s.mdd <= 0.0
    @test s.ulcer >= 0.0
    @test s.span_days == Dates.value(res.daily_states[end].date - res.daily_states[1].date)

    # The daily table's drawdown column IS the headline number, not a second one computed a
    # second way from the rounded currency series.
    d = daily_returns_table(res)
    @test nrow(d) == length(res)
    @test minimum(d.drawdown_pct) == s.mdd
    @test all(d.drawdown_pct .<= 0.0)
    @test d.return_pct == 100 .* [x.pnl_frac for x in res.daily_states]
    @test d.log_growth == log.(1.0 .+ [x.pnl_frac for x in res.daily_states])

    # An explicit drawdown re-derivation, written from the definition rather than from the code.
    bk = res.trajectory.bankroll
    peak = -Inf; worst = 0.0
    for b in bk
        peak = max(peak, b)
        worst = min(worst, (b - peak) / peak * 100)
    end
    @test worst ≈ s.mdd

    # Sharpe and Sortino are on per-slate LOG returns, because a slate is the compounding unit.
    r = log.(1.0 .+ res.trajectory.slate_pl)
    @test s.sharpe ≈ mean(r) / std(r)
    @test s.growth_per_slate ≈ mean(r)
    @test s.calmar == (s.mdd < 0 ? s.total_return_pct / abs(s.mdd) : 0.0)

    # CAGR is NaN over a zero-length span rather than an invented annual rate.
    one_day = Dict{Int, UPF.FixtureInfo}(
        k => (date = Date(2025, 4, 1), score = v.score) for (k, v) in UPF_FX)
    single = simulate_portfolio(UPF_POL, build_books(UPF_SPEC, UPF_L, UPF_ODDS, one_day);
                                bootstrap = false)
    @test length(single) == 1
    @test isnan(single.summary.cagr)
    @test isnan(single.summary.sharpe_ann)

    # `win_rate` counts staked selections, pushes included in the denominator.
    @test 0.0 <= s.win_rate <= 1.0
    nt = as_namedtuple(s)
    @test nt isa NamedTuple
    @test nt.mdd == s.mdd
    @test length(nt) == fieldcount(UPF.PortfolioSummary)

    sf = states_frame(res)
    @test nrow(sf) == length(res)
    @test names(sf) == [String(f) for f in fieldnames(UPF.DailyState)]
end

# ===================================================================
# 9. The convergence gate
# ===================================================================

@testset "convergence gating" begin
    fit = upf_fit(UPF_L)
    @test convergence_verdict(fit)[1]

    books, br = build_books_reported(UPF_SPEC, fit, UPF_ODDS, UPF_FX)
    @test br.converged === true
    @test br.gated == true
    @test isempty(br.failed_gates)
    # Reading the posterior off the Fit must not change a single number.
    @test books == build_books(UPF_SPEC, UPF_L, UPF_ODDS, UPF_FX)
    @test build_books(UPF_SPEC, fit, UPF_ODDS, UPF_FX) == books

    bad = upf_unconverged(fit)
    @test !convergence_verdict(bad)[1]

    # THE GATE. Default is `true` here and `false` nowhere: an unconverged posterior is not merely
    # noisier, it is too NARROW, so every edge looks larger than it is and Kelly stake size is
    # monotone in that edge.
    @test_throws Evaluation.ConvergenceRefusal build_books_reported(UPF_SPEC, bad, UPF_ODDS,
                                                                    UPF_FX)
    @test_throws Evaluation.ConvergenceRefusal build_books(UPF_SPEC, bad, UPF_ODDS, UPF_FX)

    # The gate REFUSES; it does not change arithmetic. Lifted, the books are bit-identical and the
    # provenance travels with them.
    lifted, lr = build_books_reported(UPF_SPEC, bad, UPF_ODDS, UPF_FX;
                                      require_converged = false, quiet = true)
    @test lifted == books
    @test lr.converged === false
    @test lr.failed_gates == ["rhat"]
    @test lr.gated == false
    @test occursin("FAIL", sprint(show, MIME"text/plain"(), lr))

    # ... and is carried into the result, so a `PortfolioResult` off disk answers "should this be
    # believed" without a DataStore, the chains, or a re-audit.
    res = simulate_portfolio(UPF_POL, lifted, lr; bootstrap = false)
    @test res.converged === false
    @test res.failed_gates == ["rhat"]
    @test occursin("UNCONVERGED", sprint(show, res))
    @test occursin("NOT CONVERGED", sprint(show, MIME"text/plain"(), res))

    # An ungated source claims nothing rather than claiming health.
    plain = simulate_portfolio(UPF_POL, books; bootstrap = false)
    @test plain.converged === nothing

    # `run_portfolio_simulation` is the one-call form, gated the same way.
    r, b, rep = run_portfolio_simulation(UPF_SPEC, UPF_POL, fit, UPF_ODDS, UPF_FX;
                                         bootstrap = false)
    @test b == books
    @test rep.converged === true
    @test r.trajectory.bankroll == plain.trajectory.bankroll
    @test_throws Evaluation.ConvergenceRefusal run_portfolio_simulation(
        UPF_SPEC, UPF_POL, bad, UPF_ODDS, UPF_FX; bootstrap = false)

    sys = UPF.PortfolioSystem(UPF_SPEC, UPF_POL)
    r2, _, _ = run_portfolio_simulation(sys, UPF_L, UPF_ODDS, UPF_FX; bootstrap = false)
    @test r2.trajectory.bankroll == plain.trajectory.bankroll

    # A legacy `(latents_df, expr)` pair routes onto the fast path and gives the same books.
    r3, b3, rep3 = run_portfolio_simulation(UPF_SPEC, UPF_POL, (UPF_LEGACY_DF, UPF_EXPR),
                                            UPF_ODDS, UPF_FX; bootstrap = false)
    @test b3 == books
    @test rep3.converged === nothing        # no Fit, so nothing to gate on and nothing claimed
end

# ===================================================================
# 10. Backward compatibility
# ===================================================================

@testset "legacy compatibility" begin
    # 1. The legacy five-argument builder is UNTOUCHED and still produces its own books.
    legacy = UPF.build_books(UPF_SPEC, UPF_LEGACY_DF, UPF_EXPR, UPF_ODDS, UPF_FX)
    @test legacy isa Vector{UPF.MatchBook}
    @test length(legacy) == length(UPF_IDS)

    # 2. The reported legacy shape routes onto the typed fast path and returns the same books.
    routed, rr = build_books_reported(UPF_SPEC, UPF_LEGACY_DF, UPF_EXPR, UPF_ODDS, UPF_FX)
    @test routed == legacy
    @test rr.fallback_markets != ["legacy route"]      # it really did take the fast path

    # A model with no legacy-frame reader falls back to `src`'s builder rather than losing the
    # capability. `route` is visible in the report.
    struct_free = UPF.build_books(UPF_SPEC, UPF_LEGACY_DF, UPFMockPoisson(), UPF_ODDS, UPF_FX)
    @test struct_free == legacy

    # 3. A book built the new way IS a `MatchBook`, so every legacy consumer keeps working --
    #    not because a bridge translates it, but because there is nothing to translate.
    books = build_books(UPF_SPEC, UPF_L, UPF_ODDS, UPF_FX)
    @test eltype(books) === UPF.MatchBook
    slates = UPF.group(UPF_POL.grouping, books)
    @test slates == group_slates_by_day(books)
    @test build_slates(books) == slates
    @test UPF.simulate(UPF_POL, slates) isa UPF.Trajectory
    @test UPF.stake_slate(UPF_POL, slates[1], UPF.SlateContext(1, slates[1].window, 1.0)) isa
          UPF.SlateAllocation
    @test UPF.report(UPF.simulate(UPF_POL, slates)) isa NamedTuple

    # 4. `book_cache_key` is unchanged, so an existing serialised book cache still HITS.
    #    (The `BakerMcHale` case is the one a naive `hash` breaks -- see `component_hash`.)
    s1 = upf_spec(shrink = UPF.BakerMcHale())
    s2 = upf_spec(shrink = UPF.BakerMcHale())
    @test book_cache_key(s1) == book_cache_key(s2)
    @test book_cache_key(UPF_SPEC) != book_cache_key(s1)

    # 5. The legacy match-day sheet still works, and the typed one gives the same sheet.
    live_fx = upf_fixtures(UPF_IDS, played = false)
    sys = UPF.PortfolioSystem(UPF_SPEC, UPF_POL)
    legacy_sheet = UPF.stake_sheet(sys, UPF_LEGACY_DF, UPF_EXPR, UPF_ODDS, live_fx;
                                   bankroll = 5000.0)
    typed_sheet  = stake_sheet(sys, UPF_L, UPF_ODDS, live_fx; bankroll = 5000.0)
    @test names(typed_sheet) == names(legacy_sheet)
    @test nrow(typed_sheet) == nrow(legacy_sheet)
    @test isequal(typed_sheet, legacy_sheet)
    @test !isempty(typed_sheet)
    @test all(.!typed_sheet.settled)
    @test UPF.slate_summary(typed_sheet) isa DataFrame

    fit_sheet = stake_sheet(sys, upf_fit(UPF_L), UPF_ODDS, live_fx; bankroll = 5000.0)
    @test isequal(fit_sheet, typed_sheet)
    @test_throws Evaluation.ConvergenceRefusal stake_sheet(
        sys, upf_unconverged(upf_fit(UPF_L)), UPF_ODDS, live_fx)

    # A `DataStore`-shaped fixture table with no upcoming matches yields an empty sheet, not an
    # error -- the documented match-day trap, still documented and still not a crash.
    @test isempty(stake_sheet(sys, UPF_L, UPF_ODDS, Dict{Int, UPF.FixtureInfo}()))

    # 6. `grid_shrink_factor` is `shrink_factor` over a ScoreMatrix VIEW of the shared grid, not a
    #    second implementation.
    w = BookWorkspace(UPF_SPEC, UPF_L)
    oi = build_odds_index(UPF_ODDS)
    price_fixture!(w, UPF_L, 1)
    sels = extract_selections(w, oi, UPF_IDS[1], UPF_SPEC)
    p = vec(mean(w.S, dims = 3)[:, :, 1]); p ./= sum(p)
    R = UPF.payoff_matrix(sels, 12, 12, UPF_SPEC.exec.commission)
    bmh = UPF.BakerMcHale(n_draws = 16)
    @test grid_shrink_factor(bmh, w.S, R, p, UPF_SPEC.allocator, UPF_SPEC.exec;
                             seed_offset = UPF_IDS[1]) ==
          UPF.shrink_factor(bmh, Predictions.ScoreMatrix(w.S), R, p, UPF_SPEC.allocator,
                            UPF_SPEC.exec; seed_offset = UPF_IDS[1])
end

# ===================================================================
# 11. The briefing's names
# ===================================================================

@testset "aliases and accessors" begin
    # ONE TYPE PER NAME. `MarketBook` is `MatchBook`, not a second struct with the same shape --
    # which is what keeps a `Vector{MatchBook}` unserialised from an existing `.jls` stakeable.
    @test MarketSelection === UPF.Selection
    @test MarketBook === UPF.MatchBook
    @test MatchedMarketOdds === UPF.OddsIndex
    @test PortfolioPolicy === UPF.PolicySpec
    @test LogUtility === UPF.KellyLogUtility
    @test LogUtility() isa UPF.AbstractAllocator

    b = build_books(UPF_SPEC, UPF_L, UPF_ODDS, UPF_FX)[1]
    @test b isa MarketBook
    @test book_match_id(b) === b.m_id
    @test book_date(b) === b.date
    @test book_selections(b) === b.sels
    @test book_grid(b) === b.p_grid
    @test book_payoff(b) === b.R
    @test book_settle(b) === b.settle
    @test book_alloc(b) === b.a_kelly
    @test book_shrink(b) === b.k_shrink
    @test book_kkt(b) === b.kkt
    @test book_converged(b) === b.converged

    s = b.sels[1]
    @test s isa MarketSelection
    @test sel_name(s) === s.selection
    @test sel_odds_close(s) === s.odds_quoted
    @test sel_odds_settle(s) === s.odds_used
    @test sel_prob_model(s) === s.p_model
    @test sel_prob_market(s) === s.p_market
    @test sel_edge(s) == s.p_model - s.p_market
end

# ===================================================================
# 12. Reporting
# ===================================================================

@testset "reporting" begin
    books, br = build_books_reported(UPF_SPEC, UPF_L, UPF_ODDS, UPF_FX)
    res = simulate_portfolio(UPF_POL, books, br; initial_bankroll = 1000.0, B = 200)

    rep = portfolio_report(res, br; name = "unit test")
    @test rep isa PortfolioReport
    @test rep.name == "unit test"
    @test rep.build === br
    @test rep.result === res
    @test nrow(rep.daily) == length(res)
    @test occursin("unit test", sprint(show, rep))

    txt = sprint(show, MIME"text/plain"(), rep)
    @test occursin("PortfolioReport", txt)
    @test occursin("SETTLEMENT WINDOWS", txt)
    @test occursin("ATTRIBUTION BY FAMILY", txt)

    disp = sprint(io -> display_portfolio(res; io = io))
    @test occursin("PORTFOLIO", disp)
    @test occursin("max drawdown", disp)
    @test occursin("Sharpe (slate)", disp)
    # `max_slates` truncates the middle table only.
    @test occursin("more", sprint(io -> display_portfolio(res; io = io, max_slates = 1)))

    md = portfolio_markdown(rep)
    @test occursin("# unit test", md)
    @test occursin("## Headline", md)
    @test occursin("## Settlement windows", md)
    @test occursin("| max drawdown % |", md)
    @test occursin("| Sharpe (slate) |", md)
    @test occursin("| Sortino |", md)
    @test occursin("ROI 95% CI (by match)", md)
    @test count(==('\n'), md) > 30
    @test portfolio_markdown(portfolio_report(res)) isa String

    # A report with no build attached still renders -- books loaded off disk have no live one.
    @test portfolio_report(res).build === nothing
    @test occursin("Headline", portfolio_markdown(portfolio_report(res)))
end

end # testset
