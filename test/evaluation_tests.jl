using Test
using BayesianFootball
using BayesianFootball: Data, Models, Predictions, Experiments, Training, Evaluation
using DataFrames
using Dates
using Distributions
using MCMCChains
using Random
using Statistics

const EV = BayesianFootball.Evaluation

"""
A Poisson engine the LEGACY prediction path can price.

Subtyping `AbstractPoissonModel` rather than the bare root is what makes
`Predictions.model_inference` work on it, which is what makes the parity comparisons in
§3 real: both sides are driven end to end, not transcribed.
"""
struct EvalMockPoisson <: BayesianFootball.TypesInterfaces.AbstractPoissonModel end

"A typed Poisson posterior container over `n` fixtures × `n_draws` draws."
function eval_latents(; n = 24, n_draws = 120, seed = 7)
    rng = Xoshiro(seed)
    ids = collect(1001:(1000 + n))
    λh = 0.7 .+ 1.4 .* rand(rng, n, n_draws)
    λa = 0.5 .+ 1.2 .* rand(rng, n, n_draws)
    return CountLatents(ids, λh, λa)
end

"Settle one selection against a realised scoreline. Deliberately explicit rather than
derived from the pricing kernels: a settlement bug that agreed with the pricer would make
every metric look calibrated."
eval_is_winner(::Data.Market1X2, s, gh, ga) =
    s === :home ? gh > ga : s === :draw ? gh == ga : ga > gh
eval_is_winner(::Data.MarketBTTS, s, gh, ga) =
    s === :btts_yes ? (gh > 0 && ga > 0) : !(gh > 0 && ga > 0)
eval_is_winner(m::Data.MarketOverUnder, s, gh, ga) =
    startswith(String(s), "over_") ? (gh + ga > m.line) : (gh + ga < m.line)

"""
A `DataStore` carrying only `matches` and `odds` — the two domains every evaluation
kernel reads.

The odds are the model's OWN mean prices, perturbed in log-odds space, renormalised and
vigged, then handed to the real `Data.Markets._enrich_market_data!` so `prob_fair_close`
and the rest are computed exactly as the production fetcher computes them. Reproducing
that arithmetic here would be a second implementation of the thing the metrics are
scored against.

`noise = 0` would make market and model identical and every `diff` exactly zero, which
passes a test while testing nothing.
"""
function eval_datastore(l, markets; seed = 99, vig = 0.05, noise = 0.12)
    rng = MersenneTwister(seed)
    ids = latent_match_ids(l)
    scores = [(rand(rng, Poisson(mean(view(l.λ_home, i, :)))),
               rand(rng, Poisson(mean(view(l.λ_away, i, :))))) for i in eachindex(ids)]
    probs = market_probabilities(l, markets; keep_draws = false, threaded = false)

    mid = Int[]; mname = String[]; mline = Float64[]; sel = Symbol[]
    oopen = Float64[]; oclose = Float64[]; won = Bool[]
    for (i, id) in enumerate(ids)
        gh, ga = scores[i]
        for m in markets
            ks = market_keys(m)
            p = Float64[probs.means[i, probs.col_of[k]] for k in ks]
            pc = p .* exp.(noise .* randn(rng, length(p))); pc ./= sum(pc)
            po = p .* exp.((noise * 1.6) .* randn(rng, length(p))); po ./= sum(po)
            for (j, k) in enumerate(ks)
                push!(mid, id)
                push!(mname, Data.Markets.market_group(m))
                push!(mline, Data.Markets.market_line(m))
                push!(sel, k)
                push!(oopen, 1.0 / (po[j] * (1.0 + vig)))
                push!(oclose, 1.0 / (pc[j] * (1.0 + vig)))
                push!(won, eval_is_winner(m, k, gh, ga))
            end
        end
    end

    odds = DataFrame(match_id = mid, market_name = mname, market_line = mline,
                     selection = sel, odds_open = oopen, odds_close = oclose,
                     is_winner = won)
    Data.Markets._enrich_market_data!(odds)

    dates = [Date(2025, 1, 5) + Day(11 * (i - 1)) for i in eachindex(ids)]
    matches = DataFrame(match_id = copy(ids), match_date = dates,
                        match_month = [Dates.month(d) for d in dates],
                        home_score = [s[1] for s in scores],
                        away_score = [s[2] for s in scores],
                        tournament_id = fill(1, length(ids)),
                        season = fill("24/25", length(ids)),
                        home_team = ["T$(1 + (i - 1) % 8)" for i in eachindex(ids)],
                        away_team = ["T$(1 + i % 8)" for i in eachindex(ids)],
                        match_week = [(i - 1) ÷ 8 + 1 for i in eachindex(ids)])

    ds = Data.DataStore(Data.ScottishLower(), matches, DataFrame(), odds,
                        DataFrame(), DataFrame(), DataFrame(), DataFrame(), DataFrame())
    return (ds, scores)
end

"A `Fit` carrying `l` as its latents, with a chain healthy enough to pass the gates."
function eval_fit(l; name = "eval_mock", seed = 3, n = 400, n_chains = 4)
    ch = Chains(randn(Xoshiro(seed), n, 2, n_chains), [:a, :b])
    fss = [(BayesianFootball.FeatureSet(:n_teams => 4),
            Data.SplitMetaData(1, "23/24", "24/25", 1, 1, 0))]
    fit = fit_model(FitConfig(name = name, model = EvalMockPoisson(),
                              splitter = Data.CVConfig(target_seasons = ["24/25"]),
                              sampler = ReplaySampler([ch]),
                              execution = SequentialExecution(),
                              save_dir = mktempdir());
                    feature_sets = fss, quiet = true)
    return Fit(fit.config, fit.folds, l, fit.diagnostics, fit.metadata, fit.save_path)
end

const EVAL_MARKETS = [Data.Market1X2(), Data.MarketOverUnder(2.5), Data.MarketBTTS()]

# Built once — every testset reads the same fixture, so a number that moves between
# testsets is a real difference and not a different draw.
const EVAL_L = eval_latents()
const EVAL_DS, EVAL_SCORES = eval_datastore(EVAL_L, EVAL_MARKETS)
const EVAL_MODEL = EvalMockPoisson()
const EVAL_LS = Experiments.LatentStates(to_legacy_dataframe(EVAL_L), EVAL_MODEL)

eval_context(metrics; kwargs...) =
    build_evaluation_context(EVAL_L, EVAL_DS.odds, EVAL_DS.matches, metrics;
                             threaded = false, kwargs...)


@testset "Unified evaluation framework" begin

# ==============================================================================
# 1. ODDS VIEW, OUTCOMES AND ALIGNMENT
# ==============================================================================

@testset "OddsView" begin
    o = build_odds_view(EVAL_DS.odds)
    @test o isa OddsView
    @test length(o) == nrow(EVAL_DS.odds)
    @test o.match_id == Int.(EVAL_DS.odds.match_id)
    @test o.selection == Symbol.(EVAL_DS.odds.selection)
    @test all(o.has_fair)
    @test all(o.has_odds_close)
    @test o.is_winner[EVAL_DS.odds.is_winner .== true] == fill(Int8(1),
              count(EVAL_DS.odds.is_winner))
    @test o.is_winner[EVAL_DS.odds.is_winner .== false] == fill(Int8(0),
              count(.!EVAL_DS.odds.is_winner))
    @test o.prob_fair_close ≈ Float64.(EVAL_DS.odds.prob_fair_close)
    @test occursin("OddsView", sprint(show, o))

    # Missingness is a bitvector, not a NaN sentinel: `dropmissing!` drops `missing` and
    # KEEPS a genuine NaN, and collapsing the two would change which rows are scored.
    holed = copy(EVAL_DS.odds)
    holed.prob_fair_close = Vector{Union{Missing, Float64}}(holed.prob_fair_close)
    holed.prob_fair_close[3] = missing
    holed.is_winner = Vector{Union{Missing, Bool}}(holed.is_winner)
    holed.is_winner[5] = missing
    h = build_odds_view(holed)
    @test !h.has_fair[3]
    @test isnan(h.prob_fair_close[3])
    @test h.has_fair[4]
    @test h.is_winner[5] == -1
    @test h.is_winner[4] >= 0

    # An ABSENT column comes back all-missing rather than raising, so a store built
    # without `odds_close` still supports LogLoss.
    thin = select(EVAL_DS.odds, Not(:odds_close))
    t = build_odds_view(thin)
    @test !any(t.has_odds_close)
    @test all(isnan, t.odds_close)

    # …but a column the whole long form is defined by is a loud error naming it.
    @test_throws ErrorException build_odds_view(select(EVAL_DS.odds, Not(:selection)))
end

@testset "MatchOutcomes" begin
    outs = extract_match_outcomes(EVAL_DS.matches)
    @test outs isa MatchOutcomes
    @test length(outs) == nrow(EVAL_DS.matches)
    @test EV.outcome_of(outs, EVAL_DS.matches.match_id[1]) == EVAL_SCORES[1]
    @test EV.outcome_of(outs, -1) === nothing

    # A fixture with no result is ABSENT, not present-with-a-sentinel. The legacy CRPS
    # and RQR kernels innerjoin against `ds.matches` without dropping missing scores, so
    # a not-yet-played fixture makes `compute_crps(missing, …)` raise a MethodError
    # several minutes into an evaluation.
    partial = copy(EVAL_DS.matches)
    partial.home_score = Vector{Union{Missing, Int}}(partial.home_score)
    partial.home_score[2] = missing
    p = extract_match_outcomes(partial)
    @test length(p) == nrow(partial) - 1
    @test EV.outcome_of(p, partial.match_id[2]) === nothing

    @test length(extract_match_outcomes(DataFrame(match_id = Int[]))) == 0
end

@testset "Selection to market inverse" begin
    @test EV.market_for_selection(:home) == Data.Market1X2()
    @test EV.market_for_selection(:btts_no) == Data.MarketBTTS()
    @test EV.market_for_selection(:over_25) == Data.MarketOverUnder(2.5)
    @test EV.market_for_selection(:under_105) == Data.MarketOverUnder(10.5)
    @test EV.market_for_selection(:over_05) == Data.MarketOverUnder(0.5)
    @test EV.market_for_selection(:DC_1X) === nothing

    @test EV.selections_to_markets([:home, :draw, :away]) == [Data.Market1X2()]
    @test EV.selections_to_markets([:home, :over_25]) ==
          [Data.Market1X2(), Data.MarketOverUnder(2.5)]
    @test_throws ErrorException EV.selections_to_markets([:DC_1X])

    # The round trip: every market's own selections map back to it.
    for m in EVAL_MARKETS, s in market_keys(m)
        @test EV.market_for_selection(s) == m
    end
    @test length(EV.market_selections(EVAL_MARKETS)) == 7

    # `markets` is what to PRICE and `selections` is what to SCORE — a filtered trigger
    # tells the pricer it needs one market, not forty.
    @test EV.scored_markets(EV.LogLoss(:over_25)) == [Data.MarketOverUnder(2.5)]
    @test EV.scored_markets(EV.LogLoss()) == EV.DEFAULT_SCORED_MARKETS
    @test EV.scored_markets(EV.CRPS()) == Data.Markets.AbstractMarket[]
    @test EV.needs_outcomes(EV.CRPS()) && EV.needs_outcomes(EV.RQR())
    @test !EV.needs_outcomes(EV.LogLoss())
    @test !EV.needs_draws(EV.LogLoss()) && EV.needs_draws(EV.LPD())
    @test EV.scored_selections(EV.GLMEdge(:home)) == [:home]
end

@testset "Evaluation rows and alignment" begin
    ctx = eval_context([EV.LogLoss()])
    rows = evaluation_rows(ctx)

    # The typed join is the legacy `innerjoin` + `dropmissing!`, row for row and in the
    # same order — which is load-bearing, because every aggregate downstream is a `mean`
    # and floating-point addition is not associative.
    @test length(rows) == nrow(EVAL_DS.odds)
    @test [r.match_id for r in rows] == Int.(EVAL_DS.odds.match_id)
    @test [r.selection for r in rows] == Symbol.(EVAL_DS.odds.selection)
    @test rows[1] isa EvaluationRow
    @test isconcretetype(EvaluationRow)
    @test rows isa Vector{EvaluationRow}

    # Each row's model price really is its own fixture's.
    p = ctx.probs
    @test all(r -> p.match_ids[r.fixture] == r.match_id, rows)
    @test all(r -> p.selections[r.column] == r.selection, rows)
    @test all(r -> r.model_prob == p.means[r.fixture, r.column], rows)

    report = verify_alignment(ctx)
    @test report isa AlignmentReport
    @test report.ok
    @test report.n_odds_rows == nrow(EVAL_DS.odds)
    @test report.n_scored == length(rows)
    @test report.duplicate_keys == 0
    @test report.mismatched_ids == 0
    @test report.n_priced_fixtures == n_matches(EVAL_L)
    @test occursin("OK", sprint(show, report))
    @test occursin("duplicate keys", sprint(show, MIME"text/plain"(), report))

    # A duplicated (match_id, selection) double-weights that fixture in every mean. The
    # legacy innerjoin has the same exposure and no check; this one says so.
    dup_odds = vcat(EVAL_DS.odds, EVAL_DS.odds[1:1, :])
    dup = build_evaluation_context(EVAL_L, dup_odds, EVAL_DS.matches, [EV.LogLoss()];
                                   threaded = false)
    dup_report = verify_alignment(dup)
    @test dup_report.duplicate_keys == 1
    @test !dup_report.ok

    # An odds row for a fixture the model never priced is dropped, exactly as the inner
    # join would have dropped it.
    extra = copy(EVAL_DS.odds[1:1, :])
    extra.match_id = [999_999]
    widened = build_evaluation_context(EVAL_L, vcat(EVAL_DS.odds, extra),
                                       EVAL_DS.matches, [EV.LogLoss()]; threaded = false)
    wr = verify_alignment(widened)
    @test wr.n_unpriced == 1
    @test wr.n_scored == length(rows)

    # Filters and the two `dropmissing!` switches.
    @test length(evaluation_rows(ctx; selections = [:over_25])) == n_matches(EVAL_L)
    @test isempty(evaluation_rows(ctx; selections = [:not_a_selection]))

    # A batch of CRPS and RQR alone prices NOTHING — neither reads a market price.
    bare = eval_context([EV.CRPS(), EV.RQR()])
    @test isempty(bare.probs.selections)
    @test bare.odds.n == 0
    @test length(bare.outcomes) == nrow(EVAL_DS.matches)
end


# ==============================================================================
# 2. ZERO-ALLOCATION BATCH PRICING
# ==============================================================================

@testset "Batch pricing" begin
    probs = market_probabilities(EVAL_L, EVAL_MARKETS; threaded = false)
    @test probs isa MarketProbabilities
    @test size(probs.means) == (n_matches(EVAL_L), 7)
    @test size(probs.draws) == (n_draws(EVAL_L), n_matches(EVAL_L), 7)
    @test probs.match_ids == latent_match_ids(EVAL_L)
    @test EV.priced_selections(probs) == EV.market_selections(EVAL_MARKETS)
    @test EV.probability_bytes(probs) == sizeof(probs.draws) + sizeof(probs.means)
    @test occursin("MarketProbabilities", sprint(show, probs))

    # The stored mean IS the mean of the stored draws, in the same order — the property
    # every kernel that reads `means` instead of `draws` silently depends on.
    for i in (1, 12, n_matches(EVAL_L)), c in 1:7
        @test probs.means[i, c] === mean(view(probs.draws, :, i, c))
    end
    @test probs.means == [mean(view(probs.draws, :, i, c))
                          for i in axes(probs.means, 1), c in axes(probs.means, 2)]

    # 1X2 is a partition; BTTS and O/U are complements. All three therefore sum to the
    # SAME total mass, which is the grid's — slightly under 1, because the score grid
    # truncates at `max_goals = 12`. The sharp invariant is that the three agree with
    # each other; the loose one is how much mass the truncation costs.
    for i in (1, 9, n_matches(EVAL_L)), k in (1, 40, n_draws(EVAL_L))
        one_x_two = sum(probs.draws[k, i, 1:3])
        over_under = probs.draws[k, i, 4] + probs.draws[k, i, 5]
        btts = probs.draws[k, i, 6] + probs.draws[k, i, 7]
        @test one_x_two ≈ over_under atol = 1e-12
        @test one_x_two ≈ btts atol = 1e-12
        @test one_x_two <= 1.0 + 1e-12
        @test one_x_two > 1.0 - 1e-4
    end

    # Threading changes the wall clock and nothing else.
    threaded = market_probabilities(EVAL_L, EVAL_MARKETS; threaded = true)
    @test threaded.means == probs.means
    @test threaded.draws == probs.draws

    # The tensor holds exactly what the score-grid kernels produce — 0 ULP, not "close".
    ws = GridWorkspace(12)
    S = alloc_score_grid(EVAL_L, 12)
    for i in (1, 7, n_matches(EVAL_L))
        compute_score_grid!(S, ws, EVAL_L, i)
        off = 0
        for m in EVAL_MARKETS
            book = alloc_market_book(m, n_draws(EVAL_L))
            price_market!(book, S, m)
            for s in eachindex(book)
                @test probs.draws[:, i, off + s] == book[s]
            end
            off += length(book)
        end
    end

    # Means-only skips the tensor entirely: ~95 MB not allocated on a real fold.
    lean = market_probabilities(EVAL_L, EVAL_MARKETS; keep_draws = false, threaded = false)
    @test size(lean.draws) == (0, 0, 0)
    @test lean.means == probs.means
    @test prob_mean(lean, latent_match_ids(EVAL_L)[2], :home) == lean.means[2, 1]
    @test prob_mean(lean, 999_999, :home) === nothing
    @test prob_mean(lean, latent_match_ids(EVAL_L)[2], :nope) === nothing
    # …and asking for draws it does not have raises, rather than silently returning the
    # mean, which would turn LPD into log-loss.
    @test_throws ErrorException prob_draws(lean, latent_match_ids(EVAL_L)[2], :home)
    @test prob_draws(probs, latent_match_ids(EVAL_L)[2], :home) ==
          probs.draws[:, 2, 1]

    # No markets at all is an empty container, not an error.
    empty_probs = market_probabilities(EVAL_L, Data.Markets.AbstractMarket[])
    @test isempty(empty_probs.selections)
    @test EV.probability_bytes(empty_probs) == 0

    # A selection claimed by two markets in one list would silently overwrite a column.
    @test_throws ErrorException market_probabilities(
        EVAL_L, [Data.Market1X2(), Data.Market1X2()]; threaded = false)
end

@testset "Zero-allocation pricing kernel" begin
    wsp = alloc_evaluation_workspace(EVAL_L, EVAL_MARKETS)
    @test wsp isa EvaluationWorkspace
    @test length(wsp.markets) == 3
    @test wsp.offsets == [0, 3, 5]
    @test wsp.smile === nothing
    @test occursin("EvaluationWorkspace", sprint(show, wsp))

    probs = market_probabilities(EVAL_L, EVAL_MARKETS; threaded = false)
    reference = copy(probs.means)

    # Warm, then measure against an empty-closure baseline.
    price_match_markets!(probs, wsp, EVAL_L, 1)
    @test @allocated(price_match_markets!(probs, wsp, EVAL_L, 1)) == 0
    @test @allocated(price_match_markets!(probs, wsp, EVAL_L, 5)) == 0

    # A whole re-sweep through the same workspace allocates nothing and reproduces the
    # tensor bit for bit.
    sweep(p, w, l) = (for i in 1:n_matches(l); price_match_markets!(p, w, l, i); end; nothing)
    sweep(probs, wsp, EVAL_L)
    @test @allocated(sweep(probs, wsp, EVAL_L)) == 0
    @test probs.means == reference

    # Means-only is the same kernel with the tensor write dispatched away.
    lean = market_probabilities(EVAL_L, EVAL_MARKETS; keep_draws = false, threaded = false)
    price_match_markets!(lean, wsp, EVAL_L, 1)
    @test @allocated(price_match_markets!(lean, wsp, EVAL_L, 1)) == 0
end


# ==============================================================================
# 3. METRIC PARITY AGAINST THE LEGACY KERNELS
# ==============================================================================
#
# Not a transcription: the legacy side runs the real `model_inference` → four-column
# `innerjoin` → `dropmissing!` path over an `Experiments.LatentStates`, fed a DataFrame
# built from the same typed container the new side reads.
#
# The gate is 1e-12 rather than 0 ULP for two of the four, and that is a weaker claim
# said plainly: the legacy side accumulates its per-row scores in the order
# `DataFrames.innerjoin` emitted them, and that order is documented as unspecified. Two
# of the four come out bit-identical anyway, and the tests say which.

@testset "Metric parity" begin
    fit = eval_fit(EVAL_L)
    exp_res = experiment_from_fit(fit)
    ctx = eval_context([EV.LogLoss(), EV.LPD(), EV.CRPS(), EV.GLMEdge()])

    # --- LogLoss: BIT-IDENTICAL ------------------------------------------------
    legacy_ll = EV.compute_metric(EV.LogLoss(), exp_res, EVAL_DS, EVAL_LS)
    typed_ll = EV.compute_metric(EV.LogLoss(), ctx)
    @test typed_ll.overall.n_obs == legacy_ll.overall.n_obs
    @test typed_ll.overall.model_ll === legacy_ll.overall.model_ll
    @test typed_ll.overall.market_ll === legacy_ll.overall.market_ll
    @test typed_ll.overall.diff_ll === legacy_ll.overall.diff_ll

    # …including under a selection filter, where the legacy path prices forty markets to
    # answer a question about one.
    legacy_ou = EV.compute_metric(EV.LogLoss(:over_25), exp_res, EVAL_DS, EVAL_LS)
    typed_ou = EV.compute_metric(EV.LogLoss(:over_25), eval_context([EV.LogLoss(:over_25)]))
    @test typed_ou.overall.n_obs == n_matches(EVAL_L)
    @test typed_ou.overall.model_ll === legacy_ou.overall.model_ll
    @test typed_ou.overall.diff_ll === legacy_ou.overall.diff_ll

    # --- CRPS: BIT-IDENTICAL, and computable at all only because of the
    #     `get_latent_column_symbols(::AbstractPoissonModel)` fix (see §5).
    legacy_crps = EV.compute_metric(EV.CRPS(), exp_res, EVAL_DS, EVAL_LS)
    typed_crps = EV.compute_metric(EV.CRPS(), ctx)
    @test typed_crps.home.mean === legacy_crps.home.mean
    @test typed_crps.away.mean === legacy_crps.away.mean
    @test typed_crps.all.mean === legacy_crps.all.mean

    # --- LPD: agrees to 1e-12; the moments are the leaves that move.
    legacy_lpd = EV.compute_metric(EV.LPD(), exp_res, EVAL_DS, EVAL_LS)
    typed_lpd = EV.compute_metric(EV.LPD(), ctx)
    @test typed_lpd.overall.n_obs == legacy_lpd.overall.n_obs
    for f in (:model_lpd, :model_std, :model_skewness, :model_kurtosis,
              :market_lpd, :diff_lpd, :elpd)
        a = getproperty(typed_lpd.overall, f)
        b = getproperty(legacy_lpd.overall, f)
        @test abs(a - b) <= 1e-12 * max(abs(a), abs(b), 1.0)
    end

    # --- GLMEdge: iteratively reweighted least squares amplifies the last bits.
    legacy_glm = EV.compute_metric(EV.GLMEdge(), exp_res, EVAL_DS, EVAL_LS)
    typed_glm = EV.compute_metric(EV.GLMEdge(), ctx)
    @test typed_glm.n_obs == legacy_glm.n_obs
    for part in (:intercept, :prob_fair, :spread_fair)
        a = getproperty(typed_glm, part)
        b = getproperty(legacy_glm, part)
        for f in (:coef, :std_error, :z_score, :p_value)
            x = getproperty(a, f)
            y = getproperty(b, f)
            @test abs(x - y) <= 1e-9 * max(abs(x), abs(y), 1.0)
        end
    end

    # --- MIQ: the same twelve groups, over the same posterior.
    miq_ctx = eval_context([EV.MIQ()])
    typed_miq = EV.compute_metric(EV.MIQ(), miq_ctx)
    legacy_miq = EV.compute_metric(EV.MIQ(), exp_res, EVAL_DS, EVAL_LS)
    @test typed_miq.all.n_winners == legacy_miq.all.n_winners
    @test typed_miq.all.n_losers == legacy_miq.all.n_losers
    @test typed_miq.all.mean ≈ legacy_miq.all.mean atol = 1e-12
    @test typed_miq.all.mean_gap ≈ legacy_miq.all.mean_gap atol = 1e-12
    @test typed_miq.home.mean_gap ≈ legacy_miq.home.mean_gap atol = 1e-12
    # A store that quotes no O/U 1.5 line leaves those groups empty, and they come back
    # `missing` rather than as a fabricated zero.
    @test ismissing(typed_miq.over_15.mean_gap)
    @test typed_miq.over_15.n_winners == 0

    # --- LPD is the negative of LogLoss on a binary market, exactly.
    #     log((1/S) Σ p_s) = log(p̄) = −LL(p̄) for y = 1, and likewise for y = 0. An
    #     independent identity, so a bug in either kernel breaks it.
    @test typed_lpd.overall.model_lpd ≈ -typed_ll.overall.model_ll atol = 1e-12
    @test typed_lpd.overall.market_lpd ≈ -typed_ll.overall.market_ll atol = 1e-12

    # --- LPD(target = :score): the joint density of the realised scoreline.
    score_ctx = eval_context([EV.LPD(), EV.CRPS()])
    scored = EV.compute_metric(EV.LPD(), score_ctx; target = :score)
    @test scored.overall.n_obs == n_matches(EVAL_L)
    @test isfinite(scored.overall.model_lpd)
    @test scored.overall.model_lpd < 0
    # There is no quoted correct-score distribution, so no baseline is invented.
    @test isnan(scored.overall.market_lpd)
    @test isnan(scored.overall.diff_lpd)
end

@testset "RQR is reproducible and seeded" begin
    ctx = eval_context([EV.RQR()])
    a = EV.compute_metric(EV.RQR(), ctx)
    b = EV.compute_metric(EV.RQR(), ctx)

    # The legacy kernel draws from the unseeded global RNG, so two consecutive calls on
    # identical inputs disagree and the diagnostic cannot be re-checked.
    for part in (:home, :away, :all)
        x = getproperty(a, part)
        y = getproperty(b, part)
        for f in (:mean, :std, :skewness, :kurtosis, :shapiro_w, :shapiro_p)
            @test getproperty(x, f) === getproperty(y, f)
        end
    end

    # A different seed is a different sample; both are valid RQR draws.
    c = EV.compute_metric(EV.RQR(), ctx; seed = 1234)
    @test c.all.mean != a.all.mean

    # Evaluating never perturbs the caller's global RNG.
    Random.seed!(11)
    before = rand()
    Random.seed!(11)
    EV.compute_metric(EV.RQR(), ctx)
    @test rand() === before

    # `all` POOLS home and away (vcat), unlike `CRPSResults.all` which averages them.
    # Two conventions in two adjacent legacy files; both preserved.
    @test isfinite(a.all.std)
    # Replicates average SUMMARIES, not residuals — averaging residuals would shrink them
    # toward the mid-quantile normal score and manufacture normality.
    r5 = EV.compute_metric(EV.RQR(), ctx; n_sims = 5)
    @test isfinite(r5.all.shapiro_p)
    @test r5.all.mean != a.all.mean
end

@testset "Goal-distribution dispatch" begin
    # A container with no dispersion cannot reach the negative-binomial method at all —
    # unlike the `hasproperty(df, :r)` cascade it replaces, where a frame carrying
    # neither column reaches Poisson by falling off the end of an `if`.
    dh, da = marginals(EVAL_L, 1)
    @test dh isa Poisson
    @test da isa Poisson
    @test mean(dh) === EV.posterior_mean(EVAL_L.λ_home, 1)
    @test EV.posterior_mean(EVAL_L.λ_home, 3) == mean(EVAL_L.λ_home[3, :])

    λh, rh, λa, ra = crps_parameters(EVAL_L, 1)
    @test isinf(rh) && isinf(ra)
    @test λh === mean(dh)

    nb = CountLatents(latent_match_ids(EVAL_L), EVAL_L.λ_home, EVAL_L.λ_away,
                      (; r_h = fill(4.0, size(EVAL_L.λ_home)),
                         r_a = fill(6.0, size(EVAL_L.λ_away))))
    nh, na = marginals(nb, 2)
    @test nh isa NegativeBinomial
    @test na isa NegativeBinomial
    _, nrh, _, nra = crps_parameters(nb, 2)
    @test nrh == 4.0 && nra == 6.0
end


# ==============================================================================
# 4. CALIBRATION, BRIER, RPS AND PIT
# ==============================================================================

@testset "Brier, ECE and the calibration curve" begin
    # Hand-built rows with known answers, so the closed form is the reference rather
    # than the implementation.
    mk(p, y) = EvaluationRow(1, :home, 1, 1, p, p, Int8(y))
    rows = [mk(0.25, 0), mk(0.25, 1), mk(0.75, 1), mk(0.75, 1)]

    b, n = brier_score(rows)
    @test n == 4
    @test b ≈ (0.25^2 + 0.75^2 + 0.25^2 + 0.25^2) / 4
    @test brier_score(0.3, 1.0) ≈ 0.49

    curve = calibration_curve(rows; n_bins = 4)
    @test curve isa CalibrationCurve
    @test length(curve) == 4
    @test curve.edges == [0.0, 0.25, 0.5, 0.75, 1.0]
    @test curve.counts == [0, 2, 0, 2]
    @test isnan(curve.mean_predicted[1])          # empty bins are NaN, not 0.0
    @test isnan(curve.observed[3])
    @test curve.mean_predicted[2] ≈ 0.25
    @test curve.observed[2] ≈ 0.5
    @test curve.mean_predicted[4] ≈ 0.75
    @test curve.observed[4] ≈ 1.0
    @test sum(curve.counts) == 4

    # ECE is the count-weighted mean gap; MCE is the largest one.
    @test expected_calibration_error(curve) ≈ (2 * 0.25 + 2 * 0.25) / 4
    @test max_calibration_error(curve) ≈ 0.25
    @test expected_calibration_error(rows; n_bins = 4) ≈ expected_calibration_error(curve)

    # A perfectly calibrated set has ECE 0 to floating point.
    perfect = vcat([mk(0.2, 1)], [mk(0.2, 0) for _ in 1:4],
                   [mk(0.8, 1) for _ in 1:4], [mk(0.8, 0)])
    pc = calibration_curve(perfect; n_bins = 5)
    @test expected_calibration_error(pc) ≈ 0.0 atol = 1e-12
    @test max_calibration_error(pc) ≈ 0.0 atol = 1e-12

    # `p == 1.0` lands in the LAST bin rather than off the end.
    edgey = calibration_curve([mk(1.0, 1), mk(0.0, 0)]; n_bins = 4)
    @test edgey.counts == [1, 0, 0, 1]

    @test expected_calibration_error(EvaluationRow[]) |> isnan
    @test occursin("CalibrationCurve", sprint(show, curve))
    @test occursin("predicted", sprint(show, MIME"text/plain"(), curve))
end

@testset "Ranked probability score" begin
    ctx = eval_context([EV.PredictionScore()])
    rps, n = ranked_probability_score(ctx)
    @test n == n_matches(EVAL_L)
    @test 0.0 <= rps <= 1.0

    market_rps, mn = ranked_probability_score(ctx; source = :market)
    @test mn == n
    @test 0.0 <= market_rps <= 1.0

    # A perfect forecast scores 0 and a maximally wrong ORDERED one scores 1 — the
    # property that distinguishes RPS from a sum of binary scores.
    #   home win forecast (1,0,0), home wins  -> 0
    #   away win forecast (0,0,1), home wins  -> ((0-1)^2 + (0-1)^2)/2 = 1
    #   draw forecast     (0,1,0), home wins  -> ((0-1)^2 + (1-1)^2)/2 = 0.5
    for (probs, expected) in (([1.0, 0.0, 0.0], 0.0),
                              ([0.0, 0.0, 1.0], 1.0),
                              ([0.0, 1.0, 0.0], 0.5))
        one_l = CountLatents([1], reshape([1.0], 1, 1), reshape([1.0], 1, 1))
        odds = DataFrame(match_id = [1, 1, 1],
                         market_name = fill("1X2", 3),
                         market_line = zeros(3),
                         selection = [:home, :draw, :away],
                         prob_fair_close = probs,
                         is_winner = [true, false, false])
        matches = DataFrame(match_id = [1], home_score = [2], away_score = [0])
        c = build_evaluation_context(one_l, odds, matches, [EV.PredictionScore()];
                                     threaded = false)
        got, k = ranked_probability_score(c; source = :market)
        @test k == 1
        @test got ≈ expected atol = 1e-12
    end
end

@testset "evaluate_predictions" begin
    ctx = eval_context([EV.PredictionScore()])
    scores = evaluate_predictions(ctx)

    @test scores isa PredictionScores
    @test scores.model isa EV.CalibrationComponent
    @test scores.model.n_obs == nrow(EVAL_DS.odds)
    @test scores.market.n_obs == scores.model.n_obs
    @test 0.0 < scores.model.brier < 1.0
    @test 0.0 <= scores.model.ece <= 1.0
    @test scores.model.mce >= scores.model.ece
    @test isfinite(scores.model.rps)

    # The model's log-loss here is the same number LogLoss reports — the same rows, the
    # same formula, one entry point that also asks whether it is calibrated.
    ll = EV.compute_metric(EV.LogLoss(), eval_context([EV.LogLoss()]))
    @test scores.model.logloss ≈ ll.overall.model_ll atol = 1e-12
    @test scores.market.logloss ≈ ll.overall.market_ll atol = 1e-12

    # A filtered scope scores only that selection.
    sub = evaluate_predictions(ctx; selections = [:home])
    @test sub.model.n_obs == n_matches(EVAL_L)

    # …and the trigger form goes through a batch like any other rule.
    via_trigger = EV.compute_metric(EV.PredictionScore(), ctx)
    @test via_trigger.model.brier === scores.model.brier
    @test EV.get_metric_method_name(EV.PredictionScore()) == "predictions_all"
    @test EV.get_metric_method_name(EV.PredictionScore(:home)) == "predictions_home"
    @test EV.get_metric_method_name(via_trigger) == "predictions"
    @test EV.needs_outcomes(EV.PredictionScore())
end

@testset "PIT" begin
    ctx = eval_context([EV.CRPS()])
    r = pit_values(ctx)
    @test r isa PITReport
    @test r.n_obs == n_matches(EVAL_L)
    @test length(r.u_home) == r.n_obs
    @test all(0.0 .<= r.u_home .<= 1.0)
    @test all(0.0 .<= r.u_away .<= 1.0)
    @test pit_uniformity(r) == (r.ks_stat, r.p_value)
    @test occursin("PITReport", sprint(show, r))

    # Seeded and private: two calls agree, and the caller's global RNG is untouched.
    @test pit_values(ctx).u_home == r.u_home
    @test pit_values(ctx; rng = Xoshiro(1)).u_home != r.u_home
    Random.seed!(5); before = rand()
    Random.seed!(5); pit_values(ctx)
    @test rand() === before

    # The scores were SIMULATED from these λ, so the transform should look uniform.
    @test r.p_value > 0.01

    # A badly mis-specified predictive does not. Scoring a 3-goal-a-side reality against
    # a model that expects 0.05 pushes every PIT value to the top of the unit interval.
    tiny = CountLatents(latent_match_ids(EVAL_L),
                        fill(0.05, size(EVAL_L.λ_home)),
                        fill(0.05, size(EVAL_L.λ_away)))
    hot = DataFrame(match_id = latent_match_ids(EVAL_L),
                    home_score = fill(3, n_matches(EVAL_L)),
                    away_score = fill(3, n_matches(EVAL_L)))
    bad = pit_values(tiny, extract_match_outcomes(hot))
    @test bad.p_value < 0.01
    @test mean(bad.u_home) > 0.9
end


# ==============================================================================
# 5. UPSTREAM DEFECT FIXES
# ==============================================================================

@testset "unroll accepts missing" begin
    # `MIQStats`' fields are `Union{Missing, Float64}`, and `unroll` had methods for
    # `Real` and `AbstractMetricComponent` only — so `to_dataframe_row` on an MIQResult
    # with an empty selection group raised MethodError inside `evaluate_experiments`'
    # `try`, dropping the model's ENTIRE row with only a `@warn`. Any store that does not
    # quote O/U 1.5 and 3.5 guarantees that.
    @test EV.unroll("x", missing) === (; x = missing)
    @test EV.unroll("x", 1.5) === (; x = 1.5)

    stats = EV.MIQStats(missing, missing, missing, missing, missing, 0, 0)
    flat = EV.unroll("miq_over_15", stats)
    @test ismissing(flat.miq_over_15_mean)
    @test flat.miq_over_15_n_winners == 0

    fit = eval_fit(EVAL_L)
    exp_res = experiment_from_fit(fit)
    result = EV.compute_metric(EV.MIQ(), eval_context([EV.MIQ()]))
    row = EV.to_dataframe_row(exp_res, EV.MIQ(), result)
    @test row.model == "eval_mock"
    @test ismissing(row.miq_over_15_mean_gap)
    @test !ismissing(row.miq_all_mean_gap)
end

@testset "Poisson latent column schema" begin
    # `get_latent_column_symbols` had methods for `AbstractNegBinModel` and the
    # Frank-copula model and nothing else, while the legacy CRPS and RQR kernels call it
    # unconditionally — so every Poisson engine raised MethodError inside
    # `evaluate_experiments`' `try` and vanished from the leaderboard with no trace.
    cols = Predictions.get_latent_column_symbols(EVAL_MODEL, EVAL_LS.df)
    @test cols == [:match_id, :λ_h, :λ_a]
    @test all(c -> c in propertynames(EVAL_LS.df), cols)

    # The legacy kernels that call it now run for a Poisson model.
    fit = eval_fit(EVAL_L)
    exp_res = experiment_from_fit(fit)
    @test EV.compute_metric(EV.CRPS(), exp_res, EVAL_DS, EVAL_LS) isa EV.CRPSResults
    @test EV.compute_metric(EV.RQR(), exp_res, EVAL_DS, EVAL_LS) isa EV.RQRResult
end


# ==============================================================================
# 6. COMPATIBILITY
# ==============================================================================

@testset "as_typed_latents" begin
    @test as_typed_latents(EVAL_L) === EVAL_L
    @test as_typed_latents(EVAL_LS, EVAL_MODEL) isa CountLatents
    @test as_typed_latents(EVAL_LS) isa CountLatents            # reads `.model` itself

    rebuilt = as_typed_latents(EVAL_LS.df, EVAL_MODEL)
    @test rebuilt isa CountLatents
    @test latent_match_ids(rebuilt) == latent_match_ids(EVAL_L)
    @test rebuilt.λ_home ≈ EVAL_L.λ_home

    # A frame with no model is not enough: the family determines the schema, and it is a
    # property of the engine rather than of the columns.
    @test_throws ErrorException as_typed_latents(EVAL_LS.df)
    @test_throws ErrorException as_typed_latents(42)
end

@testset "Typed entry points" begin
    fit = eval_fit(EVAL_L)

    @test fit_latents(fit) === EVAL_L
    direct = EV.compute_metric(EV.LogLoss(), fit, EVAL_DS)
    from_ctx = EV.compute_metric(EV.LogLoss(), eval_context([EV.LogLoss()]))
    @test direct.overall.model_ll === from_ctx.overall.model_ll

    # The four-argument shape, for a caller that already holds its latents in any of the
    # three shapes.
    for lat in (EVAL_L, EVAL_LS, EVAL_LS.df)
        got = EV.compute_metric(EV.LogLoss(), fit, EVAL_DS, lat)
        @test got.overall.n_obs == from_ctx.overall.n_obs
    end

    # Keywords reach the kernel.
    @test EV.compute_metric(EV.CRPS(), fit, EVAL_DS; max_goals = 40).all.mean ≈
          EV.compute_metric(EV.CRPS(), fit, EVAL_DS; max_goals = 30).all.mean

    # A converged fit passes the gate; an unaudited container does not.
    passed, gates, _ = convergence_verdict(fit)
    @test passed
    @test isempty(gates)
    @test EV.compute_metric(EV.LogLoss(), fit, EVAL_DS; require_converged = true) isa
          EV.LogLossResult

    bad = eval_fit(EVAL_L; name = "wobbly", seed = 4, n = 40, n_chains = 2)
    bad_passed, bad_gates, _ = convergence_verdict(bad)
    @test !bad_passed
    @test !isempty(bad_gates)
    @test_throws ConvergenceRefusal EV.compute_metric(EV.LogLoss(), bad, EVAL_DS;
                                                      require_converged = true)
    @test occursin("did not converge",
                   sprint(showerror, ConvergenceRefusal("x", ["R-hat"], String[])))

    # An unaudited container counts as NOT converged, for the same reason the audit
    # abstains on an unmeasured gate.
    @test convergence_verdict((; diagnostics = nothing)) == (false, ["no audit"],
        ["this container carries no ConvergenceSummary — re-run `fit_model`, or " *
         "`audit_convergence(fit)` if you have the folds."])

    @test as_fit(fit, EVAL_DS) === fit
end

@testset "Legacy path is unchanged" begin
    fit = eval_fit(EVAL_L)
    exp_res = experiment_from_fit(fit)

    # `evaluate_experiments` is untouched and still carries its legacy signatures.
    # It is NOT driven end to end here: its first act is
    # `Experiments.extract_oos_predictions(ds, exp)`, which re-derives the split
    # boundaries and rebuilds every fold's feature set from the `DataStore` — the exact
    # re-derivation the typed path exists to avoid, and something a store assembled from
    # `matches` and `odds` alone cannot satisfy. What IS driven end to end is the legacy
    # kernel plus the legacy translator, which is where the column names come from.
    @test hasmethod(EV.evaluate_experiments,
                    Tuple{Vector{<:EV.AbstractScoringRule},
                          Vector{Experiments.ExperimentResults}, Data.DataStore})
    @test hasmethod(EV.evaluate_experiments,
                    Tuple{EV.AbstractScoringRule,
                          Vector{Experiments.ExperimentResults}, Data.DataStore})

    # The legacy runner's body: score each metric, flatten, merge.
    combined = (; model = "eval_mock")
    for m in (EV.LogLoss(), EV.CRPS())
        result = EV.compute_metric(m, exp_res, EVAL_DS, EVAL_LS)
        flat = EV.to_dataframe_row(exp_res, m, result)
        combined = merge(combined, Base.structdiff(flat, (; model = "")))
    end
    df = DataFrame([combined])
    @test nrow(df) == 1
    @test df.model == ["eval_mock"]
    for c in ("logloss_overall_model_ll", "logloss_overall_market_ll",
              "logloss_overall_diff_ll", "crps_home_mean", "crps_away_mean",
              "crps_all_mean")
        @test c in names(df)
    end

    # `display_summary_metric` still curates the legacy column list off such a frame.
    @test EV.display_summary_metric(df, :logloss) === nothing
    @test EV.display_summary_metric(df, :not_a_family) === nothing

    # …and the typed flattener produces the SAME column names, which is what makes a new
    # row and an old CSV line up.
    typed_row = EV.flatten_result("eval_mock", EV.LogLoss(),
                                  EV.compute_metric(EV.LogLoss(), eval_context([EV.LogLoss()])))
    legacy_row = EV.to_dataframe_row(exp_res, EV.LogLoss(),
                                     EV.compute_metric(EV.LogLoss(), exp_res, EVAL_DS, EVAL_LS))
    @test propertynames(typed_row) == propertynames(legacy_row)
    @test typed_row.logloss_overall_model_ll === legacy_row.logloss_overall_model_ll

    # The per-selection suffix keeps two rules of one family from overwriting each other.
    @test EV.metric_column_suffix(EV.LogLoss()) == ""
    @test EV.metric_column_suffix(EV.LogLoss(:over_25)) == "_over_25"
    filtered = EV.flatten_result("m", EV.LogLoss(:over_25),
                                 EV.compute_metric(EV.LogLoss(:over_25),
                                                   eval_context([EV.LogLoss(:over_25)])))
    @test :logloss_over_25_overall_diff_ll in propertynames(filtered)
end

@testset "evaluate_fits and the convergence gate" begin
    good = eval_fit(EVAL_L; name = "converged")
    bad = eval_fit(EVAL_L; name = "unconverged", seed = 4, n = 40, n_chains = 2)
    metrics = [EV.LogLoss(), EV.CRPS(), EV.PredictionScore()]

    # By default the gate FILTERS — and the convergence frame still has a row for every
    # submitted fit, because "two models, one of which did not converge" is a different
    # message from "one model".
    report = evaluate_fits(metrics, [good, bad], EVAL_DS; quiet = true, show_tables = false)
    @test report isa EvaluationReport
    @test nrow(report.rows) == 1
    @test report.rows.model == ["converged"]
    @test report.excluded == ["unconverged"]
    @test nrow(report.convergence) == 2
    @test sort(report.convergence.model) == ["converged", "unconverged"]
    @test report.convergence.audited == [true, true]
    @test isempty(report.errors)
    @test report.require_converged

    # The verdict travels WITH the numbers, so a row lifted into a plot or a CSV carries
    # the reason it should or should not be believed.
    @test "converged" in names(report.rows)
    @test "max_rhat" in names(report.rows)
    @test report.rows.converged == [true]

    # …and every metric's columns are there.
    for c in ("logloss_overall_diff_ll", "crps_all_mean",
              "predictions_model_brier", "predictions_market_rps")
        @test c in names(report.rows)
    end

    # `require_converged = false` scores it and flags it instead.
    flagged = evaluate_fits(metrics, [good, bad], EVAL_DS; require_converged = false,
                            quiet = true, show_tables = false)
    @test nrow(flagged.rows) == 2
    @test isempty(flagged.excluded)
    @test sort(flagged.rows.converged) == [false, true]

    # A metric that raises loses its OWN columns and nothing else. The legacy runner
    # drops the entire model on any single metric's failure.
    no_odds = Data.DataStore(Data.ScottishLower(), EVAL_DS.matches, DataFrame(),
                             select(EVAL_DS.odds, Not(:prob_fair_close)),
                             DataFrame(), DataFrame(), DataFrame(), DataFrame(),
                             DataFrame())
    partial = evaluate_fits([EV.LogLoss(), EV.CRPS()], [good], no_odds;
                            quiet = true, show_tables = false)
    @test nrow(partial.rows) == 1
    @test "crps_all_mean" in names(partial.rows)
    @test partial.rows.logloss_overall_n_obs == [0]     # scored nothing, but survived

    @test_throws ErrorException evaluate_fits(EV.AbstractScoringRule[], [good], EVAL_DS)
    @test_throws ErrorException evaluate_fits(metrics, Fit[], EVAL_DS)

    # The single-fit and single-metric convenience forms.
    one = evaluate_fits(EV.LogLoss(), [good], EVAL_DS; quiet = true, show_tables = false)
    @test nrow(one.rows) == 1
    @test nrow(evaluate_fits(metrics, good, EVAL_DS; quiet = true,
                             show_tables = false).rows) == 1
end

@testset "Reporting" begin
    good = eval_fit(EVAL_L; name = "alpha")
    other = eval_fit(EVAL_L; name = "beta", seed = 21)
    report = evaluate_fits([EV.LogLoss(), EV.PredictionScore()], [good, other], EVAL_DS;
                           quiet = true, show_tables = false)
    @test nrow(report.rows) == 2

    board = leaderboard(report, :logloss_overall_diff_ll)
    @test names(board)[1] == "model"
    @test "converged" in names(board)
    @test issorted(board.logloss_overall_diff_ll)
    desc = leaderboard(report, :logloss_overall_diff_ll; higher_is_better = true)
    @test issorted(desc.logloss_overall_diff_ll; rev = true)
    @test_throws ErrorException leaderboard(report, :not_a_column)

    tbl = report_table(report)
    @test names(tbl)[1] == "model"
    @test "converged" in names(tbl)
    @test names(report_table(report; columns = [:logloss_overall_model_ll]))[end] ==
          "logloss_overall_model_ll"

    io = IOBuffer()
    EV.display_convergence(report; io = io)
    text = String(take!(io))
    @test occursin("Convergence", text)
    @test occursin("PASS", text)
    @test occursin("alpha", text)

    md = markdown_report(report; title = "Phase 4")
    @test occursin("# Phase 4", md)
    @test occursin("## Convergence", md)
    @test occursin("## Scores", md)
    @test occursin("| model |", md)
    @test occursin("alpha", md)
    @test occursin("beta", md)

    @test occursin("EvaluationReport(2 scored", sprint(show, report))
    @test occursin("metrics", sprint(show, MIME"text/plain"(), report))
    @test DataFrame(report) === report.rows
    @test nrow(report) == 2 && length(report) == 2
    @test sort(EV.scored_models(report)) == ["alpha", "beta"]

    err = EV.EvaluationError("m", "logloss", "boom")
    @test occursin("boom", sprint(show, err))
end

end # testset "Unified evaluation framework"
