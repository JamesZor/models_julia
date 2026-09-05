# test/test_calibration_v2.jl
#
# Layer-2 calibration v2 — generative rate calibration, its book, and its two contracts.
#
# Every testset here pins one of three things:
#
#   * an EQUIVALENCE with the validated prototype. `PoolDispersion` must reproduce
#     `current_development/calibration_generative_eda/l01_generative_calibrator.jl` BIT FOR
#     BIT, because every published figure in that stream's README was measured on that
#     transform and a production module that is merely close to it is a different model.
#     The prototype is `include`d and RUN here, not transcribed — a transcription would be
#     testing that this file agrees with itself.
#
#   * a PROPERTY the construction claims. Coherence across derivative families; the
#     identity calibrator reproducing raw draws bit for bit; the anchor being exactly zero
#     on the pool map and exactly rate-matching off it; a refused inversion passing
#     through rather than being dropped or imputed.
#
#   * a CONTRACT with a downstream tier. `run_portfolio_simulation(spec, policy, cf, ...)`
#     must be field-identical to the same call on `cf.fit`, and a calibration run must
#     round-trip through `mcmc_experiments` losslessly.
#
# TIERS. Testsets T1-T9 are pure: synthetic fixtures, seeded, no database, no `.cache/`,
# no MCMC. T10 needs `mcmc_experiments` and SKIPS WITH A MESSAGE when it is out of reach —
# a "passed" line from a tier that skipped is not evidence.
#
# Run:  julia --project -t 8 -e 'using Test, BayesianFootball; include("test/test_calibration_v2.jl")'

using Test
using BayesianFootball
using BayesianFootball: Data, Features, Models, Predictions, Training, Evaluation
using DataFrames, Dates, Statistics, Random, LinearAlgebra, MCMCChains, UUIDs

const CAL = BayesianFootball.Calibration
const CPF = BayesianFootball.Portfolio

# ===================================================================
# 0. The prototype, loaded as the reference implementation
# ===================================================================
#
# Wrapped in its own module so its top-level names (`calibrate_latents`, `MarketRateFit`,
# `weight_summary`, ...) shadow the identically-named production exports INSIDE that
# module and nowhere else. Both implementations are then callable, side by side, in one
# session — which is the only way an equivalence gate means anything.

module CalV2Reference
using BayesianFootball
include(joinpath(@__DIR__, "..", "current_development", "calibration_generative_eda",
                 "l01_generative_calibrator.jl"))
include(joinpath(@__DIR__, "..", "current_development", "calibration_generative_eda",
                 "l03_variance_schemes.jl"))
end

const REF = CalV2Reference

# ===================================================================
# 1. Fixtures
# ===================================================================

"A Poisson engine the `Fit` fixture can name. Never sampled — `ReplaySampler` supplies the chain."
struct CalV2MockPoisson <: BayesianFootball.TypesInterfaces.AbstractPoissonModel end

"""
    cv2_latents(; n, n_draws, seed) -> CountLatents

A typed Poisson posterior with a DELIBERATELY ASYMMETRIC spread per side, so a dispersion
map with off-diagonal terms produces a different container from one without and the
supremacy/totals retention columns are not trivially equal.
"""
function cv2_latents(; n = 24, n_draws = 300, seed = 11)
    rng = Xoshiro(seed)
    ids = collect(9001:(9000 + n))
    mh = 1.1 .+ 0.5 .* rand(rng, n)
    ma = 0.8 .+ 0.4 .* rand(rng, n)
    lh = exp.(log.(mh) .+ 0.22 .* randn(rng, n, n_draws))
    la = exp.(log.(ma) .+ 0.31 .* randn(rng, n, n_draws))
    return CountLatents(ids, lh, la)
end

"""
    cv2_true_probs(lh, la; max_goals = 10) -> Dict{Symbol, Float64}

The exact Double-Poisson probabilities of every selection the inversion reads, computed
from `Features.build_probability_matrix` — the SAME primitive the inversion's objective
calls. A book built from these is one the inversion can reproduce to machine precision, so
an inversion refusal in these tests is a bug in the gates and never a bug in the fixture.
"""
function cv2_true_probs(lh::Float64, la::Float64; max_goals::Int = 10)
    P = Features.build_probability_matrix(Features.DoublePoissonMarketFeature(),
                                          [log(lh), log(la)], max_goals)
    home = sum(tril(P, -1)); draw = sum(diag(P)); away = sum(triu(P, 1))
    btts = sum(@views P[2:end, 2:end])
    out = Dict{Symbol, Float64}(:home => home, :draw => draw, :away => away,
                                :btts_yes => btts, :btts_no => 1.0 - btts)
    for k in 0:3
        under = 0.0
        for j in 0:max_goals, i in 0:max_goals
            (i + j) <= k && (under += P[i + 1, j + 1])
        end
        out[Symbol("under_$(k)5")] = under
        out[Symbol("over_$(k)5")] = 1.0 - under
    end
    return out
end

const CV2_MARKET_ROWS = [
    ("1X2", 0.0, [:home, :draw, :away]),
    ("BTTS", 0.0, [:btts_yes, :btts_no]),
    ("OverUnder", 0.5, [:over_05, :under_05]),
    ("OverUnder", 1.5, [:over_15, :under_15]),
    ("OverUnder", 2.5, [:over_25, :under_25]),
    ("OverUnder", 3.5, [:over_35, :under_35]),
]

"""
    cv2_book(ids; seed, as_of, edge, drop_all_for) -> DataFrame

A complete, de-vigged, point-in-time-shaped book: one market group per line, both or all
three sides present, `prob_fair_close` summing to 1 within (match, market, line), and the
schema's own column names plus `:as_of_minutes` / `:staleness_minutes`.

The market rates are the model's own PERTURBED by `edge`, deliberately: a book identical to
the model gives `delta = 0` on every fixture, `w` at the law's peak, and a calibration test
that compares two identical containers.

`drop_all_for` names fixtures the book does not quote at all — the un-invertible set T6
needs, and the reason the fallback is a tested path rather than a documented one.
"""
function cv2_book(l::CountLatents; seed = 31, as_of::Float64 = -25.0, edge = 0.16,
                  drop_all_for = Int[], truth = Dict{Int, Tuple{Float64, Float64}}())
    rng = Xoshiro(seed)
    ids = latent_match_ids(l)
    drop = Set{Int}(drop_all_for)
    rows = NamedTuple[]
    for (i, m) in enumerate(ids)
        m in drop && continue
        lh = median(view(l.λ_home, i, :)) * exp(edge * randn(rng))
        la = median(view(l.λ_away, i, :)) * exp(edge * randn(rng))
        truth[Int(m)] = (lh, la)
        p = cv2_true_probs(lh, la)
        for (name, line, sels) in CV2_MARKET_ROWS
            for s in sels
                push!(rows, (match_id = Int(m), market_name = name, market_line = line,
                             selection = s, odds_close = 1.0 / p[s],
                             prob_implied_close = p[s], prob_fair_close = p[s],
                             overround = 1.0, as_of_minutes = as_of,
                             staleness_minutes = 4.0, tick_minutes = as_of - 4.0,
                             n_ticks_before = 12))
            end
        end
    end
    book = DataFrame(rows)
    sort!(book, [:match_id, :market_name, :market_line, :selection])
    return book
end

"Realised outcomes joined onto a book, so `Portfolio` can settle it."
function cv2_settle!(book::DataFrame, scores::Dict{Int, Tuple{Int, Int}})
    win = Vector{Union{Missing, Bool}}(undef, nrow(book))
    for (r, row) in enumerate(eachrow(book))
        sc = get(scores, row.match_id, nothing)
        if sc === nothing
            win[r] = missing
            continue
        end
        h, a = sc
        s = row.selection
        win[r] = s === :home ? h > a :
                 s === :draw ? h == a :
                 s === :away ? h < a :
                 s === :btts_yes ? (h > 0 && a > 0) :
                 s === :btts_no ? !(h > 0 && a > 0) :
                 startswith(String(s), "over_") ?
                     (h + a) > parse(Float64, replace(String(s), "over_" => "")[1:1] * ".5") :
                     (h + a) < parse(Float64, replace(String(s), "under_" => "")[1:1] * ".5")
    end
    book.is_winner = win
    return book
end

"Three fixtures per settlement window, so a slate is a slate and not a single match."
function cv2_fixtures(ids; seed = 5, per_day = 3)
    rng = Xoshiro(seed)
    d0 = Date(2025, 4, 5)
    return Dict{Int, CPF.FixtureInfo}(
        Int(m) => (date = d0 + Day(div(i - 1, per_day)),
                   score = (rand(rng, 0:3), rand(rng, 0:3)))
        for (i, m) in enumerate(ids))
end

cv2_matches_frame(fx) = DataFrame(
    match_id   = sort!(collect(keys(fx))),
    match_date = [fx[k].date for k in sort!(collect(keys(fx)))],
    home_score = [fx[k].score[1] for k in sort!(collect(keys(fx)))],
    away_score = [fx[k].score[2] for k in sort!(collect(keys(fx)))])

"A `Fit` carrying `l` as its latents, with a chain healthy enough to pass the gates."
function cv2_fit(l; name = "cal_v2_mock", seed = 3, n = 400, n_chains = 4)
    ch = Chains(randn(Xoshiro(seed), n, 2, n_chains), [:a, :b])
    fss = [(BayesianFootball.FeatureSet(:n_teams => 4),
            Data.SplitMetaData(1, "23/24", "24/25", 1, 1, 0))]
    fit = fit_model(FitConfig(name = name, model = CalV2MockPoisson(),
                              splitter = Data.CVConfig(target_seasons = ["24/25"]),
                              sampler = ReplaySampler([ch]),
                              execution = SequentialExecution(),
                              save_dir = mktempdir());
                    feature_sets = fss, quiet = true)
    return Fit(fit.config, fit.folds, l, fit.diagnostics, fit.metadata, fit.save_path)
end

# Built once. Every testset reads the same fixture, so a number that moves between
# testsets is a real difference and not a different draw.
const CV2_L      = cv2_latents()
const CV2_IDS    = latent_match_ids(CV2_L)
const CV2_UNQ    = CV2_IDS[end-2:end]                       # quoted by nobody: the fallback set
const CV2_TRUTH  = Dict{Int, Tuple{Float64, Float64}}()    # the rates the book was built FROM
const CV2_BOOK   = cv2_book(CV2_L; drop_all_for = CV2_UNQ, truth = CV2_TRUTH)
const CV2_FX     = cv2_fixtures(CV2_IDS)
const CV2_MATCH  = cv2_matches_frame(CV2_FX)
const CV2_FIT    = cv2_fit(CV2_L)
cv2_settle!(CV2_BOOK, Dict(k => v.score for (k, v) in CV2_FX))

cv2_cal(; law = InverseGaussianLaw(w_base = 0.25, sigma = 0.35), kw...) =
    GenerativeRateCalibrator(name = "cv2_test", law = law, book_as_of_minutes = -25.0; kw...)

const CV2_RATES = CAL.invert_market_rates(CV2_BOOK; match_ids = CV2_IDS)

# ===================================================================
# T1. The weight laws
# ===================================================================

@testset "T1 — weight laws" begin
    inv = InverseGaussianLaw(w_base = 0.25, sigma = 0.35)
    std = StandardGaussianLaw(w_base = 0.40, sigma = 0.15, w_max = 1.0)
    sta = StaticGeometricLaw(w = 0.40)

    # The three closed forms at their two anchors.
    @test CAL.calibration_weight(inv, 0.0) ≈ 0.25
    @test CAL.calibration_weight(inv, 50.0) ≈ 1.0
    @test CAL.calibration_weight(std, 0.0) ≈ 1.0
    @test CAL.calibration_weight(std, 50.0) ≈ 0.40
    @test CAL.calibration_weight(sta, 0.0) == 0.40
    @test CAL.calibration_weight(sta, 50.0) == 0.40

    # Monotone in |delta|, in opposite directions, and symmetric in its sign.
    ds = 0.0:0.05:2.0
    @test issorted([CAL.calibration_weight(inv, d) for d in ds])
    @test issorted([CAL.calibration_weight(std, d) for d in ds]; rev = true)
    for d in (0.1, 0.7, 1.9)
        @test CAL.calibration_weight(inv, d) == CAL.calibration_weight(inv, -d)
        @test CAL.calibration_weight(std, d) == CAL.calibration_weight(std, -d)
    end

    # In [0, 1] everywhere. A weight outside it is not a pool.
    for law in (inv, std, sta), d in -5.0:0.25:5.0
        @test 0.0 <= CAL.calibration_weight(law, d) <= 1.0
    end

    # A discrepancy that could not be measured is not evidence for moving to the book.
    @test CAL.calibration_weight(inv, NaN) == 1.0
    @test CAL.calibration_weight(std, Inf) == 1.0

    # Identity detection, which `is_identity_calibrator` and the bit-identity gate rest on.
    @test CAL.is_identity_law(StaticGeometricLaw(w = 1.0))
    @test CAL.is_identity_law(InverseGaussianLaw(w_base = 1.0, sigma = 0.3))
    @test CAL.is_identity_law(StandardGaussianLaw(w_base = 1.0, sigma = 0.3, w_max = 1.0))
    @test !CAL.is_identity_law(StandardGaussianLaw(w_base = 0.9, sigma = 0.3, w_max = 1.0))

    # Constructor refusals. `w_max < w_base` is a DIFFERENT law written by accident.
    @test_throws ArgumentError StandardGaussianLaw(w_base = 0.5, sigma = 0.2, w_max = 0.3)
    @test_throws ArgumentError InverseGaussianLaw(w_base = 1.5, sigma = 0.2)
    @test_throws ArgumentError InverseGaussianLaw(w_base = 0.5, sigma = 0.0)
    @test_throws ArgumentError StaticGeometricLaw(w = -0.1)

    # Labels are filename-safe, sort-stable, and carry the parameters.
    @test CAL.law_label(inv) == "inv_w0.25_s0.35"
    @test CAL.law_label(std) == "std_w0.40_s0.15"
    @test CAL.law_label(sta) == "sta_w0.40"
end

# ===================================================================
# T2. EQUIVALENCE with the prototype — the load-bearing gate
# ===================================================================

@testset "T2 — PoolDispersion reproduces l01.calibrate_latents bit for bit" begin
    ref_rates = REF.invert_market_rates(CV2_BOOK; match_ids = CV2_IDS)

    # The inversions must agree first, or the pools would be pooling with different books
    # and an agreement downstream would mean nothing.
    @test length(ref_rates) == length(CV2_RATES)
    for m in CV2_IDS
        a = CV2_RATES[m]; b = ref_rates[m]
        @test a.accepted == b.accepted
        @test a.n_targets == b.n_targets
        if a.accepted
            @test a.lambda_home === b.λ_home
            @test a.lambda_away === b.λ_away
        end
    end

    for (method, w_base, sigma) in ((:inverse_gaussian, 0.25, 0.35),
                                    (:standard_gaussian, 0.40, 0.15),
                                    (:static_geometric, 0.40, 0.25))
        law = method === :inverse_gaussian ? InverseGaussianLaw(w_base = w_base, sigma = sigma) :
              method === :standard_gaussian ? StandardGaussianLaw(w_base = w_base, sigma = sigma) :
              StaticGeometricLaw(w = w_base)

        spec = REF.GenerativeCalibrationSpec(method = method, w_base = w_base, sigma = sigma)
        ref_l, ref_d = REF.calibrate_latents(CV2_L, ref_rates, spec)

        cal = cv2_cal(law = law, dispersion = PoolDispersion(), anchor = :pool_mean)
        got_l, got_d = CAL.calibrate_latents(cal, CV2_L, CV2_RATES)

        # `==` on Float64, not `isapprox`. A one-ULP perturbation of a single lambda is
        # invisible to any tolerance worth writing and unmistakable to `==`.
        @test got_l.λ_home == ref_l.λ_home
        @test got_l.λ_away == ref_l.λ_away
        @test latent_match_ids(got_l) == latent_match_ids(ref_l)
        @test maximum(abs, got_l.λ_home .- ref_l.λ_home) == 0.0

        # The diagnostics the two share must agree too — the weights are the transform.
        @test got_d.w_h == ref_d.w_h
        @test got_d.w_a == ref_d.w_a
        # `isequal`, not `==`: a refused fixture's delta is NaN in both frames and that
        # agreement is part of the contract, but `NaN == NaN` is false.
        @test isequal(got_d.delta_h, ref_d.delta_h)
        @test isequal(got_d.delta_a, ref_d.delta_a)
        @test collect(got_d.inverted) == collect(ref_d.inverted)
        @test got_d.lambda_model_h == ref_d.lambda_model_h
    end
end

@testset "T2b — the general kernel agrees with the pool to floating-point noise" begin
    # `PreservedDispersion` at w == 1 is the same map as `PoolDispersion` at w == 1, so the
    # two kernels must meet. They meet to noise and NOT bit for bit, because
    # `w*m + (1-w)*k + w*u` and `w*(m+u) + (1-w)*k` are one number in the reals and two in
    # Float64 — which is exactly why the pool kernel exists as its own code path.
    cal_pool = cv2_cal(law = StaticGeometricLaw(w = 0.55), dispersion = PoolDispersion(),
                       anchor = :none)
    cal_gen = cv2_cal(law = StaticGeometricLaw(w = 0.55),
                      dispersion = SupremacyDispersion(rho_s = 0.55, rho_t = 0.55),
                      anchor = :none)
    a, _ = CAL.calibrate_latents(cal_pool, CV2_L, CV2_RATES)
    b, _ = CAL.calibrate_latents(cal_gen, CV2_L, CV2_RATES)
    @test maximum(abs, a.λ_home .- b.λ_home) < 1e-12
    @test maximum(abs, a.λ_away .- b.λ_away) < 1e-12
end

@testset "T2c — the dispersion maps reproduce l03's schemes" begin
    ref_rates = REF.invert_market_rates(CV2_BOOK; match_ids = CV2_IDS)
    spec = REF.GenerativeCalibrationSpec(method = :standard_gaussian, w_base = 0.40,
                                         sigma = 0.15)
    law = StandardGaussianLaw(w_base = 0.40, sigma = 0.15)

    pairs = [("B_full", PreservedDispersion(), :none),
             ("B_anch", PreservedDispersion(), :pool_mean),
             ("C_sqrt", ConjugateDispersion(), :none),
             ("D_sup",  SupremacyDispersion(rho_s = 1.0, rho_t = :pool), :none),
             ("D_sup_anch", SupremacyDispersion(rho_s = 1.0, rho_t = :pool), :pool_mean),
             ("D_tot",  SupremacyDispersion(rho_s = :pool, rho_t = 1.0), :none)]

    schemes = Dict(s.id => s for s in REF.l03_schemes())
    for (id, disp, anchor) in pairs
        ref_l, _ = REF.apply_dispersion(CV2_L, ref_rates, spec, schemes[id])
        got_l, _ = CAL.calibrate_latents(cv2_cal(law = law, dispersion = disp,
                                                 anchor = anchor), CV2_L, CV2_RATES)
        @test maximum(abs, got_l.λ_home .- ref_l.λ_home) < 1e-12
        @test maximum(abs, got_l.λ_away .- ref_l.λ_away) < 1e-12
    end
end

# ===================================================================
# T3. The identity calibrator
# ===================================================================

@testset "T3 — the identity calibrator returns the raw draws bit for bit" begin
    for law in (StaticGeometricLaw(w = 1.0),
                InverseGaussianLaw(w_base = 1.0, sigma = 0.35),
                StandardGaussianLaw(w_base = 1.0, sigma = 0.15, w_max = 1.0))
        cal = cv2_cal(law = law)
        @test CAL.is_identity_calibrator(cal)
        out, d = CAL.calibrate_latents(cal, CV2_L, CV2_RATES)
        # `exp(log(x)) != x` in Float64, so this is only true because the kernel COPIES at
        # w == 1 rather than computing the algebra. That is the whole point of the check.
        @test out.λ_home == CV2_L.λ_home
        @test out.λ_away == CV2_L.λ_away
        @test all(d.w_h .== 1.0) && all(d.w_a .== 1.0)
        @test all(d.var_retention_h .≈ 1.0)
    end

    # A non-identity calibrator must actually move something, or the gate above is vacuous.
    out, _ = CAL.calibrate_latents(cv2_cal(), CV2_L, CV2_RATES)
    @test out.λ_home != CV2_L.λ_home
end

# ===================================================================
# T4. The Jensen anchor
# ===================================================================

@testset "T4 — the anchor is zero on the pool and rate-matching off it" begin
    law = StandardGaussianLaw(w_base = 0.40, sigma = 0.15)

    # On the pool map the two draw-sums the anchor compares are the SAME sum, so kappa is
    # identically zero and `anchor = :pool_mean` costs nothing. Asserted rather than left
    # for a reader to derive.
    a, da = CAL.calibrate_latents(cv2_cal(law = law, anchor = :pool_mean), CV2_L, CV2_RATES)
    b, db = CAL.calibrate_latents(cv2_cal(law = law, anchor = :none), CV2_L, CV2_RATES)
    @test a.λ_home == b.λ_home
    @test all(da.kappa_h .== 0.0) && all(da.kappa_a .== 0.0)
    @test da.kappa_h == db.kappa_h

    pool, _ = CAL.calibrate_latents(cv2_cal(law = law, dispersion = PoolDispersion(),
                                            anchor = :none), CV2_L, CV2_RATES)
    anch, dan = CAL.calibrate_latents(cv2_cal(law = law, dispersion = PreservedDispersion(),
                                              anchor = :pool_mean), CV2_L, CV2_RATES)
    full, dfu = CAL.calibrate_latents(cv2_cal(law = law, dispersion = PreservedDispersion(),
                                              anchor = :none), CV2_L, CV2_RATES)

    shifted = findall(dan.inverted)
    @test !isempty(shifted)

    # ANCHORED: the draw-mean rate equals the plain pool's, per fixture, per side.
    for i in shifted
        @test mean(view(anch.λ_home, i, :)) ≈ mean(view(pool.λ_home, i, :)) rtol = 1e-12
        @test mean(view(anch.λ_away, i, :)) ≈ mean(view(pool.λ_away, i, :)) rtol = 1e-12
    end
    @test all(isapprox.(dan.rate_ratio_h[shifted], 1.0; rtol = 1e-12))

    # UNANCHORED: strictly HOTTER, because restoring dispersion at a fixed log-location
    # raises E[exp(.)]. This is the Jensen term the anchor exists to remove, and if it were
    # absent here the anchored test above would be measuring nothing.
    @test all(dfu.rate_ratio_h[shifted] .> 1.0)
    @test all(dfu.rate_ratio_a[shifted] .> 1.0)
    @test maximum(dfu.rate_ratio_h[shifted]) > 1.0 + 1e-6

    # ...and identical DISPERSION to its anchored twin, in every basis. The two differ only
    # in first predictive moment, which is the contrast the scheme family was built for.
    @test dfu.var_retention_h[shifted] ≈ dan.var_retention_h[shifted]
    @test dfu.var_retention_tot[shifted] ≈ dan.var_retention_tot[shifted]

    @test_throws ArgumentError GenerativeRateCalibrator(name = "x", law = law, anchor = :mean)
end

# ===================================================================
# T5. The dispersion algebra
# ===================================================================

@testset "T5 — each map retains the variance it claims" begin
    law = StaticGeometricLaw(w = 0.36)
    shifted(d) = findall(d.inverted)

    _, dpool = CAL.calibrate_latents(cv2_cal(law = law, dispersion = PoolDispersion(),
                                             anchor = :none), CV2_L, CV2_RATES)
    _, dfull = CAL.calibrate_latents(cv2_cal(law = law, dispersion = PreservedDispersion(),
                                             anchor = :none), CV2_L, CV2_RATES)
    _, dsqrt = CAL.calibrate_latents(cv2_cal(law = law, dispersion = ConjugateDispersion(),
                                             anchor = :none), CV2_L, CV2_RATES)

    s = shifted(dpool)
    @test !isempty(s)
    # Pool: Var = w^2 sigma^2. At a constant law w == 0.36 exactly, so this is a number.
    @test all(isapprox.(dpool.var_retention_h[s], 0.36^2; rtol = 1e-10))
    @test all(isapprox.(dpool.var_retention_a[s], 0.36^2; rtol = 1e-10))
    # ...and because w_h == w_a here, the (s, t) bases contract by the same factor.
    @test all(isapprox.(dpool.var_retention_sup[s], 0.36^2; rtol = 1e-10))
    @test all(isapprox.(dpool.var_retention_tot[s], 0.36^2; rtol = 1e-10))

    @test all(isapprox.(dfull.var_retention_h[s], 1.0; rtol = 1e-10))
    @test all(isapprox.(dfull.var_retention_tot[s], 1.0; rtol = 1e-10))
    @test all(isapprox.(dsqrt.var_retention_h[s], 0.36; rtol = 1e-10))

    # The supremacy/totals split: preserve one basis, contract the other, and check in the
    # basis rather than per side — per-side numbers cannot tell the two apart.
    _, dsup = CAL.calibrate_latents(
        cv2_cal(law = law, dispersion = SupremacyDispersion(rho_s = 1.0, rho_t = 0.4),
                anchor = :none), CV2_L, CV2_RATES)
    @test all(isapprox.(dsup.var_retention_sup[s], 1.0; rtol = 1e-10))
    @test all(isapprox.(dsup.var_retention_tot[s], 0.16; rtol = 1e-10))

    # The falsification control is the exact mirror image in the (s, t) basis...
    _, dtot = CAL.calibrate_latents(
        cv2_cal(law = law, dispersion = SupremacyDispersion(rho_s = 0.4, rho_t = 1.0),
                anchor = :none), CV2_L, CV2_RATES)
    @test all(isapprox.(dtot.var_retention_sup[s], 0.16; rtol = 1e-10))
    @test all(isapprox.(dtot.var_retention_tot[s], 1.0; rtol = 1e-10))

    # ...and the two are NOT SEPARABLE from the per-side columns, which is the whole reason
    # the (s, t) columns are reported. Both maps share the diagonal exactly — only the sign
    # of the off-diagonal differs — so per-side retention differs between them only by the
    # sample covariance of the two log-rate residuals, while the supremacy retention
    # differs by 0.84.
    @test CAL.residual_map(SupremacyDispersion(rho_s = 1.0, rho_t = 0.4), 0.5, 0.5)[1] ==
          CAL.residual_map(SupremacyDispersion(rho_s = 0.4, rho_t = 1.0), 0.5, 0.5)[1]
    @test CAL.residual_map(SupremacyDispersion(rho_s = 1.0, rho_t = 0.4), 0.5, 0.5)[2] ==
          -CAL.residual_map(SupremacyDispersion(rho_s = 0.4, rho_t = 1.0), 0.5, 0.5)[2]
    @test abs(median(dsup.var_retention_h[s]) - median(dtot.var_retention_h[s])) < 0.05
    @test median(dtot.var_retention_sup[s]) - median(dsup.var_retention_sup[s]) < -0.80

    @test CAL.residual_map(PoolDispersion(), 0.3, 0.7) == (0.3, 0.0, 0.0, 0.7)
    @test CAL.residual_map(PreservedDispersion(), 0.3, 0.7) == (1.0, 0.0, 0.0, 1.0)
    @test CAL.is_pool_map(PoolDispersion())
    @test !CAL.is_pool_map(PreservedDispersion())
    @test_throws ArgumentError SupremacyDispersion(rho_s = :mean)
end

# ===================================================================
# T6. The fallback protocol
# ===================================================================

@testset "T6 — an un-invertible fixture passes through, counted and named" begin
    cal = cv2_cal()
    out, d = CAL.calibrate_latents(cal, CV2_L, CV2_RATES)
    ids = latent_match_ids(out)

    # Not dropped: the calibrated container holds EVERY fixture the raw one did. Dropping
    # would change which fixtures two calibrators score and make their rows incomparable.
    @test ids == CV2_IDS
    @test size(out.λ_home) == size(CV2_L.λ_home)

    for m in CV2_UNQ
        i = findfirst(==(m), ids)
        @test !d.inverted[i]
        @test !isempty(d.reason[i])                         # refused BY NAME
        @test d.w_h[i] == 1.0 && d.w_a[i] == 1.0
        # ...and passed through bit for bit, not re-derived through exp(log(.)).
        @test out.λ_home[i, :] == CV2_L.λ_home[i, :]
        @test out.λ_away[i, :] == CV2_L.λ_away[i, :]
    end

    # A fixture the book DID quote must actually have moved, or the above proves nothing.
    j = findfirst(d.inverted)
    @test j !== nothing
    @test out.λ_home[j, :] != CV2_L.λ_home[j, :]

    # Coverage, reported two ways because only one of them measures dilution. A fixture
    # the book never quoted is REFUSED (it was attempted and had nothing to fit), so it
    # lands in `n_refused` and `n_quoted` excludes it — which is why `coverage_quoted` is
    # 1.0 while `coverage` is not. Reading only the second number would call a book that
    # inverted perfectly a 87% success.
    cov = CAL.inversion_coverage(CV2_RATES, CV2_IDS)
    @test cov.n_fixtures == length(CV2_IDS)
    @test cov.n_refused == length(CV2_UNQ)
    @test cov.n_absent == 0                    # attempted for every id in `match_ids`
    @test cov.n_quoted == length(CV2_IDS) - length(CV2_UNQ)
    @test cov.n_accepted == length(CV2_IDS) - length(CV2_UNQ)
    @test cov.coverage_quoted ≈ 1.0            # a fixture with a book must invert cleanly
    @test cov.coverage < 1.0

    # ...and a fixture genuinely never attempted is a THIRD state, not folded into either.
    partial = CAL.invert_market_rates(CV2_BOOK)          # no `match_ids`: book fixtures only
    @test CAL.inversion_coverage(partial, CV2_IDS).n_absent == length(CV2_UNQ)

    # `:refuse` is the other half of the contract, and it names the fixtures.
    strict = cv2_cal(fallback = :refuse)
    @test_throws ErrorException CAL.calibrate_latents(strict, CV2_L, CV2_RATES)

    # There is deliberately no way to ask for an invented rate.
    @test_throws ArgumentError GenerativeRateCalibrator(name = "x", law = StaticGeometricLaw(),
                                                        fallback = :league_mean)
end

@testset "T6b — the inversion recovers a rate it was given" begin
    # The fixture book is generated FROM a Double Poisson, so the inversion must recover
    # its parameters. If this fails, every delta in every other testset is measuring the
    # optimiser rather than the market.
    frame = CAL.inversion_frame(CV2_RATES)
    acc = frame[frame.accepted, :]
    @test nrow(acc) == length(CV2_IDS) - length(CV2_UNQ)
    @test all(acc.optim_converged)
    @test all(acc.sse .< 1e-6)                 # far inside the 5e-3 acceptance gate
    @test all(acc.n_targets .== 13)            # 3 (1X2) + 2 (BTTS) + 4 lines x 2 sides

    # THE recovery check: the inverted rates must be the ones the book was generated from.
    # If this drifts, every `delta` in every other testset is measuring the optimiser
    # rather than the market.
    for r in eachrow(acc)
        th, ta = CV2_TRUTH[r.match_id]
        @test r.lambda_mkt_h ≈ th rtol = 1e-3
        @test r.lambda_mkt_a ≈ ta rtol = 1e-3
    end
    @test isempty(CAL.inversion_refusals(acc))

    # The refusals carry their reason and their count, not a bare percentage: a gate that
    # refuses 40% of a book for ONE reason is a configuration problem, and one that refuses
    # 2% across four reasons is the book being thin.
    reasons = CAL.inversion_refusals(frame)
    @test length(reasons) == 1
    @test last(reasons[1]) == length(CV2_UNQ)
    @test occursin("too few quoted selections", first(reasons[1]))
end

# ===================================================================
# T7. Derivative coherence — the claim the module exists to make
# ===================================================================

@testset "T7 — every market family sums to the same grid mass" begin
    cal = cv2_cal()
    out, _ = CAL.calibrate_latents(cal, CV2_L, CV2_RATES)
    mkts = CAL.l2_full_direction_markets()

    rep = CAL.coherence_report(out, mkts)
    @test rep.n_fixtures == length(CV2_IDS)
    @test length(rep.family_names) == length(mkts)
    # THE number. 1X2, four totals lines and BTTS are six partitions of one 12x12 tensor,
    # so their sums are one sum. A selection-level shift cannot produce this at any
    # tolerance; that is the comparison being made.
    @test rep.max_family_spread < 1e-12
    # The deficit from 1.0 is the truncated tail beyond the grid, not an incoherence.
    @test rep.max_deviation_from_one < 1e-6

    raw = CAL.coherence_report(CV2_L, mkts)
    @test raw.max_family_spread < 1e-12

    # And the comparison, run rather than asserted: the selection-level shift this module
    # replaces breaks the totals partition by construction.
    p_over, p_under = 0.51, 0.49
    shift(p, c) = 1 / (1 + exp(-(log(p / (1 - p)) + c)))
    # Two independently fitted GLM offsets. The totals partition survives only when they
    # happen to be equal and opposite, and nothing in that construction makes them so.
    incoherence(co, cu) = abs(shift(p_over, co) + shift(p_under, cu) - 1.0)
    @test incoherence(0.30, -0.30) < 1e-12           # the one case that accidentally works
    @test incoherence(0.30, -0.10) > 0.04
    @test maximum(incoherence(co, cu) for co in (0.1, 0.3, 0.5), cu in (-0.3, -0.1, 0.2)) > 0.10
end

# ===================================================================
# T8. The point-in-time book
# ===================================================================

@testset "T8 — the book refuses before it de-vigs" begin
    cfg = CAL.PointInTimeBookConfig(as_of_minutes = -25.0, max_staleness_minutes = 90.0)

    # Last tick AT OR BEFORE the cutoff. `<=`, and a later tick is unreachable.
    ticks = DataFrame(
        match_id = [1, 1, 1, 1, 1, 1],
        market_name = fill("1X2", 6),
        market_line = fill(0.0, 6),
        selection = [:home, :home, :home, :draw, :away, :home],
        minutes_to_kickoff = [-120.0, -40.0, -25.0, -30.0, -35.0, -5.0],
        traded_price = [2.5, 2.4, 2.30, 3.4, 3.6, 9.99])
    px = CAL.point_in_time_prices(ticks; config = cfg)
    home = only(px[px.selection .== :home, :])
    @test home.odds_close == 2.30                 # the tick AT the cutoff, not the -5 one
    @test home.staleness_minutes == 0.0
    @test home.n_ticks_before == 3
    @test only(px[px.selection .== :draw, :]).staleness_minutes == 5.0

    # COMPLETENESS BEFORE NORMALISATION. A lone `over_05` de-vigs to a fair probability of
    # exactly 1.0 and no error — the defect that cost the O/U 0.5 ladder its place in the
    # stream's Phase 2 book. It must be refused, not normalised.
    one_sided = DataFrame(
        match_id = [2], market_name = ["OverUnder"], market_line = [0.5],
        selection = [:over_05], odds_close = [1.05], tick_minutes = [-30.0],
        n_ticks_before = [4], staleness_minutes = [5.0], as_of_minutes = [-25.0])
    book, refused = CAL.devig_book!(one_sided; config = cfg)
    @test nrow(book) == 0
    @test nrow(refused) == 1
    @test occursin("incomplete market", refused.reason[1])
    @test CAL.book_refusal_summary(refused) == ["incomplete market" => 1]

    # ...and with the gate off it is normalised to the fabricated 1.0, which is exactly why
    # the gate is on by default.
    loose = CAL.PointInTimeBookConfig(as_of_minutes = -25.0, require_complete_markets = false)
    bad, _ = CAL.devig_book!(copy(one_sided); config = loose)
    @test nrow(bad) == 1 && bad.prob_fair_close[1] == 1.0

    # Staleness and overround, both refusing by name.
    stale = DataFrame(
        match_id = [3, 3], market_name = fill("BTTS", 2), market_line = fill(0.0, 2),
        selection = [:btts_yes, :btts_no], odds_close = [2.0, 2.0],
        tick_minutes = [-200.0, -30.0], n_ticks_before = [1, 1],
        staleness_minutes = [175.0, 5.0], as_of_minutes = fill(-25.0, 2))
    _, r2 = CAL.devig_book!(stale; config = cfg)
    @test nrow(r2) == 1 && occursin("stalest side", r2.reason[1])

    arb = DataFrame(
        match_id = [4, 4], market_name = fill("BTTS", 2), market_line = fill(0.0, 2),
        selection = [:btts_yes, :btts_no], odds_close = [1.2, 1.2],
        tick_minutes = [-30.0, -30.0], n_ticks_before = [1, 1],
        staleness_minutes = [5.0, 5.0], as_of_minutes = fill(-25.0, 2))
    _, r3 = CAL.devig_book!(arb; config = cfg)
    @test nrow(r3) == 1 && occursin("overround", r3.reason[1])

    # The instant assertion. The column names say "close" whatever the cutoff, so this is
    # the only thing standing between a T-25 experiment and a table of closing prices.
    @test CAL.assert_book_as_of(CV2_BOOK, -25.0) === CV2_BOOK
    @test_throws ErrorException CAL.assert_book_as_of(CV2_BOOK, 0.0)
    @test_throws ErrorException CAL.assert_book_as_of(select(CV2_BOOK, Not(:as_of_minutes)), -25.0)
    mixed = vcat(CV2_BOOK[1:2, :], transform(CV2_BOOK[3:4, :], :as_of_minutes => ByRow(_ -> 0.0) => :as_of_minutes))
    @test_throws ErrorException CAL.assert_book_as_of(mixed, -25.0)

    @test CAL.expected_selection_count("1X2", 0.0) == 3
    @test CAL.expected_selection_count("OverUnder", 2.5) == 2
    @test CAL.expected_selection_count("CorrectScore", 0.0) == 0
end

@testset "T8b — calibrate_fit refuses a book from the wrong instant" begin
    close_book = transform(CV2_BOOK, :as_of_minutes => ByRow(_ -> 0.0) => :as_of_minutes)
    @test_throws ErrorException calibrate_fit(cv2_cal(), CV2_FIT, close_book)
    # ...and builds when the instants agree.
    cf = calibrate_fit(cv2_cal(), CV2_FIT, CV2_BOOK)
    @test cf isa CalibratedFit
    @test cf.book_as_of_minutes == -25.0
end

# ===================================================================
# T9. The Portfolio contract
# ===================================================================

@testset "T9 — CalibratedFit is a first-class portfolio source" begin
    cal = cv2_cal()
    cf = calibrate_fit(cal, CV2_FIT, CV2_BOOK; rates = CV2_RATES)

    # The calibrated container really is the SAME concrete type, which is what makes every
    # downstream tier work with no new method. See `CalibratedLatents`.
    @test typeof(cf.latents) === typeof(CV2_L)
    @test cf.latents isa CalibratedLatents
    @test calibrated_fit(cf) isa Fit
    @test Training.fit_name(cf.fit) == Training.fit_name(CV2_FIT)
    @test "calibrated" in cf.fit.config.tags
    @test any(startswith("calibrator:"), cf.fit.config.tags)
    # The audit is carried across, not re-derived — one posterior, one verdict.
    @test cf.fit.diagnostics === CV2_FIT.diagnostics
    @test Evaluation.convergence_verdict(cf) == Evaluation.convergence_verdict(CV2_FIT)
    @test Evaluation.fit_latents(cf) === cf.latents

    spec = CPF.BookSpec(markets = Data.MarketConfig(CAL.l2_tradeable_markets()),
                        shrink = CPF.NoShrinkage())
    policy = CPF.PolicySpec()

    r_cf, books_cf, br_cf = run_portfolio_simulation(spec, policy, cf, CV2_BOOK, CV2_FX;
                                                     bootstrap = false, quiet = true)
    r_fit, books_fit, br_fit = run_portfolio_simulation(spec, policy, calibrated_fit(cf),
                                                        CV2_BOOK, CV2_FX;
                                                        bootstrap = false, quiet = true)

    # Passing the CalibratedFit and passing its inner Fit must be the same call.
    @test length(books_cf) == length(books_fit)
    @test br_cf.n_books == br_fit.n_books
    @test r_cf.trajectory.bankroll == r_fit.trajectory.bankroll
    @test r_cf.summary.total_return_pct == r_fit.summary.total_return_pct
    @test r_cf.trajectory.bets.stake == r_fit.trajectory.bets.stake

    # It priced something, or the equality above is two empty things agreeing.
    @test nrow(r_cf.trajectory.bets) > 0
    @test br_cf.n_books > 0

    # And calibration actually changed the book, so this is a calibrated portfolio and not
    # a raw one wearing a tag.
    r_raw, _, _ = run_portfolio_simulation(spec, policy, CV2_FIT, CV2_BOOK, CV2_FX;
                                           bootstrap = false, quiet = true)
    @test r_raw.summary.total_return_pct != r_cf.summary.total_return_pct

    # The identity calibrator must reproduce the RAW portfolio exactly — the in-grid
    # control that says the calibrated code path adds nothing of its own.
    cf_id = calibrate_fit(cv2_cal(law = StaticGeometricLaw(w = 1.0)), CV2_FIT, CV2_BOOK;
                          rates = CV2_RATES)
    r_id, _, _ = run_portfolio_simulation(spec, policy, cf_id, CV2_BOOK, CV2_FX;
                                          bootstrap = false, quiet = true)
    @test r_id.trajectory.bankroll == r_raw.trajectory.bankroll
    @test r_id.trajectory.bets.stake == r_raw.trajectory.bets.stake

    # Evaluation reaches it too, and the scores differ from raw for the same reason.
    sc = CAL.calibration_scores(cf, CV2_BOOK, CV2_MATCH)
    @test sc.n_obs > 0
    @test isfinite(sc.logloss) && isfinite(sc.ece) && isfinite(sc.brier)

    # The summary prints without reaching a database or a network.
    io = IOBuffer()
    CAL.calibration_summary(cf; io = io)
    s = String(take!(io))
    @test occursin("CALIBRATION", s) && occursin(cal.name, s)
end

# ===================================================================
# T10. Persistence — needs `mcmc_experiments`
# ===================================================================

"`PostgresStorage` for the throwaway test experiment, or `nothing` when it is out of reach."
function cv2_storage()
    try
        db = Training.PostgresStorage("calibration_v2_test")
        Training.ensure_schema!(db)
        return db
    catch e
        @info "T10 SKIPPED — mcmc_experiments is out of reach; this tier proves nothing " *
              "and is not counted as passing." exception = (e, catch_backtrace())
        return nothing
    end
end

@testset "T10 — round-trip through mcmc_experiments" begin
    db = cv2_storage()
    if db === nothing
        @test_skip "mcmc_experiments unreachable"
    else
        cal = cv2_cal()
        cf = calibrate_fit(cal, CV2_FIT, CV2_BOOK; rates = CV2_RATES)

        # 1. The registry. A calibrator is a canonical recipe like a BookSpec is.
        rid = Training.save_calibrator(db, "cv2_" * string(uuid4())[1:8], cal;
                                       description = "test", tags = ["test", "calibration_v2"])
        @test rid isa Integer
        back = Training.load_calibrator(db, rid)
        @test back isa GenerativeRateCalibrator
        @test CAL.calibrator_hash(back) == CAL.calibrator_hash(cal)
        @test back.law == cal.law && back.book_as_of_minutes == cal.book_as_of_minutes
        types = Training.list_configs(db; config_type = "calibrator")
        @test rid in types.id

        # 2. A model run to hang the calibration off. Real foreign key, real cascade.
        run_id = save_fit(CV2_FIT, db)

        # 3. The calibration run itself.
        sc = CAL.calibration_scores(cf, CV2_BOOK, CV2_MATCH)
        cal_run = save_calibration_db(cf, run_id, db; scores = sc,
                                      metadata = (; suite = "test_calibration_v2"))
        @test cal_run isa UUID

        loaded = load_calibration_db(cal_run, db)
        @test CAL.calibrator_hash(loaded.calibrator) == CAL.calibrator_hash(cal)
        # LOSSLESS: the container comes back bit for bit, not to a tolerance.
        @test loaded.latents.λ_home == cf.latents.λ_home
        @test loaded.latents.λ_away == cf.latents.λ_away
        @test latent_match_ids(loaded.latents) == latent_match_ids(cf.latents)
        @test loaded.diagnostics.w_h == cf.rate_diagnostics.w_h
        @test loaded.row.calibrator_name == cal.name
        @test loaded.row.book_as_of_minutes == -25.0
        @test loaded.row.n_inverted == cf.coverage.n_accepted
        @test loaded.row.log_loss ≈ sc.logloss

        runs = list_calibration_runs(db; model_run_id = run_id)
        @test string(cal_run) in string.(runs.calibration_run_id)

        # 4. The portfolio link, through metadata and with no `src/Portfolio/` change.
        spec = CPF.BookSpec(markets = Data.MarketConfig(CAL.l2_tradeable_markets()),
                            shrink = CPF.NoShrinkage())
        policy = CPF.PolicySpec()
        result, _, _ = run_portfolio_simulation(spec, policy, cf, CV2_BOOK, CV2_FX;
                                                bootstrap = false, quiet = true)
        pid = link_portfolio_run(result, run_id, cal_run, db;
                                 book_spec = spec, policy_spec = policy, calibrator = cal)
        linked = portfolio_runs_for_calibration(cal_run, db)
        @test string(pid) in string.(linked.portfolio_run_id)
        @test nrow(linked) == 1

        reloaded = load_portfolio_db(pid, db)
        @test reloaded.trajectory.bets.stake == result.trajectory.bets.stake
    end
end

# ===================================================================
# T11. The deprecated path still works
# ===================================================================

@testset "T11 — BasicLogitShift is deprecated, not broken" begin
    CAL._L2_DEPRECATION_WARNED[] = false
    m = @test_logs (:warn,) match_mode = :any BasicLogitShift()
    @test m isa CAL.AbstractLayerTwoModel
    # Once per session, not once per construction — a warning per row is a warning nobody reads.
    @test_logs BasicLogitShift()

    cfg = CalibrationConfig(name = "legacy", model = m, prob_col = :prob_mean)
    data = DataFrame(prob_mean = [0.3, 0.6, 0.45, 0.7, 0.2, 0.55],
                     is_winner = [0, 1, 0, 1, 0, 1],
                     distribution = [fill(p, 5) for p in [0.3, 0.6, 0.45, 0.7, 0.2, 0.55]])
    fitted = CAL.fit_calibrator(m, copy(data), cfg)
    scal, sdist = CAL.apply_calibration(fitted, copy(data))
    @test length(scal) == 6 && all(0 .< scal .< 1)
    @test length(sdist) == 6
end
