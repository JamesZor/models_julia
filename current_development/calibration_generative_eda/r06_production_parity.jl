# ==============================================================================
# r06 — Does the graduated `src/Calibration/` still say what this stream published?
# ==============================================================================
#
# ------------------------------------------------------------------------------
# THE QUESTION
# ------------------------------------------------------------------------------
#
# `l01_generative_calibrator.jl` and `l02_point_in_time_book.jl` graduated into
# `src/Calibration/` on `feat/modernize-calibration-layer2`
# (`docs/architecture/rfc_layer2_calibration_v2.md`). Everything §5–§8 of the README
# claims was measured with the PROTOTYPE. A production module that is merely close
# to it is a different model, and every published figure would quietly become a
# figure about code nobody runs any more.
#
# `test/test_calibration_v2.jl` T2 pins the transform itself: `PoolDispersion`
# reproduces `l01.calibrate_latents` BIT FOR BIT on a synthetic posterior. That is
# the necessary half. This runner is the sufficient half — the whole pipeline, on
# the real canonical fit, the real T−25 Betfair book and the production
# `BookSpec` / `PolicySpec`, reproducing README §7.4 row for row.
#
# It is a REGRESSION GATE, not an experiment. There is no new number here and there
# is not supposed to be: every figure it prints already exists in §7.4, and the only
# interesting outcome is a disagreement.
#
# ------------------------------------------------------------------------------
# WHAT IS HELD FIXED, AND WHY EACH ONE MATTERS
# ------------------------------------------------------------------------------
#
#   * THE FIXTURE SET. §7.4 was measured over the 40-fold 24/25 + 25/26 study. The
#     canonical runs were extended to 43 folds with the 26/27 August programme on
#     2026-09-04, so `restrict_latents` cuts the container back before anything is
#     scored. Without it this runner would be comparing two different questions and
#     the disagreement would mean nothing.
#   * THE PRICE INSTANT. T−25, asserted by `assert_book_as_of` inside
#     `calibrate_fit`. Calibration parameters do not transfer between instants
#     (§7.3), so a close book here would produce plausible wrong numbers.
#   * THE MARKET SET. `l2_tradeable_markets()` — 11 directions, no O/U 0.5. See
#     §5.6 for what that ladder's one-sided quotes did to a de-vigged fair price.
#   * THE RISK BUDGET. `SlateDrawdown(23.0)`, `FixedCap(0.25)`, `DailySlate()`,
#     2% commission, `FractionalKelly(0.30)` — §7.4's settings exactly.
#
# ------------------------------------------------------------------------------
# DATABASE BOUNDARY
# ------------------------------------------------------------------------------
#
# READS `mcmc_experiments` (the canonical fit) and `betdb` (odds and results).
# WRITES NEITHER. No run, portfolio, calibration or config registration.
# `betdb.paper_runbook` is never opened.
#
# Run on `mcmc-beast`:
#
#     julia --project -t 16
#     julia> include("current_development/calibration_generative_eda/r06_production_parity.jl")
# ==============================================================================

# %%
# ===================================================================
# 1. Packages and module aliases
# ===================================================================

using BayesianFootball
using DataFrames
using Dates
using Printf
using Statistics

const R06_PF = BayesianFootball.Portfolio
const R06_CAL = BayesianFootball.Calibration


# %%
# ===================================================================
# 2. Configuration
# ===================================================================

const R06_EXPERIMENT = "scottish_lower_joint_player_2426"
const R06_MODEL = "m12_joint_hybrid_synergy"
const R06_GATE_SEASONS = ["24/25", "25/26"]
const R06_AS_OF = -25.0
const R06_MAX_STALENESS = 90.0

"""
    R06_PUBLISHED

README §7.4, `m12`, T−25, 11 tradeable directions. The table this runner exists to
reproduce, transcribed here so a disagreement is visible in the output rather than
in a second window.
"""
const R06_PUBLISHED = Dict{Tuple{String, String}, NamedTuple}(
    ("raw", "flat")      => (bets = 1592, ret = 111.70, flat =  9.79, sharpe = 1.220, mdd = -23.45),
    ("raw", "canonical") => (bets = 1127, ret = 151.52, flat = 15.04, sharpe = 1.592, mdd = -16.15),
    ("inv", "canonical") => (bets =  975, ret =  65.64, flat = 17.44, sharpe = 1.772, mdd =  -7.76),
    ("std", "flat")      => (bets = 1503, ret =  63.35, flat =  9.63, sharpe = 1.396, mdd = -12.10),
    ("std", "canonical") => (bets = 1103, ret =  72.85, flat = 13.62, sharpe = 1.606, mdd = -11.36),
    ("sta", "canonical") => (bets =  980, ret =  49.94, flat = 14.90, sharpe = 1.697, mdd =  -6.61),
)

"The three T−25 optima of §7.3, plus the in-grid identity control."
r06_calibrators() = [
    ("raw", GenerativeRateCalibrator(name = "identity_control",
                                     law = StaticGeometricLaw(w = 1.0),
                                     book_as_of_minutes = R06_AS_OF)),
    ("inv", GenerativeRateCalibrator(name = "scot_lower_t25_inv",
                                     law = InverseGaussianLaw(w_base = 0.25, sigma = 0.35),
                                     book_as_of_minutes = R06_AS_OF)),
    ("std", GenerativeRateCalibrator(name = "scot_lower_t25_std",
                                     law = StandardGaussianLaw(w_base = 0.40, sigma = 0.15),
                                     book_as_of_minutes = R06_AS_OF)),
    ("sta", GenerativeRateCalibrator(name = "scot_lower_t25_sta",
                                     law = StaticGeometricLaw(w = 0.40),
                                     book_as_of_minutes = R06_AS_OF)),
]

"§7.4's book spec: `DeArb`, Kelly log-utility, 30% fractional shrinkage, 2% commission."
r06_book_spec() = R06_PF.BookSpec(
    markets = Data.MarketConfig(R06_CAL.l2_tradeable_markets()),
    price = DeArb(),
    allocator = KellyLogUtility(),
    shrink = R06_PF.FractionalKelly(0.30),
    exec = ExecutionConfig(commission = PerBetCommission(0.02), budget = 0.99,
                           min_selection_stake = 0.001))

"§7.4's risk settings, with only the trust model varying."
r06_policy(trust) = R06_PF.PolicySpec(trust = trust, risk = SlateDrawdown(23.0),
                                      cap = FixedCap(0.25), grouping = DailySlate())

println("=" ^ 104)
println(" r06 — PRODUCTION PARITY: src/Calibration against README §7.4")
println("=" ^ 104)
@printf(" experiment  : %s / %s\n", R06_EXPERIMENT, R06_MODEL)
@printf(" instant     : T%+.0f, staleness bound %.0f min\n", R06_AS_OF, R06_MAX_STALENESS)
@printf(" markets     : %d market groups, %d tradeable directions (no O/U 0.5)\n",
        length(R06_CAL.l2_tradeable_markets()),
        sum(length(Predictions.market_keys(m)) for m in R06_CAL.l2_tradeable_markets()))
println("=" ^ 104)


# %%
# ===================================================================
# 3. Data — the store, the canonical fit, and the T−25 book
# ===================================================================

ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 100_000)
db = PostgresStorage(R06_EXPERIMENT)

r06_gate_ids = Set(Int.(ds.matches.match_id[in.(ds.matches.season, Ref(R06_GATE_SEASONS))]))
@printf("\ndatastore   : %d matches, %d in gate seasons\n", nrow(ds.matches), length(r06_gate_ids))

r06_book, r06_refusals = R06_CAL.point_in_time_book(ds;
    config = R06_CAL.PointInTimeBookConfig(as_of_minutes = R06_AS_OF,
                                           max_staleness_minutes = R06_MAX_STALENESS))
r06_coverage = R06_CAL.book_coverage(r06_book, r06_refusals)
@printf("book        : %d rows | %d fixtures | %d markets | staleness med %.0f p90 %.0f | overround %.4f\n",
        r06_coverage.n_rows, r06_coverage.n_fixtures, r06_coverage.n_markets,
        r06_coverage.median_staleness, r06_coverage.p90_staleness,
        r06_coverage.median_overround)
for (reason, n) in R06_CAL.book_refusal_summary(r06_refusals)
    @printf("              refused %5d  %s\n", n, reason)
end

r06_raw_fit = load_fit(db, R06_MODEL)
r06_fit = let l = R06_CAL.restrict_latents(r06_raw_fit.latents, r06_gate_ids)
    Fit(r06_raw_fit.config, r06_raw_fit.folds, l, r06_raw_fit.diagnostics,
        r06_raw_fit.metadata, r06_raw_fit.save_path)
end
@printf("fit         : %d -> %d fixtures after gate-season restriction | converged=%s\n",
        Models.n_matches(r06_raw_fit.latents), Models.n_matches(r06_fit.latents),
        r06_fit.diagnostics.passed)


# %%
# ===================================================================
# 4. GATE R06-A — the inversion, computed once for every calibrator
# ===================================================================
#
# The inversion depends on the BOOK only, never on the model or the law, so a sweep
# that recomputed it per calibrator would be paying for the same answer four times
# AND risking four slightly different ones.

r06_rates = R06_CAL.invert_market_rates(first(r06_calibrators())[2], r06_book;
                                        match_ids = Models.latent_match_ids(r06_fit.latents))
r06_icov = R06_CAL.inversion_coverage(r06_rates, Models.latent_match_ids(r06_fit.latents))
@printf("inversion   : %d/%d accepted (%.1f%% of all, %.1f%% of quoted)\n",
        r06_icov.n_accepted, r06_icov.n_fixtures,
        100 * r06_icov.coverage, 100 * r06_icov.coverage_quoted)
for (reason, n) in R06_CAL.inversion_refusals(r06_rates)
    @printf("              refused %5d  %s\n", n, reason)
end


# %%
# ===================================================================
# 5. Calibration and the portfolio, one row per (container, trust)
# ===================================================================

r06_rows = NamedTuple[]
for (tag, cal) in r06_calibrators()
    cf = calibrate_fit(cal, r06_fit, r06_book; rates = r06_rates, quiet = true)
    ws = R06_CAL.weight_summary(cf.rate_diagnostics)

    # GATE R06-B — coherence, on the real container rather than a synthetic one.
    # Six market families are six partitions of one 12x12 tensor, so their per-fixture
    # sums are one sum. This is the claim the whole module exists to make.
    coh = R06_CAL.coherence_report(cf, R06_CAL.l2_full_direction_markets())

    for (tname, trust) in (("flat", FlatTrust(1.0)),
                           ("canonical", CanonicalScottishLowerTrust()))
        result, _, br = run_portfolio_simulation(r06_book_spec(), r06_policy(trust), cf,
                                                 r06_book, ds;
                                                 bootstrap = false, quiet = true)
        s = result.summary
        push!(r06_rows, (; container = tag, trust = tname, n_books = br.n_books,
                         bets = s.n_bets, ret = s.total_return_pct, flat = s.roi,
                         sharpe = s.sharpe_ann, mdd = s.mdd,
                         w_med = ws.w_median, var_ret = ws.var_retention_median,
                         coherence = coh.max_family_spread))
    end
end
r06_frame = DataFrame(r06_rows)


# %%
# ===================================================================
# 6. GATE R06-C — every published row must reproduce
# ===================================================================
#
# Tolerances are the printed precision of §7.4 and nothing looser: 0.01 on a
# percentage, 0.001 on a Sharpe, and EXACT on the bet count. A bet count that moved
# by one means the book, the fixture set or the allocator moved, and no return
# tolerance would catch it.

const R06_TOL = (bets = 0, ret = 0.01, flat = 0.01, sharpe = 0.001, mdd = 0.01)

r06_mismatches = NamedTuple[]
for r in r06_rows
    want = get(R06_PUBLISHED, (r.container, r.trust), nothing)
    want === nothing && continue
    for f in (:bets, :ret, :flat, :sharpe, :mdd)
        got = getfield(r, f)
        exp_ = getfield(want, f)
        abs(got - exp_) <= getfield(R06_TOL, f) && continue
        push!(r06_mismatches, (; container = r.container, trust = r.trust, field = f,
                               got = got, published = exp_, delta = got - exp_))
    end
end

println()
println("-" ^ 104)
@printf("%-10s %-10s %7s %6s %10s %10s %8s %8s %8s %9s %11s\n",
        "container", "trust", "books", "bets", "return %", "flat ROI %", "Sharpe",
        "MDD %", "med w", "var ret", "coherence")
println("-" ^ 104)
for r in r06_rows
    mark = haskey(R06_PUBLISHED, (r.container, r.trust)) ?
           (any(m -> m.container == r.container && m.trust == r.trust, r06_mismatches) ?
            " MISMATCH" : "") : "   (new)"
    @printf("%-10s %-10s %7d %6d %10.2f %10.2f %8.3f %8.2f %8.3f %9.3f %11.2e%s\n",
            r.container, r.trust, r.n_books, r.bets, r.ret, r.flat, r.sharpe, r.mdd,
            r.w_med, r.var_ret, r.coherence, mark)
end
println("-" ^ 104)

r06_max_coherence = maximum(r.coherence for r in r06_rows)
@printf("\nGATE R06-B  coherence : worst family spread %.2e across %d containers — %s\n",
        r06_max_coherence, length(r06_calibrators()),
        r06_max_coherence < 1e-12 ? "PASS" : "FAIL")

if isempty(r06_mismatches)
    @printf("GATE R06-C  parity    : all %d published rows reproduce within tolerance — PASS\n",
            length(R06_PUBLISHED))
else
    @printf("GATE R06-C  parity    : %d disagreement(s) with README §7.4 — FAIL\n",
            length(r06_mismatches))
    println()
    show(DataFrame(r06_mismatches); allrows = true, allcols = true)
    println()
    println("""
    A disagreement here is NOT a licence to update the README. It means the graduated
    module and the prototype are two different transforms, and the first thing to check
    is `test/test_calibration_v2.jl` T2 — which compares them directly, on a synthetic
    posterior, with `==` on Float64.""")
end

println()
println("=" ^ 104)
println(" R06_DONE  ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
println("=" ^ 104)
