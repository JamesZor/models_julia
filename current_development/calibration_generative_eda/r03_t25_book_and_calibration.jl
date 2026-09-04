# ==============================================================================
# r03 — T−25 book, rate re-inversion, and the calibration sweep at tradeable prices
# ==============================================================================
#
# ------------------------------------------------------------------------------
# WHAT THIS IS AND IS NOT
# ------------------------------------------------------------------------------
#
# r01 and r02 answered their questions at the Betfair CLOSE and both said so in
# their first paragraph: a closing price is not available when a bet is struck, so
# every number in them is an upper bound. This runner removes that caveat from the
# calibration half of the stream.
#
#   Q1  How much book is there at T−25 at all, and how old is it?
#   Q2  How far does the book move between T−25 and the close — i.e. how much of
#       a close-priced backtest's edge was information the bettor could not have?
#   Q3  Do the inverted market rates survive the move to T−25?
#   Q4  Does the calibration still work, and are r01's chosen (w_base, σ) still
#       the right ones — or are they, like the Ireland parameters, specific to the
#       book they were fitted on?
#
# It is NOT a portfolio study. No bankroll, no Kelly, no trust vector appears
# below; `r04_t25_portfolio.jl` stakes what this runner nominates. Q4's answer is
# a calibration result and entitles nobody to a return.
#
# ------------------------------------------------------------------------------
# THE COMPARABILITY CONTRACT
# ------------------------------------------------------------------------------
#
# 1. TWO BOOKS, ONE ROW SET. The T−25 book covers fewer selections than the close
#    book, so scoring the two on their own row sets would confound the price with
#    the coverage. Every head-to-head number below is computed on the INTERSECTION
#    of the two books' (match, market, line, selection) keys. The unmatched rows
#    are counted and reported, never quietly dropped.
#
# 2. NOTHING AFTER T−25 REACHES THE T−25 BOOK. `point_in_time_prices` filters to
#    `minutes_to_kickoff <= as_of` BEFORE it picks the last tick, so a later tick
#    is unreachable rather than merely unselected. `assert_book_as_of` is called at
#    every crossing into scoring, because the pipeline reads `:odds_close` by name
#    whatever the cutoff and a mislabelled frame is the failure mode this whole
#    runner exists to remove.
#
# 3. THE PARAMETERS ARE RE-SWEPT, NOT TRANSFERRED. r01 chose `std_w0.25_s0.25`
#    against closing rates. A T−25 book is less sharp, so the model-market
#    discrepancy Δ is drawn from a different distribution and the σ that shaped
#    the weight law over closing Δ has no claim on it. Assuming otherwise is
#    exactly the error the Ireland transfer made — see `notes_rqs_01.md` §4. The
#    full 68-spec grid runs again on T−25 rates and §9 says whether the optimum
#    moved.
#
# 4. THE FIXTURE SET IS r01's. 710 OOS fixtures in 24/25 + 25/26.
#
# 5. TWO SCORING BASELINES, KEPT APART. `pit_*` columns score the model against
#    the T−25 fair price — the price a bettor could actually have taken, and the
#    only honest "did we beat the market" for this stream. `close_*` columns score
#    the same model probabilities against the closing fair price, which is the
#    published benchmark and, read as a difference, a closing-line-value proxy.
#
# ------------------------------------------------------------------------------
# PERSISTENCE CAVEAT
# ------------------------------------------------------------------------------
#
# Replaceable CSVs under `results/`. Writes nothing to `mcmc_experiments`; reads
# `betdb` for odds and results only. `paper_runbook` is never opened and the
# consoles on 8085 / 8086 are untouched.
#
# ------------------------------------------------------------------------------
# USAGE
# ------------------------------------------------------------------------------
#
#   julia --project -t 16
#   julia> include("current_development/calibration_generative_eda/r03_t25_book_and_calibration.jl")
#
#   R03_SMOKE=1 ...   # m12 only, 3 specs, coverage and drift still full
#
# ==============================================================================

# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================

using BayesianFootball
using CSV
using DataFrames
using Dates
using LinearAlgebra
using Printf
using Statistics
using ThreadPinning

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "l01_generative_calibrator.jl"))
include(joinpath(@__DIR__, "l02_point_in_time_book.jl"))


# %%
# ===================================================================
# 2. Configuration
# ===================================================================

const R03_EXPERIMENT = "scottish_lower_joint_player_2426"
const R03_GATE_SEASONS = ["24/25", "25/26"]
const R03_SMOKE = get(ENV, "R03_SMOKE", "0") != "0"

const R03_MODELS = R03_SMOKE ? ["m12_joint_hybrid_synergy"] :
    ["m12_joint_hybrid_synergy", "m05_joint_production_wealth"]

"""
T−25 is the START of MatchDay's execution band (T−25 to T−12, AGENTS.md §7.2) and
therefore the earliest instant a slate is committed — the most conservative honest
cutoff, not an arbitrary one.
"""
const R03_AS_OF = -25.0

"""
The staleness bound, and the sensitivity ladder around it.

MEASURED at T−25 over 1,641 fixtures: a 30-minute bound keeps 6,879 rows across
1,341 fixtures at p90 staleness 20.4 min; 90 minutes keeps 10,373 rows across
1,572 fixtures at p90 51.0 min; unbounded keeps 17,048 rows at p90 266.7 min. 90
is the point where fixture coverage is essentially complete and the p90 is still
inside the hour — but it is a judgement, so §4 reports the ladder and the reader
can move it.
"""
const R03_MAX_STALENESS = 90.0
const R03_STALENESS_LADDER = [30.0, 60.0, 90.0, 180.0]

const R03_W_BASES = [0.25, 0.40, 0.55, 0.70, 0.85, 1.00]
const R03_SIGMAS  = [0.15, 0.25, 0.35, 0.50, 0.75, 1.00]
const R03_METHODS = [:inverse_gaussian, :standard_gaussian, :static_geometric]

const R03_EDGE_SMALL = 0.02
const R03_EDGE_LARGE = 0.05
const R03_N_BINS = 10
const R03_INVERSION = MarketInversionConfig()

"r01's optima, carried only to be TESTED against the re-sweep. Not assumed."
const R03_R01_NOMINATIONS = Dict(
    "m12_joint_hybrid_synergy" => "std_w0.25_s0.25",
    "m05_joint_production_wealth" => "std_w0.25_s0.15",
)


# %%
# ===================================================================
# 3. Runtime and output directory
# ===================================================================

const R03_OUT = joinpath(@__DIR__, "results")
mkpath(R03_OUT)

println("\n" * "="^110)
println(" r03 · T−25 BOOK, RATE RE-INVERSION AND CALIBRATION AT TRADEABLE PRICES")
println("="^110)
@printf("  started    : %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
@printf("  threads    : %d\n", Threads.nthreads())
@printf("  cutoff     : T%.1f minutes, staleness bound %.0f min\n",
        R03_AS_OF, R03_MAX_STALENESS)
R03_SMOKE && println("  MODE       : SMOKE")


# %%
# ===================================================================
# 4. The two books
#    G-A · coverage, staleness and the matched row set
# ===================================================================

ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)
db = PostgresStorage(R03_EXPERIMENT)

r03_gate_match_ids = Set{Int}(
    Int.(ds.matches.match_id[in.(ds.matches.season, Ref(R03_GATE_SEASONS))]))

close_book = l01_betfair_closing_odds(ds)
r03_config = PointInTimeBookConfig(as_of_minutes = R03_AS_OF,
                                   max_staleness_minutes = R03_MAX_STALENESS)
t25_book, t25_refusals = point_in_time_book(ds; config = r03_config)
assert_book_as_of(t25_book, R03_AS_OF)

println("\n--- G-A · book coverage at T$(R03_AS_OF) ---")
let c = book_coverage(t25_book, t25_refusals)
    @printf("  close book : %6d rows, %5d fixtures\n",
            nrow(close_book), length(unique(close_book.match_id)))
    @printf("  T−25 book  : %6d rows, %5d fixtures, %5d markets\n",
            c.n_rows, c.n_fixtures, c.n_markets)
    @printf("  staleness  : median %.1f min, p90 %.1f min, max %.1f min\n",
            c.median_staleness, c.p90_staleness, c.max_staleness)
    @printf("  overround  : median %.4f\n", c.median_overround)
    for (reason, n) in refusal_summary(t25_refusals)
        @printf("  REFUSED %-48s %5d markets\n", reason, n)
    end
end

println("\n  staleness-bound sensitivity (coverage against the bound):")
r03_ladder_rows = NamedTuple[]
for stale in R03_STALENESS_LADDER
    cfg = PointInTimeBookConfig(as_of_minutes = R03_AS_OF, max_staleness_minutes = stale)
    bk, rf = point_in_time_book(ds; config = cfg)
    c = book_coverage(bk, rf)
    push!(r03_ladder_rows, merge((; max_staleness = stale), c))
    @printf("    bound %6.0f min : %6d rows, %5d fixtures, p90 staleness %6.1f min\n",
            stale, c.n_rows, c.n_fixtures, c.p90_staleness)
end
CSV.write(joinpath(R03_OUT, "r03_staleness_ladder.csv"), DataFrame(r03_ladder_rows))

"The (match, market, line, selection) keys BOTH books carry — the comparison set."
const R03_KEY = [:match_id, :market_name, :market_line, :selection]
r03_matched_keys = innerjoin(unique(select(t25_book, R03_KEY)),
                             unique(select(close_book, R03_KEY)); on = R03_KEY)
t25_matched = innerjoin(t25_book, r03_matched_keys; on = R03_KEY)
close_matched = innerjoin(close_book, r03_matched_keys; on = R03_KEY)
assert_book_as_of(t25_matched, R03_AS_OF)
sort!(t25_matched, R03_KEY)
sort!(close_matched, R03_KEY)

@printf("\n  matched row set: %d rows over %d fixtures\n",
        nrow(r03_matched_keys), length(unique(r03_matched_keys.match_id)))
@printf("    T−25 rows outside it : %d\n", nrow(t25_book) - nrow(t25_matched))
@printf("    close rows outside it: %d\n", nrow(close_book) - nrow(close_matched))
println("  every head-to-head number below is on the matched set; the two counts")
println("  above are what each book carries that the other cannot price.")


# %%
# ===================================================================
# 5. What the book does between T−25 and the close
# ===================================================================

r03_drift = book_drift(t25_matched, close_matched)
CSV.write(joinpath(R03_OUT, "r03_book_drift.csv"), r03_drift)

println("\n--- book drift, T−25 → close ---")
if nrow(r03_drift) > 0
    @printf("  |log price drift| : median %.4f, p90 %.4f  (%d selections)\n",
            median(abs.(r03_drift.log_price_drift)),
            quantile(abs.(r03_drift.log_price_drift), 0.90), nrow(r03_drift))
    @printf("  fair prob drift   : mean %+.5f, median %+.5f, sd %.5f\n",
            mean(r03_drift.fair_drift), median(r03_drift.fair_drift),
            std(r03_drift.fair_drift))
    fam = combine(groupby(r03_drift, :market_name),
                  nrow => :n,
                  :log_price_drift => (x -> median(abs.(x))) => :median_abs_log_drift,
                  :fair_drift => (x -> std(x)) => :sd_fair_drift)
    for r in eachrow(sort(fam, :n; rev = true))
        @printf("    %-14s n=%6d  median |log drift| %.4f  sd fair drift %.4f\n",
                r.market_name, r.n, r.median_abs_log_drift, r.sd_fair_drift)
    end
end
println("  A large drift means the close carried information T−25 did not, and")
println("  that every close-priced number in r01/r02 was scored against it.")


# %%
# ===================================================================
# 6. Market rate inversion at both instants
#    G-B · does the book still invert at T−25?
# ===================================================================

r03_raw = Dict{String, Any}()
for name in R03_MODELS
    fit = load_fit(db, name)
    fit.diagnostics.passed ||
        @warn "$name did not pass strict convergence gating" gates=fit.diagnostics.failed_gates
    r03_raw[name] = restrict_latents(fit_latents(fit), r03_gate_match_ids)
end
r03_all_ids = sort!(collect(union((Set(latent_match_ids(r03_raw[n])) for n in R03_MODELS)...)))

println("\n--- G-B · market rate inversion ---")
t_pit = @elapsed r03_rates_t25 =
    invert_market_rates(t25_book; config = R03_INVERSION, match_ids = r03_all_ids)
t_cl = @elapsed r03_rates_close =
    invert_market_rates(close_book; config = R03_INVERSION, match_ids = r03_all_ids)

for (label, rates, secs) in (("T−25", r03_rates_t25, t_pit),
                             ("close", r03_rates_close, t_cl))
    f = inversion_frame(rates)
    acc = f[f.accepted, :]
    cov = inversion_coverage(rates, r03_all_ids)
    @printf("  %-6s inverted %d in %.1fs | accepted %d of %d quoted (%.1f%%) | SSE med %.3e p90 %.3e\n",
            label, nrow(f), secs, cov.n_accepted, cov.n_quoted,
            100 * cov.coverage_quoted,
            nrow(acc) == 0 ? NaN : median(acc.sse),
            nrow(acc) == 0 ? NaN : quantile(acc.sse, 0.90))
    for (reason, n) in refusal_counts(f)
        @printf("    REFUSED %-56s %4d\n", reason, n)
    end
end
CSV.write(joinpath(R03_OUT, "r03_inversion_t25.csv"), inversion_frame(r03_rates_t25))

# How far the inverted rates themselves move between the two instants. A bare
# string here would be parsed as a docstring on the `let`, which is not documentable.
let both = [m for m in r03_all_ids
            if haskey(r03_rates_t25, m) && r03_rates_t25[m].accepted &&
               haskey(r03_rates_close, m) && r03_rates_close[m].accepted]
    if !isempty(both)
        dh = [log(r03_rates_close[m].λ_home) - log(r03_rates_t25[m].λ_home) for m in both]
        da = [log(r03_rates_close[m].λ_away) - log(r03_rates_t25[m].λ_away) for m in both]
        @printf("\n  λ_mkt drift over %d fixtures inverted at BOTH instants:\n", length(both))
        @printf("    home log-rate: mean %+.4f, sd %.4f, p90 |Δ| %.4f\n",
                mean(dh), std(dh), quantile(abs.(dh), 0.90))
        @printf("    away log-rate: mean %+.4f, sd %.4f, p90 |Δ| %.4f\n",
                mean(da), std(da), quantile(abs.(da), 0.90))
        println("    This is the size of the information the close has and T−25 does not,")
        println("    measured in the same units the calibration weight law reads.")
    end
end


# %%
# ===================================================================
# 7. Baselines on the matched row set
#    G-C · the uncalibrated model against each price
# ===================================================================
#
# The SAME model probabilities, scored twice. `close_*` reproduces r01's scope on
# the matched subset — it will not equal r01's headline exactly, because the row
# set is the intersection rather than the whole close book, and that difference is
# printed rather than assumed away.

const R03_MARKETS = l01_tradeable_markets()

println("\n--- G-C · uncalibrated baselines on the matched row set ---")
r03_rows = NamedTuple[]
r03_anchors = Dict{String, Dict{Tuple{Int,Symbol}, Float64}}()
r03_identity = GenerativeCalibrationSpec(method = :static_geometric, w_base = 1.0)

"Score one container against one book, tagging which price the baseline is."
function r03_score(model, spec, latents, book, price_tag; anchor = nothing, weights = nothing)
    row, fams = score_calibration(model, spec, latents, book, ds.matches;
                                  markets = R03_MARKETS, n_bins = R03_N_BINS,
                                  anchor = anchor, edge_small = R03_EDGE_SMALL,
                                  edge_large = R03_EDGE_LARGE, weights = weights)
    return merge(row, (; price = String(price_tag))), fams
end

r03_family_frames = DataFrame[]
for name in R03_MODELS
    raw = r03_raw[name]
    # The edge anchor is the RAW model against the T−25 price: the edge a bettor
    # could actually have seen, which is what the strata must be cut on here.
    actx = build_evaluation_context(raw, t25_matched, ds.matches,
                                    L01_EVAL.AbstractScoringRule[L01_EVAL.LogLoss()];
                                    markets = R03_MARKETS, threaded = true)
    r03_anchors[name] = edge_anchor(actx)

    for (tag, book) in (("t25", t25_matched), ("close", close_matched))
        row, fams = r03_score(name, r03_identity, raw, book, tag;
                              anchor = r03_anchors[name])
        push!(r03_rows, merge(row, (; rates = "none", spec = "uncalibrated")))
        fams.spec .= "uncalibrated_" * tag
        push!(r03_family_frames, fams)
        @printf("  %-30s vs %-5s book: LogLoss %.5f (market %.5f)  ECE %.4f (market %.4f)  N=%d\n",
                name, tag, row.head_logloss, row.head_market_logloss,
                row.head_ece, row.head_market_ece, row.head_n_obs)
    end
end
println("  A market LogLoss that is WORSE at T−25 than at the close is the market")
println("  getting sharper into kick-off, which is the expected direction and the")
println("  reason a close-priced backtest flatters every model that beats it.")


# %%
# ===================================================================
# 8. The calibration sweep on T−25 rates
# ===================================================================

r03_specs = R03_SMOKE ?
    [GenerativeCalibrationSpec(method = :standard_gaussian, w_base = 0.25, sigma = 0.25),
     GenerativeCalibrationSpec(method = :inverse_gaussian, w_base = 0.25, sigma = 0.75),
     GenerativeCalibrationSpec(method = :static_geometric, w_base = 0.55)] :
    sweep_specs(w_bases = R03_W_BASES, sigmas = R03_SIGMAS, methods = R03_METHODS)

println("\n--- sweep · $(length(r03_specs)) specs × $(length(R03_MODELS)) models, T−25 rates ---")
println("  Scored against the T−25 price. Calibrated with T−25 rates. Nothing here")
println("  reads a closing price at any point.")

t_sweep = @elapsed for name in R03_MODELS
    raw = r03_raw[name]
    anchor = r03_anchors[name]
    for (j, spec) in enumerate(r03_specs)
        cal, diag = calibrate_latents(raw, r03_rates_t25, spec)
        ws = weight_summary(diag)
        row, fams = r03_score(name, spec, cal, t25_matched, "t25";
                              anchor = anchor, weights = ws)
        push!(r03_rows, merge(row, (; rates = "t25")))
        push!(r03_family_frames, fams)
        if j % 15 == 0 || j == length(r03_specs)
            @printf("    %-30s %3d/%3d  %-18s LL %.5f  ECE %.4f  w̃ %.3f\n",
                    name, j, length(r03_specs), spec_label(spec),
                    row.head_logloss, row.head_ece, ws.w_median)
        end
    end
end
@printf("  sweep complete in %s\n", Training.format_elapsed(t_sweep))

r03_summary = DataFrame(r03_rows)
CSV.write(joinpath(R03_OUT, "r03_sweep_scores_t25.csv"), r03_summary)
CSV.write(joinpath(R03_OUT, "r03_family_scores_t25.csv"), vcat(r03_family_frames...))


# %%
# ===================================================================
# 9. Did the optimum move?
# ===================================================================

"Best grid point per (model, method) that improves LogLoss AND ECE on its baseline."
function r03_joint_best(summary, price_tag)
    out = NamedTuple[]
    for name in unique(summary.model)
        b = filter(r -> r.model == name && r.spec == "uncalibrated" &&
                        r.price == price_tag, summary)
        nrow(b) == 0 && continue
        base = first(b)
        for meth in unique(filter(r -> r.spec != "uncalibrated", summary).method)
            cand = filter(r -> r.model == name && r.method == meth &&
                               r.price == price_tag && r.spec != "uncalibrated" &&
                               r.head_logloss <= base.head_logloss &&
                               r.head_ece <= base.head_ece, summary)
            nrow(cand) == 0 && continue
            push!(out, copy(cand[argmin(cand.head_logloss), :]))
        end
    end
    return isempty(out) ? DataFrame() : DataFrame(out)
end

r03_best = r03_joint_best(r03_summary, "t25")
nrow(r03_best) > 0 && CSV.write(joinpath(R03_OUT, "r03_best_per_form_t25.csv"), r03_best)

println("\n" * "="^150)
println(" OPTIMUM PER FORM ON T−25 RATES (joint LogLoss-and-ECE improvers, headline scope)")
println("="^150)
@printf(" %-28s | %-18s | %-16s | %8s | %8s | %8s | %8s | %7s | %8s\n",
        "Model", "Method", "Spec", "LogLoss", "ΔLL base", "ECE", "mkt ECE", "w̃", "var ret")
println("-"^150)
for name in R03_MODELS
    b = filter(r -> r.model == name && r.spec == "uncalibrated" && r.price == "t25", r03_summary)
    nrow(b) == 0 && continue
    base = first(b)
    @printf(" %-28s | %-18s | %-16s | %8.5f | %8s | %8.4f | %8.4f | %7.3f | %8s\n",
            name, "—", "uncalibrated", base.head_logloss, "—", base.head_ece,
            base.head_market_ece, 1.0, "—")
    for row in eachrow(filter(r -> r.model == name, r03_best))
        @printf(" %-28s | %-18s | %-16s | %8.5f | %+8.5f | %8.4f | %8.4f | %7.3f | %8.3f\n",
                "", row.method, row.spec, row.head_logloss,
                row.head_logloss - base.head_logloss, row.head_ece,
                row.head_market_ece, row.w_median, row.var_retention_median)
    end
end
println("="^150)

println("\n DID r01's CLOSE-FITTED OPTIMUM TRANSFER?")
println("-"^150)
for name in R03_MODELS
    want = get(R03_R01_NOMINATIONS, name, nothing)
    want === nothing && continue
    hit = filter(r -> r.model == name && r.spec == want && r.price == "t25", r03_summary)
    b = filter(r -> r.model == name && r.spec == "uncalibrated" && r.price == "t25", r03_summary)
    (nrow(hit) == 0 || nrow(b) == 0) && continue
    h = first(hit); base = first(b)
    best = filter(r -> r.model == name, r03_best)
    bl = nrow(best) == 0 ? NaN : minimum(best.head_logloss)
    @printf(" %-28s r01 pick %-16s on T−25 rates: LogLoss %.5f (Δ %+.5f), ECE %.4f (base %.4f)\n",
            name, want, h.head_logloss, h.head_logloss - base.head_logloss,
            h.head_ece, base.head_ece)
    @printf(" %-28s   best T−25 spec reaches %.5f, so the transferred pick gives up %+.5f\n",
            "", bl, h.head_logloss - bl)
end

println("\n GATE 1 AT TRADEABLE PRICES")
println(" Against the T−25 baseline for the same model on the same rows.")
println("-"^150)
for name in R03_MODELS
    b = filter(r -> r.model == name && r.spec == "uncalibrated" && r.price == "t25", r03_summary)
    nrow(b) == 0 && continue
    base = first(b)
    for row in eachrow(filter(r -> r.model == name, r03_best))
        ll = row.head_logloss <= base.head_logloss + 1e-9
        ece = row.head_ece <= base.head_ece + 1e-9
        mkt = row.head_ece <= row.head_market_ece
        @printf(" [%-6s] %-28s %-18s %-16s  LogLoss %s  ECE vs model %s  ECE vs T−25 book %s\n",
                (ll && ece && mkt) ? "PASS" : "REFUSE", name, row.method, row.spec,
                ll ? "ok" : "worse", ece ? "ok" : "worse", mkt ? "ok" : "worse")
    end
end


# %%
# ===================================================================
# 10. Final report
# ===================================================================

println("\n" * "="^110)
println(" ARTEFACTS")
println("="^110)
for f in ["r03_staleness_ladder.csv", "r03_book_drift.csv", "r03_inversion_t25.csv",
          "r03_sweep_scores_t25.csv", "r03_family_scores_t25.csv",
          "r03_best_per_form_t25.csv"]
    isfile(joinpath(R03_OUT, f)) && @printf("  %s\n", joinpath(R03_OUT, f))
end
println("""
  NEXT. r04 stakes the specs §9 nominates, at T−25 prices, against the same
  raw-model control. Nothing in this runner is a return; a calibration that
  improves the score at a tradeable price still has to survive the allocator, and
  r02 §6.1 is the record of that not being the same question.
""")
@printf("  finished   : %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
