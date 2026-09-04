# ==============================================================================
# r04 — The portfolio at tradeable prices
# ==============================================================================
#
# ------------------------------------------------------------------------------
# WHAT THIS IS AND IS NOT
# ------------------------------------------------------------------------------
#
# r02 measured the portfolio at the Betfair CLOSE and said, in its own header,
# that every return in it was an upper bound. This runner pays that debt. It
# stakes at T−25 — the start of MatchDay's execution band, the earliest instant a
# slate is committed — and answers:
#
#   Q1  What did closing-price staking flatter? Raw `m12` at the close returned
#       +157.50% under the canonical trust. What does the SAME model, the SAME
#       trust and the SAME fixtures return at a price a bettor could have taken?
#   Q2  Does the calibration still pay once both its input rates and its staking
#       prices are the ones actually available?
#   Q3  Does it have closing-line value — does the market move toward its bets?
#
# It is NOT a live-trading claim even now. Two things still stand between this and
# a Saturday: the fill model (a struck price is assumed available in the size the
# allocator asked for; AGENTS.md §7.4 records `LadderSweep` as an optimistic fill
# model and the live system rests at the touch), and the fact that this reads a
# traded-price archive rather than the resting ladder the console sees. Both are
# named in §9 rather than left for the reader to discover.
#
# ------------------------------------------------------------------------------
# THE ATTRIBUTION CONTRACT
# ------------------------------------------------------------------------------
#
# Two things changed between r02 and here, and they must not be allowed to hide in
# one another:
#
#   * the RATES the calibration is computed from (close → T−25), and
#   * the PRICES the bets are struck at (close → T−25).
#
# Panel A crosses them on the MATCHED key set — the selections both books carry —
# so no comparison is contaminated by coverage:
#
#       arm            rates       staking price
#       raw@close       —          close          the r02 reference, upper bound
#       raw@t25         —          T−25           Q1: the price effect, alone
#       cal@close       close      close          the r02 calibrated arm
#       cal_t25@close   T−25       close          the rate-source effect, alone
#       cal_t25@t25     T−25       T−25           Q2: the deployable arm
#
# Panel B re-runs the deployable pair on the FULL T−25 book, because that is what
# a bettor actually has on a Saturday, and the matched restriction is an analysis
# device rather than a constraint they face.
#
# EVERY ARM SHARES the book spec, `SlateDrawdown(23.0)`, `FixedCap(0.25)`,
# `DailySlate()`, `FractionalKelly(0.30)` and 2% commission. Only the container
# and the odds frame move.
#
# ------------------------------------------------------------------------------
# PERSISTENCE CAVEAT
# ------------------------------------------------------------------------------
#
# Replaceable CSVs under `results/`. Writes nothing to `mcmc_experiments`. Reads
# `betdb` for odds and results only; `paper_runbook` is never opened and the
# consoles on 8085 / 8086 are untouched.
#
# ------------------------------------------------------------------------------
# USAGE
# ------------------------------------------------------------------------------
#
#   julia --project -t 16
#   julia> include("current_development/calibration_generative_eda/r04_t25_portfolio.jl")
#
# Requires `results/r03_best_per_form_t25.csv`; run r03 first. The specs are read
# from it rather than restated here, so this runner cannot silently stake a spec
# r03 did not nominate.
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

const R04_PF = BayesianFootball.Portfolio


# %%
# ===================================================================
# 2. Configuration
# ===================================================================

const R04_EXPERIMENT = "scottish_lower_joint_player_2426"
const R04_GATE_SEASONS = ["24/25", "25/26"]
const R04_SMOKE = get(ENV, "R04_SMOKE", "0") != "0"
const R04_MODELS = R04_SMOKE ? ["m12_joint_hybrid_synergy"] :
    ["m12_joint_hybrid_synergy", "m05_joint_production_wealth"]

const R04_AS_OF = -25.0
const R04_MAX_STALENESS = 90.0
const R04_SPLIT_DATE = Date(2025, 5, 3)
const R04_INVERSION = MarketInversionConfig()
const R04_OUT = joinpath(@__DIR__, "results")

"r02's close-fitted nominations, carried as the `cal@close` link back to that run."
const R04_CLOSE_SPECS = Dict(
    "m12_joint_hybrid_synergy" =>
        GenerativeCalibrationSpec(method = :standard_gaussian, w_base = 0.25, sigma = 0.25),
    "m05_joint_production_wealth" =>
        GenerativeCalibrationSpec(method = :standard_gaussian, w_base = 0.25, sigma = 0.15),
)

# Published r02 reference points, close-priced. Context, never a comparator.
const R04_R02_M12_RAW_CANON = 157.50
const R04_R02_M12_RAW_FLAT  = 126.09

const R04_RISK_LAMBDAS =
    [23.0, 18.0, 15.0, 12.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5]


# %%
# ===================================================================
# 3. Runtime, output and the specs r03 nominated
# ===================================================================

mkpath(R04_OUT)

"""
    r04_nominations() -> Dict{String, Vector{NamedTuple}}

The T−25 specs, read from `r03_best_per_form_t25.csv` rather than restated.

r03 nominates on the JOINT criterion — LogLoss and ECE both no worse than the
model's own T−25 baseline — so a spec reaching this runner has already cleared
Gate 1 at tradeable prices. Restating them here would let the two files drift, and
the whole point of the chain is that r04 stakes what r03 chose.
"""
function r04_nominations()
    path = joinpath(R04_OUT, "r03_best_per_form_t25.csv")
    isfile(path) || error(
        "r04 needs $(path). Run r03_t25_book_and_calibration.jl first — this " *
        "runner stakes the specs r03 nominated and does not invent its own.")
    df = CSV.read(path, DataFrame)
    out = Dict{String, Vector{NamedTuple}}()
    for r in eachrow(df)
        method = Symbol(r.method)
        spec = GenerativeCalibrationSpec(
            method = method,
            w_base = Float64(r.w_base),
            sigma = method === :static_geometric ? 0.25 : Float64(r.sigma),
            w_max = Float64(r.w_max))
        label = method === :inverse_gaussian ? "inv" :
                method === :standard_gaussian ? "std" : "sta"
        push!(get!(() -> NamedTuple[], out, String(r.model)), (; label, spec))
    end
    return out
end

r04_noms = r04_nominations()

println("\n" * "="^110)
println(" r04 · PORTFOLIO AT TRADEABLE PRICES (T$(R04_AS_OF))")
println("="^110)
@printf("  started    : %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
@printf("  threads    : %d\n", Threads.nthreads())
println("  staking at T−25; the fill model still assumes the struck price was")
println("  available in size. See §9.")
for (m, ns) in r04_noms
    for n in ns
        @printf("  nominated  : %-30s %-4s %s\n", m, n.label, spec_label(n.spec))
    end
end


# %%
# ===================================================================
# 4. Data, books and containers
#    G-A · one fixture set, two prices
# ===================================================================

ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)
db = PostgresStorage(R04_EXPERIMENT)

r04_gate_ids = Set{Int}(
    Int.(ds.matches.match_id[in.(ds.matches.season, Ref(R04_GATE_SEASONS))]))

close_book = l01_betfair_closing_odds(ds)
t25_full, t25_refusals = point_in_time_book(
    ds; config = PointInTimeBookConfig(as_of_minutes = R04_AS_OF,
                                       max_staleness_minutes = R04_MAX_STALENESS))
assert_book_as_of(t25_full, R04_AS_OF)

const R04_KEY = [:match_id, :market_name, :market_line, :selection]
r04_keys = innerjoin(unique(select(t25_full, R04_KEY)),
                     unique(select(close_book, R04_KEY)); on = R04_KEY)
t25_matched = sort!(innerjoin(t25_full, r04_keys; on = R04_KEY), R04_KEY)
close_matched = sort!(innerjoin(close_book, r04_keys; on = R04_KEY), R04_KEY)
assert_book_as_of(t25_matched, R04_AS_OF)

r04_drift = book_drift(t25_full, close_book)

println("\n--- G-A · books ---")
@printf("  close (full)   : %6d rows, %5d fixtures\n",
        nrow(close_book), length(unique(close_book.match_id)))
@printf("  T−25 (full)    : %6d rows, %5d fixtures\n",
        nrow(t25_full), length(unique(t25_full.match_id)))
@printf("  matched keys   : %6d rows, %5d fixtures — Panel A runs on these\n",
        nrow(r04_keys), length(unique(r04_keys.match_id)))

r04_raw = Dict{String, Any}()
for name in R04_MODELS
    fit = load_fit(db, name)
    fit.diagnostics.passed ||
        @warn "$name did not pass strict convergence gating" gates=fit.diagnostics.failed_gates
    r04_raw[name] = restrict_latents(fit_latents(fit), r04_gate_ids)
end
r04_all_ids = sort!(collect(union((Set(latent_match_ids(r04_raw[n])) for n in R04_MODELS)...)))

r04_rates_t25 = invert_market_rates(t25_full; config = R04_INVERSION,
                                    match_ids = r04_all_ids)
r04_rates_close = invert_market_rates(close_book; config = R04_INVERSION,
                                      match_ids = r04_all_ids)

"Every (container, rate source) this runner stakes."
r04_containers = Dict{String, Vector{NamedTuple}}()
for name in R04_MODELS
    arms = NamedTuple[(label = "raw", rates = "none", latents = r04_raw[name], weights = nothing)]
    for n in get(r04_noms, name, NamedTuple[])
        cal, diag = calibrate_latents(r04_raw[name], r04_rates_t25, n.spec)
        push!(arms, (label = n.label * "_t25", rates = "t25", latents = cal,
                     weights = weight_summary(diag)))
    end
    if haskey(R04_CLOSE_SPECS, name)
        cal, diag = calibrate_latents(r04_raw[name], r04_rates_close, R04_CLOSE_SPECS[name])
        push!(arms, (label = "std_close", rates = "close", latents = cal,
                     weights = weight_summary(diag)))
    end
    r04_containers[name] = arms
    for a in arms
        a.weights === nothing && continue
        @printf("  %-30s %-10s rates=%-5s median w %.3f, variance retained %.1f%%\n",
                name, a.label, a.rates, a.weights.w_median,
                100 * a.weights.var_retention_median)
    end
end


# %%
# ===================================================================
# 5. Panel A — the 2×2 of rate source and staking price, matched keys
# ===================================================================

const R04_BOOK = l01_book_spec(l01_tradeable_markets())
const R04_FLAT = l01_policy_spec(FlatTrust(1.0))
const R04_CANON = l01_policy_spec(CanonicalScottishLowerTrust())

"Build books once per (container, odds frame); every trust model then re-stakes them."
function r04_books(latents, odds)
    books, rep = R04_PF.build_books_reported(R04_BOOK, latents, odds, ds.matches;
                                             require_result = true, quiet = true)
    return books, rep
end

function r04_run(model, container, rates, price, panel, trust_name, books, policy;
                 window = "full")
    result = R04_PF.simulate_portfolio(policy, books; bootstrap = false)
    s = result.summary
    row = (model = String(model), container = String(container), rates = String(rates),
           price = String(price), panel = String(panel), trust = String(trust_name),
           window = String(window),
           n_bets = s.n_bets, n_slates = s.n_slates,
           total_return_pct = s.total_return_pct, flat_roi_pct = s.roi,
           max_drawdown_pct = s.mdd, sharpe_ann = s.sharpe_ann,
           calmar = s.calmar, win_rate = s.win_rate,
           mean_exposure = s.mean_exposure)
    return row, result
end

r04_rows = NamedTuple[]
r04_clv_rows = NamedTuple[]
r04_direction_frames = DataFrame[]

"The Panel A arms, as (container label, odds frame, price tag)."
function r04_panel_a_arms(name)
    arms = Tuple{String, DataFrame, String}[]
    for a in r04_containers[name]
        if a.label == "raw"
            push!(arms, ("raw", close_matched, "close"))
            push!(arms, ("raw", t25_matched, "t25"))
        elseif a.rates == "close"
            push!(arms, (a.label, close_matched, "close"))
        else
            push!(arms, (a.label, close_matched, "close"))
            push!(arms, (a.label, t25_matched, "t25"))
        end
    end
    return arms
end

println("\n--- Panel A · matched keys, close vs T−25 ---")
for name in R04_MODELS
    by_label = Dict(a.label => a for a in r04_containers[name])
    for (label, odds, price) in r04_panel_a_arms(name)
        arm = by_label[label]
        books, _ = r04_books(arm.latents, odds)
        for (tname, policy) in (("flat_1.0", R04_FLAT), ("canonical_P1", R04_CANON))
            row, result = r04_run(name, label, arm.rates, price, "A", tname, books, policy)
            push!(r04_rows, row)
            if price == "t25"
                clv = bet_clv(result.trajectory.bets, r04_drift)
                cs = clv_summary(result.trajectory.bets, clv)
                push!(r04_clv_rows, merge((; model = name, container = label,
                                           trust = tname, panel = "A"), cs))
            end
            d = direction_ledger(result)
            if nrow(d) > 0
                insertcols!(d, 1, :model => name, :container => label,
                            :price => price, :trust => tname)
                push!(r04_direction_frames, d)
            end
            @printf("  %-28s %-10s @%-5s %-13s %5d bets  return %+8.2f%%  Sharpe %5.3f  MDD %7.2f%%\n",
                    name, label, price, tname, row.n_bets, row.total_return_pct,
                    row.sharpe_ann, row.max_drawdown_pct)
        end
    end
end


# %%
# ===================================================================
# 6. Panel B — the deployable configuration, full T−25 book
# ===================================================================

println("\n--- Panel B · full T−25 book (what a bettor actually has) ---")
for name in R04_MODELS
    for arm in r04_containers[name]
        arm.rates == "close" && continue     # a close-rate arm is not deployable
        books, rep = r04_books(arm.latents, t25_full)
        for (tname, policy) in (("flat_1.0", R04_FLAT), ("canonical_P1", R04_CANON))
            row, result = r04_run(name, arm.label, arm.rates, "t25_full", "B",
                                  tname, books, policy)
            push!(r04_rows, row)
            clv = bet_clv(result.trajectory.bets, r04_drift)
            push!(r04_clv_rows, merge((; model = name, container = arm.label,
                                       trust = tname, panel = "B"),
                                      clv_summary(result.trajectory.bets, clv)))
            @printf("  %-28s %-10s @full  %-13s %5d bets  return %+8.2f%%  Sharpe %5.3f  MDD %7.2f%%  (skipped %d)\n",
                    name, arm.label, tname, row.n_bets, row.total_return_pct,
                    row.sharpe_ann, row.max_drawdown_pct, R04_PF.n_skipped(rep))
        end
    end
end


# %%
# ===================================================================
# 7. Out of sample, and risk-matched
# ===================================================================

r04_window(books, w) =
    w == "selection" ? filter(b -> b.date <= R04_SPLIT_DATE, books) :
    w == "evaluation" ? filter(b -> b.date > R04_SPLIT_DATE, books) : books

println("\n--- out of sample · slates after $(R04_SPLIT_DATE), full T−25 book ---")
for name in R04_MODELS
    for arm in r04_containers[name]
        arm.rates == "close" && continue
        books, _ = r04_books(arm.latents, t25_full)
        ev = r04_window(books, "evaluation")
        isempty(ev) && continue
        for (tname, policy) in (("flat_1.0", R04_FLAT), ("canonical_P1", R04_CANON))
            row, _ = r04_run(name, arm.label, arm.rates, "t25_full", "B", tname,
                             ev, policy; window = "evaluation")
            push!(r04_rows, row)
            @printf("  %-28s %-10s %-13s %5d bets  return %+8.2f%%  Sharpe %5.3f  MDD %7.2f%%\n",
                    name, arm.label, tname, row.n_bets, row.total_return_pct,
                    row.sharpe_ann, row.max_drawdown_pct)
        end
    end
end

# Risk-matching, for the same reason r02 §7b needed it: a calibrated container
# stakes smaller, so a fixed-λ return comparison compares two amounts of risk.
# In-sample λ selection; a mechanism demonstration, not a performance claim.
println("\n--- risk-matched to the raw T−25 arm (in-sample λ; NOT a performance claim) ---")
r04_risk_rows = NamedTuple[]
for name in R04_MODELS
    base = filter(r -> r.model == name && r.container == "raw" &&
                       r.price == "t25_full" && r.trust == "flat_1.0" &&
                       r.window == "full", DataFrame(r04_rows))
    nrow(base) == 0 && continue
    target = first(base).max_drawdown_pct
    for arm in r04_containers[name]
        (arm.label == "raw" || arm.rates == "close") && continue
        books, _ = r04_books(arm.latents, t25_full)
        best = nothing
        mdds = Float64[]
        for λ in R04_RISK_LAMBDAS
            row, _ = r04_run(name, arm.label, arm.rates, "t25_full", "F", "flat_1.0",
                             books, l01_policy_spec(FlatTrust(1.0); risk_lambda = λ))
            push!(mdds, row.max_drawdown_pct)
            row.max_drawdown_pct >= target && (best = merge(row, (; risk_lambda = λ,
                                                                 target_mdd = target)))
        end
        best === nothing && continue
        span = maximum(mdds) - minimum(mdds)
        reached = best.max_drawdown_pct <= target + 0.5
        push!(r04_risk_rows, merge(best, (; mdd_span = span, risk_matched = reached)))
        @printf("  %-28s %-10s λ %5.2f  return %+8.2f%%  Sharpe %5.3f  MDD %7.2f%% (raw %7.2f%%)%s\n",
                name, arm.label, best.risk_lambda, best.total_return_pct,
                best.sharpe_ann, best.max_drawdown_pct, target,
                reached ? "" : "  [NOT MATCHED — λ span $(round(span, digits=2))pp]")
    end
end


# %%
# ===================================================================
# 8. Artefacts and the headline tables
# ===================================================================

r04_summary = DataFrame(r04_rows)
r04_clv = isempty(r04_clv_rows) ? DataFrame() : DataFrame(r04_clv_rows)
CSV.write(joinpath(R04_OUT, "r04_portfolio_summary_t25.csv"), r04_summary)
nrow(r04_clv) > 0 && CSV.write(joinpath(R04_OUT, "r04_clv.csv"), r04_clv)
isempty(r04_risk_rows) ||
    CSV.write(joinpath(R04_OUT, "r04_risk_matched_t25.csv"), DataFrame(r04_risk_rows))
isempty(r04_direction_frames) ||
    CSV.write(joinpath(R04_OUT, "r04_direction_ledger_t25.csv"),
              vcat(r04_direction_frames...; cols = :union))

println("\n" * "="^155)
println(" Q1 · WHAT DID CLOSING-PRICE STAKING FLATTER?  (matched keys, same fixtures, same trust)")
println("="^155)
@printf(" %-28s | %-10s | %-13s | %8s | %8s | %8s | %8s\n",
        "Model", "container", "trust", "@close", "@T−25", "Δ return", "Δ Sharpe")
println("-"^155)
for name in R04_MODELS, tname in ("flat_1.0", "canonical_P1")
    a = filter(r -> r.model == name && r.container == "raw" && r.panel == "A" &&
                    r.trust == tname && r.price == "close", r04_summary)
    b = filter(r -> r.model == name && r.container == "raw" && r.panel == "A" &&
                    r.trust == tname && r.price == "t25", r04_summary)
    (nrow(a) == 0 || nrow(b) == 0) && continue
    @printf(" %-28s | %-10s | %-13s | %+8.2f | %+8.2f | %+8.2f | %+8.3f\n",
            name, "raw", tname, first(a).total_return_pct, first(b).total_return_pct,
            first(b).total_return_pct - first(a).total_return_pct,
            first(b).sharpe_ann - first(a).sharpe_ann)
end
println("="^155)

println("\n Q3 · CLOSING-LINE VALUE of the bets each T−25 arm struck")
@printf(" %-28s | %-10s | %-13s | %5s | %10s | %12s | %9s\n",
        "Model", "container", "trust", "bets", "mean CLV%", "stake-wtd %", "% positive")
println("-"^120)
for r in eachrow(r04_clv)
    @printf(" %-28s | %-10s | %-13s | %5d | %+10.3f | %+12.3f | %9.1f\n",
            r.model, r.container, r.trust, r.n_matched, r.mean_clv_pct,
            r.stake_weighted_clv_pct, r.pct_positive)
end
println(" Positive CLV means the market moved TOWARD the bet after it was struck.")
println(" It is evidence about edge; settlement is evidence about edge plus variance.")


# %%
# ===================================================================
# 9. What still stands between this and a Saturday
# ===================================================================

println("\n" * "="^110)
println(" REMAINING GAPS — read before quoting any number above")
println("="^110)
println("""
  1. FILL MODEL. Every bet is struck at the archived traded price in whatever size
     the allocator asked for. The live system rests at the touch and the archive
     carries at most three levels (AGENTS.md §7.4), so a large stake on an
     illiquid Scottish League Two selection would not fill at one price. These
     returns remain an upper bound, a smaller one than r02's.
  2. TRADED PRICE, NOT THE RESTING LADDER. `betfair.odds_history` archives traded
     prices; the console prices off `betfair_live.order_book_1m`. A T−25 traded
     price is what someone paid, not necessarily what was showing on the side we
     would have taken.
  3. STALENESS. Median $(R04_MAX_STALENESS)-bounded; r03 §4 reports the
     distribution. A price 40 minutes old is the last trade, not a live quote.
  4. IN-SAMPLE SPEC SELECTION. r03 chose the calibration parameters on the full
     period. The out-of-sample rows in §7 hold the trust vector out, not the spec.
""")
@printf("  finished   : %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
