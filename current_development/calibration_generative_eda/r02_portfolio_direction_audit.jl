# ==============================================================================
# r02 — Generative rate calibration: direction and trust-vector portfolio audit
# ==============================================================================
#
# ------------------------------------------------------------------------------
# WHAT THIS IS AND IS NOT
# ------------------------------------------------------------------------------
#
# This is Gate 2. r01 established that generative rate calibration improves
# out-of-sample LogLoss on both candidates, that the gain is almost entirely a
# LARGE-EDGE phenomenon, and that the winning direction is SHRINKAGE rather than
# the Ireland stream's conviction. None of that is a bankroll claim, and the
# Ireland transfer is the standing proof that it does not have to be one: it
# improved its own league's diagnostics and still lost 16-22% of final wealth here.
#
# So this runner asks three questions and keeps them apart:
#
#   Q1  Does calibration change the BANKROLL, at a matched policy?
#   Q2  Does it rescue a direction the league has priced badly — Over 2.5 above
#       all, which the audited production trust gates to zero?
#   Q3  Does it make `CanonicalScottishLowerTrust` unnecessary — i.e. does a
#       calibrated model under FLAT trust match the tiered champion?
#
# It is NOT a live-tradeable P&L. Every price here is the Betfair CLOSE, which is
# not available at the moment a bet is struck; the numbers are an upper bound on
# what a T-25 calibration could earn. This is the single easiest way to over-read
# the stream, so it is stated before the first table rather than after the last.
#
# ------------------------------------------------------------------------------
# THE COMPARISON CONTRACT
# ------------------------------------------------------------------------------
#
# 1. ONE BOOK, ONE CAP, ONE RISK BUDGET. Experiment 06 reports its flat baseline
#    (+136.61%) at `FixedCap(0.20)` over three markets and its tiered champion
#    (+155.93%) at `FixedCap(0.25)` over thirteen directions. Those two numbers
#    differ in the book AND the cap AND the trust, so their difference attributes
#    nothing. Every arm below holds `l01_book_spec`, `SlateDrawdown(23.0)`,
#    `FixedCap(0.25)`, `DailySlate()` and `FractionalKelly(0.30)` fixed and varies
#    ONLY the latent container and the trust model. R0 reproduces the published
#    champion separately so the reader can see what the book restriction cost.
#
# 2. THE 2x2 IS COMPLETED, NOT ASSUMED. The work package lists three calibrated
#    arms against one raw one. Without `raw + FlatTrust` the calibration effect and
#    the trust effect are confounded in every comparison that follows, so arm B is
#    added. It is the only arm here that is not in the brief, and it is the one
#    that makes the brief's arms readable.
#
# 3. O/U 0.5 IS NOT STAKED, AND THE REASON IS IN THE DATA. The closing archive
#    quotes that ladder 982 over against 408 under, so 574 fixtures are one-sided
#    and `l01_betfair_closing_odds` de-vigs them to `prob_fair_close = 1.0`. The
#    closing line's own LogLoss on that family is 1.31832 against the model's
#    0.21098. Staking it would manufacture edge out of a de-vigging artefact.
#    Arm R1 stakes it anyway, once, so the exclusion is evidenced rather than
#    asserted.
#
# 4. THE TRUST RE-OPTIMISATION IS FITTED AND SCORED ON DIFFERENT SLATES. The rule
#    sees slates up to `R02_SPLIT_DATE` and is scored on the ones after it, the
#    same protocol and the same split `MARKET_LINE_EDA_REPORT.md` used. That report
#    is also the record of this rule class REVERSING SIGN across the split on two
#    of six lines, so the out-of-sample column is the one that adjudicates and the
#    full-period column is reported only for continuity with the published table.
#
# 5. THE FIXTURE SET IS r01's. 710 OOS fixtures in 24/25 + 25/26, the 26/27
#    extension restricted out, so Gate 1 and Gate 2 are answered on the same
#    matches.
#
# ------------------------------------------------------------------------------
# PERSISTENCE CAVEAT
# ------------------------------------------------------------------------------
#
# Outputs are REPLACEABLE CSVs under `results/`. This runner writes NOTHING to
# `mcmc_experiments` — no `save_portfolio_db`, no run registration — because these
# are calibrated containers, not persisted inference runs, and a portfolio row
# whose `run_id` points at a posterior it was not computed from would be worse than
# no row. `betdb` is read for odds and results only; `paper_runbook` is never
# opened and the consoles on 8085 / 8086 are untouched.
#
# ------------------------------------------------------------------------------
# USAGE
# ------------------------------------------------------------------------------
#
#   julia --project -t 16
#   julia> include("current_development/calibration_generative_eda/r02_portfolio_direction_audit.jl")
#
#   R02_SMOKE=1 ...   # m12 only, one calibrated spec, no trust re-optimisation
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

const R02_PF = BayesianFootball.Portfolio


# %%
# ===================================================================
# 2. Configuration
# ===================================================================

const R02_EXPERIMENT = "scottish_lower_joint_player_2426"
const R02_GATE_SEASONS = ["24/25", "25/26"]

const R02_SMOKE = get(ENV, "R02_SMOKE", "0") != "0"

"""
The specs nominated by r01 §5.3 — the best JOINT improver (LogLoss and ECE both no
worse than the model's own baseline) per functional form.

`r01_best_per_form.csv` selects on LogLoss alone and lands on ECE-degrading points
for `m12`; those are not carried here. `m12` has no static-form joint improver at
all — its only one is the identity — which is itself the finding that a constant
pool cannot reproduce what the Delta-dependence buys on that model.

The two `m12` arms are chosen to be a CONTROLLED CONTRAST: they land within 0.0006
LogLoss of each other while retaining 84.6% and 6.7% of posterior log-variance
respectively. Kelly stake size reads that variance, so if variance retention
matters to a bankroll, these two arms are where it has to show.
"""
const R02_NOMINATIONS = Dict(
    "m12_joint_hybrid_synergy" => [
        (label = "std", spec = GenerativeCalibrationSpec(
            method = :standard_gaussian, w_base = 0.25, sigma = 0.25)),
        (label = "inv", spec = GenerativeCalibrationSpec(
            method = :inverse_gaussian, w_base = 0.25, sigma = 0.75)),
    ],
    "m05_joint_production_wealth" => [
        (label = "std", spec = GenerativeCalibrationSpec(
            method = :standard_gaussian, w_base = 0.25, sigma = 0.15)),
        (label = "inv", spec = GenerativeCalibrationSpec(
            method = :inverse_gaussian, w_base = 0.40, sigma = 1.00)),
    ],
)

const R02_MODELS = R02_SMOKE ? ["m12_joint_hybrid_synergy"] :
    ["m12_joint_hybrid_synergy", "m05_joint_production_wealth"]

"The window the trust rule is allowed to see. `MARKET_LINE_EDA_REPORT.md`'s split."
const R02_SPLIT_DATE = Date(2025, 5, 3)

const R02_INVERSION = MarketInversionConfig()
const R02_MAX_GOALS = 12
const R02_MIN_BETS_FOR_TIER = 50

# Published experiment-06 reference points, for continuity only. §9 judges each arm
# against the arm it is matched with, never against these.
const R02_PUBLISHED_M12_TIERED_RETURN = 155.93
const R02_PUBLISHED_M12_TIERED_SHARPE = 1.636
const R02_PUBLISHED_M12_TIERED_MDD    = -19.79
const R02_PUBLISHED_M12_FLAT_RETURN   = 136.61

# Gate 2 thresholds, from the work package.
const R02_GATE_RETURN = 130.0
const R02_GATE_SHARPE = 1.416
const R02_GATE_MDD    = -20.5


# %%
# ===================================================================
# 3. Runtime and output directory
# ===================================================================

const R02_OUT = joinpath(@__DIR__, "results")
mkpath(R02_OUT)

println("\n" * "="^110)
println(" r02 · CALIBRATED PORTFOLIO AND 13-DIRECTION AUDIT — Scottish Lower")
println("="^110)
@printf("  started    : %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
@printf("  threads    : %d\n", Threads.nthreads())
@printf("  experiment : %s (read-only)\n", R02_EXPERIMENT)
@printf("  split date : %s (trust rule sees on or before; scored after)\n", R02_SPLIT_DATE)
R02_SMOKE && println("  MODE       : SMOKE — one model, one spec, no re-optimisation")
println("\n  Every price below is the Betfair CLOSE. A closing price is not available")
println("  when a bet is struck, so every return here is an UPPER BOUND.")


# %%
# ===================================================================
# 4. Data, fits, closing book and inverted market rates
#    G-A · the same fixture set r01 scored
# ===================================================================

ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)
db = PostgresStorage(R02_EXPERIMENT)

r02_gate_match_ids = Set{Int}(
    Int.(ds.matches.match_id[in.(ds.matches.season, Ref(R02_GATE_SEASONS))]))

bf_odds = l01_betfair_closing_odds(ds)

r02_raw = Dict{String, Any}()
for name in R02_MODELS
    fit = load_fit(db, name)
    fit.diagnostics.passed || @warn "$name did not pass strict convergence gating"
    r02_raw[name] = restrict_latents(fit_latents(fit), r02_gate_match_ids)
end

r02_rates = invert_market_rates(
    bf_odds; config = R02_INVERSION,
    match_ids = sort!(collect(union((Set(latent_match_ids(r02_raw[n])) for n in R02_MODELS)...))))

println("\n--- G-A · fixture set and inversion ---")
for name in R02_MODELS
    cov = inversion_coverage(r02_rates, latent_match_ids(r02_raw[name]))
    @printf("  %-30s %d fixtures | inverted %d of %d quoted (%.1f%%; %.1f%% of all)\n",
            name, cov.n_fixtures, cov.n_accepted, cov.n_quoted,
            100 * cov.coverage_quoted, 100 * cov.coverage)
end

"Every latent container this runner stakes: the raw one and each nomination."
r02_containers = Dict{String, Vector{NamedTuple}}()
for name in R02_MODELS
    arms = NamedTuple[(label = "raw", spec = nothing, latents = r02_raw[name],
                       weights = nothing)]
    noms = R02_SMOKE ? R02_NOMINATIONS[name][1:1] : R02_NOMINATIONS[name]
    for n in noms
        cal, diag = calibrate_latents(r02_raw[name], r02_rates, n.spec)
        push!(arms, (label = n.label, spec = n.spec, latents = cal,
                     weights = weight_summary(diag)))
    end
    r02_containers[name] = arms
    for a in arms
        a.weights === nothing && continue
        @printf("  %-30s %-4s %-18s median w %.3f, posterior log-variance retained %.1f%%\n",
                name, a.label, spec_label(a.spec), a.weights.w_median,
                100 * a.weights.var_retention_median)
    end
end


# %%
# ===================================================================
# 5. The books
#    G-B · one book per (model, container); policies are free after this
# ===================================================================
#
# `BookSpec` is the cache key and `PolicySpec` is not, so every trust model below
# re-stakes the SAME books. That is what makes the trust comparison exact rather
# than merely matched: two arms differing only in trust are literally reading one
# pricing pass.

const R02_BOOK_T = l01_book_spec(l01_tradeable_markets())
const R02_BOOK_F = l01_book_spec(l01_full_direction_markets())

println("\n--- G-B · book construction ---")
r02_books = Dict{Tuple{String,String}, Any}()
for name in R02_MODELS, arm in r02_containers[name]
    books, rep = R02_PF.build_books_reported(
        R02_BOOK_T, arm.latents, bf_odds, ds.matches; require_result = true, quiet = true)
    r02_books[(name, arm.label)] = books
    @printf("  %-30s %-4s tradeable book: %d fixtures priced, %d skipped\n",
            name, arm.label, length(books), R02_PF.n_skipped(rep))
end

# The diagnostic arm that evidences the O/U 0.5 exclusion.
r02_books_full, r02_rep_full = R02_PF.build_books_reported(
    R02_BOOK_F, r02_raw[first(R02_MODELS)], bf_odds, ds.matches;
    require_result = true, quiet = true)
@printf("  %-30s %-4s 13-direction book: %d fixtures priced, %d skipped\n",
        first(R02_MODELS), "raw", length(r02_books_full), R02_PF.n_skipped(r02_rep_full))


# %%
# ===================================================================
# 6. The arms
# ===================================================================

"Run one policy over one prebuilt book set and return the summary as a row."
function r02_run(label, model, container, trust_name, books, policy; window = :full)
    result = R02_PF.simulate_portfolio(policy, books; bootstrap = false)
    s = result.summary
    row = (
        arm = String(label), model = String(model), container = String(container),
        trust = String(trust_name), window = String(window),
        n_bets = s.n_bets, n_slates = s.n_slates,
        total_return_pct = s.total_return_pct, flat_roi_pct = s.roi,
        roi_1x2_pct = s.roi_1x2, max_drawdown_pct = s.mdd,
        sharpe_ann = s.sharpe_ann, calmar = s.calmar, sortino = s.sortino,
        win_rate = s.win_rate, mean_exposure = s.mean_exposure,
        total_stake = s.total_stake, total_pnl = s.total_pnl,
    )
    return row, result
end

"Books whose fixture date falls in the requested window."
r02_window_books(books, window) =
    window === :selection ? filter(b -> b.date <= R02_SPLIT_DATE, books) :
    window === :evaluation ? filter(b -> b.date > R02_SPLIT_DATE, books) : books

const R02_FLAT = l01_policy_spec(FlatTrust(1.0))
const R02_CANON = l01_policy_spec(CanonicalScottishLowerTrust())

println("\n--- arms A–D · calibration x trust, matched book and cap ---")
r02_rows = NamedTuple[]
r02_results = Dict{String, Any}()
r02_direction_frames = DataFrame[]

for name in R02_MODELS
    for arm in r02_containers[name]
        books = r02_books[(name, arm.label)]
        for (trust_name, policy) in (("flat_1.0", R02_FLAT), ("canonical_P1", R02_CANON))
            key = "$(name)|$(arm.label)|$(trust_name)"
            row, result = r02_run(key, name, arm.label, trust_name, books, policy)
            push!(r02_rows, row)
            r02_results[key] = result
            d = direction_ledger(result)
            if nrow(d) > 0
                insertcols!(d, 1, :model => name, :container => arm.label,
                            :trust => trust_name)
                push!(r02_direction_frames, d)
            end
            @printf("  %-30s %-4s %-13s %5d bets  return %+8.2f%%  Sharpe %5.3f  MDD %7.2f%%\n",
                    name, arm.label, trust_name, row.n_bets, row.total_return_pct,
                    row.sharpe_ann, row.max_drawdown_pct)
        end
    end
end

# R1 — the 13-direction diagnostic that evidences the O/U 0.5 exclusion.
println("\n--- arm R1 · the 13-direction book, raw, flat trust (diagnostic) ---")
let name = first(R02_MODELS)
    row, result = r02_run("$(name)|raw|flat_1.0|13dir", name, "raw", "flat_1.0",
                          r02_books_full, R02_FLAT)
    push!(r02_rows, merge(row, (; arm = "R1_13direction")))
    d = direction_ledger(result)
    if nrow(d) > 0
        insertcols!(d, 1, :model => name, :container => "raw", :trust => "flat_1.0_13dir")
        push!(r02_direction_frames, d)
    end
    @printf("  %-30s %-4s %-13s %5d bets  return %+8.2f%%  Sharpe %5.3f  MDD %7.2f%%\n",
            name, "raw", "13-direction", row.n_bets, row.total_return_pct,
            row.sharpe_ann, row.max_drawdown_pct)
    ou05 = filter(r -> r.selection in (:over_05, :under_05), d)
    for r in eachrow(ou05)
        @printf("    O/U 0.5 %-10s %4d bets  Kelly ROI %+8.2f%%  capital %5.2f%%  mean p_market %.4f\n",
                String(r.selection), r.n_bets, r.kelly_roi, r.capital_share, r.p_market)
    end
end


# %%
# ===================================================================
# 7. The 13-direction audit
# ===================================================================

r02_directions = isempty(r02_direction_frames) ? DataFrame() :
                 vcat(r02_direction_frames...; cols = :union)
CSV.write(joinpath(R02_OUT, "r02_direction_ledger.csv"), r02_directions)

println("\n" * "="^150)
println(" DIRECTION AUDIT — raw vs calibrated, FLAT trust (so every direction is stakeable)")
println("="^150)
@printf(" %-30s | %-11s | %5s | %9s | %9s | %8s | %8s | %9s\n",
        "Model / direction", "container", "bets", "Kelly ROI", "flat ROI",
        "capital", "effic.", "calib.")
println("-"^150)
for name in R02_MODELS
    labels = [a.label for a in r02_containers[name]]
    sels = unique(filter(r -> r.model == name && r.trust == "flat_1.0", r02_directions).selection)
    for sel in sort(sels; by = String)
        for lab in labels
            r = filter(x -> x.model == name && x.trust == "flat_1.0" &&
                            x.container == lab && x.selection == sel, r02_directions)
            nrow(r) == 0 && continue
            r = first(r)
            @printf(" %-30s | %-11s | %5d | %+9.2f | %+9.2f | %7.2f%% | %8.2f | %+9.4f\n",
                    lab == first(labels) ? "$name $(String(sel))" : "", lab,
                    r.n_bets, r.kelly_roi, r.flat_roi, r.capital_share,
                    r.efficiency, r.calibration)
        end
    end
end
println("="^150)


# %%
# ===================================================================
# 7b. Arm F — the risk-matched comparison
# ===================================================================
#
# WHY THIS ARM EXISTS. Log-linear pooling contracts posterior log-variance by w².
# Kelly stake size is monotone in that variance, so a calibrated container stakes
# LESS and therefore compounds less, at the same `SlateDrawdown` budget. Total
# return then measures how much risk each arm took as much as how well it picked,
# and the work package's Gate-2 return threshold — a SCALE criterion — is one a
# variance-contracting transform fails mechanically while improving every SHAPE
# criterion beside it.
#
# This arm separates the two. It loosens λ on the calibrated container until its
# maximum drawdown matches the raw arm's, and reads the return there.
#
# IT IS NOT A PERFORMANCE CLAIM. λ is chosen against the same slates the return is
# then read off. That is in-sample selection, and it is the one place in this stream
# where that is done deliberately, to demonstrate a mechanism rather than to
# estimate a return. Nothing in §9's Gate-2 verdict reads this table.

const R02_RISK_LAMBDAS = [23.0, 18.0, 15.0, 12.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0]

println("\n--- arm F · risk-matched (mechanism demonstration, NOT a performance claim) ---")
r02_risk_rows = NamedTuple[]
for name in R02_MODELS
    labels = [a.label for a in r02_containers[name]]
    for trust_name in ("flat_1.0", "canonical_P1")
        trust = trust_name == "flat_1.0" ? FlatTrust(1.0) : CanonicalScottishLowerTrust()
        base = filter(r -> r.model == name && r.container == "raw" &&
                           r.trust == trust_name && r.window == "full", r02_summary)
        nrow(base) == 0 && continue
        target = first(base).max_drawdown_pct
        for lab in labels
            lab == "raw" && continue
            books = r02_books[(name, lab)]
            best = nothing
            for λ in R02_RISK_LAMBDAS
                row, _ = r02_run("F_risk_matched", name, lab, trust_name, books,
                                 l01_policy_spec(trust; risk_lambda = λ))
                cand = merge(row, (; risk_lambda = λ, target_mdd = target))
                # keep the loosest budget whose drawdown is still no deeper than raw's
                if cand.max_drawdown_pct >= target
                    best = cand
                end
            end
            if best === nothing
                @printf("  %-30s %-4s %-13s no λ in the grid stays inside raw's %.2f%% drawdown\n",
                        name, lab, trust_name, target)
                continue
            end
            push!(r02_risk_rows, best)
            @printf("  %-30s %-4s %-13s λ %5.1f  return %+8.2f%%  Sharpe %5.3f  MDD %7.2f%% (raw %7.2f%%)\n",
                    name, lab, trust_name, best.risk_lambda, best.total_return_pct,
                    best.sharpe_ann, best.max_drawdown_pct, target)
        end
    end
end
r02_risk = isempty(r02_risk_rows) ? DataFrame() : DataFrame(r02_risk_rows)
nrow(r02_risk) > 0 && CSV.write(joinpath(R02_OUT, "r02_risk_matched.csv"), r02_risk)


# %%
# ===================================================================
# 8. The re-optimised trust vector
#    Fitted on slates <= split, scored on slates > split
# ===================================================================

r02_trust_tables = DataFrame[]
if !R02_SMOKE
    println("\n--- arm E · trust vector refitted on the selection window ---")
    for name in R02_MODELS
        for arm in r02_containers[name]
            books = r02_books[(name, arm.label)]
            sel_books = r02_window_books(books, :selection)
            eval_books = r02_window_books(books, :evaluation)
            (isempty(sel_books) || isempty(eval_books)) && continue

            _, sel_result = r02_run("fit", name, arm.label, "flat_1.0", sel_books, R02_FLAT)
            sel_ledger = direction_ledger(sel_result)
            nrow(sel_ledger) == 0 && continue
            trust, table = tiered_trust_from_ledger(
                sel_ledger; min_bets = R02_MIN_BETS_FOR_TIER)
            insertcols!(table, 1, :model => name, :container => arm.label)
            push!(r02_trust_tables, table)

            refit_policy = l01_policy_spec(trust)
            for (wname, wbooks) in (("evaluation", eval_books), ("full", books))
                row, result = r02_run("E_refit_trust", name, arm.label, "refit_P?",
                                      wbooks, refit_policy; window = wname)
                push!(r02_rows, row)
                @printf("  %-30s %-4s refit  %-10s %5d bets  return %+8.2f%%  Sharpe %5.3f  MDD %7.2f%%\n",
                        name, arm.label, wname, row.n_bets, row.total_return_pct,
                        row.sharpe_ann, row.max_drawdown_pct)
            end

            # The matched out-of-sample comparators for the refit vector.
            for (tname, policy) in (("flat_1.0", R02_FLAT), ("canonical_P1", R02_CANON))
                row, _ = r02_run("E_comparator", name, arm.label, tname,
                                 eval_books, policy; window = "evaluation")
                push!(r02_rows, row)
            end
        end
    end
end

r02_trust_frame = isempty(r02_trust_tables) ? DataFrame() : vcat(r02_trust_tables...)
nrow(r02_trust_frame) > 0 &&
    CSV.write(joinpath(R02_OUT, "r02_refit_trust_vectors.csv"), r02_trust_frame)

r02_summary = DataFrame(r02_rows)
CSV.write(joinpath(R02_OUT, "r02_portfolio_summary.csv"), r02_summary)


# %%
# ===================================================================
# 9. Gate 2
# ===================================================================

println("\n" * "="^150)
println(" ARM COMPARISON — matched book (11 directions), FixedCap(0.25), SlateDrawdown(23.0)")
println("="^150)
@printf(" %-30s | %-11s | %-13s | %5s | %9s | %8s | %8s | %8s | %8s\n",
        "Model", "container", "trust", "bets", "return %", "flat ROI", "Sharpe",
        "MDD %", "exposure")
println("-"^160)
for row in eachrow(filter(r -> r.window == "full" && r.arm != "R1_13direction", r02_summary))
    @printf(" %-30s | %-11s | %-13s | %5d | %+9.2f | %+8.2f | %8.3f | %8.2f | %7.2f%%\n",
            row.model, row.container, row.trust, row.n_bets, row.total_return_pct,
            row.flat_roi_pct, row.sharpe_ann, row.max_drawdown_pct,
            100 * row.mean_exposure)
end
println("="^150)

println("\n GATE 2 — bankroll, risk-adjusted return, drawdown")
println(" Thresholds: return > $(R02_GATE_RETURN)%, annual Sharpe >= $(R02_GATE_SHARPE), MDD no worse than $(R02_GATE_MDD)%")
println(" Judged on the FULL-PERIOD matched arms. Arm E's full-period column contains")
println(" the window its rule was fitted on and is NOT eligible; read its evaluation row.")
println("-"^150)
for row in eachrow(filter(r -> r.window == "full" && r.arm != "R1_13direction" &&
                               r.trust != "refit_P?", r02_summary))
    ok_r = row.total_return_pct > R02_GATE_RETURN
    ok_s = row.sharpe_ann >= R02_GATE_SHARPE
    ok_d = row.max_drawdown_pct >= R02_GATE_MDD
    verdict = (ok_r && ok_s && ok_d) ? "PASS" : "REFUSE"
    @printf(" [%-6s] %-30s %-11s %-13s  return %s  Sharpe %s  MDD %s\n",
            verdict, row.model, row.container, row.trust,
            ok_r ? "ok" : "low", ok_s ? "ok" : "low", ok_d ? "ok" : "deep")
end

println("\n PUBLISHED REFERENCE (experiment 06, different book and cap — context only)")
@printf("   m12 + CanonicalScottishLowerTrust : %+.2f%%, Sharpe %.3f, MDD %.2f%%\n",
        R02_PUBLISHED_M12_TIERED_RETURN, R02_PUBLISHED_M12_TIERED_SHARPE,
        R02_PUBLISHED_M12_TIERED_MDD)
@printf("   m12 + FlatTrust(1.0), 3 markets  : %+.2f%%\n", R02_PUBLISHED_M12_FLAT_RETURN)


# %%
# ===================================================================
# 10. Final report
# ===================================================================

println("\n" * "="^110)
println(" ARTEFACTS")
println("="^110)
for f in ["r02_portfolio_summary.csv", "r02_direction_ledger.csv",
          "r02_refit_trust_vectors.csv", "r02_risk_matched.csv"]
    isfile(joinpath(R02_OUT, f)) && @printf("  %s\n", joinpath(R02_OUT, f))
end
println("""
  READ IN THIS ORDER.
    Q1  calibration effect  = (raw, flat) vs (std|inv, flat), same book and cap.
    Q2  Over 2.5            = the over_25 rows of §7, raw against calibrated.
    Q3  trust necessity     = (std, flat) against (raw, canonical_P1).
    Q4  scale vs shape      = arm F. If the calibrated arm reaches raw's return
                              once its drawdown is loosened to match, the return
                              shortfall was exposure, not skill. In-sample λ.
  A calibrated arm that beats raw+flat but not raw+canonical has improved the
  model, not replaced the trust vector. Record the numbers in README.md §6 with
  this run's context before drawing a conclusion from any of them.
""")
@printf("  finished   : %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
