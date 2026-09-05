# ==============================================================================
# r07 (1/3) — Line forensics on calibrated latents, and the pruning audit
# ==============================================================================
#
# ------------------------------------------------------------------------------
# THE QUESTION
# ------------------------------------------------------------------------------
#
# `MARKET_LINE_EDA_REPORT.md` opened the trade ledger line by line on RAW latents
# priced at the Betfair CLOSE and pruned the book to `1X2 + O/U 2.5`. Over 2.5 was
# the most negative direction it measured (-10.27% Kelly ROI) and
# `CanonicalScottishLowerTrust` gates it — and every other totals and BTTS
# direction — to exactly zero.
#
# Both of that verdict's premises have since moved: the container is now the
# generative-rate-calibrated one, and the price is T-25 rather than the close. This
# runner re-opens the same ledger under the new premises and asks, per line and per
# side:
#
#   H1  Which previously pruned or gated directions become profitable AND capital
#       efficient under calibration? Specifically, does Over 2.5 clear
#       `Kelly ROI > 0` and `capital_efficiency >= 0.25` in BOTH windows?
#   H1b Does O/U 0.5 stay broken? The claim in `l2_tradeable_markets` is that it is
#       thin rather than merely unprofitable, so this runner measures the ladder's
#       COVERAGE in the T-25 book before it reads a single ROI off it.
#   H4  Do the verdicts agree across `m12` and `m05`?
#
# ------------------------------------------------------------------------------
# THE FORENSIC LENS, AND WHY IT IS FLAT TRUST
# ------------------------------------------------------------------------------
#
# A direction gated to zero strikes no bets and therefore has no economics to
# measure. To ask whether a gate should be lifted the allocator has to be allowed to
# stake every direction, so the forensic arm runs `FlatTrust(1.0)` over all 13
# directions — `l2_full_direction_markets()`, O/U 0.5 included.
#
# That is a MEASUREMENT policy, not a deployable one. It is also exactly the lens
# `MARKET_LINE_EDA_REPORT.md` §1 used, so its per-line table and this one differ in
# the container and the price instant and in nothing else.
#
# Risk settings are §7.4's production ones (`SlateDrawdown(23.0)`, `FixedCap(0.25)`,
# `FractionalKelly(0.30)`, 2% commission) rather than the old report's
# `FixedCap(0.20)`, so that this suite's forensic arm and its portfolio arm sit in
# the same risk regime and a difference between them is about the book and not the
# cap.
#
# ------------------------------------------------------------------------------
# DATABASE BOUNDARY
# ------------------------------------------------------------------------------
#
# READS `mcmc_experiments` and `betdb`. WRITES NEITHER. No paper ledger is opened.
#
# Run on `mcmc-beast`:
#
#     julia --project -t 16
#     julia> include("experiments/scottish_lower/07_calibrated_portfolio_and_trust_vector/r07_line_forensics_calibrated.jl")
# ==============================================================================

# %%
# ===================================================================
# 1. Packages and the loader
# ===================================================================

include(joinpath(@__DIR__, "l07_calibrated_trust_loader.jl"))

using CSV
using DataFrames
using Dates
using Printf
using Statistics


# %%
# ===================================================================
# 2. Configuration
# ===================================================================

"The measurement policy. See the header: flat trust is what un-gates a direction."
const R71_TRUST = FlatTrust(1.0)
const R71_TRUST_LABEL = "flat_1.00"

println("=" ^ 118)
println(" r07/1 — CALIBRATED LINE FORENSICS AND THE PRUNING AUDIT")
println("=" ^ 118)
@printf(" models      : %s\n", join([m.name for m in L07_MODELS], ", "))
@printf(" containers  : %s\n", join([t for (t, _) in l07_calibrators()], ", "))
@printf(" scope       : %d directions (O/U 0.5 included — the exclusion is under test)\n",
        sum(length(Predictions.market_keys(m)) for m in L07_CAL.l2_full_direction_markets()))
@printf(" lens        : %s, SlateDrawdown(%.1f), FixedCap(%.2f), FractionalKelly(%.2f)\n",
        R71_TRUST_LABEL, L07_LAMBDA, L07_CAP, L07_KELLY_FRACTION)
@printf(" split       : selection <= %s < evaluation\n", L07_SPLIT_DATE)
println("=" ^ 118)


# %%
# ===================================================================
# 3. Data — the store, the T-25 book, the inversion, the canonical fits
# ===================================================================

r71_ctx = (@isdefined(L07_CTX) && L07_CTX !== nothing) ? L07_CTX : l07_load_context()
global L07_CTX = r71_ctx


# %%
# ===================================================================
# 4. H1b, first half — what the T-25 book actually quotes, per ladder
# ===================================================================
#
# Read this BEFORE any ROI table below. A direction quoted on 44 fixtures and one
# quoted on 600 produce columns of the same width, and only one of them is evidence.

r71_book_cov = let b = r71_ctx.book
    g = combine(groupby(b, [:market_name, :market_line]),
                nrow => :rows,
                :match_id => (x -> length(unique(x))) => :fixtures,
                :staleness_minutes => median => :median_staleness,
                :overround => median => :median_overround)
    g.market_key = [l07_market_key(n, l) for (n, l) in zip(g.market_name, g.market_line)]
    gate = r71_ctx.gate_ids
    ing = combine(groupby(filter(r -> r.match_id in gate, b), [:market_name, :market_line]),
                  :match_id => (x -> length(unique(x))) => :gate_fixtures)
    g = leftjoin(g, ing, on = [:market_name, :market_line])
    g.gate_fixtures = coalesce.(g.gate_fixtures, 0)
    g.gate_coverage_pct = 100 .* g.gate_fixtures ./ length(gate)
    sort(g, :gate_fixtures; rev = true)
end

println("\n4. T-25 BOOK COVERAGE PER LADDER (gate seasons: $(length(r71_ctx.gate_ids)) fixtures)")
println("-" ^ 118)
@printf("%-10s %8s %10s %14s %16s %10s %12s\n", "ladder", "rows", "fixtures",
        "gate fixtures", "gate coverage %", "stale med", "overround")
println("-" ^ 118)
for r in eachrow(r71_book_cov)
    @printf("%-10s %8d %10d %14d %15.1f%% %10.1f %12.4f\n", r.market_key, r.rows,
            r.fixtures, r.gate_fixtures, r.gate_coverage_pct, r.median_staleness,
            r.median_overround)
end
println("-" ^ 118)


# %%
# ===================================================================
# 5. The forensic simulations — one per (model, container)
# ===================================================================

r71_spec = l07_book_spec(L07_CAL.l2_full_direction_markets())
r71_policy = l07_policy(R71_TRUST)

r71_ledgers = DataFrame[]
r71_summaries = NamedTuple[]
r71_gate1 = NamedTuple[]

println("\n5. FORENSIC SIMULATIONS")
println("-" ^ 118)
for m in L07_MODELS, (tag, cal) in l07_calibrators()
    cf = l07_container(r71_ctx, m.key, cal)
    books, br = L07_PF.build_books_reported(r71_spec, cf, r71_ctx.book, r71_ctx.ds;
                                            quiet = true)
    result = L07_PF.simulate_portfolio(r71_policy, books, br; bootstrap = false)

    ledger = l07_ledger(result, books; model = m.key, container = tag,
                        trust = R71_TRUST_LABEL)
    gate = l07_gate_ledger_accounting(result, ledger)
    ws = L07_CAL.weight_summary(cf.rate_diagnostics)

    push!(r71_ledgers, ledger)
    push!(r71_gate1, (; model = m.key, container = tag, gate...))
    push!(r71_summaries, l07_summary_row(result; model = m.key, container = tag,
                                         trust = R71_TRUST_LABEL, n_books = br.n_books,
                                         w_median = ws.w_median,
                                         var_retention = ws.var_retention_median))
    s = result.summary
    @printf("%-5s %-9s %4d books %5d bets  ret %+8.2f%%  flat ROI %+6.2f%%  Sharpe %5.3f  MDD %7.2f%%  k %.3f  gate1 %s\n",
            m.key, tag, br.n_books, s.n_bets, s.total_return_pct, s.roi, s.sharpe_ann,
            s.mdd, s.mean_k_risk, gate.ok ? "PASS" : "FAIL")
end
println("-" ^ 118)

r71_ledger = vcat(r71_ledgers...)
r71_summary = DataFrame(r71_summaries)

println("\nGATE 1 — LEDGER ACCOUNTING INVARIANTS")
println("-" ^ 118)
@printf("%-5s %-9s %12s %14s %14s %16s %8s\n", "model", "container", "non-finite",
        "stake err", "pnl err", "pnl=stake*payoff", "verdict")
println("-" ^ 118)
for g in r71_gate1
    @printf("%-5s %-9s %12d %14.2e %14.2e %16.2e %8s\n", g.model, g.container,
            g.n_nonfinite, g.stake_err, g.pnl_err, g.identity_err, g.ok ? "PASS" : "FAIL")
end
println("-" ^ 118)
r71_gate1_pass = all(g.ok for g in r71_gate1)
@printf("GATE 1 : %d/%d simulations pass — %s\n", count(g -> g.ok, r71_gate1),
        length(r71_gate1), r71_gate1_pass ? "PASS" : "FAIL")


# %%
# ===================================================================
# 6. Per-line and per-direction breakdown
# ===================================================================
#
# Three scopes. `POOLED` merges the two models so a direction's economics are read
# off twice the sample; the per-model scopes are what H4 is answered on, because a
# verdict that only survives pooling is a verdict about one model plus noise.

r71_breakdowns = DataFrame[]
for (tag, _) in l07_calibrators()
    sub = filter(:container => ==(tag), r71_ledger)
    nrow(sub) == 0 && continue
    pooled = copy(sub)
    pooled.model = fill("POOLED", nrow(pooled))
    push!(r71_breakdowns, l07_breakdown(pooled; scope = "POOLED"))
    for m in L07_MODELS
        ms = filter(:model => ==(m.key), sub)
        nrow(ms) == 0 && continue
        push!(r71_breakdowns, l07_breakdown(ms; scope = m.key))
    end
end
r71_breakdown = vcat(r71_breakdowns...)

"Print one container's pooled, full-period line table."
function r71_print_lines(bd::DataFrame, container::AbstractString)
    rows = sort(filter(r -> r.container == container && r.scope == "POOLED" &&
                            r.window == "full", bd), [:market_key, :direction])
    isempty(rows) && return
    @printf("\n  container = %s   (book Kelly ROI %+.2f%%)\n", container,
            first(rows).book_kelly_roi_pct)
    @printf("  %-8s %-10s %6s %7s %8s %8s %8s %9s %10s %7s %6s\n", "ladder",
            "direction", "bets", "win %", "avg odds", "calib", "edge", "flat ROI",
            "Kelly ROI", "cap %", "eff")
    println("  ", "-" ^ 112)
    for r in eachrow(rows)
        @printf("  %-8s %-10s %6d %7.2f %8.2f %+8.4f %+8.4f %+9.2f %+10.2f %7.2f %6s\n",
                r.market_key, r.direction, r.n_bets, r.win_rate_pct, r.mean_odds,
                r.calib_bias, r.mean_edge, r.flat_roi_pct, r.kelly_roi_pct,
                r.capital_share_pct, l07_fmt(r.capital_efficiency))
    end
end

println("\n6. PER-LINE BREAKDOWN — pooled models, full period")
println("=" ^ 118)
for (tag, _) in l07_calibrators()
    r71_print_lines(r71_breakdown, tag)
end
println("=" ^ 118)


# %%
# ===================================================================
# 7. The pruning audit — the selection-window rule and the out-of-sample gate
# ===================================================================
#
# Two verdicts per row, and they are deliberately different questions:
#
#   `verdict_selection`  `l07_classify` on the SELECTION window alone. This is the
#                        rule as `MARKET_LINE_EDA_REPORT.md` §2 wrote it, and it is
#                        fittable — it sees only data a rule-writer would have had.
#   `oos_gate`           the work package's Gate 2: `Kelly ROI > 0` AND
#                        `capital_efficiency >= 0.25` in BOTH windows. Not fittable
#                        on the selection window, and strictly harder.
#
# A direction is recommended for a non-zero tier only if it passes the second.

function r71_audit(bd::DataFrame)
    rows = NamedTuple[]
    keyed = groupby(bd, [:container, :scope, :market_key, :direction])
    for g in keyed
        full = findfirst(==("full"), g.window)
        sel = findfirst(==("selection"), g.window)
        eva = findfirst(==("evaluation"), g.window)
        full === nothing && continue
        f = NamedTuple(g[full, :])
        s = sel === nothing ? nothing : NamedTuple(g[sel, :])
        e = eva === nothing ? nothing : NamedTuple(g[eva, :])

        verdict, reason = s === nothing ?
            ("CONDITIONAL", "no bets in the selection window") :
            l07_classify(s.kelly_roi_pct, s.capital_efficiency_anchored, s.n_bets)
        gate_ok, gate_reason = l07_oos_gate(s, e)

        sign_held = (s !== nothing && e !== nothing &&
                     isfinite(s.kelly_roi_pct) && isfinite(e.kelly_roi_pct)) ?
                    (sign(s.kelly_roi_pct) == sign(e.kelly_roi_pct) ? "held" : "reversed") :
                    "—"

        push!(rows, (; container = f.container, scope = f.scope, model = f.model,
                     market_key = f.market_key, direction = f.direction,
                     full_bets = f.n_bets, full_kelly_roi = f.kelly_roi_pct,
                     full_flat_roi = f.flat_roi_pct, full_eff = f.capital_efficiency,
                     full_cap_share = f.capital_share_pct,
                     anchor_book_roi = f.anchor_book_roi_pct,
                     sel_bets = s === nothing ? 0 : s.n_bets,
                     sel_kelly_roi = s === nothing ? NaN : s.kelly_roi_pct,
                     sel_eff = s === nothing ? NaN : s.capital_efficiency,
                     sel_eff_anch = s === nothing ? NaN : s.capital_efficiency_anchored,
                     eval_bets = e === nothing ? 0 : e.n_bets,
                     eval_kelly_roi = e === nothing ? NaN : e.kelly_roi_pct,
                     eval_eff = e === nothing ? NaN : e.capital_efficiency,
                     eval_eff_anch = e === nothing ? NaN : e.capital_efficiency_anchored,
                     sign_across_split = sign_held,
                     verdict_selection = verdict, verdict_reason = reason,
                     oos_gate = gate_ok, oos_gate_reason = gate_reason))
    end
    return sort(DataFrame(rows), [:container, :scope, :market_key, :direction])
end

r71_pruning = r71_audit(r71_breakdown)

println("\n7. PRUNING AUDIT — POOLED scope, direction level")
println("""
  `eff*`    capital efficiency against the SELECTION window's book ROI — the denominator
            a rule-writer had at the split date, and the one Gate 2 tests.
  `eff(sw)` the same-window ratio the work package names. Read the two side by side: out
            of sample the book's own ROI is near zero or negative, so this column either
            explodes or is undefined and cannot be compared against 0.25.""")
println("=" ^ 146)
for (tag, _) in l07_calibrators()
    rows = filter(r -> r.container == tag && r.scope == "POOLED" && r.direction != "ALL",
                  r71_pruning)
    isempty(rows) && continue
    @printf("\n  container = %s\n", tag)
    @printf("  %-8s %-10s %6s %10s %6s | %6s %10s %7s %7s | %8s %-12s %s\n",
            "ladder", "direction", "IS n", "IS Kelly", "IS eff", "OOS n", "OOS Kelly",
            "eff*", "eff(sw)", "sign", "IS verdict", "Gate 2")
    println("  ", "-" ^ 140)
    for r in eachrow(rows)
        @printf("  %-8s %-10s %6d %+10.2f %6s | %6d %+10.2f %7s %7s | %8s %-12s %s\n",
                r.market_key, r.direction, r.sel_bets, r.sel_kelly_roi,
                l07_fmt(r.sel_eff_anch), r.eval_bets, r.eval_kelly_roi,
                l07_fmt(r.eval_eff_anch), l07_fmt(r.eval_eff),
                r.sign_across_split, r.verdict_selection,
                r.oos_gate ? "PASS" : "fail")
    end
end
println("=" ^ 146)


# %%
# ===================================================================
# 8. H1 — the rehabilitation table
# ===================================================================
#
# The one comparison this runner exists to make: what the raw container says about a
# direction, against what each calibrated container says, on the SAME fixtures,
# prices, policy and split.

println("\n8. REHABILITATION — raw against calibrated, POOLED, direction level")
println("=" ^ 138)
r71_rehab = let
    pooled = filter(r -> r.scope == "POOLED" && r.direction != "ALL", r71_pruning)
    raw = Dict((r.market_key, r.direction) => r for r in eachrow(filter(:container => ==("raw"), pooled)))
    rows = NamedTuple[]
    for r in eachrow(filter(r -> r.container != "raw", pooled))
        b = get(raw, (r.market_key, r.direction), nothing)
        b === nothing && continue
        push!(rows, (; container = r.container, market_key = r.market_key,
                     direction = r.direction,
                     raw_full_kelly = b.full_kelly_roi, cal_full_kelly = r.full_kelly_roi,
                     delta_full_kelly = r.full_kelly_roi - b.full_kelly_roi,
                     raw_oos_kelly = b.eval_kelly_roi, cal_oos_kelly = r.eval_kelly_roi,
                     raw_gate = b.oos_gate, cal_gate = r.oos_gate,
                     status = (!b.oos_gate && r.oos_gate) ? "REHABILITATED" :
                              (b.oos_gate && !r.oos_gate) ? "LOST" :
                              (b.oos_gate && r.oos_gate) ? "held (both pass)" :
                              "both fail"))
    end
    sort(DataFrame(rows), [:container, :market_key, :direction])
end

@printf("  %-9s %-8s %-10s %11s %11s %10s %11s %11s %s\n", "container", "ladder",
        "direction", "raw Kelly", "cal Kelly", "delta", "raw OOS", "cal OOS", "status")
println("  ", "-" ^ 132)
for r in eachrow(r71_rehab)
    @printf("  %-9s %-8s %-10s %+11.2f %+11.2f %+10.2f %+11.2f %+11.2f %s\n",
            r.container, r.market_key, r.direction, r.raw_full_kelly, r.cal_full_kelly,
            r.delta_full_kelly, r.raw_oos_kelly, r.cal_oos_kelly, r.status)
end
println("=" ^ 138)

r71_rehabilitated = filter(:status => ==("REHABILITATED"), r71_rehab)
println()
if nrow(r71_rehabilitated) == 0
    println("  No direction that failed Gate 2 on raw latents passes it on any calibrated container.")
else
    @printf("  %d (container, direction) pairs move from FAIL to PASS on Gate 2:\n",
            nrow(r71_rehabilitated))
    for r in eachrow(r71_rehabilitated)
        @printf("    %-9s %-8s %-10s  full Kelly ROI %+.2f%% -> %+.2f%%\n",
                r.container, r.market_key, r.direction, r.raw_full_kelly, r.cal_full_kelly)
    end
end


# %%
# ===================================================================
# 8b. PARITY — the published Over 2.5 figure, and what the split does to it
# ===================================================================
#
# The work package's premise for H1 is `README §8.9`: "Generative Rate Calibration
# with predictive rate anchoring (`:pool_mean`) repaired Over 2.5 (+14.3% Kelly ROI
# on `m12`)". That number was measured on ONE model, the `inv_B_anch` container,
# FLAT trust, the **11-direction** tradeable book, and the WHOLE period.
#
# Section 8 above measures Over 2.5 over 13 directions and both models pooled, so it
# is not the same quantity and a disagreement between them would prove nothing. This
# section rebuilds the published arm exactly, checks it reproduces, and then splits
# the SAME arm at 2025-05-03 — which is the thing §8.9 never did and the thing the
# work package is asking about.

const R71_PUBLISHED_OVER25 = (bets = 38, kelly_roi = 14.32, flat_roi = 7.10)

println("\n8b. PARITY WITH README §8.9 — m12, inv_anch, flat trust, 11 tradeable directions")
println("=" ^ 118)
r71_parity = let
    cal = last(l07_calibrators())[2]                     # inv_anch
    cf = l07_container(r71_ctx, "m12", cal)
    spec = l07_book_spec(L07_CAL.l2_tradeable_markets())
    books, br = L07_PF.build_books_reported(spec, cf, r71_ctx.book, r71_ctx.ds; quiet = true)
    result = L07_PF.simulate_portfolio(l07_policy(FlatTrust(1.0)), books, br;
                                       bootstrap = false)
    led = l07_ledger(result, books; model = "m12", container = "inv_anch",
                     trust = "flat_1.00")
    bd = l07_breakdown(led; scope = "m12_11dir")
    filter(r -> r.market_key == "OU2.5" && r.direction == "over_25", bd)
end

@printf("  %-12s %6s %12s %12s %10s\n", "window", "bets", "Kelly ROI %", "flat ROI %", "eff*")
println("  ", "-" ^ 60)
for w in ("full", "selection", "evaluation")
    r = filter(:window => ==(w), r71_parity)
    nrow(r) == 0 && continue
    row = first(r)
    @printf("  %-12s %6d %+12.2f %+12.2f %10s\n", w, row.n_bets, row.kelly_roi_pct,
            row.flat_roi_pct, l07_fmt(row.capital_efficiency_anchored))
end
println("  ", "-" ^ 60)
r71_parity_full = first(filter(:window => ==("full"), r71_parity))
@printf("  published (§8.9)  %6d %+12.2f %+12.2f\n", R71_PUBLISHED_OVER25.bets,
        R71_PUBLISHED_OVER25.kelly_roi, R71_PUBLISHED_OVER25.flat_roi)
r71_parity_ok = r71_parity_full.n_bets == R71_PUBLISHED_OVER25.bets &&
                abs(r71_parity_full.kelly_roi_pct - R71_PUBLISHED_OVER25.kelly_roi) <= 0.01 &&
                abs(r71_parity_full.flat_roi_pct - R71_PUBLISHED_OVER25.flat_roi) <= 0.01
@printf("  PARITY : %s\n", r71_parity_ok ? "reproduces §8.9 exactly" :
        "DOES NOT reproduce §8.9 — the arm above is not the arm that was published")
println("=" ^ 118)


# %%
# ===================================================================
# 9. H4 — do the two models agree, direction by direction?
# ===================================================================
#
# Agreement is on the GATE, not on the return: two models that both pass with
# different ROIs agree about the decision, which is the thing being transferred to
# production.

println("\n9. CROSS-MODEL CONSISTENCY — Gate 2 verdicts, per model")
println("=" ^ 118)
r71_consistency = let
    per = filter(r -> r.scope != "POOLED" && r.direction != "ALL", r71_pruning)
    rows = NamedTuple[]
    for g in groupby(per, [:container, :market_key, :direction])
        d = Dict(r.scope => r for r in eachrow(g))
        m12 = get(d, "m12", nothing); m05 = get(d, "m05", nothing)
        (m12 === nothing || m05 === nothing) && continue
        push!(rows, (; container = first(g.container), market_key = first(g.market_key),
                     direction = first(g.direction),
                     m12_gate = m12.oos_gate, m05_gate = m05.oos_gate,
                     agree = m12.oos_gate == m05.oos_gate,
                     m12_oos_kelly = m12.eval_kelly_roi, m05_oos_kelly = m05.eval_kelly_roi))
    end
    sort(DataFrame(rows), [:container, :market_key, :direction])
end

for (tag, _) in l07_calibrators()
    sub = filter(:container => ==(tag), r71_consistency)
    isempty(sub) && continue
    both = filter(r -> r.m12_gate && r.m05_gate, sub)
    @printf("  %-9s  %d/%d directions agree | %d pass on BOTH models: %s\n", tag,
            count(sub.agree), nrow(sub), nrow(both),
            isempty(both) ? "none" :
            join([string(r.market_key, ":", r.direction) for r in eachrow(both)], ", "))
end
println("=" ^ 118)


# %%
# ===================================================================
# 10. Artefacts
# ===================================================================

println("\n10. ARTEFACTS")
l07_write(r71_breakdown, "market_line_breakdown_calibrated.csv")
l07_write(r71_pruning, "market_pruning_audit_calibrated.csv")
l07_write(r71_rehab, "market_line_rehabilitation.csv")
l07_write(r71_consistency, "market_line_cross_model.csv")
l07_write(r71_summary, "forensic_portfolio_summary.csv")
l07_write(r71_book_cov, "t25_book_ladder_coverage.csv")
l07_write(r71_parity, "over25_parity_with_readme_8_9.csv")
l07_write(r71_ledger, "forensic_bet_ledger.csv")

println()
println("=" ^ 118)
println(" R71_DONE  ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
println("=" ^ 118)
