# ==============================================================================
# r07 (3/3) — The head-to-head: candidate policies against the production benchmarks
# ==============================================================================
#
# ------------------------------------------------------------------------------
# THE QUESTION
# ------------------------------------------------------------------------------
#
# r07/1 says which DIRECTIONS have edge. r07/2 says how much TRUST and how much
# RISK BUDGET to spend. Neither answers the question the MatchDay consoles actually
# pose, which is a single joint one: **what should the deployed policy be, and does
# it beat what is deployed today on slates it has never seen?**
#
# `MARKET_LINE_EDA_REPORT.md` §5.1 is the cautionary precedent and the reason this
# runner exists in the shape it does. That study's per-line rule selected a basket
# (`1X2 + O/U 1.5`) that finished LAST of every configuration tested out of sample —
# worse than betting 1X2 alone. The forensic table was right about why each line
# paid and wrong about which basket to hold, because Kelly re-solves over whatever
# remains and removing a line is not subtracting its P&L. **The adjudicator is a
# re-simulation, and this file is it.**
#
# ------------------------------------------------------------------------------
# WHAT IS FITTED ON WHAT
# ------------------------------------------------------------------------------
#
# Every candidate here is specified from the SELECTION window only:
#
#   * the baskets are named a priori, except `B4_is_keep`, which is read from
#     r07/1's `verdict_selection` column — the selection window's verdict, never the
#     Gate-2 column, which has seen both windows;
#   * the tuned `(t1, ratio, lambda, cap)` is r07/2's HONEST frontier pick — best
#     selection-window return inside the budget — not its in-sample-optimal one.
#
# The evaluation window is then scored once and reported. Anything in this file that
# was chosen with sight of the evaluation window is labelled `NOT DEPLOYABLE` in the
# table it appears in.
#
# ------------------------------------------------------------------------------
# DEPENDENCIES
# ------------------------------------------------------------------------------
#
# `results/market_pruning_audit_calibrated.csv` from r07/1. It is read rather than
# recomputed so that the basket this runner deploys is provably the same object the
# forensic report published.
#
# ------------------------------------------------------------------------------
# DATABASE BOUNDARY
# ------------------------------------------------------------------------------
#
# READS `mcmc_experiments` and `betdb`. WRITES NEITHER.
#
# Run on `mcmc-beast`, AFTER r07/1 and r07/2:
#
#     julia --project -t 16
#     julia> include("experiments/scottish_lower/07_calibrated_portfolio_and_trust_vector/r07_optimal_portfolio_comparison.jl")
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

const R73_AUDIT_PATH = joinpath(L07_OUT, "market_pruning_audit_calibrated.csv")
const R73_FRONTIER_PATH = joinpath(L07_OUT, "trust_lambda_frontier.csv")

"""
    R73_GATE3

The work package's §6 Gate 3, on the OUT-OF-SAMPLE window: annual Sharpe at or above
1.65, Calmar at or above 8.0, and max drawdown no worse than -18.0%.

Transcribed here so a failure is visible in this runner's own output rather than in
a second window. For context on how demanding it is: the best out-of-sample Sharpe
recorded anywhere in `calibration_generative_eda/README.md` is 1.351, over the same
50 slates.
"""
const R73_GATE3 = (sharpe = 1.65, calmar = 8.0, mdd = -18.0)

"Bootstrap resamples for the recommended rows only. Clustered by match, per `BootstrapCI`."
const R73_BOOTSTRAP_B = 4000

println("=" ^ 134)
println(" r07/3 — OPTIMAL PORTFOLIO COMPARISON")
println("=" ^ 134)


# %%
# ===================================================================
# 3. Data
# ===================================================================

r73_ctx = (@isdefined(L07_CTX) && L07_CTX !== nothing) ? L07_CTX : l07_load_context()
global L07_CTX = r73_ctx

isfile(R73_AUDIT_PATH) || error("""
    r07/3 needs r07/1's pruning audit at
      $(R73_AUDIT_PATH)
    Run `r07_line_forensics_calibrated.jl` first. The basket is READ rather than
    recomputed so the policy deployed here is provably the object the forensic
    report published.""")
r73_audit = CSV.read(R73_AUDIT_PATH, DataFrame)

r73_tuned = if isfile(R73_FRONTIER_PATH)
    CSV.read(R73_FRONTIER_PATH, DataFrame)
else
    @warn "r07/2's frontier not found at $R73_FRONTIER_PATH; the tuned arm falls back to (1.00, 1.4, 8.0, 0.25)."
    DataFrame()
end


# %%
# ===================================================================
# 4. The baskets
# ===================================================================
#
# A basket is a tier ASSIGNMENT — which directions are staked, and at which tier —
# and it is held apart from the LADDER (`t1`, `ratio`) that turns tiers into weights.
# That separation is what lets §7 attribute a gain to "we staked a different set of
# lines" rather than to "we staked harder", which are different recommendations.

"Map a `(market_key, direction)` pair from the audit CSV onto a `TieredTrust` key."
function r73_key(market_key::AbstractString, direction::AbstractString)
    market_key == "1X2" && return ("1x2", 0.0, Symbol(direction))
    market_key == "BTTS" && return ("btts", 0.0, Symbol(direction))
    startswith(market_key, "OU") || return nothing
    line = parse(Float64, market_key[3:end])
    side = startswith(direction, "over") ? :over : :under
    return ("over_under", line, side)
end

"""
    r73_is_keep_basket(audit, container) -> Dict

The data-driven basket: every direction whose SELECTION-window verdict is `KEEP` on
the POOLED scope for this container, split into two tiers at the median of their
selection-window Kelly ROI.

Only `verdict_selection` is read. The `oos_gate` column in the same file has seen
both windows and using it here would make the out-of-sample comparison in §7 a
comparison against itself.
"""
function r73_is_keep_basket(audit::DataFrame, container::AbstractString)
    rows = filter(r -> r.container == container && r.scope == "POOLED" &&
                       r.direction != "ALL" && r.verdict_selection == "KEEP", audit)
    nrow(rows) == 0 && return Dict{Tuple{String,Float64,Symbol},Int}()
    cut = median(rows.sel_kelly_roi)
    basket = Dict{Tuple{String,Float64,Symbol},Int}()
    for r in eachrow(rows)
        k = r73_key(r.market_key, r.direction)
        k === nothing && continue
        basket[k] = r.sel_kelly_roi >= cut ? 1 : 2
    end
    return basket
end

"""
    r73_baskets(audit, container) -> Vector{NamedTuple}

Seven candidates. Each one is here to answer a specific question, and the reason is
recorded beside it because a basket without a hypothesis is a fishing expedition.
"""
function r73_baskets(audit::DataFrame, container::AbstractString)
    canonical = L07_CANONICAL_TIERS
    draw_up = Dict(L07_KEY_HOME => 1, L07_KEY_UNDER_25 => 1, L07_KEY_DRAW => 1,
                   L07_KEY_AWAY => 2)
    plus_over15 = merge(canonical, Dict(L07_KEY_OVER_15 => 2))
    plus_over25 = merge(canonical, Dict(L07_KEY_OVER_25 => 2))
    minus_away = Dict(L07_KEY_HOME => 1, L07_KEY_UNDER_25 => 1, L07_KEY_DRAW => 2)
    return [
        (key = "B0_canonical", tiers = canonical, book = :tradeable, flat = false,
         deployable = true,
         why = "the production policy — CanonicalScottishLowerTrust"),
        (key = "B1_draw_promoted", tiers = draw_up, book = :tradeable, flat = false,
         deployable = true,
         why = "r07/1, SELECTION window only: the Draw is the direction calibration " *
               "improves most (raw -2.43% -> inv +28.79% Kelly ROI, a +31pt move " *
               "against +9 on Home and +2 on Away)"),
        (key = "B2_plus_over15", tiers = plus_over15, book = :tradeable, flat = false,
         deployable = true,
         why = "r07/1, SELECTION window only: Over 1.5 has the highest Kelly ROI of any " *
               "gated direction (+29 to +34%), though on 23-32 bets — below the floor"),
        (key = "B3_plus_over25", tiers = plus_over25, book = :tradeable, flat = false,
         deployable = true,
         why = "the work package's premise, stated a priori — does un-gating Over 2.5 pay?"),
        (key = "B4_minus_away", tiers = minus_away, book = :tradeable, flat = false,
         deployable = false,
         why = "NOT DEPLOYABLE — named from the sign REVERSAL across the split, which " *
               "reads the evaluation window. On the selection window Away scores +25 " *
               "to +27% Kelly ROI and verdict KEEP, so no rule fitted before the split " *
               "would have dropped it. Carried as the measurement of what dropping it " *
               "is worth, never as a recommendation"),
        (key = "B5_is_keep", tiers = r73_is_keep_basket(audit, container),
         book = :tradeable, flat = false, deployable = true,
         why = "data-driven — every selection-window KEEP, tiered at their median"),
        (key = "B6_flat11", tiers = Dict{Tuple{String,Float64,Symbol},Int}(),
         book = :tradeable, flat = true, deployable = true,
         why = "the unpruned control — flat trust over all 11 tradeable directions"),
        (key = "B7_x1x2_ou25", tiers = Dict{Tuple{String,Float64,Symbol},Int}(),
         book = :x1x2_ou25, flat = true, deployable = true,
         why = "MARKET_LINE_EDA_REPORT §5's recommendation, on a two-market book"),
    ]
end


"The two market sets the baskets need."
r73_market_set(::Val{:tradeable}) = L07_CAL.l2_tradeable_markets()
r73_market_set(::Val{:x1x2_ou25}) =
    Data.AbstractMarket[Data.Market1X2(), Data.MarketOverUnder(2.5)]


# %%
# ===================================================================
# 5. The ladders — production, and r07/2's honest pick
# ===================================================================

"""
    r73_ladders(tuned, container, model) -> Vector{NamedTuple}

Two settings per basket:

  `prod`   `t1 = 0.35`, `ratio = 1.4`, `lambda = 23`, `cap = 0.25` — what the
           MatchDay consoles run today.
  `matched`  r07/2's honest frontier pick at a budget of -16.15% — the deployed
             policy's own full-period drawdown. This is the work package's
             matched-drawdown question (§1.4, H3) as a deployable setting: spend
             exactly the risk production already spends, and no more.
  `tuned`    the same at the -18.0% budget, which is Gate 3's limit.

Both frontier picks are chosen on the SELECTION window's return subject to the
SELECTION window's drawdown; neither reads the evaluation window.

A basket is judged at all three, because a basket change and a ladder change are
separate recommendations and bundling them would make it impossible to say which one
paid.
"""
function r73_ladder_from(tuned::DataFrame, container, model, budget, name)
    hit = nrow(tuned) == 0 ? DataFrame() :
          filter(r -> r.mode == "honest" && r.budget == budget &&
                      r.container == container && r.model == model, tuned)
    nrow(hit) == 0 &&
        return (setting = name, t1 = 1.00, ratio = 1.4, lambda = 8.0, cap = 0.25)
    h = first(hit)
    t1, ratio = if startswith(h.trust, "flat_")
        (parse(Float64, h.trust[6:end]), 1.0)
    else
        parts = split(h.trust, "_")
        (parse(Float64, parts[2]), parse(Float64, parts[3][2:end]))
    end
    return (setting = name, t1 = t1, ratio = ratio, lambda = h.lambda, cap = h.cap)
end

function r73_ladders(tuned::DataFrame, container::AbstractString, model::AbstractString)
    return [
        (setting = "prod", t1 = 0.35, ratio = 1.4, lambda = L07_LAMBDA, cap = L07_CAP),
        r73_ladder_from(tuned, container, model, -16.15, "matched"),
        r73_ladder_from(tuned, container, model, -18.0, "tuned"),
    ]
end


# %%
# ===================================================================
# 6. The comparison
# ===================================================================

println("\n6. RUNNING THE COMPARISON")
println("-" ^ 134)

r73_book_cache = Dict{Tuple{String,String,Symbol},Any}()
r73_rows = NamedTuple[]

for m in L07_MODELS, (tag, cal) in l07_calibrators()
    cf = l07_container(r73_ctx, m.key, cal)
    ladders = r73_ladders(r73_tuned, tag, m.key)
    for b in r73_baskets(r73_audit, tag)
        # A basket with no assigned direction would stake nothing; report it as such
        # rather than producing an empty portfolio that looks like a zero-return
        # strategy.
        if !b.flat && isempty(b.tiers)
            push!(r73_rows, (; model = m.key, container = tag, basket = b.key,
                             setting = "n/a", t1 = NaN, ratio = NaN, lambda = NaN,
                             cap = NaN, deployable = b.deployable, n_directions = 0, n_slates = 0, n_bets = 0,
                             return_pct = NaN, cagr_pct = NaN, flat_roi_pct = NaN,
                             sharpe_ann = NaN, sortino = NaN, mdd_pct = NaN,
                             calmar = NaN, ulcer = NaN, turnover = NaN,
                             mean_exposure = NaN, max_exposure = NaN,
                             worst_slate = NaN, mean_k_risk = NaN, frac_k_pinned = NaN,
                             n_capped = 0, is_return_pct = NaN, is_sharpe = NaN,
                             is_mdd_pct = NaN, is_calmar = NaN, is_slates = 0,
                             oos_return_pct = NaN, oos_cagr_pct = NaN, oos_sharpe = NaN,
                             oos_mdd_pct = NaN, oos_calmar = NaN, oos_slates = 0,
                             oos_k_pinned = NaN))
            continue
        end

        ck = (m.key, tag, b.book)
        entry = get!(r73_book_cache, ck) do
            spec = l07_book_spec(r73_market_set(Val(b.book)))
            books, br = L07_PF.build_books_reported(spec, cf, r73_ctx.book, r73_ctx.ds;
                                                    quiet = true)
            (; books, br, slates = L07_PF.group(DailySlate(), books))
        end

        for lad in ladders
            trust = b.flat ? FlatTrust(min(1.0, lad.t1)) :
                    l07_tiered(b.tiers, lad.t1, lad.ratio)
            policy = l07_policy(trust; lambda = lad.lambda, cap = lad.cap)
            result = L07_PF.simulate_portfolio(policy, entry.slates;
                                               converged = entry.br.converged,
                                               failed_gates = entry.br.failed_gates,
                                               bootstrap = false)
            push!(r73_rows, l07_summary_row(result; model = m.key, container = tag,
                                            basket = b.key, setting = lad.setting,
                                            t1 = lad.t1, ratio = lad.ratio,
                                            lambda = lad.lambda, cap = lad.cap,
                                            deployable = b.deployable,
                                            n_directions = b.flat ?
                                                sum(length(Predictions.market_keys(x))
                                                    for x in r73_market_set(Val(b.book))) :
                                                length(b.tiers)))
        end
    end
    @printf("  %-5s %-9s done\n", m.key, tag)
end
r73_compare = DataFrame(r73_rows)
println("-" ^ 134)


# %%
# ===================================================================
# 7. The basket question, at the production ladder
# ===================================================================
#
# Holding `t1 = 0.35, ratio = 1.4, lambda = 23, cap = 0.25` fixed, so the only thing
# that moves is WHICH directions are staked.

function r73_print(df::DataFrame, title::AbstractString)
    println("\n", title)
    println("=" ^ 148)
    @printf("  %-5s %-9s %-17s %-6s %6s %10s %8s %8s %8s | %11s %10s %9s %9s\n",
            "model", "container", "basket", "set", "bets", "return %", "Sharpe",
            "MDD %", "Calmar", "OOS ret %", "OOS Sharpe", "OOS MDD", "OOS Cal")
    println("  ", "-" ^ 144)
    last_key = ""
    for r in eachrow(df)
        key = string(r.model, "/", r.container)
        key == last_key || (println("  ", "-" ^ 144); last_key = key)
        @printf("  %-5s %-9s %-17s %-6s %6d %+10.2f %8.3f %8.2f %8.2f | %+11.2f %10.3f %9.2f %9.2f%s\n",
                r.model, r.container, r.basket, r.setting, r.n_bets, r.return_pct,
                r.sharpe_ann, r.mdd_pct, r.calmar, r.oos_return_pct, r.oos_sharpe,
                r.oos_mdd_pct, r.oos_calmar, r.deployable ? "" : "  [not deployable]")
    end
    println("=" ^ 148)
end

r73_print(sort(filter(r -> r.setting == "prod", r73_compare),
               [:model, :container, :basket]),
          "7. THE BASKET, AT THE PRODUCTION LADDER (t1 0.35, ratio 1.4, λ 23, cap 0.25)")


# %%
# ===================================================================
# 8. The same baskets at r07/2's tuned ladder
# ===================================================================

r73_print(sort(filter(r -> r.setting == "matched", r73_compare),
               [:model, :container, :basket]),
          "8. THE SAME BASKETS AT THE MATCHED-DRAWDOWN LADDER (-16.15% budget, selection-window pick)")

r73_print(sort(filter(r -> r.setting == "tuned", r73_compare),
               [:model, :container, :basket]),
          "8.1 THE SAME BASKETS AT THE -18% LADDER (selection-window pick)")

# The deployable leaderboard: mean across models, so a candidate that only works on
# one of them cannot top it.
println("\n8.2 DEPLOYABLE LEADERBOARD — mean over both models, ranked on OUT-OF-SAMPLE Sharpe")
println("=" ^ 134)
r73_board = let d = filter(r -> r.deployable && isfinite(r.oos_sharpe), r73_compare)
    g = combine(groupby(d, [:container, :basket, :setting]),
                :n_bets => mean => :bets, :return_pct => mean => :ret,
                :sharpe_ann => mean => :sharpe, :mdd_pct => mean => :mdd,
                :is_return_pct => mean => :is_ret, :is_mdd_pct => mean => :is_mdd,
                :oos_return_pct => mean => :oos_ret, :oos_sharpe => mean => :oos_sharpe,
                :oos_mdd_pct => mean => :oos_mdd, :oos_calmar => mean => :oos_calmar,
                nrow => :n_models)
    sort(filter(r -> r.n_models == length(L07_MODELS), g), :oos_sharpe; rev = true)
end
@printf("  %-9s %-17s %-8s %7s %10s %8s %8s | %10s %8s | %11s %10s %9s %8s\n",
        "container", "basket", "ladder", "bets", "return %", "Sharpe", "MDD %",
        "IS ret %", "IS MDD", "OOS ret %", "OOS Sharpe", "OOS MDD", "OOS Cal")
println("  ", "-" ^ 130)
for r in eachrow(first(r73_board, 20))
    @printf("  %-9s %-17s %-8s %7.0f %+10.2f %8.3f %8.2f | %+10.2f %8.2f | %+11.2f %10.3f %9.2f %8.2f\n",
            r.container, r.basket, r.setting, r.bets, r.ret, r.sharpe, r.mdd,
            r.is_ret, r.is_mdd, r.oos_ret, r.oos_sharpe, r.oos_mdd, r.oos_calmar)
end
println("""
  Ranked on an OUT-OF-SAMPLE column, so this table is a REPORT and not a selection
  rule — nothing in §11 is chosen from it. It is here because a reader deciding what
  to deploy needs to see the whole deployable surface, not only the three cells three
  in-sample criteria happened to pick.""")
println("=" ^ 134)


# %%
# ===================================================================
# 9. Attribution — what the container, the basket and the ladder are each worth
# ===================================================================
#
# Three one-factor moves off the deployed policy (`raw`, `B0_canonical`, `prod`), each
# holding the other two fixed, then the joint move. If the joint move is close to the
# sum of the parts the three are separable; if it is much larger, they interact and
# no single-factor recommendation is safe.
#
# The basket used here is `B2_plus_over15` — the only one §11.1's pairwise test finds
# winning in BOTH windows on every paired cell. The ladder is the matched-risk one, so
# the joint row spends the same drawdown budget the deployed policy already spends and
# the comparison is not a comparison of risk appetites.

println("\n9. ATTRIBUTION — one factor at a time, off today's deployed policy")
println("=" ^ 134)
r73_attrib = let
    rows = NamedTuple[]
    for m in L07_MODELS
        pick(c, b, s) = begin
            h = filter(r -> r.model == m.key && r.container == c && r.basket == b &&
                            r.setting == s, r73_compare)
            nrow(h) == 0 ? nothing : first(h)
        end
        base = pick("raw", "B0_canonical", "prod")
        base === nothing && continue
        for (label, r) in (
            ("deployed today", base),
            ("+ calibration (inv_anch)", pick("inv_anch", "B0_canonical", "prod")),
            ("+ matched-risk ladder", pick("raw", "B0_canonical", "matched")),
            ("+ Over 1.5 in the basket", pick("raw", "B2_plus_over15", "prod")),
            ("+ all three", pick("inv_anch", "B2_plus_over15", "matched")))
            r === nothing && continue
            push!(rows, (; model = m.key, move = label, container = r.container,
                         basket = r.basket, setting = r.setting,
                         return_pct = r.return_pct, d_return = r.return_pct - base.return_pct,
                         sharpe = r.sharpe_ann, mdd = r.mdd_pct, calmar = r.calmar,
                         oos_return = r.oos_return_pct,
                         d_oos = r.oos_return_pct - base.oos_return_pct,
                         oos_sharpe = r.oos_sharpe, oos_mdd = r.oos_mdd_pct))
        end
    end
    DataFrame(rows)
end

@printf("  %-5s %-26s %10s %10s %8s %8s | %11s %10s %10s %9s\n", "model", "move",
        "return %", "Δ return", "Sharpe", "MDD %", "OOS ret %", "Δ OOS", "OOS Sharpe",
        "OOS MDD")
println("  ", "-" ^ 128)
for r in eachrow(r73_attrib)
    @printf("  %-5s %-26s %+10.2f %+10.2f %8.3f %8.2f | %+11.2f %+10.2f %10.3f %9.2f\n",
            r.model, r.move, r.return_pct, r.d_return, r.sharpe, r.mdd,
            r.oos_return, r.d_oos, r.oos_sharpe, r.oos_mdd)
end
println("=" ^ 134)


# %%
# ===================================================================
# 10. GATE 3 — the work package's production benchmark
# ===================================================================
#
# OOS annual Sharpe >= 1.65, OOS Calmar >= 8.0, OOS max drawdown no worse than -18%.
# Applied to every candidate, so a pass is found rather than argued for, and a
# universal failure is reported as a universal failure.

r73_gate3 = let
    ok = filter(r -> isfinite(r.oos_sharpe) && isfinite(r.oos_calmar) &&
                     isfinite(r.oos_mdd_pct), r73_compare)
    d = copy(ok)
    d.g3_sharpe = d.oos_sharpe .>= R73_GATE3.sharpe
    d.g3_calmar = d.oos_calmar .>= R73_GATE3.calmar
    d.g3_mdd = d.oos_mdd_pct .>= R73_GATE3.mdd
    d.gate3 = d.g3_sharpe .& d.g3_calmar .& d.g3_mdd
    d
end

println("\n10. GATE 3 — out-of-sample annual Sharpe ≥ 1.65, Calmar ≥ 8.0, MDD ≥ −18.0%")
println("=" ^ 134)
@printf("  candidates scored : %d\n", nrow(r73_gate3))
@printf("  pass all three    : %d\n", count(r73_gate3.gate3))
@printf("  pass Sharpe       : %d   (best OOS Sharpe %.3f)\n", count(r73_gate3.g3_sharpe),
        maximum(r73_gate3.oos_sharpe))
@printf("  pass Calmar       : %d   (best OOS Calmar %.2f)\n", count(r73_gate3.g3_calmar),
        maximum(r73_gate3.oos_calmar))
@printf("  pass drawdown     : %d   (shallowest OOS MDD %.2f%%)\n", count(r73_gate3.g3_mdd),
        maximum(r73_gate3.oos_mdd_pct))
println()
if count(r73_gate3.gate3) == 0
    println("""  GATE 3 : FAIL — no candidate clears all three thresholds out of sample.
           The binding constraint is named above. Read it against the fact that the
           best out-of-sample Sharpe recorded anywhere in the calibration stream over
           these same 50 slates is 1.351 (README §8.10): the threshold was set from
           FULL-PERIOD figures, and a 50-slate window cannot resolve a Sharpe of 1.65
           even when the strategy has one.""")
else
    println("  GATE 3 : PASS — candidates clearing all three:")
    for r in eachrow(sort(filter(:gate3 => identity, r73_gate3), :oos_return_pct;
                          rev = true))
        @printf("    %-5s %-9s %-17s %-6s  OOS %+7.2f%%  Sharpe %.3f  Calmar %.2f  MDD %.2f%%%s\n",
                r.model, r.container, r.basket, r.setting, r.oos_return_pct,
                r.oos_sharpe, r.oos_calmar, r.oos_mdd_pct,
                r.deployable ? "" : "   [NOT DEPLOYABLE — basket read the evaluation window]")
    end
    if !any(filter(:gate3 => identity, r73_gate3).deployable)
        println("""
      Every one of them is a basket whose DEFINITION read the evaluation window, so
      Gate 3 is cleared by construction rather than earned. The honest statement is
      that no basket specifiable before the split clears all three thresholds; §11
      reports what the ones that are specifiable before the split actually do.""")
    end
end
println("=" ^ 134)

println("\n10.1 THE BEST CANDIDATE ON EACH GATE-3 AXIS, and the deployed policy beside it")
println("-" ^ 134)
r73_best = let
    rows = NamedTuple[]
    for (axis, col, rev) in (("OOS return", :oos_return_pct, true),
                             ("OOS Sharpe", :oos_sharpe, true),
                             ("OOS Calmar", :oos_calmar, true),
                             ("OOS drawdown", :oos_mdd_pct, true))
        d = filter(r -> isfinite(r[col]), r73_compare)
        isempty(d) && continue
        r = d[rev ? argmax(d[!, col]) : argmin(d[!, col]), :]
        push!(rows, (; axis, model = r.model, container = r.container, basket = r.basket,
                     setting = r.setting, return_pct = r.return_pct, sharpe = r.sharpe_ann,
                     mdd = r.mdd_pct, oos_return = r.oos_return_pct,
                     oos_sharpe = r.oos_sharpe, oos_mdd = r.oos_mdd_pct,
                     oos_calmar = r.oos_calmar))
    end
    for m in L07_MODELS
        h = filter(r -> r.model == m.key && r.container == "raw" &&
                        r.basket == "B0_canonical" && r.setting == "prod", r73_compare)
        nrow(h) == 0 && continue
        r = first(h)
        push!(rows, (; axis = "deployed today", model = r.model, container = r.container,
                     basket = r.basket, setting = r.setting, return_pct = r.return_pct,
                     sharpe = r.sharpe_ann, mdd = r.mdd_pct, oos_return = r.oos_return_pct,
                     oos_sharpe = r.oos_sharpe, oos_mdd = r.oos_mdd_pct,
                     oos_calmar = r.oos_calmar))
    end
    DataFrame(rows)
end
@printf("  %-15s %-5s %-9s %-17s %-6s %10s %8s %8s | %11s %10s %9s %9s\n", "axis",
        "model", "container", "basket", "set", "return %", "Sharpe", "MDD %",
        "OOS ret %", "OOS Sharpe", "OOS MDD", "OOS Cal")
println("  ", "-" ^ 130)
for r in eachrow(r73_best)
    @printf("  %-15s %-5s %-9s %-17s %-6s %+10.2f %8.3f %8.2f | %+11.2f %10.3f %9.2f %9.2f\n",
            r.axis, r.model, r.container, r.basket, r.setting, r.return_pct, r.sharpe,
            r.mdd, r.oos_return, r.oos_sharpe, r.oos_mdd, r.oos_calmar)
end
println("-" ^ 134)


# %%
# ===================================================================
# 11. The recommendation, with a clustered bootstrap on its ROI
# ===================================================================
#
# One policy, re-run with `bootstrap = true` so the recommendation carries an
# interval rather than a point. `BootstrapCI` resamples CLUSTERED BY MATCH: eleven
# selections on one fixture share one scoreline, and resampling individual bets would
# divide the standard error by roughly sqrt(11) and turn an interval that spans zero
# into one that does not.

"""
    r73_recommend(compare, criterion) -> NamedTuple

The deployable pick under one stated criterion. Eligible candidates are those whose
SELECTION-window drawdown stayed inside -18%, whose basket definition never read the
evaluation window (`deployable`), and which are available on BOTH models.

TWO criteria are reported, both predeclared, because they are different theories of
what a policy is for and they do not agree:

  `is_return`      the best mean SELECTION-window return. The natural reading of
                   "best policy", and the one a return-maximising operator would use.
  `is_calmar`      the best mean SELECTION-window return per unit of selection-window
                   drawdown.
  `is_sharpe_prod` the best mean SELECTION-window Sharpe among candidates that leave
                   the RISK LADDER at its production setting. The minimal-change
                   deployment — alter what is staked, not how hard — and the
                   in-sample analogue of Gate 3's primary metric.

Neither is tuned against the evaluation window. Reporting both, and reporting where
they disagree, is the point: a selection rule chosen after seeing which one won out
of sample would be `MARKET_LINE_EDA_REPORT` §5.1's mistake committed one level up,
on the rule instead of on the line.
"""
function r73_recommend(compare::DataFrame, criterion::Symbol)
    ok = filter(r -> r.deployable && isfinite(r.is_mdd_pct) && r.is_mdd_pct >= -18.0 &&
                     isfinite(r.is_return_pct), compare)
    isempty(ok) && return nothing
    d = copy(ok)
    d.is_calmar_own = d.is_return_pct ./ abs.(d.is_mdd_pct)
    g = combine(groupby(d, [:container, :basket, :setting]),
                :is_return_pct => mean => :is_return,
                :is_calmar_own => mean => :is_calmar,
                :is_sharpe => mean => :is_sharpe,
                :is_mdd_pct => minimum => :is_mdd,
                nrow => :n_models)
    g = filter(r -> r.n_models == length(L07_MODELS), g)
    isempty(g) && return nothing
    if criterion === :is_sharpe_prod
        g = filter(r -> r.setting == "prod", g)
        isempty(g) && return nothing
        return g[argmax(g.is_sharpe), :]
    end
    col = criterion === :is_return ? :is_return : :is_calmar
    return g[argmax(g[!, col]), :]
end

"""
    r73_score(pick) -> DataFrame

Re-run one chosen (container, basket, ladder) on both models with a clustered
bootstrap, so the recommendation carries an interval rather than a point.
"""
function r73_score(pick)
    rows = NamedTuple[]
    for m in L07_MODELS
        cal = only(c for (t, c) in l07_calibrators() if t == pick.container)
        cf = l07_container(r73_ctx, m.key, cal)
        b = only(x for x in r73_baskets(r73_audit, pick.container) if x.key == pick.basket)
        lad = only(x for x in r73_ladders(r73_tuned, pick.container, m.key)
                   if x.setting == pick.setting)
        spec = l07_book_spec(r73_market_set(Val(b.book)))
        books, br = L07_PF.build_books_reported(spec, cf, r73_ctx.book, r73_ctx.ds;
                                                quiet = true)
        trust = b.flat ? FlatTrust(min(1.0, lad.t1)) : l07_tiered(b.tiers, lad.t1, lad.ratio)
        result = L07_PF.simulate_portfolio(l07_policy(trust; lambda = lad.lambda,
                                                      cap = lad.cap),
                                           books, br; bootstrap = true, B = R73_BOOTSTRAP_B)
        ci = result.bootstrap_ci
        push!(rows, l07_summary_row(result; model = m.key, container = pick.container,
                                    basket = pick.basket, setting = lad.setting,
                                    t1 = lad.t1, ratio = lad.ratio, lambda = lad.lambda,
                                    cap = lad.cap,
                                    roi_lo = ci === nothing ? NaN : ci.roi_lo,
                                    roi_hi = ci === nothing ? NaN : ci.roi_hi,
                                    p_roi_positive = ci === nothing ? NaN : ci.p_roi_positive))
    end
    return DataFrame(rows)
end

println("\n11. THE RECOMMENDATION — three predeclared selection criteria")
println("=" ^ 134)
r73_final_frames = DataFrame[]
for criterion in (:is_return, :is_calmar, :is_sharpe_prod)
    pick = r73_recommend(r73_compare, criterion)
    if pick === nothing
        @printf("\n  criterion %-15s : no candidate satisfies the constraints on both models.\n",
                criterion)
        continue
    end
    @printf("\n  criterion %-15s : container %s | basket %s | ladder %s\n",
            criterion, pick.container, pick.basket, pick.setting)
    @printf("  selection window       : %+.2f%% return, %.2f%% drawdown, Calmar %.2f, Sharpe %.3f\n",
            pick.is_return, pick.is_mdd, pick.is_calmar, pick.is_sharpe)
    df = r73_score(pick)
    df.criterion = fill(String(criterion), nrow(df))
    push!(r73_final_frames, df)
    @printf("  %-5s %6s %10s %8s %8s %8s | %11s %10s %9s %8s | %s\n", "model", "bets",
            "return %", "Sharpe", "MDD %", "Calmar", "OOS ret %", "OOS Sharpe",
            "OOS MDD", "OOS Cal", "flat ROI 95% CI")
    println("  ", "-" ^ 136)
    for r in eachrow(df)
        @printf("  %-5s %6d %+10.2f %8.3f %8.2f %8.2f | %+11.2f %10.3f %9.2f %8.2f | [%+.2f, %+.2f] P(>0)=%.3f\n",
                r.model, r.n_bets, r.return_pct, r.sharpe_ann, r.mdd_pct, r.calmar,
                r.oos_return_pct, r.oos_sharpe, r.oos_mdd_pct, r.oos_calmar,
                r.roi_lo, r.roi_hi, r.p_roi_positive)
    end
    breached = filter(r -> isfinite(r.oos_mdd_pct) && r.oos_mdd_pct < -18.0, df)
    nrow(breached) == 0 ||
        @printf("  NOTE: out-of-sample drawdown breached the -18%% budget on %s (%s). A\n        selection-window drawdown constraint bounds the window it was fitted on and\n        nothing else; it is not a risk limit.\n",
                join(breached.model, ", "),
                join([@sprintf("%.2f%%", x) for x in breached.oos_mdd_pct], ", "))
end
r73_final = isempty(r73_final_frames) ? DataFrame() : vcat(r73_final_frames...)
println("=" ^ 134)


# %%
# ===================================================================
# 11.1 The pairwise test — each basket against the production basket
# ===================================================================
#
# §11's three criteria all take an ARGMAX over the whole field of baskets, ladders and
# containers, and they disagree with each other. That is the signature of a selection
# made on too little data: with 49 selection-window slates, the best of forty-eight
# candidates is mostly the luckiest.
#
# This section asks a narrower and far more answerable question, one comparison at a
# time: **against the production basket, holding the model, the container and the
# ladder fixed, does this change win?** Twenty-four paired cells per basket (2 models
# x 4 containers x 3 ladders), scored first on the selection window and then, quite
# separately, on the evaluation window.
#
# A basket that wins most of its paired cells in the selection window AND reproduces
# that in the evaluation window is a result. A basket that wins a lot of cells in one
# window and loses them in the other is a run of settlements, which is exactly what
# `MARKET_LINE_EDA_REPORT` §2.1 caught `OU1.5` and `OU3.5` doing on raw latents.

println("\n11.1 PAIRWISE — each deployable basket against B0_canonical, cell by cell")
println("=" ^ 134)
r73_pairwise = let
    base = Dict((r.model, r.container, r.setting) => r
                for r in eachrow(filter(r -> r.basket == "B0_canonical", r73_compare)))
    rows = NamedTuple[]
    for g in groupby(filter(r -> r.basket != "B0_canonical" && r.setting != "n/a",
                            r73_compare), :basket)
        wins_is = 0; wins_oos = 0; n = 0
        d_is = Float64[]; d_oos = Float64[]; d_sharpe = Float64[]
        for r in eachrow(g)
            b = get(base, (r.model, r.container, r.setting), nothing)
            b === nothing && continue
            (isfinite(r.is_return_pct) && isfinite(b.is_return_pct)) || continue
            n += 1
            r.is_return_pct > b.is_return_pct && (wins_is += 1)
            r.oos_return_pct > b.oos_return_pct && (wins_oos += 1)
            push!(d_is, r.is_return_pct - b.is_return_pct)
            push!(d_oos, r.oos_return_pct - b.oos_return_pct)
            push!(d_sharpe, r.oos_sharpe - b.oos_sharpe)
        end
        n == 0 && continue
        push!(rows, (; basket = first(g.basket), deployable = first(g.deployable), cells = n,
                     is_wins = wins_is, oos_wins = wins_oos,
                     mean_d_is = mean(d_is), mean_d_oos = mean(d_oos),
                     mean_d_oos_sharpe = mean(d_sharpe),
                     transfers = (wins_is > n / 2) == (wins_oos > n / 2)))
    end
    sort(DataFrame(rows), :mean_d_oos_sharpe; rev = true)
end

@printf("  %-17s %-6s %6s %9s %10s %12s %12s %16s %s\n", "basket", "depl.", "cells",
        "IS wins", "OOS wins", "mean Δ IS %", "mean Δ OOS %", "mean Δ OOS Sharpe",
        "verdict")
println("  ", "-" ^ 130)
for r in eachrow(r73_pairwise)
    @printf("  %-17s %-6s %6d %6d/%2d %7d/%2d %+12.2f %+12.2f %+16.3f %s\n",
            r.basket, r.deployable ? "yes" : "NO", r.cells, r.is_wins, r.cells,
            r.oos_wins, r.cells, r.mean_d_is, r.mean_d_oos, r.mean_d_oos_sharpe,
            r.transfers ? (r.is_wins > r.cells / 2 ? "wins in both windows" :
                           "loses in both windows") : "REVERSES across the split")
end
println("""
  `IS wins` is decided on slates up to 2025-05-03 and `OOS wins` on the 50 slates
  after it; the two columns never share an observation. A basket that carries a
  majority in both columns has survived the split on a single pre-registrable
  comparison rather than on being the argmax of a field of forty-eight.""")
println("=" ^ 134)


# %%
# ===================================================================
# 12. Artefacts
# ===================================================================

println("\n12. ARTEFACTS")
l07_write(r73_compare, "optimal_portfolio_comparison.csv")
l07_write(r73_attrib, "policy_attribution.csv")
l07_write(r73_pairwise, "basket_pairwise_vs_canonical.csv")
l07_write(r73_board, "deployable_leaderboard.csv")
l07_write(r73_gate3, "gate3_scoreboard.csv")
nrow(r73_final) == 0 || l07_write(r73_final, "recommended_policy.csv")

println()
println("=" ^ 134)
println(" R73_DONE  ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
println("=" ^ 134)
