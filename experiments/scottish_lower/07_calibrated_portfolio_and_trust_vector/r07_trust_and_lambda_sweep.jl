# ==============================================================================
# r07 (2/3) — The trust vector, the conviction ratio and the risk parameter
# ==============================================================================
#
# ------------------------------------------------------------------------------
# THE QUESTION
# ------------------------------------------------------------------------------
#
#   H2  `CanonicalScottishLowerTrust` puts tier 1 at 0.35 and tier 2 at 0.25. That
#       was fitted on RAW latents, where trust was doing duty as an ad-hoc guard
#       against inflated edges. On a calibrated container the edges are already
#       shrunk toward the market. Does 0.35 now under-bet, and what is the optimal
#       tau in [0.20, 1.00]?
#   H3  Calibration compresses max drawdown from -23.5% to -7.8%. A 20% risk budget
#       is then largely unspent. What does lambda in `SlateDrawdown(lambda)` buy on
#       the way down from 28 to 8, and where is the matched-drawdown frontier?
#   H4  Do the answers agree on `m12` and `m05`?
#
# ------------------------------------------------------------------------------
# THE TRAP THIS RUNNER IS BUILT AROUND
# ------------------------------------------------------------------------------
#
# `eda/MULTITIER_TRUST_REPORT.md` §2.1 established that while `SlateDrawdown`'s
# bisected `k` is below 1 the constraint absorbs a uniform rescale of the whole
# book EXACTLY: `(0.30, 0.15)` and `(0.50, 0.25)` produce bit-identical portfolios.
# In that regime H2 has no content — absolute trust is not a parameter, only the
# tier ratio is.
#
# `calibration_generative_eda/README.md` §8.8 established the other half: `k` can
# only SHRINK a stake vector, never lever it up, so once it pins at 1 the risk model
# stops doing anything and lambda goes inert while trust becomes the only live knob.
#
# So H2 and H3 are not two independent questions. They are one question about WHICH
# REGIME a container sits in, and the answer is expected to differ between raw and
# calibrated containers precisely because calibrated ones stake less. Every row of
# every table below therefore carries `mean_k_risk` and `frac_k_pinned`, and the
# runner asserts the scale-invariance identity directly (§5) rather than citing it.
#
# ------------------------------------------------------------------------------
# WHY THERE IS NO SEPARATE KELLY-FRACTION AXIS
# ------------------------------------------------------------------------------
#
# `FractionalKelly(f)` returns a CONSTANT `k_shrink = f` and `stake_slate` applies
# trust and `k_shrink` as two successive scalar multiplications. Flat trust `tau` at
# `FractionalKelly(0.30)` is therefore the same pre-risk vector as trust 1.0 at
# `FractionalKelly(0.30*tau)`, and §5 asserts that too. README §8.7's `(lambda,
# Kelly)` surface is thus a strict SUBSET of this runner's `(lambda, trust)` grid,
# reached at higher trust rather than at a higher Kelly fraction.
#
# ------------------------------------------------------------------------------
# DATABASE BOUNDARY
# ------------------------------------------------------------------------------
#
# READS `mcmc_experiments` and `betdb`. WRITES NEITHER.
#
# Run on `mcmc-beast`:
#
#     julia --project -t 16
#     julia> include("experiments/scottish_lower/07_calibrated_portfolio_and_trust_vector/r07_trust_and_lambda_sweep.jl")
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

"The production STAKING scope — 11 directions, no O/U 0.5 (`l2_tradeable_markets`)."
const R72_MARKETS = L07_CAL.l2_tradeable_markets()

"Drawdown budgets the frontier is read at. -18.0 is the work package's Gate 3 limit."
const R72_BUDGETS = [-16.15, -18.0, -20.0]

r72_grid = l07_trust_grid()

println("=" ^ 126)
println(" r07/2 — TRUST VECTOR, CONVICTION RATIO AND SLATE-DRAWDOWN SWEEP")
println("=" ^ 126)
@printf(" scope       : %d tradeable directions\n",
        sum(length(Predictions.market_keys(m)) for m in R72_MARKETS))
@printf(" trust grid  : %d flat levels + %d tiered (t1 x ratio) = %d vectors\n",
        length(L07_FLAT_TAUS), length(L07_TIER1_TAUS) * length(L07_RATIOS), length(r72_grid))
@printf(" lambda grid : %s\n", join(L07_LAMBDAS, ", "))
@printf(" cap grid    : %s\n", join(L07_CAPS, ", "))
@printf(" cells       : %d per (model, container), %d total\n",
        length(r72_grid) * length(L07_LAMBDAS) * length(L07_CAPS),
        length(r72_grid) * length(L07_LAMBDAS) * length(L07_CAPS) *
        length(L07_MODELS) * length(l07_calibrators()))
@printf(" split       : selection <= %s < evaluation\n", L07_SPLIT_DATE)
println("=" ^ 126)


# %%
# ===================================================================
# 3. Data
# ===================================================================

r72_ctx = (@isdefined(L07_CTX) && L07_CTX !== nothing) ? L07_CTX : l07_load_context()
global L07_CTX = r72_ctx


# %%
# ===================================================================
# 4. Books — built once per (model, container), then re-solved 368 times
# ===================================================================

r72_spec = l07_book_spec(R72_MARKETS)
r72_books = Dict{Tuple{String,String},Any}()

println("\n4. BOOK CONSTRUCTION")
println("-" ^ 126)
for m in L07_MODELS, (tag, cal) in l07_calibrators()
    cf = l07_container(r72_ctx, m.key, cal)
    books, br = L07_PF.build_books_reported(r72_spec, cf, r72_ctx.book, r72_ctx.ds;
                                            quiet = true)
    slates = L07_PF.group(DailySlate(), books)
    ws = L07_CAL.weight_summary(cf.rate_diagnostics)
    r72_books[(m.key, tag)] = (; books, br, slates, w_median = ws.w_median,
                               var_retention = ws.var_retention_median)
    @printf("%-5s %-9s %4d books  %3d slates  median w %.3f  var retained %.3f\n",
            m.key, tag, br.n_books, length(slates), ws.w_median, ws.var_retention_median)
end
println("-" ^ 126)


# %%
# ===================================================================
# 5. GATE 5 — the two identities this runner's interpretation rests on
# ===================================================================
#
# Neither is a new result; both are cited above, and citing a structural claim
# without checking it on THIS book is how a suite ends up reasoning from a fact that
# stopped being true. Both are asserted on `m12 raw` at the production lambda.
#
#   5a  SCALE INVARIANCE. Two tiered vectors with the same ratio and different levels
#       must give the identical portfolio while `k_risk < 1`.
#   5b  TRUST/KELLY EQUIVALENCE. `FlatTrust(0.5)` at `FractionalKelly(0.30)` must
#       equal `FlatTrust(1.0)` at `FractionalKelly(0.15)`, bet for bet.

println("\n5. GATE 5 — STRUCTURAL IDENTITIES")
println("-" ^ 126)

r72_g5 = let e = r72_books[("m12", "raw")]
    a = L07_PF.simulate_portfolio(l07_policy(l07_tiered(L07_CANONICAL_TIERS, 0.35, 1.4)),
                                  e.slates; bootstrap = false)
    b = L07_PF.simulate_portfolio(l07_policy(l07_tiered(L07_CANONICAL_TIERS, 0.70, 1.4)),
                                  e.slates; bootstrap = false)
    (; a, b, dret = abs(a.summary.total_return_pct - b.summary.total_return_pct),
     dbets = a.summary.n_bets - b.summary.n_bets,
     ka = a.summary.mean_k_risk, kb = b.summary.mean_k_risk)
end
@printf("5a scale invariance : t1=0.35 r=1.4 -> %+.6f%% (k %.4f) | t1=0.70 r=1.4 -> %+.6f%% (k %.4f)\n",
        r72_g5.a.summary.total_return_pct, r72_g5.ka,
        r72_g5.b.summary.total_return_pct, r72_g5.kb)
@printf("                      |delta return| = %.2e, delta bets = %d — %s\n",
        r72_g5.dret, r72_g5.dbets,
        (r72_g5.dret < 1e-6 && r72_g5.dbets == 0) ?
        "IDENTICAL, so absolute trust is inert here and only the ratio is a parameter" :
        "NOT identical — this container is NOT in the k<1 regime and absolute trust IS live")

r72_g5b = let m = first(L07_MODELS)
    cf = l07_container(r72_ctx, m.key, first(l07_calibrators())[2])
    half_spec = l07_book_spec(R72_MARKETS; kelly = 0.15)
    bk2, br2 = L07_PF.build_books_reported(half_spec, cf, r72_ctx.book, r72_ctx.ds;
                                           quiet = true)
    a = L07_PF.simulate_portfolio(l07_policy(FlatTrust(0.5)),
                                  r72_books[("m12", "raw")].slates; bootstrap = false)
    b = L07_PF.simulate_portfolio(l07_policy(FlatTrust(1.0)),
                                  L07_PF.group(DailySlate(), bk2); bootstrap = false)
    (; ra = a.summary.total_return_pct, rb = b.summary.total_return_pct,
     na = a.summary.n_bets, nb = b.summary.n_bets)
end
r72_g5a2 = let e = r72_books[("m12", "raw")]
    a = L07_PF.simulate_portfolio(l07_policy(l07_tiered(L07_CANONICAL_TIERS, 0.50, 1.4)),
                                  e.slates; bootstrap = false)
    b = L07_PF.simulate_portfolio(l07_policy(l07_tiered(L07_CANONICAL_TIERS, 1.00, 1.4)),
                                  e.slates; bootstrap = false)
    (; ra = a.summary.total_return_pct, rb = b.summary.total_return_pct,
     ka = a.summary.mean_k_risk, kb = b.summary.mean_k_risk,
     na = a.summary.n_bets, nb = b.summary.n_bets)
end
@printf("5a bis, above it    : t1=0.50 r=1.4 -> %+.6f%% (k %.4f) | t1=1.00 r=1.4 -> %+.6f%% (k %.4f)\n",
        r72_g5a2.ra, r72_g5a2.ka, r72_g5a2.rb, r72_g5a2.kb)
@printf("                      |delta return| = %.2e, delta bets = %d — %s\n",
        abs(r72_g5a2.ra - r72_g5a2.rb), r72_g5a2.na - r72_g5a2.nb,
        (abs(r72_g5a2.ra - r72_g5a2.rb) < 1e-6 && r72_g5a2.na == r72_g5a2.nb) ?
        "IDENTICAL — MULTITIER_TRUST_REPORT §2.1's scale invariance holds ABOVE the binding threshold" :
        "NOT identical — scale invariance does not hold even here")
println("""
  Read 5a and 5a-bis together. Scale invariance is not a property of the allocator,
  it is a property of the REGIME: it holds wherever `SlateDrawdown`'s k is strictly
  inside (0,1) and absorbs a uniform rescale, and it fails below the level at which
  the constraint starts to bind, because there k is pinned at 1 and there is nothing
  to absorb the rescale with. Where the canonical 0.35 falls relative to that
  threshold is exactly H2, and §7 reads it off.""")
@printf("5b trust==kelly     : trust 0.50 x Kelly 0.30 -> %+.6f%% (%d bets) | trust 1.00 x Kelly 0.15 -> %+.6f%% (%d bets)\n",
        r72_g5b.ra, r72_g5b.na, r72_g5b.rb, r72_g5b.nb)
@printf("                      |delta return| = %.2e — %s\n", abs(r72_g5b.ra - r72_g5b.rb),
        (abs(r72_g5b.ra - r72_g5b.rb) < 1e-6 && r72_g5b.na == r72_g5b.nb) ?
        "IDENTICAL, so the Kelly-fraction axis is subsumed by the trust axis" :
        "NOT identical — the two knobs are not interchangeable and this runner's grid is incomplete")
println("-" ^ 126)


# %%
# ===================================================================
# 6. The sweep
# ===================================================================

println("\n6. SWEEP")
println("-" ^ 126)
r72_frames = DataFrame[]
for m in L07_MODELS, (tag, _) in l07_calibrators()
    e = r72_books[(m.key, tag)]
    t0 = time()
    df = l07_sweep(e.slates, r72_grid; br = e.br, model = m.key, container = tag)
    df.w_median = fill(e.w_median, nrow(df))
    df.var_retention = fill(e.var_retention, nrow(df))
    push!(r72_frames, df)
    @printf("%-5s %-9s %5d cells in %6.1f s | best return %+8.2f%% | best Sharpe %5.3f | shallowest MDD %7.2f%%\n",
            m.key, tag, nrow(df), time() - t0, maximum(df.return_pct),
            maximum(filter(isfinite, df.sharpe_ann)), maximum(df.mdd_pct))
end
r72_sweep = vcat(r72_frames...)
println("-" ^ 126)


# %%
# ===================================================================
# 7. H2 — is absolute trust a parameter at all, and where does it stop being one?
# ===================================================================
#
# The regime map. At the production lambda and cap, and at the canonical 1.4 ratio,
# what happens as tier-1 trust rises from 0.30 to 1.00? If the container is in the
# `k < 1` regime every column is constant and H2's premise ("0.35 under-bets") is
# void; if `k` pins, return moves and H2 has content.

println("\n7. H2 — THE TRUST REGIME MAP (ratio 1.4, lambda 23, cap 0.25)")
println("=" ^ 126)
for m in L07_MODELS
    @printf("\n  model = %s\n", m.key)
    @printf("  %-9s %6s %7s %10s %9s %8s %9s %9s %11s %10s\n", "container", "t1",
            "bets", "return %", "Sharpe", "MDD %", "mean k", "k pinned", "OOS ret %",
            "OOS Sharpe")
    println("  ", "-" ^ 116)
    for (tag, _) in l07_calibrators()
        rows = sort(filter(r -> r.model == m.key && r.container == tag &&
                                r.trust_kind == "tiered" && r.ratio == 1.4 &&
                                r.lambda == 23.0 && r.cap == 0.25, r72_sweep), :t1)
        for r in eachrow(rows)
            @printf("  %-9s %6.2f %7d %+10.2f %9.3f %8.2f %9.3f %9.2f %+11.2f %10.3f\n",
                    tag, r.t1, r.n_bets, r.return_pct, r.sharpe_ann, r.mdd_pct,
                    r.mean_k_risk, r.frac_k_pinned, r.oos_return_pct, r.oos_sharpe)
        end
    end
end
println("=" ^ 126)


# %%
# ===================================================================
# 8. H3 — the lambda ladder and its ceiling
# ===================================================================
#
# README §8.8's table, recomputed. Max drawdown against lambda at two trust levels.
# The signature to look for: a row that is FLAT from some lambda downward has hit
# `k = 1` and the risk knob has stopped working; the deepest number in that row is
# the container's drawdown CEILING at that trust.

println("\n8. H3 — MAX DRAWDOWN AGAINST LAMBDA (tiered, ratio 1.4, cap 0.25)")
println("=" ^ 126)
for m in L07_MODELS
    @printf("\n  model = %s\n", m.key)
    @printf("  %-9s %6s %s\n", "container", "t1",
            join([@sprintf("%9s", "λ=" * string(Int(l))) for l in L07_LAMBDAS], ""))
    println("  ", "-" ^ 116)
    for (tag, _) in l07_calibrators(), t1 in (0.35, 1.00)
        cells = [filter(r -> r.model == m.key && r.container == tag &&
                             r.trust_kind == "tiered" && r.ratio == 1.4 &&
                             abs(r.t1 - t1) < 1e-9 && r.lambda == l && r.cap == 0.25,
                        r72_sweep) for l in L07_LAMBDAS]
        any(isempty, cells) && continue
        @printf("  %-9s %6.2f %s\n", tag, t1,
                join([@sprintf("%9.2f", first(c).mdd_pct) for c in cells], ""))
    end
end
println("""
  A flat tail means `SlateDrawdown`'s bisected k has pinned at 1 and lambda is inert
  from there down; the last distinct value in the row is that container's deepest
  attainable drawdown at that trust level. Note `t1 = 0.35` is the canonical level:
  where its row is flat, the production policy cannot spend its own risk budget no
  matter how lambda is set, and the only way to reach the budget is more trust.""")
println("=" ^ 126)


# %%
# ===================================================================
# 9. The efficient frontier — best return inside a common drawdown
# ===================================================================
#
# A return quoted at a fixed lambda compares two different amounts of risk taken, so
# the arbiter is the best return whose realised drawdown is no deeper than a stated
# budget. Two readings, and they are different claims:
#
#   `in-sample`      the best cell by FULL-PERIOD return inside the budget. This is a
#                    mechanism demonstration and carries the selection bias README
#                    §8.12 flagged: the return is read off the same slates the cell
#                    was chosen on.
#   `honest`         the cell chosen on the SELECTION window alone (best IS return
#                    subject to IS drawdown inside the budget), then scored on the
#                    EVALUATION window it never saw. This is the deployable claim.

function r72_frontier(df::DataFrame, budget::Real)
    ok = filter(r -> isfinite(r.mdd_pct) && r.mdd_pct >= budget, df)
    isempty(ok) && return nothing
    return ok[argmax(ok.return_pct), :]
end

function r72_frontier_honest(df::DataFrame, budget::Real)
    ok = filter(r -> isfinite(r.is_mdd_pct) && r.is_mdd_pct >= budget &&
                     isfinite(r.is_return_pct), df)
    isempty(ok) && return nothing
    return ok[argmax(ok.is_return_pct), :]
end

println("\n9. THE FRONTIER — best return inside a drawdown budget")
println("=" ^ 138)
r72_frontier_rows = NamedTuple[]
for budget in R72_BUDGETS
    @printf("\n  budget = %.2f%%\n", budget)
    @printf("  %-5s %-9s %-16s %6s %6s %7s %10s %8s %8s %8s | %11s %10s %9s %8s\n",
            "model", "container", "trust", "λ", "cap", "bets", "return %", "Sharpe",
            "MDD %", "Calmar", "OOS ret %", "OOS Sharpe", "OOS MDD", "OOS Cal")
    println("  ", "-" ^ 132)
    for m in L07_MODELS, (tag, _) in l07_calibrators()
        df = filter(r -> r.model == m.key && r.container == tag, r72_sweep)
        for (mode, f) in (("in-sample", r72_frontier), ("honest", r72_frontier_honest))
            r = f(df, budget)
            r === nothing && continue
            push!(r72_frontier_rows, (; budget, mode, model = m.key, container = tag,
                                      trust = r.trust_label, lambda = r.lambda, cap = r.cap,
                                      bets = r.n_bets, return_pct = r.return_pct,
                                      sharpe = r.sharpe_ann, mdd = r.mdd_pct,
                                      calmar = r.calmar,
                                      is_return = r.is_return_pct, is_mdd = r.is_mdd_pct,
                                      is_sharpe = r.is_sharpe,
                                      oos_return = r.oos_return_pct,
                                      oos_sharpe = r.oos_sharpe, oos_mdd = r.oos_mdd_pct,
                                      oos_calmar = r.oos_calmar,
                                      k_pinned = r.frac_k_pinned))
            mode == "in-sample" || continue
            @printf("  %-5s %-9s %-16s %6.1f %6.2f %7d %+10.2f %8.3f %8.2f %8.2f | %+11.2f %10.3f %9.2f %8.2f\n",
                    m.key, tag, r.trust_label, r.lambda, r.cap, r.n_bets, r.return_pct,
                    r.sharpe_ann, r.mdd_pct, r.calmar, r.oos_return_pct, r.oos_sharpe,
                    r.oos_mdd_pct, r.oos_calmar)
        end
    end
end
r72_frontier_df = DataFrame(r72_frontier_rows)
println("=" ^ 138)

println("\n9.1 THE HONEST FRONTIER — chosen on the selection window, scored out of sample")
println("=" ^ 138)
@printf("  %-8s %-5s %-9s %-16s %6s %6s | %10s %8s | %11s %10s %9s\n",
        "budget", "model", "container", "trust", "λ", "cap", "IS ret %", "IS MDD",
        "OOS ret %", "OOS Sharpe", "OOS MDD")
println("  ", "-" ^ 132)
for r in eachrow(sort(filter(:mode => ==("honest"), r72_frontier_df),
                      [:budget, :model, :container]))
    @printf("  %8.2f %-5s %-9s %-16s %6.1f %6.2f | %+10.2f %8.2f | %+11.2f %10.3f %9.2f\n",
            r.budget, r.model, r.container, r.trust, r.lambda, r.cap,
            r.is_return, r.is_mdd, r.oos_return, r.oos_sharpe, r.oos_mdd)
end
println("=" ^ 138)


# %%
# ===================================================================
# 10. H4 — does the optimum transfer between the two models?
# ===================================================================
#
# The strongest form of the question: take the cell each model would have chosen on
# its OWN selection window at the -18% budget, and score the OTHER model on that
# same cell. A policy that only works on the model it was fitted to is a fit to the
# model, not to the league.

println("\n10. H4 — CROSS-MODEL TRANSFER OF THE CHOSEN CELL (budget -18%)")
println("=" ^ 126)
r72_transfer = let budget = -18.0
    rows = NamedTuple[]
    for (tag, _) in l07_calibrators(), src in L07_MODELS
        chosen = r72_frontier_honest(
            filter(r -> r.model == src.key && r.container == tag, r72_sweep), budget)
        chosen === nothing && continue
        for dst in L07_MODELS
            hit = filter(r -> r.model == dst.key && r.container == tag &&
                              r.trust_label == chosen.trust_label &&
                              r.lambda == chosen.lambda && r.cap == chosen.cap, r72_sweep)
            isempty(hit) && continue
            h = first(hit)
            push!(rows, (; container = tag, fitted_on = src.key, scored_on = dst.key,
                         trust = chosen.trust_label, lambda = chosen.lambda, cap = chosen.cap,
                         return_pct = h.return_pct, sharpe = h.sharpe_ann, mdd = h.mdd_pct,
                         oos_return = h.oos_return_pct, oos_sharpe = h.oos_sharpe,
                         oos_mdd = h.oos_mdd_pct))
        end
    end
    DataFrame(rows)
end

@printf("  %-9s %-11s %-11s %-16s %6s %6s %10s %8s %8s | %11s %10s\n", "container",
        "fitted on", "scored on", "trust", "λ", "cap", "return %", "Sharpe", "MDD %",
        "OOS ret %", "OOS Sharpe")
println("  ", "-" ^ 120)
for r in eachrow(r72_transfer)
    @printf("  %-9s %-11s %-11s %-16s %6.1f %6.2f %+10.2f %8.3f %8.2f | %+11.2f %10.3f%s\n",
            r.container, r.fitted_on, r.scored_on, r.trust, r.lambda, r.cap,
            r.return_pct, r.sharpe, r.mdd, r.oos_return, r.oos_sharpe,
            r.fitted_on == r.scored_on ? "  (own)" : "")
end
println("=" ^ 126)


# %%
# ===================================================================
# 11. Artefacts
# ===================================================================

println("\n11. ARTEFACTS")
l07_write(r72_sweep, "trust_lambda_grid_sweep.csv")
l07_write(r72_frontier_df, "trust_lambda_frontier.csv")
l07_write(r72_transfer, "trust_lambda_cross_model_transfer.csv")

println()
println("=" ^ 126)
println(" R72_DONE  ", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
println("=" ^ 126)
