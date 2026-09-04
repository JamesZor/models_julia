# ==============================================================================
# r05 — Variance preservation: does the pool's w² contraction cost anything?
# ==============================================================================
#
# ------------------------------------------------------------------------------
# THE QUESTION
# ------------------------------------------------------------------------------
#
# README §7.7 closed Phase 3 with an open item. The T−25 calibration halves ECE and
# earns +1.02% stake-weighted CLV, and it still compounds less than the raw model at
# a fixed risk budget, because the log-linear pool contracts posterior log-variance
# by w² and Fractional Kelly reads the whole predictive distribution. Two readings
# of that were on the table and they make opposite predictions:
#
#   H1  the contraction is an ARTEFACT of the pooling algebra. Move the location,
#       keep the width, and the compounding comes back at no cost in proper score.
#   H2  the contraction is LOAD-BEARING. Restoring width at a shifted location
#       re-inflates the Jensen tail term E[e^(−Λ)] ≥ e^(−E[Λ]) (`eda/README.md`
#       Discovery 2), manufacturing longshot mass the model then bets.
#   H3  the question is malformed. `SlateDrawdown` absorbs uniform stake changes
#       (`eda/README.md` Discovery 4), so the calibrated arm's unused drawdown
#       headroom can be spent with a risk knob, and no posterior needs touching.
#   H4  some asymmetric scheme beats all three.
#
# This runner settles it by making location and dispersion independently
# controllable (`l03_variance_schemes.jl`) and then scoring and staking seven
# dispersion transforms under TWO location laws, against the same T−25 book.
#
# ------------------------------------------------------------------------------
# THE ATTRIBUTION CONTRACT
# ------------------------------------------------------------------------------
#
# Every scheme shares:
#
#   * the fixture set (24/25 + 25/26, gate-restricted, as r03/r04),
#   * the T−25 book and the rates inverted from it,
#   * the WEIGHT LAW and therefore the pooled LOCATION c — `A_pool` reproduces
#     `l01.calibrate_latents` bit-for-bit-to-1e-9, checked by G-0 before anything
#     downstream is read,
#   * the book spec, `FixedCap(0.25)`, `DailySlate()` and 2% commission.
#
# Only the residual map M and the anchor κ move between rows. A difference in this
# table is therefore a difference in DISPERSION and nothing else — which is the one
# thing r01–r04 could not say, because there the weight moved the width and the
# location together.
#
# Two location laws are run, not one, because the effect size scales with how much
# variance the pool destroys: `std_w0.40_s0.15` retains ~70% (median w ≈ 0.84) and
# `inv_w0.25_s0.35` retains far less. A dispersion effect that is real should be
# larger under `inv`, and an artefact of one grid point should not be.
#
# ------------------------------------------------------------------------------
# WHY THE FRONTIER PANEL IS THE ARBITER
# ------------------------------------------------------------------------------
#
# A return quoted at a fixed λ compares two different amounts of risk taken, which
# is the exact confound r02 §6.2 and r04 §7 both had to work around. H1 and H3
# cannot be separated by any single-λ row: a variance-preserving scheme stakes
# bigger and therefore returns more AND draws down more, and so does raising the
# risk budget. Panel F sweeps λ for every scheme and reports the whole
# (drawdown, return) frontier, so the question becomes the only one that is
# well-posed — AT MATCHED DRAWDOWN, does any dispersion scheme buy return that the
# risk knob could not have bought more cheaply?
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
#   julia> include("current_development/calibration_generative_eda/r05_variance_experiments.jl")
#
#   R05_SMOKE=1  one model, three schemes, one location law, short λ ladder.
#
# Requires `results/r03_best_per_form_t25.csv`; run r03 first. The location laws are
# read from it rather than restated, so this runner cannot silently calibrate with a
# spec r03 did not nominate.
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
include(joinpath(@__DIR__, "l03_variance_schemes.jl"))

const R05_PF = BayesianFootball.Portfolio


# %%
# ===================================================================
# 2. Configuration
# ===================================================================

const R05_EXPERIMENT = "scottish_lower_joint_player_2426"
const R05_GATE_SEASONS = ["24/25", "25/26"]
const R05_SMOKE = get(ENV, "R05_SMOKE", "0") != "0"
const R05_MODELS = R05_SMOKE ? ["m12_joint_hybrid_synergy"] :
    ["m12_joint_hybrid_synergy", "m05_joint_production_wealth"]

const R05_AS_OF = -25.0
const R05_MAX_STALENESS = 90.0
const R05_SPLIT_DATE = Date(2025, 5, 3)
const R05_INVERSION = MarketInversionConfig()
const R05_N_BINS = 10
const R05_EDGE_SMALL = 0.02
const R05_EDGE_LARGE = 0.05
const R05_OUT = joinpath(@__DIR__, "results")

"""
The location laws, by r03's own nomination. `std` is the Phase 3 champion and the
one README §7 quotes; `inv` destroys far more posterior width and is here as the
dose-response arm — a real dispersion effect must be larger under it.
"""
const R05_FORMS = R05_SMOKE ? ["std"] : ["std", "inv"]

"""
The λ ladder. STOPS AT 8.0, and that is a measurement rather than a guess: a smoke
run over [23, 12, 6] returned bit-identical rows at 12 and 6 on every container, so
`SlateDrawdown` has stopped binding somewhere above 12 and `FixedCap(0.25)` and the
Kelly fraction are carrying the constraint from there down. Rungs below 8 would be
duplicate rows dressed as a sweep. T6 reports the saturation rather than hiding it,
because it is the boundary condition H3 has to live inside.
"""
const R05_RISK_LAMBDAS = R05_SMOKE ? [23.0, 12.0] :
    [23.0, 18.0, 15.0, 12.0, 10.0, 8.0]

"""
The Kelly ladder — the risk knob that is still live once λ has saturated.

`l01_book_spec` hard-codes 0.30. Scheme E of the work package is this knob and λ
turned together on the pool and the raw model; it is swept over EVERY container
here instead, because the arbiter in §9 is "best return at or inside a common
drawdown" and a knob offered to one arm and not another would decide that question
by construction.
"""
const R05_KELLY_FRACTIONS = R05_SMOKE ? [0.30, 0.50] : [0.30, 0.40, 0.50, 0.60]

"Directions scored. 13, not r03's 11 — the O/U 0.5 ladder IS the Jensen tail."
const R05_SCORED_MARKETS = l01_full_direction_markets()

"Directions staked. 11, unchanged from Phase 2/3. No O/U 0.5. See l01 §9."
const R05_STAKED_MARKETS = l01_tradeable_markets()

mkpath(R05_OUT)


# %%
# ===================================================================
# 3. The location laws r03 nominated
# ===================================================================

"""
    r05_location_laws() -> Dict{String, Dict{String, GenerativeCalibrationSpec}}

`model → form → spec`, read from `r03_best_per_form_t25.csv`.

Same discipline as r04: the specs are r03's nominations, not this file's opinion,
so the two cannot drift and this runner cannot calibrate with something that never
cleared Gate 1 at tradeable prices.
"""
function r05_location_laws()
    path = joinpath(R05_OUT, "r03_best_per_form_t25.csv")
    isfile(path) || error(
        "r05 needs $(path). Run r03_t25_book_and_calibration.jl first — the " *
        "location laws are r03's nominations and are not restated here.")
    df = CSV.read(path, DataFrame)
    out = Dict{String, Dict{String, GenerativeCalibrationSpec}}()
    for r in eachrow(df)
        method = Symbol(r.method)
        form = method === :inverse_gaussian ? "inv" :
               method === :standard_gaussian ? "std" : "sta"
        spec = GenerativeCalibrationSpec(
            method = method, w_base = Float64(r.w_base),
            sigma = method === :static_geometric ? 0.25 : Float64(r.sigma),
            w_max = Float64(r.w_max))
        get!(() -> Dict{String, GenerativeCalibrationSpec}(), out, String(r.model))[form] = spec
    end
    return out
end

r05_laws = r05_location_laws()
r05_schemes = R05_SMOKE ?
    filter(s -> s.id in ("A_pool", "B_full", "D_sup"), l03_schemes()) : l03_schemes()

println("\n" * "="^118)
println(" r05 · VARIANCE PRESERVATION AND DISPERSION TRANSFORMS AT T$(R05_AS_OF)")
println("="^118)
@printf("  started    : %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
@printf("  threads    : %d\n", Threads.nthreads())
R05_SMOKE && println("  MODE       : SMOKE")
println("\n  dispersion schemes under test:")
for s in r05_schemes
    @printf("    %-12s [%s]  %s\n", s.id, s.family, s.note)
end
println("\n  location laws (r03 nominations, held fixed across schemes):")
for name in R05_MODELS, form in R05_FORMS
    haskey(r05_laws, name) && haskey(r05_laws[name], form) || continue
    @printf("    %-30s %-4s %s\n", name, form, spec_label(r05_laws[name][form]))
end


# %%
# ===================================================================
# 4. Data, books, rates
# ===================================================================

ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)
db = PostgresStorage(R05_EXPERIMENT)

r05_gate_ids = Set{Int}(
    Int.(ds.matches.match_id[in.(ds.matches.season, Ref(R05_GATE_SEASONS))]))

close_book = l01_betfair_closing_odds(ds)
t25_full, t25_refusals = point_in_time_book(
    ds; config = PointInTimeBookConfig(as_of_minutes = R05_AS_OF,
                                       max_staleness_minutes = R05_MAX_STALENESS))
assert_book_as_of(t25_full, R05_AS_OF)

const R05_KEY = [:match_id, :market_name, :market_line, :selection]
r05_keys = innerjoin(unique(select(t25_full, R05_KEY)),
                     unique(select(close_book, R05_KEY)); on = R05_KEY)
t25_matched = sort!(innerjoin(t25_full, r05_keys; on = R05_KEY), R05_KEY)
assert_book_as_of(t25_matched, R05_AS_OF)

r05_drift = book_drift(t25_full, close_book)

println("\n--- books ---")
@printf("  T−25 (full)    : %6d rows, %5d fixtures  — staked on this\n",
        nrow(t25_full), length(unique(t25_full.match_id)))
@printf("  T−25 (matched) : %6d rows, %5d fixtures  — scored on this, as r03 was\n",
        nrow(t25_matched), length(unique(t25_matched.match_id)))
r05_ou05_ids = let ou05 = filter(r -> r.market_name == "OverUnder" &&
                                      r.market_line == 0.5, t25_matched)
    @printf("  O/U 0.5 rows in the scoring book: %d over %d fixtures\n",
            nrow(ou05), length(unique(ou05.match_id)))
    println("    (l02 requires both sides before de-vigging, so the r01 O/U 0.5")
    println("     defect of README §5.6 cannot occur in this book.)")
    let f = filter(r -> r.market_name == "OverUnder" && r.market_line == 0.5, t25_full)
        @printf("    the FULL T−25 book carries %d over %d — the shortfall is the close\n",
                nrow(f), length(unique(f.match_id)))
        println("     book's own one-sided O/U 0.5 quotes failing the matched-key join.")
    end
    Set{Int}(Int.(ou05.match_id))
end

r05_raw = Dict{String, Any}()
for name in R05_MODELS
    fit = load_fit(db, name)
    fit.diagnostics.passed ||
        @warn "$name did not pass strict convergence gating" gates=fit.diagnostics.failed_gates
    r05_raw[name] = restrict_latents(fit_latents(fit), r05_gate_ids)
end
r05_all_ids = sort!(collect(union((Set(latent_match_ids(r05_raw[n])) for n in R05_MODELS)...)))
r05_rates = invert_market_rates(t25_full; config = R05_INVERSION, match_ids = r05_all_ids)

@printf("\n  fixtures priced : %d;  market rates accepted on %d\n",
        length(r05_all_ids), count(f -> f.accepted, values(r05_rates)))


# %%
# ===================================================================
# 5. G-0 · the gate that keeps A_pool the production pool
# ===================================================================

println("\n--- G-0 · A_pool must reproduce l01.calibrate_latents ---")
for name in R05_MODELS, form in R05_FORMS
    haskey(r05_laws[name], form) || continue
    chk = assert_scheme_a_matches(r05_raw[name], r05_rates, r05_laws[name][form])
    @printf("  %-30s %-4s max relative departure %.2e (bound %.0e)  PASS\n",
            name, form, max(chk.max_rel_home, chk.max_rel_away), chk.rtol)
end
println("  Every row below is a difference against A_pool. If this gate had failed,")
println("  the baseline would not be the pool r01–r04 measured and no difference in")
println("  this file would mean what its column heading says.")


# %%
# ===================================================================
# 6. The containers
# ===================================================================

println("\n--- containers · dispersion applied at a frozen location ---")
r05_containers = Dict{String, Vector{NamedTuple}}()
r05_disp_rows = NamedTuple[]

for name in R05_MODELS
    arms = NamedTuple[(id = "raw", form = "none", scheme = "none",
                       spec = GenerativeCalibrationSpec(method = :static_geometric,
                                                        w_base = 1.0),
                       latents = r05_raw[name], disp = nothing)]
    for form in R05_FORMS
        haskey(r05_laws[name], form) || continue
        spec = r05_laws[name][form]
        for s in r05_schemes
            lat, diag = apply_dispersion(r05_raw[name], r05_rates, spec, s)
            summ = dispersion_summary(diag)
            push!(arms, (id = form * "_" * s.id, form = form, scheme = s.id,
                         spec = spec, latents = lat, disp = summ))
            push!(r05_disp_rows, merge((; model = name, form = form, scheme = s.id,
                                        spec = spec_label(spec), family = s.family,
                                        anchor = String(s.anchor)), summ))
        end
    end
    r05_containers[name] = arms
end

@printf(" %-30s | %-4s | %-12s | %7s | %9s | %9s | %9s | %10s\n",
        "Model", "form", "scheme", "med w", "ret side", "ret sup", "ret tot", "rate ratio")
println("-"^118)
for r in r05_disp_rows
    @printf(" %-30s | %-4s | %-12s | %7.3f | %9.3f | %9.3f | %9.3f | %10.4f\n",
            r.model, r.form, r.scheme, r.w_median, r.ret_side_median,
            r.ret_sup_median, r.ret_tot_median, r.rate_ratio_median)
end
println("  `ret *` is retained log-variance against the RAW posterior, in three")
println("  bases. `rate ratio` is the predictive rate against A_pool's — the")
println("  anchored schemes sit at 1.0000 by construction and their unanchored")
println("  twins do not, which is the whole of the H2 contrast.")

CSV.write(joinpath(R05_OUT, "r05_variance_dispersion.csv"), DataFrame(r05_disp_rows))


# %%
# ===================================================================
# 7. Gate 1 · proper scores and the Jensen tail audit
# ===================================================================

println("\n--- Gate 1 · proper scores on the T−25 book, and the tail audit ---")

r05_score_rows = NamedTuple[]
r05_family_frames = DataFrame[]
r05_jensen_rows = NamedTuple[]

t_scores = @elapsed for name in R05_MODELS
    arms = r05_containers[name]
    raw_arm = first(a for a in arms if a.id == "raw")

    actx = build_evaluation_context(raw_arm.latents, t25_matched, ds.matches,
                                    L01_EVAL.AbstractScoringRule[L01_EVAL.LogLoss()];
                                    markets = R05_SCORED_MARKETS, threaded = true)
    anchor = edge_anchor(actx)

    priced_ids = latent_match_ids(raw_arm.latents)
    realised = realised_totals(ds.matches, priced_ids)
    # `eda/README.md` Discovery 2 quotes a 3.36% realised goalless rate; this study's
    # full gated fixture set is a different scope, so the O/U 0.5-QUOTED subset — the
    # fixtures on which that ladder could actually have been traded — is measured
    # beside it rather than the two being quietly compared.
    quoted_ids = [m for m in priced_ids if m in r05_ou05_ids]
    realised_q = realised_totals(ds.matches, quoted_ids)
    @printf("\n  %s: realised goalless rate %.4f over %d priced fixtures\n",
            name, realised.under_05, realised.n)
    @printf("    and %.4f over the %d of them the O/U 0.5 ladder quotes\n",
            realised_q.under_05, realised_q.n)

    for a in arms
        row, fams = score_calibration(name, a.spec, a.latents, t25_matched, ds.matches;
                                      markets = R05_SCORED_MARKETS, n_bins = R05_N_BINS,
                                      anchor = anchor, edge_small = R05_EDGE_SMALL,
                                      edge_large = R05_EDGE_LARGE)
        jd = jensen_diagnostics(a.latents; label = a.id)
        js = jensen_summary(jd, realised; label = a.id)
        jq = jensen_summary(jd[in.(jd.match_id, Ref(r05_ou05_ids)), :], realised_q;
                            label = a.id)
        d = a.disp
        push!(r05_score_rows, merge(
            row,
            (; container = a.id, form = a.form, scheme = a.scheme,
               ret_side = d === nothing ? 1.0 : d.ret_side_median,
               ret_sup = d === nothing ? 1.0 : d.ret_sup_median,
               ret_tot = d === nothing ? 1.0 : d.ret_tot_median,
               rate_ratio = d === nothing ? NaN : d.rate_ratio_median),
            Base.structdiff(js, NamedTuple{(:scheme,)})))
        fams.spec .= a.id
        push!(r05_family_frames, fams)
        push!(r05_jensen_rows,
              merge((; model = name, container = a.id), js,
                    (; quoted_n = jq.n_fixtures,
                       quoted_p_under_05 = jq.p_under_05,
                       quoted_jensen_under_05 = jq.jensen_under_05,
                       quoted_realised_under_05 = jq.realised_under_05,
                       quoted_bias_under_05 = jq.bias_under_05)))
        @printf("  %-30s %-14s LL %.5f (mkt %.5f)  ECE %.4f  Brier %.5f  P(U0.5) %.4f (real %.4f)\n",
                name, a.id, row.head_logloss, row.head_market_logloss,
                row.head_ece, row.head_brier, js.p_under_05, js.realised_under_05)
    end
end
@printf("  scored in %s\n", Training.format_elapsed(t_scores))

r05_scores = DataFrame(r05_score_rows)
CSV.write(joinpath(R05_OUT, "r05_variance_scores.csv"), r05_scores)
CSV.write(joinpath(R05_OUT, "r05_variance_family_scores.csv"),
          vcat(r05_family_frames...; cols = :union))
CSV.write(joinpath(R05_OUT, "r05_variance_jensen.csv"), DataFrame(r05_jensen_rows))


# %%
# ===================================================================
# 8. Gate 2 · the portfolio at T−25, and the risk frontier
# ===================================================================
#
# ONE loop, not two. `build_books_reported` materialises a priced book per slate and
# is both the expensive step and the memory-heavy one, so every simulation that
# reads a given (container, Kelly fraction) — Panel P at the production budget, the
# out-of-sample window, the whole λ ladder — is run while those books are alive, and
# they are dropped before the next container is built. Caching them all instead
# would hold thirty book sets in memory at once for no gain.
#
#   Panel P  full T−25 book, λ 23.0, Kelly 0.30 — the production settings
#   Panel O  the same, restricted to slates after the split date
#   Panel F  the whole (λ, Kelly) risk surface — the frontier, and the arbiter.
#            Work-package Scheme E is the `raw` and `*_A_pool` part of this surface;
#            it is run on every container because §9 compares arms at a COMMON
#            drawdown, and a risk knob given to one arm and withheld from another
#            would settle that comparison by construction rather than by evidence.

const R05_FLAT = l01_policy_spec(FlatTrust(1.0))
const R05_CANON = l01_policy_spec(CanonicalScottishLowerTrust())

r05_trust_of(tname, λ) = tname == "flat_1.0" ?
    l01_policy_spec(FlatTrust(1.0); risk_lambda = λ) :
    l01_policy_spec(CanonicalScottishLowerTrust(); risk_lambda = λ)

function r05_row(model, arm, trust_name, panel, window, λ, kelly, s)
    return (model = String(model), container = String(arm.id), form = String(arm.form),
            scheme = String(arm.scheme), trust = String(trust_name),
            panel = String(panel), window = String(window),
            risk_lambda = Float64(λ), kelly_fraction = Float64(kelly),
            n_bets = s.n_bets, n_slates = s.n_slates,
            total_return_pct = s.total_return_pct, flat_roi_pct = s.roi,
            max_drawdown_pct = s.mdd, sharpe_ann = s.sharpe_ann,
            calmar = s.calmar, win_rate = s.win_rate,
            mean_exposure = s.mean_exposure)
end

r05_pf_rows = NamedTuple[]
r05_clv_rows = NamedTuple[]
r05_direction_frames = DataFrame[]

println("\n--- Gate 2 · Panels P / O / F / E on the full T−25 book ---")
@printf(" %-28s | %-14s | %-13s | %5s | %9s | %8s | %8s | %7s\n",
        "Model", "container", "trust", "bets", "return %", "MDD %", "Sharpe", "Calmar")
println("-"^118)

t_portfolio = @elapsed for name in R05_MODELS
    for arm in r05_containers[name]
        for kelly in R05_KELLY_FRACTIONS
            bspec = l03_book_spec(R05_STAKED_MARKETS; kelly_fraction = kelly)
            books, rep = R05_PF.build_books_reported(bspec, arm.latents, t25_full,
                                                     ds.matches; require_result = true,
                                                     quiet = true)
            ev = filter(b -> b.date > R05_SPLIT_DATE, books)

            for tname in ("flat_1.0", "canonical_P1")
                for λ in R05_RISK_LAMBDAS
                    policy = r05_trust_of(tname, λ)
                    result = R05_PF.simulate_portfolio(policy, books; bootstrap = false)
                    production = (λ == 23.0 && kelly == 0.30)
                    push!(r05_pf_rows,
                          r05_row(name, arm, tname, production ? "P" : "F", "full",
                                  λ, kelly, result.summary))

                    if production
                        clv = bet_clv(result.trajectory.bets, r05_drift)
                        push!(r05_clv_rows,
                              merge((; model = name, container = arm.id, form = arm.form,
                                     scheme = arm.scheme, trust = tname, panel = "P"),
                                    clv_summary(result.trajectory.bets, clv)))
                        d = direction_ledger(result)
                        if nrow(d) > 0
                            insertcols!(d, 1, :model => name, :container => arm.id,
                                        :form => arm.form, :scheme => arm.scheme,
                                        :trust => tname)
                            push!(r05_direction_frames, d)
                        end
                        @printf(" %-28s | %-14s | %-13s | %5d | %+9.2f | %8.2f | %8.3f | %7.3f\n",
                                name, arm.id, tname, result.summary.n_bets,
                                result.summary.total_return_pct, result.summary.mdd,
                                result.summary.sharpe_ann, result.summary.calmar)
                        if !isempty(ev)
                            es = R05_PF.simulate_portfolio(policy, ev; bootstrap = false).summary
                            push!(r05_pf_rows,
                                  r05_row(name, arm, tname, "O", "evaluation", λ, kelly, es))
                        end
                    end
                end
            end
            books = nothing
            ev = nothing
            GC.gc(false)
        end
    end
end
@printf("  Gate 2 complete in %s\n", Training.format_elapsed(t_portfolio))

r05_portfolio = DataFrame(r05_pf_rows)
CSV.write(joinpath(R05_OUT, "r05_variance_portfolio_summary.csv"), r05_portfolio)
CSV.write(joinpath(R05_OUT, "r05_variance_clv.csv"), DataFrame(r05_clv_rows))
isempty(r05_direction_frames) ||
    CSV.write(joinpath(R05_OUT, "r05_variance_direction_ledger.csv"),
              vcat(r05_direction_frames...; cols = :union))


# %%
# ===================================================================
# 9. The frontier, read at a common drawdown
# ===================================================================
#
# The comparison every earlier phase had to hedge, and the only well-posed form of
# the question. Each container has a whole (λ, Kelly) risk surface behind it; a
# return quoted at one point of that surface says nothing, because the arm beside
# it can be moved to the same risk. So:
#
#     for each arm, over its entire risk surface, take the BEST return among the
#     settings whose realised max drawdown is no deeper than the RAW arm's at the
#     production budget — and report which setting that was.
#
# That is the upper envelope, not an interpolation along one knob. It is the right
# object here because the surface is two-dimensional and NOT monotone in either
# knob alone: λ saturates (T6), after which only Kelly moves risk, and two settings
# can land on the same drawdown with different returns.
#
# An arm that cannot reach the reference drawdown at all is reported with the
# deepest drawdown its surface can produce, rather than extrapolated past the end
# of the sweep. Extrapolating a compounding curve off the end of a grid is how a
# mechanism demonstration becomes a fabricated number.

"""
    frontier_envelope(rows, target_mdd; tol) -> NamedTuple

Best return on a risk surface subject to a drawdown budget.

`tol` (0.25 percentage points) exists so a row sitting a rounding error outside the
budget is not excluded on a difference no one could act on. Drawdowns are negative
percentages throughout, so every comparison here is on the absolute value.
"""
function frontier_envelope(rows::AbstractDataFrame, target_mdd::Float64;
                           tol::Float64 = 0.25)
    nrow(rows) == 0 && return (; risk_lambda = NaN, kelly_fraction = NaN,
                               total_return_pct = NaN, sharpe_ann = NaN, calmar = NaN,
                               realised_mdd = NaN, deepest_mdd = NaN,
                               n_settings = 0, reached = false)
    t = abs(target_mdd)
    a = abs.(rows.max_drawdown_pct)
    deepest = maximum(a)
    ok = findall(<=(t + tol), a)
    if isempty(ok)                       # cannot get INSIDE the budget at all
        i = argmin(a)
        return (; risk_lambda = rows.risk_lambda[i], kelly_fraction = rows.kelly_fraction[i],
                total_return_pct = rows.total_return_pct[i], sharpe_ann = rows.sharpe_ann[i],
                calmar = rows.calmar[i], realised_mdd = rows.max_drawdown_pct[i],
                deepest_mdd = -deepest, n_settings = nrow(rows), reached = false)
    end
    # Ties are real: while `SlateDrawdown`'s k is still binding it absorbs a
    # uniform stake change exactly, so two Kelly fractions can return the same
    # number to the last bit. Break toward the SMALLER knob — the cheaper way of
    # buying a return that two settings buy equally.
    i = ok[first(sortperm(collect(zip(-rows.total_return_pct[ok],
                                      rows.kelly_fraction[ok],
                                      -rows.risk_lambda[ok]))))]
    return (; risk_lambda = rows.risk_lambda[i], kelly_fraction = rows.kelly_fraction[i],
            total_return_pct = rows.total_return_pct[i], sharpe_ann = rows.sharpe_ann[i],
            calmar = rows.calmar[i], realised_mdd = rows.max_drawdown_pct[i],
            deepest_mdd = -deepest, n_settings = nrow(rows),
            reached = deepest >= t - tol)
end

r05_frontier_rows = NamedTuple[]
for name in R05_MODELS, tname in ("flat_1.0", "canonical_P1")
    base = filter(r -> r.model == name && r.container == "raw" && r.trust == tname &&
                       r.window == "full" && r.kelly_fraction == 0.30 &&
                       r.risk_lambda == 23.0, r05_portfolio)
    nrow(base) == 0 && continue
    target = first(base).max_drawdown_pct
    for arm in r05_containers[name]
        rows = filter(r -> r.model == name && r.container == arm.id &&
                           r.trust == tname && r.window == "full", r05_portfolio)
        nrow(rows) == 0 && continue
        f = frontier_envelope(rows, target)
        push!(r05_frontier_rows,
              merge((; model = name, container = arm.id, form = arm.form,
                     scheme = arm.scheme, trust = tname, target_mdd = target), f))
    end
end
r05_frontier = DataFrame(r05_frontier_rows)
nrow(r05_frontier) > 0 &&
    CSV.write(joinpath(R05_OUT, "r05_variance_frontier.csv"), r05_frontier)


# %%
# ===================================================================
# 10. Headline tables
# ===================================================================

println("\n" * "="^140)
println(" T1 · GATE 1 — proper scores, and what the dispersion did to the tails")
println("="^140)
@printf(" %-28s | %-14s | %8s | %7s | %8s | %8s | %8s | %8s | %8s\n",
        "Model", "container", "LogLoss", "ECE", "Brier", "ret tot", "P(U0.5)", "bias U0.5",
        "bias O3.5")
println("-"^140)
for r in eachrow(r05_scores)
    @printf(" %-28s | %-14s | %8.5f | %7.4f | %8.5f | %8.3f | %8.4f | %+8.4f | %+8.4f\n",
            r.model, r.container, r.head_logloss, r.head_ece, r.head_brier,
            r.ret_tot, r.p_under_05, r.bias_under_05, r.bias_over_35)
end
println(" `bias` is mean predicted minus realised frequency over the same fixtures.")
println(" H2 predicts it grows with retained TOTALS dispersion on the under tail.")

println("\n" * "="^140)
println(" T2 · GATE 2 — the portfolio at the production risk budget (λ 23, Kelly 0.30)")
println("="^140)
@printf(" %-28s | %-14s | %-13s | %5s | %9s | %8s | %8s | %7s | %9s\n",
        "Model", "container", "trust", "bets", "return %", "MDD %", "Sharpe", "Calmar",
        "exposure")
println("-"^140)
for r in eachrow(filter(x -> x.panel == "P", r05_portfolio))
    @printf(" %-28s | %-14s | %-13s | %5d | %+9.2f | %8.2f | %8.3f | %7.3f | %9.4f\n",
            r.model, r.container, r.trust, r.n_bets, r.total_return_pct,
            r.max_drawdown_pct, r.sharpe_ann, r.calmar, r.mean_exposure)
end

if nrow(r05_frontier) > 0
    println("\n" * "="^140)
    println(" T3 · THE ARBITER — best return over the whole (λ, Kelly) surface, INSIDE the raw arm's drawdown")
    println("="^140)
    @printf(" %-28s | %-14s | %-13s | %8s | %5s | %9s | %8s | %7s | %8s | %s\n",
            "Model", "container", "trust", "λ", "Kelly", "return %", "MDD %",
            "Sharpe", "budget %", "note")
    println("-"^140)
    for r in eachrow(sort(r05_frontier, [:model, :trust, order(:total_return_pct, rev = true)]))
        @printf(" %-28s | %-14s | %-13s | %8.2f | %5.2f | %+9.2f | %8.2f | %7.3f | %8.2f | %s\n",
                r.model, r.container, r.trust, r.risk_lambda, r.kelly_fraction,
                r.total_return_pct, r.realised_mdd, r.sharpe_ann, r.target_mdd,
                r.reached ? "" : @sprintf("CANNOT REACH BUDGET (deepest %.2f%%)", r.deepest_mdd))
    end
    println(" Every arm is given the SAME drawdown budget and the whole risk surface to")
    println(" spend it with. A dispersion scheme earns its place here only by beating")
    println(" A_pool inside that budget — return it gains merely by staking bigger has")
    println(" already been offered to A_pool by λ and by the Kelly fraction.")
end

println("\n" * "="^140)
println(" T6 · THE RISK KNOB'S CEILING — max drawdown against λ, at each Kelly fraction")
println("="^140)
println(" H3 assumes drawdown headroom can be spent with the risk budget. It can, until")
println(" `SlateDrawdown` stops binding — after which λ moves nothing and only the Kelly")
println(" fraction is still a risk knob. This table is where that boundary is.")
let ref = filter(r -> r.window == "full" &&
                      (r.container == "raw" || endswith(r.container, "_A_pool")),
                 r05_portfolio)
    @printf("\n %-28s | %-14s | %-13s | %5s | %s\n", "Model", "container", "trust",
            "Kelly", join([@sprintf("%8.1f", λ) for λ in R05_RISK_LAMBDAS], " "))
    println("-"^140)
    for name in R05_MODELS, c in unique(ref.container),
        tname in ("flat_1.0", "canonical_P1"), k in R05_KELLY_FRACTIONS
        cells = String[]
        for λ in R05_RISK_LAMBDAS
            g = filter(r -> r.model == name && r.container == c && r.trust == tname &&
                            r.kelly_fraction == k && r.risk_lambda == λ, ref)
            push!(cells, nrow(g) == 0 ? "       ·" :
                  @sprintf("%8.2f", first(g).max_drawdown_pct))
        end
        all(==("       ·"), cells) && continue
        @printf(" %-28s | %-14s | %-13s | %5.2f | %s\n", name, c, tname, k,
                join(cells, " "))
    end
end
println(" Rows that repeat across λ are `SlateDrawdown`'s scaling k pinned at its")
println(" ceiling of 1.0 (`_bisect_k` returns at most 1 — the risk model can shrink")
println(" a stake vector and never lever it up). Rows that repeat across KELLY are")
println(" the opposite regime: k is still binding, and it absorbs a uniform stake")
println(" change exactly, which is `eda/README.md` Discovery 4 seen from the other")
println(" side. Only one of the two knobs is live at a time.")

println("\n" * "="^140)
println(" T4 · CLOSING-LINE VALUE (Panel P)")
println("="^140)
@printf(" %-28s | %-14s | %-13s | %5s | %10s | %12s | %10s\n",
        "Model", "container", "trust", "bets", "mean CLV%", "stake-wtd %", "% positive")
println("-"^140)
for r in eachrow(DataFrame(r05_clv_rows))
    @printf(" %-28s | %-14s | %-13s | %5d | %+10.3f | %+12.3f | %10.1f\n",
            r.model, r.container, r.trust, r.n_matched, r.mean_clv_pct,
            r.stake_weighted_clv_pct, r.pct_positive)
end

if !isempty(r05_direction_frames)
    println("\n" * "="^140)
    println(" T5 · OVER 2.5 ACROSS SCHEMES (Panel P)")
    println("="^140)
    all_dir = vcat(r05_direction_frames...; cols = :union)
    o25 = filter(r -> r.selection == :over_25, all_dir)
    @printf(" %-28s | %-14s | %-13s | %5s | %8s | %9s | %9s | %9s | %10s\n",
            "Model", "container", "trust", "bets", "win rate", "mean odds",
            "kelly ROI", "flat ROI", "cap share")
    println("-"^140)
    for r in eachrow(sort(o25, [:model, :trust, :container]))
        @printf(" %-28s | %-14s | %-13s | %5d | %8.3f | %9.3f | %+9.2f | %+9.2f | %10.2f\n",
                r.model, r.container, r.trust, r.n_bets, r.win_rate, r.mean_odds,
                r.kelly_roi, r.flat_roi, r.capital_share)
    end
    println(" `CanonicalScottishLowerTrust` gates Over 2.5 to zero, so a canonical row is")
    println(" absent by design rather than by a lack of edge; the flat rows are where")
    println(" README §7's Over 2.5 rescue is visible.")
end

println("\n" * "="^140)
println(" ARTEFACTS")
for f in ("r05_variance_dispersion.csv", "r05_variance_scores.csv",
          "r05_variance_family_scores.csv", "r05_variance_jensen.csv",
          "r05_variance_portfolio_summary.csv", "r05_variance_clv.csv",
          "r05_variance_direction_ledger.csv", "r05_variance_frontier.csv")
    p = joinpath(R05_OUT, f)
    isfile(p) && @printf("  %-42s %8d bytes\n", f, filesize(p))
end
@printf("\n  finished   : %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
println("="^140)
