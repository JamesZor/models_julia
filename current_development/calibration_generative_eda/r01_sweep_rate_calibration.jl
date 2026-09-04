# ==============================================================================
# r01 — Generative rate calibration: diagnostic and proper-score sweep
# ==============================================================================
#
# ------------------------------------------------------------------------------
# WHAT THIS IS AND IS NOT
# ------------------------------------------------------------------------------
#
# This is a CALIBRATION and PROPER-SCORING experiment. It asks one question:
#
#   Over the Scottish Lower walk-forward posteriors, is there ANY (functional
#   form, w_base, σ) at which pooling the model's posterior goal rates with the
#   rates implied by the Betfair close improves the probabilities the staking
#   layer reads — and if so, is the improvement in the small-edge regime, the
#   large-edge regime, or neither?
#
# It is NOT a betting-allocation study. No bankroll, no Kelly, no trust vector,
# no drawdown budget appears anywhere below. A calibration that improves LogLoss
# and ECE can still lose money — the Ireland result did exactly that out of
# domain — so the portfolio question is asked separately, on the winners this
# runner nominates, in `r02_portfolio_direction_audit.jl`. Nothing here entitles
# anyone to a bankroll claim.
#
# The controls are in the grid, not beside it:
#
#   * `w_base = 1.00` in every functional form is the UNCALIBRATED model, priced
#     through the identical code path. It must reproduce the baseline row exactly;
#     Gate D asserts that it does.
#   * `:static_geometric` is a constant pool, so any advantage the two Gaussian
#     forms show over it is attributable to the Δ-dependence rather than to
#     blending with the market as such.
#   * `:standard_gaussian` is the textbook shrinkage direction. It is in the grid
#     because the Ireland conclusion — that the INVERSE direction is right — was
#     drawn from one league and has already failed one out-of-domain transfer.
#
# ------------------------------------------------------------------------------
# FILTRATION AND COMPARABILITY CONTRACT
# ------------------------------------------------------------------------------
#
# 1. POSTERIORS ARE READ, NEVER SAMPLED. `load_fit` returns the completed
#    walk-forward runs from `mcmc_experiments`; every λ draw scored below was
#    produced by a fold that could not see the fixture it prices.
#
# 2. THE CALIBRATING PRICE IS THE CLOSE, AND THAT IS A DELIBERATE ANACHRONISM.
#    The market rates are inverted from the Betfair close (TWA over [−20 min,
#    kick-off]) — the same snapshot experiment 06 scored against. A closing price
#    is not available at the moment a bet is struck, so a positive result here is
#    an upper bound on what a T−25 calibration could achieve, and r02 must not be
#    read as a tradeable P&L until the price snapshot is moved back. This is
#    stated here rather than in a footnote because it is the single easiest way
#    to over-read the whole stream.
#
# 3. THE FIXTURE SET IS PINNED TO THE PUBLISHED GATE. Gate 1 of the work package
#    quotes LogLoss 0.64337 and ECE 0.0100, measured over the 40-fold 24/25 +
#    25/26 study. The canonical runs have since been extended to 43 folds with the
#    26/27 August programme, so the container is restricted to `R01_GATE_SEASONS`
#    before anything is scored. The excluded fixtures are counted and named in the
#    G-A report; the full-scope baseline is printed beside the gate-scope one so
#    the exclusion is visible rather than silent.
#
# 4. THE SCORING SCOPE IS TWO SCOPES, KEPT APART. `head_*` columns are the
#    `Evaluation.DEFAULT_SCORED_MARKETS` scope (1X2 + O/U 2.5 + BTTS) the published
#    baseline was measured on — the only scope in which the Gate-1 thresholds mean
#    anything. `wide_*` columns are the 13-direction book (1X2, O/U 0.5/1.5/2.5/3.5,
#    BTTS) r02 audits. Both come from ONE pricing pass over the wide book; the
#    headline is that pass filtered to the narrow selections.
#
# 5. EDGE BUCKETS ARE ANCHORED ON THE RAW MODEL. Calibration shrinks edges toward
#    the book by construction, so self-anchored strata would move between grid
#    points and two rows would silently score different observations. Every
#    stratified LPD below uses the UNCALIBRATED edges to assign the bucket.
#
# ------------------------------------------------------------------------------
# PERSISTENCE CAVEAT
# ------------------------------------------------------------------------------
#
# Outputs go to `current_development/calibration_generative_eda/results/` as CSV.
# They are REPLACEABLE artefacts: re-running overwrites them. Nothing here writes
# to `mcmc_experiments` — no run, no portfolio, no config registration — and
# nothing opens `betdb.paper_runbook`. The live console on 8085 is untouched.
# `betdb` is read for odds and results only.
#
# ------------------------------------------------------------------------------
# USAGE
# ------------------------------------------------------------------------------
#
#   julia --project -t 16          # mcmc-beast; -t 8 on archpc
#   julia> include("current_development/calibration_generative_eda/r01_sweep_rate_calibration.jl")
#
#   R01_SMOKE=1 julia --project -t 16 ...   # 3-spec dry run, one model
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


# %%
# ===================================================================
# 2. Configuration
# ===================================================================

const R01_EXPERIMENT = "scottish_lower_joint_player_2426"

"Primary candidate first; the team-level control second."
const R01_MODELS = [
    "m12_joint_hybrid_synergy",
    "m05_joint_production_wealth",
]

"The seasons the published Gate-1 thresholds were measured on. See contract §3."
const R01_GATE_SEASONS = ["24/25", "25/26"]

# The 2D surface of the work package, §3.2.
const R01_W_BASES = [0.25, 0.40, 0.55, 0.70, 0.85, 1.00]
const R01_SIGMAS  = [0.15, 0.25, 0.35, 0.50, 0.75, 1.00]
const R01_METHODS = [:inverse_gaussian, :standard_gaussian, :static_geometric]
const R01_W_MAX   = 1.0

# Edge strata, as the work package defines them.
const R01_EDGE_SMALL = 0.02
const R01_EDGE_LARGE = 0.05

const R01_N_BINS   = 10
const R01_MAX_GOALS = 12

# Published experiment-06 numbers for `m12` over the 40-fold study. CONTEXT, not
# assertions: §9 recomputes the baseline on the fixture set actually loaded and
# judges the sweep against that. If the two disagree the recomputed number wins
# and the disagreement is reported.
const R01_PUBLISHED_M12_LOGLOSS = 0.64337
const R01_PUBLISHED_M12_ECE     = 0.0100
const R01_PUBLISHED_BF_ECE      = 0.0139
const R01_PUBLISHED_N_OBS       = 2_899

const R01_INVERSION = MarketInversionConfig(
    feature = BayesianFootball.Features.DoublePoissonMarketFeature(
        lines = L01_INVERSION_LINES),
    max_goals = 10,        # the inversion grid; the pricing grid is R01_MAX_GOALS
    min_targets = 3,       # a bare 1X2 identifies two rates; anything less does not
    max_sse = 5.0e-3,
    lambda_bounds = (0.05, 6.0),
)

const R01_SMOKE = get(ENV, "R01_SMOKE", "0") != "0"


# %%
# ===================================================================
# 3. Runtime and output directory
# ===================================================================

const R01_OUT = joinpath(@__DIR__, "results")
mkpath(R01_OUT)

println("\n" * "="^110)
println(" r01 · GENERATIVE RATE CALIBRATION SWEEP — Scottish Lower")
println("="^110)
@printf("  started      : %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
@printf("  threads      : %d\n", Threads.nthreads())
@printf("  experiment   : %s (read-only)\n", R01_EXPERIMENT)
@printf("  output       : %s\n", R01_OUT)
R01_SMOKE && println("  MODE         : SMOKE — a 3-spec dry run, not a result")


# %%
# ===================================================================
# 4. Data snapshot, canonical fits and the closing book
#    G-A · fixture inventory and filtration
# ===================================================================

ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours = 10_000)
db = PostgresStorage(R01_EXPERIMENT)

r01_models = R01_SMOKE ? R01_MODELS[1:1] : R01_MODELS
r01_fits = Dict(name => load_fit(db, name) for name in r01_models)

for name in r01_models
    f = r01_fits[name]
    tag = f.diagnostics.passed ? "PASS" : "FAIL"
    @printf("  [%s] %-30s folds=%d  R̂=%.4f  ESS=%s  div=%d\n",
            tag, name, length(f), f.diagnostics.max_rhat,
            string(f.diagnostics.min_ess_bulk), f.diagnostics.n_divergent)
    f.diagnostics.passed || @warn(
        "$(name) did not pass strict convergence gating. A posterior that did not " *
        "converge is too NARROW, so every calibration weight below is computed " *
        "against an over-confident λ. The sweep still runs; the numbers carry this.")
end

bf_odds = l01_betfair_closing_odds(ds)
inventory = odds_inventory(bf_odds)

"Match IDs of the seasons the published gate was measured on."
r01_gate_match_ids = Set{Int}(
    Int.(ds.matches.match_id[in.(ds.matches.season, Ref(R01_GATE_SEASONS))]))

"Restrict a fit's posterior to the gate seasons, and report what was dropped."
function r01_gate_latents(name)
    full = fit_latents(r01_fits[name])
    ids = latent_match_ids(full)
    kept = count(m -> m in r01_gate_match_ids, ids)
    gate = restrict_latents(full, r01_gate_match_ids)
    return (; full, gate, n_full = length(ids), n_gate = kept,
            n_dropped = length(ids) - kept)
end

r01_latents = Dict(name => r01_gate_latents(name) for name in r01_models)

println("\n--- G-A · fixture inventory and filtration ---")
@printf("  closing book : %d rows, %d fixtures, %d (market, line, selection) keys\n",
        nrow(bf_odds), length(unique(bf_odds.match_id)), nrow(inventory))
for name in r01_models
    l = r01_latents[name]
    @printf("  %-30s OOS fixtures: %d total → %d in %s (%d dropped, 26/27 extension)\n",
            name, l.n_full, l.n_gate, join(R01_GATE_SEASONS, "+"), l.n_dropped)
end
println("\n  quoted selections:")
for row in eachrow(inventory)
    @printf("    %-14s %5.1f  %-10s  %6d rows  %5d matches  mean fair %.4f\n",
            row.market_name, row.market_line, String(row.selection),
            row.n_rows, row.n_matches, row.mean_fair)
end
CSV.write(joinpath(R01_OUT, "r01_odds_inventory.csv"), inventory)

let missing_families = [String(s) for (_, sels) in L01_FAMILIES for s in sels
                        if !(s in Set(inventory.selection))]
    isempty(missing_families) ||
        @warn "the closing book does not quote every scored selection" missing_families
end


# %%
# ===================================================================
# 5. Market rate inversion
#    G-B · inversion coverage and residual quality
# ===================================================================
#
# Done ONCE. The inverted rates depend on the book and on nothing else, so they are
# shared by every model and every grid point below.

println("\n--- G-B · market rate inversion ---")
r01_all_ids = sort!(collect(union((Set(latent_match_ids(r01_latents[n].gate))
                                   for n in r01_models)...)))

t_inv = @elapsed r01_rates = invert_market_rates(
    bf_odds; config = R01_INVERSION, match_ids = r01_all_ids)

r01_inversion = inversion_frame(r01_rates)
CSV.write(joinpath(R01_OUT, "r01_market_inversion.csv"), r01_inversion)

let acc = r01_inversion[r01_inversion.accepted, :]
    @printf("  inverted %d fixtures in %.1fs (%d threads)\n",
            nrow(r01_inversion), t_inv, Threads.nthreads())
    @printf("  accepted     : %d (%.1f%%)\n",
            nrow(acc), 100 * nrow(acc) / max(nrow(r01_inversion), 1))
    if nrow(acc) > 0
        @printf("  residual SSE : median %.3e  p90 %.3e  max %.3e\n",
                median(acc.sse), quantile(acc.sse, 0.90), maximum(acc.sse))
        @printf("  λ_mkt home   : median %.3f  [%.3f, %.3f]\n",
                median(acc.lambda_mkt_h), minimum(acc.lambda_mkt_h),
                maximum(acc.lambda_mkt_h))
        @printf("  λ_mkt away   : median %.3f  [%.3f, %.3f]\n",
                median(acc.lambda_mkt_a), minimum(acc.lambda_mkt_a),
                maximum(acc.lambda_mkt_a))
    end
    for (reason, n) in refusal_counts(r01_inversion)
        @printf("  REFUSED %-58s %4d\n", reason, n)
    end
end

for name in r01_models
    cov = inversion_coverage(r01_rates, latent_match_ids(r01_latents[name].gate))
    @printf("  %-30s coverage %.1f%%  (accepted %d, refused %d, absent %d of %d)\n",
            name, 100 * cov.coverage, cov.n_accepted, cov.n_refused,
            cov.n_absent, cov.n_fixtures)
    cov.coverage < 0.90 && @warn(
        "$(name): fewer than 90% of fixtures could be inverted. Every uninverted " *
        "fixture passes through at w = 1, so the sweep's apparent effect size is " *
        "diluted by the refusal rate, not by the calibration.", coverage = cov.coverage)
end


# %%
# ===================================================================
# 6. Derivative-market coherence
#    G-C · the claim this whole construction is for
# ===================================================================
#
# Three partitions of one 12×12 tensor must agree. What is being verified is that
# the shifted container is still a JOINT distribution over scorelines, so no
# derivative price can contradict another — the axiom selection-level shifting
# breaks. The residual is the truncated tail beyond 11 goals a side, not an error.

println("\n--- G-C · derivative-market coherence ---")
let name = first(r01_models),
    probe = GenerativeCalibrationSpec(method = :inverse_gaussian, w_base = 0.25, sigma = 0.25)

    raw = r01_latents[name].gate
    cal, _ = calibrate_latents(raw, r01_rates, probe)
    for (label, container) in (("uncalibrated", raw), ("calibrated $(spec_label(probe))", cal))
        rep = coherence_report(container, l01_wide_markets();
                               max_goals = R01_MAX_GOALS, threaded = true)
        @printf("  %-28s families=%s  max |Σ−1| = %.3e  max between-family spread = %.3e\n",
                label, join(rep.family_names, ","), rep.max_deviation_from_one,
                rep.max_family_spread)
        rep.max_family_spread < 1e-9 || error(
            "G-C FAILED: two market families disagree by $(rep.max_family_spread) on " *
            "the same fixture. That cannot happen when every price is a partition of " *
            "one score tensor, so the pricing path — not the calibration — is wrong.")
    end
    println("  the residual is the 12-goal grid truncation, and it is shared by every family.")
end


# %%
# ===================================================================
# 7. The uncalibrated baseline and the raw edge anchor
#    G-D · identity control
# ===================================================================

println("\n--- G-D · uncalibrated baseline ---")
r01_identity = GenerativeCalibrationSpec(method = :static_geometric, w_base = 1.0)
r01_baseline_rows = NamedTuple[]
r01_family_rows = DataFrame[]
r01_anchors = Dict{String, Dict{Tuple{Int,Symbol}, Float64}}()

for name in r01_models
    raw = r01_latents[name].gate

    # The anchor context is the raw model priced over the wide book: the edges that
    # assign every stratified-LPD bucket for every grid point of this model.
    anchor_ctx = build_evaluation_context(
        raw, bf_odds, ds.matches, L01_EVAL.AbstractScoringRule[L01_EVAL.LogLoss()];
        markets = l01_wide_markets(), threaded = true)
    r01_anchors[name] = edge_anchor(anchor_ctx)

    row, fams = score_calibration(name, r01_identity, raw, bf_odds, ds.matches;
                                  markets = l01_wide_markets(), n_bins = R01_N_BINS,
                                  anchor = r01_anchors[name],
                                  edge_small = R01_EDGE_SMALL,
                                  edge_large = R01_EDGE_LARGE,
                                  weights = nothing)
    push!(r01_baseline_rows, merge(row, (; scope = "gate", spec = "uncalibrated")))
    fams.spec .= "uncalibrated"   # the identity grid point carries its own label
    push!(r01_family_rows, fams)

    @printf("  %-30s headline: LogLoss %.5f (BF %.5f)  ECE %.4f (BF %.4f)  N=%d\n",
            name, row.head_logloss, row.head_market_logloss,
            row.head_ece, row.head_market_ece, row.head_n_obs)
    @printf("  %-30s wide    : LogLoss %.5f  ECE %.4f  CRPS %.4f  N=%d\n",
            "", row.wide_logloss, row.wide_ece, row.crps_all, row.wide_n_obs)
end

let m12 = findfirst(r -> r.model == "m12_joint_hybrid_synergy", r01_baseline_rows)
    if m12 !== nothing
        b = r01_baseline_rows[m12]
        @printf("\n  published 40-fold m12 : LogLoss %.5f  ECE %.4f  BF ECE %.4f  N=%d\n",
                R01_PUBLISHED_M12_LOGLOSS, R01_PUBLISHED_M12_ECE,
                R01_PUBLISHED_BF_ECE, R01_PUBLISHED_N_OBS)
        @printf("  recomputed here       : LogLoss %.5f  ECE %.4f  BF ECE %.4f  N=%d\n",
                b.head_logloss, b.head_ece, b.head_market_ece, b.head_n_obs)
        if abs(b.head_logloss - R01_PUBLISHED_M12_LOGLOSS) > 5e-4 ||
           b.head_n_obs != R01_PUBLISHED_N_OBS
            @warn(
                "the recomputed baseline does not reproduce the published 40-fold " *
                "numbers. The published thresholds therefore do NOT apply to this " *
                "fixture set; §9 judges the sweep against the recomputed baseline and " *
                "the published figures are context only.",
                published_logloss = R01_PUBLISHED_M12_LOGLOSS,
                recomputed_logloss = b.head_logloss,
                published_n = R01_PUBLISHED_N_OBS, recomputed_n = b.head_n_obs)
        end
    end
end


# %%
# ===================================================================
# 8. The calibration sweep
# ===================================================================

r01_specs = R01_SMOKE ?
    [GenerativeCalibrationSpec(method = :inverse_gaussian, w_base = 0.25, sigma = 0.25),
     GenerativeCalibrationSpec(method = :standard_gaussian, w_base = 0.55, sigma = 0.35),
     GenerativeCalibrationSpec(method = :static_geometric, w_base = 0.70)] :
    sweep_specs(w_bases = R01_W_BASES, sigmas = R01_SIGMAS,
                methods = R01_METHODS, w_max = R01_W_MAX)

println("\n--- sweep · $(length(r01_specs)) specs × $(length(r01_models)) models ---")

r01_rows = NamedTuple[]
r01_weight_rows = NamedTuple[]
t_sweep = @elapsed for name in r01_models
    raw = r01_latents[name].gate
    anchor = r01_anchors[name]
    for (j, spec) in enumerate(r01_specs)
        cal, diag = calibrate_latents(raw, r01_rates, spec)
        ws = weight_summary(diag)
        row, fams = score_calibration(name, spec, cal, bf_odds, ds.matches;
                                      markets = l01_wide_markets(),
                                      n_bins = R01_N_BINS, anchor = anchor,
                                      edge_small = R01_EDGE_SMALL,
                                      edge_large = R01_EDGE_LARGE, weights = ws)
        push!(r01_rows, merge(row, (; scope = "gate")))
        push!(r01_family_rows, fams)
        push!(r01_weight_rows, merge((; model = name, spec = spec_label(spec)), ws))

        if is_identity_spec(spec)
            base = r01_baseline_rows[findfirst(r -> r.model == name, r01_baseline_rows)]
            Δll = abs(row.head_logloss - base.head_logloss)
            Δll < 1e-12 || error(
                "G-D FAILED for $name at $(spec_label(spec)): the identity spec scored " *
                "LogLoss $(row.head_logloss) against the baseline's $(base.head_logloss). " *
                "w = 1 must copy the raw draws bit for bit; it did not.")
        end

        if j % 10 == 0 || j == length(r01_specs)
            @printf("    %-30s %3d/%3d  %-18s LL %.5f  ECE %.4f  w̃ %.3f\n",
                    name, j, length(r01_specs), spec_label(spec),
                    row.head_logloss, row.head_ece, ws.w_median)
        end
    end
end
@printf("  sweep complete in %s\n", Training.format_elapsed(t_sweep))

r01_summary = DataFrame(vcat(r01_baseline_rows, r01_rows))
r01_families = vcat(r01_family_rows...)
r01_weights = DataFrame(r01_weight_rows)

CSV.write(joinpath(R01_OUT, "r01_sweep_scores.csv"), r01_summary)
CSV.write(joinpath(R01_OUT, "r01_family_scores.csv"), r01_families)
CSV.write(joinpath(R01_OUT, "r01_weight_summary.csv"), r01_weights)


# %%
# ===================================================================
# 9. Optimum per functional form, judged against the recomputed baseline
# ===================================================================

"The best grid point per (model, method) on the gate-scope headline LogLoss."
function r01_best_per_form(summary)
    out = NamedTuple[]
    for g in groupby(filter(r -> r.spec != "uncalibrated", summary), [:model, :method])
        ok = filter(r -> isfinite(r.head_logloss), g)
        nrow(ok) == 0 && continue
        push!(out, copy(ok[argmin(ok.head_logloss), :]))
    end
    return DataFrame(out)
end

r01_best = r01_best_per_form(r01_summary)
CSV.write(joinpath(R01_OUT, "r01_best_per_form.csv"), r01_best)

println("\n" * "="^150)
println(" OPTIMUM PER FUNCTIONAL FORM (headline scope: 1X2 + O/U 2.5 + BTTS)")
println("="^150)
@printf(" %-28s | %-18s | %-16s | %8s | %8s | %8s | %8s | %7s\n",
        "Model", "Method", "Spec", "LogLoss", "ΔLL base", "ECE", "BF ECE", "w̃")
println("-"^150)
for name in r01_models
    base = r01_baseline_rows[findfirst(r -> r.model == name, r01_baseline_rows)]
    @printf(" %-28s | %-18s | %-16s | %8.5f | %8s | %8.4f | %8.4f | %7s\n",
            name, "—", "uncalibrated", base.head_logloss, "—",
            base.head_ece, base.head_market_ece, "1.000")
    for row in eachrow(filter(r -> r.model == name, r01_best))
        @printf(" %-28s | %-18s | %-16s | %8.5f | %+8.5f | %8.4f | %8.4f | %7.3f\n",
                "", row.method, row.spec, row.head_logloss,
                row.head_logloss - base.head_logloss, row.head_ece,
                row.head_market_ece, row.w_median)
    end
end
println("="^150)

println("\n EDGE-STRATIFIED LPD (buckets anchored on the RAW model's edges)")
@printf(" %-28s | %-16s | %9s | %6s | %9s | %6s | %9s\n",
        "Model", "Spec", "LPD small", "N", "LPD large", "N", "LPD all")
println("-"^110)
for name in r01_models
    base = r01_baseline_rows[findfirst(r -> r.model == name, r01_baseline_rows)]
    @printf(" %-28s | %-16s | %9.5f | %6d | %9.5f | %6d | %9.5f\n",
            name, "uncalibrated", base.lpd_small, base.n_small,
            base.lpd_large, base.n_large, base.lpd_all)
    for row in eachrow(filter(r -> r.model == name, r01_best))
        @printf(" %-28s | %-16s | %9.5f | %6d | %9.5f | %6d | %9.5f\n",
                "", row.spec, row.lpd_small, row.n_small,
                row.lpd_large, row.n_large, row.lpd_all)
    end
end

println("\n GATE 1 — proper scoring and statistical gating")
println(" Judged against the RECOMPUTED baseline on this fixture set, not the")
println(" published 40-fold constants. Gate 2 (bankroll, Sharpe, drawdown) is r02's.")
println("-"^110)
for name in r01_models
    base = r01_baseline_rows[findfirst(r -> r.model == name, r01_baseline_rows)]
    for row in eachrow(filter(r -> r.model == name, r01_best))
        ll_ok = row.head_logloss <= base.head_logloss + 1e-9
        ece_ok = row.head_ece <= base.head_ece + 1e-9
        bf_ok = row.head_ece <= row.head_market_ece
        verdict = (ll_ok && ece_ok && bf_ok) ? "PASS" : "REFUSE"
        @printf(" [%-6s] %-28s %-18s %-16s  LogLoss %s  ECE vs model %s  ECE vs close %s\n",
                verdict, name, row.method, row.spec,
                ll_ok ? "ok" : "worse", ece_ok ? "ok" : "worse", bf_ok ? "ok" : "worse")
    end
end


# %%
# ===================================================================
# 10. Final report
# ===================================================================

println("\n" * "="^110)
println(" ARTEFACTS")
println("="^110)
for f in ["r01_odds_inventory.csv", "r01_market_inversion.csv", "r01_sweep_scores.csv",
          "r01_family_scores.csv", "r01_weight_summary.csv", "r01_best_per_form.csv"]
    @printf("  %s\n", joinpath(R01_OUT, f))
end
println("""
  NEXT. Nominate the winning (method, w_base, σ) per form into
  r02_portfolio_direction_audit.jl. A Gate-1 PASS here is a CALIBRATION result
  and says nothing about bankroll: the Ireland transfer improved calibration
  diagnostics on its own league and still lost 16-22% of final wealth on this one.
  Record every number in this directory's README.md with its run context before
  drawing a conclusion from it.
""")
@printf("  finished     : %s\n", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))
