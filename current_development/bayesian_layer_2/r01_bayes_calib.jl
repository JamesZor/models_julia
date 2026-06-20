# current_development/bayesian_layer_2/r01_bayes_calib.jl
#
# Runner for the Bayesian Layer-2 calibration prototype.
# Assumes ds / results_model / posterior_predictive_distrubitons are already in the
# REPL (from current_development/betfair_closing_line/r00_basic_runner.jl).
#
# Stage 2 : Laplace-Gaussian global shift on under_25, verify the convolution.
# Stage 3 : add exp(-cΔT) decay tied to the 60-day half-life, report ESS.
# Stage 4 : add partially-pooled team effects, inspect û_j, gate on OOS log-score.

using Revise
using BayesianFootball
using DataFrames, Statistics, Dates, Random, Printf

const Cal  = BayesianFootball.Calibration
const Data = BayesianFootball.Data
const Predictions = BayesianFootball.Predictions

include(joinpath(@__DIR__, "src", "l01_bayes_calib.jl"))

# ==========================================================================
# 0. Build the L2 training frame and pick the market
# ==========================================================================
l2df   = Cal.build_l2_training_df(ds, posterior_predictive_distrubitons)
MARKET = :under_25
mdf    = filter(:selection => ==(MARKET), l2df)
dropmissing!(mdf, :is_winner)
@info "Market $MARKET" n=nrow(mdf) wins=Int(sum(mdf.is_winner)) splits=length(unique(mdf.split_id))

# ==========================================================================
# STAGE 2 — global Bayesian shift (no decay) + convolution verification
# ==========================================================================
cal2 = fit_global_shift(mdf; prob_col = :prob_mean, s_α = 1.0, decay = false)
@printf "\n[Stage 2] global shift  α̂ = %+.4f   sd(α) = %.4f   (logit scale)\n" cal2.fit.β[1] sqrt(cal2.fit.Σ[1,1])

# Verify (a) paired-N ≈ (b) N×M grid ≈ (c) analytic moments on a representative match
i  = argmax(abs.(logit.(clamp.(mdf.prob_mean, EPS, 1-EPS)) .- 0.0))   # a non-central one
μi, σi = predict_shift(cal2, mdf[i:i, :])
v = verify_convolution(mdf.distribution[i], μi[1], σi[1]; M = 400)
println("\n[Stage 2] Convolution check on match row $i  (logit scale):")
@printf "  L1 only        mean=%+.4f  var=%.4f\n"  v.l1_mean v.l1_var
@printf "  analytic       mean=%+.4f  var=%.4f   (= L1 + shift)\n" v.analytic_mean v.analytic_var
@printf "  paired-N       mean=%+.4f  var=%.4f\n"  v.paired_mean v.paired_var
@printf "  N×M grid       mean=%+.4f  var=%.4f\n"  v.grid_mean v.grid_var
@printf "  KS(grid,paired)= %.4f   (→0 confirms same 1-D law; NOT 2-D)\n" v.ks_grid_vs_paired
@printf "  widening:  Var[z_cal] − Var[z_L1] = %.4f  (= σ_shift² = %.4f)\n" (v.analytic_var - v.l1_var) σi[1]^2

# Nested sanity: σ→0 reproduces the deterministic BasicLogitShift translation
det_check = verify_convolution(mdf.distribution[i], μi[1], 0.0; M = 50)
@printf "[Stage 2] σ→0 limit: widening = %.6f (≈0 ⇒ matches current BasicLogitShift)\n" (det_check.analytic_var - det_check.l1_var)

# ==========================================================================
# STAGE 3 — time-decayed global shift, ESS report
# ==========================================================================
ref = maximum(mdf.match_date)
cal3 = fit_global_shift(mdf; prob_col = :prob_mean, s_α = 1.0, decay = true,
                        halflife = 60.0, ref_date = ref)
@printf "\n[Stage 3] decayed (HL=60d)  α̂ = %+.4f   sd(α) = %.4f\n" cal3.fit.β[1] sqrt(cal3.fit.Σ[1,1])
@printf "  raw n = %d   global ESS = %.1f   (%.0f%% of n)\n" nrow(mdf) cal3.ess 100*cal3.ess/nrow(mdf)

# ==========================================================================
# STAGE 4 — partially-pooled team residual-bias effects (empirical-Bayes τ)
# ==========================================================================
cal4, τ_best, ev_best = fit_team_shift_eb(mdf; prob_col = :prob_mean, s_α = 1.0,
                                          decay = true, halflife = 60.0, ref_date = ref,
                                          τ_grid = 0.05:0.05:0.8)
@printf "\n[Stage 4] team shift  EB τ = %.2f   (log-evidence = %.2f)\n" τ_best ev_best
te = team_effects(cal4)
println("[Stage 4] strongest residual-bias teams (|z|>1.5 ⇒ L1 systematically off):")
show(stdout, filter(:z => z -> abs(z) > 1.5, te); allrows = true, allcols = true)
println()

# ==========================================================================
# GATE — strict walk-forward OOS log-score: raw vs global vs team
# ==========================================================================
println("\n=== Walk-forward OOS log-score (higher = better) ===")
wf_global = walk_forward_logscore(mdf, (tr, rd) ->
              fit_global_shift(tr; prob_col = :prob_mean, s_α = 1.0, decay = true,
                               halflife = 60.0, ref_date = rd))
wf_team   = walk_forward_logscore(mdf, (tr, rd) ->
              fit_team_shift(tr; prob_col = :prob_mean, s_α = 1.0, τ = τ_best, decay = true,
                             halflife = 60.0, ref_date = rd))
@printf "  rows scored        : %d\n" wf_global.n
@printf "  raw L1             : %.5f\n" wf_global.raw
@printf "  + global shift     : %.5f   (Δ = %+.5f)\n" wf_global.cal wf_global.improvement
@printf "  + team shift (τ=%.2f): %.5f   (Δ = %+.5f)\n" τ_best wf_team.cal wf_team.improvement
println("\nDecision rule: ship global shift if Δ>0; add team effects only if team Δ > global Δ.")
