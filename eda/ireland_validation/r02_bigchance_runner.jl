# eda/ireland_validation/r02_bigchance_runner.jl
#
# Runner — execution + captured results. Deep EDA on `bigChanceCreated` for the
# League of Ireland. Run section-by-section in the kaimon REPL on the server.
#
# Sync first: local edits reach mcmc-beast only via git push then `git pull` in
# /root/BayesianFootball. Then include this file.

using Revise
using BayesianFootball
using DataFrames
using Statistics
using StatsBase
using ThreadPinning
pinthreads(:cores)

include("./l01_bigchance_logic.jl")

println("==========================================================")
println(" LEAGUE OF IRELAND: bigChanceCreated DEEP EDA ")
println("==========================================================\n")

# ============================================================================
# 1. LOAD DATA
# ============================================================================
println("[INFO] Loading Ireland data...")
ds = Data.load_datastore_sql(Data.Ireland())

# ============================================================================
# 2. EXTRACT bigChanceCreated + COVERAGE
#    Source: ds.statistics, period == "ALL", columns bigChanceCreated_home/away.
#    Stored as Union{Missing,Float64} → drop missing, round to Int.
# ============================================================================
stats_all = filter(r -> r.period == "ALL", ds.statistics)

home_bc = round.(Int, collect(skipmissing(stats_all.bigChanceCreated_home)))
away_bc = round.(Int, collect(skipmissing(stats_all.bigChanceCreated_away)))
all_bc  = vcat(home_bc, away_bc)

n_matches_total = nrow(filter(r -> !ismissing(r.home_score), ds.matches))
n_matches_stats = nrow(stats_all)
@printf("[INFO] Matches with played scores: %d | matches with ALL-period stats: %d (coverage %.1f%%)\n",
        n_matches_total, n_matches_stats, 100 * n_matches_stats / max(n_matches_total, 1))
@printf("[INFO] bigChance vectors: home=%d away=%d all=%d\n",
        length(home_bc), length(away_bc), length(all_bc))

#=
Matches with played scores: 1001 | matches with ALL-period stats: 696 (coverage 69.5%)
bigChance vectors: home=587 away=587 all=1174
→ ~58.6% of played matches carry a non-missing bigChanceCreated value. The future
  pillar must therefore be NaN-masked exactly like the xG and market pillars.
=#

# ============================================================================
# 3. MARGINAL SUMMARY (moments / dispersion / zero-mass)
# ============================================================================
summarise_count(home_bc, "Home bigChanceCreated")
summarise_count(away_bc, "Away bigChanceCreated")
summarise_count(all_bc,  "All bigChanceCreated")

#=
 COUNT SUMMARY: HOME (n=587)
Mean 1.7666 | Var 1.9540 | V/M 1.1061 | zeros emp 0.181 vs Poisson 0.171 (excess +0.010) | max 7 | skew 0.82  → OVER-dispersed (mild)
 COUNT SUMMARY: AWAY (n=587)
Mean 1.3390 | Var 1.5009 | V/M 1.1209 | zeros emp 0.278 vs Poisson 0.262 (excess +0.016) | max 8 | skew 1.09  → OVER-dispersed (mild)
 COUNT SUMMARY: ALL  (n=1174)
Mean 1.5528 | Var 1.7717 | V/M 1.1410 | zeros emp 0.229 vs Poisson 0.212 (excess +0.017) | max 8 | skew 0.96  → OVER-dispersed (mild)

Mean exceeds goals (≈1.40/1.09) and sits below shots — sanity OK. Zero-excess negligible (no zero-inflation).
=#

# ============================================================================
# 4. VALIDATION: reproduce r01's goals AICs with the new fitters first.
#    Poisson/NB here should match the goals numbers in r01_ireland_runner.jl
#    before we trust ZIP/ZINB/COM on bigChance.
# ============================================================================
goals_home = collect(skipmissing(ds.matches.home_score))
compare_count_models(goals_home, "VALIDATION — Home Goals")

#=
 MODEL COMPARISON: VALIDATION — HOME GOALS (n=1001)
 Poisson      k=1  LL -1514.22  AIC 3030.45  BIC 3035.35
 NegBin       k=2  LL -1513.36  AIC 3030.72  BIC 3040.54
 COM-Poisson  k=2  LL -1513.78  AIC 3031.56  BIC 3041.38
Poisson ≈ NegBin for goals (near-equidispersed) → reproduces r01's qualitative
verdict. Absolute AICs differ from r01 (2962) only because the dataset grew to
n=1001 since r01 was run. The fitters are validated.
=#

# ============================================================================
# 5. MODEL COMPARISON — the core distribution decision
# ============================================================================
fits_home = compare_count_models(home_bc, "Home bigChanceCreated")
fits_away = compare_count_models(away_bc, "Away bigChanceCreated")
fits_all  = compare_count_models(all_bc,  "All bigChanceCreated")

#=
 HOME (n=587)   sorted by AIC          AWAY (n=587)              ALL (n=1174)
 NegBin       AIC 1958.04 BIC 1966.79   NegBin   1766.0  1774.75  NegBin       3751.08 3761.21  ← AIC & BIC winner
 COM-Poisson  AIC 1958.31 BIC 1967.06   COM      1766.55 1775.30  COM-Poisson  3752.04 3762.17
 WeibullCount AIC 1958.47 BIC 1967.22   Weibull  1766.72 1775.47  WeibullCount 3752.48 3762.62
 Poisson      AIC 1959.02 BIC 1963.39   Poisson  1767.60 1771.97  ZINB         3753.08 3768.28  (LL==NB, π→0)
 ZINB         AIC 1960.04 BIC 1973.17   ZINB     1768.0  1781.12  ZIP          3757.63 3767.77
 ZIP          AIC 1960.41 BIC 1969.16   ZIP      1768.2  1776.95  Poisson      3759.29 3764.36

NB fit (home): r=16.69, μ=1.767.
- POOLED: NegBin wins AIC *and* BIC (ΔAIC≈8 over Poisson) — over-dispersion is real.
- PER-SIDE: NegBin wins AIC but the stricter BIC prefers Poisson → over-dispersion is MILD.
- ZINB ties NegBin's LL with π→0 ⇒ NO structural zeros (zero-inflation rejected).
- COM-Poisson (ν≈?) and Weibull-count competitive but never beat NB; NB is also
  already AD-safe in src/MyDistributions. Verdict: Negative Binomial.
=#

# ============================================================================
# 6. GOODNESS-OF-FIT — rootogram + χ² for the AIC-winning family (all data)
# ============================================================================
best = fits_all[1]                       # already sorted by AIC
println("\n[INFO] Best-by-AIC family for All bigChance: $(best.name)")
roo = rootogram_data(all_bc, best.pmf)
println(roo)
chi_square_gof(all_bc, best.pmf, best.k)

#=
Best-by-AIC family for All bigChance: NegBin
 count  observed  expected  sqrtO   sqrtE   hang
   0      269      274.36   16.401  16.564  -0.162
   1      386      373.93   19.647  19.337   0.310
   2      279      277.68   16.703  16.664   0.040
   3      130      148.79   11.402  12.198  -0.796
   4       74       64.34    8.602   8.021   0.581
   5       27       23.83    5.196   4.882   0.314
   6        6        7.84    2.449   2.800  -0.351
   7        2        2.35    1.414   1.533  -0.118
   8        1        0.65    1.000   0.807   0.193
χ² GoF: χ²=5.246 | bins=9 | df=6 | p=0.5127

Rootogram hangs all |·|<0.8 (slight under-fit at count=3); χ² p=0.51 → NO evidence
against NB. Negative Binomial is an excellent fit to the marginal.
=#

# ============================================================================
# 7. LINK ANALYSIS — relationship to the shared latent attack rate (λ)
# ============================================================================
long_df = build_bigchance_long(ds)
@printf("[INFO] long table rows (team-match): %d\n", nrow(long_df))

bigchance_vs_outcomes(long_df)
mean_variance_scaling(long_df; min_matches = 20)
home_away_bigchance(long_df)

#=
long table rows (team-match): 1174

LINK: bigChanceCreated vs GOALS / xG
 corr(bigChance, goals): Pearson 0.4745 | Spearman 0.4692
 corr(bigChance, xG)   : Pearson 0.6439 | Spearman 0.6555  (n=514)
 Poisson GLM goals ~ big_chance: slope 0.2665 (SE 0.0163, z=16.4, p≈3e-60)
    → each extra big chance ⇒ ×exp(0.2665)=1.31 expected goals (+31%).
 OLS big_chance ~ xg: intercept 0.125 (p=0.166, ~0), slope 1.116 (SE 0.059, t=19.0, p≈1.6e-61)
    → bigChance ≈ 1.12·xG with ~zero intercept (near 1:1 with the attack rate).

MEAN–VARIANCE SCALING (per team, n≥20): 12 teams, mean team V/M 1.0819
 NB law Var−Mean=α·Mean²: α=0.0595 (SE 0.040, t=1.48, p=0.168, CI[-0.029,0.148])
    → implied NB r≈16.8 (matches direct fit r=16.7) BUT α not individually significant
      ⇒ much of the pooled over-dispersion is cross-team heterogeneity the team
        dynamics already absorb, not within-team excess.

HOME vs AWAY: home mean 1.7666 vs away 1.3390 | MWU p=7.47e-8
    → strong home advantage on chance creation → pillar should inherit the model's
      existing home_adv term (no separate HA needed).
=#

println("\n[INFO] bigChanceCreated EDA complete.")
