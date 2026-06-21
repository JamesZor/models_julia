# eda/first_division_validation/r01_first_division_runner.jl
#
# Runner — execution + captured results. Stage-A EDA for Ireland First Division
# (tournament 718) contrasted with Ireland Premier (79). Run section-by-section in
# the kaimon REPL on the server.
#
# Sync first: local edits reach mcmc-beast only via git push then `git pull` in
# /root/BayesianFootball. The new IrelandFirstDivision segment is a Data-module
# struct → Revise will NOT pick it up; `manage_repl restart` after pulling.

using Revise
using BayesianFootball
using DataFrames
using Statistics
using StatsBase
using Printf
using ThreadPinning
pinthreads(:cores)

include("./l01_first_division_logic.jl")

const D = BayesianFootball.Data

println("==========================================================")
println(" IRELAND FIRST DIVISION (718) vs PREMIER (79) — STAGE A EDA")
println("==========================================================\n")

# ============================================================================
# 1. LOAD DATA — fresh SQL for the brand-new 718 segment; cache OK for 79.
# ============================================================================
ds718 = D.load_datastore_sql(D.IrelandFirstDivision())
ds79  = D.load_datastore_cached(D.Ireland())

#= RESULT 1
718 (fresh SQL): matches=914 statistics=680 odds=2581 lineups=32280 incidents=18867 betfair_odds=428216
 79 (cache)    : matches=1001 statistics=1972 odds=21778 lineups=39594 incidents=22572 betfair_odds=1051125
→ 718 DOES carry betfair_odds (428k rows) — the Stage-B CLV dependency is satisfied.
=#

# ============================================================================
# 2. FULL DATA AUDIT + PER-SEASON FEATURE COVERAGE (the Stage-B readiness map)
# ============================================================================
datastore_overview(ds718)
datastore_overview(ds79)

cov718 = feature_coverage_by_season(ds718)
cov79  = feature_coverage_by_season(ds79)
println("\n=== 718 First Division — per-season coverage ==="); show(cov718, allrows = true, allcols = true)
println("\n\n=== 79 Premier — per-season coverage ===");        show(cov79,  allrows = true, allcols = true)

#= RESULT 2
718 statistics has 76 cols; 79 has 98. 718 has NO bigChanceCreated columns at all,
and uses shots column `shotsOnGoal_*` (79 also has bigChance* columns).

=== 718 First Division — per-season coverage ===
 season  matches  played  stats_cov  xg_cov  bigchance_cov  shots_cov  odds_rows  betfair_rows  lineups  incidents
 2021    135      135     0.0        0.0     0.0            0.0        844        44981         5156     2098
 2022    144      144     0.326      0.0     0.0            0.007      420        72616         2059     2416
 2023    180      180     1.0        0.994   0.0            0.994      480        94710         7165     3355
 2024    180      180     0.989      0.989   0.0            0.989      471        91484         6996     3259
 2025    180      180     1.0        0.983   0.0            0.983       81        80497         7170     6102
 2026     95       95     1.0        0.979   0.0            0.979      285        43928         3734     1637

=== 79 Premier — per-season coverage ===
 season  matches  played  stats_cov  xg_cov  bigchance_cov  shots_cov  betfair_rows
 2021    180      180     0.0        0.0     0.0            0.0        179542
 2022    180      180     0.322      0.0     0.0            0.006      184183
 2023    180      180     0.983      0.0     0.883          0.983      181443
 2024    180      180     1.0        0.0     0.911          1.0        216434
 2025    180      180     1.0        0.978   0.933          1.0        194910
 2026    101      101     1.0        0.95    0.95           1.0         94613

Raw counts (ALL-period stats rows): 718 → 680 rows, 627 with xG; 79 → 696 rows, 272 with xG, 587 with bigChance.

HEADLINE coverage findings:
- 718 xG lands from 2023 (627 matches, ~99% from 2023 on) — EARLIER than the 2024 hypothesis.
- 718 has NO bigChance column ⇒ the bigChanceCreated pillar (prior work) CANNOT be used for 718.
- 718 has betfair every season ⇒ Stage-B CLV feasible.
- 79 xG only lands from 2025 (272 matches); 79 bigChance from 2023. Asymmetric feature timelines.
=#

# ============================================================================
# 3. MARGINAL MOMENTS — home / away / total goals, both leagues.
# ============================================================================
g718 = get_goals(ds718)
g79  = get_goals(ds79)

summarise_count(g718["home"],  "718 Home goals")
summarise_count(g718["away"],  "718 Away goals")
summarise_count(g718["total"], "718 All goals")
summarise_count(g79["home"],   "79 Home goals")
summarise_count(g79["away"],   "79 Away goals")
summarise_count(g79["total"],  "79 All goals")

#= RESULT 3
COUNT SUMMARY (mean | var | V/M | zeros emp vs Poisson | max | skew | regime)
718 Home (n=914):  1.4989 | 1.7026 | 1.1359 | 0.240 vs 0.223 (+0.016) | 9 | 1.13 | OVER-dispersed
718 Away (n=914):  1.3009 | 1.4592 | 1.1217 | 0.293 vs 0.272 (+0.021) | 7 | 1.02 | OVER-dispersed
718 All  (n=1828): 1.3999 | 1.5899 | 1.1357 | 0.266 vs 0.247 (+0.020) | 9 | 1.09 | OVER-dispersed
 79 Home (n=1001): 1.4016 | 1.4906 | 1.0635 | 0.247 vs 0.246 (+0.001) | 7 | 1.06 | OVER-dispersed (mild)
 79 Away (n=1001): 1.0939 | 1.0592 | 0.9682 | 0.322 vs 0.335 (-0.013) | 7 | 1.03 | ≈ equidispersed
 79 All  (n=2002): 1.2478 | 1.2979 | 1.0402 | 0.284 vs 0.287 (-0.003) | 7 | 1.10 | ≈ equidispersed

→ 718 scores MORE (1.40/side vs 1.25) and is consistently OVER-dispersed (V/M≈1.12–1.14).
  79 is near-equidispersed (V/M≈1.04), away goals actually slightly UNDER-dispersed (0.97).
  79 has the bigger home advantage (0.31 vs 0.20). First distinct-regime signal.
=#

# ============================================================================
# 4a. VALIDATION FIRST — reproduce the published 79 goal AICs with these fitters
#     before trusting them on 718.
# ============================================================================
analyze_goal_models(g79)
compare_count_models(g79["home"], "79 VALIDATION — Home goals")

#= RESULT 4a — VALIDATION on 79 (must reproduce published basic_goals/bigchance numbers)
79 Home goals: Poisson LL -1514.22 AIC 3030.45 ; NegBin LL -1513.36 AIC 3030.72 ; Weibull LL -1513.87.
  → EXACTLY matches r02_bigchance_runner.jl's validation block (Poisson AIC 3030.45, n=1001). Fitters validated.
79 Home compare_count_models winner by AIC & BIC: Poisson (λ=1.4016). NegBin/COM/Weibull all behind.
=#

# ============================================================================
# 4b. DISCRETE-COUNT LADDER — 718 then 79 (univariate + NB1/NB2 + DC bivariate).
# ============================================================================
# -- 718
analyze_goal_models(g718)
fits718_home  = compare_count_models(g718["home"],  "718 Home goals")
fits718_away  = compare_count_models(g718["away"],  "718 Away goals")
fits718_total = compare_count_models(g718["total"], "718 All goals")
compare_nb1_nb2(g718["total"], "718 All goals")
dc718 = analyze_heavyweight_models(ds718; label = "718 First Division")

# -- 79
fits79_home  = compare_count_models(g79["home"],  "79 Home goals")
fits79_total = compare_count_models(g79["total"], "79 All goals")
compare_nb1_nb2(g79["total"], "79 All goals")
dc79 = analyze_heavyweight_models(ds79; label = "79 Premier")

#= RESULT 4b — DISCRETE-COUNT LADDER
-- 718 univariate (compare_count_models, sorted by AIC) --
718 Home : NegBin AIC 2873.20 (BIC 2882.84) WINS both ; Poisson AIC 2878.08 (ΔAIC 4.9). r=11.97 μ=1.499
718 Away : NegBin AIC 2721.45 wins AIC ; BIC prefers Poisson (2730.25) — mild. r=10.78 μ=1.301
718 All  : NegBin AIC 5602.06 (BIC 5613.08) WINS both ; Poisson AIC 5614.32 (ΔAIC 12.3). r=10.79 μ=1.400
718 NB1 vs NB2 (All): identical fit marginally (ΔAIC 0.000), r=10.79 / α=0.093 / φ=1.130.

-- 79 univariate --
79 Home : Poisson AIC 3030.45 WINS AIC & BIC. NegBin r=large, no benefit.
79 All  : Poisson AIC 5760.86 WINS AIC & BIC ; NegBin AIC 5761.46. r=34.13 (α=0.029, very mild).
79 NB1 vs NB2 (All): identical marginally.

-- Dixon-Coles bivariate ladder (home,away jointly) --
718 (n=914): Indep NB AIC 5594.65 WINS ; DC NB 5596.64 ; Indep Wb 5596.87 ; Indep Pois 5603.51. ρ≈-0.0024
 79 (n=1001): Indep Poisson AIC 5724.79 WINS ; DC Pois 5726.79 ; Indep Wb 5726.92 ; Indep NB 5727.07. ρ≈-0.0022

KEY:
- 718 = genuine NegBin regime (NB beats Poisson by ~9–12 AIC on goals; NB best in the DC ladder too).
- 79  = Poisson regime (Poisson wins AIC+BIC; NB r≈34 → negligible over-dispersion).
- BOTH leagues: DC ρ ≈ 0 ⇒ NO low-score Dixon-Coles dependence; the τ correction adds nothing.
  (Note: a clamp on the NB dispersion r∈[1e-3,1e6] is required — without it the 79 DC-NB optimiser
   drives r_a→1e14 on near-equidispersed away goals and reports a spurious ~600-pt LL gain.)
=#

# ============================================================================
# 5. GOODNESS-OF-FIT for the AIC-winning family per league (total goals).
# ============================================================================
best718 = fits718_total[1]
println("\n[718] best family (total goals): $(best718.name)")
println(rootogram_data(g718["total"], best718.pmf))
chi_square_gof(g718["total"], best718.pmf, best718.k)

best79 = fits79_total[1]
println("\n[79] best family (total goals): $(best79.name)")
println(rootogram_data(g79["total"], best79.pmf))
chi_square_gof(g79["total"], best79.pmf, best79.k)
# 79's AIC winner is Poisson but it fails χ²; check whether NB rescues the tail.
nb79 = fits79_total[findfirst(f -> f.name == "NegBin", fits79_total)]
println("\n[79] NB GoF (for comparison):")
chi_square_gof(g79["total"], nb79.pmf, nb79.k)

#= RESULT 5 — GoF (rootogram + χ²) for the AIC-winning family, total goals
[718] winner NegBin: χ²=10.10 df=7 p=0.183 → NO evidence against NB. Rootogram hangs all |·|<0.76. Good fit.
[79]  winner Poisson: χ²=36.46 df=6 p≈0.000 → Poisson REJECTED. Misfit at count=7 (obs 5 vs exp 0.54,
      hang +1.50) and count=3 (hang -0.88) — a thin heavy tail of blow-out matches.
      NB for 79 total also fails: χ²=25.55 df=5 p=0.0001 (r=34) — over-dispersion too mild to win AIC,
      yet neither family captures the rare high-scoring tail. (Pooling home+away mixes V/M 1.06 & 0.97.)
=#

# ============================================================================
# 6. LEAGUE DIAGNOSTICS — overdispersion, home advantage, volatility, temporal.
# ============================================================================
println("\n######################## 718 FIRST DIVISION ########################")
test_overdispersion(g718["total"], "718 total goals")
test_home_advantage_mean(ds718.matches)
test_home_advantage_variance(ds718.matches)
tv718 = test_team_volatility(ds718.matches)
test_temporal_stability(ds718.matches)

println("\n######################## 79 PREMIER ########################")
test_overdispersion(g79["total"], "79 total goals")
test_home_advantage_mean(ds79.matches)
test_home_advantage_variance(ds79.matches)
tv79 = test_team_volatility(ds79.matches)
test_temporal_stability(ds79.matches)

#= RESULT 6 — LEAGUE DIAGNOSTICS
                              | 718 First Div            | 79 Premier
Overdispersion (total goals)  | NB justified (AIC 5602   | Poisson sufficient (AIC 5761
                              |   < Poisson 5614, Δ12.3) |   ≤ NB 5761)
Home advantage (mean)         | +0.198, MWU p=8.96e-4 *  | +0.308, MWU p=2.55e-8 *  (79 larger)
Home advantage (variance)     | ratio 1.167, F p=0.020 * | ratio 1.407, F p=7.13e-8 * (79 larger)
Temporal mean drift (KW)      | p=0.162 stable           | p=0.771 stable
Temporal var heteroscedast.   | ratio 1.37 (present)     | ratio 1.25
Team-level DI (goals conceded)| avg 1.029 (15 teams)     | avg 0.929 (14 teams)

→ Within-team residual dispersion: 718≈1.03 (slight over), 79≈0.93 (slight under). So 718's pooled
  over-dispersion is MOSTLY cross-team heterogeneity (absorbed by team strengths) PLUS a small genuine
  within-team excess; 79's mild pooled V/M is almost entirely cross-team heterogeneity. The residual
  gap (1.03 vs 0.93) is what a per-league dispersion knob should capture.
=#

# ============================================================================
# 7. 718-vs-79 CONTRAST (HEADLINE) — side-by-side regime table.
# ============================================================================
function contrast_row(ds, g, dc, tv, label)
    m = mean(g["total"]); v = var(g["total"])
    hm = mean(collect(skipmissing(ds.matches.home_score)))
    am = mean(collect(skipmissing(ds.matches.away_score)))
    (league = label,
     n = length(g["total"]),
     mean = round(m; digits = 3),
     vm = round(v / m; digits = 3),
     home_mean = round(hm; digits = 3),
     away_mean = round(am; digits = 3),
     ha = round(hm - am; digits = 3),
     within_team_DI = round(mean(tv.dispersion_index); digits = 3),
     dc_best = dc.best,
     dc_rho = round(dc.fits.dc_pois.ρ; digits = 4))
end
contrast = DataFrame([
    contrast_row(ds718, g718, dc718, tv718, "718 First Div"),
    contrast_row(ds79,  g79,  dc79,  tv79,  "79 Premier"),
])
show(contrast, allrows = true, allcols = true)

#= RESULT 7
 league         n     mean   vm     home_mean  away_mean  ha     within_team_DI  dc_best        dc_rho
 718 First Div  1828  1.4    1.136  1.499      1.301      0.198  1.029           Indep NB       -0.0024
 79 Premier     2002  1.248  1.04   1.402      1.094      0.308  0.929           Indep Poisson  -0.0022

VERDICT: 718 First Division is a DISTINCT REGIME from 79 Premier — higher-scoring (+0.15 goals/side),
genuinely over-dispersed (NB beats Poisson by 9–12 AIC; V/M 1.14 vs 1.04), with a SMALLER home
advantage. Both leagues independent (DC ρ≈0). Recommendation: STRATIFY dispersion — give First Division
its own NB dispersion parameter drawn from a shared cross-league hyperprior, while POOLING the team-
strength / home-advantage structure (whose hierarchy already absorbs the cross-team heterogeneity that
drives most of the marginal over-dispersion). Do NOT pool a single fixed dispersion across both leagues.
=#

println("\n\n[INFO] First Division Stage-A EDA complete.")
