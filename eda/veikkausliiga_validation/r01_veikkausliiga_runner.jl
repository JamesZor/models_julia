# eda/veikkausliiga_validation/r01_veikkausliiga_runner.jl
#
# Runner — execution + captured results. Stage-A EDA for Veikkausliiga (Finnish
# top flight, tournament 31). Standalone characterisation (no contrast league).
# Run section-by-section in the kaimon REPL on the server.
#
# Sync first: local edits reach mcmc-beast only via git push then `git pull` in
# /root/BayesianFootball. The new Veikkausliiga segment is a Data-module struct →
# Revise will NOT pick it up; `manage_repl restart` after pulling.
#
# Verified DB signature (betdb tournament 31): ~678 played matches, 2021–2026,
# spring–autumn calendar (~132/season; 2026 partial). xG and player ratings both
# begin 2023 (~100%); NO bigChanceCreated column; full betfair coverage (~674).

using Revise
using BayesianFootball
using DataFrames
using Statistics
using StatsBase
using Printf
using ThreadPinning
pinthreads(:cores)

include("./l01_veikkausliiga_logic.jl")

const D = BayesianFootball.Data

println("==========================================================")
println(" VEIKKAUSLIIGA (31) — STAGE A EDA")
println("==========================================================\n")

# ============================================================================
# 1. LOAD DATA — fresh SQL for the brand-new 31 segment (no cache yet).
# ============================================================================
ds = D.load_datastore_sql(D.Veikkausliiga())

#= RESULT 1
matches=678 statistics=674 odds=15133 lineups=24734 incidents=12729 betfair_odds=966834
→ 31 carries betfair every season (966k rows) — Stage-B CLV dependency satisfied.
=#

# ============================================================================
# 2. FULL DATA AUDIT + PER-SEASON FEATURE COVERAGE (Stage-B readiness map).
# ============================================================================
datastore_overview(ds)
cov = feature_coverage_by_season(ds)
println("\n=== Veikkausliiga — per-season coverage ===")
show(cov, allrows = true, allcols = true)

#= RESULT 2
 season  matches  played  stats_cov  xg_cov  bigchance_cov  shots_cov  betfair_rows  lineups
 2021    132      132     0.992      0.000   0.0            0.992      177182        4706
 2022    132      132     0.985      0.000   0.0            0.985      182179        4729
 2023    132      132     1.000      1.000   0.0            1.000      175975        4741
 2024    132      132     1.000      1.000   0.0            1.000      220268        4735
 2025    132      132     0.992      0.992   0.0            0.992      191640        5119
 2026     18       18     1.000      1.000   0.0            1.000       19590         704
→ xG (and player ratings, §8) begin 2023; NO bigChance any season; betfair every season.
=#

# ============================================================================
# 3. MARGINAL MOMENTS — home / away / total goals.
# ============================================================================
g = get_goals(ds)
summarise_count(g["home"],  "31 Home goals")
summarise_count(g["away"],  "31 Away goals")
summarise_count(g["total"], "31 All goals")

#= RESULT 3
HOME (n=678):  mean 1.466 | var 1.614 | V/M 1.101 | zeros 0.264 vs 0.231 (+0.033) | skew 0.79 | OVER (mild)
AWAY (n=678):  mean 1.325 | var 1.381 | V/M 1.042 | zeros 0.261 vs 0.266 (-0.005) | skew 1.11 | ≈ equi
ALL  (n=1356): mean 1.395 | var 1.501 | V/M 1.076 | zeros 0.263 vs 0.248 (+0.015) | skew 0.94 | OVER (mild)
→ Mildly over-dispersed (between 718's 1.14 and 79's 1.04). Home advantage only +0.142. Home goals
  carry a small zero-excess; away goals are essentially Poisson.
=#

# ============================================================================
# 4a. VALIDATION GUARD — reproduce a known-league number before trusting the
#     fitters on 31. (Re-fit Ireland 79 total goals; compare to the published
#     Poisson AIC 5760.86 / NegBin 5761.46 from first_division_eda.md.)
# ============================================================================
ds79 = D.load_datastore_cached(D.Ireland())
g79 = get_goals(ds79)
compare_count_models(g79["total"], "79 VALIDATION — All goals")

#= RESULT 4a
79 total (n=2012): Poisson AIC 5787.94 (BIC 5793.55) WINS ; NegBin 5788.74. Poisson-regime verdict
reproduced. (n=2012 vs 2002 in the 718 study → the cache gained 2026 matches, hence the small drift
from the published 5760.86; the qualitative result — Poisson wins AIC+BIC — is unchanged.) Validated.
=#

# ============================================================================
# 4b. DISCRETE-COUNT LADDER — univariate + NB1/NB2 + Dixon-Coles bivariate.
# ============================================================================
analyze_goal_models(g)
fits_home  = compare_count_models(g["home"],  "31 Home goals")
fits_away  = compare_count_models(g["away"],  "31 Away goals")
fits_total = compare_count_models(g["total"], "31 All goals")
compare_nb1_nb2(g["total"], "31 All goals")
dc = analyze_heavyweight_models(ds; label = "Veikkausliiga (31)")

#= RESULT 4b
-- univariate (compare_count_models, sorted by AIC) --
31 Home : ZIP AIC 2117.0 WINS (π=0.074, λ=1.583); Poisson 2122.42 (ΔAIC 5.4) → mild home zero-inflation. NB r=13.48
31 Away : Poisson AIC 1997.73 WINS AIC+BIC; NB r=35.9 (negligible) → clean Poisson
31 All  : COM 4121.40 / Weibull 4121.42 / NegBin 4121.44 tied; Poisson 4123.02 → NB beats Poisson by only 1.58 AIC. r=18.56
31 NB1 vs NB2 (All): IDENTICAL (ΔAIC 0.000), φ=1.075.

-- Dixon-Coles bivariate ladder (n=678), AIC --
Indep Weibull 4118.87 (WIN) | Indep Poisson 4120.14 | Indep NB 4120.31 | DC Weibull 4120.80 | DC Poisson 4122.08 | DC NB 4122.24
ρ ≈ 0.012 across variants.

KEY: near-Poisson, very mildly over-dispersed league (closer to 79 than 718). Independents beat all DC
variants and ρ≈0 ⇒ NO Dixon-Coles low-score dependence; the τ correction adds nothing.
=#

# ============================================================================
# 5. GOODNESS-OF-FIT for the AIC-winning family (total goals).
# ============================================================================
best = fits_total[1]
println("\n[31] best family (total goals): $(best.name)")
println(rootogram_data(g["total"], best.pmf))
chi_square_gof(g["total"], best.pmf, best.k)

#= RESULT 5
COM-Poisson (AIC winner): χ²=2.93 df=6 p=0.818 → excellent fit; rootogram max|hang|=0.53.
Poisson (reference)      : χ²=8.88 df=7 p=0.262 → NOT rejected. Unlike 79 (Poisson failed p≈0 on a
blow-out tail), 31 total goals have no heavy tail — plain Poisson is already an acceptable marginal.
=#

# ============================================================================
# 6. LEAGUE DIAGNOSTICS — overdispersion, home advantage, volatility, temporal.
# ============================================================================
test_overdispersion(g["total"], "31 total goals")
test_home_advantage_mean(ds.matches)
test_home_advantage_variance(ds.matches)
tv = test_team_volatility(ds.matches)
test_temporal_stability(ds.matches)

#= RESULT 6
Overdispersion (total)        : NB AIC 4121.44 < Poisson 4123.02 (Δ1.58) — NB marginally justified
Home advantage (mean)         : +0.142, MWU p=0.053 — NOT significant (weakest of the 3 leagues)
Home advantage (variance)     : ratio 1.169, F p=0.042 *
Temporal mean drift (KW)      : p=0.234 stable (months 4–9, spring–autumn). Total goals/match 2.791
Temporal var heteroscedast.   : ratio 1.58 (present)
Within-team DI (goals conceded): avg 0.985 (16–17 teams) → essentially equidispersed once team strength
  is removed; the small pooled V/M 1.08 is mostly cross-team heterogeneity the hierarchy already absorbs.
=#

# ============================================================================
# 7. PER-TEAM ATTACK / DEFENCE — goals (all seasons) then xG (2023+).
# ============================================================================
tml = build_team_match_long(ds)
ad_goals = fit_team_attack_defence(tml; min_matches = 15)
ad_xg    = fit_team_xg_attack_defence(tml; min_matches = 15)

#= RESULT 7  (BH p_adj<0.05 = *)
GOALS attack (scored/match, best first): hjk 1.867* | inter-turku 1.743* | kups 1.690* | ... |
  fc-lahti 1.044* | ifk-mariehamn 1.009* | ktp 0.939* | hifk 0.864*   (7/16 significant)
GOALS defence (conceded, best=fewest): kups 0.814* | hjk 0.885* | fc-honka 1.015* | ... |
  ac-oulu 1.699* | ktp 2.015* | eif 2.318*   (7/16 significant)
  EB shrinkage works: EIF (n=22) raw 0.864 attack → shrunk 1.090, NOT flagged on attack; its 22-match
  defensive collapse (2.32) still significant.
xG attack (2023+): hjk 1.86* | sjk 1.79* | inter-turku 1.70*  (SJK higher on xG than goals → under-finishing)
xG defence (2023+): kups 1.04* (p=5e-7) | hjk 1.15* | inter-turku 1.20* ; worst eif 2.10*, ktp 1.94*
→ Goals and xG rankings AGREE: top HJK/KuPS/Inter Turku; bottom KTP/IFK Mariehamn/EIF.
=#

# ============================================================================
# 8. PLAYER-RATING COVERAGE AUDIT.
# ============================================================================
rcov = rating_coverage_audit(ds)
println("\n=== Rating coverage by season ===")
show(rcov, allrows = true, allcols = true)
pcov = rating_position_coverage(ds)
println("\n\n=== Rating coverage by position (pooled) ===")
show(pcov, allrows = true, allcols = true)

#= RESULT 8
Per-season: ratings start 2023 — frac_any_rating ~1.0, mean rated starters/team ≈11.0 (full XI).
  2021/2022: 0.0 (no ratings). 2023:1.0/11.0 | 2024:1.0/10.99 | 2025:0.992/10.99 | 2026:1.0/10.97.
Per-position (pooled over ALL 6 seasons): G 0.72 | D 0.72 | F 0.72 | M 0.50. G/D/F parity (≈4-of-6
  rated-season fraction) ⇒ no positional bias among known positions; lower M is an artefact of pooling
  the 2 unrated seasons + clean_pos defaulting unknown labels to "M". Restricted to 2023+ the XI is full.
→ Player ratings fully model-usable from 2023 (same window as xG).
=#

# ============================================================================
# 9. PER-TEAM PLAYER-RATING DISTRIBUTIONS (squad quality, 2023+).
# ============================================================================
rl = build_team_rating_long(ds)
team_ratings = fit_team_rating_dist(rl; min_matches = 10)

#= RESULT 9  (league μ=6.968, between-team τ=0.118; BH p_adj<0.05 = *)
Best: kups 7.129* | hjk 7.124* | fc-honka 7.119* | ilves 7.053* | inter-turku 7.021 | sjk 7.006
Worst: eif 6.702* | ifk-mariehamn 6.845* | ff-jaro 6.850* | ac-oulu 6.867*
→ Squad-quality ranking matches the goals/xG strength ranking (KuPS/HJK top, EIF/IFK Mariehamn bottom).
  Ratings tightly clustered (within-team sd≈0.25), so absolute gaps small but elite clubs separate clearly.
=#

println("\n\n[INFO] Veikkausliiga Stage-A EDA complete.")
