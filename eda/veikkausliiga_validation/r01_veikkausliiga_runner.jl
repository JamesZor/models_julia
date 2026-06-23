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
(paste DataStore field row counts here)
=#

# ============================================================================
# 2. FULL DATA AUDIT + PER-SEASON FEATURE COVERAGE (Stage-B readiness map).
# ============================================================================
datastore_overview(ds)
cov = feature_coverage_by_season(ds)
println("\n=== Veikkausliiga — per-season coverage ===")
show(cov, allrows = true, allcols = true)

#= RESULT 2
Expected from DB probe: xG + ratings from 2023; bigchance_cov 0 all seasons; betfair every season.
(paste the per-season coverage table here)
=#

# ============================================================================
# 3. MARGINAL MOMENTS — home / away / total goals.
# ============================================================================
g = get_goals(ds)
summarise_count(g["home"],  "31 Home goals")
summarise_count(g["away"],  "31 Away goals")
summarise_count(g["total"], "31 All goals")

#= RESULT 3
(paste mean | var | V/M | zeros | skew | regime for home/away/total)
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
Expect: Poisson AIC ≈ 5760.86, NegBin ≈ 5761.46 (matches first_division_eda.md). Fitters validated.
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
(paste univariate winners + NB r, NB1/NB2 verdict, DC ladder winner + ρ)
=#

# ============================================================================
# 5. GOODNESS-OF-FIT for the AIC-winning family (total goals).
# ============================================================================
best = fits_total[1]
println("\n[31] best family (total goals): $(best.name)")
println(rootogram_data(g["total"], best.pmf))
chi_square_gof(g["total"], best.pmf, best.k)

#= RESULT 5
(paste rootogram + χ² p-value for the winning family)
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
(paste overdispersion verdict, HA mean/variance p-values, within-team DI, temporal stability)
=#

# ============================================================================
# 7. PER-TEAM ATTACK / DEFENCE — goals (all seasons) then xG (2023+).
# ============================================================================
tml = build_team_match_long(ds)
ad_goals = fit_team_attack_defence(tml; min_matches = 15)
ad_xg    = fit_team_xg_attack_defence(tml; min_matches = 15)

#= RESULT 7
(paste the per-team goals attack/defence tables, then xG attack/defence; note which
 teams are significantly above/below league after BH adjustment)
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
(paste per-season rating coverage + per-position coverage; confirm ~100% from 2023, 0 pre-2023)
=#

# ============================================================================
# 9. PER-TEAM PLAYER-RATING DISTRIBUTIONS (squad quality, 2023+).
# ============================================================================
rl = build_team_rating_long(ds)
team_ratings = fit_team_rating_dist(rl; min_matches = 10)

#= RESULT 9
(paste per-team minute-weighted rating table; note squad-quality ranking and how it
 lines up with the goals/xG attack rankings)
=#

println("\n\n[INFO] Veikkausliiga Stage-A EDA complete.")
