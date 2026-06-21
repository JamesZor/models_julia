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

=#

# ============================================================================
# 4a. VALIDATION FIRST — reproduce the published 79 goal AICs with these fitters
#     before trusting them on 718.
# ============================================================================
analyze_goal_models(g79)
compare_count_models(g79["home"], "79 VALIDATION — Home goals")

#= RESULT 4a

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

#= RESULT 4b

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

#= RESULT 5

=#

# ============================================================================
# 6. LEAGUE DIAGNOSTICS — overdispersion, home advantage, volatility, temporal.
# ============================================================================
println("\n######################## 718 FIRST DIVISION ########################")
test_overdispersion(g718["total"], "718 total goals")
test_home_advantage_mean(ds718.matches)
test_home_advantage_variance(ds718.matches)
test_team_volatility(ds718.matches)
test_temporal_stability(ds718.matches)

println("\n######################## 79 PREMIER ########################")
test_overdispersion(g79["total"], "79 total goals")
test_home_advantage_mean(ds79.matches)
test_home_advantage_variance(ds79.matches)
test_team_volatility(ds79.matches)
test_temporal_stability(ds79.matches)

#= RESULT 6

=#

# ============================================================================
# 7. 718-vs-79 CONTRAST (HEADLINE) — side-by-side regime table.
# ============================================================================
function contrast_row(ds, g, dc, label)
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
     dc_best = dc.best,
     dc_rho = round(dc.fits.dc_pois.ρ; digits = 4))
end
contrast = DataFrame([
    contrast_row(ds718, g718, dc718, "718 First Div"),
    contrast_row(ds79,  g79,  dc79,  "79 Premier"),
])
show(contrast, allrows = true, allcols = true)

#= RESULT 7

=#

println("\n\n[INFO] First Division Stage-A EDA complete.")
