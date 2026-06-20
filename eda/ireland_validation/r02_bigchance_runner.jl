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
PASTE COVERAGE OUTPUT HERE
=#

# ============================================================================
# 3. MARGINAL SUMMARY (moments / dispersion / zero-mass)
# ============================================================================
summarise_count(home_bc, "Home bigChanceCreated")
summarise_count(away_bc, "Away bigChanceCreated")
summarise_count(all_bc,  "All bigChanceCreated")

#=
PASTE SUMMARY OUTPUT HERE
=#

# ============================================================================
# 4. VALIDATION: reproduce r01's goals AICs with the new fitters first.
#    Poisson/NB here should match the goals numbers in r01_ireland_runner.jl
#    before we trust ZIP/ZINB/COM on bigChance.
# ============================================================================
goals_home = collect(skipmissing(ds.matches.home_score))
compare_count_models(goals_home, "VALIDATION — Home Goals")

#=
PASTE VALIDATION OUTPUT HERE (Poisson/NB AIC should match r01 ≈ 2962)
=#

# ============================================================================
# 5. MODEL COMPARISON — the core distribution decision
# ============================================================================
fits_home = compare_count_models(home_bc, "Home bigChanceCreated")
fits_away = compare_count_models(away_bc, "Away bigChanceCreated")
fits_all  = compare_count_models(all_bc,  "All bigChanceCreated")

#=
PASTE MODEL COMPARISON OUTPUT HERE
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
PASTE ROOTOGRAM + χ² OUTPUT HERE
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
PASTE LINK ANALYSIS OUTPUT HERE
=#

println("\n[INFO] bigChanceCreated EDA complete.")
