# ==============================================================================
# r59 — Deterministic bake-off of the six candidate formulations
# ==============================================================================
#
# QUESTION
#   Before spending a 40-fold MCMC grid on six two-arm joint candidates, does each
#   candidate's log-rate STRUCTURE carry any held-out signal at all? Specifically:
#   do point-in-time player RAPM lineups, a fixed bench weight, production wealth,
#   and away-travel distance beat a static team-state control on held-out goal
#   supremacy and official SofaScore-xG supremacy?
#
# CONTRACT
#   Local, deterministic ridge only. No MCMC, nothing on `mcmc-beast`, no betting.
#   RAPM vectors, the ridge coefficients, and every standardisation are fit on the
#   history block; target fixtures are scored and never fitted. Read
#   `l59_eda_loader.jl`'s header before quoting any number here: this ranks
#   formulations, it does not estimate the production models.
#
# USAGE
#   julia --project -t 8 experiments/scottish_lower/06_joint_player_lineup_fusion/r59_eda_joint_player_formulations.jl
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================
using CSV
using DataFrames
using LinearAlgebra
using Printf
using ThreadPinning

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "l59_eda_loader.jl"))

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================
const R59_HISTORY_FRACTION = 0.80
const R59_TARGET_SEASONS = ["24/25", "25/26"]
const R59_RESULTS = joinpath(@__DIR__, "r59_formulation_results.csv")
const R59_COVERAGE = joinpath(@__DIR__, "r59_coverage.csv")

# %%
# ==============================================================================
# 3. Local snapshots
# ==============================================================================
println("\n" * "="^100)
println(" EXPERIMENT 06 · r59 · SIX-FORMULATION DETERMINISTIC BAKE-OFF")
println("="^100)

full_store = j59_full_store()
official_xg = j59_official_xg()
@printf("England + Scotland snapshot: %d matches | %d lineup rows | %d official-xG fixtures\n",
        nrow(full_store.matches), nrow(full_store.lineups), length(official_xg))

# Scope A is the broad pool; B and C narrow onto the deployment target. B and C use a
# season split rather than a percentage, so they score exactly the fixtures the
# production 40-fold grid targets.
scopes = (
    (name = "A · England + Scotland", tiers = J59_TIERS_ENGLAND_SCOTLAND,
     target_seasons = nothing),
    (name = "B · Scotland 54-57", tiers = J59_TIERS_SCOTLAND,
     target_seasons = R59_TARGET_SEASONS),
    (name = "C · Scottish Lower 56/57", tiers = J59_TIERS_LOWER,
     target_seasons = R59_TARGET_SEASONS),
)

# %%
# ==============================================================================
# 4. Held-out leaderboard per scope
# ==============================================================================
all_rows = DataFrame()
coverage_rows = NamedTuple[]

for scope in scopes
    store = scope.tiers == J59_TIERS_ENGLAND_SCOTLAND ? full_store :
            j59_subset_store(full_store, scope.tiers)
    println("\n" * "="^100)
    println(" ", scope.name, " · tiers ", join(scope.tiers, ", "))
    println("="^100)
    @printf("matches: %d | lineup rows: %d\n", nrow(store.matches), nrow(store.lineups))

    result = j59_leaderboard(
        store;
        history_fraction = R59_HISTORY_FRACTION,
        target_seasons = scope.target_seasons,
        official_xg = official_xg,
    )

    coverage = j59_coverage(result.bundle)
    push!(coverage_rows, merge((scope = scope.name,), coverage))
    @printf("history: %d | held out: %d | shots-RAPM covered: %.1f%% | wealth present: %.1f%% | distance fallback: %.1f%% (sd %.3f)\n",
            coverage.n_history, coverage.n_target,
            100 * coverage.shots_rated_share, 100 * coverage.wealth_share,
            100 * coverage.distance_fallback_share, coverage.distance_sd)

    table = copy(result.table)
    insertcols!(table, 1, :scope => fill(scope.name, nrow(table)))
    append!(all_rows, table; cols = :union)
    j59_print_table(table)
end

# %%
# ==============================================================================
# 5. Artifacts and the decision-bearing view
# ==============================================================================
coverage_table = DataFrame(coverage_rows)
CSV.write(R59_RESULTS, all_rows)
CSV.write(R59_COVERAGE, coverage_table)

println("\n" * "="^100)
println(" HELD-OUT GOAL SUPREMACY, RANKED BY R² WITHIN SCOPE")
println("="^100)
goals = all_rows[all_rows.target .== "goal supremacy", :]
sort!(goals, [:scope, :r2]; rev = [false, true])
j59_print_table(goals)

println("\n" * "="^100)
println(" HELD-OUT OFFICIAL-xG SUPREMACY, RANKED BY R² WITHIN SCOPE")
println("="^100)
xg = all_rows[all_rows.target .== "official xG supremacy", :]
if nrow(xg) > 0
    sort!(xg, [:scope, :r2]; rev = [false, true])
    j59_print_table(xg)
else
    println("no scope had enough official-xG coverage to score this target")
end

println("\nCoverage:")
j59_print_table(coverage_table)
println("wrote: ", R59_RESULTS)
println("wrote: ", R59_COVERAGE)
