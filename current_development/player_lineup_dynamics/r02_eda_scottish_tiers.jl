# ==============================================================================
# r02 — Scottish-tier RAPM target and lineup-formulation transfer
# ==============================================================================
#
# QUESTION
#   Does pxG RAPM (`XGPlusMinusFeature`) or shot-volume RAPM
#   (`ShotsPlusMinusFeature`) produce the stronger held-out lineup signal in:
#     B. all Scottish tiers 54/55/56/57;
#     C. the deployment target, Scottish League One/Two 56/57?
#
# CONTRACT
#   Each scope fits its own rating vector through 23/24 and evaluates 24/25 + 25/26.
#   This produces the specified 710 Scottish-Lower target matches. No rating, scale, or
#   ridge coefficient sees those target seasons. This runner performs local ridge
#   regressions only; it launches no MCMC.
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================
include(joinpath(@__DIR__, "l01_lineup_eda.jl"))
using CSV
using Printf

# %%
# ==============================================================================
# 2. Configuration and local snapshots
# ==============================================================================
const PLD_R02_HISTORY_FRACTION = 0.80
const PLD_R02_TARGET_SEASONS = ["24/25", "25/26"]
const PLD_R02_BENCH_WEIGHT = 0.25

scope_a = pld_scope_a_store()
scopes = (
    (name="B · Scotland all tiers", tiers=PLD_TIERS_B,
     store=pld_subset_store(scope_a, PLD_TIERS_B)),
    (name="C · Scottish Lower", tiers=PLD_TIERS_C,
     store=pld_subset_store(scope_a, PLD_TIERS_C)),
)

# %%
# ==============================================================================
# 3. pxG-RAPM versus Shots-RAPM
# ==============================================================================
all_rows = DataFrame()
bench_rows = DataFrame()
for scope in scopes
    println("\n", repeat('=', 90))
    println(scope.name, " · tiers ", join(scope.tiers, ", "))
    @printf("matches: %d | lineup rows: %d\n",
            nrow(scope.store.matches), nrow(scope.store.lineups))

    for target in (:y_xg, :y_shots)
        println("\nRAPM target: ", target)
        result = pld_leaderboard(
            scope.store,
            target;
            history_fraction=PLD_R02_HISTORY_FRACTION,
            bench_weight=PLD_R02_BENCH_WEIGHT,
            target_seasons=PLD_R02_TARGET_SEASONS,
        )
        table = copy(result.table)
        table.scope = fill(scope.name, nrow(table))
        table.rapm_target = fill(String(target), nrow(table))
        select!(table, :scope, :rapm_target, Not([:scope, :rapm_target]))
        append!(all_rows, table; cols=:union)
        pld_print_table(table)

        bench = pld_select_bench_weight(result.run, result.outcomes)
        grid = copy(bench.table)
        grid.scope = fill(scope.name, nrow(grid))
        grid.rapm_target = fill(String(target), nrow(grid))
        append!(bench_rows, grid; cols=:union)
        @printf("nested-history optimal bench weight: %.3f\n", bench.winner)
    end
end

# %%
# ==============================================================================
# 4. Artifacts and compact scoreline leaderboard
# ==============================================================================
results_path = joinpath(@__DIR__, "r02_scottish_tier_results.csv")
bench_path = joinpath(@__DIR__, "r02_scottish_bench_weight_grid.csv")
CSV.write(results_path, all_rows)
CSV.write(bench_path, bench_rows)
println("\nScoreline-only leaderboard:")
scoreline = all_rows[all_rows.target .== "scoreline", :]
sort!(scoreline, [:scope, :r2], rev=[false, true])
pld_print_table(scoreline)
println("wrote: ", results_path)
println("wrote: ", bench_path)
