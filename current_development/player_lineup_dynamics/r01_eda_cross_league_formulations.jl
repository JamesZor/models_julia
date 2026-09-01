# ==============================================================================
# r01 — Cross-league lineup formulation bake-off
# ==============================================================================
#
# QUESTION
#   Which pre-match aggregation of one history-fit pxG-RAPM player vector best predicts held-out
#   scoreline and official SofaScore-xG supremacy across England and Scotland tiers
#   1/2/3/84/54/55/56/57?
#
# CONTRACT
#   The first 80% of fixtures, ordered by kickoff, fit RAPM and the small ridge calibration.
#   The last 20% are untouched evaluation fixtures. Official xG is scored only where the
#   independent r92 pull contains a non-zero measurement. This is local deterministic EDA, not
#   MCMC and not a betting study.
#
# USAGE
#   julia --project -t 8 current_development/player_lineup_dynamics/r01_eda_cross_league_formulations.jl
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
# 2. Configuration
# ==============================================================================
const PLD_R01_HISTORY_FRACTION = 0.80
const PLD_R01_BENCH_WEIGHT = 0.25
const PLD_R01_OUTPUT = joinpath(@__DIR__, "r01_lineup_formulation_results.csv")

# %%
# ==============================================================================
# 3. Local data snapshot
# ==============================================================================
println("\n", repeat('=', 90))
println("r01 · eight-tier lineup aggregation bake-off")
println(repeat('=', 90))
ds = pld_scope_a_store()
@printf("matches: %d | lineup rows: %d | tiers: %s\n",
        nrow(ds.matches), nrow(ds.lineups), join(PLD_TIERS_A, ", "))

# %%
# ==============================================================================
# 4. History-fit pxG RAPM and held-out evaluation
# ==============================================================================
result = pld_leaderboard(
    ds,
    :y_xg;
    history_fraction=PLD_R01_HISTORY_FRACTION,
    bench_weight=PLD_R01_BENCH_WEIGHT,
)
@printf("history fixtures: %d | held out: %d\n",
        length(result.run.history_ids), length(result.run.target_ids))
println("\nHeld-out leaderboard (higher r/rho/R2 and lower MAE are better):")
pld_print_table(result.table)

# %%
# ==============================================================================
# 5. Bench-weight selection inside the history block
# ==============================================================================
bench = pld_select_bench_weight(result.run, result.outcomes)
@printf("nested-history optimal w_bench: %.3f\n", bench.winner)
pld_print_table(bench.table)

# %%
# ==============================================================================
# 6. Artifact
# ==============================================================================
CSV.write(PLD_R01_OUTPUT, result.table)
CSV.write(joinpath(@__DIR__, "r01_bench_weight_grid.csv"), bench.table)
println("wrote: ", PLD_R01_OUTPUT)
