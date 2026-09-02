# experiments/scottish_lower/01_poisson_2426_grid/r23_extend_poisson_2627.jl
#
# Incremental extension of Run #24 (m00_poisson_control in scottish_lower_joint_2426)
# to include the 2026/27 season (Folds 41 and 42).

using BayesianFootball
using Dates, Printf, LinearAlgebra, ThreadPinning
import LibPQ

pinthreads(:cores)
LinearAlgebra.BLAS.set_num_threads(1)

const TT = BayesianFootball.Training
const DD = BayesianFootball.Data

println("="^90)
println("  SCOTTISH LOWER POISSON: INCREMENTAL EXTENSION TO 2026/27 SEASON")
println("="^90)

# 1. Connect to mcmc_experiments and load cached DataStore
db = TT.PostgresStorage("scottish_lower_joint_2426")
println("Connecting to database: ", db)
ds = DD.load_datastore_cached(DD.ScottishLower())
println("Loaded DataStore with $(length(ds.matches.match_id)) matches.")

# 2. Extended Splitter Configuration
splitter_2627 = DD.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons    = ["24/25", "25/26", "26/27"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = 0,
    stop_early        = true,
)

# 3. Preview Extension
println("\n--- [1/3] Previewing Extension ---")
TT.preview_extension(db, 24, ds; splitter = splitter_2627)

# 4. Execute extend_fit
println("\n--- [2/3] Executing extend_fit (QueuedExecution 16) ---")
extended_fit = TT.extend_fit(db, 24, ds;
    splitter  = splitter_2627,
    execution = TT.QueuedExecution(16)
)
println("Sampling and persistence complete! Total folds now: $(length(extended_fit.folds))")

# 5. Verification & Audit
println("\n--- [3/3] Auditing Database State ---")
verified_fit = TT.load_fit(db, 24)
println("Successfully re-loaded Run #24: $(length(verified_fit.folds)) folds, $(size(verified_fit.latents.λ_home, 1)) total OOS matches.")
println("Audit result: max R-hat = $(verified_fit.diagnostics.max_rhat), min bulk ESS = $(verified_fit.diagnostics.min_ess_bulk)")
println("="^90)
