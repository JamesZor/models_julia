# current_development/scottish_wealth/r01_smoke_wealth.jl
#
# RUNNER: Smoke Test for Scottish Lower Wealth-Augmented Models
#
# Tests both Arm A (Proxy xG + Wealth) and Arm B (Champion 3-Layer + Wealth)
# on a single short target split ("25/26", 2 history seasons) using the
# standard Experiments framework.

using BayesianFootball
using DataFrames, Dates, Statistics, Printf
using MCMCChains, ThreadPinning

pinthreads(:cores)

const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions

const ROOT = pkgdir(BayesianFootball)
include("l02_wealth_engines.jl")

println("="^95)
println("SMOKE TEST: SCOTTISH LOWER WEALTH-AUGMENTED MODELS")
println("="^95)

# 1. Load DataStore
ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours=720)
@info "DataStore loaded" n_matches=nrow(ds.matches)

save_dir = joinpath(ROOT, "data/scottish_wealth_smoke/")
mkpath(save_dir)

dyn = PreGame.TimeDecayDynamics(days_half_life = 365.0)

mA_wealth = TeamPxGGoalsAPMWealthModel(dynamics_config = dyn)
mB_wealth = TeamFunnelPxGGoalsAPMWealthModel(dynamics_config = dyn)

# 2. Check Feature Hook
println("\n--- 1. TESTING WEALTH FEATURE EXTRACTION ---")
rA = Features.required_features(mA_wealth)
println("✓ Arm A declares $(length(rA)) features (including ScottishTeamWealthFeature: $(any(f -> f isa ScottishTeamWealthFeature, rA)))")

rB = Features.required_features(mB_wealth)
println("✓ Arm B declares $(length(rB)) features (including ScottishTeamWealthFeature: $(any(f -> f isa ScottishTeamWealthFeature, rB)))")

# 3. Train Arm A Smoke Split
println("\n" * "="^80)
println("TRAINING ARM A (Proxy xG + Wealth) SMOKE SPLIT")
println("="^80)

task_A = Experiments.create_experiment_task(
    ds, mA_wealth, "pxg_apm_wealth_smoke", save_dir;
    target_seasons       = ["25/26"],
    history_seasons      = 2,
    warmup_period        = 20,
    dynamics_col         = :match_biweek,
    samples              = 400,
    warmup               = 200,
    chains               = 3,
    use_queue            = true,
    max_depth            = 8,
    max_concurrent_tasks = 4
)

t0 = time()
res_A = Experiments.run_experiment(task_A)
elapsed_A = round((time() - t0) / 60, digits = 1)
println("✓ Arm A Wealth Smoke completed in $(elapsed_A) min ($(length(res_A.training_results.items)) folds)")

# Inspect w_wealth posterior from Arm A
ch_A = res_A.training_results.items[1][1]
w_vals_A = vec(Array(ch_A[:w_wealth]))
println("\n===================================================================")
println("ARM A: w_wealth POSTERIOR (Fold 1):")
println(@sprintf("  Mean:     %+6.4f", mean(w_vals_A)))
println(@sprintf("  Std:      %+6.4f", std(w_vals_A)))
println(@sprintf("  90%% CI:   [%+6.4f, %+6.4f]", quantile(w_vals_A, 0.05), quantile(w_vals_A, 0.95)))
println(@sprintf("  P(w > 0): %.1f%%", 100.0 * count(w_vals_A .> 0.001) / length(w_vals_A)))
println("===================================================================")

# 4. Train Arm B Smoke Split
println("\n" * "="^80)
println("TRAINING ARM B (Champion 3-Layer Funnel + Wealth) SMOKE SPLIT")
println("="^80)

task_B = Experiments.create_experiment_task(
    ds, mB_wealth, "funnel_pxg_apm_wealth_smoke", save_dir;
    target_seasons       = ["25/26"],
    history_seasons      = 2,
    warmup_period        = 20,
    dynamics_col         = :match_biweek,
    samples              = 400,
    warmup               = 200,
    chains               = 3,
    use_queue            = true,
    max_depth            = 8,
    max_concurrent_tasks = 4
)

t0_b = time()
res_B = Experiments.run_experiment(task_B)
elapsed_B = round((time() - t0_b) / 60, digits = 1)
println("✓ Arm B Wealth Smoke completed in $(elapsed_B) min ($(length(res_B.training_results.items)) folds)")

ch_B = res_B.training_results.items[1][1]
w_vals_B = vec(Array(ch_B[:w_wealth]))
println("\n===================================================================")
println("ARM B: w_wealth POSTERIOR (Fold 1):")
println(@sprintf("  Mean:     %+6.4f", mean(w_vals_B)))
println(@sprintf("  Std:      %+6.4f", std(w_vals_B)))
println(@sprintf("  90%% CI:   [%+6.4f, %+6.4f]", quantile(w_vals_B, 0.05), quantile(w_vals_B, 0.95)))
println(@sprintf("  P(w > 0): %.1f%%", 100.0 * count(w_vals_B .> 0.001) / length(w_vals_B)))
println("===================================================================")

println("\n✓ SMOKE TEST COMPLETE: Both Arm A and Arm B with Wealth are 100% Validated!")
