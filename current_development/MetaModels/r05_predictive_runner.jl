# current_development/MetaModels/r05_predictive_runner.jl
#
# A true out-of-sample predictive regime filter runner.
# It trains a Meta Model for EVERY weekly fold (expanding window) and 
# uses fold w-1 to predict the regime (Good/Bad) for fold w.
#
# Run with: julia --project -t 32

using BayesianFootball
using DataFrames
using Dates
using Statistics
using LogExpFunctions: logistic

using ThreadPinning
pinthreads(:cores)

# Reload the module fresh
if isdefined(Main, :MetaModels)
    println("Reloading MetaModels module...")
end


include("./current_development/MetaModels/src/MetaModels.jl")
include("src/MetaModels.jl")
using .MetaModels

include("./current_development/MetaModels/src/staking.jl")
include("src/staking.jl")

println("="^65)
println("  META MODEL — Predictive Weekly Regime Filter (r05)")
println("="^65)

# ===========================================================================
# 1. LOAD DATA AND LAYER 1 RESULTS
# ===========================================================================
println("\n[1] Loading DataStore...")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

println("[2] Loading Layer 1 Experiment Results...")
# save_dir   = "./data/meta_model_layer1/"

save_dir = "./data/meta_model_layer1/ireland/"
saved_files = BayesianFootball.Experiments.list_experiments(save_dir, data_dir="")
exp_results = BayesianFootball.Experiments.load_experiment(saved_files, 1)

# ===========================================================================
# 3. CONFIGURE META MODEL
# ===========================================================================
println("\n[3] Configuring ConvexMixtureMetaModel...")

# We use the global hierarchy to speed up the 80 weekly folds (no team biases)
meta_model = MetaModels.ConvexMixtureMetaModel(
    dynamics_config  = MetaModels.MetaGRWDynamicsConfig(σ_prior=0.2),
    hierarchy_config = MetaModels.GlobalMetaHierarchyConfig() 
)

# Concurrency is handled by the workflow.jl (defaults to nthreads())
sampler_config = BayesianFootball.Samplers.QueuedNUTSConfig(
    n_samples = 500,
    n_chains  = 4,
    n_warmup  = 200,
    accept_rate = 0.65,
    max_depth   = 10,
    initialisation = BayesianFootball.Samplers.UniformInit(-2, 2),
)

# ===========================================================================
# 4. RUN EXPERIMENT (WEEKLY FOLDS)
# ===========================================================================
TARGET_SELECTION = :under_25

println("[4] Creating MetaExperimentTask for selection: $TARGET_SELECTION")
meta_task = MetaModels.MetaExperimentTask(
    exp_results,
    meta_model,
    sampler_config,
    exp_results.config.splitter,
    TARGET_SELECTION
)

println("[5] Running Queued Fold Experiment...")
println("    $(Threads.nthreads()) threads available\n")
_raw_result = MetaModels.run_meta_experiment(meta_task; ds=ds)

meta_results = if _raw_result isa Tuple
    _raw_result[1]
else
    _raw_result
end

# Check convergence diagnostic (R-hat) across all chains
MetaModels.check_fold_rhats(meta_results)

# ===========================================================================
# 5. PREDICTIVE STAKING & ADVANCED HURDLE EVALUATION
# ===========================================================================
println("\n" * "="^65)
println("  PREDICTIVE STAKING RESULTS (OOS)")
println("="^65)

min_edge = 0.00

# This is now fully multi-threaded and automatically joins :distribution
ledger = MetaModels.compute_predictive_stakes(meta_results, meta_results.all_data; min_edge=min_edge)

if nrow(ledger) == 0
    println("No ledger data returned.")
else
    # Run the advanced Hurdle, Sharpe and Growth metrics analysis
    MetaModels.evaluate_predictive_hurdle(ledger; min_edge=min_edge)
end


#=
Performance across 439 OOS matches: Market line: under_15
--------------------------------------------------
L1 Raw (Unfiltered) : Bets=174  PnL=0.7986  ROI=12.27%
Good Regime (Gated) : Bets=68  PnL=0.0899  ROI=3.16%
Bad Regime (Skipped): Bets=106  PnL=0.7087  ROI=19.37%
--------------------------------------------------

First 10 weeks of regime threshold evolution:
  Week 2: θ_pred=0.5159 | threshold=0.5195 -> BAD
  Week 3: θ_pred=0.5124 | threshold=0.5177 -> BAD
  Week 4: θ_pred=0.484 | threshold=0.5159 -> BAD
  Week 5: θ_pred=0.5211 | threshold=0.5142 -> GOOD
  Week 7: θ_pred=0.5357 | threshold=0.5159 -> GOOD
  Week 8: θ_pred=0.5355 | threshold=0.5177 -> GOOD
  Week 9: θ_pred=0.5381 | threshold=0.5195 -> GOOD
  Week 10: θ_pred=0.545 | threshold=0.5203 -> GOOD
  Week 11: θ_pred=0.5437 | threshold=0.5211 -> GOOD
  Week 12:
=#


#=
# over_15
Performance across 439 OOS matches:
--------------------------------------------------
L1 Raw (Unfiltered) : Bets=157  PnL=-0.7469  ROI=-4.3%
Good Regime (Gated) : Bets=41  PnL=-0.5213  ROI=-10.87%
Bad Regime (Skipped): Bets=116  PnL=-0.2256  ROI=-1.8%
--------------------------------------------------

First 10 weeks of regime threshold evolution:
  Week 2: θ_pred=0.5154 | threshold=0.5197 -> BAD
  Week 3: θ_pred=0.52 | threshold=0.5175 -> GOOD
  Week 4: θ_pred=0.4938 | threshold=0.5197 -> BAD
  Week 5: θ_pred=0.5151 | threshold=0.5175 -> BAD
  Week 7: θ_pred=0.5245 | threshold=0.5154 -> GOOD
  Week 8: θ_pred=0.5285 | threshold=0.5175 -> GOOD
  Week 9: θ_pred=0.5321 | threshold=0.5197 -> GOOD
  Week 10: θ_pred=0.5399 | threshold=0.5198 -> GOOD
  Week 11: θ_pred=0.547 | threshold=0.52 -> GOOD
  Week 12: θ_pred=0.5162 | threshold=0.5223 -> BAD
=#


#=
over_25
Performance across 439 OOS matches:
--------------------------------------------------
L1 Raw (Unfiltered) : Bets=208  PnL=2.1102  ROI=10.45%
Good Regime (Gated) : Bets=82  PnL=0.0934  ROI=1.11%
Bad Regime (Skipped): Bets=126  PnL=2.0168  ROI=17.17%
--------------------------------------------------

First 10 weeks of regime threshold evolution:
  Week 2: θ_pred=0.5127 | threshold=0.5156 -> BAD
  Week 3: θ_pred=0.5209 | threshold=0.5142 -> GOOD
  Week 4: θ_pred=0.4927 | threshold=0.5156 -> BAD
  Week 5: θ_pred=0.5132 | threshold=0.5142 -> BAD
  Week 7: θ_pred=0.5056 | threshold=0.5132 -> BAD
  Week 8: θ_pred=0.4938 | threshold=0.5129 -> BAD
  Week 9: θ_pred=0.4895 | threshold=0.5127 -> BAD
  Week 10: θ_pred=0.4773 | threshold=0.5091 -> BAD
  Week 11: θ_pred=0.4861 | threshold=0.5056 -> BAD
  Week 12: θ_pred=0.4785 | threshold=0.4997 -> BAD

julia>
=#




#=
under_25 
Performance across 439 OOS matches:
--------------------------------------------------
L1 Raw (Unfiltered) : Bets=143  PnL=0.882  ROI=9.69%
Good Regime (Gated) : Bets=42  PnL=0.6558  ROI=33.07%
Bad Regime (Skipped): Bets=101  PnL=0.2262  ROI=3.18%
--------------------------------------------------

First 10 weeks of regime threshold evolution:
  Week 2: θ_pred=0.5156 | threshold=0.5185 -> BAD
  Week 3: θ_pred=0.5141 | threshold=0.5171 -> BAD
  Week 4: θ_pred=0.4976 | threshold=0.5156 -> BAD
  Week 5: θ_pred=0.5108 | threshold=0.5149 -> BAD
  Week 7: θ_pred=0.524 | threshold=0.5141 -> GOOD
  Week 8: θ_pred=0.4979 | threshold=0.5149 -> BAD
  Week 9: θ_pred=0.4772 | threshold=0.5141 -> BAD
  Week 10: θ_pred=0.4792 | threshold=0.5125 -> BAD
  Week 11: θ_pred=0.5027 | threshold=0.5108 -> BAD
  Week 12: θ_pred=0.4792 | threshold=0.5068 -> BAD
=#

