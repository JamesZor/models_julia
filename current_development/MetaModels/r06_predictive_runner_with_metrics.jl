# current_development/MetaModels/r06_predictive_runner_with_metrics.jl
#
# Runs the true out-of-sample predictive regime filter backtest,
# then applies the Bernoulli-Gamma hurdle model to evaluate L1 Raw, Good, and Bad regimes.
#
# Run with: julia --project -t 32

using BayesianFootball
using DataFrames
using Dates
using Statistics
using LogExpFunctions: logistic

using ThreadPinning
pinthreads(:cores)

# Reload modules
if isdefined(Main, :MetaModels)
    println("Reloading MetaModels...")
end

include("./current_development/MetaModels/src/MetaModels.jl")
include("src/MetaModels.jl")
using .MetaModels

include("./current_development/MetaModels/src/staking.jl")
include("src/staking.jl")

include("./current_development/MetaModels/l06_metrics.jl")

println("="^65)
println("  META MODEL — Predictive Weekly Regime Filter with Hurdle Metrics (r06)")
println("="^65)

# ===========================================================================
# 1. LOAD DATA AND LAYER 1 RESULTS
# ===========================================================================
println("\n[1] Loading DataStore...")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

println("[2] Loading Layer 1 Experiment Results...")
save_dir = "./data/meta_model_layer1/ireland/"
saved_files = BayesianFootball.Experiments.list_experiments(save_dir, data_dir="")
exp_results = BayesianFootball.Experiments.load_experiment(saved_files, 1)

# ===========================================================================
# 3. CONFIGURE META MODEL
# ===========================================================================
println("\n[3] Configuring ConvexMixtureMetaModel...")
meta_model = MetaModels.ConvexMixtureMetaModel(
    dynamics_config  = MetaModels.MetaGRWDynamicsConfig(σ_prior=0.2),
    hierarchy_config = MetaModels.GlobalMetaHierarchyConfig() 
)

sampler_config = BayesianFootball.Samplers.QueuedNUTSConfig(
    n_samples = 500,
    n_chains  = 2,
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

# ===========================================================================
# 5. PREDICTIVE STAKING & ADVANCED HURDLE EVALUATION
# ===========================================================================
println("\n" * "="^65)
println("  PREDICTIVE STAKING & HURDLE ANALYSIS")
println("="^65)

# min_edge: user specified min edge (e.g. 0.02 = 2%)
min_edge = 0.02

# Compute stakes for the Out of Sample matches
ledger_raw = MetaModels.compute_predictive_stakes(meta_results, meta_results.all_data; min_edge=min_edge)

# Join the full distributions
ledger = innerjoin( 
    ledger_raw, 
    meta_results.all_data[!, [:match_id, :distribution]],
    on = :match_id
)

if nrow(ledger) == 0
    println("No ledger data returned.")
else
    # Run the advanced Hurdle, Sharpe and Growth metrics analysis!
    evaluate_predictive_hurdle(ledger; min_edge=min_edge)
end




#=
# under_15
julia> evaluate_predictive_hurdle(ledger; min_edge=min_edge)

===================================================================================================================
  HURDLE MODEL & PREDICTIVE STAKING METRICS (OOS)
===================================================================================================================
Group                 | Bets | Win%   | AvgStake | EmpROI   | ParamROI | EmpSharpe | ParamSharpe | EmpGrowth | ParamGrowth
-------------------------------------------------------------------------------------------------------------------
L1 Raw (Unfiltered)   | 174  | 32.8%  | 3.74%    | 8.96%    | 8.96%    | 0.0561    | 0.0563      | 0.080%    | 0.163%     
Good Regime (Gated)   | 68   | 30.9%  | 4.19%    | -0.28%   | -0.28%   | -0.0018   | -0.0019     | -0.300%   | -0.207%    
Bad Regime (Skipped)  | 106  | 34.0%  | 3.45%    | 14.89%   | 14.89%   | 0.0907    | 0.0911      | 0.323%    | 0.359%     
-------------------------------------------------------------------------------------------------------------------
Fitted Hurdle Parameters (Bernoulli-Gamma):
  * L1 Raw         : p = 0.3276 | Gamma(α = 18.5932, θ = 0.1251 ) | E[Y] = 2.3262  (ROI if win)
  * Good Regime    : p = 0.3088 | Gamma(α = 19.3882, θ = 0.1150 ) | E[Y] = 2.2290  (ROI if win)
  * Bad Regime     : p = 0.3396 | Gamma(α = 18.7037, θ = 0.1274 ) | E[Y] = 2.3829  (ROI if win)
===================================================================================================================
=#



#=
over_15
julia> evaluate_predictive_hurdle(ledger; min_edge=min_edge)

===================================================================================================================
  HURDLE MODEL & PREDICTIVE STAKING METRICS (OOS)
===================================================================================================================
Group                 | Bets | Win%   | AvgStake | EmpROI   | ParamROI | EmpSharpe | ParamSharpe | EmpGrowth | ParamGrowth
-------------------------------------------------------------------------------------------------------------------
L1 Raw (Unfiltered)   | 157  | 68.2%  | 11.05%   | -5.87%   | -5.87%   | -0.0903   | -0.0906     | -1.082%   | -0.917%    
Good Regime (Gated)   | 41   | 63.4%  | 11.70%   | -12.20%  | -12.20%  | -0.1797   | -0.1820     | -2.139%   | -1.750%    
Bad Regime (Skipped)  | 116  | 69.8%  | 10.82%   | -3.63%   | -3.63%   | -0.0567   | -0.0569     | -0.706%   | -0.643%    
-------------------------------------------------------------------------------------------------------------------
Fitted Hurdle Parameters (Bernoulli-Gamma):
  * L1 Raw         : p = 0.6815 | Gamma(α = 18.6314, θ = 0.0205 ) | E[Y] = 0.3812  (ROI if win)
  * Good Regime    : p = 0.6341 | Gamma(α = 21.0567, θ = 0.0183 ) | E[Y] = 0.3846  (ROI if win)
  * Bad Regime     : p = 0.6983 | Gamma(α = 17.9787, θ = 0.0211 ) | E[Y] = 0.3801  (ROI if win)
===================================================================================================================
=#

#=
under_25
julia> evaluate_predictive_hurdle(ledger; min_edge=min_edge)

===================================================================================================================
  HURDLE MODEL & PREDICTIVE STAKING METRICS (OOS)
===================================================================================================================
Group                 | Bets | Win%   | AvgStake | EmpROI   | ParamROI | EmpSharpe | ParamSharpe | EmpGrowth | ParamGrowth
-------------------------------------------------------------------------------------------------------------------
L1 Raw (Unfiltered)   | 143  | 55.9%  | 6.37%    | 4.08%    | 4.08%    | 0.0433    | 0.0435      | 0.126%    | 0.079%     
Good Regime (Gated)   | 42   | 57.1%  | 4.72%    | 10.51%   | 10.51%   | 0.1071    | 0.1084      | 1.177%    | 0.390%     
Bad Regime (Skipped)  | 101  | 55.4%  | 7.05%    | 1.40%    | 1.40%    | 0.0151    | 0.0152      | -0.308%   | -0.115%    
-------------------------------------------------------------------------------------------------------------------
Fitted Hurdle Parameters (Bernoulli-Gamma):
  * L1 Raw         : p = 0.5594 | Gamma(α = 15.5853, θ = 0.0552 ) | E[Y] = 0.8604  (ROI if win)
  * Good Regime    : p = 0.5714 | Gamma(α = 19.8569, θ = 0.0470 ) | E[Y] = 0.9340  (ROI if win)
  * Bad Regime     : p = 0.5545 | Gamma(α = 15.1955, θ = 0.0545 ) | E[Y] = 0.8288  (ROI if win)
===================================================================================================================
=#


#=

over_25
julia> evaluate_predictive_hurdle(ledger; min_edge=min_edge)

===================================================================================================================
  HURDLE MODEL & PREDICTIVE STAKING METRICS (OOS)
===================================================================================================================
Group                 | Bets | Win%   | AvgStake | EmpROI   | ParamROI | EmpSharpe | ParamSharpe | EmpGrowth | ParamGrowth
-------------------------------------------------------------------------------------------------------------------
L1 Raw (Unfiltered)   | 208  | 43.3%  | 9.71%    | -5.77%   | -5.77%   | -0.0526   | -0.0527     | -0.087%   | -1.117%    
Good Regime (Gated)   | 82   | 41.5%  | 10.30%   | -8.35%   | -8.35%   | -0.0753   | -0.0758     | -1.153%   | -1.495%    
Bad Regime (Skipped)  | 126  | 44.4%  | 9.32%    | -4.09%   | -4.09%   | -0.0374   | -0.0375     | 0.613%    | -0.891%    
-------------------------------------------------------------------------------------------------------------------
Fitted Hurdle Parameters (Bernoulli-Gamma):
  * L1 Raw         : p = 0.4327 | Gamma(α = 18.0623, θ = 0.0652 ) | E[Y] = 1.1778  (ROI if win)
  * Good Regime    : p = 0.4146 | Gamma(α = 20.8080, θ = 0.0582 ) | E[Y] = 1.2103  (ROI if win)
  * Bad Regime     : p = 0.4444 | Gamma(α = 16.9301, θ = 0.0684 ) | E[Y] = 1.1580  (ROI if win)
===================================================================================================================
(l1_raw = (p = 0.4326923076923077, shape = 18.062343886958224, scale = 0.0652062536926995, μ_pos = 1.177777777777778, va
=#

