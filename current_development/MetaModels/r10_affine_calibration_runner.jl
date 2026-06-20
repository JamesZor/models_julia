# current_development/MetaModels/r10_affine_calibration_runner.jl
#
# ===========================================================================
# AFFINE CALIBRATION RUNNER (M x N Joint Distribution)
# ===========================================================================
# This runner implements the fully Bayesian Affine Calibration model.
#
# Unlike the Convex Mixture model which uses a "Regime Gate", this model
# calculates a fully calibrated probability for every single match.
# It does this by executing a mathematically rigorous M x N convolution:
# for each of the M samples from the Layer 1 predictive distribution,
# it applies N affine shifts from the Layer 2 calibration posterior.
#
# The resulting M x N matrix perfectly captures both the epistemic 
# uncertainty of the match dynamics (L1) and the epistemic uncertainty 
# of the systemic calibration (L2), allowing the Kelly module to shrink 
# the bet sizes accordingly.
# ===========================================================================

using Pkg; Pkg.activate(".")
using BayesianFootball
using DataFrames
using Dates
using Serialization
using ThreadPinning

pinthreads(:cores)

println("===========================================================================")
println("  INITIALIZING M x N AFFINE CALIBRATION WORKFLOW")
println("===========================================================================")

# ===========================================================================
# 1. LOAD LAYER 1 RESULTS
# ===========================================================================
L1_EXP_DIR = "./current_development/save_dir_xg_dynamic"
println("\n[1] Loading Base L1 Experiment Results...")
exp_results = BayesianFootball.Experiments.load_experiment(L1_EXP_DIR)

# ===========================================================================
# 2. LOAD DATASTORE & ODDS
# ===========================================================================
println("\n[2] Loading Base DataStore...")
ds_raw = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

println("    -> Computing SHARP Anchor Odds (-10 to 0 mins)...")
betfair_sharp = BayesianFootball.Data.summarize_betfair_market(
    ds_raw, 
    open_window=(-100000.0, -10.0), 
    close_window=(-10.0, 0.0)
)

ds = BayesianFootball.Data.DataStore(
    ds_raw.segment, ds_raw.matches, ds_raw.statistics,
    betfair_sharp, 
    ds_raw.lineups, ds_raw.incidents, ds_raw.betfair_odds
)

println("    -> Extracting LOOSE Staking Odds (Soft Bookmaker)...")
# Since we proved Bet365/Sofascore is highly liquid and profitable to pick off:
loose_staking_odds = ds_raw.odds 

# ===========================================================================
# 3. CONFIGURE AFFINE META MODEL
# ===========================================================================
println("\n[3] Configuring Affine Calibration Model...")
using BayesianFootball.MetaModels

meta_config = AffineCalibrationMetaModel(
    dynamics_config  = BayesianFootball.MetaModels.MetaGRWDynamicsConfig(),
    hierarchy_config = BayesianFootball.MetaModels.HierarchicalMetaTeamConfig()
)

sampler_config = BayesianFootball.Samplers.QueuedNUTSConfig(
    samples=500,
    warmup=200,
    max_concurrent_tasks=Threads.nthreads()
)

# ===========================================================================
# 4. RUN MULTI-MARKET EXPERIMENTS
# ===========================================================================
TARGET_SELECTIONS = [:under_25, :over_25]
min_edge = 0.00

println("\n[4] Running M x N Calibration across $(length(TARGET_SELECTIONS)) markets...")
multi_market_results, multi_market_ledgers = MetaModels.run_multi_market_experiments(
    TARGET_SELECTIONS,
    exp_results,
    meta_config,
    sampler_config,
    ds;
    min_edge=min_edge,
    staking_odds=loose_staking_odds
)

# ===========================================================================
# 5. PERSIST RESULTS TO DISK
# ===========================================================================
save_dir = "./current_development/save_dir_affine_calibration"
mkpath(save_dir)
println("\n[5] Serializing results to disk...")
serialize(joinpath(save_dir, "multi_market_results.jls"), multi_market_results)
serialize(joinpath(save_dir, "multi_market_ledgers.jls"), multi_market_ledgers)
println("    Successfully saved to: $save_dir")

# ===========================================================================
# 6. STREAMLINED REPORTING
# ===========================================================================
println("\n" * "="^75)
println("  GENERATING CONSOLIDATED HURDLE REPORTS")
println("="^75)

markets_evaluated = collect(keys(multi_market_ledgers))
final_metrics = MetaModels.evaluate_multiple_markets(multi_market_ledgers, markets_evaluated; min_edge=min_edge)

println("\nRunner finished successfully.")
