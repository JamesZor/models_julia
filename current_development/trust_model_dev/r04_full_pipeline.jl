# current_development/trust_model_dev/r04_full_pipeline.jl

using Revise
using BayesianFootball
using DataFrames
using Dates

# Enable fast parallel CPU threads for the Queued NUTS config!
using ThreadPinning; pinthreads(:cores) 

# Include all the L2 structural dependencies we've built
include("l01_basic_main.jl")
include("l02_inference.jl")
include("l03_results_structs.jl")

const Data = BayesianFootball.Data
const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions
const Training = BayesianFootball.Training
const Samplers = BayesianFootball.Samplers
const Markets = BayesianFootball.Data.Markets

println("=========================================================")
println("🚀 INITIALIZING LAYER 2 PIPELINE")
println("=========================================================")

# 1. Load Data
ds = Data.load_datastore_cached(Data.Ireland())

# 2. Load L1 Experiment & Get PPDs
src_dir = "./data/double_poisson_smile_src_grid/"
list_of_experiments = Experiments.list_experiments(src_dir, data_dir="")
l1_experiment = Experiments.load_experiment(list_of_experiments, 5)

println("Generating Base Layer 1 PPDs...")
l1_ppd = Predictions.model_inference(ds, l1_experiment)
l1_ppd_df = l1_ppd.df

# 3. Define the L2 Trust Configuration
const TRUST_MARKETS = AbstractMarket[
    Market1X2(),
    MarketBTTS(),
    MarketOverUnder(0.5),
    MarketOverUnder(1.5),
    MarketOverUnder(2.5),
    MarketOverUnder(3.5),
    MarketOverUnder(4.5)
]

l2_config = TrustModelConfig(
    market_config = Markets.MarketConfig(TRUST_MARKETS)
)

# 4. Process Splits with Warmup Offset
println("\n=========================================================")
println("⚙️  BUILDING LAYER 2 FEATURE COLLECTION")
println("=========================================================")

splitter = l1_experiment.config.splitter
splits = Data.create_id_boundaries(ds, splitter)

l2_warmup_splits = 5 # Skip the first 5 splits so L2 has enough historical L1 predictions to train on
total_splits = length(splits)

println("Total L1 Splits: $total_splits | L2 Warmup Splits: $l2_warmup_splits")

using BayesianFootball.Features
valid_split_indices = (l2_warmup_splits + 1):total_splits

# CRITICAL: Define the static team_map ONCE using all known matches.
# If we calculate this dynamically inside the loop, the alphabetical sort order 
# will change when new teams enter, destroying the Turing chain indexing!
all_global_teams = unique(vcat(ds.matches.home_team, ds.matches.away_team))
global_team_map = Dict(name => i for (i, name) in enumerate(sort(all_global_teams)))

# Initialize an empty array of Tuples for the FeatureCollection
feature_tuples = Tuple{Features.FeatureSet, Any}[]

for s in valid_split_indices
    # To predict split `s`, L2 is allowed to train on the Out-Of-Sample L1 predictions 
    # from split 1 up to split `s-1`.
    historical_l1_match_ids = Int[]
    for prior_s in 1:(s-1)
        append!(historical_l1_match_ids, splits[prior_s][1].target_match_ids)
    end
    # Filter the L1 PPD to only include these historical matches
    l1_history_df = subset(l1_ppd_df, :match_id => ByRow(in(historical_l1_match_ids)))
    
    # Extract the features for the Trust Model
    extractor = TrustDatasetFeature(l1_history_df, TRUST_MARKETS)
    F_data = Dict{Symbol, Any}()
    Features.add_feature!(F_data, extractor, historical_l1_match_ids, global_team_map, ds)
    
    # Inject the static n_teams for the Turing model
    F_data[:n_teams] = length(all_global_teams)
    
    push!(feature_tuples, (Features.FeatureSet(F_data), nothing))
end

feature_sets_for_l2 = Features.FeatureCollection(feature_tuples)
println("Successfully built $(length(feature_sets_for_l2)) Layer 2 FeatureSets!")

# 5. Execute using QueuedNUTSConfig
println("\n=========================================================")
println("🔥 EXECUTING MULTI-THREADED L2 TRAINING QUEUE")
println("=========================================================")

# Use the identical CPU Queueing configuration that L1 uses
sampler_conf = Samplers.QueuedNUTSConfig(
    n_samples = 500, 
    n_chains = 4, 
    n_warmup = 200, 
    accept_rate = 0.65, 
    max_depth = 10,  
    initialisation = Samplers.UniformInit(-2.0, 2.0),
    show_progress = false # Set false for clean multi-threaded logs
)

train_cfg = Training.Independent(
    parallel = true,
    max_concurrent_tasks = Threads.nthreads() # Use all CPU cores available!
)

training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

# This will automatically distribute the N_splits * 4 chains across your CPU threads!
l2_training_results = Training.train(l2_config, training_config, feature_sets_for_l2)

println("\n=========================================================")
println("📦 PACKAGING UNIFIED INFERENCE RESULTS")
println("=========================================================")

# Extract the chains mapping
l2_chains = Dict{Int, Chains}()
for (i, s) in enumerate(valid_split_indices)
    # l2_training_results is an array of (Chains, Metadata) tuples that aligns with our valid_split_indices
    l2_chains[s] = l2_training_results[i][1]
end

trust_results = TrustModelResults(
    l2_config,
    l2_chains,
    Dict(:sampler => "QueuedNUTSConfig", :run_time => string(now()))
)

layered_results = LayeredInferenceResults(
    l1_experiment,
    trust_results,
    l2_warmup_splits
)

println("✅ Layered Inference Results successfully unified!")
println("L2 Calibration is now ready for Backtesting and Signals Generation.")
