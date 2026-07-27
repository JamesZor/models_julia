# current_development/trust_model_dev/r02_training.jl
println("=========================================================")
println("🚀 RUNNING TURING MODEL ON TRUST LAYER (SINGLE SPLIT)")
println("=========================================================")

# 1. Run the data pipe to get the extracted F_data and ordered_ids
include("r01_data_pipe.jl")

println("\n=========================================================")
println("⚙️ Setting up FeatureSet and Sampler Config...")
println("=========================================================")

# 2. Build the exact FeatureSet struct the Training module expects
using BayesianFootball.Features
fset = Features.FeatureSet(F_data)

# 3. Create a Fast Sampler Config for testing
using BayesianFootball.Samplers
using BayesianFootball.Training

# Fast sampler for quick prototyping (100 samples is enough to prove the engine works)
sampler_conf = Samplers.NUTSConfig(
    100, # samples
    4,   # chains
    100, # warmup
    0.65,
    10,  
    Samplers.UniformInit(-2.0, 2.0),
    true # show progress
)
train_cfg = Training.Independent(; parallel=false)
training_config = Training.TrainingConfig(sampler_conf, train_cfg, nothing, false)

# 4. Train!
println("\n🧠 Starting NUTS Sampler via Training.train()...")
println("This will compile the Turing model on the first run, which may take a minute.")
# Note: model_config was instantiated in r01_data_pipe.jl
chain = Training.train(model_config, training_config, fset)

println("\n✅ Training Complete!")
println("=========================================================")
describe(chain)
