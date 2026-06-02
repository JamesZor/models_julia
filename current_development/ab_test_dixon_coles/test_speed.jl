# current_development/ab_test_dixon_coles/test_speed.jl

using Revise
using BayesianFootball
using Turing
using ReverseDiff
using BenchmarkTools

const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Data = BayesianFootball.Data

# 1. Load tiny dataset (just first few rows to build structure)
println("[INFO] Loading Ireland DataStore...")
ds = Data.load_datastore_cached(Data.Ireland())
    split_data = filter(row -> row.season == 2026, ds.matches)

# 2. Config
model_cfg = PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel(
    interception_config    = PreGame.GlobalInterception(),
    player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0),
    dispersion_config      = PreGame.HomeAwayDispersion(),
    homeadvantage_config   = PreGame.HierarchicalTeamHomeAdvantage(),
    kappa_config           = PreGame.HierarchicalTeamKappa(),
    player_ratings_feature = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)),
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    market_weight          = 0.4
)

# 3. Build Model
println("[INFO] Building features and Turing model...")
feature_set = Features.build_feature_set(model_cfg, split_data)
model = PreGame.build_turing_model(model_cfg, feature_set)

# 4. Generate Random Parameters for testing
vi = Turing.VarInfo(model)
model(vi) # init

θ = vi[Turing.SampleFromPrior()]
f = x -> Turing.getlogp(model(vi, Turing.SampleFromPrior(), x))

# 5. Test ReverseDiff Tape Compilation and Execution
println("[INFO] Compiling ReverseDiff Tape (compile=true)...")
const tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, θ))

println("[INFO] Timing compiled tape execution...")
time_eval = @belapsed ReverseDiff.gradient!($(similar(θ)), $tape, $θ)
println("Gradient Evaluation Time: ", round(time_eval * 1000, digits=2), " ms")
