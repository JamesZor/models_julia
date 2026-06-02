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

# 2. Config
model_cfg = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
    interception_config    = PreGame.GlobalInterception(),
    player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0),
    dispersion_config      = PreGame.HomeAwayDispersion(),
    homeadvantage_config   = PreGame.HierarchicalTeamHomeAdvantage(),
    kappa_config           = PreGame.HierarchicalTeamKappa(),
    dixon_coles_config     = PreGame.HierarchicalTeamDixonColesConfig(),
    player_ratings_feature = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)),
    market_feature_config  = Features.DixonColesMarketFeature(),
    market_weight          = 0.4
)

# 3. Create temporal boundaries
cv_config = Data.GroupedCVConfig(
    tournament_groups = [[79]], # Ireland
    target_seasons = ["2026"],
    history_seasons = 1,
    dynamics_col = :match_month
)

boundaries = Data.create_id_boundaries(ds, cv_config)

# 4. Build Model
println("[INFO] Building features and Turing model...")
feature_collection = Features.create_features(boundaries, ds, model_cfg)
feature_set = feature_collection[1][2] # tuple is (SubDataFrame, FeatureSet) typically, or just FeatureSet? Let's check. 
# In README: feature_collection = Features.create_features(boundaries, ds, test_model). We'll assume feature_collection is Vector{FeatureSet} or similar, actually README doesn't specify. Wait, README says: feature_collection = Features.create_features(boundaries, ds, test_model).
# Let's see what is inside it. Usually it's Vector{FeatureSet}. Let's just do feature_set = feature_collection[1]
model = PreGame.build_turing_model(model_cfg, feature_collection[1])

# 5. Generate Random Parameters for testing
using DynamicPPL
using LogDensityProblems

vi = DynamicPPL.VarInfo(model)
model(vi) # init

θ = vi[:]
lf = DynamicPPL.LogDensityFunction(model)
f = x -> LogDensityProblems.logdensity(lf, x)

# 6. Test ReverseDiff Tape Compilation and Execution
println("[INFO] Compiling ReverseDiff Tape (compile=true)...")
const tape = ReverseDiff.compile(ReverseDiff.GradientTape(f, θ))

println("[INFO] Timing compiled tape execution...")
time_eval = @belapsed ReverseDiff.gradient!($(similar(θ)), $tape, $θ)
println("Gradient Evaluation Time: ", round(time_eval * 1000, digits=2), " ms")
