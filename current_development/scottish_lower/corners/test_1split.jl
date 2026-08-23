using Revise
using BayesianFootball
using Turing, DynamicPPL

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Training    = BayesianFootball.Training
const Experiments = BayesianFootball.Experiments
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include("l01_corner_data.jl")
include("l05_corner_recomb_pipeline.jl")

ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)

splitter = Data.GroupedCVConfig(
    tournament_groups = [[56, 57]],
    target_seasons    = ["24/25", "25/26"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = 0,
    stop_early        = true
)
boundaries = Data.create_id_boundaries(ds, splitter)
b1 = boundaries[1]

model_spec = TeamGoalsCornerRecombIntegratedModel(
    dynamics_config      = PreGame.TimeDecayDynamics(days_half_life = 365.0),
    interception_config  = PreGame.GlobalInterception(),
    homeadvantage_config = PreGame.GlobalHomeAdvantage(),
    name                 = "recomb_corner_integrated_hl365_hs2"
)

features = Features.create_features(b1[1], ds, model_spec)
println("Features created successfully. Keys: ", keys(features.data))
println("Building Turing model...")
t_mod = PreGame.build_turing_model(model_spec, features)
println("Evaluating t_mod()...")
try
    t_mod()
    println("✓ t_mod() evaluated successfully!")
catch e
    @error "Error in t_mod()" exception=(e, catch_backtrace())
end
