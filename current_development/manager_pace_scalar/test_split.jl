using BayesianFootball
using DataFrames, Dates, Distributions, Statistics, Printf, Serialization

include("l01_manager_pace_data.jl")
include("l02_manager_pace_engine.jl")
include("l03_manager_pace_predict.jl")

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Data        = BayesianFootball.Data

OUT_DIR = "./data/l2_ireland_engines"
PIN_PATH = joinpath(OUT_DIR, "ds_ire79.jls")
ds = deserialize(PIN_PATH)

function make_manager_pace_scalar_engine()
    return DynamicSmileDoublePoissonXGWealthManagerPacePlayerTimeDecayModel(
        interception_config    = PreGame.HierarchicalMonthlyInterception(),
        player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(days_half_life = 60.0),
        homeadvantage_config   = PreGame.HierarchicalTeamHomeAdvantage(),
        kappa_config           = PreGame.HierarchicalTeamKappa(),
        player_ratings_feature = Features.PlayerRatingsFeature(
            Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
        ),
        wealth_feature         = TeamWealthFeature(),
        manager_pace_feature   = ManagerPaceFeature(pseudo_matches = 15.0),
        w_wealth_prior         = truncated(Normal(0.105, 0.05), lower=0.0),
        w_pace_prior           = truncated(Normal(0.05, 0.03), lower=0.0),
        market_feature_config  = Features.DoublePoissonMarketFeature(),
        smile_feature          = Features.MarketSmileFeature(Kmax = 4),
        market_on              = false,
        supremacy_weight       = 0.4,
        smile_weight           = 0.4,
    )
end

model = make_manager_pace_scalar_engine()
task = Experiments.create_experiment_task(
    ds, model, "test_split", OUT_DIR;
    target_seasons        = ["2025", "2026"],
    history_seasons       = 2,
    warmup_period         = 0,
    dynamics_col          = :match_biweek,
    samples               = 50,
    warmup                = 20,
    chains                = 2,
    use_queue             = true,
    max_concurrent_tasks  = 2,
)

println("Running test task...")
try
    res = Experiments.run_experiment(task)
    println("Completed! Splits saved: ", length(res.training_results.items))
    if length(res.training_results.items) > 0 && !isnothing(res.training_results.items[1])
        println("Split 1 item type: ", typeof(res.training_results.items[1]))
    else
        println("Split 1 item is NOTHING!")
    end
catch e
    @error "Task failed" exception=(e, catch_backtrace())
end
