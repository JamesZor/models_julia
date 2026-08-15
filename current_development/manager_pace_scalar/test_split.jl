using BayesianFootball
using DataFrames, Dates, Distributions, Statistics, Printf, Serialization, Turing, ReverseDiff

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
cv_config = Data.GroupedCVConfig(
    tournament_groups = [Data.tournament_ids(ds.segment)],
    target_seasons    = ["2025", "2026"],
    history_seasons   = 2,
    dynamics_col      = :match_biweek,
    warmup_period     = 0,
    stop_early        = false
)
splits = Data.create_id_boundaries(ds, cv_config)
println("Split 1: ", length(splits[1][1].history_match_ids), " train, ", length(splits[1][1].target_match_ids), " test")

b, meta = splits[1]
fset = Features.create_features(b, ds, model, :match_biweek)
println("Built features. Building turing model...")
turing_mod = PreGame.build_turing_model(model, fset)

println("Testing sample with AutoReverseDiff...")
try
    ch = sample(
        turing_mod,
        NUTS(20, 0.65),
        20;
        progress = false,
        adtype = AutoReverseDiff(compile=true),
        initial_params = Turing.InitFromUniform(-2.0, 2.0)
    )
    println("SUCCESS! Chain: ", size(ch))
catch e
    println("\n!!! EXCEPTION CAUGHT !!!")
    println(sprint(showerror, e, catch_backtrace()))
end
