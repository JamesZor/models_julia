# current_development/match_day_inference/runner.jl

# ==========================================
# 1. ENVIRONMENT & IMPORTS
# ==========================================
using Pkg
Pkg.activate(".")

using Revise
# Load our mini-module loader
include("loader.jl")

# Setup CPU thread pinning
using ThreadPinning
pinthreads(:cores)

# ==========================================
# 2. DATASTORE & EXPERIMENT LOADING
# ==========================================
println("\n=== 1. Loading Datastore ===")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

# Set directory where SofaScore JSON lineups are placed (Strategy A)
json_lineups_dir = "./current_development/match_day_inference/data/lineups"
mkpath(json_lineups_dir)

# ==========================================
# 3. CHOOSE EXPERIMENT METHOD
# ==========================================
println("\n=== 2. Resolving Model / Experiment ===")

# OPTION A: Load a previously saved MCMC experiment
# (Highly recommended for player-level models to avoid long sampling times during inference)
# example:
# saved_exp_dir = "./data/matchday_inference_1/2026-05-22_XGMarketPlayerTimeDecay"
# if isdir(saved_exp_dir)
#     expr = Experiments.load_experiment(saved_exp_dir)
# else
#     ...
# end

# For demonstration / safety fallback, we define a small training setup here:
save_dir = "./data/matchday_inference_runner_temp/"
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

model = PreGame.DynamicMarketXGPlayerTimeDecayModel(
    interception_config  = inter_cfg,
    player_dynamics_config = PreGame.PositionalPlayerDynamics(days_half_life=180.0),
    dispersion_config    = disp_cfg,
    homeadvantage_config = ha_cfg,
    kappa_config         = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight        = 1.0
)

# We find the target season and target split
target_seasons = get_target_seasons_string(ds)
dynamics_col = :match_month

# Warmup to target the most recent completed split
warmup_period = last(unique(subset(ds.matches, :season => ByRow(isequal(target_seasons[1])))[dynamics_col]))

println("Creating a quick experiment task for the latest split (warmup_period = $warmup_period)...")
task = Experiments.create_experiment_task(
    ds, 
    model, 
    "runner_test_inference", 
    save_dir; 
    target_seasons=target_seasons,
    history_seasons = 3,
    dynamics_col=dynamics_col,
    samples=100,      # Small samples for fast runner testing
    warmup=50,        # Small warmup for fast runner testing
    chains=2,         # 2 chains for fast runner testing
    max_concurrent_splits = 1
)

println("Running model fit...")
expr = Experiments.run_experiment(task)
Experiments.save_experiment(expr)

# ==========================================
# 4. FETCH TODAY'S MATCHES
# ==========================================
println("\n=== 3. Fetching Today's Fixtures ===")

local todays_matches
try
    todays_matches = fetch_todays_matches(ds)
    println("Successfully fetched ", nrow(todays_matches), " matches starting today from PostgreSQL.")
catch e
    @warn "Failed to query database for today's matches. Using mock matches for runner test. Error: $e"
    # Fallback mock for testing in case DB is unreachable
    todays_matches = DataFrame(
        match_id = [15238092, 15238093],
        home_team = ["derry-city", "drogheda-united"],
        away_team = ["st-patricks-athletic", "dundalk-fc"],
        round = [18, 18],
        tournament_id = [79, 79],
        season_id = [87682, 87682],
        match_week = [999, 999],
        match_date = [today(), today()]
    )
end

show(todays_matches)
println()

# ==========================================
# 5. RUN MATCH DAY INFERENCE (PPD Generation)
# ==========================================
println("\n=== 4. Running Match Day Inference ===")
ppd = compute_todays_matches_pdds(ds, expr, todays_matches, json_lineups_dir)

println("\nRaw Model 1X2 Probabilities:")
show(subset(ppd.df, :market_name => ByRow(==("1X2"))))
println()

# ==========================================
# 6. BETFAIR LIVE STREAM & KELLY STAKING
# ==========================================
println("\n=== 5. Connecting to Redis & Kelly Staking ===")
redis_host = get(ENV, "REDIS_HOST", "127.0.0.1")
redis_port = parse(Int, get(ENV, "REDIS_PORT", "6379"))

try
    println("🔗 Connecting to Redis at $redis_host:$redis_port...")
    redis_conn = RedisConnection(host=redis_host, port=redis_port)
    
    # Run the live betting dashboard (one-shot display)
    print_live_betting_dashboard(ppd, redis_conn, todays_matches; kelly_fraction=0.5, min_edge=0.02)
    
    # Example polling loop (uncomment to run interactive live poller):
    # println("Press Ctrl+C to exit live betting monitor.")
    # while true
    #     print("\e[2J\e[H") # Clear terminal
    #     print_live_betting_dashboard(ppd, redis_conn, todays_matches; kelly_fraction=0.5, min_edge=0.02)
    #     sleep(5.0)
    # end
catch e
    @warn "Redis connection skipped or failed. Run your local Redis and Betfair light streamer to view live betting dashboard. Error: $e"
end
