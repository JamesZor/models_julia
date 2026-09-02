
using BayesianFootball
using DataFrames
using Turing
using MCMCChains

using ThreadPinning

pinthreads(:cores)

const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Diagnostics = BayesianFootball.Experiments.Diagnostics
const Evaluation = BayesianFootball.Evaluation


ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

save_dir = "./data/match_daya/"


inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()

dyn_cfg = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=45.0)

tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

model = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    dixon_coles_config     = PreGame.HierarchicalTeamDixonColesConfig(),
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DixonColesMarketFeature(),
    market_weight          = 0.4
)


task = Experiments.create_experiment_task(
    ds, 
    model, 
    "dixon_coles_12_06_26", 
    save_dir; 
    target_seasons=["2026"],
    history_seasons = 2,
    warmup_period =  20,
    dynamics_col=:match_week,
    samples=2000,      # Small samples for fast runner testing
    warmup=1000,        # Small warmup for fast runner testing
    chains=8,         # 2 chains for fast runner testing
    use_queue=true
)

#
results = Experiments.run_experiment(task)
# Experiments.save_experiment(results)
# #
# #
# Experiments.save_experiment(results)
#

# ==========================================
# DIAGNOSTICS
# ==========================================
chains_df_all = Diagnostics.extract_chains(ds, results)

println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv_diag_all = Diagnostics.check_convergence(chains_df_all)

println("\n--- Temporal Stability Diagnostics (ADF Stationarity) ---")
stab_diag_all = Diagnostics.check_stability(chains_df_all)

metrics = [
    Evaluation.LogLoss(), 
    Evaluation.CRPS(), 
    Evaluation.RQR(),
    Evaluation.GLMEdge()
]

master_eval_df = Evaluation.evaluate_experiments(metrics, [results], ds)

println("\n>>> Evaluation Results (Sorted by LogLoss):")
display(sort(master_eval_df, :logloss_overall_diff_ll))

Evaluation.display_summary_metric(master_eval_df, :logloss)
Evaluation.display_summary_metric(master_eval_df, :glmedge)
Evaluation.display_summary_metric(master_eval_df, :rqr)

# ==========================================
# 4. FETCH TODAY'S MATCHES
# ==========================================
include("./current_development/match_day_inference/loader.jl")
saved_fiels = Experiments.list_experiments(save_dir, data_dir="")
expr = Experiments.load_experiment(saved_fiels, 1)

# ==========================================
# 4. FETCH TODAY'S MATCHES
# ==========================================
println("\n=== 3. Fetching Today's Fixtures ===")

local todays_matches

todays_matches = fetch_todays_matches(ds)
show(todays_matches)
println()


json_lineups_dir = "./current_development/match_day_inference/data/lineups"
# ==========================================
# 5. RUN MATCH DAY INFERENCE (PPD Generation)
# ==========================================
println("\n=== 4. Running Match Day Inference ===")
ppd = compute_todays_matches_pdds(ds, expr, todays_matches, json_lineups_dir)

ppd = Predictions.model_inference(ds, expr)

println("\nRaw Model 1X2 Probabilities:")
show(subset(ppd.df, :market_name => ByRow(==("1X2"))))
println()

# ==========================================
# 6. BETFAIR LIVE STREAM & KELLY STAKING
# ==========================================
println("\n=== 5. Connecting to Redis & Kelly Staking ===")
redis_host = get(ENV, "REDIS_HOST", "127.0.0.1")
redis_host = get(ENV, "REDIS_HOST", "100.124.38.117")
redis_port = parse(Int, get(ENV, "REDIS_PORT", "6379"))

try
    println("🔗 Connecting to Redis at $redis_host:$redis_port...")
    redis_conn = RedisConnection(host=redis_host, port=redis_port)
    
    # Run the live betting dashboard (one-shot display)
    print_live_betting_dashboard(ppd, redis_conn, todays_matches; kelly_fraction=0.00, min_edge=0.00)
    
    # Example polling loop (uncomment to run interactive live poller):
    # println("Press Ctrl+C to exit live betting monitor.")
    # while true
    #     print("\e[2J\e[H") # Clear terminal
    #     print_live_betting_dashboard(ppd, redis_conn, todays_matches; kelly_fraction=0.5, min_edge=0.02)
    #     sleep(60.0)
    # end
catch e
    @warn "Redis connection skipped or failed. Run your local Redis and Betfair light streamer to view live betting dashboard. Error: $e"
end



# Extract MCMC chains into long-format dataframe
#
chains_df_all = Diagnostics.extract_chains(ds, expr)

println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv_diag_all = Diagnostics.check_convergence(chains_df_all)


#=
julia> print_live_betting_dashboard(ppd, redis_conn, todays_matches; kelly_fraction=0.00, min_edge=0.00)

=============================================================================================================================
 LIVE MATCHDAY BETTING DASHBOARD | Kelly: 0.0 | Min Edge: 0.0 | 19:00:48
=============================================================================================================================

> shamrock-rovers vs st-patricks-athletic (ID: 15238103) [$]
╭─────────┬─────────────┬─────────┬───────┬─────────────┬───────┬───────┬───────┬─────────┬───────┬─────────╮
│ Market  │ Selection   │ Model % │ Mid % │ Model Price │   Mid │  Back │   Lay │      EV │ Kelly │ Bayes K │
├─────────┼─────────────┼─────────┼───────┼─────────────┼───────┼───────┼───────┼─────────┼───────┼─────────┤
│ 1X2     │ * Away      │   34.5% │ 32.8% │        2.90 │  3.05 │  3.00 │  3.10 │   +3.4% │  ---- │   0.32% │
│         │ * Home      │   42.0% │ 37.6% │        2.38 │  2.66 │  2.62 │  2.70 │   +9.9% │  ---- │   2.74% │
│         │   Draw      │   23.6% │ 29.6% │        4.24 │  3.38 │  3.35 │  3.40 │  -21.0% │  ---- │    ---- │
│ O/U 0.5 │   Under 0.5 │    5.5% │  8.5% │       18.33 │ 11.75 │ 11.00 │ 12.50 │  -40.0% │  ---- │    ---- │
│         │ * Over 0.5  │   94.5% │ 91.3% │        1.06 │  1.10 │  1.09 │  1.10 │   +3.1% │  ---- │  25.16% │
│ O/U 1.5 │ * Over 1.5  │   79.4% │ 71.9% │        1.26 │  1.39 │  1.38 │  1.40 │   +9.6% │  ---- │  17.89% │
│         │   Under 1.5 │   20.6% │ 28.2% │        4.86 │  3.55 │  3.50 │  3.60 │  -28.0% │  ---- │    ---- │
│ O/U 2.5 │ * Over 2.5  │   57.8% │ 46.1% │        1.73 │  2.17 │  2.14 │  2.20 │  +23.6% │  ---- │  15.39% │
│         │   Under 2.5 │   42.2% │ 53.9% │        2.37 │  1.85 │  1.84 │  1.87 │  -22.3% │  ---- │    ---- │
│ O/U 3.5 │   Under 3.5 │   63.6% │ 75.2% │        1.57 │  1.33 │  1.32 │  1.34 │  -16.1% │  ---- │    ---- │
│         │ * Over 3.5  │   36.4% │ 24.5% │        2.74 │  4.08 │  3.95 │  4.20 │  +44.0% │  ---- │  11.30% │
│ O/U 4.5 │   Under 4.5 │   79.8% │ 88.9% │        1.25 │  1.12 │  1.12 │  1.13 │  -10.6% │  ---- │    ---- │
│         │ * Over 4.5  │   20.2% │ 10.8% │        4.95 │  9.30 │  9.00 │  9.60 │  +81.7% │  ---- │   8.16% │
│ O/U 5.5 │   Under 5.5 │   90.0% │ 96.2% │        1.11 │  1.04 │  1.03 │  1.05 │   -7.3% │  ---- │    ---- │
│         │ * Over 5.5  │   10.0% │  4.1% │       10.03 │ 24.50 │ 22.00 │ 27.00 │ +119.4% │  ---- │   4.54% │
│ BTTS    │ * Yes       │   58.8% │ 53.2% │        1.70 │  1.88 │  1.85 │  1.91 │   +8.8% │  ---- │   5.60% │
│         │   No        │   41.2% │ 46.7% │        2.43 │  2.14 │  2.10 │  2.18 │  -13.5% │  ---- │    ---- │
╰─────────┴─────────────┴─────────┴───────┴─────────────┴───────┴───────┴───────┴─────────┴───────┴─────────╯

> sligo-rovers vs bohemian (ID: 15238102) [$]
╭─────────┬─────────────┬─────────┬───────┬─────────────┬───────┬───────┬───────┬─────────┬───────┬─────────╮
│ Market  │ Selection   │ Model % │ Mid % │ Model Price │   Mid │  Back │   Lay │      EV │ Kelly │ Bayes K │
├─────────┼─────────────┼─────────┼───────┼─────────────┼───────┼───────┼───────┼─────────┼───────┼─────────┤
│ 1X2     │   Away      │   34.0% │ 61.9% │        2.94 │  1.62 │  1.61 │  1.62 │  -45.2% │  ---- │    ---- │
│         │ * Home      │   42.3% │ 14.7% │        2.36 │  6.80 │  6.60 │  7.00 │ +179.3% │  ---- │  31.13% │
│         │ * Draw      │   23.6% │ 23.0% │        4.23 │  4.35 │  4.30 │  4.40 │   +1.6% │  ---- │   0.09% │
│ O/U 0.5 │   Under 0.5 │    5.5% │  6.7% │       18.07 │ 15.00 │ 14.50 │ 15.50 │  -19.8% │  ---- │    ---- │
│         │ * Over 0.5  │   94.5% │ 93.0% │        1.06 │  1.08 │  1.07 │  1.08 │   +1.1% │  ---- │   6.86% │
│ O/U 1.5 │ * Over 1.5  │   79.2% │ 75.5% │        1.26 │  1.33 │  1.32 │  1.33 │   +4.5% │  ---- │   6.99% │
│         │   Under 1.5 │   20.8% │ 24.4% │        4.80 │  4.10 │  4.00 │  4.20 │  -16.7% │  ---- │    ---- │
│ O/U 2.5 │ * Over 2.5  │   57.4% │ 50.6% │        1.74 │  1.98 │  1.96 │  1.99 │  +12.5% │  ---- │   7.45% │
│         │   Under 2.5 │   42.6% │ 49.3% │        2.35 │  2.03 │  2.00 │  2.06 │  -14.8% │  ---- │    ---- │
│ O/U 3.5 │   Under 3.5 │   64.0% │ 71.7% │        1.56 │  1.40 │  1.38 │  1.41 │  -11.7% │  ---- │    ---- │
│         │ * Over 3.5  │   36.0% │ 28.2% │        2.78 │  3.55 │  3.50 │  3.60 │  +26.0% │  ---- │   6.60% │
│ O/U 4.5 │   Under 4.5 │   80.2% │ 87.0% │        1.25 │  1.15 │  1.14 │  1.16 │   -8.6% │  ---- │    ---- │
│         │ * Over 4.5  │   19.8% │ 13.0% │        5.05 │  7.70 │  7.40 │  8.00 │  +46.7% │  ---- │   4.95% │
│ O/U 5.5 │   Under 5.5 │   90.3% │ 94.8% │        1.11 │  1.06 │  1.05 │  1.06 │   -5.2% │  ---- │    ---- │
│         │ * Over 5.5  │    9.7% │  5.1% │       10.30 │ 19.50 │ 18.00 │ 21.00 │  +74.8% │  ---- │   3.12% │
│ BTTS    │ * Yes       │   58.5% │ 49.0% │        1.71 │  2.04 │  2.00 │  2.08 │  +16.9% │  ---- │  12.90% │
│         │   No        │   41.5% │ 51.0% │        2.41 │  1.96 │  1.93 │  1.99 │  -19.9% │  ---- │    ---- │
╰─────────┴─────────────┴─────────┴───────┴─────────────┴───────┴───────┴───────┴─────────┴───────┴─────────╯

> drogheda-united vs waterford-fc (ID: 15238099) [$]
╭─────────┬─────────────┬─────────┬───────┬─────────────┬───────┬───────┬───────┬────────┬───────┬─────────╮
│ Market  │ Selection   │ Model % │ Mid % │ Model Price │   Mid │  Back │   Lay │     EV │ Kelly │ Bayes K │
├─────────┼─────────────┼─────────┼───────┼─────────────┼───────┼───────┼───────┼────────┼───────┼─────────┤
│ 1X2     │ * Away      │   28.3% │ 21.5% │        3.54 │  4.65 │  4.50 │  4.80 │ +27.2% │  ---- │   5.20% │
│         │   Home      │   48.8% │ 52.5% │        2.05 │  1.90 │  1.89 │  1.92 │  -7.9% │  ---- │    ---- │
│         │   Draw      │   23.0% │ 25.6% │        4.35 │  3.90 │  3.85 │  3.95 │ -11.6% │  ---- │    ---- │
│ O/U 0.5 │   Under 0.5 │    5.5% │  6.9% │       18.22 │ 14.50 │ 14.00 │ 15.00 │ -23.2% │  ---- │    ---- │
│         │ * Over 0.5  │   94.5% │ 93.0% │        1.06 │  1.08 │  1.07 │  1.08 │  +1.1% │  ---- │   7.08% │
│ O/U 1.5 │ * Over 1.5  │   79.3% │ 75.2% │        1.26 │  1.33 │  1.32 │  1.34 │  +4.7% │  ---- │   7.24% │
│         │   Under 1.5 │   20.7% │ 25.0% │        4.84 │  4.00 │  3.90 │  4.10 │ -19.5% │  ---- │    ---- │
│ O/U 2.5 │ * Over 2.5  │   57.7% │ 50.4% │        1.73 │  1.98 │  1.97 │  2.00 │ +13.6% │  ---- │   8.14% │
│         │   Under 2.5 │   42.3% │ 49.6% │        2.36 │  2.02 │  1.99 │  2.04 │ -15.8% │  ---- │    ---- │
│ O/U 3.5 │   Under 3.5 │   63.6% │ 71.9% │        1.57 │  1.39 │  1.38 │  1.40 │ -12.2% │  ---- │    ---- │
│         │ * Over 3.5  │   36.4% │ 28.0% │        2.75 │  3.58 │  3.50 │  3.65 │ +27.3% │  ---- │   6.80% │
│ O/U 4.5 │   Under 4.5 │   79.8% │ 87.0% │        1.25 │  1.15 │  1.14 │  1.16 │  -9.0% │  ---- │    ---- │
│         │ * Over 4.5  │   20.2% │ 12.7% │        4.96 │  7.90 │  7.60 │  8.20 │ +53.2% │  ---- │   5.56% │
│ O/U 5.5 │   Under 5.5 │   90.0% │ 94.8% │        1.11 │  1.06 │  1.05 │  1.06 │  -5.5% │  ---- │    ---- │
│         │ * Over 5.5  │   10.0% │  5.1% │       10.04 │ 19.50 │ 18.00 │ 21.00 │ +79.2% │  ---- │   3.25% │
│ BTTS    │ * Yes       │   57.6% │ 53.3% │        1.74 │  1.88 │  1.85 │  1.90 │  +6.5% │  ---- │   3.39% │
│         │   No        │   42.4% │ 46.5% │        2.36 │  2.15 │  2.12 │  2.18 │ -10.1% │  ---- │    ---- │
╰─────────┴─────────────┴─────────┴───────┴─────────────┴───────┴───────┴───────┴────────┴───────┴─────────╯

> dundalk-fc vs derry-city (ID: 15238100) [$]
╭─────────┬─────────────┬─────────┬───────┬─────────────┬───────┬───────┬───────┬─────────┬───────┬─────────╮
│ Market  │ Selection   │ Model % │ Mid % │ Model Price │   Mid │  Back │   Lay │      EV │ Kelly │ Bayes K │
├─────────┼─────────────┼─────────┼───────┼─────────────┼───────┼───────┼───────┼─────────┼───────┼─────────┤
│ 1X2     │   Away      │   31.3% │ 32.8% │        3.20 │  3.05 │  3.00 │  3.10 │   -6.1% │  ---- │    ---- │
│         │ * Home      │   45.3% │ 37.9% │        2.21 │  2.64 │  2.62 │  2.66 │  +18.7% │  ---- │   7.49% │
│         │   Draw      │   23.4% │ 29.9% │        4.27 │  3.35 │  3.30 │  3.40 │  -22.8% │  ---- │    ---- │
│ O/U 0.5 │   Under 0.5 │    5.5% │  8.9% │       18.05 │ 11.25 │ 10.50 │ 12.00 │  -41.8% │  ---- │    ---- │
│         │ * Over 0.5  │   94.5% │ 90.9% │        1.06 │  1.10 │  1.09 │  1.11 │   +3.0% │  ---- │  24.48% │
│ O/U 1.5 │ * Over 1.5  │   79.2% │ 70.4% │        1.26 │  1.42 │  1.41 │  1.43 │  +11.6% │  ---- │  21.88% │
│         │   Under 1.5 │   20.8% │ 29.4% │        4.80 │  3.40 │  3.35 │  3.45 │  -30.2% │  ---- │    ---- │
│ O/U 2.5 │ * Over 2.5  │   57.4% │ 43.9% │        1.74 │  2.28 │  2.26 │  2.30 │  +29.7% │  ---- │  19.08% │
│         │   Under 2.5 │   42.6% │ 55.9% │        2.35 │  1.79 │  1.78 │  1.80 │  -24.2% │  ---- │    ---- │
│ O/U 3.5 │   Under 3.5 │   64.0% │ 77.2% │        1.56 │  1.29 │  1.28 │  1.31 │  -18.1% │  ---- │    ---- │
│         │ * Over 3.5  │   36.0% │ 23.0% │        2.77 │  4.35 │  4.20 │  4.50 │  +51.4% │  ---- │  12.97% │
│ O/U 4.5 │   Under 4.5 │   80.1% │ 90.1% │        1.25 │  1.11 │  1.10 │  1.12 │  -11.8% │  ---- │    ---- │
│         │ * Over 4.5  │   19.9% │  9.6% │        5.04 │ 10.40 │  9.80 │ 11.00 │  +94.5% │  ---- │   9.04% │
│ O/U 5.5 │   Under 5.5 │   90.3% │ 96.2% │        1.11 │  1.04 │  1.03 │  1.05 │   -7.0% │  ---- │    ---- │
│         │ * Over 5.5  │    9.7% │  3.7% │       10.27 │ 27.00 │ 24.00 │ 30.00 │ +133.8% │  ---- │   4.86% │
│ BTTS    │ * Yes       │   58.0% │ 51.9% │        1.72 │  1.92 │  1.89 │  1.96 │   +9.7% │  ---- │   6.37% │
│         │   No        │   42.0% │ 48.1% │        2.38 │  2.08 │  2.04 │  2.12 │  -14.4% │  ---- │    ---- │
╰─────────┴─────────────┴─────────┴───────┴─────────────┴───────┴───────┴───────┴─────────┴───────┴─────────╯

> shelbourne vs galway-united (ID: 15238101) [$]
╭─────────┬─────────────┬─────────┬───────┬─────────────┬───────┬───────┬───────┬─────────┬───────┬─────────╮
│ Market  │ Selection   │ Model % │ Mid % │ Model Price │   Mid │  Back │   Lay │      EV │ Kelly │ Bayes K │
├─────────┼─────────────┼─────────┼───────┼─────────────┼───────┼───────┼───────┼─────────┼───────┼─────────┤
│ 1X2     │ * Away      │   22.6% │ 18.3% │        4.43 │  5.45 │  5.40 │  5.50 │  +21.8% │  ---- │   2.67% │
│         │   Home      │   56.1% │ 56.8% │        1.78 │  1.76 │  1.75 │  1.77 │   -1.8% │  ---- │    ---- │
│         │   Draw      │   21.3% │ 24.8% │        4.68 │  4.03 │  3.95 │  4.10 │  -15.7% │  ---- │    ---- │
│ O/U 0.5 │   Under 0.5 │    5.0% │  6.8% │       20.05 │ 14.75 │ 14.00 │ 15.50 │  -30.2% │  ---- │    ---- │
│         │ * Over 0.5  │   95.0% │ 93.0% │        1.05 │  1.08 │  1.07 │  1.08 │   +1.7% │  ---- │  13.84% │
│ O/U 1.5 │ * Over 1.5  │   80.8% │ 75.2% │        1.24 │  1.33 │  1.32 │  1.34 │   +6.7% │  ---- │  12.86% │
│         │   Under 1.5 │   19.2% │ 24.8% │        5.21 │  4.03 │  3.95 │  4.10 │  -24.2% │  ---- │    ---- │
│ O/U 2.5 │ * Over 2.5  │   59.9% │ 50.3% │        1.67 │  1.99 │  1.98 │  2.00 │  +18.6% │  ---- │  13.12% │
│         │   Under 2.5 │   40.1% │ 49.5% │        2.50 │  2.02 │  2.00 │  2.04 │  -19.9% │  ---- │    ---- │
│ O/U 3.5 │   Under 3.5 │   61.3% │ 71.9% │        1.63 │  1.39 │  1.38 │  1.40 │  -15.4% │  ---- │    ---- │
│         │ * Over 3.5  │   38.7% │ 27.8% │        2.58 │  3.60 │  3.55 │  3.65 │  +37.4% │  ---- │  10.67% │
│ O/U 4.5 │   Under 4.5 │   78.0% │ 87.0% │        1.28 │  1.15 │  1.14 │  1.16 │  -11.1% │  ---- │    ---- │
│         │ * Over 4.5  │   22.0% │ 13.0% │        4.54 │  7.70 │  7.40 │  8.00 │  +63.1% │  ---- │   7.36% │
│ O/U 5.5 │   Under 5.5 │   88.8% │ 94.8% │        1.13 │  1.06 │  1.05 │  1.06 │   -6.8% │  ---- │    ---- │
│         │ * Over 5.5  │   11.2% │  5.1% │        8.91 │ 19.75 │ 18.50 │ 21.00 │ +107.6% │  ---- │   4.81% │
│ BTTS    │ * Yes       │   57.1% │ 51.8% │        1.75 │  1.93 │  1.91 │  1.95 │   +9.2% │  ---- │   5.60% │
│         │   No        │   42.8% │ 48.3% │        2.33 │  2.07 │  2.04 │  2.10 │  -12.6% │  ---- │    ---- │
=#

