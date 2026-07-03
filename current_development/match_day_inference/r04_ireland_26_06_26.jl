
using BayesianFootball
using DataFrames
using Turing
using MCMCChains

using ThreadPinning

using Dates

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
    "dixon_coles_$(today())", 
    save_dir; 
    target_seasons=["2026"],
    history_seasons = 2,
    warmup_period =  22,
    dynamics_col=:match_week,
    samples=2000,      # Small samples for fast runner testing
    warmup=1000,        # Small warmup for fast runner testing
    chains=16,         # 2 chains for fast runner testing
    use_queue=true
)

#
results = Experiments.run_experiment(task)
Experiments.save_experiment(results)
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
# NOTE: do NOT overwrite with `model_inference(ds, expr)` — that re-predicts the
# saved experiment's stale target matchday, not today's fixtures.

println("\nRaw Model 1X2 Probabilities:")
show(subset(ppd.df, :market_name => ByRow(==("1X2"))))
println()

# ==========================================
# 6. BETFAIR LIVE STREAM & KELLY STAKING
# ==========================================
println("\n=== 5. Connecting to Redis & Kelly Staking ===")
redis_host = get(ENV, "REDIS_HOST", "100.124.38.117")  # home server over Tailscale
redis_port = parse(Int, get(ENV, "REDIS_PORT", "6379"))

try
    println("🔗 Connecting to Redis at $redis_host:$redis_port...")
    redis_conn = RedisConnection(host=redis_host, port=redis_port)
    
    # Run the live betting dashboard (one-shot display)
    print_live_betting_dashboard(ppd, redis_conn, todays_matches; kelly_fraction=0.00, min_edge=0.03)
    
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
julia> print_live_betting_dashboard(ppd, redis_conn, todays_matches; kelly_fraction=0.00, min_edge=0.03)                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                                         
=============================================================================================================================                                                                                                                                                                                            
 LIVE MATCHDAY BETTING DASHBOARD | Kelly: 0.0 | Min Edge: 0.03 | 17:49:46                                                                                                                                                                                                                                                
=============================================================================================================================                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                         
> bohemian vs st-patricks-athletic (ID: 15238116) [$]                                                                                                                                                                                                                                                                    
╭─────────┬─────────────┬─────────┬───────┬─────────────┬───────┬───────┬───────┬────────┬───────┬─────────╮                                                                                                                                                                                                             
│ Market  │ Selection   │ Model % │ Mid % │ Model Price │   Mid │  Back │   Lay │     EV │ Kelly │ Bayes K │                                                                                                                                                                                                             
├─────────┼─────────────┼─────────┼───────┼─────────────┼───────┼───────┼───────┼────────┼───────┼─────────┤                                                                                                                                                                                                             
│ 1X2     │ * Away      │   40.1% │ 32.8% │        2.49 │  3.05 │  3.00 │  3.10 │ +20.4% │  ---- │   6.32% │                                                                                                                                                                                                             
│         │   Home      │   31.6% │ 37.7% │        3.16 │  2.65 │  2.62 │  2.68 │ -17.2% │  ---- │    ---- │                                                                                                                                                                                                             
│         │   Draw      │   28.2% │ 30.1% │        3.54 │  3.33 │  3.30 │  3.35 │  -6.8% │  ---- │    ---- │                                                
│ O/U 0.5 │ * Under 0.5 │    9.2% │  8.5% │       10.92 │ 11.75 │ 11.00 │ 12.50 │  +0.7% │  ---- │    ---- │                                                
│         │   Over 0.5  │   90.8% │ 91.7% │        1.10 │  1.09 │  1.08 │  1.10 │  -1.9% │  ---- │    ---- │                                                
│ O/U 1.5 │   Over 1.5  │   71.7% │ 71.9% │        1.39 │  1.39 │  1.38 │  1.40 │  -1.0% │  ---- │    ---- │                                                
│         │   Under 1.5 │   28.3% │ 28.0% │        3.54 │  3.58 │  3.50 │  3.65 │  -1.0% │  ---- │    ---- │                                                
│ O/U 2.5 │   Over 2.5  │   45.5% │ 45.9% │        2.20 │  2.18 │  2.16 │  2.20 │  -1.7% │  ---- │    ---- │                                                
│         │   Under 2.5 │   54.5% │ 54.3% │        1.83 │  1.84 │  1.83 │  1.85 │  -0.2% │  ---- │    ---- │                                                
│ O/U 3.5 │   Under 3.5 │   75.7% │ 76.0% │        1.32 │  1.31 │  1.30 │  1.33 │  -1.6% │  ---- │    ---- │                                                
│         │   Over 3.5  │   24.3% │ 24.4% │        4.11 │  4.10 │  4.00 │  4.20 │  -2.6% │  ---- │    ---- │                                                
│ O/U 4.5 │   Under 4.5 │   88.9% │ 89.3% │        1.12 │  1.12 │  1.11 │  1.13 │  -1.3% │  ---- │    ---- │                                                
│         │   Over 4.5  │   11.1% │ 10.6% │        9.03 │  9.40 │  9.00 │  9.80 │  -0.3% │  ---- │    ---- │                                                
│ O/U 5.5 │   Under 5.5 │   95.6% │ 96.2% │        1.05 │  1.04 │  1.03 │  1.05 │  -1.5% │  ---- │    ---- │                                                
│         │   Over 5.5  │    4.4% │  3.8% │       22.89 │ 26.00 │ 22.00 │ 30.00 │  -3.9% │  ---- │    ---- │                                                
│ BTTS    │   Yes       │   50.9% │ 53.9% │        1.97 │  1.85 │  1.85 │  1.86 │  -5.9% │  ---- │    ---- │                                                
│         │ * No        │   49.1% │ 46.7% │        2.04 │  2.14 │  2.10 │  2.18 │  +3.2% │  ---- │    ---- │                                                
╰─────────┴─────────────┴─────────┴───────┴─────────────┴───────┴───────┴───────┴────────┴───────┴─────────╯                                                
                                                                                                                                                            
> derry-city vs drogheda-united (ID: 15238117) [$]                                                                                                          
╭─────────┬─────────────┬─────────┬───────┬─────────────┬───────┬───────┬───────┬────────┬───────┬─────────╮                                                
│ Market  │ Selection   │ Model % │ Mid % │ Model Price │   Mid │  Back │   Lay │     EV │ Kelly │ Bayes K │                                                
├─────────┼─────────────┼─────────┼───────┼─────────────┼───────┼───────┼───────┼────────┼───────┼─────────┤                                                
│ 1X2     │ * Away      │   22.6% │ 13.5% │        4.42 │  7.40 │  7.20 │  7.60 │ +63.1% │  ---- │   9.54% │                                                
│         │   Home      │   51.5% │ 63.1% │        1.94 │  1.58 │  1.58 │  1.59 │ -18.7% │  ---- │    ---- │                                                
│         │ * Draw      │   25.9% │ 23.8% │        3.86 │  4.20 │  4.10 │  4.30 │  +6.1% │  ---- │    ---- │                                                
│ O/U 0.5 │ * Under 0.5 │    7.8% │  7.1% │       12.81 │ 14.00 │ 13.50 │ 14.50 │  +5.4% │  ---- │    ---- │                                                
│         │   Over 0.5  │   92.2% │ 93.0% │        1.08 │  1.08 │  1.07 │  1.08 │  -1.4% │  ---- │    ---- │                                                
│ O/U 1.5 │   Over 1.5  │   74.7% │ 74.9% │        1.34 │  1.33 │  1.32 │  1.35 │  -1.3% │  ---- │    ---- │                                                
│         │   Under 1.5 │   25.3% │ 25.0% │        3.96 │  4.00 │  3.90 │  4.10 │  -1.5% │  ---- │    ---- │                                                
│ O/U 2.5 │   Over 2.5  │   49.4% │ 49.9% │        2.02 │  2.00 │  1.99 │  2.02 │  -1.7% │  ---- │    ---- │                                                
│         │   Under 2.5 │   50.6% │ 50.1% │        1.98 │  2.00 │  1.97 │  2.02 │  -0.4% │  ---- │    ---- │                                                
│ O/U 3.5 │   Under 3.5 │   72.3% │ 72.5% │        1.38 │  1.38 │  1.37 │  1.39 │  -0.9% │  ---- │    ---- │                                                
│         │   Over 3.5  │   27.6% │ 27.4% │        3.62 │  3.65 │  3.55 │  3.75 │  -1.8% │  ---- │    ---- │                                                
│ O/U 4.5 │   Under 4.5 │   86.8% │ 87.7% │        1.15 │  1.14 │  1.13 │  1.15 │  -1.9% │  ---- │    ---- │                                                
│         │ * Over 4.5  │   13.2% │ 12.7% │        7.59 │  7.90 │  7.60 │  8.20 │  +0.1% │  ---- │    ---- │                                                
│ O/U 5.5 │   Under 5.5 │   94.6% │ 95.2% │        1.06 │  1.05 │  1.04 │  1.06 │  -1.7% │  ---- │    ---- │                                                
│         │ * Over 5.5  │    5.4% │  4.9% │       18.38 │ 20.25 │ 18.50 │ 22.00 │  +0.7% │  ---- │    ---- │                                                
│ BTTS    │ * Yes       │   51.8% │ 48.8% │        1.93 │  2.05 │  2.02 │  2.08 │  +4.5% │  ---- │    ---- │                                                
│         │   No        │   48.2% │ 51.2% │        2.07 │  1.96 │  1.92 │  1.99 │  -7.4% │  ---- │    ---- │                                                
╰─────────┴─────────────┴─────────┴───────┴─────────────┴───────┴───────┴───────┴────────┴───────┴─────────╯                                                
                                                                                                                                                            
> dundalk-fc vs waterford-fc (ID: 15238118) [$]                                                                                                             
╭─────────┬─────────────┬─────────┬───────┬─────────────┬───────┬───────┬───────┬────────┬───────┬─────────╮                                                
│ Market  │ Selection   │ Model % │ Mid % │ Model Price │   Mid │  Back │   Lay │     EV │ Kelly │ Bayes K │                                                
├─────────┼─────────────┼─────────┼───────┼─────────────┼───────┼───────┼───────┼────────┼───────┼─────────┤                                                
│ 1X2     │ * Away      │   28.4% │ 17.1% │        3.52 │  5.85 │  5.80 │  5.90 │ +64.6% │  ---- │  12.63% │                                                
│         │   Home      │   44.3% │ 61.5% │        2.25 │  1.62 │  1.62 │  1.63 │ -28.2% │  ---- │    ---- │                                                
│         │ * Draw      │   27.3% │ 21.3% │        3.67 │  4.70 │  4.60 │  4.80 │ +25.5% │  ---- │   6.07% │                                                
│ O/U 0.5 │ * Under 0.5 │    8.2% │  3.8% │       12.12 │ 26.50 │ 24.00 │ 29.00 │ +98.0% │  ---- │   3.78% │                                                
│         │   Over 0.5  │   91.8% │ 96.2% │        1.09 │  1.04 │  1.03 │  1.05 │  -5.5% │  ---- │    ---- │                                                
│ O/U 1.5 │   Over 1.5  │   73.7% │ 84.4% │        1.36 │  1.19 │  1.18 │  1.19 │ -13.0% │  ---- │    ---- │                                                
│         │ * Under 1.5 │   26.3% │ 15.9% │        3.81 │  6.30 │  6.00 │  6.60 │ +57.5% │  ---- │  10.04% │                                                
│ O/U 2.5 │   Over 2.5  │   48.1% │ 63.5% │        2.08 │  1.58 │  1.56 │  1.59 │ -24.9% │  ---- │    ---- │                                                
│         │ * Under 2.5 │   51.9% │ 36.5% │        1.93 │  2.74 │  2.70 │  2.78 │ +40.1% │  ---- │  20.99% │                                                
│ O/U 3.5 │ * Under 3.5 │   73.5% │ 59.3% │        1.36 │  1.69 │  1.67 │  1.70 │ +22.7% │  ---- │  29.82% │                                                
│         │   Over 3.5  │   26.5% │ 40.7% │        3.77 │  2.46 │  2.42 │  2.50 │ -35.8% │  ---- │    ---- │                                                
│ O/U 4.5 │ * Under 4.5 │   87.5% │ 77.8% │        1.14 │  1.29 │  1.27 │  1.30 │ +11.2% │  ---- │  35.81% │                                                
│         │   Over 4.5  │   12.5% │ 22.2% │        8.02 │  4.50 │  4.40 │  4.60 │ -45.1% │  ---- │    ---- │                                                
│ O/U 5.5 │ * Under 5.5 │   94.9% │ 89.3% │        1.05 │  1.12 │  1.11 │  1.13 │  +5.4% │  ---- │  42.14% │                                                
│         │   Over 5.5  │    5.1% │ 10.5% │       19.69 │  9.50 │  9.00 │ 10.00 │ -54.3% │  ---- │    ---- │                                                
│ BTTS    │   Yes       │   52.6% │ 59.0% │        1.90 │  1.69 │  1.68 │  1.71 │ -11.7% │  ---- │    ---- │                                                
│         │ * No        │   47.4% │ 40.8% │        2.11 │  2.45 │  2.42 │  2.48 │ +14.8% │  ---- │   7.61% │                                                
╰─────────┴─────────────┴─────────┴───────┴─────────────┴───────┴───────┴───────┴────────┴───────┴─────────╯                                                
                                                                                                                                                            
> shamrock-rovers vs galway-united (ID: 15238119) [$]                                                                                                       
╭─────────┬─────────────┬─────────┬───────┬─────────────┬───────┬───────┬───────┬────────┬───────┬─────────╮                                                
│ Market  │ Selection   │ Model % │ Mid % │ Model Price │   Mid │  Back │   Lay │     EV │ Kelly │ Bayes K │                                                
├─────────┼─────────────┼─────────┼───────┼─────────────┼───────┼───────┼───────┼────────┼───────┼─────────┤                                                
│ 1X2     │ * Away      │   21.5% │ 11.1% │        4.66 │  9.00 │  8.80 │  9.20 │ +88.9% │  ---- │  10.92% │                                                
│         │   Home      │   52.7% │ 68.5% │        1.90 │  1.46 │  1.45 │  1.47 │ -23.5% │  ---- │    ---- │                                                
│         │ * Draw      │   25.8% │ 20.2% │        3.88 │  4.95 │  4.90 │  5.00 │ +26.4% │  ---- │   5.92% │                                                
│ O/U 0.5 │ * Under 0.5 │    8.1% │  5.5% │       12.31 │ 18.25 │ 17.50 │ 19.00 │ +42.1% │  ---- │    ---- │                                                
│         │   Over 0.5  │   91.9% │ 94.8% │        1.09 │  1.06 │  1.05 │  1.06 │  -3.5% │  ---- │    ---- │                                                
│ O/U 1.5 │   Over 1.5  │   73.7% │ 78.7% │        1.36 │  1.27 │  1.26 │  1.28 │  -7.1% │  ---- │    ---- │                                                
│         │ * Under 1.5 │   26.3% │ 21.3% │        3.81 │  4.70 │  4.60 │  4.80 │ +20.9% │  ---- │   4.02% │                                                                               
│ O/U 2.5 │   Over 2.5  │   48.1% │ 55.1% │        2.08 │  1.81 │  1.80 │  1.83 │ -13.4% │  ---- │    ---- │                                                                               
│         │ * Under 2.5 │   51.9% │ 44.6% │        1.93 │  2.24 │  2.22 │  2.26 │ +15.2% │  ---- │   9.24% │                                                                               
│ O/U 3.5 │ * Under 3.5 │   73.5% │ 67.3% │        1.36 │  1.48 │  1.48 │  1.49 │  +8.8% │  ---- │  13.46% │                                                                               
│         │   Over 3.5  │   26.5% │ 32.3% │        3.78 │  3.10 │  3.05 │  3.15 │ -19.2% │  ---- │    ---- │                                                                               
│ O/U 4.5 │ * Under 4.5 │   87.6% │ 84.0% │        1.14 │  1.19 │  1.18 │  1.20 │  +3.4% │  ---- │    ---- │                                                                               
│         │   Over 4.5  │   12.4% │ 16.1% │        8.07 │  6.20 │  6.00 │  6.40 │ -25.6% │  ---- │    ---- │                                                                               
│ O/U 5.5 │ * Under 5.5 │   95.0% │ 93.5% │        1.05 │  1.07 │  1.06 │  1.08 │  +0.7% │  ---- │    ---- │                                                                               
│         │   Over 5.5  │    5.0% │  6.9% │       19.93 │ 14.50 │ 13.50 │ 15.50 │ -32.3% │  ---- │    ---- │                                                                               
│ BTTS    │ * Yes       │   50.2% │ 48.8% │        1.99 │  2.05 │  2.04 │  2.06 │  +2.3% │  ---- │    ---- │                                                                               
│         │   No        │   49.8% │ 51.2% │        2.01 │  1.96 │  1.94 │  1.97 │  -3.3% │  ---- │    ---- │                                                                               
╰─────────┴─────────────┴─────────┴───────┴─────────────┴───────┴───────┴───────┴────────┴───────┴─────────╯
=#

