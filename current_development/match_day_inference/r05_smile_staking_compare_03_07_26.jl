# current_development/match_day_inference/r05_smile_staking_compare_03_07_26.jl
#
# SMILE-DP staking comparison runner: per-bet Bayesian–McHale  vs  Unified structural Kelly.
#
# One model (DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel), two staking layers,
# two leagues (League of Ireland Premier Division = Ireland(79); First Division = 718):
#   • Bayesian–McHale (per-bet, scalar Baker–McHale shrinkage) — the "Bayes K" column of the
#     single-model live dashboard (Signals.BayesianKelly).
#   • Unified (P)-portfolio + (U-MC) shrinkage k* — the per-match panel, Over/Under + BTTS only,
#     cap 0.10 + 2% commission (see src/unified_staking.jl and docs/bets_multi/…).
#
# Run top-to-bottom, or step section by section in the REPL. Set `train=false` to reuse a
# previously-saved smile experiment in the same save_dir instead of re-training.

using BayesianFootball
using DataFrames
using Turing
using MCMCChains
using ThreadPinning
using Dates

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions
const D           = BayesianFootball.Data

include("./current_development/match_day_inference/loader.jl")

const JSON_LINEUPS_DIR = "./current_development/match_day_inference/data/lineups"
const BANKROLL = 50.0   # £ paper bankroll for the unified panel; set 0.0 for %-only display.

# ==========================================================================================
# 1. SHARED SMILE-DP MODEL (identical config to the existing r05 runners)
# ==========================================================================================
function build_smile_model()
    inter_cfg = PreGame.HierarchicalMonthlyInterception()
    disp_cfg  = PreGame.HomeAwayDispersion()
    ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
    kap_cfg   = PreGame.HierarchicalTeamKappa()
    dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=45.0)

    tracker_bayes     = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
    feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

    return PreGame.DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel(
        interception_config    = inter_cfg,
        player_dynamics_config = dyn_cfg,
        dispersion_config      = disp_cfg,
        homeadvantage_config   = ha_cfg,
        kappa_config           = kap_cfg,
        player_ratings_feature = feature_cfg_bayes,
        market_feature_config  = Features.DoublePoissonMarketFeature(),
        smile_feature          = Features.MarketSmileFeature(Kmax = 4),
        market_on              = true,
        supremacy_weight       = 1.0,
        smile_weight           = 0.4,
    )
end

# ==========================================================================================
# 2. PER-LEAGUE DRIVER — train (or load) smile, run match-day inference, show BOTH staking layers
# ==========================================================================================
"""
    run_smile_league(ds, save_dir; label, train, exp_prefix, bankroll)

Loads (default) or re-trains the smile model on `ds`, runs today's match-day inference, and
prints the per-bet Bayesian–McHale dashboard and the unified structural-Kelly panel side by
side. `exp_prefix` is the saved experiment's name prefix — the existing r05 runners save the
smile experiment as `double_poisson_smile_<date>`. Returns `(; expr, ppd, latents, todays_matches)`.
"""
function run_smile_league(ds::D.DataStore, save_dir::String;
                          label::String="Smile-DP", train::Bool=false,
                          exp_prefix::String="double_poisson_smile", bankroll::Float64=BANKROLL)
    model = build_smile_model()

    if train
        mkpath(save_dir)
        task = Experiments.create_experiment_task(
            ds, model, "$(exp_prefix)_$(today())", save_dir;
            target_seasons  = ["2026"],
            history_seasons = 2,
            warmup_period   = 23,
            dynamics_col    = :match_week,
            samples         = 2000,
            warmup          = 1000,
            chains          = 16,
            use_queue       = true,
        )
        results = Experiments.run_experiment(task)
        Experiments.save_experiment(results)
    end

    saved = Experiments.list_experiments(save_dir, data_dir="")
    expr  = Experiments.load_experiment(find_experiment_path(saved, exp_prefix))

    println("\n=== Fetching Today's Fixtures ($label) ===")
    todays_matches = fetch_todays_matches(ds)
    show(todays_matches); println()

    println("\n=== Running Match Day Inference ($label) ===")
    latents = compute_todays_matches_latents(ds, expr, todays_matches, JSON_LINEUPS_DIR)
    ppd     = Predictions.model_inference(latents)

    println("\n$label 1X2 Probabilities:")
    show(subset(ppd.df, :market_name => ByRow(==("1X2")))); println()

    # ---- staking comparison (needs live Betfair book from Redis) ----
    redis_host = get(ENV, "REDIS_HOST", "100.124.38.117")   # home server over Tailscale
    redis_port = parse(Int, get(ENV, "REDIS_PORT", "6379"))
    try
        println("\n🔗 Connecting to Redis at $redis_host:$redis_port ...")
        redis_conn = RedisConnection(host=redis_host, port=redis_port)

        # (a) Per-bet Bayesian–McHale — the "Bayes K" column (full std Kelly shown alongside).
        print_live_betting_dashboard(ppd, redis_conn, todays_matches; kelly_fraction=1.0, min_edge=0.0)

        # (b) Unified structural Kelly — per-match portfolio (Over/Under + BTTS, cap 0.10, 2% comm).
        print_unified_staking_dashboard(latents, redis_conn, todays_matches;
            label=label, cap=0.10, commission=0.02, bankroll=bankroll)
    catch e
        @warn "Redis connection skipped/failed — start Redis + the Betfair light streamer to view the live staking comparison. Error: $e"
    end

    return (; expr, ppd, latents, todays_matches)
end

# ==========================================================================================
# 3a. LEAGUE OF IRELAND — PREMIER DIVISION  (Ireland(), tournament 79)
#     Loads the smile experiment already trained by r05_ireland_03_07_26.jl
#     (save_dir "./data/match_day_ireland/july/", name "double_poisson_smile_<date>").
# ==========================================================================================
ds_premier = D.load_datastore_cached(D.Ireland())
premier = run_smile_league(ds_premier, "./data/match_day_ireland/july/";
                           label="Smile-DP · Premier", train=false);



#=
======================================================================================================================                
 UNIFIED STRUCTURAL KELLY (Smile-DP · Premier) | Over/Under + BTTS | cap=0.1  commission=2.0% | bankroll=£50.0 | 19:22:54             
 (P) joint log-optimal portfolio + (U-MC) posterior shrinkage k*  —  paper stakes, back-only                                          
======================================================================================================================                
                                                                                                                                      
> derry-city vs waterford-fc (ID: 15238121)  k*=1.0  Σa*=0.1  cash=0.9  G=0.0246                                                      
╭───────────────┬───────────┬─────────┬───────┬────────┬──────────────┬─────────╮                                                     
│ Market        │ Selection │ Model % │  Back │     EV │ Exec (k*·a*) │ £ Stake │                                                     
├───────────────┼───────────┼─────────┼───────┼────────┼──────────────┼─────────┤                                                     
│ OverUnder_0.5 │ under_05  │    8.1% │ 20.00 │ +58.3% │        1.02% │   £0.51 │                                                     
│ OverUnder_1.5 │ under_15  │   28.2% │  5.10 │ +41.5% │        3.24% │   £1.62 │                                                     
│ OverUnder_2.5 │ under_25  │   53.5% │  2.44 │ +29.0% │        5.75% │   £2.88 │                                                     
╰───────────────┴───────────┴─────────┴───────┴────────┴──────────────┴─────────╯                                                     
                                                                                                                                      
> drogheda-united vs bohemian (ID: 15238122)  k*=0.74  Σa*=0.1  cash=0.9  G=0.0048                                                    
╭───────────────┬───────────┬─────────┬───────┬────────┬──────────────┬─────────╮                                                     
│ Market        │ Selection │ Model % │  Back │     EV │ Exec (k*·a*) │ £ Stake │                                                     
├───────────────┼───────────┼─────────┼───────┼────────┼──────────────┼─────────┤                                                     
│ OverUnder_0.5 │ under_05  │    8.2% │ 15.00 │ +20.5% │        0.38% │   £0.19 │                                                     
│ OverUnder_1.5 │ under_15  │   28.5% │  4.10 │ +15.1% │        2.28% │   £1.14 │                                                     
│ OverUnder_2.5 │ under_25  │   53.9% │  2.02 │  +7.7% │        1.12% │   £0.56 │                                                     
│ OverUnder_3.5 │ under_35  │   75.1% │  1.40 │  +4.6% │        3.62% │   £1.81 │                                                     
╰───────────────┴───────────┴─────────┴───────┴────────┴──────────────┴─────────╯                                                     
                                                                                                                                      
> shelbourne vs dundalk-fc (ID: 15238123)  k*=1.0  Σa*=0.1  cash=0.9  G=0.0215                                                        
╭───────────────┬───────────┬─────────┬───────┬────────┬──────────────┬─────────╮                                                     
│ Market        │ Selection │ Model % │  Back │     EV │ Exec (k*·a*) │ £ Stake │                                                     
├───────────────┼───────────┼─────────┼───────┼────────┼──────────────┼─────────┤                                                     
│ OverUnder_0.5 │ under_05  │    8.8% │ 17.50 │ +51.2% │        0.91% │   £0.46 │                                                     
│ OverUnder_1.5 │ under_15  │   30.0% │  4.70 │ +38.6% │        4.90% │   £2.45 │                                                     
│ OverUnder_2.5 │ under_25  │   55.6% │  2.24 │ +23.3% │        4.19% │   £2.10 │                                                     
╰───────────────┴───────────┴─────────┴───────┴────────┴──────────────┴─────────╯                                                     
                                                                                                                                      
> st-patricks-athletic vs galway-united (ID: 15238124)  k*=1.0  Σa*=0.1  cash=0.9  G=0.0128                                           
╭───────────────┬───────────┬─────────┬───────┬────────┬──────────────┬─────────╮                                                     
│ Market        │ Selection │ Model % │  Back │     EV │ Exec (k*·a*) │ £ Stake │                                                     
├───────────────┼───────────┼─────────┼───────┼────────┼──────────────┼─────────┤                                                     
│ OverUnder_0.5 │ under_05  │    7.1% │ 20.00 │ +39.9% │        0.84% │   £0.42 │                                                     
│ OverUnder_1.5 │ under_15  │   25.8% │  5.00 │ +26.9% │        2.68% │   £1.34 │                                                     
│ OverUnder_2.5 │ under_25  │   50.3% │  2.36 │ +17.5% │        4.44% │   £2.22 │                                                     
│ OverUnder_3.5 │ under_35  │   72.1% │  1.55 │ +10.9% │        1.59% │   £0.80 │                                                     
│ BTTS_0.0      │ btts_yes  │   51.3% │  1.96 │  -0.5% │        0.45% │   £0.22 │                                                     
╰───────────────┴───────────┴─────────┴───────┴────────┴──────────────┴─────────╯                                                     
                                                                                                                                      
> sligo-rovers vs shamrock-rovers (ID: 15238125)  k*=0.73  Σa*=0.1  cash=0.9  G=0.0051                                                
╭───────────────┬───────────┬─────────┬───────┬────────┬──────────────┬─────────╮                                                     
│ Market        │ Selection │ Model % │  Back │     EV │ Exec (k*·a*) │ £ Stake │                                                     
├───────────────┼───────────┼─────────┼───────┼────────┼──────────────┼─────────┤                                                     
│ OverUnder_0.5 │ under_05  │    7.9% │ 15.00 │ +16.3% │        0.09% │   £0.04 │                                                     
│ OverUnder_1.5 │ under_15  │   27.8% │  4.20 │ +14.9% │        2.27% │   £1.14 │                                                     
│ OverUnder_2.5 │ under_25  │   53.0% │  2.08 │  +9.0% │        3.44% │   £1.72 │                                                     
│ BTTS_0.0      │ btts_yes  │   51.3% │  1.92 │  -2.4% │        1.49% │   £0.74 │                                                     
╰───────────────┴───────────┴─────────┴───────┴────────┴──────────────┴─────────╯
=#


#=
=============================================================================================================================                                                                                                                                                                                            
 LIVE MATCHDAY BETTING DASHBOARD | Kelly: 1.0 | Min Edge: 0.0 | 19:22:52                                                                                                                                                                                                                                                 
=============================================================================================================================                                                                                                                                                                                            
                                                                                                                                                                                                                                                                                                                         
> derry-city vs waterford-fc (ID: 15238121) [$]                                                                                                                                                                                                                                                                          
╭─────────┬─────────────┬─────────┬───────┬─────────────┬───────┬───────┬───────┬─────────┬────────┬─────────╮                                                                                                                                                                                                           
│ Market  │ Selection   │ Model % │ Mid % │ Model Price │   Mid │  Back │   Lay │      EV │  Kelly │ Bayes K │                                                                                                                                                                                                           
├─────────┼─────────────┼─────────┼───────┼─────────────┼───────┼───────┼───────┼─────────┼────────┼─────────┤                                                                                                                                                                                                           
│ 1X2     │ * Away      │   28.0% │ 13.2% │        3.57 │  7.60 │  7.40 │  7.80 │ +107.5% │ 16.79% │  16.32% │                                                                                                                                                                                                           
│         │   Home      │   45.8% │ 67.1% │        2.19 │  1.49 │  1.48 │  1.50 │  -32.3% │   ---- │    ---- │                                                                                                                                                                                                           
│         │ * Draw      │   26.2% │ 20.0% │        3.82 │  5.00 │  4.90 │  5.10 │  +28.4% │  7.28% │   6.88% │                                                                                                                                                                                                           
│ O/U 0.5 │ * Under 0.5 │    9.8% │  4.8% │       10.20 │ 21.00 │ 20.00 │ 22.00 │  +96.1% │  5.06% │   4.95% │                                                                                                                                                                                                           
│         │   Over 0.5  │   90.2% │ 95.2% │        1.11 │  1.05 │  1.04 │  1.06 │   -6.2% │   ---- │    ---- │                                                                                                                                                                                                           
│ O/U 1.5 │   Over 1.5  │   70.9% │ 80.6% │        1.41 │  1.24 │  1.23 │  1.25 │  -12.8% │   ---- │    ---- │                                                                                                                                                                                                           
│         │ * Under 1.5 │   29.1% │ 19.2% │        3.43 │  5.20 │  5.10 │  5.30 │  +48.5% │ 11.84% │  11.37% │                                                                                                                                                                                                           
│ O/U 2.5 │   Over 2.5  │   46.2% │ 59.2% │        2.16 │  1.69 │  1.68 │  1.70 │  -22.3% │   ---- │    ---- │                                                                                                                                                                                                           
│         │ * Under 2.5 │   53.8% │ 40.7% │        1.86 │  2.46 │  2.44 │  2.48 │  +31.2% │ 21.64% │  20.71% │                                                                                                                                                                                                           
│ O/U 3.5 │ * Under 3.5 │   73.9% │ 64.7% │        1.35 │  1.54 │  1.53 │  1.56 │  +13.1% │ 24.73% │  22.78% │                                                                                                                                                                                                           
│         │   Over 3.5  │   26.1% │ 35.3% │        3.84 │  2.83 │  2.78 │  2.88 │  -27.5% │   ---- │    ---- │                                                                                                                                                                                                           
│ O/U 4.5 │ * Under 4.5 │   87.3% │ 82.0% │        1.15 │  1.22 │  1.21 │  1.23 │   +5.6% │ 26.81% │  23.61% │                                                                                                                                                                                                           
│         │   Over 4.5  │   12.7% │ 18.2% │        7.87 │  5.50 │  5.30 │  5.70 │  -32.7% │   ---- │    ---- │                                                                                                                                                                                                           
│ O/U 5.5 │ * Under 5.5 │   95.4% │ 91.7% │        1.05 │  1.09 │  1.08 │  1.10 │   +3.0% │ 37.57% │  28.28% │                                                                                                                                                                                                           
│         │   Over 5.5  │    4.6% │  8.2% │       21.63 │ 12.25 │ 11.50 │ 13.00 │  -46.8% │   ---- │    ---- │                                                                                                                                                                                                           
│ BTTS    │   Yes       │   50.4% │ 52.5% │        1.98 │  1.90 │  1.90 │  1.91 │   -4.2% │   ---- │    ---- │                                                                                                                                                                                                           
│         │ * No        │   49.6% │ 47.4% │        2.02 │  2.11 │  2.08 │  2.14 │   +3.1% │  2.86% │   0.72% │                                                                                                                                                                                                           
╰─────────┴─────────────┴─────────┴───────┴─────────────┴───────┴───────┴───────┴─────────┴────────┴─────────╯                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                         
> drogheda-united vs bohemian (ID: 15238122) [$]                                                                                                                                                                                                                                                                         
╭─────────┬─────────────┬─────────┬───────┬─────────────┬───────┬───────┬───────┬────────┬───────┬─────────╮                                                                                                                                                                                                             
│ Market  │ Selection   │ Model % │ Mid % │ Model Price │   Mid │  Back │   Lay │     EV │ Kelly │ Bayes K │                                                                                                                                                                                                             
├─────────┼─────────────┼─────────┼───────┼─────────────┼───────┼───────┼───────┼────────┼───────┼─────────┤                                                                                                                                                                                                             
│ 1X2     │   Away      │   49.5% │ 54.8% │        2.02 │  1.83 │  1.81 │  1.84 │ -10.4% │  ---- │    ---- │                          
│         │ * Home      │   24.7% │ 19.6% │        4.04 │  5.10 │  5.00 │  5.20 │ +23.6% │ 5.91% │   4.80% │                          
│         │   Draw      │   25.8% │ 26.1% │        3.88 │  3.83 │  3.75 │  3.90 │  -3.4% │  ---- │    ---- │                          
│ O/U 0.5 │ * Under 0.5 │    9.9% │  6.6% │       10.06 │ 15.25 │ 15.00 │ 15.50 │ +49.1% │ 3.51% │   3.29% │                          
│         │   Over 0.5  │   90.1% │ 93.0% │        1.11 │  1.08 │  1.07 │  1.08 │  -3.6% │  ---- │    ---- │                          
│ O/U 1.5 │   Over 1.5  │   70.6% │ 76.0% │        1.42 │  1.31 │  1.31 │  1.32 │  -7.6% │  ---- │    ---- │                          
│         │ * Under 1.5 │   29.4% │ 23.8% │        3.40 │  4.20 │  4.10 │  4.30 │ +20.7% │ 6.67% │   5.68% │                          
│ O/U 2.5 │   Over 2.5  │   45.9% │ 51.3% │        2.18 │  1.95 │  1.94 │  1.96 │ -11.0% │  ---- │    ---- │                          
│         │ * Under 2.5 │   54.1% │ 49.0% │        1.85 │  2.04 │  2.02 │  2.06 │  +9.3% │ 9.16% │   6.90% │                          
│ O/U 3.5 │ * Under 3.5 │   74.3% │ 70.9% │        1.35 │  1.41 │  1.40 │  1.42 │  +4.0% │ 9.88% │   6.21% │                          
│         │   Over 3.5  │   25.7% │ 29.4% │        3.88 │  3.40 │  3.35 │  3.45 │ -13.7% │  ---- │    ---- │                          
│ O/U 4.5 │ * Under 4.5 │   87.5% │ 86.2% │        1.14 │  1.16 │  1.15 │  1.17 │  +0.6% │ 4.26% │   1.10% │                          
│         │   Over 4.5  │   12.5% │ 13.7% │        8.01 │  7.30 │  7.00 │  7.60 │ -12.6% │  ---- │    ---- │                          
│ O/U 5.5 │ * Under 5.5 │   95.5% │ 94.8% │        1.05 │  1.06 │  1.05 │  1.06 │  +0.3% │ 5.05% │   0.72% │                          
│         │   Over 5.5  │    4.5% │  5.6% │       22.12 │ 17.75 │ 16.50 │ 19.00 │ -25.4% │  ---- │    ---- │                          
│ BTTS    │   Yes       │   49.2% │ 53.6% │        2.03 │  1.86 │  1.85 │  1.88 │  -9.0% │  ---- │    ---- │                          
│         │ * No        │   50.8% │ 46.5% │        1.97 │  2.15 │  2.10 │  2.20 │  +6.7% │ 6.10% │   2.81% │                          
╰─────────┴─────────────┴─────────┴───────┴─────────────┴───────┴───────┴───────┴────────┴───────┴─────────╯                          
                                                                                                                                      
> shelbourne vs dundalk-fc (ID: 15238123) [$]                                                                                         
╭─────────┬─────────────┬─────────┬───────┬─────────────┬───────┬───────┬───────┬────────┬────────┬─────────╮                         
│ Market  │ Selection   │ Model % │ Mid % │ Model Price │   Mid │  Back │   Lay │     EV │  Kelly │ Bayes K │                         
├─────────┼─────────────┼─────────┼───────┼─────────────┼───────┼───────┼───────┼────────┼────────┼─────────┤                         
│ 1X2     │ * Away      │   37.4% │ 26.0% │        2.67 │  3.85 │  3.80 │  3.90 │ +42.2% │ 15.06% │  13.79% │                         
│         │   Home      │   35.3% │ 48.1% │        2.84 │  2.08 │  2.06 │  2.10 │ -27.4% │   ---- │    ---- │                         
│         │ * Draw      │   27.3% │ 26.1% │        3.66 │  3.83 │  3.80 │  3.85 │  +3.8% │  1.37% │   0.55% │                         
│ O/U 0.5 │ * Under 0.5 │   10.6% │  5.6% │        9.41 │ 18.00 │ 17.50 │ 18.50 │ +86.1% │  5.22% │   4.95% │                         
│         │   Over 0.5  │   89.4% │ 94.8% │        1.12 │  1.06 │  1.05 │  1.06 │  -6.2% │   ---- │    ---- │                         
│ O/U 1.5 │   Over 1.5  │   69.1% │ 79.1% │        1.45 │  1.27 │  1.26 │  1.27 │ -13.0% │   ---- │    ---- │                         
│         │ * Under 1.5 │   30.9% │ 20.8% │        3.23 │  4.80 │  4.70 │  4.90 │ +45.3% │ 12.26% │  11.23% │                         
│ O/U 2.5 │   Over 2.5  │   44.1% │ 56.0% │        2.27 │  1.79 │  1.77 │  1.80 │ -22.0% │   ---- │    ---- │                         
│         │ * Under 2.5 │   55.9% │ 43.9% │        1.79 │  2.28 │  2.24 │  2.32 │ +25.2% │ 20.35% │  18.18% │                         
│ O/U 3.5 │ * Under 3.5 │   75.7% │ 66.2% │        1.32 │  1.51 │  1.50 │  1.52 │ +13.6% │ 27.17% │  23.68% │                         
│         │   Over 3.5  │   24.3% │ 33.4% │        4.12 │  3.00 │  2.94 │  3.05 │ -28.6% │   ---- │    ---- │                         
│ O/U 4.5 │ * Under 4.5 │   88.5% │ 83.3% │        1.13 │  1.20 │  1.19 │  1.21 │  +5.3% │ 27.71% │  22.06% │                         
│         │   Over 4.5  │   11.5% │ 16.7% │        8.66 │  6.00 │  5.80 │  6.20 │ -33.1% │   ---- │    ---- │                         
│ O/U 5.5 │ * Under 5.5 │   95.9% │ 93.0% │        1.04 │  1.08 │  1.07 │  1.08 │  +2.6% │ 37.51% │  27.01% │                         
│         │   Over 5.5  │    4.1% │  7.4% │       24.46 │ 13.50 │ 13.00 │ 14.00 │ -46.9% │   ---- │    ---- │                         
│ BTTS    │   Yes       │   49.7% │ 59.2% │        2.01 │  1.69 │  1.67 │  1.71 │ -17.0% │   ---- │    ---- │                         
│         │ * No        │   50.3% │ 40.7% │        1.99 │  2.46 │  2.42 │  2.50 │ +21.7% │ 15.31% │  12.10% │                         
╰─────────┴─────────────┴─────────┴───────┴─────────────┴───────┴───────┴───────┴────────┴────────┴─────────╯                         
                                                                                                                                      
> st-patricks-athletic vs galway-united (ID: 15238124) [$]                                                                            
╭─────────┬─────────────┬─────────┬───────┬─────────────┬───────┬───────┬───────┬─────────┬────────┬─────────╮                        
│ Market  │ Selection   │ Model % │ Mid % │ Model Price │   Mid │  Back │   Lay │      EV │  Kelly │ Bayes K │                        
├─────────┼─────────────┼─────────┼───────┼─────────────┼───────┼───────┼───────┼─────────┼────────┼─────────┤                        
│ 1X2     │ * Away      │   23.3% │ 11.2% │        4.29 │  8.90 │  8.60 │  9.20 │ +100.6% │ 13.23% │  12.83% │                        
│         │   Home      │   52.1% │ 69.9% │        1.92 │  1.43 │  1.42 │  1.44 │  -26.0% │   ---- │    ---- │                        
│         │ * Draw      │   24.5% │ 18.7% │        4.07 │  5.35 │  5.20 │  5.50 │  +27.6% │  6.58% │   6.10% │                        
│ O/U 0.5 │ * Under 0.5 │    8.7% │  4.8% │       11.44 │ 21.00 │ 20.00 │ 22.00 │  +74.8% │  3.94% │   3.79% │                        
│         │   Over 0.5  │   91.3% │ 94.8% │        1.10 │  1.06 │  1.05 │  1.06 │   -4.2% │   ---- │    ---- │                        
│ O/U 1.5 │   Over 1.5  │   73.3% │ 80.6% │        1.36 │  1.24 │  1.23 │  1.25 │   -9.8% │   ---- │    ---- │                        
│         │ * Under 1.5 │   26.7% │ 19.4% │        3.75 │  5.15 │  5.00 │  5.30 │  +33.3% │  8.33% │   7.63% │                        
│ O/U 2.5 │   Over 2.5  │   49.4% │ 58.5% │        2.02 │  1.71 │  1.69 │  1.73 │  -16.5% │   ---- │    ---- │                        
│         │ * Under 2.5 │   50.6% │ 41.7% │        1.98 │  2.40 │  2.36 │  2.44 │  +19.4% │ 14.28% │  12.76% │                        
│ O/U 3.5 │ * Under 3.5 │   71.1% │ 64.3% │        1.41 │  1.56 │  1.55 │  1.56 │  +10.2% │ 18.59% │  15.95% │                        
│         │   Over 3.5  │   28.9% │ 35.2% │        3.46 │  2.84 │  2.80 │  2.88 │  -19.1% │   ---- │    ---- │                        
│ O/U 4.5 │ * Under 4.5 │   85.4% │ 82.0% │        1.17 │  1.22 │  1.21 │  1.23 │   +3.3% │ 15.60% │  10.93% │                        
│         │   Over 4.5  │   14.6% │ 18.0% │        6.83 │  5.55 │  5.40 │  5.70 │  -20.9% │   ---- │    ---- │                        
│ O/U 5.5 │ * Under 5.5 │   94.4% │ 91.7% │        1.06 │  1.09 │  1.08 │  1.10 │   +2.0% │ 24.68% │  14.25% │                        
│         │   Over 5.5  │    5.6% │  7.8% │       17.92 │ 12.75 │ 12.00 │ 13.50 │  -33.0% │   ---- │    ---- │                        
│ BTTS    │ * Yes       │   51.3% │ 50.6% │        1.95 │  1.98 │  1.96 │  1.99 │   +0.5% │  0.53% │   0.03% │                        
│         │   No        │   48.7% │ 49.6% │        2.05 │  2.02 │  1.99 │  2.04 │   -3.0% │   ---- │    ---- │                        
╰─────────┴─────────────┴─────────┴───────┴─────────────┴───────┴───────┴───────┴─────────┴────────┴─────────╯                        

> sligo-rovers vs shamrock-rovers (ID: 15238125) [$]                          
╭─────────┬─────────────┬─────────┬───────┬─────────────┬───────┬───────┬───────┬─────────┬────────┬─────────╮                                              
│ Market  │ Selection   │ Model % │ Mid % │ Model Price │   Mid │  Back │   Lay │      EV │  Kelly │ Bayes K │                                              
├─────────┼─────────────┼─────────┼───────┼─────────────┼───────┼───────┼───────┼─────────┼────────┼─────────┤                                              
│ 1X2     │   Away      │   29.9% │ 61.0% │        3.34 │  1.64 │  1.63 │  1.65 │  -51.3% │   ---- │    ---- │                                              
│         │ * Home      │   43.8% │ 16.1% │        2.28 │  6.20 │  6.00 │  6.40 │ +162.6% │ 32.52% │  32.23% │                                              
│         │ * Draw      │   26.3% │ 23.5% │        3.80 │  4.25 │  4.20 │  4.30 │  +10.6% │  3.31% │   2.60% │                                              
│ O/U 0.5 │ * Under 0.5 │    9.6% │  6.2% │       10.40 │ 16.00 │ 15.00 │ 17.00 │  +44.2% │  3.16% │   2.95% │                                              
│         │   Over 0.5  │   90.4% │ 93.9% │        1.11 │  1.06 │  1.06 │  1.07 │   -4.2% │   ---- │    ---- │                                              
│ O/U 1.5 │   Over 1.5  │   71.3% │ 76.9% │        1.40 │  1.30 │  1.29 │  1.31 │   -8.0% │   ---- │    ---- │                                              
│         │ * Under 1.5 │   28.7% │ 23.3% │        3.48 │  4.30 │  4.20 │  4.40 │  +20.5% │  6.41% │   5.54% │                                              
│ O/U 2.5 │   Over 2.5  │   46.8% │ 52.6% │        2.14 │  1.90 │  1.89 │  1.91 │  -11.6% │   ---- │    ---- │                                              
│         │ * Under 2.5 │   53.2% │ 47.6% │        1.88 │  2.10 │  2.08 │  2.12 │  +10.7% │  9.90% │   7.95% │                                              
│ O/U 3.5 │ * Under 3.5 │   73.5% │ 70.4% │        1.36 │  1.42 │  1.41 │  1.43 │   +3.6% │  8.73% │   5.29% │                                              
│         │   Over 3.5  │   26.5% │ 29.9% │        3.77 │  3.35 │  3.30 │  3.40 │  -12.4% │   ---- │    ---- │
│ O/U 4.5 │ * Under 4.5 │   87.0% │ 85.5% │        1.15 │  1.17 │  1.16 │  1.18 │   +0.9% │  5.63% │   1.95% │
│         │   Over 4.5  │   13.0% │ 14.3% │        7.68 │  7.00 │  6.80 │  7.20 │  -11.5% │   ---- │    ---- │
│ O/U 5.5 │ * Under 5.5 │   95.2% │ 93.9% │        1.05 │  1.06 │  1.06 │  1.07 │   +0.9% │ 15.73% │   6.41% │
│         │   Over 5.5  │    4.8% │  5.8% │       20.96 │ 17.25 │ 16.00 │ 18.50 │  -23.7% │   ---- │    ---- │
│ BTTS    │   Yes       │   51.3% │ 51.4% │        1.95 │  1.94 │  1.92 │  1.97 │   -1.5% │   ---- │    ---- │
│         │   No        │   48.7% │ 48.3% │        2.05 │  2.07 │  2.04 │  2.10 │   -0.7% │   ---- │    ---- │
╰─────────┴─────────────┴─────────┴───────┴─────────────┴───────┴───────┴───────┴─────────┴────────┴─────────╯
=#


# ==========================================================================================
# 3b. LEAGUE OF IRELAND — FIRST DIVISION  (IrelandFirstDivision(), tournament 718)
#     Swap the Betfair-summarized book into ds.odds (must match how r05_first_div trained),
#     then load its smile experiment from "./data/match_day_ireland_first/july/".
# ==========================================================================================
ds_fd_raw = D.load_datastore_cached(D.IrelandFirstDivision())
odds_bf   = D.summarize_betfair_market(ds_fd_raw, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds_first  = D.DataStore(ds_fd_raw.segment, ds_fd_raw.matches, ds_fd_raw.statistics,
                        odds_bf, ds_fd_raw.lineups, ds_fd_raw.incidents, ds_fd_raw.betfair_odds)
# NB: don't name this `first` — it shadows Base.first, which lineups.jl calls internally.
firstdiv = run_smile_league(ds_first, "./data/match_day_ireland_first/july/";
                            label="Smile-DP · First Div", train=false)



#=
======================================================================================================================                                      
 UNIFIED STRUCTURAL KELLY (Smile-DP · First Div) | Over/Under + BTTS | cap=0.1  commission=2.0% | bankroll=£50.0 | 19:30:35                                 
 (P) joint log-optimal portfolio + (U-MC) posterior shrinkage k*  —  paper stakes, back-only                                                                
======================================================================================================================                                      
                                                                                                                                                            
> bray-wanderers vs athlone-town (ID: 15238814)  k*=0.58  Σa*=0.092  cash=0.908  G=0.0062                                                                   
╭───────────────┬───────────┬─────────┬───────┬────────┬──────────────┬─────────╮                                                                           
│ Market        │ Selection │ Model % │  Back │     EV │ Exec (k*·a*) │ £ Stake │                                                                           
├───────────────┼───────────┼─────────┼───────┼────────┼──────────────┼─────────┤                                                                           
│ OverUnder_2.5 │ over_25   │   64.3% │  1.65 │  +5.3% │        0.53% │   £0.27 │                                                                           
│ OverUnder_3.5 │ over_35   │   42.6% │  2.66 │ +11.8% │        1.53% │   £0.77 │                                                                           
│ OverUnder_4.5 │ over_45   │   24.5% │  5.00 │ +20.5% │        1.40% │   £0.70 │                                                                           
│ OverUnder_5.5 │ over_55   │   12.4% │ 10.50 │ +27.5% │        0.75% │   £0.38 │                                                                           
│ BTTS_0.0      │ btts_yes  │   63.3% │  1.67 │  +5.0% │        1.14% │   £0.57 │                                                                           
╰───────────────┴───────────┴─────────┴───────┴────────┴──────────────┴─────────╯                                                                           
                                                                                                                                                            
> cobh-ramblers vs cork-city (ID: 15238815)  k*=1.0  Σa*=0.1  cash=0.9  G=0.032                                                                             
╭───────────────┬───────────┬─────────┬───────┬────────┬──────────────┬─────────╮                                                                           
│ Market        │ Selection │ Model % │  Back │     EV │ Exec (k*·a*) │ £ Stake │                                                                           
├───────────────┼───────────┼─────────┼───────┼────────┼──────────────┼─────────┤                                                                           
│ OverUnder_3.5 │ over_35   │   42.3% │  3.25 │ +35.6% │        3.66% │   £1.83 │                                                                           
│ OverUnder_4.5 │ over_45   │   24.3% │  6.60 │ +57.7% │        3.04% │   £1.52 │                                                                           
│ OverUnder_5.5 │ over_55   │   12.3% │ 15.50 │ +86.6% │        2.97% │   £1.49 │                                                                           
│ BTTS_0.0      │ btts_yes  │   65.0% │  1.86 │ +19.8% │        0.33% │   £0.17 │                                                                           
╰───────────────┴───────────┴─────────┴───────┴────────┴──────────────┴─────────╯                                                                           
                                                                                                                                                            
> finn-harps vs wexford-fc (ID: 15238816)  k*=0.62  Σa*=0.1  cash=0.9  G=0.0067                                                                             
╭───────────────┬───────────┬─────────┬───────┬────────┬──────────────┬─────────╮                                                                           
│ Market        │ Selection │ Model % │  Back │     EV │ Exec (k*·a*) │ £ Stake │                                                                           
├───────────────┼───────────┼─────────┼───────┼────────┼──────────────┼─────────┤                                                                           
│ OverUnder_0.5 │ over_05   │   95.3% │  1.06 │  +0.9% │        0.67% │   £0.34 │                                                                           
│ OverUnder_2.5 │ over_25   │   59.3% │  1.83 │  +7.6% │        2.41% │   £1.21 │                                                                           
│ OverUnder_3.5 │ over_35   │   37.2% │  3.10 │ +13.9% │        1.25% │   £0.62 │                                                                           
│ OverUnder_4.5 │ over_45   │   20.2% │  6.20 │ +23.1% │        1.19% │   £0.60 │                                                                           
│ OverUnder_5.5 │ over_55   │    9.6% │ 14.00 │ +31.7% │        0.69% │   £0.34 │                                                                           
╰───────────────┴───────────┴─────────┴───────┴────────┴──────────────┴─────────╯                                                                           
                                                                                                                                                            
> treaty-united-fc vs kerry-fc (ID: 15238817)  k*=1.0  Σa*=0.1  cash=0.9  G=0.0493                                                                          
╭───────────────┬───────────┬─────────┬───────┬─────────┬──────────────┬─────────╮                                                                          
│ Market        │ Selection │ Model % │  Back │      EV │ Exec (k*·a*) │ £ Stake │                                                                          
├───────────────┼───────────┼─────────┼───────┼─────────┼──────────────┼─────────┤                                                                          
│ OverUnder_3.5 │ over_35   │   37.5% │  4.30 │  +58.8% │        4.20% │   £2.10 │                                                                          
│ OverUnder_4.5 │ over_45   │   20.4% │  9.80 │  +96.3% │        3.87% │   £1.93 │                                                                          
│ OverUnder_5.5 │ over_55   │    9.7% │ 24.00 │ +128.6% │        1.93% │   £0.97 │                                                                          
╰───────────────┴───────────┴─────────┴───────┴─────────┴──────────────┴─────────╯
=#

#=
> bray-wanderers vs athlone-town (ID: 15238814) [$]                                                                                                                        19:30 [147/1811]
╭─────────┬─────────────┬─────────┬───────┬─────────────┬───────┬───────┬───────┬────────┬───────┬─────────╮                                                                               
│ Market  │ Selection   │ Model % │ Mid % │ Model Price │   Mid │  Back │   Lay │     EV │ Kelly │ Bayes K │                                                                               
├─────────┼─────────────┼─────────┼───────┼─────────────┼───────┼───────┼───────┼────────┼───────┼─────────┤                                                                               
│ 1X2     │ * Away      │   25.5% │ 21.1% │        3.92 │  4.75 │  4.60 │  4.90 │ +17.2% │ 4.79% │   3.61% │                                                                               
│         │   Home      │   52.8% │ 57.1% │        1.89 │  1.75 │  1.74 │  1.76 │  -8.1% │  ---- │    ---- │                                                                               
│         │   Draw      │   21.7% │ 22.5% │        4.60 │  4.45 │  4.40 │  4.50 │  -4.4% │  ---- │    ---- │                                                                               
│ O/U 0.5 │ * Under 0.5 │    5.7% │  4.5% │       17.48 │ 22.25 │ 19.50 │ 25.00 │ +11.6% │ 0.63% │   0.20% │                                                                               
│         │   Over 0.5  │   94.3% │ 95.7% │        1.06 │  1.04 │  1.04 │  1.05 │  -2.0% │  ---- │    ---- │                                                                               
│ O/U 1.5 │   Over 1.5  │   78.8% │ 82.0% │        1.27 │  1.22 │  1.21 │  1.23 │  -4.6% │  ---- │    ---- │                                                                               
│         │ * Under 1.5 │   21.2% │ 18.2% │        4.73 │  5.50 │  5.30 │  5.70 │ +12.1% │ 2.83% │   1.27% │                                                                               
│ O/U 2.5 │   Over 2.5  │   54.9% │ 59.9% │        1.82 │  1.67 │  1.65 │  1.69 │  -9.5% │  ---- │    ---- │                                                                               
│         │ * Under 2.5 │   45.1% │ 39.8% │        2.22 │  2.51 │  2.48 │  2.54 │ +11.9% │ 8.06% │   4.88% │                                                                               
│ O/U 3.5 │ * Under 3.5 │   67.0% │ 62.9% │        1.49 │  1.59 │  1.57 │  1.61 │  +5.2% │ 9.08% │   4.31% │                                                
│         │   Over 3.5  │   33.0% │ 37.0% │        3.03 │  2.70 │  2.66 │  2.74 │ -12.2% │  ---- │    ---- │                                                
│ O/U 4.5 │ * Under 4.5 │   82.0% │ 80.6% │        1.22 │  1.24 │  1.23 │  1.25 │  +0.9% │ 3.71% │   0.60% │                                                
│         │   Over 4.5  │   18.0% │ 19.2% │        5.55 │  5.20 │  5.00 │  5.40 │ -10.0% │  ---- │    ---- │                                                
│ O/U 5.5 │   Under 5.5 │   87.6% │ 91.3% │        1.14 │  1.10 │  1.09 │  1.10 │  -4.5% │  ---- │    ---- │                                                
│         │ * Over 5.5  │   12.4% │  8.9% │        8.09 │ 11.25 │ 10.50 │ 12.00 │ +29.9% │ 3.14% │   1.46% │                                                
│ BTTS    │ * Yes       │   63.4% │ 58.7% │        1.58 │  1.71 │  1.67 │  1.74 │  +5.8% │ 8.65% │   4.10% │                                                
│         │   No        │   36.6% │ 41.2% │        2.73 │  2.43 │  2.34 │  2.52 │ -14.2% │  ---- │    ---- │                                                
╰─────────┴─────────────┴─────────┴───────┴─────────────┴───────┴───────┴───────┴────────┴───────┴─────────╯                                                
                                                                                                                                                            
> cobh-ramblers vs cork-city (ID: 15238815) [$]                                                                                                             
╭─────────┬─────────────┬─────────┬───────┬─────────────┬───────┬───────┬───────┬────────┬────────┬─────────╮                                               
│ Market  │ Selection   │ Model % │ Mid % │ Model Price │   Mid │  Back │   Lay │     EV │  Kelly │ Bayes K │                                               
├─────────┼─────────────┼─────────┼───────┼─────────────┼───────┼───────┼───────┼────────┼────────┼─────────┤                                               
│ 1X2     │   Away      │   41.8% │ 58.3% │        2.39 │  1.71 │  1.69 │  1.74 │ -29.3% │   ---- │    ---- │                                               
│         │ * Home      │   35.3% │ 17.9% │        2.83 │  5.60 │  5.40 │  5.80 │ +90.6% │ 20.59% │  19.86% │                                               
│         │   Draw      │   22.9% │ 23.8% │        4.37 │  4.20 │  4.10 │  4.30 │  -6.2% │   ---- │    ---- │                                               
│ O/U 0.5 │   Under 0.5 │    5.8% │  5.8% │       17.22 │ 17.25 │ 15.00 │ 19.50 │ -12.9% │   ---- │    ---- │                                               
│         │   Over 0.5  │   94.2% │ 93.5% │        1.06 │  1.07 │  1.06 │  1.08 │  -0.2% │   ---- │    ---- │                                               
│ O/U 1.5 │ * Over 1.5  │   78.6% │ 76.6% │        1.27 │  1.31 │  1.29 │  1.32 │  +1.4% │  4.93% │   1.19% │                                               
│         │   Under 1.5 │   21.4% │ 23.5% │        4.68 │  4.25 │  4.10 │  4.40 │ -12.4% │   ---- │    ---- │                                               
│ O/U 2.5 │ * Over 2.5  │   54.6% │ 52.4% │        1.83 │  1.91 │  1.88 │  1.94 │  +2.7% │  3.02% │   0.64% │                                               
│         │   Under 2.5 │   45.4% │ 47.8% │        2.20 │  2.09 │  2.06 │  2.12 │  -6.5% │   ---- │    ---- │                                               
│ O/U 3.5 │   Under 3.5 │   67.2% │ 69.7% │        1.49 │  1.44 │  1.42 │  1.45 │  -4.6% │   ---- │    ---- │                                               
│         │ * Over 3.5  │   32.8% │ 30.1% │        3.05 │  3.33 │  3.25 │  3.40 │  +6.6% │  2.92% │   0.83% │                                               
│ O/U 4.5 │   Under 4.5 │   82.1% │ 85.8% │        1.22 │  1.17 │  1.15 │  1.18 │  -5.6% │   ---- │    ---- │                                               
│         │ * Over 4.5  │   17.9% │ 14.3% │        5.59 │  7.00 │  6.60 │  7.40 │ +18.0% │  3.22% │   1.41% │                                               
│ O/U 5.5 │   Under 5.5 │   87.7% │ 94.3% │        1.14 │  1.06 │  1.05 │  1.07 │  -7.9% │   ---- │    ---- │                                               
│         │ * Over 5.5  │   12.3% │  5.9% │        8.15 │ 17.00 │ 15.50 │ 18.50 │ +90.1% │  6.22% │   4.74% │                                               
│ BTTS    │ * Yes       │   65.0% │ 52.4% │        1.54 │  1.91 │  1.86 │  1.96 │ +21.0% │ 24.37% │  20.32% │                                               
│         │   No        │   35.0% │ 47.8% │        2.86 │  2.09 │  2.02 │  2.16 │ -29.4% │   ---- │    ---- │                                               
╰─────────┴─────────────┴─────────┴───────┴─────────────┴───────┴───────┴───────┴────────┴────────┴─────────╯                                               
                                                                                                                                                            
> finn-harps vs wexford-fc (ID: 15238816) [$]                                                                                                               
╭─────────┬─────────────┬─────────┬───────┬─────────────┬───────┬───────┬───────┬────────┬───────┬─────────╮                                                
│ Market  │ Selection   │ Model % │ Mid % │ Model Price │   Mid │  Back │   Lay │     EV │ Kelly │ Bayes K │                                                
├─────────┼─────────────┼─────────┼───────┼─────────────┼───────┼───────┼───────┼────────┼───────┼─────────┤                                                
│ 1X2     │   Away      │   47.4% │ 49.8% │        2.11 │  2.01 │  1.98 │  2.04 │  -6.2% │  ---- │    ---- │                                                
│         │ * Home      │   29.3% │ 25.0% │        3.42 │  4.00 │  3.90 │  4.10 │ +14.2% │ 4.90% │   3.02% │                                                
│         │   Draw      │   23.4% │ 25.3% │        4.28 │  3.95 │  3.90 │  4.00 │  -8.9% │  ---- │    ---- │                                                
│ O/U 0.5 │ * Under 0.5 │    7.1% │  5.6% │       14.17 │ 17.75 │ 15.50 │ 20.00 │  +9.4% │ 0.65% │   0.17% │                                                
│         │   Over 0.5  │   92.9% │ 93.9% │        1.08 │  1.06 │  1.06 │  1.07 │  -1.5% │  ---- │    ---- │                                                
│ O/U 1.5 │   Over 1.5  │   75.2% │ 78.1% │        1.33 │  1.28 │  1.27 │  1.29 │  -4.4% │  ---- │    ---- │                                                
│         │ * Under 1.5 │   24.8% │ 22.2% │        4.04 │  4.50 │  4.40 │  4.60 │  +8.9% │ 2.63% │   0.93% │                                                
│ O/U 2.5 │   Over 2.5  │   49.9% │ 53.9% │        2.01 │  1.85 │  1.83 │  1.88 │  -8.8% │  ---- │    ---- │                                                
│         │ * Under 2.5 │   50.1% │ 46.3% │        1.99 │  2.16 │  2.12 │  2.20 │  +6.3% │ 5.62% │   2.25% │                                                
│ O/U 3.5 │ * Under 3.5 │   71.6% │ 68.7% │        1.40 │  1.46 │  1.44 │  1.47 │  +3.2% │ 7.19% │   2.50% │                                                
│         │   Over 3.5  │   28.4% │ 32.0% │        3.53 │  3.12 │  3.10 │  3.15 │ -12.1% │  ---- │    ---- │                                                
│ O/U 4.5 │   Under 4.5 │   85.4% │ 84.7% │        1.17 │  1.18 │  1.17 │  1.19 │  -0.1% │  ---- │    ---- │                                                
│         │   Over 4.5  │   14.6% │ 15.9% │        6.84 │  6.30 │  6.20 │  6.40 │  -9.4% │  ---- │    ---- │                                                
│ O/U 5.5 │   Under 5.5 │   90.4% │ 93.5% │        1.11 │  1.07 │  1.06 │  1.08 │  -4.2% │  ---- │    ---- │                                                
│         │ * Over 5.5  │    9.6% │  6.7% │       10.43 │ 15.00 │ 14.00 │ 16.00 │ +34.2% │ 2.63% │   1.25% │                                                
│ BTTS    │ * Yes       │   60.5% │ 56.5% │        1.65 │  1.77 │  1.74 │  1.80 │  +5.4% │ 7.23% │   3.04% │                                                
│         │   No        │   39.5% │ 43.5% │        2.53 │  2.30 │  2.24 │  2.36 │ -11.6% │  ---- │    ---- │                                                
╰─────────┴─────────────┴─────────┴───────┴─────────────┴───────┴───────┴───────┴────────┴───────┴─────────╯                                                
                                                                                                                                                            
> treaty-united-fc vs kerry-fc (ID: 15238817) [$]                                                                                                           
╭─────────┬─────────────┬─────────┬───────┬─────────────┬───────┬───────┬───────┬─────────┬────────┬─────────╮                                              
│ Market  │ Selection   │ Model % │ Mid % │ Model Price │   Mid │  Back │   Lay │      EV │  Kelly │ Bayes K │                                              
├─────────┼─────────────┼─────────┼───────┼─────────────┼───────┼───────┼───────┼─────────┼────────┼─────────┤                                              
│ 1X2     │ * Away      │   34.8% │ 34.2% │        2.88 │  2.92 │  2.88 │  2.96 │   +0.1% │  0.07% │   0.00% │                                              
│         │ * Home      │   41.4% │ 36.6% │        2.41 │  2.73 │  2.70 │  2.76 │  +11.8% │  6.96% │   4.70% │                                              
│         │   Draw      │   23.8% │ 29.4% │        4.20 │  3.40 │  3.35 │  3.45 │  -20.2% │   ---- │    ---- │                                              
│ O/U 0.5 │   Under 0.5 │    7.0% │  8.9% │       14.32 │ 11.25 │ 10.50 │ 12.00 │  -26.7% │   ---- │    ---- │                                              
│         │ * Over 0.5  │   93.0% │ 91.3% │        1.08 │  1.10 │  1.09 │  1.10 │   +1.4% │ 15.45% │   8.48% │                                              
│ O/U 1.5 │ * Over 1.5  │   75.4% │ 70.7% │        1.33 │  1.42 │  1.40 │  1.43 │   +5.6% │ 14.01% │   8.74% │                                              
│         │   Under 1.5 │   24.6% │ 29.4% │        4.07 │  3.40 │  3.30 │  3.50 │  -18.9% │   ---- │    ---- │                                              
│ O/U 2.5 │ * Over 2.5  │   50.1% │ 43.9% │        2.00 │  2.28 │  2.24 │  2.32 │  +12.2% │  9.86% │   6.33% │                                              
│         │   Under 2.5 │   49.9% │ 56.2% │        2.00 │  1.78 │  1.76 │  1.80 │  -12.2% │   ---- │    ---- │                                              
│ O/U 3.5 │   Under 3.5 │   71.4% │ 77.2% │        1.40 │  1.29 │  1.28 │  1.31 │   -8.6% │   ---- │    ---- │                                              
│         │ * Over 3.5  │   28.6% │ 22.7% │        3.50 │  4.40 │  4.30 │  4.50 │  +22.8% │  6.91% │   4.67% │                                              
│ O/U 4.5 │   Under 4.5 │   85.2% │ 90.1% │        1.17 │  1.11 │  1.10 │  1.12 │   -6.2% │   ---- │    ---- │                                              
│         │ * Over 4.5  │   14.8% │  9.6% │        6.78 │ 10.40 │  9.80 │ 11.00 │  +44.6% │  5.07% │   3.69% │                                              
│ O/U 5.5 │   Under 5.5 │   90.3% │ 96.2% │        1.11 │  1.04 │  1.03 │  1.05 │   -7.0% │   ---- │    ---- │                                                                             
│         │ * Over 5.5  │    9.7% │  ---- │       10.30 │  ---- │ 24.00 │  ---- │ +133.0% │  5.78% │   4.88% │                                                                             
│ BTTS    │ * Yes       │   61.6% │ 51.7% │        1.62 │  1.94 │  1.88 │  1.99 │  +15.8% │ 17.94% │  13.62% │                                                                             
│         │   No        │   38.4% │ 47.8% │        2.60 │  2.09 │  2.00 │  2.18 │  -23.2% │   ---- │    ---- │                                                                             
╰─────────┴─────────────┴─────────┴───────┴─────────────┴───────┴───────┴───────┴─────────┴────────┴─────────╯
=#


