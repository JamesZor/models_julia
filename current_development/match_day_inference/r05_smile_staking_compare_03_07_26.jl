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
                           label="Smile-DP · Premier", train=false)

# ==========================================================================================
# 3b. LEAGUE OF IRELAND — FIRST DIVISION  (IrelandFirstDivision(), tournament 718)
#     Swap the Betfair-summarized book into ds.odds (must match how r05_first_div trained),
#     then load its smile experiment from "./data/match_day_ireland_first/july/".
# ==========================================================================================
ds_fd_raw = D.load_datastore_cached(D.IrelandFirstDivision())
odds_bf   = D.summarize_betfair_market(ds_fd_raw, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds_first  = D.DataStore(ds_fd_raw.segment, ds_fd_raw.matches, ds_fd_raw.statistics,
                        odds_bf, ds_fd_raw.lineups, ds_fd_raw.incidents, ds_fd_raw.betfair_odds)
first = run_smile_league(ds_first, "./data/match_day_ireland_first/july/";
                         label="Smile-DP · First Div", train=false)
