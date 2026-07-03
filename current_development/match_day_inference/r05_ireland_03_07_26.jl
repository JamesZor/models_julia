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

save_dir = "./data/match_day_ireland/july/"


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

# ==========================================
# 2. MODEL — src struct, keeper defaults (smile_weight=0.5, sup=1.0, Kmax=4)
# ==========================================
model_2 = PreGame.DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel(
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
KMAX = model_2.smile_feature.Kmax
nK   = KMAX + 1


warmup_period =  23

task = Experiments.create_experiment_task(
    ds, 
    model, 
    "dixon_coles_$(today())", 
    save_dir; 
    target_seasons=["2026"],
    history_seasons = 2,
    warmup_period =  warmup_period,
    dynamics_col=:match_week,
    samples=2000,      # Small samples for fast runner testing
    warmup=1000,        # Small warmup for fast runner testing
    chains=16,         # 2 chains for fast runner testing
    use_queue=true
)

task_2 = Experiments.create_experiment_task(
    ds, 
    model_2, 
    "double_poisson_smile_$(today())", 
    save_dir; 
    target_seasons=["2026"],
    history_seasons = 2,
    warmup_period =  warmup_period,
    dynamics_col=:match_week,
    samples=2000,      # Small samples for fast runner testing
    warmup=1000,        # Small warmup for fast runner testing
    chains=16,         # 2 chains for fast runner testing
    use_queue=true
)


#
results = Experiments.run_experiment(task)
Experiments.save_experiment(results)



results_2 = Experiments.run_experiment(task_2)
Experiments.save_experiment(results_2)
# #
# #
# Experiments.save_experiment(results)
#

# ==========================================
# DIAGNOSTICS
# ==========================================
chains_df_all = Diagnostics.extract_chains(ds, results_2)

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
expr   = Experiments.load_experiment(find_experiment_path(saved_fiels, "dixon_coles"))
expr_2 = Experiments.load_experiment(find_experiment_path(saved_fiels, "double_poisson_smile"))

label_1 = "DixonColes-XG"
label_2 = "Smile-DP"

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
# 5. RUN MATCH DAY INFERENCE (PPD Generation) — both models
# ==========================================
println("\n=== 4. Running Match Day Inference ===")
ppd   = compute_todays_matches_pdds(ds, expr, todays_matches, json_lineups_dir)
# Smile model: keep the latents (λ draws) so the unified structural-Kelly panel can reuse
# the same posterior without re-running MCMC extraction.
latents_2 = compute_todays_matches_latents(ds, expr_2, todays_matches, json_lineups_dir)
ppd_2 = BayesianFootball.Predictions.model_inference(latents_2)
# NOTE: do NOT overwrite with `model_inference(ds, expr)` — that re-predicts the
# saved experiment's stale target matchday, not today's fixtures.

# Paper bankroll for the unified staking panel (£); set to 0.0 for %-only display.
BANKROLL = 50.0

println("\n$label_1 1X2 Probabilities:")
show(subset(ppd.df, :market_name => ByRow(==("1X2"))))
println()

println("\n$label_2 1X2 Probabilities:")
show(subset(ppd_2.df, :market_name => ByRow(==("1X2"))))
println()

# ==========================================
# 6. BETFAIR LIVE STREAM & KELLY STAKING (MODEL COMPARISON)
# ==========================================
println("\n=== 5. Connecting to Redis & Kelly Staking ===")
redis_host = get(ENV, "REDIS_HOST", "100.124.38.117")  # home server over Tailscale
redis_port = parse(Int, get(ENV, "REDIS_PORT", "6379"))

try
    println("🔗 Connecting to Redis at $redis_host:$redis_port...")
    redis_conn = RedisConnection(host=redis_host, port=redis_port)

    # Run the live betting dashboard comparing both models side-by-side (one-shot display)
    print_live_betting_dashboard_compare(ppd, label_1, ppd_2, label_2, redis_conn, todays_matches; kelly_fraction=0.00)

    # UNIFIED structural-Kelly portfolio panel (Smile-DP, Over/Under + BTTS only, cap 0.10, 2% comm)
    print_unified_staking_dashboard(latents_2, redis_conn, todays_matches;
        label=label_2, cap=0.10, commission=0.02, bankroll=BANKROLL)

    # Example polling loop (uncomment to run interactive live poller):
    # println("Press Ctrl+C to exit live betting monitor.")
    # while true
    #     print("\e[2J\e[H") # Clear terminal
    #     print_live_betting_dashboard_compare(ppd, label_1, ppd_2, label_2, redis_conn, todays_matches; kelly_fraction=0.5)
    #     print_unified_staking_dashboard(latents_2, redis_conn, todays_matches; label=label_2, cap=0.10, commission=0.02, bankroll=BANKROLL)
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

