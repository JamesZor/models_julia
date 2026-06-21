# current_development/ab_test_first_division/r02_double_negbin.jl
#
# Double NEGATIVE-BINOMIAL outfield-player engine on First Division (718), 2025/26,
# Betfair market pillar. Adds two NB variants to the r01 DoublePoisson/DixonColes
# A/B and compares all four:
#   - FD_NegBin_HomeAway      : NB goals, HomeAwayDispersion (scalar r_h, r_a)
#   - FD_NegBin_Hierarchical  : NB goals, AdvancedVolatilityDispersion
#                               (HIERARCHICAL r: per-team + per-month volatility
#                                around a global base + home offset)
#
# Motivation: the 718 EDA (eda/first_division_validation) found First Division is a
# genuine NB regime (V/M≈1.14, NB beats Poisson by 9–12 AIC), so the goals pillar
# should be over-dispersed. This tests whether that helps OOS pricing/edge, and
# whether hierarchical (team/month) dispersion beats a single home/away r.
#
# Sync: git push then `git pull` in /root/BayesianFootball. New src/ engine →
# manage_repl restart after pulling (struct + module include).

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using ThreadPinning
using ProgressMeter
using GLM
pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Evaluation  = BayesianFootball.Evaluation
const BackTesting = BayesianFootball.BackTesting
const Predictions = BayesianFootball.Predictions
const D           = BayesianFootball.Data

# ==========================================
# 1. DATA — 718 with Betfair odds in the .odds slot (market pillar + eval)
# ==========================================
ds_raw = D.load_datastore_cached(D.IrelandFirstDivision())
odds_bf = D.summarize_betfair_market(ds_raw, open_window=(-100000.0,-10.0), close_window=(-20.0,0.0))
ds = D.DataStore(ds_raw.segment, ds_raw.matches, ds_raw.statistics, odds_bf,
                 ds_raw.lineups, ds_raw.incidents, ds_raw.betfair_odds)
ds1 = ds
save_dir = "/root/BayesianFootball/data/first_division_ab/"; mkpath(save_dir)

# ==========================================
# 2. SHARED COMPONENTS
# ==========================================
inter_cfg = PreGame.GlobalInterception()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)
tracker   = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
frat      = Features.PlayerRatingsFeature(tracker)

samples=800; warmup=300; chains=4; target_seasons=["2025","2026"]

# ==========================================
# 3. NB MODELS  (goals + xG + market pillars; NB goals dispersion varies)
# ==========================================
model_nb_ha = PreGame.DynamicDoubleNegBinXGOutfieldPlayerTimeDecayModel(
    interception_config=inter_cfg, player_dynamics_config=dyn_cfg,
    dispersion_config=PreGame.HomeAwayDispersion(),
    homeadvantage_config=ha_cfg, kappa_config=kap_cfg,
    player_ratings_feature=frat, market_weight=0.4)

model_nb_hier = PreGame.DynamicDoubleNegBinXGOutfieldPlayerTimeDecayModel(
    interception_config=inter_cfg, player_dynamics_config=dyn_cfg,
    dispersion_config=PreGame.AdvancedVolatilityDispersion(),
    homeadvantage_config=ha_cfg, kappa_config=kap_cfg,
    player_ratings_feature=frat, market_weight=0.4)

task_nb_ha = Experiments.create_experiment_task(ds, model_nb_ha, "FD_NegBin_HomeAway", save_dir;
    target_seasons=target_seasons, dynamics_col=:match_biweek,
    warmup_period=0, samples=samples, warmup=warmup, chains=chains, use_queue=true)
task_nb_hier = Experiments.create_experiment_task(ds, model_nb_hier, "FD_NegBin_Hierarchical", save_dir;
    target_seasons=target_seasons, dynamics_col=:match_biweek,
    warmup_period=0, samples=samples, warmup=warmup, chains=chains, use_queue=true)

println("--- Running FD NegBin HomeAway ---")
res_nb_ha = Experiments.run_experiment(task_nb_ha); Experiments.save_experiment(res_nb_ha)
println("--- Running FD NegBin Hierarchical ---")
res_nb_hier = Experiments.run_experiment(task_nb_hier); Experiments.save_experiment(res_nb_hier)

# ==========================================
# 4. EVALUATE ALL FOUR (load DP + DC from r01, add the two NB)
# ==========================================
saved = Experiments.list_experiments(save_dir, data_dir="")
# pick the latest of each by name
pick(nm) = Experiments.load_experiment(saved, findfirst(p -> occursin(nm, p), saved))
res_dp = pick("FD_DoublePoisson_Market")
res_dc = pick("FD_DixonColes_Market")
all_results = [res_dp, res_dc, res_nb_ha, res_nb_hier]

println("\n=== LogLoss (Betfair) ===")
ll = Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds1)
show(ll, allrows=true, allcols=true)

println("\n=== GLM Edge (Betfair) ===")
ge = Evaluation.evaluate_experiments(Evaluation.GLMEdge(), all_results, ds1)
show(ge, allrows=true, allcols=true)

#= RESULT — LogLoss + GLM (4-way)   [718, 2025/26, Betfair pillar; train times in brackets]
LogLoss (diff = model_ll - market_ll; more negative beats the market more):
  DixonColes        -0.03245   [1h48m]
  NegBin_HomeAway   -0.03226   [55m]    ← NB ≈ DC, clearly > DP
  NegBin_Hierarchical -0.03218 [2h02m]
  DoublePoisson     -0.02948   [60m]

GLM edge (spread_fair coef, p):
  DixonColes        1.477  p=0.088   ← best (marginal)
  NegBin_HomeAway   1.338  p=0.143
  NegBin_Hierarchical 1.326 p=0.146
  DoublePoisson     0.859  p=0.306

→ Both NB variants BEAT DoublePoisson and ~tie DixonColes on calibration/edge.
  HIERARCHICAL dispersion ≈ HomeAway on every metric (no gain) but costs ~2x train
  time (2h02m vs 55m) — NOT worth it; use HomeAwayDispersion.
=#

# ==========================================
# 5. BACKTEST + PER-MARKET (ROI & growth factor)
# ==========================================
ledger = BackTesting.run_backtest(ds1, all_results, [BayesianFootball.Signals.BayesianKelly()];
    market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG)
tearsheet = BackTesting.generate_tearsheet(ledger)

bets_df = filter(r -> r.stake > 0, ledger.df)
agg = combine(groupby(bets_df, :model_name),
    nrow=>:bets, :stake=>sum=>:turnover, :pnl=>sum=>:profit,
    :pnl=>(p->exp(sum(log.(max.(1e-8, 1.0 .+ p)))))=>:growth_factor)
agg.roi_pct = 100 .* agg.profit ./ agg.turnover
println("\n=== Backtest aggregate (ROI & growth factor) ==="); show(agg, allrows=true, allcols=true)

bt_market = combine(groupby(bets_df, [:model_name, :selection]),
    nrow=>:bets, :stake=>sum=>:turnover, :pnl=>sum=>:profit,
    :pnl=>(p->exp(sum(log.(max.(1e-8, 1.0 .+ p)))))=>:growth_factor,
    :is_winner=>(w->100*mean(skipmissing(w)))=>:win_pct)
bt_market.roi_pct = 100 .* bt_market.profit ./ bt_market.turnover
sort!(bt_market, [:model_name, :roi_pct], rev=[false,true])
println("\n=== Backtest per-market ==="); show(bt_market, allrows=true, allcols=true)

#= RESULT — Backtest (4-way)   BayesianKelly, Betfair odds
Aggregate (growth_factor = ∏(1+pnl): naive all-market full-Kelly compounding):
  model                bets  ROI%    growth_factor
  NegBin_HomeAway      762   12.49   0.626   ← BEST ROI
  NegBin_Hierarchical  762   12.46   0.592
  DoublePoisson        769   10.24   0.452
  DixonColes           779    9.67   0.616
→ Both NB variants have the HIGHEST ROI (12.5% vs DP 10.2%, DC 9.7%). NB's edge comes
  mostly from better HOME pricing: per-market HOME growth_factor NB-HA 6.40x vs DC 5.21x,
  DP 4.04x (home ROI ~42%). away is a near-wipeout for all (gf ~0.16x); over_45/55 noise.
  NB-HomeAway ≈ NB-Hierarchical per-market (hierarchical adds nothing).
NOTE: aggregate growth_factor<1 for ALL models — naive "bet every market at full Kelly"
  overbets (counts simultaneous bets as sequential) → value-destroying compounding despite
  positive ROI. Consistent with portfolio-kelly-partial-hedge: cap simultaneous stakes.
  Per-market isolated growth (e.g. HOME 6.4x) is the cleaner read.

VERDICT: the double-NB engine (motivated by the 718 NB EDA) is the best outfield engine for
First Division on ROI, matching DC on LogLoss/edge. Use HomeAwayDispersion (hierarchical not
worth it). NB also samples FASTER/healthier than DC (55m vs 1h48m; ε up to 0.025 vs DC's tiny steps).
=#

println("\nDone! Double-NB A/B vs DoublePoisson/DixonColes complete.")
