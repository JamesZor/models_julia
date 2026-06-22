# current_development/ab_test_first_division/r03_halflife_grid.jl
#
# Grid search over the time-decay HALF-LIFE for the double-NB HomeAway-dispersion
# outfield engine on First Division (718), 2025/26, Betfair market pillar.
# Everything else fixed (market_weight=0.4, NB goals + HomeAwayDispersion).
#
# match_weights_i = 0.5 ^ (Δdays_i / half_life). Short half-life = recency-heavy
# (less effective data, more reactive); long = gentler decay (more data, staler form).
#
# Grid: [30, 45, 60, 90, 120] days. The 60-day model already exists from r02
# (FD_NegBin_HomeAway) and is REUSED — only 30/45/90/120 are trained here.
#
# Reports, per half-life: LogLoss, GLM-edge, RQR (goals calibration), and backtest
# ROI + growth factor (aggregate AND per-market line).
#
# Sync + (no src change → no restart needed). Long run: 4 new fits × ~55m ≈ 3.7h.

using Revise, BayesianFootball, DataFrames, Distributions, Statistics, GLM
using ThreadPinning, ProgressMeter; pinthreads(:cores)

const PreGame=BayesianFootball.Models.PreGame; const Features=BayesianFootball.Features
const Experiments=BayesianFootball.Experiments; const Evaluation=BayesianFootball.Evaluation
const BackTesting=BayesianFootball.BackTesting; const Predictions=BayesianFootball.Predictions
const D=BayesianFootball.Data

# ---- data ----
ds_raw = D.load_datastore_cached(D.IrelandFirstDivision())
odds_bf = D.summarize_betfair_market(ds_raw, open_window=(-100000.0,-10.0), close_window=(-20.0,0.0))
ds = D.DataStore(ds_raw.segment, ds_raw.matches, ds_raw.statistics, odds_bf, ds_raw.lineups, ds_raw.incidents, ds_raw.betfair_odds)
save_dir = "/root/BayesianFootball/data/fd_halflife_grid/"; mkpath(save_dir)

# ---- fixed components ----
inter_cfg=PreGame.GlobalInterception(); ha_cfg=PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg=PreGame.HierarchicalTeamKappa(); tracker=Features.BayesianTracker(6.5,1.0,0.5,0.01)
frat=Features.PlayerRatingsFeature(tracker)
samples=800; warmup=300; chains=4; target_seasons=["2025","2026"]

HALF_LIVES = [30.0, 45.0, 90.0, 120.0]   # 60 reused from r02

function train_hl(hl)
    m = PreGame.DynamicDoubleNegBinXGOutfieldPlayerTimeDecayModel(
        interception_config=inter_cfg, player_dynamics_config=PreGame.OutfieldPlayerDynamicsConfig(days_half_life=hl),
        dispersion_config=PreGame.HomeAwayDispersion(), homeadvantage_config=ha_cfg, kappa_config=kap_cfg,
        player_ratings_feature=frat, market_weight=0.4)
    t = Experiments.create_experiment_task(ds, m, "FD_NB_HL$(Int(hl))", save_dir;
        target_seasons=target_seasons, dynamics_col=:match_biweek, warmup_period=0,
        samples=samples, warmup=warmup, chains=chains, use_queue=true)
    r = Experiments.run_experiment(t); Experiments.save_experiment(r); return r
end

for hl in HALF_LIVES
    println("--- Training NB HomeAway half-life=$(hl) ---")
    train_hl(hl)
end

# ==========================================
# EVALUATION across the grid
# ==========================================
# Load the grid (+ reuse the r02 60-day model)
grid_saved = Experiments.list_experiments(save_dir, data_dir="")
r02_saved  = Experiments.list_experiments("/root/BayesianFootball/data/first_division_ab/", data_dir="")
load_one(saved, key) = Experiments.load_experiment(saved, findfirst(p->occursin(key,p), saved))
results = Dict{Int,Any}()
for hl in [30,45,90,120]; results[hl] = load_one(grid_saved, "FD_NB_HL$(hl)_"); end
results[60] = load_one(r02_saved, "FD_NegBin_HomeAway")
hls = sort(collect(keys(results)))
all_results = [results[hl] for hl in hls]

# ---- LogLoss + GLM + RQR ----
ll = Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds)
ge = Evaluation.evaluate_experiments(Evaluation.GLMEdge(), all_results, ds)
rq = Evaluation.evaluate_experiments(Evaluation.RQR(), all_results, ds)
println("\n=== LogLoss ==="); show(ll[:,[:model,:logloss_overall_diff_ll]], allrows=true)
println("\n=== GLM edge ==="); show(ge[:,[:model,:glmedge_spread_fair_coef,:glmedge_spread_fair_p_value]], allrows=true)
println("\n=== RQR ==="); show(rq[:,[:model,:rqr_all_std,:rqr_all_kurtosis,:rqr_all_shapiro_p]], allrows=true)

# ---- Backtest: aggregate + per-market ROI & growth factor ----
ledger = BackTesting.run_backtest(ds, all_results, [BayesianFootball.Signals.BayesianKelly()];
    market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG)
bets_df = filter(r->r.stake>0, ledger.df)
gf(p) = exp(sum(log.(max.(1e-8, 1.0 .+ p))))
agg = combine(groupby(bets_df, :model_name), nrow=>:bets, :stake=>sum=>:turn, :pnl=>sum=>:profit, :pnl=>gf=>:growth_factor)
agg.roi_pct = 100 .* agg.profit ./ agg.turn
println("\n=== Backtest aggregate ==="); show(agg, allrows=true, allcols=true)

bt_market = combine(groupby(bets_df, [:model_name,:selection]), nrow=>:bets, :stake=>sum=>:turn, :pnl=>sum=>:profit, :pnl=>gf=>:growth_factor)
bt_market.roi_pct = 100 .* bt_market.profit ./ bt_market.turn
println("\n=== Backtest per-market ==="); show(bt_market, allrows=true, allcols=true)

#= RESULT — HALF-LIFE GRID (FD 718, 2025/26, 275 matches, full-Kelly, no costs)
# Full writeup: HALFLIFE_GRID_RESULTS.md
#
# Aggregate (monotonic — shorter HL wins on money):
#   HL   bets  ROI%   growth
#   30   760   13.82  1.044   <- only growth>1.0; BEST
#   45   763   13.10  0.816
#   60   762   12.49  0.626
#   90   769   11.08  0.323
#   120  773   10.32  0.217
#
# LogLoss diff_ll (INVERTED: longer HL marginally better, but range tiny 0.0016):
#   30 -0.0332 | 45 -0.0326 | 60 -0.0323 | 90 -0.0317 | 120 -0.0316
# GLM spread coef/p (agrees w/ ROI: short=strongest): 30:1.63/.071  60:1.34/.143  120:1.23/.183
# RQR: flat & good — all pass Shapiro (HL120 borderline .047), std~0.94-0.99. Calibration not the lever.
#
# Per-market driver: HOME ~42% ROI ALL half-lives (robust); DRAW +25-28% all. The GAP comes from
# UNDER ladder flipping with HL: under_25 HL30 +9.8% -> HL120 -10.2%; under_35 +8.4% -> -4.7%.
# AWAY loss all (HL30 least-bad -5.6%, HL120 -18.2%). over_25 loser all HLs (adverse selection).
#
# VERDICT: use HL~=30 (recency-heavy). Monotonic trend -> worth sweeping below 30 (21/14), but
# watch effective sample size (275 matches starves team hierarchy at very short HL).
=#

println("\nDone! Half-life grid complete.")
