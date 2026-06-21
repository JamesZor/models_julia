# current_development/ab_test_first_division/r01_ab_test_first_division.jl
#
# Outfield-player A/B for Ireland First Division (718), seasons 2025+2026.
# Models (pillars: goals + xG + market):
#   1. Double Poisson  (DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel)
#   2. Dixon-Coles     (DynamicDixonColesXGOutfieldPlayerTimeDecayModel)
#   (Double Negative-Binomial engine to be added next — does not exist yet.)
#
# KEY vs r02_ab_test_ireland.jl: the market pillar is anchored to the **Betfair**
# market, NOT SofaScore. SofaScore `ds.odds` only carries the 1x2 result market,
# which under-identifies the home/away goal rates; Betfair carries 1x2 + BTTS +
# Over/Under lines, giving a far better market-implied (λ_h, λ_a). We therefore
# swap `summarize_betfair_market(ds)` into the `.odds` slot BEFORE task creation,
# so the SAME Betfair frame drives both the training market pillar and evaluation.
#
# Sync first: local edits reach mcmc-beast only via git push then `git pull` in
# /root/BayesianFootball. No src/ change here → no REPL restart needed; just include.

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using ThreadPinning
using ProgressMeter

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Evaluation  = BayesianFootball.Evaluation
const BackTesting = BayesianFootball.BackTesting
const Data        = BayesianFootball.Data

# ==========================================
# 1. SETUP & DATA  (718 First Division)
# ==========================================
println("[INFO] Loading Ireland First Division (718) DataStore...")
ds_raw = Data.load_datastore_cached(Data.IrelandFirstDivision())

save_dir::String = "./data/first_division_ab/"
mkpath(save_dir)

# --- Betfair market as the odds source (for BOTH training pillar and eval) ---
# close_window = closing line (what the market feature reads via prob_fair_close).
odds_bf = Data.summarize_betfair_market(
    ds_raw,
    open_window  = (-100000.0, -10.0),
    close_window = (-20.0, 0.0),
)
ds = Data.DataStore(
    ds_raw.segment, ds_raw.matches, ds_raw.statistics,
    odds_bf, ds_raw.lineups, ds_raw.incidents, ds_raw.betfair_odds,
)

# --- Sanity: confirm Betfair provides the selections the inversion needs ---
println("[CHECK] Betfair summary rows: ", nrow(odds_bf))
if nrow(odds_bf) > 0
    println("[CHECK] selections present: ", sort(unique(string.(odds_bf.selection))))
    n_mkts = combine(groupby(odds_bf, :match_id), nrow => :n)
    println("[CHECK] matches with ≥1 betfair market row: ", nrow(n_mkts),
            " (median markets/match ", Int(round(median(n_mkts.n))), ")")
end

#= RESULT — data/coverage
Betfair summary: 6797 rows; 682/914 matches carry ≥1 market row (median 9 markets/match).
selections present: home/draw/away, btts_yes/btts_no, over_/under_ 05..55, plus cs_*/dnb_*/dc_*.
→ all selections the market-inversion needs (1x2 + BTTS + O/U) are present.
Target window: 275 played matches in 2025+2026 (xG ~99% covered from 2023).
=#

# ==========================================
# 2. SHARED COMPONENT CONFIGURATION
# ==========================================
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life = 60.0)

tracker_bayes     = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

samples = 800
warmup  = 300
chains  = 4
target_seasons = ["2025", "2026"]

# ==========================================
# 3. MODEL INITIALIZATION  (goals + xG + market pillars)
# ==========================================
# Model 1: Double Poisson + Market
model_dp_m = PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight          = 0.4,
)

# Model 2: Dixon-Coles + Market
model_dc_m = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DixonColesMarketFeature(),
    market_weight          = 0.4,
)

# ==========================================
# 4. TASK CREATION
# ==========================================
task_dp_m = Experiments.create_experiment_task(
    ds, model_dp_m, "FD_DoublePoisson_Market", save_dir;
    target_seasons = target_seasons, dynamics_col = :match_biweek,
    warmup_period = 0, samples = samples, warmup = warmup, chains = chains, use_queue = true,
)

task_dc_m = Experiments.create_experiment_task(
    ds, model_dc_m, "FD_DixonColes_Market", save_dir;
    target_seasons = target_seasons, dynamics_col = :match_biweek,
    warmup_period = 0, samples = samples, warmup = warmup, chains = chains, use_queue = true,
)

# ==========================================
# 5. RUN EXPERIMENTS
# ==========================================
println("--- Running FD Double Poisson Market ---")
res_dp_m = Experiments.run_experiment(task_dp_m)
Experiments.save_experiment(res_dp_m)

println("--- Running FD Dixon Coles Market ---")
res_dc_m = Experiments.run_experiment(task_dc_m)
Experiments.save_experiment(res_dc_m)

all_results = [res_dp_m, res_dc_m]

# ==========================================
# 6. EVALUATION & BACKTESTING  (Betfair odds)
# ==========================================
# `ds` already carries the Betfair odds frame, so it doubles as the eval store.
ds1 = ds

println("\n===========================================")
println("📊 GLM Edge Evaluation (Betfair Odds)")
println("===========================================")
eval_glmedge = Evaluation.evaluate_experiments(Evaluation.GLMEdge(), all_results, ds1)
Evaluation.display_summary_metric(eval_glmedge, :glmedge)

#= RESULT — GLM Edge  (n_obs=1635; spread_fair_coef>0 & significant ⇒ genuine edge)
 model                    intercept(p)        prob_fair_coef(p)    spread_fair_coef   p_value
 FD_DixonColes_Market     -2.496 (1.8e-67)    5.095 (1.2e-69)      1.477              0.088   ← marginal edge
 FD_DoublePoisson_Market  -2.476 (1.2e-66)    5.042 (5.7e-69)      0.859              0.306   ← not significant
→ Calibration coefs strong for both. The edge term (spread_fair) is only marginally significant
  for DC and insignificant for DP — far weaker than 79 Premier (r02: spread_fair p≈1e-4). The
  First Division market is thinner/noisier, so the model's pricing edge over Betfair is small.
=#

println("\n===========================================")
println("📉 LogLoss Evaluation (Betfair Odds)")
println("===========================================")
eval_logloss = Evaluation.evaluate_experiments(Evaluation.LogLoss(), all_results, ds1)
Evaluation.display_summary_metric(eval_logloss, :logloss)

#= RESULT — LogLoss  (n_obs=1635; diff = model_ll − market_ll, NEGATIVE beats the market)
 model                    model_ll   market_ll   diff_ll
 FD_DixonColes_Market     0.55253    0.58498     -0.03245   ← best
 FD_DoublePoisson_Market  0.55550    0.58498     -0.02948
→ BOTH beat the Betfair 1x2 market on LogLoss; Dixon-Coles slightly better. Edge (~-0.03) is a
  touch below 79 Premier (r02: DC -0.034, DP -0.036) but clearly positive.
=#

println("\n===========================================")
println("💰 Backtesting Strategy (Kelly)")
println("===========================================")
ledger = BackTesting.run_backtest(
    ds1,
    all_results,
    [BayesianFootball.Signals.BayesianKelly()];
    market_config = BayesianFootball.Data.Markets.DEFAULT_MARKET_CONFIG,
)
tearsheet = BackTesting.generate_tearsheet(ledger)

println("\n>>> Backtest Comparison Summary:")
cols_to_show = [:model_name, :selection, :opportunities, :activity_pct, :bets_placed,
                :turnover, :profit, :roi_pct, :win_rate_pct]
show(tearsheet[:, cols_to_show], allrows = true)

#= RESULT — Backtest  (BayesianKelly, Betfair odds, aggregated over all markets)
 model                    bets   turnover   profit   ROI%
 FD_DoublePoisson_Market  767    51.10      5.23     10.23
 FD_DixonColes_Market     777    49.44      4.80      9.71
→ Both profitable in-sample-temporal; DP marginally higher aggregate ROI, DC marginally better
  calibration/LogLoss. Per-market (DC): the HOME market is the big winner (ROI ~41%, profit 4.13,
  91 bets) and away is the worst (-26%). over_45/over_55 show extreme ROI on TINY samples
  (≤14 bets) → noise, not signal. Treat aggregate ROI as encouraging but thin-market and
  optimistic (small n=275 target matches, no execution costs).

CAVEATS / OPS NOTES:
- Dixon-Coles sampling is SLOW on 718: DP 60min, DC 1h48m (NUTS hit tiny step sizes ε~3e-6 on
  some splits — a posterior-geometry pathology worth reparameterising before scaling up).
- The kaimon gate "failed" the run on a 10-min no-output timeout, but the Julia process kept
  training and saved BOTH experiments to ./data/first_division_ab/ — load via list_experiments.
- Market pillar = Betfair de-vigged closing line. For thin minor leagues the pillar may be better
  anchored to Bet365 (see memory betfair-vs-bet365-market-anchor) — revisit.
=#

# ==========================================
# 7. PER-MARKET BREAKDOWN  (GLM edge + backtest ROI & growth factor)
# ==========================================
using GLM
const Predictions = BayesianFootball.Predictions

# --- 7a. Per-market backtest: ROI + growth factor ---
# Growth factor = ∏(1 + pnl_i) over placed bets (pnl is already in bankroll-fraction
# units, so each bet multiplies wealth by 1+pnl); geometric growth/bet = gf^(1/n)-1.
bets_df = filter(r -> r.stake > 0, ledger.df)
bt_market = combine(groupby(bets_df, [:model_name, :selection]),
    nrow => :bets,
    :stake => sum => :turnover,
    :pnl   => sum => :profit,
    :pnl   => (p -> exp(sum(log.(max.(1e-8, 1.0 .+ p))))) => :growth_factor,
    :is_winner => (w -> 100 * mean(skipmissing(w))) => :win_pct)
bt_market.roi_pct = 100 .* bt_market.profit ./ bt_market.turnover
bt_market.growth_per_bet_pct = 100 .* (bt_market.growth_factor .^ (1 ./ bt_market.bets) .- 1)
sort!(bt_market, [:model_name, :roi_pct], rev = [false, true])
println("\n--- Per-market backtest (ROI & growth factor) ---")
show(bt_market, allrows = true, allcols = true)

# --- 7b. Per-market GLM edge ---
# NOTE: evaluate_experiments([GLMEdge([s]) for s in sels], …) does NOT work for a
# per-market table — GLMEdgeResult flattens under the constant name "glmedge", so
# every selection's columns collide and merge keeps only the last. Compute the
# per-selection logistic edge directly from one inference per model instead.
function permarket_glm(exp)
    latents = Experiments.extract_oos_predictions(ds1, exp)
    ppd = Predictions.model_inference(latents)
    mf  = transform(ppd.df, :distribution => ByRow(mean) => :prob_model)
    mf  = select(mf, :match_id, :market_name, :market_line, :selection, :prob_model)
    adf = innerjoin(ds1.odds, mf, on = [:match_id, :market_name, :market_line, :selection])
    dropmissing!(adf, [:prob_fair_close, :is_winner])
    adf.spread_fair = adf.prob_model .- adf.prob_fair_close
    adf.Y = Float64.(adf.is_winner)
    rows = NamedTuple[]
    for sub in groupby(adf, :selection)
        n = nrow(sub); (n >= 12 && length(unique(sub.Y)) == 2) || continue
        m = glm(@formula(Y ~ prob_fair_close + spread_fair), sub, Binomial(), LogitLink())
        ct = coeftable(m); idx = findfirst(==("spread_fair"), ct.rownms)
        push!(rows, (model = exp.config.name, selection = sub.selection[1], n_obs = n,
                     spread_coef = ct.cols[1][idx], spread_p = ct.cols[4][idx]))
    end
    return DataFrame(rows)
end
glm_market = vcat(permarket_glm(res_dp), permarket_glm(res_dc))
sort!(glm_market, [:model, :spread_p])
println("\n--- Per-market GLM edge (spread_fair coef & p; p<0.10 ⇒ edge) ---")
show(glm_market, allrows = true, allcols = true)

#= RESULT — PER-MARKET BREAKDOWN

GLM EDGE per market (spread_fair coef, p; p<0.10 = model prices an edge vs Betfair):
  Dixon-Coles                              Double-Poisson
   over_45  coef 53.2  p 0.016 *            over_45  coef 39.1  p 0.026 *
   over_15  coef 15.6  p 0.031 *            under_45 coef 35.3  p 0.033 *
   btts_no  coef 36.4  p 0.051 *            btts_yes coef 15.5  p 0.099 *
   over_25  coef  8.8  p 0.076 *            btts_no  coef 15.5  p 0.099 *
   under_15 coef 16.9  p 0.092 *            (over_15/under_15 p 0.19; home p 0.38)
   home     coef  2.1  p 0.349              away     coef  0.7  p 0.76
   away     coef -0.7  p 0.78
→ The genuine pricing edge concentrates on OVER lines (over_45/15/25) and BTTS — i.e. the model
  fades the market UNDER-pricing goals, consistent with 718 being a high-scoring/over-dispersed
  league (see first-division-718-signature). DC has more significant over-market edges than DP.
  1x2 (home/draw/away) shows NO significant GLM edge for either model. Large coefs on O/U lines
  are a scale artifact (small within-line spread variance) — read the p-value, not the magnitude.

BACKTEST per market (BayesianKelly; growth_factor = terminal wealth multiple betting that market):
  Dixon-Coles            bets  ROI%    growth_factor   Double-Poisson         bets  ROI%    gf
   home                   91    41.2    5.21 ×           home                   94    38.5    4.04 ×
   under_15               36    26.8    1.11 ×           under_15               43    31.8    1.22 ×
   over_15                56     9.7    1.33 ×           btts_no                35    12.7    1.09 ×
   btts_no                29     7.8    1.01 ×           over_15                51     6.6    1.16 ×
   over_25                84   -11.0    0.46 ×           over_25                82   -17.4    0.33 ×
   away                   83   -25.6    0.10 × (≈wipe)   away                   83    -6.5    0.20 ×
   over_45                15   358     2.16 × (TINY n)   over_45                13   363     2.26 × (TINY n)
→ HOME is the dominant profitable market (DC 5.2×, DP 4.0× wealth) — the bulk of both models' edge.
  AWAY is a near-wipeout (gf 0.10–0.20×). over_25/over_35 lose despite the GLM "edge" sign (the edge
  is in the EXTREME over lines, not 2.5). over_45/55 & under_05 ROIs are noise (≤15 bets).
  Reconcile GLM vs backtest: GLM edge sign ≠ profit — Kelly profit needs the edge AND favourable
  odds/variance; the real, robust money is HOME (1x2), where GLM is insignificant but Kelly + the
  model's team-strength calibration still extract value.
=#

println("\nDone! First Division outfield A/B (DP + DC) + per-market breakdown complete.")
