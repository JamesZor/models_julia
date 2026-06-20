# current_development/ab_test_dixon_coles/r09_totals_dispersion.jl
#
# Runner for the totals dispersion / calibration diagnostic (l09).
# Quantifies how COMPRESSED each model's expected-total-goals forecast is vs the
# market and vs realised goals — i.e. the "under-lean" we've been betting — and
# whether it is over-shooting. VALIDATED on server 2026-06-19; results inline below.

using Revise
using BayesianFootball
using DataFrames
using Statistics
using ThreadPinning

pinthreads(:cores)

const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions
const Data = BayesianFootball.Data
const Features = BayesianFootball.Features

include("./current_development/ab_test_dixon_coles/l09_totals_dispersion.jl")

# ==========================================
# 1. DATA  (market reference = ds.odds bookmaker totals; Betfair has no O/U here)
# ==========================================
println("[INFO] Loading Ireland DataStore...")
ds = Data.load_datastore_cached(Data.Ireland())
odds = ds.odds            # de-vigged bookmaker O/U ladder (prob_fair_close)

# ==========================================
# 2. PICK A GRID
# ==========================================
save_dir = "./data/dixon_coles_halflife_grid/";       n_models = 4    # r06
# save_dir = "./data/dixon_coles_market_weight_grid/"; n_models = 10   # r07

saved = Experiments.list_experiments(save_dir, data_dir="")
experiments = [Experiments.load_experiment(saved, i) for i in 1:n_models]

# ==========================================
# 3. RUN
# ==========================================
report_df, permatch_df = run_totals_dispersion(ds, odds, experiments)
println("\ndisp_ratio<1 ⇒ compressed | slope>1 ⇒ compressed | cor_*_real = who predicts goals")
show(Base.sort(report_df, :disp_ratio), allrows=true, allcols=true); println()

for exp in experiments
    name = String(exp.config.name)
    sub = subset(permatch_df, :model_name => ByRow(==(name)))
    println("\n--- Calibration buckets (by market_Etot): $name ---")
    show(totals_calibration_buckets(sub; nbins=5), allrows=true); println()
end

#= ============================================================================
VALIDATED RESULTS (Ireland, n=281 OOS matches, 2026-06-19)

Market (bookmaker O/U) reference: mean E[tot] 2.58, sd 0.261, slope vs realised 0.688,
cor vs realised 0.111.  Realised mean 2.53, sd 1.63 (totals are very noisy).

HALF-LIFE grid (mw fixed 0.4):  compression is structural, ~constant.
  model            disp_ratio  slope_model  cor_model_real
  HalfLife_120        0.506        1.123         0.091
  HalfLife_60         0.521        1.262         0.106   <- best totals signal
  HalfLife_30         0.566        0.969         0.088
  HalfLife_14         0.601        0.980         0.095

MARKET-WEIGHT grid (half-life fixed 60):  the real lever. disp_ratio is U-shaped
in market_weight; cor_model_real peaks at LOW mw then collapses when over-anchored.
  mw     disp_ratio  slope_model  cor_model_real   corr_mm
  0.0       1.458       0.204         0.048          0.198   <- pure model: OVER-dispersed & noisy
  0.1       0.603       1.164         0.113          0.255
  0.25      0.537       1.333         0.115          0.261   <- best totals signal (≈ market 0.111)
  0.4       0.518       1.221         0.102          0.253
  0.5(live) ~0.51        ~1.2          ~0.10          ~0.25  (interpolated)
  0.6       0.556       0.890         0.080          0.194
  0.8       0.584       0.679         0.064          0.158
  1.0       0.629       0.555         0.056          0.117
  1.25      0.672       0.462         0.050          0.092
  1.5       0.698       0.406         0.046          0.077
  2.0       0.747       0.412         0.049          0.052

KEY FINDINGS
1. The pure structural model (mw=0) is OVER-dispersed (sd 0.381 > market 0.261) but
   its totals are NOISE (cor 0.048). It spreads totals wide and WRONG.
2. The market pillar is a DENOISER. At mw 0.1–0.4 it shrinks that spurious spread
   (disp_ratio→0.5, the "compression" seen on the live dashboard) AND lifts the
   predictive correlation to ~match the market (0.10–0.115).
3. NO config beats the market at forecasting single-match totals (best cor 0.115 vs
   0.111). The totals edge is therefore NOT superior goal prediction — it is fading
   the market's mild OVER-dispersion (market slope 0.688 < 1) by mean-reverting
   extreme totals back toward ~2.5.
4. Over-anchoring (mw>0.6) de-compresses toward the market's spread but cor COLLAPSES
   to 0.05 (worse than the market). So raising market_weight to "fix" compression is
   counterproductive — it destroys signal AND erodes the edge.
5. Sweet spot for totals = mw 0.25–0.4 (matches r07's 1X2 logloss/edge sweet spot).
   Live model at mw=0.5 is marginally over-anchored for totals; 0.4 is the better pick.
============================================================================= =#

println("\nDone — totals dispersion scan complete.")
