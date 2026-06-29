#=
RUNNER for l04_split_market_negbin.jl — SplitMarketDoubleNegBinModel.

Single-split diagnostic. The NegBin sibling of r02: same split (level/supremacy) market pillar,
but goals ~ RobustNegativeBinomial(r, λ) so the model carries a structural DISPERSION the market's
independent-Poisson template ignores (this is what moves BTTS / correct-score → the derived-market
edge). Defaults to Ireland FIRST DIVISION (718, V/M≈1.14) where the dispersion actually bites — on
the near-Poisson top flight (79, V/M 0.94) r fits large and this ≈ the double-Poisson.

Goals:
  1. CONVERGES — R-hat ≤ ~1.05 incl. σ_sup, σ_lev and the dispersion r.
  2. DISPERSION is real — fitted r is finite/small (over-dispersion), not r→∞ (=Poisson).

Run after: git push -> git pull (server) -> RESTART REPL, then:
    include("current_development/split_market_pillar/r08_run_split_negbin.jl")
=#

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
const Data        = BayesianFootball.Data

include("current_development/split_market_pillar/l04_split_market_negbin.jl")

# ==========================================
# 1. DATA — Betfair market pillar
# ==========================================
# 718 = Ireland First Division (over-dispersed, NB regime). Swap to Data.Ireland() for the
# near-Poisson top flight (there NegBin ≈ double-Poisson).
segment = Data.Ireland()
ds = Data.load_datastore_cached(segment)
odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds_market = Data.DataStore(
    ds.segment, ds.matches, ds.statistics, odds,
    ds.lineups, ds.incidents, ds.betfair_odds
)

save_dir = "./data/split_market_dev_area/"
mkpath(save_dir)

# ==========================================
# 2. SHARED COMPONENT CONFIG
# ==========================================
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()        # NegBin r (now USED)
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
feature_cfg_bayes = Features.PlayerRatingsFeature(Features.BayesianTracker(6.5, 1.0, 0.5, 0.01))
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

# ==========================================
# 3. THE SPLIT-MARKET NEGBIN MODEL (both marginals anchored)
# ==========================================
model = SplitMarketDoubleNegBinModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    market_on              = true,
    supremacy_weight       = 1.0,
    level_weight           = 1.0,    # anchor BOTH marginals (totals rate + supremacy)
)

task = Experiments.create_experiment_task(
    ds_market, model, "split_negbin_r1", save_dir;
    target_seasons  = ["2026"],
    history_seasons = 2,
    warmup_period   = 21,
    dynamics_col    = :match_week,
    samples         = 1000,
    warmup          = 500,
    chains          = 4,
    use_queue       = true,
    max_depth       = 10,
)

results = Experiments.run_experiment(task)
Experiments.save_experiment(results)

# ==========================================
# 4. CONVERGENCE + DISPERSION DIAGNOSTICS
# ==========================================
chains_obj = Experiments.Diagnostics.extract_chains(ds_market, results)
println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv = Experiments.Diagnostics.check_convergence(chains_obj)
display(conv.df)

#=
julia> display(conv.df)
35×11 DataFrame
 Row │ std         mean        ess      train_season  raw_symbol            rhat      target_season  fold   week   parameter             entity               
     │ Float64     Float64     Float64  String        Symbol                Float64   String         Int64  Int64  String                String               
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │  0.327855    3.25026        NaN  2026          ν_xg                  1.00062   2026               0      0  ν_xg                  global
   2 │  0.0353883   0.386455       NaN  2026          σ_sup                 1.00051   2026               0      0  σ_sup                 global
   3 │  0.0244765   0.159793       NaN  2026          σ_lev                 1.00131   2026               0      0  σ_lev                 global
   4 │  0.0504884   0.225327       NaN  2026          ha.γ_team_raw[1]      1.00023   2026               0      0  home_advantage        bohemian
   5 │  0.0518027   0.211407       NaN  2026          ha.γ_team_raw[2]      1.00069   2026               0      0  home_advantage        cork-city
   6 │  0.0499821   0.218334       NaN  2026          ha.γ_team_raw[3]      0.999684  2026               0      0  home_advantage        derry-city
   7 │  0.0504883   0.212006       NaN  2026          ha.γ_team_raw[4]      0.999529  2026               0      0  home_advantage        drogheda-united
   8 │  0.0539797   0.220687       NaN  2026          ha.γ_team_raw[5]      1.0014    2026               0      0  home_advantage        dundalk-fc
   9 │  0.0500719   0.229292       NaN  2026          ha.γ_team_raw[6]      1.00097   2026               0      0  home_advantage        galway-united
  10 │  0.0496657   0.219012       NaN  2026          ha.γ_team_raw[7]      1.00076   2026               0      0  home_advantage        shamrock-rovers
  11 │  0.0504003   0.22475        NaN  2026          ha.γ_team_raw[8]      1.00301   2026               0      0  home_advantage        shelbourne
  12 │  0.0499427   0.224939       NaN  2026          ha.γ_team_raw[9]      1.00322   2026               0      0  home_advantage        sligo-rovers
  13 │  0.0490406   0.224916       NaN  2026          ha.γ_team_raw[10]     1.00035   2026               0      0  home_advantage        st-patricks-athletic
  14 │  0.0499119   0.227651       NaN  2026          ha.γ_team_raw[11]     0.999766  2026               0      0  home_advantage        waterford-fc
  15 │  0.0441482   0.221234       NaN  2026          ha.γ_base             1.00145   2026               0      0  ha.γ_base             global
  16 │  0.0212589   0.0253389      NaN  2026          ha.σ_γ                1.0011    2026               0      0  ha.σ_γ                global
  17 │  0.0674408   1.09218        NaN  2026          kap.κ_team_raw[1]     1.00208   2026               0      0  kappa                 bohemian
  18 │  0.061619    1.04586        NaN  2026          kap.κ_team_raw[2]     1.00039   2026               0      0  kappa                 cork-city
  19 │  0.067265    1.02187        NaN  2026          kap.κ_team_raw[3]     1.00124   2026               0      0  kappa                 derry-city
  20 │  0.0637074   1.02807        NaN  2026          kap.κ_team_raw[4]     1.00102   2026               0      0  kappa                 drogheda-united
  21 │  0.0753695   1.06509        NaN  2026          kap.κ_team_raw[5]     1.00108   2026               0      0  kappa                 dundalk-fc
  22 │  0.0615613   1.06956        NaN  2026          kap.κ_team_raw[6]     1.00234   2026               0      0  kappa                 galway-united
  23 │  0.0622731   1.0491         NaN  2026          kap.κ_team_raw[7]     1.00075   2026               0      0  kappa                 shamrock-rovers
  24 │  0.0630706   1.05498        NaN  2026          kap.κ_team_raw[8]     0.999579  2026               0      0  kappa                 shelbourne
  25 │  0.06887     1.09419        NaN  2026          kap.κ_team_raw[9]     1.00109   2026               0      0  kappa                 sligo-rovers
  26 │  0.0635182   1.08197        NaN  2026          kap.κ_team_raw[10]    1.00102   2026               0      0  kappa                 st-patricks-athletic
  27 │  0.0705935   1.09689        NaN  2026          kap.κ_team_raw[11]    1.00038   2026               0      0  kappa                 waterford-fc
  28 │  0.0851475   0.643919       NaN  2026          kap.κ_base            1.00027   2026               0      0  kap.κ_base            global
  29 │  0.0415807   0.068086       NaN  2026          kap.σ_κ               1.00244   2026               0      0  kap.σ_κ               global
  30 │  0.0672413  -0.125539       NaN  2026          p_dyn.w_G_att         1.00049   2026               0      0  p_dyn.w_G_att         global
  31 │  0.0658728   0.169177       NaN  2026          p_dyn.w_G_def         1.00088   2026               0      0  p_dyn.w_G_def         global
  32 │  0.0155974   0.108839       NaN  2026          p_dyn.w_Outfield_att  1.00118   2026               0      0  p_dyn.w_Outfield_att  global
  33 │  0.0144181  -0.123087       NaN  2026          p_dyn.w_Outfield_def  1.00061   2026               0      0  p_dyn.w_Outfield_def  global
  34 │ 21.3183     32.5835         NaN  2026          disp.log_r            1.00059   2026               0      0  r_home                global
  35 │ 10.3723     26.3158         NaN  2026          disp.log_r            1.00059   2026               0      0  r_away                global
=#


println("\n--- Dispersion rows (the NegBin r; small r = over-dispersed, large r ≈ Poisson) ---")
disp_rows = filter(r -> occursin("disp", lowercase(string(r.raw_symbol))) ||
                        occursin("disp", lowercase(string(r.parameter))), conv.df)
isempty(disp_rows) ? println("(no 'disp' rows matched — inspect conv.df raw_symbol column)") :
                     display(disp_rows[:, [:raw_symbol, :mean, :std, :rhat]])

println("""

Read:
 • σ_sup (who-wins disagreement) and σ_lev (totals disagreement) — both sampled, should be finite.
 • Dispersion r: a SMALL finite r (say < ~15) means real over-dispersion the NegBin is capturing;
   r drifting very large means the data is ≈ Poisson and NegBin adds nothing (expected on tnmt 79).
 • Next: full-CV run + r06_per_line_eval.jl (point its include at l04) — the edge to look for is on
   BTTS / correct-score (where dispersion reshapes P(0)/tails), NOT totals.
""")
