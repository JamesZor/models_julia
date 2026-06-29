#= 
Having issues with the split market pillar model ( think claude is confusing itself) 

1. run a dixon coles (double Poisson) {goals, xG market , outfield} model and show that is converges. 
2.  Create a model for the double Poisson {goals, xG market, outfield} with the makret pillar changed 
    using the normal distributions as the weighting - do more research / details.
3. 

=#



# 1. run a model

using Revise
using BayesianFootball
using DataFrames
using Distributions
using ThreadPinning
using ProgressMeter

pinthreads(:cores)

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Evaluation  = BayesianFootball.Evaluation
const BackTesting = BayesianFootball.BackTesting
const Data        = BayesianFootball.Data
const Signals     = BayesianFootball.Signals




ds = Data.load_datastore_cached(Data.Ireland())

odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds_market = Data.DataStore(
    ds.segment, ds.matches, ds.statistics, odds,
    ds.lineups, ds.incidents, ds.betfair_odds
)


save_dir = "./data/split_market_dev_area/"
mkpath(save_dir)

# ==========================================
# 2. SHARED COMPONENT CONFIG  (matches r06)
# ==========================================
# model prior settings
inter_cfg = PreGame.HierarchicalMonthlyInterception()
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()

# player model
tracker_bayes     = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

# configs for run
target_seasons = ["2026"]
dynamics_col   = :match_biweek
half_life      = 60.0
market_weight = 0.4


dyn_cfg = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=half_life)

model = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    dixon_coles_config     = PreGame.HierarchicalTeamDixonColesConfig(),
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DixonColesMarketFeature(),
    market_weight          = market_weight
)

task = Experiments.create_experiment_task(
    ds, 
    model, 
    "dixon_coles_r1", 
    save_dir; 
    target_seasons=["2026"],
    history_seasons = 2,
    warmup_period =  21,
    dynamics_col=:match_week,
    samples=1000,      # Small samples for fast runner testing
    warmup=500,        # Small warmup for fast runner testing
    chains=4,         # 2 chains for fast runner testing
    use_queue=true
)

#
results = Experiments.run_experiment(task)
Experiments.save_experiment(results)

saved = Experiments.list_experiments(save_dir, data_dir="")
results = Experiments.load_experiment(saved, 3)

chain = results.train

# DIAGNOSTICS
chains_df_all = Experiments.Diagnostics.extract_chains(ds, results)
println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv_diag_all = Experiments.Diagnostics.check_convergence(chains_df_all)
#=
ChainDiagnostic (Convergence)
  ✅ All parameters converged (R-hat ≤ 1.05)
=#

conv_diag_all.df

#=
julia> conv_diag_all.df
32×11 DataFrame
 Row │ std        mean        ess      train_season  raw_symbol            rhat      target_season  fold   week   parameter             entity               
     │ Float64    Float64     Float64  String        Symbol                Float64   String         Int64  Int64  String                String               
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 0.319134    3.27766        NaN  2026          ν_xg                  0.999808  2026               0      0  ν_xg                  global
   2 │ 0.018124    0.182197       NaN  2026          σ_market              1.00053   2026               0      0  σ_market              global
   3 │ 0.0647047   0.226516       NaN  2026          ha.γ_team_raw[1]      1.0009    2026               0      0  home_advantage        bohemian
   4 │ 0.0774437   0.176572       NaN  2026          ha.γ_team_raw[2]      1.00002   2026               0      0  home_advantage        cork-city
   5 │ 0.0651589   0.225863       NaN  2026          ha.γ_team_raw[3]      1.00032   2026               0      0  home_advantage        derry-city
   6 │ 0.0668564   0.199874       NaN  2026          ha.γ_team_raw[4]      1.00119   2026               0      0  home_advantage        drogheda-united
   7 │ 0.0777284   0.213001       NaN  2026          ha.γ_team_raw[5]      1.00172   2026               0      0  home_advantage        dundalk-fc
   8 │ 0.0642489   0.228239       NaN  2026          ha.γ_team_raw[6]      1.00104   2026               0      0  home_advantage        galway-united
   9 │ 0.0628079   0.221367       NaN  2026          ha.γ_team_raw[7]      1.00053   2026               0      0  home_advantage        shamrock-rovers
  10 │ 0.0672467   0.222332       NaN  2026          ha.γ_team_raw[8]      1.0001    2026               0      0  home_advantage        shelbourne
  11 │ 0.0643436   0.204226       NaN  2026          ha.γ_team_raw[9]      0.999781  2026               0      0  home_advantage        sligo-rovers
  12 │ 0.0644451   0.219743       NaN  2026          ha.γ_team_raw[10]     1.00089   2026               0      0  home_advantage        st-patricks-athletic
  13 │ 0.0652195   0.200561       NaN  2026          ha.γ_team_raw[11]     1.00115   2026               0      0  home_advantage        waterford-fc
  14 │ 0.0503952   0.212728       NaN  2026          ha.γ_base             1.0004    2026               0      0  ha.γ_base             global
  15 │ 0.0356402   0.0472096      NaN  2026          ha.σ_γ                1.00182   2026               0      0  ha.σ_γ                global
  16 │ 0.14607     1.26956        NaN  2026          kap.κ_team_raw[1]     1.00029   2026               0      0  kappa                 bohemian
  17 │ 0.140938    1.15321        NaN  2026          kap.κ_team_raw[2]     1.00039   2026               0      0  kappa                 cork-city
  18 │ 0.140111    1.25689        NaN  2026          kap.κ_team_raw[3]     1.00211   2026               0      0  kappa                 derry-city
  19 │ 0.132613    1.20402        NaN  2026          kap.κ_team_raw[4]     0.999692  2026               0      0  kappa                 drogheda-united
  20 │ 0.143192    1.23591        NaN  2026          kap.κ_team_raw[5]     0.999934  2026               0      0  kappa                 dundalk-fc
  21 │ 0.136046    1.24235        NaN  2026          kap.κ_team_raw[6]     1.00226   2026               0      0  kappa                 galway-united
  22 │ 0.140223    1.25993        NaN  2026          kap.κ_team_raw[7]     0.999401  2026               0      0  kappa                 shamrock-rovers
  23 │ 0.144171    1.27006        NaN  2026          kap.κ_team_raw[8]     0.999714  2026               0      0  kappa                 shelbourne
  24 │ 0.136336    1.22529        NaN  2026          kap.κ_team_raw[9]     1.0017    2026               0      0  kappa                 sligo-rovers
  25 │ 0.142738    1.25904        NaN  2026          kap.κ_team_raw[10]    0.999848  2026               0      0  kappa                 st-patricks-athletic
  26 │ 0.133321    1.21917        NaN  2026          kap.κ_team_raw[11]    0.99992   2026               0      0  kappa                 waterford-fc
  27 │ 0.176253    0.88883        NaN  2026          kap.κ_base            1.00168   2026               0      0  kap.κ_base            global
  28 │ 0.0557246   0.0865407      NaN  2026          kap.σ_κ               1.00247   2026               0      0  kap.σ_κ               global
  29 │ 0.0920398  -0.121129       NaN  2026          p_dyn.w_G_att         0.999675  2026               0      0  p_dyn.w_G_att         global
  30 │ 0.0888112   0.119722       NaN  2026          p_dyn.w_G_def         1.00222   2026               0      0  p_dyn.w_G_def         global
  31 │ 0.0209619   0.0850194      NaN  2026          p_dyn.w_Outfield_att  1.00024   2026               0      0  p_dyn.w_Outfield_att  global
  32 │ 0.0185101  -0.125792       NaN  2026          p_dyn.w_Outfield_def  1.00103   2026               0      0  p_dyn.w_Outfield_def  global
=#







# ==========================================
# Poisson Model
# ==========================================

model_poisson = PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    market_weight          = market_weight
)



task_poisson = Experiments.create_experiment_task(
    ds, 
    model_poisson, 
    "poisson_r1", 
    save_dir; 
    target_seasons=["2026"],
    history_seasons = 2,
    warmup_period =  21,
    dynamics_col=:match_week,
    samples=1000,      # Small samples for fast runner testing
    warmup=500,        # Small warmup for fast runner testing
    chains=4,         # 2 chains for fast runner testing
    use_queue=true
)

results_poisson = Experiments.run_experiment(task_poisson)
Experiments.save_experiment(results_poisson)

chains_df_all_poisson = Experiments.Diagnostics.extract_chains(ds, results_poisson)
println("\n--- Convergence Diagnostics (R-hat & ESS) ---")
conv_diag_all_poisson = Experiments.Diagnostics.check_convergence(chains_df_all_poisson)
#=
ChainDiagnostic (Convergence)
  ✅ All parameters converged (R-hat ≤ 1.05)
=#

conv_diag_all_poisson.df


#=
julia> conv_diag_all_poisson.df
32×11 DataFrame
 Row │ std        mean        ess      train_season  raw_symbol            rhat      target_season  fold   week   parameter             entity               
     │ Float64    Float64     Float64  String        Symbol                Float64   String         Int64  Int64  String                String               
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ 0.331936    3.24536        NaN  2026          ν_xg                  1.00325   2026               0      0  ν_xg                  global
   2 │ 0.0256445   0.220281       NaN  2026          σ_market              1.00025   2026               0      0  σ_market              global
   3 │ 0.0705031   0.226855       NaN  2026          ha.γ_team_raw[1]      1.00008   2026               0      0  home_advantage        bohemian
   4 │ 0.079017    0.176824       NaN  2026          ha.γ_team_raw[2]      1.00031   2026               0      0  home_advantage        cork-city
   5 │ 0.0709066   0.22438        NaN  2026          ha.γ_team_raw[3]      1.00106   2026               0      0  home_advantage        derry-city
   6 │ 0.0724393   0.200224       NaN  2026          ha.γ_team_raw[4]      1.00075   2026               0      0  home_advantage        drogheda-united
   7 │ 0.0801798   0.212846       NaN  2026          ha.γ_team_raw[5]      1.00076   2026               0      0  home_advantage        dundalk-fc
   8 │ 0.069635    0.229528       NaN  2026          ha.γ_team_raw[6]      1.00136   2026               0      0  home_advantage        galway-united
   9 │ 0.0694918   0.219667       NaN  2026          ha.γ_team_raw[7]      1.00142   2026               0      0  home_advantage        shamrock-rovers
  10 │ 0.069916    0.223184       NaN  2026          ha.γ_team_raw[8]      1.00276   2026               0      0  home_advantage        shelbourne
  11 │ 0.0687864   0.208365       NaN  2026          ha.γ_team_raw[9]      0.999923  2026               0      0  home_advantage        sligo-rovers
  12 │ 0.0716619   0.220391       NaN  2026          ha.γ_team_raw[10]     1.00013   2026               0      0  home_advantage        st-patricks-athletic
  13 │ 0.0686847   0.202972       NaN  2026          ha.γ_team_raw[11]     1.00322   2026               0      0  home_advantage        waterford-fc
  14 │ 0.0559593   0.213413       NaN  2026          ha.γ_base             1.00211   2026               0      0  ha.γ_base             global
  15 │ 0.0373026   0.0480113      NaN  2026          ha.σ_γ                1.00119   2026               0      0  ha.σ_γ                global
  16 │ 0.0745732   1.10042        NaN  2026          kap.κ_team_raw[1]     1.00196   2026               0      0  kappa                 bohemian
  17 │ 0.0774143   1.05495        NaN  2026          kap.κ_team_raw[2]     1.00179   2026               0      0  kappa                 cork-city
  18 │ 0.0729027   1.10032        NaN  2026          kap.κ_team_raw[3]     1.00223   2026               0      0  kappa                 derry-city
  19 │ 0.0713276   1.07221        NaN  2026          kap.κ_team_raw[4]     0.999748  2026               0      0  kappa                 drogheda-united
  20 │ 0.08132     1.08821        NaN  2026          kap.κ_team_raw[5]     1.0006    2026               0      0  kappa                 dundalk-fc
  21 │ 0.0712194   1.08276        NaN  2026          kap.κ_team_raw[6]     1.00109   2026               0      0  kappa                 galway-united
  22 │ 0.0738179   1.10412        NaN  2026          kap.κ_team_raw[7]     0.999898  2026               0      0  kappa                 shamrock-rovers
  23 │ 0.0740034   1.10056        NaN  2026          kap.κ_team_raw[8]     1.00139   2026               0      0  kappa                 shelbourne
  24 │ 0.0707199   1.07696        NaN  2026          kap.κ_team_raw[9]     1.00359   2026               0      0  kappa                 sligo-rovers
  25 │ 0.0728201   1.09312        NaN  2026          kap.κ_team_raw[10]    1.0033    2026               0      0  kappa                 st-patricks-athletic
  26 │ 0.0725627   1.07693        NaN  2026          kap.κ_team_raw[11]    0.999814  2026               0      0  kappa                 waterford-fc
  27 │ 0.0912697   0.677711       NaN  2026          kap.κ_base            1.00022   2026               0      0  kap.κ_base            global
  28 │ 0.0477181   0.0642444      NaN  2026          kap.σ_κ               1.00084   2026               0      0  kap.σ_κ               global
  29 │ 0.0936577  -0.119171       NaN  2026          p_dyn.w_G_att         1.00026   2026               0      0  p_dyn.w_G_att         global
  30 │ 0.0973926   0.104629       NaN  2026          p_dyn.w_G_def         1.0002    2026               0      0  p_dyn.w_G_def         global
  31 │ 0.0215081   0.0925663      NaN  2026          p_dyn.w_Outfield_att  1.00009   2026               0      0  p_dyn.w_Outfield_att  global
  32 │ 0.0193152  -0.120725       NaN  2026          p_dyn.w_Outfield_def  1.00172   2026               0      0  p_dyn.w_Outfield_def  global
=#

