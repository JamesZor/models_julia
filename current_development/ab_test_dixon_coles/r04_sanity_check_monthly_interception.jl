# current_development/ab_test_dixon_coles/r04_sanity_check_monthly_interception.jl

using Revise
using BayesianFootball
using DataFrames
using Turing

using ThreadPinning
pinthreads(:cores)

const PreGame = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions
const Data = BayesianFootball.Data

# ==========================================
# 1. SETUP & DATA
# ==========================================
println("[INFO] Loading Ireland DataStore...")
ds = Data.load_datastore_cached(Data.Ireland())

# ==========================================
# 2. SHARED COMPONENT CONFIGURATION
# ==========================================
disp_cfg  = PreGame.HomeAwayDispersion()
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

interception_global_config = PreGame.GlobalInterception()
interception_hierarchical_config = PreGame.HierarchicalMonthlyInterception()

# ==========================================
# 3. MODEL 1: GLOBAL INTERCEPTION
# ==========================================
println("[INFO] Initializing DynamicDixonColesXGOutfieldPlayerTimeDecayModel (GLOBAL INTERCEPTION)...")
# model_global = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
#     interception_config    = PreGame.GlobalInterception(),
#     player_dynamics_config = dyn_cfg,
#     dispersion_config      = disp_cfg,
#     homeadvantage_config   = ha_cfg,
#     kappa_config           = kap_cfg,
#     dixon_coles_config     = PreGame.HierarchicalTeamDixonColesConfig(),
#     player_ratings_feature = feature_cfg_bayes,
#     market_feature_config  = Features.DixonColesMarketFeature(),
#     market_weight          = 0.4
# )
#
model_global = PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel(
    interception_config    = interception_global_config,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight          = 0.4
)


# ==========================================
# 4. RUN GLOBAL INTERCEPTION EXPERIMENT
# ==========================================
println("[INFO] Creating Task 1 (Global Interception, Season 2026)...")
task_global = Experiments.create_experiment_task(
    ds, 
    model_global, 
    "sanity_dc_global_inter", 
    "./tmp_mcmc_checkpoints/"; 
    target_seasons=["2026"], 
    dynamics_col=:match_month,
    warmup_period = 5,
    samples=1000,
    warmup=500,  
    chains=16,
    use_queue=true,
)

println("[INFO] Running Global Interception Experiment...")
results_global = Experiments.run_experiment(task_global)

println("[INFO] Global Interception Completed. Summarizing Split 1, Chain 1...")
chains_global = results_global.training_results[1][1]
display(describe(chains_global))

#=
julia> display(describe(chains_global))
Chains MCMC chain (1000×47×16 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 16
Samples per chain = 1000
Wall duration     = 2379.74 seconds
Compute duration  = 36536.22 seconds
parameters        = ν_xg, σ_market, inter.μ, ha.γ_base, ha.σ_γ, ha.γ_team_raw[1], ha.γ_team_raw[2], ha.γ_team_raw[3], ha.γ_team_raw[4], ha.γ_team_raw[5], ha.γ_team_raw[6], ha.γ_team_raw[7], ha.γ_team_raw[8], ha.γ_team_raw[9], ha.γ_team_raw[10], ha.γ_team_raw[11], kap.κ_base, kap.σ_κ, kap.κ_team_raw[1], kap.κ_team_raw[2], kap.κ_team_raw[3], kap.κ_team_raw[4], kap.κ_team_raw[5], kap.κ_team_raw[6], kap.κ_team_raw[7], kap.κ_team_raw[8], kap.κ_team_raw[9], kap.κ_team_raw[10], kap.κ_team_raw[11], p_dyn.w_G_att, p_dyn.w_G_def, p_dyn.w_Outfield_att, p_dyn.w_Outfield_def
internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, lp, logprior, loglikelihood

Summary Statistics

            parameters      mean       std      mcse     ess_bulk     ess_tail      rhat   ess_per_sec 
                Symbol   Float64   Float64   Float64      Float64      Float64   Float64       Float64 

                  ν_xg    3.2001    0.3357    0.0022   22152.6786   11041.5021    0.9999        0.6063
              σ_market    0.2170    0.0253    0.0002   20008.4719   11717.3034    1.0004        0.5476
               inter.μ    0.2050    0.0838    0.0006   17207.0388   11338.0949    1.0019        0.4710
             ha.γ_base    0.2241    0.0534    0.0004   19900.2145   12108.4779    1.0008        0.5447
                ha.σ_γ    0.0434    0.0339    0.0003   10155.2021    7282.8407    1.0006        0.2779
      ha.γ_team_raw[1]    0.2369    0.9558    0.0066   20967.8323   11856.2432    1.0007        0.5739
      ha.γ_team_raw[2]   -0.3940    0.9700    0.0067   20944.1848   12713.5052    1.0007        0.5732
      ha.γ_team_raw[3]    0.1370    0.9436    0.0067   20064.3020   11833.1612    1.0008        0.5492
      ha.γ_team_raw[4]   -0.1187    0.9409    0.0065   20785.8435   12139.7234    1.0011        0.5689
      ha.γ_team_raw[5]   -0.0134    1.0151    0.0069   21764.3293   11719.3923    1.0009        0.5957
      ha.γ_team_raw[6]    0.1963    0.9475    0.0064   21668.2636   11205.5173    1.0008        0.5931
      ha.γ_team_raw[7]    0.0575    0.9487    0.0064   22045.3352   12410.5944    1.0001        0.6034
      ha.γ_team_raw[8]    0.1520    0.9510    0.0066   21080.5247   11502.5490    1.0010        0.5770
      ha.γ_team_raw[9]   -0.1586    0.9546    0.0067   20613.2838   12125.0182    1.0006        0.5642
     ha.γ_team_raw[10]    0.0343    0.9522    0.0063   22811.3333   11852.6485    1.0016        0.6243
     ha.γ_team_raw[11]   -0.1184    0.9229    0.0060   23955.7385   11885.1245    1.0011        0.6557
            kap.κ_base    0.5005    0.0913    0.0007   18011.6138   10688.2355    1.0014        0.4930
               kap.σ_κ    0.0622    0.0460    0.0005    7938.2028    7570.7873    1.0006        0.2173
     kap.κ_team_raw[1]    0.1341    0.9359    0.0062   22677.5609   11803.8884    1.0013        0.6207
     kap.κ_team_raw[2]   -0.6224    1.0133    0.0078   16935.3944   11246.3275    1.0005        0.4635
     kap.κ_team_raw[3]    0.1894    0.9284    0.0063   21507.2279   12122.4227    1.0013        0.5887
     kap.κ_team_raw[4]   -0.3198    0.9512    0.0066   21055.4532   12789.6148    1.0004        0.5763
     kap.κ_team_raw[5]   -0.0167    1.0114    0.0070   20982.1727   11596.0452    1.0001        0.5743
     kap.κ_team_raw[6]   -0.1142    0.9486    0.0063   23020.7264   12443.1685    1.0013        0.6301
     kap.κ_team_raw[7]    0.2102    0.9420    0.0065   20987.7667   11409.7875    1.0017        0.5744
     kap.κ_team_raw[8]    0.1962    0.9315    0.0064   20905.8712   12109.9943    1.0017        0.5722
     kap.κ_team_raw[9]   -0.2290    0.9217    0.0065   20263.4289   12354.4025    1.0017        0.5546
    kap.κ_team_raw[10]    0.0213    0.9359    0.0064   21112.6805   11990.0907    1.0003        0.5779
    kap.κ_team_raw[11]   -0.2233    0.9283    0.0064   21217.9589   12507.6158    1.0004        0.5807
         p_dyn.w_G_att   -0.0986    0.0974    0.0007   19686.0564   11806.5099    1.0004        0.5388
         p_dyn.w_G_def    0.1228    0.0941    0.0007   18874.7892   11839.1524    1.0008        0.5166
  p_dyn.w_Outfield_att    0.0979    0.0202    0.0002   15447.7637   11775.0111    1.0003        0.4228
  p_dyn.w_Outfield_def   -0.1160    0.0189    0.0001   18818.8455   11726.9020    1.0014        0.5151


Quantiles

            parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
                Symbol   Float64   Float64   Float64   Float64   Float64 

                  ν_xg    2.5594    2.9709    3.1905    3.4220    3.8826
              σ_market    0.1740    0.1994    0.2150    0.2321    0.2735
               inter.μ    0.0399    0.1490    0.2044    0.2616    0.3689
             ha.γ_base    0.1194    0.1883    0.2236    0.2600    0.3280
                ha.σ_γ    0.0017    0.0169    0.0361    0.0623    0.1265
      ha.γ_team_raw[1]   -1.6576   -0.4056    0.2449    0.8764    2.0865
      ha.γ_team_raw[2]   -2.2411   -1.0606   -0.3988    0.2478    1.5500
      ha.γ_team_raw[3]   -1.7382   -0.4912    0.1438    0.7726    1.9813
      ha.γ_team_raw[4]   -1.9465   -0.7545   -0.1166    0.5064    1.7192
      ha.γ_team_raw[5]   -2.0081   -0.6871   -0.0111    0.6574    2.0049
      ha.γ_team_raw[6]   -1.6958   -0.4366    0.2077    0.8239    2.0234
      ha.γ_team_raw[7]   -1.8157   -0.5782    0.0527    0.7019    1.8834
      ha.γ_team_raw[8]   -1.6944   -0.4940    0.1609    0.7971    1.9998
      ha.γ_team_raw[9]   -2.0131   -0.8069   -0.1606    0.4781    1.7239
     ha.γ_team_raw[10]   -1.8082   -0.6159    0.0318    0.6793    1.9130
     ha.γ_team_raw[11]   -1.9130   -0.7442   -0.1230    0.4953    1.7209
            kap.κ_base    0.3237    0.4382    0.4995    0.5619    0.6823
               kap.σ_κ    0.0028    0.0256    0.0532    0.0892    0.1713
     kap.κ_team_raw[1]   -1.7306   -0.4850    0.1378    0.7606    1.9651
     kap.κ_team_raw[2]   -2.5418   -1.3258   -0.6475    0.0624    1.4093
     kap.κ_team_raw[3]   -1.6767   -0.4173    0.1975    0.8069    2.0155
     kap.κ_team_raw[4]   -2.1453   -0.9714   -0.3326    0.3241    1.5925
     kap.κ_team_raw[5]   -2.0001   -0.6957   -0.0193    0.6612    1.9713
     kap.κ_team_raw[6]   -1.9499   -0.7562   -0.1213    0.5108    1.7693
     kap.κ_team_raw[7]   -1.6489   -0.4147    0.2213    0.8462    2.0195
     kap.κ_team_raw[8]   -1.6481   -0.4162    0.1967    0.8247    2.0380
     kap.κ_team_raw[9]   -1.9767   -0.8607   -0.2433    0.3731    1.6155
    kap.κ_team_raw[10]   -1.8075   -0.6031    0.0200    0.6412    1.8656
    kap.κ_team_raw[11]   -1.9896   -0.8576   -0.2402    0.4050    1.6278
         p_dyn.w_G_att   -0.2904   -0.1638   -0.0981   -0.0332    0.0939
         p_dyn.w_G_def   -0.0646    0.0599    0.1244    0.1865    0.3052
  p_dyn.w_Outfield_att    0.0579    0.0844    0.0984    0.1113    0.1374
  p_dyn.w_Outfield_def   -0.1529   -0.1288   -0.1164   -0.1035   -0.0786
=#



# ==========================================
# 5. MODEL 2: HIERARCHICAL MONTHLY INTERCEPTION
# ==========================================
println("\n[INFO] Initializing DynamicDixonColesXGOutfieldPlayerTimeDecayModel (MONTHLY INTERCEPTION)...")
# model_monthly = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
#     interception_config    = PreGame.HierarchicalMonthlyInterception(),
#     player_dynamics_config = dyn_cfg,
#     dispersion_config      = disp_cfg,
#     homeadvantage_config   = ha_cfg,
#     kappa_config           = kap_cfg,
#     dixon_coles_config     = PreGame.HierarchicalTeamDixonColesConfig(),
#     player_ratings_feature = feature_cfg_bayes,
#     market_feature_config  = Features.DixonColesMarketFeature(),
#     market_weight          = 0.4
# )
#
model_monthly = PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel(
    interception_config    = interception_hierarchical_config,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    player_ratings_feature = feature_cfg_bayes,
    market_weight          = 0.4
)

# ==========================================
# 6. RUN HIERARCHICAL MONTHLY INTERCEPTION EXPERIMENT
# ==========================================
println("[INFO] Creating Task 2 (Monthly Interception, Season 2026)...")
task_monthly = Experiments.create_experiment_task(
    ds, 
    model_monthly, 
    "sanity_dc_monthly_inter", 
    "./tmp_mcmc_checkpoints/"; 
    target_seasons=["2026"], 
    dynamics_col=:match_month,
    warmup_period = 5,
    samples=1000,
    warmup=500,  
    chains=16,
    use_queue=true,
)

println("[INFO] Running Monthly Interception Experiment...")
results_monthly = Experiments.run_experiment(task_monthly)

println("[INFO] Monthly Interception Completed. Summarizing Split 1, Chain 1...")
chains_monthly = results_monthly.training_results[1][1]
display(describe(chains_monthly))

println("\n[INFO] Both Sanity Checks Complete! Check the trace of `inter.δ_month` to ensure convergence.")
