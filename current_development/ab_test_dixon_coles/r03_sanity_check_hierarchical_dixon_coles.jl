# current_development/ab_test_dixon_coles/r03_sanity_check_hierarchical_dixon_coles.jl

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
inter_cfg = PreGame.GlobalInterception()
disp_cfg  = PreGame.HomeAwayDispersion() 
ha_cfg    = PreGame.HierarchicalTeamHomeAdvantage()
kap_cfg   = PreGame.HierarchicalTeamKappa()
dyn_cfg   = PreGame.OutfieldPlayerDynamicsConfig(days_half_life=60.0)

tracker_bayes = Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)
feature_cfg_bayes = Features.PlayerRatingsFeature(tracker_bayes)

# ==========================================
# 3. MODEL 1: GLOBAL RHO (MARKET)
# ==========================================
println("[INFO] Initializing DynamicDixonColesXGOutfieldPlayerTimeDecayModel (GLOBAL RHO)...")
model_global = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
    interception_config    = inter_cfg,
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    dixon_coles_config     = PreGame.GlobalDixonColesConfig(),
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DixonColesMarketFeature(),
    market_weight          = 0.4
)

# ==========================================
# 4. RUN GLOBAL RHO EXPERIMENT
# ==========================================
println("[INFO] Creating Task 1 (Global Rho, Season 2026)...")
task_global = Experiments.create_experiment_task(
    ds, 
    model_global, 
    "sanity_dc_global_rho", 
    "./tmp_mcmc_checkpoints/"; 
    target_seasons=["2026"], 
    dynamics_col=:match_month,
    warmup_period = 5,
    samples=1000,
    warmup=500,  
    chains=8,
    use_queue=true,
)

println("[INFO] Running Global Rho Experiment...")
results_global = Experiments.run_experiment(task_global)

println("[INFO] Global Rho Completed. Summarizing Split 1, Chain 1...")
chains_global = results_global.training_results[1][1]
display(describe(chains_global))




#=
julia> display(describe(chains_global))
Chains MCMC chain (1000×48×8 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 8
Samples per chain = 1000
Wall duration     = 468.39 seconds
Compute duration  = 3521.13 seconds
parameters        = ν_xg, σ_market, inter.μ, ha.γ_base, ha.σ_γ, ha.γ_team_raw[1], ha.γ_team_raw[2], ha.γ_team_raw[3], ha.γ_team_raw[4], ha.γ_team_raw[5], ha.γ_team_raw[6], ha.γ_team_raw[7], ha.γ_team_raw[8], ha.γ_team_raw[9], ha.γ_team_raw[10], ha.γ_team_raw[11], kap.κ_base, kap.σ_κ, kap.κ_team_raw[1], kap.κ_team_raw[2], kap.κ_team_raw[3], kap.κ_team_raw[4], kap.κ_team_raw[5], kap.κ_team_raw[6], kap.κ_team_raw[7], kap.κ_team_raw[8], kap.κ_team_raw[9], kap.κ_team_raw[10], kap.κ_team_raw[11], p_dyn.w_G_att, p_dyn.w_G_def, p_dyn.w_Outfield_att, p_dyn.w_Outfield_def, dc.ρ_base
internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, lp, logprior, loglikelihood

Summary Statistics

            parameters      mean       std      mcse     ess_bulk    ess_tail      rhat   ess_per_sec 
                Symbol   Float64   Float64   Float64      Float64     Float64   Float64       Float64 

                  ν_xg    3.1882    0.3297    0.0031   10844.9217   5808.5851    1.0023        3.0800
              σ_market    0.1803    0.0168    0.0002    8988.3752   6032.9643    1.0003        2.5527
               inter.μ    0.1805    0.0788    0.0009    8404.0510   5808.4591    1.0015        2.3867
             ha.γ_base    0.2219    0.0483    0.0005    8939.1947   5977.6217    1.0009        2.5387
                ha.σ_γ    0.0463    0.0344    0.0005    4191.2363   3310.5839    1.0002        1.1903
      ha.γ_team_raw[1]    0.3117    0.9194    0.0093    9703.0902   6155.2171    1.0012        2.7557
      ha.γ_team_raw[2]   -0.5971    0.9630    0.0104    8533.5536   6385.4975    1.0002        2.4235
      ha.γ_team_raw[3]    0.2034    0.9130    0.0091   10181.2646   5845.3662    0.9999        2.8915
      ha.γ_team_raw[4]   -0.1825    0.9218    0.0087   11217.5341   6106.8416    0.9996        3.1858
      ha.γ_team_raw[5]   -0.0243    0.9857    0.0096   10581.4822   5598.9118    1.0006        3.0051
      ha.γ_team_raw[6]    0.2393    0.9280    0.0092   10176.1969   5657.8914    0.9999        2.8900
      ha.γ_team_raw[7]    0.1419    0.8872    0.0091    9400.7454   5930.7248    1.0011        2.6698
      ha.γ_team_raw[8]    0.2181    0.9255    0.0089   10718.9434   6145.3265    1.0008        3.0442
      ha.γ_team_raw[9]   -0.2296    0.9375    0.0090   10945.0280   6405.3369    1.0023        3.1084
     ha.γ_team_raw[10]    0.0839    0.9125    0.0089   10441.6268   5819.9598    1.0019        2.9654
     ha.γ_team_raw[11]   -0.1699    0.9241    0.0090   10472.2669   6218.6933    1.0011        2.9741
            kap.κ_base    0.6033    0.1190    0.0013    8692.1564   4827.7900    1.0009        2.4686
               kap.σ_κ    0.0739    0.0560    0.0006    5991.1172   3530.3113    1.0009        1.7015
     kap.κ_team_raw[1]    0.0233    0.9778    0.0098   10066.0879   5671.7585    1.0011        2.8588
     kap.κ_team_raw[2]   -0.1728    0.9934    0.0101    9750.4781   5914.1580    1.0013        2.7691
     kap.κ_team_raw[3]    0.0918    0.9848    0.0091   11622.4841   5727.4267    1.0009        3.3008
     kap.κ_team_raw[4]   -0.1100    0.9784    0.0092   11195.2339   5955.2014    1.0014        3.1794
     kap.κ_team_raw[5]   -0.0054    1.0027    0.0098   10460.3063   5728.2447    1.0019        2.9707
     kap.κ_team_raw[6]   -0.2074    0.9923    0.0093   11451.6366   5665.8496    1.0007        3.2523
     kap.κ_team_raw[7]   -0.0655    0.9853    0.0099    9874.1668   5715.0240    1.0012        2.8043
     kap.κ_team_raw[8]    0.0747    0.9741    0.0092   11295.9039   6051.7251    1.0000        3.2080
     kap.κ_team_raw[9]   -0.0722    1.0084    0.0093   11635.9400   5844.5985    1.0013        3.3046
    kap.κ_team_raw[10]   -0.1992    0.9871    0.0104    9100.3347   5491.0912    1.0008        2.5845
    kap.κ_team_raw[11]   -0.0169    1.0059    0.0101    9936.2998   5699.6756    1.0025        2.8219
         p_dyn.w_G_att   -0.1129    0.0850    0.0008   11073.9982   6300.0515    1.0005        3.1450
         p_dyn.w_G_def    0.1379    0.0843    0.0008   10132.8708   6299.0106    1.0022        2.8777
  p_dyn.w_Outfield_att    0.1017    0.0169    0.0002    9355.3963   5985.6915    1.0004        2.6569
  p_dyn.w_Outfield_def   -0.1212    0.0162    0.0002    9019.5323   5854.7157    1.0006        2.5615
             dc.ρ_base   -0.1708    0.1328    0.0012   11763.1642   5914.9034    1.0011        3.3407


Quantiles

            parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
                Symbol   Float64   Float64   Float64   Float64   Float64 

                  ν_xg    2.5634    2.9573    3.1829    3.4108    3.8644
              σ_market    0.1507    0.1685    0.1790    0.1908    0.2161
               inter.μ    0.0277    0.1276    0.1806    0.2334    0.3336
             ha.γ_base    0.1285    0.1898    0.2214    0.2543    0.3170
                ha.σ_γ    0.0019    0.0190    0.0398    0.0669    0.1285
      ha.γ_team_raw[1]   -1.5687   -0.2917    0.3266    0.9234    2.0991
      ha.γ_team_raw[2]   -2.4316   -1.2518   -0.6105    0.0257    1.3970
      ha.γ_team_raw[3]   -1.6244   -0.3930    0.2275    0.8086    1.9764
      ha.γ_team_raw[4]   -1.9782   -0.8101   -0.1933    0.4302    1.6608
      ha.γ_team_raw[5]   -1.9404   -0.6827   -0.0185    0.6420    1.9016
      ha.γ_team_raw[6]   -1.6474   -0.3777    0.2537    0.8700    2.0180
      ha.γ_team_raw[7]   -1.6168   -0.4451    0.1378    0.7218    1.9164
      ha.γ_team_raw[8]   -1.6541   -0.3968    0.2142    0.8391    2.0029
      ha.γ_team_raw[9]   -2.0543   -0.8597   -0.2352    0.3923    1.6166
     ha.γ_team_raw[10]   -1.7357   -0.5178    0.0934    0.6903    1.8265
     ha.γ_team_raw[11]   -1.9844   -0.7765   -0.1805    0.4425    1.6665
            kap.κ_base    0.3726    0.5223    0.6037    0.6834    0.8415
               kap.σ_κ    0.0031    0.0290    0.0625    0.1066    0.2105
     kap.κ_team_raw[1]   -1.8559   -0.6212    0.0093    0.6785    1.9216
     kap.κ_team_raw[2]   -2.1381   -0.8462   -0.1610    0.4923    1.7832
     kap.κ_team_raw[3]   -1.8436   -0.5686    0.0916    0.7471    2.0709
     kap.κ_team_raw[4]   -2.0076   -0.7787   -0.1133    0.5577    1.8004
     kap.κ_team_raw[5]   -1.9772   -0.6781   -0.0015    0.6794    1.9223
     kap.κ_team_raw[6]   -2.1346   -0.8827   -0.2147    0.4555    1.7382
     kap.κ_team_raw[7]   -1.9717   -0.7420   -0.0771    0.6057    1.8955
     kap.κ_team_raw[8]   -1.8368   -0.5850    0.0727    0.7303    1.9522
     kap.κ_team_raw[9]   -2.0234   -0.7559   -0.0708    0.6105    1.9048
    kap.κ_team_raw[10]   -2.1550   -0.8537   -0.1966    0.4598    1.7413
    kap.κ_team_raw[11]   -1.9373   -0.7084   -0.0189    0.6650    1.9568
         p_dyn.w_G_att   -0.2802   -0.1694   -0.1130   -0.0583    0.0569
         p_dyn.w_G_def   -0.0278    0.0805    0.1393    0.1950    0.3019
  p_dyn.w_Outfield_att    0.0692    0.0903    0.1015    0.1131    0.1351
  p_dyn.w_Outfield_def   -0.1539   -0.1319   -0.1213   -0.1105   -0.0891
             dc.ρ_base   -0.4357   -0.2574   -0.1693   -0.0805    0.0849

nothing
=#


mp_global = Predictions.model_inference(ds, results_global)

# ==========================================
# 5. MODEL 2: HIERARCHICAL RHO (MARKET)
# ==========================================
println("\n[INFO] Initializing DynamicDixonColesXGOutfieldPlayerTimeDecayModel (HIERARCHICAL RHO)...")
model_hierarchical = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
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
# 6. RUN HIERARCHICAL RHO EXPERIMENT
# ==========================================
println("[INFO] Creating Task 2 (Hierarchical Rho, Season 2026)...")
task_hierarchical = Experiments.create_experiment_task(
    ds, 
    model_hierarchical, 
    "sanity_dc_hierarchical_rho", 
    "./tmp_mcmc_checkpoints/"; 
    target_seasons=["2026"], 
    dynamics_col=:match_month,
    warmup_period = 5,
    samples=1000,
    warmup=500,  
    chains=8,
    use_queue=true,
)

println("[INFO] Running Hierarchical Rho Experiment...")
results_hierarchical = Experiments.run_experiment(task_hierarchical)

println("[INFO] Hierarchical Rho Completed. Summarizing Split 1, Chain 1...")
chains_hierarchical = results_hierarchical.training_results[1][1]
display(describe(chains_hierarchical))



#=
julia> display(describe(chains_hierarchical))
Chains MCMC chain (1000×60×8 Array{Float64, 3}):

Iterations        = 501:1:1500
Number of chains  = 8
Samples per chain = 1000
Wall duration     = 700.7 seconds
Compute duration  = 5173.65 seconds
parameters        = ν_xg, σ_market, inter.μ, ha.γ_base, ha.σ_γ, ha.γ_team_raw[1], ha.γ_team_raw[2], ha.γ_team_raw[3], ha.γ_team_raw[4], ha.γ_team_raw[5], ha.γ_team_raw[6], ha.γ_team_raw[7], ha.γ_team_raw[8], ha.γ_team_raw[9], ha.γ_team_raw[10], ha.γ_team_raw[11], kap.κ_base, kap.σ_κ, kap.κ_team_raw[1], kap.κ_team_raw[2], kap.κ_team_raw[3], kap.κ_team_raw[4], kap.κ_team_raw[5], kap.κ_team_raw[6], kap.κ_team_raw[7], kap.κ_team_raw[8], kap.κ_team_raw[9], kap.κ_team_raw[10], kap.κ_team_raw[11], p_dyn.w_G_att, p_dyn.w_G_def, p_dyn.w_Outfield_att, p_dyn.w_Outfield_def, dc.ρ_base, dc.σ_ρ, dc.raw_ρ[1], dc.raw_ρ[2], dc.raw_ρ[3], dc.raw_ρ[4], dc.raw_ρ[5], dc.raw_ρ[6], dc.raw_ρ[7], dc.raw_ρ[8], dc.raw_ρ[9], dc.raw_ρ[10], dc.raw_ρ[11]
internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, lp, logprior, loglikelihood

Summary Statistics

            parameters      mean       std      mcse     ess_bulk    ess_tail      rhat   ess_per_sec 
                Symbol   Float64   Float64   Float64      Float64     Float64   Float64       Float64 

                  ν_xg    3.1901    0.3299    0.0032   10288.8044   5528.1362    1.0009        1.9887
              σ_market    0.1830    0.0173    0.0002    9543.9825   5991.6580    1.0008        1.8447
               inter.μ    0.1812    0.0810    0.0008    9425.0736   5780.1459    1.0007        1.8217
             ha.γ_base    0.2224    0.0478    0.0005    9084.2730   6045.3881    1.0008        1.7559
                ha.σ_γ    0.0454    0.0344    0.0005    4241.3388   3720.3295    1.0013        0.8198
      ha.γ_team_raw[1]    0.2861    0.9513    0.0099    9319.3224   5625.3349    1.0002        1.8013
      ha.γ_team_raw[2]   -0.5455    0.9758    0.0106    8587.7604   6122.2418    1.0003        1.6599
      ha.γ_team_raw[3]    0.1924    0.9230    0.0085   11761.7006   6161.3713    1.0010        2.2734
      ha.γ_team_raw[4]   -0.1719    0.9314    0.0096    9321.1888   5958.9444    1.0005        1.8017
      ha.γ_team_raw[5]   -0.0179    0.9783    0.0090   11924.4849   6213.5422    1.0020        2.3048
      ha.γ_team_raw[6]    0.2269    0.9350    0.0092   10488.8956   5709.9675    1.0009        2.0274
      ha.γ_team_raw[7]    0.1341    0.9132    0.0100    8422.7602   5825.7002    1.0001        1.6280
      ha.γ_team_raw[8]    0.2144    0.9392    0.0092   10586.4965   6484.6199    1.0000        2.0462
      ha.γ_team_raw[9]   -0.2097    0.9318    0.0091   10436.9730   5753.5730    1.0005        2.0173
     ha.γ_team_raw[10]    0.0880    0.9251    0.0092   10183.4513   5967.3483    1.0015        1.9683
     ha.γ_team_raw[11]   -0.1567    0.9277    0.0089   10836.7655   5801.3367    1.0000        2.0946
            kap.κ_base    0.6016    0.1210    0.0014    7868.8744   4237.3855    1.0018        1.5210
               kap.σ_κ    0.0736    0.0563    0.0006    5295.0360   4212.2600    1.0002        1.0235
     kap.κ_team_raw[1]    0.0347    1.0005    0.0099   10200.5235   5844.1898    1.0012        1.9716
     kap.κ_team_raw[2]   -0.1540    0.9818    0.0097   10285.8007   5422.3404    0.9998        1.9881
     kap.κ_team_raw[3]    0.0926    0.9725    0.0095   10503.6564   5694.3269    1.0009        2.0302
     kap.κ_team_raw[4]   -0.1273    0.9929    0.0101    9616.7088   6033.0215    1.0016        1.8588
     kap.κ_team_raw[5]   -0.0094    1.0085    0.0107    8822.2744   5922.4810    1.0025        1.7052
     kap.κ_team_raw[6]   -0.1958    0.9949    0.0098   10371.7610   5814.1332    0.9998        2.0047
     kap.κ_team_raw[7]   -0.0650    0.9736    0.0095   10756.2609   6132.7931    1.0023        2.0790
     kap.κ_team_raw[8]    0.0831    0.9761    0.0088   12215.9156   6297.8566    1.0017        2.3612
     kap.κ_team_raw[9]   -0.0828    0.9756    0.0089   12060.6958   6075.9341    1.0019        2.3312
    kap.κ_team_raw[10]   -0.2174    1.0183    0.0095   11502.6250   5639.0093    1.0005        2.2233
    kap.κ_team_raw[11]   -0.0190    0.9681    0.0095   10295.1187   6267.5110    1.0001        1.9899
         p_dyn.w_G_att   -0.1133    0.0863    0.0009    8592.4985   5958.5238    1.0013        1.6608
         p_dyn.w_G_def    0.1344    0.0882    0.0009    8882.8196   5697.3925    1.0007        1.7169
  p_dyn.w_Outfield_att    0.1018    0.0176    0.0002    8325.4110   5703.2236    1.0003        1.6092
  p_dyn.w_Outfield_def   -0.1211    0.0170    0.0002    9244.0147   6016.0179    1.0006        1.7867
             dc.ρ_base   -0.1814    0.1437    0.0014   10319.3458   5743.9795    1.0011        1.9946
                dc.σ_ρ    0.1265    0.0804    0.0008    7829.9687   4660.9492    1.0000        1.5134
           dc.raw_ρ[1]   -0.0369    0.9269    0.0088   11039.1225   5853.1940    1.0021        2.1337
           dc.raw_ρ[2]    0.0320    0.9378    0.0093   10179.0561   5861.0404    1.0010        1.9675
           dc.raw_ρ[3]    0.0104    0.9449    0.0096    9763.5172   5819.2580    1.0016        1.8872
           dc.raw_ρ[4]   -0.0324    0.9389    0.0089   11054.1011   6119.6076    1.0004        2.1366
           dc.raw_ρ[5]    0.0092    0.9999    0.0096   10817.7302   5927.6634    1.0015        2.0909
           dc.raw_ρ[6]    0.0295    0.9481    0.0090   11135.2625   5744.7845    1.0020        2.1523
           dc.raw_ρ[7]   -0.0002    0.9418    0.0096    9708.5512   5816.8567    1.0007        1.8765
           dc.raw_ρ[8]   -0.0418    0.9215    0.0090   10439.8334   6165.6950    0.9999        2.0179
           dc.raw_ρ[9]    0.1045    0.9218    0.0091   10228.2597   6045.9152    1.0010        1.9770
          dc.raw_ρ[10]   -0.0554    0.9259    0.0089   10729.6868   6195.8109    1.0014        2.0739
          dc.raw_ρ[11]   -0.0353    0.9298    0.0088   11186.7365   5611.1569    1.0012        2.1623


Quantiles

            parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
                Symbol   Float64   Float64   Float64   Float64   Float64 

                  ν_xg    2.5642    2.9675    3.1815    3.4038    3.8773
              σ_market    0.1528    0.1708    0.1818    0.1937    0.2197
               inter.μ    0.0235    0.1263    0.1811    0.2378    0.3388
             ha.γ_base    0.1267    0.1908    0.2223    0.2545    0.3174
                ha.σ_γ    0.0019    0.0181    0.0386    0.0652    0.1282
      ha.γ_team_raw[1]   -1.6199   -0.3412    0.2965    0.9192    2.1457
      ha.γ_team_raw[2]   -2.3939   -1.2102   -0.5692    0.0739    1.4590
      ha.γ_team_raw[3]   -1.6298   -0.4206    0.1953    0.8231    1.9918
      ha.γ_team_raw[4]   -1.9403   -0.7958   -0.1978    0.4443    1.6935
      ha.γ_team_raw[5]   -1.9237   -0.6813   -0.0075    0.6261    1.9000
      ha.γ_team_raw[6]   -1.6697   -0.3830    0.2348    0.8555    2.0316
      ha.γ_team_raw[7]   -1.6933   -0.4650    0.1490    0.7429    1.9097
      ha.γ_team_raw[8]   -1.6340   -0.4085    0.2339    0.8325    2.0427
      ha.γ_team_raw[9]   -2.0419   -0.8446   -0.2218    0.4112    1.6441
     ha.γ_team_raw[10]   -1.7680   -0.5189    0.0991    0.7077    1.8475
     ha.γ_team_raw[11]   -1.9604   -0.7827   -0.1534    0.4518    1.6700
            kap.κ_base    0.3700    0.5183    0.6018    0.6818    0.8434
               kap.σ_κ    0.0026    0.0288    0.0618    0.1066    0.2096
     kap.κ_team_raw[1]   -1.9333   -0.6313    0.0261    0.7043    1.9520
     kap.κ_team_raw[2]   -2.0429   -0.8143   -0.1555    0.5065    1.7556
     kap.κ_team_raw[3]   -1.8112   -0.5497    0.1012    0.7496    1.9792
     kap.κ_team_raw[4]   -2.0631   -0.7926   -0.1409    0.5366    1.8114
     kap.κ_team_raw[5]   -1.9729   -0.6929   -0.0100    0.6548    1.9817
     kap.κ_team_raw[6]   -2.0986   -0.8737   -0.2064    0.4837    1.7725
     kap.κ_team_raw[7]   -1.9488   -0.7194   -0.0707    0.5819    1.8510
     kap.κ_team_raw[8]   -1.7774   -0.5975    0.0933    0.7492    1.9813
     kap.κ_team_raw[9]   -2.0128   -0.7330   -0.0834    0.5562    1.8692
    kap.κ_team_raw[10]   -2.1414   -0.9199   -0.2247    0.4643    1.8128
    kap.κ_team_raw[11]   -1.8986   -0.6799   -0.0247    0.6448    1.8822
         p_dyn.w_G_att   -0.2851   -0.1698   -0.1124   -0.0564    0.0543
         p_dyn.w_G_def   -0.0380    0.0744    0.1338    0.1959    0.3069
  p_dyn.w_Outfield_att    0.0675    0.0901    0.1020    0.1140    0.1358
  p_dyn.w_Outfield_def   -0.1547   -0.1330   -0.1210   -0.1094   -0.0880
             dc.ρ_base   -0.4772   -0.2731   -0.1773   -0.0855    0.0919
                dc.σ_ρ    0.0185    0.0679    0.1100    0.1678    0.3214
           dc.raw_ρ[1]   -1.8605   -0.6659   -0.0361    0.5918    1.7888
           dc.raw_ρ[2]   -1.8424   -0.6009    0.0380    0.6501    1.8571
           dc.raw_ρ[3]   -1.8612   -0.6239    0.0092    0.6481    1.8572
           dc.raw_ρ[4]   -1.8520   -0.6661   -0.0288    0.6001    1.7890
           dc.raw_ρ[5]   -1.9481   -0.6791    0.0117    0.6848    1.9759
           dc.raw_ρ[6]   -1.8740   -0.6020    0.0322    0.6660    1.9270
           dc.raw_ρ[7]   -1.8319   -0.6300    0.0026    0.6204    1.8596
           dc.raw_ρ[8]   -1.8617   -0.6569   -0.0365    0.5615    1.7534
           dc.raw_ρ[9]   -1.6996   -0.5094    0.1073    0.7257    1.9356
          dc.raw_ρ[10]   -1.9064   -0.6752   -0.0626    0.5856    1.7273
          dc.raw_ρ[11]   -1.8819   -0.6510   -0.0277    0.5893    1.7766

nothing
=#


mp_hier = Predictions.model_inference(ds, results_hierarchical)



println("\n[INFO] Both Sanity Checks Complete!")


function compar(mp)
  mp_summary = transform(mp.df, :distribution => ByRow(mean) => :model_prob)

  # 2. Join the model predictions with the historical odds data
  comparison_df = innerjoin(
  mp_summary[!, [:match_id, :selection, :model_prob]],
  ds.odds[!, [:match_id, :is_winner, :selection, :odds_close, :prob_implied_close, :prob_fair_close]],
  on = [:match_id, :selection]
  )

  # 3. Calculate how far off our model is from the market's true fair probability
  comparison_df.prob_diff = comparison_df.model_prob .- comparison_df.prob_fair_close

  # 4. Add the model's implied fair odds for easy reading
  comparison_df.model_odds = 1.0 ./ comparison_df.model_prob

  # Sort for readability
  # Display the 1X2 market predictions as a quick sanity check
  # Display the Mean Absolute Error (MAE) across all predicted markets
  mae = mean(abs.(comparison_df.prob_diff))
  println("\nMean Absolute Error vs Market: ", round(mae, digits=4))

  return mae, comparison_df 

end 

mae_global, comp_gloabl = compar(mp_global)
mae_hier, comp_hier = compar(mp_hier)

mae_hier
mae_global

describe(comp_gloabl.prob_diff)

describe(comp_hier.prob_diff)

#=
julia> mae_hier
0.07203281164660293

julia> mae_global
0.07214629977859958

julia> describe(comp_gloabl.prob_diff)
Summary Stats:
Length:         408
Missing Count:  0
Mean:           0.044117
Std. Deviation: 0.123844
Minimum:        -0.118874
1st Quartile:   -0.023402
Median:         0.003658
3rd Quartile:   0.047064
Maximum:        0.495230
Type:           Float64

julia> describe(comp_hier.prob_diff)
Summary Stats:
Length:         408
Missing Count:  0
Mean:           0.044117
Std. Deviation: 0.123759
Minimum:        -0.118913
1st Quartile:   -0.023121
Median:         0.003841
3rd Quartile:   0.047095
Maximum:        0.494959
Type:           Float64
=#

