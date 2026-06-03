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
model_global = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
    interception_config    = PreGame.GlobalInterception(),
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    dixon_coles_config     = PreGame.HierarchicalTeamDixonColesConfig(),
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DixonColesMarketFeature(),
    market_weight          = 0.4
)

#
# model_global = PreGame.DynamicDoublePoissonXGOutfieldPlayerTimeDecayModel(
#     interception_config    = interception_global_config,
#     player_dynamics_config = dyn_cfg,
#     dispersion_config      = disp_cfg,
#     homeadvantage_config   = ha_cfg,
#     kappa_config           = kap_cfg,
#     player_ratings_feature = feature_cfg_bayes,
#     market_weight          = 0.4
# )
#

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



#=
julia> display(describe(chains_global))                                                                                                                                                       
Chains MCMC chain (1000×60×16 Array{Float64, 3}):                                                                                                                                             
                                                                                                                                                                                              
Iterations        = 501:1:1500                                                                                                                                                                
Number of chains  = 16                                                                                                                                                                        
Samples per chain = 1000                                                                                                                                                                      
Wall duration     = 1054.77 seconds                                                                                                                                                           
Compute duration  = 16554.5 seconds                                                                                                                                                           
parameters        = ν_xg, σ_market, inter.μ, ha.γ_base, ha.σ_γ, ha.γ_team_raw[1], ha.γ_team_raw[2], ha.γ_team_raw[3], ha.γ_team_raw[4], ha.γ_team_raw[5], ha.γ_team_raw[6], ha.γ_team_raw[7], ha.γ_team_raw[8], ha.γ_team_raw[9], ha.γ_team_raw[10], ha.γ_team_raw[11], kap.κ_base, kap.σ_κ, kap.κ_team_raw[1], kap.κ_team_raw[2], kap.κ_team_raw[3], kap.κ_team_raw[4], kap.κ_team_raw[5], kap.κ_team_raw[6], kap.κ_team_raw[7], kap.κ_team_raw[8], kap.κ_team_raw[9], kap.κ_team_raw[10], kap.κ_team_raw[11], p_dyn.w_G_att, p_dyn.w_G_def, p_dyn.w_Outfield_att, p_dyn.w_Outfield_def, dc.ρ_base, dc.σ_ρ, dc.raw_ρ[1], dc.raw_ρ[2], dc.raw_ρ[3], dc.ra
w_ρ[4], dc.raw_ρ[5], dc.raw_ρ[6], dc.raw_ρ[7], dc.raw_ρ[8], dc.raw_ρ[9], dc.raw_ρ[10], dc.raw_ρ[11]                                                           
internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, lp, logprior, loglikelihood

Summary Statistics                                                                             
                                                                                               
            parameters      mean       std      mcse     ess_bulk     ess_tail      rhat   ess_per_sec 
                Symbol   Float64   Float64   Float64      Float64      Float64   Float64       Float64 
                                                                                               
                  ν_xg    3.2044    0.3349    0.0018   32709.9518   11107.4975    1.0016        1.9759
              σ_market    0.1813    0.0180    0.0001   22030.9931   12158.7362    1.0012        1.3308
               inter.μ    0.1299    0.0880    0.0005   28786.1180   12029.5101    1.0012        1.7389
             ha.γ_base    0.2162    0.0486    0.0003   29114.6061   12514.8735    1.0007        1.7587
                ha.σ_γ    0.0439    0.0337    0.0003    9467.2928    8181.6781    1.0004        0.5719
      ha.γ_team_raw[1]    0.2513    0.9387    0.0056   28236.6017   12081.4138    1.0012        1.7057
      ha.γ_team_raw[2]   -0.4428    0.9830    0.0060   26697.5105   12496.2475    1.0007        1.6127
      ha.γ_team_raw[3]    0.1643    0.9446    0.0052   33070.6412   11431.7233    1.0013        1.9977
      ha.γ_team_raw[4]   -0.1410    0.9368    0.0054   30023.2602   12070.3631    1.0014        1.8136
      ha.γ_team_raw[5]   -0.0108    0.9907    0.0053   35304.6620   11933.1175    1.0020        2.1326
      ha.γ_team_raw[6]    0.2141    0.9361    0.0054   29660.3520   11967.4023    1.0012        1.7917
      ha.γ_team_raw[7]    0.0921    0.9273    0.0055   28614.8978   12084.6653    1.0011        1.7285
      ha.γ_team_raw[8]    0.1583    0.9391    0.0055   29541.0733   12295.9789    1.0020        1.7845
      ha.γ_team_raw[9]   -0.1773    0.9286    0.0052   32318.9524   12018.6189    1.0011        1.9523
     ha.γ_team_raw[10]    0.0805    0.9338    0.0052   32704.1287   11484.1449    1.0022        1.9755
     ha.γ_team_raw[11]   -0.1707    0.9252    0.0054   29833.6025   12682.7020    1.0012        1.8021
            kap.κ_base    0.8428    0.1651    0.0010   25906.5903   10676.0113    1.0015        1.5649
               kap.σ_κ    0.0869    0.0554    0.0007    5939.7075    7489.8552    1.0022        0.3588
     kap.κ_team_raw[1]    0.5070    0.9001    0.0060   23307.9571   11413.3995    1.0006        1.4080
     kap.κ_team_raw[2]   -1.1321    1.0028    0.0092   12488.1678   11131.4346    1.0007        0.7544
     kap.κ_team_raw[3]    0.2556    0.8552    0.0049   31199.7369   12629.4484    1.0020        1.8847
     kap.κ_team_raw[4]   -0.4257    0.8939    0.0058   24276.8166   11786.6753    1.0011        1.4665
     kap.κ_team_raw[5]   -0.0299    0.9932    0.0056   31331.5808   11314.1353    1.0016        1.8926
     kap.κ_team_raw[6]    0.0282    0.8710    0.0050   30036.8030   11650.1195    1.0013        1.8144
     kap.κ_team_raw[7]    0.2793    0.8761    0.0053   27466.6103   11868.9782    1.0006        1.6592
     kap.κ_team_raw[8]    0.3828    0.8827    0.0055   25803.3262   11627.5990    1.0018        1.5587
     kap.κ_team_raw[9]   -0.2512    0.8784    0.0052   28955.5647   12490.4655    1.0012        1.7491
    kap.κ_team_raw[10]    0.2413    0.8745    0.0053   27079.1551   12463.1264    1.0008        1.6358
    kap.κ_team_raw[11]   -0.1685    0.8691    0.0051   28941.7507   11567.8991    1.0006        1.7483
         p_dyn.w_G_att   -0.1157    0.0929    0.0006   24976.8358   12335.4857    1.0020        1.5088
         p_dyn.w_G_def    0.1178    0.0873    0.0005   28550.2515   12252.3428    1.0002        1.7246
  p_dyn.w_Outfield_att    0.0828    0.0208    0.0002   16481.2279   12593.9449    1.0007        0.9956
  p_dyn.w_Outfield_def   -0.1265    0.0178    0.0001   27490.5353   12213.4803    1.0016        1.6606
             dc.ρ_base   -0.1861    0.1469    0.0009   29166.2647   10686.4430    1.0015        1.7618
                dc.σ_ρ    0.1264    0.0832    0.0006   17907.6716   10262.0122    1.0002        1.0817
           dc.raw_ρ[1]   -0.0279    0.9469    0.0054   31278.2303   11699.5991    1.0012        1.8894
           dc.raw_ρ[2]    0.0376    0.9338    0.0053   31605.9332   10952.4860    1.0020        1.9092
           dc.raw_ρ[3]   -0.0122    0.9271    0.0053   30903.8712   12062.2596    1.0012        1.8668
           dc.raw_ρ[4]   -0.0298    0.9447    0.0056   28606.5829   10774.3327    1.0005        1.7280
           dc.raw_ρ[5]    0.0021    1.0060    0.0056   31826.3686   11904.1171    1.0009        1.9225
           dc.raw_ρ[6]    0.0383    0.9363    0.0051   33158.8041   11024.7125    1.0009        2.0030
           dc.raw_ρ[7]    0.0144    0.9396    0.0051   33718.1370   11913.1216    1.0013        2.0368
           dc.raw_ρ[8]   -0.0286    0.9323    0.0053   31038.8943   11708.3054    1.0014        1.8750
           dc.raw_ρ[9]    0.0859    0.9412    0.0052   32629.7440   10784.4058    1.0024        1.9711
          dc.raw_ρ[10]   -0.0508    0.9378    0.0051   33394.3314   11699.9681    1.0018        2.0172
          dc.raw_ρ[11]   -0.0285    0.9389    0.0050   35956.3429   10822.3632    1.0018        2.1720

                                                                                               
Quantiles                                                                                                                                                     
                                                                                               
            parameters      2.5%     25.0%     50.0%     75.0%     97.5%                                                                                      
                Symbol   Float64   Float64   Float64   Float64   Float64                                                                                      

                  ν_xg    2.5657    2.9742    3.2003    3.4283    3.8702                                                                                      
              σ_market    0.1496    0.1689    0.1798    0.1924    0.2202                                                                                      
               inter.μ   -0.0424    0.0705    0.1293    0.1897    0.3010                                                                                      
             ha.γ_base    0.1209    0.1838    0.2161    0.2486    0.3125                                                                                      
                ha.σ_γ    0.0018    0.0174    0.0368    0.0627    0.1265                                                                                      
      ha.γ_team_raw[1]   -1.6107   -0.3734    0.2637    0.8852    2.0259                                                                                      
      ha.γ_team_raw[2]   -2.3097   -1.1130   -0.4625    0.2069    1.5463                                                                                      
      ha.γ_team_raw[3]   -1.7484   -0.4572    0.1788    0.7861    2.0293                                                                                      
      ha.γ_team_raw[4]   -1.9927   -0.7698   -0.1328    0.4809    1.7023                                                                                      
      ha.γ_team_raw[5]   -1.9522   -0.6799   -0.0142    0.6590    1.9290                                                                                      
      ha.γ_team_raw[6]   -1.6526   -0.4106    0.2229    0.8500    2.0207                                                                                      
      ha.γ_team_raw[7]   -1.7377   -0.5279    0.0960    0.7145    1.9052                                                                                      
      ha.γ_team_raw[8]   -1.7233   -0.4686    0.1658    0.7964    1.9807                                                                                      
      ha.γ_team_raw[9]   -1.9645   -0.8063   -0.1882    0.4473    1.6693                                                                                      
     ha.γ_team_raw[10]   -1.7787   -0.5333    0.0817    0.7122    1.9134                                                                                      
     ha.γ_team_raw[11]   -1.9498   -0.8000   -0.1704    0.4391    1.6894                                                                                      
            kap.κ_base    0.5227    0.7316    0.8397    0.9554    1.1744                                                                                      
               kap.σ_κ    0.0044    0.0423    0.0815    0.1239    0.2057                                                                                      
     kap.κ_team_raw[1]   -1.3215   -0.0799    0.5351    1.1151    2.2104                                                                                      
     kap.κ_team_raw[2]   -2.9288   -1.8195   -1.2034   -0.5031    1.0248                                                                                      
     kap.κ_team_raw[3]   -1.4632   -0.3046    0.2672    0.8175    1.9412                                                                                      
     kap.κ_team_raw[4]   -2.1371   -1.0266   -0.4406    0.1561    1.4008                                                                                      
     kap.κ_team_raw[5]   -1.9504   -0.7054   -0.0309    0.6420    1.9085                                                                                      
     kap.κ_team_raw[6]   -1.7169   -0.5344    0.0277    0.6026    1.7561                                                                                      
     kap.κ_team_raw[7]   -1.5109   -0.2817    0.2906    0.8618    1.9881                                                                                      
     kap.κ_team_raw[8]   -1.4071   -0.1889    0.3902    0.9784    2.0693                                                                                      
     kap.κ_team_raw[9]   -1.9117   -0.8463   -0.2735    0.3280    1.5321                                                                                      
    kap.κ_team_raw[10]   -1.5435   -0.3270    0.2646    0.8325    1.9359                                                                                      
    kap.κ_team_raw[11]   -1.8447   -0.7480   -0.1887    0.3908    1.5985                                                                                      
         p_dyn.w_G_att   -0.2992   -0.1787   -0.1155   -0.0538    0.0647                                                                                      
         p_dyn.w_G_def   -0.0556    0.0592    0.1182    0.1774    0.2890                                                                                      
  p_dyn.w_Outfield_att    0.0414    0.0690    0.0832    0.0971    0.1225                                                                                      
  p_dyn.w_Outfield_def   -0.1614   -0.1386   -0.1266   -0.1143   -0.0916                                                                                      
             dc.ρ_base   -0.4888   -0.2775   -0.1817   -0.0870    0.0885                                                                                      
                dc.σ_ρ    0.0173    0.0657    0.1092    0.1692    0.3306                                                                                      
           dc.raw_ρ[1]   -1.8824   -0.6543   -0.0335    0.6030    1.8337                                                                                      
           dc.raw_ρ[2]   -1.8239   -0.5938    0.0397    0.6668    1.8797                                                                                      
           dc.raw_ρ[3]   -1.8281   -0.6359   -0.0097    0.6198    1.8047                                                                                      
           dc.raw_ρ[4]   -1.8836   -0.6752   -0.0330    0.6111    1.8233                                                                                      
           dc.raw_ρ[5]   -1.9711   -0.6832    0.0079    0.6911    1.9489                                                                                      
           dc.raw_ρ[6]   -1.8044   -0.5818    0.0377    0.6591    1.8753                                                                                      
           dc.raw_ρ[7]   -1.8555   -0.6035    0.0147    0.6476    1.8487                                                                                      
           dc.raw_ρ[8]   -1.8774   -0.6531   -0.0211    0.5873    1.8170                                                                                      
           dc.raw_ρ[9]   -1.7705   -0.5315    0.0837    0.7185    1.9119                                                                                      
          dc.raw_ρ[10]   -1.9007   -0.6782   -0.0453    0.5713    1.7893                                                                                      
          dc.raw_ρ[11]   -1.8843   -0.6530   -0.0253    0.5958    1.7960
=#


# ==========================================
# 5. MODEL 2: HIERARCHICAL MONTHLY INTERCEPTION
# ==========================================
println("\n[INFO] Initializing DynamicDixonColesXGOutfieldPlayerTimeDecayModel (MONTHLY INTERCEPTION)...")
model_monthly = PreGame.DynamicDixonColesXGOutfieldPlayerTimeDecayModel(
    interception_config    = PreGame.HierarchicalMonthlyInterception(),
    player_dynamics_config = dyn_cfg,
    dispersion_config      = disp_cfg,
    homeadvantage_config   = ha_cfg,
    kappa_config           = kap_cfg,
    dixon_coles_config     = PreGame.HierarchicalTeamDixonColesConfig(),
    player_ratings_feature = feature_cfg_bayes,
    market_feature_config  = Features.DixonColesMarketFeature(),
    market_weight          = 0.4
)
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





#=
julia> display(describe(chains_monthly))                                                                                                                                                      
Chains MCMC chain (1000×61×16 Array{Float64, 3}):                                                                                                                                             
                                                                                                                                                                                              
Iterations        = 501:1:1500                                                                                                                                                                
Number of chains  = 16                                                                                                                                                                        
Samples per chain = 1000                                                                                                                                                                      
Wall duration     = 666.23 seconds                                                                                                                                                            
Compute duration  = 10454.61 seconds                                                                                                                                                          
parameters        = ν_xg, σ_market, inter.μ_base[1], inter.μ_base[2], inter.σ_month, inter.raw_month[1], inter.raw_month[2], inter.raw_month[3], inter.raw_month[4], inter.raw_month[5], inter.raw_month[6], inter.raw_month[7], inter.raw_month[8], inter.raw_month[9], inter.raw_month[10], inter.raw_month[11], inter.raw_month[12], ha.γ_base, ha.σ_γ, ha.γ_team_raw[1], ha.γ_team_raw[2], ha.γ_team_raw[3], ha.γ_team_raw[4], ha.γ_team_raw[5], ha.γ_team_raw[6], ha.γ_team_raw[7], ha.γ_team_raw[8], ha.γ_team_raw[9], ha.γ_team_raw[10], ha.γ_team_raw[11], kap.κ_base, kap.σ_κ, kap.κ_team_raw[1], kap.κ_team_raw[2], kap.κ_team_raw[3], kap.κ_te
am_raw[4], kap.κ_team_raw[5], kap.κ_team_raw[6], kap.κ_team_raw[7], kap.κ_team_raw[8], kap.κ_team_raw[9], kap.κ_team_raw[10], kap.κ_team_raw[11], p_dyn.w_G_att, p_dyn.w_G_def, p_dyn.w_Outfield_att, p_dyn.w_Outfield_def
internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, lp, logprior, loglikelihood
                                                                                                                                                                                              
Summary Statistics                                                                                                                                                                            
                                                                                                                                                                                              
            parameters      mean       std      mcse     ess_bulk     ess_tail      rhat   ess_per_sec 
                Symbol   Float64   Float64   Float64      Float64      Float64   Float64       Float64 
                                                                                                                                                                                              
                  ν_xg    3.1981    0.3314    0.0019   30349.7818   11387.2027    1.0017        2.9030
              σ_market    0.2202    0.0258    0.0002   25539.8102   11993.2121    1.0008        2.4429
       inter.μ_base[1]    0.1925    0.1618    0.0009   31333.7862   11862.7914    1.0013        2.9971
       inter.μ_base[2]    0.2152    0.1160    0.0008   23867.8289   11523.2848    1.0007        2.2830
         inter.σ_month    0.0290    0.0241    0.0002   12187.3543    8512.4855    1.0007        1.1657
    inter.raw_month[1]    0.0027    0.9962    0.0053   34994.5223   11561.4896    1.0016        3.3473
    inter.raw_month[2]   -0.0021    0.9854    0.0055   32487.5746   11739.6968    1.0012        3.1075
    inter.raw_month[3]   -0.0911    0.9687    0.0053   33847.5551   11605.6034    1.0005        3.2376
    inter.raw_month[4]   -0.0323    0.9718    0.0055   30788.8594   10548.6177    1.0010        2.9450
    inter.raw_month[5]   -0.0083    0.9412    0.0052   32246.4154   11402.6149    1.0021        3.0844
    inter.raw_month[6]   -0.0394    0.9369    0.0054   29831.0896   11717.1226    1.0013        2.8534
    inter.raw_month[7]    0.1585    0.9625    0.0055   31188.4810   11931.0114    1.0006        2.9832
    inter.raw_month[8]   -0.1326    0.9314    0.0052   31636.1400   12063.8619    1.0005        3.0260
    inter.raw_month[9]    0.1694    0.9197    0.0053   30720.4966   12206.9300    1.0006        2.9385
   inter.raw_month[10]   -0.1217    0.9236    0.0053   30573.8192   13004.6153    1.0022        2.9244
   inter.raw_month[11]    0.0854    0.9577    0.0057   28386.4610   10333.5483    1.0015        2.7152
   inter.raw_month[12]    0.0043    1.0056    0.0055   33546.7845   11887.9708    1.0001        3.2088
             ha.γ_base    0.2238    0.0546    0.0003   29043.0451   11894.4919    1.0008        2.7780
                ha.σ_γ    0.0439    0.0340    0.0003   10415.4474    9094.5998    0.9999        0.9963
      ha.γ_team_raw[1]    0.2520    0.9562    0.0056   28949.4080   11310.9140    1.0011        2.7691
      ha.γ_team_raw[2]   -0.3952    0.9811    0.0058   28520.4125   11652.1602    1.0014        2.7280
      ha.γ_team_raw[3]    0.1321    0.9356    0.0056   27878.8956   11698.7438    1.0009        2.6667
      ha.γ_team_raw[4]   -0.1272    0.9423    0.0052   32681.4894   11704.7517    1.0014        3.1260
      ha.γ_team_raw[5]   -0.0019    0.9998    0.0056   31984.1581   11346.7000    1.0013        3.0593
      ha.γ_team_raw[6]    0.2104    0.9501    0.0052   33060.1520   11499.8367    1.0019        3.1623
      ha.γ_team_raw[7]    0.0642    0.9346    0.0054   30313.4529   11495.0576    1.0015        2.8995
      ha.γ_team_raw[8]    0.1315    0.9498    0.0054   30860.5409   12410.5527    1.0009        2.9519
      ha.γ_team_raw[9]   -0.1614    0.9499    0.0058   27253.5566   10474.7345    1.0010        2.6068
     ha.γ_team_raw[10]    0.0395    0.9499    0.0053   31736.9731   12194.3685    1.0018        3.0357
     ha.γ_team_raw[11]   -0.1115    0.9308    0.0054   29693.4918   10953.8286    1.0032        2.8402
            kap.κ_base    0.5011    0.0912    0.0006   23814.7996   11425.0516    1.0009        2.2779
               kap.σ_κ    0.0630    0.0469    0.0005    8787.0136    9530.5364    1.0006        0.8405
     kap.κ_team_raw[1]    0.1436    0.9280    0.0054   29644.6224   11418.6648    1.0014        2.8356
     kap.κ_team_raw[2]   -0.6232    1.0005    0.0065   23922.2444   12526.5647    1.0006        2.2882
     kap.κ_team_raw[3]    0.1883    0.9305    0.0053   30884.4582   11590.9404    1.0007        2.9541
     kap.κ_team_raw[4]   -0.3053    0.9507    0.0055   30251.9948   12479.2164    1.0009        2.8937
     kap.κ_team_raw[5]   -0.0116    1.0116    0.0056   32928.5974   11488.6158    1.0013        3.1497
     kap.κ_team_raw[6]   -0.1209    0.9277    0.0053   30653.8601   11280.9596    1.0015        2.9321
     kap.κ_team_raw[7]    0.2123    0.9402    0.0054   29836.5517   11766.3680    1.0004        2.8539
     kap.κ_team_raw[8]    0.1923    0.9240    0.0054   29526.9370   12664.0445    1.0007        2.8243
     kap.κ_team_raw[9]   -0.2394    0.9391    0.0054   29978.3550   11538.4579    1.0009        2.8675
    kap.κ_team_raw[10]    0.0300    0.9386    0.0055   28960.0723   11644.1226    1.0006        2.7701
    kap.κ_team_raw[11]   -0.2233    0.9434    0.0057   27197.4170   11655.5080    1.0010        2.6015
         p_dyn.w_G_att   -0.0998    0.0987    0.0006   30072.9533   11580.5614    1.0011        2.8765
         p_dyn.w_G_def    0.1207    0.0946    0.0006   29682.8802   12179.9026    1.0015        2.8392
  p_dyn.w_Outfield_att    0.0968    0.0209    0.0001   24001.0312   12136.0571    1.0006        2.2957
  p_dyn.w_Outfield_def   -0.1166    0.0194    0.0001   28811.8561   11906.6552    1.0006        2.7559
                                                                                               
                                                                                               
Quantiles                                                               
                                                                                               
            parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
                Symbol   Float64   Float64   Float64   Float64   Float64                                                                                      

                  ν_xg    2.5625    2.9732    3.1945    3.4164    3.8616                                                                                      
              σ_market    0.1765    0.2016    0.2180    0.2359    0.2766                                                                                      
       inter.μ_base[1]   -0.1242    0.0830    0.1927    0.3002    0.5118                                                                                      
       inter.μ_base[2]   -0.0102    0.1356    0.2168    0.2935    0.4377                                                                                      
         inter.σ_month    0.0009    0.0107    0.0232    0.0408    0.0899                                                                                      
    inter.raw_month[1]   -1.9397   -0.6761    0.0051    0.6702    1.9660                                                                                      
    inter.raw_month[2]   -1.9407   -0.6747   -0.0009    0.6672    1.9236                                                                                      
    inter.raw_month[3]   -1.9819   -0.7383   -0.0993    0.5672    1.8040                                                                                      
    inter.raw_month[4]   -1.9410   -0.6815   -0.0352    0.6174    1.8642                                                                                      
    inter.raw_month[5]   -1.8926   -0.6255   -0.0072    0.6233    1.8611                                                                                      
    inter.raw_month[6]   -1.8784   -0.6754   -0.0412    0.5850    1.7963                                                                                      
    inter.raw_month[7]   -1.7462   -0.4836    0.1650    0.8010    2.0386                                                                                      
    inter.raw_month[8]   -1.9357   -0.7627   -0.1420    0.4921    1.7059                                                                                      
    inter.raw_month[9]   -1.6926   -0.4339    0.1788    0.7811    1.9700                                                                                      
   inter.raw_month[10]   -1.9259   -0.7377   -0.1236    0.4907    1.6930                                                                                      
   inter.raw_month[11]   -1.8046   -0.5472    0.0876    0.7252    1.9652                                                                                      
   inter.raw_month[12]   -1.9626   -0.6766    0.0023    0.6840    1.9796                                                                                      
             ha.γ_base    0.1159    0.1878    0.2237    0.2600    0.3327                                                                                      
                ha.σ_γ    0.0017    0.0175    0.0368    0.0628    0.1267                                                                                      
      ha.γ_team_raw[1]   -1.6447   -0.3920    0.2571    0.8999    2.0954                                                                                      
      ha.γ_team_raw[2]   -2.2611   -1.0627   -0.4056    0.2543    1.5865                                                                                      
      ha.γ_team_raw[3]   -1.7294   -0.4890    0.1339    0.7566    1.9609                                                                                      
      ha.γ_team_raw[4]   -1.9768   -0.7692   -0.1314    0.5039    1.7450                                                                                      
      ha.γ_team_raw[5]   -1.9729   -0.6723   -0.0021    0.6676    1.9779                                                                                      
      ha.γ_team_raw[6]   -1.7037   -0.4137    0.2200    0.8555    2.0521                                                                                      
      ha.γ_team_raw[7]   -1.7584   -0.5743    0.0632    0.7054    1.8776                                                                                      
      ha.γ_team_raw[8]   -1.7355   -0.5040    0.1486    0.7733    1.9664                                                                                      
      ha.γ_team_raw[9]   -2.0097   -0.7987   -0.1704    0.4757    1.7112                                                                                      
     ha.γ_team_raw[10]   -1.8282   -0.5999    0.0444    0.6746    1.9129                                                                                      
     ha.γ_team_raw[11]   -1.9335   -0.7464   -0.1110    0.5068    1.7229                                                                                      
            kap.κ_base    0.3246    0.4394    0.5000    0.5612    0.6820                                                                                      
               kap.σ_κ    0.0026    0.0257    0.0536    0.0909    0.1753                                                                                      
     kap.κ_team_raw[1]   -1.7184   -0.4726    0.1597    0.7643    1.9601                                                                                      
     kap.κ_team_raw[2]   -2.5405   -1.2978   -0.6426    0.0373    1.3818                                                                                      
     kap.κ_team_raw[3]   -1.6420   -0.4263    0.1940    0.8177    1.9891                                                                                      
     kap.κ_team_raw[4]   -2.1270   -0.9538   -0.3074    0.3244    1.5904                                                                                      
     kap.κ_team_raw[5]   -2.0096   -0.6888   -0.0085    0.6604    1.9799                                                                                      
     kap.κ_team_raw[6]   -1.9315   -0.7378   -0.1300    0.4907    1.7138                                                                                      
     kap.κ_team_raw[7]   -1.6545   -0.4040    0.2193    0.8535    2.0481                                                                                      
     kap.κ_team_raw[8]   -1.6463   -0.4200    0.1976    0.8215    1.9924                                                                                      
     kap.κ_team_raw[9]   -2.0635   -0.8679   -0.2557    0.3770    1.6638                                                                                      
    kap.κ_team_raw[10]   -1.8458   -0.5830    0.0245    0.6522    1.8786                                                                                      
    kap.κ_team_raw[11]   -2.0435   -0.8648   -0.2440    0.4015    1.6922                                                                                      
         p_dyn.w_G_att   -0.2931   -0.1658   -0.1005   -0.0335    0.0959                                                                                      
         p_dyn.w_G_def   -0.0644    0.0564    0.1212    0.1843    0.3067                                                                                      
  p_dyn.w_Outfield_att    0.0551    0.0828    0.0970    0.1112    0.1367                                                                                      
  p_dyn.w_Outfield_def   -0.1547   -0.1297   -0.1166   -0.1036   -0.0786
=#




#=
julia> display(describe(chains_monthly))                                                                                                                                                                                                     
Chains MCMC chain (1000×74×16 Array{Float64, 3}):                                                                                                                                                                                            
                                                                                                                                                                                                                                             
Iterations        = 501:1:1500                                                                                                                                                                                                               
Number of chains  = 16                                                                                                                                                                                                                       
Samples per chain = 1000                                                                                                                                                                                                                     
Wall duration     = 1386.99 seconds                                                                                                                                                                                                          
Compute duration  = 21793.89 seconds                                                                                                                                                                                                         
parameters        = ν_xg, σ_market, inter.μ_base[1], inter.μ_base[2], inter.σ_month, inter.raw_month[1], inter.raw_month[2], inter.raw_month[3], inter.raw_month[4], inter.raw_month[5], inter.raw_month[6], inter.raw_month[7], inter.raw_month[8], inter.raw_month[9], inter.raw_month[10], inter.raw_month[11], inter.raw_month[12], ha.γ_base, ha.σ_γ, ha.γ_team_raw[1], ha.γ_team_raw[2], ha.γ_team_raw[3], ha.γ_team_raw[4], ha.γ_team_raw[5], ha.γ_team_raw[6], ha.γ_team_raw[7], ha.γ_team_raw[8], ha.γ_team_raw[9], ha.γ_team_raw[10], ha.γ_team_raw[11], kap.κ_base, kap.σ_κ, kap.κ_team_raw[1], kap.κ_team_raw[2], kap.κ_team_raw[3], kap.κ_te
am_raw[4], kap.κ_team_raw[5], kap.κ_team_raw[6], kap.κ_team_raw[7], kap.κ_team_raw[8], kap.κ_team_raw[9], kap.κ_team_raw[10], kap.κ_team_raw[11], p_dyn.w_G_att, p_dyn.w_G_def, p_dyn.w_Outfield_att, p_dyn.w_Outfield_def, dc.ρ_base, dc.σ_ρ, dc.raw_ρ[1], dc.raw_ρ[2], dc.raw_ρ[3], dc.raw_ρ[4], dc.raw_ρ[5], dc.raw_ρ[6], dc.raw_ρ[7], dc.raw_ρ[8], dc.raw_ρ[9], dc.raw_ρ[10], dc.raw_ρ[11]
internals         = n_steps, is_accept, acceptance_rate, log_density, hamiltonian_energy, hamiltonian_energy_error, max_hamiltonian_energy_error, tree_depth, numerical_error, step_size, nom_step_size, lp, logprior, loglikelihood         
                                                                                                                                                                                                                                             
Summary Statistics                                                                                                                                                                                                                           
                                                                                                                                                                                                                                             
            parameters      mean       std      mcse     ess_bulk     ess_tail      rhat   ess_per_sec                                                                                                                                       
                Symbol   Float64   Float64   Float64      Float64      Float64   Float64       Float64                                                                                                                                       
                                                                                                                                                                                                                                             
                  ν_xg    3.1977    0.3309    0.0019   28391.0689   11089.3273    1.0006        1.3027                                                                                                                                       
              σ_market    0.1830    0.0182    0.0001   18663.3228   12324.4101    1.0009        0.8564                                                                                                                                       
       inter.μ_base[1]    0.1219    0.1567    0.0010   25831.2456   13260.6671    1.0014        1.1853                                                                                                                                       
       inter.μ_base[2]    0.0743    0.1259    0.0009   18069.3159   12244.2778    1.0006        0.8291                                                                                                                                       
         inter.σ_month    0.0267    0.0222    0.0002   11462.1353    7789.8467    1.0002        0.5259                                                                                                                                       
    inter.raw_month[1]    0.0147    0.9890    0.0056   30922.3493   12014.0082    1.0017        1.4189                                                                                                                                       
    inter.raw_month[2]   -0.0172    0.9814    0.0057   29963.6740   11969.8797    1.0005        1.3749                                                                                                                                       
    inter.raw_month[3]   -0.1192    0.9651    0.0056   30059.5664   11670.8075    1.0008        1.3793                                                                                                                                       
    inter.raw_month[4]   -0.0676    0.9599    0.0054   31534.2660   11706.1187    1.0030        1.4469                                                                                                                                       
    inter.raw_month[5]   -0.0413    0.9348    0.0055   28381.4863   11798.6030    1.0023        1.3023                                                                                                                                       
    inter.raw_month[6]   -0.0429    0.9349    0.0055   29140.1960   11797.4165    1.0005        1.3371                                                                                                                                       
    inter.raw_month[7]    0.1436    0.9723    0.0058   27771.7077   11602.2365    1.0009        1.2743                                                                                                                                       
    inter.raw_month[8]   -0.1516    0.9395    0.0055   29380.4209   11343.3662    1.0007        1.3481                                                                                                                                       
    inter.raw_month[9]    0.2159    0.9220    0.0057   26075.0323   11561.1099    1.0015        1.1964                                                                                                                                       
   inter.raw_month[10]   -0.0867    0.9150    0.0058   25310.0288   11836.1614    1.0009        1.1613                                                                                                                                       
   inter.raw_month[11]    0.1387    0.9458    0.0055   30671.3602   11938.2020    1.0012        1.4073                                                                                                                                       
   inter.raw_month[12]    0.0139    1.0151    0.0056   33185.2752   11503.0966    1.0012        1.5227                                                                                                                                       
             ha.γ_base    0.2180    0.0499    0.0003   25646.1518   11759.9602    1.0018        1.1768                                                                                                                                       
                ha.σ_γ    0.0440    0.0339    0.0003    9542.9325    8887.4246    1.0009        0.4379                                                        
      ha.γ_team_raw[1]    0.2563    0.9420    0.0059   25631.6644   11776.1970    1.0013        1.1761                                                        
      ha.γ_team_raw[2]   -0.4462    0.9942    0.0062   25788.6910   12213.3982    1.0004        1.1833                                                        
      ha.γ_team_raw[3]    0.1583    0.9343    0.0055   28738.6984   12329.2372    1.0014        1.3187                                                        
      ha.γ_team_raw[4]   -0.1370    0.9489    0.0056   28705.8295   11756.9883    1.0012        1.3172                                                        
      ha.γ_team_raw[5]   -0.0152    0.9935    0.0053   35771.0820   11235.3412    1.0020        1.6413                                                        
      ha.γ_team_raw[6]    0.2282    0.9357    0.0057   27280.7461   11925.1061    1.0014        1.2518                                                        
      ha.γ_team_raw[7]    0.1029    0.9388    0.0057   27545.5678   12115.7879    1.0009        1.2639                                                        
      ha.γ_team_raw[8]    0.1687    0.9447    0.0056   28239.7017   11592.3192    1.0011        1.2958                                                        
      ha.γ_team_raw[9]   -0.1906    0.9388    0.0057   27669.3686   11050.4131    1.0005        1.2696                                                        
     ha.γ_team_raw[10]    0.0832    0.9342    0.0053   31631.3832   12331.8261    1.0003        1.4514                                                        
     ha.γ_team_raw[11]   -0.1704    0.9270    0.0054   30002.1363   11395.6424    1.0011        1.3766                                                        
            kap.κ_base    0.8842    0.1774    0.0012   20735.7207   11200.0920    1.0010        0.9514                                                        
               kap.σ_κ    0.0828    0.0551    0.0007    5791.5036    6744.6106    1.0018        0.2657                                                        
     kap.κ_team_raw[1]    0.4894    0.9078    0.0063   21323.8790   12166.2135    1.0009        0.9784                                                        
     kap.κ_team_raw[2]   -1.0579    1.0207    0.0089   13561.0251   12176.9179    1.0015        0.6222                                                        
     kap.κ_team_raw[3]    0.2657    0.8800    0.0052   28378.8300   12210.0353    1.0006        1.3021                                                        
     kap.κ_team_raw[4]   -0.4031    0.9027    0.0061   21839.1397   11905.0702    1.0014        1.0021                                                        
     kap.κ_team_raw[5]   -0.0289    1.0101    0.0056   32516.7656   11576.7119    1.0017        1.4920                                                        
     kap.κ_team_raw[6]    0.0293    0.8635    0.0050   29507.4573   12282.6384    1.0014        1.3539                                                        
     kap.κ_team_raw[7]    0.2679    0.8815    0.0056   25108.0235   11900.4295    1.0014        1.1521                                                        
     kap.κ_team_raw[8]    0.3755    0.8896    0.0059   23214.1781   11924.4056    1.0016        1.0652                                                        
     kap.κ_team_raw[9]   -0.2348    0.8760    0.0055   25983.3645   12446.2654    1.0022        1.1922                                                        
    kap.κ_team_raw[10]    0.2221    0.8853    0.0054   27061.7302   12464.7995    1.0013        1.2417                                                        
    kap.κ_team_raw[11]   -0.1448    0.8781    0.0054   26584.1645   11463.6443    1.0006        1.2198                                                        
         p_dyn.w_G_att   -0.1112    0.0921    0.0006   23512.5312   12304.2417    1.0012        1.0789                                                        
         p_dyn.w_G_def    0.1241    0.0885    0.0005   26687.4823   12195.2187    1.0004        1.2245                                                        
  p_dyn.w_Outfield_att    0.0860    0.0214    0.0002   13906.0098   12133.2835    1.0011        0.6381                                                        
  p_dyn.w_Outfield_def   -0.1246    0.0182    0.0001   27344.8619   12602.9264    1.0017        1.2547                                                        
             dc.ρ_base   -0.1841    0.1464    0.0009   28770.5840   12047.4566    1.0006        1.3201                                                        
                     ⋮         ⋮         ⋮         ⋮            ⋮            ⋮         ⋮             ⋮                                                        
                                                                                               
                                                                                         12 rows omitted                                                      
                                                                                               
Quantiles                                                                                      
                                                                                               
            parameters      2.5%     25.0%     50.0%     75.0%     97.5% 
                Symbol   Float64   Float64   Float64   Float64   Float64 

                  ν_xg    2.5794    2.9715    3.1904    3.4142    3.8686                                                                                      
              σ_market    0.1513    0.1702    0.1815    0.1942    0.2228                                                                                      
       inter.μ_base[1]   -0.1862    0.0140    0.1224    0.2268    0.4297                                                                                      
       inter.μ_base[2]   -0.1699   -0.0109    0.0742    0.1576    0.3230                                                                                      
         inter.σ_month    0.0009    0.0099    0.0213    0.0379    0.0822                                                                                      
    inter.raw_month[1]   -1.9416   -0.6426    0.0188    0.6761    1.9301                                                                                      
    inter.raw_month[2]   -1.9459   -0.6778   -0.0218    0.6429    1.9254                                                                                      
    inter.raw_month[3]   -1.9936   -0.7759   -0.1250    0.5260    1.7860                                                                                      
    inter.raw_month[4]   -1.9282   -0.7128   -0.0729    0.5770    1.8449                                                                                      
    inter.raw_month[5]   -1.8670   -0.6751   -0.0366    0.5809    1.7895                                                                                      
    inter.raw_month[6]   -1.8977   -0.6550   -0.0415    0.5707    1.7883                                                                                      
    inter.raw_month[7]   -1.7668   -0.5056    0.1534    0.7911    2.0588                                                                                      
    inter.raw_month[8]   -1.9868   -0.7879   -0.1558    0.4662    1.7167                                                                                      
    inter.raw_month[9]   -1.6327   -0.3899    0.2217    0.8224    2.0124                                                                                      
   inter.raw_month[10]   -1.9142   -0.6881   -0.0866    0.5093    1.7165                                                                                      
   inter.raw_month[11]   -1.7734   -0.4866    0.1453    0.7794    1.9910                                                                                      
   inter.raw_month[12]   -1.9682   -0.6690    0.0059    0.6880    2.0160                                                                                      
             ha.γ_base    0.1194    0.1848    0.2179    0.2515    0.3157                                                                                      
                ha.σ_γ    0.0018    0.0176    0.0367    0.0629    0.1256                                                                                      
      ha.γ_team_raw[1]   -1.6098   -0.3761    0.2669    0.8954    2.0561                                                                                      
      ha.γ_team_raw[2]   -2.3218   -1.1269   -0.4703    0.2098    1.5600                                                                                      
      ha.γ_team_raw[3]   -1.7020   -0.4563    0.1676    0.7927    1.9587                                                                                      
      ha.γ_team_raw[4]   -1.9904   -0.7761   -0.1462    0.4870    1.7523                                                                                      
      ha.γ_team_raw[5]   -1.9629   -0.6878   -0.0117    0.6521    1.9320                                                                                      
      ha.γ_team_raw[6]   -1.6295   -0.3899    0.2370    0.8545    2.0605                                                                                      
      ha.γ_team_raw[7]   -1.7708   -0.5174    0.1033    0.7308    1.9541                                                                                      
      ha.γ_team_raw[8]   -1.6830   -0.4659    0.1726    0.8071    1.9943                                                                                      
      ha.γ_team_raw[9]   -2.0392   -0.8160   -0.2029    0.4287    1.6773                                                                                      
     ha.γ_team_raw[10]   -1.7867   -0.5286    0.0838    0.7103    1.9112                                                                                      
     ha.γ_team_raw[11]   -1.9468   -0.8090   -0.1815    0.4474    1.6815                                                                                      
            kap.κ_base    0.5382    0.7642    0.8821    1.0027    1.2384                                                                                      
               kap.σ_κ    0.0034    0.0377    0.0764    0.1196    0.2014                                                                                      
     kap.κ_team_raw[1]   -1.3737   -0.0989    0.5161    1.1019    2.2145                                                                                      
     kap.κ_team_raw[2]   -2.8982   -1.7722   -1.1139   -0.4032    1.1128                                                                                      
     kap.κ_team_raw[3]   -1.4960   -0.3098    0.2755    0.8725    1.9645                                                                                      
     kap.κ_team_raw[4]   -2.1082   -1.0263   -0.4291    0.1850    1.4480                                                                                      
     kap.κ_team_raw[5]   -1.9950   -0.7106   -0.0254    0.6504    1.9566                                                                                      
     kap.κ_team_raw[6]   -1.6538   -0.5483    0.0294    0.5972    1.7262                                                                                      
     kap.κ_team_raw[7]   -1.4931   -0.3111    0.2826    0.8588    1.9603                                                                                      
     kap.κ_team_raw[8]   -1.4291   -0.2162    0.3916    0.9735    2.0940                                                                                      
     kap.κ_team_raw[9]   -1.9040   -0.8196   -0.2578    0.3326    1.5526                                                                                      
    kap.κ_team_raw[10]   -1.5391   -0.3581    0.2366    0.8077    1.9548                                                                                      
    kap.κ_team_raw[11]   -1.8474   -0.7360   -0.1633    0.4380    1.6504                                                                                      
         p_dyn.w_G_att   -0.2917   -0.1729   -0.1110   -0.0494    0.0694                                                                                      
         p_dyn.w_G_def   -0.0499    0.0656    0.1241    0.1836    0.2974                                                                                      
  p_dyn.w_Outfield_att    0.0430    0.0716    0.0866    0.1004    0.1267                                                                                      
  p_dyn.w_Outfield_def   -0.1601   -0.1366   -0.1245   -0.1123   -0.0888                                                                                      
             dc.ρ_base   -0.4854   -0.2764   -0.1793   -0.0871    0.0928                                                                                      
                     ⋮         ⋮         ⋮         ⋮         ⋮         ⋮                                                                                      

                                                           12 rows omitted
=#


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

mp_global = Predictions.model_inference(ds, results_global)

mp_hier = Predictions.model_inference(ds, results_monthly)



mae_global, comp_gloabl = compar(mp_global)
mae_hier, comp_hier = compar(mp_hier)

mae_hier
mae_global

describe(comp_gloabl.prob_diff)

describe(comp_hier.prob_diff)

