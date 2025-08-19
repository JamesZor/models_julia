data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats/")


println(data_files.base_dir) 
println(data_files.match)
println(data_files.odds) 


data_store = DataStore(data_files) 

transformer = DataTransformers(create_list_mapping)
mapping = MappedData(data_store, transformer)
features = FeaturesBasicMaherModel(data_store, mapping)
model = create_maher_models(features)
cfg = ModelTrainingConfig(2_000, true, 4, false)

model_chains = train_maher_model(model, cfg)

results = extract_team_parameters_essential(model_chains, mapping)


results = extract_team_parameters(model_chains, mapping)

#= 
 Row │ team                          attack_ht  defense_ht  home_ht   attack_st  defense_st  home_st   attack_ft  defense_ft  home_ft 
     │ String                        Float64    Float64     Float64   Float64    Float64     Float64   Float64    Float64     Float64 
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ motherwell                     0.967462    1.00774   0.696416   1.04657     0.963764  0.810256   1.12391     1.00586    1.3577
   2 │ ross-county                    0.849708    0.990544  0.696416   0.9309      1.09301   0.810256   0.928987    1.08501    1.3577
   3 │ st-johnstone                   0.816495    0.92382   0.696416   0.801017    0.906526  0.810256   0.790175    0.906395   1.3577
   4 │ heart-of-midlothian            1.18832     0.790897  0.696416   1.24622     0.857554  0.810256   1.4384      0.790776   1.3577
   5 │ aberdeen                       1.00307     0.885927  0.696416   0.962529    0.894863  0.810256   1.08832     0.874431   1.3577
   6 │ rangers                        1.53109     0.729716  0.696416   1.55749     0.546109  0.810256   2.01191     0.528379   1.3577
   7 │ livingston                     0.952417    1.00562   0.696416   0.817081    0.912073  0.810256   0.910745    0.963763   1.3577
   8 │ kilmarnock                     0.866685    0.937867  0.696416   0.962863    0.820973  0.810256   0.970222    0.875508   1.3577
   9 │ celtic                         1.67085     0.627457  0.696416   1.67906     0.656093  0.810256   2.20476     0.533516   1.3577
  10 │ hibernian                      0.98395     0.893948  0.696416   1.0734      0.867121  0.810256   1.16357     0.865773   1.3577
  11 │ dundee-united                  0.881208    0.842143  0.696416   0.995868    0.799561  0.810256   1.0704      0.841412   1.3577
  12 │ st-mirren                      0.909637    0.906429  0.696416   0.893414    0.903854  0.810256   0.947419    0.89785    1.3577
  13 │ dundee-fc                      0.912875    1.02774   0.696416   1.03431     0.963841  0.810256   1.11433     1.10015    1.3577
  14 │ hamilton-academical            0.804576    1.22034   0.696416   0.833292    1.18278   0.810256   0.71993     1.29388    1.3577
  15 │ arbroath                       0.743543    1.02372   0.696416   0.799294    1.01751   0.810256   0.772072    1.20221    1.3577
  16 │ partick-thistle                0.969014    0.973384  0.696416   1.04915     1.11449   0.810256   1.0892      1.17065    1.3577
  17 │ inverness-caledonian-thistle   0.847687    0.809047  0.696416   0.919582    0.915663  0.810256   0.959892    0.958671   1.3577
  18 │ raith-rovers                   0.977671    0.844447  0.696416   0.902444    1.08795   0.810256   1.01235     1.09366    1.3577
  19 │ queens-park-fc                 1.16778     1.12786   0.696416   1.10093     1.34026   0.810256   1.07421     1.23055    1.3577
  20 │ greenock-morton                0.811356    0.990763  0.696416   0.839358    0.940004  0.810256   0.865088    1.12928    1.3577
  21 │ ayr-united                     0.87077     1.06468   0.696416   0.890741    0.976742  0.810256   0.93356     1.20186    1.3577
  22 │ cove-rangers                   1.17775     1.67648   0.696416   1.13318     1.84168   0.810256   0.672487    1.50086    1.3577
  23 │ queen-of-the-south             1.00987     1.29492   0.696416   0.922051    1.36332   0.810256   0.804414    1.34973    1.3577
  24 │ alloa-athletic                 1.4193      1.61444   0.696416   1.05196     1.59441   0.810256   0.812115    1.32858    1.3577
  25 │ dunfermline-athletic           0.947914    1.04259   0.696416   0.853299    1.03492   0.810256   0.875278    1.13799    1.3577
  26 │ airdrieonians                  1.33037     1.45722   0.696416   1.18515     1.25623   0.810256   0.796443    0.894397   1.3577
=#
