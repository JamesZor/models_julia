using BayesianFootball
using DataFrames
using Statistics
using Plots
using Distributions
using KernelDensity
using Plots
using StatsPlots

### Load the saved data for the basic maher model 
r1 = load_experiment("/home/james/bet_project/models_julia/experiments/maher_basic_test_20250820_231427.jld2")
# extract the results and config 
result = r1[1] 
experiment_config = r1[2]


# After running your experiment, you have:
# - result (ExperimentResult)
# - data_store (DataStore)
data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)


# Configure prediction settings
pred_config = PredictionConfig(
    quantile_levels = [0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975],
    keep_samples = false,  # Set to true if you want to keep raw samples
    n_simulations = 1000,
    max_goals = 10
)


first_split = result.chains_sequence[1]
model_params = extract_model_parameters(first_split, result.mapping, pred_config)

# Example: Look at parameter distribution for a specific team
# team_name = "celtic"
team_name = "st-mirren"
if haskey(model_params.teams, team_name)
    team = model_params.teams[team_name]
    println("\n$(team_name) Parameter Distribution:")
    println("Attack (FT):")
    println("  Mean: $(round(team.attack_ft.mean, digits=3))")
    println("  Median: $(round(team.attack_ft.median, digits=3))")
    println("  95% CI: [$(round(team.attack_ft.quantiles[1], digits=3)), " *
            "$(round(team.attack_ft.quantiles[end], digits=3))]")
    println("  IQR: [$(round(team.attack_ft.quantiles[3], digits=3)), " *
            "$(round(team.attack_ft.quantiles[5], digits=3))]")
end


# Predict a specific match
  home_team = "celtic"
  away_team = "rangers"
  
  prediction = predict_match_probabilistic(home_team, away_team, model_params, pred_config)
  
  if !isnothing(prediction)
      println("\n\nMatch Prediction: $home_team vs $away_team")
      println("="^50)
      
      println("\nExpected Goals (Full Time):")
      println("  $home_team xG: $(round(prediction.xG_home_ft.median, digits=2)) " *
              "[$(round(prediction.xG_home_ft.quantiles[1], digits=2)), " *
              "$(round(prediction.xG_home_ft.quantiles[end], digits=2))]")
      println("  $away_team xG: $(round(prediction.xG_away_ft.median, digits=2)) " *
              "[$(round(prediction.xG_away_ft.quantiles[1], digits=2)), " *
              "$(round(prediction.xG_away_ft.quantiles[end], digits=2))]")
      
      println("\nOutcome Probabilities:")
      println("  Home Win: $(round(prediction.prob_home_win * 100, digits=1))%")
      println("  Draw: $(round(prediction.prob_draw * 100, digits=1))%")
      println("  Away Win: $(round(prediction.prob_away_win * 100, digits=1))%")
      
      # Implied odds
      println("\nImplied Fair Odds:")
      println("  Home: $(round(1/prediction.prob_home_win, digits=2))")
      println("  Draw: $(round(1/prediction.prob_draw, digits=2))")
      println("  Away: $(round(1/prediction.prob_away_win, digits=2))")
  end


# Predict all matches for a round with uncertainty
cv_config = TimeSeriesSplitsConfig(["20/21", "21/22", "22/23", "23/24"], "24/25", :round)
target_matches = filter(row -> row.season == cv_config.target_season, data_store.matches)
round_1_matches = filter(row -> row.round == 1, target_matches)

predictions_df = predict_round_with_uncertainty(
    first_split,
    round_1_matches,
    result.mapping,
    pred_config
)

# Analyze predictions
summary = create_prediction_summary(predictions_df)

println("\n\nRound 1 Predictions Summary:")
println("="^50)
println("Number of matches: $(summary.n_matches)")
println("\nPrediction Interval Coverage (95% CI):")
println("  Home goals: $(round(summary.interval_coverage.home * 100, digits=1))%")
println("  Away goals: $(round(summary.interval_coverage.away * 100, digits=1))%")
println("\nAverage Interval Width:")
println("  Home: ±$(round(summary.interval_width.home/2, digits=2)) goals")
println("  Away: ±$(round(summary.interval_width.away/2, digits=2)) goals")
println("\nPrediction Accuracy:")
println("  MAE Home: $(round(summary.mae.home_ft, digits=3))")
println("  MAE Away: $(round(summary.mae.away_ft, digits=3))")
println("  Brier Score: $(round(summary.brier_score, digits=4))")



###### Prob prediction 
#
#
#= 


=#

samples_posterior = extract_posterior_samples(first_split.ft, result.mapping) 
a1 = predict_match_full_posterior("st-johnstone", "aberdeen", samples_posterior)
#=
(xG_home = (mean = 0.9480469762937985, median = 0.942288306341767, q025 = 0.7529260029527372, q975 = 1.16507291793253, iqr = 0.14230634448470525, samples = [1.1444606741775138, 0.9689955194160086, 0.8846580919973533, 0.9838067930186961, 0.8792711218750776, 0.99455
21467031542, 0.9172306984869868, 0.9154857130949108, 0.8268297478875968, 0.9594402795475448  …  0.85782598189629, 0.858130465507161, 1.1853734422459796, 0.8800273290611926, 1.103099632522014, 0.8137335811232622, 0.983812451024455, 0.855513478357179, 1.033140030943
4013, 1.13326595462812]), xG_away = (mean = 1.0320811255924496, median = 1.028560906760572, q025 = 0.8391875354626425, q975 = 1.2567892182077585, iqr = 0.1409458204722147, samples = [1.1451103675469645, 0.8841688060047301, 1.0625897607709232, 1.0653540806282684, 0
.9427063071938329, 1.1484930149610302, 0.9073320033757071, 1.1788216381152394, 0.9521528173555472, 1.1326820149775296  …  0.8561272637817396, 0.9681624777578948, 0.9395013545175992, 1.1098520177694167, 0.929073345479304, 1.0859082354373395, 1.0135645175277526, 0.9
393594062830555, 1.111085360626344, 0.9958657755706163]), prob_home_win = 0.3231043464162192, prob_draw = 0.3097096068115074, prob_away_win = 0.3671860144771423, score_distribution = [0.13964099370677 0.14251961243406902 … 6.762723646140517e-7 7.579628278072614e-8
; 0.13080238255546633 0.13351732910743394 … 6.332209494471241e-7 7.094400919119031e-8; … ; 3.3318829270832244e-7 3.399524443933857e-7 … 1.5826902994746672e-12 1.764063534424161e-13; 3.4813958250781776e-8 3.5507918807614214e-8 … 1.6451106038923148e-13 1.83183719697
74577e-14])
=#

posterior_samples = extract_posterior_samples(first_split.ht, result.mapping)








#=
round one
home_team            away_team             home_score  away_score 
String31             String31              Int64       Int64      
──────────────────────────────────────────────────────────────────
st-johnstone         aberdeen                       1           2 
heart-of-midlothian  rangers                        0           0 
dundee-united        dundee-fc                      2           2 
celtic               kilmarnock                     4           0 
st-mirren            hibernian                      3           0 
motherwell           ross-county                    0           0 
partick-thistle      greenock-morton                0           0 
livingston           dunfermline-athletic           2           0 
hamilton-academical  ayr-united                     0           2 
falkirk-fc           queens-park-fc                 2           1 
airdrieonians        raith-rovers                   1           0 
=#
