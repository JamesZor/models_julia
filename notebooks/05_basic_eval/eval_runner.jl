using BayesianFootball

##################################################
# Basic set up - loading 
##################################################
### Load the saved data for the basic maher model 
r1 = load_experiment("/home/james/bet_project/models_julia/experiments/maher_basic_test_20250820_231427.jld2")
# extract the results and config 
result = r1[1] 
experiment_config = r1[2]
mapping = result.mapping
# After running your experiment, you have:
# - result (ExperimentResult)
# - data_store (DataStore)
data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)

# Get the target season 
target_matches = filter(row -> row.season == experiment_config.cv_config.target_season, data_store.matches)
# compute the chains based off the models chains
matches_results = predict_target_season(target_matches, result, mapping)
keys(matches_results)
##################################################
# Eval
##################################################
# De-constructed 
match_id = 12476901 
match_id = 12476824
m1::MatchPredict = matches_results[match_id]

# 1x2 lines 
ft = get_1x2_odds(m1.ft)
ht = get_1x2_odds(m1.ht)
p_1x2 = get_1x2_odds(m1)

sort(filter(row -> row.sofa_match_id==match_id && row.minutes ≥ 0 && row.minutes ≤ 5, data_store.odds)[:, [:minutes, :ht_home, :ht_draw, :ht_away, :home, :draw, :away]], :minutes)

m_score = match_score_times(match_id, data_store)
# get match results
#=
5×4 DataFrame
 Row │ time   is_home  home_score  away_score 
     │ Int64  Bool?    Float64?    Float64?   
─────┼────────────────────────────────────────
   1 │     6    false         0.0         1.0
   2 │    25    false         0.0         2.0
   3 │    59     true         1.0         2.0
   4 │    81     true         2.0         2.0
   5 │    90    false         2.0         3.0
=#


model_odds = get_1x2_odds(matches_results[match_id])
market_odds = get_market_odds(match_id, data_store, 0:5, m_score)
c = compare_odds(model_odds, market_odds)
print_comparison_table(c, m_score, 0:120)

