# include("/home/james/bet_project/models_julia/notebooks/03_predictions/refactor_maher.jl")
using BayesianFootball
r1 = load_experiment("/home/james/bet_project/models_julia/experiments/maher_basic_test_20250820_231427.jld2")

result = r1[1]
experiment_config = r1[2]
mapping = result.mapping

# Load data store
data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)



target_matches = filter(row -> row.season == experiment_config.cv_config.target_season, data_store.matches)


matches_results = predict_target_season(target_matches, result, mapping)

m_id = rand(keys(matches_results))

m1 = matches_results[m_id] 

