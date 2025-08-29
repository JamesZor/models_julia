
using BayesianFootball
r1 = load_experiment("/home/james/bet_project/models_julia/experiments/maher_basic_test_20250820_231427.jld2")
result = r1[1]
experiment_config = r1[2]
mapping = result.mapping

# Load data store
data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)
target_matches = filter(row -> row.season == experiment_config.cv_config.target_season, data_store.matches)

# get random 
m_id = rand(target_matches.match_id)

# run get results for one
mr = BayesianFootball.get_match_results(data_store, m_id) 

# run for all target 
mrs = BayesianFootball.process_matches_results(data_store, target_matches)


##################################################
# Extract active minutes from incidents
##################################################
ma = get_line_active_minutes(data_store, m_id)
mas = process_matches_active_minutes(data_store, target_matches)

#  added to BayesianFootball 

