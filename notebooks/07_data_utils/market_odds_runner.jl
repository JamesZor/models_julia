include("/home/james/bet_project/models_julia/notebooks/07_data_utils/market_odds_setup.jl")

# Load data store
data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)

match_id = rand(data_store.matches.match_id)

odds = get_game_line_odds(data_store, match_id)
