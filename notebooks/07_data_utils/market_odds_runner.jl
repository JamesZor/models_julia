include("/home/james/bet_project/models_julia/notebooks/07_data_utils/market_odds_setup.jl")

# Load data store
data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)

match_id = rand(data_store.matches.match_id)

odds = get_game_line_odds(data_store, match_id)


### odds processing
#
config = default_config()
processed_odds = get_processed_game_line_odds(data_store, match_id)

filter(row -> row.sofa_match_id==match_id && row.minutes > 0 && row.minutes < 10, data_store.odds)

odds_quality_check(data_store, match_id)


timeline = display_odds_timeline(data_store, match_id, minutes_range=(-5, 90))


test_configurations(data_store, match_id)


analyze_match_odds(data_store, match_id)


# market group 
# Compare raw vs processed for specific market
raw_odds = get_game_line_odds(data_store, match_id)
processed_odds = get_processed_game_line_odds(data_store, match_id)
df = display_market_group(raw_odds, processed_odds, "ft_1x2")

odds_quality_check(data_store, match_id)
