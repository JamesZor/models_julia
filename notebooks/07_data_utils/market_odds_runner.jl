include("/home/james/bet_project/models_julia/notebooks/07_data_utils/market_odds_setup.jl")

# Load data store
data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)

match_id = rand(data_store.matches.match_id)


# Simple method to get raw odds
odds = get_game_line_odds(data_store, match_id)


### odds processing
config = default_config()
processed_odds = get_processed_game_line_odds(data_store, match_id)

filter(row -> row.sofa_match_id==match_id && row.minutes > 0 && row.minutes < 10, data_store.odds)

odds_quality_check(data_store, match_id)


timeline = display_odds_timeline(data_store, match_id, minutes_range=(-5, 15))


test_configurations(data_store, match_id)


analyze_match_odds(data_store, match_id)


# market group 
# Compare raw vs processed for specific market
raw_odds = get_game_line_odds(data_store, match_id)
processed_odds = get_processed_game_line_odds(data_store, match_id)
df = display_market_group(raw_odds, processed_odds, "ft_1x2")
df = display_market_group(raw_odds, processed_odds, "ht_1x2")
df = display_market_group(raw_odds, processed_odds, "ft_ou_05")
df = display_market_group(raw_odds, processed_odds, "ft_ou_15")
df = display_market_group(raw_odds, processed_odds, "ft_ou_25")


df = display_market_group(raw_odds, processed_odds, "ht_ou_05")
df = display_market_group(raw_odds, processed_odds, "ht_ou_15")
df = display_market_group(raw_odds, processed_odds, "ht_ou_25")

df = display_market_group(raw_odds, processed_odds, "ft_cs")  # Full-time correct scores
df = display_market_group(raw_odds, processed_odds, "ht_cs")  # Half-time correct scores

df = display_market_group(raw_odds, processed_odds, "btts")  # Half-time correct scores
odds_quality_check(data_store, match_id)


## Check odds
match_id = rand(data_store.matches.match_id)
sort(filter(row -> row.sofa_match_id==match_id && row.minutes > 0 && row.minutes < 10, data_store.odds)[:, [:minutes, :home, :draw, :away, :ht_home, :ht_draw, :ht_away]], :minutes)

# cs ft
sort(filter(row -> row.sofa_match_id==match_id && row.minutes > 0 && row.minutes < 10, data_store.odds)[:, ["minutes", "0_0", "0_1", "0_2", "0_3", "1_1", "1_2", "1_3","2_1", "2_2", "2_3","3_1", "3_2", "3_3","1_0", "2_0", "3_0", "any_other_home", "any_other_away", "any_other_draw"]], :minutes)

sort(filter(row -> row.sofa_match_id==match_id && row.minutes > 0 && row.minutes < 10, data_store.odds)[:, ["minutes", "0_0", "0_1", "0_2", "0_3", "1_1", "1_2", "1_3","2_1", "2_2", "2_3","3_1", "3_2", "3_3","1_0", "2_0", "3_0", "any_other_home", "any_other_away", "any_other_draw"]], :minutes)
sort(filter(row -> row.sofa_match_id==match_id && row.minutes > 0 && row.minutes < 10, data_store.odds)[:, vcat(["minutes"], MARKET_GROUPS["ht_cs"])], :minutes)
processed_odds = get_processed_game_line_odds(data_store, match_id)




##### proceeing a few 
r1 = load_experiment("/home/james/bet_project/models_julia/experiments/maher_basic_test_20250820_231427.jld2")
result = r1[1]
experiment_config = r1[2]
mapping = result.mapping

target_matches = filter(row -> row.season == experiment_config.cv_config.target_season, data_store.matches)


matches_odds = process_matches_odds(data_store, target_matches)

m_id = rand(target_matches.match_id)

matches_odds[m_id]



### added to bayesain football 

