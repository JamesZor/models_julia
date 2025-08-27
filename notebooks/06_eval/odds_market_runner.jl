println("Loading experiment data...")
r1 = load_experiment("/home/james/bet_project/models_julia/experiments/maher_basic_test_20250820_231427.jld2")
result = r1[1]
experiment_config = r1[2]
mapping = result.mapping

# Load data store
data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)

df = data_store.incidents
match_id = 12476901 
# Get target season matches
target_matches = filter(row -> row.season == experiment_config.cv_config.target_season, data_store.matches)

# Let's get the odds for a specific match
match_id_to_process = 12476901 
original_odds = get_game_line_odds(data_store.odds, match_id_to_process)

# Now, let's process them using the new function, aiming for a 105% book (1.05)
rescaled_odds_obj = impute_and_rescale_odds(data_store, match_id_to_process, target_sum=1.005)



# display 
function display_odd_change(o_odds, re_odds)

# You can now inspect the rescaled odds
println("--- Original FT 1x2 Odds ---")
println("Home: ", original_odds.ft.home)
println("Draw: ", original_odds.ft.draw)
println("Away: ", original_odds.ft.away)
original_probs_sum = sum(1 ./ filter(!isnothing, [original_odds.ft.home, original_odds.ft.draw, original_odds.ft.away]))
@printf "Original Implied Probability Sum: %.4f\n\n" original_probs_sum

println("--- Rescaled FT 1x2 Odds (Target Sum: 1.05) ---")
println("Home: ", rescaled_odds_obj.ft.home)
println("Draw: ", rescaled_odds_obj.ft.draw)
println("Away: ", rescaled_odds_obj.ft.away)
rescaled_probs_sum = sum(1 ./ filter(!isnothing, [rescaled_odds_obj.ft.home, rescaled_odds_obj.ft.draw, rescaled_odds_obj.ft.away]))
@printf "Rescaled Implied Probability Sum: %.4f\n" rescaled_probs_sum

mr = get_match_results(data_store.incidents, match_id)
