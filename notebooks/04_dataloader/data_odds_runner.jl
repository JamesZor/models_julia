# Load data store
data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)

match_id = rand(data_store.matches.match_id)


odds = get_game_line_odds(data_store, match_id)



odds_row = filter(row -> row.sofa_match_id == match_id && row.minutes == 0, data_store.odds)
odds_row = filter(row -> row.sofa_match_id == match_id && row.minutes > 0, data_store.odds)

a1 =odds_row[:, ["0_0", "1_0", "2_0", "3_0", "0_1", "1_1", "2_1", "3_1", "0_2", "1_2", "2_2", "3_2", "0_3", "1_3", "2_3", "3_3",  "any_other_away", "any_other_draw", "any_other_home"]]
a = odds_row[:, ["ht_0_0", "ht_1_0", "ht_2_0", "ht_0_1", "ht_1_1", "ht_2_1", "ht_0_2", "ht_1_2", "ht_2_2", "ht_any_unquoted"]] 




1 .+ a[2,:]
odds

for row in eachrow(a1)
  b = [ v for v in values(row)]
  println(sum( 1 ./ b)) 
end 

