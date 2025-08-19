##################################################
# Runner for data transformer 
##################################################

data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)

mapping_functions = MappingFunctions(create_list_mapping) 
mapping = MappedData(data_store, mapping_functions)


# TEST:
splits = time_series_splits(data_store.matches , ["20/21", "21/22", "22/23", "23/24"], "24/25")
for (i, (train_data, round_info)) in enumerate(splits)
    println("Split $i ($round_info): $(nrow(train_data)) rows")
    println( first(sort(train_data[:, [:season, :home_team, :away_team, :match_date, :home_score, :away_score]], :match_date, rev=true)))
end


