data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats/")


println(data_files.base_dir) 
println(data_files.match)
println(data_files.odds) 


data_store = DataStore(data_files) 

transformer = DataTransformers(create_list_mapping)
mapping = MappedData(data_store, transformer)
features = FeaturesBasicMaherModel(data_store, mapping)



