data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24/")
data_store = DataStore( data_files)

optimize_datastore!(data_store)
size_before = Base.summarysize(data_store)

optimize_datastore!(data_store)

size_after = Base.summarysize(data_store)

println("Size before: $(round(size_before / 1024^2, digits=2)) MB")
println("Size after:  $(round(size_after / 1024^2, digits=2)) MB")
println("Saved: $(round((size_before - size_after) / 1024^2, digits=2)) MB")
