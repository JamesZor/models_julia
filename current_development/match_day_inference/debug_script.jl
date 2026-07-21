using BayesianFootball
using DataFrames
include("loader.jl")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())
todays_matches = fetch_todays_matches(ds)
println("TODAYS MATCHES:")
show(todays_matches)
println()

save_dir = "./data/match_daya/"
saved_fiels = BayesianFootball.Experiments.list_experiments(save_dir, data_dir="")
expr = BayesianFootball.Experiments.load_experiment(saved_fiels, 1)

json_lineups_dir = "./data/lineups"
ppd = compute_todays_matches_pdds(ds, expr, todays_matches, json_lineups_dir)

println("PPD.DF match_ids:")
show(unique(ppd.df.match_id))
println()

println("PPD.DF subset for 15238109:")
show(subset(ppd.df, :match_id => ByRow(==(15238109)), :market_name => ByRow(==("1X2"))))
println()
