using BayesianFootball
using DataFrames
include("loader.jl")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland(); force=true)
todays_matches = fetch_todays_matches(ds)
println("TODAYS MATCHES:")
show(todays_matches)
