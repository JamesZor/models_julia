using BayesianFootball
using DataFrames
include("loader.jl")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland(); force=true)
