using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using DataFrames
using Dates

# 1. Load Data
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

# 2. Print matches schema and first few rows to understand structure
println("Columns in ds.matches: ", names(ds.matches))
println(first(ds.matches, 5))
