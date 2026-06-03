using Pkg; Pkg.activate(".")
using BayesianFootball
using DataFrames, Statistics, Dates, GLM, HypothesisTests

println("Loading data...")
ds = Data.load_datastore_cached(Data.Ireland())
df = ds.matches

df.month = month.(df.date)
df.total_goals = df.home_score .+ df.away_score

gdf = combine(groupby(df, :month), :total_goals => mean => :avg_goals, nrow => :count)
println("\nEDA: Average goals by month:")
display(gdf)

kw_test = KruskalWallisTest(df.total_goals, df.month)
println("\nKruskal-Wallis Test:")
display(kw_test)

df.month_cat = categorical(df.month)
glm_model = glm(@formula(total_goals ~ month_cat), df, Poisson(), LogLink())
println("\nPoisson GLM:")
display(glm_model)
