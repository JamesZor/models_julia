using Pkg; Pkg.activate("/home/james/bet_project/BayesianFootball")
using BayesianFootball
using DataFrames
using Dates
using HypothesisTests
using GLM
using Statistics

ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())
matches = ds.matches
matches = dropmissing(matches, [:home_score, :away_score, :start_timestamp])
matches.total_goals = matches.home_score .+ matches.away_score
matches.month = month.(matches.start_timestamp)

monthly_stats = combine(groupby(matches, :month), 
    :total_goals => mean => :avg_goals,
    nrow => :count
)
println("=== Monthly Goal Averages ===")
println(monthly_stats)

months_data = [matches[matches.month .== m, :total_goals] for m in unique(matches.month)]
if length(months_data) > 1
    kw_test = KruskalWallisTest(months_data...)
    println("\n=== Kruskal-Wallis Test ===")
    println(kw_test)
end

matches.month_cat = categorical(matches.month)
poisson_model = glm(@formula(total_goals ~ month_cat), matches, Poisson(), LogLink())
println("\n=== Poisson Regression (Month Effect) ===")
println(poisson_model)
