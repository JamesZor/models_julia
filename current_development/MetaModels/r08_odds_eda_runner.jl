# current_development/MetaModels/r08_odds_eda_runner.jl
#
# Runner to evaluate Betfair vs Sofascore market coverage and pricing differences.
# Run with: julia --project current_development/MetaModels/r08_odds_eda_runner.jl

using BayesianFootball
using DataFrames

include("./current_development/MetaModels/l07_odds_eda.jl")

println("=======================================================")
println("  LOADING BASE DATASTORE (SOFASCORE)")
println("=======================================================")
ds_raw = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())
sofa_odds = ds_raw.odds

println("\n=======================================================")
println("  COMPUTING BETFAIR EXCHANGE SUMMARY")
println("=======================================================")
# We take the closing window from 10 mins before kickoff to exactly kickoff
betfair_odds_summary = BayesianFootball.Data.summarize_betfair_market(
    ds_raw,
    open_window=(-100000.0, -10.0),
    close_window=(-60.0, 0.0)
)

TARGET_SELECTIONS = [:under_25, :over_25]
total_games = nrow(ds_raw.matches)

println("\nRunning Coverage Analysis for: ", TARGET_SELECTIONS)
println("Total Games in DB: ", total_games)
coverage_df, joined_df, stats_df = compare_odds_coverage(sofa_odds, betfair_odds_summary, TARGET_SELECTIONS; total_matches=total_games)

# Optional: You can inspect joined_df in the REPL directly 
# to see the exact odds differences for specific matches!
coverage_df





#=
betfair_odds_summary = BayesianFootball.Data.summarize_betfair_market(
    ds_raw,
    open_window=(-100000.0, -10.0),
    close_window=(-60.0, 0.0)
)

Row │ selection  avg_sofa_odds  avg_bf_odds  mean_odds_diff  median_odds_diff  std_odds_diff  mean_prob_diff 
     │ Symbol     Float64        Float64      Float64         Float64           Float64        Float64        
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ over_25           2.0821       2.2153          0.1332            0.1233         0.0758         -0.028
   2 │ under_25          1.7862       1.8909          0.1047            0.0961         0.0696         -0.0304)

julia> # Optional: You can inspect joined_df in the REPL directly 
       # to see the exact odds differences for specific matches!
       coverage_df
2×6 DataFrame
 Row │ selection  sofa_count  bf_count  total_matches  bf_true_coverage_pct  sofa_true_coverage_pct 
     │ Symbol     Int64       Int64     Int64          Float64               Float64                
─────┼──────────────────────────────────────────────────────────────────────────────────────────────
   1 │ over_25           955       961            984                 97.66                   97.05
   2 │ under_25          955       961            984                 97.66                   97.05




#=
betfair_odds_summary = BayesianFootball.Data.summarize_betfair_market(
    ds_raw,
    open_window=(-100000.0, -10.0),
    close_window=(-25.0, 0.0)
)
=#

#=
1751 rows omitted, 2×7 DataFrame
 Row │ selection  avg_sofa_odds  avg_bf_odds  mean_odds_diff  median_odds_diff  std_odds_diff  mean_prob_diff 
     │ Symbol     Float64        Float64      Float64         Float64           Float64        Float64        
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ over_25           2.0809       2.2134          0.1325            0.1225         0.0786         -0.0278
   2 │ under_25          1.7877       1.8924          0.1048            0.0967         0.0726         -0.0302)

julia> # Optional: You can inspect joined_df in the REPL directly 
       # to see the exact odds differences for specific matches!
       coverage_df
2×6 DataFrame
 Row │ selection  sofa_count  bf_count  total_matches  bf_true_coverage_pct  sofa_true_coverage_pct 
     │ Symbol     Int64       Int64     Int64          Float64               Float64                
─────┼──────────────────────────────────────────────────────────────────────────────────────────────
   1 │ over_25           955       935            984                 95.02                   97.05
   2 │ under_25          955       935            984                 95.02                   97.05
=#





#=
betfair_odds_summary = BayesianFootball.Data.summarize_betfair_market(
    ds_raw,
    open_window=(-100000.0, -10.0),
    close_window=(-10.0, 0.0)
)
=#

#=
Row │ selection  avg_sofa_odds  avg_bf_odds  mean_odds_diff  median_odds_diff  std_odds_diff  mean_prob_diff 
     │ Symbol     Float64        Float64      Float64         Float64           Float64        Float64        
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ over_25           2.0703       2.2018          0.1314            0.1181         0.0817         -0.0279
   2 │ under_25          1.7969       1.9017          0.1048            0.0966         0.0702         -0.0299)

julia> # Optional: You can inspect joined_df in the REPL directly 
       # to see the exact odds differences for specific matches!
       coverage_df
2×6 DataFrame
 Row │ selection  sofa_count  bf_count  total_matches  bf_true_coverage_pct  sofa_true_coverage_pct 
     │ Symbol     Int64       Int64     Int64          Float64               Float64                
─────┼──────────────────────────────────────────────────────────────────────────────────────────────
   1 │ over_25           955       846            984                 85.98                   97.05
   2 │ under_25          955       846            984                 85.98                   97.05
=#

=#

