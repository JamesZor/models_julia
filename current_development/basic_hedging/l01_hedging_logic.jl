
using BayesianFootball
using DataFrames
using Statistics
using Dates

function analyze_hedging_mvp(ds; market_line=1.5, exit_minute=30.0)
    # 1. Identify Under selection
    selection = Symbol("under_", Int(market_line*10))
    
    # 2. Extract in-play odds for the market
    ou_odds = filter(r -> r.market_name == "OverUnder" && r.market_line == market_line && r.selection == selection, ds.betfair_odds)
    
    # 3. Get kickoff prices (minutes_to_kickoff around 0)
    kickoff_odds = filter(r -> -5.0 <= r.minutes_to_kickoff <= 2.0, ou_odds)
    # Take the one closest to 0 for each match
    kickoff_odds = combine(groupby(kickoff_odds, :match_id)) do df
        df[argmin(abs.(df.minutes_to_kickoff)), :]
    end
    rename!(kickoff_odds, :traded_price => :price_t0, :minutes_to_kickoff => :min_t0)
    
    # 4. Get exit prices (minutes_to_kickoff around exit_minute + 1.64)
    # We use the FH offset from the technical brief
    target_exit_min = exit_minute + 1.64
    exit_odds = filter(r -> (target_exit_min - 5.0) <= r.minutes_to_kickoff <= (target_exit_min + 5.0), ou_odds)
    exit_odds = combine(groupby(exit_odds, :match_id)) do df
        df[argmin(abs.(df.minutes_to_kickoff .- target_exit_min)), :]
    end
    rename!(exit_odds, :traded_price => :price_texit, :minutes_to_kickoff => :min_texit)
    
    # 5. Join
    df_merged = innerjoin(kickoff_odds, exit_odds, on=:match_id, makeunique=true)
    
    # 6. Check for goals before exit
    # We need to consider the lag. If a goal happens at minute 28, it might reflect in prices at 29.64.
    # So we check incidents for goals with time < exit_minute
    goals = filter(r -> r.incident_type == "goal", ds.incidents)
    
    # Calculate goals before exit for each match
    match_goals = combine(groupby(goals, :match_id)) do df
        (goals_before_exit = sum(df.time .< exit_minute),)
    end
    
    df_final = leftjoin(df_merged, match_goals, on=:match_id)
    df_final.goals_before_exit = coalesce.(df_final.goals_before_exit, 0)
    
    # 7. Calculate Returns
    # Return = (BackPrice / LayPrice) - 1
    # Note: Laying at Betfair involves commission, but for MVP we use (P1/P2 - 1)
    df_final.roi = (df_final.price_t0 ./ df_final.price_texit) .- 1.0
    
    return df_final
end

# Load data and run
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())
res_u15 = analyze_hedging_mvp(ds; market_line=1.5, exit_minute=30.0)

println("--- Under 1.5 Hedging at Minute 30 (Ireland) ---")
println("Total Matches: ", nrow(res_u15))
println("Matches with 0 goals at 30': ", sum(res_u15.goals_before_exit .== 0))
println("Average ROI (all): ", mean(res_u15.roi))
println("Average ROI (0 goals): ", mean(res_u15.roi[res_u15.goals_before_exit .== 0]))
println("Average ROI (>0 goals): ", mean(res_u15.roi[res_u15.goals_before_exit .> 0]))

# Check Under 0.5
res_u05 = analyze_hedging_mvp(ds; market_line=0.5, exit_minute=20.0)
println("\n--- Under 0.5 Hedging at Minute 20 (Ireland) ---")
println("Total Matches: ", nrow(res_u05))
println("Matches with 0 goals at 20': ", sum(res_u05.goals_before_exit .== 0))
println("Average ROI (all): ", mean(res_u05.roi))
println("Average ROI (0 goals): ", mean(res_u05.roi[res_u05.goals_before_exit .== 0]))
