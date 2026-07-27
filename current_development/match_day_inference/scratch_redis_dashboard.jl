using DataFrames
using Dates
using Distributions

module Predictions
    struct PPD
        df::DataFrame
    end
end

using .Predictions
include("/home/james/bet_project/BayesianFootball/current_development/match_day_inference/src/live_betting.jl")

redis_conn = RedisConnection(host="100.124.38.117", port=6379)

df_matches = DataFrame(
    match_id = [15238109],
    home_team = ["bohemian"],
    away_team = ["dundalk-fc"]
)

df_ppd = DataFrame(
    match_id = [15238109, 15238109, 15238109],
    market_name = ["1X2", "1X2", "1X2"],
    market_line = [0.0, 0.0, 0.0],
    selection = [:home, :away, :draw],
    distribution = [Normal(0.5, 0.1), Normal(0.3, 0.1), Normal(0.2, 0.1)]
)

ppd = Predictions.PPD(df_ppd)

print_live_betting_dashboard(ppd, redis_conn, df_matches)
