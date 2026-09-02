using JSON3
using Redis

redis_conn = RedisConnection(host="100.124.38.117", port=6379)
include("/home/james/bet_project/BayesianFootball/current_development/match_day_inference/src/live_betting.jl")

mappings = get_live_market_mappings(redis_conn)
println("Length of markets: ", length(mappings.markets))
for (k, v) in mappings.markets
    if occursin("bohemian", k[1]) || occursin("bohemian", k[2])
        println("Found match in markets! ", k, " -> ", v)
    end
end
