using Redis
using JSON3

redis_conn = RedisConnection(host="100.124.38.117", port=6379)
meta_dict = Redis.hgetall(redis_conn, "live_market_meta")

mapping_lookup = Dict{Tuple{String, String, String}, String}()
for (market_id, raw_json) in meta_dict
    meta = JSON3.read(raw_json)
    home_slug = haskey(meta, :home_slug) && meta.home_slug !== nothing ? string(meta.home_slug) : ""
    away_slug = haskey(meta, :away_slug) && meta.away_slug !== nothing ? string(meta.away_slug) : ""
    market_type = haskey(meta, :market_type) ? string(meta.market_type) : ""

    if !isempty(home_slug) && !isempty(away_slug) && !isempty(market_type)
        mapping_lookup[(home_slug, away_slug, market_type)] = string(market_id)
    end
end

println("mapping lookup size: ", length(mapping_lookup))
if haskey(mapping_lookup, ("bohemian", "dundalk-fc", "MATCH_ODDS"))
    println("Found bohemians MATCH_ODDS!")
else
    println("Did NOT find bohemians MATCH_ODDS!")
    println("Looking for keys with bohemian:")
    for k in keys(mapping_lookup)
        if occursin("bohemian", k[1]) || occursin("bohemian", k[2])
            println("Found key: ", k)
        end
    end
end
