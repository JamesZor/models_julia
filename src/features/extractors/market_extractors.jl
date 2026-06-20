# src/features/extractors/market_extractors.jl

function add_feature!(F_data::Dict, config::AbstractMarketFeatureConfig, ordered_ids, team_map::Dict, ds::Data.DataStore)
    id_set = Set(ordered_ids)
    filtered_odds = subset(ds.odds, :match_id => ByRow(in(id_set)))
    odds_by_match = groupby(filtered_odds, :match_id)
    n_matches = length(odds_by_match)
    
    # We use Any/NamedTuple because keys vary based on config
    thread_results = Vector{Tuple{Int, NamedTuple}}(undef, n_matches)
    
    @threads for i in 1:n_matches
        match_df = odds_by_match[i]
        res = fit_market_implied_parameters(match_df, config)
        params = extract_parameters(config, res.minimizer)
        thread_results[i] = (res.match_id, params)
    end
    
    market_map = Dict{Int, NamedTuple}(r[1] => r[2] for r in thread_results)
    
    # Dynamically unroll keys and apply NaN padding
    dummy_params = extract_parameters(config, get_initial_guess(config))
    for key in keys(dummy_params)
        dict_key = Symbol("flat_market_", key)
        F_data[dict_key] = [haskey(market_map, id) ? market_map[id][key] : NaN for id in ordered_ids]
    end
end
