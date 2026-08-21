# src/predictions/inference.jl

using DataFrames
using Base.Threads
using ProgressMeter
using ..Experiments: LatentStates
# Import the interface functions we need
using ..Data: AbstractMarket, market_group, market_line, outcomes
using ..Data.Markets: DEFAULT_MARKET_CONFIG 

export model_inference

# 1. The Kernel (Process one match)
# ------------------------------------------------------------------
function predict_row(model, row, markets)
    params = extract_params(model, row)
    S = compute_score_matrix(model, params)
    
    # We use string(market) as a temporary unique key for the dictionary
    results = Dict{String, Dict{Symbol, Vector{Float64}}}()
    
    for market in markets
        # compute_market_probs now returns Dict{Symbol, Vector}
        results[string(market)] = compute_market_probs(S, market)
    end
    
    return results
end

const _PPD_CACHE = Dict{UInt64, Any}()

# 2. The Orchestrator
# ------------------------------------------------------------------
function model_inference(latents::LatentStates; market_config=DEFAULT_MARKET_CONFIG, verbose::Bool=false)
    if isnothing(market_config)
        error("market_config must be provided")
    end

    k = hash((objectid(latents.df), objectid(latents.model), hash(market_config.markets)))
    if haskey(_PPD_CACHE, k)
        return _PPD_CACHE[k]
    end

    df = latents.df
    model = latents.model
    markets = collect(market_config.markets)
    
    n_matches = nrow(df)
    verbose && println("Running Inference on $(n_matches) matches...")
    
    # A. Run Predictions (Threaded)
    results_vec = Vector{Dict}(undef, n_matches)
    rows = collect(eachrow(df))
    
    @threads for i in 1:n_matches
        results_vec[i] = predict_row(model, rows[i], markets)
    end
    
    # B. Flatten into Rich PPD Structure
    # Columns: match_id | market_name | market_line | selection | distribution
    
    v_match_ids = Int[]
    v_market_names = String[]
    v_market_lines = Float64[]
    v_selections = Symbol[]
    v_dists = Vector{Float64}[]
    
    # Iterate through the results and reconstruct the metadata
    for (i, res_dict) in enumerate(results_vec)
        # Convert match_id to String to ensure consistency
        mid = rows[i].match_id
        
        # Iterate over the *original* markets list to get metadata safely
        for market in markets
            m_key = string(market)
            
            # Check if this market was successfully computed
            if haskey(res_dict, m_key)
                outcome_dict = res_dict[m_key]
                
                # Get standardized metadata
                m_name = market_group(market) # e.g., "1X2"
                m_line = market_line(market)  # e.g., 0.0
                
                for (sel_symbol, dist) in outcome_dict
                    push!(v_match_ids, mid)
                    push!(v_market_names, m_name)
                    push!(v_market_lines, m_line)
                    push!(v_selections, sel_symbol) # e.g., :home
                    push!(v_dists, dist)
                end
            end
        end
    end
    
    ppd_df = DataFrame(
        :match_id => v_match_ids,
        :market_name => v_market_names,
        :market_line => v_market_lines,
        :selection => v_selections,
        :distribution => v_dists
    )
    
    res = PPD(ppd_df, model, market_config)
    _PPD_CACHE[k] = res
    return res
end



"""
Function to process an experiment results struct and datastore to get the 
model inference prediction of the matches ( Out of sample predictions). 
returns the predictive posterior distribution (PPD) for the 
specifed markets (has default)
"""
function model_inference(ds::Data.DataStore, exp::Experiments.ExperimentResults)::Predictions.PPD 
  return  model_inference(
                Experiments.extract_oos_predictions(ds, exp)
         )
end

