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
    model_inference(latents::AbstractPosteriorLatents, model; market_config, verbose)

Price dense typed posterior latents without converting them back to a legacy
`DataFrame`. The model is retained in the returned `PPD`; score-grid dispatch is
selected entirely by the latent container type.
"""
function model_inference(latents::AbstractPosteriorLatents,
                         model::AbstractFootballModel;
                         market_config = DEFAULT_MARKET_CONFIG,
                         verbose::Bool = false)
    isnothing(market_config) && error("market_config must be provided")

    k = hash((objectid(latents), objectid(model), hash(market_config.markets), :typed))
    haskey(_PPD_CACHE, k) && return _PPD_CACHE[k]

    markets = collect(market_config.markets)
    nm = n_matches(latents)
    verbose && println("Running typed inference on $nm matches...")

    # One reusable score workspace per Julia worker; market result vectors remain
    # fixture-owned because they are retained in the returned PPD.
    worker_slots = Base.Threads.maxthreadid()
    workspaces = [GridWorkspace() for _ in 1:worker_slots]
    grids = [alloc_score_grid(latents) for _ in 1:worker_slots]
    smile_buffers = latents isa SmileLatents ?
        [alloc_smile_buffers(latents) for _ in 1:worker_slots] : nothing

    results_vec = Vector{Dict{String, Dict{Symbol, Vector{Float64}}}}(undef, nm)
    @threads :static for i in 1:nm
        worker = threadid()
        ws = workspaces[worker]
        S = grids[worker]
        compute_score_grid!(S, ws, latents, i)

        target = if latents isa SmileLatents
            buffers = smile_buffers[worker]
            fill_smile_buffers!(buffers.λ_tot, buffers.φ, latents, i)
            SmileScoreGrid(S, buffers.λ_tot, buffers.φ, latents.strikes)
        else
            S
        end

        fixture_results = Dict{String, Dict{Symbol, Vector{Float64}}}()
        for market in markets
            probabilities = if market isa Union{Market1X2, MarketBTTS, MarketOverUnder}
                price_market(target, market)
            else
                # Typed kernels currently cover 1X2, BTTS and O/U. Preserve the
                # complete legacy MarketConfig contract for all other markets.
                compute_market_probs(ScoreMatrix(S), market)
            end
            fixture_results[string(market)] = probabilities
        end
        results_vec[i] = fixture_results
    end

    match_ids = Int[]
    market_names = String[]
    market_lines = Float64[]
    selections = Symbol[]
    distributions = Vector{Float64}[]

    ids = latent_match_ids(latents)
    for (i, fixture_results) in enumerate(results_vec)
        for market in markets
            probabilities = fixture_results[string(market)]
            for (selection, distribution) in probabilities
                push!(match_ids, ids[i])
                push!(market_names, market_group(market))
                push!(market_lines, market_line(market))
                push!(selections, selection)
                push!(distributions, distribution)
            end
        end
    end

    ppd = PPD(DataFrame(
        :match_id => match_ids,
        :market_name => market_names,
        :market_line => market_lines,
        :selection => selections,
        :distribution => distributions,
    ), model, market_config)
    _PPD_CACHE[k] = ppd
    return ppd
end

function model_inference(latents::AbstractPosteriorLatents;
                         model = nothing,
                         market_config = DEFAULT_MARKET_CONFIG,
                         verbose::Bool = false)
    model isa AbstractFootballModel || error(
        "model_inference(::AbstractPosteriorLatents) needs `model = ...` because " *
        "typed latent containers deliberately store posterior values only. " *
        "Alternatively call `model_inference(latents, model; ...)`.")
    return model_inference(latents, model;
                           market_config = market_config, verbose = verbose)
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

