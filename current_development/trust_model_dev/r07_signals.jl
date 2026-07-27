# current_development/trust_model_dev/r07_signals.jl

using Revise
using BayesianFootball
using DataFrames
using Distributions
using Base.Threads
using ProgressMeter

const Data = BayesianFootball.Data
const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions
const Evaluation = BayesianFootball.Evaluation
const Markets = BayesianFootball.Data.Markets

# ====================================================================
# 1. Monkey Patching Predictions & Evaluation
# ====================================================================

# 1a. Allow model_inference to just pass-through a PPD
Predictions.model_inference(ppd::Predictions.PPD) = ppd

# 1b. Create an evaluation entry point that takes a dictionary of PPDs
function Evaluation.evaluate_experiments(
    metrics::Vector{<:Evaluation.AbstractScoringRule}, 
    ppd_dict::Dict{String, Tuple{Predictions.PPD, Experiments.ExperimentResults}}, 
    ds::Data.DataStore
)
    master_rows = []

    println("============================================================")
    println(" 🚀 Running PPD Batch Evaluation...")
    println("============================================================")

    for (model_name, (ppd, mock_exp)) in ppd_dict
        print("Evaluating: $(model_name) ... ")

        # Start the row with the model name
        combined_row = (; model = model_name)
        success = true

        for metric in metrics
            try
                # We pass `ppd` as the latents_raw. Our monkey-patch above will just return it.
                result = Evaluation.compute_metric(metric, mock_exp, ds, ppd)
                flat_row = Evaluation.to_dataframe_row(mock_exp, metric, result)
                
                clean_row = Base.structdiff(flat_row, (; model="")) 
                combined_row = merge(combined_row, clean_row)
            catch e
                success = false
                @warn "Error evaluating $(typeof(metric)) for $model_name: $e"
            end
        end
        
        if success
            push!(master_rows, combined_row)
            println("✅ Done")
        else
            println("❌ Failed")
        end
    end

    master_df = DataFrame(master_rows)
    if nrow(master_df) > 0
        sort!(master_df, :model)
    end
    return master_df
end

# ====================================================================
# 2. Extracting Layer 1 & Layer 1+2 PPDs
# ====================================================================

function construct_layered_ppds(ds::Data.DataStore, lr::LayeredInferenceResults; K_samples::Int=2000)
    println("Constructing Layer 1 PPD...")
    
    # 1. Get raw L1 PPD restricted to Trust Markets
    l1_latents = Experiments.extract_oos_predictions(ds, lr.l1_results)
    l1_ppd_raw = Predictions.model_inference(l1_latents; market_config = lr.l2_results.config.market_config)
    
    # 2. Identify the target match IDs for splits > warmup
    splitter = lr.l1_results.config.splitter
    splits = Data.create_id_boundaries(ds, splitter)
    valid_split_indices = (lr.l2_warmup_splits + 1):length(splits)
    
    valid_match_ids = Int[]
    for s in valid_split_indices
        append!(valid_match_ids, splits[s][1].target_match_ids)
    end
    
    # 3. Filter L1 PPD to only valid matches
    l1_df = subset(l1_ppd_raw.df, :match_id => ByRow(in(valid_match_ids)))
    l1_ppd = Predictions.PPD(l1_df, l1_ppd_raw.model, l1_ppd_raw.config)
    
    println("Constructing Layer 1 + Layer 2 PPD...")
    
    # Reconstruct the global_team_map that was used for training L2
    all_global_teams = unique(vcat(ds.matches.home_team, ds.matches.away_team))
    global_team_map = Dict(name => i for (i, name) in enumerate(sort(all_global_teams)))
    
    # We need match -> home_team mapping
    matches_subset = subset(ds.matches, :match_id => ByRow(in(valid_match_ids)))
    match_home_teams = Dict(r.match_id => r.home_team for r in eachrow(matches_subset))
    
    # Build market mapping dict for Trust Model indices
    # TRUST_MARKETS index matches w0[m] index
    market_map = Dict{Tuple{String, Float64}, Int}()
    for (m_idx, market) in enumerate(lr.l2_results.config.market_config.markets)
        m_name = Data.market_group(market)
        m_line = Data.market_line(market)
        market_map[(m_name, m_line)] = m_idx
    end
    
    # Get bookie data for looking up scalar probabilities
    # We always use ds.odds because ds.betfair_odds is a raw tick log that requires the `summarize_odds` pipeline
    bookie_df = ds.odds
    bookie_lookup = select(bookie_df, :match_id, :market_name, :market_line, :selection, :prob_fair_close)
    unique!(bookie_lookup, [:match_id, :market_name, :market_line, :selection])
    
    # Build match -> split mapping for quick lookup
    match_to_split = Dict{Int, Int}()
    for s in valid_split_indices
        for mid in splits[s][1].target_match_ids
            match_to_split[mid] = s
        end
    end
    
    # Pre-extract all L2 chain data into memory so threads don't touch DataFrames
    split_chains = Dict{Int, Any}()
    n_markets = length(lr.l2_results.config.market_config.markets)
    for s in valid_split_indices
        chain_df = DataFrame(lr.l2_results.chains[s])
        split_chains[s] = (
            chain_len = nrow(chain_df),
            w0 = [chain_df[!, Symbol("w0[$m]")] for m in 1:n_markets],
            sigma = chain_df[!, Symbol("σ_team")],
            team_z = [chain_df[!, Symbol("team_z[$t]")] for t in 1:length(global_team_map)]
        )
    end
    
    # Join L1 with Bookie to get P_bookie on the same row
    # If a match doesn't have odds, it is dropped and not evaluated.
    l1_with_bookie = innerjoin(
        l1_df, 
        bookie_lookup, 
        on = [:match_id, :market_name, :market_line, :selection]
    )
    
    # Filter valid matches and markets
    valid_mask = Bool[]
    market_idx_col = Int[]
    for row in eachrow(l1_with_bookie)
        key = (row.market_name, row.market_line)
        if haskey(market_map, key) && haskey(match_to_split, row.match_id)
            push!(valid_mask, true)
            push!(market_idx_col, market_map[key])
        else
            push!(valid_mask, false)
            push!(market_idx_col, 0)
        end
    end
    
    l1_with_bookie = l1_with_bookie[valid_mask, :]
    market_indices = market_idx_col[valid_mask]
    
    N_rows = nrow(l1_with_bookie)
    println("Blending L1 & L2 with $(Threads.nthreads()) threads across $N_rows match-selections...")
    
    # Pre-allocate output arrays
    v_match_ids = Vector{Int}(undef, N_rows)
    v_market_names = Vector{String}(undef, N_rows)
    v_market_lines = Vector{Float64}(undef, N_rows)
    v_selections = Vector{Symbol}(undef, N_rows)
    v_dists = Vector{Vector{Float64}}(undef, N_rows)
    
    # Extract columns to avoid DataFrame row allocation in the loop
    col_match_id = l1_with_bookie.match_id
    col_market_name = l1_with_bookie.market_name
    col_market_line = l1_with_bookie.market_line
    col_selection = l1_with_bookie.selection
    col_distribution = l1_with_bookie.distribution
    col_prob_bookie = l1_with_bookie.prob_fair_close
    
    Threads.@threads for i in 1:N_rows
        mid = col_match_id[i]
        market_name = col_market_name[i]
        market_line = col_market_line[i]
        selection = col_selection[i]
        
        p_l1_chain = col_distribution[i]
        p_bookie = col_prob_bookie[i]
        
        home_team = match_home_teams[mid]
        t_idx = global_team_map[home_team]
        m_idx = market_indices[i]
        
        s = match_to_split[mid]
        chain_data = split_chains[s]
        
        # Sample indices randomly (with replacement) to target length K
        l1_indices = rand(1:length(p_l1_chain), K_samples)
        l2_indices = rand(1:chain_data.chain_len, K_samples)
        
        blended_dist = Vector{Float64}(undef, K_samples)
        
        for k in 1:K_samples
            idx1 = l1_indices[k]
            idx2 = l2_indices[k]
            
            # Trust Model equation
            w0_val = chain_data.w0[m_idx][idx2]
            sigma_val = chain_data.sigma[idx2]
            team_z_val = chain_data.team_z[t_idx][idx2]
            
            eta = w0_val + sigma_val * team_z_val
            w_l = 1.0 / (1.0 + exp(-eta)) # logistic
            
            p_l1 = p_l1_chain[idx1]
            
            # Blend!
            blended_dist[k] = w_l * p_l1 + (1.0 - w_l) * p_bookie
        end
        
        v_match_ids[i] = mid
        v_market_names[i] = market_name
        v_market_lines[i] = market_line
        v_selections[i] = selection
        v_dists[i] = blended_dist
    end
    
    l1_l2_df = DataFrame(
        :match_id => v_match_ids,
        :market_name => v_market_names,
        :market_line => v_market_lines,
        :selection => v_selections,
        :distribution => v_dists
    )
    
    l1_l2_ppd = Predictions.PPD(l1_l2_df, l1_ppd_raw.model, l1_ppd_raw.config)
    
    return l1_ppd, l1_l2_ppd
end

# ====================================================================
# 3. Main Execution 
# ====================================================================

# NOTE: This assumes `ds` and `layered_results` are already loaded in your REPL session!

println("Building Layered PPDs for Evaluation...")

# This generates our two PPD objects! 
# We sample 2000 items from both distributions to create a uniform length joint MC
l1_ppd, l1_l2_ppd = construct_layered_ppds(ds, layered_results, K_samples=2000)

# We use the L1 experiment config to pass as the "mock" ExperimentResults
mock_exp = layered_results.l1_results

# Build the PPD dictionary for batch_runner
ppd_dict = Dict(
    "Layer 1 (No Trust)" => (l1_ppd, mock_exp),
    "Layer 1 + Layer 2 (Trust Blend)" => (l1_l2_ppd, mock_exp)
)

# Define the metrics we want to test
metrics_to_test = [
    Evaluation.GLMEdge(), 
    Evaluation.LogLoss(), 
    Evaluation.LPD()
]

# Run the evaluation!
println("\n", "="^60, "\n📈 Evaluating Layer 1 vs Layer 1+2 (OOS)\n", "="^60)
data_of_metrics = Evaluation.evaluate_experiments(metrics_to_test, ppd_dict, ds)

println("\n", "="^60, "\n📈 GLM Edge — src grid vs li_*/dp_*\n", "="^60)
Evaluation.display_summary_metric(data_of_metrics, :glmedge)

#=
--- GLM Edge Summary ---
2×4 DataFrame
 Row │ model                            glmedge_intercept_coef  glmedge_spread_fair_coef  glmedge_spread_fair_p_value 
     │ String                           Float64                 Float64                   Float64                     
─────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ Layer 1 (No Trust)                             -2.49502                   1.88837                  0.000663767
   2 │ Layer 1 + Layer 2 (Trust Blend)                -2.50611                  10.2794                   7.91065e-7
=#

Evaluation.display_summary_metric(data_of_metrics, :logloss)

#=
--- LogLoss Summary (Lower Diff is Better) ---
2×4 DataFrame
 Row │ model                            logloss_overall_model_ll  logloss_overall_market_ll  logloss_overall_diff_ll 
     │ String                           Float64                   Float64                    Float64                 
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ Layer 1 (No Trust)                               0.55051                    0.547877               0.00263349
   2 │ Layer 1 + Layer 2 (Trust Blend)                  0.546085                   0.547877              -0.00179151
=#

Evaluation.display_summary_metric(data_of_metrics, :lpd)

#=
--- LPD Summary (Higher Diff is Better; Higher ELPD is Better) ---
2×9 DataFrame
 Row │ model                            lpd_overall_model_lpd  lpd_overall_model_std  lpd_overall_model_skewness  lpd_overall_model_kurtosis  lpd_overall_market_lpd  lpd_overall_diff_lpd  lpd_overall_elpd  lpd_overall_n_obs 
     │ String                           Float64                Float64                Float64                     Float64                     Float64                 Float64               Float64           Int64             
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │ Layer 1 (No Trust)                           -0.55051                0.451146                    -1.55164                     2.82647               -0.547877           -0.00263349          -2419.49               4395
   2 │ Layer 1 + Layer 2 (Trust Blend)              -0.546085               0.452507                    -1.58357                     2.98114               -0.547877            0.00179151          -2400.04               4395
=#

