# src/evaluation/batch_runner.jl

export evaluate_experiments, display_summary_metric

"""
    evaluate_experiments(metrics::Vector{<:AbstractScoringRule}, experiments::Vector{ExperimentResults}, ds::DataStore)

Runs a batch of evaluation metrics against a list of loaded experiments.
Returns a single, wide DataFrame containing all flattened metrics for each experiment.
"""
function evaluate_experiments(metrics::Vector{<:AbstractScoringRule}, experiments::Vector{ExperimentResults}, ds::DataStore)
    master_rows = []

    println("============================================================")
    println(" 🚀 Running Batch Evaluation...")
    println("============================================================")

    for (i, exp) in enumerate(experiments)
        model_name = exp.config.name
        print("[$i/$(length(experiments))] Evaluating: $(model_name) ... ")

        # 1. Extract Latents ONCE for the entire batch of metrics for this experiment
        latents = Experiments.extract_oos_predictions(ds, exp)

        # Start the row with the model name
        combined_row = (; model = model_name)
        success = true

        for metric in metrics
            try
                result = compute_metric(metric, exp, ds, latents)
                flat_row = to_dataframe_row(exp, metric, result)
                
                # Remove the duplicate 'model' key from the flat_row before merging
                # because we already initialized it in combined_row
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
            println("❌ Failed (Partial or complete failure)")
            # Optionally, you can still push partial results if you want
            # push!(master_rows, combined_row) 
        end
    end

    master_df = DataFrame(master_rows)
    if nrow(master_df) > 0
        sort!(master_df, :model)
    end
    return master_df
end

# Convenience method for a single metric
function evaluate_experiments(metric::AbstractScoringRule, experiments::Vector{ExperimentResults}, ds::DataStore)
    return evaluate_experiments([metric], experiments, ds)
end

"""
    display_summary_metric(df::DataFrame, metric_family::Symbol)

Displays a curated subset of columns from the master evaluation DataFrame based on the metric family.
"""
function display_summary_metric(df::DataFrame, metric_family::Symbol)
    if metric_family == :rqr
        cols = [:model, :rqr_all_mean, :rqr_all_std, :rqr_all_skewness, :rqr_all_kurtosis, :rqr_all_shapiro_w, :rqr_all_shapiro_p]
        println("\n--- RQR Summary ---")
    elseif metric_family == :logloss
        cols = [:model, :logloss_overall_model_ll, :logloss_overall_market_ll, :logloss_overall_diff_ll]
        # Also surface any per-selection logloss diff columns (e.g. logloss_over_25_overall_diff_ll)
        per_line = filter(n -> occursin(r"^logloss_.+_overall_diff_ll$", n) && n != "logloss_overall_diff_ll", names(df))
        append!(cols, Symbol.(per_line))
        println("\n--- LogLoss Summary (Lower Diff is Better) ---")
    elseif metric_family == :glmedge
        # Depending on if they passed GLMEdge() or GLMEdge(:home) it might be glmedge_all_...
        # We use regex to grab the most important intercept and spread fair columns
        cols = [:model]
        append!(cols, Symbol.(filter(n -> occursin(r"glmedge.*_intercept_coef", n), names(df))))
        append!(cols, Symbol.(filter(n -> occursin(r"glmedge.*_spread_fair_coef", n), names(df))))
        append!(cols, Symbol.(filter(n -> occursin(r"glmedge.*_spread_fair_p_value", n), names(df))))
        println("\n--- GLM Edge Summary ---")
    elseif metric_family == :crps
        cols = [:model, :crps_home_mean, :crps_away_mean, :crps_all_mean]
        println("\n--- CRPS Summary (Lower is Better) ---")
    else
        println("Unknown metric family: \$metric_family")
        return
    end
    
    # Ensure all requested columns exist in the DataFrame
    existing_cols = filter(c -> string(c) in names(df), cols)
    
    if length(existing_cols) <= 1 # Only :model exists
        @warn "No data found for metric family '\$metric_family' in the provided DataFrame."
        return
    end

    display(select(df, existing_cols...))
end
