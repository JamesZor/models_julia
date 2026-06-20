# src/evaluation/metrics_methods/logloss.jl 

export LogLoss, LogLossResult, LogLossComponent

# --- The Trigger ---
struct LogLoss <: AbstractScoringRule 
    selections::Vector{Symbol}
end

# Convenience constructors
LogLoss(selection::Symbol) = LogLoss([selection])
LogLoss() = LogLoss(Symbol[])

# --- The Components ---
struct LogLossComponent <: AbstractMetricComponent
    model_ll::Float64
    market_ll::Float64
    diff_ll::Float64    # Model - Market (Negative means YOUR model is better!)
    n_obs::Int
end

struct LogLossResult <: AbstractEvaluationResult
    overall::LogLossComponent
end

# --- Translator Mappings ---
function get_metric_method_name(::LogLossResult)::String
    return "logloss"
end

function get_metric_method_name(metric::LogLoss)::String
    if isempty(metric.selections)
        return "logloss_all"
    else
        return "logloss_" * join(String.(metric.selections), "_")
    end
end

# ==============================================================================
# MATH & HELPERS
# ==============================================================================

"""
    calc_logloss(p::Float64, y::Float64)

Calculates the binary cross-entropy (logloss) for a single prediction.
Clamps probabilities to prevent log(0) errors. Lower is better.
"""
function calc_logloss(p::Float64, y::Float64)
    # Clamp to avoid -Inf from log(0)
    p_clamped = clamp(p, 1e-15, 1.0 - 1e-15)
    return -(y * log(p_clamped) + (1.0 - y) * log(1.0 - p_clamped))
end

# ==============================================================================
# MAIN COMPUTE METHOD
# ==============================================================================

function compute_metric(metric::LogLoss, exp::ExperimentResults, ds::DataStore, latents_raw::Any)::LogLossResult
    
    # 3. Model Inference (Get the model's probabilities)
    ppd = Predictions.model_inference(latents_raw)
    model_features = transform(ppd.df, :distribution => ByRow(mean) => :prob_model)
    select!(model_features, :match_id, :market_name, :market_line, :selection, :prob_model)

    # 4. Merge Model Probs with Market Probs
    analysis_df = innerjoin(
        # REVIEW:
        ds.odds, # market_data.df,
        model_features,
        on = [:match_id, :market_name, :market_line, :selection]
    )
    
    # 5. Clean up missing odds/outcomes
    dropmissing!(analysis_df, [:prob_fair_close, :is_winner])
    
    # FILTER BY SELECTIONS
    if !isempty(metric.selections)
        filter!(:selection => s -> s in metric.selections, analysis_df)
    end
    
    # Convert outcomes to Float64 (1.0 for win, 0.0 for loss)
    analysis_df.Y = Float64.(analysis_df.is_winner)
    
    # 6. Calculate LogLoss Arrays
    ll_model_array  = calc_logloss.(analysis_df.prob_model, analysis_df.Y)
    ll_market_array = calc_logloss.(analysis_df.prob_fair_close, analysis_df.Y)
    
    # 7. Summarize
    mean_model  = mean(ll_model_array)
    mean_market = mean(ll_market_array)
    diff        = mean_model - mean_market
    n_obs       = nrow(analysis_df)

    # 8. Pack and Return
    return LogLossResult(
        LogLossComponent(mean_model, mean_market, diff, n_obs)
    )
end
