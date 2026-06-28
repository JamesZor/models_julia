# src/evaluation/metrics_methods/lpd.jl

export LPD, LPDResult, LPDComponent

# --- The Trigger ---
struct LPD <: AbstractScoringRule
    selections::Vector{Symbol}
end

LPD(selection::Symbol) = LPD([selection])
LPD() = LPD(Symbol[])

# --- The Components ---
struct LPDComponent <: AbstractMetricComponent
    model_lpd::Float64    # Mean per-match LPD (higher is better)
    market_lpd::Float64   # Mean per-match LPD from market fair odds
    diff_lpd::Float64     # Model - Market (positive = model is better)
    elpd::Float64         # Total ELPD: sum of all per-match LPDs
    n_obs::Int
end

struct LPDResult <: AbstractEvaluationResult
    overall::LPDComponent
end

# --- Translator Mappings ---
function get_metric_method_name(::LPDResult)::String
    return "lpd"
end

function get_metric_method_name(metric::LPD)::String
    if isempty(metric.selections)
        return "lpd_all"
    else
        return "lpd_" * join(String.(metric.selections), "_")
    end
end

# ==============================================================================
# MATH & HELPERS
# ==============================================================================

"""
    calc_lpd_samples(samples::Vector{Float64}, y::Float64)

Bayesian Log Predictive Density from raw posterior probability samples.

Uses log-sum-exp for numerical stability:
  LPD = log( (1/S) Σ_s p(y | θ^s) )

For binary y ∈ {0.0, 1.0}:
  p(y | θ^s) = samples[s]        if y = 1
               (1 - samples[s])  if y = 0

Higher is better. This is the negative of binary log loss computed
over the full PPD rather than a collapsed point probability.
"""
function calc_lpd_samples(samples::Vector{Float64}, y::Float64)
    if y == 1.0
        log_liks = log.(clamp.(samples, 1e-15, 1.0 - 1e-15))
    else
        log_liks = log.(clamp.(1.0 .- samples, 1e-15, 1.0 - 1e-15))
    end
    lmax = maximum(log_liks)
    return lmax + log(mean(exp.(log_liks .- lmax)))
end

"""
    calc_lpd_scalar(p::Float64, y::Float64)

LPD for a scalar probability (e.g. market fair-odds implied probability).
"""
function calc_lpd_scalar(p::Float64, y::Float64)
    p_clamped = clamp(p, 1e-15, 1.0 - 1e-15)
    return y == 1.0 ? log(p_clamped) : log(1.0 - p_clamped)
end

# ==============================================================================
# MAIN COMPUTE METHOD
# ==============================================================================

function compute_metric(metric::LPD, exp::ExperimentResults, ds::DataStore, latents_raw::Any)::LPDResult

    # 1. Extract PPD — keep the raw distribution vector, do NOT collapse to mean
    ppd = Predictions.model_inference(latents_raw)
    model_features = select(ppd.df, :match_id, :market_name, :market_line, :selection, :distribution)

    # 2. Merge with market odds (need prob_fair_close and is_winner)
    analysis_df = innerjoin(
        ds.odds,
        model_features,
        on = [:match_id, :market_name, :market_line, :selection]
    )

    dropmissing!(analysis_df, [:prob_fair_close, :is_winner])

    # 3. Filter by selections
    if !isempty(metric.selections)
        filter!(:selection => s -> s in metric.selections, analysis_df)
    end

    analysis_df.Y = Float64.(analysis_df.is_winner)

    # 4. Compute per-match LPD arrays
    # Model: log( mean_s p(y | θ^s) ) using full posterior sample vector
    lpd_model_array  = calc_lpd_samples.(analysis_df.distribution, analysis_df.Y)
    # Market: log( p_fair(y) ) — scalar market probability as baseline
    lpd_market_array = calc_lpd_scalar.(analysis_df.prob_fair_close, analysis_df.Y)

    # 5. Aggregate
    mean_model  = mean(lpd_model_array)
    mean_market = mean(lpd_market_array)
    diff        = mean_model - mean_market   # positive = model beats market
    elpd        = sum(lpd_model_array)
    n_obs       = nrow(analysis_df)

    return LPDResult(
        LPDComponent(mean_model, mean_market, diff, elpd, n_obs)
    )
end
