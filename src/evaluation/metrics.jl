# src/evaluation/metrics.jl
## Model evaluation metrics
using SpecialFunctions

using Statistics

"""
    poisson_deviance(y_true, y_pred)

Calculate Poisson deviance for count data.
"""
function poisson_deviance(y_true::Vector, y_pred::Vector)
    # Avoid log(0) issues
    y_pred_safe = max.(y_pred, 1e-8)
    
    deviance = 2 * sum(
        y_true .* log.(y_true ./ y_pred_safe) - (y_true - y_pred)
    )
    
    return deviance
end

"""
    mean_absolute_error(y_true, y_pred)

Calculate mean absolute error.
"""
function mean_absolute_error(y_true::Vector, y_pred::Vector)
    return mean(abs.(y_true - y_pred))
end

"""
    root_mean_squared_error(y_true, y_pred)

Calculate root mean squared error.
"""
function root_mean_squared_error(y_true::Vector, y_pred::Vector)
    return sqrt(mean((y_true - y_pred).^2))
end

"""
    log_loss_goals(y_true, y_pred)

Calculate log loss for goals using Poisson distribution.
"""
function log_loss_goals(y_true::Vector, y_pred::Vector)
    # Ensure predictions are positive
    y_pred_safe = max.(y_pred, 1e-8)
    
    # Calculate Poisson log likelihood
    log_probs = y_true .* log.(y_pred_safe) - y_pred_safe - loggamma.(y_true .+ 1) # TODO: loggamma is in SpecialFunctions
    
    return -mean(log_probs)
end

"""
    evaluate_predictions(predictions_df)

Evaluate predictions using multiple metrics.
"""
function evaluate_predictions(predictions_df::Union{DataFrame, SubDataFrame})
    # Remove any rows with missing predictions
    pred_complete = dropmissing(predictions_df)
    
    if nrow(pred_complete) == 0
        return nothing
    end
    
    # Calculate metrics for half-time
    ht_metrics = (
        mae_home = mean_absolute_error(
            pred_complete.actual_home_ht,
            pred_complete.pred_xG_home_ht
        ),
        mae_away = mean_absolute_error(
            pred_complete.actual_away_ht,
            pred_complete.pred_xG_away_ht
        ),
        rmse_home = root_mean_squared_error(
            pred_complete.actual_home_ht,
            pred_complete.pred_xG_home_ht
        ),
        rmse_away = root_mean_squared_error(
            pred_complete.actual_away_ht,
            pred_complete.pred_xG_away_ht
        ),
        log_loss_home = log_loss_goals(
            pred_complete.actual_home_ht,
            pred_complete.pred_xG_home_ht
        ),
        log_loss_away = log_loss_goals(
            pred_complete.actual_away_ht,
            pred_complete.pred_xG_away_ht
        )
    )
    
    # Calculate metrics for full-time
    ft_metrics = (
        mae_home = mean_absolute_error(
            pred_complete.actual_home_ft,
            pred_complete.pred_xG_home_ft
        ),
        mae_away = mean_absolute_error(
            pred_complete.actual_away_ft,
            pred_complete.pred_xG_away_ft
        ),
        rmse_home = root_mean_squared_error(
            pred_complete.actual_home_ft,
            pred_complete.pred_xG_home_ft
        ),
        rmse_away = root_mean_squared_error(
            pred_complete.actual_away_ft,
            pred_complete.pred_xG_away_ft
        ),
        log_loss_home = log_loss_goals(
            pred_complete.actual_home_ft,
            pred_complete.pred_xG_home_ft
        ),
        log_loss_away = log_loss_goals(
            pred_complete.actual_away_ft,
            pred_complete.pred_xG_away_ft
        )
    )
    
    # Calculate total goals metrics
    total_pred_ht = pred_complete.pred_xG_home_ht + pred_complete.pred_xG_away_ht
    total_actual_ht = pred_complete.actual_home_ht + pred_complete.actual_away_ht
    
    total_pred_ft = pred_complete.pred_xG_home_ft + pred_complete.pred_xG_away_ft
    total_actual_ft = pred_complete.actual_home_ft + pred_complete.actual_away_ft
    
    total_metrics = (
        mae_ht = mean_absolute_error(total_actual_ht, total_pred_ht),
        mae_ft = mean_absolute_error(total_actual_ft, total_pred_ft),
        rmse_ht = root_mean_squared_error(total_actual_ht, total_pred_ht),
        rmse_ft = root_mean_squared_error(total_actual_ft, total_pred_ft)
    )
    
    return (
        n_matches = nrow(pred_complete),
        half_time = ht_metrics,
        full_time = ft_metrics,
        total_goals = total_metrics
    )
end

"""
    evaluate_by_round(predictions_df)

Evaluate predictions grouped by round.
"""
function evaluate_by_round(predictions_df::DataFrame)
    results = DataFrame()
    
    for round_group in groupby(predictions_df, :round)
        round_num = round_group.round[1]
        metrics = evaluate_predictions(round_group)
        
        if !isnothing(metrics)
            push!(results, (
                round = round_num,
                n_matches = metrics.n_matches,
                mae_home_ht = metrics.half_time.mae_home,
                mae_away_ht = metrics.half_time.mae_away,
                rmse_home_ht = metrics.half_time.rmse_home,
                rmse_away_ht = metrics.half_time.rmse_away,
                mae_total_ht = metrics.total_goals.mae_ht,
                rmse_total_ht = metrics.total_goals.rmse_ht,
                mae_home_ft = metrics.full_time.mae_home,
                mae_away_ft = metrics.full_time.mae_away,
                rmse_home_ft = metrics.full_time.rmse_home,
                rmse_away_ft = metrics.full_time.rmse_away,
                mae_total_ft = metrics.total_goals.mae_ft,
                rmse_total_ft = metrics.total_goals.rmse_ft
            ))
        end
    end
    
    return results
end

"""
    classification_metrics(predictions_df)

Calculate classification metrics for match outcomes (Win/Draw/Loss).
"""
function classification_metrics(predictions_df::DataFrame)
    pred_complete = dropmissing(predictions_df)
    
    if nrow(pred_complete) == 0
        return nothing
    end
    
    # Calculate probabilities for each match
    outcomes_pred = String[]
    outcomes_actual = String[]
    
    for row in eachrow(pred_complete)
        # Predicted outcome based on xG
        probs = calculate_match_probabilities(row.pred_xG_home_ft, row.pred_xG_away_ft)
        
        # Most likely outcome
        if probs.home_win > probs.draw && probs.home_win > probs.away_win
            push!(outcomes_pred, "H")
        elseif probs.draw > probs.away_win
            push!(outcomes_pred, "D")
        else
            push!(outcomes_pred, "A")
        end
        
        # Actual outcome
        if row.actual_home_ft > row.actual_away_ft
            push!(outcomes_actual, "H")
        elseif row.actual_home_ft == row.actual_away_ft
            push!(outcomes_actual, "D")
        else
            push!(outcomes_actual, "A")
        end
    end
    
    # Calculate accuracy
    accuracy = mean(outcomes_pred .== outcomes_actual)
    
    # Confusion matrix
    confusion = Dict()
    for actual in ["H", "D", "A"]
        for pred in ["H", "D", "A"]
            key = "$(actual)_pred_$(pred)"
            confusion[key] = sum((outcomes_actual .== actual) .& (outcomes_pred .== pred))
        end
    end
    
    return (
        accuracy = accuracy,
        n_correct = sum(outcomes_pred .== outcomes_actual),
        n_total = length(outcomes_pred),
        confusion = confusion,
        predicted = outcomes_pred,
        actual = outcomes_actual
    )
end
