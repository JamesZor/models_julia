# Workflow for analyzing predictions from experiment results

using BayesianFootball
using DataFrames
using Statistics
using Plots

"""
    analyze_experiment_predictions(result, data_store, cv_config)

Complete workflow for analyzing predictions from an experiment.
"""
function analyze_experiment_predictions(
    result::ExperimentResult,
    data_store::DataStore,
    cv_config::TimeSeriesSplitsConfig
)
    println("="^60)
    println("Prediction Analysis")
    println("="^60)
    
    # Step 1: Generate predictions for all rounds
    println("\n1. Generating predictions for all rounds...")
    predictions = predict_all_rounds(result, data_store, cv_config)
    println("   Generated $(nrow(predictions)) match predictions")
    
    # Step 2: Overall evaluation
    println("\n2. Overall Performance Metrics:")
    overall_metrics = evaluate_predictions(predictions)
    
    if !isnothing(overall_metrics)
        println("\n   Full-Time Performance:")
        println("   - MAE (Home): $(round(overall_metrics.full_time.mae_home, digits=3))")
        println("   - MAE (Away): $(round(overall_metrics.full_time.mae_away, digits=3))")
        println("   - RMSE (Home): $(round(overall_metrics.full_time.rmse_home, digits=3))")
        println("   - RMSE (Away): $(round(overall_metrics.full_time.rmse_away, digits=3))")
        println("   - MAE (Total Goals): $(round(overall_metrics.total_goals.mae_ft, digits=3))")
        
        println("\n   Half-Time Performance:")
        println("   - MAE (Home): $(round(overall_metrics.half_time.mae_home, digits=3))")
        println("   - MAE (Away): $(round(overall_metrics.half_time.mae_away, digits=3))")
    end
    
    # Step 3: Round-by-round evaluation
    println("\n3. Performance by Round:")
    round_metrics = evaluate_by_round(predictions)
    
    if nrow(round_metrics) > 0
        for row in eachrow(round_metrics)
            println("   Round $(row.round): MAE(FT) = $(round(row.mae_total_ft, digits=3)), " *
                   "RMSE(FT) = $(round(row.rmse_total_ft, digits=3)) " *
                   "($(row.n_matches) matches)")
        end
    end
    
    # Step 4: Classification performance
    println("\n4. Match Outcome Classification:")
    class_metrics = classification_metrics(predictions)
    
    if !isnothing(class_metrics)
        println("   Accuracy: $(round(class_metrics.accuracy * 100, digits=1))% " *
               "($(class_metrics.n_correct)/$(class_metrics.n_total))")
        
        println("\n   Confusion Matrix:")
        println("   Actual\\Pred |   H   |   D   |   A   |")
        println("   ------------|-------|-------|-------|")
        for actual in ["H", "D", "A"]
            print("        $actual      |")
            for pred in ["H", "D", "A"]
                count = class_metrics.confusion["$(actual)_pred_$(pred)"]
                print("  $(lpad(count, 3))  |")
            end
            println()
        end
    end
    
    return predictions, overall_metrics, round_metrics, class_metrics
end

"""
    plot_prediction_errors(predictions_df)

Create visualization of prediction errors.
"""
function plot_prediction_errors(predictions_df::DataFrame)
    # Calculate errors
    errors_home_ft = predictions_df.actual_home_ft - predictions_df.pred_xG_home_ft
    errors_away_ft = predictions_df.actual_away_ft - predictions_df.pred_xG_away_ft
    
    # Create plots
    p1 = histogram(errors_home_ft, 
                   title="Home Goals Prediction Error (FT)",
                   xlabel="Actual - Predicted",
                   ylabel="Frequency",
                   bins=20,
                   label="Home")
    
    p2 = histogram(errors_away_ft,
                   title="Away Goals Prediction Error (FT)",
                   xlabel="Actual - Predicted",
                   ylabel="Frequency",
                   bins=20,
                   label="Away",
                   color=:red)
    
    # Scatter plot of predictions vs actuals
    p3 = scatter(predictions_df.pred_xG_home_ft, predictions_df.actual_home_ft,
                 xlabel="Predicted xG",
                 ylabel="Actual Goals",
                 title="Home Goals: Predicted vs Actual",
                 label="Matches",
                 markersize=3,
                 alpha=0.6)
    plot!(p3, [0, 6], [0, 6], label="Perfect Prediction", linestyle=:dash)
    
    p4 = scatter(predictions_df.pred_xG_away_ft, predictions_df.actual_away_ft,
                 xlabel="Predicted xG",
                 ylabel="Actual Goals",
                 title="Away Goals: Predicted vs Actual",
                 label="Matches",
                 markersize=3,
                 alpha=0.6,
                 color=:red)
    plot!(p4, [0, 6], [0, 6], label="Perfect Prediction", linestyle=:dash)
    
    # Combine plots
    plot(p1, p2, p3, p4, layout=(2,2), size=(800, 600))
end

"""
    extract_team_strength_evolution(result, team_name)

Track how a team's parameters evolve across CV splits.
"""
function extract_team_strength_evolution(result::ExperimentResult, team_name::String)
    evolution = DataFrame()
    
    for (i, chains) in enumerate(result.chains_sequence)
        params = extract_team_parameters(chains, result.mapping)
        team_params = filter(row -> row.team == team_name, params)
        
        if nrow(team_params) > 0
            push!(evolution, (
                split = i,
                round_info = chains.round_info,
                attack_ht = team_params.attack_ht[1],
                defense_ht = team_params.defense_ht[1],
                attack_ft = team_params.attack_ft[1],
                defense_ft = team_params.defense_ft[1]
            ))
        end
    end
    
    return evolution
end

# Main execution
function main()
    # Assuming you have your result from the experiment
    # result = load_experiment("experiments/your_experiment.jld2")[1]
    # data_store = DataStore(DataFiles("path/to/data"))
    # cv_config = TimeSeriesSplitsConfig(...)
    
    # If you're running this right after your experiment:
    println("Analyzing predictions from experiment...")
    
    # You would call it like:
    # predictions, overall, by_round, classification = analyze_experiment_predictions(
    #     result, data_store, cv_config
    # )
    
    # Then you can save the predictions
    # CSV.write("predictions.csv", predictions)
    
    println("\nAnalysis complete!")
end

# Export functions for use in REPL
export analyze_experiment_predictions
export plot_prediction_errors
export extract_team_strength_evolution
