# analysis_logic.jl
using Statistics, DataFrames, Plots, Printf

# Make sure the structs are available
include("odds_market_setup.jl")

# ##################################################
# 1. MODIFIED KELLY CALCULATION
# ##################################################

"""
Calculates the Bayesian Kelly fraction chain.
Based on the original `calculate_bayesian_kelly` function[cite: 3, 15].
"""
function calculate_bayesian_kelly_chain(
    model_probs::Vector{Float64},
    market_odds::Union{Float64, Nothing}
)
    # Handle invalid inputs
    if isnothing(market_odds) || market_odds <= 1.0 || isempty(model_probs)
        return Float64[] # Return an empty chain
    end

    kelly_fractions = Float64[]
    for p in model_probs
        edge = p * market_odds - 1.0
        if edge > 0
            f = edge / (market_odds - 1.0)
            push!(kelly_fractions, f)
        else
            push!(kelly_fractions, 0.0)
        end
    end
    
    # Return the full chain of fractions
    return kelly_fractions
end

# ##################################################
# 2. PROFIT CALCULATION & ITERATION
# ##################################################

"""
Calculates profit from a Kelly chain given a specific quantile.
"""
function calculate_profit_from_chain(
    chain::Vector{Float64},
    odds::Union{Float64, Nothing},
    won::Union{Bool, Nothing},
    quantile_level::Float64,
    fractional_kelly::Float64 = 1.0
)
    if isempty(chain) || isnothing(odds) || isnothing(won)
        return (profit=0.0, stake=0.0)
    end

    # Determine the stake using the specified quantile of the Kelly distribution
    stake = quantile(chain, quantile_level) * fractional_kelly

    if stake <= 0
        return (profit=0.0, stake=0.0)
    end

    profit = won ? stake * (odds - 1) : -stake
    return (profit=profit, stake=stake)
end

"""
Iterates through all target matches and calculates aggregated profit and stake
for a given quantile.
"""
function run_analysis_for_quantile(
    target_matches::DataFrame,
    matches_results_map::Dict,
    data_store::DataStore;
    quantile_level::Float64 = 0.5,
    fractional_kelly::Float64 = 1.0,
    target_sum::Float64 = 1.005
)
    # Initialize accumulators for profit and stake
    market_groups = [
        "FT_1x2", "HT_1x2", "FT_OU", "HT_OU", "FT_CS", "HT_CS"
    ]
    results = Dict(group => (profit=0.0, stake=0.0) for group in market_groups)
    
    cumulative_wealth = [1.0] # Start with a bankroll of 1.0

    for match_row in eachrow(target_matches)
        match_id = match_row.match_id
        
        # Get necessary data for the match
        model_predictions = get(matches_results_map, match_id, nothing)
        if isnothing(model_predictions) continue end
        
        match_results = get_match_results(data_store.incidents, match_id)
        norm_odds = impute_and_rescale_odds(data_store, match_id; target_sum=target_sum)
        raw_odds = get_game_line_odds(data_store.odds, match_id)
        if isnothing(norm_odds) || isnothing(raw_odds) continue end
        
        match_total_profit = 0.0

        # --- Process Full-Time Markets ---
        
        # FT 1x2
        p_home = calculate_profit_from_chain(calculate_bayesian_kelly_chain(model_predictions.ft.home, norm_odds.ft.home), norm_odds.ft.home, match_results.ft.home, quantile_level, fractional_kelly)
        p_draw = calculate_profit_from_chain(calculate_bayesian_kelly_chain(model_predictions.ft.draw, norm_odds.ft.draw), norm_odds.ft.draw, match_results.ft.draw, quantile_level, fractional_kelly)
        p_away = calculate_profit_from_chain(calculate_bayesian_kelly_chain(model_predictions.ft.away, norm_odds.ft.away), norm_odds.ft.away, match_results.ft.away, quantile_level, fractional_kelly)
        
        results["FT_1x2"] = (profit = results["FT_1x2"].profit + p_home.profit + p_draw.profit + p_away.profit, stake = results["FT_1x2"].stake + p_home.stake + p_draw.stake + p_away.stake)
        match_total_profit += p_home.profit + p_draw.profit + p_away.profit

        # FT Over/Under
        ft_ou_markets = [
            (model_predictions.ft.under_05, norm_odds.ft.under_0_5, match_results.ft.under_05),
            (1 .- model_predictions.ft.under_05, norm_odds.ft.over_0_5, !isnothing(match_results.ft.under_05) ? !match_results.ft.under_05 : nothing),
            (model_predictions.ft.under_15, norm_odds.ft.under_1_5, match_results.ft.under_15),
            (1 .- model_predictions.ft.under_15, norm_odds.ft.over_1_5, !isnothing(match_results.ft.under_15) ? !match_results.ft.under_15 : nothing),
            (model_predictions.ft.under_25, norm_odds.ft.under_2_5, match_results.ft.under_25),
            (1 .- model_predictions.ft.under_25, norm_odds.ft.over_2_5, !isnothing(match_results.ft.under_25) ? !match_results.ft.under_25 : nothing),
            (model_predictions.ft.under_35, norm_odds.ft.under_3_5, match_results.ft.under_35),
            (1 .- model_predictions.ft.under_35, norm_odds.ft.over_3_5, !isnothing(match_results.ft.under_35) ? !match_results.ft.under_35 : nothing),
        ]
        for (probs, odds, won) in ft_ou_markets
            p = calculate_profit_from_chain(calculate_bayesian_kelly_chain(probs, odds), odds, won, quantile_level, fractional_kelly)
            results["FT_OU"] = (profit = results["FT_OU"].profit + p.profit, stake = results["FT_OU"].stake + p.stake)
            match_total_profit += p.profit
        end

        # --- Process Half-Time Markets (similar logic) ---
        
        # HT 1x2
        p_home_ht = calculate_profit_from_chain(calculate_bayesian_kelly_chain(model_predictions.ht.home, norm_odds.ht.home), norm_odds.ht.home, match_results.ht.home, quantile_level, fractional_kelly)
        p_draw_ht = calculate_profit_from_chain(calculate_bayesian_kelly_chain(model_predictions.ht.draw, norm_odds.ht.draw), norm_odds.ht.draw, match_results.ht.draw, quantile_level, fractional_kelly)
        p_away_ht = calculate_profit_from_chain(calculate_bayesian_kelly_chain(model_predictions.ht.away, norm_odds.ht.away), norm_odds.ht.away, match_results.ht.away, quantile_level, fractional_kelly)

        results["HT_1x2"] = (profit = results["HT_1x2"].profit + p_home_ht.profit + p_draw_ht.profit + p_away_ht.profit, stake = results["HT_1x2"].stake + p_home_ht.stake + p_draw_ht.stake + p_away_ht.stake)
        match_total_profit += p_home_ht.profit + p_draw_ht.profit + p_away_ht.profit


        # HT Over/Under
        ht_ou_markets = [
            (model_predictions.ht.under_05, norm_odds.ht.under_0_5, match_results.ht.under_05),
            (1 .- model_predictions.ht.under_05, norm_odds.ht.over_0_5, !isnothing(match_results.ht.under_05) ? !match_results.ht.under_05 : nothing),
            (model_predictions.ht.under_15, norm_odds.ht.under_1_5, match_results.ht.under_15),
            (1 .- model_predictions.ht.under_15, norm_odds.ht.over_1_5, !isnothing(match_results.ht.under_15) ? !match_results.ht.under_15 : nothing),
            (model_predictions.ht.under_25, norm_odds.ht.under_2_5, match_results.ht.under_25),
            (1 .- model_predictions.ht.under_25, norm_odds.ht.over_2_5, !isnothing(match_results.ht.under_25) ? !match_results.ht.under_25 : nothing),
            (model_predictions.ht.under_35, norm_odds.ht.under_3_5, match_results.ht.under_35),
            (1 .- model_predictions.ht.under_35, norm_odds.ht.over_3_5, !isnothing(match_results.ht.under_35) ? !match_results.ht.under_35 : nothing),
        ]
        for (probs, odds, won) in ht_ou_markets
            p = calculate_profit_from_chain(calculate_bayesian_kelly_chain(probs, odds), odds, won, quantile_level, fractional_kelly)
            results["HT_OU"] = (profit = results["HT_OU"].profit + p.profit, stake = results["HT_OU"].stake + p.stake)
            match_total_profit += p.profit
        end

        # (Add HT Over/Under and Correct Score logic here if needed, following the FT example)

        # Update cumulative wealth
        push!(cumulative_wealth, last(cumulative_wealth) + match_total_profit)
    end

    # Calculate final ROI for each group
    rois = Dict{String, Float64}()
    for (group, data) in results
        rois[group] = data.stake > 0 ? (data.profit / data.stake) * 100 : 0.0
    end
    
    return (rois=rois, wealth_curve=cumulative_wealth)
end

# ##################################################
# 3. VISUALIZATION
# ##################################################

"""
Plots ROI vs. Quantile for different market groups.
"""
function plot_quantile_sensitivity(quantile_results::Dict)
    plot(legend=:outertopright, xlabel="Quantile Level", ylabel="ROI (%)", title="ROI Sensitivity to Kelly Quantile Choice")
    
    quantiles = sort(collect(keys(quantile_results)))
    market_groups = sort(collect(keys(quantile_results[quantiles[1]])))

    for group in market_groups
        roi_values = [quantile_results[q][group] for q in quantiles]
        plot!(quantiles, roi_values, label=group, lw=2)
    end
    
    savefig("roi_vs_quantile.png")
    println("📈 ROI vs. Quantile plot saved to roi_vs_quantile.png")
end

"""
Plots the cumulative wealth curve over all matches.
"""
function plot_wealth_curve(wealth_curve::Vector{Float64}, quantile_level::Float64)
    plot(wealth_curve, 
        label="Cumulative Wealth (Bankroll)", 
        xlabel="Match Number", 
        ylabel="Bankroll Value",
        title="Cumulative Wealth at Quantile $(quantile_level)",
        legend=:topleft,
        lw=2
    )
    hline!([1.0], linestyle=:dash, color=:red, label="Starting Bankroll", lw=2)
    
    savefig("cumulative_wealth.png")
    println("💰 Cumulative Wealth plot saved to cumulative_wealth.png")
end
