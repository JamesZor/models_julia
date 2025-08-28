# threaded analysis_logic.jl
using Statistics, DataFrames, Plots, Printf, Base.Threads

# Ensure the structs from your setup file are available
include("odds_market_setup.jl")

# ##################################################
# 1. NEW DATA STRUCTURES 
# ##################################################

# Holds the full distribution of Kelly fractions for all markets in a match
struct MatchKellyChains
    # Chains calculated using normalized odds (good for 1x2, O/U)
    norm_1x2::NamedTuple{(:home, :draw, :away), NTuple{3, Vector{Float64}}}
    norm_ou::NamedTuple{(:u05, :o05, :u15, :o15, :u25, :o25, :u35, :o35), NTuple{8, Vector{Float64}}}
    
    # Chains calculated using raw odds (often better for less efficient markets like Correct Score)
    raw_cs::Dict{Tuple{Int, Int}, Vector{Float64}}
end

# A container for all pre-processed data for a single match
struct MatchAnalysisPacket
    match_id::Int
    model_predictions::MatchPredict 
    match_results::Eval.MatchLinesResults
    norm_odds::Eval.RescaledMatchOdds
    raw_odds::Eval.MatchOdds
    kelly_chains::MatchKellyChains
end

# ##################################################
# 2. STAGE 1: DATA PREPARATION & PROCESSING (Threaded)
# ##################################################

"""
Core function to calculate all Kelly chains for a single match.
This is the heart of the "Process" stage.
"""
function calculate_match_kelly_chains(
    model_predictions::MatchPredict, 
    norm_odds::Eval.RescaledMatchOdds,
    raw_odds::Eval.MatchOdds
)
    # --- Use Normalized Odds for 1x2 and O/U Markets ---
    # Full-Time (FT)
    ft_home_chain = calculate_bayesian_kelly_chain(model_predictions.ft.home, norm_odds.ft.home)
    ft_draw_chain = calculate_bayesian_kelly_chain(model_predictions.ft.draw, norm_odds.ft.draw)
    ft_away_chain = calculate_bayesian_kelly_chain(model_predictions.ft.away, norm_odds.ft.away)
    
    ft_u25_chain = calculate_bayesian_kelly_chain(model_predictions.ft.under_25, norm_odds.ft.under_2_5)
    ft_o25_chain = calculate_bayesian_kelly_chain(1.0 .- model_predictions.ft.under_25, norm_odds.ft.over_2_5)
    
    # (Extend this for other O/U lines as needed, e.g., 0.5, 1.5, 3.5)
    
    # --- Use Raw Odds for Correct Score Market ---
    raw_cs_chains = Dict{Tuple{Int, Int}, Vector{Float64}}()
    for (idx, score_tuple) in enumerate(score_grid()) # Assuming a helper function score_grid
        score_probs = model_predictions.ft.correct_score[:, idx]
        score_odds = get(raw_odds.ft.correct_score, score_tuple, nothing)
        raw_cs_chains[score_tuple] = calculate_bayesian_kelly_chain(score_probs, score_odds)
    end

    # NOTE: This example is simplified to FT markets. You would expand this for HT markets too.
    return MatchKellyChains(
        (home=ft_home_chain, draw=ft_draw_chain, away=ft_away_chain),
        (u05=[], o05=[], u15=[], o15=[], u25=ft_u25_chain, o25=ft_o25_chain, u35=[], o35=[]), # Simplified
        raw_cs_chains
    )
end

"""
Processes a single match: fetches all data and calculates Kelly chains.
This function is designed to be called by a thread.
"""
function process_single_match(match_id::Int, matches_predictions_map, data_store)
    try
        m1 = matches_predictions_map[match_id]
        mr = get_match_results(data_store.incidents, match_id)
        norm_odds = impute_and_rescale_odds(data_store, match_id, target_sum=1.005)
        raw_odds = get_game_line_odds(data_store.odds, match_id)

        if isnothing(norm_odds) || isnothing(raw_odds)
            return nothing
        end

        kelly_chains = calculate_match_kelly_chains(m1, norm_odds, raw_odds)
        
        return MatchAnalysisPacket(match_id, m1, mr, norm_odds, raw_odds, kelly_chains)
    catch e
        @warn "Could not process match_id $match_id: $e"
        return nothing
    end
end

"""
Orchestrates the threaded processing of all target matches.
"""
function prepare_all_matches_threaded(target_matches, matches_predictions_map, data_store)
    num_matches = nrow(target_matches)
    packets = Vector{Union{MatchAnalysisPacket, Nothing}}(undef, num_matches)
    
    # Use @threads for simple, efficient parallelization
    @threads for i in 1:num_matches
        match_id = target_matches[i, :match_id]
        packets[i] = process_single_match(match_id, matches_predictions_map, data_store)
    end
    
    # Filter out any matches that failed and return a dictionary keyed by match_id
    valid_packets = filter(!isnothing, packets)
    return Dict(p.match_id => p for p in valid_packets)
end


# ##################################################
# 3. STAGE 2: ANALYSIS (Fast, Non-Threaded)
# ##################################################

"""
Calculates aggregate ROI across all matches for a given quantile.
This function operates on the pre-processed data and is very fast.
"""
function calculate_aggregate_roi(
    analysis_packets::Dict{Int, MatchAnalysisPacket};
    quantile_level::Float64 = 0.5,
    fractional_kelly::Float64 = 1.0
)
    # Initialize accumulators
    market_groups = ["FT_1x2", "FT_OU_2.5", "FT_CS"]
    results = Dict(g => (profit=0.0, stake=0.0) for g in market_groups)

    for (match_id, packet) in analysis_packets
        # Unpack data for clarity
        chains = packet.kelly_chains
        results_ = packet.match_results
        odds = packet.norm_odds # Using normalized odds for staking in 1x2 and O/U

        # --- FT 1x2 ---
        p_h = calculate_profit_from_chain(chains.norm_1x2.home, odds.ft.home, results_.ft.home, quantile_level, fractional_kelly)
        p_d = calculate_profit_from_chain(chains.norm_1x2.draw, odds.ft.draw, results_.ft.draw, quantile_level, fractional_kelly)
        p_a = calculate_profit_from_chain(chains.norm_1x2.away, odds.ft.away, results_.ft.away, quantile_level, fractional_kelly)
        results["FT_1x2"] = (profit=results["FT_1x2"].profit + p_h.profit + p_d.profit + p_a.profit, stake=results["FT_1x2"].stake + p_h.stake + p_d.stake + p_a.stake)
        
        # --- FT Over/Under 2.5 ---
        p_u25 = calculate_profit_from_chain(chains.norm_ou.u25, odds.ft.under_2_5, results_.ft.under_25, quantile_level, fractional_kelly)
        p_o25 = calculate_profit_from_chain(chains.norm_ou.o25, odds.ft.over_2_5, !isnothing(results_.ft.under_25) ? !results_.ft.under_25 : nothing, quantile_level, fractional_kelly)
        results["FT_OU_2.5"] = (profit=results["FT_OU_2.5"].profit + p_u25.profit + p_o25.profit, stake=results["FT_OU_2.5"].stake + p_u25.stake + p_o25.stake)

        # (Add logic for other markets as needed)
    end

    # Calculate final ROI
    rois = Dict{String, Float64}()
    for (group, data) in results
        rois[group] = data.stake > 0 ? (data.profit / data.stake) * 100 : 0.0
    end
    
    return rois
end


# Helper function from previous answer (no changes needed)
function calculate_profit_from_chain(chain, odds, won, q, fk)
    # ... same implementation as before ...
end

# Helper for CS market grid
function score_grid()
    return [(h, a) for h in 0:3 for a in 0:3]
end

# Plotting functions from previous answer (no changes needed)
# plot_quantile_sensitivity(quantile_results)
# plot_wealth_curve(wealth_curve, quantile_level)
