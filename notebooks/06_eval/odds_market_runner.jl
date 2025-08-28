println("Loading experiment data...")
r1 = load_experiment("/home/james/bet_project/models_julia/experiments/maher_basic_test_20250820_231427.jld2")
result = r1[1]
experiment_config = r1[2]
mapping = result.mapping

# Load data store
data_files = DataFiles("/home/james/bet_project/football_data/scot_nostats_20_to_24")
data_store = DataStore(data_files)

df = data_store.incidents
match_id = 12476901 
# Get target season matches
target_matches = filter(row -> row.season == experiment_config.cv_config.target_season, data_store.matches)

# Let's get the odds for a specific match
match_id_to_process = 12476901 
original_odds = get_game_line_odds(data_store.odds, match_id_to_process)

# Now, let's process them using the new function, aiming for a 105% book (1.05)
rescaled_odds_obj = impute_and_rescale_odds(data_store, match_id_to_process, target_sum=1.005)



# display 
# You can now inspect the rescaled odds
println("--- Original FT 1x2 Odds ---")
println("Home: ", original_odds.ft.home)
println("Draw: ", original_odds.ft.draw)
println("Away: ", original_odds.ft.away)
original_probs_sum = sum(1 ./ filter(!isnothing, [original_odds.ft.home, original_odds.ft.draw, original_odds.ft.away]))
@printf "Original Implied Probability Sum: %.4f\n\n" original_probs_sum

println("--- Rescaled FT 1x2 Odds (Target Sum: 1.05) ---")
println("Home: ", rescaled_odds_obj.ft.home)
println("Draw: ", rescaled_odds_obj.ft.draw)
println("Away: ", rescaled_odds_obj.ft.away)
rescaled_probs_sum = sum(1 ./ filter(!isnothing, [rescaled_odds_obj.ft.home, rescaled_odds_obj.ft.draw, rescaled_odds_obj.ft.away]))
@printf "Rescaled Implied Probability Sum: %.4f\n" rescaled_probs_sum

mr = get_match_results(data_store.incidents, match_id)


###

match_id = 12476901 

matches_results = predict_target_season(target_matches, result, mapping)
m1::MatchPredict = matches_results[match_id]


##################################################
# Kelly Criterion Implementation
##################################################
#
#
module ExtendedEval
  # Kelly criterion structures
  struct KellyFraction
      outcome::Symbol  # :home, :draw, :away, etc.
      fraction::Float64
      confidence_interval::Tuple{Float64, Float64}  # CI from Bayesian posterior
  end

end
using Statistics
"""
Calculate Bayesian Kelly fraction for a single outcome
"""
function calculate_bayesian_kelly(
    model_probs::Vector{Float64},  # Posterior samples
    market_odds::Float64,
    fractional_kelly::Float64 = 1.0  # Conservative fraction
)
    kelly_fractions = Float64[]
    
    for p in model_probs
        # Standard Kelly formula: f = (p*o - 1) / (o - 1)
        # where p is probability, o is decimal odds
        edge = p * market_odds - 1.0
        
        if edge > 0
            f = edge / (market_odds - 1.0)
            push!(kelly_fractions, f)
        else
            push!(kelly_fractions, 0.0)
        end
    end
    
    if isempty(kelly_fractions) || all(f -> f == 0, kelly_fractions)
        return ExtendedEval.KellyFraction(:none, 0.0, (0.0, 0.0))
    end
    
    # Use percentiles for conservative betting
    fraction = quantile(kelly_fractions, 0.5) * fractional_kelly  # Median * safety factor
    ci = (quantile(kelly_fractions, 0.25), quantile(kelly_fractions, 0.75))
    
    return ExtendedEval.KellyFraction(:single, fraction, ci)
end

# test 
#

match_id = rand(keys(matches_results))
m1 = matches_results[match_id];
o1 = impute_and_rescale_odds(data_store, match_id_to_process, target_sum=1.005)
oo1 = get_game_line_odds(data_store.odds, match_id_to_process)

mr = get_match_results(data_store.incidents, match_id)

filter(row-> row.match_id==match_id, data_store.matches)[:,[:home_team, :away_team, :home_score, :away_score, :home_score_ht, :away_score_ht]]

mr.ft.home 
mr.ft.away 
mr.ft.draw
k_ft_home = calculate_bayesian_kelly(m1.ft.home, o1.ft.home)
k_ft_draw = calculate_bayesian_kelly(m1.ft.draw, o1.ft.draw)
k_ft_away = calculate_bayesian_kelly(m1.ft.away, o1.ft.away)

mr.ht.home 
mr.ht.away 
mr.ht.draw
k_ht_home = calculate_bayesian_kelly(m1.ht.home, o1.ht.home)
k_ht_draw = calculate_bayesian_kelly(m1.ht.draw, o1.ht.draw)
k_ht_away = calculate_bayesian_kelly(m1.ht.away, o1.ht.away)


mr.ft.under_25
mr.ft.under_15 
mr.ft.under_05

mr.ft.under_35
k_ft_u35 = calculate_bayesian_kelly(m1.ft.under_35, o1.ft.under_3_5)
k_ft_o35 = calculate_bayesian_kelly(1 .-m1.ft.under_35, o1.ft.over_3_5)

mr.ft.under_25
k_ft_u25 = calculate_bayesian_kelly(m1.ft.under_25, o1.ft.under_2_5)
k_ft_o25 = calculate_bayesian_kelly(1 .-m1.ft.under_25, o1.ft.over_2_5)

mr.ft.under_15
k_ft_u15 = calculate_bayesian_kelly(m1.ft.under_15, o1.ft.under_1_5)
k_ft_o15 = calculate_bayesian_kelly(1 .-m1.ft.under_15, o1.ft.over_1_5)

mr.ft.under_05
k_ft_u05 = calculate_bayesian_kelly(m1.ft.under_05, o1.ft.under_0_5)
k_ft_o05 = calculate_bayesian_kelly(1 .-m1.ft.under_05, o1.ft.over_0_5)



mr.ht.under_25
k_ht_u25 = calculate_bayesian_kelly(m1.ht.under_25, o1.ht.under_2_5)
k_ht_o25 = calculate_bayesian_kelly(1 .-m1.ht.under_25, o1.ht.over_2_5)

mr.ht.under_15
k_ht_u15 = calculate_bayesian_kelly(m1.ht.under_15, o1.ht.under_1_5)
k_ht_o15 = calculate_bayesian_kelly(1 .-m1.ht.under_15, o1.ht.over_1_5)

mr.ht.under_05
k_ht_u05 = calculate_bayesian_kelly(m1.ht.under_05, o1.ht.under_0_5)
k_ht_o05 = calculate_bayesian_kelly(1 .-m1.ht.under_05, o1.ht.over_0_5)

# order for cs 
# 0-0, 1-0, 2-0, 3-0, 0-1, 1-1, 2-1, 3-1, 0-2, 1-2, 2-2, 3-2, 0-3, 1-3, 2-3, 3-3 
# 1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16. 

k_ft_cs_0_0 = calculate_bayesian_kelly(m1.ft.correct_score[:, 1], oo1.ft.correct_score[(0,0)])
k_ft_cs_2_1 = calculate_bayesian_kelly(m1.ft.correct_score[:, 7], oo1.ft.correct_score[(2,1)])
k_ft_cs_2_0 = calculate_bayesian_kelly(m1.ft.correct_score[:, 3], oo1.ft.correct_score[(2,1)])


#### Claudes shit attempt 


match_id = rand(keys(matches_results))
m1 = matches_results[match_id];
kelly = calculate_match_kelly_fractions(m1, data_store, match_id, fractional_kelly=1.0);
mr = get_match_results(data_store.incidents, match_id);
display_kelly_fractions_v2(kelly, match_id, data_store, mr)
kelly



###### 

function calc_profit(kf::Union{ExtendedEval.KellyFraction, Nothing}, odds::Union{Float64, Nothing}, won::Union{Bool, Nothing})
    if isnothing(kf) || isnothing(odds) || isnothing(won) || kf.fraction <= 0
        return 0.0
    end
    stake = kf.fraction  # As fraction of bankroll
    return won ? stake * (odds - 1) : -stake
end

function test_kelly_1x2_ht(kelly, match_results, norm_odds)
    profit_1x2_ht = 0.0
    # HT 1X2
    ht_1x2 = [
        (:home, "Home", kelly.ht.one_x_two.home, match_results.ht.home, norm_odds.ht.home),
        (:draw, "Draw", kelly.ht.one_x_two.draw, match_results.ht.draw, norm_odds.ht.draw),
        (:away, "Away", kelly.ht.one_x_two.away, match_results.ht.away, norm_odds.ht.away)
    ]
    
    for (field, name, kf, won, odds) in ht_1x2
        profit = calc_profit(kf, odds, won)
        profit_1x2_ht += profit
    end
    return profit_1x2_ht
end


match_id = rand(keys(matches_results))
m1 = matches_results[match_id];
mr = get_match_results(data_store.incidents, match_id);
kelly = calculate_match_kelly_fractions(m1, data_store, match_id, fractional_kelly=1.0);
norm_odds = impute_and_rescale_odds(data_store, match_id, target_sum=1.005)
raw_odds = get_game_line_odds(data_store.odds, match_id)
test_kelly_1x2_ht(kelly, mr, norm_odds)
test_kelly_1x2_ht(kelly, mr, raw_odds)


#####
#
function calculate_bayesian_kelly(
    model_probs::Vector{Float64},  # Posterior samples
    market_odds::Float64,
    fractional_kelly::Float64 = 1.0  # Conservative fraction
)
    kelly_fractions = Float64[]
    
    for p in model_probs
        # Standard Kelly formula: f = (p*o - 1) / (o - 1)
        # where p is probability, o is decimal odds
        edge = p * market_odds - 1.0
        
        if edge > 0
            f = edge / (market_odds - 1.0)
            push!(kelly_fractions, f)
        else
            push!(kelly_fractions, 0.0)
        end
    end
    
    if isempty(kelly_fractions) || all(f -> f == 0, kelly_fractions)
        return ExtendedEval.KellyFraction(:none, 0.0, (0.0, 0.0))
    end
   
    return kelly_fractions
    
end


k_ = calculate_bayesian_kelly(m1.ft.correct_score[:,6], raw_odds.ft.correct_score[(1,1)], 1.0)



quantile(k_, [0.1, 0.25, 0.5, 0.75, 0.9]) 
mean(k_)
median(k_)

using Plots
histogram(k_, bins=60)

sum( k_ .> 0)/ length(k_)

