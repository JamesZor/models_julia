# r11_staking_eda.jl

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, CSV

include("l03_risk_manager.jl") 
include("l04_vector_alpha.jl") 

@info "Loading datastore and experiment latents..."
ds = D.load_datastore_cached(D.ScottishLower())
odds = D.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))

# We use IDX 3 (funnel_apm_xg) since it was your Champion Model
src_dir = "./data/experiments/plus_minus_biweek"
list_of_experiments = E.list_experiments(src_dir, data_dir="")
expr = E.load_experiment(list_of_experiments, 3) 

latents = E.extract_oos_predictions(ds, expr)
n_matches = nrow(latents.df)

optimal_alpha_dict = Dict{String, Float64}(
    "O/U 3.5_over_35" => 0.002, "BTTS_btts_no" => 0.003, "O/U 4.5_over_45" => 0.003,
    "O/U 1.5_over_15" => 0.004, "O/U 1.5_under_15" => 0.027, "O/U 0.5_under_05" => 0.029,
    "O/U 3.5_under_35" => 0.131, "1X2_away" => 0.157, "BTTS_btts_yes" => 0.183,
    "1X2_home" => 0.432, "1X2_draw" => 0.445, "O/U 2.5_over_25" => 0.486,
    "O/U 2.5_under_25" => 0.818, "O/U 0.5_over_05" => 0.846, "O/U 4.5_under_45" => 1.000
)

v_config = VectorAlphaConfig(0.02, optimal_alpha_dict)
r_config = RiskConfig(20.0)

scalar_markets = D.AbstractMarket[D.Market1X2(), D.MarketBTTS()]
over_unders = [D.MarketOverUnder(i + 0.5) for i in 0:4]
markets_config = D.MarketConfig(reduce(vcat, (scalar_markets, over_unders)))

# DataFrame to store every individual bet placed
bets_df = DataFrame(
    match_id = Int[],
    market_selection = String[],
    stake = Float64[],
    odds = Float64[],
    is_winner = Bool[],
    pl = Float64[]
)

@info "Simulating Risk-Managed Vector Portfolio to gather staking distribution..."

for i in 1:n_matches
    row = latents.df[i, :]
    m_id = row.match_id
    
    raw_odds_map, _, winner_map = extract_market_data(odds, m_id, markets_config)
    if isempty(raw_odds_map)
        continue
    end
    
    odds_map, fair_prob_map = normalize_market_group_odds(raw_odds_map)
    
    param = Predictions.extract_params(expr.config.model, row)
    local score_matrix
    try
        score_matrix = Predictions.compute_score_matrix(expr.config.model, param)
    catch e
        continue
    end
    
    match_model_prob = Dict(
        string(m) => Predictions.compute_market_probs(score_matrix, m)
        for m in markets_config.markets
    )
    
    selections, vec_stakes, R_mat = optimize_portfolio_vector(score_matrix, match_model_prob, odds_map, fair_prob_map, winner_map, v_config)
    
    if isempty(selections)
        continue
    end
    
    P_model_grid = mean(score_matrix.data, dims=3)[:, :, 1]
    p_model_vec  = vec(P_model_grid)
    returns_vec  = R_mat * vec_stakes
    
    # Apply Lambda=20 constraint
    k_shrink = solve_drawdown_multiplier(p_model_vec, returns_vec, r_config.lambda)
    risk_stakes = vec_stakes .* k_shrink
    
    for j in 1:length(selections)
        st = risk_stakes[j]
        sel = selections[j]
        
        # Only log if the Risk Manager allowed a non-zero stake
        if st > 0
            m_str = replace(sel.market, "Market[" => "", "]" => "")
            key = "$(m_str)_$(sel.selection)"
            
            pl = 0.0
            if sel.is_winner
                pl = st * (1.0 - 0.02) * (sel.odds - 1.0)
            else
                pl = -st
            end
            
            push!(bets_df, (m_id, key, st, sel.odds, sel.is_winner, pl))
        end
    end
end

# ===================================================================
# EDA 1: Match-Level Staking (How much Bankroll is put at risk per match?)
# ===================================================================
match_group = combine(groupby(bets_df, :match_id), :stake => sum => :total_match_stake)
avg_match_stake = mean(match_group.total_match_stake)
med_match_stake = median(match_group.total_match_stake)
max_match_stake = maximum(match_group.total_match_stake)

println("\n", "="^80)
println("=== MATCH-LEVEL STAKING DISTRIBUTION (Lambda = 20.0) ===")
println("Total Matches with >= 1 bet : ", nrow(match_group))
println("Average Total Stake per match: ", round(avg_match_stake * 100, digits=2), "% of bankroll")
println("Median Total Stake per match : ", round(med_match_stake * 100, digits=2), "% of bankroll")
println("Maximum Total Stake per match: ", round(max_match_stake * 100, digits=2), "% of bankroll")
println("="^80)

# ===================================================================
# EDA 2: Market-Level Staking & Win/Loss Split
# ===================================================================

# Base summary per market
market_summary = combine(groupby(bets_df, :market_selection),
    nrow => :bet_count,
    :stake => sum => :total_volume,
    :stake => mean => :avg_stake_pct,
    :pl => sum => :total_pl,
    :is_winner => (x -> sum(x) / length(x)) => :hit_rate
)

# Split by Winning and Losing bets
wins_df = filter(r -> r.is_winner, bets_df)
losses_df = filter(r -> !r.is_winner, bets_df)

win_summary = combine(groupby(wins_df, :market_selection), :stake => mean => :avg_win_stake_pct)
loss_summary = combine(groupby(losses_df, :market_selection), :stake => mean => :avg_loss_stake_pct)

# Outer join to assemble the master table
market_summary = outerjoin(market_summary, win_summary, on=:market_selection)
market_summary = outerjoin(market_summary, loss_summary, on=:market_selection)

# Clean missing data (if a market had 0 wins or 0 losses)
market_summary.avg_win_stake_pct = coalesce.(market_summary.avg_win_stake_pct, 0.0)
market_summary.avg_loss_stake_pct = coalesce.(market_summary.avg_loss_stake_pct, 0.0)

# Formatting
market_summary.hit_rate = round.(market_summary.hit_rate .* 100, digits=1)
market_summary.total_volume = round.(market_summary.total_volume, digits=2)

# Convert all stakes from fraction to percentage for readability
market_summary.avg_stake_pct = round.(market_summary.avg_stake_pct .* 100, digits=2)
market_summary.avg_win_stake_pct = round.(market_summary.avg_win_stake_pct .* 100, digits=2)
market_summary.avg_loss_stake_pct = round.(market_summary.avg_loss_stake_pct .* 100, digits=2)
market_summary.total_pl = round.(market_summary.total_pl, digits=3)

# Sort by the markets where we put the most money to work
sort!(market_summary, :total_volume, rev=true)
select!(market_summary, :market_selection, :bet_count, :total_volume, :total_pl, :hit_rate, :avg_stake_pct, :avg_win_stake_pct, :avg_loss_stake_pct)

println("\n", "="^110)
println("=== MARKET-LEVEL STAKING DISTRIBUTION (Ranked by Total Volume wagered) ===")
println("Note: Average stakes are displayed as a % of current bankroll.")
println("="^110)
display(market_summary)

out_file = "current_development/portfolio_explore/r11_staking_eda.csv"
CSV.write(out_file, market_summary)
println("\n✓ Full EDA exported to $(out_file)")
