# r08_vector_alpha_optim.jl

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics
using ThreadPinning
using CSV
using Optim

include("l04_vector_alpha.jl")

# -------------------------------------------------------------------
# 1. Environment & Data Loading
# -------------------------------------------------------------------
pinthreads(:cores)
@info "Loading datastore and experiment latents..."
ds = D.load_datastore_cached(D.ScottishLower())
odds = D.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))

src_dir = "./data/experiments/plus_minus_biweek"
list_of_experiments = E.list_experiments(src_dir, data_dir="")
expr = E.load_experiment(list_of_experiments, 3)

latents = E.extract_oos_predictions(ds, expr)
n_matches = nrow(latents.df)

# -------------------------------------------------------------------
# 2. Map The Vector Keys (15 Dimensions)
# -------------------------------------------------------------------
scalar_markets = D.AbstractMarket[D.Market1X2(), D.MarketBTTS()]
over_unders = [D.MarketOverUnder(i + 0.5) for i in 0:4]
markets_config = D.MarketConfig(reduce(vcat, (scalar_markets, over_unders)))

# These strings must exactly match "$(sel.market)_$(sel.selection)"
alpha_keys = [
    "1X2_home", "1X2_draw", "1X2_away",
    "BTTS_btts_yes", "BTTS_btts_no",
    "O/U 0.5_over_05", "O/U 0.5_under_05",
    "O/U 1.5_over_15", "O/U 1.5_under_15",
    "O/U 2.5_over_25", "O/U 2.5_under_25",
    "O/U 3.5_over_35", "O/U 3.5_under_35",
    "O/U 4.5_over_45", "O/U 4.5_under_45"
]

DIM = length(alpha_keys)

# -------------------------------------------------------------------
# 3. Build the Memory Cache (The Speed Hack)
# -------------------------------------------------------------------
struct MatchCache
    m_id::Int
    score_matrix::Any
    match_model_prob::Any
    odds_map::Any
    fair_prob_map::Any
    winner_map::Any
end

@info "Pre-computing and caching $(n_matches) Score Matrices to RAM..."

match_cache = Vector{Union{Nothing, MatchCache}}(undef, n_matches)

Threads.@threads for i in 1:n_matches
    row = latents.df[i, :]
    m_id = row.match_id
    
    raw_odds_map, _, winner_map = extract_market_data(odds, m_id, markets_config)
    if isempty(raw_odds_map)
        match_cache[i] = nothing
        continue
    end
    
    odds_map, fair_prob_map = normalize_market_group_odds(raw_odds_map)
    
    param = Predictions.extract_params(expr.config.model, row)
    local score_matrix
    try
        score_matrix = Predictions.compute_score_matrix(expr.config.model, param)
    catch e
        match_cache[i] = nothing
        continue
    end
    
    match_model_prob = Dict(
        string(m) => Predictions.compute_market_probs(score_matrix, m)
        for m in markets_config.markets
    )
    
    match_cache[i] = MatchCache(m_id, score_matrix, match_model_prob, odds_map, fair_prob_map, winner_map)
end

# Filter out failed/missing matches
valid_cache = filter(x -> !isnothing(x), match_cache)
@info "Cache built successfully! Valid matches ready for 15-D optimization: $(length(valid_cache))"

# -------------------------------------------------------------------
# 4. Objective Function wrapper for Optim.jl
# -------------------------------------------------------------------
function evaluate_alpha_vector(x::Vector{Float64})
    # Map the numeric vector X back to specific market selections
    alpha_dict = Dict{String, Float64}(alpha_keys[i] => x[i] for i in 1:DIM)
    config = VectorAlphaConfig(0.02, alpha_dict)
    
    returns = zeros(length(valid_cache))
    
    # Fast evaluation loop over the pre-computed memory cache (MULTITHREADED)
    Threads.@threads for idx in 1:length(valid_cache)
        c = valid_cache[idx]
        selections, stakes, _ = optimize_portfolio_vector(c.score_matrix, c.match_model_prob, c.odds_map, c.fair_prob_map, c.winner_map, config)
        
        if isempty(selections)
            continue
        end
        
        match_pl = 0.0
        for i in 1:length(selections)
            st = stakes[i]
            sel = selections[i]
            if st > 0
                if sel.is_winner
                    match_pl += st * (1.0 - 0.02) * (sel.odds - 1.0)
                else
                    match_pl -= st
                end
            end
        end
        
        returns[idx] = match_pl
    end
    
    # Filter out empty zeros (matches where no selections were made)
    filter!(x -> x != 0.0, returns)
    
    # Heavy penalty for completely zero-betting vectors
    if length(returns) < 10 
        return 99999.0 
    end
    
    # Calculate Compounding and Martin Ratio
    bankroll = [1.0; cumprod(1.0 .+ returns)]
    run_max = accumulate(max, bankroll)
    dd = (bankroll .- run_max) ./ run_max
    dd_pct = dd .* 100.0
    
    fb = bankroll[end]
    mdd_pct = minimum(dd_pct)
    
    # Ulcer Index = Root Mean Square of all drawdowns
    ulcer_index = sqrt(mean(dd_pct .^ 2))
    ulcer_index = max(ulcer_index, 1e-4) # Prevent division by zero
    
    tot_ret = (fb - 1.0) * 100.0
    
    # If the strategy loses money, return a penalty
    if tot_ret <= 0.0
        return 99999.0 + abs(tot_ret)
    end
    
    martin = (tot_ret / ulcer_index)
    
    # Optim.jl MINIMIZES the objective function, so we return the negative Martin Ratio
    return -martin
end

# -------------------------------------------------------------------
# 5. Global Optimization Execution
# -------------------------------------------------------------------
println("\n", "="^80)
println("=== RUNNING 15-DIMENSIONAL ALPHA OPTIMIZATION ===")
println("Algorithm : Fminbox(NelderMead())")
println("Objective : Maximize Martin Ratio")
println("="^80)

# Start all alphas at the scalar empirical optimum (0.25)
initial_x = fill(0.25, DIM)
lower = zeros(DIM)
upper = ones(DIM)

# Run Optimization
@time result = optimize(evaluate_alpha_vector, lower, upper, initial_x, Fminbox(NelderMead()), Optim.Options(iterations=1500, show_trace=true, show_every=50))

# -------------------------------------------------------------------
# 6. Extract and Display Results
# -------------------------------------------------------------------
best_martin = -Optim.minimum(result)
best_x = Optim.minimizer(result)

println("\n", "="^80)
println("=== THEORETICAL UPPER LIMIT FOUND ===")
println("Maximized Martin Ratio: ", round(best_martin, digits=3))
println("="^80)

final_df = DataFrame(
    Market_Selection = alpha_keys,
    Optimal_Alpha = round.(best_x, digits=3)
)

# Sort from lowest alpha (markets the model is worst at) to highest (markets the model crushes)
sort!(final_df, :Optimal_Alpha)

display(final_df)

out_file = "current_development/portfolio_explore/r08_vector_alpha_results.csv"
CSV.write(out_file, final_df)
println("\n✓ Optimization results exported to $(out_file)")
