# current_development/trust_model_dev/r02_trust_mvp.jl
using Revise
using BayesianFootball
using DataFrames
using Distributions
using Statistics
using Turing
using ThreadPinning

# Lock OS threads for maximum MCMC performance
pinthreads(:cores)

# Include the loader
include("l02_trust_mvp.jl")
using .TrustMVP

const Data = BayesianFootball.Data
const Experiments = BayesianFootball.Experiments

println("1. Loading Datastore and Experiment...")
# Load the Irish Datastore as used in previous interactive sessions
ds = Data.load_datastore_cached(Data.Ireland())
src_dir = "./data/double_poisson_smile_src_grid/"
list_of_experiments = Experiments.list_experiments(src_dir, data_dir="")
expr = Experiments.load_experiment(list_of_experiments, 5)

split_id = 25
split = BayesianFootball.Data.create_data_splits(ds, expr.config.splitter)[split_id]
expr_split = expr.training_results[split_id]

println("\n2. Extracting Features for Over 2.5...")
next_matches = BayesianFootball.Data.get_next_matches(ds, expr_split, expr.config.splitter)
all_expr_inference = BayesianFootball.Predictions.model_inference(ds, expr) 
next_matches_inference = subset(all_expr_inference.df, :match_id => ByRow(x -> x in next_matches.match_id))

past_match_ids_set = Set(split[1].match_id)

# Build Team Map
all_teams_in_split = sort(unique(vcat(split[1].home_team, split[1].away_team)))
team_map = Dict(name => i for (i, name) in enumerate(all_teams_in_split))
total_teams = length(team_map)

# Feature Matrix Base
past_odds = subset(ds.odds, :match_id => ByRow(x -> x in past_match_ids_set))
odds_features = select(past_odds, 
    :match_id, 
    :selection, 
    :is_winner, 
    :prob_fair_open,
    :prob_fair_close
)

model_preds = select(all_expr_inference.df, 
    :match_id, 
    :selection, 
    :distribution => ByRow(mean) => :model_prob
)

match_teams = select(ds.matches, :match_id, :home_team, :away_team)

feature_matrix = innerjoin(odds_features, model_preds, on=[:match_id, :selection])
feature_matrix = innerjoin(feature_matrix, match_teams, on=:match_id)

feature_matrix.home_team_index = [get(team_map, t, 0) for t in feature_matrix.home_team]
feature_matrix.away_team_index = [get(team_map, t, 0) for t in feature_matrix.away_team]

# Filter specifically to Over 2.5
df_over25 = subset(feature_matrix, :selection => ByRow(==(:over_25)))
println("   Found $(nrow(df_over25)) historical matches for Over 2.5")

println("\n3. Calculating Global CLV Prior...")
dynamic_prior_mean, beta_clv = calculate_global_clv_prior(df_over25)
println("   CLV Beta: ", round(beta_clv, digits=3))
println("   Dynamic Prior Mean: ", round(dynamic_prior_mean, digits=3))

println("\n4. Building and Sampling Hierarchical Trust Model...")
rename!(df_over25, :prob_fair_close => :market_devig)
trust_model = build_trust_model(df_over25, total_teams, dynamic_prior_mean)

# MCMC Settings
n_samples   = 1000
n_chains    = 4
n_warmup    = 500
accept_rate = 0.65
max_depth   = 10

chain_trust = sample(
    trust_model, 
    NUTS(n_warmup, accept_rate; max_depth=max_depth), 
    MCMCThreads(), 
    n_samples, 
    n_chains
)

println("\n5. Processing Out-Of-Sample Matches (Distributional Staking)...")
valid_match_ids = Set(next_matches.match_id)

next_model_over25 = subset(next_matches_inference, :selection => ByRow(==(:over_25)))
match_to_l1_dist = Dict(r.match_id => r.distribution for r in eachrow(next_model_over25))

next_odds = subset(ds.odds, :match_id => ByRow(x -> x in valid_match_ids))
next_odds_features = select(next_odds, 
    :match_id, 
    :selection, 
    :prob_fair_close => :market_devig,
    :odds_close
)

next_matrix = innerjoin(select(next_model_over25, :match_id, :selection), next_odds_features, on=[:match_id, :selection])
next_matrix = innerjoin(next_matrix, match_teams, on=:match_id)

composed_dists, final_stakes = distributional_staking(
    next_matrix, 
    chain_trust, 
    match_to_l1_dist, 
    team_map
)

next_matrix.composed_distribution = composed_dists
next_matrix.kelly_stake = final_stakes

# Sort by highest stake
sort!(next_matrix, :kelly_stake, rev=true)

println("\n======================================================")
println("COMPOSED POSTERIOR UNCERTAINTY & KELLY STAKES")
println("======================================================")

for row in eachrow(next_matrix)
    comp_dist = row.composed_distribution
    l1_dist   = match_to_l1_dist[row.match_id]
    
    println("Match: ", row.home_team, " vs ", row.away_team)
    println("  Market Prob:     ", round(row.market_devig, digits=4))
    println("  L1 Mean Prob:    ", round(mean(l1_dist), digits=4))
    println("  L1 Std Dev:      ", round(std(l1_dist), digits=4))
    println("  Composed Mean:   ", round(mean(comp_dist), digits=4))
    println("  Composed Std:    ", round(std(comp_dist), digits=4))
    println("  Kelly Stake:     ", round(row.kelly_stake * 100.0, digits=2), "%")
    println("-------------------------------------------------")
end

println("Runner complete!")


# 1. Bring in the `is_winner` column from ds.odds
next_is_winner = select(ds.odds, :match_id, :selection, :is_winner)
next_matrix = innerjoin(next_matrix, next_is_winner, on=[:match_id, :selection], makeunique=true)

# 2. Calculate L1 Kelly Stake and Returns
l1_stakes = Float64[]
l1_returns = Float64[]
composed_returns = Float64[]

for row in eachrow(next_matrix)
    l1_dist = match_to_l1_dist[row.match_id]
    odds = row.odds_close
    b = odds - 1.0 # The decimal profit multiplier
    
    # Calculate Distributional Kelly for purely Layer 1
    raw_l1_draw_stakes = Float64[]
    for p in l1_dist
        stake = (p * (b + 1.0) - 1.0) / b
        push!(raw_l1_draw_stakes, max(0.0, stake)) # Floor at 0
    end
    l1_stake = mean(raw_l1_draw_stakes)
    push!(l1_stakes, l1_stake)
    
    # Calculate PnL (Returns)
    if row.is_winner === true
        # We won! Profit is stake * (odds - 1)
        push!(l1_returns, l1_stake * b)
        push!(composed_returns, row.kelly_stake * b)
    elseif row.is_winner === false
        # We lost! We lose the full stake
        push!(l1_returns, -l1_stake)
        push!(composed_returns, -row.kelly_stake)
    else
        # Match voided or unplayed
        push!(l1_returns, 0.0) 
        push!(composed_returns, 0.0)
    end
end

next_matrix.l1_kelly_stake = l1_stakes
next_matrix.l1_returns = l1_returns
next_matrix.composed_returns = composed_returns

println("=========================================================")
println("FINANCIAL PERFORMANCE: L1 Raw vs L2 Trust Calibrated")
println("Total L1 Return:       ", round(sum(next_matrix.l1_returns) * 100, digits=2), "%")
println("Total Composed Return: ", round(sum(next_matrix.composed_returns) * 100, digits=2), "%")
println("=========================================================")

# Let's view the side-by-side comparison
display(select(next_matrix, 
    :match_id, 
    :home_team, 
    :away_team, 
    :is_winner, 
    :l1_kelly_stake, 
    :kelly_stake, 
    :l1_returns, 
    :composed_returns
))


    # 1. Instantiate the optimizer from your own codebase!
    b_kelly = BayesianFootball.Signals.BayesianKelly()

    mchale_stakes = Float64[]
    mchale_returns = Float64[]

    for row in eachrow(next_matrix)
	# We pass the fully composed (L1 + L2) distribution
        dist = row.composed_distribution 
        odds = row.odds_close
        
	# Call your exact codebase function!
        final_stake = BayesianFootball.Signals.compute_stake(b_kelly, dist, odds)
        push!(mchale_stakes, final_stake)
        
	# Calculate PnL for the new stake
        if row.is_winner === true
            push!(mchale_returns, final_stake * (odds - 1.0))
        elseif row.is_winner === false
            push!(mchale_returns, -final_stake)
        else
            push!(mchale_returns, 0.0)
        end
    end

    next_matrix.mchale_stake = mchale_stakes
    next_matrix.mchale_returns = mchale_returns

    println("=========================================================")
    println("OPTIMIZER PERFORMANCE: Simple Average vs BayesianKelly")
    println("Total Simple Return: ", round(sum(next_matrix.composed_returns) * 100, digits=2), "%")
    println("Total McHale Return: ", round(sum(next_matrix.mchale_returns) * 100, digits=2), "%")
    println("=========================================================")

    # Compare the exact stakes side-by-side
    display(select(next_matrix, 
        :match_id, 
        :is_winner,
        :kelly_stake => :simple_stake, 
        :mchale_stake
    ))


# --------------------------------------------- 


# 1. Gather all predictions and odds for the entire dataset (Over 2.5 only)
all_preds_over25 = subset(all_expr_inference.df, :selection => ByRow(==(:over_25)))

# Bring in odds and match metadata
all_odds = select(ds.odds, 
    :match_id, 
    :selection, 
    :is_winner, 
    :prob_fair_close => :market_devig,
    :odds_close
)
match_teams = select(ds.matches, :match_id, :home_team, :away_team)

# 2. Build the Master Matrix
master_matrix = innerjoin(all_preds_over25, all_odds, on=[:match_id, :selection])
master_matrix = innerjoin(master_matrix, match_teams, on=:match_id)

# Only keep rows where is_winner is known (completed matches)
dropmissing!(master_matrix, :is_winner)

println("Found $(nrow(master_matrix)) total completed matches for evaluation!")

# 3. Extract Trust Model Parameters (from the chain we already trained)
w0_draws = vec(Array(chain_trust[:w0]))
σ_draws  = vec(Array(chain_trust[:σ_team]))

l1_stakes = Float64[]
l1_returns = Float64[]
mchale_stakes = Float64[]
mchale_returns = Float64[]

b_kelly = BayesianFootball.Signals.BayesianKelly()

# 4. Big Loop over ALL matches
for row in eachrow(master_matrix)
    l1_dist = row.distribution
    q_mkt = row.market_devig
    odds  = row.odds_close
    b = odds - 1.0
    
    # --- A. PURE L1 MODEL (Naive Average of Kelly) ---
    raw_l1_draw_stakes = Float64[]
    for p in l1_dist
        stake = (p * (b + 1.0) - 1.0) / b
        push!(raw_l1_draw_stakes, max(0.0, stake))
    end
    l1_stake = mean(raw_l1_draw_stakes)
    push!(l1_stakes, l1_stake)
    
    if row.is_winner === true
        push!(l1_returns, l1_stake * b)
    else
        push!(l1_returns, -l1_stake)
    end
    
    # --- B. L2 TRUST CALIBRATED MODEL (BayesianKelly Optimizer) ---
    h_idx = get(team_map, row.home_team, 0)
    a_idx = get(team_map, row.away_team, 0)
    
    n_samples = min(length(w0_draws), length(l1_dist))
    match_composed_dist = Float64[]
    
    for s in 1:n_samples
        w0_s = w0_draws[s]
        σ_s  = σ_draws[s]
        
        # Apply the learned team hierarchies (0.0 if team wasn't in training set)
        h_z_s = h_idx > 0 ? chain_trust[Symbol("z_team[$h_idx]")][s] : 0.0
        a_z_s = a_idx > 0 ? chain_trust[Symbol("z_team[$a_idx]")][s] : 0.0
        
        η_s = w0_s + σ_s * (h_z_s + a_z_s)
        w_s = logistic(η_s)
        
        p_tilde_s = w_s * l1_dist[s] + (1.0 - w_s) * q_mkt
        push!(match_composed_dist, p_tilde_s)
    end
    
    # Use exact BayesianKelly optimizer on the composed distribution
    final_mchale_stake = BayesianFootball.Signals.compute_stake(b_kelly, match_composed_dist, odds)
    push!(mchale_stakes, final_mchale_stake)
    
    if row.is_winner === true
        push!(mchale_returns, final_mchale_stake * b)
    else
        push!(mchale_returns, -final_mchale_stake)
    end
end

master_matrix.l1_kelly_stake = l1_stakes
master_matrix.l1_returns = l1_returns

master_matrix.mchale_stake = mchale_stakes
master_matrix.mchale_returns = mchale_returns

println("=========================================================")
println("FULL HISTORICAL BACKTEST: $(nrow(master_matrix)) Matches")
println("Total L1 Return:       ", round(sum(master_matrix.l1_returns) * 100, digits=2), "%")
println("Total Trust Return:    ", round(sum(master_matrix.mchale_returns) * 100, digits=2), "%")
println("=========================================================")

# Sort by biggest McHale stakes to see the model's highest conviction bets
sort!(master_matrix, :mchale_stake, rev=true)
display(select(master_matrix, :match_id, :is_winner, :l1_kelly_stake, :mchale_stake, :l1_returns, :mchale_returns)[1:20, :])



#------------------------

    # 1. Sort chronologically so our Max Drawdown calculation is realistic
    sort!(master_matrix, :match_id)

    # 2. Trackers
    l1_bankroll = 1.0
    mchale_bankroll = 1.0

    l1_peak = 1.0
    mchale_peak = 1.0

    l1_max_drawdown = 0.0
    mchale_max_drawdown = 0.0

    # 3. Geometric Compounding Loop
    for row in eachrow(master_matrix)
        # Apply the return multiplier
        global l1_bankroll *= (1.0 + row.l1_returns)
        global mchale_bankroll *= (1.0 + row.mchale_returns)
        
        # Update all-time highs
        global l1_peak = max(l1_peak, l1_bankroll)
        global mchale_peak = max(mchale_peak, mchale_bankroll)
        
        # Calculate current drawdown from the peak
        l1_dd = (l1_peak - l1_bankroll) / l1_peak
        mchale_dd = (mchale_peak - mchale_bankroll) / mchale_peak
        
        # Track the worst drawdown experienced
        global l1_max_drawdown = max(l1_max_drawdown, l1_dd)
        global mchale_max_drawdown = max(mchale_max_drawdown, mchale_dd)
    end

    println("=========================================================")
    println("GEOMETRIC COMPOUNDING BACKTEST (Sequential Settlement)")
    println("=========================================================")
    println("STARTING BANKROLL: 1.00 (100%)")
    println("")
    println("--- RAW L1 MODEL ---")
    println("Final Bankroll:    ", round(l1_bankroll, digits=4))
    println("Geometric Return:  ", round((l1_bankroll - 1.0) * 100, digits=2), "%")
    println("Max Drawdown:      ", round(l1_max_drawdown * 100, digits=2), "%")
    println("")
    println("--- L2 TRUST + MCHALE ---")
    println("Final Bankroll:    ", round(mchale_bankroll, digits=4))
    println("Geometric Return:  ", round((mchale_bankroll - 1.0) * 100, digits=2), "%")
    println("Max Drawdown:      ", round(mchale_max_drawdown * 100, digits=2), "%")
    println("=========================================================")



# -----------
# Because the Bayesian math is perfectly symmetrical, we can evaluate the entire Totals market using the exact same trust parameters we already learned!
#   Since the model will only ever find a mathematical edge on one side of the line for a given match, it will seamlessly ignore the bad side and size the Kelly stake purely for the side with value.
#   Here is the combined block that filters for both  over_25  and  under_25 , applies the trust blending, and spits out both the Arithmetic and Geometric returns in one go:
    # 1. Gather all predictions for the entire Totals market (Over AND Under)
    all_preds_totals = subset(all_expr_inference.df, :selection => ByRow(x -> x in (:over_25, :under_25)))
    
    # Bring in odds and match metadata
    all_odds = select(ds.odds, :match_id, :selection, :is_winner, :prob_fair_close => :market_devig, :odds_close)
    match_teams = select(ds.matches, :match_id, :home_team, :away_team)
    
    # 2. Build the Master Matrix
    master_matrix = innerjoin(all_preds_totals, all_odds, on=[:match_id, :selection])
    master_matrix = innerjoin(master_matrix, match_teams, on=:match_id)
    dropmissing!(master_matrix, :is_winner)
    
    # Sort chronologically for geometric compounding!
    sort!(master_matrix, :match_id)
    
    println("Found $(nrow(master_matrix)) total completed Over/Under selections for evaluation!")
    
    # 3. Extract Trust Model Parameters (from the chain trained purely on over_25)
    w0_draws = vec(Array(chain_trust[:w0]))
    σ_draws  = vec(Array(chain_trust[:σ_team]))
    
    l1_returns = Float64[]
    mchale_returns = Float64[]
    b_kelly = BayesianFootball.Signals.BayesianKelly()
    
    # 4. Big Loop over ALL matches
    for row in eachrow(master_matrix)
        l1_dist = row.distribution
        q_mkt = row.market_devig
        odds  = row.odds_close
        b = odds - 1.0
        
        # --- A. PURE L1 MODEL ---
        raw_l1_draw_stakes = Float64[]
        for p in l1_dist
            stake = (p * (b + 1.0) - 1.0) / b
            push!(raw_l1_draw_stakes, max(0.0, stake))
        end
        l1_stake = mean(raw_l1_draw_stakes)
        push!(l1_returns, row.is_winner === true ? l1_stake * b : -l1_stake)
        
        # --- B. L2 TRUST CALIBRATED MODEL ---
        h_idx = get(team_map, row.home_team, 0)
        a_idx = get(team_map, row.away_team, 0)
        
        n_samples = min(length(w0_draws), length(l1_dist))
        match_composed_dist = Float64[]
        
        for s in 1:n_samples
            w0_s = w0_draws[s]
            σ_s  = σ_draws[s]
            
            # Apply team effects
            h_z_s = h_idx > 0 ? chain_trust[Symbol("z_team[$h_idx]")][s] : 0.0
            a_z_s = a_idx > 0 ? chain_trust[Symbol("z_team[$a_idx]")][s] : 0.0
            
            η_s = w0_s + σ_s * (h_z_s + a_z_s)
            w_s = logistic(η_s)
            
            # Symmetrical trust! Works perfectly for both Over and Under
            p_tilde_s = w_s * l1_dist[s] + (1.0 - w_s) * q_mkt
            push!(match_composed_dist, p_tilde_s)
        end
        
        # Compute Exact Kelly Stake
        final_mchale_stake = BayesianFootball.Signals.compute_stake(b_kelly, match_composed_dist, odds)
        push!(mchale_returns, row.is_winner === true ? final_mchale_stake * b : -final_mchale_stake)
    end

    master_matrix.l1_returns = l1_returns
    master_matrix.mchale_returns = mchale_returns

    # 5. Calculate Geometric Compounding
    l1_bankroll = 1.0; mchale_bankroll = 1.0
    l1_peak = 1.0; mchale_peak = 1.0
    l1_max_drawdown = 0.0; mchale_max_drawdown = 0.0

    for row in eachrow(master_matrix)
        global l1_bankroll *= (1.0 + row.l1_returns)
        global mchale_bankroll *= (1.0 + row.mchale_returns)
        
        global l1_peak = max(l1_peak, l1_bankroll)
        global mchale_peak = max(mchale_peak, mchale_bankroll)
        
        global l1_max_drawdown = max(l1_max_drawdown, (l1_peak - l1_bankroll) / l1_peak)
        global mchale_max_drawdown = max(mchale_max_drawdown, (mchale_peak - mchale_bankroll) / mchale_peak)
    end

    println("=========================================================")
    println("FULL HISTORICAL TOTALS MARKET (Over & Under 2.5)")
    println("Total Selections Evaluated: ", nrow(master_matrix))
    println("=========================================================")
    println("--- RAW L1 MODEL ---")
    println("Arithmetic Return: ", round(sum(master_matrix.l1_returns) * 100, digits=2), "%")
    println("Geometric Return:  ", round((l1_bankroll - 1.0) * 100, digits=2), "%")
    println("Max Drawdown:      ", round(l1_max_drawdown * 100, digits=2), "%")
    println("")
    println("--- L2 TRUST + MCHALE ---")
    println("Arithmetic Return: ", round(sum(master_matrix.mchale_returns) * 100, digits=2), "%")
    println("Geometric Return:  ", round((mchale_bankroll - 1.0) * 100, digits=2), "%")
    println("Max Drawdown:      ", round(mchale_max_drawdown * 100, digits=2), "%")
    println("=========================================================")
