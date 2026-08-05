# r07_risk_alpha_lambda_sweep.jl

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics
using ThreadPinning
using CSV

include("l03_risk_manager.jl")

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
# 2. Configuration Setup
# -------------------------------------------------------------------
scalar_markets = D.AbstractMarket[D.Market1X2(), D.MarketBTTS()]
over_unders = [D.MarketOverUnder(i + 0.5) for i in 0:4]
markets_config = D.MarketConfig(reduce(vcat, (scalar_markets, over_unders)))

b_config = BacktestConfig(commission=0.02, alphas=[1.0])

alphas = collect(1.0:-0.05:0.05) # We skip 0.0 as it produces no stakes
lambdas = [0.0, 1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 100.0]

println("\n", "="^80)
println("=== RUNNING 2D ALPHA & LAMBDA SWEEP ===")
println("Total Matches : $(n_matches)")
println("Combinations  : $(length(alphas) * length(lambdas))")
println("="^80)
println("Optimizing loop: Matrix calculations will be cached per match for extreme speed...")

# -------------------------------------------------------------------
# 3. Highly Optimized 2D Sweep Logic (Caches the Score Matrix)
# -------------------------------------------------------------------
function evaluate_match_2D_sweep(latents_row, expr, odds_df, markets_config, b_config, alphas, lambdas)
    m_id = latents_row.match_id
    
    # Pre-allocate return structure for all combinations
    n_combs = length(alphas) * length(lambdas)
    res_df = DataFrame(
        match_id   = fill(m_id, n_combs),
        alpha      = zeros(Float64, n_combs),
        lambda     = zeros(Float64, n_combs),
        risk_stake = zeros(Float64, n_combs),
        shrink_k   = zeros(Float64, n_combs),
        risk_pl    = zeros(Float64, n_combs),
        status     = fill("", n_combs)
    )
    
    # 1. Fast-Fail if no odds
    raw_odds_map, _, winner_map = extract_market_data(odds_df, m_id, markets_config)
    if isempty(raw_odds_map)
        res_df.status .= "MISSING_ODDS"
        return res_df
    end
    
    odds_map, fair_prob_map = normalize_market_group_odds(raw_odds_map)
    
    # 2. Compute Score Matrix (Exactly ONCE per match)
    param = Predictions.extract_params(expr.config.model, latents_row)
    score_matrix = nothing
    try
        score_matrix = Predictions.compute_score_matrix(expr.config.model, param)
    catch e
        res_df.status .= "SOLVER_FAIL"
        return res_df
    end
    
    match_model_prob = Dict(
        string(m) => Predictions.compute_market_probs(score_matrix, m)
        for m in markets_config.markets
    )
    
    P_model_grid = mean(score_matrix.data, dims=3)[:, :, 1]
    p_model_vec  = vec(P_model_grid)
    
    # 3. 2D Loop 
    idx = 1
    for α in alphas
        selections, base_stakes, R_mat = optimize_portfolio(score_matrix, match_model_prob, odds_map, fair_prob_map, winner_map, b_config; alpha=α)
        
        if isempty(selections)
            for λ in lambdas
                res_df.alpha[idx] = α
                res_df.lambda[idx] = λ
                res_df.status[idx] = "NO_SELECTIONS"
                idx += 1
            end
            continue
        end
        
        returns_vec = R_mat * base_stakes
        
        for λ in lambdas
            k_shrink = solve_drawdown_multiplier(p_model_vec, returns_vec, λ)
            risk_stakes = base_stakes .* k_shrink
            
            risk_net_pl = 0.0
            for i in 1:length(selections)
                sel = selections[i]
                rst = risk_stakes[i]
                if rst > 0
                    if sel.is_winner
                        risk_net_pl += rst * (1.0 - b_config.commission) * (sel.odds - 1.0)
                    else
                        risk_net_pl -= rst
                    end
                end
            end
            
            res_df.alpha[idx] = α
            res_df.lambda[idx] = λ
            res_df.risk_stake[idx] = sum(risk_stakes)
            res_df.shrink_k[idx] = k_shrink
            res_df.risk_pl[idx] = risk_net_pl
            res_df.status[idx] = "SUCCESS"
            
            idx += 1
        end
    end
    
    return res_df
end

# Multithreaded execution across all matches
results_2D = Vector{DataFrame}(undef, n_matches)
@time Threads.@threads for i in 1:n_matches
    results_2D[i] = evaluate_match_2D_sweep(latents.df[i, :], expr, odds, markets_config, b_config, alphas, lambdas)
end
raw_sweep_df = vcat(results_2D...)

# -------------------------------------------------------------------
# 4. Aggregation and Summarization
# -------------------------------------------------------------------
println("\nAggregating $(length(alphas) * length(lambdas)) combinations...")

valid_matches = subset(raw_sweep_df, :status => ByRow(==("SUCCESS")))
sort!(valid_matches, :match_id) # chronological proxy

sweep_summary = combine(groupby(valid_matches, [:alpha, :lambda]),
    :shrink_k => mean => :Avg_Shrinkage,
    :risk_stake => sum => :Total_Stake,
    :risk_pl => sum => :Net_PL
)

sweep_summary.ROI = ifelse.(sweep_summary.Total_Stake .> 0, (sweep_summary.Net_PL ./ sweep_summary.Total_Stake) .* 100, 0.0)

# Calculate complex compounding metrics per combination
final_banks = Float64[]
max_dds = Float64[]
sharpes = Float64[]
calmars = Float64[]
martins = Float64[]

for row in eachrow(sweep_summary)
    sub = subset(valid_matches, :alpha => ByRow(==(row.alpha)), :lambda => ByRow(==(row.lambda)))
    returns = sub.risk_pl
    
    if length(returns) > 0
        bankroll = [1.0; cumprod(1.0 .+ returns)]
        run_max = accumulate(max, bankroll)
        dd = (bankroll .- run_max) ./ run_max
        dd_pct = dd .* 100.0
        
        fb = bankroll[end]
        mdd_pct = minimum(dd_pct)
        ulcer_index = sqrt(mean(dd_pct .^ 2))
        
        tot_ret = (fb - 1.0) * 100.0
        calmar = mdd_pct < 0.0 ? (tot_ret / abs(mdd_pct)) : 0.0
        martin = ulcer_index > 0.0 ? (tot_ret / ulcer_index) : 0.0
        
        push!(final_banks, fb)
        push!(max_dds, mdd_pct)
        push!(sharpes, std(returns) > 0 ? (mean(returns) / std(returns)) : 0.0)
        push!(calmars, calmar)
        push!(martins, martin)
    else
        push!(final_banks, 1.0)
        push!(max_dds, 0.0)
        push!(sharpes, 0.0)
        push!(calmars, 0.0)
        push!(martins, 0.0)
    end
end

sweep_summary.Final_Bankroll = final_banks
sweep_summary.Max_Drawdown = max_dds
sweep_summary.Sharpe = sharpes
sweep_summary.Calmar = calmars
sweep_summary.Martin = martins

# Format
sweep_summary.Avg_Shrinkage  = round.(sweep_summary.Avg_Shrinkage, digits=3)
sweep_summary.Total_Stake    = round.(sweep_summary.Total_Stake, digits=1)
sweep_summary.Net_PL         = round.(sweep_summary.Net_PL, digits=2)
sweep_summary.ROI            = round.(sweep_summary.ROI, digits=2)
sweep_summary.Final_Bankroll = round.(sweep_summary.Final_Bankroll, digits=2)
sweep_summary.Max_Drawdown   = round.(sweep_summary.Max_Drawdown, digits=2)
sweep_summary.Sharpe         = round.(sweep_summary.Sharpe, digits=3)
sweep_summary.Calmar         = round.(sweep_summary.Calmar, digits=3)
sweep_summary.Martin         = round.(sweep_summary.Martin, digits=3)

# -------------------------------------------------------------------
# 5. Display & Export
# -------------------------------------------------------------------
println("\n", "="^80)
println("=== 2D SWEEP COMPLETE: TOP 20 CONFIGURATIONS BY MARTIN RATIO ===")
println("="^80)

# Sort by Martin Ratio descending
sorted_summary = sort(sweep_summary, :Martin, rev=true)
display(first(sorted_summary, 20))

out_file = "current_development/portfolio_explore/r07_alpha_lambda_sweep.csv"
CSV.write(out_file, sorted_summary)
println("\n✓ Full 2D sweep results exported to $(out_file)")
