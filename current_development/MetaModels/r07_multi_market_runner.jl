# current_development/MetaModels/r07_multi_market_runner.jl
#
# Sequential multi-market runner for the Meta Model regime filter.
# Trains the Meta Model across 80 folds for each market in the TARGET_SELECTIONS list.
# Results and ledgers are persisted to disk via Serialization for later analysis.
#
# Run with: julia --project -t 32

using BayesianFootball
using DataFrames
using Dates
using Statistics
using Serialization
using LogExpFunctions: logistic

using ThreadPinning
pinthreads(:cores)

include("./current_development/MetaModels/src/MetaModels.jl")
# include("src/MetaModels.jl")
using .MetaModels

include("./current_development/MetaModels/src/staking.jl")
# include("src/staking.jl")

include("./current_development/MetaModels/l06_metrics.jl")

println("="^75)
println("  META MODEL — Multi-Market Sequential Runner (r07)")
println("="^75)

# ===========================================================================
# 1. LOAD DATA AND LAYER 1 RESULTS
# ===========================================================================
println("\n[1] Loading DataStore...")
ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

# use betfair odds instead 
println("\n[1] Loading Base DataStore...")
ds_raw = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.Ireland())

    println("\n[1b] Converting to Betfair Exchange Odds...")
    # Collapse the Betfair time series into a single sharp closing price 10 mins before kickoff
    betfair_odds_summary = BayesianFootball.Data.summarize_betfair_market(
	ds_raw,
	open_window=(-100000.0, -10.0),
	close_window=(-20.0, 0.0)
    )

    # Create the new DataStore using the Betfair summary instead of Sofascore
    ds = BayesianFootball.Data.DataStore(
	ds_raw.segment,
	ds_raw.matches,
	ds_raw.statistics,
	betfair_odds_summary, # <--- The Meta Model will now use Betfair!
	ds_raw.lineups,
	ds_raw.incidents,
	ds_raw.betfair_odds
    )

    # IMPORTANT: Ensure 'is_winner' is correctly populated for the new Betfair odds
    # You can paste the repair_is_winner!(ds) function we just wrote here, and run it:
    repair_is_winner!(ds)



println("[2] Loading Layer 1 Experiment Results...")
save_dir = "./data/meta_model_layer1/ireland/"
saved_files = BayesianFootball.Experiments.list_experiments(save_dir, data_dir="")
exp_results = BayesianFootball.Experiments.load_experiment(saved_files, 1)

# ===========================================================================
# 3. CONFIGURE META MODEL
# ===========================================================================
println("\n[3] Configuring ConvexMixtureMetaModel...")
meta_model = MetaModels.ConvexMixtureMetaModel(
    dynamics_config  = MetaModels.MetaGRWDynamicsConfig(σ_prior=0.2),
    hierarchy_config = MetaModels.GlobalMetaHierarchyConfig() 
)

sampler_config = BayesianFootball.Samplers.QueuedNUTSConfig(
    n_samples = 500,
    n_chains  = 2,
    n_warmup  = 200,
    accept_rate = 0.65,
    max_depth   = 10,
    initialisation = BayesianFootball.Samplers.UniformInit(-2, 2),
)

# ===========================================================================
# 4. RUN EXPERIMENTS ACROSS TARGET MARKETS
# ===========================================================================
# Define the markets to evaluate
# TARGET_SELECTIONS = [:under_15, :under_25, :under_35, :over_15, :over_25, :over_35]
# TARGET_SELECTIONS = [:under_15, :under_25]
# TARGET_SELECTIONS = [:DC_1X, :DC_12, :DC_X2]
#
TARGET_SELECTIONS = [:home, :draw, :away]


min_edge = 0.00
multi_market_results, multi_market_ledgers = MetaModels.run_multi_market_experiments(
    TARGET_SELECTIONS,
    exp_results,
    meta_model,
    sampler_config,
    ds;
    min_edge=min_edge
)

# ===========================================================================
# 5. PERSIST RESULTS TO DISK
# ===========================================================================
println("\n[5] Serializing all multi-market results to disk...")
serialize(joinpath(save_dir, "multi_market_results.jls"), multi_market_results)
serialize(joinpath(save_dir, "multi_market_ledgers.jls"), multi_market_ledgers)
println("Successfully saved to: $save_dir")

# ===========================================================================
# 6. STREAMLINED REPORTING
# ===========================================================================
println("\n" * "="^75)
println("  GENERATING CONSOLIDATED HURDLE REPORTS")
println("="^75)

# Evaluate and print the reports for all successfully evaluated markets
markets_evaluated = collect(keys(multi_market_ledgers))
final_metrics = MetaModels.evaluate_multiple_markets(multi_market_ledgers, markets_evaluated; min_edge=min_edge)






#=
julia> final_metrics = MetaModels.evaluate_multiple_markets(multi_market_ledgers, markets_evaluated; min_edge=min_edge)

===================================================================================================================
  HURDLE METRICS (OOS) — Target Market: under_35
===================================================================================================================
Group                 | Bets | Win%   | AvgStake | EmpROI   | ParamROI | EmpSharpe | ParamSharpe | EmpGrowth | ParamGrowth
-------------------------------------------------------------------------------------------------------------------
L1 Raw (Unfiltered)   | 123  | 74.8%  | 10.50%   | 2.12%    | 2.12%    | 0.0352    | 0.0354      | -0.642%   | 0.014%     
Good Regime (Gated)   | 31   | 77.4%  | 10.96%   | 5.58%    | 5.58%    | 0.0941    | 0.0961      | 1.303%    | 0.398%     
Bad Regime (Skipped)  | 92   | 73.9%  | 10.34%   | 0.96%    | 0.96%    | 0.0157    | 0.0158      | -1.288%   | -0.107%    
-------------------------------------------------------------------------------------------------------------------
Fitted Hurdle Parameters (Bernoulli-Gamma):
  * L1 Raw         : p = 0.7480 | Gamma(α = 11.5851, θ = 0.0315 ) | E[Y] = 0.3653  (ROI if win)
  * Good Regime    : p = 0.7742 | Gamma(α = 8.1999 , θ = 0.0444 ) | E[Y] = 0.3638  (ROI if win)
  * Bad Regime     : p = 0.7391 | Gamma(α = 13.5808, θ = 0.0269 ) | E[Y] = 0.3659  (ROI if win)
===================================================================================================================

===================================================================================================================
  HURDLE METRICS (OOS) — Target Market: over_35
===================================================================================================================
Group                 | Bets | Win%   | AvgStake | EmpROI   | ParamROI | EmpSharpe | ParamSharpe | EmpGrowth | ParamGrowth
-------------------------------------------------------------------------------------------------------------------
L1 Raw (Unfiltered)   | 220  | 21.4%  | 7.69%    | -6.47%   | -6.47%   | -0.0347   | -0.0348     | -1.315%   | -1.394%    
Good Regime (Gated)   | 52   | 23.1%  | 8.06%    | 2.62%    | 2.62%    | 0.0131    | 0.0133      | -2.089%   | -0.880%    
Bad Regime (Skipped)  | 168  | 20.8%  | 7.58%    | -9.29%   | -9.29%   | -0.0507   | -0.0510     | -1.074%   | -1.544%    
-------------------------------------------------------------------------------------------------------------------
Fitted Hurdle Parameters (Bernoulli-Gamma):
  * L1 Raw         : p = 0.2136 | Gamma(α = 10.3428, θ = 0.3266 ) | E[Y] = 3.3778  (ROI if win)
  * Good Regime    : p = 0.2308 | Gamma(α = 7.0607 , θ = 0.4882 ) | E[Y] = 3.4469  (ROI if win)
  * Bad Regime     : p = 0.2083 | Gamma(α = 12.3547, θ = 0.2715 ) | E[Y] = 3.3541  (ROI if win)
===================================================================================================================
=#




#=
close_window=(-20.0, 0.0)
julia> markets_evaluated = collect(keys(multi_market_ledgers))
2-element Vector{Symbol}:
 :over_25
 :under_25

julia> final_metrics = MetaModels.evaluate_multiple_markets(multi_market_ledgers, markets_evaluated; min_edge=min_edge)

===================================================================================================================
  HURDLE METRICS (OOS) — Target Market: over_25
===================================================================================================================
Group                 | Bets | Win%   | AvgStake | EmpROI   | ParamROI | EmpSharpe | ParamSharpe | EmpGrowth | ParamGrowth
-------------------------------------------------------------------------------------------------------------------
L1 Raw (Unfiltered)   | 234  | 44.0%  | 11.27%   | 1.16%    | 1.16%    | 0.0100    | 0.0100      | 0.207%    | -0.703%    
Good Regime (Gated)   | 114  | 43.9%  | 10.92%   | -1.23%   | -1.23%   | -0.0108   | -0.0108     | -0.234%   | -0.889%    
Bad Regime (Skipped)  | 120  | 44.2%  | 11.59%   | 3.44%    | 3.44%    | 0.0290    | 0.0291      | 0.628%    | -0.516%    
-------------------------------------------------------------------------------------------------------------------
Fitted Hurdle Parameters (Bernoulli-Gamma):
  * L1 Raw         : p = 0.4402 | Gamma(α = 17.1718, θ = 0.0756 ) | E[Y] = 1.2983  (ROI if win)
  * Good Regime    : p = 0.4386 | Gamma(α = 17.1631, θ = 0.0729 ) | E[Y] = 1.2519  (ROI if win)
  * Bad Regime     : p = 0.4417 | Gamma(α = 17.8869, θ = 0.0750 ) | E[Y] = 1.3420  (ROI if win)
===================================================================================================================

===================================================================================================================
  HURDLE METRICS (OOS) — Target Market: under_25
===================================================================================================================
Group                 | Bets | Win%   | AvgStake | EmpROI   | ParamROI | EmpSharpe | ParamSharpe | EmpGrowth | ParamGrowth
-------------------------------------------------------------------------------------------------------------------
L1 Raw (Unfiltered)   | 175  | 53.7%  | 8.04%    | 4.26%    | 4.26%    | 0.0432    | 0.0433      | -0.091%   | 0.028%     
Good Regime (Gated)   | 43   | 53.5%  | 5.28%    | 10.85%   | 10.85%   | 0.1019    | 0.1032      | 1.463%    | 0.418%     
Bad Regime (Skipped)  | 132  | 53.8%  | 8.94%    | 2.12%    | 2.12%    | 0.0219    | 0.0221      | -0.591%   | -0.182%    
-------------------------------------------------------------------------------------------------------------------
Fitted Hurdle Parameters (Bernoulli-Gamma):
  * L1 Raw         : p = 0.5371 | Gamma(α = 15.2508, θ = 0.0617 ) | E[Y] = 0.9411  (ROI if win)
  * Good Regime    : p = 0.5349 | Gamma(α = 16.6449, θ = 0.0644 ) | E[Y] = 1.0724  (ROI if win)
  * Bad Regime     : p = 0.5379 | Gamma(α = 16.7722, θ = 0.0536 ) | E[Y] = 0.8985  (ROI if win)
===================================================================================================================
=#




#=

close_window=(-60.0, 0.0)
julia> markets_evaluated = collect(keys(multi_market_ledgers))
2-element Vector{Symbol}:
 :over_25
 :under_25

julia> final_metrics = MetaModels.evaluate_multiple_markets(multi_market_ledgers, markets_evaluated; min_edge=min_edge)

===================================================================================================================
  HURDLE METRICS (OOS) — Target Market: over_25
===================================================================================================================
Group                 | Bets | Win%   | AvgStake | EmpROI   | ParamROI | EmpSharpe | ParamSharpe | EmpGrowth | ParamGrowth
-------------------------------------------------------------------------------------------------------------------
L1 Raw (Unfiltered)   | 245  | 44.1%  | 11.33%   | 1.72%    | 1.72%    | 0.0147    | 0.0148      | 0.015%    | -0.656%    
Good Regime (Gated)   | 89   | 42.7%  | 11.41%   | -0.21%   | -0.21%   | -0.0017   | -0.0018     | -0.535%   | -0.898%    
Bad Regime (Skipped)  | 156  | 44.9%  | 11.28%   | 2.82%    | 2.82%    | 0.0242    | 0.0243      | 0.330%    | -0.519%    
-------------------------------------------------------------------------------------------------------------------
Fitted Hurdle Parameters (Bernoulli-Gamma):
  * L1 Raw         : p = 0.4408 | Gamma(α = 16.9946, θ = 0.0769 ) | E[Y] = 1.3075  (ROI if win)
  * Good Regime    : p = 0.4270 | Gamma(α = 17.7664, θ = 0.0753 ) | E[Y] = 1.3373  (ROI if win)
  * Bad Regime     : p = 0.4487 | Gamma(α = 16.7203, θ = 0.0772 ) | E[Y] = 1.2914  (ROI if win)
===================================================================================================================

===================================================================================================================
  HURDLE METRICS (OOS) — Target Market: under_25
===================================================================================================================
Group                 | Bets | Win%   | AvgStake | EmpROI   | ParamROI | EmpSharpe | ParamSharpe | EmpGrowth | ParamGrowth
-------------------------------------------------------------------------------------------------------------------
L1 Raw (Unfiltered)   | 186  | 54.8%  | 7.98%    | 5.95%    | 5.95%    | 0.0607    | 0.0609      | 0.265%    | 0.169%     
Good Regime (Gated)   | 50   | 52.0%  | 6.49%    | 5.78%    | 5.78%    | 0.0555    | 0.0561      | 0.668%    | 0.151%     
Bad Regime (Skipped)  | 136  | 55.9%  | 8.53%    | 6.02%    | 6.02%    | 0.0625    | 0.0629      | 0.117%    | 0.177%     
-------------------------------------------------------------------------------------------------------------------
Fitted Hurdle Parameters (Bernoulli-Gamma):
  * L1 Raw         : p = 0.5484 | Gamma(α = 15.3377, θ = 0.0608 ) | E[Y] = 0.9321  (ROI if win)
  * Good Regime    : p = 0.5200 | Gamma(α = 18.5286, θ = 0.0558 ) | E[Y] = 1.0342  (ROI if win)
  * Bad Regime     : p = 0.5588 | Gamma(α = 15.6553, θ = 0.0573 ) | E[Y] = 0.8972  (ROI if win)
===================================================================================================================
=#








println("\nRunner finished successfully.")


ds.odds
unique(ds.odds.selection)


  using DataFrames
    function repair_is_winner!(ds)
        println("Cross-referencing ds.odds with ds.matches actual scores...")
        # Create a lookup dictionary for match scores for O(1) access
        score_lookup = Dict(r.match_id => (r.home_score, r.away_score) for r in eachrow(ds.matches))
        fixed_count = 0
        missing_count = 0
        for i in 1:nrow(ds.odds)
            match_id = ds.odds.match_id[i]
            sel = ds.odds.selection[i]
            # Skip if match data is somehow missing
            if !haskey(score_lookup, match_id)
                continue
            end
            h, a = score_lookup[match_id]
            total = h + a
            computed_winner = missing
            # 1. Standard 1X2
            if sel == :home
                computed_winner = h > a
            elseif sel == :draw
                computed_winner = h == a
            elseif sel == :away
                computed_winner = h < a
            # 2. Double Chance (DC)
            elseif sel == :DC_1X
                computed_winner = h >= a
            elseif sel == :DC_X2
                computed_winner = h <= a
            elseif sel == :DC_12
                computed_winner = h != a

            # 3. Both Teams to Score (BTTS)
            elseif sel == :btts_yes
		computed_winner = (h > 0) && (a > 0)
            elseif sel == :btts_no
		computed_winner = (h == 0) || (a == 0)

            # 4. Over / Under Markets
            else
                sel_str = string(sel)
		if startswith(sel_str, "over_") || startswith(sel_str, "under_")
                    # Parse the line (e.g. "under_25" -> 2.5)
                    line = parse(Float64, sel_str[end-1:end]) / 10.0

                    if startswith(sel_str, "over_")
			computed_winner = total > line
                    else
                        computed_winner = total < line
                    end
		end
            end

            # Apply the fix if it's a known market
            if !ismissing(computed_winner)
		# Check if it was wrong (or missing previously)
		if ismissing(ds.odds.is_winner[i]) || ds.odds.is_winner[i] != computed_winner
                    ds.odds.is_winner[i] = computed_winner
                    fixed_count += 1
		end
            else
                missing_count += 1
            end
	end

	println("Repair Complete!")
	println("  -> Fixed/Updated: $fixed_count rows.")
	println("  -> Ignored (Messy strings): $missing_count rows.")
    end

    # Run the repair function
    repair_is_winner!(ds)
