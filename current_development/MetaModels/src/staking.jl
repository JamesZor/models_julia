# current_development/MetaModels/src/staking.jl
#
# Generates the posterior distribution of Q_i = θ_t * p_L1 + (1 - θ_t) * m_i
# for each match in joined_data, then applies BayesianKelly staking.
#
# This is the key mathematical advantage of the full Bayesian approach:
# instead of a point estimate of Q, we pass the full posterior vector to
# BayesianKelly, which uses McHale's shrinkage to account for uncertainty.

export compute_meta_stakes, meta_ledger

using Statistics
using LogExpFunctions: logistic
using BayesianFootball.Signals: BayesianKelly, AnalyticalShrinkageKelly, compute_stake
using DataFrames

"""
    extract_Q_posterior(chain, joined_data) -> Matrix{Float64}

For each match in `joined_data`, reconstruct the full posterior distribution
of the mixture probability Q_i = θ_t * p_L1_i + (1 - θ_t) * m_i.

Returns a Matrix of size (n_samples, n_matches).
Each column is the posterior distribution of Q for that match.
"""
function extract_Q_posterior(chain, joined_data::DataFrame)
    n_matches = nrow(joined_data)
    n_weeks   = maximum(joined_data.W)

    # Extract raw posterior samples for all parameters
    α_samples     = vec(chain[:α_intercept])
    σ_GRW_samples = vec(chain[Symbol("dyn_θ_logit.σ_GRW")])
    n_samples     = length(α_samples)

    # Extract all z_w samples: (n_samples, n_weeks)
    z_w_mat = hcat([vec(chain[Symbol("dyn_θ_logit.z_w[$w]")]) for w in 1:n_weeks]...)

    # Reconstruct weekly θ_t for all posterior samples
    # Non-centered: cumsum(z_w .* σ_GRW) centered, then logistic(α + drift)
    θ_t_mat = Matrix{Float64}(undef, n_samples, n_weeks)
    for s in 1:n_samples
        drift = cumsum(z_w_mat[s, :] .* σ_GRW_samples[s])
        drift_centered = drift .- mean(drift)
        θ_t_mat[s, :] = logistic.(α_samples[s] .+ drift_centered)
    end

    # For each match, Q_i^(s) = θ_t_w^(s) * p_L1_i + (1 - θ_t_w^(s)) * m_i
    Q_posterior = Matrix{Float64}(undef, n_samples, n_matches)
    for i in 1:n_matches
        w    = joined_data.W[i]
        p_l1 = clamp(joined_data.p_L1[i], 1e-5, 1.0 - 1e-5)
        m_i  = clamp(joined_data.prob_fair_close[i], 1e-5, 1.0 - 1e-5)
        Q_posterior[:, i] = θ_t_mat[:, w] .* p_l1 .+ (1.0 .- θ_t_mat[:, w]) .* m_i
    end

    return Q_posterior
end

"""
    compute_meta_stakes(chain, joined_data; signal, min_edge) -> DataFrame

Applies the given staking signal to the full Meta Model posterior Q distribution.
Outputs a ledger DataFrame with stakes and PnL.

Arguments:
- `chain`: The MCMC chain from run_meta_experiment
- `joined_data`: The joined DataFrame (must have :W, :p_L1, :prob_fair_close, :odds_close, :is_winner)
- `signal`: An AbstractSignal (default: BayesianKelly with min_edge)
- `min_edge`: Minimum edge threshold (as a fraction, e.g. 0.02 = 2%)
"""
function compute_meta_stakes(
    chain,
    joined_data::DataFrame;
    signal = BayesianKelly(min_edge=0.02),
    verbose::Bool = true
)
    verbose && println("Extracting full Q posterior ($(nrow(joined_data)) matches)...")
    Q_post = extract_Q_posterior(chain, joined_data)

    n_matches = nrow(joined_data)
    stakes    = Vector{Float64}(undef, n_matches)
    q_means   = Vector{Float64}(undef, n_matches)
    q_stds    = Vector{Float64}(undef, n_matches)

    verbose && println("Computing $(typeof(signal)) stakes...")
    for i in 1:n_matches
        q_dist     = Q_post[:, i]
        odds       = joined_data.odds_close[i]
        stakes[i]  = compute_stake(signal, q_dist, odds)
        q_means[i] = mean(q_dist)
        q_stds[i]  = std(q_dist)
    end

    # Build ledger
    ledger = copy(joined_data[!, [:match_id, :selection, :home_team, :away_team,
                                   :match_date, :W, :p_L1, :prob_fair_close,
                                   :odds_close, :is_winner]])
    ledger.Q_mean  = q_means
    ledger.Q_std   = q_stds
    ledger.Q_edge  = q_means .- (1.0 ./ ledger.odds_close)   # vs market implied
    ledger.L1_edge = ledger.p_L1 .- (1.0 ./ ledger.odds_close)
    ledger.stake   = stakes
    ledger.pnl     = [
        stakes[i] > 0 ?
        (Bool(joined_data.is_winner[i]) ? stakes[i] * (joined_data.odds_close[i] - 1.0) : -stakes[i]) :
        0.0
        for i in 1:n_matches
    ]

    return ledger
end

"""
    meta_ledger_summary(ledger::DataFrame) -> Nothing

Prints a headless-friendly summary of the Meta Model staking ledger.
"""
function meta_ledger_summary(ledger::DataFrame; label::String="Meta Model")
    bets    = subset(ledger, :stake => ByRow(>(0.0)))
    n_bets  = nrow(bets)
    n_total = nrow(ledger)

    println("\n" * "="^65)
    println("  STAKING LEDGER: $label")
    println("="^65)
    println("  Matches evaluated: $n_total")
    println("  Bets placed:       $n_bets  ($(round(n_bets/n_total*100, digits=1))% bet rate)")

    if n_bets == 0
        println("  >> No bets placed (edge filter too tight or no Q > implied)")
        return
    end

    total_stake = sum(bets.stake)
    total_pnl   = sum(bets.pnl)
    roi         = total_pnl / total_stake * 100
    win_rate    = mean(Bool.(bets.is_winner)) * 100
    avg_Q       = mean(bets.Q_mean)
    avg_impl    = mean(1.0 ./ bets.odds_close)
    avg_edge    = mean(bets.Q_edge)

    println("  Total Turnover:    $(round(total_stake, digits=4)) units")
    println("  Total PnL:         $(round(total_pnl,   digits=4)) units")
    println("  ROI:               $(round(roi, digits=2))%")
    println("  Win Rate:          $(round(win_rate, digits=1))%")
    println("  Avg Q_mean:        $(round(avg_Q, digits=4))")
    println("  Avg Implied Prob:  $(round(avg_impl, digits=4))")
    println("  Avg Q Edge:        $(round(avg_edge, digits=4))")

    # Weekly breakdown
    println("\n  Weekly PnL (first 10 active weeks):")
    weekly = combine(groupby(bets, :W),
        :pnl   => sum   => :week_pnl,
        :stake => sum   => :week_stake,
        :match_id => length => :n_bets
    )
    sort!(weekly, :W)
    for row in first(eachrow(weekly), 10)
        cum_roi = row.week_pnl / row.week_stake * 100
        bar = row.week_pnl > 0 ? "▲" : "▼"
        println("  $bar Week $(lpad(row.W, 3)): PnL=$(rpad(string(round(row.week_pnl, digits=3)), 8)) | $(row.n_bets) bets | ROI=$(round(cum_roi, digits=1))%")
    end

    # Cumulative PnL by week
    sort!(bets, :match_date)
    cum_pnl = cumsum(bets.pnl)
    best_cum  = round(maximum(cum_pnl), digits=4)
    worst_cum = round(minimum(cum_pnl), digits=4)
    final_cum = round(last(cum_pnl), digits=4)
    println("\n  Cumulative PnL: Best=$best_cum | Worst=$worst_cum | Final=$final_cum")
    println("="^65)
end

"""
    compute_predictive_stakes(meta_results, all_data; min_edge=0.02) -> DataFrame

A 1-step-ahead regime filter.
Uses fold `k` to predict regime for fold `k+1`.
Applies the Bet Gate: only bets if predicted θ >= expanding median.
Returns a ledger of performance on the OOS folds.
"""
function compute_predictive_stakes(
    meta_results,
    all_data::DataFrame;
    min_edge=0.02,
    staking_odds::Union{DataFrame, Nothing}=nothing
)
    return compute_predictive_stakes(
        meta_results, 
        all_data, 
        meta_results.task.model; 
        min_edge=min_edge, 
        staking_odds=staking_odds
    )
end

function compute_predictive_stakes(
    meta_results,
    all_data::DataFrame,
    model::ConvexMixtureMetaModel;
    min_edge=0.02,
    staking_odds::Union{DataFrame, Nothing}=nothing
)
    n_folds = length(meta_results.fold_results)
    
    # 1. Establish the "Cold Start" prior from Fold 1
    fold1_chain = meta_results.fold_results[1].chain
    α_prior_mean = mean(fold1_chain[:α_intercept])
    θ_prior = logistic(α_prior_mean)
    
    # Pre-compute θ_pred for all folds concurrently
    θ_preds = zeros(n_folds - 1)
    Threads.@threads for k in 1:(n_folds - 1)
        chain = meta_results.fold_results[k].chain
        fold_data = meta_results.fold_results[k].fold_data
        w_k = maximum(fold_data.W)
        
        α_samples     = vec(chain[:α_intercept])
        σ_GRW_samples = vec(chain[Symbol("dyn_θ_logit.σ_GRW")])
        n_samples     = length(α_samples)
        
        # We only need z_w up to w_k
        z_w_mat = hcat([vec(chain[Symbol("dyn_θ_logit.z_w[$w]")]) for w in 1:w_k]...)
        
        θ_wk_samples = zeros(n_samples)
        for s in 1:n_samples
            drift = cumsum(z_w_mat[s, :] .* σ_GRW_samples[s])
            drift_centered = drift .- mean(drift)
            θ_wk_samples[s] = logistic(α_samples[s] + drift_centered[end])
        end
        
        θ_preds[k] = mean(θ_wk_samples)
    end
    
    history_of_θ = Float64[θ_prior]
    ledgers = DataFrame[]
    
    # Optional Staking Odds Override
    eval_data = copy(all_data)
    if !isnothing(staking_odds)
        stake_df = staking_odds[!, [:match_id, :selection, :odds_close]]
        rename!(stake_df, :odds_close => :stake_odds_close)
        eval_data = innerjoin(eval_data, stake_df, on=[:match_id, :selection])
    else
        eval_data.stake_odds_close = eval_data.odds_close
    end
    
    # We evaluate matches starting from fold 2
    for k in 1:(n_folds - 1)
        θ_pred = θ_preds[k]
        
        # The threshold is the expanding median of all history seen so far
        current_threshold = median(history_of_θ)
        is_good_regime = θ_pred >= current_threshold
        
        # Evaluate on the NEXT fold (fold k+1)
        next_fold_data = subset(eval_data, :fold_idx => ByRow(==(k+1)))
        
        if nrow(next_fold_data) > 0
            stakes = zeros(nrow(next_fold_data))
            if is_good_regime
                for i in 1:nrow(next_fold_data)
                    dist = next_fold_data.distribution[i]
                    odds = next_fold_data.stake_odds_close[i]
                    stakes[i] = compute_stake(BayesianKelly(min_edge=min_edge), dist, odds)
                end
            end
            
            pnls = [s > 0 ? (Bool(w) ? s*(o-1.0) : -s) : 0.0
                    for (s,w,o) in zip(stakes, next_fold_data.is_winner, next_fold_data.stake_odds_close)]
                    
            cols_to_copy = intersect(names(next_fold_data), 
                ["match_id", "selection", "home_team", "away_team", "match_date", 
                 "W", "p_L1", "prob_fair_close", "odds_close", "stake_odds_close", "is_winner", "distribution"])
            
            ledger = copy(next_fold_data[!, cols_to_copy])
            ledger.θ_pred    .= θ_pred
            ledger.threshold .= current_threshold
            ledger.regime    .= is_good_regime ? "GOOD" : "BAD"
            ledger.stake     = stakes
            ledger.pnl       = pnls
            
            # Revert column name for standard reporting downstream
            rename!(ledger, :stake_odds_close => :odds_close, :odds_close => :anchor_odds)

            
            push!(ledgers, ledger)
        end
        
        # Append the new prediction to the history for the next iteration
        push!(history_of_θ, θ_pred)
    end
    
    if isempty(ledgers)
        return DataFrame()
    end
    
    return vcat(ledgers...)
end

function compute_predictive_stakes(
    meta_results,
    all_data::DataFrame,
    model::AffineCalibrationMetaModel;
    min_edge=0.02,
    staking_odds::Union{DataFrame, Nothing}=nothing
)
    n_folds = length(meta_results.fold_results)
    ledgers = DataFrame[]
    
    # Optional Staking Odds Override
    eval_data = copy(all_data)
    if !isnothing(staking_odds)
        stake_df = staking_odds[!, [:match_id, :selection, :odds_close]]
        rename!(stake_df, :odds_close => :stake_odds_close)
        eval_data = innerjoin(eval_data, stake_df, on=[:match_id, :selection])
    else
        eval_data.stake_odds_close = eval_data.odds_close
    end

    # We evaluate matches starting from fold 2
    for k in 1:(n_folds - 1)
        chain = meta_results.fold_results[k].chain
        fold_data = meta_results.fold_results[k].fold_data
        w_k = maximum(fold_data.W)
        
        # 1. Extract Posterior of the L2 Model
        α_samples     = vec(chain[:α_intercept])
        β_samples     = vec(chain[:β_intercept])
        
        n_samples = length(α_samples)
        
        # Determine if we have dynamic GRW
        has_dynamics = Symbol("dyn_α_logit.σ_GRW") in names(chain, :parameters)
        α_wk_samples = copy(α_samples)
        if has_dynamics
            σ_GRW_samples = vec(chain[Symbol("dyn_α_logit.σ_GRW")])
            z_w_mat = hcat([vec(chain[Symbol("dyn_α_logit.z_w[$w]")]) for w in 1:w_k]...)
            for s in 1:n_samples
                drift = cumsum(z_w_mat[s, :] .* σ_GRW_samples[s])
                drift_centered = drift .- mean(drift)
                α_wk_samples[s] += drift_centered[end]
            end
        end
        
        has_teams = Symbol("δ_team.σ_team") in names(chain, :parameters)

        next_fold_data = subset(eval_data, :fold_idx => ByRow(==(k+1)))
        
        if nrow(next_fold_data) > 0
            stakes = zeros(nrow(next_fold_data))
            
            for i in 1:nrow(next_fold_data)
                h = next_fold_data.home_idx[i]
                a = next_fold_data.away_idx[i]
                
                p_L1 = clamp(next_fold_data.p_L1[i], 1e-5, 1.0 - 1e-5)
                logit_p = logit(p_L1)
                
                # Generate full posterior of calibrated probabilities
                Q_posterior = zeros(n_samples)
                
                for s in 1:n_samples
                    t_bias = 0.0
                    if has_teams && h > 0 && a > 0
                        σ_team = chain[Symbol("δ_team.σ_team")][s]
                        z_h = chain[Symbol("δ_team.z_team[$h]")][s]
                        z_a = chain[Symbol("δ_team.z_team[$a]")][s]
                        t_bias = (z_h + z_a) * σ_team
                    end
                    
                    α_s = α_wk_samples[s] + t_bias
                    β_s = β_samples[s]
                    
                    logit_Q = α_s + (β_s * logit_p)
                    Q_posterior[s] = logistic(logit_Q)
                end
                
                # Bet Gate: Affine is fully calibrated, so ALL matches are valid to check Kelly
                odds = next_fold_data.stake_odds_close[i]
                stakes[i] = compute_stake(BayesianKelly(min_edge=min_edge), Q_posterior, odds)
            end
            
            pnls = [s > 0 ? (Bool(w) ? s*(o-1.0) : -s) : 0.0
                    for (s,w,o) in zip(stakes, next_fold_data.is_winner, next_fold_data.stake_odds_close)]
                    
            cols_to_copy = intersect(names(next_fold_data), 
                ["match_id", "selection", "home_team", "away_team", "match_date", 
                 "W", "p_L1", "prob_fair_close", "odds_close", "stake_odds_close", "is_winner", "distribution"])
            
            ledger = copy(next_fold_data[!, cols_to_copy])
            ledger.θ_pred    .= 1.0 # Affine doesn't use theta regime
            ledger.threshold .= 0.0
            ledger.regime    .= "CALIBRATED"
            ledger.stake     = stakes
            ledger.pnl       = pnls
            
            rename!(ledger, :stake_odds_close => :odds_close, :odds_close => :anchor_odds)
            
            push!(ledgers, ledger)
        end
    end
    
    if isempty(ledgers)
        return DataFrame()
    end
    
    return vcat(ledgers...)
end

