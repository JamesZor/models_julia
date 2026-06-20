# current_development/MetaModels/src/metrics.jl
#
# Implements the Bernoulli-Gamma Hurdle model, Sharpe ratios (parametric and empirical), 
# and growth rates (parametric and empirical) for evaluation of regime filter outputs.

using DataFrames
using Statistics
using Distributions
using Random
using Printf
import BayesianFootball

export evaluate_predictive_hurdle, evaluate_multiple_markets, fit_hurdle_roi

function fit_hurdle_roi(stakes, pnls, odds, is_winner; mc_samples=100000)
    active_indices = findall(>(0.0), stakes)
    n_bets = length(active_indices)
    
    if n_bets == 0
        return (
            p = 0.0, shape = NaN, scale = NaN, μ_pos = 0.0, var_pos = 0.0,
            E_R_param = NaN, σ_R_param = NaN, Sharpe_param = NaN,
            E_R_emp = NaN, σ_R_emp = NaN, Sharpe_emp = NaN,
            G_emp = 0.0, G_param = 0.0, avg_stake = 0.0, n_bets = 0
        )
    end
    
    active_stakes = stakes[active_indices]
    active_pnls   = pnls[active_indices]
    active_odds   = odds[active_indices]
    
    rois = active_pnls ./ active_stakes
    
    wins = rois .> 0.0
    n_wins = sum(wins)
    p = n_wins / n_bets
    
    avg_stake = mean(active_stakes)
    
    E_R_emp = mean(rois)
    σ_R_emp = n_bets >= 2 ? std(rois) : 0.0
    Sharpe_emp = σ_R_emp > 0.0 ? E_R_emp / σ_R_emp : NaN
    
    log_wealth_increments = log.(max.(1e-8, 1.0 .+ active_stakes .* rois))
    G_emp = exp(mean(log_wealth_increments)) - 1.0
    
    pos_rois = rois[wins]
    
    shape_val = NaN
    scale_val = NaN
    μ_pos = 0.0
    var_pos = 0.0
    E_R_param = NaN
    σ_R_param = NaN
    Sharpe_param = NaN
    G_param = NaN
    
    if n_wins > 0
        if n_wins < 2 || var(pos_rois) == 0.0
            μ_pos = mean(pos_rois)
            var_pos = n_wins >= 2 ? var(pos_rois) : 1e-4
            if var_pos == 0.0
                var_pos = 1e-4
            end
            scale_val = var_pos / μ_pos
            shape_val = μ_pos / scale_val
        else
            try
                g_fit = fit(Gamma, pos_rois)
                shape_val = shape(g_fit)
                scale_val = scale(g_fit)
                μ_pos = mean(g_fit)
                var_pos = var(g_fit)
            catch e
                μ_pos = mean(pos_rois)
                var_pos = var(pos_rois)
                if var_pos == 0.0
                    var_pos = 1e-4
                end
                scale_val = var_pos / μ_pos
                shape_val = μ_pos / scale_val
            end
        end
        
        E_R_param = p * μ_pos - (1.0 - p)
        E_R2_param = p * (var_pos + μ_pos^2) + (1.0 - p)
        
        Var_R_param = E_R2_param - E_R_param^2
        σ_R_param = sqrt(max(0.0, Var_R_param))
        Sharpe_param = σ_R_param > 0.0 ? E_R_param / σ_R_param : NaN
        
        rng = Random.MersenneTwister(42)
        y_samples = rand(rng, Gamma(shape_val, scale_val), mc_samples)
        mean_log_wealth = mean(log.(max.(1e-8, 1.0 .+ avg_stake .* y_samples)))
        g_param = (1.0 - p) * log(max(1e-8, 1.0 - avg_stake)) + p * mean_log_wealth
        G_param = exp(g_param) - 1.0
    else
        E_R_param = -1.0
        σ_R_param = 0.0
        Sharpe_param = NaN
        g_param = log(max(1e-8, 1.0 - avg_stake))
        G_param = exp(g_param) - 1.0
    end
    
    return (
        p = p, shape = shape_val, scale = scale_val, μ_pos = μ_pos, var_pos = var_pos,
        E_R_param = E_R_param, σ_R_param = σ_R_param, Sharpe_param = Sharpe_param,
        E_R_emp = E_R_emp, σ_R_emp = σ_R_emp, Sharpe_emp = Sharpe_emp,
        G_emp = G_emp, G_param = G_param, avg_stake = avg_stake, n_bets = n_bets
    )
end

function _printf_row(group, bets, win_rate, avg_stake, emp_roi, param_roi, emp_sharpe, param_sharpe, emp_growth, param_growth)
    @printf(
        "%-21s | %-4s | %-6s | %-8s | %-8s | %-8s | %-9s | %-11s | %-9s | %-11s\n",
        group, bets, win_rate, avg_stake, emp_roi, param_roi, emp_sharpe, param_sharpe, emp_growth, param_growth
    )
end

function _print_metrics_row(group, m)
    if m.n_bets == 0
        _printf_row(group, "0", "-", "-", "-", "-", "-", "-", "-", "-")
    else
        win_rate_str = @sprintf("%.1f%%", m.p * 100)
        avg_stake_str = @sprintf("%.2f%%", m.avg_stake * 100)
        emp_roi_str = @sprintf("%.2f%%", m.E_R_emp * 100)
        param_roi_str = isnan(m.E_R_param) ? "NaN" : @sprintf("%.2f%%", m.E_R_param * 100)
        emp_sharpe_str = isnan(m.Sharpe_emp) ? "NaN" : @sprintf("%.4f", m.Sharpe_emp)
        param_sharpe_str = isnan(m.Sharpe_param) ? "NaN" : @sprintf("%.4f", m.Sharpe_param)
        emp_growth_str = @sprintf("%.3f%%", m.G_emp * 100)
        param_growth_str = isnan(m.G_param) ? "NaN" : @sprintf("%.3f%%", m.G_param * 100)
        
        _printf_row(
            group, string(m.n_bets), win_rate_str, avg_stake_str,
            emp_roi_str, param_roi_str, emp_sharpe_str, param_sharpe_str,
            emp_growth_str, param_growth_str
        )
    end
end

function _print_regime_params(group, m)
    if m.n_bets == 0
        println("  * $(rpad(group, 15)): No bets placed.")
    elseif isnan(m.shape)
        @printf("  * %-15s: p = %.4f | Gamma fit failed (no positive ROI variance)\n", group, m.p)
    else
        @printf("  * %-15s: p = %.4f | Gamma(α = %-7.4f, θ = %-7.4f) | E[Y] = %-7.4f (ROI if win)\n", 
                group, m.p, m.shape, m.scale, m.μ_pos)
    end
end

function evaluate_predictive_hurdle(ledger::DataFrame; min_edge=0.00, mc_samples=100000, market_name::String="Market")
    if !hasproperty(ledger, :distribution)
        error("Ledger DataFrame must contain the :distribution column. Please join with meta_results.all_data first.")
    end

    all_stakes = [BayesianFootball.Signals.compute_stake(BayesianFootball.Signals.BayesianKelly(min_edge=min_edge), dist, odds)
                  for (dist, odds) in zip(ledger.distribution, ledger.odds_close)]
    all_pnls = [s > 0.0 ? (Bool(w) ? s*(o-1.0) : -s) : 0.0
                for (s,w,o) in zip(all_stakes, ledger.is_winner, ledger.odds_close)]
    
    good_ledger = subset(ledger, :regime => ByRow(==("GOOD")))
    good_stakes = good_ledger.stake
    good_pnls = good_ledger.pnl
    good_odds = good_ledger.odds_close
    good_winner = good_ledger.is_winner

    bad_ledger = subset(ledger, :regime => ByRow(==("BAD")))
    bad_stakes = [BayesianFootball.Signals.compute_stake(BayesianFootball.Signals.BayesianKelly(min_edge=min_edge), dist, odds)
                  for (dist, odds) in zip(bad_ledger.distribution, bad_ledger.odds_close)]
    bad_pnls = [s > 0.0 ? (Bool(w) ? s*(o-1.0) : -s) : 0.0
                for (s,w,o) in zip(bad_stakes, bad_ledger.is_winner, bad_ledger.odds_close)]

    metrics_all  = fit_hurdle_roi(all_stakes, all_pnls, ledger.odds_close, ledger.is_winner; mc_samples=mc_samples)
    metrics_good = fit_hurdle_roi(good_stakes, good_pnls, good_odds, good_winner; mc_samples=mc_samples)
    metrics_bad  = fit_hurdle_roi(bad_stakes, bad_pnls, bad_ledger.odds_close, bad_ledger.is_winner; mc_samples=mc_samples)

    println("\n" * "="^115)
    println("  HURDLE METRICS (OOS) — Target Market: $market_name")
    println("="^115)
    
    _printf_row("Group", "Bets", "Win%", "AvgStake", "EmpROI", "ParamROI", "EmpSharpe", "ParamSharpe", "EmpGrowth", "ParamGrowth")
    println("-"^115)
    _print_metrics_row("L1 Raw (Unfiltered)", metrics_all)
    _print_metrics_row("Good Regime (Gated)", metrics_good)
    _print_metrics_row("Bad Regime (Skipped)", metrics_bad)
    println("-"^115)
    
    println("Fitted Hurdle Parameters (Bernoulli-Gamma):")
    _print_regime_params("L1 Raw", metrics_all)
    _print_regime_params("Good Regime", metrics_good)
    _print_regime_params("Bad Regime", metrics_bad)
    println("="^115)

    return (l1_raw = metrics_all, good_regime = metrics_good, bad_regime = metrics_bad)
end

function evaluate_multiple_markets(multi_ledger_dict::Dict{Symbol, DataFrame}, target_markets::Vector{Symbol}; min_edge=0.02)
    all_metrics = Dict{Symbol, NamedTuple}()
    for mkt in target_markets
        if haskey(multi_ledger_dict, mkt)
            ledger = multi_ledger_dict[mkt]
            if nrow(ledger) > 0
                res = evaluate_predictive_hurdle(ledger; min_edge=min_edge, market_name=string(mkt))
                all_metrics[mkt] = res
            else
                println("\nMarket $mkt has no ledger data.")
            end
        else
            println("\nMarket $mkt not found in the provided dictionary.")
        end
    end
    return all_metrics
end
