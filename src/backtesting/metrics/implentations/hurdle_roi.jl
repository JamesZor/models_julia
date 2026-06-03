# src/backtesting/metrics/implentations/hurdle_roi.jl
#
# Bernoulli-Gamma Hurdle Model for Betting ROI
#
# Mathematical Model:
#   ROI ~ p · Gamma(α, β) + (1-p) · δ(-1)
#
# Where:
#   p     : Probability of a winning bet (Bernoulli parameter)
#   α, β  : Shape and scale of the Gamma distribution over positive ROIs
#   δ(-1) : Point mass at -1 (losing the entire stake)
#
# Key Outputs:
#   - Parametric Sharpe:  E[R] / σ[R]  derived from the fitted distribution
#   - Parametric Growth:  G = exp(E[log(1 + f·R)]) - 1  via Monte Carlo integration

using Distributions
using Statistics
using Random

export BernoulliGammaHurdle

"""
    BernoulliGammaHurdle <: AbstractDistributionalMetric

Fits a Bernoulli-Gamma Hurdle model to per-bet ROI data.

The model decomposes bet outcomes into:
- Win probability `p` (Bernoulli)
- Positive return distribution `Gamma(α, β)` 
- Loss at `-1` (full stake lost)

Returns fitted parameters, parametric/empirical Sharpe ratios, and geometric growth rates.

# Fields
- `mc_samples::Int`: Number of Monte Carlo samples for parametric growth rate (default: 100,000)
- `min_bets::Int`: Minimum active bets required to attempt fit (default: 5)
"""
Base.@kwdef struct BernoulliGammaHurdle <: AbstractDistributionalMetric
    mc_samples::Int = 100_000
    min_bets::Int = 5
end

function metric_description(m::BernoulliGammaHurdle)::String
    return "Bernoulli-Gamma Hurdle: ROI ~ p·Gamma(α,β) + (1-p)·δ(-1). " *
           "Estimates parametric Sharpe and geometric growth rate via MC integration."
end

"""
    compute_distributional_metric(metric::BernoulliGammaHurdle, sub_df::AbstractDataFrame)

Fits the hurdle model to the grouped sub-DataFrame.
Returns a NamedTuple with `hurdle_`-prefixed keys for tearsheet integration.
"""
function compute_distributional_metric(metric::BernoulliGammaHurdle, sub_df::AbstractDataFrame)
    # --- 1. Extract Active Bets ---
    active_mask = sub_df.stake .> 1e-6
    n_bets = count(active_mask)

    # Return empty results if insufficient data
    if n_bets < metric.min_bets
        return _empty_hurdle_result(n_bets)
    end

    active_stakes = sub_df.stake[active_mask]
    active_pnls   = sub_df.pnl[active_mask]

    # --- 2. Compute Per-Bet ROI ---
    rois = active_pnls ./ active_stakes

    # Data integrity guard: ROI should never be below -1.0 in standard betting
    if any(r -> r < -1.0 - 1e-6, rois)
        @warn "BernoulliGammaHurdle: ROI below -1.0 detected — check stake/pnl data integrity"
        rois = clamp.(rois, -1.0, Inf)
    end

    # --- 3. Bernoulli Component ---
    wins = rois .> 0.0
    n_wins = sum(wins)
    p = n_wins / n_bets
    avg_stake = mean(active_stakes)

    # --- 4. Empirical Metrics ---
    E_R_emp = mean(rois)
    σ_R_emp = n_bets >= 2 ? std(rois) : 0.0
    Sharpe_emp = σ_R_emp > 0.0 ? E_R_emp / σ_R_emp : NaN

    # Empirical geometric growth rate: G = exp(E[log(1 + stake_i · roi_i)]) - 1
    log_wealth_increments = log.(max.(1e-8, 1.0 .+ active_stakes .* rois))
    G_emp = exp(mean(log_wealth_increments)) - 1.0

    # --- 5. Gamma Component (Positive ROIs) ---
    pos_rois = rois[wins]

    shape_val = NaN
    scale_val = NaN
    μ_pos     = 0.0
    var_pos   = 0.0
    E_R_param     = NaN
    σ_R_param     = NaN
    Sharpe_param  = NaN
    G_param       = NaN

    if n_wins > 0
        μ_pos = mean(pos_rois)
        
        if n_wins < 2 || var(pos_rois) == 0.0
            # Method of Moments fallback with scaled variance floor
            var_pos = max(n_wins >= 2 ? var(pos_rois) : 0.0, (0.01 * μ_pos)^2)
            scale_val = var_pos / μ_pos
            shape_val = μ_pos / scale_val
        else
            try
                g_fit = fit(Gamma, pos_rois)
                shape_val = shape(g_fit)
                scale_val = scale(g_fit)
                μ_pos     = mean(g_fit)
                var_pos   = var(g_fit)
            catch
                # MLE failed — Method of Moments fallback
                var_pos = max(var(pos_rois), (0.01 * μ_pos)^2)
                scale_val = var_pos / μ_pos
                shape_val = μ_pos / scale_val
            end
        end

        # --- 6. Parametric Moments ---
        # E[R] = p · E[R|win] + (1-p) · (-1) = p·μ_pos - (1-p)
        E_R_param = p * μ_pos - (1.0 - p)

        # E[R²] = p · (Var[R|win] + E[R|win]²) + (1-p) · 1
        E_R2_param = p * (var_pos + μ_pos^2) + (1.0 - p)
        Var_R_param = E_R2_param - E_R_param^2
        σ_R_param = sqrt(max(0.0, Var_R_param))
        Sharpe_param = σ_R_param > 0.0 ? E_R_param / σ_R_param : NaN

        # --- 7. Parametric Growth Rate (Monte Carlo) ---
        # G = exp( (1-p)·log(1-f) + p·E[log(1+f·Y)] ) - 1
        # where Y ~ Gamma(α, β) and f = avg_stake
        rng = Random.MersenneTwister(42)
        y_samples = rand(rng, Gamma(shape_val, scale_val), metric.mc_samples)
        mean_log_wealth = mean(log.(max.(1e-8, 1.0 .+ avg_stake .* y_samples)))
        g_log = (1.0 - p) * log(max(1e-8, 1.0 - avg_stake)) + p * mean_log_wealth
        G_param = exp(g_log) - 1.0
    else
        # All losses: E[R] = -1, growth is purely loss
        E_R_param = -1.0
        σ_R_param = 0.0
        Sharpe_param = NaN
        g_log = log(max(1e-8, 1.0 - avg_stake))
        G_param = exp(g_log) - 1.0
    end

    # --- 8. Return NamedTuple ---
    return (
        hurdle_p          = round(p, digits=4),
        hurdle_shape      = round(shape_val, digits=4),
        hurdle_scale      = round(scale_val, digits=4),
        hurdle_E_R        = round(E_R_param, digits=4),
        hurdle_sharpe     = round(Sharpe_param, digits=4),
        hurdle_G          = round(G_param, digits=6),
        hurdle_G_emp      = round(G_emp, digits=6),
        hurdle_n_bets     = n_bets,
        hurdle_avg_stake  = round(avg_stake, digits=4),
    )
end

"""
    _empty_hurdle_result(n_bets::Int)

Returns a zero-filled NamedTuple matching the hurdle output schema.
Used when there are insufficient bets to fit the model.
"""
function _empty_hurdle_result(n_bets::Int)
    return (
        hurdle_p          = 0.0,
        hurdle_shape      = NaN,
        hurdle_scale      = NaN,
        hurdle_E_R        = NaN,
        hurdle_sharpe     = NaN,
        hurdle_G          = 0.0,
        hurdle_G_emp      = 0.0,
        hurdle_n_bets     = n_bets,
        hurdle_avg_stake  = 0.0,
    )
end
