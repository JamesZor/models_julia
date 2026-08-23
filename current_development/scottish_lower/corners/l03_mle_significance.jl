# current_development/scottish_lower/corners/l03_mle_significance.jl
#
# Frequentist MLE & GLM Significance Diagnostics for Corner Generation and Conversion

using DataFrames
using Statistics
using Distributions
using Optim
using ForwardDiff
using LinearAlgebra
using Printf
using SpecialFunctions

"""
    fit_corner_generation_mle(df::DataFrame; model_type::Symbol = :negbin)

Fits an MLE model for corner counts with team attack (alpha) and defense (beta) parameters,
plus home advantage (gamma) and base intercept (mu).
Returns parameter estimates, standard errors (via Hessian inversion), p-values, and LRT vs null.
"""
function fit_corner_generation_mle(df::DataFrame; model_type::Symbol = :negbin)
    # 1. Map teams to indices
    teams = sort(unique(vcat(df.home_team, df.away_team)))
    K = length(teams)
    team_dict = Dict(t => i for (i, t) in enumerate(teams))
    
    # 2. Extract match data
    h_idx = [team_dict[t] for t in df.home_team]
    a_idx = [team_dict[t] for t in df.away_team]
    y_h = Float64.(coalesce.(df.corners_h, 0))
    y_a = Float64.(coalesce.(df.corners_a, 0))
    N = length(y_h)
    
    # Parameter vector theta:
    # [1] mu
    # [2] gamma_ha
    # [3 : K+1] alpha[1:K-1] (with alpha[K] = -sum(alpha[1:K-1]))
    # [K+2 : 2K] beta[1:K-1] (with beta[K] = -sum(beta[1:K-1]))
    # [2K+1] log_phi (if negbin)
    
    n_params = model_type == :negbin ? (2K + 1) : 2K
    
    function unpack_params(theta)
        mu = theta[1]
        gamma = theta[2]
        
        alpha_free = theta[3:(K+1)]
        alpha = vcat(alpha_free, -sum(alpha_free))
        
        beta_free = theta[(K+2):(2K)]
        beta = vcat(beta_free, -sum(beta_free))
        
        phi = model_type == :negbin ? exp(theta[2K+1]) : 1e6
        return (mu, gamma, alpha, beta, phi)
    end
    
    # Negative Log-Likelihood
    function nll(theta)
        mu, gamma, alpha, beta, phi = unpack_params(theta)
        
        # Guard against extreme values
        abs(mu) > 10.0 && return 1e8
        abs(gamma) > 5.0 && return 1e8
        maximum(abs.(alpha)) > 5.0 && return 1e8
        maximum(abs.(beta)) > 5.0 && return 1e8
        
        ll = 0.0
        for i in 1:N
            th = h_idx[i]
            ta = a_idx[i]
            
            log_lam_h = mu + gamma + alpha[th] - beta[ta]
            log_lam_a = mu + alpha[ta] - beta[th]
            
            lam_h = exp(clamp(log_lam_h, -10.0, 10.0))
            lam_a = exp(clamp(log_lam_a, -10.0, 10.0))
            
            if model_type == :poisson
                ll += logpdf(Poisson(lam_h), y_h[i])
                ll += logpdf(Poisson(lam_a), y_a[i])
            else
                # NegBin with mean lambda and overdispersion phi: r = phi, p = phi / (phi + lam)
                p_h = phi / (phi + lam_h)
                p_a = phi / (phi + lam_a)
                ll += logpdf(NegativeBinomial(phi, p_h), y_h[i])
                ll += logpdf(NegativeBinomial(phi, p_a), y_a[i])
            end
        end
        return -ll
    end
    
    # Initial values
    init_theta = zeros(n_params)
    init_theta[1] = log(mean(vcat(y_h, y_a)))
    init_theta[2] = 0.15 # slight home advantage
    if model_type == :negbin
        init_theta[2K+1] = log(10.0) # initial phi = 10
    end
    
    res = optimize(nll, init_theta, LBFGS(), Optim.Options(iterations = 1000, show_trace = false))
    theta_hat = Optim.minimizer(res)
    ll_full = -Optim.minimum(res)
    
    # Fit Null Model (mu + gamma only, no team parameters)
    function nll_null(theta_null)
        mu = theta_null[1]
        gamma = theta_null[2]
        phi = model_type == :negbin ? exp(theta_null[3]) : 1e6
        
        ll = 0.0
        for i in 1:N
            lam_h = exp(mu + gamma)
            lam_a = exp(mu)
            if model_type == :poisson
                ll += logpdf(Poisson(lam_h), y_h[i]) + logpdf(Poisson(lam_a), y_a[i])
            else
                p_h = phi / (phi + lam_h)
                p_a = phi / (phi + lam_a)
                ll += logpdf(NegativeBinomial(phi, p_h), y_h[i]) + logpdf(NegativeBinomial(phi, p_a), y_a[i])
            end
        end
        return -ll
    end
    
    init_null = model_type == :negbin ? [init_theta[1], init_theta[2], log(10.0)] : [init_theta[1], init_theta[2]]
    res_null = optimize(nll_null, init_null, LBFGS(), Optim.Options(iterations = 500))
    ll_null = -Optim.minimum(res_null)
    
    # Likelihood Ratio Test
    df_lrt = 2 * (K - 1)
    lrt_stat = 2.0 * (ll_full - ll_null)
    lrt_p = 1.0 - cdf(Chisq(df_lrt), max(0.0, lrt_stat))
    
    # Compute Hessian for Standard Errors
    H = ForwardDiff.hessian(nll, theta_hat)
    cov_mat = try
        inv(H)
    catch
        pinv(H)
    end
    
    se_theta = sqrt.(abs.(diag(cov_mat)))
    
    mu_hat, gamma_hat, alpha_hat, beta_hat, phi_hat = unpack_params(theta_hat)
    
    # Build team results DataFrame
    team_results = DataFrame(
        team = teams,
        alpha_att = alpha_hat,
        beta_def = beta_hat,
        mult_att = exp.(alpha_hat),
        mult_def = exp.(beta_hat)
    )
    
    return (
        teams = teams,
        mu = mu_hat,
        se_mu = se_theta[1],
        gamma_ha = gamma_hat,
        se_gamma = se_theta[2],
        phi = phi_hat,
        ll_full = ll_full,
        ll_null = ll_null,
        lrt_stat = lrt_stat,
        df_lrt = df_lrt,
        lrt_p = lrt_p,
        team_df = team_results,
        aic_full = 2 * n_params - 2 * ll_full,
        aic_null = 2 * length(init_null) - 2 * ll_null
    )
end

"""
    fit_corner_conversion_mle(df::DataFrame)

Fits an MLE Logistic Binomial model for corner goal conversion:
logit(q_{h,a}) = logit(q_base) + eta_att[h] - zeta_def[a]
Tests whether team differences are statistically significant vs a single global conversion rate.
"""
function fit_corner_conversion_mle(df::DataFrame)
    # Filter matches where at least one corner occurred
    sub = filter(r -> (r.corners_h > 0 || r.corners_a > 0), df)
    teams = sort(unique(vcat(sub.home_team, sub.away_team)))
    K = length(teams)
    team_dict = Dict(t => i for (i, t) in enumerate(teams))
    
    h_idx = [team_dict[t] for t in sub.home_team]
    a_idx = [team_dict[t] for t in sub.away_team]
    
    c_h = sub.corners_h
    c_a = sub.corners_a
    g_h = sub.corner_goals_h
    g_a = sub.corner_goals_a
    N = length(c_h)
    
    # Null Model: Single global conversion rate q_base
    total_corners = sum(c_h) + sum(c_a)
    total_corner_goals = sum(g_h) + sum(g_a)
    q_null = total_corner_goals / total_corners
    
    ll_null = 0.0
    for i in 1:N
        c_h[i] > 0 && (ll_null += logpdf(Binomial(c_h[i], q_null), g_h[i]))
        c_a[i] > 0 && (ll_null += logpdf(Binomial(c_a[i], q_null), g_a[i]))
    end
    
    # Full Model: logit(q) = mu_q + eta_att[th] - zeta_def[ta]
    # theta: [1] mu_q, [2:K] eta[1:K-1], [K+1:2K-1] zeta[1:K-1]
    n_params = 2K - 1
    
    function unpack_conv(theta)
        mu_q = theta[1]
        eta_free = theta[2:K]
        eta = vcat(eta_free, -sum(eta_free))
        zeta_free = theta[(K+1):(2K-1)]
        zeta = vcat(zeta_free, -sum(zeta_free))
        return (mu_q, eta, zeta)
    end
    
    function nll_conv(theta)
        mu_q, eta, zeta = unpack_conv(theta)
        
        # Ridge penalty (L2 shrinkage) to prevent infinite MLEs when a team has 0 goals
        l2_reg = 0.1 * (sum(eta.^2) + sum(zeta.^2))
        
        ll = 0.0
        for i in 1:N
            th = h_idx[i]
            ta = a_idx[i]
            
            if c_h[i] > 0
                q_h = 1.0 / (1.0 + exp(-(mu_q + eta[th] - zeta[ta])))
                q_h = clamp(q_h, 1e-5, 0.99999)
                ll += logpdf(Binomial(c_h[i], q_h), g_h[i])
            end
            if c_a[i] > 0
                q_a = 1.0 / (1.0 + exp(-(mu_q + eta[ta] - zeta[th])))
                q_a = clamp(q_a, 1e-5, 0.99999)
                ll += logpdf(Binomial(c_a[i], q_a), g_a[i])
            end
        end
        return -ll + l2_reg
    end
    
    init_conv = zeros(n_params)
    init_conv[1] = log(q_null / (1.0 - q_null))
    
    res = optimize(nll_conv, init_conv, LBFGS(), Optim.Options(iterations = 1000))
    ll_full = -Optim.minimum(res)
    mu_q_hat, eta_hat, zeta_hat = unpack_conv(Optim.minimizer(res))
    
    df_lrt = 2 * (K - 1)
    lrt_stat = 2.0 * (ll_full - ll_null)
    lrt_p = 1.0 - cdf(Chisq(df_lrt), max(0.0, lrt_stat))
    
    team_conv_df = DataFrame(
        team = teams,
        eta_att = eta_hat,
        zeta_def = zeta_hat,
        est_conv_rate = [1.0 / (1.0 + exp(-(mu_q_hat + eta_hat[i]))) for i in 1:K]
    )
    
    return (
        teams = teams,
        q_null = q_null,
        mu_q = mu_q_hat,
        ll_full = ll_full,
        ll_null = ll_null,
        lrt_stat = lrt_stat,
        df_lrt = df_lrt,
        lrt_p = lrt_p,
        team_df = team_conv_df,
        aic_full = 2 * n_params - 2 * ll_full,
        aic_null = 2 - 2 * ll_null
    )
end
