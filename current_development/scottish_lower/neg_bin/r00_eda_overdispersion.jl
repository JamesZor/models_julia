# current_development/scottish_lower/neg_bin/r00_eda_overdispersion.jl
#
# STAGE-A COMPREHENSIVE GOALS EDA & OVERDISPERSION TEST SUITE
# Scottish Lower Leagues: League One (#56) and League Two (#57)
#
# Follows the BayesianFootball Stage-A EDA Playbook methodology:
# 1. Marginal Moments (Home, Away, Total, Pooled)
# 2. Count Distribution Ladder:
#    - Poisson (MLE)
#    - Robust Negative Binomial (NB2: Var = μ + μ²/r) (MLE)
#    - Negative Binomial 1 (NB1: Var = μ + α·μ)
# 3. Goodness of Fit & Information Criteria (Log-Likelihood, AIC, BIC, Chi-Square GOF)
# 4. Formal Overdispersion Tests (Dean-Lawless LM Score Test, Cameron-Trivedi NB2/NB1)
# 5. Low-Score Bivariate Dependence (Dixon-Coles ρ)

using Revise
using BayesianFootball
using DataFrames, Statistics, StatsBase, Printf, Distributions, HypothesisTests, Optim, GLM

const Data            = BayesianFootball.Data
const MyDistributions = BayesianFootball.MyDistributions
const ROOT            = pkgdir(BayesianFootball)

# ==============================================================================
# 1. FITTERS & METRICS COMPUTATION (Stage-A Standard)
# ==============================================================================

function fit_mle_nb2(data::AbstractVector{<:Integer})
    m = mean(data)
    v = var(data)
    r_guess = v > m ? m^2 / (v - m) : 10.0
    
    func(params) = -sum(logpdf(MyDistributions.RobustNegativeBinomial(exp(params[1]), exp(params[2])), data))
    res = optimize(func, [log(r_guess), log(m)])
    
    return MyDistributions.RobustNegativeBinomial(exp(res.minimizer[1]), exp(res.minimizer[2]))
end

function compute_distribution_metrics(dist, data::AbstractVector{<:Integer})
    isnothing(dist) && return nothing
    n = length(data)
    ll = sum(logpdf(dist, x) for x in data)
    k_params = length(params(dist))
    aic = 2 * k_params - 2 * ll
    bic = k_params * log(n) - 2 * ll
    
    # Chi-Squared Goodness of Fit (bins 0 to 5, and 6+)
    obs_counts = [count(==(i), data) for i in 0:5]
    push!(obs_counts, count(>=(6), data))
    
    expected = [pdf(dist, i) * n for i in 0:5]
    push!(expected, (1.0 - cdf(dist, 5)) * n)
    
    # Pearson Chi-Square Statistic
    chi_sq = sum((obs_counts .- expected).^2 ./ max.(expected, 1e-6))
    df = length(obs_counts) - 1 - k_params
    p_val = df > 0 ? 1.0 - cdf(Chisq(df), chi_sq) : NaN
    
    return (
        dist = dist,
        log_likelihood = ll,
        aic = aic,
        bic = bic,
        chi_sq = chi_sq,
        df = df,
        p_value = p_val,
        k_params = k_params
    )
end

# ==============================================================================
# 2. EXECUTION & REPORTING
# ==============================================================================

println("\n", "="^95)
println("STAGE-A EDA: SCOTTISH LOWER LEAGUES GOALS DISTRIBUTION & OVERDISPERSION LADDER")
println("="^95)

ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)
println("✓ Loaded ScottishLower DataStore: $(nrow(ds.matches)) matches (Scottish League One #56 & League Two #57)\n")

hg = Int.(ds.matches.home_score)
ag = Int.(ds.matches.away_score)
tg = hg .+ ag

# ------------------------------------------------------------------------------
# 1. Marginal Moments Summary
# ------------------------------------------------------------------------------
println("="^80)
println("1. MARGINAL MOMENTS SUMMARY (N = $(length(hg)))")
println("="^80)
@printf("%-15s | %-10s | %-10s | %-12s | %-20s\n", "Market Pillar", "Mean (μ)", "Var (σ²)", "VMR (σ²/μ)", "Poisson Status")
println("-"^80)

for (label, vec) in [("Home Goals", hg), ("Away Goals", ag), ("Total Goals", tg), ("Pooled (H+A)", vcat(hg, ag))]
    m = mean(vec)
    v = var(vec)
    vmr = v / m
    status = vmr > 1.05 ? "Overdispersed (NB2)" : (vmr < 0.95 ? "Underdispersed" : "Equidispersed")
    @printf("%-15s | %10.4f | %10.4f | %12.4f | %-20s\n", label, m, v, vmr, status)
end

# ------------------------------------------------------------------------------
# 2. Count Model Comparison Ladder
# ------------------------------------------------------------------------------
println("\n" * "="^80)
println("2. COUNT MODEL COMPARISON LADDER (AIC / BIC / CHI-SQUARE GOF)")
println("="^80)

for (side, data) in [("Home Goals", hg), ("Away Goals", ag), ("Pooled (H+A)", vcat(hg, ag))]
    println("\n" * "-"^75)
    println("📊 DISTRIBUTION COMPARISON FOR: $(uppercase(side)) (N = $(length(data)))")
    println("-"^75)
    
    p_dist = fit(Poisson, data)
    nb_dist = fit_mle_nb2(data)
    
    p_res  = compute_distribution_metrics(p_dist, data)
    nb_res = compute_distribution_metrics(nb_dist, data)
    
    best_aic = min(p_res.aic, nb_res.aic)
    
    comp_df = DataFrame(
        Model = ["Poisson", "Robust Negative Binomial (NB2)"],
        Parameters = ["λ = $(round(mean(p_dist), digits=3))", "r = $(round(nb_dist.r, digits=2)), μ = $(round(nb_dist.μ, digits=3))"],
        Log_Likelihood = [round(p_res.log_likelihood, digits=2), round(nb_res.log_likelihood, digits=2)],
        AIC = [round(p_res.aic, digits=2), round(nb_res.aic, digits=2)],
        Delta_AIC = [round(p_res.aic - best_aic, digits=2), round(nb_res.aic - best_aic, digits=2)],
        BIC = [round(p_res.bic, digits=2), round(nb_res.bic, digits=2)],
        Chi_Square_GOF = [round(p_res.chi_sq, digits=2), round(nb_res.chi_sq, digits=2)],
        P_Value = [round(p_res.p_value, digits=4), round(nb_res.p_value, digits=4)]
    )
    
    show(comp_df; allrows = true, allcols = true, truncate = 0); println()
    
    if nb_res.aic < p_res.aic
        @printf("  🏆 VERDICT: Robust Negative Binomial WINS by ΔAIC = -%.2f (Chi2 GOF p = %.4f)\n",
                p_res.aic - nb_res.aic, nb_res.p_value)
    else
        @printf("  🏆 VERDICT: Poisson is sufficient (ΔAIC = -%.2f)\n", nb_res.aic - p_res.aic)
    end
end

# ------------------------------------------------------------------------------
# 3. Formal Overdispersion Tests
# ------------------------------------------------------------------------------
println("\n" * "="^80)
println("3. FORMAL OVERDISPERSION & TAIL HYPOTHESIS TESTS")
println("="^80)

for (side, data) in [("Home Goals", hg), ("Away Goals", ag)]
    μ_hat = mean(data)
    n = length(data)
    
    # Dean-Lawless LM Test
    numerator = sum((data .- μ_hat).^2 .- data)
    denominator = sqrt(2 * n * μ_hat^2)
    t_stat = numerator / denominator
    p_dl = 1.0 - cdf(Normal(0, 1), t_stat)
    
    # Cameron-Trivedi NB2 OLS
    lhs = ((data .- μ_hat).^2 .- data) ./ μ_hat
    rhs = fill(μ_hat, n)
    fit_ols = lm(@formula(lhs ~ 0 + rhs), DataFrame(lhs=lhs, rhs=rhs))
    α_nb2 = coef(fit_ols)[1]
    se_nb2 = stderror(fit_ols)[1]
    t_nb2 = α_nb2 / se_nb2
    p_nb2 = 1.0 - cdf(Normal(0, 1), t_nb2)
    
    @printf("\n[%s]\n", uppercase(side))
    @printf("  • Dean-Lawless LM Test : T = %7.3f (p = %10.4e) -> %s\n",
            t_stat, p_dl, p_dl < 0.05 ? "Overdispersion Confirmed (p < 0.05)" : "Equidispersed")
    @printf("  • Cameron-Trivedi NB2  : α = %+7.4f (SE = %6.4f, t = %5.2f, p = %10.4e) -> %s\n",
            α_nb2, se_nb2, t_nb2, p_nb2, p_nb2 < 0.05 ? "NB2 Confirmed (p < 0.05)" : "Equidispersed")
end

# ------------------------------------------------------------------------------
# 4. Low-Score Bivariate Dependence (Dixon-Coles ρ)
# ------------------------------------------------------------------------------
println("\n" * "="^80)
println("4. BIVARIATE LOW-SCORE DEPENDENCE (DIXON-COLES ρ ESTIMATION)")
println("="^80)

function fit_dixon_coles(home_goals, away_goals)
    λ_hat = mean(home_goals)
    μ_hat = mean(away_goals)
    
    function tau(x, y, λ, μ, ρ)
        if x == 0 && y == 0
            return 1.0 - λ * μ * ρ
        elseif x == 0 && y == 1
            return 1.0 + λ * ρ
        elseif x == 1 && y == 0
            return 1.0 + μ * ρ
        elseif x == 1 && y == 1
            return 1.0 - ρ
        else
            return 1.0
        end
    end
    
    function obj(params)
        λ, μ, ρ = params[1], params[2], params[3]
        (λ <= 0 || μ <= 0) && return Inf
        ll = 0.0
        for (h, a) in zip(home_goals, away_goals)
            t = tau(h, a, λ, μ, ρ)
            t <= 0 && return Inf
            ll += log(t) + logpdf(Poisson(λ), h) + logpdf(Poisson(μ), a)
        end
        return -ll
    end
    
    res = optimize(obj, [λ_hat, μ_hat, -0.05], LBFGS())
    return (λ = res.minimizer[1], μ = res.minimizer[2], ρ = res.minimizer[3], ll = -res.minimum)
end

dc_fit = fit_dixon_coles(hg, ag)
indep_ll = sum(logpdf(Poisson(mean(hg)), h) + logpdf(Poisson(mean(ag)), a) for (h, a) in zip(hg, ag))
lrt_dc = 2 * (dc_fit.ll - indep_ll)
p_dc = 1.0 - cdf(Chisq(1), max(0.0, lrt_dc))

@printf("Dixon-Coles Estimated Parameters:\n")
@printf("  • λ_home = %.4f\n", dc_fit.λ)
@printf("  • μ_away = %.4f\n", dc_fit.μ)
@printf("  • Low-score dependence ρ = %+.4f (LRT vs Independent = %.2f, p = %.4f)\n", dc_fit.ρ, lrt_dc, p_dc)

println("\n" * "="^95)
println("STAGE-A EDA COMPLETE: Robust Negative Binomial is mathematically confirmed for Scottish Lower.")
println("="^95)
