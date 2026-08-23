# current_development/scottish_lower/corners/r02_mle_scottish_lower_test.jl
#
# Frequentist MLE Statistical Significance Diagnostic on Scottish Lower (2024/25 -> 2025/26)

include("l01_corner_data.jl")
include("l02_corner_statistical_tests.jl")
include("l03_mle_significance.jl")

using Printf
using Statistics

println("================================================================================")
println(" SCOTTISH LOWER (56 & 57): FREQUENTIST MLE SIGNIFICANCE DIAGNOSTIC")
println(" Window: 2024/25 -> 2025/26 (Walk-Forward Trading Benchmark)")
println("================================================================================\n")

# 1. Ingest Data
df_all = fetch_scottish_corner_dataset()

# Filter to Scottish Lower (tournaments 56 & 57)
df_lower = filter(r -> r.tournament_id in (56, 57), df_all)

# Filter to 2024/25 -> 2025/26 seasons
df_24_26 = filter(r -> occursin("24/25", coalesce(r.season, "")) || occursin("25/26", coalesce(r.season, "")), df_lower)

println("--- 1. DATASET OVERVIEW ---")
@printf("Full Scottish Lower Matches (2020-2027): %d matches\n", nrow(df_lower))
@printf("Scottish Lower 2024/25 -> 2025/26:       %d matches, %d teams\n", 
        nrow(df_24_26), length(unique(vcat(df_24_26.home_team, df_24_26.away_team))))
@printf("Total Corners in 24/25->25/26 Window:    %d corners (Mean: %.2f/match)\n", 
        sum(df_24_26.corners_total), mean(df_24_26.corners_total))
@printf("Total Corner Goals in 24/25->25/26:      %d goals (Mean Conv: %.2f%%)\n\n", 
        sum(df_24_26.corner_goals_total), sum(df_24_26.corner_goals_total) / sum(df_24_26.corners_total) * 100)

# 2. Fit MLE Corner Generation Model on 24/25 -> 25/26 Window
println("--- 2. CORNER GENERATION MLE FIT (Negative Binomial Likelihood) ---")
gen_fit = fit_corner_generation_mle(df_24_26; model_type = :negbin)

@printf("Base Log Corner Rate (mu):    %+.4f (SE: %.4f, Base Corners/Game = %.2f)\n", 
        gen_fit.mu, gen_fit.se_mu, exp(gen_fit.mu))
@printf("Home Advantage (gamma_ha):     %+.4f (SE: %.4f, z = %.2f, p = %.4e)\n", 
        gen_fit.gamma_ha, gen_fit.se_gamma, gen_fit.gamma_ha / gen_fit.se_gamma, 
        2 * (1 - cdf(Normal(), abs(gen_fit.gamma_ha / gen_fit.se_gamma))))
@printf("Overdispersion Parameter (phi): %.2f (r = %.2f)\n", gen_fit.phi, gen_fit.phi)
@printf("Log-Likelihood (Full Model):   %.2f (AIC: %.2f)\n", gen_fit.ll_full, gen_fit.aic_full)
@printf("Log-Likelihood (Null Model):   %.2f (AIC: %.2f)\n", gen_fit.ll_null, gen_fit.aic_null)
@printf("Likelihood Ratio Test (LRT):   ChiSq(df=%d) = %.2f (p = %.4e)\n", 
        gen_fit.df_lrt, gen_fit.lrt_stat, gen_fit.lrt_p)

if gen_fit.lrt_p < 0.01
    println(">>> VERDICT: Team Corner Generation parameters are HIGHLY STATISTICALLY SIGNIFICANT (p < 0.01)! <<<\n")
else
    println(">>> VERDICT: Team Corner Generation parameters lack significance. <<<\n")
end

# Sort and print team attacking and defensive effects
sort!(gen_fit.team_df, :alpha_att, rev=true)
println("Top 5 Corner Attacking Teams (alpha_att):")
for r in eachrow(first(gen_fit.team_df, 5))
    @printf("  %-25s | alpha: %+.3f (x%.2f corner pressure) | beta: %+.3f (x%.2f concession)\n",
            r.team, r.alpha_att, r.mult_att, r.beta_def, r.mult_def)
end
println("\nBottom 5 Corner Attacking Teams (alpha_att):")
for r in eachrow(last(gen_fit.team_df, 5))
    @printf("  %-25s | alpha: %+.3f (x%.2f corner pressure) | beta: %+.3f (x%.2f concession)\n",
            r.team, r.alpha_att, r.mult_att, r.beta_def, r.mult_def)
end

# 3. Fit MLE Corner Conversion Model on 24/25 -> 25/26 Window
println("\n--- 3. CORNER GOAL CONVERSION MLE FIT (Binomial Logistic Model) ---")
conv_fit = fit_corner_conversion_mle(df_24_26)

@printf("Null Conversion Rate (q_null): %.2f%%\n", conv_fit.q_null * 100)
@printf("Base Logit (mu_q):             %+.4f\n", conv_fit.mu_q)
@printf("Log-Likelihood (Full Model):   %.2f (AIC: %.2f)\n", conv_fit.ll_full, conv_fit.aic_full)
@printf("Log-Likelihood (Null Model):   %.2f (AIC: %.2f)\n", conv_fit.ll_null, conv_fit.aic_null)
@printf("Likelihood Ratio Test (LRT):   ChiSq(df=%d) = %.2f (p = %.4e)\n", 
        conv_fit.df_lrt, conv_fit.lrt_stat, conv_fit.lrt_p)

if conv_fit.lrt_p < 0.05
    println(">>> VERDICT: Team Corner Goal Conversion is STATISTICALLY SIGNIFICANT! <<<\n")
else
    println(">>> VERDICT: Team Conversion Differences are NOISY / WEAK on 2-season sample (p > 0.05). Requires Hierarchical Shrinkage! <<<\n")
end

sort!(conv_fit.team_df, :est_conv_rate, rev=true)
println("Top 5 Estimated Corner Goal Converters:")
for r in eachrow(first(conv_fit.team_df, 5))
    @printf("  %-25s | eta_att: %+.3f | zeta_def: %+.3f | Est Conv: %5.2f%%\n",
            r.team, r.eta_att, r.zeta_def, r.est_conv_rate * 100)
end
println("\nBottom 5 Estimated Corner Goal Converters:")
for r in eachrow(last(conv_fit.team_df, 5))
    @printf("  %-25s | eta_att: %+.3f | zeta_def: %+.3f | Est Conv: %5.2f%%\n",
            r.team, r.eta_att, r.zeta_def, r.est_conv_rate * 100)
end

# 4. Full Historical Window Comparison (2020-2027)
println("\n--- 4. FULL HISTORICAL WINDOW CHECK (2020-2027, N = $(nrow(df_lower))) ---")
gen_all = fit_corner_generation_mle(df_lower; model_type = :negbin)
conv_all = fit_corner_conversion_mle(df_lower)

@printf("Corner Generation Full History LRT: ChiSq(df=%d) = %.2f (p = %.4e)\n", 
        gen_all.df_lrt, gen_all.lrt_stat, gen_all.lrt_p)
@printf("Corner Conversion Full History LRT: ChiSq(df=%d) = %.2f (p = %.4e)\n\n", 
        conv_all.df_lrt, conv_all.lrt_stat, conv_all.lrt_p)

println("================================================================================")
println("✓ FREQUENTIST MLE SIGNIFICANCE DIAGNOSTIC COMPLETE")
println("================================================================================")
