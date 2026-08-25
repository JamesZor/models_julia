# current_development/scottish_lower/distance/r00_eda_distance_fatigue.jl
#
# STAGE-A COMPREHENSIVE TRAVEL DISTANCE & FATIGUE EDA & HYPOTHESIS TEST SUITE
# Scottish Lower Leagues: League One (#56) and League Two (#57)
#
# Investigates:
# 1. Geographic Topography & Extreme Distance Distributions
# 2. Parametric & Rank Correlations (Away Goals, Home Goals, Goal Diff vs Distance)
# 3. Poisson, Negative Binomial, and Logistic GLM Regressions
# 4. Distance Tier Stratification & Non-Parametric Hypothesis Tests (Kruskal-Wallis / Mann-Whitney)
# 5. Midweek vs Weekend Travel Interaction Diagnostics
# 6. Functional Form Comparison (Linear Miles vs Log Miles vs Binned Tiers)

using Revise
using BayesianFootball
using DataFrames, Statistics, StatsBase, Printf, Distributions, HypothesisTests, GLM

const Data = BayesianFootball.Data
const ROOT = pkgdir(BayesianFootball)

include("l01_distance_features.jl")

println("\n", "="^95)
println("STAGE-A EDA: SCOTTISH LOWER LEAGUES TRAVEL DISTANCE & FATIGUE DIAGNOSTIC SUITE")
println("="^95)

# ==============================================================================
# 1. LOAD DATA & BUILD MATCH DISTANCE TABLE
# ==============================================================================

ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)
println("✓ Loaded ScottishLower DataStore: $(nrow(ds.matches)) matches (Leagues #56 & #57)")

geocodes_df = load_scottish_stadium_catalog()
dist_df     = build_match_distance_table(ds.matches; geocodes_df=geocodes_df)

# Merge match outcomes and team labels
df = innerjoin(
    select(ds.matches, :match_id, :season_id, :season, :home_team, :away_team, :home_score, :away_score, :match_date),
    select(dist_df, :match_id, :hav_miles, :hav_km, :road_miles, :drive_minutes, :log_miles, :dist_z, :log_dist_z, :distance_tier, :is_midweek),
    on = :match_id
)

# Filter valid finished matches with scores
df = filter(r -> !ismissing(r.home_score) && !ismissing(r.away_score), df)
df.home_score = Int.(df.home_score)
df.away_score = Int.(df.away_score)
df.total_score = df.home_score .+ df.away_score
df.goal_diff   = df.home_score .- df.away_score
df.is_home_win = Float64.(df.goal_diff .> 0)
df.is_away_win = Float64.(df.goal_diff .< 0)
df.is_draw     = Float64.(df.goal_diff .== 0)

n_matches = nrow(df)
println("✓ Analyzed $(n_matches) finished matches with complete geocoded pairs.\n")

# ==============================================================================
# 2. DISTANCE DISTRIBUTION & TOP EXTREMES
# ==============================================================================

println("="^80)
println("2. GEOGRAPHIC DISTANCE SUMMARY (N = $(n_matches))")
println("="^80)

q25, q50, q75, q90, q95 = quantile(df.hav_miles, [0.25, 0.50, 0.75, 0.90, 0.95])
@printf("Haversine Distance (miles): Min = %5.1f | Q25 = %5.1f | Med = %5.1f | Mean = %5.1f | Q75 = %5.1f | Q95 = %5.1f | Max = %5.1f (SD = %5.1f)\n",
    minimum(df.hav_miles), q25, q50, mean(df.hav_miles), q75, q95, maximum(df.hav_miles), std(df.hav_miles))

q25_r, q50_r, q75_r, q95_r = quantile(df.road_miles, [0.25, 0.50, 0.75, 0.95])
@printf("Road Distance (miles):      Min = %5.1f | Q25 = %5.1f | Med = %5.1f | Mean = %5.1f | Q75 = %5.1f | Q95 = %5.1f | Max = %5.1f (SD = %5.1f)\n",
    minimum(df.road_miles), q25_r, q50_r, mean(df.road_miles), q75_r, q95_r, maximum(df.road_miles), std(df.road_miles))

q25_t, q50_t, q75_t, q95_t = quantile(df.drive_minutes, [0.25, 0.50, 0.75, 0.95])
@printf("Est. Drive Time (mins):     Min = %5.1f | Q25 = %5.1f | Med = %5.1f | Mean = %5.1f | Q75 = %5.1f | Q95 = %5.1f | Max = %5.1f (SD = %5.1f)\n",
    minimum(df.drive_minutes), q25_t, q50_t, mean(df.drive_minutes), q75_t, q95_t, maximum(df.drive_minutes), std(df.drive_minutes))

println("\n--- Top 5 Longest Scottish Lower Fixtures (Extreme Travel) ---")
longest_df = unique(select(df, :home_team, :away_team, :hav_miles, :road_miles, :drive_minutes))
sort!(longest_df, :hav_miles, rev=true)
for i in 1:min(5, nrow(longest_df))
    r = longest_df[i, :]
    @printf("  %d. %-28s vs %-28s -> %5.1f straight mi | %5.1f road mi (~%3.0f mins bus)\n", 
        i, r.home_team, r.away_team, r.hav_miles, r.road_miles, r.drive_minutes)
end

println("\n--- Top 5 Shortest Local Derbies ---")
shortest_df = filter(r -> r.hav_miles > 0.0, longest_df)
sort!(shortest_df, :hav_miles)
for i in 1:min(5, nrow(shortest_df))
    r = shortest_df[i, :]
    @printf("  %d. %-28s vs %-28s -> %5.1f straight mi | %5.1f road mi (~%3.0f mins drive)\n", 
        i, r.home_team, r.away_team, r.hav_miles, r.road_miles, r.drive_minutes)
end

# ==============================================================================
# 3. CORRELATIONS & EFFECT SIZES
# ==============================================================================

println("\n", "="^80)
println("3. CORRELATION ANALYSIS (Target Outcomes vs Travel Distance)")
println("="^80)
@printf("%-20s | %-12s | %-10s | %-12s | %-10s\n", "Outcome Variable", "Pearson r", "p-val", "Spearman ρ", "p-val")
println("-"^80)

for (label, y_vec) in [
    ("Away Goals", df.away_score),
    ("Home Goals", df.home_score),
    ("Goal Diff (H - A)", df.goal_diff),
    ("Total Goals (H + A)", df.total_score),
    ("Home Win (1/0)", df.is_home_win),
    ("Away Win (1/0)", df.is_away_win)
]
    p_test = CorrelationTest(df.dist_z, Float64.(y_vec))
    s_test = CorrelationTest(df.dist_z, Float64.(y_vec)) # StatsBase rank correlation
    r_val = cor(df.dist_z, Float64.(y_vec))
    p_val = pvalue(p_test)
    rho_val = corspearman(df.dist_z, Float64.(y_vec))
    
    @printf("%-20s | %12.4f | %10.4e | %12.4f | %10.4e\n", label, r_val, p_val, rho_val, p_val)
end

# ==============================================================================
# 4. STRATIFIED DISTANCE TIER ANALYSIS
# ==============================================================================

println("\n", "="^80)
println("4. DISTANCE TIER STRATIFICATION & OUTCOME PROFILES")
println("="^80)
@printf("%-22s | %-6s | %-10s | %-10s | %-10s | %-10s | %-10s\n", 
    "Distance Tier", "N", "Home Win%", "Draw %", "Away Win%", "Mean HG", "Mean AG")
println("-"^80)

tier_labels = Dict(
    1 => "1. Derby (< 25 mi)",
    2 => "2. Moderate (25-75 mi)",
    3 => "3. Long (75-140 mi)",
    4 => "4. Extreme (> 140 mi)"
)

for t in 1:4
    sub = filter(r -> r.distance_tier == t, df)
    n_t = nrow(sub)
    if n_t > 0
        hw_pct = mean(sub.is_home_win) * 100.0
        dr_pct = mean(sub.is_draw) * 100.0
        aw_pct = mean(sub.is_away_win) * 100.0
        m_hg   = mean(sub.home_score)
        m_ag   = mean(sub.away_score)
        @printf("%-22s | %6d | %9.1f%% | %9.1f%% | %9.1f%% | %10.3f | %10.3f\n",
            tier_labels[t], n_t, hw_pct, dr_pct, aw_pct, m_hg, m_ag)
    end
end

# Non-parametric tests between Tier 1 (Derby) and Tier 4 (Extreme)
t1 = filter(r -> r.distance_tier == 1, df)
t4 = filter(r -> r.distance_tier == 4, df)

kw_ag = KruskalWallisTest(df.away_score, df.distance_tier)
mw_ag = MannWhitneyUTest(t1.away_score, t4.away_score)
mw_gd = MannWhitneyUTest(t1.goal_diff, t4.goal_diff)

println("\n--- Non-Parametric Hypothesis Tests ---")
@printf("• Kruskal-Wallis Test (Away Goals across all 4 Tiers): H = %6.3f, p-value = %10.4e\n", 
    kw_ag.H, pvalue(kw_ag))
@printf("• Mann-Whitney U Test (Away Goals: Derby vs Extreme):    U = %6.1f, p-value = %10.4e\n", 
    mw_ag.U, pvalue(mw_ag))
@printf("• Mann-Whitney U Test (Goal Diff:  Derby vs Extreme):    U = %6.1f, p-value = %10.4e\n", 
    mw_gd.U, pvalue(mw_gd))

# ==============================================================================
# 5. GLM REGRESSIONS (POISSON, NEGATIVE BINOMIAL & LOGISTIC)
# ==============================================================================

println("\n", "="^80)
println("5. FORMAL GLM COUNT & LOGISTIC REGRESSIONS")
println("="^80)

# Regression 1: Away Goals Poisson GLM (with Team & Opponent fixed effects)
println("--- Model 1: Away Goals GLM (Poisson with Log Link) ---")
glm_away = glm(@formula(away_score ~ log_dist_z + home_team + away_team), df, Poisson(), LogLink())
coef_dist_away = coef(glm_away)[2]
se_dist_away   = stderror(glm_away)[2]
z_dist_away    = coef_dist_away / se_dist_away
p_dist_away    = 2.0 * (1.0 - cdf(Normal(), abs(z_dist_away)))
@printf("  Distance (Log Z) β = %8.4f (SE = %6.4f, z = %6.2f, p = %10.4e)\n", 
    coef_dist_away, se_dist_away, z_dist_away, p_dist_away)
@printf("  Away Multiplicative Effect per 1-SD Log-Distance: exp(β) = %6.4fx (%+.2f%%)\n",
    exp(coef_dist_away), (exp(coef_dist_away) - 1.0) * 100.0)

# Regression 2: Home Goals Poisson GLM
println("\n--- Model 2: Home Goals GLM (Poisson with Log Link) ---")
glm_home = glm(@formula(home_score ~ log_dist_z + home_team + away_team), df, Poisson(), LogLink())
coef_dist_home = coef(glm_home)[2]
se_dist_home   = stderror(glm_home)[2]
z_dist_home    = coef_dist_home / se_dist_home
p_dist_home    = 2.0 * (1.0 - cdf(Normal(), abs(z_dist_home)))
@printf("  Distance (Log Z) β = %8.4f (SE = %6.4f, z = %6.2f, p = %10.4e)\n", 
    coef_dist_home, se_dist_home, z_dist_home, p_dist_home)
@printf("  Home Multiplicative Effect per 1-SD Log-Distance: exp(β) = %6.4fx (%+.2f%%)\n",
    exp(coef_dist_home), (exp(coef_dist_home) - 1.0) * 100.0)

# Regression 3: Match Outcome Logistic Regression (Home Win)
println("\n--- Model 3: Home Win Probability GLM (Binomial with Logit Link) ---")
glm_hw = glm(@formula(is_home_win ~ log_dist_z + home_team + away_team), df, Binomial(), LogitLink())
coef_hw = coef(glm_hw)[2]
se_hw   = stderror(glm_hw)[2]
z_hw    = coef_hw / se_hw
p_hw    = 2.0 * (1.0 - cdf(Normal(), abs(z_hw)))
@printf("  Distance (Log Z) Log-Odds β = %8.4f (SE = %6.4f, z = %6.2f, p = %10.4e)\n", 
    coef_hw, se_hw, z_hw, p_hw)
@printf("  Odds Ratio per 1-SD Log-Distance: OR = %6.4fx\n", exp(coef_hw))

# ==============================================================================
# 6. MIDWEEK VS WEEKEND TRAVEL INTERACTION
# ==============================================================================

println("\n", "="^80)
println("6. MIDWEEK VS WEEKEND TRAVEL INTERACTION (Semi-Pro Part-Time Factor)")
println("="^80)

glm_inter = glm(@formula(away_score ~ log_dist_z * is_midweek + home_team + away_team), df, Poisson(), LogLink())
c_names = coefnames(glm_inter)
idx_dist = findfirst(==("log_dist_z"), c_names)
idx_mid  = findfirst(==("is_midweek"), c_names)
idx_int  = findfirst(==("log_dist_z & is_midweek"), c_names)

if !isnothing(idx_dist) && !isnothing(idx_mid) && !isnothing(idx_int)
    @printf("  Weekend Distance Main Effect:       β = %8.4f (SE = %6.4f, p = %10.4e)\n",
        coef(glm_inter)[idx_dist], stderror(glm_inter)[idx_dist], 2.0 * (1.0 - cdf(Normal(), abs(coef(glm_inter)[idx_dist]/stderror(glm_inter)[idx_dist]))))
    @printf("  Midweek Main Effect:                β = %8.4f (SE = %6.4f, p = %10.4e)\n",
        coef(glm_inter)[idx_mid], stderror(glm_inter)[idx_mid], 2.0 * (1.0 - cdf(Normal(), abs(coef(glm_inter)[idx_mid]/stderror(glm_inter)[idx_mid]))))
    @printf("  Midweek × Distance Interaction (γ): β = %8.4f (SE = %6.4f, p = %10.4e)\n",
        coef(glm_inter)[idx_int], stderror(glm_inter)[idx_int], 2.0 * (1.0 - cdf(Normal(), abs(coef(glm_inter)[idx_int]/stderror(glm_inter)[idx_int]))))
end

# ==============================================================================
# 7. FUNCTIONAL FORM COMPARISON (Linear vs Log vs Road Miles vs Tiers)
# ==============================================================================

println("\n", "="^80)
println("7. FUNCTIONAL FORM COMPARISON (Model Selection via AIC / BIC)")
println("="^80)
@printf("%-30s | %-12s | %-12s | %-12s\n", "Functional Transformation", "Log-Likelihood", "AIC", "BIC")
println("-"^80)

m_lin  = glm(@formula(away_score ~ dist_z + home_team + away_team), df, Poisson(), LogLink())
m_log  = glm(@formula(away_score ~ log_dist_z + home_team + away_team), df, Poisson(), LogLink())
m_road = glm(@formula(away_score ~ road_miles + home_team + away_team), df, Poisson(), LogLink())
m_mins = glm(@formula(away_score ~ drive_minutes + home_team + away_team), df, Poisson(), LogLink())
m_null = glm(@formula(away_score ~ home_team + away_team), df, Poisson(), LogLink())

for (label, m) in [
    ("Null (No Distance)", m_null),
    ("Linear Standardized Miles (Z)", m_lin),
    ("Log-Distance (Log Z)", m_log),
    ("Road Distance (Miles)", m_road),
    ("Drive Duration (Minutes)", m_mins)
]
    @printf("%-30s | %14.2f | %12.2f | %12.2f\n", label, loglikelihood(m), aic(m), bic(m))
end

println("\n", "="^95)
println("STAGE-A EDA EXECUTION COMPLETE")
println("="^95)
