# current_development/scottish_lower/corners/r00_eda_corner_distributions.jl
#
# Stage 1 EDA: Distributional Properties, Home Advantage, & Goal Breakdown

include("l01_corner_data.jl")
include("l02_corner_statistical_tests.jl")

using Printf

println("================================================================================")
println(" SCOTTISH FOOTBALL: CORNER DISTRIBUTIONS & GOAL DECOMPOSITION EDA")
println("================================================================================")

# 1. Ingest Data
df = fetch_scottish_corner_dataset()
println("Total Matches Ingested: ", nrow(df))
println("Seasons Covered: ", sort(unique(df.season)))
println("Tournaments: ", sort(unique(df.tournament_id)), " (54: Prem, 55: Champ, 56: L1, 57: L2)\n")

# 2. Overall Goal Decomposition Breakdown
total_g = sum(df.goals_total)
total_open = sum(df.open_goals_h .+ df.open_goals_a)
total_pen = sum(df.pen_goals_h .+ df.pen_goals_a)
total_og = sum(df.og_goals_h .+ df.og_goals_a)
total_corner = sum(df.corner_goals_total)

println("--- 1. 4-WAY GOAL DECOMPOSITION ACROSS ALL SCOTTISH TIERS ---")
@printf("Total Match Goals: %d (100.0%%)\n", total_g)
@printf("  ├─ Open-Play Goals:  %5d (%5.2f%%)\n", total_open, total_open / total_g * 100)
@printf("  ├─ Corner Goals:     %5d (%5.2f%%)\n", total_corner, total_corner / total_g * 100)
@printf("  ├─ Penalty Goals:    %5d (%5.2f%%)\n", total_pen, total_pen / total_g * 100)
@printf("  └─ Own Goals:        %5d (%5.2f%%)\n\n", total_og, total_og / total_g * 100)

# 3. Corner Generation Statistics by Tier
println("--- 2. CORNER COUNT GENERATION BY TIER ---")
tiers = [
    (54, "Scottish Premiership (54)"),
    (55, "Scottish Championship (55)"),
    (56, "Scottish League One (56)"),
    (57, "Scottish League Two (57)")
]

for (tid, tname) in tiers
    sub = filter(r -> r.tournament_id == tid, df)
    nrow(sub) == 0 && continue
    
    m_h = mean(sub.corners_h)
    v_h = var(sub.corners_h)
    m_a = mean(sub.corners_a)
    v_a = var(sub.corners_a)
    m_tot = mean(sub.corners_total)
    v_tot = var(sub.corners_total)
    
    ha_test = test_corner_home_advantage(sub.corners_h, sub.corners_a)
    disp_h = compute_dispersion_stats(sub.corners_h)
    disp_tot = compute_dispersion_stats(sub.corners_total)
    
    @printf("[%s] N = %d matches\n", tname, nrow(sub))
    @printf("  Mean Corners/Match: %.2f (Home: %.2f, Away: %.2f) | Var/Mean: %.2f\n", m_tot, m_h, m_a, disp_tot.dispersion_index)
    @printf("  Home Adv Ratio: %.2fx (Diff: +%.2f corners, t = %.2f, p = %.4e)\n", ha_test.ratio_ha, ha_test.mean_diff, ha_test.t_stat, ha_test.p_value)
    @printf("  Home Overdispersion Index: %.2f (%s, p = %.4e)\n\n", disp_h.dispersion_index, disp_h.is_overdispersed ? "OVERDISPERSED" : "POISSON EQUIDISPERSED", disp_h.p_value)
end

# 4. Correlation: Do Corners Correlate with Total Goals?
corr_corners_goals = cor(df.corners_total, df.goals_total)
corr_corners_open = cor(df.corners_total, df.open_goals_h .+ df.open_goals_a)
corr_corners_corner_goals = cor(df.corners_total, df.corner_goals_total)

println("--- 3. CORNER COUNT VS GOAL CORRELATIONS ---")
@printf("Cor(Total Corners, Total Goals):       %+.4f\n", corr_corners_goals)
@printf("Cor(Total Corners, Open-Play Goals):   %+.4f\n", corr_corners_open)
@printf("Cor(Total Corners, Corner Goals):      %+.4f\n\n", corr_corners_corner_goals)

println("================================================================================")
println("✓ STAGE 1 DISTRIBUTIONAL EDA COMPLETE")
println("================================================================================")
