# current_development/scottish_lower/corners/r01_eda_team_corner_signals.jl
#
# Stage 2 EDA: Team-Level Generation, Conversion Efficiency, Defensive Prevention, & YoY Persistence

include("l01_corner_data.jl")
include("l02_corner_statistical_tests.jl")

using Printf
using Statistics

println("================================================================================")
println(" SCOTTISH FOOTBALL: TEAM CORNER SIGNALS & CONVERSION EFFICIENCY EDA")
println("================================================================================")

# 1. Ingest Data
df = fetch_scottish_corner_dataset()
println("Total Matches: ", nrow(df), "\n")

# 2. Team-Level Metrics
team_df = compute_team_corner_metrics(df)
filter!(r -> r.matches >= 30, team_df) # Only teams with at least 30 matches
println("Total Established Teams (>= 30 matches): ", nrow(team_df), "\n")

# Sort by Corner Creation Rate
sort!(team_df, :corner_rate_for, rev=true)
println("--- 1. TOP 5 & BOTTOM 5 CORNER CREATING TEAMS ---")
println("Top 5 Corner Creators (Attacking Pressure):")
for r in eachrow(first(team_df, 5))
    @printf("  %-25s | Matches: %3d | Corners/G: %5.2f | Conceded/G: %5.2f | Corner Goals: %2d (Conv: %4.2f%%)\n",
            r.team, r.matches, r.corner_rate_for, r.corner_rate_against, r.corner_goals_for, r.corner_conv_for * 100)
end
println("\nBottom 5 Corner Creators:")
for r in eachrow(last(team_df, 5))
    @printf("  %-25s | Matches: %3d | Corners/G: %5.2f | Conceded/G: %5.2f | Corner Goals: %2d (Conv: %4.2f%%)\n",
            r.team, r.matches, r.corner_rate_for, r.corner_rate_against, r.corner_goals_for, r.corner_conv_for * 100)
end

# Sort by Corner Offensive Conversion
sort!(team_df, :corner_conv_for, rev=true)
println("\n--- 2. CORNER OFFENSIVE CONVERSION EFFICIENCY (Goals / Corner Won) ---")
println("Top 5 Deadliest Set-Piece Finishers:")
for r in eachrow(first(team_df, 5))
    @printf("  %-25s | Corners: %4d | Goals: %2d | Conversion: %5.2f%%\n",
            r.team, r.corners_for, r.corner_goals_for, r.corner_conv_for * 100)
end
println("\nBottom 5 Set-Piece Finishers:")
for r in eachrow(last(team_df, 5))
    @printf("  %-25s | Corners: %4d | Goals: %2d | Conversion: %5.2f%%\n",
            r.team, r.corners_for, r.corner_goals_for, r.corner_conv_for * 100)
end

# Sort by Defensive Prevention (Lowest conversion against = best defense)
sort!(team_df, :corner_conv_against)
println("\n--- 3. CORNER DEFENSIVE RESISTANCE (Opponent Goals / Corner Conceded) ---")
println("Top 5 Best Set-Piece Defending Teams (Lowest Opponent Conversion):")
for r in eachrow(first(team_df, 5))
    @printf("  %-25s | Conceded: %4d | Opp Goals: %2d | Opp Conversion: %5.2f%%\n",
            r.team, r.corners_against, r.corner_goals_against, r.corner_conv_against * 100)
end
println("\nBottom 5 Worst Set-Piece Defending Teams (Highest Opponent Conversion):")
for r in eachrow(last(team_df, 5))
    @printf("  %-25s | Conceded: %4d | Opp Goals: %2d | Opp Conversion: %5.2f%%\n",
            r.team, r.corners_against, r.corner_goals_against, r.corner_conv_against * 100)
end

# 4. Variance & Signal-to-Noise Ratio Analysis
mean_conv = mean(team_df.corner_conv_for)
std_conv = std(team_df.corner_conv_for)
mean_gen = mean(team_df.corner_rate_for)
std_gen = std(team_df.corner_rate_for)

println("\n--- 4. CROSS-TEAM VARIATION & SIGNAL SPREAD ---")
@printf("Corner Generation Rate:       Mean = %.2f corners/game, Std = %.2f (CV = %.2f%%)\n",
        mean_gen, std_gen, std_gen / mean_gen * 100)
@printf("Corner Conversion Efficiency: Mean = %.2f%%, Std = %.2f%% (CV = %.2f%%)\n\n",
        mean_conv * 100, std_conv * 100, std_conv / mean_conv * 100)

# 5. Year-over-Year (YoY) Persistence (r_{t, t+1})
yoy = compute_yoy_persistence(df)
println("--- 5. YEAR-OVER-YEAR (YoY) AUTOCORRELATION & PERSISTENCE ---")
@printf("Sample Size: %d consecutive team-season pairs\n", yoy.n_pairs)
@printf("  1. Corner Generation (Corners Won / Game):      r = %+.4f  (HIGH PERSISTENCE)\n", yoy.r_corners_for)
@printf("  2. Corner Concession (Corners Conceded / Game):  r = %+.4f  (HIGH PERSISTENCE)\n", yoy.r_corners_against)
@printf("  3. Corner Goal Conversion (Goals / Corner):     r = %+.4f  (NOISE / LOW PERSISTENCE)\n", yoy.r_corner_conv)
@printf("  4. Total Corner Goals / Game:                   r = %+.4f  (MODERATE PERSISTENCE)\n\n", yoy.r_corner_goals_per_game)

println("================================================================================")
println("✓ STAGE 2 TEAM-LEVEL EDA COMPLETE")
println("================================================================================")
