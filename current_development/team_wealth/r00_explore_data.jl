# current_development/team_wealth/r00_explore_data.jl
#
# ==============================================================================
# RUNNER / PLAYGROUND: Step-by-Step Data Validation for Team Wealth
# ==============================================================================
#
# Execute line-by-line in Julia REPL or run as a standalone script:
#   julia --project=. current_development/team_wealth/r00_explore_data.jl
#
# ==============================================================================

include("l01_wealth_data.jl")

println("===================================================================")
println("STEP 1: TESTING DATABASE CONNECTION & VALUATION EXTRACTOR")
println("===================================================================\n")

# 1. Connect
conn = wealth_db_connect()
println("✓ Connected to PostgreSQL.")

# 2. Fetch Catalog
println("\nFetching player valuations from match_incidents and provisional lineups...")
val_cat = fetch_player_valuations(conn, tournament_ids=[79, 718])
println("✓ Found $(nrow(val_cat)) unique players with market valuations.")
println("\nTop 5 Valued Players in Catalog:")
sort!(val_cat, :market_value, rev=true)
for r in eachrow(first(val_cat, 5))
    @printf("  ID: %6d | %-25s | Pos: %2s | Val: €%10.0f\n", 
            r.player_id, r.player_name, coalesce(r.player_position, "?"), r.market_value)
end

# 3. Match Lineups
println("\nFetching match starting lineups for Tournament 79 (Premier Division)...")
lineup_vals = fetch_match_lineup_values(conn, val_cat, tournament_id=79)
println("✓ Loaded $(nrow(lineup_vals)) starting player rows.")

# Coverage statistics:
n_with_val = count(r -> !ismissing(r.market_value), eachrow(lineup_vals))
pct_cov = 100.0 * n_with_val / nrow(lineup_vals)
@printf("✓ Player-level valuation match rate: %d / %d (%.1f%%)\n", n_with_val, nrow(lineup_vals), pct_cov)

# 4. Match Wealth Table
println("\nAggregating match-level starting XI wealth and computing seasonal Z-scores...")
match_wealth = build_match_wealth_table(lineup_vals)
println("✓ Successfully processed $(nrow(match_wealth)) matches.")

# 5. Team Hierarchy Display
println("\n===================================================================")
println("TEAM WEALTH RANKINGS (Mean Starting XI Valuation in EUR)")
println("===================================================================")
team_summary = combine(groupby(lineup_vals, :home_team),
    :clean_value => (v -> sum(v)/11) => :mean_starting_xi_val,
    :clean_value => mean => :mean_player_val,
    :match_id => (v -> length(unique(v))) => :n_matches
)
sort!(team_summary, :mean_starting_xi_val, rev=true)

for r in eachrow(team_summary)
    @printf("  %-25s : Starting XI: €%10.0f  (Avg Player: €%8.0f | %2d games)\n", 
            r.home_team, r.mean_starting_xi_val, r.mean_player_val, r.n_matches)
end

# 6. Sample Output of Processed Feature
println("\n===================================================================")
println("SAMPLE MATCH-LEVEL WEALTH DELTAS (ΔW = w_home - w_away)")
println("===================================================================")
for r in eachrow(first(match_wealth, 8))
    @printf("  Match %8d | %-20s (€%6.2fM) vs %-20s (€%6.2fM) | ΔW = %+6.3f\n",
            r.match_id, r.home_team, r.home_xi_val/1e6, r.away_team, r.away_xi_val/1e6, r.delta_w)
end

println("\n===================================================================")
println("✓ STEP 1 DATA PROCESSOR VERIFIED & READY FOR LAYER 1 FEATURE SET")
println("===================================================================")

close(conn)
