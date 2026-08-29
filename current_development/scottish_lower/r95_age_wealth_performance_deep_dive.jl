# current_development/scottish_lower/r95_age_wealth_performance_deep_dive.jl
#
# Comprehensive Deep Dive: Squad Age vs Market Wealth Interaction on Fouls,
# Possession, Cards, Shots on Target, and Goal Differentials across Scottish Lower & English Tiers.

using BayesianFootball
using DataFrames
using Dates
using Statistics
using StatsBase
using Printf
using LibPQ

ENV["BF_DB_URL"] = get(ENV, "BF_DB_URL", "postgresql://admin:CpPhGzIZ2qHtAh6cJT%2FHHFovs0CqfTx6@192.168.1.88:5433/betdb")

println("="^100)
println(" SQUAD AGE × SQUAD WEALTH DEEP DIVE: PERFORMANCE, DISCIPLINE & MATCH DYNAMICS")
println("="^100)

# ==============================================================================
# 1. Direct PostgreSQL Extraction Across Tournaments
# ==============================================================================
conn = LibPQ.Connection(ENV["BF_DB_URL"])

query_sql = """
WITH starting_xi AS (
    SELECT 
        m.match_id,
        m.tournament_id,
        t.country || ' - ' || t.name AS league_name,
        m.start_timestamp,
        m.home_team,
        m.away_team,
        m.home_score,
        m.away_score,
        (m.home_score - m.away_score)::float8 AS goal_diff,
        
        -- Home Starting XI Metrics
        AVG(CASE WHEN l.is_home_team THEN 
            (EXTRACT(EPOCH FROM m.start_timestamp) - (l.raw_data->'player'->>'dateOfBirthTimestamp')::bigint) / (365.25 * 86400) 
        END)::float8 AS home_avg_age,
        
        SUM(CASE WHEN l.is_home_team THEN 
            COALESCE(l.proposed_market_value, 100000) 
        END)::float8 AS home_total_wealth,
        
        SUM(CASE WHEN l.is_home_team AND (EXTRACT(EPOCH FROM m.start_timestamp) - (l.raw_data->'player'->>'dateOfBirthTimestamp')::bigint) / (365.25 * 86400) < 23 THEN 
            COALESCE(l.proposed_market_value, 100000) ELSE 0 
        END)::float8 AS home_young_wealth,

        SUM(CASE WHEN l.is_home_team AND (EXTRACT(EPOCH FROM m.start_timestamp) - (l.raw_data->'player'->>'dateOfBirthTimestamp')::bigint) / (365.25 * 86400) >= 23 
                                     AND (EXTRACT(EPOCH FROM m.start_timestamp) - (l.raw_data->'player'->>'dateOfBirthTimestamp')::bigint) / (365.25 * 86400) < 29 THEN 
            COALESCE(l.proposed_market_value, 100000) ELSE 0 
        END)::float8 AS home_peak_wealth,

        SUM(CASE WHEN l.is_home_team AND (EXTRACT(EPOCH FROM m.start_timestamp) - (l.raw_data->'player'->>'dateOfBirthTimestamp')::bigint) / (365.25 * 86400) >= 29 THEN 
            COALESCE(l.proposed_market_value, 100000) ELSE 0 
        END)::float8 AS home_vet_wealth,

        COUNT(CASE WHEN l.is_home_team AND l.proposed_market_value IS NOT NULL THEN 1 END) AS home_known_wealth_count,
        COUNT(CASE WHEN l.is_home_team AND (l.raw_data->'player'->>'dateOfBirthTimestamp') IS NOT NULL THEN 1 END) AS home_known_dob_count,
        COUNT(CASE WHEN l.is_home_team THEN 1 END) AS home_starter_count,

        -- Away Starting XI Metrics
        AVG(CASE WHEN NOT l.is_home_team THEN 
            (EXTRACT(EPOCH FROM m.start_timestamp) - (l.raw_data->'player'->>'dateOfBirthTimestamp')::bigint) / (365.25 * 86400) 
        END)::float8 AS away_avg_age,
        
        SUM(CASE WHEN NOT l.is_home_team THEN 
            COALESCE(l.proposed_market_value, 100000) 
        END)::float8 AS away_total_wealth,

        SUM(CASE WHEN NOT l.is_home_team AND (EXTRACT(EPOCH FROM m.start_timestamp) - (l.raw_data->'player'->>'dateOfBirthTimestamp')::bigint) / (365.25 * 86400) < 23 THEN 
            COALESCE(l.proposed_market_value, 100000) ELSE 0 
        END)::float8 AS away_young_wealth,

        SUM(CASE WHEN NOT l.is_home_team AND (EXTRACT(EPOCH FROM m.start_timestamp) - (l.raw_data->'player'->>'dateOfBirthTimestamp')::bigint) / (365.25 * 86400) >= 23 
                                         AND (EXTRACT(EPOCH FROM m.start_timestamp) - (l.raw_data->'player'->>'dateOfBirthTimestamp')::bigint) / (365.25 * 86400) < 29 THEN 
            COALESCE(l.proposed_market_value, 100000) ELSE 0 
        END)::float8 AS away_peak_wealth,

        SUM(CASE WHEN NOT l.is_home_team AND (EXTRACT(EPOCH FROM m.start_timestamp) - (l.raw_data->'player'->>'dateOfBirthTimestamp')::bigint) / (365.25 * 86400) >= 29 THEN 
            COALESCE(l.proposed_market_value, 100000) ELSE 0 
        END)::float8 AS away_vet_wealth,

        COUNT(CASE WHEN NOT l.is_home_team AND l.proposed_market_value IS NOT NULL THEN 1 END) AS away_known_wealth_count,
        COUNT(CASE WHEN NOT l.is_home_team AND (l.raw_data->'player'->>'dateOfBirthTimestamp') IS NOT NULL THEN 1 END) AS away_known_dob_count,
        COUNT(CASE WHEN NOT l.is_home_team THEN 1 END) AS away_starter_count

    FROM sofascore.matches m
    JOIN sofascore.tournaments t ON t.tournament_id = m.tournament_id
    JOIN sofascore.match_player_lineups l ON l.match_id = m.match_id
    WHERE m.tournament_id IN (1, 2, 3, 84, 56, 57)
      AND (l.substitute IS FALSE OR l.substitute IS NULL)
    GROUP BY m.match_id, m.tournament_id, t.country, t.name, m.start_timestamp, m.home_team, m.away_team, m.home_score, m.away_score
    HAVING COUNT(CASE WHEN l.is_home_team THEN 1 END) >= 9 
       AND COUNT(CASE WHEN NOT l.is_home_team THEN 1 END) >= 9
)
SELECT 
    s.*,
    (s.home_avg_age - s.away_avg_age) AS delta_age,
    (LN(s.home_total_wealth) - LN(s.away_total_wealth)) AS delta_wealth_log,
    (LN(s.home_young_wealth + 10000) - LN(s.away_young_wealth + 10000)) AS delta_young_wealth,
    (LN(s.home_peak_wealth + 10000) - LN(away_peak_wealth + 10000)) AS delta_peak_wealth,
    (LN(s.home_vet_wealth + 10000) - LN(away_vet_wealth + 10000)) AS delta_vet_wealth
FROM starting_xi s
WHERE s.home_avg_age IS NOT NULL AND s.away_avg_age IS NOT NULL
  AND s.home_total_wealth > 0 AND s.away_total_wealth > 0
ORDER BY s.start_timestamp;
"""

df_xi = DataFrame(LibPQ.execute(conn, query_sql))
println("Loaded $(nrow(df_xi)) match starting-XI profiles across 6 divisions.")

# Fetch detailed BBC / Sofascore match stats (fouls, possession, yellow cards, shots)
stats_sql = """
SELECT 
    s.match_id,
    s.stat_type,
    s.home_value,
    s.away_value
FROM bbc.match_stats s
JOIN sofascore.matches m ON s.match_id = m.match_id
WHERE m.tournament_id IN (1, 2, 3, 84, 56, 57)
  AND s.stat_type IN ('foulsCommitted', 'possessionPercentage', 'totalYellowCard', 'shotsTotal', 'shotsOnTarget', 'cornersWon')
"""
stats_raw = DataFrame(LibPQ.execute(conn, stats_sql))
close(conn)
println("Loaded $(nrow(stats_raw)) detailed match statistics from bbc.match_stats.")

# Unstack match stats
stats_clean = unique(stats_raw, [:match_id, :stat_type])
stats_h = unstack(stats_clean, :match_id, :stat_type, :home_value, renamecols = x -> "$(x)_h")
stats_a = unstack(stats_clean, :match_id, :stat_type, :away_value, renamecols = x -> "$(x)_a")
stats_wide = innerjoin(stats_h, stats_a, on=:match_id)

df_all = leftjoin(df_xi, stats_wide, on=:match_id)
df_all.date = [Date(DateTime(r.start_timestamp)) for r in eachrow(df_all)]
df_all.is_scot_lower = [r.tournament_id in (56, 57) for r in eachrow(df_all)]

# Standardize Delta Wealth and Delta Age by League
gdf = groupby(df_all, :league_name)
transform!(gdf, :delta_wealth_log => (x -> (x .- mean(x)) ./ max(std(x), 1e-4)) => :delta_z_wealth)
transform!(gdf, :delta_age => (x -> (x .- mean(x)) ./ max(std(x), 1e-4)) => :delta_z_age)

# Net stats
df_all.delta_fouls = [(!ismissing(r.foulsCommitted_h) && !ismissing(r.foulsCommitted_a)) ? Float64(r.foulsCommitted_h - r.foulsCommitted_a) : missing for r in eachrow(df_all)]
df_all.delta_poss  = [(!ismissing(r.possessionPercentage_h) && !ismissing(r.possessionPercentage_a)) ? Float64(r.possessionPercentage_h - r.possessionPercentage_a) : missing for r in eachrow(df_all)]
df_all.delta_yellows = [(!ismissing(r.totalYellowCard_h) && !ismissing(r.totalYellowCard_a)) ? Float64(r.totalYellowCard_h - r.totalYellowCard_a) : missing for r in eachrow(df_all)]
df_all.delta_shots_on_target = [(!ismissing(r.shotsOnTarget_h) && !ismissing(r.shotsOnTarget_a)) ? Float64(r.shotsOnTarget_h - r.shotsOnTarget_a) : missing for r in eachrow(df_all)]

# ==============================================================================
# 2. League-by-League Correlation Matrix
# ==============================================================================
println("\n" * "="^100)
println(" 1. UNIVARIATE & MULTIVARIATE CORRELATION ACROSS LEAGUES")
println("="^100)
@printf("%-28s | %5s | %7s | %7s | %7s | %7s | %7s | %7s\n",
        "League", "N", "r(GD,ΔW)", "r(GD,ΔA)", "r(ΔW,ΔA)", "β_wealth", "β_age", "r(NetFoul,ΔW)")
println("-"^100)

for sub_gdf in groupby(df_all, :league_name)
    lname = first(sub_gdf.league_name)
    n = nrow(sub_gdf)
    r_w = cor(sub_gdf.goal_diff, sub_gdf.delta_wealth_log)
    r_a = cor(sub_gdf.goal_diff, sub_gdf.delta_age)
    r_wa = cor(sub_gdf.delta_wealth_log, sub_gdf.delta_age)
    
    # OLS coefficients: GD ~ ΔW + ΔA
    var_w = var(sub_gdf.delta_wealth_log)
    var_a = var(sub_gdf.delta_age)
    cov_wa = cov(sub_gdf.delta_wealth_log, sub_gdf.delta_age)
    cov_wy = cov(sub_gdf.delta_wealth_log, sub_gdf.goal_diff)
    cov_ay = cov(sub_gdf.delta_age, sub_gdf.goal_diff)
    det = var_w * var_a - cov_wa^2
    b_w = (cov_wy * var_a - cov_ay * cov_wa) / det
    b_a = (cov_ay * var_w - cov_wy * cov_wa) / det

    foul_sub = filter(r -> !ismissing(r.delta_fouls), sub_gdf)
    r_f = nrow(foul_sub) >= 20 ? cor(foul_sub.delta_wealth_log, foul_sub.delta_fouls) : NaN

    @printf("%-28s | %5d | %+7.4f | %+7.4f | %+7.4f | %+7.4f | %+7.4f | %+7.4f\n",
            lname, n, r_w, r_a, r_wa, b_w, b_a, r_f)
end
println("="^100)

# ==============================================================================
# 3. Age-Bracket Wealth Breakdown: Young vs Peak vs Veteran
# ==============================================================================
println("\n" * "="^100)
println(" 2. CORRELATION OF GOAL DIFFERENTIAL BY AGE-WEALTH BRACKET")
println("="^100)
@printf("%-28s | %5s | %10s | %10s | %10s | %10s\n",
        "League", "N", "Total ΔW", "Peak(23-28)", "Young(<23)", "Vet(29+)")
println("-"^100)

for sub_gdf in groupby(df_all, :league_name)
    lname = first(sub_gdf.league_name)
    n = nrow(sub_gdf)
    r_tot = cor(sub_gdf.goal_diff, sub_gdf.delta_wealth_log)
    r_peak = cor(sub_gdf.goal_diff, sub_gdf.delta_peak_wealth)
    r_young = cor(sub_gdf.goal_diff, sub_gdf.delta_young_wealth)
    r_vet = cor(sub_gdf.goal_diff, sub_gdf.delta_vet_wealth)
    @printf("%-28s | %5d | %+10.4f | %+10.4f | %+10.4f | %+10.4f\n",
            lname, n, r_tot, r_peak, r_young, r_vet)
end
println("="^100)

# ==============================================================================
# 4. 2D Cross-Tabulation: Squad Wealth Tier × Squad Age Tier (Scottish Lower vs All)
# ==============================================================================
function classify_wealth_tier(z)
    if z < -0.75
        return "1. Away Wealth Advantage (Δz < -0.75)"
    elseif z <= 0.75
        return "2. Wealth Parity (|Δz| ≤ 0.75)"
    else
        return "3. Home Wealth Advantage (Δz > +0.75)"
    end
end

function classify_age_tier(z)
    if z < -0.75
        return "1. Home Younger (Δz_age < -0.75)"
    elseif z <= 0.75
        return "2. Age Parity (|Δz_age| ≤ 0.75)"
    else
        return "3. Home Older/Experienced (Δz_age > +0.75)"
    end
end

df_all.wealth_tier = [classify_wealth_tier(z) for z in df_all.delta_z_wealth]
df_all.age_tier    = [classify_age_tier(z) for z in df_all.delta_z_age]

function print_2d_crosstab(df_subset, dataset_name)
    println("\n" * "="^100)
    println(" 3. 2D CROSS-TABULATION: WEALTH TIER × AGE PROFILE ($dataset_name)")
    println("="^100)
    @printf("%-36s | %-32s | %4s | %6s | %6s | %7s | %7s | %6s\n",
            "Wealth Tier", "Age Tier", "N", "H-Win%", "A-Win%", "Avg GD", "NetFouls", "H-Poss")
    println("-"^100)
    
    cross_df = combine(groupby(df_subset, [:wealth_tier, :age_tier]),
        nrow => :n,
        :goal_diff => (g -> mean(g .> 0) * 100) => :h_win_pct,
        :goal_diff => (g -> mean(g .< 0) * 100) => :a_win_pct,
        :goal_diff => mean => :avg_gd,
        :delta_fouls => (f -> isempty(skipmissing(f)) ? NaN : mean(skipmissing(f))) => :avg_net_fouls,
        :possessionPercentage_h => (p -> isempty(skipmissing(p)) ? NaN : mean(skipmissing(p))) => :avg_h_poss
    )
    sort!(cross_df, [:wealth_tier, :age_tier])
    
    for r in eachrow(cross_df)
        @printf("%-36s | %-32s | %4d | %5.1f%% | %5.1f%% | %+7.3f | %+7.2f | %5.1f%%\n",
                r.wealth_tier, r.age_tier, r.n, r.h_win_pct, r.a_win_pct, r.avg_gd, r.avg_net_fouls, r.avg_h_poss)
    end
    println("="^100)
end

print_2d_crosstab(filter(r -> r.is_scot_lower, df_all), "SCOTTISH LOWER (League One & Two)")
print_2d_crosstab(df_all, "ALL LEAGUES POOLED (England PL, Champ, L1, L2 + Scotland L1, L2)")

# ==============================================================================
# 5. Discipline & Match Dynamics (Fouls, Cards, Possession, SOT)
# ==============================================================================
println("\n" * "="^100)
println(" 4. REGRESSION SLOPES: SQUAD WEALTH & AGE vs MATCH DYNAMICS")
println("="^100)

for (label, df_sub) in [("Scottish Lower", filter(r -> r.is_scot_lower, df_all)), ("All Leagues", df_all)]
    println("  [$label]")
    
    # Net Fouls
    df_f = filter(r -> !ismissing(r.delta_fouls), df_sub)
    if nrow(df_f) >= 50
        r_fw = cor(df_f.delta_z_wealth, df_f.delta_fouls)
        r_fa = cor(df_f.delta_z_age, df_f.delta_fouls)
        @printf("    Net Fouls Committed (Home - Away):    r(Δz_wealth) = %+6.3f,  r(Δz_age) = %+6.3f  (N = %d)\n",
                r_fw, r_fa, nrow(df_f))
    end

    # Possession
    df_p = filter(r -> !ismissing(r.delta_poss), df_sub)
    if nrow(df_p) >= 50
        r_pw = cor(df_p.delta_z_wealth, df_p.delta_poss)
        r_pa = cor(df_p.delta_z_age, df_p.delta_poss)
        @printf("    Net Possession %% (Home - Away):       r(Δz_wealth) = %+6.3f,  r(Δz_age) = %+6.3f  (N = %d)\n",
                r_pw, r_pa, nrow(df_p))
    end

    # Yellow Cards
    df_c = filter(r -> !ismissing(r.delta_yellows), df_sub)
    if nrow(df_c) >= 50
        r_cw = cor(df_c.delta_z_wealth, df_c.delta_yellows)
        r_ca = cor(df_c.delta_z_age, df_c.delta_yellows)
        @printf("    Net Yellow Cards (Home - Away):       r(Δz_wealth) = %+6.3f,  r(Δz_age) = %+6.3f  (N = %d)\n",
                r_cw, r_ca, nrow(df_c))
    end

    # Shots on Target
    df_s = filter(r -> !ismissing(r.delta_shots_on_target), df_sub)
    if nrow(df_s) >= 50
        r_sw = cor(df_s.delta_z_wealth, df_s.delta_shots_on_target)
        r_sa = cor(df_s.delta_z_age, df_s.delta_shots_on_target)
        @printf("    Net Shots on Target (Home - Away):    r(Δz_wealth) = %+6.3f,  r(Δz_age) = %+6.3f  (N = %d)\n",
                r_sw, r_sa, nrow(df_s))
    end
    println()
end
println("="^100)
