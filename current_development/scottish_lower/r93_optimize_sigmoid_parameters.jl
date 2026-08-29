# current_development/scottish_lower/r93_optimize_sigmoid_parameters.jl
#
# Grid Search / Convexity Check for Sigmoid Parameters: (x0: inflection age, k: transition slope)
# Evaluated on Scottish Lower & All Leagues to check parameter stability and basin robustness.

using BayesianFootball
using DataFrames
using Dates
using Statistics
using StatsBase
using Printf
using LibPQ

ENV["BF_DB_URL"] = get(ENV, "BF_DB_URL", "postgresql://admin:CpPhGzIZ2qHtAh6cJT%2FHHFovs0CqfTx6@192.168.1.88:5433/betdb")

conn = LibPQ.Connection(ENV["BF_DB_URL"])

sql = """
SELECT 
    m.match_id,
    m.tournament_id,
    t.country || ' - ' || t.name AS league_name,
    m.start_timestamp,
    m.home_score,
    m.away_score,
    (m.home_score - m.away_score)::float8 AS goal_diff,
    l.is_home_team,
    l.player_id,
    COALESCE(l.proposed_market_value, 100000)::float8 AS market_value,
    (EXTRACT(EPOCH FROM m.start_timestamp) - (l.raw_data->'player'->>'dateOfBirthTimestamp')::bigint) / (365.25 * 86400)::float8 AS age
FROM sofascore.matches m
JOIN sofascore.tournaments t ON t.tournament_id = m.tournament_id
JOIN sofascore.match_player_lineups l ON l.match_id = m.match_id
WHERE m.tournament_id IN (1, 2, 3, 84, 56, 57)
  AND (l.substitute IS FALSE OR l.substitute IS NULL)
  AND (l.raw_data->'player'->>'dateOfBirthTimestamp') IS NOT NULL
ORDER BY m.start_timestamp;
"""

df_players = DataFrame(LibPQ.execute(conn, sql))
close(conn)

matches_grouped = groupby(df_players, [:match_id, :tournament_id, :league_name, :home_score, :away_score, :goal_diff])

# Grid search space:
# x0 (inflection age): 20.0 to 26.0 in steps of 0.5
# k (slope): 0.3 to 1.5 in steps of 0.15
x0_grid = 20.5:0.5:25.5
k_grid  = 0.4:0.2:1.4

println("="^90)
println(" GRID SEARCH OPTIMIZATION: SIGMOID PARAMETERS (x0 = Inflection Age, k = Steepness)")
println("="^90)

results = []

for x0 in x0_grid
    for k in k_grid
        phi(a) = 1.0 / (1.0 + exp(-k * (a - x0)))
        
        deltas = Float64[]
        gds    = Float64[]
        is_scot = Bool[]

        for grp in matches_grouped
            hp = filter(r -> r.is_home_team, grp)
            ap = filter(r -> !r.is_home_team, grp)
            (nrow(hp) >= 9 && nrow(ap) >= 9) || continue
            
            wh = sum(r.market_value * phi(coalesce(r.age, 26.3)) for r in eachrow(hp))
            wa = sum(r.market_value * phi(coalesce(r.age, 26.3)) for r in eachrow(ap))
            (wh > 0 && wa > 0) || continue
            
            push!(deltas, log(wh) - log(wa))
            push!(gds, grp.goal_diff[1])
            push!(is_scot, grp.tournament_id[1] in (56, 57))
        end

        # Correlations
        r_all = cor(gds, deltas)
        
        scot_idx = findall(is_scot)
        r_scot = cor(gds[scot_idx], deltas[scot_idx])

        push!(results, (x0=x0, k=k, r_scot=r_scot, r_all=r_all))
    end
end

df_res = DataFrame(results)

println("\nTOP 10 PARAMETER COMBINATIONS FOR SCOTTISH LOWER:")
println("-"^90)
top_scot = first(sort(df_res, :r_scot, rev=true), 10)
for r in eachrow(top_scot)
    @printf("  x0 (Inflection) = %4.1f yrs | k (Slope) = %4.2f | Scot Lower r = %+7.4f | All Leagues r = %+7.4f\n",
            r.x0, r.k, r.r_scot, r.r_all)
end

println("\nTOP 10 PARAMETER COMBINATIONS FOR ALL LEAGUES:")
println("-"^90)
top_all = first(sort(df_res, :r_all, rev=true), 10)
for r in eachrow(top_all)
    @printf("  x0 (Inflection) = %4.1f yrs | k (Slope) = %4.2f | Scot Lower r = %+7.4f | All Leagues r = %+7.4f\n",
            r.x0, r.k, r.r_scot, r.r_all)
end

println("\n" * "="^90)
println(" BASELINE COMPARISONS (Raw Wealth without Age Adjustment):")
println("  - Scottish Lower Raw r = +0.1535")
println("  - All Leagues Raw r    = +0.2534")
println("="^90)
