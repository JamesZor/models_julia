# current_development/scottish_lower/r94_age_adjusted_wealth_quick_benchmark.jl
#
# Fast Pre-MCMC Validation of Age-Adjusted Production Wealth Candidates
# Tests various player-level age weighting functions ϕ(Age) via:
# 1. Pearson r & Spearman ρ with Match Goal Difference
# 2. Fast Poisson GLM Log-Loss / Deviance (IRLS in milliseconds)
# 3. Correlation with On-Pitch Net Dominance (Shots on Target, Possession, Net Fouls)

using BayesianFootball
using DataFrames
using Dates
using Statistics
using StatsBase
using Printf
using LibPQ
using GLM

ENV["BF_DB_URL"] = get(ENV, "BF_DB_URL", "postgresql://admin:CpPhGzIZ2qHtAh6cJT%2FHHFovs0CqfTx6@192.168.1.88:5433/betdb")

println("="^100)
println(" FAST PRE-MCMC BENCHMARK: AGE-ADJUSTED PRODUCTION WEALTH CANDIDATE FORMULATIONS")
println("="^100)

conn = LibPQ.Connection(ENV["BF_DB_URL"])

# Fetch player-level lineup valuations and ages
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

println("Loaded $(nrow(df_players)) player-match entries across 6 leagues.")

# Define Candidate Age-Weighting Curves ϕ(Age)
# 1. Baseline: Raw Wealth (No age adjustment)
phi_raw(age) = 1.0

# 2. Gaussian Bell Curve centered at Peak Physical Age (26.5y, σ=4.5y)
phi_gaussian_prime(age; mu=26.5, sigma=4.5) = exp(-((age - mu)^2) / (2 * sigma^2))

# 3. Asymmetric Experience Curve (Penalizes raw youth potential, sustains veteran game IQ)
# Youth (<23) heavily discounted, peak (23-29) at 1.0, veteran (30+) gentle taper
function phi_experience_taper(age)
    if age < 23.0
        return (age / 23.0)^2.0       # e.g., age 19 -> 0.68, age 21 -> 0.83
    elseif age <= 29.0
        return 1.0                   # Peak production plateau
    else
        return max(0.40, 1.0 - 0.04 * (age - 29.0)) # Age 34 -> 0.80, Age 37 -> 0.68
    end
end

# 4. Logistic Sigmoid Experience Transition (Youth discounted, all mature pros full weight)
phi_experience_sigmoid(age) = 1.0 / (1.0 + exp(-0.8 * (age - 23.5)))

# 5. Peak-Power Curve (Age / 26.0)^1.5 * exp(-0.08 * max(0, Age - 27.0))
phi_peak_power(age) = (age / 26.0)^1.5 * exp(-0.06 * max(0.0, age - 27.0))

candidates = [
    ("1. Raw Squad Wealth (Baseline)", phi_raw),
    ("2. Gaussian Peak Prime (μ=26.5, σ=4.5)", phi_gaussian_prime),
    ("3. Experience Taper (Youth Disc + Peak Plateau)", phi_experience_taper),
    ("4. Experience Sigmoid (Maturation Ramp)", phi_experience_sigmoid),
    ("5. Peak-Power Density Curve", phi_peak_power)
]

# Match level evaluation
matches_grouped = groupby(df_players, [:match_id, :tournament_id, :league_name, :home_score, :away_score, :goal_diff])

function evaluate_candidate(phi_fn)
    deltas = Float64[]
    gds    = Float64[]
    tourns = Int[]
    lnames = String[]
    h_scores = Int[]
    a_scores = Int[]

    for grp in matches_grouped
        home_players = filter(r -> r.is_home_team, grp)
        away_players = filter(r -> !r.is_home_team, grp)
        
        # Require reasonable starter count per side
        (nrow(home_players) >= 9 && nrow(away_players) >= 9) || continue
        
        # Compute Age-Adjusted Squad Wealth
        w_h = sum(r.market_value * phi_fn(coalesce(r.age, 26.3)) for r in eachrow(home_players))
        w_a = sum(r.market_value * phi_fn(coalesce(r.age, 26.3)) for r in eachrow(away_players))
        
        (w_h > 0 && w_a > 0) || continue
        
        delta_w = log(w_h) - log(w_a)
        push!(deltas, delta_w)
        push!(gds, grp.goal_diff[1])
        push!(tourns, grp.tournament_id[1])
        push!(lnames, grp.league_name[1])
        push!(h_scores, grp.home_score[1])
        push!(a_scores, grp.away_score[1])
    end

    df_eval = DataFrame(
        tournament_id = tourns,
        league_name = lnames,
        delta_w = deltas,
        goal_diff = gds,
        home_score = h_scores,
        away_score = a_scores
    )

    # Standardize by league
    gdf = groupby(df_eval, :league_name)
    transform!(gdf, :delta_w => (x -> (x .- mean(x)) ./ max(std(x), 1e-4)) => :delta_z)
    
    # Fast Poisson GLM for Home & Away goals
    # Log-intensity: η_h = μ_h + w * delta_z, η_a = μ_a - w * delta_z
    df_eval.is_scot = [t in (56, 57) for t in df_eval.tournament_id]
    
    return df_eval
end

println("\n" * "="^100)
println(" 1. SCOTTISH LOWER PERFORMANCE COMPARISON (League One & Two)")
println("="^100)
@printf("%-46s | %10s | %10s | %10s | %10s\n",
        "Candidate Formulation", "Pearson r", "Spearman ρ", "OLS Slope", "Deviance")
println("-"^100)

for (name, phi_fn) in candidates
    df_ev = evaluate_candidate(phi_fn)
    df_scot = filter(r -> r.is_scot, df_ev)
    
    r = cor(df_scot.goal_diff, df_scot.delta_z)
    rho = corspearman(df_scot.goal_diff, df_scot.delta_z)
    slope = cov(df_scot.goal_diff, df_scot.delta_z) / var(df_scot.delta_z)
    
    # Fast Poisson regression deviance
    glm_fit = glm(@formula(home_score ~ delta_z), df_scot, Poisson(), LogLink())
    dev = deviance(glm_fit)

    @printf("%-46s | %+10.4f | %+10.4f | %+10.4f | %10.2f\n",
            name, r, rho, slope, dev)
end
println("="^100)

println("\n" * "="^100)
println(" 2. ALL LEAGUES POOLED PERFORMANCE (10,100+ Matches)")
println("="^100)
@printf("%-46s | %10s | %10s | %10s | %10s\n",
        "Candidate Formulation", "Pearson r", "Spearman ρ", "OLS Slope", "Deviance")
println("-"^100)

for (name, phi_fn) in candidates
    df_ev = evaluate_candidate(phi_fn)
    
    r = cor(df_ev.goal_diff, df_ev.delta_z)
    rho = corspearman(df_ev.goal_diff, df_ev.delta_z)
    slope = cov(df_ev.goal_diff, df_ev.delta_z) / var(df_ev.delta_z)
    
    glm_fit = glm(@formula(home_score ~ delta_z), df_ev, Poisson(), LogLink())
    dev = deviance(glm_fit)

    @printf("%-46s | %+10.4f | %+10.4f | %+10.4f | %10.2f\n",
            name, r, rho, slope, dev)
end
println("="^100)
