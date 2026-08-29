# current_development/scottish_lower/r92_skewed_age_distributions.jl
#
# Testing Skewed & Asymmetric Age-Weighting Distributions:
# 1. Shifted Gamma Distribution (Steep youth ramp, heavy veteran right-tail)
# 2. Log-Normal Distribution
# 3. Skew-Normal Distribution
# 4. Generalized Richards Sigmoid (Asymmetric transition)
# 5. Gaussian Benchmark & Pure Sigmoid Benchmark

using BayesianFootball
using DataFrames
using Dates
using Statistics
using StatsBase
using Printf
using LibPQ
using GLM
using SpecialFunctions

ENV["BF_DB_URL"] = get(ENV, "BF_DB_URL", "postgresql://admin:CpPhGzIZ2qHtAh6cJT%2FHHFovs0CqfTx6@192.168.1.88:5433/betdb")

println("="^100)
println(" SKEWED & ASYMMETRIC AGE DISTRIBUTION BENCHMARK (GAMMA, LOG-NORMAL, SKEW-NORMAL, RICHARDS)")
println("="^100)

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

# ------------------------------------------------------------------------------
# Candidate Curves Definition
# ------------------------------------------------------------------------------

# 1. Standard Shifted Gamma Distribution normalized to peak at 1.0
# Mode = A_peak, Career entry A0 = 16.0
function phi_shifted_gamma(age; A_peak=26.5, A0=16.0, alpha=3.5)
    age <= A0 && return 0.0
    x = age - A0
    x_peak = A_peak - A0
    beta = (alpha - 1.0) / x_peak
    # Normalized gamma mode: (x / x_peak)^(alpha - 1) * exp(-beta * (x - x_peak))
    val = (x / x_peak)^(alpha - 1.0) * exp(-beta * (x - x_peak))
    return clamp(val, 0.0, 1.0)
end

# 2. Shifted Log-Normal Distribution normalized to peak at 1.0
# Mode = A_peak, Career entry A0 = 16.0
function phi_shifted_lognormal(age; A_peak=26.5, A0=16.0, sigma=0.40)
    age <= A0 && return 0.0
    x = age - A0
    x_peak = A_peak - A0
    # Mode of lognormal is exp(mu - sigma^2) => mu = log(x_peak) + sigma^2
    mu = log(x_peak) + sigma^2
    # f(x) / f(x_peak)
    val = (x_peak / x) * exp(-((log(x) - mu)^2 - (log(x_peak) - mu)^2) / (2 * sigma^2))
    return clamp(val, 0.0, 1.0)
end

# 3. Asymmetric Skew-Normal Distribution (Azzalini formulation)
# Positive alpha_skew produces right-tail skewness (long veteran tail)
function phi_skew_normal(age; xi=25.0, omega=5.0, alpha_skew=2.5)
    z = (age - xi) / omega
    # Standard normal PDF
    pdf_z = exp(-0.5 * z^2) / sqrt(2 * pi)
    # Standard normal CDF
    cdf_z = 0.5 * (1.0 + erf(alpha_skew * z / sqrt(2.0)))
    val = 2.0 * pdf_z * cdf_z
    return val
end

# 4. Generalized Richards Sigmoid (Asymmetric S-curve with parameter nu)
function phi_richards_sigmoid(age; x0=23.0, k=0.8, nu=0.5)
    # nu < 1 produces a slower, more gradual upper approach
    return 1.0 / (1.0 + exp(-k * (age - x0)))^(1.0 / nu)
end

# 5. Baseline Benchmarks
phi_raw(age) = 1.0
phi_gaussian(age) = exp(-((age - 26.5)^2) / (2 * (4.5)^2))
phi_standard_sigmoid(age) = 1.0 / (1.0 + exp(-0.8 * (age - 23.5)))

# Parameter sweep on Gamma: test alpha in [2.0, 3.0, 4.0, 5.0, 6.0] and A_peak in [25.5, 26.5, 27.5]
gamma_candidates = [
    ("Shifted Gamma (Peak=26.5, α=2.5, heavy tail)", a -> phi_shifted_gamma(a; A_peak=26.5, alpha=2.5)),
    ("Shifted Gamma (Peak=26.5, α=3.5, mod tail)",   a -> phi_shifted_gamma(a; A_peak=26.5, alpha=3.5)),
    ("Shifted Gamma (Peak=26.5, α=5.0, tight tail)", a -> phi_shifted_gamma(a; A_peak=26.5, alpha=5.0)),
    ("Shifted Gamma (Peak=27.5, α=3.5, veteran peak)", a -> phi_shifted_gamma(a; A_peak=27.5, alpha=3.5)),
    ("Shifted Gamma (Peak=25.5, α=3.5, early peak)", a -> phi_shifted_gamma(a; A_peak=25.5, alpha=3.5))
]

lognormal_candidates = [
    ("Shifted Log-Normal (Peak=26.5, σ=0.35)", a -> phi_shifted_lognormal(a; A_peak=26.5, sigma=0.35)),
    ("Shifted Log-Normal (Peak=26.5, σ=0.50)", a -> phi_shifted_lognormal(a; A_peak=26.5, sigma=0.50))
]

richards_candidates = [
    ("Richards Sigmoid (x0=23.0, k=0.8, ν=0.5 - heavy upper)", a -> phi_richards_sigmoid(a; x0=23.0, k=0.8, nu=0.5)),
    ("Richards Sigmoid (x0=23.0, k=0.8, ν=2.0 - sharp upper)", a -> phi_richards_sigmoid(a; x0=23.0, k=0.8, nu=2.0))
]

all_models = [
    ("1. Baseline: Raw Wealth (No Age Adj)", phi_raw),
    ("2. Symmetric Gaussian (μ=26.5, σ=4.5)", phi_gaussian),
    ("3. Standard Sigmoid (x0=23.5, k=0.8)", phi_standard_sigmoid),
    gamma_candidates...,
    lognormal_candidates...,
    richards_candidates...
]

function eval_curve(phi_fn)
    deltas = Float64[]
    gds    = Float64[]
    tourns = Int[]
    lnames = String[]

    # Pre-compute mode scaling if needed
    test_ages = 16.0:0.1:40.0
    max_val = maximum(phi_fn(a) for a in test_ages)
    norm_phi(a) = max_val > 0 ? (phi_fn(a) / max_val) : phi_fn(a)

    for grp in matches_grouped
        hp = filter(r -> r.is_home_team, grp)
        ap = filter(r -> !r.is_home_team, grp)
        (nrow(hp) >= 9 && nrow(ap) >= 9) || continue
        
        wh = sum(r.market_value * norm_phi(coalesce(r.age, 26.3)) for r in eachrow(hp))
        wa = sum(r.market_value * norm_phi(coalesce(r.age, 26.3)) for r in eachrow(ap))
        (wh > 0 && wa > 0) || continue
        
        push!(deltas, log(wh) - log(wa))
        push!(gds, grp.goal_diff[1])
        push!(tourns, grp.tournament_id[1])
        push!(lnames, grp.league_name[1])
    end

    df_eval = DataFrame(tournament_id = tourns, league_name = lnames, delta_w = deltas, goal_diff = gds)
    gdf = groupby(df_eval, :league_name)
    transform!(gdf, :delta_w => (x -> (x .- mean(x)) ./ max(std(x), 1e-4)) => :delta_z)
    df_eval.is_scot = [t in (56, 57) for t in df_eval.tournament_id]
    return df_eval
end

println("\n" * "="^100)
println(" SKEWED DISTRIBUTION BENCHMARK RESULTS")
println("="^100)
@printf("%-52s | %10s | %10s | %10s | %10s\n",
        "Model Formulation", "Scot Lower r", "Scot Lower ρ", "All Leagues r", "All Leagues ρ")
println("-"^100)

for (name, phi_fn) in all_models
    df_ev = eval_curve(phi_fn)
    
    # Scottish Lower
    scot_sub = filter(r -> r.is_scot, df_ev)
    r_scot = cor(scot_sub.goal_diff, scot_sub.delta_z)
    rho_scot = corspearman(scot_sub.goal_diff, scot_sub.delta_z)
    
    # All Leagues
    r_all = cor(df_ev.goal_diff, df_ev.delta_z)
    rho_all = corspearman(df_ev.goal_diff, df_ev.delta_z)

    @printf("%-52s | %+10.4f | %+10.4f | %+10.4f | %+10.4f\n",
            name, r_scot, rho_scot, r_all, rho_all)
end
println("="^100)
