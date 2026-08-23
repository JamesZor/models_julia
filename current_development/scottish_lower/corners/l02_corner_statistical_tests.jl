# current_development/scottish_lower/corners/l02_corner_statistical_tests.jl
#
# Statistical Testing Framework for Corner Kick & Set-Piece Decomposition

using DataFrames
using Statistics
using Distributions
using HypothesisTests

"""
    compute_dispersion_stats(counts::Vector{<:Real})

Computes mean, variance, index of dispersion (Var/Mean), and chi-squared p-value vs Poisson null.
"""
function compute_dispersion_stats(counts::Vector{<:Real})
    n = length(counts)
    m = mean(counts)
    v = var(counts)
    idx = v / m
    # Chi-sq test statistic for Poisson dispersion: (n-1) * s^2 / m ~ Chisq(n-1)
    chisq_stat = (n - 1) * v / m
    p_val = 1.0 - cdf(Chisq(n - 1), chisq_stat)
    return (mean = m, var = v, dispersion_index = idx, chisq_stat = chisq_stat, p_value = p_val, is_overdispersed = (idx > 1.05 && p_val < 0.01))
end

"""
    test_corner_home_advantage(home_counts::Vector{<:Real}, away_counts::Vector{<:Real})

Tests if home corners significantly exceed away corners via paired t-test.
"""
function test_corner_home_advantage(home_counts::Vector{<:Real}, away_counts::Vector{<:Real})
    diffs = home_counts .- away_counts
    tt = OneSampleTTest(diffs)
    m_h = mean(home_counts)
    m_a = mean(away_counts)
    ratio = m_h / m_a
    return (
        mean_home = m_h,
        mean_away = m_a,
        mean_diff = mean(diffs),
        ratio_ha = ratio,
        t_stat = tt.t,
        p_value = pvalue(tt)
    )
end

"""
    compute_team_corner_metrics(df::DataFrame)

Aggregates team-level corner creation, concession, offensive conversion, and defensive prevention.
"""
function compute_team_corner_metrics(df::DataFrame)
    teams = unique(vcat(df.home_team, df.away_team))
    records = []
    
    for t in teams
        home_m = filter(r -> r.home_team == t, df)
        away_m = filter(r -> r.away_team == t, df)
        
        n_matches = nrow(home_m) + nrow(away_m)
        n_matches == 0 && continue
        
        # Corners For & Against
        corners_for = sum(home_m.corners_h) + sum(away_m.corners_a)
        corners_against = sum(home_m.corners_a) + sum(away_m.corners_h)
        
        # Corner Goals For & Against
        corner_goals_for = sum(home_m.corner_goals_h) + sum(away_m.corner_goals_a)
        corner_goals_against = sum(home_m.corner_goals_a) + sum(away_m.corner_goals_h)
        
        rate_for = corners_for / n_matches
        rate_against = corners_against / n_matches
        
        conv_for = corners_for > 0 ? (corner_goals_for / corners_for) : 0.0
        conv_against = corners_against > 0 ? (corner_goals_against / corners_against) : 0.0
        
        push!(records, (
            team = t,
            matches = n_matches,
            corners_for = corners_for,
            corners_against = corners_against,
            corner_rate_for = rate_for,
            corner_rate_against = rate_against,
            corner_goals_for = corner_goals_for,
            corner_goals_against = corner_goals_against,
            corner_conv_for = conv_for,
            corner_conv_against = conv_against
        ))
    end
    
    return DataFrame(records)
end

"""
    compute_yoy_persistence(df::DataFrame)

Computes year-over-year correlation (r_t,t+1) for corner generation, conceding, and goal conversion.
"""
function compute_yoy_persistence(df::DataFrame)
    seasons = sort(unique(df.season))
    pairs_for = Tuple{Float64, Float64}[]
    pairs_against = Tuple{Float64, Float64}[]
    pairs_conv = Tuple{Float64, Float64}[]
    pairs_goals = Tuple{Float64, Float64}[]
    
    for i in 1:(length(seasons) - 1)
        s1 = seasons[i]
        s2 = seasons[i + 1]
        
        df1 = filter(r -> r.season == s1, df)
        df2 = filter(r -> r.season == s2, df)
        
        t1 = compute_team_corner_metrics(df1)
        t2 = compute_team_corner_metrics(df2)
        
        merged = innerjoin(t1, t2, on = :team, makeunique = true)
        filter!(r -> r.matches >= 15 && r.matches_1 >= 15, merged)
        
        for r in eachrow(merged)
            push!(pairs_for, (r.corner_rate_for, r.corner_rate_for_1))
            push!(pairs_against, (r.corner_rate_against, r.corner_rate_against_1))
            push!(pairs_conv, (r.corner_conv_for, r.corner_conv_for_1))
            push!(pairs_goals, (r.corner_goals_for / r.matches, r.corner_goals_for_1 / r.matches_1))
        end
    end
    
    r_for = isempty(pairs_for) ? 0.0 : cor(first.(pairs_for), last.(pairs_for))
    r_against = isempty(pairs_against) ? 0.0 : cor(first.(pairs_against), last.(pairs_against))
    r_conv = isempty(pairs_conv) ? 0.0 : cor(first.(pairs_conv), last.(pairs_conv))
    r_goals = isempty(pairs_goals) ? 0.0 : cor(first.(pairs_goals), last.(pairs_goals))
    
    return (
        n_pairs = length(pairs_for),
        r_corners_for = r_for,
        r_corners_against = r_against,
        r_corner_conv = r_conv,
        r_corner_goals_per_game = r_goals
    )
end
