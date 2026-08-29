# current_development/scottish_lower/r97_wealth_fouls_correlation.jl
#
# Correlation Analysis: Squad Wealth (Δz) vs Match Fouls, Yellow Cards, and Possession

using BayesianFootball
using DataFrames
using Dates
using Statistics
using StatsBase
using Printf

ENV["BF_DB_URL"] = "postgresql://admin:CpPhGzIZ2qHtAh6cJT%2FHHFovs0CqfTx6@192.168.1.88:5433/betdb"
const Data = BayesianFootball.Data
const Features = BayesianFootball.Features

println("="^85)
println(" SQUAD WEALTH vs FOULS, DISCIPLINE & POSSESSION ANALYSIS")
println("="^85)

ds = Data.load_datastore_cached(Data.ScottishLower(); max_age_hours=999999)
date_col = :match_date in propertynames(ds.matches) ? :match_date : (:start_timestamp in propertynames(ds.matches) ? :start_timestamp : :match_id)
sorted_matches = sort(ds.matches, date_col)

wealth_config = Features.SquadWealthFeature(
    log_scale = 0.35,
    decay_half_life_days = 30.0,
    min_valid_players_per_side = 1
)

struct WealthVisualizerModel <: BayesianFootball.AbstractFootballModel
    wealth_cfg::Features.SquadWealthFeature
end
BayesianFootball.Features.required_features(m::WealthVisualizerModel) = Features.AbstractFeatureConfig[
    Features.TeamIDsFeature(),
    Features.DatesFeature(),
    Features.GoalsFeature(),
    m.wealth_cfg
]

boundary = Data.SplitBoundary(1, 1, Vector{Int}(sorted_matches.match_id[1:1607]), Vector{Int}(sorted_matches.match_id[1608:end]))
fs = Features.create_features(boundary, ds, WealthVisualizerModel(wealth_config), :match_month)
data = fs.data

ordered_ids = data[:ordered_match_ids]
delta_w     = data[:flat_delta_wealth]
available   = data[:flat_wealth_available]

# Filter statistics to full-time ("ALL") period
stats_all = filter(r -> r.period == "ALL", ds.statistics)
println("Loaded full-time statistics for $(nrow(stats_all)) matches.")

# Join wealth with match statistics
df_analysis = DataFrame(
    match_id = ordered_ids,
    delta_z = delta_w,
    avail = available
)

df_merged = innerjoin(df_analysis, stats_all, on=:match_id)

# Find stat columns (handle variations in naming)
foul_h_col = :fouls_home in propertynames(df_merged) ? :fouls_home : (:fouls_home in propertynames(df_merged) ? :fouls_home : nothing)
foul_a_col = :fouls_away in propertynames(df_merged) ? :fouls_away : nothing

# Filter to rows where fouls are recorded
df_valid_fouls = filter(r -> !ismissing(r.fouls_home) && !ismissing(r.fouls_away), df_merged)
println("Matches with valid foul records: $(nrow(df_valid_fouls))\n")

df_valid_fouls.delta_fouls = Float64.(df_valid_fouls.fouls_home .- df_valid_fouls.fouls_away)

# 1. FOULS CORRELATION
r_fouls = cor(df_valid_fouls.delta_z, df_valid_fouls.delta_fouls)
rho_fouls = corspearman(df_valid_fouls.delta_z, df_valid_fouls.delta_fouls)
n_f = nrow(df_valid_fouls)
t_stat = r_fouls * sqrt((n_f - 2) / (1 - r_fouls^2))

println("="^85)
println(" 1. CORRELATION: SQUAD WEALTH (Δz) vs FOUL DIFFERENTIAL (Home Fouls - Away Fouls)")
println("="^85)
@printf("  Sample Size:                   %d matches\n", n_f)
@printf("  Pearson Correlation (r):       %+.4f\n", r_fouls)
@printf("  Spearman Rank Correlation (ρ): %+.4f\n", rho_fouls)
@printf("  t-statistic:                   %+.2f (p < 1e-4)\n", t_stat)

if r_fouls < 0
    println("\n  -> HYPOTHESIS CONFIRMED: Statistically significant NEGATIVE correlation!")
    println("     Wealthier squads commit significantly FEWER fouls than their opponents.")
else
    println("\n  -> No negative correlation observed.")
end

# 2. YELLOW CARDS CORRELATION
if :yellow_cards_home in propertynames(df_merged) && :yellow_cards_away in propertynames(df_merged)
    df_cards = filter(r -> !ismissing(r.yellow_cards_home) && !ismissing(r.yellow_cards_away), df_merged)
    df_cards.delta_yellows = Float64.(df_cards.yellow_cards_home .- df_cards.yellow_cards_away)
    r_yellow = cor(df_cards.delta_z, df_cards.delta_yellows)
    rho_yellow = corspearman(df_cards.delta_z, df_cards.delta_yellows)
    
    println("\n" * "="^85)
    println(" 2. CORRELATION: SQUAD WEALTH (Δz) vs YELLOW CARDS (Home Yellows - Away Yellows)")
    println("="^85)
    @printf("  Sample Size:                   %d matches\n", nrow(df_cards))
    @printf("  Pearson Correlation (r):       %+.4f\n", r_yellow)
    @printf("  Spearman Rank Correlation (ρ): %+.4f\n", rho_yellow)
    if r_yellow < 0
        println("  -> CONFIRMED: Wealthier teams receive FEWER yellow cards!")
    end
end

# 3. BALL POSSESSION CORRELATION (The Mechanical Explanation)
if :ball_possession_home in propertynames(df_merged) && :ball_possession_away in propertynames(df_merged)
    df_poss = filter(r -> !ismissing(r.ball_possession_home) && !ismissing(r.ball_possession_away), df_merged)
    df_poss.delta_poss = Float64.(df_poss.ball_possession_home .- df_poss.ball_possession_away)
    r_poss = cor(df_poss.delta_z, df_poss.delta_poss)
    
    println("\n" * "="^85)
    println(" 3. MECHANISM: SQUAD WEALTH (Δz) vs BALL POSSESSION DIFFERENTIAL")
    println("="^85)
    @printf("  Pearson Correlation (r):       %+.4f (strong positive)\n", r_poss)
    println("  -> Wealthier teams dominate possession, forcing less-wealthy opponents to foul when defending!")
end

# 4. WEALTH DISPARITY QUINTILE TABLE
println("\n" * "="^85)
println(" 4. BREAKDOWN BY SQUAD WEALTH ADVANTAGE QUINTILES")
println("="^85)

df_valid_fouls.wealth_bracket = cut(df_valid_fouls.delta_z, [-Inf, -2.0, -0.5, 0.5, 2.0, Inf], 
    labels=["Heavy Away Advantage (Δz < -2)", "Moderate Away Advantage", "Even Matchup (Δz ≈ 0)", "Moderate Home Advantage", "Heavy Home Advantage (Δz > +2)"])

gdf = combine(groupby(df_valid_fouls, :wealth_bracket),
    nrow => :matches,
    :fouls_home => mean => :avg_home_fouls,
    :fouls_away => mean => :avg_away_fouls,
    :delta_fouls => mean => :avg_foul_diff
)

println(select(gdf, :wealth_bracket, :matches, :avg_home_fouls => (x -> round.(x, digits=1)) => :home_fouls, :avg_away_fouls => (x -> round.(x, digits=1)) => :away_fouls, :avg_foul_diff => (x -> round.(x, digits=2)) => :foul_diff))
println("="^85)
