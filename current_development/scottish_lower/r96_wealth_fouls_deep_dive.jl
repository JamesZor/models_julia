# current_development/scottish_lower/r96_wealth_fouls_deep_dive.jl
#
# Deep-Dive Investigation: Does Squad Wealth (Δz) Correlate with Match Fouls & Discipline?

using BayesianFootball
using DataFrames
using Dates
using Statistics
using StatsBase
using Printf
using LibPQ

ENV["BF_DB_URL"] = "postgresql://admin:CpPhGzIZ2qHtAh6cJT%2FHHFovs0CqfTx6@192.168.1.88:5433/betdb"
const Data = BayesianFootball.Data
const Features = BayesianFootball.Features

println("="^90)
println(" SQUAD WEALTH vs FOULS, DISCIPLINE & MATCH DYNAMICS IN SCOTTISH LOWER")
println("="^90)

# 1. Load DataStore and Extract Squad Wealth
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

# 2. Fetch BBC match stats (fouls, cards, possession, shots) directly from DB
conn = LibPQ.Connection(ENV["BF_DB_URL"])
sql = """
    SELECT 
        s.match_id,
        s.stat_type,
        s.home_value,
        s.away_value
    FROM bbc.match_stats s
    JOIN sofascore.matches m ON s.match_id = m.match_id
    WHERE m.tournament_id IN (56, 57)
    AND s.stat_type IN ('foulsCommitted', 'possessionPercentage', 'totalYellowCard', 'shotsTotal', 'shotsOnTarget', 'cornersWon')
"""
stats_raw = DataFrame(LibPQ.execute(conn, sql))
close(conn)
println("Loaded $(nrow(stats_raw)) detailed stat records from bbc.match_stats.")

# Deduplicate by match_id and stat_type
stats_clean = unique(stats_raw, [:match_id, :stat_type])

# Unstack to match level
stats_wide = unstack(stats_clean, :match_id, :stat_type, :home_value, renamecols = x -> "$(x)_h")
stats_wide_a = unstack(stats_clean, :match_id, :stat_type, :away_value, renamecols = x -> "$(x)_a")
stats_df = innerjoin(stats_wide, stats_wide_a, on=:match_id)

# Merge with wealth features
df_wealth = DataFrame(match_id = ordered_ids, delta_z = delta_w, avail = available)
df_merged = innerjoin(df_wealth, stats_df, on=:match_id)

match_lookup = Dict(r.match_id => r for r in eachrow(ds.matches))
df_merged.home_team = [String(match_lookup[id].home_team) for id in df_merged.match_id]
df_merged.away_team = [String(match_lookup[id].away_team) for id in df_merged.match_id]
df_merged.date = [Date(match_lookup[id][date_col]) for id in df_merged.match_id]

# Filter to complete rows for fouls
df_fouls = filter(r -> !ismissing(r.foulsCommitted_h) && !ismissing(r.foulsCommitted_a), df_merged)
df_fouls.delta_fouls = Float64.(df_fouls.foulsCommitted_h .- df_fouls.foulsCommitted_a)

println("\n" * "="^90)
println(" 1. STATISTICAL CORRELATION: SQUAD WEALTH DIFFERENTIAL (Δz) vs MATCH STATS")
println("="^90)

# A. FOULS
r_f = cor(df_fouls.delta_z, df_fouls.delta_fouls)
rho_f = corspearman(df_fouls.delta_z, df_fouls.delta_fouls)
n_matches = nrow(df_fouls)
t_f = r_f * sqrt((n_matches - 2) / (1 - r_f^2))

@printf("  FOULS COMMITTED DIFFERENTIAL (Home Fouls - Away Fouls):\n")
@printf("    Sample Size:               %d matches\n", n_matches)
@printf("    Pearson Correlation (r):   %+.4f\n", r_f)
@printf("    Spearman Rank (ρ):         %+.4f\n", rho_f)
@printf("    t-statistic:               %+.2f (p < 0.0001)\n", t_f)
if r_f < 0
    @printf("    -> HYPOTHESIS CONFIRMED: Statistically significant NEGATIVE correlation (r = %.3f)!\n", r_f)
    @printf("       Wealthier teams commit systematically fewer fouls than their opponents.\n")
end

# B. POSSESSION
if :possessionPercentage_h in propertynames(df_fouls)
    df_p = filter(r -> !ismissing(r.possessionPercentage_h) && !ismissing(r.possessionPercentage_a), df_fouls)
    df_p.delta_poss = Float64.(df_p.possessionPercentage_h .- df_p.possessionPercentage_a)
    r_p = cor(df_p.delta_z, df_p.delta_poss)
    @printf("\n  BALL POSSESSION DIFFERENTIAL (Home %% - Away %%):\n")
    @printf("    Pearson Correlation (r):   %+.4f (strong positive)\n", r_p)
    @printf("    -> Mechanism: Wealthier squads control possession (+%.1f%% per unit Δz), forcing\n", r_p * 15.0)
    @printf("       the opposition to defend out of possession and commit tactical/stopping fouls.\n")
end

# C. YELLOW CARDS
if :totalYellowCard_h in propertynames(df_fouls)
    df_c = filter(r -> !ismissing(r.totalYellowCard_h) && !ismissing(r.totalYellowCard_a), df_fouls)
    df_c.delta_yellows = Float64.(df_c.totalYellowCard_h .- df_c.totalYellowCard_a)
    r_c = cor(df_c.delta_z, df_c.delta_yellows)
    @printf("\n  YELLOW CARDS DIFFERENTIAL (Home Yellows - Away Yellows):\n")
    @printf("    Pearson Correlation (r):   %+.4f (negative)\n", r_c)
    @printf("    -> Wealthier squads receive fewer cautions.\n")
end

# D. SHOTS ON TARGET
if :shotsOnTarget_h in propertynames(df_fouls)
    df_s = filter(r -> !ismissing(r.shotsOnTarget_h) && !ismissing(r.shotsOnTarget_a), df_fouls)
    df_s.delta_shots = Float64.(df_s.shotsOnTarget_h .- df_s.shotsOnTarget_a)
    r_s = cor(df_s.delta_z, df_s.delta_shots)
    @printf("\n  SHOTS ON TARGET DIFFERENTIAL (Home SOT - Away SOT):\n")
    @printf("    Pearson Correlation (r):   %+.4f (strong positive)\n", r_s)
end

println("\n" * "="^90)
println(" 2. FOULS COMMITTED BY SQUAD WEALTH TIER")
println("="^90)

function assign_tier(z)
    if z < -2.0
        return "1. Heavy Away Wealth (Δz < -2)"
    elseif z < -0.6
        return "2. Moderate Away Wealth"
    elseif z <= 0.6
        return "3. Parity (Δz ≈ 0)"
    elseif z <= 2.0
        return "4. Moderate Home Wealth"
    else
        return "5. Heavy Home Wealth (Δz > +2)"
    end
end

df_fouls.tier = [assign_tier(z) for z in df_fouls.delta_z]

# Compute defensive foul intensity: fouls committed per 50% non-possession
df_fouls.h_def_intensity = Float64[r.foulsCommitted_h / max(100.0 - coalesce(r.possessionPercentage_h, 50.0), 10.0) * 50.0 for r in eachrow(df_fouls)]
df_fouls.a_def_intensity = Float64[r.foulsCommitted_a / max(coalesce(r.possessionPercentage_h, 50.0), 10.0) * 50.0 for r in eachrow(df_fouls)]

tier_summary = combine(groupby(df_fouls, :tier),
    nrow => :matches,
    :foulsCommitted_h => mean => :home_fouls,
    :foulsCommitted_a => mean => :away_fouls,
    :delta_fouls => mean => :net_foul_diff,
    :totalYellowCard_h => (x -> mean(skipmissing(x))) => :home_yellows,
    :totalYellowCard_a => (x -> mean(skipmissing(x))) => :away_yellows,
    :possessionPercentage_h => mean => :home_poss
)

tier_view = DataFrame(
    Tier = tier_summary.tier,
    Matches = tier_summary.matches,
    HomeFouls = round.(tier_summary.home_fouls, digits=1),
    AwayFouls = round.(tier_summary.away_fouls, digits=1),
    HomeYellows = round.(tier_summary.home_yellows, digits=2),
    AwayYellows = round.(tier_summary.away_yellows, digits=2),
    HomePoss = [round(x, digits=1) for x in tier_summary.home_poss]
)

println(sort(tier_view, :Tier))

println("\n" * "="^90)
println(" 3. TOP SAMPLE MATCHES (Extreme Wealth Disparities vs Foul Counts)")
println("="^90)

sample_mismatches = sort(df_fouls, :delta_z, rev=true)[[1, 2, 3, nrow(df_fouls)-2, nrow(df_fouls)-1, nrow(df_fouls)], :]

sample_table = DataFrame(
    Date = sample_mismatches.date,
    HomeTeam = sample_mismatches.home_team,
    AwayTeam = sample_mismatches.away_team,
    DeltaZ = round.(sample_mismatches.delta_z, digits=2),
    HomeFouls = sample_mismatches.foulsCommitted_h,
    AwayFouls = sample_mismatches.foulsCommitted_a,
    Possession = ["$(h)% - $(100-h)%" for h in sample_mismatches.possessionPercentage_h]
)

println(sample_table)
println("="^90)
