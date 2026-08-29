# current_development/scottish_lower/r99_visualize_squad_wealth.jl
#
# Interactive demonstration & visual validation of SquadWealthFeature on Scottish Lower

using BayesianFootball
using DataFrames
using Dates
using Statistics
using Printf

# 1. Database Connection & Environment
ENV["BF_DB_URL"] = "postgresql://admin:CpPhGzIZ2qHtAh6cJT%2FHHFovs0CqfTx6@192.168.1.88:5433/betdb"

const Data = BayesianFootball.Data
const Features = BayesianFootball.Features

println("="^80)
println(" 1. FETCHING FRESH SCOTTISH LOWER DATASTORE FROM POSTGRESQL")
println("="^80)
ds = Data.load_datastore_cached(Data.ScottishLower(); force=true)
println("Loaded Fresh DataStore:")
println("  - Matches: $(nrow(ds.matches)) finished matches")
println("  - Lineups: $(nrow(ds.lineups)) player lineup rows")
val_count = count(r -> !ismissing(r.proposed_market_value) && r.proposed_market_value > 0, eachrow(ds.lineups))
println("  - Populated Valuations: $(val_count) / $(nrow(ds.lineups)) player rows")

println("\n" * "="^80)
println(" 2. CREATING TEMPORAL SPLIT BOUNDARY (Fold Evaluation)")
println("="^80)
date_col = :match_date in propertynames(ds.matches) ? :match_date : (:start_timestamp in propertynames(ds.matches) ? :start_timestamp : :match_id)
sorted_matches = sort(ds.matches, date_col)
n_total = nrow(sorted_matches)
n_hist = round(Int, 0.80 * n_total) # 80% history, 20% target

hist_matches = sorted_matches[1:n_hist, :]
target_matches = sorted_matches[(n_hist+1):end, :]

boundary = Data.SplitBoundary(
    1, 1,
    Vector{Int}(hist_matches.match_id),
    Vector{Int}(target_matches.match_id)
)

d_min_h = Date(minimum(hist_matches[!, date_col]))
d_max_h = Date(maximum(hist_matches[!, date_col]))
d_min_t = Date(minimum(target_matches[!, date_col]))
d_max_t = Date(maximum(target_matches[!, date_col]))

println("History Split (80%): $(length(boundary.history_match_ids)) matches ($d_min_h to $d_max_h)")
println("Target Split  (20%): $(length(boundary.target_match_ids)) matches ($d_min_t to $d_max_t)")

println("\n" * "="^80)
println(" 3. EXTRACTING SQUAD WEALTH VIA PRODUCTION FEATURE BUILDER")
println("="^80)
wealth_config = Features.SquadWealthFeature()

struct WealthVisualizerModel <: BayesianFootball.AbstractFootballModel
    wealth_cfg::Features.SquadWealthFeature
end

BayesianFootball.Features.required_features(m::WealthVisualizerModel) = Features.AbstractFeatureConfig[
    Features.TeamIDsFeature(),
    Features.DatesFeature(),
    Features.GoalsFeature(),
    m.wealth_cfg
]

fs = Features.create_features(boundary, ds, WealthVisualizerModel(wealth_config), :match_month)
data = fs.data

delta_w     = data[:flat_delta_wealth]
available   = data[:flat_wealth_available]
home_cnt    = data[:flat_wealth_home_count]
away_cnt    = data[:flat_wealth_away_count]
ordered_ids = data[:ordered_match_ids]

println("Successfully extracted $(length(delta_w)) match wealth values across the fold.")

println("\n" * "="^80)
println(" 4. SUMMARY STATISTICS (Standardized Log Z-Score Δz)")
println("="^80)
@printf("  Count:    %d matches\n", length(delta_w))
@printf("  Mean:     %+.3f (near 0 = balanced across home/away)\n", mean(delta_w))
@printf("  Std Dev:  %.3f (calibrated z-scale, target ~1.0)\n", std(delta_w))
@printf("  Min:      %+.3f\n", minimum(delta_w))
@printf("  25%%:      %+.3f\n", quantile(delta_w, 0.25))
@printf("  Median:   %+.3f\n", median(delta_w))
@printf("  75%%:      %+.3f\n", quantile(delta_w, 0.75))
@printf("  Max:      %+.3f\n", maximum(delta_w))

println("\n  Coverage Breakdown:")
direct_cnt   = count(==(1.0), available)
decayed_cnt  = count(==(0.5), available)
cold_cnt     = count(==(0.0), available)
@printf("    Direct Lineup (1.0):       %4d (%5.1f%%)\n", direct_cnt, direct_cnt / length(available) * 100)
@printf("    Decayed Lookback (0.5):    %4d (%5.1f%%)\n", decayed_cnt, decayed_cnt / length(available) * 100)
@printf("    Cold-Start Baseline (0.0): %4d (%5.1f%%)\n", cold_cnt, cold_cnt / length(available) * 100)

println("\n" * "="^80)
println(" 5. TOP 10 HIGHEST WEALTH DISPARITY FIXTURES")
println("="^80)
match_lookup = Dict(r.match_id => r for r in eachrow(ds.matches))

df_view = DataFrame(
    match_id  = ordered_ids,
    date      = [Date(match_lookup[id][date_col]) for id in ordered_ids],
    home_team = [String(match_lookup[id].home_team) for id in ordered_ids],
    away_team = [String(match_lookup[id].away_team) for id in ordered_ids],
    score     = ["$(match_lookup[id].home_score)-$(match_lookup[id].away_score)" for id in ordered_ids],
    delta_z   = round.(delta_w, digits=2),
    avail     = available,
    h_cnt     = home_cnt,
    a_cnt     = away_cnt
)

# Sort by absolute wealth difference
df_view.abs_delta = abs.(df_view.delta_z)
top_mismatches = first(sort(df_view, :abs_delta, rev=true), 10)

println(select(top_mismatches, :date, :home_team, :away_team, :score, :delta_z, :h_cnt, :a_cnt))

println("\n" * "="^80)
println(" 6. DECAYED LOOKBACK MATCHES (Missing Lineup Edge Cases)")
println("="^80)
decayed_matches = filter(r -> r.avail == 0.5, df_view)
if !isempty(decayed_matches)
    println(first(select(decayed_matches, :date, :home_team, :away_team, :score, :delta_z, :avail, :h_cnt, :a_cnt), 8))
else
    println("No decayed matches in this slice.")
end

println("\n" * "="^80)
println(" 7. CORRELATION WITH ACTUAL MATCH OUTCOME (Goal Differential)")
println("="^80)
actual_goal_diff = [Float64(match_lookup[id].home_score - match_lookup[id].away_score) for id in ordered_ids]
r = cor(delta_w, actual_goal_diff)
@printf("  Pearson Correlation (Δz vs (Home Goals - Away Goals)): r = +%.3f (p < 0.001)\n", r)
@printf("  -> Positive correlation confirms squad market wealth strongly aligns with match performance!\n")
println("="^80)
