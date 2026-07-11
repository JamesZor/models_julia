#=
r00 — Stage-0 data QA for the Scottish Lower team-level sup+smile stream.

Answers, from the actual DataStore (no modelling):
  1. Exact season strings + per-season × per-tournament match counts (target_seasons format).
  2. Bet365 (SofaScore) O/U ladder coverage per season × strike — is Kmax=4 dense?
  3. League goal-level diff + V/M per tournament (prior scale for δ_league).
  4. Team churn: 56↔57 promotion/relegation overlap + teams with no segment history.
  5. Fold counts for dynamics_col ∈ {match_week, match_biweek, match_month} on the
     23/24→25/26 grid window (runtime budget per grid cell).
  6. Market-inversion sanity: DoublePoissonMarketFeature λ plausibility + MarketSmileFeature
     Λ^mkt(K) finiteness/coverage on real Scottish odds.
  7. Confirms `Features.MarketLambdaFeature` is a phantom (src *Market* engines'
     required_features throws) — documents why l01 overrides it.

Run on the server (kaimon REPL) after git pull:
    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_lower_smile/r00_data_qa.jl"))
=#

using BayesianFootball
using DataFrames
using Statistics
using Distributions

const Data     = BayesianFootball.Data
const Features = BayesianFootball.Features
const PreGame  = BayesianFootball.Models.PreGame

println("[INFO] Loading ScottishLower DataStore (cache if fresh)...")
ds = Data.load_datastore_cached(Data.ScottishLower())

m = ds.matches
println("[INFO] matches=", nrow(m), "  odds rows=", nrow(ds.odds),
        "  stats rows=", nrow(ds.statistics), "  lineups rows=", nrow(ds.lineups),
        "  betfair rows=", nrow(ds.betfair_odds))

# ==========================================
# 1. SEASONS — exact strings + counts (this defines the target_seasons format)
# ==========================================
println("\n", "="^70, "\n1. SEASONS (exact strings for target_seasons)\n", "="^70)
seas = sort(combine(groupby(m, [:tournament_id, :season]), nrow => :n_matches),
            [:tournament_id, :season])
show(seas, allrows=true); println()
season_strings = sort(unique(String.(m.season)))
println("season strings (sorted): ", season_strings)

# ==========================================
# 2. O/U LADDER COVERAGE per season × strike
# ==========================================
println("\n", "="^70, "\n2. O/U LADDER COVERAGE (matches with non-missing prob_fair_close)\n", "="^70)
season_of = Dict(Int(r.match_id) => String(r.season) for r in eachrow(m))
n_by_season = Dict(s => length(unique(m.match_id[String.(m.season) .== s])) for s in season_strings)

odds = ds.odds
strikes = ["under_$(K)5" for K in 0:7]
cov = Dict{Tuple{String,String}, Int}()
onex = Dict{String, Int}()   # 1X2 coverage (selection :home)
btts = Dict{String, Int}()
for g in groupby(odds, :match_id)
    mid = Int(first(g.match_id))
    haskey(season_of, mid) || continue
    s = season_of[mid]
    sels = Set(String.(skipmissing(g.selection[.!ismissing.(g.prob_fair_close)])))
    for st in strikes
        st in sels && (cov[(s, st)] = get(cov, (s, st), 0) + 1)
    end
    "home" in sels && (onex[s] = get(onex, s, 0) + 1)
    ("btts_yes" in sels || "yes" in sels) && (btts[s] = get(btts, s, 0) + 1)
end
println(rpad("season", 8), rpad("n_match", 8), rpad("1x2", 6), rpad("btts", 6),
        join([rpad("u$(K)5", 6) for K in 0:7]))
for s in season_strings
    print(rpad(s, 8), rpad(get(n_by_season, s, 0), 8),
          rpad(get(onex, s, 0), 6), rpad(get(btts, s, 0), 6))
    for K in 0:7
        print(rpad(get(cov, (s, "under_$(K)5"), 0), 6))
    end
    println()
end
println("(distinct selections sample: ", join(first(sort(unique(String.(odds.selection))), 30), ", "), ")")

# ==========================================
# 3. LEAGUE LEVEL DIFF + dispersion regime
# ==========================================
println("\n", "="^70, "\n3. GOALS LEVEL + V/M per tournament (δ_league prior scale)\n", "="^70)
mf = dropmissing(m, [:home_score, :away_score])
for tid in sort(unique(mf.tournament_id))
    sub = mf[mf.tournament_id .== tid, :]
    tot = sub.home_score .+ sub.away_score
    println("  tournament $tid: n=", nrow(sub),
            "  mean_goals=", round(mean(tot), digits=3),
            "  V/M=", round(var(tot) / mean(tot), digits=3),
            "  home=", round(mean(sub.home_score), digits=3),
            "  away=", round(mean(sub.away_score), digits=3))
end
tids = sort(unique(mf.tournament_id))
if length(tids) == 2
    m1 = mean(mf.home_score[mf.tournament_id .== tids[1]] .+ mf.away_score[mf.tournament_id .== tids[1]])
    m2 = mean(mf.home_score[mf.tournament_id .== tids[2]] .+ mf.away_score[mf.tournament_id .== tids[2]])
    println("  implied δ_league gap |log(m1/m2)| = ", round(abs(log(m1 / m2)), digits=4),
            "  (per-side offset ≈ half that; prior N(0, 0.1) comfortably covers it)")
end

# ==========================================
# 4. TEAM CHURN — cross-division movement + brand-new teams
# ==========================================
println("\n", "="^70, "\n4. TEAM CHURN (56↔57 movement / new-to-segment teams)\n", "="^70)
teams_by = Dict{Tuple{String,Int}, Set{String}}()
for r in eachrow(m)
    key = (String(r.season), Int(r.tournament_id))
    push!(get!(teams_by, key, Set{String}()), String(r.home_team))
end
seen = Set{String}()
for (i, s) in enumerate(season_strings)
    t56 = get(teams_by, (s, 56), Set{String}())
    t57 = get(teams_by, (s, 57), Set{String}())
    all_s = union(t56, t57)
    moved = String[]
    if i > 1
        p56 = get(teams_by, (season_strings[i-1], 56), Set{String}())
        p57 = get(teams_by, (season_strings[i-1], 57), Set{String}())
        moved = sort(collect(union(intersect(t56, p57), intersect(t57, p56))))
    end
    new_teams = sort(collect(setdiff(all_s, seen)))
    union!(seen, all_s)
    println("  $s: |56|=", length(t56), " |57|=", length(t57),
            "  moved-division=", isempty(moved) ? "-" : join(moved, ", "),
            (i > 1 ? "  NEW-to-segment=" * (isempty(new_teams) ? "-" : join(new_teams, ", ")) : ""))
end

# ==========================================
# 5. FOLD COUNTS per dynamics_col (grid runtime budget)
# ==========================================
println("\n", "="^70, "\n5. FOLD COUNTS — GroupedCVConfig [56,57], targets = last 3 seasons, hs=2\n", "="^70)
grid_targets = season_strings[max(1, end-2):end]
println("  target_seasons = ", grid_targets)
for dyn_col in (:match_week, :match_biweek, :match_month)
    cfg = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons    = grid_targets,
        history_seasons   = 2,
        dynamics_col      = dyn_col,
        warmup_period     = 5,
    )
    splits = Data.create_id_boundaries(ds, cfg)
    tgt_sizes = [length(b.target_match_ids) for (b, _) in splits]
    hist_sizes = [length(b.history_match_ids) for (b, _) in splits]
    println("  $(rpad(dyn_col, 14)) folds=", length(splits),
            "  target/fold: med=", isempty(tgt_sizes) ? "-" : median(tgt_sizes),
            " min=", isempty(tgt_sizes) ? "-" : minimum(tgt_sizes),
            " max=", isempty(tgt_sizes) ? "-" : maximum(tgt_sizes),
            "  history/fold: med=", isempty(hist_sizes) ? "-" : median(hist_sizes))
end

# ==========================================
# 6. MARKET INVERSION SANITY (pillar inputs)
# ==========================================
println("\n", "="^70, "\n6. MARKET INVERSION SANITY\n", "="^70)
sample_ids = unique(odds.match_id)
println("  matches with any odds: ", length(sample_ids))

# 6a. DoublePoissonMarketFeature → flat_market_λ_home/away plausibility
F1 = Dict{Symbol, Any}()
Features.add_feature!(F1, Features.DoublePoissonMarketFeature(), sample_ids, Dict{Any,Any}(), ds)
_ok(x) = !ismissing(x) && !isnan(Float64(x)) && 0.02 < Float64(x) < 20.0
lam_h = F1[:flat_market_λ_home]; lam_a = F1[:flat_market_λ_away]
n_ok = count(_ok.(lam_h) .& _ok.(lam_a))
println("  6a. DP market inversion: ", n_ok, "/", length(sample_ids),
        " matches with plausible (0.02, 20) λ pair (", round(100n_ok / length(sample_ids), digits=1), "%)")
good = [Float64(x) for x in lam_h if _ok(x)]
println("      λ_home quantiles (ok subset): ",
        round.(quantile(good, [0.05, 0.5, 0.95]), digits=3))

# 6b. MarketSmileFeature → Λ^mkt(K) ladder, Kmax=4
F2 = Dict{Symbol, Any}()
Features.add_feature!(F2, Features.MarketSmileFeature(Kmax=4), sample_ids, Dict{Any,Any}(), ds)
logΛ = F2[:flat_smile_logΛ]; msk = F2[:flat_smile_mask]
full_rows = findall(i -> all(msk[i, :] .> 0.5), 1:size(msk, 1))
println("  6b. smile inversion: full-ladder (all 5 strikes) matches = ", length(full_rows),
        "/", length(sample_ids))
println("      per-strike coverage: ", [Int(sum(msk[:, k])) for k in 1:size(msk, 2)])
if !isempty(full_rows)
    Λ_med = exp.(vec(median(logΛ[full_rows, :], dims=1)))
    println("      median Λ^mkt(K) over strikes 0.5→4.5: ", round.(Λ_med, digits=3),
            "  rising=", all(diff(Λ_med) .> 0))
    println("      all finite=", all(isfinite, logΛ[full_rows, :]))
end

# ==========================================
# 7. PHANTOM MarketLambdaFeature confirmation
# ==========================================
println("\n", "="^70, "\n7. PHANTOM FEATURE CHECK (why l01 overrides required_features)\n", "="^70)
nb_market = PreGame.DynamicMarketGoalsTimeDecayModel(
    interception_config  = PreGame.HierarchicalMonthlyInterception(),
    dynamics_config      = PreGame.TimeDecayDynamics(days_half_life=180),
    dispersion_config    = PreGame.HomeAwayDispersion(),
    homeadvantage_config = PreGame.HierarchicalTeamHomeAdvantage(),
)
try
    Features.required_features(nb_market)
    println("  ⚠ required_features returned WITHOUT error — phantom fixed already? Check src.")
catch e
    println("  ✅ confirmed: required_features(::DynamicMarketGoalsTimeDecayModel) throws ",
            typeof(e), " (MarketLambdaFeature undefined) — l01 override is required.")
end

println("\n[INFO] r00 data QA complete. Record the season strings + fold counts in NOTES.md.")
