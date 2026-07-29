#=
r00 — Stage-0 data QA for the Scottish Upper (54 Premiership / 55 Championship) bake-off.

Adapted from current_development/scottish_lower_smile/r00_data_qa.jl. Answers, from the actual
DataStore (no modelling):

  1. Exact season strings + per-season × per-tournament match counts. ⚠ CRITICAL: `sofascore.seasons`
     names these tiers "Premiership 23/24" / "Championship 23/24". If the matches fetcher does NOT
     strip the competition prefix, `target_seasons = ["24/25", ...]` matches NOTHING and every grid
     cell silently trains on zero folds. This section decides the exact strings the runners use.
  2. Bet365 (SofaScore) O/U ladder coverage per season × strike — is Kmax=4 dense?
  3. Goals level + V/M per tournament → δ_league prior scale and Poisson-vs-NegBin base.
  4. BBC shots/SoT coverage per season × tournament (the funnel arm's ONLY viable shot source —
     SofaScore shots are missing for 55 before 23/24).
  5. SofaScore player-rating coverage per season × tournament. THE DECIDING CHECK: with
     history_seasons=2 the 24/25 target folds look back into 22/23, where tournament 55 has no
     ratings. If coverage there is bad enough to distort the ratings arm, the fallback is
     history_seasons=1 applied UNIFORMLY to every cell (see NOTES.md).
  6. Team churn: 54↔55 promotion/relegation overlap + teams new to the segment.
  7. Fold counts for dynamics_col ∈ {match_week, match_biweek, match_month} on the grid window
     (targets 24/25→25/26, hs=2) — the per-cell runtime budget.
  8. Market-inversion sanity: DoublePoissonMarketFeature λ plausibility + MarketSmileFeature
     Λ^mkt(K) finiteness/coverage on real 54/55 odds.
  9. Confirms ds.betfair_odds is EMPTY (54/55 were never collected) so no runner depends on it.

Run on the server (kaimon REPL) after git pull:
    include(joinpath(pkgdir(BayesianFootball), "current_development/scottish_upper/r00_data_qa.jl"))

First call fetches fresh from SQL (no cache for this segment yet) — needs ENV["BF_DB_URL"].
=#

using BayesianFootball
using DataFrames
using Statistics
using Distributions

const Data     = BayesianFootball.Data
const Features = BayesianFootball.Features
const PreGame  = BayesianFootball.Models.PreGame

println("[INFO] Loading ScottishUpper DataStore (fresh SQL fetch on first call)...")
ds = Data.load_datastore_cached(Data.ScottishUpper())

m = ds.matches
println("[INFO] matches=", nrow(m), "  odds=", nrow(ds.odds), "  stats=", nrow(ds.statistics),
        "  lineups=", nrow(ds.lineups), "  incidents=", nrow(ds.incidents),
        "  betfair=", nrow(ds.betfair_odds), "  bbc=", nrow(ds.bbc),
        "  bbc_events=", nrow(ds.bbc_events))

# ==========================================
# 1. SEASONS — exact strings (this is the gate for target_seasons)
# ==========================================
println("\n", "="^70, "\n1. SEASONS — exact strings for target_seasons\n", "="^70)
seas = sort(combine(groupby(m, [:tournament_id, :season]), nrow => :n_matches),
            [:tournament_id, :season])
show(seas, allrows=true); println()

season_strings = sort(unique(String.(m.season)))
println("\nseason strings (sorted): ", season_strings)

# The pooled design REQUIRES both tournaments to share a season string. If 54 says
# "Premiership 24/25" and 55 says "Championship 24/25", a single `target_seasons` vector cannot
# select both and the pooled fold builder will silently produce half (or zero) the data.
s54 = sort(unique(String.(m.season[m.tournament_id .== 54])))
s55 = sort(unique(String.(m.season[m.tournament_id .== 55])))
shared = intersect(Set(s54), Set(s55))
println("  54 seasons: ", s54)
println("  55 seasons: ", s55)
if length(shared) == length(s54) == length(s55)
    println("  ✅ season strings are SHARED across both tournaments — pooled target_seasons works.")
else
    println("  ⛔ BLOCKER: season strings DIFFER between 54 and 55 (shared = ", sort(collect(shared)), ").")
    println("     The matches fetcher is not stripping the competition prefix. Fix that before ANY grid,")
    println("     or every pooled fold silently drops one tournament.")
end

# ==========================================
# 2. O/U LADDER COVERAGE per season × strike
# ==========================================
println("\n", "="^70, "\n2. O/U LADDER COVERAGE (matches with non-missing prob_fair_close)\n", "="^70)
season_of = Dict(Int(r.match_id) => String(r.season) for r in eachrow(m))
n_by_season = Dict(s => length(unique(m.match_id[String.(m.season) .== s])) for s in season_strings)

odds = ds.odds
strikes = ["under_$(K)5" for K in 0:7]
cov  = Dict{Tuple{String,String}, Int}()
onex = Dict{String, Int}()
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
println(rpad("season", 22), rpad("n_match", 9), rpad("1x2", 6), rpad("btts", 6),
        join([rpad("u$(K)5", 6) for K in 0:7]))
for s in season_strings
    print(rpad(s, 22), rpad(get(n_by_season, s, 0), 9),
          rpad(get(onex, s, 0), 6), rpad(get(btts, s, 0), 6))
    for K in 0:7
        print(rpad(get(cov, (s, "under_$(K)5"), 0), 6))
    end
    println()
end
println("READ: Kmax=4 needs u05..u45 dense in every GRID season (24/25, 25/26).")

# ==========================================
# 3. GOALS LEVEL + dispersion regime  → δ_league prior scale
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
    g1 = mf.home_score[mf.tournament_id .== tids[1]] .+ mf.away_score[mf.tournament_id .== tids[1]]
    g2 = mf.home_score[mf.tournament_id .== tids[2]] .+ mf.away_score[mf.tournament_id .== tids[2]]
    gap = abs(log(mean(g1) / mean(g2)))
    println("  implied δ_league gap |log(m1/m2)| = ", round(gap, digits=4),
            "  (per-side offset ≈ half that)")
    # On 56/57 the gap was 0.047 and N(0, 0.1) covered it comfortably. δ_league is a LEVEL offset —
    # it cannot absorb a difference in strength SPREAD between the tiers.
    println("  vs league_offset_sd=0.1 prior: ", gap < 0.15 ? "✅ comfortably covered" :
            "⚠ WIDEN league_offset_sd (gap is a large fraction of the prior sd)")
    sd1, sd2 = std(g1), std(g2)
    println("  goal-total sd per tier: ", round(sd1, digits=3), " vs ", round(sd2, digits=3),
            "  ratio=", round(sd1 / sd2, digits=3),
            (abs(log(sd1 / sd2)) > 0.15 ? "  ⚠ tiers differ in SPREAD, not just level — a level offset cannot fix that" : ""))
end

# ==========================================
# 4. BBC SHOTS COVERAGE (the funnel arm's input)
# ==========================================
println("\n", "="^70, "\n4. BBC SHOTS / SoT COVERAGE per season × tournament\n", "="^70)
if nrow(ds.bbc) == 0
    println("  ⛔ ds.bbc is EMPTY — the funnel arm cannot run. Check the BBC fetcher.")
else
    bbc = leftjoin(select(ds.bbc, :match_id, :shots_h, :shots_a, :sot_h, :sot_a),
                   select(m, :match_id, :tournament_id, :season), on = :match_id)
    dropmissing!(bbc, [:tournament_id, :season])
    println(rpad("season", 22), rpad("tid", 6), rpad("n_bbc", 8), rpad("shots_ok", 10), rpad("sot_ok", 8))
    for g in sort(collect(groupby(bbc, [:season, :tournament_id])), by = x -> (String(first(x.season)), first(x.tournament_id)))
        n = nrow(g)
        ns = count(i -> !ismissing(g.shots_h[i]) && !ismissing(g.shots_a[i]), 1:n)
        nt = count(i -> !ismissing(g.sot_h[i])   && !ismissing(g.sot_a[i]),   1:n)
        println(rpad(String(first(g.season)), 22), rpad(first(g.tournament_id), 6),
                rpad(n, 8), rpad(ns, 10), rpad(nt, 8))
    end
    println("READ: expect ~100% on BOTH tiers for all seasons — this is why the funnel arm uses")
    println("      ShotsFunnelFeature (BBC) and NOT ShotsFeature (SofaScore, half-missing on 55).")
end

# ==========================================
# 5. PLAYER-RATING COVERAGE — decides the history_seasons fallback
# ==========================================
println("\n", "="^70, "\n5. SofaScore PLAYER-RATING COVERAGE per season × tournament\n", "="^70)
if nrow(ds.lineups) == 0 || !("rating" in names(ds.lineups))
    println("  ⛔ no `rating` column in ds.lineups — the ratings arm cannot run.")
else
    lu = leftjoin(select(ds.lineups, :match_id, :rating),
                  select(m, :match_id, :tournament_id, :season), on = :match_id)
    dropmissing!(lu, [:tournament_id, :season])
    rated = combine(groupby(lu, [:season, :tournament_id]),
                    nrow => :n_player_rows,
                    :rating => (r -> count(!ismissing, r)) => :n_rated)
    # matches with at least one rated player, per season × tournament
    lu.has = .!ismissing.(lu.rating)
    permatch = combine(groupby(lu, [:season, :tournament_id, :match_id]), :has => any => :any_rated)
    mcov = combine(groupby(permatch, [:season, :tournament_id]),
                   nrow => :n_matches, :any_rated => sum => :n_matches_rated)
    out = leftjoin(mcov, rated, on = [:season, :tournament_id])
    sort!(out, [:season, :tournament_id])
    show(out, allrows=true, allcols=true, truncate=0); println()

    # The specific cell the plan hinges on: the 22/23 history block that hs=2 pulls in for the
    # 24/25 target folds.
    hist_seasons = filter(s -> occursin("22/23", s), season_strings)
    for hs_ in hist_seasons, tid in (54, 55)
        r = out[(out.season .== hs_) .& (out.tournament_id .== tid), :]
        isempty(r) && continue
        pct = 100 * r.n_matches_rated[1] / max(r.n_matches[1], 1)
        println("  $hs_ / tid $tid: ", r.n_matches_rated[1], "/", r.n_matches[1],
                " matches rated (", round(pct, digits=1), "%)",
                pct < 50 ? "  ⚠ thin history for the ratings arm" : "")
    end
    println("DECISION RULE: if the 22/23 block is thin on 55, drop to history_seasons=1 for ALL")
    println("               cells (uniformly — never per-arm, that breaks comparability).")
end

# ==========================================
# 6. TEAM CHURN — cross-division movement + new teams
# ==========================================
println("\n", "="^70, "\n6. TEAM CHURN (54↔55 movement / new-to-segment teams)\n", "="^70)
teams_by = Dict{Tuple{String,Int}, Set{String}}()
for r in eachrow(m)
    push!(get!(teams_by, (String(r.season), Int(r.tournament_id)), Set{String}()), String(r.home_team))
end
seen = Set{String}()
for (i, s) in enumerate(season_strings)
    t54 = get(teams_by, (s, 54), Set{String}())
    t55 = get(teams_by, (s, 55), Set{String}())
    all_s = union(t54, t55)
    moved = String[]
    if i > 1
        p54 = get(teams_by, (season_strings[i-1], 54), Set{String}())
        p55 = get(teams_by, (season_strings[i-1], 55), Set{String}())
        moved = sort(collect(union(intersect(t54, p55), intersect(t55, p54))))
    end
    new_teams = sort(collect(setdiff(all_s, seen)))
    union!(seen, all_s)
    println("  $s: |54|=", length(t54), " |55|=", length(t55),
            "  moved-division=", isempty(moved) ? "-" : join(moved, ", "),
            (i > 1 ? "  NEW-to-segment=" * (isempty(new_teams) ? "-" : join(new_teams, ", ")) : ""))
end

# ==========================================
# 7. FOLD COUNTS per dynamics_col (grid runtime budget)
# ==========================================
println("\n", "="^70, "\n7. FOLD COUNTS — GroupedCVConfig [54,55], grid window\n", "="^70)
# The grid window is the LAST TWO seasons (targets 24/25 → 25/26), hs=2, warmup_period=0.
grid_targets = season_strings[max(1, end-1):end]
println("  target_seasons = ", grid_targets, "   history_seasons = 2   warmup_period = 0")
for dyn_col in (:match_week, :match_biweek, :match_month)
    cfg = Data.GroupedCVConfig(
        tournament_groups = [Data.tournament_ids(ds.segment)],
        target_seasons    = grid_targets,
        history_seasons   = 2,
        dynamics_col      = dyn_col,
        warmup_period     = 0,
    )
    splits = Data.create_id_boundaries(ds, cfg)
    tgt  = [length(b.target_match_ids)  for (b, _) in splits]
    hist = [length(b.history_match_ids) for (b, _) in splits]
    println("  $(rpad(dyn_col, 14)) folds=", length(splits),
            "  target/fold: med=", isempty(tgt) ? "-" : median(tgt),
            " min=", isempty(tgt) ? "-" : minimum(tgt),
            " max=", isempty(tgt) ? "-" : maximum(tgt),
            "  history/fold: med=", isempty(hist) ? "-" : median(hist))
end
println("READ: :match_biweek is the chosen granularity. Its fold count sets the grid budget —")
println("      cell_wall ≈ r01_smoke_wall × (folds / 5).")

# ==========================================
# 8. MARKET INVERSION SANITY (pillar inputs)
# ==========================================
println("\n", "="^70, "\n8. MARKET INVERSION SANITY\n", "="^70)
sample_ids = unique(odds.match_id)
println("  matches with any odds: ", length(sample_ids))

F1 = Dict{Symbol, Any}()
Features.add_feature!(F1, Features.DoublePoissonMarketFeature(), sample_ids, Dict{Any,Any}(), ds)
_ok(x) = !ismissing(x) && !isnan(Float64(x)) && 0.02 < Float64(x) < 20.0
lam_h = F1[:flat_market_λ_home]; lam_a = F1[:flat_market_λ_away]
n_ok = count(_ok.(lam_h) .& _ok.(lam_a))
println("  8a. DP market inversion: ", n_ok, "/", length(sample_ids),
        " matches with plausible (0.02, 20) λ pair (",
        round(100n_ok / max(length(sample_ids), 1), digits=1), "%)")
good = [Float64(x) for x in lam_h if _ok(x)]
isempty(good) || println("      λ_home quantiles (ok subset): ",
                         round.(quantile(good, [0.05, 0.5, 0.95]), digits=3))

F2 = Dict{Symbol, Any}()
Features.add_feature!(F2, Features.MarketSmileFeature(Kmax=4), sample_ids, Dict{Any,Any}(), ds)
logΛ = F2[:flat_smile_logΛ]; msk = F2[:flat_smile_mask]
full_rows = findall(i -> all(msk[i, :] .> 0.5), 1:size(msk, 1))
println("  8b. smile inversion: full-ladder (all 5 strikes) matches = ", length(full_rows),
        "/", length(sample_ids))
println("      per-strike coverage: ", [Int(sum(msk[:, k])) for k in 1:size(msk, 2)])
if !isempty(full_rows)
    Λ_med = exp.(vec(median(logΛ[full_rows, :], dims=1)))
    println("      median Λ^mkt(K) over strikes 0.5→4.5: ", round.(Λ_med, digits=3),
            "  rising=", all(diff(Λ_med) .> 0))
    println("      all finite=", all(isfinite, logΛ[full_rows, :]))
end

# ==========================================
# 9. BETFAIR — expected EMPTY for 54/55
# ==========================================
println("\n", "="^70, "\n9. BETFAIR COVERAGE (expected: none)\n", "="^70)
if nrow(ds.betfair_odds) == 0
    println("  ✅ ds.betfair_odds is empty, as expected — 54/55 were never collected")
    println("     (core.tournament_config: betfair_competition_id = NULL, no betfair component).")
    println("     Benchmark AND training anchor are both the de-vigged Bet365 close. No runner in")
    println("     this stream may reference ds.betfair_odds.")
else
    println("  ⚠ UNEXPECTED: ", nrow(ds.betfair_odds), " betfair rows over ",
            length(unique(ds.betfair_odds.match_id)), " matches.")
    println("     Betfair history has landed since the plan was written → re-open the anchor A/B")
    println("     at r04 (score BOTH arms against the SAME benchmark; anchoring to Betfair and then")
    println("     scoring against Betfair mechanically compresses the spread).")
end

println("\n[INFO] r00 complete. Record season strings, biweek fold count, δ_league gap and the")
println("       history_seasons decision in NOTES.md before running r01.")
