# ==============================================================================
# r95 — pxG model forensics
# What the BBC live-text proxy is made of, and whether the form window is tuned
# ==============================================================================
#
# WHAT THIS IS
#   A descriptive and diagnostic pass over `Features.PxGFeature` on Scottish
#   tiers 56/57 — the tiers it is actually deployed on.
#
#   FOUR QUESTIONS.
#     1. What is each match measured WITH?  (the commentary / shot-count / goals ladder)
#     2. What does the parser actually see?  (zone x body x context, and the fitted cells)
#     3. Is the proxy calibrated against real scorelines?
#     4. Is the form window tuned, or is 8 matches an unexamined default?
#
# WHAT THIS IS NOT
#   Not an external-validity study. Tiers 56/57 carry NO SofaScore statistics, no
#   player ratings and no xG — that absence is why the proxy exists. Agreement with
#   official xG is r92's job, on the six tiers where a reference exists.
#
#   Not a betting study, and not a model fit: nothing here samples.
#
# ⚠ ONE TRAP THIS RUNNER IS BUILT AROUND
#   The measurement ladder's bottom rung IS the scoreline. On a match measured by
#   the goals fallback, "pxG vs goals" is a tautology and would report a perfect
#   correlation that means nothing. Section 5 therefore restricts every calibration
#   check to COMMENTARY-SOURCED matches, and says so in the output.
#
# USAGE
#   julia --project -t 8
#   julia> include("current_development/scottish_lower/r95_pxg_model_forensics.jl")
# ==============================================================================

# %%
# ==============================================================================
# 1. Packages and implementation
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using LinearAlgebra
using Printf
using Statistics

include(joinpath(@__DIR__, "l93_eda_toolkit.jl"))

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================

const R95_SEGMENT = Data.ScottishLower()
const R95_K = 25.0                    # cell-table shrinkage; the production default
const R95_BASE = PxGFeature()         # the shipped defaults, for reference

# The sweep grid. Flat windows and exponential half-lives are swept together so the
# two decay families can be compared on one axis.
const R95_WINDOWS = [1, 2, 3, 4, 6, 8, 12, 19, 0]      # 0 = every earlier match
const R95_HALF_LIVES = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0]
const R95_PRIOR_WEIGHTS = [0.0, 1.0, 3.0, 6.0, 12.0]

eda_banner("r95 · pxG model forensics (Scottish tiers 56/57)")
println("  shipped defaults: ", R95_BASE)

# %%
# ==============================================================================
# 3. Data snapshot
# ==============================================================================

eda_section("1/7", "Data snapshot")

ds = Data.load_datastore_cached(R95_SEGMENT; max_age_hours = 100_000)
@printf("  matches %d | lineups %d | incidents %d | bbc %d | bbc_events %d\n",
        nrow(ds.matches), nrow(ds.lineups), nrow(ds.incidents),
        nrow(ds.bbc), nrow(ds.bbc_events))
@printf("  seasons: %s\n", join(sort(unique(skipmissing(ds.matches.season))), ", "))
@printf("  SofaScore statistics rows: %d  <- the reason a PROXY exists at all\n",
        nrow(ds.statistics))

frame = eda_match_frame(ds)
played = frame[frame.played, :]
@printf("  played matches: %d | mean goals/match: %.3f | home win %.1f%% draw %.1f%% away %.1f%%\n",
        nrow(played), mean(played.total),
        100 * mean(played.home_win), 100 * mean(played.draw), 100 * mean(played.away_win))

# %%
# ==============================================================================
# 4. The measurement ladder
# ==============================================================================

eda_section("2/7", "Measurement ladder — what is each match measured with?")

observations = Features.pxg_match_observations(ds, R95_BASE)
source_of = Dict(k => v.source for (k, v) in observations)

by_source = Dict{Symbol,Int}()
for v in values(observations)
    by_source[v.source] = get(by_source, v.source, 0) + 1
end
total = nrow(ds.matches)
println("  Overall:")
for rung in (:commentary, :shot_counts, :goals)
    n = get(by_source, rung, 0)
    @printf("    %-14s %6d  (%5.1f%%)\n", String(rung), n, 100 * n / total)
end
@printf("    %-14s %6d  (%5.1f%%)\n", "unmeasured",
        total - sum(values(by_source)), 100 * (total - sum(values(by_source))) / total)

println("\n  By season — BBC live text starts in 23/24, and the cutover should be visible:")
@printf("  %-10s | %8s | %11s | %12s | %8s | %11s\n",
        "season", "matches", "commentary", "shot counts", "goals", "commentary%")
println("  " * repeat('-', 78))
for season in sort(unique(skipmissing(ds.matches.season)))
    sub = ds.matches[coalesce.(ds.matches.season .== season, false), :]
    ids = Int.(sub.match_id)
    counts = Dict(r => count(id -> get(source_of, id, :none) === r, ids)
                  for r in (:commentary, :shot_counts, :goals))
    @printf("  %-10s | %8d | %11d | %12d | %8d | %10.1f%%\n",
            season, length(ids), counts[:commentary], counts[:shot_counts],
            counts[:goals], 100 * counts[:commentary] / max(length(ids), 1))
end

println("\n  By tier:")
for tier in sort(unique(ds.matches.tournament_id))
    ids = Int.(ds.matches[ds.matches.tournament_id .== tier, :match_id])
    n_comm = count(id -> get(source_of, id, :none) === :commentary, ids)
    @printf("    tier %d: %d matches, %.1f%% commentary-sourced\n",
            tier, length(ids), 100 * n_comm / max(length(ids), 1))
end

commentary_ids = Set(k for (k, v) in observations if v.source === :commentary)
println("\n  [GATE] Commentary-sourced matches are the only ones on which a pxG-vs-goals")
println("         calibration check is meaningful; the goals rung IS the scoreline.")
@printf("         usable for calibration: %d matches\n", length(commentary_ids))

# %%
# ==============================================================================
# 5. Shot-type distributions and the fitted cell table
# ==============================================================================

eda_section("3/7", "Shot descriptors and the fitted cell model")

shots = Features.build_shots(ds)
@printf("  attempts parsed from live text: %d over %d matches\n",
        nrow(shots), length(unique(shots.match_id)))
@printf("  parse coverage: %.2f%% | side unresolved: %.2f%% | penalties: %d\n",
        100 * count(shots.parsed) / max(nrow(shots), 1),
        100 * count(ismissing, shots.is_home) / max(nrow(shots), 1),
        count(shots.is_penalty))

println()
eda_freq_table(shots.zone; title = "Zone:", top = 12)
println()
eda_freq_table(shots.body_part; title = "Body part:", top = 6)
println()
eda_freq_table(shots.context; title = "Set-piece context:", top = 6)

model = Features.fit_shot_xg(shots; k = R95_K)
@printf("\n  Fitted cell model: base rate %.4f | penalty xG %.4f | %d cells | k = %.1f\n",
        model.base_rate, model.penalty_xg, length(model.cells), model.k)

# Cell-level calibration: does the shrunk cell value match the raw conversion rate?
# A systematic gap here is over- or under-shrinkage, which `k` controls.
open_play = shots[.!shots.is_penalty .& shots.parsed, :]
cell_rows = NamedTuple[]
for g in groupby(open_play, [:zone, :body_part, :context])
    n = nrow(g)
    n < 30 && continue
    key = (g.zone[1], g.body_part[1], g.context[1])
    push!(cell_rows, (
        cell = "$(key[1])/$(key[2])/$(key[3])",
        n = n,
        raw = mean(g.is_goal),
        fitted = get(model.cells, key, model.base_rate),
    ))
end
cells = sort!(DataFrame(cell_rows), :n, rev = true)
@printf("\n  Cell calibration over %d cells with n >= 30 (shrunk value vs raw conversion):\n",
        nrow(cells))
@printf("  %-46s | %7s | %8s | %8s | %8s\n", "cell", "n", "raw", "fitted", "shrink")
println("  " * repeat('-', 90))
for r in eachrow(first(cells, 16))
    @printf("  %-46s | %7d | %8.4f | %8.4f | %+8.4f\n",
            r.cell, r.n, r.raw, r.fitted, r.fitted - r.raw)
end
@printf("\n  across all n>=30 cells: r(raw, fitted) = %.4f | mean |shrink| = %.4f\n",
        eda_pearson(cells.raw, cells.fitted), mean(abs, cells.fitted .- cells.raw))
println("  (shrinkage should pull small cells toward the base rate and leave large ones alone)")

# %%
# ==============================================================================
# 6. Match-level pxG distribution
# ==============================================================================

eda_section("4/7", "Match-level pxG distribution")

pxg_rows = NamedTuple[]
for r in eachrow(played)
    obs = get(observations, r.match_id, nothing)
    obs === nothing && continue
    push!(pxg_rows, (match_id = r.match_id, source = obs.source,
                     pxg_h = obs.h, pxg_a = obs.a,
                     pxg_total = obs.h + obs.a, pxg_diff = obs.h - obs.a,
                     goals_h = r.goals_home, goals_a = r.goals_away,
                     supremacy = r.supremacy, total = r.total,
                     home_win = r.home_win))
end
pxg = DataFrame(pxg_rows)
comm = pxg[pxg.source .=== :commentary, :]

eda_print_describe([
    eda_describe("pxG home (commentary)", comm.pxg_h),
    eda_describe("pxG away (commentary)", comm.pxg_a),
    eda_describe("pxG total (commentary)", comm.pxg_total),
    eda_describe("pxG diff (commentary)", comm.pxg_diff),
    eda_describe("goals total", comm.total),
    eda_describe("goal supremacy", comm.supremacy),
]; title = "  Commentary-sourced matches only:")

println()
eda_histogram(comm.pxg_total; label = "pxG total per match (commentary-sourced):", bins = 20)
println()
eda_histogram(comm.pxg_diff; label = "pxG difference (home - away):", bins = 20)

@printf("\n  home share of pxG: %.3f (goals: %.3f) — the proxy's own home advantage\n",
        sum(comm.pxg_h) / sum(comm.pxg_total),
        sum(comm.goals_h) / max(sum(comm.total), 1))

# %%
# ==============================================================================
# 7. Calibration against real scorelines
# ==============================================================================

eda_section("5/7", "Calibration against actual scorelines (commentary-sourced only)")

println("  Match-level agreement between the proxy and what actually happened.")
println("  Perfect agreement is NOT expected and would be alarming: xG is a measure of")
println("  chance quality, and the gap between it and the scoreline is the finishing")
println("  variance the count model exists to represent.\n")

eda_print_agreement([
    eda_agreement("pxG total -> goals total", comm.pxg_total, comm.total),
    eda_agreement("pxG diff  -> goal supremacy", comm.pxg_diff, comm.supremacy),
]; title = "Proxy vs scoreline:")

println()
eda_print_decile(eda_decile_table(comm.pxg_total, comm.total; k = 10);
                 xlab = "pxG total", ylab = "goals",
                 title = "Does more pxG mean more goals?")
println()
eda_print_decile(eda_decile_table(comm.pxg_diff, comm.supremacy; k = 10);
                 xlab = "pxG diff", ylab = "goal sup",
                 title = "Does the pxG difference order real scorelines?")

@printf("\n  AUC, pxG difference -> home win: %.4f\n", eda_auc(comm.pxg_diff, comm.home_win))
@printf("  Poisson check — mean pxG total %.3f vs mean goals %.3f (ratio %.3f)\n",
        mean(comm.pxg_total), mean(comm.total), mean(comm.pxg_total) / mean(comm.total))
println("  (a ratio near 1 means the proxy is on the goals scale; it is a level check,")
println("   not an accuracy check)")

println()
eda_scatter(comm.pxg_diff, comm.supremacy;
            xlab = "pxG difference", ylab = "goal supremacy",
            title = "Chance quality vs outcome — the spread IS finishing variance:")

# %%
# ==============================================================================
# 8. Rolling-window / decay half-life sweep
# ==============================================================================

eda_section("6/7", "Form-window sweep — is `lookback = 8` tuned or assumed?")

println("  Each configuration is rebuilt end to end and its supremacy column scored")
println("  against realised goal supremacy. Every value is point-in-time, so this is a")
println("  pseudo-out-of-sample measure: no match ever sees itself or a later one.\n")

sweep_rows = NamedTuple[]
for window in R95_WINDOWS
    config = PxGFeature(lookback = window, decay = :window, prior_weight = 3.0,
                        min_matches = 3, fallback = :goals, scale = 1.0, k = R95_K)
    fs = eda_features(ds, [config])
    ids = Int.(fs.data[:ordered_match_ids])
    aligned = eda_match_frame(ds; ordered_ids = ids)
    signal = eda_signal("window=$(window == 0 ? "all" : window)",
                        fs.data[:flat_pxg_supremacy], aligned)
    push!(sweep_rows, signal)
end
for half_life in R95_HALF_LIVES
    config = PxGFeature(lookback = 0, decay = :exponential, half_life_matches = half_life,
                        prior_weight = 3.0, min_matches = 3, fallback = :goals,
                        scale = 1.0, k = R95_K)
    fs = eda_features(ds, [config])
    ids = Int.(fs.data[:ordered_match_ids])
    aligned = eda_match_frame(ds; ordered_ids = ids)
    push!(sweep_rows, eda_signal("half-life=$(half_life)", fs.data[:flat_pxg_supremacy], aligned))
end
eda_print_signal(sweep_rows; title = "Decay sweep (supremacy column vs goal supremacy):")

let best = sweep_rows[argmax([isnan(r.r_supremacy) ? -Inf : r.r_supremacy for r in sweep_rows])]
    @printf("\n  strongest association: %s (r = %+.4f, AUC = %.4f)\n",
            best.feature, best.r_supremacy, best.auc_home_win)
    shipped = filter(r -> r.feature == "window=8", sweep_rows)
    if !isempty(shipped)
        @printf("  shipped default window=8: r = %+.4f, AUC = %.4f (gap %+.4f)\n",
                shipped[1].r_supremacy, shipped[1].auc_home_win,
                best.r_supremacy - shipped[1].r_supremacy)
    end
end

println("\n  Shrinkage sweep at the shipped window (prior_weight pulls a thin team history")
println("  toward the league mean; 0 trusts a two-match sample completely):\n")
shrink_rows = NamedTuple[]
for prior_weight in R95_PRIOR_WEIGHTS
    config = PxGFeature(lookback = 8, decay = :window, prior_weight = prior_weight,
                        min_matches = 3, fallback = :goals, scale = 1.0, k = R95_K)
    fs = eda_features(ds, [config])
    aligned = eda_match_frame(ds; ordered_ids = Int.(fs.data[:ordered_match_ids]))
    push!(shrink_rows, eda_signal("prior_weight=$(prior_weight)",
                                  fs.data[:flat_pxg_supremacy], aligned))
end
eda_print_signal(shrink_rows)

println("\n  Ladder sensitivity — how much does the goals fallback contribute?\n")
ladder_rows = NamedTuple[]
for fb in (:none, :shots, :goals)
    config = PxGFeature(lookback = 8, decay = :window, prior_weight = 3.0,
                        min_matches = 3, fallback = fb, scale = 1.0, k = R95_K)
    fs = eda_features(ds, [config])
    aligned = eda_match_frame(ds; ordered_ids = Int.(fs.data[:ordered_match_ids]))
    push!(ladder_rows, eda_signal("fallback=$(fb)", fs.data[:flat_pxg_supremacy], aligned))
end
eda_print_signal(ladder_rows)

# %%
# ==============================================================================
# 9. Verdict
# ==============================================================================

eda_section("7/7", "Verdict")

let
    base_fs = eda_features(ds, [R95_BASE])
    aligned = eda_match_frame(ds; ordered_ids = Int.(base_fs.data[:ordered_match_ids]))
    shipped = eda_signal("shipped default", base_fs.data[:flat_pxg_supremacy], aligned)
    @printf("  Shipped PxGFeature(): r(supremacy) = %+.4f, AUC(home win) = %.4f, %.1f%% neutral\n",
            shipped.r_supremacy, shipped.auc_home_win, 100 * shipped.zero_share)
    @printf("  Commentary coverage : %.1f%% of matches | cell model: %d cells over %d attempts\n",
            100 * get(by_source, :commentary, 0) / total, length(model.cells), nrow(shots))
    println()
    println("  Read alongside r92, which scores this same proxy against official SofaScore xG")
    println("  on the six tiers where a reference exists. Nothing here measures agreement with")
    println("  a true xG, because on tiers 56/57 no such measurement exists.")
end
eda_rule(100, '=')
