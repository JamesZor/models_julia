# ==============================================================================
# r94 — pxG-APM / RAPM forensics
# What the stint ridge is fitted on, what it produces, and how hard it is shrunk
# ==============================================================================
#
# WHAT THIS IS
#   A descriptive and diagnostic pass over `Features.PxGRapmFeature` on Scottish
#   tiers 56/57.
#
#   FIVE QUESTIONS.
#     1. What does the stint segmentation actually produce, and what does it reject?
#     2. How do the four plus-minus targets differ — and is `:y_xg` the right default?
#     3. What does lambda do to the rating spread, and where is reliability maximised?
#     4. What do the player ratings look like, by position and by exposure?
#     5. Does the starting-XI differential — the covariate itself — carry signal?
#
# WHAT THIS IS NOT
#   Not a validation against an external player rating. Tiers 56/57 carry ZERO
#   SofaScore player ratings (verified: `lineups.rating` is missing on all 74,225
#   rows) — that absence is the entire reason RAPM was built for these tiers. The
#   research validated the method on tiers 54/55 where the yardstick exists; this
#   runner measures internal properties only: reliability, shrinkage, spread, signal.
#
#   Not a model fit. Nothing here samples.
#
# ⚠ THE MEASUREMENT THAT MATTERS MOST IS SPLIT-HALF RELIABILITY.
#   A ridge coefficient can look beautifully distributed and still be noise. Section 5
#   refits the ratings on two disjoint halves of the match set and correlates the
#   players who appear in both. That number is the ceiling on everything downstream.
#
# USAGE
#   julia --project -t 8
#   julia> include("current_development/scottish_lower/r94_pxg_rapm_forensics.jl")
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
using Random
using Statistics

include(joinpath(@__DIR__, "l93_eda_toolkit.jl"))

# %%
# ==============================================================================
# 2. Configuration
# ==============================================================================

const R94_SEGMENT = Data.ScottishLower()
const R94_SEED = 20260830
const R94_TARGETS = (:y_xg, :y_goals, :y_shots, :y_sot)
const R94_LAMBDAS = [50.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10_000.0, 20_000.0, 50_000.0]
const R94_HALF_LIFE = 730.0
const R94_EXPOSURE_FLOOR = 40         # segments a player needs before entering reliability
const R94_TOP_N = 15
const R94_SHIPPED = PxGRapmFeature()

eda_banner("r94 · pxG-APM / RAPM forensics (Scottish tiers 56/57)")
println("  shipped defaults: target=", R94_SHIPPED.target, " lambda=", R94_SHIPPED.lambda,
        " w_sim=", R94_SHIPPED.w_sim, " half_life=", R94_SHIPPED.half_life_days,
        " shrink_segments=", R94_SHIPPED.shrink_segments)

# %%
# ==============================================================================
# 3. Data snapshot and stint construction
# ==============================================================================

eda_section("1/8", "Data snapshot")

ds = Data.load_datastore_cached(R94_SEGMENT; max_age_hours = 100_000)
@printf("  matches %d | lineups %d | incidents %d | bbc_events %d\n",
        nrow(ds.matches), nrow(ds.lineups), nrow(ds.incidents), nrow(ds.bbc_events))
@printf("  SofaScore player ratings present: %d / %d  <- the reason RAPM exists here\n",
        count(!ismissing, ds.lineups.rating), nrow(ds.lineups))

prep = Features.pm_prepared(ds)
segments = prep.segments
segments_all, rejects = Features.build_segments(ds)
@printf("  segments built: %d over %d matches | rejected matches: %d\n",
        nrow(segments), length(unique(segments.match_id)), nrow(rejects))

# %%
# ==============================================================================
# 4. Stint distributions
# ==============================================================================

eda_section("2/8", "Stint segmentation — what the ridge is fitted on")

per_match = combine(groupby(segments, :match_id), nrow => :n_segments)
eda_print_describe([
    eda_describe("segments per match", per_match.n_segments),
    eda_describe("segment duration (min)", segments.duration),
    eda_describe("home players on pitch", [length(p) for p in segments.home_players]),
    eda_describe("away players on pitch", [length(p) for p in segments.away_players]),
    eda_describe("goal diff at start", segments.gd_start),
]; title = "  Segment geometry:")

println()
eda_histogram(segments.duration; label = "Segment duration (minutes):", bins = 20)

@printf("\n  segments with a dismissal in force: %d (%.2f%%) — home %d, away %d\n",
        count(r -> r.red_home > 0 || r.red_away > 0, eachrow(segments)),
        100 * count(r -> r.red_home > 0 || r.red_away > 0, eachrow(segments)) / nrow(segments),
        count(>(0), segments.red_home), count(>(0), segments.red_away))
@printf("  live-text covered segments        : %d (%.1f%%)\n",
        count(segments.covered), 100 * count(segments.covered) / nrow(segments))
println("  (shot-based targets are restricted to covered segments — feeding the ridge a")
println("   fake 0 for every uncovered match would be tens of thousands of phantom rows)")

println("\n  Rejected matches by reason (a rejection loses the whole match, not one stint):")
eda_freq_table(rejects.reason; top = 8)

println("\n  Segments per season:")
@printf("  %-10s | %10s | %10s | %12s | %10s\n",
        "season", "matches", "segments", "seg/match", "covered %")
println("  " * repeat('-', 62))
for season in sort(unique(segments.season))
    sub = segments[segments.season .== season, :]
    n_matches = length(unique(sub.match_id))
    @printf("  %-10s | %10d | %10d | %12.2f | %9.1f%%\n",
            season, n_matches, nrow(sub), nrow(sub) / n_matches,
            100 * count(sub.covered) / nrow(sub))
end

exposure = Features.player_exposure(segments)
@printf("\n  distinct players: %d | median segments/player: %.0f | median minutes: %.0f\n",
        nrow(exposure), median(exposure.n_segments), median(exposure.minutes))
@printf("  players below the research's 540-minute analysis floor: %d (%.1f%%)\n",
        count(<(540), exposure.minutes), 100 * count(<(540), exposure.minutes) / nrow(exposure))
println()
eda_histogram(exposure.minutes; label = "Player on-pitch minutes:", bins = 20)

# %%
# ==============================================================================
# 5. The four targets
# ==============================================================================

eda_section("3/8", "Targets — is `:y_xg` the right default?")

covered = segments[segments.covered, :]
eda_print_describe([eda_describe(String(t), covered[!, t]) for t in R94_TARGETS];
                   title = "  Per-segment response, on live-text covered segments:")

println("\n  Zero-inflation — a target that is 0 most of the time carries little per stint:")
for t in R94_TARGETS
    v = covered[!, t]
    @printf("    %-10s exactly zero on %.1f%% of covered segments\n",
            String(t), 100 * count(iszero, v) / length(v))
end

println("\n  Correlation between targets (covered segments):")
target_cols = [Float64.(covered[!, t]) for t in R94_TARGETS]
eda_print_corr(String.(collect(R94_TARGETS)),
               eda_corr_matrix(String.(collect(R94_TARGETS)), target_cols); flag = 0.75)

println("\n  [READ] `:y_goals` is the base paper's plus-minus and is the most zero-inflated.")
println("         `:y_xg` is the same event stream weighted by chance quality, so it is far")
println("         less lumpy per stint — which is why it is the default for a covariate that")
println("         has to say something about a 20-minute personnel window.")

# %%
# ==============================================================================
# 6. Lambda sweep and split-half reliability
# ==============================================================================

eda_section("4/8", "Ridge lambda — shrinkage and reliability")

comp_sets = Features.competition_sets(ds)
T_rating = maximum(segments.match_date)

"Fit one rating vector over a segment subset."
function r94_fit(segs::DataFrame; target::Symbol, lambda::Float64)
    return Features.fit_ratings(segs; target = target, λ = lambda, w_sim = 0.0,
                                half_life = R94_HALF_LIFE, T_rating = T_rating,
                                comp_sets = comp_sets)
end

# Split-half: two disjoint halves of the MATCH set, so a player's two ratings come
# from entirely different games. Splitting on segments would leak — the same match
# would inform both halves.
Random.seed!(R94_SEED)
match_ids = shuffle(unique(Int.(segments.match_id)))
half_a = Set(match_ids[1:fld(end, 2)])
half_b = Set(setdiff(match_ids, half_a))
segs_a = segments[in.(Int.(segments.match_id), Ref(half_a)), :]
segs_b = segments[in.(Int.(segments.match_id), Ref(half_b)), :]
@printf("  split-half: %d vs %d matches (%d vs %d segments), seed %d\n\n",
        length(half_a), length(half_b), nrow(segs_a), nrow(segs_b), R94_SEED)

reliable_players = Set(Int.(exposure[exposure.n_segments .>= R94_EXPOSURE_FLOOR, :player_id]))

sweep = NamedTuple[]
for target in (:y_xg, :y_goals, :y_shots)
    for lambda in R94_LAMBDAS
        full = r94_fit(segments; target = target, lambda = lambda)
        full === nothing && continue
        fit_a = r94_fit(segs_a; target = target, lambda = lambda)
        fit_b = r94_fit(segs_b; target = target, lambda = lambda)

        r_split, rho_split, n_common = NaN, NaN, 0
        if fit_a !== nothing && fit_b !== nothing
            ra = Dict(Int(r.player_id) => r.rapm for r in eachrow(fit_a))
            rb = Dict(Int(r.player_id) => r.rapm for r in eachrow(fit_b))
            common = [p for p in intersect(keys(ra), keys(rb)) if p in reliable_players]
            n_common = length(common)
            if n_common >= 20
                va = [ra[p] for p in common]
                vb = [rb[p] for p in common]
                r_split = eda_pearson(va, vb)
                rho_split = eda_spearman(va, vb)
            end
        end
        push!(sweep, (target = String(target), lambda = lambda, n_players = nrow(full),
                      sd = std(full.rapm), iqr = quantile(full.rapm, 0.75) - quantile(full.rapm, 0.25),
                      absmax = maximum(abs, full.rapm),
                      n_common = n_common, split_r = r_split, split_rho = rho_split))
    end
end

@printf("  %-9s | %8s | %8s | %8s | %8s | %8s | %9s | %9s\n",
        "target", "lambda", "players", "sd", "IQR", "|max|", "n_common", "split-half")
println("  " * repeat('-', 96))
for r in sweep
    @printf("  %-9s | %8.0f | %8d | %8.4f | %8.4f | %8.4f | %9d | %9s\n",
            r.target, r.lambda, r.n_players, r.sd, r.iqr, r.absmax, r.n_common,
            eda_fmt(r.split_r, "%.4f"))
end

println("\n  Shrinkage is monotone in lambda by construction; reliability is not. The cell to")
println("  pick is the one that maximises split-half correlation, NOT the one with the")
println("  widest spread — a wide spread of noise is still noise.")
println()
println("  [READ THE PLATEAU, NOT THE ARGMAX] As lambda grows the ridge solution converges in")
println("  DIRECTION to the unregularised gradient X'Wy, so split-half reliability rises and")
println("  then FLATTENS rather than peaking. Picking the argmax therefore drives lambda to the")
println("  edge of whatever grid happens to be written. The usable reading is where the curve")
println("  flattens; the binding decision is r40's out-of-sample log loss.\n")
for target in ("y_xg", "y_goals", "y_shots")
    rows = filter(r -> r.target == target && !isnan(r.split_r), sweep)
    isempty(rows) && continue
    best = rows[argmax([r.split_r for r in rows])]
    shipped = filter(r -> r.lambda == R94_SHIPPED.lambda, rows)
    @printf("    %-8s best split-half %.4f at lambda = %.0f", target, best.split_r, best.lambda)
    if !isempty(shipped)
        @printf("   (shipped lambda=%.0f gives %.4f)", R94_SHIPPED.lambda, shipped[1].split_r)
    end
    println()
end

# %%
# ==============================================================================
# 7. Player rating distributions
# ==============================================================================

eda_section("5/8", "Player rating distributions")

ratings_xg = r94_fit(segments; target = :y_xg, lambda = R94_SHIPPED.lambda)
ratings_goals = r94_fit(segments; target = :y_goals, lambda = R94_SHIPPED.lambda)

# Position and name come from the lineups, taking each player's modal position.
position_of = Dict{Int,String}()
name_of = Dict{Int,String}()
position_counts = Dict{Int,Dict{String,Int}}()
for r in eachrow(ds.lineups)
    ismissing(r.player_id) && continue
    pid = Int(r.player_id)
    ismissing(r.player_name) || (name_of[pid] = String(r.player_name))
    pos = Features.pm_clean_position(r.position)
    d = get!(position_counts, pid, Dict{String,Int}())
    d[pos] = get(d, pos, 0) + 1
end
for (pid, counts) in position_counts
    position_of[pid] = argmax(counts)
end

rated = DataFrame(
    player_id = Int.(ratings_xg.player_id),
    rapm_xg = Float64.(ratings_xg.rapm),
)
goals_map = Dict(Int(r.player_id) => Float64(r.rapm) for r in eachrow(ratings_goals))
exposure_map = Dict(Int(r.player_id) => (segs = r.n_segments, mins = r.minutes)
                    for r in eachrow(exposure))
rated.rapm_goals = [get(goals_map, p, NaN) for p in rated.player_id]
rated.position = [get(position_of, p, "?") for p in rated.player_id]
rated.name = [get(name_of, p, "player $p") for p in rated.player_id]
rated.n_segments = [get(exposure_map, p, (segs = 0, mins = 0.0)).segs for p in rated.player_id]
rated.minutes = [get(exposure_map, p, (segs = 0, mins = 0.0)).mins for p in rated.player_id]

eda_print_describe([
    eda_describe("RAPM :y_xg", rated.rapm_xg),
    eda_describe("RAPM :y_goals", rated.rapm_goals),
]; title = "  Player ratings at the shipped lambda:")
println()
eda_histogram(rated.rapm_xg; label = "RAPM (:y_xg) across all rated players:", bins = 22)

@printf("\n  agreement between the two targets' ratings: r = %.4f, rho = %.4f\n",
        eda_pearson(rated.rapm_xg, rated.rapm_goals),
        eda_spearman(rated.rapm_xg, rated.rapm_goals))

println("\n  By modal position — the research found goalkeeper ratings worthless (rho ~ 0")
println("  against the SofaScore rating), so a GK spread near the outfield spread is a")
println("  warning, not a feature:")
@printf("  %-6s | %8s | %10s | %10s | %10s | %10s\n",
        "pos", "players", "mean", "sd", "min", "max")
println("  " * repeat('-', 66))
for pos in ("G", "D", "M", "F", "?")
    sub = rated[rated.position .== pos, :]
    nrow(sub) == 0 && continue
    @printf("  %-6s | %8d | %+10.5f | %10.5f | %+10.5f | %+10.5f\n",
            pos, nrow(sub), mean(sub.rapm_xg), std(sub.rapm_xg),
            minimum(sub.rapm_xg), maximum(sub.rapm_xg))
end

println("\n  By exposure quintile — a rating from few segments should be nearer zero if the")
println("  ridge is doing its job:")
sorted_exposure = sort(rated, :n_segments)
n = nrow(sorted_exposure)
@printf("  %-10s | %8s | %12s | %10s | %10s\n",
        "quintile", "players", "segments", "mean |r|", "sd")
println("  " * repeat('-', 60))
for q in 1:5
    lo = 1 + fld((q - 1) * n, 5)
    hi = fld(q * n, 5)
    sub = sorted_exposure[lo:hi, :]
    @printf("  %-10d | %8d | %12.0f | %10.5f | %10.5f\n",
            q, nrow(sub), median(sub.n_segments), mean(abs, sub.rapm_xg), std(sub.rapm_xg))
end

# %%
# ==============================================================================
# 8. Top and bottom rated players
# ==============================================================================

eda_section("6/8", "Top and bottom rated Scottish Lower players")

qualified = rated[rated.minutes .>= 540, :]
@printf("  %d players clear the 540-minute analysis floor.\n\n", nrow(qualified))

function r94_print_players(frame::DataFrame, title::AbstractString)
    println("  ", title)
    @printf("  %4s | %-30s | %4s | %8s | %9s | %10s | %10s\n",
            "#", "player", "pos", "minutes", "segments", "RAPM xg", "RAPM goals")
    println("  " * repeat('-', 94))
    for (i, r) in enumerate(eachrow(frame))
        @printf("  %4d | %-30s | %4s | %8.0f | %9d | %+10.5f | %+10s\n",
                i, first(r.name, 30), r.position, r.minutes, r.n_segments,
                r.rapm_xg, eda_fmt(r.rapm_goals, "%+.5f"))
    end
end

r94_print_players(first(sort(qualified, :rapm_xg, rev = true), R94_TOP_N),
                  "Highest-rated (:y_xg, 540+ minutes):")
println()
r94_print_players(first(sort(qualified, :rapm_xg), R94_TOP_N),
                  "Lowest-rated (:y_xg, 540+ minutes):")

println("\n  [CAVEAT] These are ridge coefficients on a stint xG differential, not a scouting")
println("           verdict. A player on a dominant side inherits some of his team's rating;")
println("           w_sim = 0 limits but does not eliminate that. Read the split-half number")
println("           in section 4 before reading any individual name here.")

# %%
# ==============================================================================
# 9. The starting-XI differential — the covariate itself
# ==============================================================================

eda_section("7/8", "Starting-XI differential")

fs = eda_features(ds, [R94_SHIPPED])
column = fs.data[:flat_pxg_rapm]
aligned = eda_match_frame(ds; ordered_ids = Int.(fs.data[:ordered_match_ids]))

@printf("  rated players %d | auto scale %.5f | available on %.1f%% of matches\n",
        length(fs.data[:pxg_rapm_ratings]), fs.data[:pxg_rapm_scale],
        100 * mean(fs.data[:flat_pxg_rapm_available]))

eda_print_describe([eda_describe("pxg_rapm column", column)];
                   title = "\n  The design column:")
println()
eda_histogram(column; label = "Starting-XI RAPM differential (standardised):", bins = 22)

println()
println("  ⚠ IN-SAMPLE vs HELD-OUT. The block above fits the ridge on every match, so")
println("  correlating it against those same matches lets a goal-differential target")
println("  memorise the goal differences it is scored on. Both are printed; only the")
println("  held-out block is evidence.\n")

in_sample = NamedTuple[eda_signal("shipped (:y_xg)", column, aligned)]
for target in (:y_goals, :y_shots, :y_sot)
    fsv = eda_features(ds, [PxGRapmFeature(target = target, lambda = R94_SHIPPED.lambda)])
    av = eda_match_frame(ds; ordered_ids = Int.(fsv.data[:ordered_match_ids]))
    push!(in_sample, eda_signal("target=$(target)", fsv.data[:flat_pxg_rapm], av))
end
eda_print_signal(in_sample; title = "IN-SAMPLE (ridge fitted on the scored matches — inflated):")

println()
signal_rows = NamedTuple[]
for target in (:y_xg, :y_goals, :y_shots, :y_sot)
    push!(signal_rows, eda_holdout_signal(
        ds, PxGRapmFeature(target = target, lambda = R94_SHIPPED.lambda),
        :flat_pxg_rapm, "target=$(target)"))
end
for lambda in (200.0, 1000.0, 5000.0, 20_000.0)
    push!(signal_rows, eda_holdout_signal(
        ds, PxGRapmFeature(target = :y_xg, lambda = lambda),
        :flat_pxg_rapm, "y_xg lambda=$(Int(lambda))"))
end
for shrink in (0.0, 20.0, 60.0)
    push!(signal_rows, eda_holdout_signal(
        ds, PxGRapmFeature(shrink_segments = shrink),
        :flat_pxg_rapm, "shrink_segments=$(shrink)"))
end
eda_print_signal(signal_rows; title = "HELD-OUT (ridge fitted on the first 80% only — the honest read):")

println()
eda_print_decile(eda_decile_table(column, aligned.supremacy; k = 10);
                 xlab = "XI RAPM", ylab = "goal sup",
                 title = "Does the starting-XI differential order real scorelines?")

# %%
# ==============================================================================
# 10. Verdict
# ==============================================================================

eda_section("8/8", "Verdict")

let
    shipped_signal = signal_rows[1]   # held-out :y_xg
    best_split = filter(r -> r.target == "y_xg" && !isnan(r.split_r), sweep)
    @printf("  Covariate: r(supremacy) = %+.4f, AUC(home win) = %.4f, available on %.1f%%\n",
            shipped_signal.r_supremacy, shipped_signal.auc_home_win,
            100 * mean(fs.data[:flat_pxg_rapm_available]))
    if !isempty(best_split)
        best = best_split[argmax([r.split_r for r in best_split])]
        @printf("  Reliability: best split-half %.4f at lambda = %.0f (shipped lambda = %.0f)\n",
                best.split_r, best.lambda, R94_SHIPPED.lambda)
        if best.split_r < 0.4
            println("  [WARN] Split-half reliability below 0.40. The rating is substantially noise;")
            println("         treat any individual player ordering as indicative at best.")
        end
    end
    @printf("  Segments: %d over %d matches, %.1f%% live-text covered\n",
            nrow(segments), length(unique(segments.match_id)),
            100 * count(segments.covered) / nrow(segments))
end
eda_rule(100, '=')
