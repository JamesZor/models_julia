# current_development/plus_minus_ratings/r03_targets_qa.jl
#
# RUNNER. WP4 — build the five targets and check they are what we think they are.
#
# GATES:
#   T1 clock alignment  — BBC shot times and SofaScore incident times MUST share a clock, or
#                         shots land in the wrong segment and every shot-based target is quietly
#                         wrong. Checked by comparing goal timings, which both sources record.
#   T2 attribution      — do the per-segment shot counts sum back to the match totals?
#   T3 sparsity         — %(y = 0) per target. This is the whole justification for the ladder:
#                         the base paper reports 72% for goals and we reproduced 72.2% in WP2.
#   T4 signal shape     — variance, and correlation between targets on the common subset
#   T5 in-play model    — coefficient sanity (does being a goal up lower your scoring rate? does
#                         a red card hurt?) and the t=0 expected points vs the empirical base rate
#   T6 coverage         — the shot-based targets only exist from 23/24. Every comparison must run
#                         on the COMMON subset or the denser targets get flattered by recency.

using DataFrames
using Statistics
using Printf

include(joinpath(@__DIR__, "l03_targets.jl"))

_hdr(s) = println("\n", "="^78, "\n", s, "\n", "="^78)

_hdr("Building segments and shots")
@time SEG, REJ = build_segments()
@time SH = build_shots()
XGM = fit_shot_xg(SH)
SH.xg = predict_xg(XGM, SH)
@printf("segments %d | shots %d | matches with shots %d\n",
        nrow(SEG), nrow(SH), length(unique(SH.match_id)))

# ==========================================
# T1 — CLOCK ALIGNMENT  (must pass before anything else means anything)
# ==========================================
_hdr("T1 — do BBC and SofaScore share a clock?")
inc = PM_INCIDENTS[]
sofa_goals = inc[coalesce.(inc.incident_type .== "goal", false), :]
bbc_goals  = SH[SH.is_goal, :]

# Build the per-match time lists BY HAND. `combine` flattens a vector-valued return into one row
# per element, which silently turns the subsequent join into a cartesian product — the first run
# of this gate "failed" with p95 |diff| = 68 min purely from that.
function goal_times_by_match(df)
    d = Dict{Int, Vector{Float64}}()
    for r in eachrow(df)
        t = pm_clock(r.time, r.added_time, 0.0)
        isnan(t) && continue
        push!(get!(d, Int(r.match_id), Float64[]), t)
    end
    for v in values(d); sort!(v); end
    return d
end
sg = goal_times_by_match(sofa_goals)
bg = goal_times_by_match(bbc_goals)

diffs = Float64[]; n_pairs = 0; n_len_mismatch = 0
for (mid, a) in sg
    b = get(bg, mid, nothing); b === nothing && continue
    if length(a) != length(b); n_len_mismatch += 1; continue; end
    n_pairs += 1
    append!(diffs, a .- b)
end
@printf("matches with goals in both sources: %d | dropped for differing goal counts: %d\n",
        n_pairs, n_len_mismatch)
if isempty(diffs)
    println("NO comparable matches — investigate before trusting any shot-based target.")
else
    @printf("paired goal timings: %d | mean diff %.3f min | median %.1f | %% within ±1 min: %.1f\n",
            length(diffs), mean(diffs), median(diffs), 100 * mean(abs.(diffs) .<= 1.0))
    @printf("%% within ±2 min: %.1f | p95 |diff| %.1f\n",
            100 * mean(abs.(diffs) .<= 2.0), quantile(abs.(diffs), 0.95))
    println(abs(mean(diffs)) < 0.5 && mean(abs.(diffs) .<= 1.0) > 0.9 ?
            "GATE PASSED: the two sources agree on the clock." :
            "GATE FAILED: systematic offset — shots would be misassigned to segments.")
end

# ==========================================
# BUILD THE TARGETS
# ==========================================
_hdr("Fitting the in-play hazard and building all five targets")
@time LONG = build_state_intervals()
@printf("state intervals: %d rows (%d match-intervals × 2 sides)\n", nrow(LONG), nrow(LONG) ÷ 2)
@time HAZ = fit_inplay_hazard(LONG)
println(coeftable(HAZ))
println("\nRead: tbin coefficients should RISE with time (more goals late); mp_c should be")
println("clearly positive (a man up scores more); is_home positive. NOTE gd_f is a strength")
println("PROXY as much as a game-state effect — the model is deliberately team-blind, so a team")
println("3 goals up looks high-scoring partly because it is the better team. That is the base")
println("paper's intended behaviour (§4.2), not a defect, but it does mean xPPM rewards players")
println("for being ahead in a way that partly reflects their team.")
@time XP = xp_table(HAZ)
@time add_targets!(SEG, SH, XP)

# ==========================================
# T5 — IN-PLAY MODEL SANITY
# ==========================================
_hdr("T5 — in-play model sanity")
h0, a0 = expected_points(XP, 0.0, 0, 0)
@printf("expected points at kickoff (0-0, 11v11): home %.3f  away %.3f\n", h0, a0)
emp = combine(groupby(unique(select(PM_LINEUPS[], :match_id, :home_score, :away_score)), []),
              [:home_score, :away_score] =>
                 ((h, a) -> mean(skipmissing(3 .* (h .> a) .+ (h .== a)))) => :xp_h,
              [:home_score, :away_score] =>
                 ((h, a) -> mean(skipmissing(3 .* (a .> h) .+ (h .== a)))) => :xp_a)
println("empirical points per match: ", emp)
println("(base paper's EPL figures were 1.63 home / 1.11 away — a lower league should be flatter)")

for (t, gd, mp, lbl) in ((0.0, 0, 0, "kickoff level"), (45.0, 1, 0, "HT, 1 up"),
                         (45.0, 0, 1, "HT, level, opp down to 10"), (80.0, 1, 0, "80', 1 up"),
                         (80.0, -1, 0, "80', 1 down"))
    h, a = expected_points(XP, t, gd, mp)
    @printf("  %-30s xP_home %.3f  xP_away %.3f\n", lbl, h, a)
end

# ==========================================
# T2 — ATTRIBUTION
# ==========================================
_hdr("T2 — do per-segment shots sum back to match totals?")
per_match_seg = combine(groupby(SEG[SEG.covered, :], :match_id),
                        [:shots_h, :shots_a] => ((a, b) -> sum(a) + sum(b)) => :seg_shots)
attributable = SH[.!ismissing.(SH.is_home), :]
per_match_raw = combine(groupby(attributable, :match_id), nrow => :raw_shots)
mm = innerjoin(per_match_seg, per_match_raw, on = :match_id)
@printf("matches compared: %d | exact match: %.2f%% | mean shortfall %.3f shots\n",
        nrow(mm), 100 * mean(mm.seg_shots .== mm.raw_shots),
        mean(mm.raw_shots .- mm.seg_shots))
println("Any shortfall is shots whose timestamp fell outside every segment — should be ~0.")

# ==========================================
# T6 / T3 — COVERAGE AND SPARSITY
# ==========================================
_hdr("T6 — coverage: the shot-based targets exist on a smaller match set")
cov = combine(groupby(SEG, [:tournament_id, :season]),
              nrow => :segments, :covered => (c -> round(100 * mean(c), digits = 1)) => :pct_covered)
println(sort(cov, [:tournament_id, :season]))
COM = SEG[SEG.covered, :]
@printf("\ncommon subset (live_text present): %d of %d segments (%.1f%%), %d matches\n",
        nrow(COM), nrow(SEG), 100 * nrow(COM) / nrow(SEG), length(unique(COM.match_id)))

_hdr("T3 — sparsity of each target, ON THE COMMON SUBSET")
rows = NamedTuple[]
for t in TARGETS
    v = COM[!, t]
    push!(rows, (target = String(t), pct_zero = round(100 * mean(v .== 0), digits = 1),
                 sd = round(std(v), digits = 4), mean_abs = round(mean(abs.(v)), digits = 4),
                 p01 = round(quantile(v, 0.01), digits = 3),
                 p99 = round(quantile(v, 0.99), digits = 3)))
end
S3 = DataFrame(rows)
println(S3)
println("\nGoals sparsity is the number to beat: the base paper reports 72%, WP2 reproduced 72.2%.")
println("A target that is materially denser AND still correlated with goals is the point of WP4.")

# ==========================================
# T4 — SIGNAL SHAPE
# ==========================================
_hdr("T4 — correlation between targets (common subset)")
M = Matrix(COM[!, TARGETS])
C = round.(cor(M), digits = 3)
println(DataFrame(hcat(String.(TARGETS), C), vcat(:target, TARGETS...)))

println("\nPer-90-normalised signal-to-noise by target (sd of the per-90 rate; higher = more")
println("discriminating between segments, but only meaningful alongside the correlation to goals):")
for t in TARGETS
    v = COM[!, t] ./ (COM.duration ./ 90)
    @printf("  %-9s sd(per-90) %8.3f   cor with goals-per-90 %6.3f\n", String(t), std(v),
            cor(v, COM.y_goals ./ (COM.duration ./ 90)))
end

const TQA = (segments = SEG, shots = SH, hazard = HAZ, xp = XP, sparsity = S3, corr = C)
_hdr("WP4 done — inspect `TQA`, then write the verdict into NOTES.md")
