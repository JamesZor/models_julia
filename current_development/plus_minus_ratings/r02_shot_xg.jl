# current_development/plus_minus_ratings/r02_shot_xg.jl
#
# RUNNER. WP3 — the BBC-text xG model and its gates.
#
# GATES:
#   X1 parse coverage   — ≥95% of shots must yield a zone, per tier × season
#   X2 face validity    — do the cell rates order the way football says they should?
#   X3 model ladder     — leave-one-season-out CV, Brier + LogLoss, against the per-shot-type
#                         empirical-frequency baseline the base paper used as its floor
#   X4 calibration      — decile plot of predicted vs realised
#   X5 team-level       — our team xG vs SofaScore's on 54/55. The PRIMARY calibration gate,
#                         because it needs no name matching.
#   X6 player-level     — same at player level, which ALSO measures the BBC-name→player_id match
#                         rate that WP1 flagged as unmeasured risk
#   X7 tier transfer    — fit Prem → score Champ (the downward direction that mimics 56/57)
#
# Base-paper anchors to beat/compare (Table 3, coordinate-based): open play 0.0739 vs baseline
# 0.0866; headers 0.0872 vs 0.0994; free kicks 0.0575 vs 0.0584; penalties 0.1848 vs 0.1848
# (no model beat the penalty base rate — hence one constant).

using DataFrames
using Statistics
using Printf

include(joinpath(@__DIR__, "l02_shot_parser.jl"))

_hdr(s) = println("\n", "="^78, "\n", s, "\n", "="^78)

_hdr("Building the shot table")
@time SH = build_shots()
@printf("shots: %d over %d matches | goals: %d (%.2f%%) | penalties: %d\n",
        nrow(SH), length(unique(SH.match_id)), sum(SH.is_goal),
        100 * mean(SH.is_goal), sum(SH.is_penalty))

# ==========================================
# X1 — PARSE COVERAGE
# ==========================================
_hdr("X1 — parse coverage (gate: ≥95%)")
x1 = combine(groupby(SH, [:tournament_id, :season]),
             nrow => :shots,
             :parsed     => (p -> round(100 * mean(p), digits = 1)) => :pct_parsed,
             :zone       => (z -> round(100 * mean(z .!== :unknown), digits = 1)) => :pct_zone,
             :body_part  => (b -> round(100 * mean(b .!== :unknown), digits = 1)) => :pct_body)
sort!(x1, [:tournament_id, :season])
println(x1)
bad = x1[x1.pct_parsed .< 95.0, :]
println(nrow(bad) == 0 ? "\nGATE PASSED: every tier × season ≥95% parsed." :
        "\nGATE FAILED on:\n$(bad)")

println("\nUnparsed examples (what the closed vocabulary is missing):")
let unparsed_txt = PM_LIVETEXT[][coalesce.(in.(PM_LIVETEXT[].event_type, Ref(SHOT_EVENTS)), false), :],
    shown = 0
    for t in skipmissing(unparsed_txt.text)
        shown >= 8 && break
        parse_shot("attempt_missed", t).parsed && continue
        println("  · ", first(String(t), 150)); shown += 1
    end
end

# ==========================================
# X2 — FACE VALIDITY
# ==========================================
_hdr("X2 — empirical conversion rates (do they order sensibly?)")
op = SH[.!SH.is_penalty .& SH.parsed, :]
for f in (:zone, :body_part, :context)
    t = combine(groupby(op, f), nrow => :shots, :is_goal => mean => :conv)
    t.conv = round.(100 .* t.conv, digits = 2)
    sort!(t, :conv, rev = true)
    println("\nby $f:"); println(t)
end
@printf("\npenalties: %d shots, %.1f%% converted (this becomes the single constant xG)\n",
        sum(SH.is_penalty), 100 * mean(SH.is_penalty .& SH.is_goal) / mean(SH.is_penalty))

# ==========================================
# X3 — MODEL LADDER, LEAVE-ONE-SEASON-OUT
# ==========================================
_hdr("X3 — model ladder (leave-one-season-out CV, pooled OOS)")
SH.block = string.(SH.tournament_id, "_", SH.season)
blocks = unique(SH.block)

LADDER = [
    ("m0 base rate"                => (:_, :_, :_)),
    ("m1 zone"                     => (:zone, :_, :_)),
    ("m2 zone+body"                => (:zone, :body_part, :_)),
    ("m3 zone+body+context"        => (:zone, :body_part, :context)),
]

res = NamedTuple[]
for (name, feats) in LADDER
    preds = Float64[]; ys = Bool[]
    for b in blocks
        tr = SH[SH.block .!= b, :]; te = SH[SH.block .== b, :]
        (isempty(tr) || isempty(te)) && continue
        m = fit_shot_xg(tr; features = feats)
        append!(preds, predict_xg(m, te; features = feats)); append!(ys, te.is_goal)
    end
    push!(res, (model = name, n = length(ys), brier = brier(preds, ys),
                logloss = logloss(preds, ys), mean_pred = mean(preds), mean_obs = mean(ys)))
end
L = DataFrame(res)
L.brier = round.(L.brier, digits = 5); L.logloss = round.(L.logloss, digits = 5)
L.mean_pred = round.(L.mean_pred, digits = 4); L.mean_obs = round.(L.mean_obs, digits = 4)
println(L)
@printf("\nm3 vs m0 Brier improvement: %.2f%%\n",
        100 * (L.brier[1] - L.brier[end]) / L.brier[1])
println("For scale, the base paper's COORDINATE-based open-play model improved on its baseline")
println("by 0.0866 → 0.0739 = 14.7%. We should land materially below that, not above it.")

# The winner, refit on everything.
BEST = (:zone, :body_part, :context)
XGM = fit_shot_xg(SH; features = BEST)
SH.xg = predict_xg(XGM, SH; features = BEST)
@printf("\nfull-sample model: base rate %.4f, penalty xG %.4f, %d cells\n",
        XGM.base_rate, XGM.penalty_xg, length(XGM.cells))
@printf("total xG %.0f vs total goals %d (ratio %.3f)\n",
        sum(SH.xg), sum(SH.is_goal), sum(SH.xg) / sum(SH.is_goal))

# ==========================================
# X4 — CALIBRATION
# ==========================================
_hdr("X4 — calibration deciles (out-of-sample predictions from X3's m3)")
oos_p = Float64[]; oos_y = Bool[]
for b in blocks
    tr = SH[SH.block .!= b, :]; te = SH[SH.block .== b, :]
    (isempty(tr) || isempty(te)) && continue
    m = fit_shot_xg(tr; features = BEST)
    append!(oos_p, predict_xg(m, te; features = BEST)); append!(oos_y, te.is_goal)
end
cal = DataFrame(p = oos_p, y = oos_y)
# Rank-based deciles: predictions are heavily tied (a cell model emits few distinct values), so
# quantile cut-points would collapse. Ranking spreads the ties across bins instead.
cal.dec = min.(10, floor.(Int, 10 .* ((sortperm(sortperm(cal.p)) .- 1) ./ nrow(cal))) .+ 1)
c4 = combine(groupby(cal, :dec), nrow => :n, :p => mean => :pred, :y => mean => :obs)
c4.pred = round.(c4.pred, digits = 4); c4.obs = round.(c4.obs, digits = 4)
c4.diff = round.(c4.obs .- c4.pred, digits = 4)
println(sort(c4, :dec))

# ==========================================
# X5 — TEAM-LEVEL CALIBRATION vs SOFASCORE   (primary gate: no name matching)
# ==========================================
_hdr("X5 — team xG vs SofaScore team xG (54/55)")
lu = PM_LINEUPS[]
# A handful of live_text rows carry a team slug that matches neither side (or none at all), so
# `is_home_event` is missing — those shots cannot be attributed and are excluded here. Reported
# below, because the same rows will be unusable for the WP4 segment targets.
@printf("shots with an unattributable side: %d (%.2f%%)\n",
        sum(ismissing.(SH.is_home)), 100 * mean(ismissing.(SH.is_home)))
SHA = SH[.!ismissing.(SH.is_home), :]
SHA.is_home = Bool.(SHA.is_home)

sofa = combine(groupby(lu[.!ismissing.(lu.expected_goals), :], [:match_id, :is_home_team]),
               :expected_goals => sum => :sofa_xg)
ours = combine(groupby(SHA, [:match_id, :is_home]), :xg => sum => :our_xg,
               :is_goal => sum => :goals, nrow => :shots)
rename!(sofa, :is_home_team => :side); rename!(ours, :is_home => :side)
cmp = innerjoin(sofa, ours, on = [:match_id, :side])
cmp = innerjoin(cmp, unique(select(lu, :match_id, :tournament_id, :season)), on = :match_id)
@printf("matched team-innings: %d over %d matches\n", nrow(cmp), length(unique(cmp.match_id)))
x5 = combine(groupby(cmp, [:tournament_id, :season]),
             nrow => :n,
             [:our_xg, :sofa_xg] => ((a, b) -> round(cor(a, b), digits = 3))            => :cor,
             [:our_xg, :sofa_xg] => ((a, b) -> round(mean(abs.(a .- b)), digits = 3))   => :mae,
             [:our_xg, :sofa_xg] => ((a, b) -> round(mean(a .- b), digits = 3))         => :bias,
             :our_xg  => (a -> round(mean(a), digits = 3)) => :mean_ours,
             :sofa_xg => (b -> round(mean(b), digits = 3)) => :mean_sofa)
println(sort(x5, [:tournament_id, :season]))
@printf("\nPOOLED: cor %.3f | MAE %.3f | bias %.3f\n",
        cor(cmp.our_xg, cmp.sofa_xg), mean(abs.(cmp.our_xg .- cmp.sofa_xg)),
        mean(cmp.our_xg .- cmp.sofa_xg))
println("Reference: bbc_xg_proxy's frozen team-level GLM reached Spearman ≈ 0.715, R² 0.442")
println("against the same SofaScore target — but from MATCH-AGGREGATE counts, not shot events.")

# ==========================================
# X6 — PLAYER-LEVEL + NAME MATCH RATE
# ==========================================
_hdr("X6 — player-level xG, and the BBC-name → player_id match rate")
lu2 = lu[.!ismissing.(lu.player_name), :]
lu2.key = string.(lu2.match_id, "|", strip_name.(lu2.player_name))
namemap = Dict(zip(lu2.key, lu2.player_id))

sh_named = SH[SH.shooter .!= "", :]
sh_named.key = string.(sh_named.match_id, "|", sh_named.shooter)
sh_named.pid = [get(namemap, k, missing) for k in sh_named.key]
@printf("shots with a shooter name: %d | resolved to a player_id: %.2f%%\n",
        nrow(sh_named), 100 * mean(.!ismissing.(sh_named.pid)))
byt = combine(groupby(sh_named, :tournament_id),
              nrow => :shots,
              :pid => (p -> round(100 * mean(.!ismissing.(p)), digits = 2)) => :pct_resolved)
println(byt)
println("\nThis is the number WP1 could not measure. It gates the tier-56 live_text hole-fill:")
println("that fallback resolves substitution NAMES to ids by exactly this mechanism.")

resolved = sh_named[.!ismissing.(sh_named.pid), :]
pxg = combine(groupby(resolved, [:match_id, :pid]), :xg => sum => :our_xg)
rename!(pxg, :pid => :player_id)
psofa = lu[.!ismissing.(lu.expected_goals), [:match_id, :player_id, :expected_goals]]
pj = innerjoin(pxg, psofa, on = [:match_id, :player_id])
if nrow(pj) > 0
    @printf("\nplayer-match pairs with both: %d | cor %.3f | MAE %.3f | bias %.3f\n",
            nrow(pj), cor(pj.our_xg, pj.expected_goals),
            mean(abs.(pj.our_xg .- pj.expected_goals)),
            mean(pj.our_xg .- pj.expected_goals))
end

# ==========================================
# X7 — TIER TRANSFER
# ==========================================
_hdr("X7 — tier transfer (does the model survive moving down a division?)")
function eval_fit(train_tiers, test_tiers)
    tr = SH[in.(SH.tournament_id, Ref(train_tiers)), :]
    te = SH[in.(SH.tournament_id, Ref(test_tiers)), :]
    (isempty(tr) || isempty(te)) && return (brier = NaN, logloss = NaN, n = 0)
    m = fit_shot_xg(tr; features = BEST)
    p = predict_xg(m, te; features = BEST)
    return (brier = brier(p, te.is_goal), logloss = logloss(p, te.is_goal), n = nrow(te))
end
tr_rows = NamedTuple[]
for (nm, a, b) in (("Prem→Champ", [54], [55]), ("Champ→Prem", [55], [54]),
                   ("Upper→L1",   [54,55], [56]), ("Upper→L2", [54,55], [57]),
                   ("in-sample Champ", [55], [55]), ("in-sample L1", [56], [56]))
    r = eval_fit(a, b)
    push!(tr_rows, (fit = nm, n = r.n, brier = round(r.brier, digits = 5),
                    logloss = round(r.logloss, digits = 5)))
end
println(DataFrame(tr_rows))
println("\nUpper→L1/L2 vs in-sample L1 is the gate that matters: it is exactly the operation we")
println("perform in production, and the two should be close.")

const XQA = (shots = SH, model = XGM, ladder = L, calib = c4, team = cmp, transfer = DataFrame(tr_rows))
_hdr("WP3 done — inspect `XQA`, then write the verdict into NOTES.md")
