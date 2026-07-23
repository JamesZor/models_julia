# current_development/plus_minus_ratings/r04_ridge_fit.jl
#
# RUNNER. WP5 — tune and fit the reference ridge RAPM.
#
# STAGES:
#   R0 floor    — what Brier does a NO-RATING model get? Every number below is meaningless
#                 without this. (An ordered logit with the strength coefficient pinned to 0,
#                 i.e. league base rates only.)
#   R1 grid     — λ × half-life at w_SIM = 0, all five targets, on a COMMON set of evaluation
#                 seasons so the comparison is like-for-like.
#   R2 fairness — the shot-based targets only exist on 52.2% of segments (WP4/T6). Refit the
#                 GOALS target on that same subset, so xG is not credited for being the more
#                 recent data.
#   R3 w_SIM    — teammate-similarity shrinkage at the best (λ, half-life) per target.
#                 This is RESEARCH_rapm.md §1.1's headline low-minutes treatment.
#   R4 final    — fit the winner, compare the red-card / home-advantage coefficients to the base
#                 paper's Table 4 (a specification check: wrong signs ⇒ the segment builder is
#                 wrong), and look at who the ratings actually like.
#
# Runtime is dominated by 2,700×2,700 Cholesky solves; expect tens of minutes. Redirect stdout.

using DataFrames
using Statistics
using Printf
using LinearAlgebra

include(joinpath(@__DIR__, "l04_ridge_apm.jl"))

_hdr(s) = println("\n", "="^78, "\n", s, "\n", "="^78)

_hdr("Building segments, shots and targets")
@time SEG, REJ = build_segments()
@time SH = build_shots()
XGM = fit_shot_xg(SH); SH.xg = predict_xg(XGM, SH)
LONG = build_state_intervals(); HAZ = fit_inplay_hazard(LONG); XP = xp_table(HAZ)
add_targets!(SEG, SH, XP)
CS = competition_sets()
@printf("segments %d (covered %d) | matches %d\n",
        nrow(SEG), sum(SEG.covered), length(unique(SEG.match_id)))

# Evaluation seasons: the shot targets need training data, which only exists from 23/24, so
# 24/25 and 25/26 are the only seasons every arm can be scored on. Use them for ALL arms.
const EVAL_SEASONS = ["24/25", "25/26"]
println("evaluation seasons (common to every arm): ", EVAL_SEASONS)

# ==========================================
# R0 — THE NO-RATING FLOOR
# ==========================================
_hdr("R0 — floor: what does a model with NO rating information score?")
"""Match outcomes alone, no ratings involved — the input to the no-information floor."""
function match_outcomes(segments)
    meta = pm_match_meta()
    sc = Dict(Int(r.match_id) => (r.home_score, r.away_score) for r in eachrow(meta))
    rows = NamedTuple[]
    for g in groupby(segments, :match_id)
        mid = Int(g.match_id[1])
        haskey(sc, mid) || continue
        h, a = sc[mid]
        (ismissing(h) || ismissing(a)) && continue
        push!(rows, (match_id = mid, season = String(g.season[1]),
                     y = h > a ? 3 : (h == a ? 2 : 1)))
    end
    return DataFrame(rows)
end

let rows = NamedTuple[], ms = match_outcomes(SEG)
    for s in EVAL_SEASONS
        ms_tr = ms[ms.season .!= s, :]; ms_ev = ms[ms.season .== s, :]
        (nrow(ms_tr) < 100 || nrow(ms_ev) < 50) && continue
        θ = fit_ordered_logit(zeros(nrow(ms_tr)), ms_tr.y)   # strength ≡ 0 ⇒ base rates only
        push!(rows, (season = s, n = nrow(ms_ev),
                     brier = multiclass_brier(θ, zeros(nrow(ms_ev)), ms_ev.y),
                     logloss = multiclass_logloss(θ, zeros(nrow(ms_ev)), ms_ev.y)))
    end
    F = DataFrame(rows)
    println(F)
    global FLOOR_BRIER = sum(F.brier .* F.n) / sum(F.n)
    global FLOOR_LL    = sum(F.logloss .* F.n) / sum(F.n)
    @printf("\nFLOOR: Brier %.5f | LogLoss %.5f  — every arm below must beat this.\n",
            FLOOR_BRIER, FLOOR_LL)
    println("(The base paper's tuned model reached Brier 0.292 on EPL, against 0.295 for the")
    println("de-vigged bet365 close. A lower league is less predictable, so expect worse.)")
end

# ==========================================
# R1 — λ × HALF-LIFE GRID
# ==========================================
_hdr("R1 — λ × half-life grid at w_SIM = 0")
LAMBDAS = [0.1, 0.5, 1.0, 5.0, 20.0]
HALFLIVES = [180.0, 365.0, 730.0]
@time G1 = sweep(SEG, EVAL_SEASONS; targets = [:y_goals, :y_sot, :y_shots, :y_xg, :y_xp],
                 lambdas = LAMBDAS, half_lives = HALFLIVES, comp_sets = CS)
G1.d_brier = round.(G1.brier .- FLOOR_BRIER, digits = 5)
for c in (:brier, :logloss); G1[!, c] = round.(G1[!, c], digits = 5); end
for c in (:ha, :red1, :sd_players); G1[!, c] = round.(G1[!, c], digits = 4); end
sort!(G1, :brier)
println(first(G1, 20))

println("\nBest cell per target (negative d_brier = beats the no-rating floor):")
best = combine(groupby(G1, :target), sdf -> first(sort(sdf, :brier), 1))
println(sort(best, :brier))

# ==========================================
# R2 — FAIRNESS: goals refit on the covered subset
# ==========================================
_hdr("R2 — goals on the SAME 52.2% subset the shot targets live on")
@time G2 = sweep(SEG, EVAL_SEASONS; targets = [:y_goals], lambdas = LAMBDAS,
                 half_lives = HALFLIVES, comp_sets = CS, covered_only = true,
                 label = "_cov")
G2.d_brier = round.(G2.brier .- FLOOR_BRIER, digits = 5)
for c in (:brier, :logloss); G2[!, c] = round.(G2[!, c], digits = 5); end
println(first(sort(G2, :brier), 5))
println("\nCompare this row against y_xg / y_sot / y_shots above — THAT is the honest test of")
println("whether a denser target beats goals, with sample size held constant.")

# ==========================================
# R3 — TEAMMATE-SIMILARITY SHRINKAGE
# ==========================================
_hdr("R3 — w_SIM sweep at each target's best (λ, half-life)")
W_SIMS = [0.0, 0.25, 0.5, 0.75, 0.9]
rows3 = DataFrame()
for r in eachrow(best)
    tgt = Symbol(r.target)
    g = sweep(SEG, EVAL_SEASONS; targets = [tgt], lambdas = [r.lambda],
              half_lives = [r.half_life], w_sims = W_SIMS, comp_sets = CS)
    rows3 = vcat(rows3, g)
end
rows3.d_brier = round.(rows3.brier .- FLOOR_BRIER, digits = 5)
for c in (:brier, :logloss); rows3[!, c] = round.(rows3[!, c], digits = 5); end
for c in (:ha, :red1, :sd_players); rows3[!, c] = round.(rows3[!, c], digits = 4); end
println(sort(rows3, [:target, :w_sim]))
println("\nw_SIM = 0 is plain ridge. If the tuned value does not beat it, the teammate prior is")
println("not earning its place on THIS data and we say so — RESEARCH_rapm.md §2.1 already warns")
println("that informed priors buy less in football than the basketball literature claims.")

# ==========================================
# R4 — FINAL FIT AND SPECIFICATION CHECK
# ==========================================
_hdr("R4 — final fit on all data")
WIN = first(sort(rows3, :brier), 1)[1, :]
@printf("winner: target=%s λ=%.2f half_life=%.0f w_SIM=%.2f (Brier %.5f, floor %.5f)\n",
        WIN.target, WIN.lambda, WIN.half_life, WIN.w_sim, WIN.brier, FLOOR_BRIER)

wt = Symbol(replace(String(WIN.target), "_cov" => ""))
# The winner is fit on the covered subset if it is a shot-based target, or if it is the
# "_cov"-labelled goals arm from R2.
use_covered = (wt in SHOT_TARGETS) || occursin("_cov", String(WIN.target))
FIT_SEG = use_covered ? SEG[SEG.covered, :] : SEG
println("fitting on ", nrow(FIT_SEG), " segments (covered_only = ", use_covered, ")")
wcfg = SegmentWeights(; half_life_days = WIN.half_life)
T_end = DateTime(maximum(FIT_SEG.start_timestamp))
X, y, w, COLS = build_design(FIT_SEG; target = wt, weights = wcfg, T_rating = T_end, comp_sets = CS)
S = WIN.w_sim == 0.0 ? nothing : similarity_matrix(FIT_SEG, COLS; k = 10)
A = Matrix(Symmetric(Matrix(X' * Diagonal(w) * X))); b = Vector(X' * (w .* y))
@time BETA = ridge_solve(A, b, penalty_matrix(COLS, S, WIN.w_sim), WIN.lambda)

_hdr("Specification check vs the base paper's Table 4")
@printf("home advantage : %+.4f      [paper: +0.006 PM, +0.005 xGPM, +0.0004 xPPM]\n", BETA[COLS.ha])
for k in 1:3
    @printf("red card %d     : %+.4f      [paper: %s]\n", k, BETA[COLS.reds[k]],
            ("-1.25", "-0.16", "-0.012")[k])
end
println("\nSigns matter more than magnitudes (our target scale differs). A red card must be")
println("clearly NEGATIVE and monotone-decreasing in severity; home advantage small positive.")
println("A wrong sign here means the segment builder is wrong, not the ridge.")

println("\nleague coefficients (tier strength, identified only via cross-tier players):")
for (i, l) in enumerate(COLS.league_ids)
    @printf("  tier %d : %+.4f\n", l, BETA[COLS.leagues[i]])
end

_hdr("Ratings")
EXP = player_exposure(FIT_SEG)
R = DataFrame(player_id = COLS.player_ids, rating = BETA[1:length(COLS.player_ids)])
R = innerjoin(R, EXP, on = :player_id)
names_map = Dict{Int, String}()
for r in eachrow(PM_LINEUPS[]); ismissing(r.player_name) || (names_map[Int(r.player_id)] = String(r.player_name)); end
R.name = [get(names_map, p, "?") for p in R.player_id]
@printf("ratings: sd %.4f | range [%.3f, %.3f]\n", std(R.rating), minimum(R.rating), maximum(R.rating))
@printf("cor(rating, log minutes) = %.3f   ← should be SMALL; a large value means the rating is\n",
        cor(R.rating, log.(max.(R.minutes, 1.0))))
println("   mostly measuring playing time (selection), not contribution.")

R900 = R[R.minutes .>= 900, :]
println("\nTop 15 (≥900 minutes, the base paper's top-N floor):")
println(first(sort(R900, :rating, rev = true), 15)[:, [:name, :rating, :minutes, :n_matches, :n_tiers]])
println("\nBottom 10 (≥900 minutes):")
println(first(sort(R900, :rating), 10)[:, [:name, :rating, :minutes, :n_matches, :n_tiers]])

const RQA = (grid = G1, grid_cov = G2, wsim = rows3, winner = WIN, beta = BETA,
             cols = COLS, ratings = R, floor_brier = FLOOR_BRIER)
_hdr("WP5 done — inspect `RQA`, then write the verdict into NOTES.md")
