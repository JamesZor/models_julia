# current_development/plus_minus_ratings/r06_vs_sofascore.jl
#
# RUNNER. WP7 (first pass) — how do our RAPM ratings relate to the SofaScore player rating?
#
# READ THE EXPECTATION BEFORE THE RESULT, or you will misread it. Gelade & Hvattum (2020)
# measured how much of plus-minus variance bottom-up event statistics explain: **22%–38% by
# position** (goalkeepers lowest, forwards highest). The SofaScore rating IS a bottom-up
# event-based rating. So the target band is:
#
#     ρ ≈ 0.47 – 0.62, ordered GK < midfielders < forwards
#
#   * ρ ≈ 0        ⇒ our rating is broken, or pure noise.
#   * ρ ≈ 0.5      ⇒ EXACTLY RIGHT. This is what a working top-down rating looks like against a
#                    bottom-up one — they measure genuinely different things.
#   * ρ ≈ 0.9      ⇒ BAD. We would not have built a top-down rating; we would have rebuilt the
#                    box score, and could have saved ourselves the trouble.
#
# So a middling correlation is the SUCCESS criterion here, not the failure criterion.
#
# ALSO TESTED — the team-strength pitfall flagged at the end of WP5. `w_SIM` shrinks each player
# toward his most frequent teammates, which improves match-outcome Brier *precisely by making
# ratings more team-like*. Here we measure that directly: what share of rating variance is
# explained by a team fixed effect, and how much within-team spread survives. A rating that is
# 95% team is not a player rating.

using DataFrames
using Statistics
using StatsBase: mode
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "l04_ridge_apm.jl"))

_hdr(s) = println("\n", "="^78, "\n", s, "\n", "="^78)

_spearman(a, b) = cor(sortperm(sortperm(a)) ./ length(a), sortperm(sortperm(b)) ./ length(b))

_hdr("Rebuilding segments and targets")
SEG, REJ = build_segments()
SH = build_shots(); XGM = fit_shot_xg(SH); SH.xg = predict_xg(XGM, SH)
LONG = build_state_intervals(); HAZ = fit_inplay_hazard(LONG); XP = xp_table(HAZ)
add_targets!(SEG, SH, XP)
CS = competition_sets()
COV = SEG[SEG.covered, :]

# ==========================================
# THE SOFASCORE YARDSTICK
# ==========================================
# Minute-weighted mean rating per player, over the SAME window the RAPM is fit on. Ratings only
# exist for tiers 54/55 (and 55 only from 23/24), which defines the comparison population.
lu = PM_LINEUPS[]
covered_matches = Set(COV.match_id)
lr = lu[coalesce.(in.(lu.match_id, Ref(covered_matches)), false) .& .!ismissing.(lu.rating), :]
lr.mins = coalesce.(passmissing(Float64).(lr.minutes_played), 90.0)
lr.mins = ifelse.(lr.mins .<= 0, 1.0, lr.mins)

SOFA = combine(groupby(lr, :player_id),
               [:rating, :mins] => ((r, m) -> sum(r .* m) / sum(m)) => :sofa_rating,
               :mins => sum => :sofa_minutes,
               nrow => :n_rated,
               :team_id => (t -> mode(collect(skipmissing(t)))) => :team_id,
               :position => (p -> mode(pm_clean_position.(p))) => :pos)
@printf("players with a SofaScore rating in the fitted window: %d\n", nrow(SOFA))
@printf("SofaScore rating: mean %.3f sd %.3f range [%.2f, %.2f]\n",
        mean(SOFA.sofa_rating), std(SOFA.sofa_rating),
        minimum(SOFA.sofa_rating), maximum(SOFA.sofa_rating))

# ==========================================
# FIT RAPM AT SEVERAL CELLS
# ==========================================
# NOT just WP5's Brier-optimal cell. WP5 flagged that the tuning criterion rewards recovering
# team strength, so we look across w_SIM rather than inheriting the winner.
CELLS = [(:y_shots, 1000.0, 0.0), (:y_shots, 1000.0, 0.5), (:y_shots, 1000.0, 0.9),
         (:y_xg,     200.0, 0.0), (:y_xg,     200.0, 0.5), (:y_xg,     200.0, 0.75),
         (:y_goals, 1000.0, 0.0), (:y_goals, 1000.0, 0.9)]

T_end = DateTime(maximum(COV.start_timestamp))
rows = NamedTuple[]; RATINGS = Dict{String, DataFrame}()

for (tgt, λ, ws) in CELLS
    wcfg = SegmentWeights(; half_life_days = 730.0)
    X, y, w, cols = build_design(COV; target = tgt, weights = wcfg,
                                 T_rating = T_end, comp_sets = CS)
    S = ws == 0.0 ? nothing : similarity_matrix(COV, cols; k = 10)
    A = Matrix(Symmetric(Matrix(X' * Diagonal(w) * X))); b = Vector(X' * (w .* y))
    β = ridge_solve(A, b, penalty_matrix(cols, S, ws), λ)

    R = DataFrame(player_id = cols.player_ids, rapm = β[1:length(cols.player_ids)])
    R = innerjoin(R, SOFA, on = :player_id)
    R = innerjoin(R, select(player_exposure(COV), :player_id, :minutes), on = :player_id)
    R = R[R.minutes .>= 540, :]            # the analysis floor from RESEARCH_rapm.md §2.2
    key = "$(tgt)_w$(ws)"
    RATINGS[key] = R

    # How much of the rating is TEAM? R² of a team fixed effect.
    grp = groupby(R, :team_id)
    ss_tot = sum((R.rapm .- mean(R.rapm)) .^ 2)
    ss_res = sum(sum((g.rapm .- mean(g.rapm)) .^ 2) for g in grp)
    team_r2 = 1 - ss_res / ss_tot

    push!(rows, (target = String(tgt), w_sim = ws, n = nrow(R),
                 pearson  = cor(R.rapm, R.sofa_rating),
                 spearman = _spearman(R.rapm, R.sofa_rating),
                 team_r2  = team_r2,
                 sd_rapm  = std(R.rapm),
                 within_team_sd = sqrt(ss_res / nrow(R))))
end

_hdr("Correlation with the SofaScore rating (players ≥540 min)")
C = DataFrame(rows)
for c in (:pearson, :spearman, :team_r2, :sd_rapm, :within_team_sd)
    C[!, c] = round.(C[!, c], digits = 3)
end
println(C)
println("\nTARGET BAND from Gelade & Hvattum (2020): ρ ≈ 0.47–0.62.")
println("`team_r2` is the share of rating variance explained by a team fixed effect — the")
println("WP5 pitfall made measurable. A rating that is nearly all team is not a player rating.")

# ==========================================
# BY POSITION  (the ordering is the real test)
# ==========================================
_hdr("By position group — expected ordering GK < D/M < F")
for key in ("y_shots_w0.0", "y_xg_w0.0", "y_shots_w0.9")
    haskey(RATINGS, key) || continue
    R = RATINGS[key]
    t = combine(groupby(R, :pos), nrow => :n,
                [:rapm, :sofa_rating] => ((a, b) -> length(a) < 8 ? NaN : cor(a, b)) => :pearson,
                [:rapm, :sofa_rating] => ((a, b) -> length(a) < 8 ? NaN : _spearman(a, b)) => :spearman)
    t.pearson = round.(t.pearson, digits = 3); t.spearman = round.(t.spearman, digits = 3)
    println("\n", key, ":"); println(sort(t, :pos))
end

# ==========================================
# IS THE RELATIONSHIP MONOTONE?
# ==========================================
_hdr("Shape: mean RAPM by SofaScore decile (the closest thing to a calibration curve)")
println("Two ratings on different scales cannot be 'calibrated' to each other — what matters is")
println("whether the relationship is MONOTONE and how much RAPM spread survives within a decile.")
for key in ("y_shots_w0.0", "y_xg_w0.0")
    haskey(RATINGS, key) || continue
    R = copy(RATINGS[key])
    R.dec = min.(10, floor.(Int, 10 .* ((sortperm(sortperm(R.sofa_rating)) .- 1) ./ nrow(R))) .+ 1)
    t = combine(groupby(R, :dec), nrow => :n,
                :sofa_rating => mean => :sofa, :rapm => mean => :rapm_mean,
                :rapm => std => :rapm_sd)
    for c in (:sofa, :rapm_mean, :rapm_sd); t[!, c] = round.(t[!, c], digits = 3); end
    println("\n", key, ":"); println(sort(t, :dec))
end

# ==========================================
# WHO DOES EACH SYSTEM LIKE THAT THE OTHER DOESN'T?
# ==========================================
_hdr("Biggest disagreements (y_xg, w_SIM=0, ≥900 min)")
R = RATINGS["y_xg_w0.0"]; R = R[R.minutes .>= 900, :]
R.z_rapm = (R.rapm .- mean(R.rapm)) ./ std(R.rapm)
R.z_sofa = (R.sofa_rating .- mean(R.sofa_rating)) ./ std(R.sofa_rating)
R.gap = R.z_rapm .- R.z_sofa
nm = Dict{Int,String}()
for r in eachrow(lu); ismissing(r.player_name) || (nm[Int(r.player_id)] = String(r.player_name)); end
R.name = [get(nm, p, "?") for p in R.player_id]
show_cols = [:name, :pos, :rapm, :sofa_rating, :z_rapm, :z_sofa, :minutes]
println("RAPM likes far more than SofaScore does:")
println(first(sort(R, :gap, rev = true), 8)[:, show_cols])
println("\nSofaScore likes far more than RAPM does:")
println(first(sort(R, :gap), 8)[:, show_cols])

const VQA = (corr = C, ratings = RATINGS, sofa = SOFA)
_hdr("WP7 first pass done — inspect `VQA`")
