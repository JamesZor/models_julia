# current_development/plus_minus_ratings/r01_segment_qa.jl
#
# RUNNER. WP2 — segment QA and the IDENTIFIABILITY GATE.
#
# This is the gate the whole stream turns on. RAPM works when the design matrix carries enough
# independent lineup variation to separate players. Our budget is thinner than the base paper's
# (they had ~130k segments / ~11k players ≈ 12:1, on leagues with more rotation than ours), so
# the honest question is not "is the ratio big enough" — nobody in the literature gives a
# threshold, see RESEARCH_rapm.md §4 — but "which players are actually estimable, and how many".
#
# GATES:
#   S1 rejects      — which matches could not be segmented, and why
#   S2 shape        — segments/match, durations, goals per segment, % of segments with y=0
#                     (the base paper's 72% is the number every denser target must beat)
#   S3 exposure     — segments and minutes per player; share below the 540-minute floor
#   S4 clusters     — EXACT always-together groups (identical design columns ⇒ ridge is
#                     mathematically forced to give them identical ratings)
#   S5 spectrum     — eigenvalues of XᵀWX: condition number, rank deficiency, effective degrees
#                     of freedom vs λ, and PER-PLAYER posterior variance. This is the real
#                     evidence; the ratio is just a headline.
#
# Run (redirect: the eigendecomposition takes a few minutes and the kaimon gate is cosmetic):
#   open(".../r01_out.txt","w") do io; redirect_stdout(io) do
#       include(".../r01_segment_qa.jl") end end

using DataFrames
using LinearAlgebra
using SparseArrays
using Statistics
using Printf

include(joinpath(@__DIR__, "l01_segments.jl"))

_hdr(s) = println("\n", "="^78, "\n", s, "\n", "="^78)

# ==========================================
# BUILD
# ==========================================
_hdr("Building segments over all four Scottish tiers")
@time SEG, REJ = build_segments()
@printf("segments: %d over %d matches | rejected matches: %d\n",
        nrow(SEG), length(unique(SEG.match_id)), nrow(REJ))

# ==========================================
# S1 — REJECTS
# ==========================================
_hdr("S1 — matches that could not be segmented")
if nrow(REJ) > 0
    s1 = combine(groupby(REJ, [:tournament_id, :reason]), nrow => :matches)
    println(unstack(s1, :tournament_id, :reason, :matches))
    println("\nBy season (the tier-56 incident holes should dominate `no_incidents`):")
    println(unstack(combine(groupby(REJ, [:season, :reason]), nrow => :n),
                    :season, :reason, :n))
end
println("\nNOTE: `no_incidents` is the cost of NOT wiring the BBC live_text fallback yet — WP1")
println("showed the text parses 100%, but name→player_id matching is still unmeasured risk.")

# ==========================================
# S2 — SEGMENT SHAPE
# ==========================================
_hdr("S2 — segment shape")
segs_per_match = combine(groupby(SEG, [:match_id, :tournament_id]), nrow => :n)
s2 = combine(groupby(segs_per_match, :tournament_id),
             nrow => :matches, :n => mean => :segs_per_match,
             :n => minimum => :min_segs, :n => maximum => :max_segs)
s2.segs_per_match = round.(s2.segs_per_match, digits = 2)
println(s2)

SEG.y = SEG.goals_home .- SEG.goals_away
@printf("\nduration (min): mean %.1f  median %.1f  p10 %.1f  p90 %.1f  |  <5 min: %.1f%%\n",
        mean(SEG.duration), median(SEG.duration),
        quantile(SEG.duration, 0.1), quantile(SEG.duration, 0.9),
        100 * mean(SEG.duration .< 5))
@printf("goals in segment: mean %.3f  |  segments with y = 0: %.1f%%   [base paper: 72%%]\n",
        mean(SEG.goals_home .+ SEG.goals_away), 100 * mean(SEG.y .== 0))
@printf("exposure: %.0f match-minutes total (%.1f matches' worth)\n",
        sum(SEG.duration), sum(SEG.duration) / 90)

garbage = (abs.(SEG.gd_start) .>= 2) .& (abs.(SEG.gd_end) .>= 2)
red_imb = (SEG.red_home .> 0) .⊻ (SEG.red_away .> 0)
@printf("garbage-time segments (start AND end ≥2 apart): %.1f%% of segments, %.1f%% of minutes\n",
        100 * mean(garbage), 100 * sum(SEG.duration[garbage]) / sum(SEG.duration))
@printf("segments with a manpower imbalance: %.1f%% of segments, %.1f%% of minutes\n",
        100 * mean(red_imb), 100 * sum(SEG.duration[red_imb]) / sum(SEG.duration))

# ==========================================
# S3 — PLAYER EXPOSURE
# ==========================================
_hdr("S3 — exposure per player")
EXP = player_exposure(SEG)
np = nrow(EXP); ns = nrow(SEG)
@printf("players: %d | segments: %d | ratio %.1f : 1   [base paper ≈ 12 : 1]\n",
        np, ns, ns / np)
for q in (0.1, 0.25, 0.5, 0.75, 0.9)
    @printf("  minutes q%-4.2f = %7.0f    segments q%-4.2f = %5.0f\n",
            q, quantile(EXP.minutes, q), q, quantile(EXP.n_segments, q))
end
@printf("\nbelow the 540-minute analysis floor : %5d players (%.1f%%)\n",
        sum(EXP.minutes .< 540), 100 * mean(EXP.minutes .< 540))
@printf("below the 900-minute top-N floor    : %5d players (%.1f%%)\n",
        sum(EXP.minutes .< 900), 100 * mean(EXP.minutes .< 900))
@printf("players appearing in >1 tier        : %5d (%.1f%%)  ← identify the league columns\n",
        sum(EXP.n_tiers .> 1), 100 * mean(EXP.n_tiers .> 1))
println("\nRatio above the 540-minute floor only:")
E540 = EXP[EXP.minutes .>= 540, :]
@printf("  players %d → ratio %.1f : 1\n", nrow(E540), ns / nrow(E540))

# ==========================================
# DESIGN MATRIX
# ==========================================
_hdr("Design matrix")
CS = competition_sets()
@time X, y, w, COLS = build_design(SEG; comp_sets = CS)
@printf("X: %d × %d, %d nonzeros (%.4f%% dense)\n",
        size(X, 1), size(X, 2), nnz(X), 100 * nnz(X) / prod(size(X)))
@printf("weights: mean %.3f  min %.4f  max %.3f\n", mean(w), minimum(w), maximum(w))

# ==========================================
# S4 — ALWAYS-TOGETHER CLUSTERS
# ==========================================
_hdr("S4 — exact always-together clusters (identical design columns)")
# Two players whose columns are elementwise identical are mathematically indistinguishable:
# ridge MUST assign them the same rating. This is the base paper's §6 caveat, measured.
Xc = X[:, 1:length(COLS.player_ids)]
sig = Dict{UInt64, Vector{Int}}()
rv = rowvals(Xc); nzv = nonzeros(Xc)
for j in 1:size(Xc, 2)
    r = nzrange(Xc, j)
    h = hash((view(rv, r), view(nzv, r)))
    push!(get!(sig, h, Int[]), j)
end
clusters = [v for v in values(sig) if length(v) > 1]
sort!(clusters, by = length, rev = true)
@printf("players in a tied cluster: %d of %d (%.1f%%) across %d clusters\n",
        sum(length, clusters; init = 0), size(Xc, 2),
        100 * sum(length, clusters; init = 0) / size(Xc, 2), length(clusters))
if !isempty(clusters)
    sizes = length.(clusters)
    println("cluster sizes: ", sort(unique(sizes)), "  (largest ", maximum(sizes), ")")
    println("\nMinutes of the players in the 5 largest clusters (tied ⇒ identical ratings):")
    emap = Dict(r.player_id => r.minutes for r in eachrow(EXP))
    for c in first(clusters, 5)
        mins = [get(emap, COLS.player_ids[j], 0.0) for j in c]
        @printf("  n=%d  minutes: %s\n", length(c), string(round.(mins, digits = 0)))
    end
end

# ==========================================
# S5 — SPECTRUM AND PER-PLAYER PRECISION
# ==========================================
_hdr("S5 — spectrum of XᵀWX, effective df, and per-player variance")
A = Matrix(Symmetric(Matrix(X' * Diagonal(w) * X)))
@printf("forming the %d × %d Gram matrix … ", size(A, 1), size(A, 2)); flush(stdout)
@time F = eigen(Symmetric(A))
d = max.(F.values, 0.0)                     # numerical negatives are zeros
@printf("eigenvalues: max %.4g  min %.4g\n", maximum(d), minimum(d))
@printf("rank-deficient directions (λ_i < 1e-8 · λ_max): %d of %d\n",
        sum(d .< 1e-8 * maximum(d)), length(d))
pos = d[d .> 1e-12 * maximum(d)]
@printf("condition number over the non-null space: %.4g\n", maximum(pos) / minimum(pos))

# Weights are normalised to mean 1 in `build_design`, so λ is on a scale set by the data volume
# and these numbers are comparable across weight configurations.
println("\neffective degrees of freedom  Σ dᵢ/(dᵢ+λ)   [ceiling = ", length(d), "]")
V2 = F.vectors .^ 2
npl = length(COLS.player_ids)
mins_v = let m = Dict(r.player_id => r.minutes for r in eachrow(EXP))
    [max(get(m, p, 1.0), 1.0) for p in COLS.player_ids]
end
lams = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
println(rpad("λ", 9), rpad("edf", 10), rpad("%params", 10), rpad("med postvar", 14),
        rpad("%prior-dominated", 18), "cor(log var, log min)")
for λ in lams
    edf = sum(d ./ (d .+ λ))
    pv  = (V2 * (1.0 ./ (d .+ λ)))[1:npl]
    @printf("%-9.2f%-10.1f%-10.1f%-14.4g%-18.1f%.3f\n",
            λ, edf, 100 * edf / length(d), median(pv),
            100 * mean(pv .>= 0.95 / λ), cor(log.(pv), log.(mins_v)))
end
println("\n`%prior-dominated` is the honest identifiability number: the share of players whose")
println("posterior variance is ≥95% of the prior variance 1/λ — players about whom the data says")
println("essentially NOTHING, whose rating is pure shrinkage. The last column should be strongly")
println("NEGATIVE (more minutes ⇒ better identified); if it is near zero, the variance has")
println("saturated at the prior for most players and λ is too large to learn anything.")

λ_diag = 1.0
dv10 = (V2 * (1.0 ./ (d .+ λ_diag)))[1:npl]
prec = DataFrame(player_id = COLS.player_ids, post_var = dv10)
prec = leftjoin(prec, select(EXP, :player_id, :minutes, :n_segments), on = :player_id)
sort!(prec, :post_var)
println("\nBest-identified players (λ=$λ_diag):");  println(first(prec, 5))
println("\nWorst-identified players (λ=$λ_diag):"); println(last(prec, 5))

const SQA = (segments = SEG, rejects = REJ, exposure = EXP, X = X, y = y, w = w,
             cols = COLS, eig = (values = d,), clusters = clusters, precision = prec)
_hdr("WP2 done — inspect `SQA`, then write the verdict into NOTES.md")
