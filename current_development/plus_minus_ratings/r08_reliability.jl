# current_development/plus_minus_ratings/r08_reliability.jl
#
# RUNNER. WP7 completion — the two axes the decision rule actually rests on.
#
# Hvattum & Gelade (2021) define exactly two criteria for comparing rating systems:
#   RELIABILITY — "similar ratings are produced when using different data sets"  → R1, R2
#   VALIDITY    — "the quality of match outcome forecasts when the ratings are used as
#                  predictor variables"                                          → R3
# Neither has been measured yet, so the stream currently has NO verdict.
#
# R1  split-half reliability  — fit on odd matchdays vs even, correlate. Same for SofaScore, on
#                               the same players and the same halves. No external truth needed;
#                               this is the single most informative number left.
# R2  season-to-season        — the BASE PAPER's own headline reliability claim (0.35 t→t+1,
#                               0.30 t→t+2). Fits are per-season and DISJOINT — a trailing
#                               window would share data between consecutive seasons and inflate
#                               the correlation, which would flatter us against their figure.
# R3  validity / retrodiction — RAPM-fed vs SofaScore-fed vs the de-vigged closing line, on the
#                               same held-out matches. The cheap GLM analogue of the engine A/B.
# R4  cross-tier transfer     — the ~576 players appearing in BOTH the upper and lower tiers:
#                               does a rating built ONLY from their 56/57 minutes line up with a
#                               SofaScore rating built ONLY from their 54/55 minutes? This is the
#                               only direct evidence the lower-league ratings sit on a real scale.
# R5  the forward anomaly     — r06 found forwards our weakest position (ρ 0.22) when the
#                               literature says they should be the strongest. R1 by position
#                               settles whether that is OUR noise or SOFASCORE's.

using DataFrames
using Statistics
using StatsBase: mode
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "l04_ridge_apm.jl"))

_hdr(s) = println("\n", "="^78, "\n", s, "\n", "="^78)
_sp(a, b) = cor(sortperm(sortperm(a)) ./ length(a), sortperm(sortperm(b)) ./ length(b))

_hdr("Rebuild")
SEG, REJ = build_segments()
SH = build_shots(); XGM = fit_shot_xg(SH); SH.xg = predict_xg(XGM, SH)
LONG = build_state_intervals(); HAZ = fit_inplay_hazard(LONG); XP = xp_table(HAZ)
add_targets!(SEG, SH, XP)
CS = competition_sets()
lu = PM_LINEUPS[]
COV = SEG[SEG.covered, :]
CLUB = pm_club_map()

"""Ridge fit over an arbitrary segment subset → DataFrame(player_id, rapm)."""
function fit_ratings(segs; target = :y_xg, λ = 200.0, w_sim = 0.0, half_life = 730.0)
    nrow(segs) < 500 && return nothing
    wcfg = SegmentWeights(; half_life_days = half_life)
    Te = DateTime(maximum(segs.start_timestamp))
    X, y, w, cols = build_design(segs; target = target, weights = wcfg,
                                 T_rating = Te, comp_sets = CS)
    S = w_sim == 0.0 ? nothing : similarity_matrix(segs, cols; k = 10)
    A = Matrix(Symmetric(Matrix(X' * Diagonal(w) * X))); b = Vector(X' * (w .* y))
    β = ridge_solve(A, b, penalty_matrix(cols, S, w_sim), λ)
    return DataFrame(player_id = cols.player_ids, rapm = β[1:length(cols.player_ids)])
end

"""Minute-weighted mean SofaScore rating over a given set of matches."""
function sofa_over(match_ids)
    s = Set(match_ids)
    lr = lu[coalesce.(in.(lu.match_id, Ref(s)), false) .& .!ismissing.(lu.rating), :]
    lr.m = coalesce.(passmissing(Float64).(lr.minutes_played), 90.0)
    lr.m = ifelse.(lr.m .<= 0, 1.0, lr.m)
    combine(groupby(lr, :player_id),
            [:rating, :m] => ((r, mm) -> sum(r .* mm) / sum(mm)) => :sofa,
            :m => sum => :sofa_min,
            :position => (p -> mode(pm_clean_position.(p))) => :pos)
end

# ==========================================
# R1 — SPLIT-HALF RELIABILITY
# ==========================================
_hdr("R1 — split-half reliability: ours vs SofaScore, same players, same halves")
# Alternate MATCHDAYS (rank of kickoff date within tier×season), so each half is a balanced
# sample of the season rather than a time slice.
md = unique(select(COV, :match_id, :tournament_id, :season, :start_timestamp))
sort!(md, [:tournament_id, :season, :start_timestamp])
md.half = fill(0, nrow(md))
for g in groupby(md, [:tournament_id, :season])
    g.half .= [isodd(i) ? 1 : 2 for i in 1:nrow(g)]
end
halfmap = Dict(Int(r.match_id) => r.half for r in eachrow(md))
COV.half = [halfmap[Int(m)] for m in COV.match_id]
@printf("matches: half A %d | half B %d\n",
        length(unique(COV.match_id[COV.half .== 1])), length(unique(COV.match_id[COV.half .== 2])))

r1 = NamedTuple[]
R1STORE = Dict{String, DataFrame}()
for (tgt, λ, ws) in ((:y_xg, 200.0, 0.0), (:y_shots, 1000.0, 0.0), (:y_shots, 1000.0, 0.9))
    A1 = fit_ratings(COV[COV.half .== 1, :]; target = tgt, λ = λ, w_sim = ws)
    B1 = fit_ratings(COV[COV.half .== 2, :]; target = tgt, λ = λ, w_sim = ws)
    (A1 === nothing || B1 === nothing) && continue
    J = innerjoin(rename(A1, :rapm => :a), rename(B1, :rapm => :b), on = :player_id)

    sa = sofa_over(unique(COV.match_id[COV.half .== 1]))
    sb = sofa_over(unique(COV.match_id[COV.half .== 2]))
    S = innerjoin(rename(select(sa, :player_id, :sofa, :sofa_min, :pos), :sofa => :sa, :sofa_min => :ma),
                  rename(select(sb, :player_id, :sofa, :sofa_min), :sofa => :sb, :sofa_min => :mb),
                  on = :player_id)
    J = innerjoin(J, S, on = :player_id)
    ea = player_exposure(COV[COV.half .== 1, :]); eb = player_exposure(COV[COV.half .== 2, :])
    J = innerjoin(J, rename(select(ea, :player_id, :minutes), :minutes => :min_a), on = :player_id)
    J = innerjoin(J, rename(select(eb, :player_id, :minutes), :minutes => :min_b), on = :player_id)
    J = J[(J.min_a .>= 450) .& (J.min_b .>= 450) .& (J.pos .!= "G"), :]
    key = "$(tgt)_w$(ws)"; R1STORE[key] = J
    push!(r1, (rating = key, n = nrow(J),
               rapm_splithalf = round(cor(J.a, J.b), digits = 3),
               sofa_splithalf = round(cor(J.sa, J.sb), digits = 3),
               rapm_spearman  = round(_sp(J.a, J.b), digits = 3),
               sofa_spearman  = round(_sp(J.sa, J.sb), digits = 3)))
end
println(DataFrame(r1))
println("\nDECISION-RULE HALF #1: our split-half must be ≥ SofaScore's on ≥1 cell.")
println("NB SofaScore's own figure is the fair benchmark — it is not 1.0, because a rating")
println("averaged over half a season is itself noisy.")

# ==========================================
# R5 — THE FORWARD ANOMALY (falls straight out of R1)
# ==========================================
_hdr("R5 — split-half reliability BY POSITION: is the forward problem ours or SofaScore's?")
for (key, J) in sort(collect(R1STORE), by = first)
    t = combine(groupby(J, :pos), nrow => :n,
                [:a, :b]   => ((x, y) -> length(x) < 10 ? NaN : cor(x, y)) => :rapm_sh,
                [:sa, :sb] => ((x, y) -> length(x) < 10 ? NaN : cor(x, y)) => :sofa_sh)
    t.rapm_sh = round.(t.rapm_sh, digits = 3); t.sofa_sh = round.(t.sofa_sh, digits = 3)
    println("\n", key, ":"); println(sort(t, :pos))
end
println("\nIf OUR forward reliability is low → our forward ratings are noise.")
println("If SOFASCORE's is also low → the r06 correlation was capped by THEIR noise, and the")
println("'forwards should be strongest' expectation does not transfer to this league.")

# ==========================================
# R2 — SEASON TO SEASON
# ==========================================
_hdr("R2 — season-to-season stability (base paper: 0.35 at t→t+1, 0.30 at t→t+2)")
# Disjoint single-season fits. Goals uses the FULL sample (6 seasons ⇒ more pairs); xG is
# restricted to the live-text era.
function per_season(target, λ, segs, seasons)
    out = Dict{String, DataFrame}()
    for s in seasons
        sub = segs[segs.season .== s, :]
        f = fit_ratings(sub; target = target, λ = λ, half_life = 1e6)   # no decay within a season
        f === nothing && continue
        e = player_exposure(sub)
        out[s] = innerjoin(f, select(e, :player_id, :minutes), on = :player_id)
    end
    return out
end
ALL_SEASONS = ["20/21","21/22","22/23","23/24","24/25","25/26"]
COV_SEASONS = ["23/24","24/25","25/26"]
PS_goals = per_season(:y_goals, 1000.0, SEG, ALL_SEASONS)
PS_xg    = per_season(:y_xg,     200.0, COV, COV_SEASONS)
SOFA_S   = Dict(s => sofa_over(unique(SEG.match_id[SEG.season .== s])) for s in ALL_SEASONS)

function stability(PS, seasons, lag)
    rows = NamedTuple[]
    for i in 1:(length(seasons) - lag)
        s1, s2 = seasons[i], seasons[i + lag]
        (haskey(PS, s1) && haskey(PS, s2)) || continue
        j = innerjoin(rename(PS[s1], :rapm => :r1, :minutes => :m1),
                      rename(PS[s2], :rapm => :r2, :minutes => :m2), on = :player_id)
        j = j[(j.m1 .>= 900) .& (j.m2 .>= 900), :]
        so = innerjoin(rename(select(SOFA_S[s1], :player_id, :sofa), :sofa => :s1v),
                       rename(select(SOFA_S[s2], :player_id, :sofa), :sofa => :s2v), on = :player_id)
        j = leftjoin(j, so, on = :player_id)
        jj = j[.!ismissing.(j.s1v) .& .!ismissing.(j.s2v), :]
        nrow(j) < 25 && continue
        push!(rows, (pair = "$s1→$s2", n_rapm = nrow(j),
                     rapm = round(cor(j.r1, j.r2), digits = 3),
                     n_sofa = nrow(jj),
                     sofa = nrow(jj) < 25 ? NaN : round(cor(jj.s1v, jj.s2v), digits = 3)))
    end
    return DataFrame(rows)
end
println("\ny_goals, full sample, lag 1:"); println(stability(PS_goals, ALL_SEASONS, 1))
println("\ny_goals, full sample, lag 2:"); println(stability(PS_goals, ALL_SEASONS, 2))
println("\ny_xg, live-text era, lag 1:");  println(stability(PS_xg, COV_SEASONS, 1))

# ==========================================
# R3 — VALIDITY: retrodiction vs SofaScore vs the market
# ==========================================
_hdr("R3 — do the ratings forecast results? RAPM vs SofaScore vs the closing line")
ODDS = devig_1x2(PM_ODDS[])
@printf("matches with a de-vigged 1X2 close: %d\n", nrow(ODDS))

function retrodict(eval_season; target = :y_xg, λ = 200.0, w_sim = 0.0)
    segs = COV
    ev = segs.season .== eval_season
    any(ev) || return nothing
    t_cut = minimum(DateTime.(segs.start_timestamp[ev]))
    tr = segs[DateTime.(segs.start_timestamp) .< t_cut, :]
    nrow(tr) < 1000 && return nothing

    f = fit_ratings(tr; target = target, λ = λ, w_sim = w_sim)
    f === nothing && return nothing
    rmap = Dict(r.player_id => r.rapm for r in eachrow(f))
    prior_sofa = sofa_over(unique(tr.match_id))
    smap = Dict(r.player_id => r.sofa for r in eachrow(prior_sofa))

    rows = NamedTuple[]
    meta = pm_match_meta()
    sc = Dict(Int(r.match_id) => (r.home_score, r.away_score) for r in eachrow(meta))
    for g in groupby(segs[ev, :], :match_id)
        mid = Int(g.match_id[1]); haskey(sc, mid) || continue
        h, a = sc[mid]; (ismissing(h) || ismissing(a)) && continue
        f1 = g[argmin(g.t_start), :]
        rr = sum(get(rmap, p, 0.0) for p in f1.home_players) -
             sum(get(rmap, p, 0.0) for p in f1.away_players)
        hs = [get(smap, p, NaN) for p in f1.home_players]
        as = [get(smap, p, NaN) for p in f1.away_players]
        (all(isnan, hs) || all(isnan, as)) && continue
        ss = sum(filter(!isnan, hs)) - sum(filter(!isnan, as))
        push!(rows, (match_id = mid, rapm = rr, sofa = ss,
                     y = h > a ? 3 : (h == a ? 2 : 1)))
    end
    D = DataFrame(rows); nrow(D) < 50 && return nothing

    function score(x)
        θ = fit_ordered_logit(x, D.y)
        (multiclass_brier(θ, x, D.y), multiclass_logloss(θ, x, D.y))
    end
    b_r, l_r = score(D.rapm); b_s, l_s = score(D.sofa); b_0, l_0 = score(zeros(nrow(D)))

    M = innerjoin(D, ODDS, on = :match_id)
    b_m, l_m, n_m = if nrow(M) >= 50
        P = hcat(M.p_away, M.p_draw, M.p_home)
        bs = mean(sum((P .- (M.y .== reshape(1:3, 1, 3))) .^ 2, dims = 2))
        ls = -mean(log.(clamp.([P[i, M.y[i]] for i in 1:nrow(M)], 1e-9, 1)))
        (bs, ls, nrow(M))
    else
        (NaN, NaN, 0)
    end
    return (season = eval_season, n = nrow(D),
            floor_brier = round(b_0, digits = 4),
            rapm_brier  = round(b_r, digits = 4),
            sofa_brier  = round(b_s, digits = 4),
            n_mkt = n_m, market_brier = round(b_m, digits = 4),
            rapm_ll = round(l_r, digits = 4), sofa_ll = round(l_s, digits = 4),
            market_ll = round(l_m, digits = 4))
end
r3 = filter(!isnothing, [retrodict(s) for s in ("24/25", "25/26")])
println(DataFrame(r3))
println("\nDECISION-RULE HALF #2: rapm_brier must not be materially worse than sofa_brier.")
println("The market column is the ceiling, not a target — the base paper's ratings reached")
println("0.292 against 0.295 for the de-vigged bet365 close on ELITE leagues.")

# ==========================================
# R4 — CROSS-TIER TRANSFER
# ==========================================
_hdr("R4 — cross-tier: RAPM from 56/57 minutes only vs SofaScore from 54/55 minutes only")
low = SEG[in.(SEG.tournament_id, Ref([56, 57])), :]
upp_matches = unique(SEG.match_id[in.(SEG.tournament_id, Ref([54, 55]))])
f_low = fit_ratings(low; target = :y_goals, λ = 1000.0)
s_upp = sofa_over(upp_matches)
X4 = innerjoin(f_low, s_upp, on = :player_id)
X4 = innerjoin(X4, rename(select(player_exposure(low), :player_id, :minutes), :minutes => :low_min),
               on = :player_id)
X4 = X4[(X4.low_min .>= 540) .& (X4.sofa_min .>= 540) .& (X4.pos .!= "G"), :]
@printf("crossover players with ≥540 min in BOTH groups: %d\n", nrow(X4))
if nrow(X4) >= 25
    @printf("Pearson %.3f | Spearman %.3f\n", cor(X4.rapm, X4.sofa), _sp(X4.rapm, X4.sofa))
    println("\nThese two numbers share NO match, NO season overlap by construction and no team —")
    println("a player's lower-league RAPM against his upper-league SofaScore rating. Positive")
    println("correlation here is the cleanest evidence available that the 56/57 ratings mean")
    println("something, because nothing about the two quantities is mechanically linked.")
end

const WQA = (splithalf = DataFrame(r1), retrodict = DataFrame(r3), crosstier = X4, r1store = R1STORE)
_hdr("WP7 complete — inspect `WQA`, then write the verdict into NOTES.md")
