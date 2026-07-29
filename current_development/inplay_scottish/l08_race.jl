#=
l08_race.jl — WP-F loader: one interface over every race arm, so the arms differ ONLY in
what is being tested.

THE ARMS (arm 3 is absent — WP-E was skipped, see NOTES):

  | arm       | timeline           | clock / exposure          | model              |
  |-----------|--------------------|---------------------------|--------------------|
  | 0a        | ds.incidents       | l01 (Tend 95, flat)       | incumbent goals    |
  | 0b_naive  | BBC                | l01 (Tend 95, flat)       | incumbent goals    |
  | 0b_clock  | BBC                | BBC (Tend 90, stoppage)   | incumbent goals    |
  | mvp1      | BBC                | BBC                       | shot-flow + p2     |
  | mvp2      | BBC                | BBC                       | nowcast (surplus)  |

**Why 0b is SPLIT.** The plan's arm 0b exists to separate "BBC as a better data source" from
"BBC as a new model". But WP-A found BBC also carries a better CLOCK (real stoppage; 88 of
2,965 goals change half), so an undivided 0b would confound coverage with clock — the same
mistake arm 0b exists to prevent. `0b_naive` isolates coverage; `0b_clock` isolates the clock.

**The common currency is the COMPOSED BOOK, not the slice likelihood.** MVP-1 counts shots,
so its slice log-likelihood is not comparable to a goals-NHPP's. Every arm does produce a
final-score matrix P_t, so all arms are scored on realised OUTCOMES through that matrix —
per market family, never aggregated across families (the Double Chance defect reversed a
headline in the APM stream by exactly that route).

Requires l01–l07 to be included first.
=#

using DataFrames, Statistics, Random, Distributions

# ---------------------------------------------------------------------------
# 1. Arm definition
# ---------------------------------------------------------------------------

"""
    RaceArm(name, kind, chain, config, mseqs, draws)

`kind` selects the kernel route:
  `:goals_flat` — l01's integrator, Tend 95, flat Δt exposure (arms 0a, 0b_naive)
  `:goals_expo` — same terms, Tend 90, per-match stoppage exposure (arm 0b_clock)
  `:shots`      — shot intensity thinned by p2 (mvp1)
  `:nowcast`    — goals plus the shot-surplus term (mvp2)
"""
struct RaceArm
    name::String
    kind::Symbol
    chain::Any
    config::Any
    mseqs::Vector{NamedTuple}
    draws::Any
end

by_mid(arm::RaceArm) = Dict(m.mid => m for m in arm.mseqs)

# `remaining_intensity_expo` now lives in l09_ingame.jl (the in-game model owns the
# stoppage-aware integrator); include l09 before using the :goals_expo arm.

"""
    arm_lambdas_and_kernels(arm, ms, t_now, gh, ga, rh, ra)
        -> (λh_draws, λa_draws, K_h, K_a)

Returns whatever pair multiplies to the remaining GOAL intensity, so that
`compose_score_matrix(λh, λa, K_h, K_a)` is identical across arms. For the shot arm the
pregame vector is the SHOT intensity and the kernel absorbs `p2` — Poisson thinning makes
that exactly equivalent to a goals model, which is why no new score-matrix type is needed.
"""
function arm_lambdas_and_kernels(arm::RaceArm, ms, t_now, gh, ga, rh, ra)
    d = arm.draws[ms.mid]
    if arm.kind === :goals_flat
        K_h, K_a = remaining_intensity(arm.chain, arm.config; pg_h = 1.0, pg_a = 1.0,
            gh = gh, ga = ga, reds_h = rh, reds_a = ra, t_now = t_now)
        return d.λ_h, d.λ_a, K_h, K_a
    elseif arm.kind === :goals_expo
        K_h, K_a = remaining_intensity_expo(arm.chain, arm.config; pg_h = 1.0, pg_a = 1.0,
            gh = gh, ga = ga, reds_h = rh, reds_a = ra, t_now = t_now,
            at1 = ms.at1, at2 = ms.at2)
        return d.λ_h, d.λ_a, K_h, K_a
    elseif arm.kind === :shots
        K_h, K_a = remaining_shot_intensity(arm.chain, arm.config; pg_s_h = 1.0, pg_s_a = 1.0,
            gh = gh, ga = ga, reds_h = rh, reds_a = ra, t_now = t_now,
            at1 = ms.at1, at2 = ms.at2)
        p2 = d.p2 === nothing ? fill(ms.p2, length(K_h)) : mean(d.p2)
        return d.λ_s_h, d.λ_s_a, K_h .* mean(p2), K_a .* mean(p2)
    elseif arm.kind === :nowcast
        sh = count(s ->  s.home && s.t < t_now, ms.shots)
        sa = count(s -> !s.home && s.t < t_now, ms.shots)
        K_h, K_a = remaining_intensity_nowcast(arm.chain, arm.config; pg_h = 1.0, pg_a = 1.0,
            λs_h = ms.λsh, λs_a = ms.λsa, shots_h = sh, shots_a = sa,
            gh = gh, ga = ga, reds_h = rh, reds_a = ra, t_now = t_now,
            at1 = ms.at1, at2 = ms.at2)
        return d.λ_h, d.λ_a, K_h, K_a
    end
    error("unknown arm kind $(arm.kind)")
end

"""
    arm_matrix(arm, ms, t_now; gh, ga, rh, ra, n_pairs, G) -> P̄ (G×G, sums to 1)

The composed mean final-score matrix — the single object every arm is scored through.
"""
function arm_matrix(arm::RaceArm, ms, t_now; gh, ga, rh, ra,
                    n_pairs::Int = 800, G::Int = 13)
    λh, λa, K_h, K_a = arm_lambdas_and_kernels(arm, ms, t_now, gh, ga, rh, ra)
    S = compose_score_matrix(λh, λa, K_h, K_a; gh = gh, ga = ga, n_pairs = n_pairs,
                             max_goals = G - 1,
                             rng = Xoshiro(ms.mid + Int(round(t_now * 10))))
    dropdims(mean(S.data; dims = 3); dims = 3)
end

"Match state at `t_now` from an arm's own event sequences."
function state_at(ms, t_now)
    (gh = count(g ->  g.home && g.t < t_now, ms.goals),
     ga = count(g -> !g.home && g.t < t_now, ms.goals),
     rh = count(c ->  c.home && c.t < t_now, ms.reds),
     ra = count(c -> !c.home && c.t < t_now, ms.reds))
end

# ---------------------------------------------------------------------------
# 2. Outcome scoring — per market family, match-clustered
# ---------------------------------------------------------------------------

const RACE_SELS = vcat([:home, :draw, :away, :btts_yes, :btts_no],
                       [Symbol("over_$(k)5") for k in 0:3],
                       [Symbol("under_$(k)5") for k in 0:3])

family(sel::Symbol) = sel in (:home, :draw, :away) ? :x12 :
                      sel in (:btts_yes, :btts_no) ? :btts : :ou

"Realised truth for the scored selections, from the FINAL score."
function truth_of(fh::Int, fa::Int)
    t = Dict{Symbol, Int}(:home => Int(fh > fa), :draw => Int(fh == fa),
                          :away => Int(fh < fa),
                          :btts_yes => Int(fh > 0 && fa > 0),
                          :btts_no => Int(!(fh > 0 && fa > 0)))
    for k in 0:3
        t[Symbol("over_$(k)5")]  = Int(fh + fa > k)
        t[Symbol("under_$(k)5")] = Int(fh + fa <= k)
    end
    return t
end

"""
    score_arm(arm, mids, checkpoints; n_pairs, G) -> DataFrame

One row per (match, checkpoint, selection): composed probability and realised outcome.
Scored later per FAMILY — never pooled across families.
"""
function score_arm(arm::RaceArm, mids, checkpoints = (0.0, 30.0, 60.0, 80.0);
                   n_pairs::Int = 800, G::Int = 13)
    idx = by_mid(arm)
    rows = DataFrame(mid = Int[], t0 = Float64[], sel = Symbol[],
                     p = Float64[], y = Int[])
    for mid in mids
        haskey(idx, mid) || continue
        ms = idx[mid]
        fh = count(g -> g.home, ms.goals); fa = count(g -> !g.home, ms.goals)
        tr = truth_of(fh, fa)
        for t0 in checkpoints
            st = state_at(ms, t0)
            P̄ = arm_matrix(arm, ms, t0; gh = st.gh, ga = st.ga, rh = st.rh, ra = st.ra,
                           n_pairs = n_pairs, G = G)
            for sel in RACE_SELS
                push!(rows, (mid, t0, sel, sum(P̄[cells_for(sel, G)]), tr[sel]))
            end
        end
    end
    rows.family = family.(rows.sel)
    return rows
end

_ll(p, y) = y * log(clamp(p, 1e-9, 1)) + (1 - y) * log(clamp(1 - p, 1e-9, 1))

"Per-family Brier / logloss, plus the per-MATCH mean logloss used for the clustered test."
function family_scores(rows::DataFrame)
    rows = copy(rows); rows.ll = _ll.(rows.p, rows.y)
    agg = combine(groupby(rows, [:t0, :family]), nrow => :n,
        [:p, :y] => ((p, y) -> mean((p .- y) .^ 2)) => :brier,
        :ll => (x -> -mean(x)) => :logloss,
        [:p, :y] => ((p, y) -> mean(p .- y)) => :bias)
    per_match = combine(groupby(rows, [:t0, :family, :mid]), :ll => mean => :ll)
    return sort(agg, [:t0, :family]), per_match
end

"""
    clustered_delta(pm_a, pm_b) -> DataFrame

Paired difference in per-match mean log-likelihood, **clustered by match** (n = matches,
not slices and not folds). The fold-paired t used elsewhere in this stream is inflated
~2.4–3×; see l07. Positive `mean` = arm A better.
"""
function clustered_delta(pm_a::DataFrame, pm_b::DataFrame)
    j = innerjoin(pm_a, pm_b, on = [:t0, :family, :mid], makeunique = true)
    combine(groupby(j, [:t0, :family])) do g
        d = g.ll .- g.ll_1
        se = std(d) / sqrt(length(d))
        (n = length(d), mean = mean(d), se = se, t = mean(d) / se)
    end
end
