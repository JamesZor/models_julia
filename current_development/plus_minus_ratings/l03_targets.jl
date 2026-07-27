# current_development/plus_minus_ratings/l03_targets.jl
#
# LOADER (temporary module). WP4 — the four plus-minus targets over the WP2 segments.
#
#   y_goals  home−away goal difference            (base paper's PM; 72.2% of segments are 0)
#   y_shots  home−away shot difference            (repo-specific; the bbc_xg_proxy funnel lesson)
#   y_sot    home−away shots-on-target difference
#   y_xg     home−away xG difference              (base paper's xGPM, via the WP3 zonal model)
#   y_xp     home−away change in expected points  (base paper's xPPM)
#
# The first four are counting exercises once shots are attributed to segments. `y_xp` needs an
# in-play win/draw/loss model, built here as a self-contained reduced form.
#
# WHY NOT REUSE `inplay_scottish`: the plan authorises falling back to a local reduced form if
# that stream's NHPP artefact is not ready (it is mid-WP2/WP3 in its own NOTES). Blocking three
# working targets on another stream's unfinished work would be the wrong trade. The form here
# follows the same shape its research settled on — observable covariates, global (no team
# hierarchy), linear game state, time bins.
#
# THE TEAM-STRENGTH-BLIND RULE (base paper §4.2, load-bearing): the in-play model deliberately
# does NOT condition on team strength. If it did, we would be crediting players for playing on a
# good team — double-counting the very thing the plus-minus regression is trying to estimate.
# So the intensities depend only on game state (time, goal difference, manpower) and home/away.

using DataFrames
using Statistics
using GLM
using StatsModels

include(joinpath(@__DIR__, "l01_segments.jl"))
include(joinpath(@__DIR__, "l02_shot_parser.jl"))

# ==========================================
# 1. ATTRIBUTE SHOTS TO SEGMENTS
# ==========================================
"""
    attach_shot_targets!(segments, shots) -> segments

Adds `shots_h/a`, `sot_h/a`, `xg_h/a` per segment by placing each shot on the SAME clock the
segments were built on (`pm_clock`, with the first-half stoppage offset).

Shots that cannot be attributed to a side are dropped — WP3 measured this at 2.44%.

`covered` marks matches that have live_text at all: tiers 54/55 only have it from 23/24, so the
shot-based targets exist on a strictly smaller match set than the goals target. Every downstream
comparison MUST be run on the common subset, or the denser targets get flattered by also being
the more recent ones.
"""
function attach_shot_targets!(segments::DataFrame, shots::DataFrame)
    for c in (:shots_h, :shots_a, :sot_h, :sot_a, :xg_h, :xg_a)
        segments[!, c] = zeros(Float64, nrow(segments))
    end
    segments.covered = falses(nrow(segments))

    it1_by_match = Dict{Int, Float64}()
    for r in eachrow(pm_match_meta())
        it1_by_match[Int(r.match_id)] = _num(r.injury_time1)
    end

    seg_by_match = Dict{Int, Vector{Int}}()
    for (i, r) in enumerate(eachrow(segments))
        push!(get!(seg_by_match, Int(r.match_id), Int[]), i)
    end

    covered_matches = Set(shots.match_id)
    for (mid, idxs) in seg_by_match
        mid in covered_matches || continue
        for i in idxs; segments.covered[i] = true; end
    end

    for sh in eachrow(shots)
        ismissing(sh.is_home) && continue
        mid = Int(sh.match_id)
        idxs = get(seg_by_match, mid, nothing); idxs === nothing && continue
        t = pm_clock(sh.time, sh.added_time, get(it1_by_match, mid, 0.0))
        isnan(t) && continue

        # Half-open [t_start, t_end); the last segment takes anything at or past its end so
        # stoppage-time shots are never lost off the back of the match.
        hit = 0
        for i in idxs
            if t >= segments.t_start[i] && t < segments.t_end[i]
                hit = i; break
            end
        end
        if hit == 0
            last_i = idxs[argmax([segments.t_end[i] for i in idxs])]
            t >= segments.t_end[last_i] && (hit = last_i)
        end
        hit == 0 && continue

        home = sh.is_home === true
        segments[hit, home ? :shots_h : :shots_a] += 1.0
        sh.is_on_target && (segments[hit, home ? :sot_h : :sot_a] += 1.0)
        segments[hit, home ? :xg_h : :xg_a] += sh.xg
    end
    return segments
end

# ==========================================
# 2. IN-PLAY HAZARD  (for xPPM)
# ==========================================
"""
    build_state_intervals(; tournaments, seasons) -> DataFrame

Intervals of constant GAME STATE — split on goals and dismissals, but NOT on substitutions,
because the hazard model is team-strength- and personnel-blind. One row per (interval × side)
with the goals that side scored and the exposure in minutes.

Note this is a different decomposition from the plus-minus segments: those split on personnel
and ignore goals; these split on goals and ignore personnel.
"""
function build_state_intervals(; tournaments::Vector{Int} = PM_TIERS)
    ensure_pm_data!()
    lu  = PM_LINEUPS[]; inc = PM_INCIDENTS[]
    keep = coalesce.(in.(lu.tournament_id, Ref(tournaments)), false)
    meta = unique(select(lu[keep, :], :match_id, :injury_time1, :injury_time2))
    valid = Set(meta.match_id)

    rows = NamedTuple[]
    for g in groupby(inc, :match_id)
        mid = Int(g.match_id[1]); mid in valid || continue
        mrow = meta[findfirst(==(mid), meta.match_id), :]
        it1 = _num(mrow.injury_time1); it2 = _num(mrow.injury_time2)
        T = pm_match_length(it1, it2)

        evs = NamedTuple[]
        for r in eachrow(g)
            t = pm_clock(r.time, r.added_time, it1); isnan(t) && continue
            t = clamp(t, 0.0, T)
            if r.incident_type == "goal"
                push!(evs, (t = t, kind = :goal, home = r.is_home === true))
            elseif r.incident_type == "card" && !ismissing(r.incident_class) &&
                   String(r.incident_class) in DISMISSAL_CLASSES
                push!(evs, (t = t, kind = :red, home = r.is_home === true))
            end
        end
        sort!(evs, by = e -> e.t)

        gd = 0; mp = 0; t0 = 0.0
        function emit!(t1)
            d = t1 - t0
            d <= 0 && return
            push!(rows, (match_id = mid, t_start = t0, duration = d, gd = gd, mp = mp,
                         goals_h = 0, goals_a = 0))
            t0 = t1
        end
        for e in evs
            emit!(e.t)
            if e.kind === :goal
                e.home ? (gd += 1) : (gd -= 1)
                # credit the goal to the interval that just closed
                if !isempty(rows) && rows[end].match_id == mid
                    r = rows[end]
                    rows[end] = merge(r, e.home ? (goals_h = r.goals_h + 1,) :
                                                  (goals_a = r.goals_a + 1,))
                end
            else
                e.home ? (mp -= 1) : (mp += 1)
            end
        end
        emit!(T)
    end

    iv = DataFrame(rows)
    # long form: one row per (interval, side)
    long = vcat(
        DataFrame(match_id = iv.match_id, t_start = iv.t_start, duration = iv.duration,
                  gd = iv.gd, mp = iv.mp, goals = iv.goals_h, is_home = true),
        DataFrame(match_id = iv.match_id, t_start = iv.t_start, duration = iv.duration,
                  gd = iv.gd, mp = iv.mp, goals = iv.goals_a, is_home = false))
    # from the scoring side's perspective
    long.gd_own = ifelse.(long.is_home, long.gd, .-long.gd)
    long.mp_own = ifelse.(long.is_home, long.mp, .-long.mp)
    long.gd_c   = clamp.(long.gd_own, -3, 3)
    long.mp_c   = clamp.(long.mp_own, -3, 3)
    long.tbin   = string.(min.(5, floor.(Int, long.t_start ./ 15)))
    long.logexp = log.(max.(long.duration, 1e-3) ./ 90)
    return long
end

"""
    fit_inplay_hazard(long) -> StatsModels model

Poisson log-link with a log-exposure offset. Goal difference enters as a FACTOR (the effect of
being 1 up is not half the effect of being 2 up), manpower linearly, plus time bins and a home
term. Global — no team parameters, by design (see the team-strength-blind rule above).
"""
function fit_inplay_hazard(long::DataFrame)
    d = copy(long)
    d.gd_f = string.(d.gd_c)
    return glm(@formula(goals ~ tbin + gd_f + mp_c + is_home), d, Poisson(), LogLink();
               offset = d.logexp)
end

"""
    xp_table(model; T=96.0, dt=0.5, gd_range=-6:6, mp_range=-3:3) -> Array

Backward induction over the match clock giving `P(home win), P(draw), P(away win)` from any
state `(t, gd, mp)`. Returns `V[ti, gdi, mpi, outcome]`.

Manpower is held fixed over the remaining match — the same simplification the base paper makes
by treating manpower as a covariate evaluated at time t. Future dismissals are unforecastable
here and rare enough not to distort expected points materially.
"""
function xp_table(model; T::Float64 = 96.0, dt::Float64 = 0.5,
                  gd_range = -6:6, mp_range = -3:3)
    nt = Int(ceil(T / dt)) + 1
    ngd = length(gd_range); nmp = length(mp_range)
    gd_i(g) = clamp(g - first(gd_range) + 1, 1, ngd)

    # λ(t, gd, mp, side) on the grid
    grid = DataFrame(tbin = String[], gd_f = String[], mp_c = Int[], is_home = Bool[],
                     goals = Int[], logexp = Float64[])
    tbins = 0:5
    for tb in tbins, g in gd_range, m in mp_range, h in (true, false)
        push!(grid, (string(tb), string(clamp(g, -3, 3)), clamp(m, -3, 3), h, 0, 0.0))
    end
    # The model carries a log(duration/90) offset, so predicting at offset 0 returns the rate
    # per 90 minutes; dividing by 90 below gives the per-minute intensity the DP needs.
    grid.rate = predict(model, grid; offset = zeros(nrow(grid)))

    key = Dict{Tuple{Int,Int,Int,Bool}, Float64}()
    for r in eachrow(grid)
        key[(parse(Int, r.tbin), parse(Int, r.gd_f), r.mp_c, r.is_home)] = r.rate
    end
    rate(tb, g, m, h) = key[(tb, clamp(g, -3, 3), clamp(m, -3, 3), h)] / 90.0  # per minute

    V = zeros(Float64, nt, ngd, nmp, 3)
    for (gi, g) in enumerate(gd_range), mi in 1:nmp
        V[nt, gi, mi, :] .= (g > 0 ? (1.0, 0.0, 0.0) : g == 0 ? (0.0, 1.0, 0.0) : (0.0, 0.0, 1.0))
    end
    for ti in (nt - 1):-1:1
        t  = (ti - 1) * dt
        tb = min(5, floor(Int, t / 15))
        for (mi, m) in enumerate(mp_range), (gi, g) in enumerate(gd_range)
            ph = rate(tb, g, m, true)  * dt
            pa = rate(tb, g, m, false) * dt
            p0 = max(0.0, 1.0 - ph - pa)
            for o in 1:3
                V[ti, gi, mi, o] = p0 * V[ti + 1, gi, mi, o] +
                                   ph * V[ti + 1, gd_i(g + 1), mi, o] +
                                   pa * V[ti + 1, gd_i(g - 1), mi, o]
            end
        end
    end
    return (V = V, dt = dt, T = T, gd_range = gd_range, mp_range = mp_range)
end

"""
    expected_points(tab, t, gd, mp) -> (xp_home, xp_away)

`xP = 3·P(win) + 1·P(draw)` for each side.
"""
function expected_points(tab, t::Float64, gd::Int, mp::Int)
    ti = clamp(Int(round(t / tab.dt)) + 1, 1, size(tab.V, 1))
    gi = clamp(gd - first(tab.gd_range) + 1, 1, length(tab.gd_range))
    mi = clamp(mp - first(tab.mp_range) + 1, 1, length(tab.mp_range))
    pw, pd, pl = tab.V[ti, gi, mi, 1], tab.V[ti, gi, mi, 2], tab.V[ti, gi, mi, 3]
    return (3pw + pd, 3pl + pd)
end

# ==========================================
# 3. ASSEMBLE ALL FIVE TARGETS
# ==========================================
"""
    add_targets!(segments, shots, xp) -> segments

`y_xp` follows the base paper exactly: `(xP_h(end) − xP_h(start)) − (xP_a(end) − xP_a(start))`,
evaluated at the segment boundaries with the goal difference and manpower prevailing there.
"""
function add_targets!(segments::DataFrame, shots::DataFrame, xp)
    attach_shot_targets!(segments, shots)
    segments.y_goals = Float64.(segments.goals_home .- segments.goals_away)
    segments.y_shots = segments.shots_h .- segments.shots_a
    segments.y_sot   = segments.sot_h   .- segments.sot_a
    segments.y_xg    = segments.xg_h    .- segments.xg_a

    y_xp = Vector{Float64}(undef, nrow(segments))
    for (i, s) in enumerate(eachrow(segments))
        mp = s.red_away - s.red_home           # + means home has the manpower advantage
        h0, a0 = expected_points(xp, s.t_start, s.gd_start, mp)
        h1, a1 = expected_points(xp, s.t_end,   s.gd_end,   mp)
        y_xp[i] = (h1 - h0) - (a1 - a0)
    end
    segments.y_xp = y_xp
    return segments
end

const TARGETS = [:y_goals, :y_shots, :y_sot, :y_xg, :y_xp]
