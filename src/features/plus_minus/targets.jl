# src/features/plus_minus/targets.jl
#
# The four plus-minus responses, computed over the personnel segments.
#
#   y_goals  home-away goal difference            (base paper's PM; 72.1% of segments are 0)
#   y_shots  home-away shot difference            (the GREEN-LIT target; 34.3% zero)
#   y_sot    home-away shots-on-target difference (52.2% zero)
#   y_xg     home-away xG difference              (base paper's xGPM, via the zonal shot model)
#
# FAITHFUL PORT of the counting half of current_development/plus_minus_ratings/l03_targets.jl.
#
# `y_xp` (expected-points PM) is deliberately NOT ported. It was the weakest of the five targets in
# the research (Delta Brier -0.00319, last place), 0.881 correlated with `y_goals` (i.e. largely a
# game-state reweighting of goals rather than new information), and it needs an in-play hazard GLM
# plus a backward-induction table — a large dependency for the worst arm. See the stream NOTES.md.

using DataFrames

"""
    attach_shot_targets!(segments, shots, it1_by_match) -> segments

Adds `shots_h/a`, `sot_h/a`, `xg_h/a` per segment by placing each shot on the SAME clock the
segments were built on (`pm_clock`, with the first-half stoppage offset).

`covered` marks matches that have BBC live-text at all. Live text only starts in 23/24, so the
shot-based targets exist on a strictly smaller match set than the goals target.

⚠ CRITICAL: on a match with no live text the shot columns are 0 by construction, which is
indistinguishable from a genuine 0-0 shot difference. Fitting a shot-based target over uncovered
segments would feed the regression tens of thousands of fake zeros — so those targets MUST be
restricted to `segments.covered` (enforced in `fit_ratings`).
"""
function attach_shot_targets!(segments::DataFrame, shots::DataFrame,
                              it1_by_match::Dict{Int, Float64})
    for c in (:shots_h, :shots_a, :sot_h, :sot_a, :xg_h, :xg_a)
        segments[!, c] = zeros(Float64, nrow(segments))
    end
    segments.covered = falses(nrow(segments))
    nrow(shots) == 0 && return segments

    seg_by_match = Dict{Int, Vector{Int}}()
    for (i, mid) in enumerate(segments.match_id)
        push!(get!(seg_by_match, Int(mid), Int[]), i)
    end

    covered_matches = Set(Int.(shots.match_id))
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

"""
    add_targets!(segments, shots, it1_by_match) -> segments

Attaches the shot counts and materialises the four `y_*` response columns.
"""
function add_targets!(segments::DataFrame, shots::DataFrame, it1_by_match::Dict{Int, Float64})
    attach_shot_targets!(segments, shots, it1_by_match)
    segments.y_goals = Float64.(segments.goals_home .- segments.goals_away)
    segments.y_shots = segments.shots_h .- segments.shots_a
    segments.y_sot   = segments.sot_h   .- segments.sot_a
    segments.y_xg    = segments.xg_h    .- segments.xg_a
    return segments
end

# Targets that only exist on live-text-covered matches.
const PM_SHOT_TARGETS = (:y_shots, :y_sot, :y_xg)
const PM_TARGETS      = (:y_goals, :y_shots, :y_sot, :y_xg)
