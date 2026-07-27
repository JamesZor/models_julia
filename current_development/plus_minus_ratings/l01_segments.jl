# current_development/plus_minus_ratings/l01_segments.jl
#
# LOADER (temporary module). WP2 — segments and the sparse design matrix.
#
# A SEGMENT is a maximal interval of a match during which the set of players on the pitch does
# not change. A new segment starts at kickoff, at every substitution, and at every dismissal.
# Goals do NOT start a new segment (the personnel are unchanged) but they are tracked, because
# the score at the start and end of a segment drives the garbage-time weight.
#
# WHAT THIS FILE PRODUCES
#   build_segments(...)  -> DataFrame, one row per segment, with the on-pitch player vectors
#   build_design(...)    -> (X::SparseMatrixCSC, y, w, cols) ready for the ridge solve in l04
#
# SPEC SOURCES. The base paper is Kharrat, López Peña & McHale (2019) §3.2/§4.3, but WP0 found a
# more developed formulation — Hvattum, Arntzen & Pantuso (2020), Appl. Sci. 10(20) 7345 — and
# RESEARCH_rapm.md §6 adopted five changes from it. The ones that land here:
#
#   * Segment weight is a PRODUCT of three factors, not just time decay:
#         w = w_time · w_duration · w_goals
#     with w_duration downweighting short (noisy) segments and w_goals discounting garbage time
#     (segments that both start AND end ≥2 goals apart).
#   * The design row is scaled by d/90, so coefficients read as contributions per 90 minutes,
#     and the response stays a RAW goal count. This is strictly better than dividing the
#     response by d/90, which would inflate the variance of short segments enormously.
#   * League adjustment is Hvattum's per-player competition average (1/|C_p|)·Σ_{c∈C_p} β_c,
#     not the base paper's "adapted to competition" counting rule. Better suited to our
#     four-tier pool, where 576 of 3,192 players cross between upper and lower tiers.
#   * Home advantage and the dismissal dummies are NOT penalised (enforced in l04, flagged here
#     via `cols.penalised`).
#
# Downstream: `r01_segment_qa.jl` (the WP2 identifiability gate), then `l03_targets.jl` swaps in
# the other three targets over the SAME segments.

using DataFrames
using Dates
using SparseArrays
using Statistics

include(joinpath(@__DIR__, "l00_pm_data.jl"))

# ==========================================
# 1. THE MATCH CLOCK
# ==========================================
# SofaScore reports an event at 45+2 as `time=45, added_time=2`, and one at 90+4 as
# `time=90, added_time=4`. Naively adding them is NOT monotone: a 45+2 event would land at 47,
# *after* the 46th minute of the second half. Half-time substitutions are common, so this is a
# real ordering bug, not a hypothetical one. Offsetting everything after half-time by the
# first-half stoppage restores monotonicity.
"""
    pm_clock(time, added_time, it1) -> Float64

Monotone match minute. `it1` is first-half stoppage time (`matches.injury_time1`).
"""
function pm_clock(time, added_time, it1::Float64)
    ismissing(time) && return NaN
    t  = Float64(time)
    at = ismissing(added_time) ? 0.0 : Float64(added_time)
    return t + at + (t > 45.0 ? it1 : 0.0)
end

_num(x, default = 0.0) = (ismissing(x) || !isfinite(Float64(x))) ? default : Float64(x)

"""
    pm_match_length(meta_row) -> Float64

Total playable minutes: 90 plus both stoppage periods when recorded. Segments are measured on
this clock, so a match with 6 minutes of stoppage genuinely contributes 96 minutes of exposure.
"""
pm_match_length(it1::Float64, it2::Float64) = 90.0 + it1 + it2

# ==========================================
# 2. PER-MATCH SEGMENT CONSTRUCTION
# ==========================================
"""
Outcome of trying to segment one match. `ok=false` rows are excluded from the design but kept
so `r01` can report exactly why, rather than silently shrinking the sample.
"""
struct MatchSegmentation
    match_id::Int
    ok::Bool
    reason::String
    segments::Vector{NamedTuple}
end

const DISMISSAL_CLASSES = ("red", "yellowRed")

"""
    segment_match(lu, inc, it1, it2; max_reds=3) -> MatchSegmentation

Walk one match's incidents, maintaining the on-pitch sets, the score and the dismissal counts.

Rejection reasons (each is a real data defect seen in WP1's G2/G5):
  * `starters_ne_11`   — teamsheet does not have exactly 11 starters per side (WP1 found
                         teamsheets averaging 9.2 players on the matches whose substitution ids
                         failed to resolve — an incomplete bench scrape; the whole match is
                         unusable, not just the player).
  * `sub_in_unknown`   — a player entered who is not on the teamsheet.
  * `sub_out_off`      — a player left who was not on the pitch.
  * `no_incidents`     — nothing to segment with.
"""
function segment_match(lu::AbstractDataFrame, inc::Union{Nothing, AbstractDataFrame},
                       it1::Float64, it2::Float64; max_reds::Int = 3)
    mid = Int(lu.match_id[1])
    T   = pm_match_length(it1, it2)

    home_start = Int[]; away_start = Int[]; sheet = Set{Int}()
    for r in eachrow(lu)
        ismissing(r.player_id) && continue
        pid = Int(r.player_id)
        push!(sheet, pid)
        r.is_starter === true || continue
        (r.is_home_team === true ? home_start : away_start) |> v -> push!(v, pid)
    end
    if length(home_start) != 11 || length(away_start) != 11
        return MatchSegmentation(mid, false, "starters_ne_11", NamedTuple[])
    end
    inc === nothing && return MatchSegmentation(mid, false, "no_incidents", NamedTuple[])

    # --- collect and order the events -------------------------------------------------
    evs = NamedTuple[]
    for r in eachrow(inc)
        t = pm_clock(r.time, r.added_time, it1)
        isnan(t) && continue
        t = clamp(t, 0.0, T)
        if r.incident_type == "substitution"
            (ismissing(r.player_in_id) || ismissing(r.player_out_id)) && continue
            push!(evs, (t = t, kind = :sub, home = r.is_home === true,
                        pin = Int(r.player_in_id), pout = Int(r.player_out_id)))
        elseif r.incident_type == "card" && !ismissing(r.incident_class) &&
               String(r.incident_class) in DISMISSAL_CLASSES
            ismissing(r.player_id) && continue
            push!(evs, (t = t, kind = :red, home = r.is_home === true,
                        pin = 0, pout = Int(r.player_id)))
        elseif r.incident_type == "goal"
            push!(evs, (t = t, kind = :goal, home = r.is_home === true, pin = 0, pout = 0))
        end
    end
    sort!(evs, by = e -> (e.t, e.kind === :goal ? 0 : 1))   # goals resolve before personnel

    # --- walk ---------------------------------------------------------------------------
    on_h = Set(home_start); on_a = Set(away_start)
    red_h = 0; red_a = 0
    gh = 0; ga = 0                    # running score
    seg_gh = 0; seg_ga = 0            # goals inside the current segment
    t_open = 0.0
    gd_open = 0
    segs = NamedTuple[]

    function close_segment!(t_close)
        d = t_close - t_open
        d <= 0.0 && return                       # simultaneous events: no interval between them
        push!(segs, (match_id = mid,
                     t_start = t_open, t_end = t_close, duration = d,
                     home_players = sort(collect(on_h)), away_players = sort(collect(on_a)),
                     red_home = red_h, red_away = red_a,
                     goals_home = seg_gh, goals_away = seg_ga,
                     gd_start = gd_open, gd_end = gh - ga))
        t_open = t_close; gd_open = gh - ga; seg_gh = 0; seg_ga = 0
    end

    i = 1
    while i <= length(evs)
        t = evs[i].t
        # Take ALL events at this instant together — a triple substitution at 60' is one
        # boundary, not three zero-length segments.
        j = i
        while j <= length(evs) && evs[j].t == t
            j += 1
        end
        batch = @view evs[i:j-1]

        # Goals at this instant belong to the segment that is closing.
        for e in batch
            e.kind === :goal || continue
            e.home ? (gh += 1; seg_gh += 1) : (ga += 1; seg_ga += 1)
        end
        if any(e -> e.kind !== :goal, batch)
            close_segment!(t)
            for e in batch
                e.kind === :goal && continue
                on, other = e.home ? (on_h, on_a) : (on_a, on_h)
                if e.kind === :sub
                    e.pout in on || return MatchSegmentation(mid, false, "sub_out_off", NamedTuple[])
                    e.pin in sheet || return MatchSegmentation(mid, false, "sub_in_unknown", NamedTuple[])
                    delete!(on, e.pout); push!(on, e.pin)
                else  # :red
                    delete!(on, e.pout)
                    e.home ? (red_h += 1) : (red_a += 1)
                end
            end
        end
        i = j
    end
    close_segment!(T)

    isempty(segs) && return MatchSegmentation(mid, false, "no_incidents", NamedTuple[])
    return MatchSegmentation(mid, true, "ok", segs)
end

# ==========================================
# 3. ALL MATCHES
# ==========================================
"""
    build_segments(; tournaments, seasons=nothing, max_reds=3) -> (segments, rejects)

`segments` is one row per segment with match metadata attached; `rejects` is a per-match table
of why a match was dropped. Matches without incidents are rejected here — the BBC live_text
fallback for tier 56's holes is deliberately NOT wired in yet, because WP1 flagged that its
name→player_id matching is an unmeasured risk. `r01` reports the cost of that choice.
"""
function build_segments(; tournaments::Vector{Int} = PM_TIERS,
                          seasons::Union{Nothing, Vector{String}} = nothing,
                          max_reds::Int = 3)
    ensure_pm_data!()
    lu  = PM_LINEUPS[]
    inc = PM_INCIDENTS[]

    keep = in.(lu.tournament_id, Ref(tournaments))
    seasons === nothing || (keep = keep .& in.(lu.season, Ref(seasons)))
    lu = lu[coalesce.(keep, false), :]

    inc_idx = Dict{Int, SubDataFrame}()
    for g in groupby(inc, :match_id)
        inc_idx[Int(g.match_id[1])] = g
    end

    rows = NamedTuple[]; rejects = NamedTuple[]
    for g in groupby(lu, :match_id)
        mid  = Int(g.match_id[1])
        it1  = _num(g.injury_time1[1]); it2 = _num(g.injury_time2[1])
        res  = segment_match(g, get(inc_idx, mid, nothing), it1, it2; max_reds = max_reds)
        meta = (tournament_id = Int(g.tournament_id[1]), season = String(g.season[1]),
                start_timestamp = g.start_timestamp[1])
        if !res.ok
            push!(rejects, (match_id = mid, reason = res.reason, meta...))
            continue
        end
        for s in res.segments
            push!(rows, merge(s, meta))
        end
    end
    return DataFrame(rows), DataFrame(rejects)
end

# ==========================================
# 4. WEIGHTS
# ==========================================
"""
    SegmentWeights(; half_life_days, rho2, rho3, dur_pow, rho4)

`w = w_time · w_duration · w_goals` (Hvattum et al. 2020 §3.1; RESEARCH_rapm.md §1.2).

  * `w_time     = 0.5 ^ ((T_rating − t_match) / half_life_days)` — the base paper's ζ,
    reparameterised as a half-life to match this repo's convention (hl365 etc.).
  * `w_duration = ((d + rho2) / rho3) ^ dur_pow` — a 3-minute segment is nearly pure noise.
    `dur_pow = 0` disables it. NOTE the sign convention: Hvattum weight *increases* with
    duration, which is a modelling choice to emphasise long segments, NOT variance-optimal
    weighting — with a raw-count response the WLS-optimal weight would go the other way. Left
    as a tunable exponent so WP5 can settle it empirically rather than by assertion.
  * `w_goals    = rho4` when the segment starts AND ends ≥2 goals apart, else 1 — garbage time.
"""
Base.@kwdef struct SegmentWeights
    half_life_days::Float64 = 365.0
    rho2::Float64           = 0.0
    rho3::Float64           = 90.0
    dur_pow::Float64        = 1.0
    rho4::Float64           = 0.5
    gd_threshold::Int       = 2
end

function segment_weight(seg, cfg::SegmentWeights, T_rating::DateTime)
    age_days = max(0.0, (T_rating - DateTime(seg.start_timestamp)) / Millisecond(86_400_000))
    w_time = 0.5 ^ (age_days / cfg.half_life_days)
    w_dur  = cfg.dur_pow == 0.0 ? 1.0 : ((seg.duration + cfg.rho2) / cfg.rho3) ^ cfg.dur_pow
    garbage = abs(seg.gd_start) >= cfg.gd_threshold && abs(seg.gd_end) >= cfg.gd_threshold
    return w_time * w_dur * (garbage ? cfg.rho4 : 1.0)
end

# ==========================================
# 5. PLAYER COMPETITION HISTORY  (the league adjustment)
# ==========================================
"""
    competition_sets(; tournaments) -> Dict{Int, Dict{Int, Set{Int}}}

`C_p` as of each match: `out[match_id][player_id] -> set of tournaments the player has appeared
in up to AND INCLUDING this match`. Accumulated chronologically, so a player's league history
never contains information from the future — the naive version (compute `C_p` over the whole
sample) leaks, and would quietly flatter any predictive validation.

Including the *current* match's tournament is not leakage: which competition this match belongs
to is known at kickoff.
"""
function competition_sets(; tournaments::Vector{Int} = PM_TIERS)
    ensure_pm_data!()
    lu = PM_LINEUPS[]
    lu = lu[coalesce.(in.(lu.tournament_id, Ref(tournaments)), false), :]
    sort!(lu, :start_timestamp)

    seen = Dict{Int, Set{Int}}()
    out  = Dict{Int, Dict{Int, Set{Int}}}()
    for g in groupby(lu, :match_id, sort = false)
        mid = Int(g.match_id[1]); tid = Int(g.tournament_id[1])
        local_map = Dict{Int, Set{Int}}()
        for r in eachrow(g)
            ismissing(r.player_id) && continue
            pid = Int(r.player_id)
            s = get!(seen, pid, Set{Int}())
            push!(s, tid)
            local_map[pid] = copy(s)
        end
        out[mid] = local_map
    end
    return out
end

# ==========================================
# 6. THE DESIGN MATRIX
# ==========================================
"""
Column bookkeeping for the design matrix: which column is what, and which columns the ridge is
allowed to penalise. Home advantage and the dismissal dummies are NOT penalised
(Hvattum et al. 2020: "Regularization is not applied to ... the home field advantages and the
effects of red cards").
"""
struct DesignCols
    player_ids::Vector{Int}          # column j ↔ player_ids[j]
    player_index::Dict{Int, Int}
    ha::Int
    reds::UnitRange{Int}
    leagues::UnitRange{Int}
    league_ids::Vector{Int}
    penalised::Vector{Bool}
    n::Int
end

"""
    build_design(segments; weights, T_rating, max_reds=3, normalize_manpower=false, comp_sets=nothing)

Returns `(X, y, w, cols)`.

  * `y`  — RAW goal difference in the segment (not per-90).
  * `X`  — every row scaled by `d/90`, so coefficients read as per-90 contributions. Doing it
           this way round (rather than dividing `y` by `d/90`) keeps short segments from
           acquiring an enormous response variance.
  * player columns: `+1` home, `−1` away, `0` absent.
  * dismissal dummies `n = 1..max_reds`: `+1` if home has ≥n dismissals and away has none,
    `−1` if away has ≥n and home has none, `0` otherwise. That single rule IS the base paper's
    cancellation behaviour — a red card each cancels out.
  * league columns: Hvattum's per-player competition average, `(1/|C_p|)` added for each
    competition the player has appeared in, summed over home players minus away players. In a
    match where every player has only ever played in this one tier the column is exactly 0, so
    league strength is identified ONLY through players who move between tiers — which is the
    intent, and why the 576 cross-tier players matter.

`normalize_manpower` multiplies each side's player entries by `11/|P_side|`, so that when a team
is down to ten the remaining players each carry more of the team's output. Defaults to `false`
to keep the reference arm faithful to the base paper's plain ±1 design; the dismissal dummies
already absorb the average manpower effect.
"""
function build_design(segments::DataFrame;
                      target::Symbol = :y_goals,
                      weights::SegmentWeights = SegmentWeights(),
                      T_rating::DateTime = DateTime(maximum(segments.start_timestamp)),
                      max_reds::Int = 3,
                      normalize_manpower::Bool = false,
                      comp_sets::Union{Nothing, Dict{Int, Dict{Int, Set{Int}}}} = nothing)

    pids = sort(unique(vcat(reduce(vcat, segments.home_players; init = Int[]),
                            reduce(vcat, segments.away_players; init = Int[]))))
    pidx = Dict(p => i for (i, p) in enumerate(pids))
    np = length(pids)

    lids = sort(unique(segments.tournament_id))
    nl = comp_sets === nothing ? 0 : length(lids)
    lidx = Dict(l => i for (i, l) in enumerate(lids))

    ha_col   = np + 1
    red_cols = (np + 2):(np + 1 + max_reds)
    lg_cols  = (np + 2 + max_reds):(np + 1 + max_reds + nl)
    ncol     = np + 1 + max_reds + nl

    penalised = trues(ncol)
    penalised[ha_col] = false
    penalised[red_cols] .= false          # per Hvattum et al. 2020

    I = Int[]; J = Int[]; V = Float64[]
    n = nrow(segments)
    y = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)

    # `target` selects which of the WP4 responses to fit (`:y_goals`, `:y_xg`, `:y_xp`, ...).
    # Fall back to the goal difference so l01 stays usable before l03 has run.
    yvec = hasproperty(segments, target) ? Float64.(segments[!, target]) :
           Float64.(segments.goals_home .- segments.goals_away)

    for (r, seg) in enumerate(eachrow(segments))
        scale = seg.duration / 90.0
        y[r] = yvec[r]
        w[r] = segment_weight(seg, weights, T_rating)

        nh = length(seg.home_players); na = length(seg.away_players)
        fh = normalize_manpower ? 11.0 / max(nh, 1) : 1.0
        fa = normalize_manpower ? 11.0 / max(na, 1) : 1.0

        for p in seg.home_players
            push!(I, r); push!(J, pidx[p]); push!(V,  scale * fh)
        end
        for p in seg.away_players
            push!(I, r); push!(J, pidx[p]); push!(V, -scale * fa)
        end

        push!(I, r); push!(J, ha_col); push!(V, scale)

        for k in 1:max_reds
            v = (seg.red_home >= k && seg.red_away == 0) ?  1.0 :
                (seg.red_away >= k && seg.red_home == 0) ? -1.0 : 0.0
            v == 0.0 && continue
            push!(I, r); push!(J, red_cols[k]); push!(V, scale * v)
        end

        if comp_sets !== nothing
            cmap = get(comp_sets, Int(seg.match_id), nothing)
            if cmap !== nothing
                acc = Dict{Int, Float64}()
                for (players, sgn) in ((seg.home_players, 1.0), (seg.away_players, -1.0))
                    for p in players
                        cs = get(cmap, p, nothing)
                        (cs === nothing || isempty(cs)) && continue
                        inv = sgn / length(cs)
                        for c in cs
                            haskey(lidx, c) || continue
                            acc[c] = get(acc, c, 0.0) + inv
                        end
                    end
                end
                for (c, v) in acc
                    v == 0.0 && continue
                    push!(I, r); push!(J, lg_cols[lidx[c]]); push!(V, scale * v)
                end
            end
        end
    end

    X = sparse(I, J, V, n, ncol)

    # NORMALISE THE WEIGHTS TO MEAN 1. Without this, λ and the half-life are confounded: a
    # shorter half-life shrinks every weight, shrinks XᵀWX with them, and so makes the SAME λ
    # behave like a much heavier penalty. Tuning (λ, half-life) jointly would then be partly
    # chasing a units artefact rather than a modelling trade-off. With mean-1 weights, λ sits on
    # a scale set by the data volume, and the two hyper-parameters move independently.
    # (Measured on the first WP2 run: half_life=365d over six seasons gave mean(w)=0.041, i.e.
    # a silent ×24 inflation of every λ on the grid.)
    mw = mean(w)
    mw > 0 && (w ./= mw)

    cols = DesignCols(pids, pidx, ha_col, red_cols, lg_cols, lids, penalised, ncol)
    return X, y, w, cols
end

# ==========================================
# 7. EXPOSURE BOOKKEEPING
# ==========================================
"""
    player_exposure(segments) -> DataFrame

Per player: number of segments, total on-pitch minutes, matches, and the tiers they appeared in.
Drives the WP2 identifiability gate and the 540-minute analysis floor from RESEARCH_rapm.md §2.2.
"""
function player_exposure(segments::DataFrame)
    acc = Dict{Int, NamedTuple{(:segs, :mins, :matches, :tiers),
                               Tuple{Int, Float64, Set{Int}, Set{Int}}}}()
    for seg in eachrow(segments)
        for players in (seg.home_players, seg.away_players)
            for p in players
                a = get!(acc, p, (segs = 0, mins = 0.0, matches = Set{Int}(), tiers = Set{Int}()))
                push!(a.matches, Int(seg.match_id)); push!(a.tiers, Int(seg.tournament_id))
                acc[p] = (segs = a.segs + 1, mins = a.mins + seg.duration,
                          matches = a.matches, tiers = a.tiers)
            end
        end
    end
    return sort!(DataFrame(
        player_id = collect(keys(acc)),
        n_segments = [v.segs for v in values(acc)],
        minutes    = [round(v.mins, digits = 1) for v in values(acc)],
        n_matches  = [length(v.matches) for v in values(acc)],
        n_tiers    = [length(v.tiers) for v in values(acc)],
    ), :minutes, rev = true)
end
