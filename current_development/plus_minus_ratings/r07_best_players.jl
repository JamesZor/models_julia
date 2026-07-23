# current_development/plus_minus_ratings/r07_best_players.jl
#
# RUNNER. WP7b — "who were the best players this season?", and does SofaScore agree?
#
# This is the base paper's §5.2.3 (Table 5 / Table 6): rank players by rating within a season,
# apply a ≥900-minute floor, and look at the names. The paper judged its ratings partly on
# whether the top of the list was believable.
#
# WHY THIS IS NOT REDUNDANT WITH THE OVERALL CORRELATION (r06). A correlation of 0.20 across
# 800 players is compatible with two very different worlds:
#   (a) the two systems agree loosely everywhere, or
#   (b) they disagree in the muddled middle but AGREE STRONGLY at the extremes.
# Only (b) would make the rating useful for team selection or recruitment, which is what a
# top-N list is actually for. r06's decile table hinted at (b) — it was flat across the bottom
# 70% and only turned up in the top three deciles — so this measures the top end directly.
#
# THE TEAM CONFOUND IS CARRIED THROUGH. r06 established that raw agreement is ~all team
# strength (within-team ρ pinned at ~0.20 whatever we tune). So every measure here is reported
# BOTH raw and within-team, i.e. after subtracting each player's own team's mean. Raw top-N
# lists in a league containing Celtic and Rangers will simply enumerate Celtic and Rangers.

using DataFrames
using Statistics
using StatsBase: mode
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "l04_ridge_apm.jl"))

_hdr(s) = println("\n", "="^78, "\n", s, "\n", "="^78)
_spearman(a, b) = cor(sortperm(sortperm(a)) ./ length(a), sortperm(sortperm(b)) ./ length(b))
_pct(v) = (sortperm(sortperm(v)) .- 0.5) ./ length(v)      # percentile rank in [0,1]

_hdr("Rebuild")
SEG, REJ = build_segments()
SH = build_shots(); XGM = fit_shot_xg(SH); SH.xg = predict_xg(XGM, SH)
LONG = build_state_intervals(); HAZ = fit_inplay_hazard(LONG); XP = xp_table(HAZ)
add_targets!(SEG, SH, XP)
CS = competition_sets()
lu = PM_LINEUPS[]

# team_id → readable name, and player_id → name
TEAMS = let conn = pm_connect()
    df = DataFrame(LibPQ.execute(conn, """
        SELECT DISTINCT l.team_id,
               CASE WHEN l.is_home_team THEN m.home_team ELSE m.away_team END AS team_name
        FROM sofascore.match_player_lineups l
        JOIN sofascore.matches m ON m.match_id = l.match_id
        WHERE m.tournament_id IN (54,55,56,57)"""))
    close(conn)
    Dict(Int(r.team_id) => String(r.team_name) for r in eachrow(df) if !ismissing(r.team_id))
end
PNAME = Dict{Int,String}()
for r in eachrow(lu); ismissing(r.player_name) || (PNAME[Int(r.player_id)] = String(r.player_name)); end

# ==========================================
# PER-SEASON RATINGS
# ==========================================
"""
Rate players AS OF the end of `season`, fitting on a two-year trailing window with a 365-day
half-life anchored at the season end — the base paper's approach (a rolling window, two years of
data, more recent matches weighted higher). Fitting on the single season alone would be truer to
"best player OF that season" but far noisier in a league this size.
"""
function season_ratings(season::String; target = :y_xg, λ = 200.0, w_sim = 0.0, covered = true)
    segs = covered ? SEG[SEG.covered, :] : SEG
    in_season = segs.season .== season
    any(in_season) || return nothing
    t_end   = maximum(DateTime.(segs.start_timestamp[in_season]))
    t_start = minimum(DateTime.(segs.start_timestamp[in_season])) - Day(730)
    win = segs[(DateTime.(segs.start_timestamp) .<= t_end) .&
               (DateTime.(segs.start_timestamp) .>= t_start), :]
    nrow(win) < 1000 && return nothing

    wcfg = SegmentWeights(; half_life_days = 365.0)
    X, y, w, cols = build_design(win; target = target, weights = wcfg,
                                 T_rating = t_end, comp_sets = CS)
    S = w_sim == 0.0 ? nothing : similarity_matrix(win, cols; k = 10)
    A = Matrix(Symmetric(Matrix(X' * Diagonal(w) * X))); b = Vector(X' * (w .* y))
    β = ridge_solve(A, b, penalty_matrix(cols, S, w_sim), λ)

    R = DataFrame(player_id = cols.player_ids, rapm = β[1:length(cols.player_ids)])

    # Minutes and team must come from THE SEASON ITSELF, not the fitting window.
    exp_s = player_exposure(segs[in_season, :])
    R = innerjoin(R, select(exp_s, :player_id, :minutes), on = :player_id)

    # SofaScore yardstick, also from this season only.
    lr = lu[(lu.season .== season) .& .!ismissing.(lu.rating), :]
    lr.mins = coalesce.(passmissing(Float64).(lr.minutes_played), 90.0)
    lr.mins = ifelse.(lr.mins .<= 0, 1.0, lr.mins)
    sofa = combine(groupby(lr, :player_id),
                   [:rating, :mins] => ((r, m) -> sum(r .* m) / sum(m)) => :sofa,
                   :team_id  => (t -> mode(collect(skipmissing(t)))) => :team_id,
                   :position => (p -> mode(pm_clean_position.(p)))   => :pos)
    R = innerjoin(R, sofa, on = :player_id)        # ⇒ tiers 54/55 only, where a rating exists
    R.name = [get(PNAME, p, "?") for p in R.player_id]
    R.team = [get(TEAMS, Int(t), "?") for t in R.team_id]
    return R
end

const SEASONS = ["23/24", "24/25", "25/26"]
const MIN_MINUTES = 900.0          # the base paper's top-N floor

# ==========================================
# TOP-N LISTS AND AGREEMENT AT THE TOP
# ==========================================
function agreement_at_top(R::DataFrame; k::Int = 20)
    R = R[R.minutes .>= MIN_MINUTES, :]
    R = R[R.pos .!= "G", :]                    # r06: goalkeeper RAPM is unidentified (ρ ≈ 0)
    nrow(R) < 3k && return nothing

    R.p_rapm = _pct(R.rapm); R.p_sofa = _pct(R.sofa)
    # within-team: subtract each player's own team mean, so we compare teammates
    tm = combine(groupby(R, :team_id), :rapm => mean => :mr, :sofa => mean => :ms)
    R = innerjoin(R, tm, on = :team_id)
    R.dr = R.rapm .- R.mr; R.ds = R.sofa .- R.ms
    R.pd_rapm = _pct(R.dr); R.pd_sofa = _pct(R.ds)

    top_r = partialsortperm(R.rapm, 1:k, rev = true)
    top_s = partialsortperm(R.sofa, 1:k, rev = true)
    topd_r = partialsortperm(R.dr, 1:k, rev = true)
    topd_s = partialsortperm(R.ds, 1:k, rev = true)
    exp_overlap = k^2 / nrow(R)

    return (R = R, n = nrow(R),
            # where do OUR top-k sit on SofaScore's scale? 0.5 = chance
            sofa_pct_of_rapm_top = mean(R.p_sofa[top_r]),
            rapm_pct_of_sofa_top = mean(R.p_rapm[top_s]),
            overlap = length(intersect(Set(top_r), Set(top_s))),
            exp_overlap = exp_overlap,
            w_sofa_pct_of_rapm_top = mean(R.pd_sofa[topd_r]),
            w_overlap = length(intersect(Set(topd_r), Set(topd_s))),
            spearman = _spearman(R.rapm, R.sofa),
            w_spearman = _spearman(R.dr, R.ds))
end

for (tgt, λ, ws, lbl) in ((:y_xg, 200.0, 0.0, "y_xg  w_SIM=0  (lowest team-loading)"),
                          (:y_shots, 1000.0, 0.9, "y_shots w_SIM=0.9 (WP5 Brier winner)"))
    _hdr("TARGET: $lbl")
    rows = NamedTuple[]
    for s in SEASONS
        R = season_ratings(s; target = tgt, λ = λ, w_sim = ws)
        R === nothing && continue
        a = agreement_at_top(R)
        a === nothing && continue

        println("\n───── $s : top 15 by RAPM (≥900 min, outfield) ─────")
        T = sort(a.R, :rapm, rev = true)
        T.sofa_pct = round.(100 .* T.p_sofa, digits = 0)
        println(first(T, 15)[:, [:name, :pos, :team, :rapm, :sofa, :sofa_pct, :minutes]])

        println("\n───── $s : top 10 by RAPM *within their own team* ─────")
        T2 = sort(a.R, :dr, rev = true)
        T2.sofa_within_pct = round.(100 .* T2.pd_sofa, digits = 0)
        println(first(T2, 10)[:, [:name, :pos, :team, :dr, :ds, :sofa_within_pct, :minutes]])

        push!(rows, (season = s, n = a.n,
                     sofa_pct_of_rapm_top20 = round(100 * a.sofa_pct_of_rapm_top, digits = 1),
                     rapm_pct_of_sofa_top20 = round(100 * a.rapm_pct_of_sofa_top, digits = 1),
                     overlap = a.overlap, chance = round(a.exp_overlap, digits = 1),
                     w_sofa_pct = round(100 * a.w_sofa_pct_of_rapm_top, digits = 1),
                     w_overlap = a.w_overlap,
                     spearman = round(a.spearman, digits = 3),
                     w_spearman = round(a.w_spearman, digits = 3)))
    end
    println("\n───── agreement at the top, $lbl ─────")
    println(DataFrame(rows))
end

println("""

HOW TO READ THE SUMMARY
  sofa_pct_of_rapm_top20 : average SofaScore percentile of the players OUR rating ranks top-20.
                           50 = no better than picking at random. 90 = strong agreement.
  overlap vs chance      : how many of the two top-20s are the same player, against the number
                           expected by chance (k²/n).
  w_* columns            : the same quantities computed WITHIN TEAM. r06 showed raw agreement is
                           essentially all team strength, so these are the honest numbers — they
                           ask whether we pick the right player out of a given squad.
""")
_hdr("WP7b done")
