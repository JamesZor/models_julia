#=
l00_position_helpers.jl — Phase 1 EDA helpers for position-aware player ratings.

Pure analysis (no Turing). Builds the taxonomy + metrics behind the four decision gates:
  Gate 1 coverage, Gate 2 multi-positionality, Gate 3 out-of-position Δ, Gate 4 A-vs-B.

KEY DISCIPLINE (see NOTES.md / PROMPT.md): the live extractor silently defaults a missing
position to "M". Here `canonical_pos` returns `missing` for anything unmappable so we can keep
*true*-M separate from *defaulted*-M; downstream stats are computed on real positions only.

Reuses the live Kalman: `BayesianFootball`'s `calculate_player_ratings(BayesianTracker, ratings)`.

    include("current_development/position_aware_ratings/l00_position_helpers.jl")
=#

using BayesianFootball
using DataFrames
using Dates
using Statistics
using StatsBase
using GLM

const Data = BayesianFootball.Data

# The live extractor's tracker config (src/features/extractors/player_extractors.jl callers).
const LIVE_TRACKER = BayesianFootball.Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)

# ----------------------------------------------------------------------------
# 1. Position taxonomy — missing/unknown -> `missing` (NOT defaulted to "M")
# ----------------------------------------------------------------------------
"""
    canonical_pos(raw) -> Union{String,Missing}

Map a raw SofaScore position string to one of "G","D","M","F". Returns `missing` for
empty/unknown so a *defaulted*-M never masquerades as a real midfield appearance.
Raw vocab seen in this DB: G/Goalkeeper/GK, D/Defender/DF, M/Midfielder/MF, F/Forward/FW/A.
"""
function canonical_pos(raw)
    (ismissing(raw) || raw == "") && return missing
    s = uppercase(strip(string(raw)))
    s in ("G", "GK", "GOALKEEPER")              && return "G"
    s in ("D", "DF", "DEFENDER", "B")           && return "D"
    s in ("M", "MF", "MIDFIELDER")              && return "M"
    s in ("F", "FW", "A", "FORWARD", "ATTACKER")&& return "F"
    # single-letter fallbacks for compound codes (e.g. "DM","CM","RW")
    first(s) == 'G' && return "G"
    first(s) == 'D' && return "D"
    first(s) == 'M' && return "M"
    (first(s) == 'F' || first(s) == 'A') && return "F"
    return missing
end

# ----------------------------------------------------------------------------
# 2. Starter frame — chronological, 2023+, real-position flagged
# ----------------------------------------------------------------------------
"""
    prepare_starter_lineups(ds; from_year=2023) -> DataFrame

Join lineups→match_date, keep STARTERS (`is_substitute == false`, `minutes_played > 0`) from
`from_year` onward. Adds: `pos` (canonical, may be missing), `pos_is_real` (Bool),
`pos_eda` (pos, or "M_DEFAULT" placeholder so the live-pipeline mix is also visible).
Sorted by `match_date`.
"""
function prepare_starter_lineups(ds::Data.DataStore; from_year::Int=2023)
    isempty(ds.lineups) && return DataFrame()
    lu = select(ds.lineups, :match_id, :player_id, :player_name, :team_side,
                :position, :rating, :minutes_played, :is_substitute)
    md = select(ds.matches, :match_id, :match_date)
    df = innerjoin(lu, md, on = :match_id)

    # 2023+ starters only
    yr = year.(df.match_date)
    sub = coalesce.(df.is_substitute, false)
    mins = coalesce.(df.minutes_played, 0.0)
    keep = (yr .>= from_year) .& .!sub .& (mins .> 0)
    df = df[keep, :]
    isempty(df) && return df

    df.pos         = canonical_pos.(df.position)
    df.pos_is_real = .!ismissing.(df.pos)
    df.pos_eda     = [ismissing(p) ? "M_DEFAULT" : p for p in df.pos]
    df.has_rating  = map(x -> !ismissing(x) && !isnan(x), df.rating)
    sort!(df, :match_date)
    return df
end

# ----------------------------------------------------------------------------
# 3. Gate 1 — coverage
# ----------------------------------------------------------------------------
function coverage_stats(df::DataFrame)
    n = nrow(df)
    n == 0 && return (n=0,)
    real = count(df.pos_is_real)
    rated = count(df.has_rating)
    posmix = countmap(df.pos_eda)
    realmix = countmap(skipmissing(df.pos))
    (; n,
       pct_real_pos   = round(100real/n, digits=1),
       pct_rated      = round(100rated/n, digits=1),
       pct_defaultM   = round(100 * get(posmix, "M_DEFAULT", 0) / n, digits=1),
       n_players      = length(unique(df.player_id)),
       n_matches      = length(unique(df.match_id)),
       mix_real       = Dict(k => round(100v/max(real,1), digits=1) for (k,v) in realmix),
    )
end

# ----------------------------------------------------------------------------
# 4. Gate 2 — multi-positionality
# ----------------------------------------------------------------------------
"""
    add_modal_position!(df; min_apps=5) -> DataFrame

On REAL-position rows only: per player compute modal position, distinct-position count, and a
position entropy (bits). Adds row-level `modal_pos` and `is_off_modal` for players with
`>= min_apps` real appearances (others get `missing`, excluded from off-modal stats).
"""
function add_modal_position!(df::DataFrame; min_apps::Int=5)
    # per-player modal position + appearance count over REAL-position rows (copy avoids view pitfalls)
    modal = Dict{Any,String}(); napps = Dict{Any,Int}()
    for sub in groupby(df[df.pos_is_real, :], :player_id)
        cnts = countmap(String.(sub.pos))
        pid = first(sub.player_id)
        napps[pid] = sum(values(cnts))
        modal[pid] = argmax(cnts)          # ties -> first by Dict order; fine for EDA
    end

    df.player_napps = [get(napps, pid, 0) for pid in df.player_id]
    df.modal_pos    = Vector{Union{Missing,String}}(missing, nrow(df))
    df.is_off_modal = Vector{Union{Missing,Bool}}(missing, nrow(df))
    for i in 1:nrow(df)
        (df.pos_is_real[i] && df.player_napps[i] >= min_apps) || continue
        m = modal[df.player_id[i]]
        df.modal_pos[i]    = m
        df.is_off_modal[i] = String(df.pos[i]) != m
    end
    return df
end

"""
    multipositionality_stats(df; min_apps=5) -> NamedTuple

Per-player distinct positions / entropy (players with >= min_apps real apps), plus the
appearance-weighted share of starter player-matches played OFF the player's modal position.
This is the make-or-break gate: ~0 ⇒ per-position ratings ≡ the single rating.
"""
function multipositionality_stats(df::DataFrame; min_apps::Int=5)
    real = df[df.pos_is_real .& (df.player_napps .>= min_apps), :]
    nrow(real) == 0 && return (n_players=0, off_modal_share=NaN)
    # per-player summaries
    pdistinct = Int[]; pentropy = Float64[]
    for sub in groupby(real, :player_id)
        cnts = collect(values(countmap(String.(sub.pos))))
        p = cnts ./ sum(cnts)
        push!(pdistinct, length(cnts))
        push!(pentropy, -sum(x -> x > 0 ? x*log2(x) : 0.0, p))
    end
    offm = collect(skipmissing(real.is_off_modal))
    (; n_players          = length(pdistinct),
       n_player_matches    = nrow(real),
       mean_distinct_pos   = round(mean(pdistinct), digits=3),
       pct_multipos_players= round(100 * mean(pdistinct .> 1), digits=1),
       mean_entropy_bits   = round(mean(pentropy), digits=3),
       off_modal_share     = round(100 * mean(offm), digits=2),   # % of player-matches off modal
       n_off_modal         = count(offm),
    )
end

# ----------------------------------------------------------------------------
# 5. Opponent-strength control (within-match, from the lineups themselves)
# ----------------------------------------------------------------------------
"""
    add_opponent_strength!(df) -> DataFrame

Per (match_id, team_side) sum the starters' realised `rating` as a team-strength proxy; the
opponent's sum becomes each row's `opp_strength`. A within-player control for Gate 3 so the
off-modal Δ isn't just "fielded a weakened XI vs a strong side".
"""
function add_opponent_strength!(df::DataFrame)
    g = combine(groupby(df, [:match_id, :team_side]),
                :rating => (x -> sum(skipmissing(x))) => :team_strength)
    other(side) = side == "home" ? "away" : "home"
    out = copy(df)
    out.opp_side = other.(out.team_side)
    opp = rename(g, :team_side => :opp_side, :team_strength => :opp_strength)
    out = leftjoin(out, opp, on = [:match_id, :opp_side])
    out.opp_strength = coalesce.(out.opp_strength, NaN)
    return out
end

# ----------------------------------------------------------------------------
# 6. Gate 3 — within-player out-of-position regression
# ----------------------------------------------------------------------------
"""
    out_of_position_regression(df) -> (model, table)

Within-player FE via group-demeaning: regress demeaned `rating` on demeaned `is_off_modal`
and demeaned controls (`is_home`, `minutes_played`, `opp_strength`). The off-modal coefficient
is the rating-point penalty for playing off your modal position, net of who you played and how
long. |t| ≫ 2 with a consistent sign ⇒ Gate 3 passes.
"""
function out_of_position_regression(df::DataFrame)
    d = df[df.pos_is_real .& (.!ismissing).(df.is_off_modal) .& df.has_rating, :]
    d = d[(!isnan).(coalesce.(d.opp_strength, NaN)), :]
    nrow(d) < 50 && return (nothing, DataFrame())

    d = DataFrame(
        player_id = d.player_id,
        rating    = Float64.(d.rating),
        off_modal = Float64.(coalesce.(d.is_off_modal, false)),
        is_home   = Float64.(d.team_side .== "home"),
        minutes   = Float64.(coalesce.(d.minutes_played, 0.0)),
        opp_str   = Float64.(d.opp_strength),
    )
    # group-demean each numeric column within player (absorbs player fixed effect)
    demean!(d, :rating); demean!(d, :off_modal)
    demean!(d, :is_home); demean!(d, :minutes); demean!(d, :opp_str)

    m = lm(@formula(rating ~ 0 + off_modal + is_home + minutes + opp_str), d)
    ct = coeftable(m)
    tbl = DataFrame(term=ct.rownms, coef=round.(ct.cols[1], digits=4),
                    se=round.(ct.cols[2], digits=4), t=round.(ct.cols[3], digits=2),
                    p=round.(ct.cols[4], digits=4))
    return (m, tbl)
end

function demean!(d::DataFrame, col::Symbol)
    gd = groupby(d, :player_id)
    transform!(gd, col => (x -> x .- mean(x)) => col)
    return d
end

# ----------------------------------------------------------------------------
# 7. Gate 4 — constructions A & B, online pre-match ratings
# ----------------------------------------------------------------------------
"""
    add_constructions!(df; tracker=LIVE_TRACKER, min_pos_apps=4, delta_table=nothing) -> DataFrame

Adds three PRE-match rating estimates (each computed before seeing the row's realised rating),
all on real-position rated rows, sorted chronologically:

  * `pre_overall` — baseline: the live single per-player Kalman pre-match value.
  * `pre_A`       — Construction A: per-(player×position) Kalman pre-match; falls back to
                     `pre_overall` when that player has < `min_pos_apps` prior apps in the position.
  * `pre_B`       — Construction B: `pre_overall` + δ(played_pos) on off-modal rows, where δ comes
                     from `delta_table` (Dict played_pos=>offset). Pass a TRAIN-estimated δ to
                     avoid leakage; if `nothing`, B == overall (δ=0).
"""
function add_constructions!(df::DataFrame; tracker=LIVE_TRACKER, min_pos_apps::Int=4,
                            delta_table::Union{Nothing,Dict}=nothing)
    d = df[df.pos_is_real .& df.has_rating, :]
    sort!(d, :match_date)
    n = nrow(d)
    d.pre_overall = fill(NaN, n)
    d.pre_A       = fill(NaN, n)
    d.pre_B       = fill(NaN, n)

    # baseline overall pre-match (one sequence per player) — mirrors the live extractor
    for sub in groupby(d, :player_id)
        idx = parentindices(sub)[1]
        d.pre_overall[idx] .= calculate_player_ratings(tracker, Float64.(sub.rating))
    end
    # Construction A: one Kalman sequence per (player, position)
    for sub in groupby(d, [:player_id, :pos])
        idx = parentindices(sub)[1]
        pre = calculate_player_ratings(tracker, Float64.(sub.rating))
        napp_prior = 0:(length(idx)-1)                # prior apps in this position
        for (k, gi) in enumerate(idx)
            d.pre_A[gi] = napp_prior[k] >= min_pos_apps ? pre[k] : d.pre_overall[gi]
        end
    end
    # Construction B: overall + δ(played pos) when off modal
    for (k, row) in enumerate(eachrow(d))
        δ = 0.0
        if delta_table !== nothing && !ismissing(row.is_off_modal) && row.is_off_modal == true
            δ = get(delta_table, row.pos, 0.0)
        end
        d.pre_B[k] = d.pre_overall[k] + δ
    end
    return d
end

"""
    estimate_delta_table(train_df) -> Dict{String,Float64}

Per played-position average residual (realised rating − overall pre-match) on OFF-modal training
rows. The δ Construction B applies. Needs `pre_overall` + `is_off_modal` already on `train_df`.
"""
function estimate_delta_table(train_df::DataFrame)
    d = train_df[(.!ismissing).(train_df.is_off_modal) .& train_df.is_off_modal .&
                 (!isnan).(train_df.pre_overall), :]
    out = Dict{String,Float64}()
    for sub in groupby(d, :pos)
        out[String(first(sub.pos))] = mean(Float64.(sub.rating) .- sub.pre_overall)
    end
    return out
end

"""
    ab_holdout_eval(df; test_frac=0.3, min_pos_apps=4) -> NamedTuple

Chronological train/test split. Estimate B's δ on train, build all three pre-match estimates,
then score RMSE/MAE of each against the realised rating on TEST rows — reported overall and on
OFF-MODAL test rows (where A and B can differ from the baseline). Lower = better next-match
prediction. Also: how often A / B differ materially (>ε) from the baseline.
"""
function ab_holdout_eval(df::DataFrame; test_frac::Float64=0.3, min_pos_apps::Int=4, ε::Float64=0.25)
    base = df[df.pos_is_real .& df.has_rating .& (.!ismissing).(df.is_off_modal), :]
    sort!(base, :match_date)
    nrow(base) < 100 && return (n=nrow(base), note="too few rated off-modal-eligible rows")

    dates = sort(unique(base.match_date))
    cut = dates[clamp(floor(Int, (1-test_frac)*length(dates)), 1, length(dates))]

    full = add_constructions!(base; min_pos_apps=min_pos_apps,
                              delta_table=estimate_delta_table(base[base.match_date .<= cut, :]))
    test = full[full.match_date .> cut, :]
    nrow(test) == 0 && return (n=0, note="empty test")

    y = Float64.(test.rating)
    rmse(p) = sqrt(mean((y .- p).^2))
    mae(p)  = mean(abs.(y .- p))
    offm = coalesce.(test.is_off_modal, false)

    score(mask) = (
        n     = count(mask),
        rmse_overall = round(rmse(test.pre_overall[mask]), digits=4),
        rmse_A       = round(rmse(test.pre_A[mask]),       digits=4),
        rmse_B       = round(rmse(test.pre_B[mask]),       digits=4),
        mae_overall  = round(mae(test.pre_overall[mask]),  digits=4),
        mae_A        = round(mae(test.pre_A[mask]),        digits=4),
        mae_B        = round(mae(test.pre_B[mask]),        digits=4),
    )
    (; cut, n_test=nrow(test),
       all_rows  = score(trues(nrow(test))),
       off_modal = score(offm),
       pct_A_differs = round(100*mean(abs.(test.pre_A .- test.pre_overall) .> ε), digits=2),
       pct_B_differs = round(100*mean(abs.(test.pre_B .- test.pre_overall) .> ε), digits=2),
    )
end

# re-export the src Kalman for convenience
const calculate_player_ratings = BayesianFootball.Features.calculate_player_ratings
