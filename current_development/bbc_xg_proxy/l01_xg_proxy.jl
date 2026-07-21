#=
l01 — loader for the bbc xG-proxy stream (WP1/WP2).

Provides:
  - fetch_matches_wide(conn; tournaments)  : one row per match — bbc stats pivoted
    (prefer genuine over backfilled rows), sofascore xG (nullable), meta.
  - to_team_rows(df)                       : two team-match rows per match
    (own attacking stats + opponent fouls + home flag + xG target).
  - fit ladder: fit_proxy (Gamma log-link or lognormal OLS via GLM.jl).
  - eval: r2/mae/spearman, season-blocked CV, Champ↔Prem transfer splits,
    decile calibration table.

Design decisions (see NOTES.md WP0):
  - `filled=true` rows are backfilled-but-usable; de-dup prefers genuine rows.
  - Feature vector: shotsOnTarget, shotsOffTarget, shotsBlocked, cornersWon,
    possessionPercentage, hitWoodwork (own) + foulsCommitted (opponent) + home.
  - Target: sofascore expectedGoals per side, floored at 1e-2 for Gamma/log.
=#

using LibPQ
using DataFrames
using Statistics
using GLM
using StatsBase: corspearman, ordinalrank

# ==========================================
# Data extraction
# ==========================================

const PROXY_STATS = ["shotsOnTarget", "shotsOffTarget", "shotsBlocked",
                     "cornersWon", "possessionPercentage", "foulsCommitted",
                     "hitWoodwork"]

"""One row per match with bbc stats pivoted wide + xG (missing where absent)."""
function fetch_matches_wide(conn; tournaments::Vector{Int} = [54, 55, 56, 57])
    stats_in = join(("'" * s * "'" for s in PROXY_STATS), ",")
    t_in = join(tournaments, ",")
    piv_cols = join(
        ["""max(home_value) FILTER (WHERE stat_type = '$s') AS $(lowercase(s))_h,
            max(away_value) FILTER (WHERE stat_type = '$s') AS $(lowercase(s))_a"""
         for s in PROXY_STATS], ",\n")
    sql = """
    WITH st AS (
        SELECT DISTINCT ON (match_id, stat_type)
               match_id, stat_type, home_value, away_value
        FROM bbc.match_stats
        WHERE stat_type IN ($stats_in)
        ORDER BY match_id, stat_type, filled   -- genuine (filled=false) wins de-dup
    ),
    piv AS (SELECT match_id, $piv_cols FROM st GROUP BY match_id),
    xg AS (
        SELECT match_id, home_value::float8 AS xg_h, away_value::float8 AS xg_a
        FROM sofascore.match_statistics
        WHERE stat_key = 'expectedGoals' AND period = 'ALL'
          AND home_value IS NOT NULL AND away_value IS NOT NULL
    )
    SELECT m.match_id, m.tournament_id, s.name AS season,
           m.start_timestamp, m.home_team, m.away_team,
           m.home_score, m.away_score,
           $(join(["piv.$(lowercase(s))_h, piv.$(lowercase(s))_a" for s in PROXY_STATS], ", ")),
           xg.xg_h, xg.xg_a
    FROM bbc.match_meta b
    JOIN sofascore.matches m ON m.match_id = b.match_id
    JOIN sofascore.seasons s ON s.season_id = m.season_id AND s.tournament_id = m.tournament_id
    JOIN piv ON piv.match_id = b.match_id
    LEFT JOIN xg ON xg.match_id = b.match_id
    WHERE m.tournament_id IN ($t_in) AND b.scores_match
    ORDER BY m.start_timestamp
    """
    df = DataFrame(execute(conn, sql))
    # LibPQ returns bbc integer stats as Union{Missing,Int32}; keep missing, cast to Float64
    for c in names(df)
        if endswith(c, "_h") || endswith(c, "_a")
            df[!, c] = passmissing(Float64).(df[!, c])
        end
    end
    return df
end

"""Two team-match rows per match. Own attacking stats + opponent fouls + home flag."""
function to_team_rows(df::DataFrame)
    rows = NamedTuple[]
    for r in eachrow(df)
        for (side, opp) in (("h", "a"), ("a", "h"))
            get_(base) = r[Symbol(lowercase(base) * "_" * side)]
            push!(rows, (
                match_id      = r.match_id,
                tournament_id = r.tournament_id,
                season        = String(r.season),
                kickoff       = r.start_timestamp,
                team          = side == "h" ? String(r.home_team) : String(r.away_team),
                opponent      = side == "h" ? String(r.away_team) : String(r.home_team),
                is_home       = side == "h" ? 1.0 : 0.0,
                goals         = side == "h" ? r.home_score : r.away_score,
                goals_against = side == "h" ? r.away_score : r.home_score,
                sot           = get_("shotsOnTarget"),
                soff          = get_("shotsOffTarget"),
                sblock        = get_("shotsBlocked"),
                corners       = get_("cornersWon"),
                poss          = get_("possessionPercentage"),
                woodwork      = get_("hitWoodwork"),
                fouls_opp     = r[Symbol("foulscommitted_" * opp)],
                xg            = side == "h" ? r.xg_h : r.xg_a,
            ))
        end
    end
    return DataFrame(rows)
end

"""Drop rows unusable for fitting (missing core stats); floor xG for log/Gamma."""
function training_rows(team_df::DataFrame)
    tr = dropmissing(team_df, [:sot, :soff, :corners, :poss, :xg])
    tr = tr[tr.xg .> 0, :]
    tr.sblock   = coalesce.(tr.sblock, 0.0)      # partially-missing; 0 + weak signal
    tr.woodwork = coalesce.(tr.woodwork, 0.0)
    tr.xg_floor = max.(tr.xg, 1e-2)
    return tr
end

# ==========================================
# Model ladder (GLM.jl)
# ==========================================

const F_M0 = @formula(xg_floor ~ sot)                                # naive SoT scaler
const F_M1 = @formula(xg_floor ~ sot + soff + sblock + corners + poss + woodwork + fouls_opp + is_home)
const F_M2 = @formula(xg_floor ~ sot + soff + sblock + corners + poss + woodwork + fouls_opp + is_home +
                                 sqrt(sot) + sqrt(soff) + sot & poss)  # diminishing returns + tempo interaction

"""fit_proxy(df, f; link=:gamma) → fitted GLM. link ∈ (:gamma, :lognormal)."""
function fit_proxy(df::DataFrame, f::FormulaTerm; link::Symbol = :gamma)
    if link === :gamma
        return glm(f, df, Gamma(), LogLink())
    elseif link === :lognormal
        lhs, rhs = f.lhs, f.rhs
        return lm(FormulaTerm(FunctionTerm(log, [lhs], :(log($(lhs.sym)))), rhs), df)
    end
    error("unknown link $link")
end

"""Predictions on the xG scale (lognormal gets a smearing-free naive exp — fine for ranking/MAE comparison)."""
predict_xg(m, df; link::Symbol = :gamma) =
    link === :gamma ? predict(m, df) : exp.(predict(m, df))

# ==========================================
# Evaluation
# ==========================================

function eval_metrics(y::AbstractVector, ŷ::AbstractVector)
    ok = .!ismissing.(ŷ) .& isfinite.(Float64.(coalesce.(ŷ, NaN)))
    y = Float64.(y[ok]); ŷ = Float64.(ŷ[ok])
    ss_res = sum(abs2, y .- ŷ)
    ss_tot = sum(abs2, y .- mean(y))
    (n = length(y),
     r2 = round(1 - ss_res / ss_tot, digits=4),
     mae = round(mean(abs.(y .- ŷ)), digits=4),
     spearman = round(corspearman(y, ŷ), digits=4))
end

"""Season-blocked CV: each (tournament, season) block held out once."""
function blocked_cv(tr::DataFrame, f::FormulaTerm; link::Symbol = :gamma)
    out = NamedTuple[]
    blocks = sort(unique(collect(zip(tr.tournament_id, tr.season))))
    for (tid, seas) in blocks
        test = (tr.tournament_id .== tid) .& (tr.season .== seas)
        m = fit_proxy(tr[.!test, :], f; link)
        ŷ = predict_xg(m, tr[test, :]; link)
        push!(out, (block = "$(tid) $(seas)", eval_metrics(tr.xg[test], ŷ)...))
    end
    pooled_pred = similar(tr.xg, Union{Missing,Float64})
    for (tid, seas) in blocks
        test = (tr.tournament_id .== tid) .& (tr.season .== seas)
        m = fit_proxy(tr[.!test, :], f; link)
        pooled_pred[test] = predict_xg(m, tr[test, :]; link)
    end
    push!(out, (block = "POOLED-OOS", eval_metrics(tr.xg, pooled_pred)...))
    DataFrame(out)
end

"""Champ↔Prem transfer: the tier-invariance gate."""
function transfer_test(tr::DataFrame, f::FormulaTerm; link::Symbol = :gamma)
    champ = tr[tr.tournament_id .== 55, :]
    prem  = tr[tr.tournament_id .== 54, :]
    m_c = fit_proxy(champ, f; link)
    m_p = fit_proxy(prem, f; link)
    DataFrame([
        (direction = "Champ→Prem",  eval_metrics(prem.xg,  predict_xg(m_c, prem;  link))...),
        (direction = "Prem→Champ",  eval_metrics(champ.xg, predict_xg(m_p, champ; link))...),
        (direction = "Champ→Champ(in)", eval_metrics(champ.xg, predict_xg(m_c, champ; link))...),
        (direction = "Prem→Prem(in)",   eval_metrics(prem.xg,  predict_xg(m_p, prem;  link))...),
    ])
end

"""Decile calibration: mean predicted vs mean actual xG per predicted-decile."""
function calibration_deciles(y::AbstractVector, ŷ::AbstractVector; k::Int = 10)
    y = Float64.(y); ŷ = Float64.(ŷ)
    dec = ceil.(Int, ordinalrank(ŷ) ./ (length(ŷ) / k))
    combine(groupby(DataFrame(y=y, ŷ=ŷ, dec=clamp.(dec, 1, k)), :dec),
            nrow => :n,
            :ŷ => (x -> round(mean(x), digits=3)) => :pred_mean,
            :y => (x -> round(mean(x), digits=3)) => :actual_mean)
end
