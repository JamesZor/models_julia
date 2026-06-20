#=
l06_momentum_feature.jl  —  CAUSAL SofaScore-momentum feature for the in-play model.

SofaScore `match_graph.points` is a per-minute signed momentum series (+ve = home pressing).
For in-play prediction we must use ONLY the minutes up to the current bin (no future leak), so we
build a *causal* decay-weighted Area-Under-the-Curve up to t_m and take the side's net dominance.

Verified on Ireland: all 253 modelling matches have momentum; the standardised per-side feature is
only weakly tied to current score (r≈0.085 — not just re-encoding game state) and modestly predicts
realized remaining goals (r≈0.156). In the Turing model β_mom ≈ +0.14 (90% CI [0.09, 0.18], ×1.15/SD),
and held-out count elpd improves (−1.0753 → −1.0689). See r06 / momentum_feature_report.md.

Reuses l01_momentum.jl (connect_to_db, fetch_momentum_data, parse_points_to_vector).
=#

using DataFrames, Statistics, LibPQ, JSON3
include(joinpath(@__DIR__, "..", "l01_momentum.jl"))   # connect_to_db / fetch_momentum_data / parse_points_to_vector

"""
    causal_momentum_auc(v, t_m; decay=0.03) -> (home_auc, away_auc)

Decay-weighted momentum AUC over minutes 1:min(t_m, len) ONLY (causal). Later minutes weight higher:
`w_t = exp(-decay·(T − t))`, `home += max(0, v_t)·w_t`, `away += max(0, −v_t)·w_t`.
"""
function causal_momentum_auc(v::AbstractVector, t_m::Real; decay::Float64 = 0.03)
    T = min(round(Int, t_m), length(v))
    T < 1 && return (0.0, 0.0)
    h = a = 0.0
    @inbounds for t in 1:T
        w = exp(-decay * (T - t))
        h += max(0.0,  Float64(v[t])) * w
        a += max(0.0, -Float64(v[t])) * w
    end
    return (h, a)
end

"""
    build_momentum_lookup(tournament_ids) -> Dict(match_id => Vector{Int})

Fetch and parse the per-minute momentum series for the given tournament ids (one DB round-trip).
"""
function build_momentum_lookup(tournament_ids::Vector{Int})
    conn = connect_to_db()
    df = try
        q = """SELECT mg.match_id, mg.points FROM match_graph mg
               INNER JOIN matches mm ON mg.match_id = mm.match_id
               WHERE mm.tournament_id = ANY(\$1)"""
        DataFrame(LibPQ.execute(conn, q, [tournament_ids]))
    finally
        close(conn)
    end
    lookup = Dict{Int,Vector{Int}}()
    for r in eachrow(df)
        v = parse_points_to_vector(r.points)
        (ismissing(v) || isempty(v)) && continue
        lookup[Int(r.match_id)] = v
    end
    return lookup
end

"Net causal momentum from a side's perspective (+ve = that side dominant) at (match_id, t_m)."
function row_net_momentum(lookup, match_id, t_m, is_home; decay = 0.03)
    v = get(lookup, Int(match_id), nothing)
    v === nothing && return 0.0
    h, a = causal_momentum_auc(v, t_m; decay = decay)
    return is_home == 1.0 ? (h - a) : (a - h)
end
