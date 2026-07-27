#=
LOADER l09 — RealSource: real L1 smile-engine OOS predictions + Betfair close book → StakingMatch.

The real-data adapter (was staking_real/l01_real_books.jl + l02_real_ext_book.jl). Turns the
`src_sup40_sw40` engine's real out-of-sample matches into the SAME StakingMatch the SimSource
produces, so the whole staking pipeline runs verbatim on real books.

THE SMILE SUBTLETY: `src_sup40_sw40` prices O/U through Λ = λ_tot·φ(K), NOT the plain (λ_h,λ_a)
grid. So StakingMatch.P = plain double-Poisson grids from the posterior λ draws (the coherent
substrate; 1X2/BTTS are grid-correct), while the per-unit MODEL targets for the trust blend come
from the smile PPD, carried as `model_sel` (11) and `model_dists` (11 × S). The policy imprints
`model_sel` onto the grid via coherent_multiplier (cycles=50) so O/U is priced exactly as b21 certified.

Team ids are mapped from the string team names (for the hierarchical trust model / r05 EDA).
Commission c folds into the payout: d_eff = 1 + (odds_close − 1)(1 − c). Depends on l01–l04 +
BayesianFootball.Data.Markets (Asian-handicap parsing in the extended book).
=#

using DataFrames
using Distributions
using Statistics
using LinearAlgebra: dot

const _MKTS = BayesianFootball.Data.Markets

const REAL_SEL_KEYS = [
    ("1X2",       0.0, :home),     ("1X2",       0.0, :draw),     ("1X2",       0.0, :away),
    ("OverUnder", 1.5, :over_15),  ("OverUnder", 1.5, :under_15),
    ("OverUnder", 2.5, :over_25),  ("OverUnder", 2.5, :under_25),
    ("OverUnder", 3.5, :over_35),  ("OverUnder", 3.5, :under_35),
    ("BTTS",      0.0, :btts_yes), ("BTTS",      0.0, :btts_no),
]

"Plain double-Poisson 144-grid for one posterior draw (λh, λa scalars), truncation-renormalized."
function _plain_grid(λh::Float64, λa::Float64)
    ph = pdf.(Poisson(λh), 0:GG-1)
    pa = pdf.(Poisson(λa), 0:GG-1)
    g = vec(ph * pa')
    return g ./ sum(g)
end

"Evenly-spaced subsample indices of a length-n posterior into ≤ S_keep draws (deterministic)."
_thin_idx(n::Int, S_keep::Int) = n <= S_keep ? collect(1:n) : round.(Int, range(1, n, length=S_keep))

"Map team names → dense integer ids (sorted, stable). Returns (id_of::Dict, names::Vector)."
function _team_map(matches_df::DataFrame)
    names = sort(unique(vcat(String.(matches_df.home_team), String.(matches_df.away_team))))
    id_of = Dict(n => i for (i, n) in enumerate(names))
    return id_of, names
end

# ---------- source contract ----------

"Real match source: L1 latents + PPD + Betfair frame + matches frame. c = commission, S_keep = draws."
Base.@kwdef struct RealSource <: AbstractMatchSource
    lat::Any
    ppd::Any
    odds_bf::DataFrame
    matches_df::DataFrame
    c::Float64 = 0.02
    S_keep::Int = 200
end

"""
    load_matches(src::RealSource) -> (; matches, model_sel, model_dists, teams, mids, team_names, prehist)

`model_sel[i]` = 11 per-sel MODEL probs (O/U = smile PPD); `model_dists[i]` = 11 × S model draws.
StakingMatch.P/pbar are the plain grids (coherent substrate). `prehist` is empty (real = cold start).
"""
function load_matches(src::RealSource)
    lat, ppd, odds_bf, matches_df, c, S_keep = src.lat, src.ppd, src.odds_bf, src.matches_df, src.c, src.S_keep
    latdf, ppddf = lat.df, ppd.df
    id_of, team_names = _team_map(matches_df)

    odds_ids = Set(odds_bf.match_id)
    keep = [mid for mid in latdf.match_id if mid in odds_ids]
    mdate = Dict(r.match_id => r.match_date for r in eachrow(matches_df))
    sort!(keep, by = mid -> mdate[mid])

    lat_row = Dict(r.match_id => r for r in eachrow(latdf))
    score   = Dict(r.match_id => (r.home_score, r.away_score) for r in eachrow(matches_df))
    teamid  = Dict(r.match_id => (id_of[String(r.home_team)], id_of[String(r.away_team)]) for r in eachrow(matches_df))
    core = Set(("1X2", "OverUnder", "BTTS"))
    osub = odds_bf[in.(odds_bf.market_name, Ref(core)), :]
    odds_by = Dict(k.match_id => v for (k, v) in pairs(groupby(osub, :match_id)))
    psub = ppddf[in.(ppddf.market_name, Ref(core)), :]
    ppd_by = Dict(k.match_id => v for (k, v) in pairs(groupby(psub, :match_id)))

    n = length(keep)
    matches     = Vector{StakingMatch}(undef, n)
    model_sel   = Vector{Vector{Float64}}(undef, n)
    model_dists = Vector{Matrix{Float64}}(undef, n)

    for (ix, mid) in enumerate(keep)
        lr = lat_row[mid]
        λh, λa = lr.λ_h, lr.λ_a
        idx = _thin_idx(length(λh), S_keep)
        S = length(idx)
        P = Matrix{Float64}(undef, GG * GG, S)
        @inbounds for (j, s) in enumerate(idx)
            P[:, j] = _plain_grid(Float64(λh[s]), Float64(λa[s]))
        end
        pbar = vec(mean(P, dims=2))

        psel = zeros(11)
        sdist = Matrix{Float64}(undef, 11, S)
        pg = get(ppd_by, mid, nothing)
        for m in 1:11
            mn, ml, sel = REAL_SEL_KEYS[m]
            dist = nothing
            if pg !== nothing
                r = findfirst(i -> pg.market_name[i] == mn && pg.market_line[i] == ml && pg.selection[i] == sel, 1:nrow(pg))
                r !== nothing && (dist = pg.distribution[r])
            end
            if dist === nothing
                gm = Float64.(SEL_MASKS[m])
                psel[m] = dot(gm, pbar)
                sdist[m, :] = (gm' * P)'
            else
                psel[m] = mean(dist)
                sdist[m, :] = dist[idx]
            end
        end

        d = ones(11)                               # missing line ⇒ d=1 ⇒ pure-loss col ⇒ never staked
        q = copy(psel)                             # missing ⇒ model prob ⇒ blend no-op
        og = get(odds_by, mid, nothing)
        won_bf = Vector{Union{Missing,Bool}}(missing, 11)
        if og !== nothing
            for m in 1:11
                mn, ml, sel = REAL_SEL_KEYS[m]
                r = findfirst(i -> og.market_name[i] == mn && og.market_line[i] == ml && og.selection[i] == sel, 1:nrow(og))
                r === nothing && continue
                oc = og.odds_close[r]
                d[m] = 1.0 + (oc - 1.0) * (1.0 - c)
                q[m] = og.prob_fair_close[r]
                won_bf[m] = og.is_winner[r]
            end
        end

        h, a = score[mid]
        won_score = settle_score(Int(h), Int(a))
        won = Bool[ismissing(won_bf[m]) ? won_score[m] : won_bf[m] for m in 1:11]
        hid, aid = teamid[mid]

        matches[ix]     = StakingMatch(d, q, P, pbar, won, return_matrix(d), hid, aid, (Int(h), Int(a)))
        model_sel[ix]   = psel
        model_dists[ix] = sdist
    end

    teams = [(m.home, m.away) for m in matches]
    return (; matches, model_sel, model_dists, teams, mids=keep, team_names,
            prehist=(matches=StakingMatch[], model_sel=Vector{Float64}[]))
end

# ==========================================================================================
# EXTENDED multi-market book (v2): all 7 families priced off the coherently-tilted grid.
# net-return-per-state settlement (push for DNB/whole-AH, half for quarter-AH). ExtMatch carries
# a variable-length book; the r03 runner stakes it jointly with unified Kelly. Trust units are the
# SAME 7 core lines (fit_trust on the core reps) — CS/DC/DNB/AH inherit the tilt via the grid.
# ==========================================================================================

const EXT_FAMILIES = Set(["1X2", "OverUnder", "BTTS", "CorrectScore", "DoubleChance",
                          "DrawNoBet", "AsianHandicap"])
const FAM_ID = Dict("1X2"=>1, "OverUnder"=>2, "BTTS"=>3, "CorrectScore"=>4,
                    "DoubleChance"=>5, "DrawNoBet"=>6, "AsianHandicap"=>7)
const FAM_LABEL = ["1X2", "totals", "BTTS", "CorrectScore", "DoubleChance", "DrawNoBet", "AsianHandicap"]

"Net return per unit stake for `sel` at score (h,a) given commission-adjusted odds `d`."
function sel_payoff(mn::AbstractString, sel::Symbol, ml::Float64, h::Int, a::Int, d::Float64)::Float64
    W(b::Bool) = b ? (d - 1.0) : -1.0
    if mn == "1X2"
        sel === :home && return W(h > a)
        sel === :draw && return W(h == a)
        return W(h < a)
    elseif mn == "OverUnder"
        tot = h + a
        return startswith(String(sel), "over") ? W(tot > ml) : W(tot < ml)
    elseif mn == "BTTS"
        return sel === :btts_yes ? W(h >= 1 && a >= 1) : W(!(h >= 1 && a >= 1))
    elseif mn == "CorrectScore"
        s = String(sel)
        if occursin(r"^cs_\d\d$", s)
            return W(h == parse(Int, s[4]) && a == parse(Int, s[5]))
        elseif s == "cs_any_other_home"; return W(h > a && (h >= 4 || a >= 4))
        elseif s == "cs_any_other_away"; return W(h < a && (h >= 4 || a >= 4))
        elseif s == "cs_any_other_draw"; return W(h == a && h >= 4)
        end
        return -1.0
    elseif mn == "DoubleChance"
        sel === :DC_1X && return W(h >= a)
        sel === :DC_X2 && return W(a >= h)
        return W(h != a)
    elseif mn == "DrawNoBet"
        h == a && return 0.0
        return sel === :dnb_home ? W(h > a) : W(a > h)
    elseif mn == "AsianHandicap"
        side, L = _MKTS.parse_ah_selection(sel)
        pay(LL) = begin
            margin = side === :home ? (h - a) : (a - h)
            adj = margin + LL
            adj > 0 ? (d - 1.0) : adj < 0 ? -1.0 : 0.0
        end
        return _MKTS.ah_is_quarter(L) ? 0.5 * pay(L - 0.25) + 0.5 * pay(L + 0.25) : pay(L)
    end
    return -1.0
end

const _HG = Int.(HGRID)
const _AG = Int.(AGRID)

"144-state model net-return column for one selection."
function _return_column(mn, sel, ml, d)
    r = Vector{Float64}(undef, GG * GG)
    @inbounds for s in 1:(GG * GG)
        r[s] = sel_payoff(mn, sel, ml, _HG[s], _AG[s], d)
    end
    return r
end

"One match's extended book: every present Betfair selection, priced off the grid + settled."
struct ExtMatch
    pbar::Vector{Float64}
    R::Matrix{Float64}
    settle::Vector{Float64}
    d::Vector{Float64}
    q::Vector{Float64}
    fam::Vector{Int}
    core_sel::Vector{Float64}
    q_core::Vector{Float64}
    over_lines::Vector{Float64}
    over_smile::Vector{Float64}
    over_qmkt::Vector{Float64}
    home::Int
    away::Int
    score::Tuple{Int,Int}
end

const CORE_REP = [("1X2",0.0,:home), ("1X2",0.0,:draw), ("1X2",0.0,:away),
                  ("OverUnder",1.5,:over_15), ("OverUnder",2.5,:over_25),
                  ("OverUnder",3.5,:over_35), ("BTTS",0.0,:btts_yes)]

_market_prob(og, mn, ml, sel, fallback) = begin
    og === nothing && return fallback
    r = findfirst(i -> og.market_name[i] == mn && og.market_line[i] == ml && og.selection[i] == sel, 1:nrow(og))
    r === nothing ? fallback : og.prob_fair_close[r]
end

"""
    build_ext_books(src::RealSource) -> (matches::Vector{ExtMatch}, mids, team_names)

Extended analogue of load_matches: the FULL bettable Betfair book (all EXT_FAMILIES) per match
plus the smile O/U ladder + 1X2/BTTS reps for the trust tilt.
"""
function build_ext_books(src::RealSource)
    lat, ppd, odds_bf, matches_df, c, S_keep = src.lat, src.ppd, src.odds_bf, src.matches_df, src.c, src.S_keep
    latdf, ppddf = lat.df, ppd.df
    id_of, team_names = _team_map(matches_df)
    odds_ids = Set(odds_bf.match_id)
    keep = [mid for mid in latdf.match_id if mid in odds_ids]
    mdate = Dict(r.match_id => r.match_date for r in eachrow(matches_df))
    sort!(keep, by = mid -> mdate[mid])

    lat_row = Dict(r.match_id => r for r in eachrow(latdf))
    score   = Dict(r.match_id => (Int(r.home_score), Int(r.away_score)) for r in eachrow(matches_df))
    teamid  = Dict(r.match_id => (id_of[String(r.home_team)], id_of[String(r.away_team)]) for r in eachrow(matches_df))
    osub = odds_bf[in.(odds_bf.market_name, Ref(EXT_FAMILIES)), :]
    odds_by = Dict(k.match_id => v for (k, v) in pairs(groupby(osub, :match_id)))
    psub = ppddf[in.(ppddf.market_name, Ref(EXT_FAMILIES)), :]
    ppd_by = Dict(k.match_id => v for (k, v) in pairs(groupby(psub, :match_id)))

    _ppd_mean(pg, mn, ml, sel) = begin
        pg === nothing && return nothing
        r = findfirst(i -> pg.market_name[i] == mn && pg.market_line[i] == ml && pg.selection[i] == sel, 1:nrow(pg))
        r === nothing ? nothing : mean(pg.distribution[r])
    end

    out = Vector{ExtMatch}(undef, length(keep))
    for (ix, mid) in enumerate(keep)
        lr = lat_row[mid]; λh, λa = lr.λ_h, lr.λ_a
        idx = _thin_idx(length(λh), S_keep)
        P = Matrix{Float64}(undef, GG * GG, length(idx))
        @inbounds for (j, s) in enumerate(idx); P[:, j] = _plain_grid(Float64(λh[s]), Float64(λa[s])); end
        pbar = vec(mean(P, dims=2))
        pg = get(ppd_by, mid, nothing)
        h, a = score[mid]

        og = get(odds_by, mid, nothing)
        Rcols = Vector{Float64}[]; ds = Float64[]; qs = Float64[]; fams = Int[]; settle = Float64[]
        over_lines = Float64[]; over_smile = Float64[]; over_qmkt = Float64[]
        if og !== nothing
            for row in eachrow(og)
                mn, ml, sel = row.market_name, row.market_line, row.selection
                haskey(FAM_ID, mn) || continue
                de = 1.0 + (row.odds_close - 1.0) * (1.0 - c)
                push!(Rcols, _return_column(mn, sel, ml, de))
                push!(ds, de); push!(qs, row.prob_fair_close); push!(fams, FAM_ID[mn])
                push!(settle, sel_payoff(mn, sel, ml, h, a, de))
                if mn == "OverUnder" && startswith(String(sel), "over")
                    sm = _ppd_mean(pg, mn, ml, sel)
                    sm === nothing && (sm = dot(Float64.(mask_for(mn, ml, "over")), pbar))
                    push!(over_lines, ml); push!(over_smile, sm); push!(over_qmkt, row.prob_fair_close)
                end
            end
        end
        R = isempty(Rcols) ? zeros(GG * GG, 0) : reduce(hcat, Rcols)

        core_sel = zeros(7); q_core = zeros(7)
        for (u, (mn, ml, sel)) in enumerate(CORE_REP)
            sm = _ppd_mean(pg, mn, ml, sel)
            core_sel[u] = sm === nothing ? dot(Float64.(SEL_MASKS[UNIT_REP_SEL[u]]), pbar) : sm
            q_core[u] = _market_prob(og, mn, ml, sel, core_sel[u])
        end
        hid, aid = teamid[mid]
        out[ix] = ExtMatch(pbar, R, settle, ds, qs, fams, core_sel, q_core,
                           over_lines, over_smile, over_qmkt, hid, aid, (h, a))
    end
    return (matches=out, mids=keep, team_names=team_names)
end

# each present over line maps to the nearest core totals trust unit (u4=o15,u5=o25,u6=o35)
_over_unit(line) = line <= 1.5 ? 4 : line == 2.5 ? 5 : 6

"Coherent IPF multiplier over the full O/U ladder + 1X2 supremacy + BTTS (ext-book version)."
function ext_tilt_multiplier(em::ExtMatch, w::AbstractVector; cycles=50, tol=1e-8)
    t1 = w[1]*em.core_sel[1] + (1-w[1])*em.q_core[1]
    t2 = w[2]*em.core_sel[2] + (1-w[2])*em.q_core[2]
    t3 = w[3]*em.core_sel[3] + (1-w[3])*em.q_core[3]
    s = t1 + t2 + t3; t1 /= s; t2 /= s
    masks = BitVector[BitVector(mask_for("1X2",0.0,"home")), BitVector(mask_for("1X2",0.0,"draw"))]
    targets = Float64[t1, t2]
    for (k, line) in enumerate(em.over_lines)
        line == 5.5 && continue
        u = _over_unit(line)
        push!(masks, BitVector(mask_for("OverUnder", line, "over")))
        push!(targets, w[u]*em.over_smile[k] + (1-w[u])*em.over_qmkt[k])
    end
    push!(masks, BitVector(mask_for("BTTS",0.0,"btts_yes")))
    push!(targets, w[7]*em.core_sel[7] + (1-w[7])*em.q_core[7])

    g = copy(em.pbar); mult = ones(length(g))
    for _ in 1:cycles
        moved = 0.0
        for j in eachindex(masks)
            m = masks[j]; cur = sum(@view g[m])
            t = clamp(targets[j], 1e-9, 1-1e-9)
            δ = logit(t) - logit(clamp(cur, 1e-9, 1-1e-9)); e = exp(δ)
            @views g[m] .*= e; @views mult[m] .*= e
            z = sum(g); g ./= z; mult ./= z; moved = max(moved, abs(δ))
        end
        moved < tol && break
    end
    return mult
end

ext_tilted_pbar(em::ExtMatch, w) = (g = em.pbar .* ext_tilt_multiplier(em, w); g ./ sum(g))

"Realized win (Bool) of the 7 rep selections — for the EB/Bayes trust fit."
function core_wins(em::ExtMatch)
    h, a = em.score; tot = h + a
    Bool[h>a, h==a, h<a, tot>1.5, tot>2.5, tot>3.5, (h>=1 && a>=1)]
end

"Push an ExtMatch onto TrustHist (7 core units + team ids)."
function push_hist_ext!(hst::TrustHist, em::ExtMatch)
    y = core_wins(em)
    for u in 1:7
        push!(hst.p[u], em.core_sel[u]); push!(hst.q[u], em.q_core[u]); push!(hst.y[u], Float64(y[u]))
        push!(hst.home[u], em.home); push!(hst.away[u], em.away)
    end
end
