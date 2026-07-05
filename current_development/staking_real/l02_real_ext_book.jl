#=
LOADER — EXTENDED multi-market book (v2): generalize the core-11 staking book to EVERY market
family we can price off the 12×12 score grid, with net-return-per-state settlement that handles
pushes (Draw-No-Bet, whole-line Asian) and half-wins (quarter-line Asian).

Design (see experiments.md §v2): the trust blend + coherent IPF tilt still run on the validated
independent directions (1X2 supremacy, the O/U totals ladder, BTTS). CorrectScore, DoubleChance,
DrawNoBet and AsianHandicap are DETERMINISTIC functions of the score grid, so they are priced off
the SAME coherently-tilted grid and staked jointly by the unified Kelly — no new tilt units, no
IPF conflicts. Only the settlement generalizes.

`sel_payoff` is the single engine for BOTH the model return matrix R (over the 144 grid states)
and the realized settlement (at the actual score) — win → d−1, lose → −1, push → 0,
AH-quarter → mean of its two component lines. It mirrors src/Data/betfair_util.jl `grade_selection`
exactly, extended from a win/lose Bool to a net-return Float.

Betfair coverage note (Ireland 2025-26 OOS, 275 matches): only 1X2, OverUnder(0.5–5.5), BTTS and
CorrectScore have books; AsianHandicap/DrawNoBet are absent and DoubleChance has no OOS liquidity.
The AH/DC/DNB code below is correct and auto-activates on any league/match that carries them.

Depends on l01_real_books.jl (SimMatch, MMASK, SEL_*, GG/HGRID/AGRID, mask_for, build helpers).
=#

using DataFrames
using Distributions
using Statistics
using LinearAlgebra: dot
using LogExpFunctions: logit

if !@isdefined(build_real_books)
    include(joinpath(@__DIR__, "l01_real_books.jl"))
end

const _MKTS = BayesianFootball.Data.Markets

# families we stake in the extended book (priced off the tilted grid)
const EXT_FAMILIES = Set(["1X2", "OverUnder", "BTTS", "CorrectScore", "DoubleChance",
                          "DrawNoBet", "AsianHandicap"])
const FAM_ID = Dict("1X2"=>1, "OverUnder"=>2, "BTTS"=>3, "CorrectScore"=>4,
                    "DoubleChance"=>5, "DrawNoBet"=>6, "AsianHandicap"=>7)
const FAM_LABEL = ["1X2", "totals", "BTTS", "CorrectScore", "DoubleChance", "DrawNoBet", "AsianHandicap"]

"""
    sel_payoff(mn, sel, ml, h, a, d) -> Float64

Net return per unit stake for selection `sel` (market `mn`, line `ml`) at final score (h,a),
given commission-adjusted decimal odds `d`. win → d−1, lose → −1, push → 0. Quarter-line Asian
splits the stake across its two adjacent whole/half lines (mean of the two payoffs).
"""
function sel_payoff(mn::AbstractString, sel::Symbol, ml::Float64, h::Int, a::Int, d::Float64)::Float64
    W(b::Bool) = b ? (d - 1.0) : -1.0
    if mn == "1X2"
        sel === :home && return W(h > a)
        sel === :draw && return W(h == a)
        return W(h < a)                                   # :away
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
        return W(h != a)                                  # :DC_12
    elseif mn == "DrawNoBet"
        h == a && return 0.0                              # draw → stake refunded
        return sel === :dnb_home ? W(h > a) : W(a > h)
    elseif mn == "AsianHandicap"
        side, L = _MKTS.parse_ah_selection(sel)
        pay(LL) = begin
            margin = side === :home ? (h - a) : (a - h)
            adj = margin + LL
            adj > 0 ? (d - 1.0) : adj < 0 ? -1.0 : 0.0    # ==0 → push
        end
        return _MKTS.ah_is_quarter(L) ? 0.5 * pay(L - 0.25) + 0.5 * pay(L + 0.25) : pay(L)
    end
    return -1.0
end

# grid-state score vectors (h,a per flat state) once
const _HG = Int.(HGRID)
const _AG = Int.(AGRID)

"144-state model net-return column for one selection (payoff evaluated at every grid score)."
function _return_column(mn, sel, ml, d)
    r = Vector{Float64}(undef, GG * GG)
    @inbounds for s in 1:(GG * GG)
        r[s] = sel_payoff(mn, sel, ml, _HG[s], _AG[s], d)
    end
    return r
end

"One match's extended book: every present Betfair selection, priced off the grid + settled."
struct ExtMatch
    pbar::Vector{Float64}          # 144 plain grid mean (tilt substrate)
    R::Matrix{Float64}             # 144 × M model net-return per state
    settle::Vector{Float64}        # M realized net-return per unit stake (from actual score)
    d::Vector{Float64}             # M commission-adjusted odds
    q::Vector{Float64}             # M de-vigged market prob
    fam::Vector{Int}               # M family id (FAM_ID)
    core_sel::Vector{Float64}      # 7 rep smile model probs [home,draw,away,o15,o25,o35,btts_yes]
    q_core::Vector{Float64}        # 7 rep market probs
    over_lines::Vector{Float64}    # O/U over lines present (for the tilt)
    over_smile::Vector{Float64}    # smile model prob per over line
    over_qmkt::Vector{Float64}     # market prob per over line
    score::Tuple{Int,Int}
end

const CORE_REP = [("1X2",0.0,:home), ("1X2",0.0,:draw), ("1X2",0.0,:away),
                  ("OverUnder",1.5,:over_15), ("OverUnder",2.5,:over_25),
                  ("OverUnder",3.5,:over_35), ("BTTS",0.0,:btts_yes)]

"""
    build_ext_books(lat, ppd, odds_bf, matches_df; c=0.02, S_keep=200) -> (matches, mids)

Extended analogue of `build_real_books`: one `ExtMatch` per match (kickoff order) carrying the
FULL bettable Betfair book (all EXT_FAMILIES) plus the smile O/U ladder + 1X2/BTTS reps for the
trust tilt. Model O/U probs are smile-correct (Λ=λ_tot·φ); everything else prices off the grid.
"""
function build_ext_books(lat, ppd, odds_bf::DataFrame, matches_df::DataFrame;
                         c::Float64=0.02, S_keep::Int=200)
    latdf, ppddf = lat.df, ppd.df
    odds_ids = Set(odds_bf.match_id)
    keep = [mid for mid in latdf.match_id if mid in odds_ids]
    mdate = Dict(r.match_id => r.match_date for r in eachrow(matches_df))
    sort!(keep, by = mid -> mdate[mid])

    lat_row = Dict(r.match_id => r for r in eachrow(latdf))
    score   = Dict(r.match_id => (Int(r.home_score), Int(r.away_score)) for r in eachrow(matches_df))
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

        # bettable selections (the full Betfair book for this match)
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

        # 7 core reps for the trust blend (model = smile PPD, market = de-vig close)
        core_sel = zeros(7); q_core = zeros(7)
        for (u, (mn, ml, sel)) in enumerate(CORE_REP)
            sm = _ppd_mean(pg, mn, ml, sel)
            core_sel[u] = sm === nothing ? dot(Float64.(SEL_MASKS[UNIT_REP_SEL[u]]), pbar) : sm
            q_core[u] = _market_prob(og, mn, ml, sel, core_sel[u])
        end

        out[ix] = ExtMatch(pbar, R, settle, ds, qs, fams, core_sel, q_core,
                           over_lines, over_smile, over_qmkt, (h, a))
    end
    return (matches=out, mids=keep)
end

_market_prob(og, mn, ml, sel, fallback) = begin
    og === nothing && return fallback
    r = findfirst(i -> og.market_name[i] == mn && og.market_line[i] == ml && og.selection[i] == sel, 1:nrow(og))
    r === nothing ? fallback : og.prob_fair_close[r]
end

# ---------- generalized trust tilt over the full O/U ladder ----------

# each present over line maps to the nearest core totals trust unit (u4=o15,u5=o25,u6=o35)
_over_unit(line) = line <= 1.5 ? 4 : line == 2.5 ? 5 : 6

"""
    ext_tilt_multiplier(em, w; cycles=50) -> (mult, tilt_masks, targets)

Coherent IPF multiplier that imprints the blended per-line targets onto the plain grid:
home + draw (1X2 supremacy), the full smile O/U over-ladder, and BTTS. `w` is the 7-unit trust
vector (home,draw,away,o15,o25,o35,btts). CorrectScore/DC/DNB/AH inherit this tilt via the grid.
"""
function ext_tilt_multiplier(em::ExtMatch, w::Vector{Float64}; cycles=50, tol=1e-8)
    # 1X2 supremacy: blend then renormalize the triple, keep home+draw as constraints
    t1 = w[1]*em.core_sel[1] + (1-w[1])*em.q_core[1]
    t2 = w[2]*em.core_sel[2] + (1-w[2])*em.q_core[2]
    t3 = w[3]*em.core_sel[3] + (1-w[3])*em.q_core[3]
    s = t1 + t2 + t3; t1 /= s; t2 /= s
    masks = BitVector[BitVector(mask_for("1X2",0.0,"home")), BitVector(mask_for("1X2",0.0,"draw"))]
    targets = Float64[t1, t2]
    for (k, line) in enumerate(em.over_lines)
        line == 5.5 && continue                          # beyond smile → grid ≈ target, skip
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

# ---------- trust history on the 7 core units (reuses fit_trust_eb) ----------

"Realized win (Bool) of the 7 rep selections from the final score — for the EB trust fit."
function core_wins(em::ExtMatch)
    h, a = em.score; tot = h + a
    Bool[h>a, h==a, h<a, tot>1.5, tot>2.5, tot>3.5, (h>=1 && a>=1)]
end

function push_hist_ext!(hst::TrustHist, em::ExtMatch)
    y = core_wins(em)
    for u in 1:7
        push!(hst.p[u], em.core_sel[u]); push!(hst.q[u], em.q_core[u]); push!(hst.y[u], Float64(y[u]))
    end
end
