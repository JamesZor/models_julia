#=
LOADER l01 — book schema + score-grid geometry + the StakingMatch object.

The neutral, source-agnostic foundation of the staking module. Everything downstream (the Kelly
solver, the coherent pricing, the trust models, the policies) speaks in terms of the objects
defined here. NOTHING here knows whether a match came from the simulator or from real L1 + Betfair
data — that is exactly why the two sources can share the whole pipeline.

`StakingMatch` (was `SimMatch` in the old staking_sim lab — renamed because it carries REAL data
just as happily as simulated data) is one match's book: quoted odds, de-vigged market probs, the
model's posterior score-grid, settlement, the return matrix, and the two team ids (needed by the
hierarchical trust model; 0 = unknown).

Book layout = the core 11 selections: 1X2 (3) + O/U 1.5/2.5/3.5 (6) + BTTS (2), fixed order.
7 trust *units*: complements (under_x, btts_no) share their over/yes unit's weight — they carry no
independent information. `settle_score` uses true-score semantics (avoids GG truncation edges).
=#

# ---------- score-grid geometry ----------

const GG = 12                                              # goals 0..11 per side → 144 states
const HGRID = vec([h for h in 0:GG-1, a in 0:GG-1])
const AGRID = vec([a for h in 0:GG-1, a in 0:GG-1])

"Binary indicator over the 144 grid states for a (market_name, market_line, selection) row."
function mask_for(mname, mline, sel)
    s = String(sel)
    if mname == "1X2"
        s == "home" && return HGRID .> AGRID
        s == "draw" && return HGRID .== AGRID
        return HGRID .< AGRID
    elseif mname == "OverUnder"
        return startswith(s, "over") ? (HGRID .+ AGRID .> mline) : (HGRID .+ AGRID .< mline)
    elseif mname == "BTTS"
        yes = (HGRID .>= 1) .& (AGRID .>= 1)
        return s == "btts_yes" ? yes : .!yes
    end
    error("mask_for: unknown market ($mname, $mline, $sel)")
end

# ---------- fixed book layout (11 selections) ----------

const SEL_NAMES = ["home", "draw", "away",
                   "over_15", "under_15", "over_25", "under_25", "over_35", "under_35",
                   "btts_yes", "btts_no"]
const SEL_SPECS = [("1X2", 0.0, "home"), ("1X2", 0.0, "draw"), ("1X2", 0.0, "away"),
                   ("OverUnder", 1.5, "over"), ("OverUnder", 1.5, "under"),
                   ("OverUnder", 2.5, "over"), ("OverUnder", 2.5, "under"),
                   ("OverUnder", 3.5, "over"), ("OverUnder", 3.5, "under"),
                   ("BTTS", 0.0, "btts_yes"), ("BTTS", 0.0, "btts_no")]
const SEL_MASKS = [BitVector(mask_for(n, l, s)) for (n, l, s) in SEL_SPECS]
const MMASK = Float64.(hcat(SEL_MASKS...))                 # 144 × 11
const FAM_OF_SEL = [1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3]       # 1=1X2, 2=OU, 3=BTTS

# trust units: 7 lines (complement shares its unit's w); representative sel per unit
const UNIT_OF_SEL  = [1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7]
const UNIT_REP_SEL = [1, 2, 3, 4, 6, 8, 10]                # home, draw, away, o15, o25, o35, btts_yes
const UNIT_NAMES   = ["home", "draw", "away", "over_15", "over_25", "over_35", "btts_yes"]

# ---------- the match object ----------

"""
One match's book + model posterior, source-agnostic. Fields:
  `d`      quoted decimal odds (commission-adjusted), 11
  `q_mkt`  de-vigged market prob per selection, 11
  `P`      model posterior score-grids, 144 × S
  `pbar`   mean grid over draws, 144  (coherent substrate for the tilt)
  `won`    settlement per selection, 11
  `R`      return matrix d'.*mask − 1, 144 × 11
  `home`   home team id (0 = unknown)  — for hierarchical trust
  `away`   away team id (0 = unknown)
  `score`  (home_goals, away_goals)
"""
struct StakingMatch
    d::Vector{Float64}
    q_mkt::Vector{Float64}
    P::Matrix{Float64}
    pbar::Vector{Float64}
    won::Vector{Bool}
    R::Matrix{Float64}
    home::Int
    away::Int
    score::Tuple{Int,Int}
end

"True-score settlement of the 11 core selections."
function settle_score(h::Int, a::Int)
    Bool[h > a, h == a, h < a,
         h + a > 1.5, h + a < 1.5, h + a > 2.5, h + a < 2.5, h + a > 3.5, h + a < 3.5,
         h >= 1 && a >= 1, !(h >= 1 && a >= 1)]
end

"Return matrix over the 144 states for a quoted-odds vector d (11): d'.*MMASK − 1."
return_matrix(d::Vector{Float64}) = d' .* MMASK .- 1.0

"Per-selection posterior-draw probabilities (11 × S) from a match's grid draws."
sel_dists(P::Matrix{Float64}) = MMASK' * P
sel_dists(m::StakingMatch) = MMASK' * m.P
