#=
LOADER — REAL-data adapter: L1 smile-engine OOS predictions + Betfair close book → SimMatch.

Turns the `src_sup40_sw40` engine's real out-of-sample matches (Ireland 2025-26, Betfair
close) into the SAME `SimMatch` objects the staking_sim lab races, so the sim l02 staking
machinery (trust blend → coherent IPF grid tilt → capped unified Kelly, EB trust fit) runs
verbatim on real books. Reuse, don't fork.

THE SMILE SUBTLETY (see the plan + src/predictions/score_computation/smile_poisson.jl):
`src_sup40_sw40` prices O/U through Λ = λ_tot·φ(K), NOT the plain (λ_h,λ_a) grid. So:
  • `SimMatch.P`  = plain double-Poisson grids from the posterior λ draws (144 × S) — the
    coherent-grid substrate the unified solver needs (1X2/BTTS are grid-correct here).
  • the per-UNIT model targets for the trust blend + the raw-unified (w=1) tilt come from the
    smile PPD (`model_inference`) — carried alongside as `smile_sel` (11) and `smile_dists`
    (11 × S). O/U entries are the smile probs; 1X2/BTTS equal the grid by construction.
The runner imprints `smile_sel` onto the grid via `coherent_multiplier` (l02), so EVERY
unified strategy — including raw U (w=1) — prices O/U exactly as b21 certified.

Book = the core 11 selections (1X2, O/U 1.5/2.5/3.5, BTTS), fixed order = sim SEL_NAMES.
Commission c folds into the payout: d_eff = 1 + (odds_close − 1)(1 − c), applied to BOTH
decisions and settlement (net-winnings commission). A missing line ⇒ d = 1.0 (pure-loss
column ⇒ the solver never stakes it, R-return 0) and q_mkt = model prob (blend no-op).

Depends on ../staking_sim/l01_sim_market_model.jl (SimMatch, MMASK, SEL_*, settle_score,
GG/HGRID/AGRID via unified_staking/l01). Guard-included below.
=#

using DataFrames
using Distributions
using Statistics
using LinearAlgebra: dot

if !@isdefined(SimMatch)
    include(joinpath(@__DIR__, "..", "staking_sim", "l01_sim_market_model.jl"))
end

# Core-11 book keys, in sim SEL_NAMES order:
#   home, draw, away, over_15, under_15, over_25, under_25, over_35, under_35, btts_yes, btts_no
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

"""
    build_real_books(lat, ppd, odds_bf, matches_df; c=0.02, S_keep=200)

Returns a NamedTuple:
  • `matches`      :: Vector{SimMatch}          — one per match with betfair odds, match_date order
  • `smile_sel`    :: Vector{Vector{Float64}}   — 11 per-sel MODEL probs (O/U = smile PPD)
  • `smile_dists`  :: Vector{Matrix{Float64}}   — 11 × S per-sel MODEL draws (O/U = smile PPD)
  • `mids`         :: Vector{Int}               — match ids, aligned with the above

`lat`  = LatentStates from extract_oos_predictions (needs λ_h, λ_a draw vectors).
`ppd`  = PPD from model_inference(lat)             (smile-correct per-sel distributions).
`odds_bf` = summarize_betfair_market frame         (odds_close, prob_fair_close, is_winner).
`matches_df` = ds.matches                          (home_score, away_score, match_date).
"""
function build_real_books(lat, ppd, odds_bf::DataFrame, matches_df::DataFrame;
                          c::Float64=0.02, S_keep::Int=200)
    latdf = lat.df
    ppddf = ppd.df

    # matches present in BOTH latents and the betfair book, ordered by kickoff
    odds_ids = Set(odds_bf.match_id)
    keep = [mid for mid in latdf.match_id if mid in odds_ids]
    mdate = Dict(r.match_id => r.match_date for r in eachrow(matches_df))
    sort!(keep, by = mid -> mdate[mid])

    # fast lookups
    lat_row = Dict(r.match_id => r for r in eachrow(latdf))
    score   = Dict(r.match_id => (r.home_score, r.away_score) for r in eachrow(matches_df))
    core = Set(("1X2", "OverUnder", "BTTS"))
    osub = odds_bf[in.(odds_bf.market_name, Ref(core)), :]
    odds_by = Dict(k.match_id => v for (k, v) in pairs(groupby(osub, :match_id)))
    psub = ppddf[in.(ppddf.market_name, Ref(core)), :]
    ppd_by = Dict(k.match_id => v for (k, v) in pairs(groupby(psub, :match_id)))

    matches     = Vector{SimMatch}(undef, length(keep))
    smile_sel   = Vector{Vector{Float64}}(undef, length(keep))
    smile_dists = Vector{Matrix{Float64}}(undef, length(keep))

    for (ix, mid) in enumerate(keep)
        # --- model: plain grids (P, pbar) from posterior λ draws ---
        lr = lat_row[mid]
        λh, λa = lr.λ_h, lr.λ_a
        idx = _thin_idx(length(λh), S_keep)
        S = length(idx)
        P = Matrix{Float64}(undef, GG * GG, S)
        @inbounds for (j, s) in enumerate(idx)
            P[:, j] = _plain_grid(Float64(λh[s]), Float64(λa[s]))
        end
        pbar = vec(mean(P, dims=2))

        # --- model: smile per-selection probs + draws (11 sels) from the PPD ---
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
            if dist === nothing                    # PPD miss (shouldn't happen for core) → grid mask mean
                gm = Float64.(SEL_MASKS[m])
                psel[m] = dot(gm, pbar)
                sdist[m, :] = (gm' * P)'
            else
                psel[m] = mean(dist)
                sdist[m, :] = dist[idx]
            end
        end

        # --- book: odds, de-vigged market prob, settlement (commission into payout) ---
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

        # settlement: prefer graded is_winner, fall back to the true-score mask
        h, a = score[mid]
        won_score = settle_score(Int(h), Int(a))
        won = Bool[ismissing(won_bf[m]) ? won_score[m] : won_bf[m] for m in 1:11]

        R = d' .* MMASK .- 1.0
        matches[ix]     = SimMatch(d, q, P, pbar, won, R, fill(NaN, 11), (Int(h), Int(a)))
        smile_sel[ix]   = psel
        smile_dists[ix] = sdist
    end

    return (matches=matches, smile_sel=smile_sel, smile_dists=smile_dists, mids=keep)
end
