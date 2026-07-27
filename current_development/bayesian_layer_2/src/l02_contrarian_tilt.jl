# current_development/bayesian_layer_2/src/l02_contrarian_tilt.jl
#
# Contrarian stake tilt — validated prototype (see docs/archive/l3_meta_model_research.md §10.5).
#
# Finding: per-market fold-to-fold realised performance is MEAN-REVERTING
# (demeaned lag-1 autocorr ≈ −0.16). A momentum regime-gate performance-chases and LOSES;
# a contrarian tilt WINS. Recently-cold curated-good markets out-perform hot ones ~4× per bet
# (G_emp 0.0171 vs 0.0042 OOS). Tilting Kelly stake up on cold / down on hot, within the curated
# good set, lifts G_emp +26% and profit +9% at β≈1 while preserving bet volume.
#
# Requires a per-bet table `M` with columns: selection, split_id, match_date, stake (Kelly at
# min_edge=0), roi (=odds-1 if win else -1). Build M by joining the L1 PPD to the Betfair odds
# (prob_fair / odds_close / is_winner) and ds.matches (match_date, split_id) — see the runner.

using DataFrames, Statistics, Dates

_geomp(s, r) = isempty(s) ? NaN : exp(mean(log.(max.(1e-8, 1 .+ s .* r)))) - 1
_summ(s, r)  = (bets=length(s), roi=isempty(s) ? NaN : 100*sum(s.*r)/sum(s),
                G_emp=_geomp(s, r), profit=sum(s.*r), turnover=sum(s))

"""
    recent_form(tr, ref_date; hl=45.0)

Recency-discounted realised per-bet log-growth of a market over its past bets `tr`
(those strictly before the current fold). `ref_date` = start of the held-out fold.
Negative ⇒ the market is "cold" right now.
"""
function recent_form(tr::AbstractDataFrame, ref_date::Date; hl::Float64=45.0)
    v = log.(max.(1e-8, 1 .+ tr.stake .* tr.roi))
    w = exp.(-log(2)/hl .* Float64.(Dates.value.(ref_date .- tr.match_date)))
    return sum(w .* v) / sum(w)
end

"""
    contrarian_tilt(wm; β=1.0, scale=0.01, lo=0.3, hi=2.0)

Mean-reversion multiplier on the Kelly stake. `wm`=recent_form. Cold (wm<0) ⇒ >1, hot ⇒ <1.
`β/scale` is the effective strength; clamp keeps it mild. β=0 reproduces static staking.
"""
contrarian_tilt(wm::Real; β=1.0, scale=0.01, lo=0.3, hi=2.0) = clamp(1 - β*wm/scale, lo, hi)

"""
    tilt_backtest(M, markets; β, scale, hl, lo, hi, start_k, min_bets)

Walk-forward: within `markets`, bet every Kelly selection but resize by the contrarian tilt
computed from past folds only. Returns aggregate (bets, roi, G_emp, profit, turnover, mean_tilt).
"""
function tilt_backtest(M, markets; β::Float64=1.0, scale::Float64=0.01, hl::Float64=45.0,
                       lo::Float64=0.3, hi::Float64=2.0, start_k::Int=4, min_bets::Int=12)
    splits = sort(unique(M.split_id)); s=Float64[]; r=Float64[]; tl=Float64[]
    for k in (start_k+1):length(splits), mk in markets
        sub = M[M.selection .== mk, :]
        tr  = sub[in.(sub.split_id, Ref(Set(splits[1:k-1]))) .& (sub.stake .> 1e-6), :]
        te  = sub[(sub.split_id .== splits[k]) .& (sub.stake .> 1e-6), :]
        nrow(te) == 0 && continue
        tilt = nrow(tr) >= min_bets ? contrarian_tilt(recent_form(tr, minimum(te.match_date); hl=hl);
                                                      β=β, scale=scale, lo=lo, hi=hi) : 1.0
        push!(tl, tilt); append!(s, tilt .* te.stake); append!(r, te.roi)
    end
    return merge(_summ(s, r), (mean_tilt = isempty(tl) ? NaN : mean(tl),))
end

"""
    cold_hot_decomposition(M, markets; hl, start_k, min_bets)

Diagnostic: split OOS bets by decision-time state (recent_form < 0 = cold) and compare growth.
Cold ≫ hot confirms the mean-reversion edge is real (not a gate artifact).
"""
function cold_hot_decomposition(M, markets; hl::Float64=45.0, start_k::Int=4, min_bets::Int=12)
    splits = sort(unique(M.split_id)); cs=Float64[]; cr=Float64[]; hs=Float64[]; hr=Float64[]
    for k in (start_k+1):length(splits), mk in markets
        sub = M[M.selection .== mk, :]
        tr  = sub[in.(sub.split_id, Ref(Set(splits[1:k-1]))) .& (sub.stake .> 1e-6), :]
        te  = sub[(sub.split_id .== splits[k]) .& (sub.stake .> 1e-6), :]
        (nrow(te) == 0 || nrow(tr) < min_bets) && continue
        if recent_form(tr, minimum(te.match_date); hl=hl) < 0
            append!(cs, te.stake); append!(cr, te.roi)
        else
            append!(hs, te.stake); append!(hr, te.roi)
        end
    end
    return (cold = _summ(cs, cr), hot = _summ(hs, hr))
end
