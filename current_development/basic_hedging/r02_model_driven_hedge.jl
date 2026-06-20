#=
r02_model_driven_hedge.jl  —  PHASE 2: in-game-model-driven position management.

Goal (user's): use the in-play intensity model to decide when to HEDGE OUT / hold / add to a position
placed by the pre-game DCMH model, as game state and time-decay change the EV vs the live market.

Pipeline per OPEN bet (selection s, match m, entry odds O, stake S):
  walk the match's in-play ticks (panel rows: t_w wall-clock for pricing, t_m match-minute for the model,
  plus live state gh,ga,reds,pregame λ). At each tick:
    1. in-play model (ch, :linear config) -> posterior-mean expected REMAINING goals per side (μ_h, μ_a)
    2. -> remaining-goals score matrix -> P_model(s | current score)   (`model_prob`)
    3. live market -> implied prob 1/price        (`exit_price` from bf_ip)
    4. model edge e = P_model - 1/price
    5. EXIT (lay φ of the position) the first time e <= τ; else hold to settlement.

KEY RESULT (Ireland DCMH backtest, 258 matches, portfolio-capped sequential log-growth):
  * AT-signal execution prints 6.8 but that is STALE-PRICE LOOKAHEAD (the l04 lesson). With realistic
    FORWARD execution (lay at the next price >= signal+lag) it is STABLE across 1/3/5-min lag (~3.74),
    NOT collapsing like the l04 trading backtest — because we manage an EXISTING position on a genuine
    edge-decay signal, not a microstructure blip.
  * Honest comparison (cap 0.20, full exit φ=1, forward 3-min):
        hold                3.076
        fixed clock @70'    2.667   (WORSE than hold)
        fixed clock @80'    3.613
        model exit τ=0      3.734
        model exit τ=-0.05  4.423   <- best threshold
    Co-sizing: τ=-0.05 full exit optimal at cap 0.25 -> 4.549 (≈4.3× hold terminal wealth).
  * Optimal τ is a small NEGATIVE edge (exit only once the market clearly overtakes the model, ~5pts),
    not zero — avoids churning out on noise.
  * When the model fires, FULL exit beats partial (4.42 > 3.82 at φ=0.5). This REVERSES the fixed-clock
    finding: the clock hedges partially out of ignorance; the model KNOWS the edge is gone, so it exits
    decisively. Partial hedging is what you do WITHOUT a model.

CAVEATS: in-sample, single league, 258 matches; τ & cap tuned in-sample (need CV/OOS — use l07 harness);
forward-fill 3-min lag + ±6-min price match is realistic but Betfair LTP != tradeable lay (real lay pays
the spread). Signal is causal (score/time known at the tick; execution lagged).

INPUTS (built upstream, see r00_explore / r01): `ch` (fitted :linear InPlayIntensityConfig chain),
`inp` (InPlayInputs — for x_center/x_scale standardisation), `panel` (per-tick state w/ t_w,t_m,gh,ga,
reds,pg_λ_h,pg_λ_a), `bf_ip` (in-play ticks), `L2` (DCMH ledger w/ :h held-pnl, :date), `cashout_pnl`,
`exit_price`, `COMM`. Model loader: ../match_inplay_explore/l03_inplay_turing.jl.
=#

using Distributions, Statistics, LinearAlgebra, DataFrames

# ---- in-play model -> expected remaining goals per side (posterior mean of the :linear chain) ----
const ΑV = vec(Array(ch[:α]))
const BM = vec(mean(reduce(hcat, [vec(Array(ch[Symbol("β[$i]")])) for i in 1:7]), dims = 1))  # 7 means
const XC = inp.x_center; const XS = inp.x_scale

"Expected remaining goals for one side given the live state (standardised design = model's)."
function side_mu(t_m, is_home, trailing, leading, man_adv, log_pg)
    raw  = [t_m, t_m^2, Float64(is_home), Float64(trailing), Float64(leading), Float64(man_adv), log_pg]
    xstd = (raw .- XC) ./ XS
    off  = log(max((90.0 - t_m) / 90.0, 0.05))
    exp(clamp(mean(ΑV) + dot(BM, xstd) + off, -20, 20))
end

"Both sides at a tick -> (μ_home, μ_away) remaining-goal intensities."
function tick_mu(gh, ga, hreds, areds, pg_h, pg_a, t_m)
    (side_mu(t_m, 1, gh < ga, gh > ga, areds - hreds, log(pg_h)),
     side_mu(t_m, 0, ga < gh, ga > gh, hreds - areds, log(pg_a)))
end

score_mat(μh, μa; K = 12) = pdf.(Poisson(μh), 0:K) * pdf.(Poisson(μa), 0:K)'

"P_model for a market selection over the remaining-goals matrix P, given current score (gh,ga)."
function model_prob(sel::Symbol, P, gh, ga)
    K = size(P, 1) - 1; s = String(sel)
    if startswith(s, "over_") || startswith(s, "under_")
        L = parse(Float64, replace(s, "over_" => "", "under_" => "")) / 10
        T = gh + ga; pov = 0.0
        for i in 0:K, j in 0:K; (T + i + j) > L && (pov += P[i+1, j+1]); end
        return startswith(s, "over_") ? pov : 1 - pov
    elseif sel in (:home, :draw, :away)
        ph = pd = pa = 0.0
        for i in 0:K, j in 0:K
            d = (gh + i) - (ga + j); d > 0 ? (ph += P[i+1, j+1]) : d == 0 ? (pd += P[i+1, j+1]) : (pa += P[i+1, j+1])
        end
        return sel === :home ? ph : sel === :draw ? pd : pa
    elseif sel in (:btts_yes, :btts_no)
        py = 0.0
        for i in 0:K, j in 0:K; ((gh + i) >= 1 && (ga + j) >= 1) && (py += P[i+1, j+1]); end
        return sel === :btts_yes ? py : 1 - py
    end
    error("unknown selection $sel")
end

# ---- model-driven exit with FORWARD (realistic) execution ----
const PMAP = Dict(mid => sort(panel[panel.match_id .== mid, :], :t_m) for mid in unique(panel.match_id))

"""
    model_exit_fwd(r; τ, φ, t_min, lag_min) -> per-unit-stake pnl

Signal: first tick (t_m ≥ t_min) where model edge `P_model − 1/price ≤ τ`. EXECUTE the lay of fraction φ
at the next available price ≥ signal+lag_min (forward execution — no stale-price lookahead). Else hold.
`r.h` is the held-to-settlement pnl per unit stake.
"""
function model_exit_fwd(r; τ = -0.05, φ = 1.0, t_min = 25.0, lag_min = 3.0)
    haskey(PMAP, r.match_id) || return r.h
    signalled = false; t_sig = 0.0
    for row in eachrow(PMAP[r.match_id])
        row.t_m > 85 && break
        px = exit_price(r.match_id, r.selection, row.t_w; window = 6.0); ismissing(px) && continue
        if !signalled
            row.t_m < t_min && continue
            P = score_mat(tick_mu(row.gh, row.ga, row.home_reds, row.away_reds, row.pg_λ_h, row.pg_λ_a, row.t_m)...)
            (model_prob(r.selection, P, row.gh, row.ga) - 1/px <= τ) && (signalled = true; t_sig = row.t_m)
        elseif row.t_m >= t_sig + lag_min
            return φ * cashout_pnl(r, px) + (1 - φ) * r.h
        end
    end
    return r.h
end

# ---- ADD side (symmetric): back MORE when the model edge GROWS past τ_add ----
# FINDING: adding HURTS growth (3.08 hold -> 2.33) even though the adds are +EV (ROI +11.5%, hit 45.6%
# @ avg odds 4.95). Textbook ROI-vs-growth: +EV but growth-NEGATIVE because it piles high-variance
# correlated exposure onto an already-cap-sized book; the convex log penalty exceeds the EV. The in-game
# model is valuable on the REDUCE side only — exit on edge-decay (variance↓, growth↑), never press adds.
"Back extra size `a` (units of original stake) at the forward live price the first tick model edge ≥ τ_add."
function model_add_fwd(r; τ_add = 0.06, a = 0.5, t_min = 25.0, lag_min = 3.0, t_max = 80.0)
    haskey(PMAP, r.match_id) || return r.h
    signalled = false; t_sig = 0.0
    for row in eachrow(PMAP[r.match_id])
        row.t_m > t_max && break
        px = exit_price(r.match_id, r.selection, row.t_w; window = 6.0); ismissing(px) && continue
        if !signalled
            row.t_m < t_min && continue
            P = score_mat(tick_mu(row.gh, row.ga, row.home_reds, row.away_reds, row.pg_λ_h, row.pg_λ_a, row.t_m)...)
            (model_prob(r.selection, P, row.gh, row.ga) - 1/px >= τ_add) && (signalled = true; t_sig = row.t_m)
        elseif row.t_m >= t_sig + lag_min
            return r.h + a * (r.is_winner ? (px - 1.0) : -1.0)   # added back bet, settled at final outcome
        end
    end
    return r.h
end

# ---- DISTRIBUTIONAL sizing (Baker–McHale BayesianKelly over per-draw P_model) ----
# FINDING: does NOT improve growth (4.29 vs 4.42 mean-based full-exit). At the τ=−0.05 trigger the target
# chooses FULL exit 86% of the time (mean φ=0.88) because a gone edge ⇒ f_target≈0 regardless of
# confidence; the :linear posterior is tight (P_model std≈0.035) so shrinkage (~13%) rarely reaches the
# partial zone. Distributional sizing only matters with a WIDER posterior (sparse/hierarchical/multi-league).
# Keep the simple validated rule. Needs full posterior draws (αd, Bd thinned to ~200) + Optim.
using Optim
const ΑD = let s = length(ΑV); αraw = vec(Array(ch[:α])); αraw[1:max(1, s ÷ 200):s] end
const BD = let s = length(ΑV); Bm = reduce(hcat, [vec(Array(ch[Symbol("β[$i]")])) for i in 1:7]); Bm[1:max(1, s ÷ 200):s, :] end
const SD = length(ΑD)

function side_mu_draws(t_m, is_home, trailing, leading, man_adv, log_pg)
    raw = [t_m, t_m^2, Float64(is_home), Float64(trailing), Float64(leading), Float64(man_adv), log_pg]
    xstd = (raw .- XC) ./ XS; off = log(max((90.0 - t_m) / 90.0, 0.05))
    exp.(clamp.(ΑD .+ BD * xstd .+ off, -20, 20))
end
function pmodel_draws(sel, gh, ga, hreds, areds, pg_h, pg_a, t_m)
    μh = side_mu_draws(t_m, 1, gh < ga, gh > ga, areds - hreds, log(pg_h))
    μa = side_mu_draws(t_m, 0, ga < gh, ga > gh, hreds - areds, log(pg_a))
    [model_prob(sel, pdf.(Poisson(μh[s]), 0:12) * pdf.(Poisson(μa[s]), 0:12)', gh, ga) for s in 1:SD]
end
"Baker–McHale BayesianKelly target fraction over a posterior P-distribution (kelly.jl, min_edge=0)."
function bkelly(dist, odds)
    b = odds - 1.0; b <= 0 && return 0.0
    p = mean(dist); s_mean = max(0.0, p - (1 - p)/b); s_mean <= 1e-6 && return 0.0
    naive = [max(0.0, q - (1 - q)/b) for q in dist]
    obj(k) = (u = 0.0; for sq in naive; a = k*sq; a >= 0.999 && return Inf; u += p*log(1 + b*a) + (1 - p)*log(1 - a); end; -u/length(naive))
    s_mean * Optim.minimizer(optimize(obj, 0.0, 1.0))
end
"Continuous lay-off toward the uncertainty-aware target f_target = bkelly(P_model_draws, live odds)."
function dist_exit_fwd(r; τ_act = -0.05, t_min = 25.0, lag_min = 3.0, t_max = 85.0)
    haskey(PMAP, r.match_id) || return r.h
    f_entry = r.stake; signalled = false; t_sig = 0.0
    for row in eachrow(PMAP[r.match_id])
        row.t_m > t_max && break
        px = exit_price(r.match_id, r.selection, row.t_w; window = 6.0); ismissing(px) && continue
        if !signalled
            row.t_m < t_min && continue
            Pm = score_mat(tick_mu(row.gh, row.ga, row.home_reds, row.away_reds, row.pg_λ_h, row.pg_λ_a, row.t_m)...)
            (model_prob(r.selection, Pm, row.gh, row.ga) - 1/px <= τ_act) && (signalled = true; t_sig = row.t_m)
        elseif row.t_m >= t_sig + lag_min
            d = pmodel_draws(r.selection, row.gh, row.ga, row.home_reds, row.away_reds, row.pg_λ_h, row.pg_λ_a, row.t_m)
            φ = 1 - clamp(bkelly(d, px) / max(f_entry, 1e-9), 0, 1)
            return φ * cashout_pnl(r, px) + (1 - φ) * r.h
        end
    end
    return r.h
end

# ---- portfolio-capped sequential log-growth on a per-unit-stake pnl column ----
function growth_col(df, col; cap = 0.2)
    order = unique(df.match_id); bym = Dict(k.match_id => v for (k, v) in pairs(groupby(df, :match_id)))
    rs = Float64[]
    for mid in order
        sub = bym[mid]; tot = sum(sub.stake); sc = tot > cap ? cap/tot : 1.0
        push!(rs, sum(sub.stake[i] * sc * sub[i, col] for i in 1:nrow(sub)))
    end
    any(rs .<= -1.0) ? -Inf : sum(log1p.(rs))
end

# ----------------------------------------------------------------- run ----
L2.me_fwd = [model_exit_fwd(r; τ = -0.05, φ = 1.0, lag_min = 3.0) for r in eachrow(L2)]
println("hold              : ", round(growth_col(L2, :h),      digits = 3))
println("model exit τ=-.05 : ", round(growth_col(L2, :me_fwd), digits = 3))
for c in 0.15:0.05:0.35
    println("  cap=$c -> ", round(growth_col(L2, :me_fwd; cap = c), digits = 3))
end
