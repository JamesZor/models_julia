#=
LOADER — staking strategies + trust-blend machinery + season runner for the simulation lab.

Pure functions over `SimMatch` (no session globals — fixes the r02 template's flaw).
All strategies bet the SAME books; sequential bankroll compounding; ruin = wealth <
cfg.ruin_floor ⇒ betting frozen for the rest of the campaign (uniform convention).

Strategies raced (registry in `run_season`):
  FLAT_1pct · K_full/K_half/K_quarter (Signals.KellyCriterion, external 3% edge filter)
  BM_ana (AnalyticalShrinkageKelly 3%) · BM_num (BayesianKelly 3%, Baker–McHale numerical)
  U_cap02/U_cap05/U_cap100 (structural (P) at pbar) · U_UMC (cap 1 + (U-MC) k*)
  TRUST_U_cap02 / TRUST_UMC (per-line trust blend → coherent grid via closed-form IPF tilts
                             → unified solve; w fit walk-forward by EB partial pooling —
                             Julia port of docs/bets_multi/_verify_trust.py T4 fit_ws)

Trust-blend maths: docs/bets_multi/trust_blend_notes.md §3 (EB fit), §4 (I-projection = tilt).
The per-line intercept tilt has a CLOSED FORM per step: for mask m with current mass c and
target t, γ = logit(t) − logit(c) hits t exactly holding the rest fixed; cycling masks = IPF.
=#

using Random
using Statistics
using LogExpFunctions: logit, logistic
using BayesianFootball
const SIG = BayesianFootball.Signals

# ---------- shared per-match pieces ----------

"Per-selection posterior-draw probabilities (11 × S) — the r02 `Mmask' * P` trick."
sel_dists(sm::SimMatch) = MMASK' * sm.P

match_return(a::Vector{Float64}, sm::SimMatch) =
    1.0 - sum(a) + sum(a[m] * sm.d[m] * sm.won[m] for m in 1:11)

"Rescale per-bet stakes if they exceed a total-exposure guard (full Kelly across 11 sels)."
function guard!(a::Vector{Float64}; cap::Float64=0.98)
    s = sum(a)
    s > cap && (a .*= cap / s)
    return a
end

# ---------- per-bet strategies ----------

function stakes_flat(sm::SimMatch, pbar_sel; f=0.01, min_edge=0.03)
    a = zeros(11)
    for m in 1:11
        pbar_sel[m] - 1.0 / sm.d[m] >= min_edge && (a[m] = f)
    end
    return guard!(a)
end

"Any src/signals AbstractSignal per selection; optional external edge filter (KellyCriterion
has no built-in one — keeps the 3% filter uniform across per-bet strategies)."
function stakes_signal(sm::SimMatch, dists, sig; min_edge_ext=0.0)
    a = zeros(11)
    for m in 1:11
        if min_edge_ext > 0.0
            mean(view(dists, m, :)) - 1.0 / sm.d[m] < min_edge_ext && continue
        end
        a[m] = SIG.compute_stake(sig, view(dists, m, :), sm.d[m])
    end
    return guard!(a)
end

# ---------- unified strategies ----------

stakes_unified(sm::SimMatch; cap=0.2, p=sm.pbar) = solve_P(p, sm.R; cap=cap)

"(U-MC): deterministic stride subsample of draws (no RNG — keeps strategies independent)."
function stakes_umc(sm::SimMatch; cap=1.0, S_dec=50, kgrid=0.05:0.05:1.0,
                    P=sm.P, pbar=sm.pbar)
    astar = solve_P(pbar, sm.R; cap=cap)
    sum(astar) < 1e-10 && return astar
    idx = round.(Int, range(1, size(P, 2), length=min(S_dec, size(P, 2))))
    A = Matrix{Float64}(undef, 11, length(idx))
    for (j, s) in enumerate(idx)
        A[:, j] = solve_P(view(P, :, s), sm.R; cap=cap, a0=astar, iters=600)
    end
    ks = collect(kgrid)
    Ψ = [mean(G_growth(k .* view(A, :, j), pbar, sm.R) for j in 1:length(idx)) for k in ks]
    return ks[argmax(Ψ)] .* astar
end

# ---------- trust blend: targets, coherent grid, EB fit ----------

"Blended per-unit targets w·p_model + (1−w)·q_mkt; 1X2 triple renormalized to sum to 1."
function blend_targets(pbar_sel, q_sel, w::Vector{Float64})
    t = [w[u] * pbar_sel[UNIT_REP_SEL[u]] + (1.0 - w[u]) * q_sel[UNIT_REP_SEL[u]] for u in 1:7]
    s = t[1] + t[2] + t[3]
    t[1] /= s; t[2] /= s; t[3] /= s
    return t
end

# constrained masks: home, draw (away implied by 1X2 renorm), O1.5/2.5/3.5 over, btts_yes
const TILT_MASKS = (SEL_MASKS[1], SEL_MASKS[2], SEL_MASKS[4], SEL_MASKS[6], SEL_MASKS[8], SEL_MASKS[10])
const TILT_UNIT = (1, 2, 4, 5, 6, 7)   # unit index of each constrained mask's target

"""
Closed-form IPF: cycle intercept tilts γ_j = logit(t_j) − logit(current mass) until all six
blended targets are hit (I-projection onto the target marginals — trust_blend_notes §4).
Returns the multiplier vector over the 144 states (apply to pbar and/or each draw column).
"""
function coherent_multiplier(pbar::Vector{Float64}, targets::Vector{Float64};
                             cycles=10, tol=1e-8)
    g = copy(pbar)
    mult = ones(length(g))
    for _ in 1:cycles
        moved = 0.0
        for j in 1:6
            m = TILT_MASKS[j]
            cur = sum(view(g, m))
            t = clamp(targets[TILT_UNIT[j]], 1e-9, 1 - 1e-9)
            δγ = logit(t) - logit(clamp(cur, 1e-9, 1 - 1e-9))
            e = exp(δγ)
            g[m] .*= e
            mult[m] .*= e
            z = sum(g)
            g ./= z
            mult ./= z
            moved = max(moved, abs(δγ))
        end
        moved < tol && break
    end
    return mult
end

"Trust history: per unit, (p_model, q_mkt, y) of the representative selection."
struct TrustHist
    p::Vector{Vector{Float64}}
    q::Vector{Vector{Float64}}
    y::Vector{Vector{Float64}}
end
TrustHist() = TrustHist([Float64[] for _ in 1:7], [Float64[] for _ in 1:7], [Float64[] for _ in 1:7])

function push_hist!(h::TrustHist, sm::SimMatch, pbar_sel)
    for u in 1:7
        m = UNIT_REP_SEL[u]
        push!(h.p[u], pbar_sel[m]); push!(h.q[u], sm.q_mkt[m]); push!(h.y[u], Float64(sm.won[m]))
    end
end

"""
EB partial-pooled per-unit trust (port of _verify_trust.py T4 fit_ws): per-unit Bernoulli
log-lik over a w grid, logit-normal prior with (w0, τ) by marginal likelihood, posterior mean.
`halflife` (in observations) exponentially down-weights old observations in the log-lik —
Inf (default) = the untouched static fit; finite H targets drifting per-line bias (r03/E2b).
"""
function fit_trust_eb(h::TrustHist; wgrid=collect(0.0:0.005:1.0),
                      w0grid=range(-2.0, 2.0, length=17), τgrid=(0.25, 0.5, 1.0, 2.0),
                      halflife::Real=Inf)
    nw = length(wgrid)
    LL = zeros(7, nw)
    for u in 1:7
        n = length(h.y[u])
        n == 0 && continue
        for (wi, w) in enumerate(wgrid)
            s = 0.0
            @inbounds for i in 1:n
                p̃ = clamp(w * h.p[u][i] + (1.0 - w) * h.q[u][i], 1e-9, 1 - 1e-9)
                ll = h.y[u][i] * log(p̃) + (1.0 - h.y[u][i]) * log1p(-p̃)
                s += isinf(halflife) ? ll : exp2(-(n - i) / halflife) * ll
            end
            LL[u, wi] = s
        end
    end
    zg = logit.(clamp.(wgrid, 1e-4, 1 - 1e-4))
    best, bw0, bτ = -Inf, 0.0, 1.0
    for w0 in w0grid, τ in τgrid
        lp = @. -0.5 * ((zg - w0) / τ)^2
        marg = 0.0
        for u in 1:7
            mx = maximum(view(LL, u, :))
            marg += mx + log(sum(exp.(view(LL, u, :) .- mx .+ lp)))
        end
        marg > best && ((best, bw0, bτ) = (marg, w0, τ))
    end
    lp = @. -0.5 * ((zg - bw0) / bτ)^2
    w = zeros(7)
    for u in 1:7
        mx = maximum(view(LL, u, :))
        post = exp.(view(LL, u, :) .- mx .+ lp)
        post ./= sum(post)
        w[u] = sum(post .* wgrid)
    end
    return w, (w0=bw0, τ=bτ)
end

# ---------- season runner ----------

const STRATEGY_NAMES = ["FLAT_1pct", "K_full", "K_half", "K_quarter", "BM_ana", "BM_num",
                        "U_cap02", "U_cap05", "U_cap100", "U_UMC",
                        "TRUST_U_cap02", "TRUST_UMC"]

"""
run_season(cfg, seed; ...) → (results = Dict(name => (logw, n_bets, turnover, ruined)),
                              w_trace = Vector{(i, w)}, w_final)
Identical books for all strategies; sequential compounding; ruin-freeze at cfg.ruin_floor.
Trust w warm-started on cfg.n_prehist no-betting matches, refit every `refit_every`.
"""
function run_season(cfg::SimConfig, seed::Integer; S_dec=50, kgrid=0.05:0.05:1.0,
                    refit_every=30, strategies=STRATEGY_NAMES)
    rng = Xoshiro(seed)
    hist = TrustHist()
    if cfg.n_prehist > 0
        for sm in simulate_campaign(cfg, rng; n_matches=cfg.n_prehist, S=16)
            push_hist!(hist, sm, MMASK' * sm.pbar)
        end
    end
    w_units = cfg.n_prehist > 0 ? fit_trust_eb(hist)[1] : fill(logistic(-1.0), 7)

    ms = simulate_campaign(cfg, rng)
    ns = length(strategies)
    logw = Dict(s => Float64[] for s in strategies)
    nbets = Dict(s => 0 for s in strategies)
    turn = Dict(s => 0.0 for s in strategies)
    ruined = Dict(s => false for s in strategies)
    cumW = Dict(s => 1.0 for s in strategies)
    w_trace = Vector{Tuple{Int,Vector{Float64}}}()
    needs_eb = "TRUST_U_cap02" in strategies || "TRUST_UMC" in strategies
    needs_t05 = "TRUST05_U_cap02" in strategies       # hard-coded w = 0.5 control (E3)
    needs_cur = "CURATED05_U_cap02" in strategies     # w = 0 on 1X2, 0.5 on totals/BTTS (E4)
    w_half = fill(0.5, 7)
    w_cur = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5]

    for (i, sm) in enumerate(ms)
        dists = sel_dists(sm)
        pbar_sel = MMASK' * sm.pbar
        if needs_eb && i > 1 && (i - 1) % refit_every == 0
            w_units = fit_trust_eb(hist)[1]
            push!(w_trace, (i, copy(w_units)))
        end
        mult = needs_eb ?
            coherent_multiplier(sm.pbar, blend_targets(pbar_sel, sm.q_mkt, w_units)) : ones(1)
        mult05 = needs_t05 ?
            coherent_multiplier(sm.pbar, blend_targets(pbar_sel, sm.q_mkt, w_half)) : ones(1)
        mult_cur = needs_cur ?
            coherent_multiplier(sm.pbar, blend_targets(pbar_sel, sm.q_mkt, w_cur)) : ones(1)
        for s in strategies
            if ruined[s]
                push!(logw[s], 0.0)
                continue
            end
            a =
                s == "FLAT_1pct"  ? stakes_flat(sm, pbar_sel) :
                s == "K_full"     ? stakes_signal(sm, dists, SIG.KellyCriterion(1.0); min_edge_ext=0.03) :
                s == "K_half"     ? stakes_signal(sm, dists, SIG.KellyCriterion(0.5); min_edge_ext=0.03) :
                s == "K_quarter"  ? stakes_signal(sm, dists, SIG.KellyCriterion(0.25); min_edge_ext=0.03) :
                s == "BM_ana"     ? stakes_signal(sm, dists, SIG.AnalyticalShrinkageKelly(0.03)) :
                s == "BM_num"     ? stakes_signal(sm, dists, SIG.BayesianKelly(0.03)) :
                s == "U_cap02"    ? stakes_unified(sm; cap=0.2) :
                s == "U_cap05"    ? stakes_unified(sm; cap=0.5) :
                s == "U_cap100"   ? stakes_unified(sm; cap=1.0) :
                s == "U_UMC"      ? stakes_umc(sm; cap=1.0, S_dec=S_dec, kgrid=kgrid) :
                s == "TRUST_U_cap02" ? stakes_unified(sm; cap=0.2, p=normalize_mult(sm.pbar, mult)) :
                s == "TRUST05_U_cap02" ? stakes_unified(sm; cap=0.2, p=normalize_mult(sm.pbar, mult05)) :
                s == "CURATED05_U_cap02" ? stakes_unified(sm; cap=0.2, p=normalize_mult(sm.pbar, mult_cur)) :
                s == "TRUST_UMC"  ? stakes_umc(sm; cap=1.0, S_dec=S_dec, kgrid=kgrid,
                                               P=apply_mult(sm.P, mult),
                                               pbar=normalize_mult(sm.pbar, mult)) :
                error("unknown strategy $s")
            r = max(match_return(a, sm), 1e-12)
            push!(logw[s], log(r))
            cumW[s] *= r
            nbets[s] += count(>(1e-8), a)
            turn[s] += sum(a)
            cumW[s] < cfg.ruin_floor && (ruined[s] = true)
        end
        push_hist!(hist, sm, pbar_sel)   # settled AFTER betting — no leakage
    end
    results = Dict(s => (logw=logw[s], n_bets=nbets[s], turnover=turn[s], ruined=ruined[s])
                   for s in strategies)
    return (results=results, w_trace=w_trace, w_final=w_units)
end

normalize_mult(p::Vector{Float64}, mult) = (g = p .* mult; g ./ sum(g))
function apply_mult(P::Matrix{Float64}, mult)
    Q = P .* mult
    for j in 1:size(Q, 2)
        Q[:, j] ./= sum(view(Q, :, j))
    end
    return Q
end

# ---------- metrics ----------

"Per-strategy summary of one logw vector: terminal W, growth/match, maxDD (r02 formulas)."
function summarize_logw(logw::Vector{Float64})
    cw = cumsum(logw)
    peak = accumulate(max, cw)
    return (terminal_W=exp(sum(logw)), G_per_match=mean(logw),
            max_dd=1.0 - exp(minimum(cw .- peak)))
end

# ---------- oracle trust (for the p4 plot) ----------

"""
Growth-maximising per-unit w on a large offline sample (truth known). Uses the model's
pbar and market q per match; expected growth under TRUE selection probabilities.
"""
function oracle_trust(cfg::SimConfig; n=40_000, seed=1, wgrid=collect(0.0:0.02:1.0))
    rng = Xoshiro(seed)
    ms = simulate_campaign(cfg, rng; n_matches=n, S=16)
    p = [Float64[] for _ in 1:7]
    q = [Float64[] for _ in 1:7]
    pt = [Float64[] for _ in 1:7]
    dd = [Float64[] for _ in 1:7]
    for sm in ms
        ps = MMASK' * sm.pbar
        for u in 1:7
            m = UNIT_REP_SEL[u]
            push!(p[u], ps[m]); push!(q[u], sm.q_mkt[m])
            push!(pt[u], sm.p_true[m]); push!(dd[u], sm.d[m])
        end
    end
    worac = zeros(7)
    for u in 1:7
        best, bw = -Inf, 0.0
        for w in wgrid
            g = 0.0
            @inbounds for i in eachindex(pt[u])
                b = dd[u][i] - 1.0
                p̃ = w * p[u][i] + (1.0 - w) * q[u][i]
                f = max(0.0, (p̃ * dd[u][i] - 1.0) / b)
                f = min(f, 0.98)
                g += pt[u][i] * log1p(b * f) + (1.0 - pt[u][i]) * log1p(-f)
            end
            g /= length(pt[u])
            g > best && ((best, bw) = (g, w))
        end
        worac[u] = bw
    end
    return worac
end

# ---------- sanity checks (verification item 2) ----------

"""
sanity_checks(): (a) zero-noise/zero-bias world ⇒ every strategy stakes ≈ nothing (vig
kills all edges when model = market = truth); (b) σ_mod = 0 (model knows truth) ⇒ plug-in
Kelly strongly profitable in expectation. Prints PASS/FAIL lines.
"""
function sanity_checks(; seed=42)
    cfg0 = SimConfig(σ_mkt=0.0, σ_mod=0.0, γ_tot=0.0, γ_btts=0.0, σ_post=1e-3,
                     n_matches=200, n_prehist=0)
    rng = Xoshiro(seed)
    tot = 0.0
    for sm in simulate_campaign(cfg0, rng)
        tot += sum(stakes_signal(sm, sel_dists(sm), SIG.KellyCriterion(1.0); min_edge_ext=0.0))
        tot += sum(stakes_unified(sm; cap=1.0))
    end
    println(tot < 1e-6 ? "PASS" : "FAIL", " (a) no-edge world total stake = $tot")

    cfg1 = SimConfig(σ_mkt=0.08, σ_mod=0.0, γ_tot=0.0, γ_btts=0.0, σ_post=1e-3,
                     n_matches=2000, n_prehist=0)
    rng = Xoshiro(seed + 1)
    ev = 0.0; nb = 0
    for sm in simulate_campaign(cfg1, rng)
        a = stakes_signal(sm, sel_dists(sm), SIG.KellyCriterion(1.0); min_edge_ext=0.03)
        for m in 1:11
            a[m] > 1e-8 && (ev += a[m] * (sm.p_true[m] * sm.d[m] - 1.0); nb += 1)
        end
    end
    println(ev > 0 ? "PASS" : "FAIL", " (b) oracle-model Kelly staked EV = $(round(ev, digits=3)) over $nb bets")
end
