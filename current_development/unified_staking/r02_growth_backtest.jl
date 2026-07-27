#=
RUNNER — r02: sequential growth backtest over ALL OOS matches (Ireland, r19 iso_flat posterior):
unified structural Kelly (P)+(U-MC) vs the per-bet BayesianKelly baseline vs fixed stake.

Fairness design (everything shared across strategies):
  - same model probabilities (iso_flat per-draw 144-state score grids → p̄ and per-selection dists)
  - same commission-adjusted odds  d_eff = 1 + (d_close − 1)(1 − c)   [decisions AND settlement]
  - same book (1X2 + O/U ladder + BTTS at Betfair close), same matches, same date order
  - same settlement: W ← W · (1 − Σa + Σ a_m · d_eff_m · won_m), sequential compounding

Strategies:
  U_cap100  unified (P) at cap=1.0, k* per match via (U-MC) (S_dec=100 draws, warm-started)
  U_cap02   unified (P) at cap=0.2 (portfolio-cap memory: dominant risk lever)
  PB_full   per-bet Signals.BayesianKelly(min_edge=0.03) per selection, scaled down if Σa>1
  PB_cap02  same, proportionally scaled to Σa≤0.2
  FIX_1pct  flat 1% on every selection with p̄ − 1/d_eff ≥ 0.03

Requires the r19 session state (`ar19`, `odf`=ds1_19.odds, `lat`, `lat_ids`, `hgrid/agrid`,
`solve_P`, `mask_for`, `state_draws`, `G_growth` from l01/r01 definitions). Ran live on kaimon
2026-07-02; results pasted at the bottom.
=#

const Signals = BayesianFootball.Signals
using Random

function prep_match(mid; c=0.02, families=Set(["1X2","OverUnder","BTTS"]))
    book = odf[(odf.match_id .== mid) .& in.(odf.market_name, Ref(families)), :]
    nrow(book) == 0 && return nothing
    sort!(book, [:market_name, :market_line, :selection])
    masks = [mask_for(r.market_name, r.market_line, r.selection) for r in eachrow(book)]
    d_eff = 1.0 .+ (book.odds_close .- 1.0) .* (1 - c)
    Mmask = hcat(masks...)
    P = state_draws(mid)
    pbar = vec(mean(P, dims=2))
    won = let m = ds1_19.matches[findfirst(==(mid), ds1_19.matches.match_id), :]
        st = (hgrid .== m.home_score) .& (agrid .== m.away_score)
        [any(mk .& st) for mk in masks]
    end
    R = hcat([d_eff[m] .* masks[m] .- 1.0 for m in eachindex(masks)]...)
    return (mid=mid, d_eff=d_eff, Mmask=Mmask, P=P, pbar=pbar, won=won, R=R)
end

match_return(a, d_eff, won) = 1.0 - sum(a) + sum(a .* d_eff .* won)

function stakes_unified(pm; cap=1.0, S_dec=100, seed=7)
    astar = solve_P(pm.pbar, pm.R; cap=cap)
    sum(astar) < 1e-9 && return astar
    idx = rand(Xoshiro(seed), 1:size(pm.P,2), S_dec)
    A = Matrix{Float64}(undef, length(astar), S_dec)
    for (j,s) in enumerate(idx); A[:,j] = solve_P(view(pm.P,:,s), pm.R; cap=cap, a0=astar, iters=600); end
    ks = 0.05:0.05:1.0
    kstar = ks[argmax([mean(G_growth(k .* view(A,:,j), pm.pbar, pm.R) for j in 1:S_dec) for k in ks])]
    return kstar .* astar
end

function stakes_perbet(pm; min_edge=0.03, cap=Inf)
    dists = pm.Mmask' * pm.P
    sig = Signals.BayesianKelly(min_edge)
    a = [Signals.compute_stake(sig, vec(dists[m,:]), pm.d_eff[m]) for m in 1:length(pm.d_eff)]
    s = sum(a); s > cap && (a .*= cap / s)
    return a
end

function stakes_fixed(pm; min_edge=0.03, f=0.01)
    pmask = vec(pm.Mmask' * pm.pbar)
    return [ (pmask[m] - 1/pm.d_eff[m]) >= min_edge ? f : 0.0 for m in 1:length(pm.d_eff) ]
end

function run_backtest_r02(mids; c=0.02)
    strategies = ["U_cap100", "U_cap02", "PB_full", "PB_cap02", "FIX_1pct"]
    logW = Dict(s => Float64[] for s in strategies)
    nbets = Dict(s => 0 for s in strategies); turn = Dict(s => 0.0 for s in strategies)
    for mid in mids
        pm = prep_match(mid; c=c); pm === nothing && continue
        for (s, a) in [("U_cap100", stakes_unified(pm; cap=1.0)),
                       ("U_cap02",  stakes_unified(pm; cap=0.2)),
                       ("PB_full",  stakes_perbet(pm; cap=1.0)),
                       ("PB_cap02", stakes_perbet(pm; cap=0.2)),
                       ("FIX_1pct", stakes_fixed(pm))]
            r = match_return(a, pm.d_eff, pm.won)
            push!(logW[s], log(max(r, 1e-12)))
            nbets[s] += count(>(1e-6), a); turn[s] += sum(a)
        end
    end
    n = length(logW["U_cap100"])
    df = DataFrame(strategy=strategies)
    df.terminal_W = [round(exp(sum(logW[s])), digits=3) for s in strategies]
    df.G_per_match = [round(mean(logW[s]), digits=5) for s in strategies]
    df.max_dd = [round(1 - exp(minimum(cumsum(logW[s]) .- accumulate(max, cumsum(logW[s])))), digits=3) for s in strategies]
    df.n_bets = [nbets[s] for s in strategies]
    df.turnover_pm = [round(turn[s]/n, digits=3) for s in strategies]
    return df, n
end

# matches in date order (only those with latents + betfair book)
ord = leftjoin(DataFrame(match_id=collect(lat_ids)), ds1_19.matches[:, [:match_id, :match_date]], on=:match_id)
sort!(ord, :match_date)
mids_ord = [m for m in ord.match_id if m in Set(odf.match_id)]
res02, n02 = run_backtest_r02(mids_ord; c=0.02)
res02_c0, _ = run_backtest_r02(mids_ord; c=0.0)
println("n = $n02 matches");  show(res02; allcols=true);  show(res02_c0; allcols=true)

#=
RESULTS (kaimon 2026-07-02, threaded 16 cores, n=275 matches in date order, S_dec=100)
NOTE: the live run used the threaded variant `run_backtest_r02_mt` (@threads over matches,
per-match log-returns into a preallocated matrix, aggregated after — identical math).

c = 0.02 (2% commission on net winnings, decisions AND settlement):
  strategy   terminal_W  G_per_match  max_dd  n_bets  turnover_pm
  U_cap100        0.066     -0.00987   0.996    1484        0.328
  U_cap02         0.119     -0.00775   0.992    1093        0.151
  PB_full         0.000     -0.12031   1.000     796        0.286   ← BANKRUPT
  PB_cap02        0.573     -0.00202   0.952     796        0.142
  FIX_1pct        1.255     +0.00082   0.313     796        0.029   ← only survivor

c = 0 (no commission):
  U_cap100 0.169 · U_cap02 0.171 · PB_full 0.000 · PB_cap02 0.869 · FIX_1pct 1.499

Market-curated variant (drop 1X2, totals+BTTS book only, c=0.02, cap=0.2):
  strategy      terminal_W  G_per_match  max_dd  n_bets  turnover_pm
  U_cap02_tot        4.569     +0.00554   0.770     969        0.104  ← unified WINS
  PB_cap02_tot       3.563     +0.00464   0.663     535        0.096
  FIX_1pct_tot       1.354     +0.00111   0.264     535        0.020

READS (full discussion in NOTES.md):
 1. PB_full = independent per-bet full Kelly goes BANKRUPT — exactly the
    [[portfolio-kelly-partial-hedge]] prediction, now shown on real books.
 2. Every Kelly-sized strategy loses while flat 1% on the SAME filtered bets is +25%:
    the betting stream has positive edge, but Kelly sizes stakes by the MAGNITUDE of
    model-market divergence — which on this uncalibrated model is dominated by 1X2 bias.
    Kelly adversely selects your own biases; sizing amplifies exactly what recentring
    hasn't fixed. This is notes §8.4 (shrinkage fixes estimation error, NOT bias) made real.
 3. Unified vs per-bet at the same cap: PB_cap02 > U_cap02. The per-bet baseline's
    min_edge=0.03 filter is crude market curation the unified engine lacks — it bets
    every "positive edge" the model hallucinates (1093 vs 796 bets). The structural
    layer only pays off once p is trustworthy.
 4. Commission matters exactly as feared: PB_cap02 goes 0.869 → 0.573 at c=2%.
 5. MARKET CURATION FLIPS THE WHOLE TABLE. Restricted to totals+BTTS (the families with
    certified per-line edge per RESULTS_smile_grid), everything turns profitable and the
    ORDERING INVERTS: unified ×4.57 > per-bet ×3.56 > flat ×1.35 over 275 matches at 2%
    commission. Once the bias source is removed, the structural layer (joint solve,
    hedges, mutual-exclusivity pricing) genuinely adds ~+28% terminal wealth over
    per-bet Kelly at the same cap. Max drawdown is still 77% (full-Kelly-at-cap
    aggression) — deployment needs a lower cap or fractional overlay, but the ranking
    is the finding. Curation-by-family is a crude stand-in for per-line recentring;
    the calibrated version (split_market_pillar Gap 2) should recover 1X2 as hedging
    inventory too.
=#
