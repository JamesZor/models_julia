#=
RUNNER — first real-data run of the unified staking engine (l01) on the r19 iso_flat posterior.
Ran live on the kaimon session 2026-07-02 (results pasted at the bottom).

Uses the r19 grid experiment (hierarchical-iso grid, Ireland; iso_flat = the keeper cell) as the
L1 posterior source. If the r19 session state is gone, reload iso_flat from
./data/hier_iso_grid_ireland/ via Experiments loading + include l09 first (needed to deserialize).

Run:
    include("current_development/split_market_pillar/l09_hier_iso_poisson.jl")  # only if loading from disk
    include("current_development/unified_staking/l01_structural_kelly.jl")
    include("current_development/unified_staking/r01_r19_realmatch_example.jl")
=#

using BayesianFootball
using DataFrames

const Data        = BayesianFootball.Data
const Experiments = BayesianFootball.Experiments

# ==========================================
# 1. DATA — Betfair-swapped eval store + iso_flat latents
# ==========================================
ds = Data.load_datastore_cached(Data.Ireland())
odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds1 = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)

# iso_flat experiment: from live session (`ar19`) or from disk
iso_flat_res = isdefined(Main, :ar19) ?
    ar19[findfirst(r -> r.config.name == "iso_flat", ar19)] :
    error("load iso_flat from ./data/hier_iso_grid_ireland/ (include l09 first)")
lat = Experiments.extract_oos_predictions(ds1, iso_flat_res)

# ==========================================
# 2. VALIDATE the (P) solver vs Long's closed form (notes Example A)
# ==========================================
R_A = [1.8 -1.0; -1.0 -1.0; -1.0 2.0]
p_A = [0.40, 0.25, 0.35]
aA = solve_P(p_A, R_A)
@assert maximum(abs.(aA .- [0.1115, 0.0808])) < 1e-3 "solver disagrees with Long closed form"
println("[OK] solver ≡ Long closed form: a* = ", round.(aA, digits=4))

# ==========================================
# 3. TWO REAL MATCHES (both with the full 9 market/line book at Betfair close)
# ==========================================
MID_FAV  = 13250747   # 2025-06-27 shamrock-rovers vs waterford-fc (home 1.31) — result 1-0
MID_EVEN = 13250681   # 2025-06-20 galway-united vs st-patricks-athletic (home 3.08) — result 3-1

for (tag, mid) in [("FAV", MID_FAV), ("EVEN", MID_EVEN)], cap in (1.0, 0.2)
    res = run_match(ds1.odds, lat.df, mid; cap=cap)
    stl = settle(ds1.odds, ds1.matches, mid, res)
    println("\n===== $tag  cap=$cap  k*=$(res.kstar)  Σa=$(round(res.total,digits=3))  " *
            "G=$(round(res.G,digits=4))  settled $(stl.score) → W=$(stl.W) =====")
    show(res.book; allrows=true, allcols=true)
    println()
end

#=
RESULTS (kaimon, 2026-07-02, iso_flat 3200 draws, S_dec=200)

FAV — Shamrock Rovers 1-0 Waterford (home close 1.31):
  cap=1.0: k*=0.95, Σa*=1.00 (budget BINDS, cash 0), G=0.117, settled W=1.186.
    10-market book. Includes a genuine Whelan negative-EV hedge: over_15 EV=-0.108
    staked a*=0.249 to cushion the under-ladder. Model "edges" on 1X2 are huge
    (away p=0.179 vs market 0.075, EV +141%) — UNCALIBRATED per r13 (1X2 has no
    certified edge) — the engine faithfully amplifies model bias. Caveat emptor (§8.4).
  cap=0.2: k*=1.00, settled W=1.059. The cap DROPS the -EV hedge (with a tight budget,
    capital goes to the highest-edge claims only — hedging is a luxury of the loose
    budget). k*=1 because when the cap binds, per-draw decisions all hit the same
    boundary → no decision dispersion → (U-MC) sees nothing to shrink. Cap and
    shrinkage are SUBSTITUTE risk controls; the cap dominates when it binds.

EVEN — Galway United 3-1 St Patrick's (home close 3.08):
  cap=1.0: k*=0.79, Σa*=1.00, settled W=1.076.
    The optimizer staked BOTH over_35 (4.75) AND under_35 (1.283): 1/4.75+1/1.283=0.99 —
    the de-vigged TWA close pair has NO overround (sum q < 1), so dutching both sides is
    a ~1% cash-replication arb, and (P) loaded 88% of bankroll into it. This is the §2.1
    rank/degeneracy story live: at Betfair close, complementary pairs can price to <1
    (thin book / TWA artefact). REALITY CHECK: 2-5% commission on net winnings kills a
    1% arb, and TWA quotes are not simultaneously executable — commission MUST enter R
    before any live use, and near-degenerate pairs should be screened.

Overall reads:
  1. Machinery works end-to-end on real posteriors + real Betfair books (solver exact vs
     Long; hedges and support selection behave per theory).
  2. k* ∈ [0.79, 1.0] — the MCMC posterior is too TIGHT for shrinkage to be the main
     risk control here; the real protections are (a) per-line recentring calibration
     (kills the fake 1X2 edge), (b) the portfolio cap, (c) commission-aware R.
     Matches [[staking-research-conclusions]] + [[calibrate-centre-edge-in-tails]].
  3. Both books settled profitably (W 1.06–1.19) but n=2 is anecdote, not evidence —
     the r24-style growth backtest over all 288 OOS matches is the real test.
=#
