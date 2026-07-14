# inplay_scottish — running log

In-play layer for Scottish lower leagues (56 = League One, 57 = League Two).
Plan: `~/.claude/plans/i-m-looking-at-the-shimmying-charm.md` (session 2026-07-14).

## Purpose (locked)

Fair-value in-play score matrix **P_t** for **position management of the pregame book**
(π(ω) payoff-vector rebalancer + validated exit signal), NOT in-play value betting —
the concept map closed Scottish in-play trading ("Scotland is a pre-game market").
Thin Betfair LTP is enough: we need fair value, not fills.

## Design decisions (locked with user)

1. **Bayesian form: compose posteriors.** Pregame λ posterior draws (from the Scottish
   grid winner) × NHPP multiplier posterior draws → per-draw remaining-score matrix →
   full PPD per market line. No per-match MCMC at match time. (Modular/"cut" Bayes —
   see RESEARCH.md.)
2. **Training target: outcomes via NHPP** on incident goal times (market-free).
   Betfair LTP is *evaluation only* (thin: median ~49 in-play prints MATCH_ODDS,
   ~26 OU2.5; inversion unidentified in >50% of bins per the liquidity audit).
3. **Model form settled by the Ireland stream** (`../match_inplay_explore/`, do not
   relitigate): observable-covariate regression (no latent state/filters), **global**
   (team hierarchies hurt OOS), **linear game state**, **NHPP δ_time** bins (fixes the
   late-game 3.4σ bias; post-HT spike +0.30, late surge +0.24).

## Data facts (betdb, 2026-07-14)

| | t=56 | t=57 |
|---|---|---|
| finished matches | 985 | 985 |
| with incidents | 702 — **holes: season 52605 (71/180), 77129 (16/175)** | 985 complete |
| betfair LTP | ~890 matches, 9-market ladder | **only 140** |
| bet365 closing (sofascore) | complete incl. 11-line goals ladder | same |

⇒ train on incidents (57 full + 56's 4 good seasons); evaluate vs Betfair on 56.
56 incident holes flagged to user for a re-scrape.

## Infrastructure

- **homelab archpc** (16 GB, 8c/16t) via kaimon-remote: EDA + light checks ONLY.
- **mcmc-beast** (128 GB, 16c/32t): artifacts + heavy sampling later. Currently running
  the pregame smile grid — do not disturb.
- Sync: git push (laptop) → pull (server). Artifacts move by scp over tailscale.
- Pregame winner artifact: `data/scottish_decay_grid/none_pois_hl365_hs2_20260712_212831`
  (Grid A winner hl365_hs2, 164M) — scp'd from mcmc-beast. Grid B (smile) winner: pending.

## Work packages

- **WP0** RESEARCH.md — deep-research grounding (Vecer multipliers, cut Bayes,
  Davis–Norman, risk-constrained Kelly). IN PROGRESS.
- **WP1** `r00_data_qa.jl` — incidents QA, score-path reconstruction, betfair density,
  clock-map anchoring sanity on thin prints. IN PROGRESS.
- **WP2** `l01_nhpp_scottish.jl` / `r01_nhpp_transfer.jl` — port l08 NHPP + l07 CV
  harness; paired CV pregame-only vs +δ_time vs +state; pooled 56+57 vs separate.
- **WP3** `l02_ppd_compose.jl` / `r02_ppd_calibration.jl` — draw pairing → market-line
  PPDs via `compute_market_probs(MarketOverUnder(L − current_total))`.
  Gates: t=0 reproduces pregame prices; per-bin ECE/Brier; eval vs identifiable
  Betfair bins (l01 filters: ≥6 sels + full 1X2, λ_rem<6, residual<0.06).
- **WP4** `l03_rebalancer.jl` / `r03_rebalance_backtest.jl` — π(ω) state + convex Δa*
  with ℓ1 crossing cost; race: hold vs exit-rule τ=−0.05 vs rebalancer.
  Benchmark to beat: exit-rule uplift +0.306±0.059 (t=5.21).

## Log

- 2026-07-14: stream created. Deep-research batch dispatched. r00 written.
