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
- 2026-07-14: **r00 ran on homelab kaimon** (fresh datastore fetched from betdb, cache
  rebuilt). Verdict:
  - Training seasons: 57 all six (20/21–25/26); 56 four (20/21, 21/22, 22/23, 24/25);
    56's 23/24 (71/180) + 25/26 (16/175) excluded pending incident re-scrape.
  - Score-path reconstruction exact for 98.9% (56) / 99.4% (57) of incident matches.
  - **Stoppage clamping**: ALL stoppage goals sit at exactly mm=45 (129) or mm=90 (302)
    with added_time=0 → terminal NHPP slices need extended exposure (injury_time1/2
    exist in sofascore.matches but are NOT fetched into the DataStore; use league-mean
    stoppage or extend the fetcher).
  - Betfair (eval only): median 49 1X2 prints/match, in-match gap p50 ≈ 1.05 min,
    p90 ≈ 4.1–4.6 min; **identifiable full-1X2 5-min bins only 23% (56) / 18% (57)**
    (4-min LOCF + ≥6 selections) — harsher than the concept-map audit; confirms
    eval-only role. Goal-jump anchoring finds a price jump for 81%/76% of goals;
    off1≈3.0±1.5, off2≈18.0±2.9 (medians sit on the priors → fallback-heavy on thin
    prints; acceptable for eval binning).
  - Red cards: 305 matches (18% of incident matches), 327 total — enough to fit β_man.
- 2026-07-14: **WP0 done** → RESEARCH.md. Key: Vecer/Kopriva/Ichiba 2009 market-implied
  red-card opponent multiplier ×1.2 ≈ our Ireland man_adv fit (+0.18 log) — use as
  prior centre; compose-posteriors = cut Bayes (draw pairing, Jacob et al. 2017);
  BRB λ = log β/log α composes with ℓ1 costs in one DCP program.
- 2026-07-14: **WP2 done (r01 on homelab, 7m39s, all R̂ ≤ 1.008).** 715 matches /
  27,170 slices / 1,934 goals / 1,046 red-affected rows. **Scotland ≠ Ireland:**
  - GLM race: time>pg_only t=6.29; **state−time t=0.51 (NULL — Ireland was 3.15)**;
    +man_adv t=8.85 (dominant).
  - Posterior: α=−4.584 (net level 0.99 after shape — see r02 gate), β=0.163 (rates
    rise), γ_tr=+0.02±0.06 (nil), **γ_ld=+0.09 (sign FLIPPED vs Ireland −0.24)**,
    **γ_man=0.53 ⇒ ×1.70** (Titman-scale, crushed the ×1.2 Vecer prior — red cards
    are THE in-play repricing event in these leagues), σ_time=0.08 (first-5-min dip
    −0.12, post-HT bump +0.09).
  - Late-checkpoint remaining-goals bias: +0.020/+0.014/+0.003 (60'/75'/85') — clean.
  - Open question (r01b candidate): γ_ld>0 may be a frailty/selection artifact
    (teams outperforming their pregame λ both lead AND keep scoring).
- 2026-07-14: **WP3 gates i+ii PASSED (r02, 22s).** t=0 kernel = 0.988 (composed
  kickoff intensity within 1.2% of pregame λ; consistent with the known ~5% pregame
  hot-ness); max composed-vs-pregame price gap < 0.01 across all 17 selections.
  Outcome calibration: Brier tightens 0.14→0.05 (OU) / 0.25→0.07 (BTTS) from KO to
  80'; **in-play conditioning beats frozen-pregame at every checkpoint** (BTTS Brier
  0.246→0.074 at 80'). Per-selection biases within ~1 SE (n=300). Outstanding:
  gate iii (Betfair identifiable-bin comparison on 56 24/25) = r02b.
- 2026-07-14: **r01b (user Q: hierarchical team δ_lead?) = NULL.** Non-centred team
  slopes on leading/trailing: σ_ld = 0.075 [0.007, 0.178], σ_trl = 0.10 [0.011, 0.222]
  — both hug zero (hierarchical-σ-null pattern again); shrunk team spread only
  −0.05…+0.06; Stage-B race auto-skipped. In the hier fit the GLOBAL γ_ld CI widens to
  [−0.02, 0.19] ⇒ not credibly non-zero. Verdict: keep global linear spec; γ_ld flip
  is most consistent with pregame-λ frailty, not team character. max R̂ 1.008.
- 2026-07-14: **r02b (WP3 gate iii) PASSED — with two data-validity lessons.**
  56 24/25, 143 matches, 661 identifiable bins. Raw comparison is POISONED by
  (a) settled selections and (b) ONE-SIDED thin markets (single traded side → vig-strip
  normalises p_fair to 1.0; raw market logloss 2.9 OU / 5.9 BTTS is an artifact —
  filters now built into r02b). On live two-sided rows: agreement corr 0.94/0.98/0.95
  (1X2/OU/BTTS), MAE 3.6–5.1 pts; reality-pricing is a statistical TIE (t = 0.15 / 1.05
  / −2.10, the last on only 27 matches). The composed engine is market-grade fair value
  — and prices the ~⅓ of OU rows + ~½ of BTTS rows where the exchange is one-sided or
  absent, which is exactly the coverage position management needs.
- NEXT: WP4 r03 rebalancer backtest (l03 solver written+pushed): book v1 from the
  decay-grid winner (unified-Kelly, curated totals/BTTS) — awaiting user's call on
  v1-now vs after the smile grid; race = hold vs exit-rule τ=−0.05 vs rebalancer.
  Full-fat NUTS re-runs and grids belong on mcmc-beast once the pregame grid finishes.
