# Ireland First Division (718) — Consolidated Research Notes

Master index + findings for all First Division (tournament **718**, `Data.IrelandFirstDivision()`)
work. Pulls together the Stage-A EDA, the three outfield-engine A/B runs, and the time-decay
half-life grid. All runs done in the kaimon REPL on `mcmc-beast`; sync via git push→pull.

## Report map

| Report | Location | What it covers |
|---|---|---|
| Stage-A EDA | `eda/first_division_validation/first_division_eda.md` (r01) | 718 as a DGP; pool-vs-stratify; Stage-B readiness |
| DP/DC A/B | `current_development/ab_test_first_division/r01_ab_test_first_division.jl` (RESULT blocks) | DoublePoisson vs DixonColes outfield engines |
| Double-NB A/B | `current_development/ab_test_first_division/r02_double_negbin.jl` (RESULT blocks) | NB (HomeAway + Hierarchical) vs DP/DC; RQR |
| Half-life grid | `current_development/ab_test_first_division/HALFLIFE_GRID_RESULTS.md` (r03) | time-decay half-life sweep {30,45,60,90,120} |
| This note | `718_FIRST_DIVISION_NOTES.md` | consolidated summary + cross-report verdicts |

Memory: `first-division-718-signature`, `outfield-xg-engine-gotchas`.

---

## 1. Stage-A EDA — 718 is a distinct NB regime

- **Over-dispersed:** NB beats Poisson by **9–12 AIC** (V/M ≈ 1.14, NB r ≈ 10.8, μ ≈ 1.40/side).
  The Premier (79) is by contrast a **Poisson** league (V/M ≈ 1.04). → 718 needs its own dispersion.
- Scores **more** (1.40 vs 1.25/side) but **smaller home advantage** (0.20 vs 0.31).
- **Dixon-Coles ρ ≈ 0** in both tiers → drop the τ low-score correction for Irish leagues.
- Most pooled over-dispersion is **cross-team heterogeneity** (absorbed by hierarchical team strength);
  residual within-team gap ~0.10 is what a per-league NB `r` knob captures.
- **Verdict:** stratify dispersion (per-league `r` from a shared hyperprior); pool team-strength/home.
- **Stage-B readiness:** betfair_odds present (428k rows, CLV feasible); xG from 2023 (~99% coverage);
  **NO `bigChanceCreated` column** → bigChance pillar unusable for 718.

## 2. Engine A/B — double-NB is the best outfield engine

Engines: goals + xG + market pillars, Betfair de-vigged market pillar, 2025/26, 275 matches,
`market_weight=0.4`, full-Kelly, no costs. (r01 = DP/DC; r02 added NB variants.)

| Engine | LogLoss diff | GLM spread (p) | ROI % | Train |
|---|---|---|---|---|
| **NB HomeAway** | −0.0323 | 1.34 (0.143) | **12.5** | 55m |
| NB Hierarchical | −0.0322 | 1.33 (0.146) | 12.5 | 2h02m |
| DixonColes | −0.0325 (best) | 1.48 (0.088, best) | 9.7 | 1h48m |
| DoublePoisson | −0.0295 | 0.86 (0.306) | 10.2 | 60m |

- **NB wins ROI**, ~ties DC on LogLoss/edge, and samples faster & healthier than DC
  (ε up to 0.025 vs DC's pathological ε~3e-6).
- **Hierarchical dispersion adds NOTHING** over simple `HomeAwayDispersion` on every metric but
  costs 2× train time → **use HomeAwayDispersion**.
- **RQR (goals calibration):** NB **passes** Shapiro normality (std 0.95, kurt 0.21, p=0.48);
  Poisson-based DP/DC **fail** (std>1, heavy tails) — exactly the over-dispersion the EDA flagged.
  The NB goals *distribution* is well-calibrated; the models are only "bad at goals" as a *betting*
  signal on totals (market already prices the predictable part → O/U is adverse selection).

## 3. Half-life grid — run short (≈30 days)

ROI and growth are **monotonic in half-life — shorter wins** (HL30 ROI 13.8%/growth 1.044, the only
setting that compounds; HL120 10.3%/0.217). GLM edge agrees (short = strongest, near-significant).
LogLoss *aggregate* mildly prefers long, but that's a **tail-line artifact** (the near-certain
`over_05`/`under_55` lines): on contested/bettable markets short HL is at least as well calibrated.
RQR is flat & good across the grid. Full tables in `HALFLIFE_GRID_RESULTS.md`.

## 4. Cross-report verdict — where the edge actually lives

Every lens (ROI, growth, GLM, LogLoss) converges on the same per-market structure:

- **HOME 1x2 is the edge.** ~42% ROI, growth ~6×, robust across engines AND half-lives; the only
  contested market the model also beats market on LogLoss. This is the bulk of all profit.
- **DRAW** is a steady secondary winner (+25–28%).
- **AWAY is a near-wipeout** (−6% to −18% ROI, growth 0.1–0.2×) at every setting — adverse.
- **Totals (O/U 2.5/3.5) are adverse selection** — model is calibrated but has no per-match edge
  over the market on goals; `over_25` loses despite a positive GLM sign.
- **The under ladder is the half-life lever:** `under_15/25/35` flip from profitable (short HL) to
  loss-making (long HL). This is what moves the aggregate ROI between half-lives.
- **`over_45/55`, `under_05` ROIs are noise** (≤22 bets) — ignore.

**Operating recommendation for 718:** double-NB engine, `HomeAwayDispersion`, half-life ≈ 30 days,
Betfair market pillar (`market_weight` 0.4). Curate to the markets with real edge (home, draw,
short-HL unders) and drop the structural losers (away, over_25/35/55, under_05). Apply a
portfolio-Kelly cap for live sizing — the sub-1.0 aggregate growth at long HL / all-market full-Kelly
is overbetting, not model failure (see `portfolio-kelly-partial-hedge`).

## 5. Open threads

- Sweep half-life **below 30** (21/14) — trend is monotonic, but watch effective sample size
  (275 matches starves the team hierarchy at very short HL).
- Market-inverse pillar currently anchors **λ only** (DoublePoisson inversion, Poisson-implied).
  Optional: switch to `RegularizedDoubleNegativeBinomialMarketFeature` for NB-consistent λ, and/or
  add a market-dispersion (`r`) anchoring term. Not yet wired in.
- Consider anchoring the market pillar to **Bet365 de-vigged** rather than Betfair for this thin
  minor league (see `betfair-vs-bet365-market-anchor`), executing bets on Betfair.
