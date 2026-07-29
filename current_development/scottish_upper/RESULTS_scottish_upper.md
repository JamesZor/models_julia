# Scottish Upper (54/55) — bake-off results

> Template. Fill each section as its runner completes. Nothing here is a result until the
> corresponding convergence gate has passed.

## 0. Scope and benchmark

- Segment: `Data.ScottishUpper()` = tournaments **54 (Premiership)** + **55 (Championship)**, pooled,
  with a zero-sum `δ_league` offset on log-λ.
- CV: targets **24/25 → 25/26**, `history_seasons = 2`, `dynamics_col = :match_biweek`,
  `warmup_period = 0`.
- **Benchmark = de-vigged Bet365 (SofaScore) close.** No Betfair exists for these tiers, so:
  - there is **no CLV-vs-exchange check** and no exchange-priced backtest;
  - the benchmark is **self-referential for market-anchored cells** (`iso_*`, `smile_*`), which are
    trained toward the same close they are scored against. Structural cells (`none_*`, `funnel`,
    `rating`) do not have that circularity.
  Absolute numbers here are therefore **not comparable** to the 56/57 or Ireland streams.

## 1. r00 — data QA ✅ PASSED (2026-07-29, `r00_out.txt`)

| item | value | note |
|---|---|---|
| season strings | `"20/21"…"25/26"`, **shared by both tiers** | fetcher strips the competition prefix — pooled `target_seasons` works |
| matches / season | 198 (54, rounds 1–33) · 180 (55) | 54's 5 post-split rounds are absent from the DB |
| O/U ladder density | u05–u55 ≈376/378 in both grid seasons | **Kmax=4 comfortable**; u75 thin |
| mean goals 54 / 55 | 2.694 / 2.620 | |
| `δ_league` gap | **0.0279** | well inside `league_offset_sd = 0.1` — no widening needed |
| goal-total sd ratio 54:55 | **0.954** | tiers differ in LEVEL not SPREAD ⇒ a level offset is the right tool; the planned "wider gap" risk did not materialise |
| **V/M 54 / 55** | **0.965 / 1.09** | **⚠ different dispersion regimes** — 54 sub-Poisson, 55 over-dispersed. Unlike 56/57, `none_nb` is a genuinely informative reference here |
| BBC shots coverage | **2218/2218 = 100%**, both tiers, all 6 seasons | best shot data in the project |
| rating coverage 22/23 (tier 55) | **0/180 (0%)** | exactly the case the src masking fix targets |
| biweek folds (grid window) | **40** (week 77 / month 22), history ≈756/fold | sets the per-cell budget |
| market inversion | DP **2213/2213 (100%)**; smile full-ladder 2113/2213, Λ rising 2.297→2.759 | textbook market smile |
| betfair | **536 matches, tier 54 only** (23/24–25/26) | ⚠ see correction below |

### ⚠ Betfair correction

The stream was designed on a verified "0 Betfair rows" reading. **A backfill for 54/55 began
2026-07-29 15:26 and is in progress**: tier 54 at 598 SUCCESS / 392 PENDING (536 usable matches),
tier 55 at 0 SUCCESS / 895 PENDING.

- The pooled grid **keeps Bet365 as the uniform training anchor**. A Betfair-on-54 / Bet365-on-55
  pillar would confound the anchor with the league.
- Betfair on tier 54 is now usable as a **secondary CLV check** on the winner (536 matches — more
  than 56/57 ever had).
- The **full anchor A/B re-opens once tier 55 completes** — that is the open Ireland counter-test.
- **⚠ The DataStore cache was written mid-ingest.** Re-fetch with `force = true` before any run whose
  result depends on Betfair.

**Decisions taken:** `history_seasons = 2` (the mask makes the unrated 22/23 block safe — r01 CHECK 4
must confirm), `league_offset_sd = 0.1`, `INCLUDE_SMILE` = pending r01's runtime table.

## 2. r01 — smoke + runtime calibration

Checks passed: `__ / __`.

| cell | smoke folds | smoke wall | projected r02 wall |
|---|---|---|---|
| none_pois_hl365 | | | |
| none_pois_hl180 | | | |
| rating_pois_hl365 | | | |
| funnel_pois_hl365 | | | |
| none_nb_hl365 | | | |
| iso_pois_mw40_hl365 | | | |
| smile_pois_sup100_sw50 | | | |
| **total** | | | |

Rating-pillar scale: centred rating sd = `___` → implied pillar sd at `w_sd = 0.05` = `___`.
Unmasked `−10·base` mode present? `___` (must be **no**).

`δ_league` read: `___` (expect 54 > 55 on goal level).

## 3. r02 / r03 — family bake-off

### 3.1 Convergence gate (`r02_convergence.txt`)

| cell | folds | ≤1.01 | % | worst | verdict |
|---|---|---|---|---|---|

### 3.2 Family-pooled LogLoss diff vs close (lower better)

| cell | 1X2 | totals | BTTS |
|---|---|---|---|

### 3.3 GLMEdge — significant lines only

### 3.4 Money — hurdle_G / ROI per selection

### 3.5 Routing verdict

| family | winner | evidence | act? |
|---|---|---|---|
| 1X2 | | | expected: ABSTAIN |
| totals | | | |
| BTTS | | | |

Reference checks:
- `none_nb` vs `none_pois` — NB should be inert on sub-Poisson data. Result: `___`
- `hl180` vs `hl365` — does the 56/57 long-memory gradient reproduce? Result: `___`

## 4. r04 / r05 — anchor strength

### 4.1 Convergence gate (`r04_convergence.txt`) — including WHICH params offend

### 4.2 Sweep results

| cell | 1X2 | totals | BTTS | ROI | hurdle_G | gate |
|---|---|---|---|---|---|---|

### 4.3 Which pattern does 54/55 show?

- [ ] **Ireland pattern** — interior optimum near `mw ≈ 0.25–0.40`, over-anchoring hurts.
- [ ] **56/57 pattern** — axis flat on scoring; tiebreak on sampler health; offenders are the team
      ratings, not the market σ.
- [ ] Neither / something new: `___`

## 5. Production decision

| | |
|---|---|
| Engine | |
| Weights | |
| `hl` / `hs` | |
| Route | |
| Abstain | |
| Retrain cadence | weekly, in-season |

**Caveats to carry into live use:** `___`

## 6. Open threads

- Betfair history for 54/55 → enables the anchor A/B and real CLV. Currently absent.
- xG backfill for 54 (only 25/26 present) → would open the Ireland `outfield_*` family.
- 54's missing post-split rounds (30/season) — end-of-season folds are structurally thin.
