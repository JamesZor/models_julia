# r03 bigChance A/B — O/U + BTTS markets & hurdle growth function

Follow-up to the r03 backtest (M1 `{goals,market,xG}` · M2 `{goals,market,bigChance}` ·
M3 `{goals,market,bigChance,xG}`). The 1X2 backtest used **Betfair** odds; here we look at
**Over/Under, BTTS** and the **hurdle growth function** of the ROI.

## Data caveat — odds source
Ireland's **Betfair exchange only lists 1X2 + Double-Chance** (no O/U or BTTS liquidity), so the
backtest here runs on `ds.odds` = **SofaScore bookmaker closing odds, vig included** (`:odds_close`,
realistic takeable prices — not de-vigged). Small sample: **281 matches**, ~60–110 bets per selection.
Relative comparison is valid; treat absolute ROIs as indicative.

## Market-group ROI (bookmaker closing odds)
| Market | M1 (xG) | M2 (bigChance) | M3 (both) |
|---|---|---|---|
| **BTTS** | +29.7% | +29.2% | +24.7% |
| OverUnder (all lines) | +2.6% | +3.8% | +2.3% |
| 1X2 | +0.8% | +3.3% | +0.5% |
| Double-Chance | −25.7% | −26.2% | −25.9% |

Double-Chance is a consistent loser (low odds, no edge, pays the margin) — exclude from staking.

## Hurdle growth function `G`
Hurdle model: `R ~ p·Gamma(α,β) + (1−p)·δ(−1)` (per-bet ROI). The **growth function** is the per-bet
geometric (Kelly) growth rate
```
G = exp( E[log(1 + f·R)] ) − 1        f = avg stake
```
`hurdle_G` = parametric (MC over the fitted Gamma); `hurdle_G_emp` = realized. **`G>0` compounds the
bankroll; `G<0` bleeds it even when average ROI is positive** (variance drag from Kelly-sized losses).
Defined in `src/backtesting/metrics/implentations/hurdle_roi.jl`.

Per-selection `G` (≈ mean across the 3 models):
| Selection | ROI | `G` (param) | `G_emp` | Verdict |
|---|---|---|---|---|
| **btts_yes** | +31% | +0.0075 | +0.009 | ✅ genuinely compounds |
| **draw** | +42% | +0.0070 | +0.008 | ✅ compounds |
| **under_35** | +6% | +0.0025 | +0.001 | ✅ small + |
| btts_no | +24% | +0.0007 | +0.006 | ≈ flat/slightly + |
| under_25 | +12% | ~0 | +0.003 | marginal |
| over_25 | −6% | +0.003 | **−0.006** | ✗ param + but emp − (noise) |
| **away (1X2)** | **+16%** | **−0.016** | **−0.010** | ⚠️ +ROI but NEGATIVE growth |
| home (1X2) | −17% | −0.025 | −0.026 | ✗ bleeds |

**Headline:** `away` 1X2 shows **+16% ROI yet negative G** — a few large Kelly stakes that lose drag
log-wealth down, so it loses money compounded. ROI hides this; the growth function exposes it. The
markets that actually **compound** are **BTTS-yes, draws, and under-goals**; home/over bleed.

## bigChance vs xG across markets
The earlier M2 (bigChance) edge was **1X2/Betfair-specific** and does NOT carry here:
- **btts_yes:** M1 (xG) `G≈+0.0080` ≥ M3 +0.0078 ≥ M2 +0.0065 (xG marginally best)
- **draw:** M3 +0.0094 > M1≈M2 +0.0059
- **O/U:** all three within noise

⇒ No consistent bigChance-vs-xG winner across the goal-derived markets, and M3 (both) is no longer
the worst here. Net: **bigChance ≈ xG as the attacking pillar** — a viable substitute, not a clear
upgrade. [[bigchancecreated-eda-findings]]

## Next (optional)
Re-run focused on the compounding markets only (BTTS-yes, draws, under-goals) for cleaner per-model
`G`; and confirm on a larger sample / other leagues before acting.
