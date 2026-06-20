# Paper Track — League of Ireland — 19 Jun 2026

**Model:** DCMH HalfLife ~45 / market_weight 0.5 (Dixon-Coles XG outfield, today's r03 runner)
**Backtest reference:** HL-60 grid (r06) green-zone filter — bet only positive `hurdle_G` + Sharpe markets.
**Dashboard generated:** 17:15 | **KO:** 20:00 (Fri)
**Bankroll:** £15.00 (Betfair, £1 min) | **Staking:** fractional Bayes-K, rounded to £1, portfolio-capped ~0.2.
**Strategy filter:** GREEN = totals unders (U1.5/U2.5/U3.5) + over_2.5 + BTTS-Yes. RED = all 1X2 + btts_no + overs ≥3.5 (negative G).

Backtest green ranking (HL-60, by growth): **btts_yes** (G .0088, Sh .20) > **under_15** (G .0069, Sh .145, ROI 42%) > over_25 > under_25 > under_35. Note: under_15 is a top-2 growth market, not a flyer — but ~32% hit-rate, so £1 sizing.

---

## Bets tracked

`Entry` = back price at 17:15 dashboard (single snapshot; close not re-pulled this week).
Rungs spread across matches to avoid nesting (U1.5 ⊂ U2.5 in same game = correlated, not diversified).

| # | Match | Selection | Model % | Entry | EV @ entry | Stake £ | FT | Goals | Result | P/L £ |
|---|-------|-----------|--------:|------:|-----------:|--------:|---:|------:|--------|------:|
| 1 | Bohemian v Dundalk | Under 1.5 | 25.8% | 5.00 | +29.0% | **1** | 1–2 | 3 | ❌ LOST | **−1.00** |
| 2 | St Patrick's v Sligo | Under 2.5 | 49.9% | 2.42 | +20.8% | **1** | 2–0 | 2 | ✅ WON | **+1.42** |
| 3 | St Patrick's v Sligo | BTTS Yes | 48.9% | 2.34 | +14.4% | **1** | 2–0 | — | ❌ LOST | **−1.00** |
| 4 | Waterford v Shamrock | Under 3.5 | 75.0% | 1.45 | +8.7% | **1** | 0–2 | 2 | ✅ WON | **+0.45** |

**Live book P/L (as placed): −1.00 +1.42 −1.00 +0.45 = −£0.13** (2W / 2L on £4 staked → −3.3% ROI). Bankroll £15 → **£14.87**.

**⚠️ Counterfactual — two deviations from a clean unders thesis cost the night.** Core legs fixed (Bohemian U1.5 LOST, St Pat's U2.5 WON). The two levers are (a) the Waterford gut-hedge (U2.5→U3.5) and (b) the off-thesis BTTS-Yes leg:

| Book | Waterford rung | BTTS-Yes | Staked | Net | ROI |
|------|----------------|----------|-------:|----:|----:|
| As placed | U3.5 +0.45 | in (−1.00) | £4 | **−£0.13** | −3% |
| Un-hedged | U2.5 +1.18 | in (−1.00) | £4 | +£0.60 | +15% |
| As placed, no BTTS | U3.5 +0.45 | out | £3 | +£0.87 | +29% |
| **Coherent unders** (un-hedged, no BTTS) | U2.5 +1.18 | out | £3 | **+£1.60** | **+53%** |

The thesis-pure book — back the unders, don't step the rung down, drop the contradictory both-score leg — is the clear winner at **+£1.60 on £3**. The gut-hedge cost £0.73; the off-thesis BTTS-Yes cost £1.00.

**Coherence note (the real lesson):** BTTS-Yes is directionally a *goals* bet — it needs both teams to score (≥2 goals), so it contradicts Under 1.5 (mutually exclusive) and only co-wins with Under 2.5 on exactly 1-1. Pairing it with U2.5 on St Pat's was a self-cancelling straddle (splits unless 1-1) — exactly what happened (2–0: under won, BTTS lost). It's the best backtest market by Sharpe and makes money over time (its loss here was variance), but it's a **separate thesis** from totals-unders. Don't stack both-score against an under on the *same* match, and on a low-goals night the unders are the clean core. **Carry forward: pick one coherent goal-direction per match.**

### Leg-by-leg learnings
- **Waterford (the model-vs-gut test): MODEL WON, and the hedge actively cost us the night.** Gut feared a Shamrock blowout (bottom v top → over). Actual 0–2 = controlled low-scoring away win, exactly the model's read. Stepping down U2.5 → U3.5 didn't just shave upside — it turned the whole book from **+£0.60 to −£0.13**. The blowout fear was unfounded; the model's "controlled low win ≠ goal-fest" logic was the correct, more profitable call. **Carry forward: on top-vs-bottom mismatches, trust the totals read and don't hedge the rung down.**
- **Bohemian under: genuine miss.** 1–2 = 3 goals; both U1.5 and U2.5 would have lost here. Model liked under (51.4%), market was right. U1.5 flyer missing is within its ~32% hit-rate, but the under-lean itself was wrong on this game.
- **St Pat's straddle split** as expected: U2.5 won (2 goals), BTTS-Yes lost (Sligo blanked, 2–0). Net +£0.42 on the match. BTTS-Yes is the best backtest market but variance bit on a single game.

---

## Deliberate skips (counterfactual — log result to check the filter)

| Match | Selection | Entry | EV | Why skipped | Result |
|-------|-----------|------:|---:|-------------|--------|
| St Patrick's v Sligo | Away | 15.50 | +167.2% | 1X2 neg G — headline trap | ✅ LOST (St Pat's won 2–0) — good skip |
| Waterford v Shamrock | Home | 5.00 | +38.9% | 1X2 neg G | ✅ LOST (0–2) — good skip |
| Galway v Derry | Home | 3.30 | +27.8% | 1X2 neg G | ❌ WON (Galway 2–1) — would've won |
| Bohemian v Dundalk | Away (Dundalk) | 4.10 | +37.3% | 1X2 neg G | ❌ WON (Dundalk 1–2) — would've won |

**1X2 skip filter: 2 busted / 2 would've won** (vs 3/3 busting on 12 Jun). Muddier night — but the biggest flagged number (St Pat's Away +167%) was still the worst bet, and the negative-G read is a season-long average, not a single-night promise. Don't over-update on either week.

---

## Settlement

| Match | FT Score | Total Goals | BTTS |
|-------|---------|------------:|------|
| Drogheda v Shelbourne | 2–2 | 4 | Yes |
| Galway v Derry | 2–1 | 3 | Yes |
| St Patrick's v Sligo | 2–0 | 2 | No |
| Waterford v Shamrock | 0–2 | 2 | No |
| Bohemian v Dundalk | 1–2 | 3 | Yes |

**Night-wide totals:** 14 goals / 5 games = **2.8 avg** (model leaned ~2.5, market ~2.9 → actual nearer the market). Model's blanket under-lean slightly too aggressive this round (Under 2.5 went 2/5), but it picked its two strongest unders correctly (St Pat's, Waterford both landed) which kept the book out of red.

**Verdict:** break-even night (−£0.13). Selection was good (both placed U2.5s won); the drag was the Bohemian U1.5 flyer + the BTTS-Yes leg of the straddle. Key takeaway carried forward: **on top-vs-bottom mismatches, trust the model's low-scoring totals read** — Waterford confirmed it cleanly.

---

### Notes
- Only one dashboard snapshot this week (17:15) — no close re-pull, so no CLV captured. Next time re-pull at ~KO to track totals-edge stability through team news.
- Kelly displayed 0.0 on dashboard; stakes derived from Bayes-K column, £1 floor.
- Bankroll carry: **£14.87** into next track.
