# Paper Track — League of Ireland — 12 Jun 2026

**Model:** DCMH_HalfLife_60 (Dixon-Coles XG outfield, market_weight 0.4)
**Dashboard generated:** ~17:38 | **KO:** 20:00 (Fri)
**Bankroll:** £20.00 (Betfair, £1 min) | **Staking:** fractional Bayes-K, rounded to £1
**Strategy filter:** only markets with positive backtested `hurdle_G` + Sharpe (HL-60 grid).
Green markets = totals (unders + over_2.5) and BTTS-Yes. RED = all 1X2 + btts_no (negative G).

---

## Bets tracked

`Entry` = back at ~17:38 (pre-lineup). `Close` = back at 20:11 (XIs confirmed, ~KO). `CLV` = did the market move toward us (price shortened = +) or away (drifted = −)?

| # | Match | Selection | Model % | Entry | Close | CLV | EV @ close | Stake £ | Status | Result | P/L £ |
|---|-------|-----------|--------:|------:|------:|:---:|----------:|--------:|--------|--------|------:|
| 1 | Galway v Dundalk | Under 3.5 | 72.1% | 1.54 | 1.53 | + | +10.3% | — | not placed | — | — |
| 2 | Galway v Dundalk | Under 2.5 | 50.3% | 2.36 | **2.36** | 0 | +18.7% | **2** | ✅ PLACED | ✅ WON (0-1, 1 gl) | **+2.72** |
| 3 | Galway v Dundalk | Under 1.5 | 25.1% | 5.10 | 5.00 | + | +25.3% | — | not placed | — | — |
| 4 | Derry v Bohemian | Over 2.5 | 49.9% | 2.32 | 2.30 | + | +14.7% | — | not placed | — | — |
| 5 | Derry v Bohemian | BTTS Yes | 54.6% | 1.94 | **1.95** | − | +6.4% | **1** | ✅ PLACED | ✅ WON (4-1) | **+0.95** |
| 6 | St Patrick's v Drogheda | BTTS Yes | 49.9% | 2.16 | **2.16** | 0 | +7.7% | **1** | ✅ PLACED | ❌ LOST (2-0) | **−1.00** |

**Live book:** £4 staked (rows 2, 5, 6). Bankroll £20 → £16 dry.
**CLV verdict:** totals edges held or crept toward us through the lineup re-price (mild net +CLV). Nothing blew up post-team-news → green markets are stable.

### Settled scenarios for the live £4
- **Galway Under 2.5 @ 2.36** (£2): wins if Galway+Dundalk total goals ≤ 2 → returns £4.72 (+£2.72); loses if ≥3 → −£2.
- **Derry BTTS Yes @ 1.95** (£1): wins if both Derry & Bohemian score → returns £1.95 (+£0.95); else −£1.
- **St Pats BTTS Yes @ 2.16** (£1): wins if both St Pats & Drogheda score → returns £2.16 (+£1.16); else −£1.
- Max win if all three land: **+£4.83** (book → £8.83). Max loss if all three miss: **−£4.00**.

**Early exposure:** £6 (rows 1–3) · **Pending:** £4 (rows 4–6) · **Total at risk if all placed:** £10 / £20 (keep ~£10 dry).

Row 3 (Under 1.5) is the high-variance flyer — £1 only.

---

## Deliberate skips (counterfactual — log result to check the filter)

These flagged big EV on the dashboard but sit in RED markets (negative `hurdle_G`). Tracking to confirm we were right to pass.

| Match | Selection | Entry | Close | EV @ close | Why skipped | Did it win? |
|-------|-----------|------:|------:|-----------:|-------------|-------------|
| Waterford v Sligo | Away | 3.80 | 4.00 ↑ | +58.5% | 1X2 neg G; **drifted AWAY post-lineup** | ❌ LOST (Waterford won 4-0) — good skip |
| St Patrick's v Drogheda | Away | 9.00 | 9.20 ↑ | +39.3% | 1X2 neg G; drifted away | ❌ LOST (St Pats won 2-0) — good skip |
| Galway v Dundalk | Home | 2.90 | 2.94 ↑ | +19.3% | 1X2 neg G; drifted away | ❌ LOST (Galway lost 0-1) — good skip |
| Galway v Dundalk | BTTS No | 2.54 | 2.58 ↑ | +17.6% | btts_no neg G | ✅ would have WON (0-1) — the one miss |

**Key finding:** every skipped 1X2 selection **drifted further from the model** after lineups confirmed — the informed close disagrees with the model's match-result view *more*, not less. Confirms the backtest's negative-G read on 1X2. The "+58% Waterford Away" is the market telling us we're on the wrong side, not a missed bet.

---

## Settlement (fill in after FT)

| Match | FT Score | Total Goals | BTTS |
|-------|---------|------------:|------|
| Galway v Dundalk | 0–1 | 1 | No |
| Derry v Bohemian | 4–1 | 5 | Yes |
| St Patrick's v Drogheda | 2–0 | 2 | No |
| Waterford v Sligo | 4–0 | 4 | No |
| Shelbourne v Shamrock | 2–1 | 3 | Yes |

**Live book P/L: +£2.72 +£0.95 −£1.00 = +£2.67** (2W / 1L on £4 staked → **+66.8% ROI**). Bankroll £20 → **£22.67**.

**Skips counterfactual:** 3 of 4 lost (all three 1X2 traps — Waterford Away, St Pats Away, Galway Home — busted exactly as the negative-G read + post-lineup drift predicted). The lone miss was Galway BTTS No, which would have won.

**CLV vindication:** Waterford Away drifted 3.80→4.00 into the close *and lost 4-0*. The market's post-lineup move away from the model was the correct signal — the "+58% EV" was a trap, confirmed by result.

**Model read of the night:** good. Won the totals/BTTS calls (its green zone), and every 1X2 it liked lost. Also note Derry Over 2.5 / Over 3.5 (model liked, not placed) both landed 4-1 — correct direction, missed winners.

---

### Notes
- Open question at time of writing: does the live model use **confirmed** or **projected** lineups? If projected, EARLY rows carry team-news risk and PENDING rows should only fire post-XI.
- Kelly displayed as 0.0 on dashboard; stakes here derived from Bayes-K column.
