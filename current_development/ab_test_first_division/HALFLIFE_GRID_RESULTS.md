# Time-Decay Half-Life Grid — Double-NB HomeAway Outfield Engine (First Division 718)

**Model:** `DynamicDoubleNegBinXGOutfieldPlayerTimeDecayModel`, goals (NB) + xG + market pillars,
HomeAway dispersion, `market_weight=0.4`, Betfair market pillar. Everything fixed **except** the
player-dynamics time-decay half-life. Target 2025/26, 28 splits, 275 in-sample matches, full-Kelly,
no costs. Match weight = `0.5 ^ (Δdays / half_life)`: short = recency-heavy/reactive, long = more
(staler) data. Grid = **{30, 45, 60, 90, 120}** days (60 reused from r02). Runner: `r03_halflife_grid.jl`.

## Headline: shorter half-life wins on money, longer wins (marginally) on probabilistic fit

| Half-life | Bets | ROI % | Growth factor |
|---|---|---|---|
| **30** | 760 | **13.82** | **1.044** |
| 45 | 763 | 13.10 | 0.816 |
| 60 | 762 | 12.49 | 0.626 |
| 90 | 769 | 11.08 | 0.323 |
| 120 | 773 | 10.32 | 0.217 |

Perfectly **monotonic** in ROI and growth — shorter is better. **HL30 is the only setting with
aggregate growth > 1.0** (all-market full-Kelly; the others overbet into compounding losses, the
familiar portfolio-Kelly caveat). The form signal in this league decays fast; weighting recent
matches harder pays off.

## The other metrics tell a more nuanced story

**LogLoss (model − market diff_ll; closer to 0 = better, all negative = below market):**

| HL | 30 | 45 | 60 | 90 | 120 |
|---|---|---|---|---|---|
| diff_ll | −0.0332 | −0.0326 | −0.0323 | −0.0317 | −0.0316 |

**Inverted vs ROI** — *longer* half-life is marginally better calibrated against the market overall.
But the spread is tiny (0.0016 across the whole grid). Probabilistic-fit and betting-edge pull in
opposite directions: more data → smoother average probabilities; recency → sharper, more exploitable
mispricings.

**GLM edge (spread_fair coef, p-value; higher coef / lower p = stronger edge):**

| HL | 30 | 45 | 60 | 90 | 120 |
|---|---|---|---|---|---|
| coef | **1.633** | 1.445 | 1.338 | 1.212 | 1.231 |
| p | **0.071** | 0.112 | 0.143 | 0.188 | 0.183 |

Agrees with ROI: shortest half-life has the strongest (near-significant) edge signal.

**RQR (goals calibration; std→1, kurtosis→0, Shapiro p>0.05 = Normal = good):**

| HL | std | kurtosis | Shapiro p |
|---|---|---|---|
| 30 | 0.938 | −0.17 | 0.519 ✓ |
| 45 | 0.962 | 0.07 | 0.774 ✓ |
| 60 | 0.967 | −0.08 | 0.574 ✓ |
| 90 | 0.939 | 0.00 | 0.925 ✓ |
| 120 | 0.994 | 0.04 | 0.047 (borderline) |

Goals calibration is **good and essentially flat** across the grid — every setting passes Shapiro
(HL120 borderline). Half-life does **not** break the NB goals model's calibration; the std slightly
< 1 everywhere (mild residual tightness). Calibration is not the lever here.

## Per-market LogLoss (model − market diff; negative = model beats market)

| Market | HL30 | HL60 | HL120 | n |
|---|---|---|---|---|
| home | **−0.0038** | −0.0032 | −0.0012 | 166 |
| draw | −0.0005 | −0.0003 | −0.0001 | 166 |
| away | +0.0153 | +0.0146 | +0.0143 | 165 |
| btts (yes≡no) | +0.0037 | +0.0046 | +0.0035 | 60 |
| over_05 | −0.2724 | −0.2715 | −0.2710 | 108 |
| o/u_15 | +0.0001 | −0.0006 | −0.0001 | 95 |
| o/u_25 | +0.0132 | +0.0139 | +0.0123 | 147 |
| o/u_35 | +0.0065 | +0.0101 | +0.0134 | 107 |
| over_45 | −0.0433 | −0.0323 | −0.0229 | 39 |
| under_55 | −0.6373 | −0.6398 | −0.6415 | 48 |
| **ALL** | **−0.0332** | −0.0323 | −0.0316 | 1635 |

Complementary lines (over_25≡under_25, btts_yes≡btts_no) have identical diffs — same binary event.
The aggregate "ALL" is **carried by the near-certain tail lines** `over_05` (−0.27) and `under_55`
(−0.64), where the model is sharply better calibrated than the de-vigged market. On the *contested*
markets the model only genuinely beats market on **home** (and a whisker on draw); away/totals are
worse. The "longer HL helps LogLoss" aggregate is a tail artifact — on the bettable lines (home,
over_35, over/under_45) **short HL is better calibrated**, consistent with ROI.

## Where the ROI difference actually comes from (per-market)

The dominant profit engine is **HOME 1x2**, and it's ~**42% ROI for every half-life** (growth
5.9–6.8×, slightly higher at short HL). The home edge is robust to decay. **DRAW** is also a steady
winner (+25–28%, all HLs). What *changes* with half-life:

- **UNDER lines flip with half-life.** `under_25`: HL30 **+9.8%** → HL120 **−10.2%**.
  `under_35`: HL30 **+8.4%** → HL120 **−4.7%**. `under_15`: HL30 +28.9% → HL120 +11.8%.
  Recency-weighting makes the under ladder profitable — this is the main driver of the aggregate gap.
- **AWAY** loses for all (adverse), but HL30 is least-bad (−5.6%) vs HL120 worst (−18.2%).
- **over_25** is a consistent loser everywhere (−13% to −18%) — the known adverse-selection line;
  half-life doesn't rescue it.
- Structural losers regardless of HL: `over_35`, `over_55`, `under_05`.
- `over_45` shows +280–315% ROI but on tiny stakes (n≈14–22) — variance, not a real edge.

## Verdict

**Use a short half-life (≈30 days) for this engine on First Division.** It maximises ROI (13.8%) and
is the only setting that compounds (growth 1.044), driven by turning the under ladder profitable and
softening the away leg while leaving the dominant home edge intact. The cost is a negligible LogLoss
penalty (0.0016) and no calibration harm (RQR still passes). The trend is monotonic, so **a sweep
below 30 (e.g. 21, 14) is worth trying** — but watch effective sample size: with only 275 in-sample
matches, very short half-lives starve the team-strength hierarchy. HL30 is a sensible operating point.

Caveat: in-sample, no costs, all-market full-Kelly (the sub-1.0 growth at long HL is overbetting,
not pure model failure — apply the portfolio-Kelly cap for live sizing).
