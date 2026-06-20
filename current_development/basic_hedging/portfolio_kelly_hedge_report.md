# Portfolio Kelly + Continuous Partial Hedging — Report

*Ireland DCMH backtest, 258 matches / 1,574 bets, staked per-bet with `BayesianKelly` (Baker–McHale
distributional optimiser). Sequential per-match compounding of log-growth. Code:
`r01_portfolio_kelly_hedge.jl`.*

## What changed (correction to the earlier "hedging adds nothing" finding)

The earlier conclusion compared **hold** vs **full-exit** at a flat **¼-Kelly**. Both choices were wrong:

1. **Wrong staking baseline.** `BayesianKelly` is correct *per bet*, but a match fires ~6 **simultaneous,
   correlated** bets (nested OU lines + BTTS + 1X2). Summing independent Kelly fractions over-stakes the
   **joint** position. A flat ¼-Kelly just hides that by scaling everything; the right fix is a
   **portfolio cap** (Kelly on the joint book).
2. **Wrong hedge action space.** Hedging is **continuous** — lay off a fraction φ∈[0,1] (cash out part /
   reduce risk) — not binary hold-vs-full-exit. Full exit (φ=1) is essentially never optimal, which is
   precisely why the binary test saw no value.

## 1. The dominant lever is portfolio sizing, not hedging

Independent per-bet `BayesianKelly` summed across a match:

| metric | value |
|---|---|
| mean joint stake / match | 0.40 of bankroll |
| **max joint stake / match** | **2.17 (217%)** |
| matches staking >25% | 57% |
| worst-match return | **−143% → BANKRUPTCY** (growth = −∞, bank → 0) |

Capping the **summed** stake per match (scale all simultaneous stakes so Σ ≤ cap) gives the textbook
Kelly parabola — but at the **portfolio** level, the object that was never sized:

| joint cap | 0.05 | 0.10 | 0.15 | **0.20** | 0.25 | 0.30 | 0.40 | 0.50 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|
| log-growth | 1.34 | 2.33 | 2.89 | **3.08** | 3.02 | 2.46 | 0.77 | −1.09 | −6.05 |

Growth-optimal joint cap ≈ **0.20** (×21.7 terminal bank). Past ~0.45 it goes negative. **This single
change — sizing the joint book — turns bankruptcy into ×21.7 growth.** It is far larger than any hedging
effect.

## 2. Partial hedging adds growth on a correctly-sized book — and the optimum is interior

On the cap-0.20 book, scanning the hedge fraction φ at a fixed exit minute:

| φ (fraction laid off) | hold 0.0 | 0.25 | 0.50 | 0.75 | full 1.0 |
|---|---|---|---|---|---|
| **exit @70'** | 3.076 | 3.341 | **3.377** | 3.164 | 2.667 |
| **exit @80'** | 3.076 | 3.522 | 3.768 | **3.805** | 3.613 |

**Full exit is worse than holding at 70'** and only marginal at 80' — the reason the binary exit/hold
test found nothing. The value is in the **interior**: reduce risk on *part* of the position, let the rest
run (φ ≈ 0.5–0.75).

## 3. Synergy: hedging is variance insurance → it lets you size the book bigger

Because partial hedging cuts variance, the growth-optimal **cap rises** once you hedge. Joint cap × φ:

| joint cap | hold-only | best partial-hedge | best φ |
|---|---|---|---|
| 0.20 | 3.076 | 3.805 | 0.75 |
| 0.25 | 3.022 | 4.235 | 0.75 |
| **0.30** | 2.459 | **4.302** | 0.75 (@80') |
| 0.40 | 0.767 | 3.773 | 0.75 |
| 0.45 | −0.291 | 3.312 | 1.0 |

- **Hold-only optimum:** cap 0.20 → 3.08.
- **Hedged optimum (80'):** cap 0.30, φ=0.75 → **4.30** (≈ **3.4×** the hold-only-optimum terminal wealth).
- **Hedged optimum (70', cleaner):** cap 0.25, φ=0.50 → **3.60** (≈1.7×).
- Where an **un-hedged** book at cap 0.40–0.45 collapses toward ruin, the **hedged** book stays strongly
  positive — the insurance buys you the right to run a larger book.

## Recommendations

1. **Size the joint book, not each bet.** Replace independent per-bet Kelly with a **portfolio cap**
   (Σ simultaneous stakes ≤ ~0.20–0.25), or a joint log-growth optimisation across the correlated
   selections. This is the #1 growth fix (bankruptcy → ×20+).
2. **Hedge partially, never fully.** A standing rule of φ≈0.5 around the 70' mark already adds growth and
   reduces drawdown; φ≈0.75 nearer the end captures more but with thinner liquidity.
3. **Co-size cap and hedge.** With partial hedging the optimal cap rises (~0.25–0.30). Treat staking and
   hedging as one joint problem, not two stages.

## Caveats
- **In-sample**, 258 backtest matches, single league (Ireland) — magnitudes will shrink out-of-sample.
- Wall-clock 80' ≈ match-minute ~65 (late; thin liquidity, possible mild near-settlement lookahead) —
  **trust the 70' result more** than 80'.
- Betfair **LTP ≠ tradeable lay price**; a real lay would pay the spread.
- This is a **rule-based** hedge (fixed φ at a fixed minute). The model-driven version sets φ from the
  in-play model's updated posterior vs the live price (Phase 2).

## Phase 2 — model-driven position management (BUILT, `r02_model_driven_hedge.jl`)

Replaces the fixed (minute, φ) clock rule with an **in-game-model** exit signal. Per open bet, walk the
match's ticks; at each, the in-play intensity model (`ch`, `:linear` config) gives expected **remaining**
goals per side → remaining-goals score matrix → **P_model(selection | live score)**; compare to the live
market **1/price**. Exit (lay) the first time the model edge `P_model − 1/price ≤ τ`.

**Lookahead discipline (critical).** Executing *at* the signal price prints 6.8 log-growth — but that is
**stale-price lookahead** (the l04 lesson: LTP at the signal tick already moved on the goal that fired
the signal). With realistic **forward execution** (lay at the next price ≥ signal+lag) it settles at
**~3.74 and is stable across 1/3/5-min lag** — it does *not* collapse like the l04 trading backtest,
because we manage an existing position on a genuine edge-decay signal, not a microstructure blip.

**Honest results** (cap 0.20, full exit, forward 3-min):

| strategy | log-growth |
|---|---|
| hold to settlement | 3.076 |
| fixed clock exit @70' | 2.667 (worse than hold) |
| fixed clock exit @80' | 3.613 |
| model exit, τ=0 | 3.734 |
| **model exit, τ=−0.05** | **4.423** |
| model exit + partial φ=0.5 | 3.819 |

Co-sizing: τ=−0.05 full exit is optimal at **cap 0.25 → 4.549** (≈4.3× hold terminal wealth).

**Two conceptual findings:**
1. **Optimal τ is a small *negative* edge (−0.05), not zero** — exit only once the market clearly
   overtakes the model (~5 pts), not on the first wobble; avoids churning out on noise.
2. **When the model fires, FULL exit beats partial (4.42 > 3.82).** This *reverses* the fixed-clock
   finding (§ above), and it is the whole point: the clock hedges *partially* out of ignorance; the model
   *knows* the edge is gone, so it exits *decisively*. **Partial hedging is what you do without a model —
   the in-game model buys you the right to exit fully and correctly.**

**Caveats:** in-sample, single league, 258 matches; τ & cap tuned in-sample (validate via the l07 CV
harness); forward-fill 3-min + ±6-min price match is realistic but Betfair LTP ≠ tradeable lay (real lay
pays the spread). Signal is causal (score/time known at the tick; execution lagged).

### Out-of-fold validation (5-fold × 6 repeats over matches)

The in-sample τ/cap were a worry. Two CV designs give a sharp methodological lesson:

- **Tuning (τ, cap) per train fold, scoring on test:** uplift +0.056 ± 0.109, **t = 0.51 (n.s.)** — the
  per-fold grid-selection of τ on ~200-match folds injects more selection noise than signal.
- **Pre-committed rule τ = −0.05 (no per-fold tuning):** OOS uplift **+0.306 ± 0.059 per fold, t = 5.21**,
  positive in 73% of folds. Per-fold mean × 5 folds (≈1.53) **exactly matches the full-sample uplift
  (1.528)** ⇒ at a fixed threshold there is **essentially no overfitting** — the in-sample gain replicates
  OOS. Robust across cap: τ=−0.05 gives t = 5.25 / 5.21 / 5.02 at cap 0.20 / 0.25 / 0.30.

**Takeaway:** the exit overlay genuinely generalizes, but **pre-commit the threshold from theory** (exit
once the market overtakes the model by ~5 pts); do *not* data-mine τ per fold. τ=0 is only marginal
(t=1.84) — the small negative buffer matters.

### Add-to-position side — built, HURTS growth (asymmetry)

Backing more when the model edge *grows* past τ_add lowers growth (3.08 → 2.33 at size 0.5; → 0.55 at
size 1.0) **even though the adds are +EV** (ROI +11.5%, hit 45.6% @ avg odds 4.95, n=688). Textbook
ROI-vs-growth: +EV but growth-negative because it piles high-variance correlated exposure onto an
already-cap-sized book; the convex log penalty exceeds the EV. **The in-game model is valuable on the
REDUCE side only** — exit on edge-decay (variance↓, growth↑), never press adds. A two-sided rebalancer
should be asymmetric. (Caveat: this is *net* adding; a risk-neutral Kelly-target rebalance that funds adds
by trimming elsewhere — total risk flat — is untested.)

### Distributional `BayesianKelly` sizing — built, does NOT improve growth (negative result)

Replaced the binary full-exit action with a *continuous* lay-off sized from the model's uncertainty:
per tick compute **P_model as a distribution** (one value per posterior draw of μ_h, μ_a), feed into the
Baker–McHale `BayesianKelly` to get the uncertainty-shrunk target exposure `f_target`, and lay off
`φ = 1 − clamp(f_target / stake, 0, 1)`. Aligned at the validated trigger (τ=−0.05):

| strategy | log-growth |
|---|---|
| hold | 3.076 |
| mean-based full exit (τ=−0.05) | **4.423** |
| distributional `BayesianKelly` sizing | 4.288 |

It **underperforms** the simple rule. Diagnostic: at the exec tick the target chooses **full exit 86% of
the time** (mean φ=0.88); only 2.9% become partial, and those marginally hurt. Two reasons the uncertainty
adds nothing: (1) the `:linear` posterior is **tight** (P_model std ≈0.035) so the shrinkage is small
(~13%) and rarely reaches the partial zone; (2) the trigger fires *because* the edge is gone, and a gone
edge ⇒ `f_target≈0` (full exit) regardless of confidence. Distributional sizing would only matter with a
**wider posterior** (sparse data / hierarchical / multi-league model where parameter uncertainty is real).
**Keeper = the simple validated rule: full exit at mean edge ≤ −0.05.** Code: `dist_exit_fwd` / `bkelly` /
`pmodel_draws` in r02.

### Still open
Risk-neutral Kelly-target rebalance for the add side; validation on a second league (ScottishLower has
Betfair in-play) to confirm τ=−0.05 transfers across competitions.
