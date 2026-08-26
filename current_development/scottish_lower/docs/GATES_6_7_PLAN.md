# Gates 6 and 7 — plan of record

Written 2026-08-26, after the 24/25 grid landed (360 OOS fixtures, convergence 7/7).

This is the design *before* the code, so the choices can be argued with rather than
reverse-engineered from an implementation. Everything here is a decision, not a discovery.

---

## The shape of the remaining work

```
gate 6   are the probabilities any good?          outcomes + market as a BASELINE
gate 7   does it make money?                      Betfair close + staking + commission
```

They are separate because they fail differently, and because **the ranking can flip between
them.** This is not hypothetical in this repository:

- the bigChance A/B had log loss preferring the bigChance pillar, and CLV against Betfair
  close reversed it;
- the APM comparison came out statistically indistinguishable on log loss and separated only
  on growth.

So gate 6 is a **filter**, gate 7 is the **verdict**. If gate 6 fails, gate 7 is measuring
the staking rule rather than the model, which is why the order is fixed.

---

## Gate 6 — Evaluation

### The question

*Are these probabilities well-formed and roughly as good as the market's, per line?*

### The pass criterion — and what it is NOT

**Passing does not mean beating the market.** This has to be stated up front because it is
the easiest way to design a gate that rejects a working model. Prior work on this book found
the model losing narrowly to market on 1X2 in absolute proper-scoring terms while still
producing positive CLV and growth. A gate set at "beats market log loss" would have thrown
that model away.

Gate 6 passes when the model is **not broken**:

| check | threshold |
|---|---|
| calibration slope per line | within [0.7, 1.3] |
| no line catastrophically worse than market | Δ log loss ≤ +0.02 on any single line |
| probabilities well-formed | sum to 1 per market, no 0 or 1, no NaN |
| alignment | identical fixture × selection sets, model vs market |

Being *better* than market on a line is recorded as a finding, not a pass condition.

### Sub-gates

**6a — Book integrity.** Before any score is computed: exactly one winner per market per
fixture (1 of 3 for 1X2, 1 of 2 for BTTS and each O/U line); de-vigged probabilities sum to
1; no missing `is_winner`.

This is where the known Double Chance defect would surface — DC marks 1 of 2 selections as
winner and halves its fair probabilities. DC is deliberately **not in the contract book**, and
6a is what would catch it if someone added it.

**6b — Alignment.** Identical `match_id × selection` sets between model prices and market
prices, asserted *before* any ranking is printed. Two models scored on different fixture sets
are not comparable, and the difference is invisible in the output.

**6c — Model probability sanity.** The score grid sums to `1 - truncation_mass` (gate 5
measured 4.4e-5), so model probabilities are normalised per market before scoring. The
normalisation magnitude is **reported**, not hidden — a large one would mean `max_goals` is
too small for the league.

**6d — Proper scores, per line.** Log loss and Brier for: model, de-vigged Bet365 close,
Betfair close.

Three commitments:

- **Per line, never aggregated across the selections of one market.** Aggregating over
  selections is what reversed the APM headline once already.
- **Fixture-weighted, not fold-averaged.** OOS blocks range from 2 to 24 fixtures; a fold
  average lets a 2-fixture block outvote a 24-fixture one. Pooling all fixtures into one
  table does this automatically — the trap is only in averaging per-fold scores, so both are
  reported side by side to show the size of the difference.
- **Paired differences, not two independent means.** Model and market score the *same*
  fixtures, so the statistic is the per-fixture difference and its standard error. Treating
  them as independent samples inflates the confidence interval to the point of saying nothing.

**6e — Calibration.** Reliability curve per line, plus the slope and intercept of the
calibration regression. Slope < 1 means over-confident, > 1 under-confident. This is the
check that distinguishes "wrong" from "under-dispersed", and prior work on this book found
the model's dispersion running at roughly **half the market's** — so a slope well under 1 is
the expected finding rather than a surprise.

**6f — The draw question.** Gate 5 found mean draw ≈ 0.243 against an empirical Scottish
L1/L2 rate of ~0.25-0.27. Conditionally independent goals with no Dixon-Coles term
under-predict draws, and this is what that looks like.

Tested directly rather than assumed: realised draw rate over the 360 fixtures vs mean
predicted, with a binomial interval. If the deficit is real and significant it is the
strongest available argument for a DC or copula variant, and it belongs in the gate-6
variant comparison rather than in a hunch.

### Inputs and outputs

```
in    tp_grid_latents.df          360 fixtures x posterior λ, r
      ds.odds                     360/360 coverage, odds_close + is_winner + overround_close
      ds.betfair_odds             324/360, ticks -> close price needs summarising
out   tp_book                     tidy: match_id, market, line, selection, p_model,
                                  p_bet365, p_betfair, is_winner, fold
      tp_gate6                    PASS/FAIL table
      tp_score_table              per-line log loss / Brier, model vs both baselines
```

### Known caveat, recorded not resolved

De-vigging is **proportional** (`prob_implied_close / overround_close`), which the package
computes upstream. The 1X2 overround is **10.65%** on this league — chunky next to a
top-flight 5-7% — and proportional de-vigging is known to be biased against favourites at
that level. Shin or power de-vigging would give a different baseline. This shifts the
*baseline*, not the model, so it affects every Δ reported. Recorded so the number is read
with it in mind.

---

## Gate 7 — Growth / CLV

### The question

*Does this model, priced into a real book at real prices with real commission, make money —
and does it beat the closing line?*

### What already exists

`src/Portfolio` was audited and graduated: `MatchBook`, slate grouping (`DailySlate`,
`SingleMatchSlate`), `stake_slate(policy, slate, ctx)`, `path_metrics`, `bootstrap_roi`,
`attribution`. Gate 7 **assembles** these. It does not write staking maths.

### Decisions this gate needs, which gate 6 does not

1. **Entry price and time.** CLV is entry price versus closing price, so "when do we bet"
   must be defined. Betfair data is tick-level with `minutes_to_kickoff`. Prior work anchored
   per match because that field is wall-clock, not match-clock.
2. **Curation.** Prior results are unambiguous that this dominates the model: on the full
   book every Kelly variant lost, and curating to totals + BTTS inverted the result. Which
   lines are eligible is a gate-7 parameter, and must be **declared before** the numbers are
   looked at, not chosen after.
3. **Portfolio cap.** Σ simultaneous stakes ≤ ~0.2. Independent per-bet full Kelly went
   bankrupt in prior sims; the cap is the dominant lever, not the edge estimate.
4. **Slate grouping.** `DailySlate` matches how the cap is supposed to work — simultaneous
   exposure is per matchday, not per match.
5. **Coverage.** Betfair reaches 324/360 fixtures (90%). The 36 missing are dropped, and
   **which** 36 must be reported — if they are systematically the small-market fixtures, the
   backtest is measuring a slightly different league.

### Primary and secondary

**CLV is the discriminator; growth `G` is secondary.** Growth over 360 fixtures on one season
is far too noisy to rank models — prior work put the honest edge interval at
[-1.5%, +20.9%], which contains zero comfortably. CLV is the lower-variance signal and the
one that survives out of sample.

### Traps carried in from prior work

- Alpha fitted on the same data it is evaluated on overfits catastrophically (-91% OOS).
  Any tuned parameter must be walk-forward or fixed in advance.
- Path metrics computed on a scrambled bet order silently produce nonsense drawdowns.
- No slate cap → negative bankroll.

---

## After gate 7

Model 01 becomes the **reference baseline**, and everything after is judged against it.

1. **Variants of 01**, if gate 6 justifies them — Dixon-Coles or copula for the draw deficit
   (6f), and league-varying home advantage for 56 vs 57, which is a real modelling question
   since two divisions are pooled and there is no reason to assume a shared home edge. Both
   need the same gates 0-7, and the comparison is per line at gate 6 and on CLV at gate 7.
2. **`02_apm_player_poisson`** — rebuilt from scratch; the graduated APM is not trusted.
3. **`03_open_play_recombination`** — last, because it is the largest and has the most
   variants.

Open tickets do not block any of this: T002 is AD cost only, T003 mis-prices 0.56% of
fixtures on one side.
