# DataStore Feature Review — better signals than momentum

Computed live on `sofascrape_db` via kaimon REPL, 2026-06-05. Same rigour as the
momentum study: split-half **reliability** (is it a stable team trait?) + **OOS
incremental value over rolling xG** (does it tell us about latent strength that xG
doesn't?). Team-game panel: 16,918 team-games, **130 teams / multiple leagues**
(much broader than the 20-team Irish momentum subset).

> **Maths/methods:** every formula, estimator, and test used here is defined in
> `docs/feature_validation_methodology.md` (§5 maps each result below to its test).

## The framework (unchanged)
A feature helps the model's latent attack/defence strength iff it is an extra
*measurement of that latent* carrying information **beyond xG**. The model already
treats xG as a second, low-noise measurement of λ; the question is what else is a
good measurement. Reliability tells us how noisy the measurement is; OOS partial
correlation (controlling rolling xG) tells us how much *new* information it adds.

## Reliability (Spearman-Brown, split-half on odd/even games)

| Feature | reliability | vs xG (0.86) |
|---|---|---|
| touchesInOppBox | **0.97** | cleaner than xG |
| ballPossession | 0.96 | cleaner |
| finalThirdEntries | 0.95 | cleaner |
| totalShotsInsideBox | 0.94 | cleaner |
| cornerKicks | 0.92 | cleaner |
| bigChanceCreated | 0.92 | cleaner |
| shotsOnGoal | 0.88 | ~xG |
| expectedGoals | 0.86 | (reference) |
| goals (gf) | 0.86 | (reference) |

Several stats are **less-noisy team traits than xG itself.**

## OOS incremental value over rolling xG (partial correlation)

Predictor = pre-match rolling-diff of the stat (past games only); control =
rolling xG-diff; target = this match's goal-diff / xG-diff. N ≈ 4,000–8,000.

| Feature | → future goals | → future xG | DB coverage (matches) |
|---|---|---|---|
| **bigChanceCreated** | **0.190** | **0.239** | 4,477 |
| **touchesInOppBox** | **0.144** | **0.205** | 4,232 |
| totalShotsInsideBox | 0.131 | 0.156 | **7,557** |
| ballPossession | 0.112 | 0.149 | **7,909** |
| shotsOnGoal | 0.118 | 0.141 | **8,047** |
| cornerKicks | 0.111 | 0.131 | **8,459** |
| finalThirdEntries | 0.102 | 0.142 | 4,694 |
| *momentum AUC (reference)* | *0.107* | *0.130* | *1,146* |
| xG coverage for comparison | (baseline) | (baseline) | 4,282 |

**Every candidate beats or matches momentum's incremental signal**, and the top
two beat it substantially. Crucially the volume stats (shots, corners, possession)
have **~2× the coverage of xG**.

### Combined (they are mutually correlated — don't stack all 7)
Predicting future xG-diff, nested OLS, N=1,672:

| Model | R² |
|---|---|
| rolling xG only | 0.094 |
| + touchesInOppBox | 0.142 |
| + bigChanceCreated | 0.163 |
| + totalShotsInsideBox | 0.164 (adds ~0, collinear) |

A parsimonious **{xG, touchesInOppBox, bigChanceCreated}** set ~doubles explained
variance of future xG-diff vs xG alone (+73% relative).

## Two distinct, both-useful roles

1. **Better latent-strength regularization** (the xG-pillar logic, extended).
   `bigChanceCreated` and `touchesInOppBox` are additional, lower-noise
   measurements of attacking threat. Adding them as measurement pillars (or as a
   composite pre-match "attacking threat" prior) pins the latent attack/defence
   down better than xG alone — especially valuable for low-data teams (promoted
   sides, season openers) where shrinkage matters most.

2. **xG coverage extension.** `shotsOnGoal` / `totalShotsInsideBox` /
   `cornerKicks` / `ballPossession` cover ~8,000 matches vs xG's 4,282. Where xG
   is missing, a shot/territory composite can stand in as the Pillar-A signal with
   real predictive content (far better than the momentum fallback idea).

## Caveats — applied honestly, same as for momentum

- **Market subsumption (for EDGE).** These are public signals; the closing line
  prices them. Against an efficient market they will not generate edge on their
  own. Their value is (a) better *measurement* of latent strength (helps NoMarket
  core + low-data teams), (b) coverage, and (c) possible residual edge only in
  thin lower-league markets — which needs its own per-league test. The signal here
  is ~2× momentum's, so that residual is more plausible, but not assumed.
- **Game-state confound / endogeneity.** Territory metrics (touchesInOppBox,
  possession, finalThirdEntries) are score-state-distorted like momentum — a
  leading team cedes territory. At the *rolling* level this washes out (why the
  OOS signal is real), but **do not enter them per-match into the likelihood** —
  same bad-control risk. Event counts (bigChanceCreated, shots, corners) are
  safer per-match. Preferred entry for all: rolling-history prior, not a per-match
  observation pillar.
- **Coverage of the betting leagues.** bigChance/touches coverage (~4.2–4.5k) may
  be thinner in the specific curated lower leagues that are bet — check coverage
  per `tournament_id` before committing.

## Poisson-GLM sandbox (faithful to the Dixon-Coles likelihood)

All three are non-negative counts; on the modelling subset mean goals = 1.37,
var = 1.45 (ratio 1.06) → **Poisson is appropriate** (negligible overdispersion;
NegBin not needed yet). Structured as the real model is: predict a team's goals
from *its* rolling attacking counts × *opponent's* rolling defensive counts (log
link), chronological 70/30 split, scored on **held-out per-game Poisson
log-likelihood** (the count analogue of the model's log-loss). N = 3,428
team-games (limited by bigChance/touches overlap).

| Model | held-out LL | lift vs null | in–out gap (overfit) |
|---|---|---|---|
| null (intercept) | −1.5252 | — | — |
| xG only | −1.4883 | +0.0369 | 0.015 |
| shotsInsideBox only | −1.4875 | +0.0377 | 0.013 |
| bigChance only | −1.4935 | +0.0317 | 0.015 |
| touches only | −1.5009 | +0.0243 | 0.020 |
| xG + bigChance | −1.4811 | +0.0441 | 0.010 |
| **xG + bigChance + touches** | **−1.4781** | **+0.0471** | 0.009 |
| all 4 | −1.4772 | +0.0478 | 0.010 |

**Reading it:**
- xG alone lifts held-out LL +0.037 over null. Adding **bigChance + touches lifts
  another +0.010** (~+28% on top of what xG delivers) — real out-of-sample signal,
  and the small in–out gap (~0.01) says it's **not overfit noise**.
- **shotsInsideBox *alone* ≈ xG alone** (−1.4875 vs −1.4883) — equally predictive,
  with ~2× the coverage. Strong support for using it as the xG coverage-extender.
- **Diminishing returns:** after {xG, bigChance, touches}, the 4th feature adds
  ~nothing (collinear). Parsimonious set confirmed.
- *Caveats:* single split, N limited by bigChance/touches coverage; the
  shots≈xG comparison is on the all-4 subset (coverage advantage realised only on
  the full ~7.5k). Re-run as rolling-origin CV before committing.

## Recommendation (ranked)

1. **bigChanceCreated** — strongest incremental signal (+0.24 on future xG), good
   reliability, safe (event count). Best single addition.
2. **touchesInOppBox** — highest reliability (0.97), strong signal, complements
   bigChance (the two together carry most of the joint value). Territory metric →
   rolling-prior only.
3. **High-coverage trio** (totalShotsInsideBox / cornerKicks / ballPossession) —
   use primarily for **xG coverage extension** to the ~4k extra matches.

Go/no-go is the same as before: **held-out 1X2 log-loss** in the real model,
xG-only vs xG+{bigChance,touches} priors, with attention to low-data teams.
