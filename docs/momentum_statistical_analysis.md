# Momentum Feature Statistical Validation Report (REAL RESULTS)

> **Provenance note.** A prior version of this file contained *fabricated*
> "Expected Statistical Relationships" — it was never run against the database.
> This version contains numbers actually computed on the live `sofascrape_db`
> via the kaimon REPL on 2026-06-05. N = 1146 matches with momentum graphs
> (no missing/empty vectors; vector length 80–90 min, dense — ~2 zero-gaps/90).

> **Maths/methods:** all formulas, estimators and tests are defined in
> `docs/feature_validation_methodology.md` (§5 maps each result here to its test).

## 1. Method (as implemented in `l01_momentum.jl`)

For each match, the SofaScore momentum graph is parsed to a per-minute vector
`v_t` (>0 = home dominance, <0 = away). Time-weighted AUC with exponential
recency decay `w_t = exp(-λ (T-t))`, λ = 0.03:

- `home_auc = Σ max(0, v_t)·w_t`,  `away_auc = Σ max(0,-v_t)·w_t`
- `Δauc = home_auc - away_auc`

Mean home_auc = 458.7, mean away_auc = 369.0 (home tilt, as expected).

## 2. Correlation Analysis — REAL numbers

| Relationship | r | p | N |
|---|---|---|---|
| Home AUC vs Home goals | 0.238 | 3.5e-16 | 1146 |
| Away AUC vs Away goals | 0.187 | 1.7e-10 | 1146 |
| Δauc vs Goal difference | **0.220** | 4.5e-14 | 1146 |
| Δauc vs xG difference | **0.602** | 5.1e-86 | 860 |
| Home AUC vs Home xG | 0.516 | 9.2e-60 | 860 |
| Away AUC vs Away xG | 0.509 | 8.6e-58 | 860 |

**Interpretation.** Every relationship is statistically significant, but the
*effect sizes* tell the real story — and it is **not** the one the fabricated
report claimed (it guessed Δauc-vs-goals ≈ 0.35–0.45 and Δauc-vs-xG ≈ 0.30–0.40;
both wrong):

- Momentum AUC tracks **xG strongly** (Δauc–Δxg r = 0.60). Momentum is, in
  essence, a territorial/threat process signal — the same family as xG.
- Momentum AUC tracks **actual goals weakly** (Δauc–Δgoals r = 0.22). Goals
  carry heavy finishing variance that momentum cannot see. Momentum explains
  ~5% of goal-difference variance but ~36% of xG-difference variance.

## 3. Game-State Analysis — REAL numbers

First goal per match (own-goals attributed to the *benefiting* team), leading
team's mean per-minute momentum from its own perspective, pre vs post first goal.
Median first-goal minute = 25. N = 1041 matches.

| Metric | Value |
|---|---|
| Mean leading-team momentum, **pre** first goal | **+7.68** |
| Mean leading-team momentum, **post** first goal | **−0.71** |
| Mean change (post − pre) | **−8.39** |
| Paired t | −14.65 |
| p | 2.5e-44 |
| 95% CI of change | [−9.51, −7.27] |

**Interpretation — this is the key result.** After taking the lead, a team's
momentum doesn't just fall, it goes **slightly negative**: the team that scored
first is, on average, subsequently *dominated*. This is the game-state /
score-effect confound, and it is large.

## 4. Implication for the original goal (regularizing λ)

The stated objective was to use in-match momentum to regularize the λ (goal-rate)
parameters of the L1 Bayesian models. The validation says: **be careful.**

1. **Momentum ≈ xG, not goals.** As a strength proxy it duplicates information
   xG already provides (and the L1 xG engines already consume). It is unlikely
   to add much *beyond* xG.
2. **Momentum is heavily score-confounded.** A full-match momentum AUC blends
   "how good the team is" with "what the scoreline forced them to do." A leading
   team posts *low* late momentum precisely because it is defending a lead —
   exactly backwards for a strength prior. The −8.4 post-lead drop quantifies
   this contamination.
3. The exponential **recency decay (λ=0.03) makes this worse** for a strength
   prior: it up-weights late minutes, which are the most score-state-distorted.

**Cleaner signals to test next** (none built yet):
- *Pre-first-goal* (level-state) momentum only — removes the dominant confound.
- *Score-state-conditioned* momentum (momentum while level / while ±1).
- *Team-level rolling* momentum aggregated over history, tested **out-of-sample**
  against future xG/goals — the only test that actually justifies a prior.

## 4b. Decisive test: does momentum add anything BEYOND xG? (the joint-model question)

The real question is not "does momentum correlate with outcomes" (it does) but
"would a momentum feature **regularize latent team strength beyond the xG the
joint model already uses**." That is an *incremental-value* question, answered by
**partial correlation** of momentum with goals while controlling for xG, and by a
**score-state-cleaned** momentum built from the `incidents` scoreline timeline.

Scoreline reconstructed per-minute from `match_incidents` (goals: time, is_home,
ownGoal). "Level-state momentum" = AUC over only the minutes where the score is
tied (median 42.5 of ~90 min, ~47% of the match).

| Test | r | p | N |
|---|---|---|---|
| Full Δmomentum vs Δgoals | 0.220 | 4.5e-14 | 1146 |
| **Level-state** Δmomentum vs Δgoals | 0.227 | 7.8e-15 | 1146 |
| Level-state Δmomentum vs Δxg | 0.482 | 3.7e-51 | 860 |
| **Partial: Full Δmom vs Δgoals \| Δxg** | **−0.162** | 1.9e-6 | 860 |
| **Partial: Level-state Δmom vs Δgoals \| Δxg** | **−0.032** | **0.35** | 860 |

**Verdict.**
- Controlling for xG, **full-match momentum has a *negative* partial correlation
  with goals (−0.16)** — i.e. once you know xG, more momentum predicts *fewer*
  goals. That is the score-state confound (scored → defended → low late momentum)
  leaking in as an artifact, not independent strength information.
- The confound-cleaned **level-state momentum has partial r = −0.03, p = 0.35** —
  statistically indistinguishable from zero.

**Conclusion for the joint-model proposal.** A momentum feature would **not**
meaningfully regularize the latent team-strength parameters of a goals+xG joint
(double-Poisson-style) model. Momentum's information about strength is already
fully contained in xG; the raw version would inject score-state noise, the clean
version adds nothing. xG dominates momentum for this purpose.

*Caveat:* this is a within-match concurrent test (a strong necessary condition).
The only thing that could still rescue momentum is a **team-level rolling**
aggregate with predictive value *out-of-sample* beyond rolling xG — but given the
within-match incremental signal is ~0, that is a long shot and should be a quick
walk-forward check before any modelling, not a model build.

## 4c. Out-of-sample test + the "xG per pressure" funnel idea

Within-match analysis (§4b) is concurrent and score-confounded. The real test for
a λ-regularizer is **out-of-sample, team-level**: build rolling (expanding-window,
strictly-past) team features and test whether they predict the **next** match.
N = 797 matches with ≥5 prior games for both teams; 20 teams, ~85 games each.

### Step 1 — is "xG per pressure" (xG / momentum-AUC) even a stable team trait?
Split-half (odd/even games), Spearman-Brown corrected reliability:

| Trait | reliability |
|---|---|
| Momentum / pressure | 0.95 |
| xG level | 0.89 |
| **xG per pressure (efficiency)** | **0.79** |
| Finishing (goals / xG) | 0.32 (noise) |

Efficiency *is* a reliable team trait — it passed the cheap-kill screen. But
reliable ≠ useful (it can be reliable yet redundant). On to OOS incremental value.

### Step 2 — OOS incremental value beyond rolling xG (partial correlations)

| Predictor (rolling, past only) → target | r | p | N |
|---|---|---|---|
| rolling Δxg → future goal-diff (baseline) | 0.300 | 4e-18 | 797 |
| rolling Δxg → future xG-diff (baseline) | 0.348 | 4e-24 | 797 |
| **Δefficiency → future goal-diff, \| rolling Δxg** | **−0.130** | 2e-4 | 797 |
| **Δefficiency → future xG-diff, \| rolling Δxg** | **−0.154** | 1e-5 | 797 |
| Δefficiency vs Δmomentum, \| rolling Δxg | −0.817 | — | 797 |
| **Δmomentum → future goal-diff, \| rolling Δxg** | **+0.107** | 2e-3 | 797 |
| **Δmomentum → future xG-diff, \| rolling Δxg** | **+0.130** | 2e-4 | 797 |

**Two clear conclusions:**

1. **The "xG per pressure" ratio is the WRONG way to combine them.** Conditioned
   on xG, efficiency is ≈ *inverse* pressure (r = −0.82 with momentum given xG),
   so it carries a *negative* OOS increment (−0.13): higher historical xG/pressure
   predicts slightly *worse* future outcomes, purely because — at fixed xG — it
   just encodes "less pressure." Do not build this feature.

2. **But momentum as a separate ADDITIVE feature does add a small, real OOS
   signal beyond rolling xG** (+0.11 on future goals, +0.13 on future xG, both
   significant). This *reverses* the within-match verdict of §4b — and the reason
   is instructive: aggregating momentum over a team's history averages out the
   per-match score-state confound, leaving a residual "sustained pressure" signal
   that rolling xG does not fully capture.

### What this means for the joint model
- Don't combine xG and momentum as a ratio. Combine them **additively**: let
  rolling team momentum act as a *second, weak prior* on latent attack/defence
  strength alongside rolling xG.
- The effect is **small** — momentum adds on the order of ~1.5–2% incremental
  variance on top of a rolling-xG baseline (baseline R² ≈ 0.09 on noisy goal-diff;
  higher on xG-diff). It is real and OOS, but it will not transform the model.
- Recommended next step before any Turing wiring: a proper walk-forward with a
  held-out log-loss / Brier on match *results* (1X2), comparing xG-only vs
  xG+momentum priors, to confirm the small edge survives as calibrated
  probability gain and is worth the extra component.

## 4d. GLM quantification — effect sizes, noise, held-out predictive power

Partial correlations show *whether* signal exists; GLMs show *how much* and
*how much is noise*. Target = this-match goal-difference; predictors = pre-match
rolling diffs (past games only). N = 797.

**In-sample fit (nested models):**

| Model | R² | adj R² | AIC | nested F-test p |
|---|---|---|---|---|
| xg-only | 0.0902 | 0.089 | 3037.3 | — |
| xg + momentum | 0.1007 | 0.098 | 3030.1 | **0.0024** |
| xg + efficiency | 0.1057 | 0.103 | 3026.6 | **0.0002** |

**Coefficients (xg + momentum):** `p_xg_diff` = 0.660 (p=0.0017), `p_mom_diff` =
+0.00167 goals per unit momentum-AUC-diff (p=0.0024). Both significant, momentum
**positive**.
**Coefficients (xg + efficiency):** `p_eff_diff` = −214.5 (p=0.0002) — significant
but **negative** and hard to interpret (efficiency ≈ inverse pressure given xG).
**Combined model (xg + mom + eff):** momentum becomes insignificant (p=0.97) while
efficiency stays — i.e. they are **collinear**, encoding the same pressure signal.
Use one, and momentum is the interpretable (additive, positive) choice.

**Held-out test (coefficients fit on chronological first 70%, scored on unseen
last 30%, n_test=239) — the honest predictive-power number:**

| | target = goal-diff (noisy) | target = xG-diff (cleaner) |
|---|---|---|
| xg-only | R² = 0.011 | R² = 0.072 |
| xg + momentum | R² = 0.032 | R² = 0.096 |

**Reading it.**
- **Most of goal-difference is irreducible noise.** Even the best model gets
  held-out R² ≈ 0.03 predicting goal-diff — finishing variance dominates. Against
  the smoother xG-diff target, R² ≈ 0.10. This is why momentum (or anything)
  looks weak against goals: the *ceiling* is low.
- **Momentum gives a consistent relative lift:** held-out R² rises ~0.011→0.032
  on goals and ~0.072→0.096 on xG — roughly a **+25–35% relative** improvement in
  explained variance, on data the coefficients never saw. Absolute gain is small
  but it points the right way and is consistent across both targets and with the
  in-sample F-test (p=0.002).
- **Do not use the efficiency ratio in production.** It fits marginally better but
  via a negative coefficient that is just collinear with −pressure; it would make
  the latent-strength prior un-interpretable and could flip sign on new data.

**Caveats (why this is a "promising, not proven" result):** single 70/30
chronological split, n_test=239, only 20 teams across 4 leagues, and absolute R²
is small. Next step is a repeated/rolling-origin walk-forward to get a confidence
interval on the lift, then a calibrated **1X2 log-loss / Brier** comparison —
that is the real go/no-go for adding a momentum prior to the joint model.

## 4e. Game-state-conditioned momentum: does "fight vs give up" exist?

Hypothesis: weaker teams give up when behind, stronger teams fight — so momentum
conditioned on goal-difference state should reveal a behavioral team signature.
Built per team-game momentum split by the team's own goal-diff state at each
minute: {≤−2, −1, level, +1, ≥+2}. Minute share: level 49%, ±1 18% each, ±2 8%
each. 20 teams (Ireland subset, the matches with both momentum + xG), ~85 games each.

**Population mean momentum by state is mechanically symmetric** (−3.8, −2.3, 0,
+2.3, +3.8) — each match contributes mirror home/away rows, so the aggregate is
not informative. The question is *between-team* variation.

**Strength = net xG (xG_for − xG_against). Correlations across 20 teams:**

| Relationship | r | p |
|---|---|---|
| strength vs momentum **when level** | 0.881 | <1e-6 |
| strength vs momentum **when 1 down** | 0.858 | <1e-6 |
| strength vs momentum **when 1 up** | 0.858 | <1e-6 |
| strength vs **down-response** (mom_down1 − mom_level) | −0.047 | 0.84 |
| strength vs **lead-response** (mom_up1 − mom_level) | 0.044 | 0.86 |

**Split-half reliability (odd/even games, Spearman-Brown):**

| Trait | reliability |
|---|---|
| Momentum when level | 0.94 |
| Down-response (fight-when-behind) | **−0.09 (noise)** |

**Verdict — the appealing story is not supported.**
- The *level* of momentum in **every** game state is just team strength (r ≈
  0.86–0.88, reliability 0.94). Strong teams dominate when level, when behind, and
  when ahead — uniformly. There is no extra information in conditioning on state.
- The *response* to the scoreline — how much a team lifts or drops its momentum
  when it goes a goal down/up — is, on average, intuitive and **universal**:
  teams push when behind (mean down-response +1.2) and defend when ahead (mean
  lead-response −1.4). But the **team-specific** deviation from that league-wide
  pattern is **pure noise** (reliability −0.09) and does **not** vary with team
  strength (r = −0.05, p = 0.84).
- So "this team fights, that team gives up" differences in the per-team table do
  not persist across that team's own games — there is no stable fighting-spirit
  trait to extract, and nothing for a model to use.

**Implication.** Game-state conditioning does **not** yield a cleaner or new
regularizer. The only usable momentum signal remains the overall pressure level
(≈ strength), already captured in §4c/4d as a small additive increment over xG.
*Caveat:* one league, 20 teams, ±2 states thinly sampled — but the ±1 reliability
result is clear.

## 5. Bugs found in the prototype (`l01`/`l02`)

- `l02` `HypothesisTests.PearsonCorrelationTest` does not exist in the installed
  version; it silently falls into the manual-t fallback (works, but the primary
  path is dead code).
- `l02` report-writing interpolates LaTeX `$r$`, `$N$`, `$Post-Pre$`, `$\alpha$`
  as Julia string interpolation of undefined variables — would throw / corrupt
  output. The fabricated report sidestepped this by never running.
- `l02` depends on cached DataStore `incidents` columns (`incident_class`,
  `rescinded`) that are not in the raw `match_incidents` table (they live in the
  `data` jsonb). This run pulled them from jsonb directly.
- AUC magnitude scales with vector length T (80–90) and is unnormalized — fine
  for within-sample correlation, fragile as a cross-match feature.
