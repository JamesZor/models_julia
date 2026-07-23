# RESEARCH — RAPM methodology and player-rating validation

WP0 output. Batch `batch-20260723-123823-91a46909` (5 parallel workers, Gemini 3.1 Pro),
**then adversarially verified by Claude against primary sources**. Date: 2026-07-23.

> **Read this section first.** The research fanout produced several **fabricated or wrong
> citations**, listed in §7. Everything below is tagged:
> **[VERIFIED]** = checked against the primary source by fetching it;
> **[PLAUSIBLE]** = consistent across ≥2 workers and with the source paper, not independently
> checked; **[UNVERIFIED]** = single weak source, treat as a lead only.

---

## 1. The single most useful find: Hvattum's similarity regularization

**[VERIFIED — read directly from the paper PDF]**
Hvattum, Arntzen & Pantuso (2020), *Offensive and Defensive Plus–Minus Player Ratings for
Soccer*, **Applied Sciences 10(20), 7345**. https://www.mdpi.com/2076-3417/10/20/7345

This is the most methodologically developed plus-minus formulation in the football literature,
and it solves **our** central problem (§NOTES "Central statistical risk") more directly than
anything the workers proposed.

### 1.1 Shrink toward similar players, not toward zero

Plain ridge shrinks a low-minute player's rating to **zero** — i.e. to the global average — which
is the well-known collapse. Hvattum et al. instead penalise

```
f^REG.PLAYER(p) = ( f^AUX(p, T, 1) − (w^SIM / |P^SIM_p|) · Σ_{p′ ∈ P^SIM_p} f^AUX(p′, t^SIM(p,p′), w^AGE) )²
```

where `P^SIM_p` is the set of players **who have played alongside `p` on the same team for the
most minutes**, and `w^SIM` "controls the extent to which overall ratings of players with few
minutes played are shrunk towards **zero** or towards the **overall ratings of similar
players**".

Why this matters for us: it is a **non-zero prior mean that requires no extra data**. No box
score, no age curve, no external rating — just who played with whom, which we already have from
the teamsheets. Compare the workers' suggestion (build a statistical-plus-minus prior from
box-score rates, §3), which needs event data the lower leagues barely have. For Scottish League
One/Two this is strictly the better lever, and it is published, soccer-native precedent.

Note the direction of the shrinkage is *teammates*, so it partially recovers "this player is on a
good team" — which is exactly the pitfall in §5.4. The paper accepts that trade; `w^SIM` is the
dial. **We should treat `w^SIM` as a tuned hyper-parameter, not a constant**, and report ratings
at `w^SIM = 0` (pure ridge) alongside the tuned value.

### 1.2 Segment weights are a PRODUCT of three factors, not just time decay

**[VERIFIED]** Our plan (following the 2019 paper) had only exponential time decay. The 2020
formulation uses `w(m,s) = w^TIME · w^DURATION · w^GOALS`:

| factor | form | purpose |
|---|---|---|
| `w^TIME(m,s)` | `exp(ρ₁ (T − t^MATCH(m)))` | recency (the 2019 paper's ζ) |
| `w^DURATION(m,s)` | `(d(m,s) + ρ₂) / ρ₃` | a 3-minute segment is nearly pure noise; weight it down |
| `w^GOALS(m,s)` | `ρ₄` if `|g^S| ≥ 2` **and** `|g^E| ≥ 2`, else `1` | **garbage-time discount**: segments that start *and* end ≥2 goals apart |

The duration and garbage-time weights are cheap to add and directly attack variance, which is
our binding constraint. **Adopt both.**

### 1.3 Other verified specification details

- **Two observations per segment**, one per team's perspective (dependent variable = goals
  scored *by that team*, not the difference). This is what enables **separate offensive and
  defensive ratings** per player. Goalkeepers are given **no offensive rating** (`V_p = {GK}` is
  special-cased throughout).
- **Ratings are per-90 contributions**: explanatory terms scaled by `d(m,s)/90` and normalised by
  `11/|P|` for the number of contributing players.
- **Regularization is deliberately NOT applied** to home-field advantage or red-card effects —
  only to player ratings, the offensive/defensive split, and the age curve. Confirms the plan's
  "do not penalise the intercept".
- **Red cards**: `r(m,s,n) ∈ {−1,0,1}` for `n = 1..4`, signed by which side is short-handed, with
  a separate offensive and defensive coefficient. Injured players who cannot be replaced because
  substitutions are exhausted are **modelled identically to a red card** — a nice touch we can
  copy for free.
- **League/competition adjustment** is per-player: `(1/|C_p|) Σ_{c ∈ C_p} β_c^COMP`, averaged over
  the set of domestic competitions the player has appeared in. This is *simpler* than the 2019
  paper's `m_il` "adapted to competition" counting rule and is a better fit for our four-tier
  pool, where players move between 54/55 and 56/57 (**576 of 3,192 players do**).
- **Age** enters as a piecewise-linear function over knot ages, with the exact age expressed as a
  convex combination of the two nearest knots, plus a second-difference smoothness penalty. We
  likely lack birth dates — **out of scope**, but the smoothness-penalty trick is worth
  remembering.
- Solved as an **unconstrained quadratic program** by parallel gradient descent (C++), because
  the dimensions defeat a direct solve.

---

## 2. Validation methodology — the reliability / validity framework

**[VERIFIED]** Hvattum & Gelade (2021), *Comparing bottom-up and top-down ratings for individual
soccer players*, **International Journal of Computer Science in Sport 20(1)**.
https://reference-global.com/article/10.2478/ijcss-2021-0002

This is the canonical reference for **our WP7** and it defines exactly two axes:

- **Reliability** — "whether similar ratings are produced when using **different data sets**".
  This is precisely the split-half / odd-even construction in the plan's WP7.2. One worker
  claimed split-half is "unsuitable for RAPM" (§7); that claim is **contradicted** by the
  discipline's own reference paper, which builds its comparison on exactly this axis.
- **Validity** — "the quality of **match outcome forecasts** generated when the ratings are used
  as **predictor variables**". This is the plan's WP7.4, and it is also the criterion the 2019
  paper used to tune `(λ, ζ)`. One harness, two jobs — as planned.

**Headline result:** plus-minus (top-down) ratings **outperformed** bottom-up action-valuation
ratings on both reliability and validity. Encouraging for this stream's premise: we are replacing
a bottom-up rating (SofaScore) with a top-down one, and the literature says top-down is the
stronger construct.

### 2.1 What correlation with SofaScore should we actually expect?

**[VERIFIED]** Gelade & Hvattum (2020), *On the relationship between +/− ratings and event-level
performance statistics*, **Journal of Sports Analytics 6(4)**.
https://journals.sagepub.com/doi/10.3233/JSA-200432

Event-level (bottom-up) statistics explain only **22%–38% of the variance in plus-minus
ratings**, by position:

| position | R² |
|---|---|
| Goalkeepers | 22.4% |
| Defensive midfielders | 30.7% |
| Midfielders | 35.0% |
| Defenders | 36.1% |
| Attacking midfielders | 36.4% |
| Forwards | 37.9% |

Strongest individual correlates are modest: touches/90 `r = 0.35`, successful passes/90
`r = 0.34`, save-to-shot ratio `r = 0.30`, pass completion % `r = 0.27`.

**This converts our WP7.1 from a vibe into a quantitative expectation.** A SofaScore rating *is*
a bottom-up event-based rating. So the expected agreement band between our RAPM and SofaScore is
roughly **ρ ≈ 0.47–0.62** (√R²), lowest for goalkeepers, highest for forwards. Concretely:

- ρ near 0 ⇒ our RAPM is broken (or pure noise).
- ρ in ≈0.45–0.65, ordered GK < mid < forwards ⇒ **exactly right**; that is what a working
  top-down rating looks like against a bottom-up one.
- ρ near 0.9 ⇒ suspicious. We would not have built a top-down rating; we would have rediscovered
  the box score.

The correct reading is therefore that **low-ish correlation is the success criterion, not the
failure criterion** — which inverts the naive interpretation. The decisive evidence has to come
from reliability (WP7.2) and validity (WP7.4), not from agreement (WP7.1).

**Also important, and a caution for WP6:** the same paper finds that "incorporating the
event-level statistics only **marginally** improves the predictive power of plus-minus ratings."
So a box-score-informed prior may buy less than the basketball literature advertises. Another
reason to prefer Hvattum's teammate-similarity prior (§1.1), which is cheaper and does not depend
on event coverage we lack in 56/57.

### 2.2 Minimum minutes

**[VERIFIED]** Gelade & Hvattum (2020) **excluded players with fewer than 540 minutes** (six full
matches) from their regression analyses, because per-90 statistics are unstable below that.
The 2019 paper separately used **≥900 minutes** for its "best players" tables.

⇒ Use **540 minutes** as the analysis floor and **900** for any published top-N list. The
workers' claim of "500–1,000 minutes" **[PLAUSIBLE]** is consistent with both.

---

## 3. Prior-informed / box-score-informed plus-minus

**[PLAUSIBLE]** The mathematics is standard and safe: replace the ridge penalty `λ‖β‖²` with
`λ‖β − β₀‖²`, which shrinks toward an informed prior mean `β₀` instead of zero. Equivalent to a
Gaussian prior `β ~ N(β₀, σ²)`, so it drops straight into the Bayesian arm (WP6) as a prior mean.

**[UNVERIFIED]** The specific basketball implementations the workers cited (BPM, LEBRON, DARKO,
ESPN RPM) are practitioner systems, not peer-reviewed, and I did not verify the descriptions.
The claimed pattern — two-stage estimators that regress box-score features onto multi-year RAPM
and use the fit as the prior — is consistent across workers and is a reasonable lead, but the
soccer-specific evidence in §2.1 says the payoff here is likely small.

**[FABRICATED]** The cited soccer analogue `github.com/Torvaney/panna` **returns HTTP 404**. Do
not chase it.

**Recommendation:** implement the prior mean as a *pluggable* `β₀` and run three settings —
`β₀ = 0` (plain ridge), `β₀ =` teammate-similarity (§1.1, the published soccer method), and
`β₀ =` box-score SPM (the basketball idea) — and let WP7 decide. The teammate-similarity arm is
the one with actual football evidence behind it.

---

## 4. Identifiability and collinearity

**[PLAUSIBLE]** The qualitative picture is uncontroversial and matches the source paper's own
§6 caveats: few substitutions ⇒ highly collinear design; always-together clusters (centre-back
pairings, keeper/backup) are not separable by construction; ridge responds by splitting credit
equally among them.

**[UNVERIFIED — and this is the weakest area of the fanout]** Worker `sub_02` returned only bare
domain names (`liverpool.ac.uk`, `arxiv.org`, `cmu.edu`, `oup.com`) rather than real URLs, so
none of its specific numbers can be trusted. In particular:

- "commercial databases require 500–1,000 minutes" — **superseded** by the verified 540/900
  figures in §2.2.
- "single-season RAPM is too noisy; standard practice is a rolling 2–3 season window" — matches
  the 2019 paper's own 2-year window, so **[PLAUSIBLE]** by corroboration with the primary source.
- No source gave a defensible **observations-to-players ratio** threshold. Our plan's ≥5:1 gate is
  therefore **our own engineering judgement, not a literature standard** — it should be described
  as such in NOTES.md, and the real evidence should come from the empirical standard errors and
  the effective degrees of freedom we measure in WP2, not from a rule of thumb.

**[VERIFIED]** Structured penalization is a live research direction:
Bajons & Hornik (2024), *Regularized Adjusted Plus-Minus Models for Evaluating and Scouting
Football (Soccer) Players using Possession Sequences*, arXiv:2407.17832.
https://arxiv.org/abs/2407.17832 — explores four penalization schemes exploiting football-specific
structure, "the grouping of players into **position groups** as well as into **common strength
groups**", evaluated on 2017/18 La Liga. This is **published precedent for our WP6 design**
(hierarchical `τ` by position group). It also uses **possession sequences** rather than
substitution intervals as the observation unit — a much denser segmentation, but it needs on-ball
event data we do not have for 56/57.

---

## 5. Target variables

**[VERIFIED via the source paper]** The 2019 paper's own motivation: **72% of goal-differential
segments are exactly 0**. Denser targets are the entire point.

**[PLAUSIBLE, strongly corroborated]** Consensus across all workers and consistent with both
source papers:

- Goal differential is too sparse; it produces high-variance, weakly discriminating ratings.
- Ice hockey solved this first, with shot-attempt targets (Corsi/Fenwick) — Macdonald (2012),
  arXiv:1209.1179 — reporting materially reduced standard errors.
- xG is the modern soccer default: it keeps shot *volume* (≈10× goals) while retaining
  discrimination through shot *quality*, and is argued to be the best volume/discrimination
  trade-off.
- Unweighted shot counts sit between goals and xG: denser than goals, but blind to chance quality.

This directly supports the plan's four-target ladder, and specifically supports our
repo-specific **SoT** arm as a legitimate midpoint (the `bbc_xg_proxy` funnel lesson).

### 5.1 Zonal xG without coordinates

**[PLAUSIBLE]** Categorical xG from `zone × body part × set-piece context` is an established
fallback (it is essentially the pre-tracking-era method). The documented cost is "implicit
averaging" — a tight-angle shot and a central shot inside the same coarse zone get the same
value. Expect a real but bounded accuracy loss versus a coordinate model, and expect it to still
comfortably beat unweighted shot counts.

**Sanity anchor from the source paper (Table 3)**, which we should hold our model against: their
coordinate-based Brier scores were **open play 0.0739, headers 0.0872, free kicks 0.0575,
penalties 0.1848**, against baselines of 0.0866 / 0.0994 / 0.0584 / 0.1848. Two things to note:

1. Even *with* coordinates, the gain over the naive per-type base rate is small (open play
   0.0866 → 0.0739). Our zonal model has less headroom to lose than it first appears.
2. **Penalties: no model beat the baseline at all** (0.1848 everywhere). The paper concludes
   penalty outcomes are conditionally random and should all get one constant xG. We should not
   waste effort modelling them.

---

## 6. Consequences for this stream's plan

Concrete changes to make, all justified above:

1. **WP2** — add `w^DURATION` and `w^GOALS` (garbage-time) to the segment weights; currently only
   time decay. Model unreplaceable injured players as dismissals. (§1.2, §1.3)
2. **WP2** — switch the league adjustment to Hvattum's per-player competition average
   `(1/|C_p|) Σ_c β_c^COMP`; simpler than the 2019 `m_il` rule and better suited to our four-tier
   pool. (§1.3)
3. **WP5** — add **teammate-similarity shrinkage** (`w^SIM`) as the headline low-minutes
   treatment, tuned, with `w^SIM = 0` reported alongside. This displaces the box-score prior as
   the primary approach. (§1.1, §2.1)
4. **WP5/WP6** — consider the two-observations-per-segment form to get **offensive/defensive split
   ratings**, with goalkeepers excluded from offensive ratings. Optional; costs a doubling of the
   response but no extra data. (§1.3)
5. **WP7.1** — replace "expect modest correlation" with the quantitative band **ρ ≈ 0.47–0.62,
   ordered GK < midfielders < forwards**, and state explicitly that a *very high* correlation
   would be a red flag, not a success. (§2.1)
6. **WP7** — adopt **540 minutes** as the analysis floor and **900** for top-N tables. (§2.2)
7. **WP7** — reframe the decision rule around the literature's **reliability + validity** axes,
   which is what WP7.2 and WP7.4 already measure. (§2)
8. **WP2** — demote the ≥5:1 ratio gate from "the standard" to "our engineering judgement", and
   make the real evidence the measured standard errors / effective degrees of freedom. (§4)
9. **WP3** — do not model penalties; assign one constant xG. Expect only a small Brier gain over
   per-type base rates, and set expectations accordingly. (§5.1)

---

## 7. Citation errors found in the research fanout

Recorded so nobody re-trusts these:

| Claim | Cited as | Reality |
|---|---|---|
| "Hvattum's xG plus-minus validation" | `mdpi.com/2073-4441/13/2/170` | ISSN 2073-4441 is the journal **Water**. Fabricated. |
| Kharrat et al. 2019 preprint | `arxiv.org/abs/1706.07176` | Actual preprint is **arXiv:1706.04943**. |
| Soccer RAPM framework "panna" | `github.com/Torvaney/panna` | **HTTP 404**. |
| Identifiability sources (`sub_02`) | `liverpool.ac.uk`, `arxiv.org`, `cmu.edu`, `oup.com`, `datamb.football` | Bare domains, not real article URLs. All numbers from that worker are unverified. |
| "Split-half reliability is unsuitable for RAPM" | `metricgate.com` (generic stats site) | **Contradicted** by Hvattum & Gelade (2021), whose reliability axis is exactly this construction. |
| "RAPM does not correlate with commercial ratings" | bare `pubmed.ncbi.nlm.nih.gov` | **Wrong.** Gelade & Hvattum (2020) measure R² = 0.22–0.38, i.e. ρ ≈ 0.47–0.62. Not zero. |

---

## 8. Primary sources worth keeping

- Kharrat, López Peña & McHale (2019/2020), *Plus-Minus Player Ratings For Soccer* — the stream's
  base paper. Preprint **arXiv:1706.04943**; published in *European Journal of Operational
  Research*, https://www.sciencedirect.com/science/article/abs/pii/S0377221719309373
- Hvattum, Arntzen & Pantuso (2020), *Offensive and Defensive Plus–Minus Player Ratings for
  Soccer*, Appl. Sci. 10(20) 7345 — **the most useful methodological source for us**.
  https://www.mdpi.com/2076-3417/10/20/7345
- Hvattum & Gelade (2021), *Comparing bottom-up and top-down ratings for individual soccer
  players*, IJCSS 20(1) — the validation framework.
  https://reference-global.com/article/10.2478/ijcss-2021-0002
- Gelade & Hvattum (2020), *On the relationship between +/− ratings and event-level performance
  statistics*, J. Sports Analytics 6(4) — the expected-agreement band.
  https://journals.sagepub.com/doi/10.3233/JSA-200432
- Bajons & Hornik (2024), *RAPM ... using Possession Sequences*, arXiv:2407.17832 — structured
  penalization by position/strength group. https://arxiv.org/abs/2407.17832
- Sæbø & Hvattum (2015), *Evaluating the efficiency of the association football transfer market
  using regression based player ratings* — first rigorous soccer RAPM.
- Macdonald (2012), *Adjusted Plus-Minus for NHL players using ridge regression with goals,
  shots, Fenwick, and Corsi*, arXiv:1209.1179 — the denser-target precedent.
- Sill (2010), *Improved NBA adjusted +/− using regularization and out-of-sample testing* — the
  origin of RAPM.
  https://www.sloansportsconference.com/research-papers/improved-nba-adjusted-using-regularization-and-out-of-sample-testing
