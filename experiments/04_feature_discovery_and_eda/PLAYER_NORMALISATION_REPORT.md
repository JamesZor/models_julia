# Player Rating Normalisation — Findings

> **Question:** should the RAPM starting-XI covariate be normalised by age, position or wealth?
> **Laboratory:** England tiers 1 / 2 / 3 / 84 — 8,192 matches, 306,974 lineup rows, the only tiers carrying a reference rating, market values and dates of birth together.
> **Deployment target:** Scottish tiers 56 / 57.
> **Scripts:** `l50_player_norm.jl`, `r50_rating_structure.jl`, `r51_rapm_normalisation_bench.jl`
> **Snapshot:** 30 August 2026

---

## 0. Answer in one paragraph

**No normalisation is worth shipping.** Nine candidates were built and scored; a paired bootstrap
over held-out matches shows none of them distinguishable from the production baseline on both
stores. Two are actively harmful: residualising on age and wealth costs signal on *both* stores,
and **dropping the goalkeeper — the intervention the position analysis most strongly implied —
costs 0.024 correlation on the deployment store**, with a confidence interval well clear of zero.
The informative result is *why*: the SofaScore rating is position- and league-normalised, so it
deliberately removes exactly the team and context information a match covariate needs. Normalising
our rating to look more like it makes it worse at its job.

---

## 1. Your hypothesis was right

SofaScore ratings are position-normalised. Across 218,478 rated starter rows:

| position | n | mean | sd | p10 / p50 / p90 |
|---|---|---|---|---|
| M | 87,194 | 6.9176 | 0.548 | 6.3 / 6.8 / 7.6 |
| D | 73,239 | 6.9136 | 0.547 | 6.3 / 6.9 / 7.6 |
| F | 38,173 | 6.9063 | 0.617 | 6.3 / 6.8 / 7.7 |
| G | 19,872 | 6.9100 | 0.678 | 6.1 / 6.9 / 7.8 |

The position means span **0.011** against a pooled sd of 0.55. They are also **league**-normalised:
Premier League defenders average 6.909, League Two defenders 6.891. Only the *dispersion* is
allowed to vary — keepers swing widest (0.68), midfielders narrowest (0.55), which is a real
property of the positions rather than a scaling choice.

**Consequence.** The rating is a within-position, within-league *relative* score. It cannot be used
as an absolute cross-context quality measure, and any comparison against it must be made within
position. That is the premise the rest of this study rests on — and, as §5 shows, the reason it
turns out to be the wrong target.

---

## 2. Age and wealth explain almost nothing of a rating

Within position, on the English tiers:

| position | n | r(age) | r(log value) |
|---|---|---|---|
| G | 16,382 | +0.0068 | +0.0724 |
| D | 60,240 | +0.0336 | +0.0643 |
| M | 72,070 | +0.0480 | +0.1169 |
| F | 31,464 | +0.0370 | +0.0963 |

Pooled OLS `rating ~ age + age² + log(value)` gives **R² = 0.0143** over 174,394 rows. The age
profile is real but shallow — 6.717 at 16–18, peaking at 6.932 in the 26–30 band, back to 6.889 by
36–38 — a total swing of 0.22 against a sd of 0.55.

This bounds what a demographic normalisation of a *rating* could possibly achieve. It does not make
age and wealth useless; it makes them near-**orthogonal**, and therefore additive rather than
redundant information. That is an argument for combining them with a rating, not for dividing one
by the other.

---

## 3. RAPM's own structure

Fitted on 41,898 covered English stints (`:y_xg`, λ = 5,000), 3,993 players, of whom 2,903 carry at
least ten SofaScore ratings:

| position | n | mean RAPM | **sd RAPM** | sd SofaScore | r(RAPM, SofaScore) |
|---|---|---|---|---|---|
| G | 240 | +0.000026 | **0.005451** | 0.1383 | **+0.032** |
| D | 937 | +0.000015 | 0.005112 | 0.1499 | +0.287 |
| M | 1,218 | +0.000023 | 0.004714 | 0.1778 | +0.291 |
| F | 508 | +0.000158 | 0.004371 | 0.1918 | +0.312 |
| pooled | 2,903 | — | — | — | +0.266 |

Two things stand out.

**Goalkeepers have the widest RAPM spread and the least agreement with the reference.** In a
starting-XI *sum*, the keeper therefore contributes the most variance and the least player signal —
the worst possible combination, and the reason §4 tested dropping him.

**RAPM correlates with market value more strongly than the SofaScore rating does** (+0.107 to
+0.196, against +0.064 to +0.117). RAPM is picking up team quality: better-resourced sides have both
more expensive players and better xG differentials. Whether that is contamination or useful signal
is precisely what §5 settles.

For reference, the same measurement on Scottish Upper (54/55, where ratings also exist) gives a
different and stronger profile — D +0.509, M +0.432, F +0.291, G −0.005 — so the position ordering
is not stable across countries. Only the goalkeeper result replicates.

---

## 4. Nine normalisations, two criteria

All fitted on the first 80% of the fixture list and scored on the last 20%. The ridge, the position
means, the demographic regressions and the standardising scales are all history-only.

### Criterion A — agreement with the SofaScore rating (English tiers)

| strategy | pooled | D | M | F |
|---|---|---|---|---|
| raw | +0.2587 | +0.2833 | +0.2864 | +0.2895 |
| **exposure** *(baseline)* | +0.2562 | +0.2794 | +0.2853 | +0.2875 |
| drop_gk | +0.2709 | +0.2794 | +0.2853 | +0.2875 |
| zpos | +0.2624 | +0.2794 | +0.2853 | +0.2875 |
| zpos_nogk | +0.2736 | +0.2794 | +0.2853 | +0.2875 |
| **rank_pos** | **+0.2751** | +0.2998 | +0.2812 | +0.2740 |
| resid_demo | +0.2151 | +0.2459 | +0.2372 | +0.2400 |
| prior_demo | +0.2567 | +0.2782 | +0.2876 | +0.2888 |
| **prior_zpos** | **+0.2747** | +0.2782 | +0.2876 | +0.2888 |

### Criterion B — held-out association with the scoreline, both stores

Paired bootstrap of the change against the baseline, 4,000 resamples over held-out matches
(England n = 1,638; Scottish Lower n = 402):

| strategy | England Δr [90% CI] | Scot Lower Δr [90% CI] | verdict |
|---|---|---|---|
| raw | +0.0011 [+0.0003, +0.0019] | +0.0010 [−0.0024, +0.0044] | one store only |
| **drop_gk** | +0.0036 [−0.0010, +0.0083] | **−0.0241 [−0.0360, −0.0123]** | **harmful on target** |
| zpos | +0.0009 [−0.0003, +0.0020] | **+0.0057 [+0.0027, +0.0088]** | one store only |
| zpos_nogk | +0.0039 [−0.0010, +0.0088] | **−0.0218 [−0.0338, −0.0102]** | **harmful on target** |
| rank_pos | +0.0080 [−0.0061, +0.0222] | **−0.0207 [−0.0402, −0.0019]** | **harmful on target** |
| **resid_demo** | **−0.0105 [−0.0122, −0.0088]** | **−0.0083 [−0.0158, −0.0008]** | **harmful on both** |
| prior_demo | +0.0002 [−0.0006, +0.0009] | −0.0013 [−0.0041, +0.0015] | indistinguishable |
| prior_zpos | +0.0038 [−0.0011, +0.0086] | **−0.0229 [−0.0340, −0.0118]** | **harmful on target** |

---

## 5. The result that matters: the two criteria are inverted

Rank the strategies by criterion A and by criterion B on the deployment store, and the orderings are
close to reversed:

| strategy | agrees with SofaScore | predicts Scottish matches |
|---|---|---|
| rank_pos | **1st** (+0.2751) | **8th** (−0.0207) |
| prior_zpos | 2nd (+0.2747) | 9th (−0.0229) |
| zpos_nogk | 3rd (+0.2736) | 7th (−0.0218) |
| drop_gk | 4th (+0.2709) | 10th (−0.0241) |
| zpos | 6th (+0.2624) | **1st** (+0.0057) |
| resid_demo | 9th (+0.2151) | 6th (−0.0083) |

**Every strategy that improves agreement with the SofaScore rating makes the covariate worse at
predicting matches on the store it is deployed on.** That is not a paradox once §1 is taken
seriously: the SofaScore rating is *constructed* to be position- and league-neutral. It answers
"how well did this player perform for a player of his type, in his league?" — and to do that it
strips out exactly the team-strength and context information a match covariate needs.

The goalkeeper case is the clearest instance. GK RAPM is worthless as a *player* rating (r = +0.032
against the reference). But removing it costs 0.024 correlation on the deployment store, with an
interval clear of zero. The most plausible reading — stated as a hypothesis, not a measurement — is
that a keeper's RAPM is absorbing *team* quality rather than goalkeeping: he is on the pitch for
every stint, so his coefficient soaks up whatever the side's xG differential was. Useless as
scouting, useful as a team proxy.

---

## 6. Recommendations

1. **Ship nothing from this study.** The production `exposure` shrink is not improved on by any of
   the nine candidates at a level distinguishable from noise.
2. **Do not drop the goalkeeper**, despite the position analysis appearing to demand it. It is the
   single most harmful intervention tested on the deployment store.
3. **Do not residualise on age or wealth.** It is the only strategy that is significantly harmful on
   *both* stores, which follows from §2: there is almost nothing there to remove, so the operation
   mostly removes real variance.
4. **`zpos` is the only candidate worth revisiting** — a small but real +0.0057 on the deployment
   store, nothing on England. If it is tried, it should be tried on r40's out-of-sample log loss,
   not on linear association.
5. **Stop treating the SofaScore rating as the target.** §5 shows optimising toward it is actively
   counterproductive for this covariate. It remains useful as a *diagnostic* — it is how we know GK
   RAPM carries no player signal — but not as an objective.
6. **The orthogonality finding in §2 is the live thread.** Age and wealth explain 1.4% of a rating,
   which means an age-adjusted wealth feature and a RAPM feature are close to independent sources.
   `r93_feature_synergy_and_correlations.jl` already measured them at r = +0.336 with both adding
   significantly beyond the other. Combining them is more promising than normalising either.

---

## 7. Limitations

- **The deployment store's held-out block is 402 matches.** Differences below roughly 0.02
  correlation are not resolvable there, which is why the bootstrap rather than the point estimate
  drives every conclusion above.
- **The English laboratory is a different country.** Position agreement does not replicate between
  England (F > M > D) and Scottish Upper (D > M > F); only the goalkeeper null replicates. Any
  finding transported from England to Scotland carries that risk.
- **The team-loading explanation in §5 is a hypothesis.** It is consistent with the evidence and
  with RAPM's stronger correlation to market value than the reference rating has, but it was not
  measured directly.
- **Linear association is a weak proxy for what the count model does.** The engine already carries
  team strength in `dyn.α`/`dyn.β`, so a covariate's real job is to explain what those cannot.
  Only r40 decides.
- Scottish Upper (54/55) could not be used as the laboratory: it has **zero** market-value and
  date-of-birth coverage.

---

## 8. Reproducing

```bash
source .env          # BF_DB_URL — required, the English store is fetched from the database
julia --project -t 8
```
```julia
include("current_development/scottish_lower/r50_rating_structure.jl")
include("current_development/scottish_lower/r51_rapm_normalisation_bench.jl")
```

`l50_store()` caches the English pull to `l50_english_store.jls` (~30s on a cold fetch, instant
afterwards). `r51` additionally reads the cached `ScottishLower` store.
