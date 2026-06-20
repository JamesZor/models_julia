# `bigChanceCreated` — Deep EDA (League of Ireland)

**Status:** maths & methodology written; **results sections marked `⟨RESULTS PENDING⟩`** are
filled after running `r02_bigchance_runner.jl` in the kaimon REPL on `mcmc-beast`.

**Files:** `l01_bigchance_logic.jl` (functions), `r02_bigchance_runner.jl` (execution + captured
output), this report.

---

## 1. Motivation & framing

The outfield engine `DynamicDixonColesXGOutfieldPlayerTimeDecayModel`
(`src/models/pregame/engines/player_level/time_decay/outfield_xg_dixon_coles.jl`) is a **joint**
model: three observation "pillars" all hang off the *same* latent per-team attack rate
`λ_h, λ_a` (in log-space `log λ = intercept + home_adv + attack − defence`):

| Pillar | Observable | Likelihood | Tie to λ |
|--------|-----------|------------|----------|
| A | xG | `Gamma(ν, λ/ν)` | mean = λ |
| B | Goals | Dixon-Coles `Poisson(λ)` + low-score τ | rate = λ |
| C | Market | `Normal(log λ + log κ, σ)` | log-rate |

We want to add **`bigChanceCreated`** as a *fourth* pillar. A "big chance" is SofaScore's label for
a clear goalscoring opportunity, so it is a **discrete count** of high-quality attacking events per
team per match — conceptually a denser, higher-frequency view of the same attacking intensity that
drives goals and xG. If `E[bigChance] ≈ c·λ` for some scale `c`, it is a clean extra signal on λ
with more events per match than goals (≈ tighter information per game).

Before writing any likelihood we must answer, **as a random variable**, four questions:

1. **Expectation & variance** — what are the moments, and is it over- or under-dispersed
   (Index of Dispersion `V/M`)?
2. **Zero-mass** — are there structural zeros (matches with no big chance) beyond what Poisson
   predicts? That decides whether a zero-inflated family is needed.
3. **Best marginal family** — Poisson / NB / Weibull-count / ZIP / ZINB / COM-Poisson?
4. **Link to λ** — does `bigChance` scale monotonically with goals & xG (so it can share λ with a
   learned multiplier), and what dispersion does its likelihood need?

This session is **EDA + report only** — no Turing changes. The chosen family + link feed the next
session's pillar implementation.

## 2. Data & coverage

- **Source:** `ds.statistics`, rows with `period == "ALL"` (full-match aggregate).
- **Columns:** `bigChanceCreated_home`, `bigChanceCreated_away` (`Union{Missing, Float64}`).
- **Segment:** `Data.Ireland()` → `tournament_id = 79` (has stats + betfair coverage).
- **Cleaning:** drop `missing`, `round(Int, …)` (values are integer counts stored as floats).
- **Coverage caveat:** not every played match has an `ALL`-period stats row; we report the fraction
  to flag selection effects (the future pillar must be masked like xG/market are, via a `NaN`-mask).

`⟨RESULTS PENDING — coverage %, n_home / n_away / n_all⟩`

## 3. Marginal moments

For counts the diagnostic is the **Index of Dispersion** `D = Var/Mean`:

- `D ≈ 1` → equidispersed (Poisson plausible),
- `D > 1` → **over-dispersed** (NB / ZINB / COM ν<1),
- `D < 1` → **under-dispersed** (COM ν>1 / Weibull-count c>1).

For reference, Ireland **goals** were essentially equidispersed in `r01`
(home `V/M ≈ 1.07`, away `≈ 0.96`, all `≈ 1.04`). bigChance has a higher mean (more events/match),
so its dispersion regime is an open question we measure directly. We also compare the **empirical
zero fraction** to the Poisson-implied `e^{-μ̂}`: a large positive excess is the signature of
zero-inflation.

`⟨RESULTS PENDING — mean / var / V/M / zero-mass / skew for home, away, all⟩`

## 4. Candidate distributions — the maths

All are fit by **maximum likelihood** (`Optim`, log/logit-transformed parameters to enforce
positivity / [0,1]). Selection uses

```
AIC = 2k − 2ℓ̂        BIC = k·ln(n) − 2ℓ̂
```

where `ℓ̂` is the maximised log-likelihood and `k` the parameter count. BIC's heavier penalty
(`ln n` vs `2`) punishes the 3-parameter ZINB harder — a split AIC/BIC verdict warns that extra
complexity is only marginally justified.

**Poisson** (`k=1`). `P(Y=k)=e^{−λ}λ^k/k!`, `E=Var=λ`. The equidispersed baseline.

**Negative Binomial** — `RobustNegativeBinomial(r, μ)` (`k=2`).
`E=μ`, `Var=μ + μ²/r`. The shape `r` injects over-dispersion (`r→∞` ⇒ Poisson). Cannot model
under-dispersion. Project type is AD-safe (avoids the `p=r/(r+μ)` instability).

**Weibull-count** — `WeibullCount(c, λ)` (`k=2`). A renewal-count distribution whose inter-event
hazard follows a Weibull with shape `c`:
- `c = 1` → Poisson,
- `c < 1` → decreasing hazard ⇒ **over-dispersion**,
- `c > 1` → increasing hazard ⇒ **under-dispersion**.
The single family spans both regimes — useful if bigChance turns out underdispersed.

**Zero-Inflated Poisson (ZIP)** (`k=2`). Mixture of a structural-zero state (prob `π`) and a
Poisson(λ) count state:
```
P(Y=0) = π + (1−π)e^{−λ}
P(Y=k) = (1−π)·Poisson(k; λ),  k ≥ 1
```
Captures *excess* zeros without inflating the whole tail.

**Zero-Inflated Negative Binomial (ZINB)** (`k=3`). As ZIP but with an NB count state — handles
excess zeros **and** over-dispersion simultaneously:
```
P(Y=0) = π + (1−π)·NB(0; r, μ)
P(Y=k) = (1−π)·NB(k; r, μ),  k ≥ 1
```

**Conway-Maxwell-Poisson (COM-Poisson)** (`k=2`). Generalises Poisson with a dispersion exponent
`ν`:
```
P(Y=j) = (λ^j / (j!)^ν) / Z(λ,ν),   Z(λ,ν) = Σ_{i≥0} λ^i/(i!)^ν
```
- `ν = 1` → Poisson, `ν < 1` → over-dispersed, `ν > 1` → under-dispersed.
The normaliser `Z` has no closed form; we truncate the series at `J = max(data)+50` and compute
`log Z` via log-sum-exp for stability.

### Goodness of fit
Beyond AIC/BIC we check the **shape** of the fit on the winning family:
- **Hanging rootogram** (√-scale): `hang_j = √O_j − √(n·p̂_j)`; bars near zero across all counts ⇒
  good fit, systematic sign patterns ⇒ misfit (e.g. under-predicting zeros).
- **Pearson χ²**: `χ² = Σ (O−E)²/E` with a pooled tail bin, `df = #bins − 1 − k`. Large `p` ⇒ no
  evidence against the distribution.

## 5. Model comparison results

`⟨RESULTS PENDING — LL/k/AIC/BIC table for home, away, all; AIC- and BIC-winners⟩`

**Validation guard:** the runner first re-fits Poisson/NB on the *goals* vector; these must
reproduce `r01`'s AICs (home goals ≈ 2962) before the ZIP/ZINB/COM fitters are trusted on
bigChance.

`⟨RESULTS PENDING — rootogram + χ² for the winning family⟩`

**Provisional reasoning (to confirm with numbers):** if `V/M > 1` with no large zero-excess →
expect **NB**; if there is a real zero-excess → **ZINB**; if `V/M < 1` → **COM-Poisson (ν>1)** or
**Weibull-count (c>1)**.

## 6. Link to the shared attack rate

To justify hanging bigChance off λ we test that it is a monotone, scaled view of attacking output:

- **Correlations** (Pearson + Spearman) of `bigChance` with `goals` and with `xg` on the
  per-team-per-match long table. Positive and material ⇒ shared latent.
- **Poisson GLM** `goals ~ big_chance`: a positive, significant slope confirms big chances forecast
  goals (predictive validity of the signal).
- **OLS** `big_chance ~ xg`: slope estimates the linear scaling between the two attacking proxies.
- **Mean-variance scaling law** across teams: fit `Var − Mean = α·Mean²` (through origin); `α>0`
  confirms team-level over-dispersion and gives an implied NB `r ≈ 1/α`, telling us what dispersion
  the pillar likelihood must carry.
- **Home/away asymmetry**: Mann-Whitney U on home vs away bigChance — if a home edge exists (as it
  does for goals), the pillar should inherit the model's `home_adv` term rather than a separate one.

`⟨RESULTS PENDING — correlations, GLM/OLS coefficients, α & implied r, home/away test⟩`

## 7. Recommendation for the next session

`⟨RESULTS PENDING — fill after the run. Template:⟩`

- **Marginal family:** `⟨winner⟩` (AIC `⟨..⟩` / BIC `⟨..⟩`), dispersion regime `V/M = ⟨..⟩`.
- **Pillar link:** model `E[bigChance_side] = c · λ_side` with a learned scalar multiplier `c`
  (analogous to the xG Gamma pillar's `mean = λ`), since bigChance correlates `⟨..⟩` with goals/xG.
- **Likelihood:** `⟨e.g. RobustNegativeBinomial(r_bc, c·λ) — already AD-safe in src/MyDistributions⟩`.
- **Caveats:**
  - If **ZINB/ZIP/COM** wins, we need a *new* AD-safe `MyDistributions` type with a vectorised
    `logpdf` honouring `docs/turing_ad_performance_guide.md` (no branches in `@model`; the zero/
    non-zero split must be a precomputed Float64 mask, like the existing xG/market masks).
  - bigChance coverage `< 100%` ⇒ the pillar needs a `NaN`-mask + `@addlogprob!` exactly like
    Pillar A (xG) and Pillar C (market) so absent matches contribute nothing.
  - Add a `BigChanceCreatedFeature`-style entry to `required_features` (the extractor already exists:
    `src/features/extractors/stats_extractors.jl:30`, keys `:flat_home_big_chances` /
    `:flat_away_big_chances`).
