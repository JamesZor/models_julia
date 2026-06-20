# `bigChanceCreated` — Deep EDA (League of Ireland)

**Status:** complete. Results from `r02_bigchance_runner.jl` run in the kaimon REPL on `mcmc-beast`
(Ireland, `Data.load_datastore_sql(Data.Ireland())`, n=1001 played matches).

**TL;DR:** bigChanceCreated is a **mildly over-dispersed count** (V/M ≈ 1.11–1.14), **not**
zero-inflated. **Negative Binomial** is the best marginal family (wins AIC *and* BIC on pooled data;
χ² GoF p=0.51) and is already AD-safe in `src/MyDistributions`. It is a **near-proportional view of
the attack rate**: `bigChance ≈ 1.12·xG` (corr 0.64 with xG, 0.47 with goals), so the next-session
pillar should be `bigChance_side ~ RobustNegativeBinomial(r_eff, c·λ_side)` with a learned scale
`c ≈ 1.1`, sharing the existing λ and home-advantage, NaN-masked for ~59% coverage. The **NB1-vs-NB2
variance function** (fixed `r` vs `r = μ/α`) is statistically indistinguishable marginally (§5b) and
should be A/B'd inside the joint model.

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

**Coverage (Ireland):** 1001 played matches; 696 carry an `ALL`-period stats row (69.5%); of those,
**587** have a non-missing `bigChanceCreated` per side ⇒ **n = 587 home, 587 away, 1174 pooled**
(≈ 58.6% of played matches). Non-trivial missingness ⇒ the pillar must be NaN-masked exactly like
the xG (Pillar A) and market (Pillar C) pillars.

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

| Split | n | Mean | Var | V/M | zeros emp | zeros Poisson | excess | max | skew |
|-------|---|------|-----|-----|-----------|---------------|--------|-----|------|
| Home | 587 | 1.767 | 1.954 | **1.106** | 0.181 | 0.171 | +0.010 | 7 | 0.82 |
| Away | 587 | 1.339 | 1.501 | **1.121** | 0.278 | 0.262 | +0.016 | 8 | 1.09 |
| All  | 1174 | 1.553 | 1.772 | **1.141** | 0.229 | 0.212 | +0.017 | 8 | 0.96 |

**Reading it:** mildly **over-dispersed** everywhere (V/M ≈ 1.11–1.14). Mean (1.55) sits above goals
(≈1.24 combined) and below shots — a sensible "denser attacking signal." The zero-excess is
**negligible** (+0.01 to +0.02): empirical zeros barely exceed the Poisson prediction, so there are
**no structural zeros** → zero-inflated models are unlikely to be needed. Clear home tilt
(1.77 vs 1.34), examined formally in §6.

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

**Validation guard (home goals, n=1001):** Poisson AIC 3030.45 ≈ NegBin 3030.72 ≈ COM 3031.56 —
goals are near-equidispersed, reproducing `r01`'s qualitative verdict (absolute AICs differ from
`r01`'s 2962 only because the dataset has grown to n=1001). The new ZIP/ZINB/COM fitters are
therefore trustworthy.

**bigChanceCreated — AIC/BIC (lower = better), sorted by AIC:**

| Model | k | **Pooled** LL | AIC | BIC | Home AIC | Away AIC |
|-------|---|------|-----|-----|----------|----------|
| **NegBin** | 2 | −1873.54 | **3751.08** | **3761.21** | **1958.04** | **1766.00** |
| COM-Poisson | 2 | −1874.02 | 3752.04 | 3762.17 | 1958.31 | 1766.55 |
| Weibull-count | 2 | −1874.24 | 3752.48 | 3762.62 | 1958.47 | 1766.72 |
| ZINB | 3 | −1873.54 | 3753.08 | 3768.28 | 1960.04 | 1768.00 |
| ZIP | 2 | −1876.82 | 3757.63 | 3767.77 | 1960.41 | 1768.20 |
| Poisson | 1 | −1878.64 | 3759.29 | 3764.36 | 1959.02 | 1767.60 |

Fitted NB (home): `r = 16.69, μ = 1.767`.

**Verdict — Negative Binomial.**
- On **pooled** data NB wins **both** AIC and BIC (ΔAIC ≈ 8 over Poisson) — the over-dispersion is
  real and worth a parameter.
- **Per-side**, NB still wins AIC but the stricter BIC narrowly prefers Poisson → the over-dispersion
  is **mild** (consistent with V/M ≈ 1.11 and the team-level analysis in §6).
- **ZINB ties NB's log-likelihood with π → 0** ⇒ no structural zeros; zero-inflation is firmly
  rejected (matches the negligible zero-excess in §3).
- COM-Poisson and Weibull-count are competitive but never beat NB, and NB is already implemented
  AD-safely (`RobustNegativeBinomial`) — the pragmatic and statistical choice.

**Goodness of fit (NB, pooled, n=1174):** χ² = 5.25, df = 6, **p = 0.51** — no evidence against NB.
Hanging rootogram residuals `√O − √E` are all small (|hang| < 0.8; mild under-fit only at count 3):

```
count  obs   exp     hang        count  obs  exp    hang
  0    269  274.36  -0.162         5    27   23.83  +0.314
  1    386  373.93  +0.310         6     6    7.84  -0.351
  2    279  277.68  +0.040         7     2    2.35  -0.118
  3    130  148.79  -0.796         8     1    0.65  +0.193
  4     74   64.34  +0.581
```

## 5b. NB1 vs NB2 — which variance function? (Cameron & Trivedi §3.3)

There are two standard negative-binomial parameterisations, differing in how the
variance scales with the mean:

| | Variance | Index of dispersion `V/M` | In our code |
|---|---|---|---|
| **NB2** (standard) | `μ + μ²/r = μ + α·μ²` | `1 + μ/r` — **grows with μ** | `RobustNegativeBinomial(r, μ)` (fixed `r`) |
| **NB1** (GLM/quasi-Poisson) | `μ + α·μ = (1+α)·μ = φ·μ` | `φ` — **constant in μ** | `RobustNegativeBinomial(μ/α, μ)` |

NB1 needs **no new distribution** — it is NB2 with the dispersion made
mean-proportional (`r = μ/α`). The distinction matters for the pillar because the
model's predicted `λ` spans ~0.6 (weak away) to ~3 (strong home): NB2 makes high-λ
matches much noisier, NB1 keeps `V/M` flat.

**Result — they are indistinguishable in this marginal EDA:**

| Split | NB2 (r, α) | NB1 (α, φ) | LL | ΔAIC(NB1−NB2) |
|-------|-----------|-----------|----|---------------|
| Home | r=16.69, α=0.060 | α=0.106, φ=1.106 | −977.02 | 0.000 |
| Away | r=11.71, α=0.085 | α=0.114, φ=1.114 | −881.00 | 0.000 |
| All  | r=11.15, α=0.090 | α=0.139, φ=1.139 | −1873.54 | 0.000 |

The fits are **identical**. This is structural, not luck: on a sample with a single
mean, NB1 and NB2 are *the same two-parameter family* (just a relabelling of the
dispersion). The two variance functions only diverge once **μ varies across
observations** — i.e. in the regression / model setting, never in a marginal fit.

The discriminating test is the cross-team `V/M ~ mean` slope (NB2 ⇒ slope ≈ 1/r ≈
0.06–0.09; NB1 ⇒ slope 0). It is **underpowered**: slope **+0.205, p = 0.37**
(CI [−0.28, 0.69]; 12 teams, narrow μ range 1.3–1.8) — cannot separate them.

**Implication:** the NB1/NB2 choice is a **modelling decision deferred to the joint
model**, where `λ_i` varies widely enough to matter, and should be settled there by
in-model WAIC / predictive comparison — not by the marginal EDA. Both are one-line
variants of the same `RobustNegativeBinomial`.

**AD-safety note (for the in-model NB1).** `RobustNegativeBinomial.logpdf`
(`src/MyDistributions/negative_binomial.jl:44`) is **clamp-free** — pure
`loggamma`/`log` arithmetic, fully ReverseDiff-differentiable. The constructor's
`max(r,1e-6)`/`max(μ,1e-6)` and the `clamp` in `cdf`/`rand` are **off the gradient
path** (cdf/rand unused in the likelihood; the EDA MLE was gradient-free). They did
**not** affect the NB1/NB2 fits — both converged to the *same* `(r≈11.15, μ≈1.55)`,
hence the identical LL. The genuine in-model subtlety is that NB1 makes `r_i = c·λ_i/α`
**vary per match**, so (i) `loggamma(k_i+r_i)`/`loggamma(r_i)` are evaluated &
`digamma`-differentiated for every match — new `loggamma` on this engine's tape
(goals pillar is Poisson), so re-benchmark vs the 0.64 ms/grad target in
`docs/turing_ad_performance_guide.md`; (ii) `λ_i` must stay positive *before* forming
`r` — the existing `clamp.(log_λ,-20,20)` → `exp + 1e-6` already guarantees this, so
don't reorder it; (iii) keep `r_bc`/`α_bc` priors `truncated(…, lower>0)` so the
`max(r,1e-6)` branch never locks the wrong side of its kink into a compiled tape.

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

**Correlations (per-team-per-match, n=1174; xG subset n=514):**

| Pair | Pearson | Spearman |
|------|---------|----------|
| bigChance ↔ goals | 0.474 | 0.469 |
| bigChance ↔ **xG** | **0.644** | **0.656** |

bigChance correlates more strongly with **xG** than with goals — expected, since both are
chance-*quality* proxies whereas goals add finishing noise.

**Predictive validity — Poisson GLM `goals ~ big_chance`:** slope **0.2665** (SE 0.0163, z = 16.4,
p ≈ 3e-60). Each additional big chance multiplies expected goals by `exp(0.2665) = 1.31` (**+31%**) —
big chances genuinely forecast goals.

**Scaling vs xG — OLS `big_chance ~ xg`:** intercept 0.125 (p = 0.17, ≈ 0), slope **1.116**
(SE 0.059, t = 19.0, p ≈ 2e-61). So `E[bigChance] ≈ 1.12 · xG` with essentially **zero intercept** —
bigChance is a **near 1:1, near-proportional view of the attacking rate**. This is the key result
licensing a shared-λ pillar with a single learned scale `c ≈ 1.1`.

**Mean-variance scaling (per team, n ≥ 20; 12 teams):** mean team V/M = 1.082. Fitting
`Var − Mean = α·Mean²` through the origin gives **α = 0.0595** ⇒ implied NB **r ≈ 16.8** — almost
exactly the direct fit (r = 16.7). However α is **not individually significant** (t = 1.48,
p = 0.17, CI [−0.029, 0.148]). Interpretation: much of the *pooled* over-dispersion is **cross-team
heterogeneity** (teams differ in attacking rate) rather than within-team excess — and the model's
team dynamics already absorb that. The residual within-team dispersion that the NB pillar must carry
is therefore mild (large r).

**Home advantage:** home mean 1.767 vs away 1.339; Mann-Whitney U **p = 7.5e-8** — a strong,
significant home edge on chance creation, mirroring goals. The pillar should **inherit the model's
existing `home_adv`** rather than introduce a separate term.

## 7. Recommendation for the next session

- **Marginal family:** **Negative Binomial** (pooled AIC 3751.1 / BIC 3761.2, both winners; χ² GoF
  p = 0.51). Dispersion regime mild over-dispersion, V/M ≈ 1.11–1.14, fitted `r ≈ 16.7`. No
  zero-inflation (ZINB → π=0); COM-Poisson/Weibull-count offer no improvement.
- **Pillar link:** model `E[bigChance_side] = c · λ_side` with a single learned scalar `c ≈ 1.1`
  (from `bigChance ≈ 1.12·xG`, ~zero intercept), analogous to the xG Gamma pillar's `mean = λ`.
  bigChance correlates 0.64 with xG and 0.47 with goals, and its GLM slope on goals is highly
  significant — a clean shared-λ signal, denser than goals (≈1.55 vs 1.24 events/match).
- **Likelihood (proposed):** `bigChance_side ~ RobustNegativeBinomial(r_eff, c·λ_side)` — already
  AD-safe in `src/MyDistributions`. Inherit the existing `home_adv` (strong HA, p=7.5e-8) — do
  **not** add a separate home term.
- **NB1 vs NB2 (variance function) — test in-model:** the marginal EDA cannot separate them (§5b);
  decide inside the joint model by WAIC/predictive. Both are one-liners on the same distribution:
  - **NB2** (fixed dispersion): `r_eff = r_bc`, prior `r_bc ~ truncated(Normal(12, 8), lower=1)`.
  - **NB1** (constant V/M): `r_eff = (c·λ_side)/α_bc`, prior `α_bc ~ truncated(Normal(0.12, 0.1), lower=0)`.
  Suggested scale prior `c ~ truncated(Normal(1.1, 0.3), lower=0)`. Start with NB2 (natural fixed-r
  Turing form); A/B against NB1, which is arguably more physical (no variance blow-up for strong teams).
- **Caveats:**
  - NB won, so **no new distribution is required** — reuse `RobustNegativeBinomial` (AD-safe, with
    `mean = μ`, `var = μ + μ²/r`). Apply it via vectorised `logpdf.` + `@addlogprob!`, no branches
    in the `@model`, per `docs/turing_ad_performance_guide.md`.
  - bigChance coverage ≈ 58.6% ⇒ the pillar needs a `NaN`-mask + `@addlogprob!` exactly like
    Pillar A (xG) and Pillar C (market) so absent matches contribute nothing.
  - Add a `BigChanceCreatedFeature`-style entry to `required_features` (the extractor already exists:
    `src/features/extractors/stats_extractors.jl:30`, keys `:flat_home_big_chances` /
    `:flat_away_big_chances`).
