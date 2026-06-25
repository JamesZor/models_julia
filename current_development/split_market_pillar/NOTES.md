# Split-market pillar — level vs supremacy

Research stream for the time-decay **outfield** engines. Lives in its own folder because
`ab_test_dixon_coles/` has been stretched beyond its original purpose.

## Why

The current market pillar (every `*Market*` engine) anchors the two log goal-rates
**independently** with one shared, sampled `σ_market` and one scalar `market_weight`:

```julia
ll_market_h = logpdf.(Normal.(log_λ_h .+ log κ_h, σ_market), market_log_λ_h)
ll_market_a = logpdf.(Normal.(log_λ_a .+ log κ_a, σ_market), market_log_λ_a)
@addlogprob! sum((ll_market_h + ll_market_a) .* w .* mask) * market_weight
```

Anchoring `h` and `a` independently with one σ is — exactly — an **isotropic** Gaussian
penalty on the rotated pair (level, supremacy). So one global knob forces the same trust on
"who wins" and "how many goals". That's the problem: when the model's favourite is wrong it
corrupts correct-score / BTTS / derived 1X2, while on totals the documented edge is *fading
the market's over-dispersion* (see `totals-compression-is-denoising`). We want to trust the
market on supremacy and keep our own view on level.

## What

Rotate the pillar into the level/supremacy basis and make it **anisotropic** with two FIXED
per-axis standard deviations (these replace sampled `market_σ` + `market_weight`, and become
the grid axes):

```
model_sup   = (log_λ_h + log κ_h) − (log_λ_a + log κ_a)      m_sup   = market_log_λ_h − market_log_λ_a
model_level = (log_λ_h + log κ_h) + (log_λ_a + log κ_a)      m_level = market_log_λ_h + market_log_λ_a
ll = N(model_sup; m_sup, σ_supremacy)  +  level_active · N(model_level; m_level, σ_level)
```

- `σ_supremacy` tight → hard "who wins" anchor.
- `σ_level` loose; `σ_level = Inf` → level pillar OFF, model fully owns totals.
- Branch-free in the `@model`: `build_turing_model` converts `σ_level = Inf` into
  `level_active = 0.0` + a dummy finite σ, so the model body stays pure broadcast (AD-safe).
- No new feature extraction — `m_level`/`m_sup` are derived from the already-inverted
  `flat_market_λ_home/away/ρ`.
- Dixon-Coles ρ is anchored at `σ_supremacy` (ρ sharpens correct-score, same goal as supremacy).

## Files

- `l01_split_market.jl` — loader (temporary module). Four engine rungs + `build_turing_model`,
  `required_features`, `extract_parameters`, and prediction overrides.
- `r01_smoke_and_grid.jl` — smoke test then σ-grid backtest + evaluation.

## The ladder (rungs)

| Rung | Struct | Goals likelihood | xG | Notes |
|---|---|---|---|---|
| R1 | `SplitMarketPoissonGoalsModel` | Poisson | no | isolates the split from xG noise |
| R2 | `SplitMarketPoissonXGModel` | Poisson | yes | |
| R3 | `SplitMarketNegBinXGModel` | RobustNegativeBinomial | yes | over-dispersed |
| R4 | `SplitMarketDixonColesXGModel` | DC Poisson | yes | adds ρ anchored at σ_supremacy (= r06 model) |

## Integration gotchas (carried from memory)

- **Prediction dispatch** (`dixoncoles-prediction-dispatch-union`): all player models are
  `<: AbstractNegBinModel`; the Poisson/DC scorers are more-specific Union overrides. Our new
  structs aren't in those Unions, so R1/R2 (Poisson) and R4 (DC) ship explicit
  `extract_params`+`compute_score_matrix` overrides in the loader. R3 (NegBin) routes on the
  `AbstractNegBinModel` default via its `r_h`/`r_a` columns — no override.
- **Betfair training pillar** (`outfield-xg-engine-gotchas`): swap
  `summarize_betfair_market` into `ds.odds` **before** `create_experiment_task` so the
  supremacy anchor is built from Betfair, not SofaScore 1X2. The runner does this.
- **xG=0 → Gamma −Inf**: present xG floored to 1e-3, missing masked (R2–R4). R1 has no xG.
- Data: `Data.Ireland()` (full betfair + xG).

## Grid (per rung)

`days_half_life = 60` fixed. `σ_supremacy ∈ {0.05, 0.1, 0.2}` × `σ_level ∈ {0.5, 1.0, Inf}`.
Judge on **hurdle_G / CLV vs Betfair** per market family (not LogLoss alone). Expect:
supremacy markets (home/draw/away, BTTS) sharpen as σ_sup tightens; totals (over/under)
best where σ_level is loose/Inf if the totals edge is real.

## Findings log

### 2026-06-24 — R1 smoke test PASSED (Ireland, σ_sup=0.1, σ_level=Inf)

- Loader compiles; R1 builds, declares correct features (no XGFeature).
- Model is **correct and not pathological**. Isolated single-chain timings on one split
  (360 obs): ForwardDiff 31.9s (98.9% compile), **ReverseDiff(compile=true) 12.0s
  (98.6% compile)**. Sampling itself is fast; cost is first-call tape compile.
- Full 28-split CV run (50 warmup / 50 samples × 2 chains = 56 chains) completed in
  **15m45s** → ~**16.8 s/chain**. The `AutoReverseDiff(compile=true)` tape is rebuilt
  **per chain** and does NOT amortize, so wall-time ≈ n_chains × ~12s compile. This is
  inherent to the framework (existing engines share it), NOT specific to the split pillar.
- End-to-end PPD via `model_inference(ds, smoke_res)`: 286 matches → **8580 rows, 4 markets,
  30 selections, no `:r` ArgumentError** ⇒ the Poisson-route `extract_params` /
  `compute_score_matrix` overrides dispatch correctly.
- ⚠️ A couple of early splits showed tiny initial step sizes (ϵ ~6e-6); expected with little
  history. Re-check rhat/divergences once running the real grid (samples ≥ 800).

### Runtime implication for the grid

A single full-CV experiment ≈ ~16 min (dominated by per-chain compile, ~independent of
sample count). The 9-cell σ-grid for one rung ≈ **~2.5 h**.

### 2026-06-24 — STALL diagnosis + fix (important)

Tight **fixed** σ_supremacy makes the posterior **stiff**: NUTS picks tiny step sizes
(ϵ down to ~3e-6) and, at the default `max_depth=10`, blows the leapfrog tree up to ~1024
gradient evals on a single iteration. Under slow stiff-region gradients this hangs a chain
silently → the gate's 10-min watchdog kills the eval and the runaway chain wedges the gate
(cooperative cancel can't stop it → REPL restart). Observed:
- σ_sup=0.05: full stall (twice, even at 200/100×2 — the stall is a *single-iteration* tree
  blow-up driven by random UniformInit, NOT a function of total sample count).
- σ_sup=0.2 at 500/200×3: didn't stall but crawled (~35min+, deep trees).

**Fix that works: cap `max_depth=6`** (≤64 leapfrogs/iter) via the `max_depth` kwarg of
`create_experiment_task`. Bounds per-iteration cost absolutely → no silent hang. Validated:
σ_sup=0.1 cell completed in **11m6s**, all 4 cells ~10–11min each, zero stalls. (Fixed σ has
no release valve; the original engine *samples* `market_σ`, which is why it doesn't hit this.)

### 2026-06-24 — R1 trimmed σ-grid results (Ireland, goals-only Poisson, 200/100×2, max_depth=6)

Grid {σ_sup 0.1, 0.2} × {σ_level 1.0, Inf}. Backtest **hurdle_G** (higher=better; all cells
NEGATIVE = none profitable on this weak goals-only base):

| metric | s02_levInf | s02_lev10 | s01_levInf | s01_lev10 |
|---|---|---|---|---|
| TOTAL_G | **−0.0040** | −0.0175 | −0.2038 | −0.1657 |
| 1X2 | **−0.0025** | −0.0072 | −0.0083 | −0.0139 |
| BTTS | **−0.0001** | −0.0174 | −0.0044 | −0.0200 |
| Totals | −0.0014 | **+0.0071** | −0.1911 | −0.1318 |

LogLoss diff (model−market, lower better): s02_levInf **−0.0078** (only one beating market),
s02_lev10 −0.0002, s01_levInf +0.0122, s01_lev10 +0.0307.
GLMEdge spread_fair_coef (signal): s02 cells +0.59/+0.70 (p≈0.07–0.08); s01 cells ≈0/neg.

**Interpretation (R1 only):**
- **Tighter supremacy anchoring (σ_sup 0.1) is much WORSE than looser (0.2)** — opposite of
  the premise. The damage is concentrated in **Totals** (s01 totals collapse to −0.13/−0.19),
  i.e. hard-anchoring supremacy on a weak structural model also distorts the level/totals and
  kills the totals edge. σ_sup=0.05 (excluded; stalls) would be worse still.
- **σ_level=Inf (level free) generally best** (best BTTS & 1X2 & TOTAL_G); the one exception is
  Totals, where light level anchoring (lev10) is slightly +. Consistent with
  [[totals-compression-is-denoising]]: let the model own totals.
- Best cell **σ_sup=0.2, σ_level=Inf** ≈ breakeven — but this is the *least-anchored* corner,
  so on the goals-only base the surgical supremacy anchor does **not** add edge; less market
  anchoring is better.

**Caveat / next step.** R1 has no xG (deliberately weak, to isolate the split). The premise
(hard supremacy anchor helps) may still hold on a stronger base where the structural model is
trustworthy enough that anchoring only fixes the favourite without wrecking totals. Test R2
(Poisson+xG) and R4 (DixonColes+xG, the r06 model) next, same grid + max_depth=6.

### 2026-06-24 — xG Gamma NaN bug (BLOCKER, fixed)

R2/R3 (and the src no-market DoublePoisson engine) recomputed `xg_rate = exp(log_λ)` from the
RAW (unsanitized) `log_λ` and fed it to `Gamma(...)`. On extreme NUTS inits a NaN `log_λ`
makes the **Gamma constructor throw** before the `is_bad → -Inf` guard rejects the sample.
The queued trainer (`independent.jl:157`) only records a split once *all* chains succeed, so
broad chain failures left **`training_results.items` empty** → every cell looked "DONE" but
held no data → eval crashed on `vcat()` of an empty list. R1 (no xG) was immune.
**Fix:** sanitize `xg_rate` with the `ifelse.(isnan|isinf, 1.0, ...)` pattern (mirrors the λ
guard / gold-standard `outfield_xg_dixon_coles.jl`, which feeds a sanitized λ to Gamma). After
the fix all cells produced 28 items. (The DC R4 engine already used sanitized λ → fine.)

### 2026-06-24 — R2 results + no-market control (Poisson+xG, 200/100×2, max_depth=6)

Added a **market-off control** (σ_sup=1e6 → supremacy penalty ~flat, σ_level=Inf) using the
same fixed engine. Backtest **hurdle_G** (higher better):

| metric | **MARKET-OFF** | sup02_levInf | sup02_lev10 | sup01_levInf | sup01_lev10 |
|---|---|---|---|---|---|
| TOTAL_G | **+0.0236** | −0.1636 | −0.0341 | −0.2222 | −0.2169 |
| 1X2 | −0.0233 | −0.0029 | −0.0045 | −0.0109 | −0.0034 |
| BTTS | **+0.0246** | +0.0008 | +0.0016 | −0.0200 | −0.0114 |
| Totals | **+0.0223** | −0.1615 | −0.0312 | −0.1913 | −0.2022 |

LogLoss diff (model−market): MARKET-OFF **−0.0235** (best), sup02_lev10 −0.0134,
sup02_levInf +0.0085, sup01_levInf +0.0197, sup01_lev10 +0.2034.

**VERDICT (answers "is the market pillar worth it?"): NO, on this base.**
- The **no-market xG model is the only profitable one** (TOTAL_G +0.024) and beats *every*
  market-anchored cell, on both backtest and LogLoss. Adding any market pillar net-hurts.
- **But the premise has a kernel of truth on 1X2:** market cells beat market-off on
  home/away (e.g. home −0.013 vs market-off −0.019; away ≈−0.001 vs −0.012) — the market
  anchor genuinely improves "who wins". The problem is it **simultaneously wrecks Totals**
  (market-off Totals +0.022 vs −0.03…−0.20), and the totals loss dwarfs the 1X2 gain.
- **The split doesn't protect totals as intended.** Even σ_level=Inf (level fully free) still
  has totals collapse (sup02_levInf Totals −0.162), because supremacy and level are **not
  independent in the structural model** — they share the same team attack/defence params, so
  anchoring the supremacy axis bleeds into the level/totals. The clean (level,supremacy)
  rotation exists at the *rate* layer but not at the *parameter* layer.
- Tighter supremacy (sup01) still worse than looser (sup02) — R1's finding **holds** with xG.
- Least-bad market cell: **sup02_lev10** (loose supremacy + light level anchor) — the only one
  with positive Totals (over/under_25 +0.012/+0.010) and TOTAL_G −0.034, closest to market-off.

**Implication.** For totals/BTTS, run the model market-free (let it fade the market — matches
[[totals-compression-is-denoising]] and [[staking-research-conclusions]] market-curation). If
you want the market's 1X2 accuracy, take it at the PRICE/selection layer (bet only where model
& market agree on the favourite), NOT by anchoring the latent rates — anchoring corrupts the
totals edge. The surgical-supremacy-anchor idea, as a latent-rate prior, does not pay off here.

### 2026-06-24 — ⚠️ CONVERGENCE FAILS — the R1/R2 verdicts above are NOT reliable

`Diagnostics.check_convergence` on the R2 cells (200 samp / 100 warmup / 2 chains, max_depth=6):
across 904 params **median R-hat 1.08, max 2.45, only 15% < 1.01, 21% > 1.2; ESS NaN**. The
chains did NOT converge → every verdict above (no-market-wins, tighter-worse) may be a SAMPLING
ARTIFACT, not a real effect. The short settings + max_depth=6 (needed to dodge the stall) also
truncate NUTS trees → poor mixing on the stiff posterior.

**Root tension:** fixed tight σ ⇒ stiff posterior ⇒ stalls at max_depth=10 but won't converge at
max_depth=6. The ORIGINAL engines SAMPLE `market_σ` (truncated Normal) — that release valve is
why r07 (DC, 800/300×4, max_depth=10) converges and the fixed-σ split doesn't. **Fixing σ created
the stiffness.**

**Correct path (not yet run):** make σ_supremacy/σ_level *sampled* from tight priors (sweep the
prior MEANS, not fixed σ) so it's sampleable at full max_depth; rerun at ~800/300×4; verify
R-hat<1.01 + ESS; then compare on backtest+RQR+LogLoss: (a) split-market, (b) ORIGINAL un-split
goal+xG+market (the real "did splitting help" baseline), (c) market-off. Until then split-vs-unsplit
is OPEN.

### 2026-06-24 — Convergence root-cause investigation (single split 18) — UNRESOLVED, base-model issue

Sampled σ FIXED the stall (depth-10 fits complete). But convergence is still bad and it's the
BASE player-rating engine, not the split. Test split: Ireland Premier (seg **79**), **540
matches, 12 teams, 3 seasons** (train 2025 + 2 history, target biweek 17/2025), **only 176/540
have xG**. Single-split fits, 300 warmup / 500-800 samp × 4 chains, depth 10:

| variant (split 18) | max R-hat | median R-hat | median ESS |
|---|---|---|---|
| split, prior-mean centring, 800/300 (lucky) | 1.54 | 1.055 | 268 |
| split, empirical-demean ratings | 2.33 | 1.44 | 8 |
| split, GlobalInterception | 1.85 | 1.53 | 7 |
| split, ZeroSumTeamKappa | 3.60 | 1.98 | 5.6 |
| ORIGINAL un-split (HierKappa) | 2.85 | 1.57 | 8 |

**None converge** (want R-hat<1.01, ESS≥100s). Run-to-run ESS swings 5↔268 = metastable posterior.
Targeted single-ridge fixes (demean ratings, global intercept, zero-sum kappa) ALL failed —
worst components: kappa (κ multiplicative on λ ↔ intercept ridge), ν_xg, σ_sup, then rating
weights + ha. It's the JOINT weakly-identified structure, not one term. **⇒ r06/r07 and all
split-market backtests are non-converged / screening-quality.** Resolving this is a dedicated
base-model task: long-warmup sweep (≥2000), joint decorrelation/QR reparam, or fewer params for
the data. The split-market question stays parked until the base model samples reliably.

### 2026-06-25 — UNPARKED: sampled-σ double-Poisson converges; supremacy-only κ blow-up is an ARTIFACT

New self-contained loader `l02_split_market_poisson.jl` (`SplitMarketDoublePoissonModel`, R2:
goals+xG+split-market+outfield), σ_sup/σ_lev SAMPLED from tight priors, independent `market_on`/
`level_on` toggles. Runner `r02_*` (single) + `r03_split_controls.jl` (3 variants in parallel via
`@sync`/`@spawn` — 3×4 chains across 16 pinned cores). Ireland, single split, 1000samp/500warmup×4,
depth 10. **All three variants CONVERGE** (max R-hat 1.004/1.003/1.007) — the sampled-σ release valve
resolves the stall; depth-6 cap no longer needed.

| variant | max R-hat | κ_std | σ_κ | κ range | σ_sup | σ_lev |
|---|---|---|---|---|---|---|
| A supremacy-only (mkt on, lvl off) | 1.004 | **0.232** | **0.300** | 0.69–1.39 | 0.224 | 0.535 (=prior) |
| B market-OFF                        | 1.003 | **0.007** | **0.073** | 1.09–1.11 | 0.138 (=prior) | 0.537 (=prior) |
| C supremacy+level (both anchored)   | 1.007 | 0.027 | 0.070 | 1.02–1.10 | **0.386** | **0.160** |

**The big team-strength κ spread under supremacy-only anchoring is a CONSTRAINT ARTIFACT, not real
team finishing signal.** Ordering is B≈C ≪ A (predicted C<A<B — WRONG). Evidence + mechanism:
- **B (market-off) = the model's own pure goals+xG view shows κ essentially UNIFORM** (σ_κ 0.073,
  all teams ≈1.10). No per-team finishing variation is detectable in the data. If it were real, B
  would show it.
- **Only the half-anchored config A blows κ up** (σ_κ 0.30). In A the market pins the supremacy
  DIFFERENCE `(log_λ_h+log κ_h)−(log_λ_a+log κ_a)` while log_λ is tied to xG and the LEVEL is free,
  so κ is the only free param that can reconcile market-supremacy with xG-level → it absorbs the
  per-team market-vs-xG disagreement (spurious variance). Add the level anchor (C) → κ gets a 2nd
  constraint and collapses back (σ_κ 0.070); remove the market (B) → κ pooled (0.073).
- ⇒ my earlier read ("supremacy anchor frees κ to express finishing efficiency") is WRONG; κ is
  soaking up half-anchored tension, not skill.

**Bonus (C, both axes anchored): the model AGREES with the market on LEVEL (σ_lev 0.16, tight) and
DISAGREES on SUPREMACY (σ_sup 0.386, loose) — the REVERSE of the split premise.** The model's biggest
divergence from the market is on who-wins, not totals; so hard-anchoring supremacy forces the largest
distortion (which A then dumps into κ). This argues against the surgical-supremacy-anchor thesis at
the latent-rate layer. Still single-split / screening-quality, but the B-vs-A contrast is clean and
mechanistically explained. NEXT (if pursued): OOS backtest A/B/C + the original un-split baseline on
a converged grid before any verdict.
