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
