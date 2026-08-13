# smile_negbin — NegBin sibling of the `sup40_sw40` smile engine

## What this stream is

`DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel` — the engine behind `src_sup40_sw40`,
the cell judged throughout `orderbook_layer2` — scores goals with plain Poisson, forcing
Var = E. Its `dispersion_config` field is declared but inert
(`# config-compat; unused by the Poisson likelihood`). WP2's diagnostics flagged the consequence
structurally: the engine's 1X2 dispersion measured at roughly half the market's.

This stream builds the over-dispersed sibling: `RobustNegativeBinomial(r, λ)` in place of
`Poisson(λ)` in the goals pillar, everything else byte-for-byte identical.

| File | What it is |
|---|---|
| `l01_smile_negbin_engine.jl` | `DynamicSmileDoubleNegBinXGOutfieldPlayerTimeDecayModel` — Turing model + extractor |
| `l02_smile_negbin_predict.jl` | prediction-side dispatch, returns a `SmileScoreMatrix` |
| `r01_smoke.jl` | AD-safety + round-trip test, synthetic data, no DB |
| `r03_pipeline_smoke.jl` | the real pipeline over two Ireland 79 folds |
| `r04_prior_ladder.jl` | dispersion prior ladder — **the run that settles the stream** |
| `r02_train_ireland.jl` | the full two-league run — **written, deliberately NOT run** |

> **Bottom line (r04):** the engine is correct and converges, but there is no over-dispersion in
> the data for it to capture. Ireland 79's goal counts are if anything mildly *under*-dispersed
> (raw V/M 0.966), which a NegBin cannot represent at any `r`. The full two-league run was not
> started. See [§ r04](#r04--dispersion-prior-ladder-the-decisive-run).

### Why changing only the goals pillar is safe

The xG (Gamma), supremacy, and local-intensity smile pillars only ever constrain a **mean**, and
`E[NegBin(r,μ)] = μ = E[Poisson(μ)]`. None of them says anything about variance, so none is
disturbed by decoupling variance from the mean.

O/U deliberately still prices as `cdf(Poisson(λ_tot·φ(K)), K)`. That is not the goals likelihood
leaking back in — `Λ(K) = λ_tot·φ(K)` is the market-inverted per-strike intensity, and the
market's own total-goals dispersion is already encoded in the shape `φ(K)`. Pricing it as NegBin
too would double-count. So the NegBin changes 1X2 / BTTS / correct-score only.

## r01 — AD-safety smoke (synthetic, passed)

All 25 checks green. Compiles and samples under `AutoReverseDiff(compile=true)`; `r_h`/`r_a` live
in the chain; returns a `SmileScoreMatrix` so the container-dispatched O/U rule fires unmodified.

The check that matters: at `r = 1e6` the NegBin grid matches src's `_smile_poisson_grid` to
**1.5e-7**, and at the fitted `r` it differs by **2.2e-2**. The kernel is a genuine
generalization, and the dispersion is load-bearing rather than decorative.

## r03 — pipeline smoke, Ireland 79, 2026 biweeks 11–12

Real pipeline end to end on the pinned `ds_ire79`. Folds restricted via the splitter's own
`warmup_period`/`end_dynamics`, which leaves each fold's *training set* at full size — so these
are the two largest, slowest folds and the geometry is production-faithful. Production-exact on
DataStore / history / chains (4) / `max_depth` (10) / `UniformInit` / queued execution; reduced to
300 samples + 250 warmup (vs 800 + 300), which makes the R-hat test harder, not easier.

**3/3 folds trained in 28.1 min, no dropped splits, 17 OOS matches, 1139 PPD rows across 7
markets, every probability valid, 1X2 and O/U summing to 1.**

### Convergence — marginal, and it belongs to the parent

| family | max R-hat | min ESS | n |
|---|---|---|---|
| `log_φ` | **1.0219** | 195 | 15 |
| `disp` | 1.0148 | 1257 | 6 |
| `inter` | 1.0134 | 328 | 47 |
| `kap` | 1.0129 | 366 | 39 |
| `ha` | 1.0114 | 366 | 39 |
| `ν_xg` | 1.0103 | 1223 | 3 |
| `p_dyn` | 1.0086 | 717 | 12 |
| `σ_sup` | 1.0053 | 1248 | 3 |
| `σ_smile` | 1.0023 | 513 | 3 |

18 / 167 parameters at R-hat ≥ 1.01, worst 1.0219. Divergences ≈ 0 (0.17% on one fold, 0.0 on the
other two).

The worst family is `log_φ` — the smile shape, byte-identical to the Poisson parent. Of the two
parameters this edit *adds*, `disp.log_r` converged cleanly and `disp.δ_r_home` came in at 1.0148
with the highest ESS in the run (1257). So the NegBin edit is not what is straining the sampler.

Convergence was checked with a raw `MCMCChains.summarize` sweep, **not** `Diagnostics.extract_chains`
alone — that walker has no branch for `log_φ`, `σ_smile`, `σ_sup` or `ν_xg` and would silently
never check them.

### The finding that actually matters: `r` is unidentified by the data

| fold | posterior sd `log_r` | prior sd | ratio | posterior sd `δ_r_home` | prior sd | ratio |
|---|---|---|---|---|---|---|
| 0 | 0.383 | 0.4 | **0.96** | 0.473 | 0.5 | **0.95** |
| 11 | 0.392 | 0.4 | **0.98** | 0.458 | 0.5 | **0.92** |
| 12 | 0.387 | 0.4 | **0.97** | 0.483 | 0.5 | **0.97** |

The posterior *is* the prior. Posterior median `r_h ≈ 27.5`, `r_a ≈ 24.5`, against a prior median
of `e^3.1 = 22.2` — the likelihood moves it almost not at all.

**Mechanism.** The goals pillar is time-decayed at a 60-day half-life. On the largest fold, 360
history matches carry an effective N of **53.7** — about 107 goal counts at λ ≈ 1.3. That cannot
pin down a dispersion parameter; the standard error on `r` at that sample size swamps the signal.

**Consequence.** As configured, the engine prices ≈5% over-dispersion (`Var/E = 1 + λ/r` at
`r ≈ 25`, `λ ≈ 1.35`) because the *prior* says so, not because the data does. Measured price
impact against the Poisson parent on the same λ: 1X2 moves **0.28pp** mean, **0.69pp** max.

This is a structural finding, not a bug — but it means a full two-league run would be measuring a
prior, and that decision should be made deliberately rather than discovered afterwards.

## r04 — dispersion prior ladder (the decisive run)

Ireland 79, 2026 biweek 12 (the largest fold), one shared feature set, three dispersion priors,
warmup 800 / samples 500, chains 4 / 6 / 6 = 16 tasks in one wave. 77 min.

The top rung is **bounded**-flat on purpose. `r → ∞` *is* the Poisson limit, so on
non-over-dispersed counts the likelihood is monotone in `log r` with no interior maximum and an
improper prior gives an improper posterior — the chain drifts, R-hat explodes, and it reads as a
sampler failure when it is the answer. `Uniform(-2, 8)` is flat across `r ∈ [0.14, 2981]` —
violent over-dispersion through numerically-Poisson (`Var/E = 1.0005`) — so a drift to the Poisson
end becomes readable posterior mass at the upper edge instead.

| rung | prior sd | post sd | **ratio** | max R-hat | R-hat `log_r` | ESS | div % |
|---|---|---|---|---|---|---|---|
| A tight `N(3.1, 0.4)` | 0.400 | 0.367 | **0.92** | 1.0188 | 1.0000 | 3351 | 0.00 |
| B wide `N(3.1, 2.0)` | 2.000 | 1.489 | **0.74** | 1.0057 | 0.9992 | 3332 | 0.00 |
| C flat `U(-2, 8)` | 2.887 | 1.712 | **0.59** | 1.0097 | 1.0019 | 1947 | 0.57 |

| rung | median `r_h` | median `r_a` | `r_a` 90% CI | P(Var/E>1.05) | P(r>403) | prior P(r>403) | clamp hits |
|---|---|---|---|---|---|---|---|
| A tight | 27.1 | 24.0 | [13.0, 44.6] | 0.62 | 0.000 | 2e-13 | 0.000 |
| B wide | 152.8 | 70.7 | [8.3, 1036] | 0.25 | 0.133 | 0.074 | 0.001 |
| C flat | 278.7 | 262.6 | [10.7, 2451] | 0.14 | **0.421** | 0.200 | 0.000 |

### What it says

**The ratio falls (0.92 → 0.74 → 0.59), so the data *does* inform `r`.** The r03 reading of
"unidentified" was wrong — it was an artefact of a prior tight enough that the posterior had
nowhere to go.

**And what the data says is Poisson.** Under the flat prior, 42.1% of the posterior sits at
`r > 403` (`Var/E < 1.004`) against 20.0% under the prior — a **2.1× lift toward the Poisson
limit**. Median `Var/E = 1.0051`, 90% CI [1.0006, 1.1265]; `P(Var/E > 1.05) = 0.14`,
`P(Var/E > 1.14) = 0.04`.

Note how the medians move: `r_a` 24 → 71 → 263 as the prior loosens. **The tight prior was
manufacturing the over-dispersion**, not revealing it.

### Corroboration that needs no model at all

Raw variance-to-mean of observed goal counts, 2024–2026:

| league | home V/M | away V/M | pooled V/M | mean goals | n |
|---|---|---|---|---|---|
| 79 Ireland Premier | 0.972 | 0.921 | **0.966** | 1.225 | 494 |
| 718 Ireland First Division | 1.006 | 1.006 | **1.031** | 1.351 | 495 |

This is the *marginal* ratio, which is **inflated** by λ heterogeneity across matches — the
conditional ratio the model sees can only be lower. Both leagues are at or below Poisson before
that correction, and 79 is mildly **under**-dispersed, which a NegBin cannot represent at any `r`
(its variance is `μ + μ²/r ≥ μ`). Only something like a Conway–Maxwell–Poisson could.

The `V/M ≈ 1.14` cited in `outfield_xg_double_negbin.jl`'s header for 718 does not reproduce on
the pinned store over these seasons.

### Consequence: the full run was not started

`r02_train_ireland.jl` is written and correct, but running it would spend ~11h estimating a
parameter the data has already been shown to push toward its degenerate limit, and would ship
prices carrying ~5% over-dispersion that only the prior believes in. Measured price impact of the
tight-prior version vs the Poisson parent was 0.28pp mean / 0.69pp max on 1X2 — small, and in the
wrong direction.

### The redirection this implies

WP2's finding that the engine's **1X2 dispersion is about half the market's** is a *different
quantity* from what this stream tested. That is the spread of predicted outcomes **across**
matches; `r` governs count variance **within** a match. r04 shows the within-match component is
already Poisson-consistent, so it was never the source of the deficit.

The remaining candidates for the 1X2 under-dispersion are all on the λ side:

- the market pillars (supremacy + smile, both at weight 0.4) shrinking `λ_h`/`λ_a` toward the
  market and compressing their spread across matches;
- team-strength posteriors (`ha.`, `kap.`, `p_dyn.`) being too tight, so `λ` varies too little
  fixture to fixture;
- the 60-day half-life over-smoothing form.

That is where the next experiment belongs. Adding dispersion to the goals likelihood was the
wrong lever for the symptom.

### Sampler note

The extra warmup and chains materially improved convergence over r03: max R-hat 1.006–1.019 here
(warmup 800, 4–6 chains) against 1.022 there (warmup 250, 4 chains), with `disp.log_r` at R-hat
≈ 1.000 and ESS 1947–3351 on every rung. The clamp-plateau contamination guarded against in r04's
header did not materialise (0.1% of draws at the clamp on rung B, 0 elsewhere).
