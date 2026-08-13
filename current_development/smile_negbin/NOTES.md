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
| `r02_train_ireland.jl` | the full two-league run — **written, not yet run** |

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

### Open question, and the cheap test that settles it

Widen the prior (`log_r ~ Normal(3.1, 1.5)` instead of `Normal(3.1, 0.4)`) and re-run one fold:

- if posterior sd / prior sd stays ≈ 1.0, the data genuinely cannot identify `r` on this league,
  and any NegBin result is a prior in disguise;
- if it drops materially, the tight prior was the binding constraint and the engine can learn.

Roughly 10 minutes for one fold. Candidate follow-ups if the data does carry signal but not
enough: lengthen the half-life for the goals term, or pool `r` across 79 and 718 so one parameter
sees both leagues.
