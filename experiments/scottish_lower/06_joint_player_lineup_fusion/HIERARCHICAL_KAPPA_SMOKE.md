# Hierarchical Team Kappa — Extended Smoke Test

**Status:** PASS — all smoke gates G1–G8 passed for both candidates on `mcmc-beast`; the 40-fold grid runner passed its prepare-only preflight and is ready to launch.

**Executed:** 2026-09-03 on `mcmc-beast`, 16 Julia threads, BLAS restricted to one thread.  
**Data snapshot:** cached `ScottishLower` `DataStore`, 2,009 matches, 74,225 lineup rows and 45,759 odds rows.  
**Smoke folds:** production boundaries 2 and 20, 1,800 fitted-match rows in total and 39 held-out fixtures.  
**Sampler:** `QueuedNUTSConfig`, 4 chains, 800 warmup and 800 retained draws per chain, target acceptance 0.90.  
**Execution:** `QueuedExecution()`.

## Outcome

The final extended smoke completed with **350 / 350 assertions passing**.

| Candidate | Runtime | Parameters | Tape | Gradient | Max R̂ | Min bulk ESS | Min tail ESS | Divergences | OOS latents | Run UUID |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `m05_hierarchical_kappa` | 978.1 s | 83 | 232 instructions | 0.0706 ms | 1.0070 | 744 | 716 | 0 / 6,400 | 39 × 3,200 draws | `539d5873-2400-4b35-9e2a-6555a7b64cb6` |
| `m12_hierarchical_kappa` | 1,263.8 s | 85 | 257 instructions | 0.0781 ms | 1.0122 | 1,169 | 946 | 0 / 6,400 | 39 × 3,200 draws | `a3b5bfef-d72e-4167-a47f-c32d8e77420f` |

Both compiled gradients are comfortably below the 0.15 ms budget. Tape instruction counts were invariant to fold size, confirming that the `home_idx` / `away_idx` gathers remained vectorised rather than unrolling over matches.

## Gate evidence

| Gate | Requirement | Result |
|---|---|---|
| **G1** | Compiled ReverseDiff gradient below 0.15 ms; tape size independent of fold rows | **PASS.** 0.0706 ms / 232 instructions for `m05`; 0.0781 ms / 257 instructions for `m12`. |
| **G2** | Structural parameter contract and hierarchical chain sites | **PASS.** 83 parameters for `m05`, 85 for `m12`; `obs.σ_κ` and all 25 `obs.κ_team_raw[t]` columns were present. |
| **G3** | Four NUTS chains complete on both selected folds | **PASS.** Eight chains per candidate completed: 2 folds × 4 chains, 800 retained draws each. |
| **G4** | R̂ < 1.05; bulk ESS > 400; tail ESS > 300; zero divergences; BFMI ≥ 0.30; depth-cap rate < 5% | **PASS.** See the run and fold tables below. No trajectory reached the configured maximum tree depth. |
| **G5** | Identified hierarchical kappa extraction and 90% HPDIs | **PASS.** The per-draw sum of `δ_κ` was below `1e-10`; the posterior summaries and team ranking were produced. |
| **G6** | Finite, positive held-out `CountLatents` | **PASS.** Each candidate produced 39 fixtures × 3,200 posterior draws; all extracted rates were finite and positive. |
| **G7** | `SmileScoreGrid` construction and pricing | **PASS.** A neutral-smile wrapper (`φ(K) = 1`) produced finite score grids and totals intensities. |
| **G8** | PostgreSQL `save_fit` / `load_fit` bit-identical round trip | **PASS.** Chain names, full chain arrays, latent match IDs and every latent matrix were exactly equal after reload. Portfolio artefacts also reloaded with an identical bet ledger. |

The independent G9 check also passed for `m05`: the engine log-joint matched the separately derived `builder/equations.jl` implementation at the initial and perturbed parameter points.

### Fold-level convergence

| Candidate | Smoke fold | Max R̂ | Worst R̂ parameter | Min bulk ESS | Worst bulk parameter | Min tail ESS | Worst tail parameter | Divergences | Min BFMI | Max depth / capped |
|---|---:|---:|---|---:|---|---:|---|---:|---:|---|
| `m05` | 1 | 1.0070 | `dyn.σ_d` | 1,166 | `dyn.σ_d` | 1,446 | `obs.σ_κ` | 0 | 0.759 | 5 / 0 |
| `m05` | 2 | 1.0066 | `dyn.raw_a[20]` | 744 | `dyn.σ_a` | 716 | `production_wealth.w` | 0 | 0.679 | 7 / 0 |
| `m12` | 1 | 1.0122 | `obs.κ_team_raw[2]` | 1,197 | `dyn.σ_d` | 946 | `production_wealth.w` | 0 | 0.738 | 5 / 0 |
| `m12` | 2 | 1.0053 | `dyn.raw_a[15]` | 1,169 | `dyn.σ_d` | 1,123 | `obs.σ_κ` | 0 | 0.818 | 6 / 0 |

## Sampler correction found by the smoke test

The first complete production-settings attempt inherited the shared-kappa target acceptance of 0.65. `m05` passed, but `m12` recorded **one divergence in 6,400 transitions** on the later fold. Its other diagnostics were healthy (max R̂ 1.0065, minimum ESS 796, BFMI 0.803, maximum depth 6 and no depth caps), but the strict zero-divergence gate correctly failed and portfolio promotion was refused.

The gate was not weakened. The hierarchical production sampler was instead made explicit at target acceptance **0.90**, retaining the same model, priors, folds, four chains, 800 warmup draws and 800 retained draws. The rerun then passed with zero divergences for both candidates. The dedicated sampler is registered as:

```text
queued_nuts_4x800_hierarchical_kappa
```

This is the sampler used by the staged 40-fold grid.

## Posterior finishing-factor estimates

The summaries below use the later smoke fold, which has the deeper history and is therefore the more informative fold for naming team contrasts.

| Candidate | κ_global mean | κ_global 90% HPDI | σ_κ mean | σ_κ 90% HPDI | P(σ_κ > 0.05) |
|---|---:|---|---:|---|---:|
| `m05_hierarchical_kappa` | 1.1136 | [1.0350, 1.1856] | 0.0501 | [0.0001, 0.1016] | 0.438 |
| `m12_hierarchical_kappa` | 1.1099 | [1.0360, 1.1849] | 0.0497 | [0.0000, 0.1019] | 0.425 |

### Interpretation

- The league finishing multiplier is stable across the control and hybrid: approximately **1.11**, consistent with the existing shared-kappa result that goals run about 11–13% above the BBC proxy-xG cell table.
- The posterior for `σ_κ` remains concentrated near zero and its 90% HPDI reaches the boundary. The probability that `σ_κ` exceeds 0.05 is only 0.43–0.44.
- Therefore the smoke provides **no strong evidence of persistent team-specific conversion skill**. It validates the model and sampler plumbing; it does not establish that the extra hierarchy is predictively useful.
- The top and bottom team intervals all cross zero. The ordering below is a posterior ranking, not a list of statistically separated clubs.
- `m05` and `m12` produce almost the same spread and broadly the same ordering. Adding the player-lineup pillar does not reveal a hidden team-finishing signal in this smoke fold.

### Team conversion ranking — `m05` control

| Rank | Team | Mean δ_κ | 90% HPDI | Mean κ_team | P(δ_κ > 0) |
|---:|---|---:|---|---:|---:|
| 1 | Falkirk | +0.0249 | [-0.0726, +0.1276] | 1.1441 | 0.638 |
| 2 | Arbroath | +0.0226 | [-0.0694, +0.1161] | 1.1412 | 0.634 |
| 3 | Alloa Athletic | +0.0213 | [-0.0568, +0.1199] | 1.1393 | 0.627 |
| 23 | Kelty Hearts | -0.0218 | [-0.1170, +0.0648] | 1.0913 | 0.369 |
| 24 | Stranraer | -0.0222 | [-0.1221, +0.0617] | 1.0911 | 0.375 |
| 25 | Forfar Athletic | -0.0305 | [-0.1335, +0.0546] | 1.0821 | 0.344 |

### Team conversion ranking — `m12` hybrid

| Rank | Team | Mean δ_κ | 90% HPDI | Mean κ_team | P(δ_κ > 0) |
|---:|---|---:|---|---:|---:|
| 1 | Arbroath | +0.0226 | [-0.0516, +0.1395] | 1.1373 | 0.634 |
| 2 | Alloa Athletic | +0.0223 | [-0.0674, +0.1168] | 1.1366 | 0.640 |
| 3 | Cove Rangers | +0.0215 | [-0.0734, +0.1167] | 1.1359 | 0.636 |
| 23 | Kelty Hearts | -0.0202 | [-0.1221, +0.0658] | 1.0895 | 0.376 |
| 24 | Bonnyrigg Rose | -0.0216 | [-0.1173, +0.0641] | 1.0878 | 0.371 |
| 25 | Forfar Athletic | -0.0322 | [-0.1440, +0.0471] | 1.0766 | 0.305 |

## Persistence evidence

| Candidate | Fit run UUID | Portfolio UUID | Smoke bets | Smoke return |
|---|---|---|---:|---:|
| `m05_hierarchical_kappa` | `539d5873-2400-4b35-9e2a-6555a7b64cb6` | `77f84ef8-a043-46a6-8b53-ca453e47a3a7` | 57 | +5.37% |
| `m12_hierarchical_kappa` | `a3b5bfef-d72e-4167-a47f-c32d8e77420f` | `1a941183-6911-40e6-a3e1-29409ff0e1ec` | 60 | +10.01% |

These portfolio figures cover only 39 smoke fixtures and are persistence/plumbing checks, not performance estimates.

## 40-fold production staging

Runner:

```text
experiments/scottish_lower/06_joint_player_lineup_fusion/r65_train_hierarchical_kappa_40fold.jl
```

The runner:

- uses the two canonical hierarchical candidates from `l64_hierarchical_kappa_loader.jl`;
- uses `QueuedExecution()` and the native flattened queue;
- schedules 40 folds × 4 chains = **160 fold-chain tasks per candidate**, 320 in total;
- retains per-fold checkpoints for resumability;
- requires the same strict promotion thresholds as the smoke, including zero divergences;
- checks PostgreSQL for an already-completed exact recipe before spending compute;
- defaults to **prepare only**, so merely including/executing the file cannot accidentally launch the grid.

The prepare-only preflight completed successfully on `mcmc-beast`:

```text
40 FeatureSets per candidate
710 OOS fixtures per candidate
all structural parameter counts matched
no existing completed production recipe for either candidate
no MCMC launched
```

Launch only when `mcmc-beast` is otherwise idle:

```bash
cd /root/BayesianFootball
L65_RUN_GRID=true /root/.juliaup/bin/julia --project -t 16 \
  experiments/scottish_lower/06_joint_player_lineup_fusion/r65_train_hierarchical_kappa_40fold.jl
```

## Logs on `mcmc-beast`

```text
/root/BayesianFootball/r64_smoke_20260903.log
    Initial target-acceptance 0.65 attempt; documents the single m12 divergence.

/root/BayesianFootball/r64_smoke_20260903_accept090.log
    Final passing 350/350 smoke run.

/root/BayesianFootball/r65_hierarchical_kappa_preflight_20260903.log
    Final passing 40-fold prepare-only preflight.
```

## Production-readiness decision

**Ready to launch the 40-fold grid.** The implementation is structurally correct, vectorised, fast under compiled ReverseDiff, convergent at the dedicated 0.90 target acceptance, extractable into production latents, priceable, and losslessly persisted. The scientific decision remains open: because `σ_κ` is weak and every named team HPDI crosses zero, promotion as a superior model must depend on the full 710-match OOS evaluation rather than on the smoke posterior ranking.
