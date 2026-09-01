# Experiment 03: Scottish Lower Two-Arm Joint (Gamma Proxy xG + Poisson Goals)

## 1. Overview & Mathematical Architecture
A unified two-arm count model evaluating shared latent team strength ($\mu_{\text{home}}, \mu_{\text{away}}$) against both chance creation quality (proxy xG via Gamma likelihood) and match conversion (actual goals via Poisson likelihood):

- **Arm 1 (Proxy xG)**: $\text{pxG} \sim \text{Gamma}(\nu, \mu / \nu)$ with precision $\nu \sim \text{truncated}(\mathcal{N}(4.0, 1.5), 0.5, \infty)$.
  * *Masked Likelihood*: Evaluated for 23/24+ matches with BBC live-text commentary; skipped on pre-23/24 history.
- **Arm 2 (Goals)**: $\text{Goals} \sim \text{Poisson}(\kappa \cdot \mu)$ with conversion scale $\log \kappa \sim \mathcal{N}(0, 0.2)$.
- **Latents & Scoring**: Pricing rate $\lambda = \kappa \cdot \mu$ packaged into `CountLatents`. $\mu$ is retained as `true_xg_*` — this is the one observation family where the pricing rate and the expected xG genuinely differ, separated exactly by $\kappa$.

Neither arm is a fallback for the other. The proxy arm sharpens $\mu$ on the seasons carrying live text; the goals arm, which needs no text, carries that sharpened $\mu$ across the whole history.

## 2. Models Evaluated
- `m00_joint_baseline`: GlobalInterception + TimeDecay (180d) + GlobalHomeAdv + JointGammaPoissonObservation
- `m02_joint_squad_wealth`: Baseline + WealthCovariate
- `m03_joint_distance`: Baseline + DistanceCovariate
- `m04_joint_wealth_distance`: Baseline + WealthCovariate + DistanceCovariate
- `m05_joint_production_wealth`: Baseline + ProductionWealthCovariate (Richards Sigmoid)
- `m07_joint_bench_depth`: Baseline + BenchDepthCovariate
- `m08_joint_composite`: Baseline + ProductionWealthCovariate + BenchDepthCovariate
- `m00_poisson_control` **(control)**: the identical spine with **no Gamma arm**. This is the decision comparison — it isolates the second likelihood from every other difference.

## 3. Results & Evaluation Metrics (710 Out-of-Sample Fixtures)
| Model | LogLoss | Brier | RPS | ECE | γ (HA) | κ | ν | ΔLogLoss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `m05_joint_production_wealth` 🏆 | **0.6571** | **0.2326** | **0.2256** | 0.0160 | +0.122 | 1.126 | 3.90 | **−0.0032** |
| `m08_joint_composite` | 0.6572 | 0.2327 | 0.2260 | 0.0134 | +0.124 | 1.126 | 3.92 | −0.0030 |
| `m04_joint_wealth_distance` | 0.6574 | 0.2327 | 0.2259 | 0.0161 | +0.113 | 1.127 | 3.93 | −0.0029 |
| `m02_joint_squad_wealth` | 0.6575 | 0.2328 | 0.2259 | 0.0144 | +0.124 | 1.128 | 3.92 | −0.0028 |
| `m03_joint_distance` | 0.6575 | 0.2328 | 0.2260 | 0.0145 | +0.114 | 1.127 | 3.95 | −0.0028 |
| `m00_joint_baseline` | 0.6575 | 0.2328 | 0.2261 | 0.0161 | +0.127 | 1.130 | 3.94 | −0.0028 |
| `m07_joint_bench_depth` | 0.6577 | 0.2329 | 0.2265 | 0.0129 | +0.129 | 1.129 | 3.94 | −0.0026 |
| `m00_poisson_control` | 0.6603 | 0.2341 | 0.2273 | **0.0099** | +0.146 | — | — | 0.0000 |
| *Betfair closing line* | *0.6568* | *0.2326* | — | — | — | — | — | *−0.0035* |

**The gain is the likelihood, not the covariates.** `m00_joint_baseline` carries no covariates and already achieves −0.0028. The entire seven-arm spread is 0.0006, noise beside the 0.0028 step down from the control. The second likelihood is worth roughly **five times** what the best covariate bought in the Poisson family (experiment 01's `m05` managed −0.0006 over its own baseline).

**The joint model closes almost the whole gap to the market**, reaching Brier parity (0.2326) and landing 0.0003 LogLoss behind the closing line, where the Poisson control sat 0.0035 behind.

**Caveat — the control is better calibrated.** ECE 0.0099 against the joint arms' 0.0129–0.0161. The joint model is *sharper* but slightly less calibrated, which matters directly for Kelly staking and is visible in §4.

### 3.1 Identified parameters, not assumed ones
- $\kappa = 1.126$–$1.130$ across all seven arms: Scottish League One/Two convert about **13% more goals than the BBC shot-xG cell table predicts**. Posterior shrinkage ~76% against the prior.
- $\nu = 3.90$–$3.95$, posterior sd ~0.28 against a prior sd of 1.45 — **~80% shrinkage**. The proxy arm genuinely identifies its own precision; $\nu$ is not sampling its prior. (A posterior *mean* near the prior mean of 4.0 cannot distinguish these; the sd is the discriminator.)
- **Squad wealth's weight shrank when the proxy arm was added**: $w_{\text{raw}} = +0.125$ in experiment 01 → $+0.082$ here. Wealth was partly proxying for chance quality that the Gamma arm now measures directly. Home advantage moved the same way, $\gamma$ +0.146 → ~+0.12.

### 3.2 Convergence (40-fold walk-forward, 4 chains, 800/800)
All eight arms passed the full six-gate audit. Max $\hat{R} \le 1.013$, min ESS 635, worst divergence rate 12/128,000 (9.4e-5, against a 1e-3 threshold), BFMI 0.66–0.71, zero tree-depth saturation.

### 3.3 Proxy-xG coverage
Live-text commentary only (`fallback = :none`), which is a hard cliff at 23/24:

| Season | 20/21 | 21/22 | 22/23 | 23/24 | 24/25 | 25/26 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Commentary coverage | 0% | 0% | 0% | 100% | 100% | 100% |

Folds carry ~50% of matches observed but **~81% of the decayed evidence weight** — the covered matches are the recent ones, which the time-decay kernel already weights most. `fallback = :shots` was rejected: BBC match pages carry shot counts back to 20/21, so that rung reaches ~100% coverage while feeding the Gamma arm `shots × a league constant` — volume, not chance quality.

## 4. Betfair Exchange Portfolio Backtest (2% Net Commission, Fractional Kelly)
*Pending — `r47_portfolio_betfair_joint.jl` running. This section is the decision test: a log-loss gain that the closing line has already priced is worth nothing in money.*

## 5. Scripts
- `r45_smoke_joint_gamma_poisson.jl`: Single-fold smoke test with 55 gates — convergence audit, κ/ν bands, ν identification, λ = κ·μ extraction, clamp headroom, and the AD guide's §10.1/§10.3 battery.
- `r46_train_5models_2426_joint.jl`: 40-fold multi-season walk-forward grid on `mcmc-beast` (~19h, 8 arms).
- `r47_portfolio_betfair_joint.jl`: Betfair Exchange portfolio backtest against the Poisson control.
- `l45_joint_gamma_poisson.jl`: shared loader (arm assembly, coverage accounting, per-fold preflight, gates, leaderboard).
- `experiments/regate_r45.jl`: re-runs the r45 gate battery against SAVED fits without re-sampling.
- `experiments/r46_metrics.jl`: full evaluation metrics from saved fits, no re-sampling.

## 6. Implementation Notes & Traps
- **`MatchProxyXGFeature` refuses `fallback = :goals`.** Feeding goals to the Gamma arm would hand it the counts the Poisson arm already reads and double-count every goal. The feature errors rather than allowing it.
- **The mask key is `:flat_pxg_obs_available`, not `:flat_pxg_available`.** `PxGFeature` already owns the latter for pre-match *form* availability; a model carrying both feeds would have had one silently overwrite the other.
- **`JointGammaPoissonObservation` requires a `ClampGuard` with finite `lo`.** The Gamma arm's $e^{-\eta}$ term is unbounded below, and a compiled ReverseDiff tape has no branch left to catch the overflow. Measured headroom on posterior draws: $\eta \in [-0.79, +1.10]$ against a $[-10, 10]$ clamp, so the guard never binds.
- **`end_dynamics = nothing`, never `0`.** `0` means "stop at step zero" and yields a split with **zero** out-of-sample fixtures. r46 now gates on scored-fold count before sampling.
- **`target_match_ids` is fitted, not held out** (`splitting/methods.jl:221` names the union `fitted_ids`). It is the walk-forward's expanding training block; genuine OOS comes from `get_next_matches`, ~19 fixtures per fold, 710 across the grid.
- **AD**: 56 parameters, 191 tape instructions, 0.0626 ms warmed-minimum gradient (guide target 0.1 ms), tape length independent of fold size, compiled == fresh ReverseDiff == ForwardDiff.
