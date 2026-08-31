# Experiment 03: Scottish Lower Two-Arm Joint (Gamma Proxy xG + Poisson Goals)

## 1. Overview & Mathematical Architecture
A unified two-arm count model evaluating shared latent team strength ($\mu_{\text{home}}, \mu_{\text{away}}$) against both chance creation quality (proxy xG via Gamma likelihood) and match conversion (actual goals via Poisson likelihood):

- **Arm 1 (Proxy xG)**: $\text{pxG} \sim \text{Gamma}(\nu, \mu / \nu)$ with precision $\nu \sim \text{truncated}(\mathcal{N}(4.0, 1.5), 0.5, \infty)$.
  * *Masked Likelihood*: Evaluated for 23/24+ matches with BBC commentary; skipped on pre-23/24 history.
- **Arm 2 (Goals)**: $\text{Goals} \sim \text{Poisson}(\kappa \cdot \mu)$ with conversion scale $\log \kappa \sim \mathcal{N}(0, 0.2)$.
- **Latents & Scoring**: Pricing rate $\lambda = \kappa \cdot \mu$ packaged into `CountLatents`.

## 2. Models in Benchmark
- `m00_joint_baseline`: Joint Gamma + Poisson Baseline
- `m02_joint_squad_wealth`: Joint + Squad Wealth
- `m03_joint_distance`: Joint + Travel Distance
- `m04_joint_wealth_distance`: Joint + Squad Wealth + Distance
- `m05_joint_production_wealth`: Joint + Production Wealth
- `m07_joint_bench_depth`: Joint + Bench Depth
- `m08_joint_composite`: Joint + Production Wealth + Bench Depth

## 3. Scripts
- `r45_smoke_joint_gamma_poisson.jl`: Single-fold MCMC parameter recovery and smoke test.
- `r46_train_5models_2426_joint.jl`: 40-fold multi-season walk-forward grid for `mcmc-beast`.
  Fits all seven joint arms plus `m00_poisson_control`, the identical spine with no Gamma
  arm — that control is the decision comparison, since it isolates the second likelihood
  from every other difference.
- `l45_joint_gamma_poisson.jl`: shared loader (arm assembly, coverage accounting, preflight,
  gates, leaderboard).
- `experiments/regate_r45.jl`: re-runs the r45 gate battery against SAVED fits without
  re-sampling.

## 4. Smoke test result (fold 1, commentary-only feed)
All 55 gates passed. ν is identified by the proxy arm at 80.4-80.8% posterior shrinkage
against its prior, and κ = 1.104-1.106 at ~76% shrinkage — the league converts about 10%
more than the BBC shot-xG cell table predicts. R̂ <= 1.009, BFMI 0.66-0.71, no tree-depth
saturation, gradient 0.0626 ms over 191 tape instructions.

Note the proxy feed is `fallback = :none` (live-text commentary only, 23/24+). `:shots`
reaches ~100% coverage via BBC match-page shot counts, but that rung is `shots x a league
constant` — volume, not chance quality — and it silently becomes the majority of the Gamma
arm's evidence.
