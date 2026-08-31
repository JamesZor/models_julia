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
- `m04_joint_joint_wealth_dist`: Joint + Squad Wealth + Distance
- `m05_joint_production_wealth`: Joint + Production Wealth
- `m07_joint_bench_depth`: Joint + Bench Depth
- `m08_joint_composite`: Joint + Production Wealth + Bench Depth

## 3. Scripts
- `r45_smoke_joint_gamma_poisson.jl`: Single-fold MCMC parameter recovery and smoke test.
- `r46_train_5models_2426_joint.jl`: 40-fold multi-season walk-forward grid for `mcmc-beast`.
