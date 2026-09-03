# TASK: HIERARCHICAL TEAM KAPPA SMOKE TEST & 40-FOLD GRID STAGING

<objective>
Take over the Hierarchical Team Kappa task from Claude (which hit its session limit). Execute the extended smoke test `r64_smoke_hierarchical_kappa.jl` on `mcmc-beast`, verify all 8 verification ladder gates (MCMC convergence R̂ < 1.05, ESS > 400, 0 divergences, ReverseDiff tape timing, parameter extraction), analyze the posterior estimates of κ_global and σ_κ, stage the 40-fold grid runner `r65_train_hierarchical_kappa_40fold.jl`, and document everything in `HIERARCHICAL_KAPPA_SMOKE.md`.
</objective>

<context>
1. **Model Formulation**:
   In the production Two-Arm Joint model:
   - Arm 1 (proxy xG): pxg_s ~ Gamma(ν, μ_s / ν)  (evaluated on matches with commentary)
   - Arm 2 (goals):    y_s   ~ Poisson(κ_s · μ_s) (evaluated on all matches)
   Under Hierarchical Team Kappa:
   log_κ_t = log_κ_global + δ_κ[t]
   δ_κ = σ_κ .* (raw_t .- mean(raw_t)), where raw_t ~ Normal(0, 1), σ_κ ~ HalfNormal(0, 0.10)
   This guarantees zero-sum identification (∑ δ_κ = 0) while letting each team have a finishing multiplier.

2. **Current Code State**:
   - `src/models/pregame/builder/components.jl` implements `HierarchicalKappa{S}`.
   - `src/models/pregame/builder/engine.jl` implements `_observe` with vectorised team index lookups `home_idx` and `away_idx`.
   - `test/test_joint_gamma_poisson.jl` has passed 100% locally.
   - Code has been synced to `mcmc-beast` and precompilation verified (`isdefined(Main, :HierarchicalKappa) == true`).
   - Loader: `experiments/scottish_lower/06_joint_player_lineup_fusion/l64_hierarchical_kappa_loader.jl`
   - Smoke runner: `experiments/scottish_lower/06_joint_player_lineup_fusion/r64_smoke_hierarchical_kappa.jl`

3. **Remote Machine Details**:
   - Host: `mcmc-beast` (reach via `ssh root@mcmc-beast`)
   - Directory: `/root/BayesianFootball`
   - Julia path: `/root/.juliaup/bin/julia`
   - Experiment DB: `BF_EXPERIMENTS_DB_URL=postgresql://postgres:football_mcmc_secure@localhost:5432/mcmc_experiments` (available locally on beast).
</context>

<execution_instructions>
1. Run `r64_smoke_hierarchical_kappa.jl` on `mcmc-beast`:
   ```bash
   ssh root@mcmc-beast "cd /root/BayesianFootball && /root/.juliaup/bin/julia --project -t 16 experiments/scottish_lower/06_joint_player_lineup_fusion/r64_smoke_hierarchical_kappa.jl"
   ```
2. Verify that all 8 gates (G1 through G8) pass:
   - G1: ReverseDiff compiled tape gradient eval < 0.15 ms.
   - G2: Structural parameter contract (obs.σ_κ and obs.κ_team_raw[1:n_teams] present).
   - G3: NUTS completes with no crashed chain across 4 chains x 800 warmup x 800 retained draws.
   - G4: Strict convergence audit: max R̂ < 1.05, min bulk ESS > 400, min tail ESS > 300, divergences == 0.
   - G5: κ extraction is identified (∑ δ_κ = 0.0) with 90% HPDIs reported.
   - G6: `CountLatents` extraction is finite and positive.
   - G7: `SmileScoreGrid` prices successfully off these latents.
   - G8: `save_fit` and `load_fit` roundtrip bit-identically through PostgreSQL `mcmc_experiments`.
3. If any issue occurs, inspect and fix it cleanly in both the local repo and `mcmc-beast`.
4. Stage and run the 40-fold production grid runner:
   `experiments/scottish_lower/06_joint_player_lineup_fusion/r65_train_hierarchical_kappa_40fold.jl`
   configured with `QueuedExecution()` for parallel execution on beast across all 40 folds.
   **CRITICAL REQUIREMENT**: The 40-fold grid MUST be launched inside a persistent tmux session on `mcmc-beast` (e.g., `tmux new-session -d -s r65_hierarchical_kappa '...'`), NOT as a bare foreground SSH command. This ensures the 40-fold run is detached, survives SSH network hiccups, and can be inspected via `tmux attach -t r65_hierarchical_kappa` or log tailing.
5. Create `experiments/scottish_lower/06_joint_player_lineup_fusion/HIERARCHICAL_KAPPA_SMOKE.md` documenting:
   - Convergence audit metrics.
   - Posterior summaries for κ_global, σ_κ, and the team conversion ranking (top over-converters and under-converters).
   - Confirmation that the 40-fold grid is ready for production.
</execution_instructions>
