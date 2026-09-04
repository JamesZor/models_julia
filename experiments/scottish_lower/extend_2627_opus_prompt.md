# WORK BRIEF: EXTENDING SCOTTISH LOWER PRODUCTION MODELS TO SEASON 2026/27

<role>
You are an autonomous quantitative research and MCMC engineering agent in the BayesianFootball framework.
Your task is to incrementally extend our top-performing Scottish Lower joint-player models into the ongoing 2026/27 season, perform convergence audits, verify database persistence on `mcmc_experiments`, and confirm operational readiness for tomorrow's (Saturday 2026-09-05) MatchDay paper card.
</role>

<context>
1. **Target Models (Tournaments 56 & 57, Experiment `scottish_lower_joint_player_2426`)**:
   - Primary Production Standard: `m12_joint_hybrid_synergy` (UUID: `132df5c2-c742-4e95-8693-3aeb2b2cbaef`)
     Two-arm Gamma-Poisson observation + team time decay (180d) + `PlayerLineupPillar` (shots-RAPM starters + bench 0.10) + `ProductionWealthCovariate`.
   - Baseline Control: `m05_joint_production_wealth` (UUID: `ed541a7c-01e2-447e-a771-783517728d47`)
     Two-arm observation + team time decay (180d) + `ProductionWealthCovariate` (no lineup pillar).
   - Sensitivity Companion: `m13_joint_composite` (UUID: `5474e824-8c9d-4613-8e39-841426c3f80f`)
     Adds travel distance covariate.

2. **Why Extension is Needed for Tomorrow**:
   Tomorrow (Saturday 2026-09-05 @ 15:00 BST) has a full 10-fixture slate in Scottish League One and League Two.
   The historical 40 folds only trained up through May 2026 (end of 2025/26). In August 2026, 49 matches were played.
   Extending the fits adds Folds 41, 42, and 43, updating the latent team attack/defence states ($\alpha, \beta$) with the latest August form. Tomorrow's MatchDay execution will condition on Fold 43 (the August 29 boundary).

3. **Extension Runner**:
   A dedicated runner has been prepared at:
   `experiments/scottish_lower/06_joint_player_lineup_fusion/r68_extend_joint_player_2627.jl`
   It includes the 3-field deserialization compatibility shim for `JointGammaPoissonObservation` (mirroring `l66`), widens the splitter with `["24/25", "25/26", "26/27"]`, executes via `QueuedExecution(16)` on `mcmc-beast`, and validates database reloading and `MatchDay.canonical_fit`.
</context>

<tasks>

### Task 1: Verify Pre-Conditions & Preview
Run the preview mode:
```bash
julia --project=. -t 16 experiments/scottish_lower/06_joint_player_lineup_fusion/r68_extend_joint_player_2627.jl --preview
```
Confirm that both `m12_joint_hybrid_synergy` and `m05_joint_production_wealth` report 40 existing folds and 3 new folds to fit (Folds 41, 42, 43; 49 matches).

### Task 2: Execute Incremental MCMC Extension
Run the extension runner:
```bash
julia --project=. -t 16 experiments/scottish_lower/06_joint_player_lineup_fusion/r68_extend_joint_player_2627.jl
```
*(If you wish to include `m13_joint_composite`, pass `--all`).*
Monitor the queued execution across the 16 threads.
Ensure MCMC completes cleanly with 0 crashes.

### Task 3: Convergence & Database Audit
Verify the six-part convergence audit on the reloaded 43-fold fits:
- Split R̂ ≤ 1.05 across all parameters
- Bulk ESS ≥ 100, Tail ESS ≥ 100
- Divergences: exactly 0
- Total OOS fixtures: exactly 759 (710 historical + 49 new)
- Relational rows in `fold_results` and `match_latents` appended correctly
- Fit artifact updated in `fit_artifacts`

### Task 4: MatchDay Live Operational Verification
Write and execute a small verification script (or Julia expression) to verify:
1. `MatchDay.canonical_fit(PostgresStorage("scottish_lower_joint_player_2426"), "132df5c2-c742-4e95-8693-3aeb2b2cbaef"; require_converged = true)` loads and reports `converged = true` with 43 folds.
2. Simulate `MatchDay.select_split` against tomorrow's Saturday fixtures (date `2026-09-05`, 10 fixtures). Confirm it selects Fold 43 with zero warnings.
3. Test `MatchDay.matchday_latents` or price a dummy card using the extended fit to ensure zero crashes, finite rates, and valid team/lineup mappings.

### Task 5: Summary Report & Commit
1. Document the results in `experiments/scottish_lower/06_joint_player_lineup_fusion/EXTEND_2627_REPORT.md` (runtime, convergence table, diagnostic summaries).
2. Commit all artifacts to branch `feat/extend-scottish-lower-2627`.
</tasks>
