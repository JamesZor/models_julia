# Findings ledger

One line per completed gate run or result, newest first. Details live in each model's
`FINDINGS.md`. Nothing is quoted elsewhere unless it appears here.

| Date | Model | Config hash | What | Result |
|---|---|---|---|---|
| 2026-08-25 | 01_team_poisson | `54080fde` | Gates 0–2 on 19 dev folds | **PASS** 5/5, 4/4, 6/6 |
| 2026-08-25 | src (via model 01) | — | Pooled groups walk a per-tournament clock → contaminated fold 6 | **Fixed** in src as T001 (`edd5eba`); re-verified 20 folds, 0 dropped |
| 2026-08-25 | 01_team_poisson | `54080fde` | Gates 0–2 re-run on the T001 shared calendar clock | **PASS** 5/5, 4/4, 7/7 |
| 2026-08-25 | 01_team_poisson | `54080fde` | Gate 3a/3b — equation parity and gradients | **PASS** 3/3, 7/7; parity Δ = 0 exactly |

## Carried forward from `archive/` (pre-protocol, not gated)

| Date | Source | Claim | Status |
|---|---|---|---|
| 2026-08-24 | `archive/open_play_rebuild` | 38 walk-forward folds converged: max Rhat 1.0099, min bulk ESS 1196, 0 divergences, 710 OOS fixtures | Convergence trusted; **no evaluation performed** |
| 2026-08-24 | `archive/open_play/AUDIT_2026-08-24.md` | All prior open-play/wealth/pxG leaderboards invalid | Accepted; results quarantined |
| 2026-08-2x | `archive/open_play` EDA | Penalties + own goals ≈ 9.6% of goals; NP-NOG target raises season-to-season persistence +26% | Descriptive EDA, believed |
