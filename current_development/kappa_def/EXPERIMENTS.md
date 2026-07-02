# kappa_def — experiment tracker

> Update this table + the log below after every run. Status: `planned` → `running` →
> `done` / `failed`. Keep verdicts one line; long analysis goes in `NOTES.md` findings log.

| id | file | what | league | status | verdict |
|---|---|---|---|---|---|
| r00 | `r00_persistence_gate.jl` | EDA gate: per-team (goals−xG) attack + (conceded−xG_conceded) defense residual persistence (split-half + season t→t+1) | Ireland + 718 | planned | — |
| r01 | `r01_single_split_shakedown.jl` | single split, 3 κ modes in parallel: convergence + per-team κ posterior inspection | Ireland (79) | planned | — |
| r02 | (future) | r01 on 718 (flip SEGMENT) if r01 converges | 718 | planned | — |
| r03 | (future) | full-CV winner-mode vs V0 control, judged vs Betfair close (LPD/GLMEdge per line, r13/r14 pattern) | 718 first | planned | — |

## Decision rules (agreed up front, so we don't rationalize after)

- **r00 gate:** defensive residual persistence (split-half corr AND/OR season-over-season
  corr) > 0 with permutation p < 0.10 ⇒ κ_def structure is real, proceed. Both ≈ 0 ⇒ the
  attack-only V0 is already right; park the stream (record and stop).
- **r01:** any mode with max R-hat > 1.05 on its κ params (checked on RAW chains, not the
  curated conv.df) is out. τ posteriors pulled to ~0 with team multipliers ≈ 1.00 ⇒ that
  mode learned nothing (σ-hierarchy-null pattern) — note it, don't eval it.
- **r03:** judge vs market (per-line LPD/GLMEdge), not vs goals; the market may already
  price team defense ([[position-aware-ratings-rejected]] precedent).

## Run log

(append: date, id, config hash/settings, wall time, one-line result)
