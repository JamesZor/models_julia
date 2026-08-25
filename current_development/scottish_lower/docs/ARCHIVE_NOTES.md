# What is in `archive/`, and how much of it to believe

Everything under `archive/` predates the 2026-08-25 restructure. It is kept for reference,
equations, and EDA — **not** for its results. No leaderboard, portfolio number, or ranking in
any of these directories should be quoted or reused.

## Trust status

| Directory | Status | Notes |
|---|---|---|
| `open_play/` | **Quarantined** | The 2026-08-24 audit (`open_play/AUDIT_2026-08-24.md`) found the reported champion was priced with zero team effects and dropped hierarchical scales, plus leakage in the pxG conversion model and the wealth features. Read the audit; it is the best statement of what to guard against |
| `open_play_rebuild/` | **Partially trusted** | Clean-room rebuild that was genuinely staged (data contract → features → equations → gradients → extraction → smoke → 38-fold run). Convergence is real: 38 folds, max Rhat 1.0099, 0 divergences, 710 OOS fixtures. No evaluation was ever done. Chains may be reused only after Gates 4–5 pass against them |
| `neg_bin/`, `wealth/`, `distance/`, `corners/`, `proxy_xg/` | **Reference only** | Same prediction-adapter family as `open_play/`; their rankings inherit the same defects. The EDA and the equations are still useful reading |
| `portfolio/` | **Reference only** | Kelly/backtest machinery. Superseded by `src/Portfolio` |
| `bug_fix/` | **Reference only** | Ad hoc |

## What was genuinely learned (and survives)

- Penalties + own goals are ~9.6% of all goals in 56/57, and filtering them raises year-over-year
  team persistence (`r ≈ 0.228` vs `0.180` on raw goals). The decomposition idea is sound; it was
  the plumbing that failed.
- Own goals look like a flat Poisson rate (~0.0276 per team-match) with no team persistence.
- Referee penalty rates vary ~4.4× across officials, with substantial home bias.
- Scottish Lower (56/57) has **no SofaScore player ratings** — which is why the APM stream exists.
- Tournaments 54–57 have BBC commentary data (`ds.bbc`), the basis for proxy-xG.

## The audit's lessons, generalised

These became Gates 2, 4 and 5 in [`PROTOCOL.md`](PROTOCOL.md):

1. A prediction adapter that re-implements `src` extraction will silently disagree with it.
2. Hierarchical scales (`tau`) applied in training and omitted in prediction produce a
   plausible-looking, wrong model.
3. Any sub-model fitted on the full sample (shot→xG conversion, player valuations, ridge
   ratings) leaks the future into every historical fold.
4. Caches keyed by model label alone survive changes to the model.
5. Models compared without asserting identical fixtures and markets are not comparable.
6. Policy tuned on the reporting period is not out-of-sample.
