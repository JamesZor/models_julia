# The Protocol — how a Scottish Lower model gets built and trusted

**Status:** authoritative. Adopted 2026-08-25.

Every pregame model in this package has the same five anatomical parts:

| # | Part | Where it lives in `src` (baseline example) |
|---|---|---|
| ① | **Config** | the model struct, e.g. `DynamicGoalsTimeDecayModel` — components + hyperparameters |
| ② | **Features** | `Features.required_features(model)` + the extractors under `src/features/extractors/` |
| ③ | **Sampling** | `build_turing_model(model, feature_set)` — the maths |
| ④ | **Extraction / Inference** | `extract_parameters(model, df, feature_set, chain)` — chain → distributions |
| ⑤ | **Score Matrix** | `compute_score_matrix` in `src/predictions/score_computation/` — distributions → market prices |

Nothing structural is missing from the codebase. What was missing was **a gate between each
part**, so that a defect in ④ could not survive into ⑤ and out into a leaderboard.

That is exactly what happened in `archive/open_play/`: the prediction adapter looked up a
*string* team name in an *integer*-keyed map, got `-1`, and silently priced every match with
zero attack/defence effect. The reported champion was not the model that was fitted. See
`archive/open_play/AUDIT_2026-08-24.md`.

## Principle: extend the package, never re-implement it

`src`'s `extract_parameters` (`src/models/pregame/engines/team_level/time_decay/goals.jl:148`)
keys `team_map` by `row.home_team` and gets it right. The bug was born in prototype code that
re-implemented what `src` already did.

**A prototype should add a component, an extractor, or a config — and dispatch into the package
API for everything else.** If you find yourself rewriting extraction or score-matrix
construction inside `current_development/`, stop: either the package needs a new method, or you
are about to reproduce the 2026-08-24 audit.

---

## The gates

Gates 0–5 are cheap (minutes, one short MCMC smoke) and live in a single per-model file,
`v01_walkthrough.jl`, as numbered REPL blocks. Gates 6–7 are separate runners.

### Gate 0 — Contract

Establishes what is being modelled and on what. Runs once per model.

- Pinned `DataStore` snapshot with a recorded fingerprint.
- Fold inventory **printed**: number of folds, target season, boundary dates, train/OOS counts
  per fold.
- Season `25/26` is **sealed**. All development, selection, and policy tuning happens on `24/25`.

*Catches:* models being compared on different fixtures.

### Gate 1 — Config

- The config prints **every** hyperparameter and component choice. No hidden defaults.
- The config hashes deterministically; that hash appears in every artifact name it produces.
- `Features.required_features(model)` resolves — every declared feature has a live extractor.

*Catches:* under-keyed caches, ambiguous artifact selection.

### Gate 2 — Features (the anti-leakage gate)

- **Kickoff filtration:** `max(training kickoff) < min(OOS kickoff)` for every fold.
  Nominal round/biweek grouping is not enough — postponements cross the cutoff.
- **Perturbation test:** append future matches to the `DataStore`; an earlier fold's
  `FeatureSet` must come back **bit-identical**. This is the test that catches a
  conversion/rating model fitted on the full sample.
- **Type purity:** no `missing` and no `NaN` in any vector destined for a `@model`.
  `Int` or `Float64` only.
- **Map identity:** `team_map` round-trips name → index → name, and its key type matches
  exactly what `extract_parameters` will look up.
- **Coverage:** unknown / newly promoted teams are counted and reported per fold, never
  silently mapped to zero.

*Catches:* temporal leakage, AD failures, silently zeroed team effects.

### Gate 3 — Sampling

- **Equation parity:** evaluate the Turing model's `logjoint` at a fixed parameter vector and
  compare against an independent pure-Julia reimplementation in the model's `l02_equations.jl`.
  If these disagree, the documented model is not the fitted model.
- **Gradient diff:** compiled `ReverseDiff` vs `ForwardDiff` vs finite differences at the same
  point. Report max relative error and **median gradient time** (target ≈ 0.6 ms; see
  `docs/turing_ad_performance_guide.md` at repo root).
- **Init safety:** no `-Inf` / `NaN` log-joint at prior draws.
- **Smoke run:** one fold, 4 chains × 500 warmup / 500 retained, **persisted through
  `src/experiments`**. Report Rhat, bulk/tail ESS, divergences, tree depth, BFMI, wall time,
  and the extrapolated cost of the full grid.

The saved smoke chain is the input to Gate 4. Nothing about it is throwaway.

*Catches:* the maths not matching the write-up; AD regressions; init failures that silently
drop splits from a queued run.

### Gate 4 — Extraction / Inference

Two checks, both required — one exact, one real.

- **Synthetic-chain parity (exact):** fabricate a `Chains` object with known parameter values,
  run `extract_parameters`, and require the resulting λ to match `l02_equations.jl` to ~1e-10.
  This is what catches a dropped hierarchical scale (`tau`) or a wrong chain variable name.
- **Real-chain load (plumbing):** load the Gate 3 experiment back off disk and run the ordinary
  `Experiments.extract_oos_predictions` path. Proves the persisted artifact is loadable and the
  production route works, not just a hand-built object.
- Every sampled site in the model is either consumed by extraction or **explicitly declared
  unused** in `MODEL.md`.
- Population-fallback team-sides are counted and reported.
- No `NaN` / `Inf`; PPD row count equals fixtures × draws.

*Catches:* dropped posterior scales, wrong chain names, zeroed effects, unloadable artifacts.

### Gate 5 — Score Matrix

- The model type is **asserted** to be in the relevant `compute_score_matrix` dispatch `Union`.
  A missing entry surfaces much later as a confusing shape-parameter error; assert it here.
- Every draw's matrix sums to 1 ± tol; truncation mass at `max_goals` is reported, not assumed
  negligible.
- Market identities hold: `P(1)+P(X)+P(2)=1`; O/U and BTTS computed from the matrix agree with
  direct cell sums.
- Moment check: `E[total goals]` from the matrix ≈ `λ_h + λ_a`. Catches mis-plumbed dispersion.

### Gate 6 — Evaluation (development season `24/25` only)

- Assert **identical match IDs and market IDs** across every model being compared, before any
  ranking is printed.
- Per-line log loss / Brier / CRPS and calibration curves, against de-vigged Bet365 and against
  Betfair close.
- Score **per line**, never aggregated across the selections of one market.
- Book: **1X2, O/U 0.5 / 1.5 / 2.5 / 3.5, BTTS.** Nothing else enters the ranking.

### Gate 7 — Growth / CLV (development season `24/25` only, for now)

- Betfair close, 2% commission, portfolio-cap Kelly (Σ simultaneous stakes ≤ ~0.2), curated per
  line.
- **CLV is the primary discriminator; growth `G` is secondary.**
- `25/26` stays sealed. It is opened once, per model family, after selection is frozen —
  currently deferred to keep iteration fast.

---

## Gate reporting convention

Each block prints a PASS/FAIL table and ends with a single `@assert all_pass`. You read the
table first; the assert is the tripwire, not the report.

Every gate run appends a dated entry to that model's `FINDINGS.md`, with the config hash beside
it. **A result that is not in `FINDINGS.md` does not exist.**

## Build order

1. `01_team_poisson` — establishes the protocol on existing `src` code, and gives every later
   model a reference opponent on identical fixtures.
2. `02_apm_player_poisson` — rebuilt from scratch. The risk here is not the Turing model, it is
   the rating construction: the ridge fit must be history-only, and the Gate 2 perturbation test
   is run **on the ratings themselves**.
3. `03_open_play_recombination` — retrofit of `archive/open_play_rebuild`. Its Stage 8 chains
   converged over 38 folds; run Gates 4 and 5 against those saved chains before trusting a
   single number. If parity passes, skip resampling and go to Gates 6–7.
