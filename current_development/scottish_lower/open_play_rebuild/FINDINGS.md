# Stage 2 findings

**Status:** history-only component audit completed on `mcmc-beast` at commit `63785ab`; no sampling and no files written by the runner.

## Boundary and integrity

```text
fold: 38
history matches: 720
target matches excluded: 339
ledger rows: 720
duplicate incident IDs: 0
missing incident IDs: 0
missing-side component events: 0
rescinded component events: 0
missing history matches: 0
duplicate match rows: 0
```

The loader asserted disjoint history/target IDs and restricted incidents before classification.

## Own-goal convention

Among informative matches containing own-goal events where exactly one convention reconciled:

| Tournament | Season | Informative matches | Beneficiary-only valid | Committing-only valid |
|---:|:---:|---:|---:|---:|
| 57 | 23/24 | 8 | 8 | 0 |
| 56 | 23/24 | 14 | 14 | 0 |
| 56 | 24/25 | 6 | 6 | 0 |
| 57 | 24/25 | 11 | 11 | 0 |
| **Total** | | **39** | **39** | **0** |

This snapshot uniquely supports the provider `is_home` field as the **beneficiary/scoring side** for `ownGoal` incidents. Stage 3 may use that convention, while retaining reconciliation as a mandatory QA gate.

Overall, the beneficiary convention reconciled 718/720 history matches; the committing convention reconciled 679/720.

## Quarantine

Two matches failed component reconciliation under both conventions:

```text
match 11395473 (tournament 57, 23/24): official 2-0; ordinary incidents 2-1
match 12477131 (tournament 56, 24/25): official 1-3; ordinary incidents 2-3
```

Both have zero recorded penalties and own goals. Read-only BBC/SofaScore database reconciliation established:

- For `11395473`, SofaScore's official score is 2–0 while both the SofaScore incident progression and BBC report a third, away goal and 2–1. BBC text contains no explicit disallowed/VAR explanation. This unresolved provider disagreement remains quarantined.
- For `12477131`, SofaScore and BBC agree on 1–3. SofaScore incident `206671` duplicates the minute-86 score state of player-bearing incident `206670`, has no player, and has no BBC counterpart. It is strong duplicate evidence, but the match remains quarantined until a versioned semantic-deduplication policy is approved.

See `DATABASE_EVIDENCE.md`. Neither row is guessed, clipped, residual-adjusted, or silently overridden from another provider.

## Decision

Approve the beneficiary own-goal convention for the audited snapshot and proceed to Stage 3 using 718 reconciled rows for this boundary. Preserve the two-match quarantine, snapshot/boundary provenance, and all reconciliation checks in every future split.

# Stage 3 findings

**Status:** history-only canonical identity and feature validation passed remotely at commit `9ddc398`; no model, sampling, or default file writes.

```text
boundary: 38
history IDs: 720
target IDs: 339
registry SHA256: 5405533c43583627caf87ef2bb6a53b12a31e7c3a493a7cc3e9fd0c4651cd3af
canonical aliases: 46
included reconciled history rows: 718
quarantined history rows: 2
history-seen posterior teams: 22
```

All history/target disjointness, outcome filtration, sorted posterior-column, concrete vector,
finite weight, and `56→1`/`57→2` league assertions passed. Target-only East Kilbride (canonical
SofaScore ID `170622`) correctly resolved to column zero with `:target_only_population_fallback`;
a history-seen team resolved to its stored column, an unknown identity resolved explicitly to
`:unknown_identity`, and a conflicting ID/name pair raised an error.

The only two unique name/slug diagnostics were benign display-name punctuation differences:
`Edinburgh City F.C.` and `Kelty Hearts F.C.` both matched their DataStore/provider slugs exactly.
The normal feature builder consumed the validated registry DataFrame and performed no database I/O.

# Stage 4 findings

**Status:** pure maths implementation validated remotely at commit `02a4414`; no Turing model, sampling, or writes.

The production-shaped Stage-3 FeatureSet contained 718 reconciled rows and 22 history-seen teams
across both leagues. The deterministic interior point produced weighted data-only log likelihood
`-1500.1880932674035`. All checks passed:

- primitive manifest, support, dimensions, and flatten/unflatten parity;
- exactly centered attack, defensive-vulnerability, month, and league effects;
- vectorized versus independent scalar per-row rate equations;
- vectorized versus scalar complete weighted likelihood within `1e-10`;
- Poisson-thinning identities for converted penalties;
- no mutation of parameters or FeatureSet data;
- finite smooth-saturation/floor behavior under extreme log rates; and
- finite ForwardDiff gradient with central finite-difference spot checks.

The implementation freezes the primitive/deterministic manifest, validates support and dimensions
outside differentiable functions, and keeps the hot likelihood broadcast-vectorized with priors
explicitly excluded.

# Stage 5 findings

**Status:** Turing wrapper and compiled ReverseDiff profile remotely validated at commit `c929bf0`. No sampling, extraction/recombination, or writes occurred.

At commit `9d35609`, adapter parity, the 13 sampled groups/66-parameter manifest, finite log density,
initial gradient comparisons, and three nearby probes passed. A deliberately forced hard-clamp
regime failed fresh-versus-compiled ReverseDiff agreement. This confirms that hard `clamp` records
parameter-dependent control flow in the compiled tape. The shared equation layer now uses
`s(x)=20tanh(x/20)` for all NP-NOG and penalty log-rate bounding; training and future extraction
therefore retain one branch-free equation.

After precomputing observation-only likelihood constants and exact weighted sufficient statistics
for global penalty/own-goal rates, the final production-shaped profile passed all manifest,
finite-density, fresh/compiled ReverseDiff, ForwardDiff, finite-difference, nearby-point, and
smooth-saturation probes:

```text
rows / teams / parameters: 718 / 22 / 66
tape compilation:          1.024 s
median gradient:           0.633 ms
p95 gradient:              0.695 ms
allocations / bytes:       18 / 1,920
classification:            target met (<1 ms)
```

Stage-4 scalar likelihood and gradient parity were rerun after optimization and remained exact to
the configured tolerances.

# Stage 6 findings

**Status:** deterministic multi-chain extraction, OOS recombination, and standard inference validated
remotely at commit `6b59173`; no MCMC, persistence, or writes.

```text
synthetic posterior: 12 iterations × 2 chains = 24 draws
primitive parameters: 66
OOS fixtures: 3 across tournaments 56 and 57
score tensor: 30 × 30 × 24
explicit-convolution vs direct-Poisson maximum difference: 8.67e-17
PPD rows: 201
```

The fixtures included known-known teams in both leagues and target-only East Kilbride against a
history-seen team. East Kilbride received the explicit population fallback. Exact chain-label/order,
iteration×chain stacking, every primitive and transformed shape, draw-wise centering, no chain
mutation, Stage-4 OOS equation parity, team-swap sensitivity, score-tail support, nonnegativity,
per-draw normalization, and ordinary `Predictions.model_inference` all passed. Markets exercised
were 1X2, BTTS, DoubleChance, DrawNoBet, CorrectScore, OverUnder, and AsianHandicap.

**Promotion note:** loader-local dispatch currently returns scalar identity/status/provenance fields
alongside draw vectors. Before generic experiment persistence is used in Stage 7, the latent
serialization contract must explicitly preserve those scalar diagnostics. Real sampler output must
also pass the exact 66-column parameter-manifest check already validated on the pinned synthetic
`MCMCChains` representation.

# Stage 7 findings

**Status:** four-chain mid-size NUTS smoke and real-chain OOS inference passed remotely at commit
`580c82d`. This is a single-split diagnostic artifact, not a leaderboard experiment.

Runtime contract:

```text
Julia threads: 16, pinned to physical cores
BLAS threads: 1
concurrent NUTS chains: 4
warmup / retained per chain: 800 / 800
sampling wall time: approximately 79 s
combined chain: 800 × 80 columns × 4 chains
primitive model parameters: 66
posterior draws used by extraction: 3,200
```

Convergence and sampler diagnostics:

```text
maximum Rhat:       1.00525  (zD[20])
minimum bulk ESS:   1874.98  (kappa_A)
minimum tail ESS:   1853.59  (lambda_og)
divergences:        0
mean acceptance:    0.90120
maximum tree depth: 7
depth-cap hits:     0
BFMI by chain:      0.7641, 0.7819, 0.8698, 0.8085
preferred gate:     passed
hard gate:          passed
```

The exact real-chain primitive manifest, transformed extraction shapes, and centered sums passed.
The three metadata-only fixtures were boundary-held extraction/recombination checks, not genuine walk-forward OOS (the cumulative `target_match_ids` semantic was corrected in Stage 8); this does not invalidate the reported convergence diagnostics. They included
East Kilbride with `:target_only_population_fallback`. All latent vectors had 3,200 finite draws.
Adaptive score tensors had shapes `17×17×3200`, `19×19×3200`, and `17×17×3200`; ordinary
`model_inference` produced 201 PPD rows.

Durable credential-free artifacts are stored remotely at:

```text
/root/BayesianFootball/data/scottish_open_play_rebuild/stage7_midsize_580c82d
```

The directory contains four per-chain checkpoints, `combined_chain.jls`,
`manifest_diagnostics.jls`, and `oos_smoke.jls`. No full temporal experiment, old leaderboard,
or production OOS cache was created.

## Stage 8 implementation status (inventory validated; sampling not run)

`l07_rebuild_full_experiment.jl` and `r07_remote_full_experiment.jl` provide pooled temporal orchestration and the first genuine walk-forward OOS generation. The remote dry run at commit `cde5844` found **38 folds**, not the previously assumed 40: 19 folds for `24/25` and 19 for `25/26`. Their next-step OOS sets contain 710 fixtures in total (360 and 350 respectively), ranging from 10 to 25 fixtures per fold, over one 1,430-match canonical registry snapshot. All season/tournament/step and training-overlap checks passed.

The runner persists nonempty t+1 metadata-only OOS inventory (alignment, non-overlap, IDs/count/hash), uses one registry query over fitted plus OOS IDs, and constructs an explicit rebuild boundary whose history is every observation through `t` and whose held-out target is exactly `t+1`. This translation is required because the generic splitter's cumulative `target_match_ids` do not have the held-out semantics expected by the rebuild feature contract. No Stage 8 sampling, outcome evaluation, or leaderboard has yet run. It uses the native flattened global 38-fold × four-chain queue (up to 16 dynamically filled single-thread tasks), retains per-fold diagnostics/error artifacts, and supports exact atomic split-checkpoint resume.
