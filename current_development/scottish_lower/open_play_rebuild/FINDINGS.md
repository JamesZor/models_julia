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

**Status:** first remote AD run found hard-clamp compiled-tape invalidation; branch-free smooth saturation is implemented and pending rerun. No sampling, extraction/recombination, or writes occurred.

At commit `9d35609`, adapter parity, the 13 sampled groups/66-parameter manifest, finite log density,
initial gradient comparisons, and three nearby probes passed. A deliberately forced hard-clamp
regime failed fresh-versus-compiled ReverseDiff agreement. This confirms that hard `clamp` records
parameter-dependent control flow in the compiled tape. The shared equation layer now uses
`s(x)=20tanh(x/20)` for all NP-NOG and penalty log-rate bounding; training and future extraction
therefore retain one branch-free equation.
