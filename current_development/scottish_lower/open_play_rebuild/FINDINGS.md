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

Both have zero recorded penalties and own goals. Their incident rows include excess `goal/regular` events despite official scores. For match `12477131`, two distinct home regular-goal incidents occur at minute 86; match `11395473` contains an away regular-goal event despite the official away score being zero. Provider fields do not mark these rows rescinded. They remain quarantined rather than being guessed, clipped, or residual-adjusted.

## Decision

Approve the beneficiary own-goal convention for the audited snapshot and proceed to Stage 3 using 718 reconciled rows for this boundary. Preserve the two-match quarantine, snapshot/boundary provenance, and all reconciliation checks in every future split.
