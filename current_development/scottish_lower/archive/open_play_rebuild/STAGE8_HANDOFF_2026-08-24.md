# Stage 8 handoff — 2026-08-24

## Current state

Stage 8 is complete. The clean NP-NOG + penalty + own-goal rebuild was trained and inferred over all genuine walk-forward folds for pooled Scottish tournaments 56 and 57 and target seasons `24/25` and `25/26`.

The remotely validated inventory contains **38 folds**, not the previously assumed 40:

- `24/25`: 19 folds
- `25/26`: 19 folds
- true next-step OOS fixtures: 710 total
- canonical registry rows: 1,430

No outcome evaluation, leaderboard, backtest, Layer-2 calibration, or production cache has been generated yet.

## Final Stage 8 result

The native project queue ran four NUTS chains per fold with:

```text
Julia threads:               16, pinned to physical cores
BLAS threads:                1
QueuedNUTS retained/warmup:  800 / 800 per chain
Maximum concurrent tasks:    16
Total folds/chains:          38 / 152
Imported preflight chains:   fold 1 (4 chains)
Remaining queued tasks:      148
Queue elapsed:               41m41s
```

All 38 folds passed hard and preferred convergence gates:

```text
maximum Rhat:       1.0099490750  (fold 21)
minimum bulk ESS:   1196.0719     (fold 34)
minimum tail ESS:   1556.3239     (fold 12)
divergences:        0
depth-cap hits:     0
maximum tree depth: 7
minimum BFMI:       0.6627994
```

Inference output:

```text
OOS matches:             710
PPD rows:                47,570
population fallback:     6 / 1,420 team sides (0.423%)
hard-gate failures:      0
post-processing errors:  0
pending folds:           0
```

## Remote artifacts

All credential-free artifacts are under:

```text
/root/BayesianFootball/data/scottish_open_play_rebuild/stage8_overnight_3ec85cd
```

Important contents:

```text
run_manifest.jls
native_queue_manifest.jls
progress.jls
queued_checkpoints/split_001.jls ... split_038.jls
fold_01/fold_result.jls ... fold_38/fold_result.jls
fold_XX/diagnostics.jls
```

Two `split_001.jls.invalid-*` files remain as audit-preserved artifacts from validator development. The canonical `split_001.jls` is valid. Do not treat the `.invalid-*` copies as active checkpoints.

Prototype `Serialization` artifacts contain types from `RebuildExtractionRecombination`. Load the defining file before deserializing, for example:

```julia
using BayesianFootball, Serialization
include("current_development/scottish_lower/open_play_rebuild/l05_rebuild_extraction_recombination.jl")
using .RebuildExtractionRecombination
x = deserialize("data/scottish_open_play_rebuild/stage8_overnight_3ec85cd/fold_01/fold_result.jls")
```

Trying to deserialize before including the prototype module produces an expected `UndefVarError: RebuildExtractionRecombination not defined`; it does not indicate artifact corruption.

## Important corrections made during Stage 8

### 1. Genuine OOS split semantics

Generic `SplitBoundary.target_match_ids` are cumulative observations through step `t`; they are not held-out OOS IDs. Stage 8 now obtains true OOS fixtures using:

```julia
Data.get_next_matches(ds, (boundary, metadata), splitter)
```

For the rebuild feature contract it constructs an explicit boundary with:

- fitted history: all eligible observations through `t`
- held-out target: only genuine `t+1` fixtures

Stage 7's former “OOS” fixtures were boundary-held extraction/recombination checks rather than genuine next-step walk-forward inference. Its convergence result remains valid.

### 2. Kickoff-time filtration

`match_biweek` groups can overlap in actual dates due to postponements. Stage 8 filters nominal prior-step observations to require:

```text
training kickoff < earliest OOS kickoff
```

Excluded not-yet-played IDs are recorded in checkpoint metadata and fold manifests. This prevents postponed matches from leaking across the prediction cutoff.

### 3. Global immutable registry with fold subsets

The native queue requires one model across all FeatureSets. The rebuild model therefore owns one immutable 1,430-match canonical registry, while its specialized feature adapter validates and passes only the exact fold subset to the strict builder. FeatureSets record both:

- fold subset registry fingerprint
- global model registry fingerprint

The likelihood/team map uses fitted rows only. OOS registry rows provide identity metadata, not outcomes.

### 4. Native project queue

Stage 8 now uses the existing project path:

```julia
Training.train(
    global_model,
    TrainingConfig(
        QueuedNUTSConfig(...),
        Independent(parallel=true, max_concurrent_tasks=16),
        checkpoint_dir,
        false,
    ),
    feature_collection,
)
```

This flattens folds × chains into one queue, so a slow chain does not leave cores waiting for its fold.

### 5. False incomplete report after sampling

The first post-sampling report said folds 2–38 were incomplete even though all 38 checkpoint files existed. Sampling was not lost.

Two validator defects caused this:

1. It compared total chain columns (parameters + internals) with parameter-only names.
2. It compared deserialized project metadata structs by object identity.

Fixes:

- exact primitive manifest validation remains authoritative;
- shape validation checks retained iterations and chain count separately;
- checkpoint metadata is compared structurally by fold index, boundary SHA, and OOS provenance;
- audit-renamed checkpoints can be uniquely validated and recovered.

All 38 checkpoints were recovered and post-processed without resampling.

## Relevant commits

```text
cde5844  initial resumable Stage 8 runner
3ec85cd  pin inventory to 38 actual folds
3b2983d  translate generic boundary to rebuild walk-forward semantics
dcd39d7  fold-1 full-size launch gate passed
aed95c2  native flattened project queue integration
a88f95d  concrete FeatureCollection context construction
75058a3  versioned native queue launch manifest
333b275  strict kickoff-time filtration; sampling run commit
86c4ea1  correct parameter/internal chain shape validation
c3af3df  structural checkpoint resume and recovery
bce898b  record completed Stage 8 results
```

## Recommended next work

Stage 9 should evaluate the new Stage 8 artifacts only. Do not reuse legacy open-play/pxG/wealth leaderboards or Layer-2/portfolio outputs.

Suggested order:

1. Build a read-only Stage 8 artifact loader/aggregator with strict manifest and checkpoint validation.
2. Join the 710 OOS predictions to outcomes by `match_id`, preserving fold/cutoff provenance.
3. Audit season/league/fold coverage, fallback fixtures, quarantines, and duplicate OOS IDs.
4. Evaluate proper scoring rules and calibration for 1X2, totals, BTTS, and score distributions.
5. Compare against explicit baselines (market and simple Poisson) on identical fixtures.
6. Only then create a new leaderboard and consider calibration/backtesting.
7. Define a production-safe schema for prototype model/PPD persistence before graduation to `src/`.

## Session state

- Branch: `design/matchday-layer`
- Completed-result documentation commit: `bce898b` (this handoff note is committed afterward)
- Remote tmux target: `scottish_runner:1.1`
- Remote shell is idle; no Julia sampling process is running.
- Database credentials were never persisted or reported.
