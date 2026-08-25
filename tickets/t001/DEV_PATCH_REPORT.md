# T001 package-interface dev patch report

Implementation loader: [`l01_dev_patch.jl`](l01_dev_patch.jl)  
Readable runner: [`r01_dev_patch.jl`](r01_dev_patch.jl)  
Kaimon project: `/root/BayesianFootball`  
Production `src/` edits: **none**

## What was monkey-patched

The loader installs temporary methods into the loaded package for:

- `Data.create_id_boundaries(::GroupedCVConfig)`;
- `Data.create_data_splits(::GroupedCVConfig)`;
- `Data.get_next_matches(::GroupedSplitMetaData, ::GroupedCVConfig)`;
- the collection-level `Features.create_features` API.

The experiment runner and OOS reconstruction already call those APIs, so this exercises the
proposed behavior at the package boundary without editing `src/`. The patch disappears when the
Julia session restarts.

## Pooled API comparison

All shared seasons in the current Kaimon caches were evaluated with `:match_biweek` and no
historical baseline. Incumbent counts include one terminal empty fold per season; patched folds
only exist when a next observed card exists.

| Segment | Incumbent folds | Incumbent empty | Incumbent contaminated | Patched folds | Patched empty | Patched contaminated | Patched max held-out span |
|---|---:|---:|---:|---:|---:|---:|---:|
| ScottishLower | 106 | 7 | 53 | 105 | 0 | 0 | 11d 19h |
| ScottishUpper | 110 | 7 | 75 | 116 | 0 | 0 | 12d 0h |
| IrelandAll | 97 | 6 | 72 | 102 | 0 | 0 | 13d 1h |
| SouthKorea | 92 | 6 | 44 | 94 | 0 | 0 | 13d 5h |
| Norway | 80 | 6 | 51 | 86 | 0 | 0 | 13d 5h |

Package-API gates:

- zero patched contamination: **PASS**;
- zero patched empty folds: **PASS**;
- every patched biweek shorter than 14 elapsed days: **PASS**.

## Singleton controls

For singleton groups the patched splitter delegates to the incumbent internal implementation and
uses the stored dynamics column. Complete boundary and next-match snapshots were compared.

| Segment | Folds compared | Exactly identical |
|---|---:|---|
| Ireland | 97 | yes |
| IrelandFirstDivision | 94 | yes |
| Veikkausliiga | 62 | yes |

## Feature-time gate

A ScottishLower `24/25` fold with two history seasons was built through the monkey-patched
`Features.create_features` interface.

| Check | Result |
|---|---|
| fitted matches mapped | 820 |
| every shared raw bin maps to one model state | PASS |
| model states are contiguous `1:n_rounds` | PASS |
| `match_id → time_index` survives shuffled DataStore row order | PASS |

This directly addresses the existing positional builder defect without changing or promoting the
GRW model. Models that do not consume dynamic time indices still benefit from safe fold
membership and OOS selection.

## Legacy split-view gate

For pooled ScottishLower `24/25` with two history seasons:

- legacy dynamic split views: 19;
- relational dynamic boundaries: 19;
- effective steps identical: **PASS**;
- fitted match-ID sets identical: **PASS**.

The relational API additionally has its intended history-only baseline fold.

## Scottish local mitigation

The existing scoped `tp_build_folds` mitigation was run unchanged against the monkey-patched
package APIs:

- folds: 20;
- empty OOS folds: 0;
- matches removed by its kickoff trim: **0**.

This is the desired result: after the central fix, the local mitigation remains present but
becomes a no-op.

## Interpretation

The dev patch demonstrates that the proposed solution can be integrated through existing package
interfaces without changing `SplitBoundary` or stored match columns. It fixes the observed pooled
leakage, removes empty prediction folds, aligns feature time by match ID, keeps fixed calendar
windows, and preserves singleton snapshots.

## Remaining before production

- Convert the monkey-patch into maintainable `src/Data/splitting/` and `src/features/` code.
- Freeze `warmup_period`, `end_dynamics`, and `stop_early` behavior with synthetic tests. The dev
  patch deliberately omits terminal folds that cannot predict an observed card.
- Add deterministic tests rather than relying on remote caches.
- Update experiment, diagnostics, and MatchDay call sites to pass the complete splitter contract;
  the monkey patch infers grouped semantics from metadata and the dynamics symbol.
- Run focused tests, one lightweight model/API smoke, the full package suite, and the real-cache
  report again after production integration.

## Conclusion

All dev-patch gates passed. The approach is ready to be translated into production code, subject
to locking the remaining configuration semantics in tests first.
