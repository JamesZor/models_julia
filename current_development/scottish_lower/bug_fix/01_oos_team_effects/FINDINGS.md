# Issue 01 findings — OOS team effects silently disappear

**Experiment artifact:** _paste exact folder basename_
**Git commit:** _paste `git rev-parse HEAD`_
**Datastore/cache date:** _record here_
**Run date:** _record here_
**Status:** Awaiting reproduction output

## 1. Schema evidence

Paste the `schema_report` from block 4:

```text

```

Expected defect signature:

- OOS has `home_team`/`away_team`.
- OOS lacks `home_team_id`/`away_team_id`.
- Feature `team_map` keys are integers.

## 2. Selected-fold mapping

Paste `mapping_summary(selected_mapping)`:

```text

```

## 3. Team-name swap behavioral test

Paste `swap_test`:

```text

```

If all rate changes are exactly zero, the current extractor is invariant to known team identity in
that row, confirming the effects are not active.

## 4. All-fold extent

Paste `all_fold_totals`:

```text

```

Attach or summarize `mapping_all_folds.csv`. Distinguish:

- mappings broken because of schema mismatch;
- genuinely unseen teams absent from a fold's training history.

## 5. Magnitude of omitted effects

Paste both contribution summaries from block 8:

```text
Current-extractor-semantics:

Fitted-model semantics:

```

Record whether `tau_alpha` and `tau_beta` were present.

## 6. Candidate bridge review

Record known and unknown OOS teams for the selected fold:

```text

```

## 7. Phase-2 existing-chain validation

Run `r02_validate_corrected_extraction.jl` against the same pinned artifact/fold used in r01. Paste its
block-10 template here. Do not describe this as a production rerun: it is a local bridge reconstruction.

```text
artifact:
fold:
unknown diagnostics:
old vs mapping-only (issue 01):
mapping-only score-matrix difference:
mapping-only vs fitted (deferred issue 02/extraction parity):
mapping-only swap sensitivity:
```

Acceptance checklist:

- [ ] `assert_bridge_invariants!` passed (no posterior-column permutation).
- [ ] Known OOS names resolve to their existing integer posterior columns.
- [ ] Only genuinely unfitted/unseen names are in `unknown_names` and return `-1`.
- [ ] Old vs mapping-only differs for a known-known fixture; this is the isolated issue-01 result.
- [ ] Mapping-only swap of known names changes open and/or penalty latents.
- [ ] Mapping-only vs fitted is recorded separately as tau/clamp/floor/league extraction-parity evidence.

Do not attribute a mapping-only-vs-fitted difference solely to the name bridge.

## Conclusion

- [ ] Issue reproduced.
- [ ] Issue measured across all folds.
- [ ] Candidate name-to-training-index bridge maps established teams.
- [ ] Genuinely unseen-team behavior identified separately.
- [ ] Evidence is sufficient to begin an implementation notebook.

## Proposed implementation decision

_To complete after reviewing output._

Recommended default is to store a canonical `team_name_to_index::Dict{String,Int}` in the custom
FeatureSet and use it directly in extraction. For compatibility with saved chains, its index values
must exactly match the integer-ID map used during fitting. Unknown teams should trigger a count/warning
and use an explicit population effect—not silently pass through the same branch as a schema failure.
