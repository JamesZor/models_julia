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
