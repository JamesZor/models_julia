# Issue 01 findings — OOS team effects silently disappear

**Experiment artifact:** `recomb_pxg_wealth_integrated_hl365_hs2_20260823_075833`
**Validation commit:** `0951072`
**Datastore cache age at run:** 48.4 hours
**Run date:** 2026-08-24
**Status:** Defect confirmed; bridge validated; permanent implementation blocked on coupled tau fix

## 1. Schema evidence

Paste the `schema_report` from block 4:

```text
has_home_team = true
has_home_team_id = false
team_map_type = Dict{Int64,Int64}
n_training_teams = 22
```

Expected defect signature:

- OOS has `home_team`/`away_team`.
- OOS lacks `home_team_id`/`away_team_id`.
- Feature `team_map` keys are integers.

## 2. Selected-fold mapping

Fold 39 mapped none of its 11 fixtures under the legacy lookup and all 11 under the bridge:

```text
(matches=11, current_known=0, candidate_known=11,
 current_known_pct=0.0, candidate_known_pct=100.0)
```

## 3. Team-name swap behavioral test

The legacy extractor was exactly invariant when the known home/away team names were swapped:

```text
max_abs_open_home_change = 0.0
max_abs_open_away_change = 0.0
max_abs_total_home_change = 0.0
max_abs_total_away_change = 0.0
```

If all rate changes are exactly zero, the current extractor is invariant to known team identity in
that row, confirming the effects are not active.

## 4. All-fold extent

```text
folds = 38
matches = 710
current_known_matches = 0
candidate_known_matches = 607
genuinely_unknown_sides = 107
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
artifact: recomb_pxg_wealth_integrated_hl365_hs2_20260823_075833
fold: 39 (11 OOS matches, 3,600 draws)
unknown diagnostics: 0 sides / 0 matches
old mean open rates: 1.3242 home, 1.0325 away
mapping-only mean open rates: 0.4276 home, 1.0441 away
mapping-only score-matrix max difference: 0.62187
fitted mean open rates: 1.0808 home, 0.9632 away
mapping-only vs fitted score-matrix max difference: 0.58859
mapping-only swap max open change: 19.4329 home, 15.3880 away
mapping-only swap max penalty change: 0.2440 home, 0.1921 away
all score tensors retained mass 1.0
```

Acceptance checklist:

- [x] `assert_bridge_invariants!` passed (no posterior-column permutation).
- [x] Known OOS names resolve to their existing integer posterior columns.
- [x] Only genuinely unfitted/unseen names are in `unknown_names` and return `-1`.
- [x] Old vs mapping-only differs for a known-known fixture; this is the isolated issue-01 result.
- [x] Mapping-only swap of known names changes open and penalty latents.
- [x] Mapping-only vs fitted is recorded separately as tau/clamp/floor/league extraction-parity evidence.

Do not attribute a mapping-only-vs-fitted difference solely to the name bridge.

## Conclusion

- [x] Issue reproduced.
- [x] Issue measured across all folds.
- [x] Candidate name-to-training-index bridge maps established teams.
- [x] Genuinely unseen-team behavior identified separately.
- [x] Evidence is sufficient to begin an implementation notebook.

## Proposed implementation decision

Store a canonical `team_name_to_index::Dict{String,Int}` in the custom FeatureSet and use it directly
in extraction. For compatibility with saved chains, its values must come from the existing integer-ID
map so parameter columns cannot be permuted. Unknown teams must be reported and use an explicit
population effect.

**Coupled blocker:** mapping-only activation exposes the separate missing-`tau_alpha`/`tau_beta`
extraction defect. Raw unscaled team effects generated extreme draw-level changes (up to roughly
10–15 goals in the selected comparison and 15–19 under a name swap). Therefore the permanent OOS
method must not ship the identity bridge alone: it must reconstruct the fitted hierarchical scales at
the same time. The notebook keeps the two effects separate analytically, but deployment must correct
both before rebuilding latents.
