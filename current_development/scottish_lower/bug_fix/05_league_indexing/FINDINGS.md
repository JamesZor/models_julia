# Findings — issue 05 league indexing

## Execution status

Not executed in this change. The requested work is syntax/static diagnostics only; no heavy MCMC or remote run was started.

## Hypothesis under test

For a one-column l05 saved artifact, current DataFrame prediction maps tournament 56 to `delta_league[1]` but maps 57 to requested column 2 and then silently uses zero. The compatible interpretation of that artifact pools both ScottishLower tournaments into column 1.

## Expected evidence from `r01_validate_league_indexing.jl`

- exactly one fitted `delta_league` column;
- posterior summaries for delta, its multiplier, and pooled base log-rate;
- tournament counts for the selected OOS fold;
- exact equality for both paths on 56;
- `candidate_open/current_open == exp(delta_league[1])` on 57 for home and away;
- normalized score grids and market summaries;
- an explicit error for an unknown tournament ID.

## Caveats

- The runner is intentionally pinned to the l05 pxG champion and a nonempty OOS fold; artifact availability and selected fold must be recorded after a remote run.
- This isolates league semantics only after composing the issue-01 bridge and l03 tau helper. It does not repair unknown-team fallback, training features, or any l03-l05 production behavior.
- Total goal-rate ratios do not equal `exp(delta)` because penalty conversion and own-goal noise are additive.
