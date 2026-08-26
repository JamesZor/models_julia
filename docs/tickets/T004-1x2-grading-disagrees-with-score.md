# T004 — `is_winner` disagrees with the recorded score on 3 fixtures

| | |
|---|---|
| **Status** | open |
| **Severity** | low — 4 rows in 6,024; isolated to 2022; no current evaluation window affected |
| **Area** | `ds.odds` grading (`src/Data/` fetch/process for odds), `grade_selection` |
| **Raised** | 2026-08-26, by model 01 gate 6a in `current_development/scottish_lower/` |
| **Verified on** | Scottish 56+57, full odds table |

## Summary

Three fixtures have 1X2 selections whose `is_winner` contradicts `home_score` /
`away_score` in `ds.matches`. On two of them the match was a **2-2 draw and no 1X2
selection is marked as a winner at all** — while BTTS, O/U 2.5 and O/U 3.5 grade correctly
on the very same fixtures, so the score is right and the 1X2 grader disagreed with it.

## Evidence

```
match_id   date         score   1X2 grading
10388068   2022-08-13   2-2     home false, draw false, away false   ← no winner
10388074   2022-08-13   2-2     home false, draw false, away false   ← no winner
10107290   2022-02-08   —       1 row mismatched
```

Same fixtures, other markets, graded correctly:

```
10388068  BTTS btts_yes  true      (2-2: correct)
10388068  OverUnder 2.5  over  true    (4 goals: correct)
10388068  OverUnder 3.5  over  true    (4 goals: correct)
10388068  OverUnder 4.5  under true    (4 goals: correct)
```

Scope over the whole table:

| | count |
|---|---|
| 1X2 rows | 6,024 |
| rows where `is_winner` ≠ score implies | **4** |
| distinct fixtures affected | 3 |
| draws in the table | 509 |
| affected fixtures in the 24/25 evaluation window | **0** |

So this is not a systematic draw-grading failure — 506 of 509 draws grade correctly.

## Related, and worth checking at the same time

`DoubleChance` on those same two 2-2 fixtures marks **zero** winners (`DC_1X` and `DC_X2`
should both be true for a draw). That is consistent with the separately known DC defect —
DC `is_winner` marks 1 of 2 winners and its fair probabilities are halved. DC is
deliberately excluded from the Scottish Lower contract book for that reason.

If DC grading is fixed, fix it with this ticket rather than separately; if it is left
broken, it should be documented as unusable rather than merely unused.

## Reproduction

```julia
sc = select(ds.matches, :match_id, :match_date, :home_score, :away_score)
x  = innerjoin(filter(r -> r.market_name == "1X2", ds.odds), sc, on = :match_id)
x.actual = [r.home_score > r.away_score ? :home :
            r.home_score < r.away_score ? :away : :draw for r in eachrow(x)]
filter(r -> coalesce(r.is_winner, false) != (r.actual == r.selection), x)
```

## Proposed fix

Find why `grade_selection` returned `false` for all three 1X2 selections on a 2-2 result.
Most likely the score was absent or of a different type at grading time and has since been
corrected, in which case grading is stale rather than wrong — check whether re-running the
odds process step reproduces it.

Add an invariant to the odds QA step: **every graded market must have exactly one winning
selection per fixture** (with a declared exception list for markets where that is not true
by construction, e.g. Double Chance and Asian handicap pushes). That invariant is what
caught this, and it belongs in the pipeline rather than in a downstream prototype.

## Acceptance criteria

- [ ] The three fixtures grade consistently with `ds.matches`.
- [ ] A QA check asserts one-winner-per-market across the odds table, with declared
      exceptions.
- [ ] The check is run over all segments, not only Scottish Lower — the same grader serves
      every league.
- [ ] All 403 package tests pass.

## Scope guard

Grading only. Do not change odds fetching, de-vigging, or market definitions.
