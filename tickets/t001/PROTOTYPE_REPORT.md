# T001 prototype comparison report

Prototype: [`prototype.jl`](prototype.jl)  
Kaimon checkout: `/root/BayesianFootball`  
Branch/commit tested: `fix/t001-pooled-tournament-clock` at `d6fa452`  
Clock compared: `:match_biweek`

## Executive result

The calendar-anchored pooled clock removed every observed fit/predict overlap in this
prototype run, produced no empty folds, and bounded every held-out block to less than 14 days.
The three named singleton controls were exactly unchanged.

This validates the design direction, but it is not yet a production fix: the prototype does
not alter `src/`, does not yet align model feature-time indexing, and has not run the package
test suite.

## Methods compared

### Incumbent

Each tournament independently dense-ranks only the weeks in which it played. The pooled
splitter trains through local step `t` and predicts rows labelled local step `t + 1`.
Identical labels can therefore refer to different calendar intervals.

### Proposed prototype

For each pooled `(tournament group, season)`:

1. Anchor week 1 at the Sunday-ending week containing the group's earliest fixture.
2. Compute elapsed calendar weeks from that shared anchor.
3. Map two consecutive calendar weeks to one biweek.
4. Train on bins before the held-out bin.
5. Move to the next **observed** bin. An unplayed bin keeps its calendar position but does
   not create an empty fold or empty prediction job.
6. Keep singleton groups on the incumbent path without recomputing their clock.

A proposed biweek is therefore a fixed calendar window. Fixture counts can legitimately differ
between tournaments, but the dates represented by the bin cannot drift.

## All pooled segments

The comparison covers every season shared by both tournaments in each current cache. It counts
target-season transitions only; it excludes the production splitter's history-only baseline
folds.

| Segment | Old folds | Old contaminated | Old maximum held-out span | New folds | New contaminated | New maximum held-out span |
|---|---:|---:|---:|---:|---:|---:|
| ScottishLower | 99 | 53 | 77d 0h | 105 | 0 | 11d 19h |
| ScottishUpper | 103 | 75 | 87d 22h | 116 | 0 | 12d 0h |
| IrelandAll | 91 | 72 | 29d 22h | 102 | 0 | 13d 1h |
| SouthKorea | 86 | 44 | 48d 0h | 94 | 0 | 13d 5h |
| Norway | 74 | 51 | 60d 21h | 86 | 0 | 13d 5h |
| **Total** | **453** | **295** | — | **503** | **0** | **< 14d everywhere** |

Neither method happened to generate an empty evaluated fold in this cache. The proposed method
explicitly selects the next observed calendar bin, so a wholly blank bin would also be skipped.

The increased pooled fold count (453 → 503, about 11%) is expected: the old “biweek” means two
*tournament-playing weeks*, while the proposed biweek means two fixed calendar weeks. More
calendar windows contain at least one match across a pooled group. This increases experiment
runtime and changes pooled evaluation composition; the ticket already declares old pooled
results incomparable after the fix.

## Exact Scottish 56/57 example

The ticket's key date is 2024-10-19.

### Incumbent fold

| | Value |
|---|---|
| transition | local biweek 5 → 6 |
| last fitted kickoff | **2024-10-19 16:00** |
| first held-out kickoff | **2024-10-19 14:00** |
| last held-out kickoff | 2024-11-02 15:00 |
| held-out fixtures | 20 |
| safety result | **FAIL** — training reaches two hours into the prediction slate |

Tournament composition:

| Tournament | Fixtures | First | Last |
|---|---:|---|---|
| 56 | 10 | 2024-10-26 14:00 | 2024-11-02 15:00 |
| 57 | 10 | 2024-10-19 14:00 | 2024-11-02 15:00 |

The equal fixture counts are misleading: the two tournaments' “biweek 6” starts on different
dates, so the pooled block spans over two calendar weeks and overlaps its own fitted data.

### Proposed fold containing 2024-10-19

| | Value |
|---|---|
| transition | shared calendar biweek 5 → 6 |
| last fitted kickoff | **2024-10-05 14:00** |
| first held-out kickoff | **2024-10-12 12:00** |
| last held-out kickoff | **2024-10-19 16:00** |
| held-out fixtures | 11 |
| safety result | **PASS** |

Tournament composition:

| Tournament | Fixtures | First | Last |
|---|---:|---|---|
| 56 | 5 | 2024-10-19 14:00 | 2024-10-19 16:00 |
| 57 | 6 | 2024-10-12 12:00 | 2024-10-19 14:00 |

Both tournaments now occupy the same fixed calendar window. The counts differ 5 versus 6 because
League Two played an additional fixture date; that is genuine scheduling, not clock drift.

## Regression-risk checks

### Passed by the prototype

- **Temporal safety:** 0 contaminated proposed transitions across all five pooled segments.
- **Bounded windows:** every proposed biweek spans less than 14 elapsed days.
- **No empty dynamics jobs:** 0 empty proposed folds; absent calendar bins are skipped.
- **Singleton preservation:** exact incumbent/proposed fold equality for:

| Segment | Seasons | Folds compared | Exactly identical |
|---|---:|---:|---|
| Ireland (79) | 6 | 91 | yes |
| IrelandFirstDivision (718) | 6 | 88 | yes |
| Veikkausliiga (31) | 6 | 56 | yes |

### Still to resolve before production

1. `src/features/builder.jl` currently groups model time using the stored tournament-local
   `dynamics_col`. Production code must use the same effective pooled bins as the splitter, or
   explicitly compress observed shared bins to consecutive model indices.
2. Add a strict production assertion using `match_date + match_hour`; the calendar clock should
   prevent overlap, while the assertion catches bad or exceptional source dates.
3. Add deterministic synthetic tests for the 2024-10-19 case, a wholly blank pooled bin,
   simultaneous kickoffs, multiple seasons/groups, and singleton golden snapshots.
4. Decide and document `end_dynamics`/`stop_early` semantics against non-contiguous calendar labels.
5. Run focused tests, the real-cache report again through production APIs, and the full package
   suite.

## Conclusion

The prototype strongly supports a shared calendar clock with empty bins skipped at the fold
level. It fixes both demonstrated failure modes: temporal leakage and ragged, drifting held-out
windows. No regression was observed in the singleton controls, but production integration and
regression tests remain required before T001 can be closed.
