# Defects found in `src` by the protocol

Issues in the package itself, found while gating a model. Recorded here because
they affect work outside this stream and should not be rediscovered.

---

## 1. Pooled tournament groups step through a per-tournament clock

**Found:** 2026-08-25, by model 01 gate 2.
**Severity:** real, data-dependent. Caused genuine train/predict contamination in
1 of 19 Scottish `24/25` folds.
**Status:** **RESOLVED** in `src` 2026-08-25 (merged to main as `edd5eba`). Kept here as the record of how it was found.
**Ticket:** [`docs/tickets/T001-pooled-tournament-clock.md`](../../../docs/tickets/T001-pooled-tournament-clock.md)

### Mechanism

`add_match_week_column!` (`src/Data/preprocessing.jl:38-62`) assigns `match_week`
as a dense rank over the distinct weeks **that tournament actually played**,
grouped by `[:tournament_id, :season]`. `match_biweek = cld(match_week, 2)` and
`match_month = cld(match_week, 4)` inherit that (`src/Data/fetchers/sql/matches.jl:62-63`).

Each of those is correct in isolation and documented as such.

`GroupedCVConfig` then pools several tournaments into one group and walks forward
through `dynamics_col`, treating that per-tournament counter **as if it were a
shared clock**. Nothing asserts comparability across the pooled tournaments.

It is not shared. Any week one tournament plays and another does not — a midweek
round, a single rescheduled fixture — inserts a phantom step into that
tournament's counter and shifts it permanently for the rest of the season.

### Worked example (Scottish `24/25`)

League Two played a midweek round on **Tuesday 2024-09-17**; League One did not.
`sunday_of_week` gives it its own week, so from that point 57's counter runs one
ahead of 56's. (57's week 10 is also a lone straggler on 2024-10-12 — the same
effect from a single match.)

| date | 56 | 57 |
|---|---|---|
| 2024-10-19 | week 10 → **biweek 5** | week 11 → **biweek 6** |

So the pooled step 5 fits on five League One matches kicking off at 14:00 and
16:00 on 2024-10-19, then predicts five League Two matches kicking off at 14:00
**the same day**. Four fitted matches are simultaneous with the target; one kicks
off after it.

Day resolution cannot detect this — `match_date` is a `Date`. Only
`match_date` + `match_hour` separates the 16:00 fitted match from the 14:00 target.

### Second-order consequence

Even without an overlap, a pooled step is not a coherent slate. Fold 6's held-out
block spans 2024-10-19 → 2024-11-02 (15 days), and biweek sizes diverge sharply
— biweek 11 is 3 matches in 56 against 8 in 57.

### Blast radius

Every segment pooling more than one tournament:

| Segment | Tournaments | Exposed |
|---|---|---|
| `ScottishLower` | 56, 57 | yes |
| `ScottishUpper` | 54, 55 | yes |
| `IrelandAll` | 79, 718 | yes |
| `SouthKorea` | 3284, 6230 | yes |
| `Norway` | 5, 6 | yes |
| `Ireland`, `IrelandFirstDivision`, `Veikkausliiga` | single | no — one clock cannot drift from itself |

Only 56/57 `24/25` has been checked. Whether the others actually overlap (as
opposed to merely drifting) is unverified.

Note the archived `open_play_rebuild` Stage 8 independently arrived at kickoff
filtration as its correction #2, but attributed it to postponements. Two
independent arrivals at the same mitigation; the cause is now identified.

### Mitigation in this stream

`tp_build_folds` (`01_team_poisson/l03_gates.jl`) builds a kickoff instant from
`match_date + match_hour` and drops any nominally-prior observation not strictly
before the fold's earliest OOS kickoff. Drops are recorded in
`TPFold.dropped_ids`, printed in the fold table, and reported by gate 2. Gate 2
then verifies no violation remains.

Cost on Scottish `24/25`: **5 observations, all in fold 6** (815 → 810 fitted).

### Proper fix, and why it is not being done here

The principled fix is a clock shared by the whole pooled group — either
`match_week` computed per `(group, season)`, or better, a calendar-anchored index
so blank weeks simply produce empty steps and no dense-rank drift is possible.

That changes fold composition for **every pooled segment**, which would break
comparability with every past result on ScottishLower, ScottishUpper, IrelandAll,
SouthKorea and Norway — funnel, APM, smile, staking backtests included.

It therefore needs to be its own scoped piece of work with its own validation, not
a side effect of building model 01. The gate-enforced trim is sufficient and
provably so in the meantime: gate 2 fails loudly if it ever stops being enough.


### Resolution (2026-08-25)

Fixed in `src` under T001. `src/Data/splitting/clock.jl` gives multi-tournament pooled
groups fixed calendar bins (7 / 14 / 28 days) anchored to each season's first kickoff,
plus a temporal-safety assertion on `match_date + match_hour`. Feature construction takes
the splitter (`Features.create_features(splits, ds, model, config)`) so feature time uses
the same effective clock, and observed bins compress to consecutive model states.
Contract: `docs/guides/grouped_splitting.md`.

Verified against this stream's gates on 2026-08-25:

| | before | after |
|---|---|---|
| folds (`24/25`) | 19 | 20 |
| OOS fixtures | 360 | 360 |
| observations dropped by the local trim | 5 (fold 6) | **0** |
| gate 2 | 6/6 after mitigation | **7/7 unmitigated** |

The local kickoff trim in `tp_build_folds` is now a clean no-op and is retained purely as
a defensive check. Fold composition changed, so nothing computed under the old splitter is
comparable — this stream had produced no results yet, which is why fixing it before model
01's leaderboard was the cheaper order.

Note the T001 report measures contamination across cached pooled segments at 50–70% before
the fix. This stream independently measured only 1 fold in 19 for Scottish `24/25` at
`:match_biweek`; the wider figure spans more segments, seasons and bin widths and has not
been reproduced here.
