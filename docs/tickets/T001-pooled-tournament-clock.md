# T001 — Pooled tournament groups walk a per-tournament clock

| | |
|---|---|
| **Status** | in progress |
| **Severity** | high — causes genuine train/predict contamination |
| **Area** | `src/Data/preprocessing.jl`, `src/Data/fetchers/sql/matches.jl`, `src/Data/splitting/` |
| **Raised** | 2026-08-25, by model 01 gate 2 in `current_development/scottish_lower/` |
| **Verified on** | Scottish `24/25` (tournaments 56 + 57), 19 folds |

## Summary

`GroupedCVConfig` pools several tournaments into one group and walks forward through
`dynamics_col` (`:match_week` / `:match_biweek` / `:match_month`). Those columns are
**per-tournament dense ranks**, not a shared clock. Pooled tournaments therefore drift
apart, and a fold can end up fitted on matches that kick off at the same time as — or
after — the matches it is predicting.

## Root cause

Three pieces, each correct in isolation:

1. **`src/Data/preprocessing.jl:38-62`** — `add_match_week_column!` assigns `match_week`
   as a dense rank over the distinct weeks **that tournament actually played**, grouped
   by `[:tournament_id, :season]`. Documented behaviour: *"resets for every season and
   tournament."*

2. **`src/Data/fetchers/sql/matches.jl:62-63`** — `match_month = cld(match_week, 4)` and
   `match_biweek = cld(match_week, 2)` inherit that per-tournament counter.

3. **`src/Data/splitting/methods.jl`** — `_process_tournament_group_ids` and
   `get_next_matches` pool tournaments and step through `dynamics_col` **as if the index
   were comparable across them**. Nothing asserts that it is.

Assumption (3) is the defect.

Because the rank counts only weeks that were *played*, any week one tournament plays and
another does not — a midweek round, a single rescheduled fixture — inserts a phantom step
and shifts that tournament's counter **permanently for the rest of the season**.

## Evidence

League Two (57) played a midweek round on **Tuesday 2024-09-17** that League One (56) did
not. `sunday_of_week` gives it its own week, so 57's counter runs one ahead from then on.
(57's week 10 is also a lone straggler on 2024-10-12 — one match is enough.)

Weeks up to 2024-10-20:

```
 56: wk 9 → 2024-10-05,  wk 10 → 2024-10-19
 57: wk 9 → 2024-10-05,  wk 10 → 2024-10-12,  wk 11 → 2024-10-19
```

So on **2024-10-19**: 56 is at `biweek 5`, 57 is at `biweek 6`.

The pooled walk-forward step 5 therefore fits on five League One matches kicking off at
14:00 and 16:00 on 2024-10-19, and predicts five League Two matches kicking off at 14:00
**the same day**. Four fitted matches are simultaneous with the target; one kicks off
after it.

`match_date` is a `Date`, so day resolution cannot see this. Only `match_date` +
`match_hour` separates the 16:00 fitted match from the 14:00 target.

### Second-order consequence

Even where no overlap occurs, a pooled step is not a coherent slate:

- fold 6's held-out block spans **2024-10-19 → 2024-11-02** (15 days);
- biweek sizes diverge — biweek 11 is **3** matches in 56 against **8** in 57.

This makes the model predict up to two weeks ahead with no information update, unevenly
across folds, which depresses measured skill in a way that varies fold to fold.

## Reproduction

On the server (`/root/BayesianFootball`), any branch containing the stream:

```julia
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
using BayesianFootball, DataFrames, Dates

ds = BayesianFootball.Data.load_datastore_cached(BayesianFootball.Data.ScottishLower())
d24 = subset(ds.matches, :season => ByRow(x -> !ismissing(x) && x == "24/25"))

# the drift
g = combine(groupby(d24, [:match_biweek, :tournament_id]),
            :match_date => minimum => :first,
            :match_date => maximum => :last, nrow => :n)
sort!(g, [:match_biweek, :tournament_id])
# biweek 5: 56 runs 10-05..10-19 ; biweek 6: 57 runs 10-19..11-02
```

The gate that catches it is `tp_gate_features` in
`current_development/scottish_lower/01_team_poisson/l03_gates.jl`. Reverting
`tp_build_folds` to use `vcat(boundary.history_match_ids, boundary.target_match_ids)`
without kickoff filtration reproduces the FAIL on fold 6.

## Blast radius

Every segment pooling more than one tournament (`src/Data/fetchers/segments.jl`):

| Segment | Tournaments | Exposed |
|---|---|---|
| `ScottishLower` | 56, 57 | yes — **confirmed contaminated** |
| `ScottishUpper` | 54, 55 | yes — unverified |
| `IrelandAll` | 79, 718 | yes — unverified |
| `SouthKorea` | 3284, 6230 | yes — unverified |
| `Norway` | 5, 6 | yes — unverified |
| `Ireland` (79), `IrelandFirstDivision` (718), `Veikkausliiga` (31) | single | no — one clock cannot drift from itself |

**Part of this ticket is checking the other four.** Drift is certain; whether it produces
an actual same-day overlap is data-dependent and must be measured, not assumed.

Historical note: `current_development/scottish_lower/archive/open_play_rebuild` hit this
symptom in its Stage 8 and mitigated it (correction #2), but attributed it to
postponements. The cause was misdiagnosed until 2026-08-25.

## Proposed fix

Give a pooled group a clock its members share. Two candidates:

**(a) Group-scoped dense rank.** Compute `match_week` per `(tournament_group, season)`
rather than per `(tournament_id, season)`. Minimal change to the existing idea, but the
group has to be known at preprocessing time, which it currently is not — `match_week` is
added in the matches fetcher, before any splitter exists.

**(b) Calendar-anchored index (preferred).** Derive the step index from the calendar —
e.g. weeks elapsed since the season's first fixture — so it is shared by construction and
no dense-rank drift is possible. Blank weeks simply produce empty steps, which the
splitter must then skip rather than mis-number.

(b) is preferred: it removes the drift class entirely rather than making the existing
counter agree by convention, and it makes step width a fixed calendar quantity, which
also fixes the ragged-slate problem.

Whichever is chosen, **the splitter should additionally assert** that within a pooled
group, every fitted kickoff precedes the earliest held-out kickoff. Belt and braces: the
assertion is cheap, and it catches postponements too, which no clock design prevents.

## Acceptance criteria

1. For every pooled segment and every fold: `max(fitted kickoff) < min(held-out kickoff)`,
   using `match_date` **+ `match_hour`**, not date alone.
2. Within a fold, held-out fixtures span a bounded, documented window (state what it is).
3. Step sizes are comparable across pooled tournaments — no 3-vs-8 biweeks.
4. Single-tournament segments (`Ireland`, `IrelandFirstDivision`, `Veikkausliiga`) produce
   **identical folds to today**. If they change, the fix has overreached.
5. The other four pooled segments are measured and the results reported, whether or not
   they were contaminated.
6. `julia --project -e 'using Pkg; Pkg.test()'` passes.
7. A regression test covers the 56/57 `24/25` case specifically: construct the pooled
   folds and assert no same-day fit/predict overlap at 2024-10-19.

## Scope guard — what NOT to do

- **Do not** touch anything under `current_development/scottish_lower/01_team_poisson/`.
  Its `tp_build_folds` kickoff trim is the deliberate local mitigation and is gate-enforced;
  it should keep passing after the fix, and should then become a no-op (0 dropped).
- **Do not** rebuild any leaderboard or rerun any backtest. This fix changes fold
  composition for pooled segments, so past results on those segments become incomparable —
  that is expected and is handled separately.
- **Do not** change `add_match_week_column!`'s documented per-tournament behaviour if other
  callers rely on it. Prefer adding a new column/index over redefining the existing one.

## Reporting back

Update `Status` in [`README.md`](README.md), and record what changed plus the measured
blast radius (criterion 5) in this file under a `## Resolution` heading.
