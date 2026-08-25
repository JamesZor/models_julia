# Model 01 — Findings

Append-only. Every gate run gets a dated entry with the config hash. A result that
is not written here does not exist.

---

## 2026-08-25 — files written, nothing run

Status: **no gate has been executed.** The walkthrough covers gates 0–2; blocks
for gates 3–5 are not yet written.

| File | State |
|---|---|
| `MODEL.md` | complete for the default component set |
| `l01_model.jl` | complete |
| `l02_equations.jl` | complete for the default component set; refuses others |
| `l03_gates.jl` | gates 0–2 |
| `v01_walkthrough.jl` | blocks 0–2 |

### Findings from reading `src` (not yet executed)

1. **`src` extraction applies the hierarchical scale correctly.**
   `components/dynamics/team_level/time_decay.jl:57-61` computes
   `α_scaled = raw_a .* σ_a` then centres, matching the training submodel exactly.
   The audit's defect #2 (dropped `tau`) exists only in the archived prototype that
   reimplemented extraction. This is the evidence for the "extend the package,
   never reimplement it" rule.

2. **`src` `team_map` is name-keyed.** `goals.jl:148` looks up `row.home_team`
   against a `Dict{String,Int}` built in `features/builder.jl:45-46`. The audit's
   defect #1 (integer-keyed double lookup) likewise does not exist in `src`.

3. **Dispersion has a genuine train/predict asymmetry.** Training clamps
   (`dispersion.jl:26-30`: `exp(clamp(log_r, -10, 10))`); extraction does not
   (`dispersion.jl:75-78`: `exp(log_r)`). Benign under `Normal(3.1, 0.4)`, but real.
   Gate 4 must report the observed `|log_r|` range rather than assume it.

4. **Fold semantics are easy to misread.** `create_features` fits on
   `history_match_ids` **+** `target_match_ids` — all observations through step `t`.
   Held-out fixtures are `t+1` via `Data.get_next_matches`. Mistaking
   `target_match_ids` for a test set is what made the archived Stage 7 report a
   non-OOS "OOS" check.

5. **Half-life is unresolved.** `src` defaults to 180 days; the archived rebuild
   used 365. Neither came from Scottish evidence. Provisional until a gate-6 sweep.

### Next

- Verify blocks 0–2 execute on the server.
- Write gates 3–5: equation parity against `l02`, gradient diff, smoke run
  persisted via `src/experiments`, synthetic-chain extraction parity, score matrix.

---

## 2026-08-25 — Gates 0–2 executed, config `54080fde`

Run on the server (`/root/BayesianFootball`, Julia 1.12.4, 16 threads) at commit
`0743c9c`. Contract: pooled `[56, 57]`, target `24/25`, 2 history seasons,
`match_biweek`, `stop_early = true`.

| Gate | Result | Note |
|---|---|---|
| 0. Contract | **PASS 5/5** | 19 folds, 360 OOS fixtures, no duplicates |
| 1. Config | **PASS 4/4** | hash `54080fde`, 5 required features resolve |
| 2. Features | **PASS 6/6** | after the kickoff-filtration fix below |

Fold inventory: 19 folds for `24/25`, fitted growing 720 → 1070, OOS 10–25 per
fold. This matches the archived Stage 8 inventory of 19 folds per target season.

### Defect found and fixed: pooled biweeks are misaligned

Gate 2 failed on first run: **fold 6 violated kickoff filtration.**

Cause is not postponement (the archived Stage 8's diagnosis) but **cross-tournament
biweek misalignment inside a pooled group**. On 2024-10-19:

| tournament | biweek | matches |
|---|---|---|
| 56 | 5 | 5 |
| 57 | 6 | 5 |

The pooled step therefore fitted five League One matches kicking off at 14:00 and
16:00 on 2024-10-19, then predicted five League Two matches kicking off at 14:00
**the same day**. Four fitted matches are simultaneous with the target and one
kicks off after it.

Day resolution cannot see this — `match_date` is a `Date`. The fix builds a
kickoff instant from `match_date` + `match_hour` and drops any nominally-prior
observation whose kickoff is not strictly before the fold's earliest OOS kickoff.

Effect: **5 observations dropped, all in fold 6** (815 → 810 fitted). Every other
fold is untouched. Drops are recorded in `TPFold.dropped_ids`, shown in the fold
table, and reported by gate 2 — correct behaviour, but never silent.

This also covers postponements, which produce the same failure by another route.

### Also fixed

`tp_describe` now reads component priors out by field. The package's compact
`show` (`pregame/display.jl`) printed `GlobalInterception()` with no fields, so
gate 1's "nothing is hidden" claim was not actually being met.

### Other observations

- **OOS team coverage:** 4 of 720 sides (0.56%) hit the population fallback —
  `arbroath` and `inverness-caledonian-thistle`, both legitimately absent from the
  fitted window early in the season. Reported, not enforced.
- **The datastore now contains `26/27`.** Added to `sealed_seasons` alongside
  `25/26` so a widened season list cannot reach it without tripping the assertion.
- The local `.cache/` copy was 69.7 hours stale and refetched from the database
  during this run.

### Next

Gates 3–5: equation parity against `l02`, gradient diff, smoke run persisted via
`src/experiments`, synthetic-chain extraction parity, score matrix.
