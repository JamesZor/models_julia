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

---

## 2026-08-25 — Gates 0–2 re-run on the T001 shared calendar clock, config `54080fde`

T001 was fixed in `src` and merged to main (`edd5eba`): pooled groups now step through
fixed calendar bins anchored to each season's first kickoff, with a temporal-safety
assertion, and feature construction takes the splitter so feature time uses the same
effective clock.

Re-run at `ae8b3a6` after a REPL restart (new `clock.jl` in the module, so Revise cannot
pick it up).

| | before T001 | after T001 |
|---|---|---|
| folds (`24/25`) | 19 | **20** |
| OOS fixtures | 360 | 360 |
| dropped by local trim | 5 (all fold 6) | **0** |
| Gate 0 | 5/5 | **5/5** |
| Gate 1 | 4/4 | **4/4** |
| Gate 2 | 6/6 *after mitigation* | **7/7 unmitigated** |

The local kickoff trim is now a clean no-op, kept as a defensive check.

### What changed in the folds

Fold count moved 19 → 20 and composition changed, because calendar bins are not
dense-rank bins. Total OOS coverage is unchanged at 360 fixtures, so no matches were
lost or double-counted.

Fit→predict gaps are much tighter. Previously fold 6's held-out block spanned 15 days;
now the largest gap between last fitted and first held-out kickoff is a normal
inter-round interval (e.g. fold 10: fitted to 2024-11-30, predicts from 2024-12-03).

### Code change required

Gate 2 moved from `Features.create_features(boundary, ds, model, dynamics_col)` to the
collection overload `Features.create_features(splits, ds, model, splitter)`. The
symbol-only overload skips `_align_splitter_time!`, so per-fold construction would have
produced time indices off the pooled effective clock — silently, since it still returns a
valid FeatureSet. Grouped paths must pass the complete splitter object.

Added a gate for the new contiguity promise: `time_indices` must be `1..K` with no gaps.
Passes in all 20 folds.

### Observation worth carrying forward

Calendar bins are ragged in a new way: OOS block sizes now range from **2** (fold 9) to
**24** (fold 16) fixtures. That is honest — thin bins are real calendar gaps, e.g.
international breaks — but a fold predicting 2 matches contributes very little evidence
while still costing a full fit. Worth remembering when gate 6 weights folds; an unweighted
mean over folds would over-weight fold 9 by a factor of ~12 relative to its information.

---

## 2026-08-25 — Gate 3a/3b executed, config `54080fde`

Fold 1 (season-opening, 720 rows, 23 teams, 51 parameters). Gate 3c (smoke) not
yet run — it samples, and is James's to launch.

| Gate | Result |
|---|---|
| 3a. Equation parity | **PASS 3/3** |
| 3b. Gradient health | **PASS 7/7** |

### 3a — the load-bearing result

```
log density parity (Turing vs l02)   max |Δ| = 0.000e+00 over 3 prior draws
sampled-site manifest                7 sites, as documented
parameter count                      51 = 5 scalars + 2 x 23 team effects
```

**The difference is exactly zero, not merely within tolerance.** DynamicPPL's log
density and the independent `l02_equations.jl` implementation — written from
MODEL.md rather than from the engine — agree bit-for-bit at three independent
prior draws (e.g. `-1140.8620713133826` both ways).

So for this configuration the fitted model and the documented model are the same
object. Gate 4 can now use `l02` as its reference for the priced model.

This also settles the open question flagged in `l02_equations.jl`:
`DynamicPPL.VarInfo(model)` is **unlinked**, so the log density is in the original
constrained space and no Jacobian correction is needed for the Gamma-distributed
scales. Had the space been linked, parity would have failed by exactly those
log-Jacobian terms.

### 3b — gradients

```
compiled tape == fresh ReverseDiff    relerr = 0.000e+00
ReverseDiff == ForwardDiff            relerr = 3.387e-15
finite differences agree              max |Δ| = 2.379e-07
static tape safe at perturbed points  max relerr = 0.000e+00 over 3 probes
compiled gradient latency             median 1.448 ms (compile 6.70 s)
```

All four routes agree. The static-tape probe is clean, which is what licenses
using a compiled tape at all: the model has no data-dependent branch that would
freeze the tape on one path.

**Latency is 1.448 ms against the guide's < 1 ms target** — inside "acceptable"
(< 3 ms) but not at target. That figure is fold 1's 720 rows; later folds carry
~1060, so expect roughly 2.1 ms there. Reported, not gated: a slow model is a
cost, a wrong one is a bug. Worth revisiting only if the full grid proves painful.

### Sampler initialisation was a hidden default

`QueuedNUTSConfig` defaults to `UniformInit(-0.001, 0.001)` — far tighter than the
repository's own presets (`±0.1`), and it starts the positive-constrained scales
`σ_a`/`σ_d` almost on their support boundary. Now stated explicitly in the
contract as `init_range = 0.1`, since gate 1's premise is that no configuration is
invisible.

## 2026-08-26 — Why the gradient is slow (diagnostic, raised as T002)

Profiled `tp_grad_profile` after the latency number looked poor against the
archived benchmark table. The answer is not the maths.

**The tape holds 35,421 instructions for 51 parameters — 35,387 of them scalar,
24.6 per observation.** Self time contains no maths kernel at all: `getproperty`
10.6%, `CallWrapper` 9.7%, `pull_value!` 7.0%, `setproperty!` 6.8%,
`increment_deriv!` 3.4%. `loggamma` does not appear in the top 12. The cost is
per-instruction bookkeeping for 35k tiny nodes.

Two compounding causes, both in our code, neither in the data:

| stage | tape nodes/obs | attributable to |
|---|---|---|
| λ only (`view` + `exp`) | 2.0 | engine `view` |
| + `Poisson` logpdf | 5.0 | — |
| + stdlib `NegativeBinomial` | 11.9 | — |
| + `RobustNegativeBinomial` | 21.0 | our distribution |

1. `view(dyn.α, idx)` on a `TrackedArray` yields `SubArray{TrackedReal}`, which
   forces every downstream broadcast onto the scalar path. Isolated: `view` = 1439
   instructions, `getindex` = **3**, same value. This is the **opposite** of Rule 4
   in `docs/turing_ad_performance_guide.md`, which the engines follow.
2. `RobustNegativeBinomial` costs ~1.9x stdlib `NegativeBinomial` on that path:
   `log(r+μ)` twice per observation, and `loggamma(r)`/`log(r)` recomputed 1,440
   times per gradient despite `r` being a single global scalar.

Switching to `getindex` collapses the tape 15,120 → **5** instructions for the
identical value, and then the gradient **throws** `InexactError` from
`negative_binomial.jl:79` (`Int(k)` on a ForwardDiff dual). So the slow path is
currently masking a crash on the fast one.

Raised as [T002](../../../docs/tickets/T002-scalar-taped-likelihood.md). Not fixed
here: it touches 10 of 28 engines and a shared distribution, and gate 3a must stay
bit-identical through any fix.

**Comparability note.** The archived table
(`archive/open_play/r04_benchmark_ad_recomb.jl`) uses `@belapsed`, which reports the
**minimum**; gate 3b reports the **median**. Our min is 1.140 ms against a median of
1.150 ms, so the gap to their 0.572 ms NegBin is real and not a statistic artifact —
their `_negbin_vector_loglik` was hand-vectorised.

### The AD backend is not configurable

`QueuedNUTSConfig` has **no `adtype` field**. Both `run_sampler` methods hardcode
`AutoReverseDiff(compile=true)` (`src/samplers/engines/nuts.jl:101,120`). Gate 3b's
measurement is therefore representative of real sampling, but the contract cannot
override it. Asserted from source by `tp_ad_backend_matches_src`.

### Next

Gate 3c: run the smoke (block 7 of the walkthrough) with `-t 16`, ThreadPinning
and single-threaded BLAS. It persists a chain to
`data/scottish_lower/01_team_poisson/54080fde/`, which gate 4 then reloads.

T002 is a cost ticket, not a blocker — the posterior is correct, so gates 3c
onward can proceed at current speed.
