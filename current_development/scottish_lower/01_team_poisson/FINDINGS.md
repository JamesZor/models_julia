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

## 2026-08-26 — Gate 3c PASS 6/6, gate 4 PASS 11/13

### Gate 3c — the posterior is clean

Fold 1 (season opener, 720 matches), 4 chains x 500/500, 53.5 s wall.

| | |
|---|---|
| Rhat | max 1.00806 (threshold 1.01) |
| ESS | min bulk 606, min tail 711 (threshold 400) |
| divergences | 0 |
| tree depth | max 6 against a cap of 10 |
| BFMI | min 0.743 (threshold 0.30) |

Max tree depth 6 against the cap of 10 means the sampler never truncated a
trajectory — the geometry is easy and the non-centred parameterisation is doing
its job. Artifact: `data/scottish_lower/01_team_poisson/54080fde/tp01_smoke_54080fde_20260826_104111`.

Worth noting for the full grid: two chains found ε = 0.2 and two found ε = 0.025,
an 8x spread. Rhat 1.008 says they still agree on the posterior, so this is warmup
path dependence rather than multimodality — but if the grid ever shows Rhat drift,
this is the first place to look.

### Gate 4a — the priced model IS the fitted model

Fabricated an 8-draw / 2-chain synthetic chain with known parameters, priced it
through the package's own `extract_parameters`, compared against `l02`:

```
λ parity vs l02      max |Δλ| = 2.220e-16   (one ulp, over 8 draws x 6 fixtures x 2 sides)
r parity vs l02      max |Δr| = 0.000e+00
draws not collapsed  8 distinct λ_h across 8 draws
```

Two chains rather than one is deliberate. Every extractor flattens with
`vec(Array(chain[name]))` — column-major over `(n_iter, n_chains)` — and nothing
checks that separate components agree on that convention. With a single chain,
row-major and column-major coincide and a mixing bug would pass unnoticed.

This closes the audit's worst finding from the other direction: `src`'s extraction
keys `team_map` by NAME and applies the hierarchical scale correctly. The
re-implemented prototype was the broken one.

### Gate 4b — plumbing, on the real chain loaded from disk

```
OOS fixtures priced          20 rows priced, 20 OOS fixtures at t+1
match ids match the OOS set  0 missing, 0 unexpected
posterior depth preserved    2000 draws per fixture (chain is 500 x 4)
λ finite and positive        80000 values, range [0.549, 3.835]
r finite and positive        range [6.17, 91.97] — clamp at exp(±10) never binds
λ plausible for the league   median 1.358 goals/side (league average ~1.3)
```

The dispersion clamp asymmetry flagged in MODEL.md is now settled empirically: `r`
lands in [6.17, 91.97], nowhere near `exp(±10)`, so training clamping and
extraction not clamping cannot disagree under this prior.

### Gate 4c — FAIL 1/2: unmapped teams lose home advantage

```
[FAIL]  unmapped team keeps global home advantage
        γ_global DROPPED: λ_h is 0.849x the population value
[PASS]  month index inert for this config    max |Δλ_h| = 0.000e+00
```

`goals.jl:158` substitutes `zeros` for γ when a team is missing from `team_map`.

Under `GlobalHomeAdvantage` **γ has no team dimension**: one sampled site
`ha.γ_global`, which `extract_home_advantage` then `repeat`s into n_teams identical
columns. Verified on the gate 3c chain — `ha_mat` is 2000x23 with exactly one
distinct column, γ mean 0.1469. So `ha_mat[:, h_idx]` returns the same vector for
every index, the `h_idx > 0` guard guards nothing, and its else-branch discards a
value that was never team-specific and never missing.

Zero *is* the correct substitute for α and β, where the guard does real work: the
zero-sum constraint makes zero the population mean by construction. On the real
posterior the home side is priced at 0.863x.

Fires on 2 of 360 fixtures (0.56%) in Scottish 24/25 — `arbroath` and
`inverness-caledonian-thistle`, sides appearing in a fold's OOS window before its
fitted history. Rare, but concentrated at season boundaries where the model is
already weakest, and silent. 28 call sites across the engines.

Raised as [T003](../../../docs/tickets/T003-home-advantage-population-fallback.md).

## 2026-08-26 — Gate 5 PASS 16/16

### 5a — dispatch resolves to the intended pricer

`DynamicGoalsTimeDecayModel <: AbstractTimeDecayTeamModel <: AbstractNegBinModel`
routes to `negativebinomial.jl:48`, asserted **by resolved method file** rather
than by the call merely succeeding. That is the point: engines are selected by
supertype and by `Union` membership, and one omitted from a Union does not error
at definition time — it falls through to a default meant for another likelihood.
`extract_params` takes the separate `:r_h`/`:r_a` route, not the shared `:r` one.

### 5b — the grid is the documented distribution

```
grid parity vs stock NegativeBinomials    max |ΔP| = 5.121e-15 over 125 grids
orientation [home, away]                  E[home] 2.0895 vs λ_h 2.0900
                                          E[away] 1.4159 vs λ_a 1.4160
                                          marginals separated by 0.6740
moment vs truncated expectation           |Δ| = 1.066e-14
```

Orientation is tested RELATIVE to the separation between the marginals, not
against an absolute tolerance. A transposed grid is the nastiest bug available at
this stage — it yields perfectly well-formed probabilities that are simply the
wrong way round, and nothing downstream can detect it. It is catchable here only
because γ > 0 makes the home and away marginals genuinely differ.

**Truncation at `max_goals = 12` is defensible, and now measured rather than
assumed.** Nothing in `src` normalises the NegBin grid (only `frank_copula.jl:51`
does), so every market sums to `1 - truncation_mass`. On the highest-scoring
fixture (λ_tot 3.32), widening the grid moves the most-affected price by:

| max_goals | worst-draw mass | P(over 3.5) |
|---|---|---|
| 12 | 4.05e-4 | 0.42005309 |
| 14 | 4.14e-5 | 0.42006291 |
| 16 | 4.27e-6 | 0.42006352 |
| 20 | 5.19e-8 | 0.42006356 |

**1.05e-5** between 12 and 20 — three orders of magnitude below any edge worth
betting. Keeping 12. A higher-scoring league inheriting this default must re-check:
the quantity that matters is the price shift, not the discarded mass.

### 5c — markets partition the same grid

```
1X2   |home+draw+away - sum(grid)|   5.551e-16
BTTS  |yes+no - sum(grid)|           9.992e-16
O/U   |over+under - sum(grid)|       9.992e-16   across [0.5, 1.5, 2.5, 3.5]
cross-family disagreement            1.332e-15
O/U vs direct cell sum              0.000e+00
under(line) monotone                 yes
```

Asserting "1X2 sums to 1" would assert something false, since nothing normalises.
Internal consistency is both true and stronger: a pricer that dropped a cell,
double-counted the diagonal, or used `>=` for `>` would break consistency while
still summing to something near 1. The contract's O/U lines are all half-lines, so
no push mass exists — `over_under.jl:31-36` silently discards exact-integer
totals, which would break this identity on an integer line, so the gate refuses
one.

### Two gate bugs found in my own reference, not in src

Both worth recording, because both initially looked like defects:

1. Gating truncation on raw discarded mass at 1e-6. Wrong quantity — raw mass is
   dominated by a few extreme draws, while what matters is the price shift.
2. The moment identity omitted the away dimension. The grid is truncated in BOTH
   directions, so `E[home]` equals the truncated home first moment scaled by the
   away marginal's RETAINED mass. Dropping that factor left a residual of
   λ_h × away_tail ≈ 2.6e-6 that looked exactly like a real disagreement.

### First look at prices

| λ_h | λ_a | home | draw | away | over 2.5 | BTTS |
|---|---|---|---|---|---|---|
| 1.733 | 1.590 | 0.417 | 0.222 | 0.361 | 0.629 | 0.624 |
| 1.619 | 1.236 | 0.461 | 0.241 | 0.299 | 0.533 | 0.543 |
| 1.561 | 1.140 | 0.467 | 0.248 | 0.285 | 0.498 | 0.512 |

All in the right neighbourhood. One thing to carry into gate 6: mean draw is
**~0.243** against an empirical Scottish L1/L2 rate of ~0.25-0.27. Conditionally
independent goals with no Dixon-Coles term systematically under-predict draws, and
this is what that looks like. Test it directly rather than assuming it is noise —
it is the strongest available argument for a DC or copula variant.

## 2026-08-26 — Full 24/25 grid sampled, convergence PASS 7/7

`tp01_grid_54080fde`: 20 folds x 4 chains, 800 warmup + 800 samples, 64,000
post-warmup draws, 360 OOS fixtures extracted.

```
Rhat                      max 1.00749 (fold 3) — 20/20 folds under 1.01
effective sample size     min bulk 1012 (fold 7), min tail 1180
divergences               8 = 0.0125% of 64000 draws, in folds [8, 15, 17]
divergences not a funnel  σ at divergent draws is 0.80-1.97x the bulk mean
tree depth                max 8, 0 hits at cap 10
BFMI                      min 0.663 (fold 8)
```

### The convergence gate was only looking at fold 1

Found when wiring the grid into the walkthrough. `tp_gate_convergence` asserted
`length(chains) == 1` and then summarised `first(chains)` — on a 20-fold grid it
would have reported PASS while **never examining folds 2-20**. Now every fold is
walked, the worst reported, and the offending fold named. Re-run against the smoke
chain it reproduces the identical numbers, so the rewrite is behaviour-neutral on
the case that already passed.

This is the failure mode the protocol exists to catch, and it was in the protocol's
own code.

### The 8 divergences are integrator noise, not a funnel

Diagnosed before deciding what to do about them:

| fold | divs | σ_a div / bulk | σ_d div / bulk | mean accept |
|---|---|---|---|---|
| 8 | 1 | 0.1233 / 0.1463 | 0.2108 / **0.1069** | 0.729 |
| 15 | 6 | 0.0911 / 0.0960 | 0.0699 / 0.0878 | 0.781 |
| 17 | 1 | 0.1119 / 0.1068 | 0.1125 / 0.1110 | 0.721 |

A hierarchical funnel puts divergent draws at **small** σ. These are at 0.80-1.97x
the bulk mean — fold 8's is at *high* σ_d, the opposite signature — and the σ_a 5th
percentile (0.024-0.046) shows the posterior is not pressing against zero. With
BFMI 0.663 and no depth saturation, the geometry is fine.

Achieved acceptance is 0.72-0.78 against a 0.65 target, so the integrator is
already taking smaller steps than asked.

**Protocol amended** rather than the model re-run: divergences now gate on rate
(≤0.1%) AND absence of small-σ clustering, instead of on count == 0. Reasoning in
[PROTOCOL.md](../docs/PROTOCOL.md) § Amendment 2026-08-26. Re-running at δ = 0.9
was considered and rejected: tree depth is already at 8 against a cap of 10, so
raising δ would push toward saturation — trading 8 divergences for a worse and
slower problem — and would not have changed a posterior that is not biased.

## 2026-08-26 — Gate 6 PASS 26/26. The model is competitive with the closing line.

360 fixtures, 4,680 model prices, scored against de-vigged Bet365 close (360/360
fixtures) and Betfair close (320/360).

### Headline: it is not worse than the market

Per-line log loss, model minus de-vigged Bet365 close. Negative = model better.

| line | n | base rate | Δ log loss | t |
|---|---|---|---|---|
| O/U 0.5 | 357 | 0.927 | **-0.0083** | -1.82 |
| 1X2 home | 360 | 0.456 | **-0.0068** | -0.77 |
| O/U 3.5 | 357 | 0.280 | **-0.0030** | -0.79 |
| O/U 2.5 | 359 | 0.490 | **-0.0021** | -0.61 |
| 1X2 draw | 360 | 0.233 | **-0.0014** | -0.54 |
| O/U 1.5 | 357 | 0.756 | 0.0000 | +0.01 |
| BTTS | 359 | 0.490 | +0.0021 | +0.65 |
| 1X2 away | 360 | 0.311 | +0.0051 | +0.58 |

Better on 6 of 8 lines, nothing significant either way, worst line +0.0051 against a
+0.02 threshold. Betfair close reproduces the same ordering on its 320 fixtures.

**This is the expected shape, not a triumph.** A structural model level with the
closing line on proper scoring is what a model that might have CLV looks like; gate 7
decides whether it does.

### The dispersion finding, now measured

`sd(p_model) / sd(p_market) = 0.55`, averaged across lines. The model differentiates
fixtures at roughly **half** the market's spread — confirming the prior observation,
now on 360 fixtures across 13 lines rather than as an impression.

That has a direct consequence for gate 7 worth stating in advance: a model with half
the market's spread that nevertheless scores level will find its edges where the
market moves furthest from the base rate. Its bets will lean **against** strong
favourites and heavy overs. That is a structural property of the dispersion, not a
strategy, and it should be recognised as such when the staking results appear.

### RQR says the count distribution is right

```
RQR mean   -0.0239   (target 0)
RQR sd      0.9855   (target 1)   → well dispersed
home/away   0.0015 / -0.0493, gap 0.0508
```

Randomised quantile residuals are exactly N(0,1) under correct specification, so
both moments mean something. sd 0.9855 says the NegBin predictive for team goal
counts is essentially correctly shaped.

**Note this does not contradict the 0.55 dispersion figure above** — they measure
different things. RQR asks whether the predicted distribution of goals for a given
fixture has the right spread (it does). The 0.55 asks whether predicted
probabilities vary enough BETWEEN fixtures (they do not). The model has the right
uncertainty about each match and too little conviction about which match is which.

LPD (scoreline log predictive density, averaged over the posterior): mean **-2.9816**,
total -1073.4. Meaningless alone; it exists to rank variants of this model.

### My gate-5 draw hypothesis was wrong

Gate 5 saw mean predicted draw ≈ 0.243 on 8 smoke fixtures and I argued that
conditionally independent goals under-predict draws, making it the strongest case
for a Dixon-Coles term. Tested on all 360:

```
observed 0.2333 vs predicted 0.2523, z = -0.85
```

The model **over**-predicts draws slightly, and not significantly. The direction is
opposite to the hypothesis. There is no draw-based argument for a Dixon-Coles or
copula term on this data, and the 8-fixture reading was noise.

### Calibration slopes: the point estimates are uninterpretable here

Slopes ranged over [-1.12, 1.73] and only 1 of 13 fell inside a [0.7, 1.3] band —
which looked alarming and is an artifact. Standard errors run 0.85 to 2.97, and
**all 13 lines are within 2 se of a perfectly calibrated slope of 1** (worst |z| =
0.92). With `sd(p_model)` as low as 0.008 on the 1X2 draw, the regression has almost
no leverage and returns noise.

The gate now tests significance rather than banding a point estimate. Banding it
would reject a model for being under-dispersed, which is a real property worth
reporting and is not miscalibration.

### glm_edge: no incremental information yet

Encompassing regression `y ~ logit(p_market) + logit(p_model)` per line. Best model
coefficient is O/U 0.5 at z = 1.2; nothing reaches z = 2. On this sample the model
does not demonstrably know anything the closing line does not — which, with 360
fixtures and Δ log loss around 0.005, is what the power available can detect.

### Fold weighting: the trap is real but small here

Pooled 1X2 log loss 0.6188 vs fold-averaged 0.6182, difference **-0.0007** across 20
folds ranging 6 to 72 rows. Worth measuring rather than asserting: it is an order of
magnitude below the Δll effects being discussed, so on this data the choice does not
change a conclusion. Fixture-weighted remains the default because that ordering is
not guaranteed on a more uneven grid.

### Three defects found by building this gate

1. **T005 (high)** — `summarize_betfair_market` inner-joins an open window 24h before
   kickoff, returning **30 of 360** fixtures where 322 exist. Gate 7 uses Betfair
   close as its primary discriminator and would have run on 8% of the data.
2. **T004 (low)** — `is_winner` contradicts the score on 3 fixtures, incl. two 2-2
   draws with no 1X2 winner at all. None in the 24/25 window.
3. **Partial markets de-vig to p = 1.0.** 143 of 930 Betfair markets had one leg;
   the overround is computed over what is present, so a single leg de-vigs to
   certainty. Clamped, one losing p=1 costs ~20.7 log loss — enough to make a sound
   model look catastrophic with every other diagnostic reading healthy. Now detected
   and dropped.

Two gate-design errors of my own, both caught by their own output: joining all
baselines into one table cut the evaluation from 4,658 rows to 96 while reporting
PASS, and coverage measured in rows rather than fixtures failed a legitimate baseline
for being a thin book.

### Book-level summary

Multiclass log loss per market — `-log p(what happened)` — so binary markets are not
double-counted through their yes/no legs. Negative Δ means the model is better.

| baseline | market | n | model | market | Δ | t |
|---|---|---|---|---|---|---|
| bet365 | 1X2 | 360 | 1.0637 | 1.0656 | -0.0018 | -0.18 |
| bet365 | BTTS | 359 | 0.6947 | 0.6927 | +0.0021 | +0.65 |
| bet365 | O/U 0.5 | 357 | 0.2598 | 0.2682 | **-0.0083** | -1.82 |
| bet365 | O/U 1.5 | 357 | 0.5541 | 0.5540 | 0.0000 | +0.01 |
| bet365 | O/U 2.5 | 359 | 0.6942 | 0.6964 | -0.0021 | -0.61 |
| bet365 | O/U 3.5 | 357 | 0.5903 | 0.5933 | -0.0030 | -0.79 |
| **bet365** | **BOOK (6 markets)** | **357** | **3.8591** | **3.8728** | **-0.0137** | **-0.93** |
| betfair | 1X2 | 304 | 1.0622 | 1.0648 | -0.0026 | -0.20 |
| betfair | O/U 2.5 | 188 | 0.6890 | 0.6894 | -0.0004 | -0.07 |
| betfair | O/U 3.5 | 110 | 0.5790 | 0.5820 | -0.0030 | -0.43 |
| betfair | BOOK (6 markets) | **2** | — | — | — | — |

Pricing the whole six-market book costs the model **3.8591 nats per fixture against
the closing line's 3.8728** — 0.0137 better, t = -0.93. Not significant, and not
expected to be on 357 fixtures.

**The Betfair BOOK row is n = 2.** Only two fixtures of 320 have all six markets
priced on the exchange. The exchange book on this league is thin, so book-level
comparison against Betfair is meaningless while its per-market rows are sound. This
matters for gate 7: a staking book built on Betfair is mostly 1X2 (304 fixtures) and
O/U 2.5 (188), with BTTS essentially absent (42).

Three bugs in the summary table itself, each caught by its own output:
positional column assignment printed Betfair's O/U 1.5 log loss on the O/U 0.5 row;
a `zip` against `unique(match_id)` desynced whenever an ungraded fixture was skipped;
and the BOOK row summed however many markets each fixture happened to have, which
made the thin baseline look 2 nats cheaper. All three produced entirely plausible
numbers.

## 2026-08-26 — Gate 7 PASS 17/17. No demonstrated betting edge.

320 books at Betfair closing prices, 2% commission, portfolio cap 0.20, daily slates.
Built by assembling `src/Portfolio`; no staking mathematics written here.

### The numbers, and why they do not mean what they appear to

| policy | bets | final | ROI% | CI | **top10%** | win rate | growth |
|---|---|---|---|---|---|---|---|
| full book | 908 | 2.015 | 21.31 | [2.1, 40.9] | **108.3** | 0.338 | 0.0140 |
| totals only | 338 | 1.039 | 5.69 | [-16.1, 28.9] | **411.0** | 0.382 | 0.0008 |
| totals + BTTS | 370 | 1.015 | 2.30 | [-18.9, 24.9] | **919.4** | 0.384 | 0.0003 |
| 1X2 only | 538 | 1.979 | 26.83 | [1.9, 53.1] | **111.0** | 0.307 | 0.0137 |

Two policies double the bankroll with a bootstrap interval excluding zero. **That is
not evidence of an edge**, and the column that says so is `top10_pct`.

**The top ten bets are 108% of the full book's total P&L.** The other 898 are net
negative in aggregate. On `totals + BTTS` the top ten are 919%.

### Gate 6 and gate 7 disagree, and gate 6 wins

Gate 6 found no information advantage on 1X2: Δ log loss -0.0018 (t = -0.18) and an
encompassing-regression coefficient of z = -0.09. Gate 7 shows 1X2-only returning
26.8%.

The family breakdown resolves it. `1X2_away` contributed **0.548 of the 0.758 total
P&L (72%)** from 204 bets — and away is the line where gate 6 found the model at its
**worst** (Δll +0.0051, the only clearly positive line). The model prices away wins
less well than the market and made most of its money backing them.

That is the signature of backing longshots at long odds and having a few land. Gate 6
has far more statistical power to detect an information advantage than a 320-match
bankroll path, so where they conflict, gate 6 is the one to believe.

### Honest status of model 01

Sound, correctly implemented, competitive with the closing line on proper scoring,
and **with no demonstrated betting edge on one season**. That is a perfectly good
baseline — it is what models 02 and 03 must beat, and it is now measured rather than
assumed.

Anyone quoting "the baseline returned 21%" without `top10_pct` beside it is quoting
noise.

### Notes on the machinery

- `DeArb` settles at `d * min(overround, 1)`: the quoted price under real vig, shaved
  where the recorded book implies an arbitrage. ROI is not inflated by data artifacts.
- `require_complete_markets` drops partial markets inside `extract_selections`, so the
  p = 1.0 trap from gate 6 cannot reach here.
- Settlement uses the real score via `settle_vector`, not `is_winner`, so T004 cannot
  reach these numbers either.
- KKT residual: 319 of 320 books at ~1.2e-6 as documented. The single 2.7e-4 outlier
  is a book whose optimum is "bet nothing" — the solution is on the boundary, no stake
  is placed, and the residual is not the relevant diagnostic. The gate now checks only
  books that allocate.
- Betfair prices 5.9 selections per fixture on average, and the exchange book is thin:
  BTTS appears on 42 fixtures, O/U 0.5 on 61. Curation to totals is therefore also a
  curation to a much smaller sample, which is part of why those rows are noisier.

### Next

Model 01 is complete through all seven gates. Options, in order of what the evidence
supports:

1. **`02_apm_player_poisson`** — the planned next model, now with a measured baseline
   to beat.
2. **Variants of 01** — league-varying home advantage for 56 vs 57 is a real modelling
   question (two divisions pooled, no reason to assume a shared home edge). The
   Dixon-Coles case is NOT supported: gate 6 refuted the draw deficit.
3. **More seasons before more models.** The binding constraint on every conclusion
   above is 360 fixtures. `25/26` is sealed by design, but 23/24 and earlier are not,
   and widening the development window would sharpen every comparison that follows. — OOS blocks range from 2 to 24
fixtures, so a fold average would let a 2-fixture block outvote a 24-fixture one.
Then gate 7, growth and CLV against Betfair closing prices.

Open tickets stand and neither blocks gate 6: T002 is cost only; T003 mis-prices
0.56% of fixtures on one side.
