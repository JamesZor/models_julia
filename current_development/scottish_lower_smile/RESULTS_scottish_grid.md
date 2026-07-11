# RESULTS — Scottish Lower team-level sup+smile grids

> TEMPLATE — fill sections as the runners complete. Tables paste directly from r03/r05 output.
> Convention: cells below the convergence gate are struck through, never silently dropped.

## 0. Run log

| runner | date | wall time | notes |
|---|---|---|---|
| r01 smoke | | | |
| r02 Grid A | | | |
| r03 eval A | | | |
| r04 Grid B | | | |
| r05 eval B | | | |
| r06 src smoke | | | |

## 1. r01 smoke verdicts

(paste the R01 SMOKE SUMMARY ✅/❌ block + the δ_league / σ readouts)

## 2. Grid A — decay × history (r02/r03)

**Convergence gate** (from `r02_convergence.txt`; disqualify < 95%):

(paste)

**Family-pooled LogLoss diff** (model−market vs Bet365 close; lower better):

(paste the 📊 table)

**Per-line notes** (over/under symmetry, GLMEdge stability, RQR centring):

- ...

**DECISION — Grid-A winner:** `hl* = ___`, `hs* = ___`
Rationale (1–3 lines):

**nb reference check:** none_nb vs none_pois at (180, 2) — expected ≈ equal (V/M < 1, r inert).
Observed: ...

## 3. Grid B — supremacy × smile (r04/r05)

**Convergence gate** (from `r04_convergence.txt`):

(paste)

**Family routing table** (family-pooled LogLoss diff per cell — the headline):

(paste)

**Per-strike O/U read** (does φ win at the 0.5/1.5 and 3.5/4.5 tails?):

(paste / notes)

**Kelly tearsheet highlights** (hurdle_G per selection; money lens, read last):

(paste)

**Betfair 25/26 secondary CLV** (informational, ~315 matches):

(paste / n/a)

**DECISION — per market family:**

| family | winning cell | note |
|---|---|---|
| 1X2 | | |
| Totals (O/U) | | |
| BTTS | | |

Does the Ireland routing (supremacy→1X2, smile→totals/BTTS) transfer? ...

## 4. Graduation record (Stage 4)

- [ ] r05 winner weights baked into `DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel`
      defaults (`supremacy_weight`, `smile_weight`, `TimeDecayDynamics(days_half_life=…)`)
      in `src/models/pregame/engines/team_level/time_decay/goals_smile_league.jl`.
- [ ] r06 src smoke: all checks ✅ (paste summary).
- [ ] `julia --project -e 'using Pkg; Pkg.test()'` green after src changes.
- [ ] NOTES.md findings log updated; memory updated.

## 5. Open threads / follow-ups

- Live `match_day_inference` wiring for ScottishLower (separate session).
- Betfair historical download for 56/57 (user) → re-run the r05 secondary with full coverage.
- CMP escalation ONLY if none_pois showed a per-line bias the smile pricing could not fix.
