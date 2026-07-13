# RESULTS — Scottish Lower team-level sup+smile grids

> TEMPLATE — fill sections as the runners complete. Tables paste directly from r03/r05 output.
> Convention: cells below the convergence gate are struck through, never silently dropped.

## 0. Run log

| runner | date | wall time | notes |
|---|---|---|---|
| r01 smoke | 2026-07-12 | none 11m / iso 40m / smile 3h35m | 29/29 ✅; smile 20× runtime → r01b probe |
| r02 Grid A | 2026-07-12→13 | 30.75 h | 14 cells × 60 folds, warmup_period=0 |
| r03 eval A | 2026-07-13 | ~2 h | run with stdout→r03_out.txt (gate-timeout-proof) |
| r04 Grid B | | | |
| r05 eval B | | | |
| r06 src smoke | | | |

## 1. r01 smoke verdicts

29/29 ✅ (r01_out.txt). δ₅₆−δ₅₇ ≈ +0.02–0.035 (expected +0.047; smile CI excludes 0);
σ_smile ≈ 0.052 (market-hugging, = Ireland), σ_sup ≈ 0.249 (loose), iso σ_market ≈ 0.134;
smile-vs-grid O/U max Δ 0.042 (φ prices genuinely). Runtime finding → r01b depth probe.

## 2. Grid A — decay × history (r02/r03)

**Convergence gate** (from `r02_convergence.txt`; disqualify < 95%): 12/14 pass.
~~none_pois_hl60_hs3~~ (91.7%), ~~iso_nb_mw100_hl180_hs2~~ (75.0%, worst 1.021).

**Family-pooled LogLoss diff** (model−market vs Bet365 close; lower better):

| model | x12 | btts | totals |
|---|---|---|---|
| none_pois_hl60_hs1 | 0.0168 | 0.0017 | 0.0012 |
| none_pois_hl60_hs2 | 0.0168 | 0.0016 | 0.0011 |
| ~~none_pois_hl60_hs3~~ | 0.0168 | 0.0016 | 0.0011 |
| none_pois_hl120_hs1 | 0.0157 | 0.0016 | 0.0010 |
| none_pois_hl120_hs2 | 0.0156 | 0.0018 | 0.0008 |
| none_pois_hl120_hs3 | 0.0157 | 0.0019 | 0.0010 |
| none_pois_hl180_hs1 | 0.0156 | 0.0017 | 0.0007 |
| none_pois_hl180_hs2 | 0.0154 | 0.0019 | 0.0007 |
| none_pois_hl180_hs3 | 0.0156 | 0.0019 | 0.0006 |
| none_pois_hl365_hs1 | 0.0157 | 0.0016 | 0.0007 |
| **none_pois_hl365_hs2** | **0.0143** | **0.0014** | **0.0002** |
| none_pois_hl365_hs3 | 0.0150 | 0.0014 | 0.0003 |
| none_nb_hl180_hs2 | 0.0162 | 0.0026 | 0.0012 |
| ~~iso_nb_mw100_hl180_hs2~~ | 0.0083 | 0.0014 | −0.0030 |

**Per-line notes** (over/under symmetry, GLMEdge stability, RQR centring):

- over_K / under_K LogLoss diffs identical by construction (binary log score) — reads are per-strike.
- GLMEdge: short half-lives actively pathological — significant NEGATIVE away coefs (hl60 ≈ −5.2,
  hl120 ≈ −4.4, hl180_hs1 −3.4): over-reactive ratings anti-predict vs the close. At hl365 all
  coefs n.s. and nearest 0.
- RQR: every cell well-centred (|mean| ≤ 0.03, std ≈ 1) — no goal-calibration pathology.
- Structural cells beat the close only at the extreme strikes (over/under_05, _45, _55 negative)
  — the market's tail vig; middle strikes ≈ market (totals pooled 0.0002 at the winner).

**DECISION — Grid-A winner:** `hl* = 365`, `hs* = 2`
Rationale: best family-pooled LogLoss on ALL three families; monotone hl gradient on x12
(0.0168 → 0.0143) — long memory wins despite promotion churn (sub-Poisson, stable strengths);
hs3 adds nothing, hs1 truncates the decay; only hl365 clears the GLMEdge pathology check.

**nb reference check:** none_nb vs none_pois at (180, 2) — expected ≈ equal (V/M < 1, r inert).
Observed: nb ≤ pois on every family (x12 0.0162 vs 0.0154, btts 0.0026 vs 0.0019, totals 0.0012
vs 0.0007) → **Poisson base confirmed**, no dispersion escalation.

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
