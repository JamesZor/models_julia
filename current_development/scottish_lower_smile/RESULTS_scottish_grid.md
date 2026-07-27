# RESULTS — Scottish Lower team-level sup+smile grids

> TEMPLATE — fill sections as the runners complete. Tables paste directly from r03/r05 output.
> Convention: cells below the convergence gate are struck through, never silently dropped.

## 0. Run log

| runner | date | wall time | notes |
|---|---|---|---|
| r01 smoke | 2026-07-12 | none 11m / iso 40m / smile 3h35m | 29/29 ✅; smile 20× runtime → r01b probe |
| r02 Grid A | 2026-07-12→13 | 30.75 h | 14 cells × 60 folds, warmup_period=0 |
| r03 eval A | 2026-07-13 | ~2 h | run with stdout→r03_out.txt (gate-timeout-proof) |
| r04 Grid B | 2026-07-14→16 | 58.1 h | fast-rank redesign: 8 cells, 2 seasons/3ch; smile d6 cells 10h ea |
| r05 eval B | 2026-07-16 | ~50 m | stdout→r05_out.txt; tearsheet re-run in-session |
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

### 3.0 Design actually run (fast-rank redesign, 2026-07-13; see NOTES.md)

r01b depth probe made the original 10-cell/depth-10 grid infeasible (~25h/cell). Run spec:
targets **24/25→25/26** (40 folds), 1200/300 × **3 chains**, hl365/hs2, warmup_period=0.
Cells: `sup{40,70,100} × sw{0,50}` (sw>0 at depth 6 = RANKING only; sw=0 at depth 10)
+ `iso_pois_mw100` (d10) + `none_pois` ctl (d10, re-run at this spec for comparability).
Wall: 58.1h total (none 1h23, iso 5h36, sw0 5h44–7h36, sw50 **10h11 each**).

### 3.1 Convergence gate (`r04_convergence.txt`)

```
smile_pois_sup40_sw0   [d10]: 40/40 ≤1.01 (100%)  worst=1.0088  ✅ hard
smile_pois_sup70_sw0   [d10]: 40/40 ≤1.01 (100%)  worst=1.0082  ✅ hard
smile_pois_sup100_sw0  [d10]: 40/40 ≤1.01 (100%)  worst=1.0089  ✅ hard
iso_pois_mw100         [d10]: 24/40 ≤1.01 (60%); 40/40 ≤1.05    worst=1.0266  ⚠ marginal
none_pois_ctl          [d10]: 39/40 ≤1.01 (97.5%)               worst=1.0109  ✅ hard
smile_pois_sup40_sw50  [d6]:  0/40 ≤1.01; 20/40 ≤1.05 (50%)     worst=1.2687  ❌ FAILED ranking gate
smile_pois_sup70_sw50  [d6]:  0/40 ≤1.01; 19/40 ≤1.05 (47.5%)   worst=1.2830  ❌ FAILED ranking gate
smile_pois_sup100_sw50 [d6]:  0/40 ≤1.01; 24/40 ≤1.05 (60%)     worst=1.3097  ❌ FAILED ranking gate
```

**Depth-6 failure diagnosis** (per-fold, per-param): bad folds are scattered across ALL 40 (not
season-start-only); `log_φ` itself is implicated in ~25–55% of bad folds with **min φ ESS 19–30**
— the φ tail-pricing runs on ~20 effective draws there. The r01b probe (worst 1.077) was
misleading: it sampled 5 mid-season folds (warmup 16) at 4 chains; Grid B's season-start folds
(warmup 0) are far stiffer. **Lesson: depth caps do NOT work for the smile pillar on this data.**
sw50 rows below are shown for completeness — indicative only (r01b unbiasedness argument), never
gate-grade.

### 3.2 Per-line LogLoss diff (model−market vs Bet365 close; NEGATIVE beats close; 710 matches)

| model | home | draw | away | btts_y | btts_n | o05 | o15 | o25 | o35 | o45 | o55 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| iso_mw100 ⚠ | 0.0043 | −0.0017 | 0.0062 | 0.0022 | 0.0022 | −0.0094 | −0.0013 | −0.0042 | −0.0025 | −0.0114 | −0.0186 |
| none_ctl | 0.0093 | 0.0000 | 0.0085 | 0.0009 | 0.0009 | −0.0084 | 0.0033 | 0.0003 | 0.0021 | −0.0117 | −0.0182 |
| sup100_sw0 | 0.0054 | −0.0005 | 0.0063 | −0.0001 | −0.0001 | 0.0027 | 0.0126 | −0.0015 | 0.0057 | **0.0185** | −0.0191 |
| sup70_sw0 | 0.0053 | −0.0005 | 0.0061 | −0.0001 | −0.0001 | 0.0028 | 0.0126 | −0.0012 | 0.0056 | 0.0174 | −0.0191 |
| sup40_sw0 | 0.0052 | −0.0005 | 0.0058 | −0.0001 | −0.0001 | 0.0028 | 0.0128 | −0.0016 | 0.0060 | 0.0180 | −0.0191 |
| sup100_sw50 ❌ | 0.0042 | −0.0016 | 0.0056 | 0.0034 | 0.0034 | −0.0009 | 0.0008 | −0.0032 | −0.0011 | −0.0032 | −0.0177 |
| sup70_sw50 ❌ | 0.0043 | −0.0016 | 0.0052 | 0.0036 | 0.0036 | −0.0009 | 0.0008 | −0.0033 | −0.0011 | −0.0032 | −0.0179 |
| sup40_sw50 ❌ | 0.0042 | −0.0016 | 0.0051 | 0.0035 | 0.0035 | −0.0010 | 0.0008 | −0.0033 | −0.0012 | −0.0033 | −0.0181 |

(over_K = under_K by construction — binary log score. Full 17-selection table in `r05_out.txt`.)

### 3.3 Family routing table (family-pooled mean LogLoss diff — the headline)

| model | x12 | btts | totals | totals_tails | gate |
|---|---|---|---|---|---|
| iso_pois_mw100 | 0.0029 | 0.0022 | **−0.0058** | **−0.0062** | ⚠ 60% |
| none_pois_ctl | 0.0059 | 0.0009 | −0.0029 | −0.0037 | ✅ |
| sup40_sw0 | 0.0035 | **−0.0001** | +0.0076 | +0.0099 | ✅ |
| sup70_sw0 | 0.0036 | −0.0001 | +0.0074 | +0.0096 | ✅ |
| sup100_sw0 | 0.0037 | −0.0001 | +0.0076 | +0.0099 | ✅ |
| sup40_sw50 | **0.0026** | 0.0035 | −0.0016 | −0.0012 | ❌ |
| sup70_sw50 | 0.0026 | 0.0036 | −0.0015 | −0.0011 | ❌ |
| sup100_sw50 | 0.0027 | 0.0034 | −0.0015 | −0.0011 | ❌ |

Key facts:
1. **Supremacy-weight axis is dead flat** — sup 40/70/100 differ ≤0.0002 on every family.
2. **Supremacy-only anchoring is TOXIC for totals** (+0.0076; worst rows in the table) —
   anchoring the log-ratio with no level anchor distorts λ_tot.
3. **iso wins totals AND totals_tails outright** and effectively ties best on 1X2.
4. The smile pillar repairs sw0's totals damage (−0.0015) but never approaches iso (−0.0058) —
   consistent with the established *no-pregame-smile* finding (φ≈1, sub-Poisson FT totals).
5. BTTS flips: sw0 best (−0.0001), none 0.0009, iso worst-ish (0.0022).

### 3.4 GLMEdge (spread_fair coef; ·=p≥0.10)

Almost everything n.s. Significant: iso over/under_2.5 (+5.97); sw50 over/under_2.5 (+5.6);
sw0 over/under_5.5 (+13.2 — echo of the totals distortion, not edge). No 1X2 line significant
for any model. Full table in `r05_out.txt`.

### 3.5 Kelly tearsheet (BayesianKelly at Bet365 close; money lens, read LAST)

ROI % (bets placed):

| model | home | draw | btts_no | u15 | u25 | o35 | u45 | o45 |
|---|---|---|---|---|---|---|---|---|
| iso_mw100 | −5.2 (252) | +42.3 (40) | −4.5 (40) | **+29.7 (102)** | **+17.0 (182)** | **+27.2 (101)** | +9.7 (90) | **+19.3 (23)** |
| none_ctl | −17.3 (226) | +26.7 (81) | −4.2 (136) | −3.3 (302) | +0.5 (325) | +9.9 (87) | +2.4 (157) | −33.7 (44) |
| sup100_sw0 | −6.9 (232) | +40.0 (93) | +3.9 (152) | −8.1 (534) | +0.7 (242) | +2.9 (389) | +24.7 (5) | **−18.9 (576!)** |
| sup100_sw50 ❌ | −4.7 (237) | +41.8 (38) | +3.7 (39) | +5.1 (179) | +13.5 (149) | +7.4 (221) | +19.7 (14) | −10.0 (223) |

hurdle_G (per-bet growth vs hurdle): iso positive across the totals mid-ladder
(u25 +0.003, o25 +0.002, u35 +0.002, u45 +0.005), 1X2 NEGATIVE for every model (home
−0.006…−0.011), BTTS positive only for sw0 (+0.001). sw0's over_45: 576 bets (81% of matches)
at −18.9% ROI — the totals distortion bleeding real money, matches §3.3 fact 2.

### 3.6 Betfair 25/26 secondary CLV

Ran (315 matches) but `display_summary_metric` printed empty under redirect — numbers not
captured; informational only, does not affect the verdict. Re-run on the winner if needed.

**DECISION — per market family:**

| family | winning cell | note |
|---|---|---|
| 1X2 | *none — abstain* | no model makes Kelly money on 1X2 (home −ROI everywhere); Ireland lesson repeats |
| Totals (O/U) | **iso_pois_mw100** | wins LogLoss (−0.0058), tails (−0.0062), and ROI/hurdle_G across the ladder; gate ⚠ 60% needs a confirm run |
| BTTS | sup*_sw0 (marginal) | only cell with hurdle_G ≥ 0 on BTTS; margin tiny (−0.0001 vs none 0.0009) |

Does the Ireland routing (supremacy→1X2, smile→totals/BTTS) transfer? **NO — this is the
headline.** At team level with goals-only data: the supremacy anchor helps nothing (1X2 money
is not extractable, and it wrecks totals when unaccompanied); the smile pillar prices nothing
(no pregame smile to fit); the plain isotropic LEVEL anchor is the only pillar that pays, and
only on totals. The market pillar's entire value here is level denoising (the r09
totals-compression lesson, again).

### 3.8 ENDGAME — iso mw sweep (r07/r08, 2026-07-17/18) → **mw = 0.40**

Option A run: `iso_pois_mw{25,40,70}` @3ch + `iso_pois_mw100_c4` @4ch, depth 10, Grid-B spec
(24/25→25/26, 40 folds, 1200/300, hl365/hs2). Wall **21.3 h** (3h16 / 4h22 / 5h15 / 8h22).

**Gate (`r07_convergence.txt`, HARD ≥95% folds ≤1.01):**

| cell | ≤1.01 | worst | offending params (occurrences) |
|---|---|---|---|
| iso_mw25 (3ch) | 92.5% ⚠ | 1.0112 | dyn.σ_d ×2, dyn.raw_d ×1 (3 bad folds) |
| **iso_mw40 (3ch)** | **95.0% ✅** | 1.0132 | dyn.raw_d ×5, dyn.σ_d ×1 (2 bad folds) |
| iso_mw70 (3ch) | 65.0% ⚠ | 1.0214 | dyn.raw_a ×84, dyn.raw_d ×46 (14 bad folds) |
| iso_mw100_c4 (4ch) | 67.5% ⚠ | 1.0178 | dyn.raw_d ×87, dyn.raw_a ×68 (13 bad folds) |

Two structural reads: (a) **offenders are the team ratings, never the market pillar σ** — heavier
anchoring stiffens the latent-rating geometry, the "raising mw backfires" lesson appearing in the
SAMPLER; (b) 4 chains did NOT rescue mw100 (67.5% vs 60% @3ch) — R-hat with more chains is a
stricter test, so the marginal miss is real, not noise. All worst values ≤1.021 (mild; contrast
the depth-6 smile cells at 1.27–1.31).

**Reproduction check ✅** — `mw100_c4` vs Grid-B `mw100`: totals −0.0058 both, tails −0.0062 both,
ladder ROI 7.56% vs 7.55%, bets 908 vs 904. The pipeline reproduces; tables are trustworthy.

**Family-pooled LogLoss — the mw axis is FLAT:**

| model | x12 | btts | totals | totals_tails |
|---|---|---|---|---|
| iso_mw25 | 0.0029 | **0.0017** | −0.0057 | −0.0061 |
| iso_mw40 | 0.0029 | 0.0020 | −0.0058 | −0.0061 |
| iso_mw70 | 0.0030 | 0.0021 | −0.0058 | −0.0062 |
| iso_mw100_c4 | 0.0030 | 0.0022 | −0.0058 | −0.0062 |
| none_ctl | 0.0059 | 0.0009 | −0.0029 | −0.0037 |
| sup100_sw0 | 0.0037 | −0.0001 | +0.0076 | +0.0099 |

Totals spread across a 4× range of mw = **0.0001** (noise). **The Ireland interior optimum
(mw 0.25–0.4, "raising it backfires") does NOT transfer to 56/57** — mw is a no-op on scoring.

**Money lens — whole O/U ladder aggregated (12 selections, 710 matches):**

| model | bets | turnover | profit | ROI % |
|---|---|---|---|---|
| iso_mw70 | 894 | 30.0 | 2.3 | **7.69** |
| iso_mw100_c4 | 908 | 32.4 | 2.4 | 7.56 |
| iso_mw100 | 904 | 32.4 | 2.4 | 7.55 |
| iso_mw40 | 854 | 26.0 | 1.9 | 7.19 |
| iso_mw25 | 842 | 23.2 | 1.4 | 6.07 |
| none_ctl | 1440 | 48.2 | −1.4 | −2.93 |
| sup100_sw0 | 2353 | 25.6 | −1.4 | −5.30 |

Every iso cell earns +6…+8% on the ladder; both non-pillar controls LOSE money. Within iso the
ROI spread (6.07–7.69) is mildly increasing in mw but is noise at ~850 bets / ~26 units of
turnover. Turnover rises with mw (23→32) — heavier anchoring ⇒ larger deviations ⇒ bigger stakes.
hurdle_G agrees and is equally flat (under_25 +0.002…+0.003, under_45 +0.002…+0.005 across mw).

**DECISION — production mw = 0.40.** Justification: scoring cannot separate the mw cells
(Δtotals = 0.0001) and the money difference is inside noise, so the tiebreaker is the only
axis that DOES separate them — sampler health. mw40 is the sole cell clearing the hard
convergence gate, and it sits mid-range so it is not a boundary artefact. Choosing mw70/mw100
would buy ≤0.5pp of unreliable ROI in exchange for 13–14 badly-mixed folds per run, which
matters operationally since this model retrains weekly in-season.

### 3.7 Endgame options as offered (resolved 2026-07-17 → option A; see §3.8)

Options on the table (2026-07-17):
- **A (recommended): iso mw sweep + confirm** — iso mw{25,40,70} @3ch + mw100 @4ch hard-gate
  re-run (~24h ≈ one overnight). Tests the Ireland mw-optimum (0.25–0.4, "raising it
  backfires") on 56/57 and fixes the marginal gate. Graduate iso Poisson+league engine at the
  winning mw; route totals only (BTTS optionally from sw0).
- **B: no more compute** — graduate iso_mw100 with the gate caveat documented.
- **C: smile fair-shot** — sup100_sw50 at depth 10 (~25h); r01b unbiasedness says its −0.0015
  totals won't catch iso's −0.0058, so this buys certainty, not a different verdict.

## 4. Graduation record (Stage 4)

> ⚠ REVISED by the Grid-B verdict (§3): the production candidate is the **iso Poisson+league
> engine** (TeamIsoDPGoalsModel → src), NOT the smile engine. The already-landed
> `DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel` stays in src (harmless; Union dispatch
> shared with the Ireland player engine) but is NOT the ScottishLower production model.
> Checklist below to be rewritten once the §3.7 endgame decision is made.

- [ ] ~~r05 winner weights baked into `DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel`
      defaults (`supremacy_weight`, `smile_weight`, `TimeDecayDynamics(days_half_life=…)`)
      in `src/models/pregame/engines/team_level/time_decay/goals_smile_league.jl`.~~
- [ ] r06 src smoke: all checks ✅ (paste summary).
- [ ] `julia --project -e 'using Pkg; Pkg.test()'` green after src changes.
- [ ] NOTES.md findings log updated; memory updated.

## 5. Open threads / follow-ups

- Live `match_day_inference` wiring for ScottishLower (separate session).
- Betfair historical download for 56/57 (user) → re-run the r05 secondary with full coverage.
- CMP escalation ONLY if none_pois showed a per-line bias the smile pricing could not fix.
