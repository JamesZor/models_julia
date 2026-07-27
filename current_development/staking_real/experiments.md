# staking_real — Real-Data Staking MVP (run log)

Backtest the sim-validated staking layer — per-line trust blend → coherent IPF grid tilt →
capped unified Kelly (P), with the EB trust fit as a junk-line alarm — on the **real OOS
matches** of the `src_sup40_sw40` L1 engine (League of Ireland Premier, 2025–26, Betfair
close), and answer the MVP question:

> **Does the EB fit pull trust `w` DOWN on the markets the model is bad at (home/away 1X2)
> while HOLDING the good ones (unders, BTTS)?**

Answer (275 matches): **yes, decisively** — see §Results. This is signature-reading, not
ranking; the EB alarm moving the right way matters more than any single strategy's P/L.

Background: sim lab `../staking_sim/` (E1–E4) + `docs/bets_multi/staking_sim_report.pdf`;
memory note `staking-sim-mc-race`. The sim's "sup-blind" world (E4) predicted exactly this
signature and ordering; this run reproduces it on real books.

---

## Preflight (Hetzner / kaimon)

- Server: `/root/BayesianFootball`, `Threads.nthreads()=16` (32 HT / 16 cores). kaimon REPL.
- Payload present: `data/double_poisson_smile_src_grid/src_sup40_sw40_20260704_050900/`.
- Data: `Data.Ireland()` datastore → `summarize_betfair_market(open=(-1e5,-10), close=(-20,0))`
  swapped into `ds1.odds` (exactly as r21:157). 293 OOS matches in the engine's latents;
  **275** have a Betfair close book (the backtest set). `is_winner` is clean `Bool`.
- `extract_oos_predictions(ds1, res)` → per-draw `λ_h/λ_a/λ_tot/φ` (3200 draws = 800×4).
  `Predictions.model_inference(lat)` → smile-correct per-selection PPD.
  Both cached to `results/_lat_ppd_cache.jls` (gitignored) so the race reloads in seconds.

## Design

**Adapter** (`l01_real_books.jl`, `build_real_books`): real L1 predictions + Betfair close →
the SAME `SimMatch` the sim lab races, so the sim `l02` machinery runs verbatim.
- `SimMatch.P` = plain double-Poisson grids (144×S) from the posterior λ draws — the coherent
  substrate the unified solver needs. `pbar` = grid mean.
- **Smile subtlety** (the key design point): `src_sup40_sw40` prices O/U through
  `Λ = λ_tot·φ(K)` (`SmileScoreMatrix`), NOT the plain grid — confirmed on match 1:
  grid over-2.5 = 0.267 vs smile PPD = 0.289 (≈2.2pp). So the per-UNIT **model targets** for
  the trust blend + the raw-unified (w=1) tilt come from the smile PPD (carried as
  `smile_sel` [11] and `smile_dists` [11×S]); O/U = smile probs, 1X2/BTTS = grid by
  construction. The runner imprints `smile_sel` onto the grid via `coherent_multiplier` (IPF),
  so **every** unified strategy — incl. raw U (w=1) — prices O/U exactly as b21 certified.
- Book = core 11 selections (1X2, O/U 1.5/2.5/3.5, BTTS), fixed sim order. Commission `c`
  folds into payout `d_eff = 1+(odds_close−1)(1−c)` for BOTH decisions and settlement.
  Missing line ⇒ `d=1.0` (pure-loss column ⇒ solver never stakes it) and `q_mkt=model prob`
  (blend no-op). Settlement prefers graded `is_winner`, falls back to the true-score mask.

**Strategies** (`r01_race_src_sup40.jl`, pluggable registry — new system = one `elseif`):
`FLAT_1pct` · `PB_BK_cap02` (per-bet Bayesian–McHale Kelly 0.03 on smile draws, Σa≤0.2 — the
b21-comparable baseline) · `U_cap02` (raw unified, w=1, smile-tilted) · `TRUST05_U_cap02`
(flat w=0.5) · `CURATED05_U_cap02` (w=[0,0,0,.5,.5,.5,.5]) · `TRUST_EB_U_cap02` (EB-learned w,
cold-start 0.5, refit every 25). All bet from match 1, share books, compound sequentially,
ruin-freeze < 0.01. Race run at c=0.02 and c=0.

---

## Results (n=275, c=0.02 unless noted)

### Race table

| strategy | term_W | G/match ± SE | maxDD | n_bets | turnover | ruined |
|---|---:|---:|---:|---:|---:|:--:|
| FLAT_1pct | 1.87 | +0.00228 ± 0.00178 | 0.26 | 692 | 6.9 | – |
| PB_BK_cap02 | 0.57 | −0.00205 ± 0.01223 | 0.98 | 701 | 41.9 | – |
| U_cap02 | **0.01** | −0.01686 ± 0.01308 | 1.00 | 827 | 49.4 | **YES** |
| TRUST05_U_cap02 | 0.71 | −0.00123 ± 0.00984 | 0.95 | 1052 | 41.8 | – |
| **CURATED05_U_cap02** | **26.86** | **+0.01197 ± 0.00402** | 0.42 | 787 | 25.9 | – |
| TRUST_EB_U_cap02 | 2.98 | +0.00397 ± 0.00867 | 0.87 | 1049 | 39.8 | – |

c=0 (frictionless) same ordering, wider spread: CURATED 44.6, TRUST_EB 4.71, FLAT 1.95,
TRUST05 1.28, PB_BK 0.82, U 0.02 (ruined). Full tables: `results/e_real_summary_c020.txt`,
`results/e_real_summary_c000.txt`.

**Ordering = sim E4 exactly: `CURATED ≻ TRUST_EB ≻ FLAT ≻ TRUST05 ≻ PB_BK ≻ raw U (ruin)`.**
Raw unified (w=1) over-stakes the model's bad 1X2 and goes bankrupt; flat-0.5 still bleeds on
1X2; curation (abstain 1X2, half-trust totals/BTTS) wins by a mile; the EB fit lands between
flat and curated because it *learns* the curation over the season rather than being handed it.

### EB trust w-trajectory — the money shot (identical at c=0.02 and c=0; w-fit is commission-blind)

| match | home | draw | away | over_15 | over_25 | over_35 | btts_yes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (cold) | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 |
| 101 | 0.294 | 0.389 | 0.461 | 0.510 | 0.469 | 0.484 | 0.597 |
| 201 | 0.209 | 0.384 | 0.299 | 0.530 | 0.454 | 0.478 | 0.580 |
| **251 (final)** | **0.180** | 0.374 | **0.330** | 0.546 | 0.520 | 0.503 | **0.638** |

Pooled `w̄0` drifts +0.56 → 0.44/0.38 as the bad 1X2 evidence accumulates. **Home w 0.50→0.18,
away 0.50→0.33 (both FALL hard); over_15/25/35 hold ≈0.50–0.55; btts_yes rises to 0.64.**
→ **The EB alarm pulls trust off exactly the markets b21 flagged bad and holds the good ones.**
Plots: `plots/p1_eb_trust_trajectory.png` (money shot) + `plots/p2_wealth_race.png` — regenerate
with `save_plots(out02.rs)` (needs `ENV["GKSwstype"]="100"` set BEFORE `using Plots`, else GR
`savefig` hangs headless). PNGs live server-side; not in the local commit because the Hetzner box
can't push (see §Infra note) — the data is in `results/w_trace_c020.csv` + the tables above.

### Per-family P/L attribution (net staked £/unit, linear; c=0.02)

- FLAT: 1X2 +0.01 (roi +0.3%), totals +0.40 (+13.3%), BTTS +0.34 (+32.4%).
- CURATED05: 1X2 +0.06 (only residual from the renorm, w=0 ⇒ ~no 1X2 bets), totals +1.14,
  BTTS +2.73 — its edge is totals+BTTS, 1X2 exposure ≈ removed.
- TRUST_EB: 1X2 +0.71, totals +0.81, BTTS +2.67 — still carries some 1X2 early (before w falls).
  BTTS is the dominant profit centre for every unified strategy (roi ≈ +25–35%).
(Attribution is additive net-P/L; joint log-growth is not family-separable — read directional.)

### b21 adapter cross-check — PB_BK_cap02 per-selection ROI vs b21 `src_sup40_sw40`

**11/11 sign agreement** (c=0.02): home −7.3% (b21 −9.4), draw +17.6 (+33.5), away +24.9
(+22.7), over_15 −0.3 (−7.5), under_15 +25.6 (+42.0), over_25 +0.7 (+1.8), under_25 +17.1
(+12.2), over_35 +23.6 (+16.0), under_35 +8.0 (+9.0), btts_yes +33.7 (+32.6), btts_no +58.8
(+48.2). Same signs + same ballpark magnitudes ⇒ the adapter prices/settles like the certified
r21 pipeline. (11/11 also at c=0.)

## Verification checklist

1. Solver / include sanity — `build_real_books` + race run clean after includes. ✔
2. **b21 cross-check** — PB_BK_cap02 per-line signs match b21 (11/11). ✔
3. **Smile w=1 tilt** — post-IPF grid reproduces the smile PPD over-probs to
   `max|Δ| = 1.9e-9 < 1e-6` (needs `cycles=50`: 3 nested over-constraints overlap 1X2/BTTS;
   the sim default 10 under-converges to ~1e-3 — immaterial to stakes but fixed for hygiene). ✔
4. **Missing-line handling** — match 1 has no Betfair BTTS line ⇒ `d(btts)=1.0`, `q=model`,
   zero stake on BTTS; stakes only on present selections. ✔
5. **MVP question answered with numbers** — final EB w: home 0.18 / away 0.33 (< 0.5) vs
   over/btts 0.50–0.64 (≥ 0.5). ✔

---

## Reads / verdict

- **The staking layer transfers from sim to real.** The E4 verdict — *curated per-line w ≻
  EB-learned ≻ flat 0.5 ≻ raw model* — reproduces on 275 real Ireland matches with the actual
  `src_sup40_sw40` engine and Betfair close books. Raw unified Kelly at w=1 **bankrupts**
  (ruin) by over-trusting the model's bad 1X2; the vig-moat abstention (w→0 on 1X2) is what
  turns the model into a profitable book.
- **The EB alarm works on real data.** With no hand-tuning it drove home/away trust to
  0.18/0.33 and held totals/BTTS at ≈0.5+, i.e. it *discovered* the curation. It underperforms
  hand-set CURATED only because it pays a learning cost (bets some 1X2 in the first ~100
  matches before the evidence lands). That's the intended role: a junk-line alarm, not an oracle.
- **The smile pricing matters and is faithfully carried.** O/U priced by Λ=λ_tot·φ (not the
  plain grid) shifts totals ~2pp; imprinting it via the IPF tilt keeps every unified strategy
  consistent with the pricing b21 certified (cross-check 11/11).
- **Caveat — sample.** 275 matches, one league, one season: signature-reading. G/match SEs
  overlap zero for several strategies; CURATED's +0.012 ± 0.004 is the cleanest signal. Don't
  over-rank on terminal W (path-dependent, single realisation).

## Infra note (update memory)

The Hetzner kaimon box has **no git push credentials** (public-repo fetch works; push has no
credential helper / token / gh / ssh key → `could not read Username`). The documented
"commit results server-side → push → pull locally" loop is currently broken. Workaround used:
run + serialize results on the server, transfer plots via base64, **commit everything from
local** (which has working push creds). After pushing from local, `git reset --hard
origin/…` on the server to realign. See memory `server-file-sync-workflow`.

## v2 — Extended multi-market book (`l02_real_ext_book.jl` + `r02_ext_book.jl`)

Generalized the core-11 book to EVERY market family we can price off the score grid. The one new
primitive is `sel_payoff(mn, sel, ml, h, a, d)` → net-return-per-state (win d−1 / lose −1 /
**push 0** / **AH-quarter = mean of its two component lines**), driving BOTH the model return
matrix `R` (over the 144 grid states) and settlement (at the real score). It mirrors
`grade_selection` extended from a Bool to a Float. CorrectScore/DoubleChance/DrawNoBet/AsianHandicap
are deterministic grid functions, so they're priced off the SAME coherently-tilted grid and staked
jointly by the unified Kelly — **no new trust units, no IPF conflicts.** The tilt is extended to the
full smile O/U ladder (over_05..over_45), reproduced to **7.5e-10** (verified).

**Betfair coverage (Ireland OOS, 275 matches):** only 1X2 + O/U(0.5–5.5) + BTTS + **CorrectScore**
have books. **AsianHandicap and DrawNoBet are absent; DoubleChance has no OOS liquidity.** The
AH/DC/DNB code is correct and auto-activates on any league/match that carries them. Book size:
min 3 / median 15 / max 36 selections per match.

**Result (c=0.02; `results/e_ext_summary_c020.txt`, and c=0):**

| strategy | term_W (with CS) | CorrectScore ROI | term_W (CS excluded) |
|---|---:|---:|---:|
| CURATED05_U_cap02 | 3.08 | −16.9% | **34.07** |
| TRUST_EB_U_cap02 | 0.74 | −26.8% | 5.16 |
| FLAT_1pct | 0.36 | −10.4% | 2.13 |
| TRUST05_U_cap02 | 0.20 | −27.3% | 1.19 |
| U_cap02 (w=1) | 0.01 (ruin) | −25.5% | 0.01 (ruin) |

**Two findings:**
1. **The extended O/U ladder ADDS value.** With CorrectScore excluded, CURATED hits **34.1× > the
   core-11 book's 26.8×** (71× at c=0) — the extra 0.5/4.5/5.5 totals lines are profitable
   (U_cap02 totals ROI +54%). The generalized engine is correct and worth keeping.
2. **CorrectScore is a systematic P/L drag.** Negative ROI for EVERY strategy (−10% to −27%); the
   model has **no exact-score edge vs Betfair**. Staking it (turnover 8–13 units) destroys the
   totals/BTTS edge — CURATED 34.1× → 3.1×. This matches the b21 CS rows (most cs_* lines negative;
   several −100% ROI). **Curate CorrectScore out, exactly like the 1X2 vig-moat abstention.**

The EB trust trajectory is identical to r01 (same 7 core units) — the alarm guards 1X2/totals/BTTS
but NOT CorrectScore, because CS isn't a trust unit. **Follow-up:** give CorrectScore (and each new
family) its own EB trust unit so the alarm learns to abstain on it automatically, rather than
hand-excluding. The engine already supports this — it's a book/unit wiring change in the runner.

**Recommended book for this engine:** full O/U ladder + BTTS (bet), 1X2 curated toward market,
**CorrectScore excluded**. On Ireland this is the whole extended universe; AH/DC/DNB await a league
with liquidity.

## Follow-ups — low-trust cold start + cap sweep (`r03_cap_and_init.jl`, core-11 book)

**A. Low-trust cold start (w0=0.3) — "which markets earn trust?"** Start every unit distrusting
the model (w=0.3, defer to market) and watch which the EB fit raises. From the 0.3 baseline
(`results/e_init_w03.txt`):

| unit | 0.3 → final | Δ |
|---|---|---:|
| btts_yes | 0.30 → 0.638 | **+0.34** ↑ |
| over_15 | 0.30 → 0.546 | +0.25 ↑ |
| over_25 | 0.30 → 0.520 | +0.22 ↑ |
| over_35 | 0.30 → 0.503 | +0.20 ↑ |
| draw | 0.30 → 0.374 | +0.07 ↑ |
| away | 0.30 → 0.330 | +0.03 · flat |
| **home** | 0.30 → 0.180 | **−0.12 ↓** |

From "trust nothing", the model **earns** trust on BTTS + the whole totals ladder, a little on
draw, ~none on away, and **home is the only market that loses trust**. (The EB estimate is
empirical-Bayes/data-pooled, so the *learned* w is cold-start-invariant — 0.3 only reframes the
baseline + drives matches 1-25 staking. The content is the ranking, which cleanly separates the
markets the model has edge on from the ones it doesn't.)

**B. Cap sweep (Σa ≤ cap) — growth/risk curve** (`results/cap_sweep.csv`, `e_cap_sweep.txt`):

| cap | CURATED W | CURATED G | CURATED maxDD | TRUST_EB W | PB_BK W |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 5.7 | +0.0064 | 0.40 | 1.01 | 1.78 |
| 0.10 | 12.4 | +0.0091 | 0.43 | 1.66 | 1.29 |
| 0.20 | 26.8 | +0.0120 | 0.42 | 3.00 | 0.57 |
| **0.30** | 36.3 | +0.0131 | 0.40 | 3.69 | 0.08 |
| **0.40** | **38.0** | **+0.0132** | **0.39** | **4.33** | ruin |
| 0.50 | 33.1 | +0.0127 | 0.39 | 4.00 | ruin |
| 1.00 | 27.5 | +0.0121 | 0.45 | 4.21 | ruin |

- **Curation makes the cap safe.** CURATED's max-drawdown holds ~0.40 across the WHOLE cap range
  (0.05→1.0) while growth peaks at **cap ≈ 0.30–0.40** (W 38×, G +0.0132), then declines past 0.5
  (mild over-betting). Removing the bad markets means aggressive sizing just compounds the
  totals/BTTS edge without blowing up drawdown.
- **Without curation the cap is a substitute for shrinkage:** PB_BK degrades monotonically to ruin
  by cap 0.4; raw U ruins from cap 0.2. Both are best at the tightest cap (0.05).
- **TRUST_EB** peaks at cap ≈ 0.40 (W 4.3×) but at ~0.88 drawdown — it still carries 1X2 early,
  so it can't be sized as fat as hand-curated. FLAT_1pct is cap-invariant (≈7 bets × 1%).
- **Operating point:** curated per-line w + **cap ≈ 0.3–0.4** (growth-optimal, drawdown ~0.40).
  Confirms [[portfolio-kelly-partial-hedge]] on real data: the cap only pays off once the junk
  markets are curated out; on the raw book it just accelerates ruin.

## Trust-learning dynamics — learning rate × refit cadence (`r04_wlearn_cadence.jl`)

Two hyperparameters of the EB trust *estimator* were fixed arbitrarily in r01 (full memory, refit
every 25 matches). r04 sweeps both learning-rate mechanisms against the **weekly** refit cadence
(Ireland ≈ 5 matches/week; refit on ISO-week boundaries, robust to the 97-day off-season gap),
evaluating TRUST_EB. `run_real_season` gained backward-compatible `halflife`/`ema_alpha`/`refit_at`
kwargs (the `(H=∞, α=1, ~5wk)` cell reproduces r01 exactly: termW 2.999, home_w 0.180).

**Grid A — forgetting half-life H (weeks) × cadence K.** Full memory dominates every cell:

| | K=1 | K=4 | K=8 |
|---|---:|---:|---:|
| **H=∞** termW / home_w | 3.71 / 0.19 | 2.94 / 0.18 | 3.83 / 0.18 |
| H=4wk | 1.27 / 0.44 | 1.08 / 0.38 | 1.60 / 0.38 |
| H=1wk | 0.93 / 0.51 | 0.90 / 0.45 | 1.09 / 0.45 |

**Grid B — EMA step-size α × cadence K.** α=1 (no smoothing) best-or-tied; smoothing cuts jitter
but lags:

| | termW (K=1) | termW (K=8) | jitter (K=1) |
|---|---:|---:|---:|
| **α=1.0** | 3.71 | 3.83 | 0.0152 |
| α=0.5 | 3.57 | 2.66 | 0.0083 |
| α=0.15 | 3.12 | 1.34 | 0.0038 |

**Verdict — the per-line bias is STATIONARY** (confirms the staking-sim E2 "time-decay = no-op" on
real data):
1. **Full memory (H=∞) dominates** — shortening the half-life strictly hurts growth (termW 3.7→0.9,
   G +0.005→−0.000), drawdown (0.86→0.94) *and* reactivity. Counter-intuitively, forgetting makes
   the alarm react **less**: a short effective sample makes the EB pool each unit back to the 0.5
   prior, so home w stalls at 0.44–0.51 instead of falling to 0.18. Full memory crosses home-w<0.35
   at match 75; short memory **never** crosses. Reactivity to a *persistent* bias comes from
   *accumulating* evidence, not forgetting.
2. **No smoothing (α=1) is best-or-tied.** EMA is the real *stability* knob (jitter 0.031→0.004,
   a beautifully smooth w-path) but it **lags** the persistent signal and costs growth as cadence
   slows. Use it only if you want a calmer path, and keep cadence frequent if so.
3. **Cadence barely matters at full memory** — termW 2.9–3.9 across K=1..8, maxDD ~0.86 flat. Refit
   anywhere from weekly to 8-weekly.

**Operating point:** full memory (H=∞), no EMA (α=1), refit every ~4–6 weeks — i.e. the r01 default
is already near-optimal; the estimator doesn't need tuning because the bias is stable.
(`results/e_wlearn.txt`, `wlearn_halflife.csv`, `wlearn_ema.csv`.)

## v2 backlog (remaining)

Per-family EB trust units for CS/DC/AH (auto-abstention, above); `U_UMC`/`TRUST_UMC` k* shrinkage
overlay; partial-hedge φ overlay; growth-fit w (vs EB); Bet365-anchored `q_mkt` variant
([[betfair-vs-bet365-market-anchor]]); block-bootstrap CIs on terminal W; multi-league (718 First
Division, Veikkausliiga — and any top division that carries AsianHandicap) once those engines exist.
The registry is pluggable: a new staking system is one `elseif`; a new market family is automatic
via `sel_payoff` + Betfair coverage.
