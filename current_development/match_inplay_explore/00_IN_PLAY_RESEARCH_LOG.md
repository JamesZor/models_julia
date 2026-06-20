# In-Play Modelling — Consolidated Research Log

*A single map of everything built and learned in `current_development/match_inplay_explore/`, with the
maths, logic, results and insights. Read this first; the per-topic reports go deeper.*

---

## 0. The big picture

**Goal.** Turn live Betfair in-play odds into market-implied scoring rates, build an independent in-play
goal-intensity model, and see whether any of it (a) is well-calibrated and (b) makes money.

**One-line verdict.** We built a **well-calibrated, uncertainty-aware in-play fair-value model**, but it
has **no tradeable edge** (the exchange reprices faster than the signal can execute). The model is the
asset; the betting bot is not.

### File map
| File | Role |
|---|---|
| `l01_inplay_inverse.jl` / `r01_inplay_runner.jl` | Invert in-play odds → market-implied remaining-λ; clock alignment; game state; decay EDA |
| `l02_inplay_intensity.jl` / `r02_inplay_intensity_runner.jl` | Frequentist Poisson-GLM intensity model |
| `l03_inplay_turing.jl` / `r03_inplay_turing_runner.jl` | Bayesian (Turing) version + configurable hierarchical effects |
| `l04_backtest.jl` | In-play betting backtest |
| `r05_inplay_calibration_compare.jl` | Game-state model comparison by Over/Under calibration |
| `l06_momentum_feature.jl` / `r06_momentum_compare.jl` | Causal SofaScore-momentum covariate |
| `r07_heavytail_diagnosis.jl` | NegBin heavy-tail test → OU "under-prediction" was sampling noise |
| `l07_cv_harness.jl` / `r08_cv_compare.jl` | Repeated k-fold CV: mean ± SE + **paired** spec comparison |
| Reports: `in_play_backtest_report.md`, `game_state_calibration_report.md`, `momentum_feature_report.md`, `heavytail_diagnosis_report.md` | Deep-dives |

Pipeline: **odds → λ_rem (l01) → long-format dataset (l02) → GLM (l02) / Bayesian (l03) → calibration (r05) / backtest (l04) / momentum (l06)**.

---

## 1. Stage 0 — feasibility (the foundations, verified on the server)

- **Betfair clock sign.** `ds.betfair_odds.minutes_to_kickoff = (tick − kickoff)/60000`. Despite the name,
  **positive = in-play (after kickoff)**; the in-play window is `(0, ~125]`. Match-minute 90 lands at
  wall-clock ~105–118 (half-time + stoppage baked in).
- **Exchange vig ≈ 0.** Per-market overround `Σ 1/price ≈ 0.997–1.008`. We still normalise per market.
- **Coverage.** 976/977 Ireland matches have in-play ticks; median 690 ticks and 33/36 selections per match;
  correct-score ≈ 20% of ticks.

---

## 2. The inverse problem (l01) — odds → market-implied remaining-λ

**Idea.** At each time-bin, find the home/away scoring rates for the *rest of the match* that best explain
the live market prices, **conditioned on the current score**.

### 2.1 Clock alignment (the hard part)
The feed is wall-clock; events (`ds.incidents`) are match-minute; stoppage time is not stored.
- **Game state** = cumulative goals & red cards by match-minute from `ds.incidents`.
- **Goal-jump anchoring.** A goal makes the scoring team's 1X2 price *drop*; we locate each goal's
  wall-clock time as the largest signed log-return near its expected position, then assign the live score
  per tick from those anchors.
- **Structural clock map** (wall-clock → match-minute): two segments with the half-time gap baked in
  (`off1` from first-half anchors, `off2 = off1 + ~15 min`), so it's correct even when all goals are in one
  half. (See memory `betfair-inplay-clock-map`.)

### 2.2 Per-bin fair probabilities
LOCF within a ~10-min staleness window → strip vig per market group:
`prob_fair = (1/price) / Σ_market(1/price)`. Require a full 1X2 (≥6 selections) to identify the home/away split.

### 2.3 Score-conditioned inversion (the maths)
Model the **remaining** goals as a scoreline matrix `P[i,j]` (independent double-Poisson, `max_goals=8`).
Final score = current `(g_h, g_a)` + remaining `(i, j)`. For each market selection we compute a model
probability and fit `(λ_h, λ_a)` by minimising SSE to the fair probabilities (Nelder-Mead):

- Over line L: `P(Σ remaining ≥ (L+1) − (g_h+g_a))`
- 1X2: compare `(g_h+i)` vs `(g_a+j)`; BTTS: `(g_h+i>0) ∧ (g_a+j>0)`; CS `i,j` exact: `P[X_h−g_h, X_a−g_a]`.

Two arms: **conditioned** (current score) and **naive** (score = 0-0). Decay metrics:
`λ_rem(t)` (remaining expected goals) and the detrended per-90 rate `μ(t) = λ_rem · 90/(90 − t_m)` (NaN for t_m ≥ 80).

### 2.4 Validation
- **λ_rem is calibrated**: bucketed conditioned `λ_rem_h` vs actual remaining home goals → 0.5→0.50,
  1.0→0.97, 1.5→1.53. Mean bias **+0.013**.
- **μ@kickoff correlates 0.77/0.71** with the independent pre-game Bayesian λ.
- **Conditioning is essential**: the naive arm is biased **+0.73 goals** mid-match.
- **Gotcha:** ~0.2% degenerate Nelder-Mead fits blow up (λ → 100s) when the market is nearly settled and
  *pass* the residual filter → always filter `λ_rem < 6` and aggregate by **median**.

---

## 3. EDA — decay & game state (r01)

- **Mechanical decay.** Total `λ_rem` falls monotonically: **2.39 (0–10′) → 0.41 (80′+)**.
- **Detrended hazard rises late.** At level state, no reds, total `μ` is flat ~2.6 early then **3.1 (60′),
  3.7 (70′)** — the market prices an increasing late-game hazard.
- **Score effect (median μ).** Home trailing by 1 → 1.67 (vs level 1.44) — trailing teams attack more.
- **Red cards (man advantage).** Down a man → own/opp μ = **1.25 / 1.76**; up a man → **2.42 / 0.91**. Textbook.

---

## 4. Frequentist intensity model (l02)

**Form.** Poisson GLM, **log link**, **time-exposure offset** — long format, one row per (match × bin × side):

```
rem_goals_side ~ Poisson( λ_inst · rem_frac )
log λ_inst = α + f(t_m) + β_home·is_home + β_trail·trailing + β_lead·leading + β_man·man_adv + γ·log(pregame_λ_side)
offset = log(rem_frac),  rem_frac = (90 − t_m)/90
```
Target = **realized remaining goals** (ground truth, independent of live odds). The offset removes the trivial
"less time left" effect so `λ_inst` is a per-90 intensity.

**Coefficients** (effects significant): **trailing +0.25, leading −0.24** (score-effect spread ≈ 0.43),
man_adv +0.18, **log_pregame slope ≈ 1.26** (quality dominates); time-trend & home wash out (absorbed by the
offset / pregame λ).

**Held-out log-score** (higher better): **model −1.0245 > market λ_rem −1.0352 > pregame-only −1.0344** — i.e.
using *no live odds*, the smooth model slightly **beats the market's own noisy inversion** at predicting
realized remaining goals. Well-calibrated by decile.

---

## 5. Backtest (l04) — does it make money? No.

**Strategy.** Model prob vs Betfair price → back-bet EV with commission → fractional-Kelly stake → settle on outcome.

**The trap & the truth.**
- Filling at the price *as-of* the bin gives **+29% ROI** — but it **scales with price staleness**
  (1-min +12% → 10-min +29%) and a 2–5% spread haircut barely dents it (claimed "edge" ~30% — implausible).
  This is **stale-price lookahead**: the model knows a goal the stale quote hasn't yet absorbed.
- **Realistic execution** (fill at the *next* available price after the signal) collapses ROI:
  **+5.3% (0.5 min) → +1.6% (2 min) → +0.8% (5 min)** ≈ noise.
- Yet the model's probabilities are **well-calibrated** (OU 0.500 = 0.500, 1X2 0.333 = 0.333).

**Conclusion: no demonstrated tradeable edge** — the market reprices faster than the signal executes. See
`in_play_backtest_report.md`.

---

## 6. Bayesian model + hierarchical experiments (l03, r03, r05)

**Why Bayesian:** posterior uncertainty (for honest Kelly later) and partial pooling. Written to
`docs/turing_ad_performance_guide.md` (broadcast + `view`, no loops, config-flag branches on constants).

`InPlayIntensityConfig` toggles, independently: **team** effects (`use_team_attack/defense/trailing/leading`,
non-centred `z~Normal(0,1); θ=z·σ`), a **game-state** mode (`:none|:linear|:hier_replace|:hier_addon`,
δ_state over goal-diff buckets ±3), and **`use_momentum`**.

**Findings.**
- **Baseline recovers the GLM** (trailing +0.33, leading −0.25, log_pregame 1.25; R̂ ≈ 1.00).
- **Team hierarchical effects hurt out-of-sample** on 11 teams / 253 matches (full −1.159 vs baseline −1.075):
  partial pooling can't rescue weakly-identified team deltas. **Use the global model.**
- **Game-state representation barely matters for OU calibration** (`none`/`linear`/`hier_*` all within
  0.004 ECE). The learned δ_state is ~**linear/monotone** (down-1 +0.33 … level 0 … up-2 −0.40), so the simple
  trailing/leading dummies already suffice. Hierarchy adds nothing here. See `game_state_calibration_report.md`.
- ~~The real OU error is structural: all variants under-predict Overs by ~5 pts…~~ **CORRECTED (§9.5 /
  `heavytail_diagnosis_report.md`):** this ~5-pt under-prediction is **test-split sampling noise**, not a
  structural bias — the model is mean-unbiased (train 1.328 vs 1.330; held-out bias +0.030±0.087 over 15
  splits). A Negative-Binomial heavy tail does not help. The single r05 split was just goal-heavy.

Odds for the calibration comparison are derived with the **project pipeline** (posterior-preserving):
`compute_score_matrix` (poisson.jl) → `compute_market_probs(S, MarketOverUnder(L − current_total))` — the
in-play tweak is the **line shift** by the current total.

---

## 7. Momentum feature (l06, r06)

**Causal** SofaScore momentum: decay-weighted net AUC over `points[1:t_m]` only (the old whole-match AUC
leaks the future). Added as a global covariate `β_mom · momentum_side`.

- All 253 modelling matches have momentum. Feature is only weakly tied to score (r≈0.085 — not just
  re-encoding game state) and modestly predicts remaining goals (r≈0.156).
- **β_mom = +0.14, 90% CI [0.09, 0.18]** (×1.15 per SD) — credibly non-zero controlling for everything.
- **Held-out count elpd improves: −1.0753 → −1.0689** (a bigger lift than any game-state variant).
- **But OU calibration is a wash** (ECE 0.062→0.059, Brier/LogLoss flat) — OU is dominated by the structural
  total under-prediction momentum doesn't fix.
- **Verdict:** keep `use_momentum=true` for the **count/intensity** model; it is **not** an OU edge. See
  `momentum_feature_report.md`. (Consistent with the pre-game finding that momentum ≈ xG and adds nothing to
  team strength beyond xG — memory `momentum-validation-findings`.)

---

## 8. Correlation / Dixon-Coles analysis (the τ question)

**Question:** should the in-play model be Dixon-Coles with a correlation τ that varies with game state?

**What the data say (realized outcomes):**
- `corr(remaining_home, remaining_away) = +0.207` overall — independence **is** violated; the total is
  **overdispersed** (var/mean ≈ 1.35).
- But the correlation is **flat across game states** (level 0.24, −1 0.23, +1 0.18, +2 0.17); the extreme
  states differ only on n=10–12 (noise).

**What the market says (invert in-play odds with `DixonColesMarketFeature` → per-bin ρ):**
- Market-implied DC ρ ≈ **0 everywhere** (overall median −0.001), within ±0.04 across populated states. The
  market does **not** price a state-varying ρ.

**Why both agree on "DC τ ≈ 0, not state-varying":** the DC τ is a narrow *low-score-cell* correction; the
real +0.2 dependence is a **broad** count correlation that τ structurally cannot represent — so the market's
DC ρ sits at zero even though the broad correlation is real.

**Pre-game vs in-play τ are different objects.** The best pre-game model (`DCMH_HalfLife_60`, Dixon-Coles with
per-team hierarchical ρ regularised toward the market's DC inverse, `market_weight=0.4`) carries per-match
ρ ≈ **−0.10** (negative — classic draw-inflation, full-match final scores). That is *not* the in-play
remaining-goal dependence (+0.2, broad). So carrying pre-game τ into the in-play model is of questionable
value.

**Serialization note.** DC pre-game artifacts in `./data/dixon_coles_ab/` **fail to deserialize**
(`DynamicDixonColesXGOutfieldPlayerTimeDecayModel` struct drifted since 2026-06-02). The newer
`./data/dixon_coles_halflife_grid/` DCMH models (saved 2026-06-04) **load fine** and give λ + per-match ρ —
use those if re-anchoring.

**Verdict:** don't make τ state-varying (unsupported by outcomes *and* market). Keep the independent
double-Poisson in-play. If you want the genuine +0.2 dependence, use a **global** copula / bivariate-Poisson
(not DC τ) — but it fattens the *tails*, not the *mean*, so it won't fix the OU under-prediction.

---

## 9. Bottom line & priorities

**What we have:** a calibrated, uncertainty-aware in-play fair-value model (independent of live odds),
plus tooling to test any strategy against it — including catching our own stale-price traps.

**What works / what to keep:**
- Independent double-Poisson intensity, global coefficients (team & game-state hierarchies don't help here).
- Pregame λ as the quality anchor; score-state (trailing/leading) effects; **momentum** for the count model.

**What doesn't:** no tradeable in-play edge; hierarchical team/game-state effects; state-varying τ.

**Highest-leverage next steps (in order):**
1. **Cross-validation is now built (`l07`/`r08`) — use it.** Repeated k-fold with mean ± SE and **paired**
   spec comparison (§9.6). It already settled the open questions; run it for any new spec before trusting a gap.
2. **Live forward-test** — the only true test of whether *any* execution-speed edge exists.
3. **(Optional) global correlation** (Frank copula / bivariate-Poisson) for the +0.2 dependence — small, and
   §9.5 shows the related heavy-tail (NegBin) effect is within noise here; revisit with more data.
4. Re-anchor on DCMH (`dixon_coles_halflife_grid` index 2 = hl60) for coherence — low priority given §8.

### §9.5 Heavy-tail / OU re-diagnosis (correction) — `heavytail_diagnosis_report.md`
Implemented a Negative-Binomial total (the thesis "intensity smile" fix). Result: conditional
overdispersion is mild (var/mean ≈ 1.09; raw 1.35 is mostly the model mean varying across bins), NegBin
barely changes OU calibration, and crucially the **~5-pt Over under-prediction was test-split sampling
noise** (held-out mean bias +0.030 ± 0.087 over 15 splits; model unbiased on train). This **supersedes**
the earlier "stoppage-time exposure / independence" explanation in §6 and the game-state report.

### §9.6 Repeated k-fold CV harness (`l07_cv_harness.jl` / `r08_cv_compare.jl`)
Reports each metric as **mean ± SE over folds** and **paired** spec differences (same folds → a t-stat),
the correct way to beat sampling noise. GLM workhorse (≈ Bayesian baseline). Ireland, 5-fold × 4 repeats:
- **Goal-COUNT held-out log-likelihood:** `+game_state` beats `pregame_only` (paired **t=3.15**); `+momentum`
  beats `+game_state` (**t=2.15**) → both are **real** signals.
- **Over/Under MARKET calibration (ECE):** game-state and momentum differences are noise / trivially worse
  (t=1.3 / t=2.28-but-worse-by-0.0013) → neither helps the OU market.
- **mean_bias ≈ 0 ± 0.03** for all specs → model unbiased (confirms §9.5).
Takeaway: the live signals (game state, momentum) pay off in the **count** likelihood, not the binary OU
market; and marginal SEs overlap (≈0.02 on count LL, ≈0.007 on ECE), so only the **paired** test resolves them.

---

## 10. Conventions & knobs (quick reference)
- In-play filter: `0 < minutes_to_kickoff ≤ 130`. Bins: 5 min default (try 2/3). Staleness (LOCF): ~10 min.
- Min selections per bin: 6 (need full 1X2). `max_goals` = 8 (inversion) / 12–13 (scoreline matrix).
- Always filter `residual < 0.06–0.08` and `λ_rem < 6`; aggregate by **median**; μ only for `t_m < 80`.
- Modelling split: 75/25 **by match**. Pregame λ available for ~276 OOS matches → ~253 modelling matches.
- Sampler: `Samplers.run_sampler(model, NUTSConfig(1000/500/4))`, `AutoReverseDiff(compile=true)`, pin 16 cores.

## 11. Thesis comparison — "Constant/Local Intensity" financial-maths model vs ours

Reference: a financial-maths in-play betting thesis (Ch 3 *Constant Intensity Model*, Ch 4 *Local
Intensity Model*). It prices/hedges in-play bets by modelling each team's score as an independent Poisson
process and invoking the Fundamental Theorems of Asset Pricing. Overlaps with us in the *maths* but differs
fundamentally in *purpose*. (Earlier §8 notes corroborated its empirical claims; this section maps the
machinery explicitly.)

### 11.1 What we share
- **§3.5 least-squares calibration** of (λ₁, λ₂) to in-play OU/1X2/CS prices, score-conditioned, per
  minute (their Eq 3.26) **= our market inversion** (`l01_inplay_inverse.jl`): vig-strip + LSQ fit of a
  2-λ double-Poisson to remaining-goal prices each tick. We omit their bid-ask-spread weighting.
- **Ch 3 European-bet pricing** (Table 3.1: home/away/draw/OU/CS = Poisson-matrix sums over remaining
  goals) **= our `model_prob`** over the remaining-goals score matrix. Identical formulas (their Prop 18).

### 11.2 What we deliberately do NOT use
- **FTAP risk-neutral framework (Ch 3):** arbitrage-free + complete market, replicate any bet by
  delta-hedging two liquid bets (Prop 23: weights = δ₁,δ₂ finite differences). We do **not** replicate.
- **Local Intensity / Dupire (Ch 4):** λ_Loc(N_t, t) engineered to reproduce the *intensity smile* →
  heavy-tailed marginals that reprice every OU strike exactly. We do **not** use it — see §11.4.

### 11.3 The fundamental difference — Q-replication vs P-forecast
The thesis works under the **risk-neutral measure Q** to *price and replicate*; its hedging **assumes the
market price is correct**, calibrates to it, then trades home/away-win to replicate a contract's δ₁/δ₂
sensitivities. By construction this **cannot make money** — it is market-making / risk-neutralisation.
We work under the **physical measure P**: `l03` is fit to **realised remaining goals** to find where our
forecast **disagrees** with the market, then we Kelly-bet / Kelly-exit the gap. Our hedging is
**log-growth-driven** (exit when model edge `P_model − 1/price ≤ −0.05`), **not** delta-replication.
⇒ They assume the market is right and neutralise risk; we assume it is sometimes wrong and trade it. This
is why our exit rule shows OOS value (t=5.21, see `../basic_hedging/portfolio_kelly_hedge_report.md`),
which their replication framework explicitly assumes away.

### 11.4 On "non-homogeneous" specifically (and why no heavy-tail layer)
- Their **homogeneous** (Ch 3): λ constant over remaining time.
- Their **Local Intensity** (Ch 4): λ depends on **total goals N_t** and t → drives the heavy-tail *smile*.
- **Ours:** constant *remaining* rate within a tick (via the `log((90−t)/90)` offset), but **re-estimated
  every tick as a function of state** → effectively time- & state-varying across the match. Crucially our
  state variable is the **goal *difference*** (sign → trailing/leading *behaviour*), NOT the **total goal
  count** (which is what produces their smile). Different state variable, different purpose; our marginals
  stay Poisson.
- We tested the heavy-tail extension (`heavytail_diagnosis_report.md`, §9.5): the apparent OU
  under-prediction was **sampling noise** (multi-seed bias +0.030 ± 0.087, n.s.); a NegBin tail barely
  moved ECE (0.062→0.062); market-implied in-play ρ ≈ 0. The smile is real on their 25k-game sample but
  within noise on our ~250-match league → a Dupire local-intensity surface would fit noise here.

**Takeaway:** we borrowed the thesis's robust *calibration + pricing formulas* but run them under **P to
forecast and Kelly-trade**, not under **Q to replicate**, and left the local-intensity heavy-tail layer on
the shelf (unsupported by our data). The thesis is the right reference for *market-making/replication*;
ours is built for *edge-finding and growth*. The full position-management work (portfolio-Kelly sizing,
model-driven exits, the add-side and distributional-sizing negative results) lives in
`../basic_hedging/` (`r01_portfolio_kelly_hedge.jl`, `r02_model_driven_hedge.jl`,
`portfolio_kelly_hedge_report.md`).
