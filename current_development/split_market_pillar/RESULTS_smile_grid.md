# Smile / split-market grid — results & reference

> **Purpose:** self-contained write-up of the smile-pillar experiment (model, grid, diagnostics,
> verdict, open threads) so a future session can pick it up without re-deriving. Companion to
> `NOTES.md` (the running design log). League = **Ireland (79)**, eval market = **Betfair close**.
> Last updated 2026-06-28.
>
> **Naming:** model = two axes, **pillar** `{iso, split, smile, none}` × **dispersion**
> `{pois, nb, cmp, dc}`; reference `<pillar>-<disp>` (e.g. `smile-pois`). The `dp_*`/`li_*` cell names
> below are the saved-experiment aliases — see the **Canonical naming** table in `NOTES.md` for the full
> map. This doc covers the **pois** row (`none/iso/split/smile`-pois).

---

## 1. The question

The double-Poisson goals engine prices the whole Over/Under ladder from a single total rate
λ_tot = λ_h+λ_a (one Poisson). The per-line eval (r06/r12) showed that's too thin-tailed: the market
prices a **rising per-strike implied intensity** Λ(K) (a "smile" / local-intensity / Dupire analogy).
So: does adding a **smile pillar** that anchors the model's per-strike intensity to the market's
inverted per-strike intensity buy us calibration or, better, **edge** (per-match deviations that beat
the closing line) — and on which markets?

Tied to the project philosophy ([[calibrate-centre-edge-in-tails]], unified_kelly §6): single-bet
log-growth is **linear in E[p]**, so edge = the per-match deviation `E[p] − p_market`; the posterior
only *sizes* the stake. We therefore judge on **per-line proper scoring (LogLoss) + GLMEdge**, never
grouped backtest P/L.

---

## 2. The models

Two loader-defined engines plus the src baseline, all double-Poisson `{goals + xG + outfield-player}`:

| Loader | Engine | Market pillar |
|---|---|---|
| `src` (none) | production double-Poisson | OLD **isotropic** market weight (or off) |
| `l02_split_market_poisson.jl` | `SplitMarketDoublePoissonModel` | **anisotropic** in rotated basis: a **supremacy** ("who wins", λ_h−λ_a) anchor + a **level** ("how many", λ_h+λ_a) anchor, each with a **sampled** σ release-valve; `level_weight` sweeps the level anchor strength |
| `l03_local_intensity_poisson.jl` | `LocalIntensitySmileDoublePoissonModel` | keeps the **supremacy** anchor, **replaces the level anchor with a per-strike SMILE pillar**: global shape `log_φ ∈ ℝ^{Kmax+1}` (φ≡1 ⇒ Poisson); per strike K it anchors `log(λ_tot)+log_φ(K)` to the market-inverted `Λ^mkt(K)` (Poisson-CDF inversion of the de-vigged fair O/U, done **off the AD path**). σ_smile is sampled. φ is a **pricing object** — it does NOT enter the goals likelihood; per-line O/U is priced with its own intensity `P(N≤K)=cdf(Poisson(λ_tot·φ(K)),K)`, while 1X2/BTTS/correct-score still come from the unchanged (λ_h,λ_a) grid. `Kmax=4` (strikes 5,6 are thin/selection-biased per r07). |

**Shared config (identical across all cells, for comparability):**
`HierarchicalMonthlyInterception`, `HomeAwayDispersion`, `HierarchicalTeamHomeAdvantage`,
`HierarchicalTeamKappa`, `PlayerRatingsFeature(BayesianTracker(6.5,1.0,0.5,0.01))`,
`OutfieldPlayerDynamicsConfig(days_half_life=60)`, `DoublePoissonMarketFeature`,
`MarketSmileFeature(Kmax=4)`. Sampler: `samples=800, warmup=300, chains=4, max_depth=10`,
`target_seasons=["2025","2026"]`, `history_seasons=2`, `dynamics_col=:match_biweek`. Train pillar from
SofaScore `ds.odds`; **eval CLV vs Betfair** `ds1` (no leakage).

### Grid cells (11)

| cell | loader | spec |
|---|---|---|
| `dp_nomarket` | src | structural only, market OFF |
| `dp_old_mw50` / `dp_old_mw100` | src | OLD isotropic market, weight 0.5 / 1.0 |
| `dp_split_lw0/25/50/100` | l02 | split (sup+level), `level_weight` = 0 / 0.25 / 0.5 / 1.0 |
| `li_sup_only` | l03 | supremacy=1, **smile=0** (smile-off control) |
| `li_smile50` / `li_smile100` | l03 | supremacy=1, **smile=0.5 / 1.0** |
| `li_smile_only` | l03 | **supremacy=0**, smile=1.0 (totals-smile alone) |

---

## 3. Files (what ran, in order)

| File | Role |
|---|---|
| `l02_split_market_poisson.jl` | `SplitMarketDoublePoissonModel` definition (needed to deserialize the split cells) |
| `l03_local_intensity_poisson.jl` | `LocalIntensitySmileDoublePoissonModel` definition (needed to deserialize the smile cells) |
| `r10_grid_search_smile.jl` | **Trains** the smile grid → `./data/double_poisson_smile_grid/`; loads the r05 grid (`./data/double_poisson_market_grid/`) alongside; eval = overall GLMEdge / LogLoss / Kelly tearsheet. Output pasted in-file. |
| `r12_per_line_bias_edge.jl` | Per-line **bias (t)** + **GLMEdge(coef,p)** on the **dp_\*** grid only (no retrain). The diagnostic template. |
| `r13_per_line_bias_edge_smile.jl` | Same per-line diagnostic across **all 11** cells (loads both grids) + a **cross-model GLMEdge matrix** (line × model). |
| `r14_pooled_totals_edge.jl` | **Pooled** over/under ladder GLMEdge per model: `is_over ~ line_FE + market_logit + spread`, **SE clustered on `match_id` (CR1)** to defeat pseudo-replication; reports `p_naive` vs `p_cluster`; runs OVER + UNDER (unders = algebraic mirror = symmetry check). |

Saved experiments: `./data/double_poisson_smile_grid/` (li_\*), `./data/double_poisson_market_grid/` (dp_\*).

**Re-run (server):** `git pull`, restart REPL, then
`include("current_development/split_market_pillar/r13_per_line_bias_edge_smile.jl")` (diagnostics, no
retrain) or `r10_...` (full overnight retrain).

---

## 4. Results

### 4a. Overall (r10) — smile tops BOTH summary tables

Rare: GLMEdge (residual edge) and LogLoss (calibration) usually trade off; here the smile leads both.

| metric | best cells |
|---|---|
| **GLMEdge** spread_fair_coef (p) | **li_smile50 1.89 (p=0.0002)** · li_smile100 1.85 (0.0007) · dp_nomarket 1.76 (0.0007) |
| **LogLoss** diff vs market | dp_split_lw100 −0.0246 · **li_smile100 −0.0245** · li_smile50 −0.0240 |

`li_smile50` has the single best GLMEdge in the grid while joint-top on LogLoss → it adds *informative*
deviation, not noise.

### 4b. Per-line (r13) — decompose by market family

- **BTTS = the only family with significant per-match GLMEdge.** `btts_yes`: li_smile50 **7.24 (p=0.01)**
  — best in grid — vs li_smile_only 6.34 (0.05), li_smile100 5.96 (0.057), dp_nomarket 8.83 (0.039),
  dp_split_lw100 10.41 (0.093). **`li_sup_only` is dead on BTTS (0.35, p=0.85)** → the **smile pillar**
  creates the BTTS edge, the supremacy axis doesn't. Bias *grows* with smile weight (li_smile50 t=5.3 <
  li_smile100 8.8 < li_smile_only 9.5) → **lighter smile = more centred + keeps the edge.**
- **Totals (over/under) — no per-LINE cell clears p<0.1**, but every smile cell is **systematically
  biased** (compression toward under: li_smile50 over_15 t=−6.7) AND the coefs are **consistently
  positive** across the ladder (li_smile50: 4/4 liquid over-lines positive). At n≈210–250/line the test
  is **under-powered**, so per-line p>0.1 is *not* a null (see 4c).
- **1X2 (home/draw/away) — no edge for anyone** (all p>0.25; the lone `away` dp_split_lw50 p=0.055 is
  one multiple-comparison-fragile cell). `li_sup_only` totals rows are wildly biased (no totals pillar)
  → it's a 1X2-only cell.

### 4c. Pooled totals ladder (r14) — the power-aware test

Stack the ladder into one logit per model with line FE + market control; SE **clustered on match_id**.

| | li_smile50 | runners-up |
|---|---|---|
| **OVER CORE** (1.5/2.5/3.5) | coef **3.15**, p_naive 0.067, **p_cluster 0.20** | li_smile100 3.08 (0.31) · dp_split_lw100 3.38 (0.38) |
| **OVER FULL** (+0.5) | coef **2.88**, p_naive 0.049, **p_cluster 0.14** | dp_split_lw100 3.29 (0.33) · li_smile100 2.78 (0.33) |
| **UNDER** (mirror) | identical (CORE 3.15/0.20, FULL 2.85/0.14) | — symmetry check **passes** |

`li_smile50` is the **clear frontrunner** (top coef + rank both ladders, robust over↔under). The
clustering **knocked its naive p (0.049) back to 0.14** — exactly the pseudo-replication we built it to
catch (4 ladder rows per match = the same goal total at different strikes, correlated). So: a
**positive, best-in-grid totals coef that one league's n cannot certify** (p_cluster ~0.14–0.20). Two
under-powered-but-aligned signals (consistent-positive coefs + positive backtest P/L) say the edge is
**real but small**, not a bias artifact.

*(Mirror caveat: `prob_fair_close` is de-vigged per selection so over+under fair probs don't sum to
exactly 1 → `spread_under` ≈ −`spread_over` but not exactly; cells with extreme compressed probs
(dp_split_lw50/100) show small over↔under coef gaps. Not a bug. li_smile50's moderate spreads make it
invariant.)*

---

## 5. Verdict

- **The smile earns keep — as a derivative-market (BTTS + totals) tool, not a 1X2 one.** It is the first
  cell to push **BTTS to p=0.01** (vs dp_split_lw100's marginal 0.093) and leads the pooled totals edge.
- **`li_smile50` is the keeper:** best-in-grid BTTS GLMEdge **and** top pooled-totals coef, at the
  **lowest bias** of the edge-carrying cells. More smile (smile100 / smile_only) adds bias without edge;
  removing it (`li_sup_only`) kills the edge → optimum is "some smile, not much". 0.5 looks near-best;
  **lighter (0.2–0.4) likely trims the BTTS bias further** and is worth mapping.
- **Routing, not a global weight:** supremacy pillar drives **1X2**; smile pillar drives **totals/BTTS**.
  No single blended weight is good at both (smile distorts 1X2: home/away P/L worst in grid).
- **1X2 has no model edge** over the close (confirms prior findings / [[staking-research-conclusions]]).
- BTTS is the textbook [[calibrate-centre-edge-in-tails]] case: model over-predicts `btts_yes` ~2.4pp
  systematically (**recentre** that per-line) while the per-match spread genuinely predicts (**keep**
  that for staking).

---

## 6. Open threads / next steps

1. **Cross-league pool (the decisive totals test) — a future runner.** Ireland alone can't certify the totals
   edge (p_cluster ~0.14). Train the smile/split grid on each of Ireland + First-Div(718) + SouthKorea +
   Norway + Veikkausliiga, then run the r14 pooled-ladder GLMEdge with a **`league_FE`** and
   `match_id` clustering. ~5× n_match → either pushes li_smile50 past p_cluster<0.1 or honestly shows
   the edge is too small to bank. **Prereq:** only Ireland's grid is saved; this is an overnight
   per-league retrain, not a pure diagnostic.
2. **Map smile weight below 0.5 — `li_smile20/30/40`.** Add cells to r10; check whether the BTTS p=0.01
   edge survives lighter weight while the bias (t) drops. Cheap.
3. **Per-line BTTS recentring calibrator.** If li_smile50 ships, strip the ~+2.4pp `btts_yes` skew
   (L2-style per-line recentre) and keep the deviations for Kelly.
4. **Don't** chase 1X2 or read totals backtest P/L as model edge — it's a market-fade
   ([[totals-compression-is-denoising]]), real for staking but not a per-match signal.

---

## 7. Glossary for a fresh session

- **GLMEdge** `spread_fair_coef (p)`: coefficient on `spread = model_p − market_p` in
  `logit(outcome) ~ market_logit + spread`. >0 & p<0.1 ⇒ the model's per-match deviation predicts the
  outcome **beyond the market** = real edge. This is the project's edge yardstick.
- **bias (t)**: `mean(model_p − market_p)` with t = bias/se. |t|≫2 ⇒ systematic per-line skew (a
  **calibration** target, not edge). |t|≈0 ⇒ centred.
- **dev_std**: `std(model_p − market_p)` = per-match deviation budget = the size of the edge you bet.
- **p_naive vs p_cluster** (r14): naive SE treats correlated ladder rows as independent (pseudo-
  replication, optimistic); **p_cluster** (Huber–White on match_id) is the honest test.
- **level / supremacy / smile pillars**: anisotropic market anchors — supremacy = who-wins (λ_h−λ_a),
  level = how-many (λ_h+λ_a, isotropic), smile = per-strike how-many (replaces level with a rising
  Λ(K) curve).
