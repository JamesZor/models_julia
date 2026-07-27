# kappa_def — team conversion residuals beyond the player-rating projection

Research stream testing whether the goals-vs-xG conversion residual (κ) should carry
**per-team defensive structure**, not just the current attack-only factor.

> **Context for a fresh session/LLM:** read this file + `EXPERIMENTS.md` first. The base
> engine is the src no-market double-Poisson goals+xG outfield-player model
> (`src/models/pregame/engines/player_level/time_decay/outfield_xg_double_poisson_no_market.jl`);
> the loader here (`l01`) is that engine with the κ layer swapped out. Market pillar is
> deliberately **OFF** for the whole stream.

## Why (the motivating asymmetry)

In the player-rating engines, `att` and `def` are NOT Maher latents — they are two linear
projections of **one general player-rating sum per team**:

```
att_h = w_G_att·G_h + w_Outfield_att·O_h
def_h = w_G_def·G_h + w_Outfield_def·O_h     ← SAME G_h, O_h, different global weights
```

So a team that defends/keeps goal better than its general rating predicts has **nowhere to
put that** — the projection can't represent a team-specific defensive residual. The only free
per-team latent in the goal layer is κ, and today κ:

- is multiplicative on the **goals pillar only** (`λ_goals = κ[team]·exp(log_λ)`, xG pillar
  uses the raw rate) ⇒ κ is exactly the **xG→goals conversion residual**;
- is indexed by the **attacking team only**.

Every team gets an attacking finishing-residual; no team gets a defensive/keeper one
(conceded-goals suppressed beyond xG-conceded). That's the gap this stream tests.

## The κ ladder (all in `l01`, one struct, `kappa_mode` switch)

| mode | name | goals rate | free per-team params (n teams) |
|---|---|---|---|
| `:attack_only` | **V0 control** | `λ_h = κ[home]·exp(log_λ_h)` — the exact current `HierarchicalTeamKappa` (softplus multiplicative) | n |
| `:net` | **V2 Dixon-Coles-style net** | `λ_h = exp(log_λ_h + κ0 + τ·(δ[home]−δ[away]))` — one net conversion strength per team | n |
| `:attdef` | **V1 full Maher split** | `λ_h = exp(log_λ_h + κ0 + τ_att·z_att[home] − τ_def·z_def[away])` | 2n |

All modes: xG pillar untouched (raw `exp(log_λ)`), so κ0 (global log-conversion) is identified
by the goals/xG contrast, NOT confounded with the intercept (which enters both pillars).

**Identifiability notes:**
- `:net`: only *differences* δ[i]−δ[j] enter ⇒ the δ mean is a flat direction; we **centre**
  `δ .- mean(δ)` in-model (smooth, AD-safe) — the DC sum-to-zero convention.
- `:attdef`: the z_att/z_def means are confounded with κ0 ⇒ both are **centred** in-model.
- Non-centred parameterization (τ·z with z~N(0,1)) everywhere; τ ~ half-Normal(0, 0.10)
  (±10% team effects — deliberately tight; the σ-hierarchy null taught us 11-team leagues
  can't support loose per-team scales).

## Gates (in order)

1. **r00 persistence gate (EDA, no MCMC):** does a team's `conceded − xG_conceded` residual
   persist (within-season split-half, season t→t+1)? If defensive residuals don't repeat,
   κ_def is noise no matter how well it converges — the [[hierarchical-smile-sigma-null]]
   lesson. Attack residuals computed too as the reference/positive control.
2. **r01 shakedown (single split, Ireland):** all three modes converge (R-hat ≤ 1.01 incl.
   the NEW raw params — `check_convergence`'s curated df DROPS unfamiliar params, so r01
   reads the raw chains directly); inspect per-team κ posterior (mean, std, spread,
   att-def correlation).
3. Only if 1–2 pass: full-CV on 718 (more dispersion) and judge **vs the market**
   (LPD/GLMEdge per line, r13/r14 pattern from `split_market_pillar`), never vs goals alone.

## Conventions / gotchas carried from other streams

- **Market OFF everywhere here.** The split-pillar stream proved market anchors leak tension
  *into* κ (the r03 constraint artifact — κ spread under half-anchoring is fake). Testing κ
  structure with a market pillar on would confound exactly what we measure.
- xG Gamma pillar: floor present xG at 1e-3, mask missing; sanitize `xg_rate` separately from
  λ ([[xg-pillar-nan-and-sampler-gotchas]], [[outfield-xg-engine-gotchas]]).
- Sampler: sampled scales only (no fixed tight σ — stall risk), depth 10, 1000/500×4.
- Server: 16 pinned cores; 3 variants × 4 chains = 12 concurrent chains via `@sync`/`@spawn`
  (r03 pattern; per-experiment queues hold 4 items each, no oversubscription).
- Local edits reach the server only via `git push` → `git pull` on the server + REPL restart.
- Prediction dispatch: the loader struct is not in the src score-matrix Unions ⇒ l01 ships
  explicit `extract_params`/`compute_score_matrix` overrides (Poisson route), like l02 did.

## Files

- `NOTES.md` — this file (design + gotchas).
- `EXPERIMENTS.md` — experiment log/tracker: update it every run.
- `r00_persistence_gate.jl` — EDA gate: per-team attack/defense conversion-residual
  persistence, Ireland + 718.
- `l01_kappa_def_models.jl` — loader: `KappaDefDoublePoissonModel` (3 κ modes, market OFF)
  + extractor + prediction overrides + `kappa_team_summary` helper.
- `r01_single_split_shakedown.jl` — Ireland single split, 3 modes in parallel, convergence +
  κ inspection.
- `r03_grid_backtest.jl` — full-CV backtest grid (r15 pattern): kd_none_src control vs
  kd_net vs kd_attdef; GLMEdge/LogLoss/LPD vs Betfair close + BayesianKelly tearsheet +
  pooled-τ across splits. Run AFTER the r00/r02 gates.

## Related memories / streams

[[split-market-pillar-findings]] (κ artifact mechanism), [[hierarchical-smile-sigma-null]]
(per-team params collapse on thin leagues), [[first-division-718-signature]] (the follow-up
league), [[calibrate-centre-edge-in-tails]] (judging philosophy), `split_market_pillar/NOTES.md`
(the full convergence saga incl. why sampled scales / why ZeroSumTeamKappa once failed in a
market-on context).

## Findings log

### 2026-07-02 — r01 shakedown (Ireland, single split, 1000/500×4, depth 10): V1/V2 sample superbly, V0 control blows up; no per-team structure on 79

| mode | max κ R-hat | ESS (κ params) | att_spread | def_spread | attdef_cor | κ0_conv |
|---|---|---|---|---|---|---|
| V0_attack_only (softplus mult.) | **1.53 ❌** | 14–54 | 1.19 (fake) | — | — | — |
| V2_net (log-additive, centred) | **1.005 ✅** | 3000–6500 | 0.037 | 0.037 | — | 0.975 |
| V1_attdef (log-additive, centred) | **1.002 ✅** | 2500–7500 | 0.036 | 0.037 | 0.10 | 0.974 |

**Read:**
1. **V0 (the current production κ parameterization) hit the documented base-model
   metastability** — the whole model blew, not just κ: ha.γ_team_raw means ±5–8 (σ_γ mean
   4.35!), ν_xg/p_dyn/κ all R-hat ≈ 1.52 with ESS ~15. The R-hat ≈ 1.52–1.53 cluster across
   unrelated params = ONE chain stuck in a degenerate mode (κ↑ ↔ ha↓ compensation), the
   run-to-run metastability recorded in `split_market_pillar/NOTES.md` (ESS swings 5↔268).
   Same model family converged fine in r03-B — this is initialization luck, not a new bug.
2. **The new log-additive centred parameterizations are simply better-behaved**: identical
   data, same settings, ESS ~5000 and R-hat ≤ 1.005 on every κ param. κ0 identified by the
   goals/xG contrast + tight half-Normal(0,0.1) τ + centred z = no degenerate direction.
   This is a genuine engineering finding: **if κ survives at all, it should be reparameterized
   log-additively in src regardless of the att/def question.**
3. **CORRECTED READ (after inspecting τ posteriors — the first write-up wrongly said
   "learned nothing"):** τ_net mean 0.053 [90%: 0.004–0.131], τ_att 0.064 [0.004–0.163],
   τ_def 0.063 [0.006–0.154]; P(τ>0.05) = 0.44/0.51/0.53 vs prior 0.617. The posterior has
   **barely moved off the prior** — this is *uninformative*, NOT the smile-σ null (where τ
   was actively pulled toward 0 from a wider prior). One Ireland split cannot identify
   effects of this size: the 60-day decay half-life means the goals pillar effectively sees
   ~10–15 recent matches/team. **τ_def ≈ 0.06 remains fully consistent with the data — and
   it is economically material** (a +1.6σ defensive team suppresses opponent λ ~10% ⇒
   ~2–3pp on totals lines = min_edge scale). The question stays OPEN, and the power must
   come from elsewhere: r00 (un-decayed, all seasons — sees far more data than the decayed
   likelihood) and r02/pooling.
4. **Global conversion κ0_conv ≈ 0.974 in both modes** (teams score ~2.6% below what xG
   implies, stably estimated). Small but consistent; worth remembering when reading totals
   calibration.

**Decisions:**
- V0 as *control for eval* is replaced by the src production engine
  (`DynamicDoublePoissonXGOutfieldPlayerTimeDecayNoMarketModel`) — same model,
  battle-tested convergence; don't burn time re-rolling the metastable softplus variant.
- The stream's live question moves to **718 (r02)** and the **r00 persistence gate** —
  Ireland cannot answer it (insufficient heterogeneity), 718 has the dispersion regime.
- If r00 says defensive residuals don't persist anywhere, park after r02 regardless of
  convergence (converging ≠ learning).
