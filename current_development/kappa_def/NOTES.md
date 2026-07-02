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

## Related memories / streams

[[split-market-pillar-findings]] (κ artifact mechanism), [[hierarchical-smile-sigma-null]]
(per-team params collapse on thin leagues), [[first-division-718-signature]] (the follow-up
league), [[calibrate-centre-edge-in-tails]] (judging philosophy), `split_market_pillar/NOTES.md`
(the full convergence saga incl. why sampled scales / why ZeroSumTeamKappa once failed in a
market-on context).

## Findings log

(append dated entries here — none yet)
