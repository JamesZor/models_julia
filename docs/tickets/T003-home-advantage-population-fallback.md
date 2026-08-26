# T003 — Unmapped teams silently lose home advantage at extraction

| | |
|---|---|
| **Status** | open |
| **Severity** | medium — genuine mispricing, small blast radius, silent |
| **Area** | `src/models/pregame/engines/` (28 call sites), all `extract_parameters` |
| **Raised** | 2026-08-26, by model 01 gate 4c in `current_development/scottish_lower/` |
| **Verified on** | `DynamicGoalsTimeDecayModel`, Scottish 56+57, 24/25 |

## Summary

When a fixture's home team is absent from the fitted fold's `team_map`, extraction
substitutes **zero for the global home advantage**, not the population value. The home
side is then priced at `exp(μ + α_h + β_a)` instead of `exp(μ + γ + α_h + β_a)`.

Measured: **λ_h is 0.849x what it should be** on synthetic draws, **0.863x** on the real
gate 3c posterior (γ mean 0.1469) — a ~14% under-price on the home side, which propagates
into 1X2, totals and BTTS.

Under this model's `GlobalHomeAdvantage` component, γ is a **single global scalar with no
team dimension**. The guard that drops it is therefore not a defensible fallback choice; it
discards a known value. The attack/defence fallback in the same expression *is* correct and
should be left alone.

## Root cause

`src/models/pregame/engines/team_level/time_decay/goals.jl:154-158`:

```julia
α_h = h_idx > 0 ? dyn_nt.α[:, h_idx] : zeros(n_samples)   # correct
β_h = h_idx > 0 ? dyn_nt.β[:, h_idx] : zeros(n_samples)   # correct
γ_h = h_idx > 0 ? ha_mat[:, h_idx]   : zeros(n_samples)   # WRONG
```

**Under `GlobalHomeAdvantage`, γ has no team dimension at all.** There is a single
sampled site, `ha.γ_global`. `extract_home_advantage`
(`components/home_advantage.jl:49-54`) then does:

```julia
val = vec(Array(chain[Symbol("ha.γ_global")]))
return repeat(val, 1, n_teams)      # n_teams identical copies
```

Verified on the gate 3c chain: `ha_mat` is 2000x23 with **exactly one distinct column**,
γ posterior mean 0.1469.

So `ha_mat[:, h_idx]` returns the same vector for every `h_idx`, and the `h_idx > 0` guard
is guarding nothing. Its else-branch then destroys a quantity that was never
team-specific and never missing. This is not a subtle question about what the right
population value is — for this configuration there is no question: the value is known
whether or not the team was seen.

Contrast with `α`/`β`, where the guard is doing real work. Those genuinely are per-team,
and zero is the correct substitute because `extract_dynamics` applies a **zero-sum
constraint** (`components/dynamics/team_level/time_decay.jl:59-60`), making zero the
population mean by construction. Leave them alone.

The same wrapper pattern is used for home-advantage components that *are* team-varying, so
the fix must dispatch rather than hardcode:

| component | γ for an unmapped team |
|---|---|
| `GlobalHomeAdvantage` | `γ_global` — not a fallback, just the value |
| `HierarchicalTeamHomeAdvantage` | `γ_base`, the hierarchical mean (a genuine fallback) |
| `HierarchicalLeagueHomeAdvantage` | the league mean, or `γ_base` if the league is unknown |

Only the middle two rows involve an actual choice.

## Evidence

Gate 4c prices a fabricated fixture with an unmapped home team against the correct
population value:

```
[FAIL]  unmapped team keeps global home advantage
        γ_global DROPPED: λ_h is 0.849x the population value (max |Δ| 1.082e+00)
[PASS]  month index inert for this config
```

Gate 4a confirms everything else in the same extraction path is exact
(max |Δλ| = 2.220e-16 against an independent reference), so this is the only disagreement.

## Blast radius

Occurrences of `γ_h = h_idx > 0 ? ha_mat[:, h_idx] : zeros(n_samples)` (and the `h_id`
spelling): **28 call sites** across team-level and player-level engines, standard and
time-decay. Effectively every engine that has a home-advantage component.

The same mistake appears for **league** home advantage in the `_league` engines:

```
team_level/time_decay/goals_smile_league.jl:227    γlg = l_idx > 0 ? γ_mat[:, l_idx] : zeros(n_samples)
team_level/time_decay/goals_funnel_league.jl:253
player_level/time_decay/goals_funnel_plus_minus_league.jl:249
player_level/time_decay/goals_plus_minus_league.jl:269
```

**How often it fires.** On Scottish 56+57 across 24/25, 20 folds, 360 OOS fixtures:

| | count | share |
|---|---|---|
| unmapped home sides | 2 | 0.56% of fixtures |
| unmapped away sides | 2 | (unaffected — the away side has no γ) |
| total sides | 720 | |

Teams involved: `arbroath`, `inverness-caledonian-thistle` — sides appearing in a fold's
OOS window before they appear in its fitted history, i.e. promotion/relegation and
season-boundary folds. Rare, but concentrated exactly where a model is already weakest.

This scales with how much team turnover a segment has, and with how short the fitted
history is. It will be worse on the first fold of a season than the twentieth.

## Proposed fix

Add a population accessor alongside the existing extractors, dispatching on the
home-advantage config, e.g. in `src/models/pregame/components/home_advantage.jl`:

```julia
population_home_advantage(chain, ::GlobalHomeAdvantage, n_samples)            # γ_global
population_home_advantage(chain, ::HierarchicalTeamHomeAdvantage, n_samples)  # γ_base
population_home_advantage(chain, ::HierarchicalLeagueHomeAdvantage, n_samples)
```

then replace the 28 fallbacks with it. Do **not** change the `α`/`β` fallbacks.

Consider whether an unmapped team should be loud rather than silent. Gate 2 already
counts them, but nothing in `src` reports that a fixture was priced off a fallback. A
`fallback_used` column on the extracted DataFrame would let backtests exclude or flag
those fixtures instead of silently trusting them.

## Reproduction

```julia
include("current_development/scottish_lower/01_team_poisson/v01_walkthrough.jl")  # blocks 0-4
tp_gate4c = tp_gate_extraction_fallbacks(tp_engine, tp_features[1])
sl_gate_table("4c. Extraction fallbacks", tp_gate4c)
```

## Acceptance criteria

- [ ] Gate 4c passes: `max |Δ| <= 1e-12` between the unmapped-team price and the
      population price. For `GlobalHomeAdvantage` this means the unmapped price must be
      **identical** to a mapped one, since γ does not vary by team.
- [ ] Gate 4a still passes at `max |Δλ| <= 1e-12` — mapped teams must be unaffected.
- [ ] `α`/`β` fallbacks unchanged (still zero, still correct).
- [ ] All 28 call sites updated, including the four league-level `γlg` sites.
- [ ] All 403 package tests pass.

## Scope guard

Do not change any model's mathematics, priors, or components. This changes only what is
substituted for a team the fold never saw. Prices for mapped teams must be bit-identical.

Do not fold in T002 (AD performance) — different concern, different files.
