# Findings — Player-Level Lineup Dynamics, Bench Depth, and Multi-Tier Formulations

## Executive decision

The lineup formulations contain a small held-out signal in the broad cross-league and all-Scotland pools, but **none clears a useful predictive gate on the specified 710-match Scottish-Lower target (24/25 + 25/26)**. The target-scope result is the decision-bearing one: all eight RAPM-target/formulation combinations have negative held-out R².

Therefore:

1. **Do not promote a pure player-lineup model to production on EDA alone.**
2. Retain `OutfieldPlayerAggregation()` as the simplest experimental default.
3. If bench depth is tested in a count-model bake-off, use **`w_bench = 0.10`**, the lower boundary selected independently in every nested-history grid—not 0.25.
4. Prefer `ShotsPlusMinusFeature()` over `XGPlusMinusFeature()` for the Scottish-Lower follow-up. Its signal is still weak, but it dominates pxG RAPM on the target scope.
5. Do not prefer the higher-dimensional positional model without shrinkage/strong priors: it wins narrowly in some pools and fails most severely on the exact deployment window.

## Filtration and snapshot

All work ran locally on the laptop. No MCMC or other work was launched on `mcmc-beast`.

- **Scope A:** tiers 1/2/3/84/54/55/56/57, current local snapshot of 12,441 matches; chronological 80/20 split (9,952 history, 2,489 held out).
- **Scope B:** Scottish tiers 54/55/56/57; ratings fit through 23/24 and evaluated on 1,461 fixtures in 24/25 + 25/26.
- **Scope C:** Scottish tiers 56/57; ratings fit through 23/24 and evaluated on the specified **710 fixtures** in 24/25 + 25/26.
- Official SofaScore xG comes from the independent cached r92 pull. Zero-filled 0.000/0.000 placeholders are excluded. Tiers 56/57 have no official xG.
- RAPM, aggregation scales, and ridge calibration are history-fit. Target matches do not fit ratings or coefficients.

The Scope-A count is larger than the design document's approximate 9,000 because the current local snapshot has advanced. No rows were removed merely to reproduce a stale approximate count.

## Formulation leaderboard

### Scope A — pxG RAPM, held-out scoreline supremacy

| formulation | n | Pearson r | Spearman ρ | MAE | R² |
|---|---:|---:|---:|---:|---:|
| Positional vectors | 2,489 | **0.1656** | **0.1770** | 1.2830 | **0.0190** |
| Expected minutes | 2,489 | 0.1616 | 0.1738 | **1.2824** | 0.0185 |
| Starters + bench (0.25) | 2,489 | 0.1626 | 0.1762 | 1.2835 | 0.0179 |
| Outfield starters | 2,489 | 0.1626 | 0.1763 | 1.2834 | 0.0175 |

Differences are tiny. Positional vectors gain only 0.0015 R² over outfield starters.

### Scope A — official SofaScore-xG supremacy

| formulation | n | Pearson r | Spearman ρ | MAE | R² |
|---|---:|---:|---:|---:|---:|
| Starters + bench (0.25) | 2,132 | **0.2490** | **0.2726** | 0.8866 | **0.0610** |
| Expected minutes | 2,132 | 0.2477 | 0.2718 | **0.8861** | 0.0605 |
| Outfield starters | 2,132 | 0.2468 | 0.2695 | 0.8878 | 0.0595 |
| Positional vectors | 2,132 | 0.2355 | 0.2504 | 0.8915 | 0.0516 |

Bench and expected-minute aggregation help slightly against official xG, but the gain does not transfer to Scottish Lower.

### Scope B — scoreline supremacy, 24/25 + 25/26

| RAPM target | best formulation | n | Pearson r | Spearman ρ | MAE | R² |
|---|---|---:|---:|---:|---:|---:|
| pxG | Expected minutes | 1,461 | **0.1896** | 0.1014 | **1.3616** | **0.0277** |
| Shots | Expected minutes | 1,461 | 0.1788 | 0.0774 | 1.3668 | 0.0225 |

The pooled Scottish result favors pxG RAPM and expected minutes. It does **not** survive restriction to tiers 56/57.

### Scope C — decision-bearing 710-match Scottish-Lower target

| RAPM target | formulation | Pearson r | Spearman ρ | MAE | R² |
|---|---|---:|---:|---:|---:|
| Shots | Outfield starters | **0.0379** | **0.0415** | **1.3917** | -0.0030 |
| Shots | Expected minutes | 0.0350 | 0.0245 | 1.3924 | **-0.0028** |
| Shots | Starters + bench | 0.0361 | 0.0357 | 1.3928 | -0.0033 |
| Shots | Positional vectors | 0.0099 | 0.0051 | 1.4043 | -0.0190 |
| pxG | Outfield starters | 0.0109 | 0.0085 | 1.3992 | -0.0122 |
| pxG | Expected minutes | 0.0051 | -0.0072 | **1.3963** | **-0.0108** |
| pxG | Starters + bench | 0.0055 | 0.0011 | 1.3995 | -0.0126 |
| pxG | Positional vectors | 0.0102 | 0.0072 | 1.4025 | -0.0230 |

No formulation beats the held-out mean baseline in R². Claims of a production gain would therefore be unsupported.

## Bench weight

A nested validation block wholly inside history selected:

> **Optimal tested `w_bench = 0.10`**

This was the winner for Scope A and for both RAPM targets in Scopes B and C. Performance declined monotonically from 0.10 to 0.35 in Scope A. Because 0.10 is the grid's lower boundary, the evidence is better read as **“bench contribution is weak and should be heavily shrunk”** than as precise identification of 0.10.

The architecture retains the documented constructor default `0.25` for compatibility, while allowing either an explicit `0.10` or a learned bounded prior.

## Engineering verification

The production component supports all four strategies and all currently wired count observations:

- `PoissonObservation()`
- `NegativeBinomialObservation()`
- `JointGammaPoissonObservation()`

For the 700-match outfield mock model on the laptop:

- ReverseDiff tape instructions: **99**
- warmed minimum compiled gradient: **0.0298 ms**
- warmed median compiled gradient: **0.0317 ms**
- compiled ReverseDiff agreed with fresh ReverseDiff and ForwardDiff within the required tolerances
- tape instruction count remained independent of observation count
- lineup design construction was `@inferred` type-stable

The Turing/LogDensityProblems wrapper reported allocations around compiled-gradient replay; the likelihood itself contains no dictionaries, scalar observation loops, or mutable global buffers, and consumes pre-extracted contiguous `Vector{Float64}` inputs.

## Expected-minute caveat

The source has a structural provider ambiguity: tiers 56/57 record `minutes_played == 0` across old periods where minutes are unavailable, while a real unused substitute is also legitimately zero. The EDA therefore uses the mean of the previous five **positive recorded** minutes, defaulting a new starter to 90 and a new bench player to zero. This is point-in-time, but it overstates expected minutes for historically used substitutes and is not a clean substitution-probability model. Its strong Scope-B result should not be treated as proof that minute weighting is intrinsically superior.

## Recommendation for the next model bake-off

If a single local one-fold count-model comparison is authorized, use:

```julia
PlayerLineupDynamics(
    feature = Features.ShotsPlusMinusFeature(),
    aggregation = OutfieldPlayerAggregation(),
)
```

Compare it against the clean team-dynamics baseline on the exact same 710 fixtures. Treat it as a falsification exercise, not a likely promotion. A secondary bench arm may use `BenchWeightedPlayerAggregation(w_bench = 0.10)`. Do not spend a multi-fold MCMC grid on the positional or expected-minute variants until deterministic target-scope evidence improves.

## Reproducibility artifacts

- `r01_eda_cross_league_formulations.jl`
- `r01_lineup_formulation_results.csv`
- `r01_bench_weight_grid.csv`
- `r02_eda_scottish_tiers.jl`
- `r02_scottish_tier_results.csv`
- `r02_scottish_bench_weight_grid.csv`
