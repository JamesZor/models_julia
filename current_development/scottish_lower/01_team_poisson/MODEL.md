# Model 01 — Team-level baseline

**Engine:** `Models.PreGame.DynamicGoalsTimeDecayModel` (already in `src`).
**Role:** the reference opponent. Every later model is judged against this one on
identical fixtures and identical markets.

This model is not novel and is not meant to be. It exists so that (a) the protocol is
exercised end-to-end against code that already works, and (b) there is an honest number to
beat before anyone claims an improvement.

## What it is

A hierarchical attack/defence rating model on full-time goals, with exponential time decay
on the likelihood. Goals are modelled as negative binomial rather than Poisson, with the
dispersion `r` estimated. With the default prior `log r ~ Normal(3.1, 0.4)` the mass sits
near `r ≈ 22`, which is close to Poisson — consistent with the finding that pregame
full-time totals in these leagues are near-Poisson (V/M ≈ 0.94–1.09). Poisson is therefore a
limiting case of this model, not a separate one.

## What it is NOT

- Not a market-anchored model. There is no market pillar; it sees only goals and dates.
- Not a decomposition. Penalties and own goals are included in the target (that is model 03).
- Not player-aware (that is model 02).
- Not a Dixon–Coles low-score correction. Home and away goals are conditionally independent
  given the intensities.

## Equations

For match `i` between home team `h(i)` and away team `a(i)`, in season `s(i)` and calendar
month `m(i)`:

```
λ_h(i) = exp( μ_base[s(i)] + δ_month[m(i)] + γ_h(i) + α_h(i) + β_a(i) )
λ_a(i) = exp( μ_base[s(i)] + δ_month[m(i)]           + α_a(i) + β_h(i) )
```

Team effects are non-centred and zero-sum:

```
α_scaled = raw_a .* σ_a        α = α_scaled .- mean(α_scaled)
β_scaled = raw_d .* σ_d        β = β_scaled .- mean(β_scaled)

raw_a, raw_d ~ Normal(0, 1)^n_teams
σ_a ~ Gamma(2.0, 0.15)         σ_d ~ Gamma(2.0, 0.15)
```

`α` is attacking strength, `β` is defensive leak: a large `β` means the team concedes more,
because it enters the *opponent's* intensity.

Likelihood, time-decayed with half-life `H` days and `Δ_i` days before the boundary:

```
w_i = 0.5 ^ (Δ_i / H)

logprob += Σ_i w_i · logpdf( RobustNegativeBinomial(r_h, λ_h(i)), y_h(i) )
logprob += Σ_i w_i · logpdf( RobustNegativeBinomial(r_a, λ_a(i)), y_a(i) )
```

Note this is an `@addlogprob!` weighted pseudo-likelihood, not a generative statement — the
weights are not counts and the posterior is not a Bayesian posterior for any single data set.
That is intentional and standard for time-decayed rating models, but it means the model
cannot be scored by its marginal likelihood.

## Component menu

The engine is assembled from four components. The **default** column is what
`tp_model()` builds; the walkthrough prints the menu so alternatives can be swapped in the
REPL and blocks ②–⑤ re-run.

| Slot | Default | Alternatives |
|---|---|---|
| Interception | `GlobalInterception(μ = Normal(0.2, 0.1))` | `SeasonalInterception`, `HierarchicalMonthlyInterception` |
| Dispersion | `GlobalDispersion(log_r = Normal(3.1, 0.4))` | `HomeAwayDispersion`, `AdvancedVolatilityDispersion` |
| Home advantage | `GlobalHomeAdvantage(γ_global = Normal(0.2, 0.2))` | `HierarchicalTeamHomeAdvantage`, `HierarchicalLeagueHomeAdvantage` |
| Dynamics | `TimeDecayDynamics(days_half_life = 180)` | any half-life; `σ_att`/`σ_def` priors |

With `GlobalInterception`, `μ_base` is constant across seasons and `δ_month` is exactly zero,
so the month index is carried but unused. With `GlobalDispersion`, `r_h = r_a = r`.

**Half-life is unresolved.** 180 days is the `src` default; the archived Scottish rebuild used
365. Neither was chosen from Scottish evidence. Treat the default as provisional until a
half-life sweep is run under gate 6.

## Required features

`Features.required_features` returns `TeamIDsFeature`, `GoalsFeature`, `DatesFeature`,
`MonthFeature`, `TimeIndicesFeature`.

## Chain variable names

Extraction reads these; the parity gate asserts nothing else is silently ignored.

| Component | Sampled sites |
|---|---|
| Interception | `inter.μ` |
| Dispersion | `disp.log_r` |
| Home advantage | `ha.γ_global` |
| Dynamics | `dyn.σ_a`, `dyn.σ_d`, `dyn.raw_a[i]`, `dyn.raw_d[i]` |

## Known asymmetries to watch

1. **Dispersion clamp.** Training computes `r = exp(clamp(log_r, -10, 10))`
   (`components/dispersion.jl:26-30`); extraction computes `exp(log_r)` with no clamp
   (`dispersion.jl:75-78`). Under the default prior the clamp never binds, so the two agree in
   practice — but it is a genuine train/predict asymmetry and gate 4 should report the maximum
   observed `|log_r|` rather than assume it stays in range.
2. **Unmapped teams.** `extract_parameters` falls back to `zeros(n_samples)` for a team not in
   `team_map` (`goals.jl:150-157`). That fallback is correct behaviour for a promoted side, but
   it is also exactly how the 2026-08-24 defect hid. Gate 2 counts unmapped sides per fold and
   gate 4 reports them again; a nonzero count must be explained, never absorbed.
3. **Season index fallback.** If an OOS row has no `season_idx`, extraction uses `n_seasons`
   (`goals.jl:161`). With `GlobalInterception` this is harmless because all seasons share `μ`;
   under `SeasonalInterception` it silently assigns the newest season. Re-check if the
   interception component is swapped.

## Fold semantics

`create_features` builds the `FeatureSet` from `history_match_ids` **plus** `target_match_ids`
and fits on all of it. Those are the observations through step `t`. The held-out fixtures are
step `t+1`, retrieved with `Data.get_next_matches(ds, (boundary, meta), splitter)`.

Anything calling `target_match_ids` "the test set" is wrong, and that error is what made the
archived Stage 7 report an OOS check that was not out of sample.
