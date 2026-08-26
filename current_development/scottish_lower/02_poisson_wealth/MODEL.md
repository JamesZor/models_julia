# Model 02 — Pure Poisson + Squad Wealth ΔW

This extension retains Model 00's independent, time-decayed Poisson likelihood and
adds `w_wealth · ΔW` directly to the log-intensity. A positive coefficient makes the richer home XI more likely to score and the richer away XI less likely to score.

For every match, the fitted equations are:

```text
η_h = μ + γ_h + α_home + β_away + w_wealth · ΔW
η_a = μ       + α_away + β_home - (w_wealth · ΔW)
λ_h = exp(η_h),  λ_a = exp(η_a)
```

`ΔW` is a starting-XI market-value log differential computed only from valuations
strictly preceding the fixture kickoff.  The extraction API has no `DataStore`, so
wealth arms materialize a point-in-time OOS bridge in the FeatureSet. It is not an
input to fitting, and Gate 2 compares every retained bridge value exactly.

`z_dist` uses the versioned Scottish stadium catalog and catalog-fixed log-distance
standardisation. Missing grounds use the documented deterministic fallback.

The score matrix remains an independent Poisson grid and is checked at Gate 5.
