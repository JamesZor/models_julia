# Model 04 — Pure Poisson + Joint Wealth + Distance

This extension retains Model 00's independent, time-decayed Poisson likelihood and
adds `w_wealth · ΔW + w_distance · z_dist` directly to the log-intensity. Both directional pillars are estimated jointly against the same team-strength baseline.

For every match, the fitted equations are:

```text
η_h = μ + γ_h + α_home + β_away + w_wealth · ΔW + w_distance · z_dist
η_a = μ       + α_away + β_home - (w_wealth · ΔW + w_distance · z_dist)
λ_h = exp(η_h),  λ_a = exp(η_a)
```

`ΔW` is a starting-XI market-value log differential computed only from valuations
strictly preceding the fixture kickoff.  The extraction API has no `DataStore`, so
wealth arms materialize a point-in-time OOS bridge in the FeatureSet. It is not an
input to fitting, and Gate 2 compares every retained bridge value exactly.

`z_dist` uses the versioned Scottish stadium catalog and catalog-fixed log-distance
standardisation. Missing grounds use the documented deterministic fallback.

The score matrix remains an independent Poisson grid and is checked at Gate 5.
