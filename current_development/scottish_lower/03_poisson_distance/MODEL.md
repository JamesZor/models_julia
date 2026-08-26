# Model 03 — Pure Poisson + Travel Distance z_dist

This extension retains Model 00's independent, time-decayed Poisson likelihood and
adds `w_distance · z_dist` directly to the log-intensity. Distance is catalog-standardised log Haversine distance; a positive coefficient encodes away-travel fatigue.

For every match, the fitted equations are:

```text
η_h = μ + γ_h + α_home + β_away + w_distance · z_dist
η_a = μ       + α_away + β_home - (w_distance · z_dist)
λ_h = exp(η_h),  λ_a = exp(η_a)
```

`z_dist` uses the versioned Scottish stadium catalog and catalog-fixed log-distance
standardisation. Missing grounds use the documented deterministic fallback.

The score matrix remains an independent Poisson grid and is checked at Gate 5.
