# Issue 02 findings — extraction omits hierarchical tau scales

**Status:** Permanent tau patch validated against saved artifacts on `mcmc-beast`; no resampling performed.

## Static affected-method manifest

| File | Method | Tau status |
|---|---|---|
| `open_play/l03_recombination_models.jl` | `TeamGoalsRecombIntegratedNegBinModel` | required |
| `open_play/l04_recomb_wealth_models.jl` | `TeamGoalsRecombIntegratedPoisWealthModel` | required |
| `open_play/l05_recomb_pxg_models.jl` | `TeamPxGRecombWealthIntegratedModel` | required |
| `open_play/l03_recombination_models.jl` | Poisson / OpenPlay Poisson / recombination Poisson models | no tau by design; not applicable |

## Remote validation record

Phase-1 isolated pxG validation used fold 39 (11 matches, 3,600 draws) from
`recomb_pxg_wealth_integrated_hl365_hs2_20260823_075833`:

```text
unknowns: none
tau_alpha: q05=0.05209, median=0.10813, mean=0.11021, q95=0.17286
tau_beta:  q05=0.11021, median=0.15228, mean=0.15509, q95=0.21065
exact log team effect: q05=-0.18968, median=-0.01064, mean≈0, q95=0.23396
exact rate multiplier: q05=0.82723, median=0.98941, mean=1.00823, q95=1.26359
unscaled mean open rates: (0.42764, 1.04412)
tau-scaled mean open rates: (1.08075, 0.96318)
score-matrix maximum difference: 0.58859; both masses=1.0
tau-only swap max open change: (0.94253, 0.71899)
```

Permanent helper validation at commit `7463d80`:

| Artifact | Fold | Matches | Draws | Teams | Tau equals independent diagnostic | Feature route | DataFrame route |
|---|---:|---:|---:|---:|:---:|:---:|:---:|
| `recomb_negbin_integrated_hl365_hs2_20260822_160843` | 39 | 11 | 3,600 | 22 | yes | blocked by existing referee-parameter drift | blocked by existing referee-parameter drift |
| `recomb_pois_wealth_integrated_hl365_hs2_20260822_213446` | 39 | 11 | 3,600 | 22 | yes | passed | passed |
| `recomb_pxg_wealth_integrated_hl365_hs2_20260823_075833` | 39 | 11 | 3,600 | 22 | yes | passed | passed |

The NegBin artifact reached and passed the tau equality assertion before its independent
`raw_gamma_ref[2:57]` artifact/feature-dimension error.

## Required interpretation

- Confirm zero-sum assertions and no parameter-column mutation before interpreting magnitude.
- A nonzero corrected-mapping unscaled-vs-tau-only difference is tau-only evidence: league fallback, no clamps/floors, kappa, penalty/noise, wealth, and month arithmetic are intentionally identical.
- Do not apply the result to l03 Poisson methods; they have no tau parameters by design.
- This does not validate the separate l03 NegBin penalty hierarchy, referee centering, issue-01 mapping deployment, or training/prediction league parity.

## Permanent patch checklist

- [x] Added `_tau_scaled_team_effects` in l03: saved-column-order preserving, read-only, shape/draw-count validating, and explicit about absent tau parameters.
- [x] Routed both l03 integrated NegBin extractors through it.
- [x] Routed both l04 wealth extractors through it.
- [x] Routed both l05 pxG extractors through it.
- [x] Left tau-free l03 Poisson extraction unchanged.
- [x] Added `r02_validate_tau_patch.jl` for all three saved artifact types where available.
- [x] Ran r02 on beast: all three artifacts matched the independent tau matrices.

## Decision

Accept the tau extraction correction. It reconstructs
`alpha=(raw_alpha-rowmean).*reshape(tau_alpha,:,1)` and beta analogously before existing rate
construction, and matches an independent reconstruction for all three available artifact types.

**DataFrame OOS mapping remains issue 01 until patched.** League, clamp, penalty/referee, wealth, and
kappa behavior remain intentionally untouched. The NegBin referee-dimension failure is a separate
confirmed artifact-drift/penalty-extraction issue and does not invalidate the tau matrix result.
