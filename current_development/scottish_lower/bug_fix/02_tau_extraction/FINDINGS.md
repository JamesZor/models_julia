# Issue 02 findings — extraction omits hierarchical tau scales

**Status:** Permanent patch implemented locally; pending remote saved-artifact validation (no sampling).

## Static affected-method manifest

| File | Method | Tau status |
|---|---|---|
| `open_play/l03_recombination_models.jl` | `TeamGoalsRecombIntegratedNegBinModel` | required |
| `open_play/l04_recomb_wealth_models.jl` | `TeamGoalsRecombIntegratedPoisWealthModel` | required |
| `open_play/l05_recomb_pxg_models.jl` | `TeamPxGRecombWealthIntegratedModel` | required |
| `open_play/l03_recombination_models.jl` | Poisson / OpenPlay Poisson / recombination Poisson models | no tau by design; not applicable |

## Remote validation record

Paste block 8 output here:

```text
artifact:
fold:
unknowns:
tau posterior:
team-effect/rate-multiplier posterior:
current-unscaled vs tau-only:
score matrix:
known-team tau-only swap:
```

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
- [ ] Run r02 on beast and paste output below.

## Decision

Code correction is ready for saved-artifact validation. It reconstructs `alpha=(raw_alpha-rowmean).*reshape(tau_alpha,:,1)` and beta analogously before existing rate construction. **DataFrame OOS mapping remains issue 01 until patched**; no result from this issue should be interpreted as an issue-01 mapping fix. League, clamp, penalty/referee, wealth, and kappa behavior remain intentionally untouched.
