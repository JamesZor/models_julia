# Issue 02 findings — extraction omits hierarchical tau scales

**Status:** Pending remote existing-chain validation.

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

## Decision

Pending validation. If confirmed, the permanent extraction correction must scale centered raw attack/defence draw-wise before rate construction and have synthetic-chain regression coverage for each tau-required method.
