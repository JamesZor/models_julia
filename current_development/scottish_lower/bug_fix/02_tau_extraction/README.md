# Issue 02 — extraction omits hierarchical tau scales

**Status:** Phase-1 existing-chain investigation ready; no production change.
**Audit source:** `open_play/AUDIT_2026-08-24.md`, blocker 2.

## Hypothesis

l03 NegBin, l04, and l05 fit `alpha = centered(raw_alpha) .* tau_alpha` and the analogous beta term, but their extraction methods use centered raw columns. This changes draw-level rates. l03 Poisson models are **tau-free by design** and must not be changed merely to make this diagnostic uniform.

## Files

- `l01_tau_extraction_diagnostics.jl` — shape-safe chain reconstruction, assertions, posterior summaries, tau-only pxG reconstruction, and static affected-method manifest.
- `r01_validate_tau_extraction.jl` — notebook runner for the saved pxG champion; no sampling.
- `FINDINGS.md` — remote output record and decision log.

The loader imports `ExistingChainTeamBridge` from issue 01. It does not rebuild name identity or posterior-column ordering.

## Acceptance criteria

1. Matrix extraction correctly stacks one-column, two-dimensional, and iteration × parameter × chain arrays.
2. Centered raw alpha/beta sum to zero on every draw; exact effects equal `(raw - row_mean) .* reshape(tau,:,1)`.
3. Raw parameter matrices are unchanged by diagnostics.
4. Tau, exact team effects, and exp(team-effect) multipliers are reported.
5. For the pxG champion, corrected mapping/current-unscaled semantics are compared only with an otherwise identical tau-only reconstruction, including score grids and known-team swap sensitivity.

## Remote use

Commit/push only when a manager requests it, then use the documented beast persistent-REPL workflow. Send runner `# %%` blocks in order. Set `BF_BUGFIX_EXPERIMENT` to an exact saved artifact basename; otherwise the latest matching pxG champion is selected. Do not run MCMC locally or remotely for this phase.

## Scope caveat

This establishes tau extraction parity only. It deliberately retains current l05 league fallback, no clamps/floors, kappa, penalty, wealth, and month behavior to isolate tau. It is not a production patch and does not resolve l03 NegBin penalty/referee reconstruction or other audit issues.
