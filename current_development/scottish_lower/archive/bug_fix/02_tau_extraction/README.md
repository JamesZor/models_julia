# Issue 02 — extraction omits hierarchical tau scales

**Status:** Permanent extraction patch implemented; remote saved-artifact validation pending.
**Audit source:** `open_play/AUDIT_2026-08-24.md`, blocker 2.

## Hypothesis

l03 NegBin, l04, and l05 fit `alpha = centered(raw_alpha) .* tau_alpha` and the analogous beta term, but their extraction methods use centered raw columns. This changes draw-level rates. l03 Poisson models are **tau-free by design** and must not be changed merely to make this diagnostic uniform.

## Files

- `l01_tau_extraction_diagnostics.jl` — shape-safe chain reconstruction, assertions, posterior summaries, tau-only pxG reconstruction, and static affected-method manifest.
- `r01_validate_tau_extraction.jl` — phase-1 notebook runner for the saved pxG champion; no sampling.
- `r02_validate_tau_patch.jl` — permanent-patch notebook: recreates one fold for each available l03 NegBin, l04 wealth, and l05 pxG saved artifact, compares production helper matrices to the independent diagnostic, and smoke-tests the affected DataFrame extractor without sampling.
- `FINDINGS.md` — remote output record and decision log.

The loader imports `ExistingChainTeamBridge` from issue 01. It does not rebuild name identity or posterior-column ordering.

## Patch checklist

- [x] One l03 shared helper shape-validates raw matrices/draw counts and requires `tau_alpha`/`tau_beta` with a clear incompatible-artifact error.
- [x] Both l03 integrated NegBin extractors reconstruct scaled alpha/beta.
- [x] Both l04 wealth extractors reconstruct scaled alpha/beta through that helper.
- [x] Both l05 pxG extractors reconstruct scaled alpha/beta through that helper.
- [x] l03 tau-free Poisson extractors remain unchanged.
- [x] Saved-chain label selection/order is preserved; the helper is read-only.
- [ ] Run `r02_validate_tau_patch.jl` against available saved artifacts on beast (no sampling).

The independent diagnostic retains its prior acceptance checks: shape-safe stacking, per-draw centering, exact tau scale equation, and no chain-column mutation.

## Remote use

Commit/push only when a manager requests it, then use the documented beast persistent-REPL workflow. Send runner `# %%` blocks in order. Set `BF_BUGFIX_EXPERIMENT` to an exact saved artifact basename; otherwise the latest matching pxG champion is selected. Do not run MCMC locally or remotely for this phase.

## Scope caveat

This establishes tau extraction parity only. It deliberately retains current league fallback, clamps/floors, kappa, penalty/referee, wealth, and month behavior. **DataFrame OOS name/team mapping remains issue 01 until that patch is deployed**; r02 therefore asserts the production tau matrices against the independent issue-02 matrices and treats affected OOS extractor execution as a smoke check. This does not resolve the separate l03 NegBin penalty/referee reconstruction or other audit issues.
