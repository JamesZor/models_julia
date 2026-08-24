# Issue 05 — legacy league indexing

## Status

Phase-1 diagnostic ready; no open_play code is changed and no sampling is performed.

## Scope

The l05 pxG champion was trained with one `delta_league[1]` column while ScottishLower includes tournaments 56 and 57. Current l05 DataFrame prediction semantics select column 2 for tournament 57; on a one-column saved artifact that selection falls back to zero. This notebook compares that legacy path with the artifact-compatible pooled path.

It deliberately composes prior remediation diagnostics:

- issue 01 `ExistingChainTeamBridge`: names map to their original saved-chain columns;
- l03 permanent `_tau_scaled_team_effects`: alpha/beta are centred and tau-scaled.

Only league semantics differ between arms. Kappa, penalty/referee noise, wealth, month, no prediction-time clamp/floor (the current l05 behavior), and every other l05 arithmetic step are held identical.

## Run

Include `r01_validate_league_indexing.jl` block-by-block in the remote persistent Julia REPL. It selects the newest `recomb_pxg_wealth_integrated_hl365_hs2` artifact unless `BF_BUGFIX_EXPERIMENT` pins a folder. It is reconstruction-only.

## Acceptance criteria

1. Saved legacy champion has exactly one `delta_league` column.
2. Posterior summaries report `delta_league[1]`, `exp(delta)`, and `base_mu + delta`.
3. Tournament/fold fixture counts are printed.
4. Tournament 56 is byte/draw-wise unchanged.
5. Tournament 57 pooled/current **open-play** rate ratio equals `exp(delta_league[1])`; total rates are not ratio-tested because penalty/noise is additive.
6. Both score grids normalize per draw and basic 1X2/BTTS/O2.5 summaries are printed.
7. Unknown tournament IDs are rejected, never defaulted.

## Future chain-aware contract

- **1 league column:** legacy pooled artifact; 56 and 57 both map to column 1.
- **2 league columns:** canonical stored map; 56=>1 and 57=>2.
- **Any other/missing tournament ID:** explicit error.

`LeagueIndexDiagnostics` keeps the contract and reconstruction independent of l05-specific artifact selection so l03/l04 artifacts can be added later. This is a diagnostic contract, not a production behavior change.
