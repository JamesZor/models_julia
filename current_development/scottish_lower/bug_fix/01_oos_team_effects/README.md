# Issue 01 — OOS team effects silently disappear

**Status:** Reproduction notebook ready; defect not yet fixed.
**Audit source:** `open_play/AUDIT_2026-08-24.md`, blocker 1.

## Hypothesis

The custom recombination feature builders train with an integer-keyed map:

```julia
team_map::Dict{Int,Int}  # internal team ID => posterior column
```

The repository OOS pipeline returns rows from `ds.matches`. Those rows contain string
`home_team`/`away_team` columns and do not contain `home_team_id`/`away_team_id`.
The custom extraction adapters try to query the integer-keyed map with those strings, obtain `-1`,
and silently replace attack/defence (and related team-level effects) with zero vectors.

## Investigation files

- `l01_team_mapping_diagnostics.jl` — side-effect-free helpers for reproducing current lookup behavior,
  deriving a candidate name-to-index bridge, scanning folds, and measuring posterior contributions.
- `r01_confirm_oos_team_effects.jl` — notebook-style REPL investigation. Run one `# %%` section at a
  time in order.
- `FINDINGS.md` — paste compact output tables and record the conclusion.

## How to run remotely

First synchronize the server checkout so this issue directory exists there (commit/push locally,
then `git pull` on the server). Verify with:

```bash
test -f current_development/scottish_lower/bug_fix/01_oos_team_effects/l01_team_mapping_diagnostics.jl
```

Then start a **fresh** Julia REPL from the repository root on the server:

```bash
julia --project -t auto
```

A fresh REPL is required because the legacy prototype loaders define `const ROOT` and model types in
`Main`. If `ROOT` was assigned during an earlier failed attempt, exit and restart Julia.

Then highlight/send each `# %%` block from `r01_confirm_oos_team_effects.jl` using
`send_to_kitty`. Keep the same REPL alive so variables created by earlier blocks remain available.
Restart Julia before rerunning the loader block because the prototype files define concrete types in
`Main`.

If more than one matching experiment artifact exists, set the exact folder basename before starting:

```bash
export BF_BUGFIX_EXPERIMENT='recomb_pxg_wealth_integrated_hl365_hs2_YYYYMMDD_HHMMSS'
```

## Confirmation criteria

The issue is confirmed if all are true:

1. `ds.matches` has team-name columns but no team-ID columns.
2. `feature_set.data[:team_map]` has integer keys.
3. The current adapter lookup maps known OOS teams to `-1`.
4. A name-to-index bridge maps those same teams to valid posterior columns.
5. Posterior team contributions under the corrected lookup are materially nonzero.

## Fix acceptance criteria

A permanent correction must satisfy all of these:

- Use exactly the same team identity/index mapping in training and prediction.
- Preserve explicit population-level fallback for genuinely unseen teams.
- Report unknown-team counts instead of silently substituting zero.
- Numerically reconstruct the Turing team contribution for known teams.
- Pass every fold without accidental unknown mappings.
- Add tests for known teams, genuinely unseen teams, and string/integer schema mismatch.

This issue should be fixed before interpreting any existing OOS latent, evaluation, portfolio, or
Layer-2 result from `l03`–`l05`.
