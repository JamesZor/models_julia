# Scottish Lower correctness remediation notebooks

This directory turns each finding from
[`../open_play/AUDIT_2026-08-24.md`](../open_play/AUDIT_2026-08-24.md) into an isolated,
REPL-driven investigation.

The clean-room successor design is documented in
[`../open_play_rebuild/README.md`](../open_play_rebuild/README.md). It does not
modify, reuse, or supersede the diagnostic artifacts in this directory.

Each issue directory should contain:

- `README.md` — hypothesis, acceptance criteria, and current status.
- `lXX_*.jl` — reusable diagnostic/fix helpers with no experiment execution.
- `rXX_*.jl` — notebook-style runner split into independently sendable REPL blocks.
- `FINDINGS.md` — observed server output, conclusion, and decisions.

## Workflow

1. Reproduce the suspected defect without changing production/prototype behavior.
2. Measure its extent over every temporal fold.
3. construct the smallest candidate correction.
4. Add a regression test that fails before the correction.
5. Re-extract OOS latents and compare old versus corrected outputs.
6. Only then retrain models if the defect also contaminated fitting/features.

## Issue index

| ID | Issue | Status |
|---:|:---|:---|
| 01 | OOS team effects silently disappear | Fixed and cumulatively saved-chain validated; latent rebuild pending |
| 02 | Extraction omits hierarchical tau scales | Fixed and validated against three saved artifact types |
| 05 | Legacy one-column league indexing | Pooled legacy fix validated; latent rebuild pending |

Additional issue directories should be added as each audit finding is started rather than creating empty placeholders.
