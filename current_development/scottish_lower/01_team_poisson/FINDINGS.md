# Model 01 — Findings

Append-only. Every gate run gets a dated entry with the config hash. A result that
is not written here does not exist.

---

## 2026-08-25 — files written, nothing run

Status: **no gate has been executed.** The walkthrough covers gates 0–2; blocks
for gates 3–5 are not yet written.

| File | State |
|---|---|
| `MODEL.md` | complete for the default component set |
| `l01_model.jl` | complete |
| `l02_equations.jl` | complete for the default component set; refuses others |
| `l03_gates.jl` | gates 0–2 |
| `v01_walkthrough.jl` | blocks 0–2 |

### Findings from reading `src` (not yet executed)

1. **`src` extraction applies the hierarchical scale correctly.**
   `components/dynamics/team_level/time_decay.jl:57-61` computes
   `α_scaled = raw_a .* σ_a` then centres, matching the training submodel exactly.
   The audit's defect #2 (dropped `tau`) exists only in the archived prototype that
   reimplemented extraction. This is the evidence for the "extend the package,
   never reimplement it" rule.

2. **`src` `team_map` is name-keyed.** `goals.jl:148` looks up `row.home_team`
   against a `Dict{String,Int}` built in `features/builder.jl:45-46`. The audit's
   defect #1 (integer-keyed double lookup) likewise does not exist in `src`.

3. **Dispersion has a genuine train/predict asymmetry.** Training clamps
   (`dispersion.jl:26-30`: `exp(clamp(log_r, -10, 10))`); extraction does not
   (`dispersion.jl:75-78`: `exp(log_r)`). Benign under `Normal(3.1, 0.4)`, but real.
   Gate 4 must report the observed `|log_r|` range rather than assume it.

4. **Fold semantics are easy to misread.** `create_features` fits on
   `history_match_ids` **+** `target_match_ids` — all observations through step `t`.
   Held-out fixtures are `t+1` via `Data.get_next_matches`. Mistaking
   `target_match_ids` for a test set is what made the archived Stage 7 report a
   non-OOS "OOS" check.

5. **Half-life is unresolved.** `src` defaults to 180 days; the archived rebuild
   used 365. Neither came from Scottish evidence. Provisional until a gate-6 sweep.

### Next

- Verify blocks 0–2 execute on the server.
- Write gates 3–5: equation parity against `l02`, gradient diff, smoke run
  persisted via `src/experiments`, synthetic-chain extraction parity, score matrix.
