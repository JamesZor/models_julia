# Workflow — how to build and drive a model folder

## The ergonomics this is built for

James reads and runs research from **nvim with `kitty-runner.nvim`**, sending blocks into a
terminal ssh'd to the server's Julia REPL. That constrains every file in this stream:

- **Blocks must be independently sendable.** Each `# %%` block runs on its own, given that the
  earlier blocks in the file have run. No block depends on something defined halfway down a
  later block.
- **Numbered, banner-separated sections.** You should be able to scroll and see where data,
  model, sampling, diagnostics and inference are controlled without reading the code.
- **One statement per line.** No semicolon-compressed setup lines.
- **Prefixed globals** (`TP_SEASONS`, `TP_SAMPLES`, …) because prototype runners share a
  long-lived REPL.
- **Results land in globals you can poke at.** Every block leaves its output bound to a named
  variable, so you can inspect it after the gate table prints.

The repo-wide reference for this style is `docs/prototype_runner_style_guide.md` at the root.

## Files in a model folder

```
NN_model_name/
├── MODEL.md            What it is, the equations, the modelling decisions, what it is NOT
├── FINDINGS.md         Dated gate results + config hashes. Append-only.
├── l01_model.jl        Config struct / adapter into the src engine. Definitions only.
├── l02_equations.jl    Independent pure-Julia reimplementation of λ, for the parity gate
├── l03_gates.jl        Gates 0-2 for THIS model (lifted to _protocol/ on second use)
├── l04_sampling_gates.jl  Gate 3: equation parity, gradients, smoke run
├── v01_walkthrough.jl  ★ Gates 0–5 as numbered REPL blocks. Cheap. Run this first, always.
├── r01_train.jl        Full walk-forward grid (James launches)
├── r02_evaluate.jl     Gate 6 — OOS scoring vs market on identical fixtures
└── r03_growth_clv.jl   Gate 7 — CLV and portfolio-Kelly growth
```

`lXX` = loader (definitions, no execution). `rXX` = runner (execution). `v01` is a runner too,
but named separately because it is the **validation** walkthrough and gets run far more often
than the training runners.

### Why `l02_equations.jl` exists

It is a deliberate, independent second implementation of the model's intensity equations in
plain Julia — no Turing, no components. Gates 3 and 4 check the Turing model and the extraction
code against it. If all three agree, the fitted model, the documented model, and the priced
model are the same model. That single check is what the 2026-08-24 audit was missing.

It is not dead code and it must not be refactored to call the thing it is checking.

## The walkthrough blocks

```
0. Contract        — data snapshot fingerprint, fold inventory, sealed season
1. Config          — component menu, chosen config, config hash
2. Features        — leakage, perturbation, type purity, map identity, coverage
3. Sampling        — equation parity, gradient diff, init safety, smoke run → saved chain
4. Extraction      — synthetic-chain parity (exact) + real-chain load (plumbing)
5. Score matrix    — dispatch, normalisation, market identities, moments
```

Blocks 0–2 and 4–5 are seconds-to-minutes. Block 3's smoke is the only MCMC in the file, and
James runs it.

Each block ends:

```julia
tp_report_gate("2. Features", gate_results)   # prints the PASS/FAIL table
@assert all(g -> g.pass, gate_results)
```

Read the table; the assert is the tripwire, not the report.

## Recording results

After a gate run, append to the model's `FINDINGS.md`:

```markdown
## 2026-08-25 — Gates 0–5, config `a3f19c`

| Gate | Result | Note |
|---|---|---|
| 2. Features | PASS | 19 folds, 0 unknown teams, perturbation bit-identical |
| 3. Sampling | PASS | max rel grad err 3.1e-9, median 0.61 ms; Rhat 1.004, 0 divergences |
```

Then add one line to [`FINDINGS_INDEX.md`](FINDINGS_INDEX.md). **A result that is not written
down does not exist** — that is what made the previous round unrecoverable.

## Adding a new model

1. Copy the folder skeleton, rename the globals prefix.
2. Write `MODEL.md` **first** — equations and decisions before code. If you cannot write the
   equations, you are not ready to write the `@model`.
3. Write `l02_equations.jl` from `MODEL.md`, not from the Turing model.
4. Write `l01_model.jl`. Prefer adding a component under `src/models/pregame/components/` or an
   extractor under `src/features/extractors/` over writing bespoke prototype machinery.
5. Point `v01_walkthrough.jl` at it and start at block 0.
