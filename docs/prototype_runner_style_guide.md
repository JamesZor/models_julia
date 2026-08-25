# Human-readable prototype runner style guide

This guide defines the preferred style for research code under `current_development/`. It is both a contributor reference and an operating contract for AI coding agents.

The goal is not merely code that executes. A researcher must be able to open a runner, understand the experiment from top to bottom, execute it section by section, and identify where data, model, training, diagnostics, inference, and persistence are controlled.

The reference style is exemplified by:

```text
current_development/smile_negbin/r02_train_ireland.jl
```

## 1. Loader/runner contract

Follow the repository's paired prototype convention:

```text
lXX_name.jl   loader: types, equations, helpers, infrastructure
rXX_name.jl   runner: the readable scientific workflow
```

### Loader responsibilities

A loader may contain technical complexity that is necessary but not central to reading the experiment:

- model and configuration types;
- pure mathematical functions;
- feature-building helpers;
- checkpoint validation and recovery;
- serialization helpers;
- reusable diagnostics;
- API integration methods;
- infrastructure-specific error handling.

Loader functions should still be documented and tested, but the runner should not reproduce their internals.

### Runner responsibilities

A runner should answer, visibly and in order:

1. What question is this experiment answering?
2. What data snapshot is used?
3. What model is being fitted?
4. What split and filtration contract is used?
5. What sampler and execution strategy are used?
6. Where are outputs written?
7. What gates determine success?
8. What inference or report is produced?

The runner is a research notebook expressed as a Julia file. It should describe the experiment rather than implement a framework.

## 2. Required runner structure

Use long, visible numbered section banners and optional `# %%` cell markers:

```julia
# %%
# ===================================================================
# 1. Packages and implementation
# ===================================================================
```

A substantial training runner should normally use this order:

```text
WHAT THIS IS AND IS NOT
FILTRATION / COMPARABILITY CONTRACT
PERSISTENCE CAVEAT
USAGE

1. Packages and implementation
2. Configuration
3. Runtime and output directory
4. Data snapshot and temporal splits
5. Engine / model construction
6. Feature construction and preflight gates
7. Checkpoint preparation
8. Training
9. Convergence diagnostics
10. OOS inference / evaluation
11. Final report
```

Not every runner needs every section, but omissions should be intentional.

## 3. Put the research question first

Begin with a plain-language header explaining:

- the hypothesis;
- the control or baseline;
- what is held fixed;
- what result would change the research decision;
- what the runner deliberately does not claim.

Example:

```julia
# This is a convergence and predictive-fit experiment.
# It is not a betting allocation study.
#
# Question: does adding history-only player strength improve genuine
# walk-forward log loss relative to the clean team-level baseline?
```

Do not make the reader reverse-engineer the question from constructors and paths.

## 4. Make configuration visible

Use descriptive, prefixed constants rather than scattered literals or repeated environment lookups:

```julia
const OP8_TARGET_SEASONS = ["24/25", "25/26"]
const OP8_SAMPLES        = 800
const OP8_WARMUP         = 800
const OP8_CHAINS          = 4
const OP8_MAX_DEPTH       = 10
const OP8_QUEUE_TASKS     = 16
```

For configurable remote runners, parse environment variables once into a named configuration object. Downstream code should read `config.samples`, not call `get(ENV, ...)` repeatedly.

Prefix globals because prototype runners often share a long-lived REPL.

## 5. Show the model constructor explicitly

The model should be easy to find:

```julia
function op8_model(registry)
    return ScottishLowerNPNOGRecombinedPoissonModel(
        registry;
        half_life_days = 365.0,
        own_goal_policy = :beneficiary,
    )
end
```

Keep one keyword per line when the constructor expresses modelling decisions. Add comments explaining non-obvious choices, not restating field names.

A model should not be constructed inside checkpoint or persistence logic.

## 6. Prefer small named workflow functions

Good runner functions correspond to research actions:

```julia
op8_load_data(config)
op8_build_folds(ds, config)
op8_build_model(registry, config)
op8_build_features(ds, folds, model, config)
op8_train(model, features, config)
op8_check_convergence(results, contexts, config)
op8_generate_oos(results, contexts, model, config)
op8_print_summary(report)
```

Avoid names such as `process`, `do_work`, or `handle` without a scientific object.

Pass important dependencies explicitly. Avoid functions that silently capture many mutable globals.

## 7. Keep technical machinery out of the narrative

The runner should not contain large inline implementations of:

- atomic temporary-file replacement;
- checkpoint deserialization and recovery;
- credential redaction;
- parameter-manifest parsing;
- registry hashing;
- generic DataFrame persistence conversion.

Put these in the loader and expose a descriptive call:

```julia
checkpoint_report = prepare_stage8_checkpoints!(contexts, config)
```

The runner may print and inspect the returned report.

## 8. One statement per line

Avoid machine-compressed Julia:

```julia
samples=parse(Int,get(ENV,"SAMPLES","800")); warmup=parse(Int,get(ENV,"WARMUP","800"))
```

Prefer:

```julia
samples = parse(Int, get(ENV, "SAMPLES", "800"))
warmup  = parse(Int, get(ENV, "WARMUP", "800"))
```

Avoid dense one-line `if`, `try`, comprehensions with side effects, and multiple mutations separated by semicolons.

Long mathematical expressions may remain compact when that improves correspondence with the documented equation. Infrastructure code generally should not.

## 9. Separate filtration from feature construction

Temporal semantics must be visible before training.

A runner should explicitly report:

- fitted match IDs or count;
- held-out match IDs or count;
- prediction cutoff;
- season and time step;
- overlap count;
- postponed/not-yet-played exclusions;
- fallback identity count where available.

Never assume generic names such as `target_match_ids` mean held-out OOS. Translate package split semantics into the model's filtration contract in one named, tested function.

Required invariant:

```text
training kickoff < prediction cutoff <= held-out kickoff
```

Any exception must be quarantined or explicitly versioned.

## 10. Use the project queue visibly

When using `QueuedNUTSConfig`, show the queue configuration in the training section:

```julia
sampler = Samplers.QueuedNUTSConfig(
    n_samples  = config.samples,
    n_warmup   = config.warmup,
    n_chains   = config.chains,
    max_depth  = config.max_depth,
)

strategy = Training.Independent(
    parallel = true,
    max_concurrent_tasks = config.max_concurrent_tasks,
)
```

Explain that NUTS chains are single-threaded and that the flattened fold × chain queue dynamically fills available Julia threads. Do not build a second scheduler around an existing queue unless there is a documented API limitation.

Keep BLAS at one thread during concurrent MCMC to avoid oversubscription.

## 11. Gates belong next to the stage they protect

Use explicit gate names and output:

```text
G-A  fold inventory and filtration
G-B  feature/model manifest
G-C  checkpoint integrity
G-D  convergence
G-E  OOS extraction and score mass
G-F  artifact coverage
```

A gate should report enough context to diagnose failure: fold, parameter, match ID, expected value, and observed value.

Do not print `PASS` when required diagnostics are unavailable.

## 12. Separate training, diagnostics, inference, and evaluation

These are different scientific stages:

- **Training:** produce posterior chains.
- **Diagnostics:** decide whether a chain may be promoted.
- **Inference:** produce held-out predictive distributions.
- **Evaluation:** compare predictions with outcomes or baselines.
- **Backtesting:** apply decision and staking rules.

A runner may perform several stages, but each must have its own section and artifact. A failed convergence gate must block inference promotion. Evaluation must not silently regenerate predictions.

## 13. Persistence must be understandable

At the top of the runner, document:

- output directory;
- immutable versus replaceable artifacts;
- resume behavior;
- prototype-module deserialization requirements;
- whether data snapshots are pinned or cache-backed.

Use immutable artifacts for chains, manifests, and accepted fold results. Use atomic replacement only for progress summaries.

Prototype model types often live outside `src`; document the required `include(...)` before deserialization.

## 14. Extended validation workflow

Use the following progression for new models and major runner changes.

### Stage A — contract review

Before implementation, freeze:

- equations and priors;
- source columns and provider semantics;
- canonical identities;
- split meanings and cutoff rules;
- unknown-team policy;
- evaluation fixtures and baselines.

The researcher—not an agent—owns this contract.

### Stage B — pure implementation

Implement equations and transforms without database access, Turing, persistence, or experiment orchestration.

Validate:

- scalar/vector parity;
- support and normalization;
- finite differences or ForwardDiff;
- no mutation of inputs;
- synthetic edge cases.

### Stage C — feature and filtration audit

Build one representative real FeatureSet and inspect:

- exact match IDs;
- cutoff dates;
- team maps;
- missing/fallback identities;
- quarantines;
- no future outcomes.

### Stage D — AD/model smoke

Validate log density, gradients, compiled-tape parity, allocations, and parameter manifest.

### Stage E — synthetic extraction

Use deterministic chains to validate transformations, score tensors, and inference interfaces.

### Stage F — one real fold

Run all production-sized chains on one fold. Require convergence and full OOS inference before scaling.

### Stage G — queue preflight

Build every FeatureSet and validate every checkpoint key without sampling. Print the exact fold × chain task count and concurrency.

### Stage H — full queue

Launch the native queue with immutable checkpoints and resumability. Do not change code while interpreting artifacts from the running commit.

### Stage I — post-run audit

Verify checkpoint count, parameter/internal chain sections, diagnostics, OOS fixture count, fallback rate, and PPD coverage before claiming completion.

### Stage J — evaluation and ablation

Evaluate only accepted OOS artifacts. Add features one at a time against the same fixtures and baseline.

## 15. Agent operating contract

When an AI agent edits a research runner, it must:

1. Read this guide and the local `CLAUDE.md`.
2. Explain the proposed runner sections before editing.
3. Preserve the mathematical and filtration contract unless explicitly authorized.
4. Prefer loader helpers over hiding technical code in the runner.
5. Avoid subagents when the user requests direct control.
6. Make behavior-preserving readability refactors separately from scientific changes.
7. Show changed paths and validation commands.
8. Use dry-run or prepare-only validation before sampling.
9. Never interpret file existence or a progress bar as scientific success.
10. Record defects and corrections honestly in the handoff/findings.

The agent should optimize for the researcher's understanding, not for minimum line count.

## 16. Review checklist

Before accepting a runner, ask:

- Can a researcher identify the hypothesis in one minute?
- Is the model constructor visible?
- Are data and split choices visible?
- Can each stage be executed as a notebook cell?
- Is the native queue call easy to find?
- Are convergence and promotion gates explicit?
- Is technical recovery logic in the loader?
- Are literals centralized in configuration?
- Are outputs and resume rules documented?
- Can the runner be read without mentally executing nested control flow?

If the answer to several questions is no, refactor before adding more model complexity.
