# T007 — Parallel feature test has a hidden dependency on `splitting_tests.jl`

**Status:** open  
**Severity:** low  
**Area:** test harness  
**Raised:** 2026-08-29

## Evidence

The documented fast full-suite command fails one worker while the sequential suite passes:

```bash
julia --project -t 8 test/run_parallel_tests.jl
```

Observed result: 13/14 files passed. `features_tests.jl` failed immediately with:

```text
UndefVarError: `SplitClockProbe` not defined in `Main`
  at test/features_tests.jl:26
```

The sequential command passes all 2,509 assertions because it includes
`test/splitting_tests.jl` before `test/features_tests.jl` in the same `Main` module.

## Root cause

`test/features_tests.jl:26,37` constructs `SplitClockProbe()`, but that type and its
`required_features` method are defined only at `test/splitting_tests.jl:10-13`.
`test/run_parallel_tests.jl:14-28` intentionally starts each test file in a fresh Julia
process, so definitions from the splitting worker cannot exist in the feature worker.
The feature test is therefore order-dependent and not independently runnable.

## Reproduction

```bash
# Fails
julia --project -e 'using Test, BayesianFootball, DataFrames, Dates, InlineStrings; include("test/features_tests.jl")'

# Passes only because the first include leaks the probe into Main
julia --project -e 'using Test, BayesianFootball, DataFrames, Dates, InlineStrings; include("test/splitting_tests.jl"); include("test/features_tests.jl")'
```

## Blast radius

- The advertised concurrent full suite always reports failure.
- `features_tests.jl` cannot be used as the fastest isolated validation tier.
- Production code and package behavior are unaffected.

## Proposed fix and trade-offs

Make `features_tests.jl` self-contained by defining a feature-test-specific probe
(e.g. `FeaturesClockProbe`) and its `required_features` method in that file. An
alternative is a small `test/test_support.jl` included explicitly by both suites;
that avoids duplication but creates a shared helper whose imports must remain stable.
The local probe is preferable because it is only a few lines and keeps worker inputs
obvious.

## Acceptance criteria

1. The isolated `features_tests.jl` command above passes.
2. `julia --project -t 8 test/run_parallel_tests.jl` passes every listed file.
3. `julia --project -t 8 test/runtests.jl` remains green.
4. No test relies on definitions leaked by an earlier include.

## Scope guard

Do not change feature extraction, split-clock production code, test thresholds, or the
parallel scheduler. This ticket is only about making the feature test's probe local and
its file independently executable.
