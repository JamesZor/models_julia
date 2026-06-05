## 2026-06-05T16:03:27Z
You are worker_stats_2, a data scientist / developer.
Your working directory is /home/james/bet_project/BayesianFootball/.agents/worker_stats_2/.

Please refine the statistical validation code in `current_development/l02_momentum_analysis.jl` and verify testing:
1. In `current_development/l02_momentum_analysis.jl`:
   - Replace the use of `HypothesisTests.CorrelationTest` with `HypothesisTests.PearsonCorrelationTest` in `pearson_correlation_test`.
   - In the manual t-test fallback inside `pearson_correlation_test`, add clamping to $r$ to avoid `DomainError` when `1 - r^2` becomes negative due to floating-point rounding issues (e.g. clamp $r$ to `[-1.0 + 1e-15, 1.0 - 1e-15]`).
   - In `run_full_validation_pipeline`, ensure that if `expectedGoals_home` or `expectedGoals_away` columns are missing from the `stats` DataFrame (e.g., if no segment has them), we create them as all `missing` before selecting them, preventing a potential crash.
2. In `test/momentum_tests.jl`:
   - Clean up any duplicate imports of `l01_momentum.jl` if it is already included via `l02_momentum_analysis.jl` to prevent redefinition warnings.
3. Propose running the runner script `r02_momentum_analysis.jl` to generate `momentum_statistical_analysis.md` at the project root:
   - `julia --project current_development/r02_momentum_analysis.jl`
4. Propose running the tests:
   - `julia --project test/runtests.jl`
5. Write a handoff.md in your working directory.

MANDATORY INTEGRITY WARNING:
> DO NOT CHEAT. All implementations must be genuine. DO NOT
> hardcode test results, create dummy/facade implementations, or
> circumvent the intended task. A Forensic Auditor will independently
> verify your work. Integrity violations WILL be detected and your
> work WILL be rejected.

When done, send a message back to the Project Orchestrator (conversation ID 429c198b-bf9f-4617-ab4a-a7c770a4b4c1) detailing the results.
