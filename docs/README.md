# docs/

Written notes for `BayesianFootball`. Runner-level findings live with their stream
in `current_development/<stream>/NOTES.md` / `RESULTS_*.md`; this directory holds the
longer-lived material.

## Top level

- [`turing_ad_performance_guide.md`](turing_ad_performance_guide.md) — **the one to read
  before touching a `@model`.** AD-safety rules, why no `if`/`for` inside `@model`, and how
  to get a compiled `ReverseDiff.GradientTape` down to ~0.64 ms per gradient. Referenced
  throughout `src/`, so it stays at the top level.

## `guides/` — how to do a thing

- [`julia_coding_context_for_agents.md`](guides/julia_coding_context_for_agents.md) — language
  traps, style, Turing API facts and the verification ladder. Read before writing Julia.
- [`experiment_database_and_config_truth_guide.md`](guides/experiment_database_and_config_truth_guide.md)
  — the `mcmc_experiments` database on `mcmc-beast:5432`: schema reference, config truth
  engine, password-safe connection, incremental extension. **§2 draws the line between it and
  `betdb`, the operational database on `archpc:5433`** — read that section first if you are
  unsure which database a task needs.
- [`feature_validation_methodology.md`](guides/feature_validation_methodology.md) — the test
  battery a candidate feature has to pass before it earns a place in a model.
- [`hurdle_columns_guide.md`](guides/hurdle_columns_guide.md) — column reference for the
  Hurdle ROI distributional metric (`src/backtesting/metrics/hurdle_roi.jl`).

## `architecture/` — how the system is put together

- [`ai_agent_infrastructure_and_execution_context.md`](architecture/ai_agent_infrastructure_and_execution_context.md)
  — the two-host topology, the **two databases**, thread pinning, rsync and cache safety, the
  model equations, the `05 → 10` pipeline stages, and the standard agent prompting block.
- [`composable_model_builder_specification.md`](architecture/composable_model_builder_specification.md)
  — the `CountModelBuilder` contract.
- [`feature_engineering_protocol.md`](architecture/feature_engineering_protocol.md) — the
  extractor contract and point-in-time guards.

For the operational layer, the live and replay consoles are documented where they live:
[`current_development/match_day_inference/README.md`](../current_development/match_day_inference/README.md)
(architecture, routes, keyboard map, what the consoles refuse to pretend) and
[`QUICKSTART_LIVE.md`](../current_development/match_day_inference/QUICKSTART_LIVE.md)
(operator loop).

## `setup/` — environment and tooling

- [`claude_code_kaimon_setup.md`](setup/claude_code_kaimon_setup.md) — driving the remote
  Julia REPL on the mcmc-beast server through the kaimon MCP.
- [`kaimon_semantic_search_setup.md`](setup/kaimon_semantic_search_setup.md) — Qdrant
  code-index setup.
- [`kaimon_antigravity_playbook.md`](setup/kaimon_antigravity_playbook.md) — delegating
  token-heavy work to the Antigravity MCP.
- [`agy_remote_execution_guide.md`](setup/agy_remote_execution_guide.md) — the two-host
  execution protocol, pre-flight connectivity checks and the tmux window map.
- [`agy_tmux_agent_and_repl_control_guide.md`](setup/agy_tmux_agent_and_repl_control_guide.md)
  — driving subagent panes and the warm Julia REPL over tmux.

## `research/` — findings and background reading

- [`intensity_fusion_primer.tex`](research/intensity_fusion_primer.tex) /
  [`.pdf`](research/intensity_fusion_primer.pdf) — continuous-time intensity fusion, the
  theory behind the in-play NHPP work.
- [`continuous_time_intensity_fusion_cross_domain.md`](research/continuous_time_intensity_fusion_cross_domain.md)
  — the same idea traced through other domains.
- [`intensity_fusion_artifact.html`](research/intensity_fusion_artifact.html) — rendered
  companion to the primer.
- [`concept_map_and_reading_list.tex`](research/concept_map_and_reading_list.tex) /
  [`.pdf`](research/concept_map_and_reading_list.pdf) — map of the whole modelling stack
  plus the paper reading list.
- [`a_bayesian_in_play_prediction_model_for_football.md`](research/a_bayesian_in_play_prediction_model_for_football.md)
  — in-play model background.
- [`momentum_statistical_analysis.md`](research/momentum_statistical_analysis.md) — the
  SofaScore momentum investigation. Verdict: momentum tracks xG, not goals, with a large
  game-state confound.
- [`liquidity_audit_prompt.md`](research/liquidity_audit_prompt.md) — brief for the
  exchange-liquidity audit.

## `archive/` — shelved layers, kept for the reasoning

These describe work that was built, measured and then **not adopted**. They are kept
because they record *why*, which is the expensive part. Do not treat them as a
description of the current system.

- [`meta_model_design.md`](archive/meta_model_design.md),
  [`l3_meta_model_research.md`](archive/l3_meta_model_research.md),
  [`meta_model_research_notes.md`](archive/meta_model_research_notes.md) — the Layer 3
  meta-model. Blending the market at L3 did not beat blending it inside L1.
- [`l2_bayesian_calibration_research.md`](archive/l2_bayesian_calibration_research.md) —
  Layer 2 Bayesian calibration. Did not improve betting returns.
- [`hurdle_metric_plan.md`](archive/hurdle_metric_plan.md),
  [`hurdle_results_analysis.md`](archive/hurdle_results_analysis.md) — design notes and
  results for the Hurdle metric. The metric itself is live in `src/`; these are its
  working papers.
