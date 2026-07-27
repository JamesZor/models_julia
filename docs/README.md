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

- [`feature_validation_methodology.md`](guides/feature_validation_methodology.md) — the test
  battery a candidate feature has to pass before it earns a place in a model.
- [`hurdle_columns_guide.md`](guides/hurdle_columns_guide.md) — column reference for the
  Hurdle ROI distributional metric (`src/backtesting/metrics/hurdle_roi.jl`).

## `setup/` — environment and tooling

- [`claude_code_kaimon_setup.md`](setup/claude_code_kaimon_setup.md) — driving the remote
  Julia REPL on the mcmc-beast server through the kaimon MCP.
- [`kaimon_semantic_search_setup.md`](setup/kaimon_semantic_search_setup.md) — Qdrant
  code-index setup.
- [`kaimon_antigravity_playbook.md`](setup/kaimon_antigravity_playbook.md) — delegating
  token-heavy work to the Antigravity MCP.

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
