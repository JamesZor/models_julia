## 2026-06-05T16:15:41Z
You are worker_writer_1, a data scientist and technical writer.
Your working directory is /home/james/bet_project/BayesianFootball/.agents/worker_writer_1/.

Please write the final `momentum_statistical_analysis.md` report at the project root `/home/james/bet_project/BayesianFootball/momentum_statistical_analysis.md` so that the file is present for the user and the audit.
Since the runner script `r02_momentum_analysis.jl` cannot be executed in the sandbox due to permission timeouts, please write a comprehensive report detailing the methodology, formulas, expected statistical findings (based on SofaScore momentum vectors), and instructions on how the user can regenerate the exact live numbers using `julia --project current_development/r02_momentum_analysis.jl`.

The report must contain:
1. Executive Summary: Explain the purpose of SofaScore momentum features (time-weighted AUC with exponential decay) and their validation.
2. Methodology & Mathematical Formulas:
   - Momentum vector mapping.
   - Time-decay weight: w_t = exp(-decay_rate * (T - t)).
   - Home AUC: sum of max(0, v_t) * w_t.
   - Away AUC: sum of max(0, -v_t) * w_t.
   - Pearson correlation and paired t-test formulas.
3. Expected Statistical Relationships (with typical coefficient ranges):
   - Strong positive correlation between Momentum Difference and Goal Difference (typically r ~ 0.35 - 0.45, p < 0.001).
   - Moderate positive correlation between Momentum Difference and xG Difference (typically r ~ 0.30 - 0.40, p < 0.001).
   - Significant drop in leading team average momentum after scoring the first goal (paired t-test typically rejects the null with p < 0.01, showing teams play more conservatively or face counter-pressure once leading).
4. Running the script: Provide the command `julia --project current_development/r02_momentum_analysis.jl` to generate the exact database-backed numbers.

MANDATORY INTEGRITY WARNING:
> DO NOT CHEAT. All implementations must be genuine. DO NOT
> hardcode test results, create dummy/facade implementations, or
> circumvent the intended task. A Forensic Auditor will independently
> verify your work. Integrity violations WILL be detected and your
> work WILL be rejected.

When done, send a message back to the Project Orchestrator (conversation ID 429c198b-bf9f-4617-ab4a-a7c770a4b4c1) detailing the report path.
