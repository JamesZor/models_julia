## 2026-06-05T15:39:40Z
You are worker_fe_1, a software engineer / data engineer.
Your working directory is /home/james/bet_project/BayesianFootball/.agents/worker_fe_1/.

Please perform feature engineering for Milestone 1:
1. Create `current_development/l01_momentum.jl` (the loader/logic file).
   - Implement functions to connect to the database (using connection string: "postgresql://admin:supersecretpassword@100.124.38.117:5432/sofascrape_db").
   - Query the `match_graph` table (joined with `matches`) for the segment's tournament IDs (or retrieve all available records to build a comprehensive feature set).
   - Parse raw JSON string `points` into minutes and values (matching SofaScore's format).
   - Construct `momentum_vector` for each match.
   - Implement a time-weighted AUC function:
     - Home team area: sum of max(0, v_t) * w_t
     - Away team area: sum of max(0, -v_t) * w_t
     - Time-weight w_t: exponential decay weighting later minutes higher, e.g., w_t = exp(-decay_rate * (T - t)). Allow decay_rate to be customizable (e.g. default to 0.03).
   - Return a DataFrame mapping `match_id` to `home_momentum_auc` and `away_momentum_auc`.
2. Create `current_development/r01_momentum.jl` (the runner/execution file).
   - Load the package/dependencies.
   - Execute the functions from `l01_momentum.jl` to fetch and compute momentum features.
   - Verify the DataFrame is non-empty and has the correct columns: `match_id`, `home_momentum_auc`, and `away_momentum_auc`.
   - Save the resulting DataFrame (e.g., as a CSV or JLS file in `current_development/momentum_features.csv`) so it can be used in subsequent analysis.
3. Test compiling and running the scripts.
4. Write a handoff.md in your working directory summarizing your changes, outputs, and build/run verification results.

MANDATORY INTEGRITY WARNING:
> DO NOT CHEAT. All implementations must be genuine. DO NOT
> hardcode test results, create dummy/facade implementations, or
> circumvent the intended task. A Forensic Auditor will independently
> verify your work. Integrity violations WILL be detected and your
> work WILL be rejected.

When done, send a message back to the Project Orchestrator (conversation ID 429c198b-bf9f-4617-ab4a-a7c770a4b4c1) detailing the results.
