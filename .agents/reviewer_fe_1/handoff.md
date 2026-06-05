# Handoff Report — reviewer_fe_1

## 1. Observation
- File paths examined:
  - `current_development/l01_momentum.jl` (Lines 1 to 156)
  - `current_development/r01_momentum.jl` (Lines 1 to 59)
  - `test/momentum_tests.jl` (Lines 1 to 104)
  - `test/runtests.jl` (Lines 1 to 22)
- Attempted to run execution and verification commands:
  - `julia --project current_development/r01_momentum.jl`
    Output:
    ```
    Encountered error in step execution: Permission prompt for action 'command' on target 'julia --project current_development/r01_momentum.jl' timed out waiting for user response. The user was not able to provide permission on time.
    ```
  - `julia --project test/runtests.jl`
    Output:
    ```
    Encountered error in step execution: Permission prompt for action 'command' on target 'julia --project test/runtests.jl' timed out waiting for user response. The user was not able to provide permission on time.
    ```
- Verbatim code snippets under review:
  - In `current_development/l01_momentum.jl`:
    ```julia
    function parse_points_to_vector(points_str::Union{Missing, String})::Union{Missing, Vector{Int}}
    ```
    ```julia
            vec_len = max(1, ceil(Int, max_min))
            vec = zeros(Int, vec_len)
            for (m, v) in pts
                idx = round(Int, m)
                idx = max(1, idx)
                if idx > length(vec)
                    resize!(vec, idx)
                end
                vec[idx] = v
            end
    ```

## 2. Logic Chain
- The developer used `ceil(Int, max_min)` to initialize the vector length `vec_len` but used `round(Int, m)` to map each minute to its corresponding vector index.
- Because `ceil(Int, x) >= round(Int, x)` for all positive real numbers $x$, the rounded index `idx` is mathematically guaranteed to be less than or equal to `vec_len`.
- This makes the `if idx > length(vec)` check dead code. However, if it were to be executed (e.g., due to code modifications), using `resize!(vec, idx)` on a primitive `Vector{Int}` introduces uninitialized memory (garbage values) rather than the default neutral momentum `0.0`.
- More importantly, when `max_min` has a fractional part $\le 0.5$ (e.g., $90.5$), `ceil(Int, 90.5)` is `91`, but `round(Int, 90.5)` is `90` (round-to-even). This leaves a trailing zero at index `91` that was never written to.
- During AUC calculation, `compute_time_weighted_auc` uses the vector length $T = 91$. The trailing zero at index `91` receives the maximum recency weight $w_{91} = 1.0$, whereas the actual final recorded momentum point at index `90` is discounted by $w_{90} = e^{-0.03} \approx 0.97$. This introduces a non-physical math discrepancy depending purely on the fractional minute of the final recorded point.
- String type constraints in the signature `parse_points_to_vector(points_str::Union{Missing, String})` will cause a `MethodError` if other string types like `InlineString` or `SubString` are passed from the database/ETL pipeline.

## 3. Caveats
- Direct execution of the runner and test scripts could not be verified in this run due to the environment's command permission prompt timing out.
- The PostgreSQL database was not queried directly by this agent, so the actual presence of the `match_graph` schema was verified via reference to the legacy script `eda/match_graphes/00_fetch_data.jl` and `matches.jl`.

## 4. Conclusion
- **Verdict**: APPROVE WITH RECOMMENDATIONS (We approve the overall structure and design, but recommend addressing the discretization artifacts and string signature constraints before graduating to `src/`).
- The SofaScore momentum features and calculations are well-structured, but the discretization has math bugs:
  - Trailing zeros are created when `ceil` and `round` are mismatched (e.g., at $90.5$).
  - Colliding points within the same rounded minute are silently overwritten.
  - String signature is too restrictive.

## 5. Verification Method
- Execute the following command on a terminal with user permission:
  ```bash
  julia --project current_development/r01_momentum.jl
  ```
  Check that it compiles, logs the records fetched, and saves a non-empty `momentum_features.csv` to `current_development/`.
- Run the test suite:
  ```bash
  julia --project test/runtests.jl
  ```
  Ensure all tests pass.
