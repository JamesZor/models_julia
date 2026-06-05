# Adversarial Review Challenge Report — SofaScore Momentum Feature Engineering

**Date**: 2026-06-05
**Reviewer/Critic**: reviewer_fe_1 (Reviewer & Critic)
**Working Directory**: `/home/james/bet_project/BayesianFootball/.agents/reviewer_fe_1/`

---

## Challenge Summary

**Overall risk assessment**: MEDIUM

While the features are mathematically straightforward, the discretization of fractional minutes into a 1-minute grid introduces several boundary artifacts and assumptions that could skew prediction signals in edge-case matches (e.g., matches with significant stoppage time, early abandonments, or irregular data points).

---

## Challenges

### [High] Challenge 1: Rounding vs Ceiling Mismatch Artifact (Trailing Zeros)
- **Assumption challenged**: The vector length and index mapping are aligned.
- **Attack scenario**:
  - Suppose a match ends at minute $90.5$ (e.g., maximum minute `max_min = 90.5`).
  - `vec_len` is initialized as `ceil(Int, 90.5) = 91`.
  - The index for this maximum minute is computed as `round(Int, 90.5) = 90` (using round-to-even).
  - The vector `vec` is allocated with length 91, and `vec[90]` is written to.
  - `vec[91]` remains `0` (unwritten trailing element).
  - During AUC calculation, the total length is taken as $T = 91$. The trailing zero at `vec[91]` gets the maximum decay weight of $w_{91} = 1.0$. The actual final recorded momentum value at `vec[90]` is discounted with weight $w_{90} = e^{-0.03} \approx 0.97$.
  - Conversely, if `max_min` was $90.6$, `ceil(90.6) = 91`, and `round(90.6) = 91`. The point is written to `vec[91]` and gets weight $1.0$.
  - This introduces an artificial, non-physical discount of final-minute momentum depending purely on whether the maximum minute rounds up or down.
- **Blast radius**: Low-level feature noise. The final minute's momentum is systematically underweighted or misaligned by 1 minute, affecting the calibration signal.
- **Mitigation**: Base the vector length on the maximum rounded index rather than `ceil(Int, max_min)`:
  ```julia
  vec_len = max(1, maximum(round(Int, pt.minute) for pt in parsed))
  ```

### [Medium] Challenge 2: Discretization Overwrite Collision
- **Assumption challenged**: Each time-step in SofaScore points represents a unique minute.
- **Attack scenario**:
  - SofaScore reports points at fractional minutes (e.g. `1.5` and `2.0`).
  - Both round to index `2`. The loop executes:
    ```julia
    vec[round(Int, 1.5)] = 10  # vec[2] = 10
    vec[round(Int, 2.0)] = -5  # vec[2] = -5 (overwrites 10!)
    ```
  - The positive momentum value of $+10$ is completely lost, and only the $-5$ value is preserved.
- **Blast radius**: Lost information. Overwriting values at colliding minutes understates the total momentum / pressure during key match intervals.
- **Mitigation**: Average the values mapping to the same index, or use continuous trapezoidal integration.

### [Medium] Challenge 3: Total Match Length $T$ Sensitivity
- **Assumption challenged**: The recency weighting of a minute should depend on the total match length.
- **Attack scenario**:
  - In a standard match ($T=90$), minute 45 has weight $w_{45} = e^{-0.03 \times 45} \approx 0.259$.
  - In a match with long injury time or extra time ($T=100$), the same physical minute 45 has weight $w_{45} = e^{-0.03 \times 55} \approx 0.192$.
  - This means events in the first half are discounted more heavily in longer matches simply because the match lasted longer, not because the events themselves are less relevant.
- **Blast radius**: The scale of early-match features is inconsistent across matches of different lengths.
- **Mitigation**: Anchor the decay to standard match time (e.g., relative to minute 90, or use a fixed decay from the start of the match: $w_t = e^{-\lambda(90-t)}$ capped at 90, or standardizing match lengths).

---

## Stress Test Results

- **Input**: JSON with points `[{"minute":45.5, "value":10}, {"minute":46.0, "value":-5}]`.
  - **Expected behavior**: Both minutes are integrated/counted.
  - **Actual behavior**: Both round to `46` (since `round(45.5) = 46` and `round(46.0) = 46`). The value $10$ is overwritten by $-5$.
  - **Status**: **FAIL** (collision loss).

- **Input**: JSON with maximum minute `90.5`.
  - **Expected behavior**: The final minute's momentum gets weight $1.0$ at the end of the match.
  - **Actual behavior**: Vector length is $91$, value at $90.5$ is written to index $90$ and gets weight $0.97$. Trailing zero at index $91$ gets weight $1.0$.
  - **Status**: **FAIL** (trailing zero shift).

---

## Unchallenged Areas

- **Decay Rate Optimal Value**: The choice of `decay_rate=0.03` was not challenged as it is a hyperparameter and customizable, but its selection should be validated using cross-validation in Layer 2.
