# Vectorized Alpha Optimization Plan

## 1. Goal
We want to discover the "Theoretical Upper Limit" of the model by optimizing a separate fractional Kelly shrinkage parameter ($\alpha$) for every single selection type (e.g., Home, Away, Draw, Over 2.5, BTTS Yes, etc.). This will reveal exactly which markets the model prices best, and which ones it should size down or ignore entirely.

## 2. Granularity & Dimensionality
The $\alpha$ vector will map directly to the specific outcomes in the `markets_config`.
Based on our standard setup (1X2, BTTS, and 5 O/U lines), the optimizer will control a 15-dimensional vector:
*   `1X2_Home`, `1X2_Draw`, `1X2_Away`
*   `BTTS_Yes`, `BTTS_No`
*   `O/U 0.5_Over`, `O/U 0.5_Under`, ... up to `4.5`.

## 3. The Objective Function
We will construct a loss function that `Optim.jl` can minimize:
1.  Take the 15-dimensional vector $X \in [0, 1]^{15}$ provided by the optimizer.
2.  Map $X$ into an `alpha_dict`.
3.  Run the full backtest sequence, scaling the unconstrained stakes `a_net` by the specific $\alpha$ for each selection.
4.  Calculate the sequential Bankroll trajectory.
5.  Calculate the **Martin Ratio** (or Final Bankroll, depending on configuration).
6.  Return `-Martin_Ratio` (so the optimizer minimizes it).

## 4. Extreme Performance Optimization (The Cache)
If we just ran the normal backtest inside an `Optim` loop, evaluating 1,000 parameter combinations would take $\sim 15$ minutes because calculating the `score_matrix` is mathematically expensive. 

To make this blazing fast:
1.  We will do a **Pre-computation Pass** over the 710 matches, extracting the `score_matrix`, `odds_map`, and `match_model_prob` exactly once. 
2.  We will cache these in a fast array in memory.
3.  The Objective Function will simply loop over this cache and run the instantaneous Kelly solver (`IPNewton` or `Fminbox`).
4.  This should reduce the objective function evaluation time to a few milliseconds, allowing `Optim.jl` to explore thousands of combinations instantly!

## 5. File Architecture
*   **Modify `l02_portfolio_backtest.jl`**: Add a `VectorAlphaConfig` struct and overload `optimize_portfolio` to apply the vectorized shrinkage `a_blended_net = a_net .* alpha_vec`.
*   **Create `r08_vector_alpha_optim.jl`**: The new runner script that handles the caching, constructs the objective function wrapper, and executes the `Optim.jl` search using `Fminbox(NelderMead())` bounded between `0.0` and `1.0`.
