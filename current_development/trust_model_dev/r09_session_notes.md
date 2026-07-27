# Session Notes: The Layer 2 Trust Blend & Kelly Staking Engine

## 1. The Core Architecture Philosophy
The most important realization from this session is understanding *why* the pipeline is split into two layers, and where the `tilt_core_grid` fits in.

*   **Layer 1 (The MCMC Physics Engine):** A pure Bayesian Turing model. It is completely blind to bookmaker odds. It learns the physical laws of the game (Home Advantage, Gaussian Random Walk dynamics, Copula correlations).
*   **Layer 2 (The Frequentist Calibrator):** A Logistic GLM. It evaluates how Layer 1 performed historically against the betting market and generates mathematically optimal targets to minimize Log-Loss. 
*   **The Bridge (`tilt_core_grid`):** The Iterative Proportional Fitting (IPF) algorithm. This is the physical mechanism of the Trust Blend. It takes the 4,000 MCMC samples from Layer 1 and physically shifts the entire distribution to align with the Layer 2 targets. **Crucially, this retains the deep covariance and uncertainty of the MCMC distribution while perfectly calibrating its mean.**

*Note: The tilt matrix is NOT part of the Kelly Staking system. It executes entirely within the predictive modeling phase to finalize the probabilities before betting.*

---

## 2. The Staking Math & The Hessian (`r09_math_demo.jl`)
We isolated a single match (Drogheda vs Shamrock Rovers) to observe the Kelly Staking math in a vacuum:
1.  **The Return Matrix ($R$):** Defines the payoffs for the 144 scorelines across all markets.
2.  **The Gradient ($\nabla G$):** The directional vector pointing toward the highest expected log-wealth. We saw that Layer 1's gradient heavily favored longshots. Layer 2 tilted the grid, instantly pointing the gradient toward the heavy favorites.
3.  **The Hessian ($\nabla^2 G$):** The covariance matrix of the bets. It mathematically penalizes correlated risks. If the Gradient is the gas pedal, the Hessian is the brakes.
4.  **U-MC Shrinkage ($k^*$):** Because the optimizer is hyper-sensitive to the input probabilities (Jacobian), $k^*$ dynamically shrinks the final stakes based on the variance of the 4,000 MCMC samples.

---

## 3. The Grand Portfolio Backtest (`r09_ou_btts_backtest.jl`)
We ran a chronological backtest combining the three core families (`1X2`, `OverUnder`, `BTTS`) to evaluate the true Portfolio Sharpe and Wealth Multiplier.

**The Counter-Intuitive Result:**
*   **Layer 1 (No Trust):** Made **11.13 units** of profit, but suffered a **Negative Empirical Geometric Growth Rate (`-0.0048`)**.
*   **Layer 1 + Layer 2 (Trust Blend):** Made only **3.40 units** of profit, but achieved a **Positive Empirical Geometric Growth Rate (`+0.0005`)**.

**Conclusion:** Raw arithmetic profit is an illusion in structural betting. Layer 1 had an insanely lucky run on a massive 98-unit turnover. The mathematics of the Kelly Hurdle (`hurdle_G`) perfectly predicted that Layer 1's reckless overconfidence would eventually cause a geometric bankruptcy. Layer 2 slashed the turnover to 32 units, sacrificed raw PnL, and mathematically rescued the portfolio from a negative-growth death spiral.

---

## 4. Validating the Tilt Matrix (`scratch/eda_tilt_calibration.jl`)
To dispel doubts that the `tilt_core_grid` was artificially distorting the model, we ran a deep EDA across the entire dataset, bucketing matches by the bookmaker's implied probability.

**The Finding:**
Layer 1 suffers from a massive **Mean-Reverting Bias** (Favorite-Longshot Bias). It failed to respect elite teams (predicting heavy favorites at 47% when they empirically won 70% of the time). 

The Tilt Matrix effortlessly cured this bias:
*   On heavy underdogs (19% empirical win), Layer 1 wildly overestimated them at 35%. The Tilt Matrix violently dragged the prediction down to **21%**.
*   On heavy favorites (70% empirical win), Layer 1 wildly underestimated them at 47%. The Tilt Matrix boosted the prediction up to **67%**.

The Tilt Matrix is not a mathematical hack. It is the exact cure to Layer 1's fundamental calibration defect, bridging the gap between a theoretical physics engine and a fully operational, compounding betting syndicate.
