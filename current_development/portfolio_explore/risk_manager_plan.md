# Risk Manager Portfolio Architecture Plan

## 1. Mathematical Translation of the Drawdown Constraint

The paper defines a probabilistic drawdown constraint to prevent catastrophic ruin:
*   **$D$**: Maximum acceptable drawdown (e.g., `0.8` for a 20% drawdown limit).
*   **$\beta$**: Probability tolerance of breaching $D$ (e.g., `0.01` for a 1% chance).
*   **$\lambda$**: The derived risk multiplier $\lambda = \frac{\ln(\beta)}{\ln(D)}$.

The constraint states that the expected value of the penalty function over the match outcomes $\Omega$ must be $\le 1$:
$$ E\left[ (1 + k \cdot R)^{-\lambda} \right] \le 1 $$

Where:
*   $R$ is the net return of our proposed stakes across all 144 possible match scores.
*   $k \in (0, 1]$ is the **Kelly shrinkage factor**.

### The Root-Finding Algorithm
Since $E[(1 + kR)^{-\lambda}]$ is convex with respect to $k$, equals $1.0$ at $k=0$, and has a negative slope at $k=0$ (because it's a positive Expected Value bet), the function dips below 1.0 and then climbs back up.
1. We first test $k=1.0$ (using our standard blended stakes $a_{net}$). If $E \le 1.0$, the bet is naturally safe, and $k=1.0$.
2. If $E > 1.0$, the variance is too high. We use a fast Bisection root-finding algorithm to find the exact $k \in (0, 1)$ where $E = 1.0$.
3. We multiply our final stakes by this dynamic $k$.

## 2. Loader Implementation (`l03_risk_manager.jl`)

We will build this entirely on top of our existing `l02` architecture to ensure absolute consistency.

**New Structs:**
```julia
struct RiskConfig
    D::Float64
    beta::Float64
end
```

**Core Functions:**
*   `solve_drawdown_multiplier(p_model_vec, returns_vec, lambda)`: Executes the bisection search to find $k$.
*   `evaluate_match_risk_managed(...)`: 
    1. Runs the standard `optimize_portfolio` to get base stakes (using our optimal $\alpha = 0.35$).
    2. Calculates the 144-element return vector $R = \text{Return Matrix} \times \text{Stakes}$.
    3. Calls `solve_drawdown_multiplier` to get $k$.
    4. Applies $k$ to the stakes and calculates P/L.
    5. Returns a DataFrame tracking both the *Base Stake* and the *Risk-Managed Stake* so we can compare them side-by-side.

## 3. Smoke Test (`r04_risk_smoke.jl`)

We will run a 50-match mini-batch with a strict constraint (e.g., $D=0.85$ (15% drawdown), $\beta=0.01$). 
We will verify that:
1. The solver successfully finds $k < 1.0$ on highly volatile matches (e.g., betting heavy on longshot Correct Scores or Underdogs).
2. The logic correctly leaves $k = 1.0$ on safe, low-variance matches (e.g., betting on heavy favorites where variance is low).

## 4. Full Backtest (`r05_risk_full_backtest.jl`)

We will run the full 710 match dataset. The output DataFrame will have side-by-side metrics:
*   `base_net_pl` vs `risk_net_pl`
*   `base_max_dd` vs `risk_max_dd`
*   `base_sharpe` vs `risk_sharpe`

We expect the Risk Manager to slightly reduce total profit, but significantly reduce the Maximum Drawdown and improve the Sharpe ratio by automatically slamming the brakes on dangerous, high-variance bets while letting safe bets run at full optimal $\alpha=0.35$ Kelly.

---
**Does this mathematical interpretation of the line-search for $k$ align with your vision? If so, I will immediately generate `l03_risk_manager.jl`.**
