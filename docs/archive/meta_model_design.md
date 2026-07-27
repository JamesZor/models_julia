# Meta Model (Layer 2) Architecture & Design

This document breaks down the design, mathematics, and processing pipeline of the predictive Meta Model system.

## 1. System Architecture (ASCII Overview)

```text
========================================================================================
                       THE META MODEL (LAYER 2) REGIME FILTER
========================================================================================

[ LAYER 1: Turing NUTS ]            [ THE MARKET ]
   Historical Data                    Odds Data
          │                               │
          ▼                               ▼
  L1 OOS Predictions (p_L1)       Implied Fair Odds (m_i)
          │                               │
          └─────────────┬─────────────────┘
                        │
                        ▼
            [ THE META MODEL MIXER ]
            Q_i = θ_t * p_L1 + (1 - θ_t) * m_i
                        │
                        ▼
      [ GAUSSIAN RANDOM WALK (GRW) DYNAMICS ]
      How much do we trust L1 (θ_t) in week W?
                        │
                        ▼
        [ PREDICTIVE THRESHOLD (BET GATE) ]
        Expected θ for NEXT week compared to 
        the historical median of all past θ's.
                        │
            ┌───────────┴───────────┐
            │                       │
      θ_pred >= Median         θ_pred < Median
      [ GOOD REGIME ]           [ BAD REGIME ]
            │                       │
     Bet using L1 PPD!            Skip Bet
========================================================================================
```

---

## 2. The Mathematics

The Meta Model asks a simple question: **"In any given week, how much should we trust our Layer 1 model relative to the bookmaker?"**

### The Mixture Equation
For a specific match $i$ in week $w$, the true probability $Q_i$ is a weighted blend:
$$Q_i = \theta_{w} \cdot p^{L1}_i + (1 - \theta_{w}) \cdot m_i$$
- $p^{L1}_i$ = Our Layer 1 prediction
- $m_i$ = The bookmaker's fair probability
- $\theta_{w}$ = **The Trust Weight (0 to 1)** for week $w$.

### The Dynamics (How $\theta$ changes over time)
We don't assume L1 is perfectly consistent. We model $\theta_w$ as a **Gaussian Random Walk (GRW)** in logit-space. This means the trust weight "drifts" week to week.

$$\text{logit}(\theta_w) = \alpha + \sum_{k=1}^{w} (z_k \cdot \sigma_{\text{GRW}})$$
- **$\alpha$**: The baseline global trust (intercept).
- **$\sigma_{\text{GRW}}$**: The volatility of the regime changes. A high $\sigma$ means L1 goes through wild swings of being highly accurate to highly inaccurate. A low $\sigma$ means L1 is stable.
- **$z_k$**: The standard normal shock for week $k$.

Because the shocks $z_k \sim \mathcal{N}(0, 1)$ have a mean of 0, the expected value of next week's drift is exactly this week's drift! 

> [!TIP]
> This mathematical property of the GRW is why the 1-step-ahead prediction for week $W$ is simply the terminal value from the chain trained on week $W-1$. 

### The Likelihood (Learning from Outcomes)
The model observes the actual match outcomes $Y_i \in \{0, 1\}$ (e.g., did the match go Under 2.5?). It learns the parameters by maximizing the Bernoulli likelihood:
$$Y_i \sim \text{Bernoulli}(Q_i)$$
If L1 predicted 0.8 and the market predicted 0.4, and the outcome was 1, the model learns to push $\theta$ higher for that week.

---

## 3. The Processing Workflow (How the Code Runs)

To evaluate the Meta Model without "look-ahead bias" (peeking into the future), we use an **Expanding Window Rolling Workflow**.

### Step 1: Weekly Fold Generation (`workflow.jl`)
We group the 1,400+ historical matches by **Match Week** ($W=1, 2, \ldots, 80$). 
This creates 80 distinct "folds".

### Step 2: The Training Queue
For fold $K$, we give the Meta Model all matches from weeks $1$ to $K$.
- **Task 1**: Train Meta on Week 1.
- **Task 2**: Train Meta on Weeks 1 + 2.
- **Task 3**: Train Meta on Weeks 1 + 2 + 3.
...
- **Task 80**: Train Meta on Weeks 1 to 80.

Because these tasks don't depend on each other, `workflow.jl` throws all $80 \times 4 \text{ chains} = 320$ tasks into a queue and processes them concurrently on your 32 CPU threads.

### Step 3: Predictive Extraction (`staking.jl`)
Now we simulate walking through time as a bettor:
1. We are about to bet on the matches in **Week 2**.
2. We look at the Meta Model trained *only* on **Week 1** (Fold 1).
3. We extract $\theta_{Week 1}$ from that model. This is our prediction for Week 2 ($\theta_{pred}$).
4. We compare $\theta_{pred}$ to our threshold (the median of all historical $\theta$s).
5. If it's a **GOOD REGIME**, we use Layer 1's distribution and `BayesianKelly` to place stakes.
6. We repeat this process: to bet on Week 3, we look at the model trained on Weeks 1+2.

### Step 4: The "Cold Start" Prior
At the very beginning (Week 2), we don't have a history of $\theta$'s to calculate a median threshold. 
**The Solution:** We extract the global baseline trust ($\alpha$) from Fold 1 and use that to seed our history array. This sets our starting threshold to whatever historical baseline the model mathematically deduced from past seasons.
