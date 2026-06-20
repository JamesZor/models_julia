# In-Play Position Management & the NHPP Intensity Model — Findings + Mathematics

*A worked consolidation of the in-play position-management research and the Non-Homogeneous Poisson
Process (NHPP) intensity model. Ireland Premier Division, DCMH pre-game ledger (258 matches / 1,574 bets)
+ in-play Betfair ticks + goal-time incidents. Code: `current_development/basic_hedging/`
(`r01_portfolio_kelly_hedge.jl`, `r02_model_driven_hedge.jl`) and
`current_development/match_inplay_explore/l08_nhpp_turing.jl`.*

---

## 0. The question

The pre-game model (`DCMH_HalfLife_60`, a player-level Dixon-Coles market-hierarchical engine) places bets
at the closing line, staked with `BayesianKelly`. **Goal: use an in-game model to manage those positions —
hedge out / hold / add as game state and time change the expected value — to improve the *growth rate*.**

Two measurement principles used throughout:
- **Judge by log-growth, not ROI.** ROI ignores stake size, frequency and variance; a +EV action can
  still destroy geometric growth.
- **Validate out-of-sample.** With ~258 matches a single split is dominated by noise; we use repeated
  k-fold CV with *pre-committed* (not per-fold-tuned) parameters.

---

## Part I — Position Management (the betting side)

### 1. Log-growth and the Kelly objective

A bankroll compounding multiplicatively across sequential matches grows at rate

$$ G = \sum_{m} \ln\!\left(1 + r_m\right), \qquad r_m = \sum_{i \in \text{bets}(m)} s_i \, o_i, $$

where for bet $i$, $s_i$ is the stake as a fraction of bankroll and $o_i = (\text{odds}_i - 1)$ if it wins,
$-1$ if it loses. Maximising $\mathbb{E}[\ln(1+r)]$ is the Kelly criterion. For a single bet with win
prob $p$ and net odds $b$,

$$ f^\star = \arg\max_f \; p\ln(1+bf) + (1-p)\ln(1-f) = p - \frac{1-p}{b}. $$

**`BayesianKelly` (Baker–McHale)** sizes over the *posterior* of $p$. Given posterior draws $\{q_s\}$, it
takes each draw's naive Kelly stake $s_q = \max(0, q - (1-q)/b)$ and shrinks all of them by one factor $k$:

$$ k^\star = \arg\max_k \; \frac{1}{S}\sum_s \Big[\bar p\,\ln(1 + b\,k\,s_q) + (1-\bar p)\ln(1 - k\,s_q)\Big],
\qquad \text{stake} = s_{\bar p}\cdot k^\star, $$

with $\bar p = \mathbb{E}[q]$. The spread of $\{s_q\}$ penalises uncertainty → smaller $k^\star$.
(Code: `src/signals/implementations/kelly.jl`.)

### 2. Finding 1 — size the *joint* book, not each bet (the dominant lever)

`BayesianKelly` is correct **per bet**, but a match fires ~6 **simultaneous, correlated** bets (nested
Over/Under lines + BTTS + 1X2). Summing independent Kelly fractions over-stakes the joint position:

| metric | value |
|---|---|
| mean joint stake / match | 0.40 of bankroll |
| **max joint stake / match** | **2.17 (217%)** |
| worst-match return | **−143% → BANKRUPTCY** ($G = -\infty$) |

Capping the *summed* per-match stake at $c$ (scale all simultaneous stakes by $\min(1, c/\sum_i s_i)$) gives
the textbook Kelly parabola at the **portfolio** level:

| joint cap $c$ | 0.10 | 0.15 | **0.20** | 0.25 | 0.30 | 0.45 | 1.0 |
|---|---|---|---|---|---|---|---|
| log-growth $G$ | 2.33 | 2.89 | **3.08** | 3.02 | 2.46 | −0.29 | −6.05 |

Optimal joint cap $\approx 0.20$ (×21.7 terminal bank). **Independent per-bet Kelly bankrupts; capping the
joint book is the single biggest lever** — far larger than any hedging effect.

### 3. Finding 2 — select markets by growth contribution, not ROI

The marginal value of a selection is its **leave-one-out log-growth contribution**
$\Delta G_j = G(\text{full}) - G(\text{without } j)$, *not* its ROI. This flips the ranking:

- `1X2 home`: $\Delta G = -1.71$ (worst — drop).
- `over_15`: ROI **+1.2 %** but $\Delta G = -0.29$ → **drop despite positive ROI** (a low-odds favourite
  that adds variance without edge).
- `under_45`, `over_35`: **negative ROI** but $\Delta G > 0$ → **keep** (they pay off when the rest of the
  book loses — internal diversifiers).

Dropping the four growth-negative selections raises hold-only $G$ from **3.08 → 5.18** (vs 4.96 if you
(wrongly) drop by ROI). The model is good on the underdog/unders side, weak on home favourites and
high-over tails.

### 4. Finding 3 — the in-game model times exits; the math of the signal

Hold a back bet on selection $X$ entered at odds $O$. At each in-play tick the in-game model gives
$P_{\text{model}}(X \mid \text{score}, t)$; the live market implies $1/\text{price}$. Define the **edge**

$$ e_t = P_{\text{model}}(X) - \frac{1}{\text{price}_t}. $$

**Exit rule:** lay off the position the first tick $e_t \le \tau$. Partial cash-out of fraction $\varphi$
of a back bet (lay at live price $o_x$) has per-unit-stake payoff (commission $\kappa$ on positive locked
profit):

$$ \text{pnl}(\varphi) = \varphi\,c + (1-\varphi)\,h, \quad
c = \begin{cases}(O/o_x - 1)(1-\kappa) & O/o_x - 1 > 0\\ O/o_x - 1 & \text{else}\end{cases},\;\;
h = \begin{cases}O-1 & \text{win}\\ -1 & \text{lose}\end{cases}. $$

**Execution honesty (critical):** filling *at* the signal price is **stale-price lookahead** (the l04
lesson). Executing at the **next** price ≥ signal+lag is realistic; the result is stable across 1/3/5-min
lag (it does *not* collapse), because we manage an existing position on a genuine edge-decay signal.

| strategy (cap 0.20, full exit, fwd 3-min) | log-growth |
|---|---|
| hold to settlement | 3.076 |
| fixed clock exit @70' | 2.667 *(worse than hold)* |
| model exit, $\tau=0$ | 3.734 |
| **model exit, $\tau=-0.05$** | **4.423** |

Two structural results:
- **Optimal $\tau$ is a small *negative* edge** (exit once the market clearly overtakes the model, ~5 pts),
  not 0 — avoids churning on noise.
- **When the model fires, full exit beats partial** (4.42 > 3.82 at $\varphi{=}0.5$). The fixed clock
  hedges *partially* out of ignorance; the model *knows* the edge is gone, so it exits decisively.
  *Partial hedging is what you do without a model.*

**Out-of-sample validation (5-fold × 6 repeats over matches, paired exit-vs-hold uplift):**
- *Tuning* $(\tau,c)$ per fold: uplift $+0.056 \pm 0.109$, $t=0.51$ — selection noise swamps signal.
- *Pre-committed* $\tau=-0.05$: uplift $\mathbf{+0.306 \pm 0.059}$, $\mathbf{t=5.21}$, 73 % folds positive;
  per-fold mean × 5 ≈ 1.53 = the full-sample uplift ⇒ **no overfitting at a fixed threshold.**

**Lesson:** pre-commit the exit threshold from theory; do *not* data-mine it.

### 5. Finding 4 — the asymmetry: exit yes, add no

Symmetric add rule (back more when $e_t \ge \tau_{\text{add}}$):

| | log-growth |
|---|---|
| hold | 3.08 |
| add (size 0.5) | 2.33 |
| add (size 1.0) | 0.55 |

Adding **destroys growth even though the adds are +EV** (ROI +11.5 %, hit 45.6 % @ avg odds 4.95). The
reason is the convex log penalty: extra correlated high-variance exposure on an already-capped book costs
more than the EV gains. **The model is valuable on the *reduce* side only.** Every wealth-compounding
lever here is exposure-*reducing* (cap, curate, exit); the exposure-*increasing* ones bleed growth.

### 6. Findings 5–6 — two refinements that didn't help (honest negatives)

- **Distributional `BayesianKelly` sizing** (continuous $\varphi$ from the uncertainty-aware target,
  per-draw $P_{\text{model}}$): OOS 4.29 vs 4.42 — *no gain*. At the trigger it chooses full exit 86 % of
  the time (a gone edge ⇒ target ≈ 0 regardless of confidence); the posterior is tight (std ≈ 0.035) so
  shrinkage rarely reaches the partial zone. Would only matter with a wider posterior.
- **Margin-aware game state** (2-up vs 1-up): the *behaviour* term is binary (behind/level/ahead); the
  *score magnitude* already enters correctly via the score arithmetic. A continuous `goal_diff` fits
  *worse* (the effect is a threshold, not a gradient) and hedge growth is unchanged.

---

## Part II — The In-Play Intensity Model (the mathematics)

### 7. What we had: a flat-rate count model

Per (match × tick × side): target = remaining goals, with a **time-exposure offset**:

$$ \log \mu = \alpha + \beta^\top X + \underbrace{\log\frac{90 - t}{90}}_{\text{offset}}, \qquad
y \sim \text{Poisson}(\mu), $$

with $X$ = (time, time$^2$, is\_home, trailing, leading, log pre-game $\lambda$). This assumes a
**constant rate over the entire remaining window** — the rate may depend on the state *at the tick*, but it
is held flat from $t$ to 90, and the offset $(90-t)/90$ **ignores stoppage time**.

### 8. The bias diagnostic

For each tick, residual $= (\text{actual remaining goals}) - \mu$. If the flat-rate assumption is fine the
residual is ≈ 0 at every minute. In-sample (so overall ≈ 0 by construction), binned by match-minute:

| minute bin | residual | ±SE |
|---|---|---|
| 00–60 | ≈ +0.03 | — |
| 60–75 | −0.006 | 0.040 |
| **75–88** | **+0.108** | **0.032 (3.4σ)** |

The model **under-predicts late-game goals by ~37 %** ($0.108/0.295$), exactly where the flat rate (a)
applies a constant rate to a run-in whose true rate is rising and (b) ignores stoppage. This biases live
**Over** prices down — the miscalibration we had twice dismissed as noise.

### 9. The Non-Homogeneous Poisson Process (NHPP)

Model each side's goals as a point process with a **time-varying intensity** $\lambda(t)$. For a realised
path with goals at times $\{t_i\}_{i=1}^k$ on $[T_0,T_1]$, the NHPP log-likelihood is

$$ \boxed{\;\ln \mathcal{L} = \sum_{i=1}^{k} \ln \lambda(t_i) \;-\; \int_{T_0}^{T_1} \lambda(t)\,dt\;} $$

— **reward** for high intensity at goal times, **penalty** (survival) for intensity over the empty time.
Intensity (per side $s$, anchored to the pre-game rate $\lambda^{\text{pg}}_s$):

$$ \lambda_s(t) = \lambda^{\text{pg}}_s \, \exp\!\Big(\alpha + \beta\, z(t) + \gamma_{\text{tr}}\,\mathbb{1}[\text{trailing}] + \gamma_{\text{ld}}\,\mathbb{1}[\text{leading}]\Big),
\quad z(t) = \tfrac{t-45}{45}. $$

$\beta$ is the **time drift** the flat-rate model lacked. Expected remaining goals (what the hedge needs):

$$ \Lambda_s(t\to T) = \int_t^T \lambda_s(s)\,ds = \lambda^{\text{pg}}_s\,e^{\alpha+\gamma\cdot\text{state}}\,\frac{45}{\beta}\Big(e^{\beta z(T)} - e^{\beta z(t)}\Big), $$

integrated to $T = 96$ to **include stoppage**. State changes only at goals, so the integral splits into
piecewise segments (state constant within each); each goal's event term uses the **pre-goal** state.

**MLE result** (Optim, 255 matches, 631 goals): $\beta = 0.077 > 0$ ⇒ scoring rate **rises ~16 %** from
kick-off to minute 90 (rate ratio $e^{2\beta} = 1.165$). Re-running the residual test:

| 75–88' bin | old (flat) | **NHPP** |
|---|---|---|
| residual | +0.108 (3.4σ) | **+0.005 (gone)** |

**The bug is fixed.** Two side-findings:
- Adding game state gave $\gamma_{\text{tr}}, \gamma_{\text{ld}} = +0.11, -0.11$ — **much smaller** than the
  count model's $+0.33,-0.25$. *The count model's strong state effect was partly a flat-rate artifact*
  (its "trailing" soaked up game-flow over the whole remaining window).
- A **half-time term didn't help** (mid-game miss unchanged). After one global rescale (×0.949), *all*
  bins flatten ⇒ the residual mid-game miss is a **~5 % pre-game-λ level over-prediction**, orthogonal to
  the in-play time shape.

### 10. The discretised NHPP = Poisson regression on time-bins (for Turing)

Split each match into $\Delta t$-minute slices. For each (match × side × slice):

$$ y = \#\{\text{goals in slice}\}, \quad \text{offset} = \log \Delta t, \quad
y \sim \text{Poisson}\big(\lambda_s(t)\,\Delta t\big). $$

Summing the Poisson log-likelihood over slices equals $\sum_i \ln\lambda(t_i) - \int \lambda\,dt$ as
$\Delta t \to 0$ — the **exact NHPP likelihood**, but AD-safe (pure Poisson, broadcastable) and with
per-slice game state handled automatically. This is the bridge that lets us put the NHPP in Turing.jl.

### 11. The hierarchical Turing NHPP

$$ \log\lambda = \alpha + \log\lambda^{\text{pg}}_s + \beta\,z(t) + \gamma_{\text{tr}}\mathbb{1}_{\text{tr}} + \gamma_{\text{ld}}\mathbb{1}_{\text{ld}}
\; + \; \underbrace{\delta^{\text{time}}_{b(t)}}_{\text{shape}} \; + \; \underbrace{\delta^{\text{team}}_{j}}_{\text{team}} \; + \; \underbrace{\delta^{\text{state}}_{g}}_{\text{state}} \; + \; \log\Delta t, $$

each hierarchical block **non-centred**: $\delta = z\,\sigma$, $z \sim \mathcal{N}(0,1)^n$,
$\sigma \sim \text{Half-}\mathcal{N}$. (`global + delta[index]` per the design.) Fitted with NUTS /
ReverseDiff(compile), **$\hat R = 1.009$**, 2,400 draws.

**What the time hierarchy learned** ($\delta^{\text{time}}_b$ — a *non-monotonic* curve no single $\beta$
or half-time term can make):

| period | 0–5' | 20–25' | 40–45' | **45–50'** | 65–70' | **90–95'** |
|---|---|---|---|---|---|---|
| $\delta^{\text{time}}$ | −0.08 | −0.11 | +0.09 | **+0.30** | −0.03 | **+0.24** |

A **post-half-time spike** and a **late surge** — well-known football phenomenology, recovered from the
data. Residual test: 75–88' bin = **−0.001** (bias gone); after the ×0.947 rescale all bins are within 1σ.

*Honest notes:* $\beta$'s 90 % CI is **[−0.03, 0.31]** (includes 0) — *not* a contradiction: the time
*hierarchy* now carries the shape, so the drift is split between $\beta$ and $\delta^{\text{time}}$. The
game-state hierarchy $\delta^{\text{state}}$ is small/noisy — the real state signal stays in the linear
$\gamma_{\text{tr}}/\gamma_{\text{ld}}$.

### 12. Pricing a bet from the intensities

Remaining-goals score matrix from independent Poissons, $P_{ij} = \text{Pois}(i;\Lambda_h)\,\text{Pois}(j;\Lambda_a)$,
mapped onto the **current** scoreline $(g_h, g_a)$ (this is Proposition 18 / Table 3.1 of the financial-maths
thesis — see `00_IN_PLAY_RESEARCH_LOG §11`):

$$ P_{\text{model}}(\text{Over } L) = \!\!\sum_{i,j:\,g_h+g_a+i+j > L}\!\! P_{ij}, \qquad
P_{\text{model}}(\text{home win}) = \!\!\sum_{i,j:\,g_h+i > g_a+j}\!\! P_{ij}, \;\dots $$

This $P_{\text{model}}$ feeds the edge $e_t$ of §4. We work under the **physical measure** (forecast and
trade the disagreement), *not* the thesis's risk-neutral replication (which assumes the market is right).

### 13. Wiring the NHPP into the hedge — does better calibration pay?

Replace the flat-rate $\mu$ in the exit pricing with the NHPP $\Lambda_s(t)$; re-run the OOS exit test:

| | mean uplift/fold | ±SE | t-stat | folds positive |
|---|---|---|---|---|
| count model (cap 0.25) | +0.306 | 0.059 | 5.21 | 73 % |
| **NHPP** (cap 0.25) | +0.264 | 0.047 | **5.63** | **83 %** |

**Nuanced, honest result:** the NHPP does *not* raise the headline return (mean slightly lower) — it makes
the exit signal **more reliable** (higher $t$, 83 % vs 73 % folds positive, tighter SE). The exit is a
binary threshold and the ~5 % pre-game hot-ness biases both models equally, so fixing the in-play *shape*
polishes the *timing* rather than rewriting the economics.

---

## Part III — Takeaways & open levers

**What is solid:**
1. **Portfolio-cap Kelly is the dominant growth lever** (bankruptcy → ×20+). Size the *joint* book
   (Σ simultaneous stakes ≤ ~0.2–0.3), not each bet.
2. **Curate by growth contribution, not ROI** (3.08 → 5.18).
3. **Model-driven full exit at $e_t \le -0.05$ improves growth, validated OOS** ($t=5.21$, no overfit at a
   pre-committed threshold). Hedge to *reduce*, never to add (+EV adds still kill growth).
4. **A real model bug was found and fixed** — the flat-rate late-game under-prediction (3.4σ → 0) — by the
   NHPP, which also *learned* the post-half-time spike + late surge, with honest uncertainty.

**Open levers (in priority order):**
1. **Pre-game ~5 % hot-ness + `min_edge = 0`.** The pre-game λ over-predicts goals ~5 % on this subset;
   combined with no edge filter it inflates Over edges and feeds the over-betting / bad-market problem.
   This is *orthogonal* to all the in-play work and is likely the **next real money lever**
   (try `min_edge ≈ 0.03`, recalibrate the pre-game level).
2. **Second-league transfer test** (ScottishLower has Betfair in-play) — confirm $\tau=-0.05$ and the NHPP
   transfer across competitions; everything here is one league / 258 matches.
3. **NHPP refinements** — team hierarchy, full-posterior (not point-mean) pricing, risk-neutral
   Kelly-target rebalance for the add side (fund adds by trimming elsewhere; the only way adding might pay).

**Caveats throughout:** single league, 258 matches; Betfair LTP ≠ tradeable lay (real lay pays the
spread); signals are causal (score/time known at the tick; execution lagged); in-play state depends on the
wall-clock→match-minute clock map (validated loosely, not unit-tested on goal times).

---

### File map
- `basic_hedging/r01_portfolio_kelly_hedge.jl` — portfolio-cap sizing + partial-hedge scans (§2–4).
- `basic_hedging/r02_model_driven_hedge.jl` — model-driven exit / add / distributional sizing (§4–6).
- `basic_hedging/portfolio_kelly_hedge_report.md` — the position-management write-up.
- `match_inplay_explore/l08_nhpp_turing.jl` — the hierarchical NHPP (dataset builder, model, prediction).
- `match_inplay_explore/00_IN_PLAY_RESEARCH_LOG.md` — full prior log incl. §11 thesis comparison.
