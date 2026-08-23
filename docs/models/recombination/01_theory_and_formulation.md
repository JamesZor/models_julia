# 1. Theory & Goal Decomposition

**Module:** `BayesianFootball.Models.PreGame`  
**Related Models:** `DynamicRecombinedGoalsModel`, `DynamicPxGRecombModel`

---

## 🎯 The Motivation for Decomposition

In football, traditional models fit bivariate or double Poisson/Negative Binomial distributions directly to **gross match scores** (e.g., 2–1, 1–0):
$$Y_{\text{gross}, h} \sim \text{Poisson}(\mu_h), \quad Y_{\text{gross}, a} \sim \text{Poisson}(\mu_a)$$

In lower tiers (such as the Scottish Championship, League 1, and League 2), gross goals are contaminated by severe non-systemic noise:
1. **Penalties:** Awarded in $\sim 20\%$ of matches, converting at $\sim 76.8\%$. Penalty awards are dominated by referee whistle sensitivity and isolated box incidents rather than sustained tactical chance creation.
2. **Own Goals:** Occur in $\sim 2.7\%$ of matches, representing pure accidental variance.
3. **Low Sample Noise:** A team might win 1–0 via a dubious penalty while generating zero open-play threat, deceiving standard gross-goals models.

---

## 📐 The Three-Way Additive Decomposition

We decompose total realized goals $Y_{\text{total}}$ into three independent generative processes:
$$Y_{\text{total}, h} = Y_{\text{open}, h} + Y_{\text{pen}, h} + Y_{\text{og}, h}$$

### 1. Open-Play Goals ($Y_{\text{open}}$)
Open-play goals reflect true tactical dominance:
$$Y_{\text{open}, h} \sim \text{Poisson}(\mu_{\text{goal\_open}, h})$$
$$\log \mu_{\text{open}, h} = \mu_{\text{base}} + \delta_{\text{month}}[t_m] + \delta_{\text{league}}[l_m] + \gamma_{\text{home}} + \alpha_h - \beta_a + w_{\text{wealth}} \Delta W_m$$
$$\log \mu_{\text{open}, a} = \mu_{\text{base}} + \delta_{\text{month}}[t_m] + \delta_{\text{league}}[l_m] + \alpha_a - \beta_h - w_{\text{wealth}} \Delta W_m$$

### 2. Penalty Awards ($N_{\text{pen}}$) and Scored Penalties ($Y_{\text{pen}}$)
Penalties are modeled as a hierarchical Poisson process of awards followed by a Bernoulli/Binomial conversion probability $q_{\text{pen}} \approx 0.768$:
$$N_{\text{pen}, h} \sim \text{Poisson}(\lambda_{\text{pen}, h})$$
$$\log \lambda_{\text{pen}, h} = \mu_{\text{pen}} + \gamma_{\text{ha, pen}} + \theta_{\text{ref}}[r_m]$$
$$\theta_{\text{ref}} \sim \mathcal{N}(0, \sigma_{\text{ref}}), \quad \sigma_{\text{ref}} \sim \text{TruncatedNormal}(0.10, 0.05, a=0.01)$$

The expected penalty goal rate is:
$$\mathbb{E}[Y_{\text{pen}, h}] = q_{\text{pen}} \cdot \lambda_{\text{pen}, h}$$

### 3. Own Goals ($Y_{\text{og}}$)
Modeled as an uninformative Poisson background rate:
$$Y_{\text{og}} \sim \text{Poisson}(\lambda_{\text{og}}), \quad \lambda_{\text{og}} \approx 0.0276 \text{ goals/match}$$

---

## 🔄 Recombination

Because the sum of independent Poisson variables is itself Poisson:
$$\mu_{\text{total}, h} = \mu_{\text{goal\_open}, h} + q_{\text{pen}} \cdot \lambda_{\text{pen}, h} + \lambda_{\text{og}}$$
$$\mu_{\text{total}, a} = \mu_{\text{goal\_open}, a} + q_{\text{pen}} \cdot \lambda_{\text{pen}, a} + \lambda_{\text{og}}$$

This cleanly isolates referee biases and accidental deflections from underlying team ratings ($\alpha, \beta$).
