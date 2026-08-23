# Scottish Lower Leagues: Stage-A Goals Overdispersion EDA Notes
**Tournaments Analyzed:** Scottish League One (`#56`) and Scottish League Two (`#57`)  
**Sample Size:** $1,990$ Matches ($3,980$ Team-Match Observations, Seasons `20/21` through `25/26`)  
**Methodology:** BayesianFootball Stage-A EDA Playbook (`STAGE_A_EDA_PLAYBOOK.md`)

---

## 1. Summary Moments & Variance-to-Mean Ratio (VMR)

Under the standard Poisson distribution, variance is strictly constrained to equal the mean ($\sigma^2 / \mu = 1.00$). In Scottish Lower League matches, goals exhibit systemic **overdispersion** ($\sigma^2 > \mu$):

| Market Pillar | Observations ($N$) | Mean ($\mu$) | Variance ($\sigma^2$) | Dispersion Index ($\text{VMR} = \sigma^2/\mu$) | Poisson Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Home Goals** | $1,990$ | $1.4633$ | $1.5580$ | **`1.0647`** | **Overdispersed ($+6.5\%$ excess variance)** |
| **Away Goals** | $1,990$ | $1.2879$ | $1.4641$ | **`1.1367`** | **Strongly Overdispersed ($+13.7\%$ excess variance)** |
| **Pooled (H+A)** | $3,980$ | $1.3756$ | $1.5195$ | **`1.1046`** | **Overdispersed ($+10.5\%$ excess variance)** |
| **Total Goals** | $1,990$ | $2.7513$ | $2.6133$ | $0.9499$ | Slight negative cross-team covariance |

### Breakdown by Division:
- **Scottish League One (#56, N = 995 matches):**
  - Home Goals: $\mu = 1.486, \sigma^2 = 1.634 \implies \mathbf{\text{VMR} = 1.100}$
  - Away Goals: $\mu = 1.324, \sigma^2 = 1.497 \implies \mathbf{\text{VMR} = 1.131}$
  - Total Goals: $\mu = 2.810, \sigma^2 = 2.651 \implies \text{VMR} = 0.943$
- **Scottish League Two (#57, N = 995 matches):**
  - Home Goals: $\mu = 1.440, \sigma^2 = 1.482 \implies \mathbf{\text{VMR} = 1.029}$
  - Away Goals: $\mu = 1.252, \sigma^2 = 1.430 \implies \mathbf{\text{VMR} = 1.142}$
  - Total Goals: $\mu = 2.692, \sigma^2 = 2.571 \implies \text{VMR} = 0.955$

---

## 2. Complete Count Distribution Fit Ladder (MLE)

We fit standard Poisson vs the **Robust Negative Binomial (NB2: $\text{Var} = \mu + \mu^2/r$)** via Maximum Likelihood Estimation across all empirical goal distributions:

| Goal Distribution | Model | Fitted Parameters | Log-Likelihood | AIC | $\Delta\text{AIC}$ | BIC | Chi² GOF $p$-value | Empirical Verdict |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Home Goals** | Poisson | $\lambda = 1.463$ | $-3,066.97$ | $6,135.93$ | $+1.74$ | $6,141.53$ | $0.0009$ (Rejected) | Poisson fails |
| | **Robust NegBin (NB2)** | **$r = 23.66, \mu = 1.463$** | **$-3,065.10$** | **`6,134.19`** | **`0.00`** | $6,145.39$ | **`0.0034`** | **NB2 WINS** 🏆 |
| **Away Goals** | Poisson | $\lambda = 1.288$ | $-2,963.10$ | $5,928.19$ | $+14.77$ | $5,933.79$ | $0.0010$ (Rejected) | Severe Poisson failure |
| | **Robust NegBin (NB2)** | **$r = 9.25, \mu = 1.288$** | **$-2,954.71$** | **`5,913.42`** | **`0.00`** | **`5,924.61`** | **`0.8035` (Ideal Fit)** | **NB2 WINS (ΔAIC = -14.8)** 🏆 |
| **Pooled (H+A)**| Poisson | $\lambda = 1.376$ | $-6,041.19$ | $12,084.40$ | $+17.38$ | $12,090.70$ | $0.0000$ (Rejected) | Poisson fails |
| | **Robust NegBin (NB2)** | **$r = 13.37, \mu = 1.376$** | **$-6,031.50$** | **`12,067.00`** | **`0.00`** | **`12,079.60`** | **`0.1889`** | **NB2 WINS (ΔAIC = -17.4)** 🏆 |

---

## 3. Formal Statistical Hypothesis Tests

### A. Dean-Lawless (1989) Lagrange Multiplier (LM) Score Test
Tests $H_0: \text{Var}(Y) = \mu$ (Poisson) vs $H_1: \text{Var}(Y) = \mu + \alpha \mu^2$ ($\alpha > 0$, NB2):
$$T = \frac{\sum_i \left[ (y_i - \hat{\mu})^2 - y_i \right]}{\sqrt{2 n \hat{\mu}^2}} \sim \mathcal{N}(0, 1)$$

- **Away Goals:** $T = 4.295 \implies p = \mathbf{8.72 \times 10^{-6}}$ $\implies$ **Decisive rejection of Poisson ($p < 0.0001$)**.
- **Home Goals:** $T = 2.024 \implies p = \mathbf{0.0215}$ $\implies$ **Rejects Poisson ($p < 0.05$)**.

### B. Cameron-Trivedi (1990) Auxiliary Regression Test
Fits the auxiliary regression $\frac{(y_i - \hat{\mu})^2 - y_i}{\hat{\mu}} = \alpha \hat{\mu} + \varepsilon_i$:
- **Away Goals:** $\hat{\alpha} = +0.1057$ ($SE = 0.0275, t = 3.85, p = 5.98 \times 10^{-5}$) $\implies$ Confirms quadratic overdispersion.
- **Home Goals:** $\hat{\alpha} = +0.0438$ ($SE = 0.0245, t = 1.79, p = 0.0369$).

### C. Bivariate Low-Score Dependence (Dixon-Coles $\rho$)
Evaluates whether bivariate low-score inflation $(0,0), (1,0), (0,1), (1,1)$ exists beyond the marginal distributions:
- $\hat{\lambda}_{\text{home}} = 1.4634, \hat{\mu}_{\text{away}} = 1.2876, \hat{\rho} = -0.0265$
- Likelihood Ratio Test vs Independent Marginals: $\text{LRT} = 0.91, p = 0.3403$.
- **Conclusion:** Independent Negative Binomial marginals are structurally sufficient (no Dixon-Coles copula/interaction required).

---

## 4. Score Frequency Calibration (Clean Sheets vs Blowouts)

Comparing empirical match outcomes against theoretical distributions demonstrates why Negative Binomial is critical for betting markets:

### Away Goals Distribution:
| Goals ($k$) | Empirical % | Poisson % | Robust NegBin % | Poisson Bias | NegBin Lift |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0 (Clean Sheet)** | **`29.90%`** | $27.58\%$ | **`29.90%`** 🎯 | Underestimates by $-2.32\%$ | **Exact Calibration (0.00% error)** |
| **1** | $34.32\%$ | $35.53\%$ | $33.88\%$ | Overestimates by $+1.21\%$ | Corrects toward empirical |
| **2** | $20.45\%$ | $22.88\%$ | $21.23\%$ | Overestimates by $+2.43\%$ | Corrects toward empirical |
| **3** | $9.75\%$ | $9.82\%$ | $9.72\%$ | Neutral | Neutral |
| **4** | $4.02\%$ | $3.16\%$ | $3.63\%$ | Underestimates blowouts | $+0.47\%$ blowout recovery |
| **5** | $1.16\%$ | $0.81\%$ | $1.17\%$ | Underestimates blowouts | $+0.36\%$ blowout recovery |
| **6+** | $0.40\%$ | $0.21\%$ | $0.46\%$ | Underestimates blowouts | $+0.24\%$ blowout recovery |

---

## 5. Architectural Conclusions & Modeling Implications

1. **Away Goals are Heavily Overdispersed ($r = 9.25$):**
   - Away performance in Scottish lower divisions is volatile—defensive collapses and away clean sheets occur much more frequently than Poisson can allow.
2. **Home Goals Exhibit Mild Overdispersion ($r = 23.66$):**
   - Home performance has higher baseline stability, confirming the utility of a **Home/Away asymmetric dispersion specification** (`disp.log_r` and `disp.δ_r_home`).
3. **Pillar Independence & Mean Preservation:**
   - Because $\mathbb{E}[\text{NegBin}(r, \mu)] = \mu \equiv \mathbb{E}[\text{Poisson}(\mu)]$, swapping the goals likelihood pillar to `RobustNegativeBinomial` preserves the exact mathematical calibration of Proxy xG Gamma co-training and RAPM player strengths, while allowing the model to accurately price 0-goal clean sheets and high-total Over/Under lines.
