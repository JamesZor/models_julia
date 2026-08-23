# Scottish Lower Football: Corner & Set-Piece Decomposition EDA & Statistical Findings

**Dataset:** Scottish Pyramid (Premiership 54, Championship 55, League One 56, League Two 57)  
**Sample Sizes:**  
- **Full History (2020–2027):** 4,228 Matches across 4 tiers; 1,988 Matches in Scottish Lower (56 & 57)  
- **Trading Walk-Forward Window (2024/25 $\to$ 2025/26):** 710 Scottish Lower Matches, 22 Teams, 6,521 Corners, 259 Corner Goals  
**Execution Node:** `mcmc-beast` (32 cores)

---

## 📊 1. Macro 4-Way Goal Decomposition Breakdown

Across 4,228 matches in Scottish football, total match goals decompose into four orthogonal processes:

| Goal Component | Total Goals | % of All Match Goals | Mean Goals / Match | Modeling Treatment |
| :--- | :---: | :---: | :---: | :--- |
| **1. Open-Play Tactical Goals ($Y_{\text{open}}$)** | 9,419 | **82.47%** | 2.228 | Continuous Proxy xG + Wealth ($\Delta W$) co-training |
| **2. Penalty Whistle Goals ($Y_{\text{pen}}$)** | 924 | **8.09%** | 0.219 | Hierarchical Referee Tendency ($\gamma_{\text{ref}}$) |
| **3. Corner Set-Piece Goals ($Y_{\text{corner}}$)** | 813 | **7.12%** | 0.192 | **Hierarchical Negative Binomial + $z$-Score Conversion** |
| **4. Accidental Own Goals ($Y_{\text{og}}$)** | 265 | **2.32%** | 0.063 | Constant Poisson Background ($\lambda_{\text{og}} = 0.063$) |
| **Total Gross Match Goals ($Y_{\text{total}}$)** | **11,421** | **100.00%** | **2.701** | **Exact Discrete Poisson Convolution** |

> [!IMPORTANT]
> **Key Finding 1:** Corner goals account for **7.12% of all match goals** (comparable to penalties at 8.09%). Isolating corners cleans the open-play target down to **82.47% pure tactical goals**, removing high-variance aerial scrambles from tactical attacking/defensive ratings.

---

## 📈 2. Corner Generation Distribution & Overdispersion Analysis

| Tournament Tier | Matches | Mean Corners/Match | Home Corners | Away Corners | Home Adv Ratio | Dispersion Index ($\frac{\text{Var}}{\mu}$) | $p$-value (vs Poisson Null) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Scottish Premiership (54)** | 1,200 | 10.46 | 5.60 | 4.86 | **1.15x** (+0.74) | **1.82** | $p < 10^{-15}$ (Overdispersed) |
| **Scottish Championship (55)** | 1,040 | 9.32 | 5.05 | 4.28 | **1.18x** (+0.77) | **2.07** | $p < 10^{-15}$ (Overdispersed) |
| **Scottish League One (56)** | 994 | 9.74 | 5.11 | 4.62 | **1.11x** (+0.49) | **1.67** | $p < 10^{-15}$ (Overdispersed) |
| **Scottish League Two (57)** | 994 | 9.33 | 4.94 | 4.39 | **1.13x** (+0.56) | **1.51** | $p < 10^{-15}$ (Overdispersed) |

### Key Mathematical Takeaways:
1. **Negative Binomial Requirement:** Across all four divisions, the index of dispersion ($\frac{\text{Var}}{\mu}$) is significantly greater than $1.0$ ($1.51\text{--}2.07, p < 10^{-15}$). Corner count generation **must be modeled with a Negative Binomial likelihood** ($\text{NegBin}(\lambda, \phi)$) rather than standard Poisson.
2. **Consistent Home Advantage:** Home teams consistently generate **1.11x to 1.18x more corners** than away teams ($+0.49$ to $+0.77$ corners, $t = 3.45\text{--}5.78, p < 0.0001$).
3. **Corner Total vs Goal Total Independence:** $\text{Cor}(\text{Total Corners}, \text{Total Goals}) = +0.0096$. Match corner volume does *not* imply high gross goal volume because open-play finishing and corner award volume are orthogonal tactical processes.

---

## 🔬 3. Frequentist MLE Statistical Significance Diagnostics

We fitted frequentist Maximum Likelihood Estimation (MLE) models with numerical Hessians and Likelihood Ratio Tests (LRT) on the target Scottish Lower 2024/25 $\to$ 2025/26 trading benchmark (710 matches, 22 teams):

### A. Corner Count Generation Model ($\text{NegBin}(\lambda_{h,m}, \phi)$)
$$\log \lambda_{h, m} = \mu_{\text{corner}} + \gamma_{\text{ha}} + \alpha_{\text{att}, h} - \beta_{\text{def}, a}$$
$$\log \lambda_{a, m} = \mu_{\text{corner}} + \alpha_{\text{att}, a} - \beta_{\text{def}, h}$$

- **Base Log Corner Rate ($\mu$):** $+1.4464 \pm 0.0238$ ($4.25$ corners/team/match)
- **Home Advantage ($\gamma_{\text{ha}}$):** $+0.1292 \pm 0.0317$ ($z = 4.08, \mathbf{p = 4.54 \times 10^{-5}}$)
- **Overdispersion Parameter ($\phi$):** $7.45$ (strongly rejecting Poisson $\phi \to \infty$)
- **Likelihood Ratio Test (LRT):** $\chi^2(\text{df}=42) = 119.96, \mathbf{p = 1.98 \times 10^{-9}}$ (AIC: 6803.32 vs Null 6839.28)
- **Verdict:** Team attacking creation ($\alpha$) and defensive concession ($\beta$) are **statistically massive, genuine signals ($p < 10^{-8}$)**.

#### Top & Bottom Corner Creators in 24/25 $\to$ 25/26:
- `Arbroath`: $\alpha = +0.246$ ($\times 1.28$ corner pressure, $5.44$ corners/match)
- `Inverness CT`: $\alpha = +0.233$ ($\times 1.26$ corner pressure)
- `Dumbarton`: $\alpha = +0.140$ ($\times 1.15$ corner pressure)
- `Bonnyrigg Rose`: $\alpha = -0.194$ ($\times 0.82$ corner pressure)
- `Edinburgh City`: $\alpha = -0.345$ ($\times 0.71$ corner pressure, $3.01$ corners/match)

---

### B. Corner Goal Conversion Model ($\text{Binomial}(C_{h,m}, q_{h,a})$)
$$\text{logit}(q_{h, a}) = \text{logit}(\bar{q}) + \eta_{\text{att}, h} - \zeta_{\text{def}, a}$$

- **Global Baseline Conversion ($\bar{q}$):** **$3.79\%$** ($\approx 1$ goal per 26 corners in Scottish Lower)
- **2-Season LRT (710 matches, 259 goals):** $\chi^2(\text{df}=42) = 52.92, p = 0.1205$ ($p > 0.05$)
- **Full History LRT (1,988 matches, 813 goals):** $\chi^2(\text{df}=60) = 216.24, \mathbf{p < 10^{-15}}$
- **Key Diagnostic Insight:** In short walk-forward windows ($\sim 1\text{--}2$ seasons), raw corner goals per team ($\sim 10\text{--}15$) have high binomial sampling variance. Unregularized fixed-effects MLE suffers from noise.
- **Architectural Requirement:** We **must use Hierarchical Bayesian Shrinkage with $z$-score parameterization** ($\tilde{\eta} \sim \mathcal{N}(0, 1)$) anchored at $\text{logit}(\bar{q}) = -3.23$.

---

## 🔄 4. Year-over-Year (YoY) Autocorrelation & Signal Persistence

Across **146 consecutive team-season pairs**:

| Metric | YoY Correlation ($r_{t, t+1}$) | Signal Quality | Modeling Architecture |
| :--- | :---: | :---: | :--- |
| **Corner Generation Rate** | **$+0.6718$** | **Extremely High** | Dynamic GRW team attacking latent $\alpha_{\text{corner}, i, t}$ |
| **Corner Concession Rate** | **$+0.6130$** | **Extremely High** | Dynamic GRW team defensive latent $\beta_{\text{corner}, j, t}$ |
| **Total Corner Goals / Game** | **$+0.6763$** | **Very High** | Strong predictive signal for 4-way goal recombination |
| **Corner Goal Conversion ($q_{\text{corner}}$)** | **$+0.6767$** | **High** | Hierarchical team finishing random effect $\eta_{\text{corner}, i}$ |
| *Benchmark: Gross Goals Scored* | *$+0.2200$* | *Low / Noisy* | *Standard goal models suffer from noise* |

> [!TIP]
> **Key Finding 2 (The Signal Discovery):** Corner generation ($r = +0.6718$) and conceding ($r = +0.6130$) have **nearly 3x higher persistence than gross goals ($r \approx 0.22$)**. Corners provide an exceptionally stable measure of field tilt and territorial dominance.
