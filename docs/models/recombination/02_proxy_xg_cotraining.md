# 2. Multi-Task Proxy xG (pxG) Co-Training

**Module:** `BayesianFootball.Models.PreGame`  
**Related Models:** `DynamicPxGRecombModel`, `GammaPxGObservationConfig`

---

## 🎯 Motivation

While open-play goals eliminate penalty and own-goal distortion, football remains a low-scoring sport (mean $\sim 1.35$ open-play goals per team per match). Fitting exclusively to realized goals suffers from **finishing variance**.

**Proxy xG ($\text{pxG}$)** aggregates open-play continuous shot quality (excluding penalties). By co-training the latent open-play intensity $\mu_{\text{open}}$ simultaneously against:
1. **Continuous Proxy xG ($\text{pxG}$)**, and
2. **Discrete Realized Open-Play Goals ($Y_{\text{open}}$)**,

the model learns team attacking and defensive strengths with vastly tighter posterior credible intervals.

---

## 📐 Mathematical Formulation

### 1. Continuous Observation Likelihood (Gamma)
Proxy xG is strictly positive and right-skewed. We model $\text{pxG}$ using a Gamma distribution parameterized by shape $\nu_{\text{xg}}$ and scale $\theta = \frac{\mu_{\text{open}}}{\nu_{\text{xg}}}$:
$$\text{pxG}_{\text{open}, h} \sim \text{Gamma}\left(\nu_{\text{xg}},\, \frac{\mu_{\text{open}, h}}{\nu_{\text{xg}}}\right)$$
$$\text{pxG}_{\text{open}, a} \sim \text{Gamma}\left(\nu_{\text{xg}},\, \frac{\mu_{\text{open}, a}}{\nu_{\text{xg}}}\right)$$

Here, $\mathbb{E}[\text{pxG}] = \nu_{\text{xg}} \cdot \frac{\mu_{\text{open}}}{\nu_{\text{xg}}} = \mu_{\text{open}}$, matching the tactical open-play intensity.

Prior on precision:
$$\nu_{\text{xg}} \sim \text{TruncatedNormal}(3.5, 0.5, a=0.5)$$

### 2. Team Finishing Efficiency Factor ($\kappa_i$)
Teams differ in their conversion efficiency from xG into realized goals. We introduce a team-specific finishing multiplier $\kappa_i$:
$$\mu_{\text{goal\_open}, h} = \kappa_h \cdot \mu_{\text{open}, h}$$
$$\mu_{\text{goal\_open}, a} = \kappa_a \cdot \mu_{\text{open}, a}$$

Prior on finishing factor:
$$\log \kappa_{\text{raw}, i} \sim \mathcal{N}(0, 0.10)$$
$$\log \kappa_i = \log \kappa_{\text{raw}, i} - \frac{1}{N}\sum_{j=1}^N \log \kappa_{\text{raw}, j} \quad (\text{Sum-to-Zero Constrained})$$
$$\kappa_i = \exp\left(\text{clamp}(\log \kappa_i, -0.50, 0.50)\right)$$

### 3. Discrete Goals Observation
$$Y_{\text{open}, h} \sim \text{Poisson}(\mu_{\text{goal\_open}, h})$$
$$Y_{\text{open}, a} \sim \text{Poisson}(\mu_{\text{goal\_open}, a})$$

---

## ⚡ Zero-Allocation Binary Masking in Turing

When historical matches lack shot data, the feature extractor passes a binary mask `mask_pxg_h ∈ {0.0, 1.0}`:
```julia
ll_pxg_h = logpdf.(Gamma.(ν_xg, scale_pxg_h), pxg_open_h)
Turing.@addlogprob! sum(ll_pxg_h .* mask_pxg_h .* match_weights)
```
This guarantees ReverseDiff tape compilation never triggers branch re-evaluation.
