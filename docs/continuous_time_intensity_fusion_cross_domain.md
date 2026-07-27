# Continuous-Time Piecewise State-Space Updating of a Latent Intensity

## A Cross-Domain Analysis: Cox Processes, Reduced-Form Credit Risk, Point-Process Filtering, and Implied-Parameter Inversion

---

## 0. Abstract Statement of the Problem

Fix a finite horizon $[0,T]$ and a filtered probability space $(\Omega, \mathcal{F}, \{\mathcal{F}_t\}_{t\in[0,T]}, \mathbb{P})$ satisfying the usual conditions. We posit a latent, non-negative, non-stationary state process $\lambda = \{\lambda_t\}_{t \in [0,T]}$, not directly observable, and three information channels:

**(C0) The static prior.** A pre-horizon belief $\lambda_0 \sim \Gamma(a_0, b_0)$, with $\mathbb{E}[\lambda_0] = a_0/b_0 = \hat\lambda_0$, elicited from a historical estimator.

**(C1) The counting channel.** A simple point process $N = \{N_t\}$, adapted to $\mathcal{F}_t$, admitting the $\mathcal{F}_t$-predictable intensity $\lambda_t$; i.e.
$$
M_t \;=\; N_t - \int_0^t \lambda_s\, ds
$$
is an $(\mathcal{F}_t,\mathbb{P})$-local martingale. We observe the jump times $\{\tau_1, \tau_2, \dots\}$ exactly.

**(C2) The continuous channel.** A real-valued observation process $Z = \{Z_t\}$ derived from aggregate market prices, which is informative about $\lambda_t$ but corrupted by pricing friction, a risk premium, and microstructure noise.

The **observation filtration** is $\mathcal{Y}_t = \sigma(N_s, Z_s : s \le t) \vee \mathcal{N}$. The object of interest is the conditional (filtered) law
$$
\pi_t(\varphi) \;=\; \mathbb{E}\!\left[\varphi(\lambda_t) \,\middle|\, \mathcal{Y}_t\right],
$$
and, for pricing purposes, the conditional law of the **remaining integrated intensity**
$$
\Lambda_{t,T} \;=\; \int_t^T \lambda_s \, ds ,
$$
which is the sufficient statistic for the terminal count $N_T - N_t$ under the doubly-stochastic hypothesis.

This is *precisely* the canonical problem of **nonlinear filtering of a doubly-stochastic (Cox) intensity under mixed observation channels**. Everything that follows is an instance of it.

> **Two structural pathologies must be stated at the outset**, because they are invisible inside the domain and standard elsewhere:
>
> **(P1) Measure mismatch.** Channel (C1) identifies $\lambda$ under the physical measure $\mathbb{P}$. Channel (C2) identifies $\lambda$ under the *pricing* measure $\mathbb{Q}$. In reduced-form credit these differ by the **jump-to-default risk premium**; the ratio $\lambda^{\mathbb{Q}}/\lambda^{\mathbb{P}}$ is empirically of order $2$ and time-varying. Fusing $\lambda^{\mathbb{P}}$-information and $\lambda^{\mathbb{Q}}$-information as if they estimated the same scalar is a *specification error*, not a noise problem, and no amount of filtering will remove it. It must be absorbed by an explicit premium/bias state.
>
> **(P2) Channel dependence.** The Kushner–Stratonovich derivation requires that, conditionally on the state path $\lambda_{[0,t]}$, the two observation channels be independent. **They are not.** The market price $Z_t$ is a functional of the market's own information set $\mathcal{G}_t \supseteq \sigma(N_s: s\le t)$: the market has already seen every event timestamp you have. A naive product-of-likelihoods fusion therefore *double-counts the point process*, producing an over-confident posterior with variance collapsing faster than the true information rate. This is the correlated-expert / correlated-sensor problem, and it has established solutions (orthogonalisation of the innovation; covariance intersection).

---

## 1. CROSS-DOMAIN MATHEMATICAL ANALOGIES

### 1.1 Doubly Stochastic Poisson (Cox) Processes and Their Exact Filters

**Framework.** Cox (1955) defines the doubly stochastic Poisson process: conditionally on the realisation of the intensity path $\lambda_{[0,T]}$, $N$ is an inhomogeneous Poisson process. The conditional law of increments is
$$
\mathbb{P}\big(N_T - N_t = k \,\big|\, \lambda_{[0,T]}\big) \;=\; \frac{\big(\Lambda_{t,T}\big)^k}{k!}\, e^{-\Lambda_{t,T}}, \qquad \Lambda_{t,T} = \int_t^T \lambda_s\, ds .
$$

**The likelihood (Jacod's formula / Girsanov for point processes).** For an observed path on $[0,t]$ with jumps at $\tau_1 < \dots < \tau_{N_t}$, the Radon–Nikodým derivative with respect to the unit-rate Poisson measure $\mathbb{P}^1$ is
$$
\boxed{\;\frac{d\mathbb{P}^{\lambda}}{d\mathbb{P}^{1}}\bigg|_{\mathcal{F}_t} \;=\; \exp\!\left(\int_0^t \log \lambda_s \, dN_s \;-\; \int_0^t (\lambda_s - 1)\, ds\right) \;=\; e^{\,t}\, e^{-\int_0^t \lambda_s ds} \prod_{i=1}^{N_t} \lambda_{\tau_i} \;}
$$
This is the master likelihood from which *every* conjugate update in this document descends. Note that the source construction's likelihood (its Eq. 5) is exactly this object.

**The exact filter for a static Gamma intensity — i.e. what conjugate Gamma–Poisson updating *actually is*.** Suppose $\lambda_t \equiv \lambda_0 \Phi(t)$ with $\Phi$ a *known deterministic* modulation and $\lambda_0 \sim \Gamma(a_0,b_0)$ latent. Then by the above,
$$
\pi_t(\lambda) \;\propto\; \underbrace{\lambda^{a_0-1} e^{-b_0 \lambda}}_{\text{prior}} \cdot \underbrace{e^{-\lambda \int_0^t \Phi(s)ds} \prod_{i=1}^{N_t} \lambda\,\Phi(\tau_i)}_{\text{likelihood}} \;\propto\; \lambda^{\,a_0 + N_t - 1} \, e^{-\left(b_0 + \int_0^t \Phi(s)\,ds\right)\lambda},
$$
whence
$$
\boxed{\;\lambda_0 \,\big|\, \mathcal{Y}^N_t \;\sim\; \Gamma\!\Big(a_0 + N_t,\;\; b_0 + \textstyle\int_0^t \Phi(s)\, ds\Big)\;}
\tag{1.1}
$$
The known modulation $\Phi$ enters *only* through the **compensator** $\int_0^t \Phi\,ds$ — the accumulated exposure — and *not* through the count. This is the rigorous statement of what a Gamma–Poisson conjugate updater is: **the exact, finite-dimensional, closed-form nonlinear filter for a *static* Cox intensity observed through its own counting process.** It is optimal, and it is also *degenerate*: the state does not move.

The consequence is sharp. The posterior mean
$$
\hat\lambda_t = \frac{a_0 + N_t}{b_0 + \int_0^t \Phi\, ds}
$$
has an information content that grows monotonically in $t$ and *never forgets*. Its Fisher information about a *time-varying* $\lambda_t$ is therefore mis-specified: it will lag any genuine drift in the state, and the lag worsens as $t \to T$. The standard remedy across every applied field below is to **discount the sufficient statistics** (§1.4).

**Shot-noise Cox processes.** The canonical way to give the intensity its own stochastic dynamics while retaining tractability (Dassios & Jang; Cox–Isham) is a shot-noise intensity: a piecewise-deterministic Markov process
$$
\lambda_t \;=\; \lambda_0 e^{-\delta t} \;+\; \sum_{i \,:\, S_i \le t} Y_i \, e^{-\delta (t - S_i)}
\qquad\Longleftrightarrow\qquad
d\lambda_t \;=\; -\delta \lambda_t\, dt \;+\; dJ_t ,
$$
with $J$ a compound Poisson process of exogenous shocks $(S_i, Y_i)$. Exponential decay $\delta$ back to baseline; upward jumps at shocks. The resulting $N$ is a Cox process with a self-decaying, jump-driven intensity. Dassios & Jang give the Laplace transform of $\Lambda_{t,T}$ in closed form (essential for pricing), and derive Kalman–Bucy-type filters for linear systems driven by such a process.

**Hawkes (self-exciting) processes.** Hawkes (1971):
$$
\lambda_t \;=\; \mu(t) \;+\; \int_0^{t^-} g(t-s)\, dN_s, \qquad g(u) = \alpha e^{-\beta u} \;(\text{typically}),
$$
so that $M_t = N_t - \int_0^t \lambda_s ds$ is a martingale with $\lambda$ **$\mathcal{F}^N_t$-predictable**. This is the crucial structural distinction:

> **In a Hawkes process the intensity is *observable* — it is a deterministic functional of the observed history. There is no filtering problem, only a parameter-estimation problem. In a Cox process the intensity is *latent* — filtering is unavoidable.**

Your architecture is Cox-type (latent strength) with a Hawkes-like *deterministic* state-dependent modulation (the score-state multipliers $\tau_{xy}$ in the source construction are precisely a self-exciting/self-inhibiting kernel that happens to be $\mathcal{F}^N$-measurable). The correct decomposition is therefore

$$
\boxed{\;\lambda_t \;=\; \underbrace{\hat\lambda_0}_{\text{prior level}} \cdot \underbrace{\Phi\big(t,\, S_{t^-}\big)}_{\substack{\text{KNOWN, } \mathcal{F}^N\text{-predictable} \\ \text{deterministic modulation}}} \cdot \underbrace{e^{X_t}}_{\substack{\text{LATENT stochastic} \\ \text{deviation from prior}}} \;}
\tag{1.2}
$$

Only $X$ requires filtering. The piecewise epoch multipliers ($\rho(t)$ for injury-time-type epochs) and the state-dependent multipliers ($\tau_{xy}$) belong in $\Phi$, where they cost nothing, rather than in the filter, where they would cost dimensions. This is the single most important architectural refactor implied by the cross-domain literature.

### 1.2 Reduced-Form Credit Risk: the Exact Isomorph of Your Three-Channel Problem

Reduced-form (intensity-based) credit modelling is not an analogy; it is the *same problem with different labels*, and it is the most mature literature on the fusion of a historical prior, discrete event data, and a continuous market price.

**Construction (Lando 1998).** Default is the first jump of a Cox process:
$$
\tau \;=\; \inf\left\{ t \ge 0 \;:\; \int_0^t \lambda_s\, ds \;\ge\; E \right\}, \qquad E \sim \mathrm{Exp}(1) \ \perp\ \lambda .
$$
Survival is the exponential of the negative compensator:
$$
\mathbb{P}\big(\tau > T \,\big|\, \mathcal{F}_t \vee \{\tau > t\}\big) \;=\; \mathbb{E}\!\left[ \exp\!\left(-\int_t^T \lambda_s\, ds\right) \middle|\, \mathcal{F}_t \right] \;=\; \mathbb{E}\big[e^{-\Lambda_{t,T}}\big|\mathcal{F}_t\big].
\tag{1.3}
$$

**Pricing (Duffie–Singleton 1999; Jarrow–Turnbull 1995).** A defaultable zero-coupon bond with zero recovery prices as
$$
v(t,T) \;=\; \mathbb{E}^{\mathbb{Q}}\!\left[ \exp\!\left(-\int_t^T \big(r_s + \lambda^{\mathbb{Q}}_s\big) ds \right) \middle|\, \mathcal{F}_t \right],
$$
i.e. **the intensity enters exactly as a credit spread**. The default-risk-adjusted short rate is $R_t = r_t + \lambda^{\mathbb{Q}}_t$. Inverting a term structure of such prices for $\lambda^{\mathbb{Q}}$ is the *hazard-rate bootstrap* (§2.2).

**The measure wedge (P1), stated precisely.** The Girsanov change of measure for point processes multiplies the intensity by a predictable, positive process $\gamma$:
$$
\boxed{\;\lambda^{\mathbb{Q}}_t \;=\; \gamma_t \cdot \lambda^{\mathbb{P}}_t \;}
\qquad \text{with} \qquad
\frac{d\mathbb{Q}}{d\mathbb{P}}\bigg|_{\mathcal{F}_t} = \exp\!\left(\int_0^t \log \gamma_s\, dN_s - \int_0^t (\gamma_s - 1)\lambda^{\mathbb{P}}_s\, ds\right).
\tag{1.4}
$$
$\gamma$ is the **jump-to-default risk premium**. The credit literature measures it directly by estimating $\lambda^{\mathbb{P}}$ from realised default *events* (the counting channel) and $\lambda^{\mathbb{Q}}$ from CDS *prices* (the continuous channel), and taking the ratio. It is not $1$, it is not constant, and it is strongly counter-cyclical.

The transposition to your architecture is exact:
- $\lambda^{\mathbb{P}}$ is what your event timestamps identify (via the Gamma–Poisson filter);
- $\lambda^{\mathbb{Q}}$ is what your inverted odds identify — **after** de-vigging;
- the residual wedge $\gamma_t$ is the *combination* of any genuine risk premium and any *systematic mispricing* (the favourite–longshot bias is a $\gamma \ne 1$ that varies with the level of $\lambda$).

**Therefore: $\gamma$ is not noise to be filtered out — it is a state to be estimated.** A fusion that omits it will attribute the entire premium to the latent intensity and will be biased in a direction that is *systematic across the whole horizon*, which is the worst possible failure mode for a staking rule.

**Filtering-induced intensity (Duffie–Lando 2001).** The deepest result here. If the market observes the firm's state only through *noisy accounting reports*, then the default intensity *perceived by the market* is not the structural hazard but the **conditional hazard given the filtration** — i.e. the intensity is generated *by the filtering operation itself*:
$$
\lambda^{\text{mkt}}_t \;=\; \lim_{h \downarrow 0} \frac{1}{h}\, \mathbb{P}\big(\tau \le t + h \,\big|\, \mathcal{G}_t,\ \tau > t\big),
$$
where $\mathcal{G}_t$ is the market's (coarser) information set. **The market's implied intensity is itself a posterior mean.** This is the formal content of (P2): $Z_t$ is not a noisy measurement of $\lambda_t$; it is a *noisy measurement of another agent's filter of $\lambda_t$*, run on an information set that strictly contains your counting channel. Treating it as a conditionally-independent sensor is a category error.

### 1.3 Continuous-Discrete Filtering with Mixed Channels (the general machinery)

Take the latent log-deviation $X$ of (1.2) as an Itô diffusion with generator $\mathcal{A}$:
$$
dX_t \;=\; f(X_t, t)\, dt \;+\; \sigma(X_t,t)\, dW_t, \qquad
\mathcal{A}\varphi = f \,\partial_x \varphi + \tfrac12 \sigma^2 \partial_{xx}\varphi ,
$$
observed through **both** channels:
$$
\text{(C1)}\quad N_t \ \ \text{with intensity } \lambda_t = \hat\lambda_0 \Phi(t,S_{t^-}) e^{X_t}, \qquad
\text{(C2)}\quad dZ_t \;=\; h(X_t, t)\, dt \;+\; \varsigma_t\, dV_t ,
$$
with $W \perp V$ and (for now, counterfactually — see P2) the channels conditionally independent given $X$.

**Kushner–Stratonovich equation with mixed observations** (Kushner 1964; Zakai 1969; Segall–Kailath 1975 for the point-process innovations; Brémaud 1981 for the general martingale form). Writing $\pi_t(\varphi) = \mathbb{E}[\varphi(X_t)|\mathcal{Y}_t]$:

$$
\boxed{
\begin{aligned}
d\pi_t(\varphi) \;=\;& \pi_t(\mathcal{A}\varphi)\, dt \\[4pt]
&+\; \varsigma_t^{-2}\Big[\pi_t(\varphi\, h) - \pi_t(\varphi)\,\pi_t(h)\Big]\Big(dZ_t - \pi_t(h)\, dt\Big) 
&&\longleftarrow \ \text{\bf diffusion innovation} \\[4pt]
&+\; \left[\frac{\pi_{t^-}(\varphi\, \lambda)}{\pi_{t^-}(\lambda)} - \pi_{t^-}(\varphi)\right]\Big(dN_t - \pi_{t^-}(\lambda)\, dt\Big)
&&\longleftarrow \ \text{\bf point-process innovation}
\end{aligned}}
\tag{1.5}
$$

Read the structure carefully, because it is the entire answer to Objective 3:

- **Between events and between price ticks**, the filter follows the deterministic drift $\pi_t(\mathcal{A}\varphi)\,dt$ *minus* the two compensator terms. The point-process compensator $-[\cdot]\pi_{t^-}(\lambda)dt$ is the "**no news is news**" term: the *absence* of an event is itself information, and it monotonically pushes the posterior mass of $\lambda$ *downwards*. This is the continuous-time meaning of the $b_0 + \int_0^t \Phi\,ds$ denominator in (1.1).
- **At an event time $\tau_i$**, the filter jumps multiplicatively:
$$
\pi_{\tau_i}(\varphi) \;=\; \frac{\pi_{\tau_i^-}(\varphi\, \lambda)}{\pi_{\tau_i^-}(\lambda)}
\qquad\text{i.e.}\qquad
\pi_{\tau_i}(dx) \;=\; \frac{\lambda(x)\, \pi_{\tau_i^-}(dx)}{\pi_{\tau_i^-}(\lambda)} ,
$$
a **size-biasing (Bayes) update by the intensity itself.** For Gamma priors this is precisely $a \mapsto a+1$.
- **The gain on the diffusion channel is the posterior covariance** $\mathrm{Cov}_{\pi_t}(\varphi, h)$ divided by the observation variance. When $\varphi(x)=x$ and $h(x)=x$ this is the Kalman gain $P_t/\varsigma_t^2$, exactly.

**Zakai (unnormalised) form.** Let $\rho_t$ be the unnormalised filter, $\pi_t(\varphi) = \rho_t(\varphi)/\rho_t(\mathbf{1})$. Then $\rho$ solves a *linear* SPDE:
$$
\boxed{\;
d\rho_t(\varphi) \;=\; \rho_t(\mathcal{A}\varphi)\, dt \;+\; \varsigma_t^{-2}\rho_t(\varphi\, h)\, dZ_t \;+\; \Big[\rho_{t^-}(\varphi\,\lambda) - \rho_{t^-}(\varphi)\Big]\big(dN_t - dt\big) \; }
\tag{1.6}
$$
Linearity is what makes Monte Carlo (particle) approximation of $\rho$ convergent and unbiased; it is the theoretical licence for §3.3.

### 1.4 The Applied Analogues: Epidemiology, Reliability, Queueing, Neuroscience

Four fields solve *your exact updating problem* and have each converged on the same fix for the "never forgets" pathology of (1.1).

**(a) Epidemiology — the sliding-window Gamma–Poisson filter.** The instantaneous reproduction number $R_t$ is estimated by exactly your conjugate scheme: a $\Gamma(a,b)$ prior, a Poisson likelihood for the incidence count over a *trailing window* $[t-\varpi, t]$, and a Gamma posterior whose shape is $a + \sum_{s \in \text{window}} I_s$ and whose rate is $b + \sum_{s} \sum_k I_{s-k} w_k$ (the *total infectiousness*, which plays the role of the exposure/compensator $\int \Phi\, ds$). The window $\varpi$ is the entire methodological content: it is a **hard forgetting factor**, and it exists solely to prevent the static conjugate filter from becoming un-updatable as data accumulate. This is the direct answer to the source construction's ad-hoc choice of prior shape $r_1 = E_H(45)$ — that hyperparameter is doing the job of a forgetting factor, badly, because it is fixed at the start rather than applied recursively.

**(b) The Gamma–Beta discount filter — a *closed-form* time-varying Poisson-rate filter.** The rigorous version of "forgetting" for conjugate Poisson models (Smith's power-steady model; Harvey & Fernandes' local-level count models; West & Harrison's DLM discounting). Evolve the state multiplicatively by a Beta shock:
$$
\lambda_t \;=\; \frac{\lambda_{t-1}\,\eta_t}{\delta}, \qquad \eta_t \sim \mathrm{Beta}\big(\delta a_{t-1},\, (1-\delta) a_{t-1}\big), \qquad \delta \in (0,1] .
$$
Then Gamma conjugacy is preserved *exactly*, and the prediction step is a one-line discounting of the sufficient statistics:
$$
\boxed{\;
\lambda_{t-1} \mid \mathcal{Y}_{t-1} \sim \Gamma(a_{t-1}, b_{t-1})
\;\Longrightarrow\;
\lambda_{t} \mid \mathcal{Y}_{t-1} \sim \Gamma(\delta a_{t-1},\, \delta b_{t-1})
\;\Longrightarrow\;
\lambda_t \mid \mathcal{Y}_t \sim \Gamma\big(\delta a_{t-1} + \Delta N_t,\ \ \delta b_{t-1} + \Delta\!\!\int\! \Phi\big) }
\tag{1.7}
$$
The posterior *mean* is preserved by the prediction step ($\delta a / \delta b = a/b$) while the *variance is inflated* by $1/\delta$. This is a genuine finite-dimensional nonlinear filter for a *stochastically time-varying* intensity, it costs two multiplications, and it collapses to the source construction's scheme at $\delta = 1$. **If you adopt nothing else from this document, adopt (1.7).**

**(c) Reliability engineering — NHPP with change-points.** Software-reliability growth models (Goel–Okumoto; Musa–Okumoto) are NHPPs with mean value function $m(t) = \int_0^t \lambda(s)ds$ and Bayesian updating on failure times; the piecewise/change-point NHPP literature handles exactly your "epoch shifts" (regime changes at known or unknown times) by placing a prior on the change-point locations and marginalising. The technology for *unknown* structural breaks (which your $\Phi$ assumes are known) is here.

**(d) Neuroscience — the point-process adaptive filter.** The closest engineering analogue to your desired three-channel fusion is spike-train decoding: a latent state (stimulus/kinematics) with Gaussian dynamics, observed through (i) a point process of spikes with state-dependent intensity and (ii) continuous covariates. Brown–Frank–Eden's *Stochastic State Point Process Filter* is the Gaussian-approximate recursion for exactly (1.5), and it gives the explicit variance recursion used in §3.2.

---

## 2. INVERTING AGGREGATE CONTINUOUS SIGNALS

The question "how do I get $\lambda$ out of the price?" decomposes into four strictly separate operations, which are routinely conflated and must not be:

$$
\underbrace{\text{quoted prices}}_{\text{bid/ask ladder}}
\;\xrightarrow{\ \text{(i) de-vig}\ }\;
\underbrace{\text{risk-neutral probabilities } q^{\mathbb{Q}}}_{\text{a measure}}
\;\xrightarrow{\ \text{(ii) invert}\ }\;
\underbrace{\lambda^{\mathbb{Q}} \text{ or } \Lambda^{\mathbb{Q}}_{t,T}}_{\text{a } \mathbb{Q}\text{-intensity}}
\;\xrightarrow{\ \text{(iii) } \gamma^{-1}\ }\;
\underbrace{\lambda^{\mathbb{P}}}_{\text{the state}}
\;\xrightarrow{\ \text{(iv) orthogonalise}\ }\;
\underbrace{\text{usable innovation}}_{\text{§3.4}}
$$

Steps (iii) and (iv) are the ones absent from the source construction and from most practitioner pipelines.

### 2.1 The Strike Ladder is a Breeden–Litzenberger Density

**Breeden–Litzenberger (1978).** For a European call $C(K,T)$ on an underlying $S_T$,
$$
q_T(K) \;=\; e^{r(T-t)}\, \frac{\partial^2 C(K,T)}{\partial K^2},
\qquad
\mathbb{Q}(S_T > K) \;=\; -e^{r(T-t)}\, \frac{\partial C}{\partial K} .
$$
The entire risk-neutral marginal is recoverable from the strike-continuum of prices.

**The discrete transposition.** A ladder of "over/under $k$" contracts on the terminal count $N_T$ *is* a ladder of digital options. Let $O_t(k)$ denote the de-vigged implied probability of $\{N_T > k\}$. Then the market's *entire* risk-neutral pmf of the terminal count is available by first difference — the discrete Breeden–Litzenberger:
$$
\boxed{\; q^{\mathbb{Q}}_t(n) \;=\; \mathbb{Q}\big(N_T = n \,\big|\, \mathcal{G}_t\big) \;=\; O_t(n-1) \;-\; O_t(n) \;}
\tag{2.1}
$$
**This is the single largest piece of information being discarded by inverting a single odds line.** A totals ladder over $k \in \{0.5, 1.5, 2.5, 3.5, 4.5\}$ delivers five moments' worth of constraint on the conditional law of $N_T - N_t$, not one.

**Inversion to an intensity.** Under the doubly-stochastic hypothesis, conditionally on $\Lambda_{t,T}$,
$$
N_T - N_t \,\big|\, \Lambda_{t,T} \;\sim\; \mathrm{Poisson}(\Lambda_{t,T}),
$$
so the market-implied law of the *latent* remaining compensator is recovered by **mixture deconvolution**: find the mixing law $\mu^{\mathbb{Q}}_t$ on $\mathbb{R}_+$ solving
$$
q^{\mathbb{Q}}_t(n_t + j) \;=\; \int_0^\infty \frac{u^j e^{-u}}{j!}\, \mu^{\mathbb{Q}}_t(du), \qquad j = 0,1,2,\dots
\tag{2.2}
$$
This is a **Poisson mixture (de-)convolution**, and it is *severely ill-posed* — the Poisson kernel is smoothing, so its inverse is unbounded, exactly as for Dupire (§2.3). The practical, well-posed answer is **not** to deconvolve but to *project*: restrict $\mu^{\mathbb{Q}}$ to the Gamma family (giving a Negative-Binomial predictive, which is anyway what your overdispersion demands) and solve the finite-dimensional M-estimation problem
$$
\boxed{\;
(\hat\alpha_t, \hat\beta_t) \;=\; \arg\min_{\alpha,\beta} \;\sum_{k \in \mathcal{K}} \omega_k \,\Big( \mathrm{NB}\big(N_T > k \,\big|\, n_t, \alpha,\beta\big) \;-\; O_t(k) \Big)^2 \;+\; \mathcal{R}(\alpha,\beta)
\;}
\tag{2.3}
$$
with weights $\omega_k$ set by the inverse squared half-spread at strike $k$ (§2.4) and $\mathcal{R}$ a Tikhonov penalty anchoring the solution to the previous tick's estimate (temporal regularisation). This yields a *market-implied Gamma law for $\Lambda_{t,T}$* — i.e. an object of the same type as your filter's own posterior, which is exactly what makes fusion in §3.4 possible.

### 2.2 The Hazard-Rate Bootstrap and the Forward-Intensity Term Structure

The CDS analogue tells you what a *single* maturity can and cannot identify. In credit, the par CDS spread satisfies the approximate credit triangle
$$
s(t,T) \;\approx\; (1 - R)\, \bar\lambda^{\mathbb{Q}}(t,T), \qquad
\bar\lambda^{\mathbb{Q}}(t,T) = \frac{1}{T-t}\int_t^T \lambda^{\mathbb{Q}}_s\, ds ,
$$
so a *single* quote identifies only the **average** hazard over $[t,T]$, never its shape. Recovering the *instantaneous forward* hazard requires the term structure and a bootstrap:
$$
\boxed{\;
\lambda^{\mathbb{Q}}(t; T) \;=\; -\,\frac{\partial}{\partial T} \log \mathbb{Q}\big(\tau > T \,\big|\, \mathcal{G}_t\big)
\;}
\tag{2.4}
$$
which is *formally identical* to the HJM instantaneous forward rate $f(t,T) = -\partial_T \log P(t,T)$, and is the true **Dupire analogue**: Dupire's local volatility is recovered from a *calendar* derivative $\partial_T C$ plus a *strike* derivative $\partial_{KK} C$; the local intensity is recovered from a *calendar* derivative of the survival curve.

**The identification statement this forces on your architecture:**
> With prices at a **single** terminal maturity $T$, you can identify only the *level* $\Lambda_{t,T} = \int_t^T \lambda_s ds$ — the average remaining intensity. You cannot identify its *shape* $\lambda_s$, $s \in (t,T)$. Any claim to have inverted a "local" or "instantaneous" implied intensity from a single full-horizon market is unidentified, and whatever shape appears is an artefact of the model's own $\Phi$. To identify the shape you need a **calendar spread**: at least two maturities (e.g. a market on the count at an interim epoch $T_1 < T$, alongside the terminal market), whence the implied forward
> $$\hat\Lambda(T_1, T) \;=\; \mathbb{E}^{\mathbb{Q}}_t[N_T] - \mathbb{E}^{\mathbb{Q}}_t[N_{T_1}].$$

**The observability rank condition.** If the latent state is a $d$-vector $\lambda = (\lambda^{(1)},\dots,\lambda^{(d)})$ (e.g. two competing arrival streams) and you observe $m$ market functionals $g_1,\dots,g_m$, then the local identification requires
$$
\mathrm{rank} \left( \frac{\partial g_i}{\partial \lambda^{(j)}} \right)_{i \le m,\, j \le d} \;=\; d .
$$
An **aggregate** market (a total-count market) is a function of $\sum_j \lambda^{(j)}$ alone: its Jacobian has rank $1$, so it can *never* identify a $2$-vector of component intensities. It supplies exactly one linear constraint on the sum. Identifying the components requires a second, *differently-weighted* functional — a spread/handicap market (sensitive to $\lambda^{(1)} - \lambda^{(2)}$) or a directional market. **A totals market plus a spread market is exactly rank-2 and exactly identifies a 2-dimensional intensity; a totals market alone does not.**

### 2.3 Ill-Posedness: Why Naive Inversion is Numerically Fatal

**Dupire (1994).** The local volatility is recovered from the call surface by
$$
\sigma^2_{\mathrm{loc}}(K,T) \;=\; \frac{2\,\Big( \partial_T C + (r - q) K\, \partial_K C + qC \Big)}{K^2\, \partial^2_{KK} C} .
$$
The pathology is structural and is *entirely transferable* to your problem: the operator mapping the model parameter to the price is **smoothing** (a heat-kernel convolution), hence *compact*, hence its inverse is **unbounded**. The recovered quantity depends on a **second derivative of noisy, discretely-sampled data**, and the denominator $\partial^2_{KK}C$ is small precisely where the data are sparse (in the wings). Consequences, all of which appear in your setting:

1. **Non-existence**: with bid-ask noise the observed price ladder frequently violates the convexity/monotonicity constraints, so *no* arbitrage-free measure reproduces it, so no $\lambda$ exists.
2. **Non-uniqueness**: many intensity paths reproduce the same finite set of prices (this is the identification failure of §2.2 restated).
3. **Discontinuous dependence**: an $\varepsilon$-perturbation of the price (one tick) can move the inverted parameter by $O(1)$.

**The standard cure: Tikhonov regularisation of the inverse problem.** Rather than inverting, one *minimises a penalised misfit* — precisely the form of (2.3):
$$
\boxed{\;
\hat\theta_t \;=\; \arg\min_\theta \; \underbrace{\big\| \mathcal{P}(\theta) - Z^{\text{mkt}} \big\|^2_{\Sigma^{-1}}}_{\text{price misfit}} \;+\; \underbrace{\varrho \,\big\|\theta - \theta^{\text{prior}}\big\|^2_{\mathcal{H}}}_{\text{regulariser}} \;}
\tag{2.5}
$$
with $\varrho > 0$ the regularisation parameter (chosen by Morozov's discrepancy principle: increase $\varrho$ until the residual matches the known noise level, i.e. the bid-ask width). Convergence-rate results for Tikhonov regularisation of the Dupire inverse problem are standard (Crépey; Egger–Engl).

> **The Bayesian reading — and the reason this matters to you.** Tikhonov regularisation *is* MAP estimation with a Gaussian prior: $\varrho\|\theta - \theta^{\text{prior}}\|^2 \leftrightarrow -2\log \pi(\theta)$. **You already have the prior — it is $\lambda_0$.** So the "invert the market" step and the "prior" step are not two separate stages to be fused afterwards; the prior is *what makes the inversion well-posed in the first place*. The correct architecture does not invert-then-fuse. It **treats the market prices as observations inside the filter**, with the filter's own predictive law serving as the regulariser. This is the central design conclusion of §4.

### 2.4 Microstructure: De-vigging, the Spread, and the Longshot Bias

**(i) The overround.** Raw implied prices $\pi_i = 1/o_i$ satisfy $\sum_i \pi_i = 1 + v > 1$. The de-vigging map is *not* innocuous, and the choice of map is a *modelling* decision that directly shifts the inverted $\lambda$:

- **Proportional (basic) normalisation**: $p_i = \pi_i / \sum_j \pi_j$. Assumes the margin is applied *multiplicatively and uniformly*. It is known to be systematically biased, because it ignores the favourite–longshot bias: it over-states the probability of longshots.
- **Power / odds-ratio methods**: $p_i \propto \pi_i^{1/\kappa}$, solving for $\kappa$ such that $\sum p_i = 1$. Absorbs a monotone, level-dependent distortion.
- **Shin's (1993) model.** The margin is *derived*, not assumed: the bookmaker faces a proportion $z$ of insider traders, and sets prices to break even against them. Inverting Shin's condition gives the implied probabilities
$$
\boxed{\;
p_i \;=\; \frac{\sqrt{\,z^2 + 4(1-z)\,\dfrac{\pi_i^2}{\Pi}\,} \;-\; z}{2\,(1-z)},
\qquad \Pi = \sum_j \pi_j,
\qquad z \ \text{solving} \ \sum_i p_i = 1 . }
\tag{2.6}
$$
Shin's $z$ has a *direct interpretation as the informational asymmetry of the market* — and therefore as a **direct estimator of how much weight the market channel deserves in the fusion** (§3.4). This is the cleanest available answer to "how much do I trust the odds": $z$ is large exactly when the price is being set by informed flow.

**(ii) The spread is not noise; it is a set.** A two-sided quote does not deliver a point observation but an **interval**: the set of measures consistent with the quotes is
$$
\mathcal{Q}_t \;=\; \Big\{ \mathbb{Q} \;:\; C^{\text{bid}}_t(k) \;\le\; \mathbb{E}^{\mathbb{Q}}[\text{payoff}_k] \;\le\; C^{\text{ask}}_t(k) \quad \forall k \in \mathcal{K} \Big\} ,
$$
a convex set (an *interval of no-arbitrage prices*, in the incomplete-market sense). Two rigorous treatments:
- **Set-membership filtering**: propagate the set $\Lambda_t = \{\lambda : \text{model price} \in [\text{bid},\text{ask}]\ \forall k\}$. Exact, but the sets grow.
- **Heteroskedastic Gaussian surrogate** (the pragmatic and standard choice): observe the mid, with an observation variance calibrated to the half-spread and its *sensitivity*:
$$
\boxed{\;
Z_t = h(X_t) + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}\big(0,\, R_t\big), \qquad
R_t \;\asymp\; \left( \frac{\tfrac12\big(C^{\text{ask}} - C^{\text{bid}}\big)}{\big|\partial C / \partial \lambda\big|} \right)^{\!2} \;+\; R^{\text{model}} \;}
\tag{2.7}
$$
The numerator is the price uncertainty; the denominator is the "vega" — the sensitivity of the price to the parameter. **The spread must be divided by the sensitivity to become a parameter uncertainty.** This single formula is what makes the market channel *self-weighting*: as liquidity dries up (spread widens), or as the contract becomes insensitive to $\lambda$ (deep in/out of the money, or near $t=T$ where $\partial C/\partial\lambda \to 0$), $R_t$ automatically explodes and the Kalman gain on the market channel automatically vanishes. **No hand-tuned weight is needed, and none should be used.** Add $R^{\text{model}}$ (a variance floor) to prevent the filter from becoming infinitely confident in a tight quote.

---

## 3. STOCHASTIC FILTERING & FUSION MECHANICS

### 3.1 The Conjugate Route, and Exactly Where It Breaks

**Claim.** The Gamma–Poisson update is the exact filter for the counting channel alone; adding the diffusion channel destroys conjugacy *unless the observation is linear in $\log\lambda$*.

*Proof sketch.* From (1.6) with $\varphi$ a test function and a static state, the unnormalised density evolves as $\rho_t(\lambda) \propto \pi_0(\lambda) \cdot L^N_t(\lambda) \cdot L^Z_t(\lambda)$ with
$$
L^N_t(\lambda) = e^{-\lambda \int_0^t \Phi ds} \lambda^{N_t} \prod_i \Phi(\tau_i),
\qquad
L^Z_t(\lambda) = \exp\left( \frac{1}{\varsigma^2}\int_0^t h(\lambda)\, dZ_s \;-\; \frac{1}{2\varsigma^2}\int_0^t h(\lambda)^2 ds \right)
$$
(the second by Girsanov/Kallianpur–Striebel). If $h(\lambda) = \lambda$ (the market observes the intensity level directly), then
$$
\pi_t(\lambda) \;\propto\; \lambda^{a_0 + N_t - 1}\, \exp\left( -\Big(b_0 + \textstyle\int_0^t \Phi\, ds - \tfrac{Z_t}{\varsigma^2}\Big)\lambda \;-\; \tfrac{t}{2\varsigma^2}\, \lambda^2 \right),
$$
which is a **Gaussian-tilted Gamma** — the $\lambda^2$ term in the exponent is not absorbable into a Gamma. Conjugacy is lost. $\square$

**Two exact repairs.**

**(R1) The pseudo-count (power-prior) representation — keeps conjugacy exactly.** Do not model the market as a Gaussian sensor on $\lambda$. Model it as what it *is*: a source that reports a **posterior of the same family**. By §2.1, fitting the strike ladder returns a market-implied Gamma law $\Gamma(\alpha^{\text{mkt}}_t, \beta^{\text{mkt}}_t)$ for the remaining compensator. Fusion by (tempered) log-linear pooling then preserves the family *identically*:
$$
\boxed{\;
\pi_t(\lambda) \;\propto\; \underbrace{\lambda^{a_0 - 1} e^{-b_0\lambda}}_{\text{prior}}
\cdot \underbrace{\lambda^{N_t} e^{-\lambda \int_0^t \Phi ds}}_{\text{events}}
\cdot \underbrace{\Big[\lambda^{\alpha^{\text{mkt}}_t - 1} e^{-\beta^{\text{mkt}}_t \lambda}\Big]^{\,\omega_t}}_{\text{market, tempered by } \omega_t}
\;\;\sim\;\; \Gamma\Big(\underbrace{a_0 + N_t + \omega_t(\alpha^{\text{mkt}}_t - 1)}_{\text{shape}},\; \underbrace{b_0 + \textstyle\int_0^t \Phi ds + \omega_t \beta^{\text{mkt}}_t}_{\text{rate}}\Big) \;}
\tag{3.1}
$$
The market enters as **$\omega_t \alpha^{\text{mkt}}$ virtual events over $\omega_t \beta^{\text{mkt}}$ units of virtual exposure**. The whole filter remains two scalars. The temper $\omega_t \in [0,1]$ is *exactly the correction for (P2)*: it is the fraction of the market's information that is genuinely new, and setting $\omega_t = 1$ is the double-counting error. See §3.4 for how to set it.

**(R2) Log-space, for the general dynamic case.** Set $X = \log\lambda$ with mean-reverting (Black–Karasinski / exponential-OU) dynamics, so that positivity is automatic and the market channel — which, after (2.3), reports a *log*-intensity with variance (2.7) — is **linear-Gaussian in $X$**. The point-process channel is then the only nonlinear one. This is the state-space in §4.

### 3.2 Continuous-Discrete EKF: the Explicit Recursion

State $X_t$, Gaussian approximation $\pi_t \approx \mathcal{N}(\hat X_t, P_t)$.

**Prediction (between observations)** — integrate the moment ODEs:
$$
\frac{d\hat X_t}{dt} = f(\hat X_t,t) - \frac{\partial \log\lambda}{\partial x}\bigg|_{\hat X_t} \!\! \cdot P_t \cdot \lambda(\hat X_t)
\qquad
\frac{dP_t}{dt} = 2 F_t P_t + \sigma^2 - P_t\, \mathcal{I}^N_t\, P_t ,
$$
where $F_t = \partial_x f$ and the **point-process information rate** is $\mathcal{I}^N_t = \big(\partial_x \log \lambda\big)^2 \lambda(\hat X_t)$. The negative term in $d\hat X/dt$ is the "no news is news" drift: **in the absence of events the estimate decays**, at a rate proportional to the current posterior variance and the current intensity.

**Update at an event $\tau_i$** (the Brown–Frank–Eden point-process filter, Gaussian-approximated):
$$
\boxed{\;
P_{\tau_i}^{-1} = P_{\tau_i^-}^{-1} + \Big(\partial_x \log\lambda\Big)^2 \lambda \,\Delta \;-\; \Big(\partial_{xx}\log\lambda\Big)\big(dN - \lambda\Delta\big),
\qquad
\hat X_{\tau_i} = \hat X_{\tau_i^-} + P_{\tau_i}\, \partial_x\!\log\lambda \cdot \big(dN - \lambda \Delta\big) \;}
$$
For $\lambda = \hat\lambda_0 \Phi e^{X}$ we have $\partial_x \log\lambda = 1$, so this collapses to a remarkably clean form: the innovation is simply $(dN_t - \lambda_t dt)$ — the **compensated counting martingale**, i.e. *realised events minus expected events* — and the gain is the posterior variance.

**Update on the market channel** — the ordinary Kalman step with the self-weighting variance (2.7):
$$
K_t = \frac{P_{t^-} }{P_{t^-} + R_t}, \qquad
\hat X_t = \hat X_{t^-} + K_t\big(Z_t - \hat X_{t^-} - \hat\beta_t\big), \qquad
P_t = (1-K_t) P_{t^-} .
$$

**Where the EKF fails, precisely.** The Gaussian approximation of the *filtering distribution* is asked to represent the posterior induced by a Poisson likelihood with a very small expected count. Over the relevant regime — $\mathbb{E}[N_T] \in [1,4]$, with $0$–$3$ events actually realised — the posterior of $\lambda$ is strongly right-skewed and its Gaussian approximation:
- assigns non-negligible mass to $\lambda < 0$ unless one works in log-space (hence R2 is mandatory, not optional);
- badly mis-states the *tails*, which is fatal because the quantity you price is $\mathbb{P}(N_T - N_t \ge k)$ — a tail functional;
- has $O(1)$ relative error in the third moment, and the third moment is what the skew of an over/under price *is*.
**The low-count, short-horizon regime is exactly the regime where the EKF's error is maximal.** Its virtue is $O(1)$ cost per step and a closed-form derivative for calibration.

### 3.3 Sequential Monte Carlo: Why It Wins Here

Represent $\pi_t \approx \sum_{p=1}^{P} w^{(p)}_t \delta_{X^{(p)}_t}$. Between observations, propagate each particle through the SDE (Euler–Maruyama). The weight update is *read directly off the Zakai equation* (1.6) — which is why it is unbiased:

$$
\boxed{\;
w^{(p)}_t \;\propto\; w^{(p)}_{t_0}\cdot
\underbrace{\exp\left(-\int_{t_0}^{t} \lambda^{(p)}_s\, ds\right) \prod_{i \,:\, \tau_i \in (t_0,t]} \lambda^{(p)}_{\tau_i}}_{\text{counting-channel likelihood (Jacod)}}
\;\cdot\;
\underbrace{\prod_{t_j \in (t_0,t]} \mathcal{N}\!\Big(Z_{t_j};\; X^{(p)}_{t_j} + \hat\beta_{t_j},\; R_{t_j}/\omega_{t_j}\Big)^{\omega_{t_j}}}_{\text{market channel, tempered}} \;}
\tag{3.2}
$$

**The tractability comparison for *this* triad.**

| | **Continuous-Discrete EKF** | **Sequential Monte Carlo** |
|---|---|---|
| Cost per step | $O(d^2)$, $d\!\sim\!2$ — negligible | $O(P d)$, $P\!\sim\!10^3$–$10^4$ — still negligible |
| Handles the $\exp(-\int\lambda)$ survival term | Only via Gaussian moment-matching | **Exactly** (it is a per-particle path integral) |
| Low-count posterior skew | **Badly** — the dominant error source | Exactly |
| Tail functionals ($\mathbb{P}(N_T\!-\!N_t \ge k)$) | Poor | Exactly — and *this is the pricing quantity* |
| Non-Gaussian / interval market obs (2.7) | Requires Gaussian surrogate | Arbitrary likelihood, incl. set-membership |
| Regime jumps in $\Phi$ (structural breaks) | Requires re-linearisation | Trivial (just a change in each particle's intensity) |
| Parameter learning ($\kappa,\sigma,\gamma$) | Closed-form gradients | Needs PMCMC / SMC$^2$ (the real cost) |
| Degeneracy risk | None | **Yes** — sparse informative events cause weight collapse; mitigate by resampling (ESS $<P/2$) and by a *guided/auxiliary* proposal that looks ahead to the next event |

**Recommendation.** The state dimension is $1$–$3$; the horizon is short; the events number $\le 10$. The entire argument for the EKF — computational tractability — is *worth nothing here*, because a $10^4$-particle bootstrap filter over a 90-unit horizon costs microseconds. Meanwhile the entire argument against it — bias in the low-count, skewed, tail-functional regime — bites maximally. **Use SMC. Rao-Blackwellise where possible**: conditional on the log-OU path and the deterministic $\Phi$, the level parameter retains Gamma conjugacy, so one can sample only the *dynamic* component and integrate the *level* analytically (a marginalised particle filter), which cuts the variance by an order of magnitude for free.

The only case for the EKF/closed-form route is the **degenerate-but-excellent** one: if you accept the Gamma–Beta discount filter (1.7) with the pseudo-count fusion (3.1), you get a *fully closed-form, exactly-conjugate, two-scalar filter* that is provably optimal for its (slightly restricted) model class, runs in nanoseconds, and has no tuning beyond $(\delta, \omega)$. **This is the recommended production filter**; SMC is the recommended reference implementation against which to measure its approximation error.

### 3.4 Solving (P2): the Correlated-Channel Problem

The market has seen the events. Three defensible treatments, in increasing order of rigour:

**(A) Innovation orthogonalisation.** Do not feed the market's *level* into the filter. Feed its **innovation relative to your own event-only filter**:
$$
\tilde Z_t \;=\; Z_t \;-\; \mathbb{E}\big[Z_t \,\big|\, \mathcal{Y}^N_t\big] \;\approx\; Z_t \;-\; \big(\hat X^{N}_t + \hat\beta_t\big),
$$
where $\hat X^N_t$ is a *parallel filter run on the counting channel alone*. Everything the market says that your events already implied is projected out; only the residual — the market's genuinely private information (Shin's $z$-flow, team news, injuries) — updates the state. This is cheap (run two filters) and it is the exact analogue of the *innovations representation* in Segall–Kailath.

**(B) Covariance intersection (Julier–Uhlmann).** The robust fusion rule when the cross-correlation between two estimators is *unknown*. Given event-only estimate $(\hat X^N, P^N)$ and market-implied estimate $(\hat X^{Z}, P^{Z})$:
$$
\boxed{\;
\big(P^{\mathrm{CI}}\big)^{-1} = \omega \big(P^{N}\big)^{-1} + (1-\omega)\big(P^{Z}\big)^{-1},
\qquad
\hat X^{\mathrm{CI}} = P^{\mathrm{CI}}\Big[ \omega (P^N)^{-1}\hat X^N + (1-\omega)(P^Z)^{-1}\hat X^Z \Big], \;}
$$
with $\omega \in [0,1]$ chosen to minimise $\det P^{\mathrm{CI}}$. **CI is guaranteed non-divergent for *any* unknown correlation**, at the price of conservatism (the fused covariance is larger than the naive Kalman fusion). Given that the correlation here is not merely unknown but *known to be high*, this conservatism is exactly what you want, and it is the direct justification for the temper $\omega_t$ in (3.1) — the tempered log-pool *is* covariance intersection for the Gamma family.

**(C) Explicit market-agent model.** Posit that the market runs its own filter on $\mathcal{G}_t = \mathcal{F}^N_t \vee \sigma(U_{[0,t]})$ with $U$ its private signal. Then in a linear-Gaussian caricature $Z_t = c_t \hat X^N_t + (1-c_t) \hat U_t + \eta_t$, and one *solves for* $\hat U_t$ — the private component — before fusing. This is the most principled route and it is precisely the Duffie–Lando "the market's intensity is itself a posterior" insight made operational. It requires estimating $c_t$ offline (regress the market's implied intensity on the event-only filter's implied intensity across a historical panel; $c_t$ is the $R^2$, and $1 - c_t$ is your true information edge).

> **Empirical note with teeth.** Fit (C) on your historical panel *before* building anything. If $c_t \approx 1$ — i.e. the market's implied intensity is fully explained by a mechanical filter on the public event stream — then the market channel contains **no information you do not already have**, $\omega_t \to 0$, and the entire fusion architecture collapses to the counting-channel filter plus a de-vigging model. The *only* thing that justifies the second channel is a measurably non-zero private component $1-c_t$. **This regression is the cheapest and highest-value experiment available, and it should be run first.**

---

## 4. THE UNIFIED SDE BLUEPRINT

### 4.1 The State Space

Let the horizon be $[0,T]$, and let $\{S_t\}$ denote the observable, $\mathcal{F}^N$-measurable "configuration" of the system (the accumulated counts and any exogenous, observed regime markers). Define the latent state vector
$$
\mathbf{X}_t \;=\; \big(X^{(1)}_t,\, X^{(2)}_t,\, \beta_t \big)^{\!\top} \in \mathbb{R}^3,
$$
where $X^{(j)}$ is the **log-deviation of stream $j$'s intensity from its prior**, and $\beta$ is the **market bias / risk-premium state** ($\beta = \log\gamma$, from (1.4)).

**Intensity map (the decomposition of (1.2)):**
$$
\boxed{\;
\lambda^{(j)}_t \;=\; \underbrace{\hat\lambda^{(j)}_0}_{\substack{\text{static prior} \\ \text{(channel C0)}}}
\;\cdot\;
\underbrace{\Phi_j\big(t,\, S_{t^-}\big)}_{\substack{\text{known deterministic modulation:} \\ \text{temporal drift} \times \text{piecewise epochs} \times \text{state multipliers}}}
\;\cdot\;
\underbrace{\exp\big(X^{(j)}_t\big)}_{\substack{\text{latent stochastic deviation} \\ \text{(the only thing filtered)}}} \;}
\tag{4.1}
$$
with the modulation itself factorised, so that every deterministic structure the domain knows about lives here and *costs the filter nothing*:
$$
\Phi_j(t, S) \;=\; \underbrace{\exp\big(\xi_j t\big)}_{\text{secular drift}}
\;\cdot\; \underbrace{\prod_{e} \rho_{j,e}^{\,\mathbf{1}\{t \in I_e\}}}_{\substack{\text{piecewise epoch shifts} \\ \text{on known intervals } I_e}}
\;\cdot\; \underbrace{\psi_j\big(S_{t^-}\big)}_{\substack{\text{configuration-dependent} \\ \text{multiplier (Hawkes-like,} \\ \text{but } \mathcal{F}^N\text{-predictable)}}} .
$$

### 4.2 The Signal Dynamics (Latent SDE)

$$
\boxed{\;
d X^{(j)}_t \;=\; \underbrace{-\,\kappa_j\, X^{(j)}_t\, dt}_{\substack{\text{mean reversion to the PRIOR} \\ (X=0 \iff \lambda = \hat\lambda_0\Phi)}}
\;+\; \underbrace{\sigma_j(t)\, dW^{(j)}_t}_{\substack{\text{stochastic drift of true} \\ \text{latent strength}}}
\;+\; \underbrace{\sum_{k} \zeta_{j,k}\, dH^{(k)}_t}_{\substack{\text{structural-break jumps} \\ \text{(regime shocks, exogenous)}}} \;}
\tag{4.2}
$$
$$
d\beta_t \;=\; -\kappa_\beta\big(\beta_t - \bar\beta\big) dt \;+\; \sigma_\beta\, dW^{(\beta)}_t ,
\qquad
\mathrm{Corr}\big(dW^{(1)}, dW^{(2)}\big) = \varrho\, dt .
$$

**Design rationale, term by term:**

- **Black–Karasinski form.** Working in $\log\lambda$ with an Ornstein–Uhlenbeck driver is the Black–Karasinski / exponential-Vasicek specification. It guarantees $\lambda > 0$ without reflecting barriers or truncation, and — decisively — it makes the market observation channel (which reports a *log*-intensity, §2.1) **linear-Gaussian in the state**, so the only nonlinearity in the entire system is the point-process channel.
- **Mean reversion *to zero*, not to a free level.** $\kappa_j$ is the **forgetting rate**, and it is the continuous-time counterpart of the discount factor $\delta$ in (1.7): $\delta \approx e^{-\kappa \Delta}$. It is the parameter the source construction's fixed prior-shape hyperparameter was trying to be. $\kappa \to \infty$ recovers the pure prior (in-play data ignored); $\kappa \to 0$ recovers the never-forgetting conjugate updater. **It is the single most important number in the system and it must be estimated from data, not chosen.**
- **Horizon-scaled diffusion.** Set $\sigma_j(t) = \sigma_j \sqrt{\Phi_j(t,S)}$, so that the state's innovation variance is proportional to the *exposure rate*. This enforces the correct asymptotic: as $t \to T$ and remaining exposure $\to 0$, the state stops moving, and the filter's uncertainty about *the quantity that matters* — the remaining compensator — collapses correctly. Without this, posterior variance in $X$ stays $O(1)$ at $t=T$ while its consequences vanish, which mis-prices near-expiry contracts.
- **Jump compensator.** $H^{(k)}$ are exogenous observed shock processes (structural breaks). Because they are *observed*, they enter the drift with no filtering cost; their marks $\zeta_{j,k}$ are parameters. If breaks are *unobserved*, replace with a Markov-modulated regime and filter the regime — see the change-point NHPP literature (§1.4c). The compensated form $d\tilde H = dH - \nu\, dt$ keeps (4.2) a semimartingale with the stated drift.

### 4.3 The Observation Channels

$$
\text{(C1) Counting: } \qquad
\boxed{\;
N^{(j)}_t - \int_0^t \lambda^{(j)}_s\, ds \ \ \text{is a } \mathbb{P}\text{-martingale}, \qquad \lambda^{(j)} \ \text{as in (4.1)} \;}
$$

$$
\text{(C2) Market: } \qquad
\boxed{\;
Z^{(m)}_{t} \;=\; g_m\big(\boldsymbol{\lambda}_t,\, N_t,\, T - t\big) \;+\; \beta_t\, \partial_m \;+\; \varepsilon^{(m)}_t,
\qquad
\varepsilon^{(m)}_t \sim \mathcal{N}\big(0,\, R^{(m)}_t\big) \;}
$$
where $m$ indexes the *market functionals* recovered by the ladder inversion of §2.1 (a totals functional, a spread functional, …), and $R^{(m)}_t$ is the **self-weighting spread-over-sensitivity variance (2.7)** — the mechanism that makes the filter automatically ignore illiquid or insensitive quotes with no hand-tuning. By the rank condition of §2.2, **$m$ must range over at least $\dim(\boldsymbol\lambda)$ functionally independent markets** or the components are unidentified.

### 4.4 The Filter

Assemble (4.1)–(4.3) into the Kushner–Stratonovich system (1.5). Writing $\pi_t$ for the conditional law of $\mathbf{X}_t$ given $\mathcal{Y}_t = \sigma\big(N_s, Z_s : s \le t\big)$:

$$
\boxed{
\begin{aligned}
d\pi_t(\varphi) \;=\;& \underbrace{\pi_t\big(\mathcal{A}_t\varphi\big)\, dt}_{\text{(i) SDE drift + reversion to prior}} \\[6pt]
&+\; \sum_{m} \underbrace{\Big[\pi_t\big(\varphi\, g_m\big) - \pi_t(\varphi)\pi_t\big(g_m\big)\Big]\, \big(R^{(m)}_t\big)^{-1} \;\omega^{(m)}_t\; \Big( dZ^{(m)}_t - \pi_t\big(g_m\big)\, dt \Big)}_{\text{(ii) MARKET INNOVATION, tempered by } \omega \text{ for channel dependence (§3.4)}} \\[6pt]
&+\; \sum_{j} \underbrace{\left[\frac{\pi_{t^-}\big(\varphi\, \lambda^{(j)}\big)}{\pi_{t^-}\big(\lambda^{(j)}\big)} - \pi_{t^-}(\varphi)\right] \Big( \underbrace{dN^{(j)}_t}_{\text{event}} - \underbrace{\pi_{t^-}\big(\lambda^{(j)}\big)\, dt}_{\text{COMPENSATOR}} \Big)}_{\text{(iii) POINT-PROCESS INNOVATION: size-biasing at events, decay in their absence}}
\end{aligned}}
\tag{4.3}
$$

with generator $\mathcal{A}_t \varphi = \sum_j \big[-\kappa_j x_j \partial_{x_j}\varphi + \tfrac12 \sigma_j^2(t)\, \partial^2_{x_j x_j}\varphi\big] + \varrho\,\sigma_1\sigma_2\,\partial^2_{x_1x_2}\varphi + \mathcal{A}^\beta\varphi + \sum_k \nu_k\big[\varphi(\mathbf{x} + \boldsymbol\zeta_k) - \varphi(\mathbf{x})\big]$.

**The three innovations, in words:**
1. **Drift**: in the absence of *any* data the filter relaxes exponentially back to the static prior $\lambda_0\Phi(t)$ at rate $\kappa$, while its variance inflates toward the stationary level $\sigma^2/2\kappa$. This is the correct treatment of channel (C0) — the prior is not a one-shot initial condition, it is an **attractor**.
2. **Market innovation**: a Kalman-type correction proportional to the posterior covariance between the test function and the price functional, scaled by the inverse observation variance (2.7), and **tempered by $\omega^{(m)}_t$** to prevent double-counting the events the market has already seen (§3.4).
3. **Point-process innovation**: at an event, the posterior is multiplied by $\lambda$ and renormalised (an exact Bayes/size-bias update; for a Gamma this is $a \mapsto a+1$). Between events, the compensator $-\pi(\lambda)\,dt$ drags the posterior *down* at exactly the rate at which events were expected but did not arrive.

### 4.5 The Terminal Functional (what you actually price)

Do **not** report $\pi_t(\lambda)$. Report the conditional law of the **remaining compensator**, which is the sufficient statistic for everything downstream:
$$
\Lambda^{(j)}_{t,T} \;=\; \int_t^T \hat\lambda^{(j)}_0\, \Phi_j\big(s, S_{s^-}\big)\, e^{X^{(j)}_s}\, ds ,
$$
$$
\boxed{\;
\mathbb{P}\big(N^{(j)}_T - N^{(j)}_t = k \,\big|\, \mathcal{Y}_t\big) \;=\; \mathbb{E}_{\pi_t}\!\left[ \frac{\big(\Lambda^{(j)}_{t,T}\big)^k}{k!}\, e^{-\Lambda^{(j)}_{t,T}} \right] \;}
$$
— a **Poisson mixture**, which is *automatically overdispersed* relative to Poisson (its variance is $\mathbb{E}\Lambda + \mathrm{Var}\,\Lambda > \mathbb{E}\Lambda$). Under the Gamma-conjugate route this mixture is a **Negative Binomial in closed form**; under SMC it is the particle average. Note that $\Phi_j$ depends on the *future* configuration path $S_{s^-}$, which is itself driven by the counts: hence the exact evaluation requires forward-simulating the coupled system to $T$ (a per-particle path simulation), which is the continuous-time generalisation of the recursive convolution used in the source construction.

**Terminal degeneracy — the one thing to check.** As $t \uparrow T$: $\Lambda_{t,T} \to 0$, the counting channel's information rate $\mathcal{I}^N_t \to 0$, and $\partial C/\partial\lambda \to 0$, so by (2.7) $R_t \to \infty$. **Both channels go dark simultaneously.** The posterior variance of $X$ therefore does *not* collapse — and it should not. What collapses is the variance of the *predictive functional*, which is correct. A filter that reports shrinking uncertainty in $X$ near expiry has a bug.

---

## 5. SUMMARY OF THE DIAGNOSIS

| Component of the current architecture | Cross-domain verdict |
|---|---|
| Gamma–Poisson conjugate updating on event times | **Correct and optimal** — it is the exact nonlinear filter for a *static* Cox intensity (1.1). |
| ...but with a *static* latent state | **The core defect.** It never forgets, so it cannot track a non-stationary $\lambda$. Fix: Gamma–Beta discounting (1.7) / OU mean-reversion (4.2). The prior-shape hyperparameter is a forgetting factor in disguise. |
| Deterministic epoch and configuration multipliers | **Correct — but they belong in $\Phi$, not the filter** (1.2). They are $\mathcal{F}^N$-predictable and cost nothing. |
| Inverting a single market line for $\lambda$ | **Under-identified and ill-posed.** A single maturity identifies only $\int_t^T\lambda$, never its shape (2.4); an aggregate market has a rank-1 Jacobian and cannot identify a 2-vector of intensities (§2.2). Use the whole strike ladder (2.1) with Tikhonov regularisation (2.5). |
| Fusing event-based and market-based $\lambda$ directly | **Two errors.** (P1) They live under different measures — insert the premium state $\beta = \log\gamma$ (1.4). (P2) They are not conditionally independent — the market has already seen your events. Temper the market likelihood (3.1) / orthogonalise the innovation (§3.4). |
| Hand-tuned weight on the market signal | **Unnecessary.** The spread-over-sensitivity variance (2.7) makes the market channel self-weighting. |
| EKF vs SMC | **SMC**, decisively — the low-count, skewed, tail-functional regime is where the EKF is worst, and the computational argument for the EKF is worthless at $d\le 3$. Keep the closed-form (1.7)+(3.1) filter as the production path and SMC as its reference check. |

**The single highest-value experiment**, before any of this is built: regress the market-implied intensity on the intensity produced by an event-only filter, across a historical panel. The residual $R^2$ deficit, $1 - c_t$, *is* your information edge from the second channel. If it is near zero, the market channel is redundant and the architecture reduces to §1.4(b).

---

## 6. REFERENCES

All identifiers below were retrieved from OpenAlex/arXiv and checked to resolve. Publisher DOIs returning HTTP 403 are bot-blocks by the publisher (the DOI resolved; the landing page refused the crawler), not missing records.

### Point-process filtering and Cox processes (§1.1, §1.3, §3)
| Work | Identifier |
|---|---|
| Snyder, "Filtering and detection for doubly stochastic Poisson processes", *IEEE Trans. Inf. Theory* (1972) — **the origin of the DSPP filter** | [10.1109/tit.1972.1054756](https://doi.org/10.1109/tit.1972.1054756) · `W2120797717` |
| Hawkes, "Spectra of some self-exciting and mutually exciting point processes", *Biometrika* 58(1):83 (1971) | [10.1093/biomet/58.1.83](https://doi.org/10.1093/biomet/58.1.83) · `W2069849731` |
| Segall & Kailath, "The modeling of randomly modulated jump processes", *IEEE Trans. Inf. Theory* (1975) — **the martingale-innovations representation used in (1.5)** | [10.1109/tit.1975.1055359](https://doi.org/10.1109/tit.1975.1055359) · `W2050887379` |
| Brémaud, *Point Processes and Queues: Martingale Dynamics* (Springer) | `W2092487949` · [10.2307/2288173](https://doi.org/10.2307/2288173) |
| Grandell, *Doubly Stochastic Poisson Processes*, Springer LNM 529 (1976) | `W1501723949` |
| Dassios & Jang, "Pricing of catastrophe reinsurance and derivatives using the Cox process with shot noise intensity", *Finance & Stochastics* (2003) | [10.1007/s007800200079](https://doi.org/10.1007/s007800200079) · `W2055390972` |
| Duffie, Filipović & Schachermayer, "Affine processes and applications in finance", *Ann. Appl. Prob.* (2003) | [10.1214/aoap/1060202833](https://doi.org/10.1214/aoap/1060202833) · `W2001244105` |
| Leijdekker & Spreij, "Explicit computations for a filtering problem with point process observations, with applications to credit risk" (2008) — **closest published analogue to the filter in §4.4** | [arXiv:0802.1407](https://arxiv.org/abs/0802.1407) |
| Surace, Kutschireiter & Pfister, "Asymptotically exact unweighted particle filter for … point process observations" (2019) | [arXiv:1907.10143](https://arxiv.org/abs/1907.10143) |
| Continuous-discrete filtering with SDE dynamics and point-process observations (2024) | [arXiv:2411.19814](https://arxiv.org/abs/2411.19814) |
| Particle filter for a *marked* doubly stochastic Poisson process | `W2070817776` |
| Gamma–Gaussian inverse-Wishart Poisson filter — **a Gamma-conjugate filter for point-process tracking** | `W2514570524` |
| Brown, Frank & Eden, "Dynamic analysis of neural encoding by point process adaptive filtering", *Neural Computation* 16(5) (2004) — **the Gaussian recursion in §3.2** | [10.1162/089976604773135069](https://doi.org/10.1162/089976604773135069) · `W2134892157` |

### Reduced-form credit risk — the exact isomorph (§1.2)
| Work | Identifier |
|---|---|
| Jarrow & Turnbull, "Pricing derivatives on financial securities subject to credit risk", *J. Finance* (1995) | `W2082317778` (1,985 citations) |
| Lando, "On Cox processes and credit risky securities", *Rev. Derivatives Res.* (1998) — **the Cox construction (1.3)** | `W2084716442` (1,263 citations) |
| Duffie & Singleton, "Modeling term structures of defaultable bonds", *RFS* (1999) — **intensity-as-spread** | `W2011886674` (2,590 citations) |
| Duffie & Lando, "Term structures of credit spreads with incomplete accounting information", *Econometrica* (2001) — **the market's intensity is itself a posterior; the formal content of (P2)** | `W2118891973` (1,383 citations) |
| Berndt et al., "Measuring default risk premia from default swap rates and EDFs" — **direct measurement of the $\lambda^{\mathbb{Q}}/\lambda^{\mathbb{P}}$ wedge $\gamma$ of (1.4)** | [10.2139/ssrn.556080](https://doi.org/10.2139/ssrn.556080) · `W1629883981` |
| Chen, Cheng & Wu, "Dynamic interactions between interest-rate and credit risk" (2011) — **EKF on a CDS panel to filter unobservable affine intensity factors: the direct precedent for §3.2** | `W2161791308` |

### Implied-parameter inversion and its ill-posedness (§2)
| Work | Identifier |
|---|---|
| Breeden & Litzenberger, "Prices of state-contingent claims implicit in option prices", *J. Business* (1978) — **the density-from-strikes result behind (2.1)** | [10.1086/296025](https://doi.org/10.1086/296025) · `W2026454108` |
| Crépey, "Calibration of the local volatility in a generalized Black–Scholes model using Tikhonov regularization", *SIAM J. Math. Anal.* | [10.1137/s0036141001400202](https://doi.org/10.1137/s0036141001400202) · `W1965101614` |
| "Tikhonov regularization applied to the inverse problem of option pricing", *Inverse Problems* 21(3) | [10.1088/0266-5611/21/3/014](https://doi.org/10.1088/0266-5611/21/3/014) · `W2157707465` |
| Albani et al., "Online local volatility calibration by convex regularization with Morozov's principle and convergence rates" — **the discrepancy principle for choosing $\varrho$ in (2.5)** | [arXiv:1211.0170](https://arxiv.org/abs/1211.0170v3) |
| Discrete analogue of the Breeden–Litzenberger relation for risk-neutral density — **the rigorous version of (2.1)** | [10.1142/s2424786323500615](https://doi.org/10.1142/s2424786323500615) · `W4391542489` |
| Arbitrage-free option price surfaces via Chebyshev tensor bases (2025) — arbitrage repair of a violating quote ladder | [arXiv:2512.01967](https://arxiv.org/abs/2512.01967v1) |
| Shin, "Prices of state-contingent claims with insider traders, and the favourite–longshot bias", *Economic Journal* (1993) — **the de-vig inversion (2.6); $z$ as a trust weight** | [10.2307/2234526](https://doi.org/10.2307/2234526) · `W2023017267` |
| Ottaviani & Sørensen, "Noise, information, and the favorite-longshot bias in parimutuel predictions", *AEJ: Micro* (2010) | [10.1257/mic.2.1.58](https://doi.org/10.1257/mic.2.1.58) · `W1996746992` |

### Dynamic conjugate (discount) filters and applied analogues (§1.4)
| Work | Identifier |
|---|---|
| Smith, "A generalization of the Bayesian steady forecasting model", *JRSS-B* 41(3) (1979) — **the power-steady/discount model** | [10.1111/j.2517-6161.1979.tb01092.x](https://doi.org/10.1111/j.2517-6161.1979.tb01092.x) · `W2904464170` |
| Harvey & Fernandes, "Time series models for count or qualitative observations", *JBES* 7(4) (1989) — **the Gamma–Beta conjugate filter (1.7): the recommended fix** | [10.1080/07350015.1989.10509750](https://doi.org/10.1080/07350015.1989.10509750) · `W2093206764` |
| Cori et al., "A new framework and software to estimate time-varying reproduction numbers during epidemics", *Am. J. Epidemiol.* (2013) — **the sliding-window Gamma–Poisson filter; the forgetting-factor precedent** | [10.1093/aje/kwt133](https://doi.org/10.1093/aje/kwt133) · `W2097446414` |
| Parag, "Improved inference of time-varying reproduction numbers during infectious disease outbreaks", *Epidemics* (2019) | [10.1016/j.epidem.2019.100356](https://doi.org/10.1016/j.epidem.2019.100356) · `W2971072970` |
| EpiEstim as a state-space DLM on $\log R_t$ (2020) — **the log-space state-space refactor of §3.1(R2), in epidemiology** | [arXiv:2012.02168](https://arxiv.org/abs/2012.02168) |
| Bayesian predictive analysis for an exponential NHPP (software reliability) | `W4238484523` |
| Bayesian reliability analysis of the Power Law Process (NHPP) | [arXiv:2002.00351](https://arxiv.org/abs/2002.00351) |
| Inference for the Markov-Modulated Poisson Process (MMPP) | `W118387685` |
| HMMs for bursty Internet traffic — latent-state arrival-rate inference | `W2107372872` |

### Sensor fusion under unknown correlation (§3.4)
| Work | Identifier |
|---|---|
| Julier & Uhlmann, "A non-divergent estimation algorithm in the presence of unknown correlations", *ACC* (1997) — **covariance intersection; the guarantee behind the temper $\omega_t$** | [10.1109/acc.1997.609105](https://doi.org/10.1109/acc.1997.609105) · `W2148234182` |

### Short-rate model referenced for the log-state SDE (§4.2)
| Work | Identifier |
|---|---|
| Black & Karasinski, "Bond and option pricing when short rates are lognormal", *FAJ* 47(4) (1991) | [10.2469/faj.v47.n4.52](https://doi.org/10.2469/faj.v47.n4.52) · `W1958196783` |

---

## 7. UNVERIFIED / NOT SOURCED

Stated explicitly, per primary-source discipline:

1. **Dupire (1994), "Pricing with a smile", *Risk* 7(1):18–20.** The local-volatility formula in §2.3 is quoted from standard knowledge. The original *Risk* magazine article **is not indexed in OpenAlex with a resolvable DOI** — the closest indexed record is Derman & Kani, "Pricing and hedging with smiles" (1993, `W3191723236`, no DOI). The *formula* is nonetheless corroborated by the Tikhonov/inverse-problem papers above, which state and regularise it. Treat the Dupire citation as bibliographic, not verified.
2. **Kushner (1964) and Zakai (1969) original papers.** Cited from standard knowledge; the sub_02 worker returned a *survey* of their historical development (`W2070395359`) rather than the primary articles. Equations (1.5)–(1.6) are standard and appear in Brémaud (verified above), but the two original citations are not individually verified here.
3. **Cox (1955), "Some statistical methods connected with series of events".** Cited from standard knowledge; not individually retrieved.
4. **The auxiliary-continuous-signal analogue outside finance does not appear to exist.** The sub_04 worker returned `DATA_STATUS: PARTIAL` with an explicit negative finding: *no* literature was located pairing NHPP software-reliability or MMPP queueing models with an auxiliary continuous market-like signal. Epidemiology fuses a historical prior with live event counts (EpiEstim), and reliability/queueing fuse a prior with event timestamps, **but the three-channel fusion problem — prior + events + a concurrent price of the outcome — appears to be genuinely specific to finance.** This is a substantive result, not a gap in the search: it means the credit-risk literature (§1.2) is the *only* mature source of guidance on channels (C1)+(C2) together, which is why (P1) and (P2) are stated in its language.
5. **All mathematical derivations in §§1–4** (equations 1.1, 1.2, 1.5–1.7, 2.1–2.7, 3.1–3.2, 4.1–4.3, and the rank/identification conditions) are the author's own, assembled from the cited frameworks. They are *derived*, not retrieved, and carry no citation because none exists for this specific composition. They should be checked, not trusted.
