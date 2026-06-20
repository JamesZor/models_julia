# Feature Validation Methodology — Framework, Formulas & Logic

A mathematical companion to `momentum_statistical_analysis.md` and
`current_development/datastore_feature_review.md`. Those hold the *results*; this
defines *how* each number was produced, *what hypothesis each test answers*, and
the *statistical logic* tying it all together. Written to be checkable line by line.

---

## 0. Notation

- Matches indexed $m = 1,\dots,M$. Each match has a home and away team.
- A **team-game** is one (team, match) pair; each match yields two team-games.
  Index a team's own games chronologically $g_1 \prec g_2 \prec \dots$.
- Per team-game observables: goals $y$, expected goals $x$ (xG), big chances $c$,
  touches in opp. box, shots in box, etc. Superscripts $H,A$ = home/away when at
  match level; subscript "for"/"ag" = a team's own vs conceded value.
- $\mathbb{1}[\cdot]$ indicator; $T_m$ = match length in minutes.

---

## 1. Feature construction

### 1.1 In-match momentum and time-weighted AUC
SofaScore gives a signed per-minute momentum series $v_{m,t}$, $t=1,\dots,T_m$,
with $v>0$ = home pressure, $v<0$ = away. The recency-weighted attacking areas:

$$
A^{H}_m = \sum_{t=1}^{T_m} \max(0,\, v_{m,t})\, e^{-\lambda (T_m - t)},
\qquad
A^{A}_m = \sum_{t=1}^{T_m} \max(0,\, -v_{m,t})\, e^{-\lambda (T_m - t)},
\qquad \lambda = 0.03.
$$

The weight $w_t = e^{-\lambda(T_m-t)}$ is a geometric recency kernel: minute $t$
counts $e^{-\lambda}\approx 0.97$ as much as minute $t{+}1$; half-weight horizon
$\ln 2/\lambda \approx 23$ min. **Momentum difference** $\Delta A_m = A^H_m - A^A_m$.

> *Logic / caveat.* $A$ is an unnormalised area, so it scales with $T_m$; fine for
> within-sample correlation, fragile as a cross-match feature. The kernel
> up-weights late minutes — which §3 shows are the most score-distorted.

### 1.2 Scoreline (game-state) reconstruction from incidents
From goal incidents $G_m$ (each with minute $\tau_g$ and scoring side), the
home-minus-away lead *entering* minute $t$:

$$
s_{m,t} \;=\; \sum_{g \in G_m} \mathbb{1}[\tau_g < t]\,
\big(\mathbb{1}[\text{home credited }g] - \mathbb{1}[\text{away credited }g]\big),
$$

own goals credited to the **beneficiary**. This is the step function of the
scoreboard over the 90 minutes.

### 1.3 Game-state–conditioned momentum
For a team with perspective sign $\sigma = +1$ (home) / $-1$ (away):
team momentum $\tilde v_t = \sigma v_{m,t}$, team goal-diff state
$\tilde s_t = \sigma s_{m,t}$. Bucket $b(t) = \mathrm{clamp}(\tilde s_t, -2, 2)$.
Mean momentum **while in state** $k$:

$$
\bar m_k \;=\; \frac{\sum_{t} \tilde v_t\,\mathbb{1}[b(t)=k]}{\sum_t \mathbb{1}[b(t)=k]},
\qquad k \in \{-2,-1,0,+1,+2\}.
$$

*Level-state* momentum is the AUC restricted to $\{t: s_{m,t}=0\}$.
*Behavioural responses:* $\text{down-resp} = \bar m_{-1}-\bar m_0$,
$\text{lead-resp} = \bar m_{+1}-\bar m_0$ (within-team change vs the level baseline,
which strips the "good teams dominate everywhere" level effect).

### 1.4 Efficiency ("xG per pressure")
Per team-game $e = x / A_{\text{for}}$ — xG produced per unit attacking momentum.
Note $\log e = \log x - \log A$, so **given $x$, the residual variation in $e$ is
$-\log A$** (pure inverse pressure). This identity predicts the §4c sign flip.

### 1.5 Rolling (expanding-window) priors — the no-leakage feature
For a team's $k$-th game, any observable $z$ becomes a *pre-match* feature by
averaging strictly earlier games:

$$
\bar z^{(k)} \;=\; \frac{1}{k-1}\sum_{j=1}^{k-1} z_{g_j}, \qquad k \ge 6 \;(\text{require } \ge 5 \text{ priors}).
$$

Because $\bar z^{(k)}$ uses only $g_1,\dots,g_{k-1}$, regressing the match outcome
on it estimates **genuine forecasting** power (no look-ahead leakage). At match
level we form differences $\Delta\bar z_m = \bar z^{H}_{\text{home}} - \bar z^{}_{\text{away}}$,
and for the Poisson model the attack/defence split
$\,a = \bar z^{\text{for}}_{\text{team}},\; d = \bar z^{\text{ag}}_{\text{opp}}$.

---

## 2. The unifying principle: information = regularization = incremental prediction

Treat team strength as a latent $\theta$ (attack/defence). Each observable is a
**measurement** of $\theta$ through its own likelihood:
goals $y\sim p_1(\cdot\mid\theta)$, xG $x\sim p_2(\cdot\mid\theta)$,
big chances $c\sim p_3(\cdot\mid\theta)$. Under conditional independence,

$$
p(\theta\mid y,x,c)\;\propto\; p(\theta)\,p_1(y\mid\theta)\,p_2(x\mid\theta)\,p_3(c\mid\theta),
\qquad
\mathcal{I}(\theta) = \sum_j \mathcal{I}_j(\theta),
$$

i.e. **Fisher information adds across channels.** A new channel $c$ sharpens the
posterior on $\theta$ **iff** $\mathcal{I}_3(\theta)>0$, i.e. iff $c$ still depends
on $\theta$ *after conditioning on the channels you already have*. The linear
proxy for "depends on $\theta$ given $x$" is the **partial correlation**
$r_{c\,\cdot\,\theta\mid x}\neq 0$. Hence the slogan we kept returning to:

> "Extra information about the latent state" **=** "non-zero conditional
> dependence given xG" **=** "incremental predictive power over xG."
> They are one quantity. You cannot have regularization value without it.

**Conditional-independence failure (the trap).** If two channels are correlated
*given* $\theta$ — e.g. $\mathrm{corr}(x,c\mid\theta)=0.69$ because a big chance
*is* high xG — then $p(x,c\mid\theta)\neq p(x\mid\theta)p(c\mid\theta)$, and the
factorised likelihood **double-counts** information: $\sum_j\mathcal I_j$
over-states the truth, the posterior on $\theta$ becomes **too tight**, and
predicted probabilities are over-confident → calibration (log-loss) can *worsen*.
Remedy = **tempering**: raise each channel to a weight, $\prod_j p_j^{\,w_j}$, with
$w_j<1$ recalibrating total information (this is what `market_weight` already does).

---

## 3. The tests — definition, statistic, null, what it answers

### 3.1 Pearson correlation
$r=\dfrac{\sum_i(x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum_i(x_i-\bar x)^2\sum_i(y_i-\bar y)^2}}$,
tested via $t = r\sqrt{\tfrac{n-2}{1-r^2}}\sim t_{n-2}$, $H_0:\rho=0$.
**Answers:** is there *any* linear association? (e.g. momentum vs goals/xG.)

### 3.2 Partial correlation — the incremental-value test
$$
r_{xy\cdot z} \;=\; \frac{r_{xy}-r_{xz}\,r_{yz}}{\sqrt{(1-r_{xz}^2)(1-r_{yz}^2)}},
\qquad t = r_{xy\cdot z}\sqrt{\tfrac{n-3}{1-r_{xy\cdot z}^2}}\sim t_{n-3}.
$$
Equivalently: regress $x$ on $z$ and $y$ on $z$, correlate the residuals. With
$z=$ rolling xG, this is exactly $\mathcal I_3(\theta)>0$ in linear form.
**Answers:** does feature $x$ tell us anything about the outcome *beyond* xG?
(Momentum: $+0.11$. bigChance: $+0.24$. Efficiency: $-0.13$ — the §1.4 identity.)

### 3.3 Split-half reliability + Spearman–Brown
Split each team's games into halves (odd/even to neutralise form drift); means
$a_i,b_i$ over $i=1..20$ teams; half-length correlation $r_{1/2}=\mathrm{cor}(a,b)$.
The **Spearman–Brown prophecy** rescales to full length ($k=2$ halves):
$$
\rho \;=\; \frac{k\,r_{1/2}}{1+(k-1)\,r_{1/2}} \;=\; \frac{2\,r_{1/2}}{1+r_{1/2}}.
$$
$\rho$ estimates the fraction of a feature's variance that is a **stable team
trait** vs within-team noise (a signal-to-noise / reliability coefficient).
**Answers:** is this even a real trait before asking if it predicts? Kills noise
features cheaply (finishing $\rho=0.32$; down-response $\rho=-0.09$ → noise;
touchesInOppBox $\rho=0.97$ → cleaner than xG).

### 3.4 Paired $t$-test (game-state change)
Per match $d_i=\mu^{\text{post}}_i-\mu^{\text{pre}}_i$ (leading team's momentum after
minus before the first goal); $t=\bar d/(s_d/\sqrt n)\sim t_{n-1}$, $H_0:\mathbb E[d]=0$.
**Answers:** does taking the lead change momentum? (Yes: $\bar d=-8.4$,
$t=-14.7$ — the score-state confound, quantified.)

### 3.5 Nested linear models (Gaussian GLM / OLS)
$R^2 = 1-\dfrac{\mathrm{SS_{res}}}{\mathrm{SS_{tot}}}$;
adjusted $\bar R^2 = 1-(1-R^2)\dfrac{n-1}{n-p-1}$ (penalises parameters);
$\mathrm{AIC}=2k-2\hat\ell$ (fit vs complexity, lower better);
nested $F$-test for adding $q$ regressors:
$$
F=\frac{(\mathrm{SS_{res}^{0}}-\mathrm{SS_{res}^{1}})/q}{\mathrm{SS_{res}^{1}}/(n-p_1-1)}\sim F_{q,\,n-p_1-1}.
$$
**Answers:** how much variance, in interpretable goal units (coefficients ± SE),
and is the extra block *jointly* significant? (xG→+xG+mom: $F$-test $p=0.0024$.)

### 3.6 Walk-forward / chronological hold-out
Two distinct sources of look-ahead, both removed:
1. **Feature leakage** — handled by the expanding-window priors (§1.5).
2. **Coefficient leakage** — fit $\hat\beta$ on the chronologically first 70%,
   evaluate on the last 30%, with the baseline mean taken from **train**:
$$
R^2_{\text{oos}}=1-\frac{\sum_{i\in\text{test}}(y_i-\hat y_i)^2}{\sum_{i\in\text{test}}(y_i-\bar y_{\text{train}})^2}.
$$
**Answers:** the honest predictive power. The in-sample → out-of-sample gap is the
**overfit/noise** measure. (Goal-diff $R^2$ fell $0.090\!\to\!0.011$ — goals are
mostly irreducible noise; the smoother xG-diff target held $\sim 0.07$–$0.10$.)

### 3.7 Poisson GLM (faithful to the Dixon-Coles likelihood)
Counts $y_i\sim\mathrm{Poisson}(\mu_i)$, log link $\log\mu_i=x_i^\top\beta$ (so
attack/defence enter **multiplicatively**, as in the engine). Per-observation
log-likelihood
$$
\ell_i = y_i\log\mu_i-\mu_i-\log(y_i!).
$$
Model comparison on **held-out mean log-likelihood** $\frac1{|\text{test}|}\sum_{\text{test}}\ell_i$
— the count analogue of the model's log-loss (closer to 0 = less surprised by
actual goals). **Answers:** in the model's *own* metric, does the feature lift
predictive likelihood out of sample, and by how much vs the noise floor?
(xG lifts $+0.037$ over null; $+$bigChance$+$touches lifts a further $+0.010$,
overfit gap $\sim0.01$.)

### 3.8 Overdispersion diagnostic (Poisson vs Negative-Binomial)
Poisson imposes $\mathrm{Var}[y]=\mathbb E[y]=\mu$. Dispersion ratio
$\hat\phi=\widehat{\mathrm{Var}}/\widehat{\text{mean}}$. If $\hat\phi>1$ use
NegBin $y\sim\mathrm{NB}(\mu,r)$ with $\mathrm{Var}=\mu+\mu^2/r$ (Gamma-mixed
Poisson). **Answers:** is Poisson calibrated? (goals $\hat\phi=1.06$ ✓;
bigChance $1.28$ borderline; shotsInsideBox $1.93$ → Poisson miscalibrated, needs
NegBin — a reason to prefer big-chances as a per-match pillar.)

---

## 4. From validation to the engine: the measurement model

Latent log-rate (already in the model):
$\log\lambda^{H}_m = \mu_{\text{season}} + \gamma_{\text{HA}} + \mathrm{att}^H + \mathrm{def}^A$,
$\lambda^H = \kappa_H\,e^{\log\lambda^H}$. Existing pillars share this $\lambda$:

$$
y^H\sim\mathrm{Poisson}(\lambda^H)\ \text{(Dixon-Coles }\tau),\quad
x^H\sim\mathrm{Gamma}(\nu,\lambda^H/\nu)\ (\mathbb E=\lambda^H),\quad
\text{market: } \log\lambda^H\sim\mathcal N.
$$

A big-chance pillar is the same idea with its own scale $\omega$:
$$
c^H \sim \mathrm{Poisson}(\omega\,\lambda^H), \qquad
\mathbb E[c^H]=\omega\,\lambda^H .
$$
Design constraints follow directly from §2–3:
- **$\omega$ global or tightly pooled** ($\sigma_\omega$ small), else a free
  per-team $\omega$ absorbs the team signal and **decouples** $c$ from
  $\mathrm{att}/\mathrm{def}$ — destroying the shared-latent regularization.
  Scale prior $\omega\sim\mathrm{TruncNormal}(1.35,0.2)$ from $\bar c/\bar y=1.85/1.37$.
- **Temper the pillar** ($w<1$) because $\mathrm{corr}(x,c\mid\theta)=0.69$ breaks
  conditional independence (§2) — otherwise the latent posterior is over-tight.
- **Poisson borderline** ($\hat\phi=1.28$) → consider `RobustNegativeBinomial`.
- **Mask** $c$ like `xg_mask` (coverage 4 477 < goals); add `BigChanceFeature`.
- **Decision rule:** sweep $w\in\{0,0.25,0.5,1\}$, choose by **held-out 1X2
  log-loss** vs the current `DCMH_HalfLife_60`. $w^\star=0$ ⇒ redundant with
  xG+market ⇒ drop. (In-sample fit *always* improves — only held-out log-loss
  exposes the over-confidence, so it is the sole valid arbiter.)

---

## 5. Result → test map (so nothing is unaccounted for)

| Claim we made | Test (section) | Key statistic |
|---|---|---|
| Momentum ≈ xG, not goals | Pearson §3.1 | $r=0.60$ vs $0.22$ |
| Leading teams' momentum drops | Paired $t$ §3.4 | $\bar d=-8.4,\ t=-14.7$ |
| Momentum adds ~0 beyond xG (within match) | Partial corr §3.2 | $r_{\cdot\mid x}\approx 0$/neg |
| Momentum adds a little (rolling, OOS) | Partial corr + held-out §3.2/3.6 | $+0.11$; $R^2_{oos}\!\uparrow$ |
| "xG per pressure" hurts | identity §1.4 + partial §3.2 | $-0.13$; $r(e,A\mid x)=-0.82$ |
| "Fight/give-up" trait isn't real | Reliability §3.3 | $\rho=-0.09$ |
| Several stats cleaner than xG | Reliability §3.3 | $\rho$ up to $0.97$ |
| bigChance/touches beat momentum | Partial corr §3.2 | $+0.24/+0.21$ |
| Counts are Poisson-suitable | Dispersion §3.8 | $\hat\phi=1.06$–$1.28$ |
| Features lift the model's own metric | Poisson held-out LL §3.7 | $+0.010$ over xG |
| Big-chance pillar design | Measurement model §4 | shared $\lambda$, tempered $w$ |

---

### Reading order for self-study
§2 (why the whole thing works) → §1 (what we built) → §3 (how each claim was
tested) → §4 (how it feeds the engine). Every number in the two result docs maps
to a row of §5.
