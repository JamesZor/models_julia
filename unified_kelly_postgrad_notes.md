# Optimal Capital Allocation over Mutually-Exclusive Score Grids under Parameter Uncertainty

**Postgraduate lecture notes — merging the structural Kelly portfolio (Long 2026 / Jacot–Mochkovitch 2023 / Whelan 2024) with Baker–McHale (2013) shrinkage.**

> **Reading note.** These notes were written as a corrected replacement for an
> earlier draft (`postgrad_betting_notes.md`, Gemini 3.1). That draft has the right
> *ambition* but a **fatal error** in its central object: the shrinkage step it
> proposes, integrated over an MCMC posterior, provably returns *no shrinkage*
> (`k*=1`) because expected log-wealth is **linear in the probability vector**. §6
> diagnoses this precisely; §5 gives the corrected construction; §7 verifies both
> numerically. Read §0 first for the executive summary of what changed and why.

---

## 0. Executive summary — what is wrong, and what is right

We want one object: a vector of stakes on overlapping football markets (e.g. *Under 1.5* and *BTTS*) that (i) respects the fact that the underlying scorelines are mutually exclusive, and (ii) is protected against the fact that our model's probabilities are MCMC estimates, not truth.

The literature gives us two clean tools:

| Layer | Tool | What it buys us |
|---|---|---|
| **Structure** | Log-optimal portfolio over the state grid (Long 2026 for *direct* state bets; Jacot–Mochkovitch 2023 for *overlapping* market bets; Whelan 2024 for the concave-utility generalisation) | Stakes that price mutual exclusivity *exactly*, never double-stake contradictory markets, and automatically include negative-EV hedges where they raise log-growth. |
| **Shrinkage** | Baker–McHale 2013 | A factor `k ∈ (0,1]` that pulls the whole portfolio toward cash to correct for estimation error in the probabilities. |

The earlier draft's mistake is in **how the two layers are joined**. It froze the stakes at their posterior-mean values and then integrated the *evaluation probability* over the posterior. Because $\sum_\omega p(\omega)\log W(\omega)$ is **linear in $p$**, that integral collapses to the posterior mean and the optimiser returns $k^\*=1$. The variance of the posterior is integrated away before it can do any work.

The fix is to put the randomness **in the decision, not in the evaluation weight**: the shrinkage must correct for the fact that the *stakes themselves* are a noisy function of the estimated probabilities. This is exactly Baker–McHale's frequentist "optimiser's curse" argument, and it survives the linearity that kills the naive version. Numerically (§7), the corrected objective gives `k*≈0.88` under a tight posterior and `k*≈0.44` under a wide one — genuine, variance-responsive protection — while the naive objective gives `k*≈1.0` in both cases.

---

## 1. Formal setting

### 1.1 State space and markets

A match has a finite set of mutually exclusive, exhaustive **states** (exact scorelines)
$$\Omega=\{\omega_1,\dots,\omega_N\},\qquad \sum_{\omega\in\Omega}p(\omega)=1 .$$
Let $p=\bigl(p(\omega)\bigr)_{\omega\in\Omega}$ be the bettor's probability vector. The market does not, in general, let us bet on individual states. Instead it offers $M$ **compound markets** $m=1,\dots,M$, each a state-contingent claim defined by an indicator
$$\mathcal I_m(\omega)\in\{0,1\},\qquad \text{decimal odds } d_m>1 .$$
Write the implied (state-price) probability $q_m=1/d_m$.

> **The two structural framings — and which paper applies.**
>
> * If the available claims are the *states themselves* (one indicator per state — a horse race, or betting exact correct-score), this is **Long (2026)**: a closed-form threshold rule and a one-pass greedy support selection.
> * If the claims are *compound markets whose indicators overlap several states* (U1.5, BTTS, Over 2.5, …), this is the **non-mutually-exclusive** problem of **Jacot–Mochkovitch (2023)**: no closed form, but a smooth concave program with an explicit gradient and Hessian.
>
> Long is the special case of Jacot–Mochkovitch in which the market matrix is the identity on states. The earlier draft attributed the *overlapping*-market resolution to Long's threshold formula; the threshold formula does **not** apply once a single stake pays in more than one state. The correct citation for the U1.5-vs-BTTS object is Jacot–Mochkovitch (with Whelan for the concave/uniqueness theory).

### 1.2 Terminal wealth

Stake fractions $a=(a_1,\dots,a_M)\ge 0$ with cash $c=1-\mathbf 1^\top a\ge 0$. If state $\omega$ occurs, each winning market returns stake×odds and the rest is lost:
$$W(\omega)=c+\sum_{m=1}^M a_m\,d_m\,\mathcal I_m(\omega)
=1+\sum_{m=1}^M a_m\bigl(d_m\mathcal I_m(\omega)-1\bigr)
=1+r(\omega)^\top a,$$
where the **per-state return vector** is
$$r_m(\omega):=d_m\mathcal I_m(\omega)-1=\begin{cases}d_m-1 & \text{market $m$ wins in $\omega$}\\[2pt]-1 & \text{market $m$ loses in $\omega$.}\end{cases}$$
This is exactly Jacot–Mochkovitch Eq. (12): $W(\omega)=1+r(\omega)^\top a$. Long's $W_i=c+x_i/q_i$ is the same expression specialised to one claim per state.

### 1.3 The two problems

1. **Structural problem.** Markets overlap and contradict. *Under 1.5* and *BTTS* are mutually exclusive ($\mathcal I_{\mathrm{U}}\cdot\mathcal I_{\mathrm B}\equiv 0$, since both teams scoring forces ≥2 goals), so treating them as independent two-outcome Kelly bets is structurally wrong.
2. **Epistemic problem.** $p$ is unknown. We have a Bayesian posterior predictive distribution (PPD) $f(p\mid\mathcal D)$ from MCMC, with posterior mean $\bar p=\mathbb E_{\mathrm{PPD}}[p]$ and posterior covariance $\Sigma=\operatorname{Cov}_{\mathrm{PPD}}(p)$.

We solve them in sequence: **structure first** (§3–4, treating $\bar p$ as if known), then **shrinkage** (§5, correcting for $\Sigma$). §6 explains *why the sequence cannot be collapsed into a single posterior-mean expectation* — that is the trap.

---

## 2. The deterministic structural program

Assume for now $p=\bar p$ is known. The log-optimal portfolio solves
$$
\boxed{\;a^\star=\arg\max_{a\ge 0,\ \mathbf 1^\top a\le 1}\;
G(a)=\sum_{\omega\in\Omega}\bar p(\omega)\,\log\!\bigl(1+r(\omega)^\top a\bigr)\;}
\tag{P}
$$
$G$ is strictly concave on the feasible polytope (it is a nonnegative combination of $\log$ of affine functions). Jacot–Mochkovitch give the gradient and Hessian in closed form:
$$
\nabla G(a)=\sum_{\omega}\frac{\bar p(\omega)}{1+r(\omega)^\top a}\,r(\omega),
\qquad
\nabla^2 G(a)=-\sum_{\omega}\frac{\bar p(\omega)}{\bigl(1+r(\omega)^\top a\bigr)^2}\,r(\omega)\,r(\omega)^\top .
\tag{2.1}
$$
The Hessian is the key to the structural intuition. $r(\omega)r(\omega)^\top$ is the outer product of the joint payoff pattern; summing it builds the **payoff covariance across markets**, weighted by inverse-wealth-squared. Two mutually-exclusive markets have $r_{\mathrm U}(\omega)\,r_{\mathrm B}(\omega)<0$ in every state (one pays $-1$ whenever the other pays $d-1$), so the off-diagonal Hessian term is strongly negative: the program *knows* it must not stack them. This is the rigorous content of "covariance awareness" — it is not a special rule, it is just the second-order behaviour of (P).

### 2.1 Existence and uniqueness (Whelan 2024)

Whelan's matrix analysis tells us exactly when (P) is well posed:

* With **no overround** ($\sum_m q_m=1$) and **lay bets allowed** (free sign on $a_m$), the first-order system $A\,U'=0$ has a rank-deficient $A$ and there is a **continuum** of optima — the $\varepsilon$-shift degeneracy (add the same stake to every outcome and nothing changes).
* Two features of real books break the degeneracy and give a **unique** optimum: (i) an **overround** $\sum_m q_m>1$ makes $A$ invertible; (ii) **no lay bets** ($a_m\ge 0$) turns it into a Karush–Kuhn–Tucker problem in which at least one nonnegativity constraint binds.

Both hold for us (we pay an overround, we cannot lay on a sportsbook), so (P) has a unique solution and standard interior-point / SLSQP solvers find it. This is also why we keep $a\ge 0$ and $\mathbf 1^\top a\le 1$ as hard constraints rather than solving an unconstrained FOC.

### 2.2 The Long special case (direct state bets)

When each market is a single state ($M=N$, $\mathcal I_m(\omega)=\mathbf 1[\omega=m]$), (P) has Long's closed form. Sort states by **edge ratio** $r_i:=p_i/q_i$ in decreasing order and greedily add states while $r_{k+1}>c_k$, where on support $A$
$$
c_A=\frac{1-P_A}{1-Q_A},\quad P_A=\sum_{i\in A}p_i,\ Q_A=\sum_{i\in A}q_i,
\qquad
x_i^\star=(p_i-c^\star q_i)_+,
\qquad
W_i^\star=\max\{c^\star,\;p_i/q_i\}.
\tag{2.2}
$$
The memorable picture (Long): **cash is an all-state claim**. Holding cash $c$ is equivalent to an implicit stake $c\,q_i$ on every state; an explicit bet only "tops up" favourable states from $c\,q_i$ to $p_i$. We use (2.2) as a sanity check on the numerical solver whenever the markets happen to be disjoint single-state claims — which, as it happens, is true for the U1.5/BTTS example below.

---

## 3. Worked Example A — resolving "U1.5 vs BTTS" on a valid grid

> **Fixing the earlier draft's example.** The previous notes used
> $\bar p(\omega_{00})+\bar p(\omega_{10})=0.5$ together with $\bar p(\omega_{11})=0.6$ and
> conceded "$\sum\bar p>1$ is impossible, but illustrates the contradiction." A portfolio
> example built on a non-distribution illustrates nothing — the optimiser's whole job is
> to allocate across a genuine partition. Here is the same lesson on a coherent state space.

Collapse the scoreline grid into the **coarsest partition that separates the two markets**:

| state | meaning | U1.5 pays? | BTTS pays? | $\bar p$ |
|---|---|:--:|:--:|---:|
| $\omega_U$ | Under 1.5 (0-0, 1-0, 0-1) | ✓ | ✗ | 0.40 |
| $\omega_O$ | Over 1.5, **not** BTTS (2-0, 3-0, 0-2, …) | ✗ | ✗ | 0.25 |
| $\omega_B$ | BTTS (1-1, 2-1, …) | ✗ | ✓ | 0.35 |

This is a genuine partition (BTTS ⟹ Over 1.5, so the three cells are disjoint and exhaustive). Note $\omega_O$ is **unbettable** — neither market pays there — which is what makes the hedging interesting.

Odds: U1.5 at $d_1=2.8$ (implied $0.357$), BTTS at $d_2=3.0$ (implied $0.333$). Both carry a positive subjective edge:
$$\text{EV}_{\mathrm U}=\bar p_U d_1-1=0.12,\qquad \text{EV}_{\mathrm B}=\bar p_B d_2-1=0.05.$$
Return matrix $r(\omega)$ (rows = states, cols = [U1.5, BTTS]):
$$
R=\begin{pmatrix} d_1-1 & -1\\ -1 & -1\\ -1 & d_2-1\end{pmatrix}
=\begin{pmatrix} 1.8 & -1\\ -1 & -1\\ -1 & 2.0\end{pmatrix}.
$$

**Naïve isolated Kelly** (each market as a stand-alone two-outcome bet, $f=p-\tfrac{1-p}{d-1}$):
$$f_{\mathrm U}=0.40-\tfrac{0.60}{1.8}=0.0667,\qquad f_{\mathrm B}=0.35-\tfrac{0.65}{2.0}=0.0250 .$$

**Structural solve of (P)** (here the markets are disjoint single-state claims, so Long's greedy (2.2) applies and agrees with the numerical optimum). Take support $A=\{U,B\}$:
$$P_A=0.75,\quad Q_A=\tfrac1{2.8}+\tfrac1{3.0}=0.6905,\quad
c^\star=\frac{1-0.75}{1-0.6905}=0.8077 .$$
Edge ratios $r_U=p_U/q_U=0.40\cdot2.8=1.12$ and $r_B=0.35\cdot3.0=1.05$ both exceed $c^\star=0.808$, so both stay active:
$$
x_U^\star=p_U-c^\star q_U=0.40-0.808\cdot0.357=\mathbf{0.1115},\qquad
x_B^\star=p_B-c^\star q_B=0.35-0.808\cdot0.333=\mathbf{0.0808}.
$$
Check the Long wealth identity $W_i^\star=\max(c^\star,p_i/q_i)$:
$$W_U=0.808+2.8\cdot0.1115=1.120=r_U,\quad
W_B=0.808+3.0\cdot0.0808=1.050=r_B,\quad
W_O=c^\star=0.808 .\ \checkmark$$

**Reading the result.**

1. **Mutual exclusivity is priced, not stacked.** The optimiser does not add two isolated Kelly fractions blindly; it solves the joint log-growth, keeping 80.8% in cash. If the two markets had been *positively* overlapping rather than exclusive, the Hessian (2.1) would have pulled the stakes the other way. The contradiction the earlier draft worried about is resolved *quantitatively* by the program, with no special-case logic.
2. **The joint solve is MORE aggressive than isolated Kelly** ($0.1115>0.0667$ and $0.0808>0.0250$). This is **Whelan's central result**: with multiple mutually-exclusive outcomes, the log-optimal stake on each *favoured* bet exceeds its two-outcome Kelly value, because a loss on U1.5 is partly cushioned by a possible win on BTTS (and vice versa) — the marginal utility in the "other" winning state is higher. Two-outcome Kelly cannot see this cushion because it lumps "U1.5 fails" into one undifferentiated losing state.
3. **Negative-EV hedges fall out for free.** In this example both bets are +EV, so none appears. But if, say, BTTS were priced at a small negative edge, (P) would still stake it once the U1.5 position is large, precisely as Whelan §5 documents — the hedge raises log-growth even though its arithmetic EV is negative. This behaviour is intrinsic to log-optimal allocation over a partition; it is **not** something the shrinkage layer adds (a point the earlier draft's conclusion confused).

---

## 4. From "known $\bar p$" to "MCMC $\bar p$" — the parameter-uncertainty problem

$\bar p$ is a posterior mean from MCMC, not the truth. Baker–McHale (2013) show that plugging a point estimate into a Kelly optimisation and reading off the maximised utility is **optimistically biased**: the realised out-of-sample growth of the plug-in policy is lower than the in-sample optimum suggests, and the cure is to **shrink** the stake. For a single bet with odds $b=d-1$ and estimate error variance $\sigma^2$, their workhorse approximation is
$$
k^\star\approx\frac{s^\star(p)^2}{s^\star(p)^2+\bigl((b+1)/b\bigr)^2\sigma^2},
\tag{4.1}
$$
i.e. shrink hard when the bet is small relative to your uncertainty, barely at all when the edge dwarfs the noise. We want the multi-market analogue, driven by the full posterior $f(p\mid\mathcal D)$ rather than a single $\sigma$.

---

## 5. The correct unified objective

Let $a^\star(\cdot)$ denote the solution map of the structural program (P): feed it a probability vector, get back the optimal stakes. The executed portfolio is $k\,a^\star(\bar p)$ for a scalar shrinkage $k\in(0,1]$. The **correct** out-of-sample objective is

$$
\boxed{\;
k^\star=\arg\max_{k\in(0,1]}\;
\underbrace{\mathbb E_{p^{(s)}\sim f(\cdot\mid\mathcal D)}}_{\text{noise in the }decision}
\Bigl[\;\sum_{\omega}\bar p(\omega)\,\log\!\bigl(1+k\,r(\omega)^\top a^\star(p^{(s)})\bigr)\Bigr]
\;}
\tag{U}
$$

MCMC approximation over $S$ posterior draws:
$$
k^\star\approx\arg\max_{k}\frac1S\sum_{s=1}^S\sum_\omega \bar p(\omega)\,
\log\!\bigl(1+k\,r(\omega)^\top a^\star(p^{(s)})\bigr).
\tag{U-MC}
$$

The single thing that distinguishes (U) from the broken version (§6) is **where the posterior draw enters**: inside $a^\star(p^{(s)})$, the *decision*. The evaluation weight stays at the best available estimate of the truth, $\bar p$.

### 5.1 Why (U) is the genuine out-of-sample value

Model the real repeated game honestly: each time we refit, MCMC hands us a slightly different probability vector, so the *stakes we will actually place* are a random variable $a^\star(p^{(s)})$ (decision draw $s$). The *truth* is also unknown; represent it by an independent posterior draw $p^{(t)}$. Out-of-sample growth is
$$
\mathbb E_{s,\,t}\Bigl[\sum_\omega p^{(t)}(\omega)\log\bigl(1+k\,r(\omega)^\top a^\star(p^{(s)})\bigr)\Bigr]
\;\overset{\text{indep.}}{=}\;
\mathbb E_{s}\Bigl[\sum_\omega \bar p(\omega)\log\bigl(1+k\,r(\omega)^\top a^\star(p^{(s)})\bigr)\Bigr],
$$
because the inner expectation over $t$ is linear in $p^{(t)}$ and $\mathbb E[p^{(t)}]=\bar p$. So evaluating at $\bar p$ with a *random decision* **is** the full double expectation — the cross-fitting (decision draw vs. truth draw) reduces exactly to (U). No approximation.

### 5.2 Shrinkage is guaranteed (Baker–McHale Theorem 1, restated)

**Claim.** If the solution map $a^\star(\cdot)$ is non-degenerate over the posterior (the draws do not all give the same stakes), then $k^\star<1$.

*Proof sketch.* Let $\Psi(k)=\mathbb E_s\bigl[g_s(k)\bigr]$ with $g_s(k)=\sum_\omega\bar p(\omega)\log(1+k\,r(\omega)^\top a^\star(p^{(s)}))$. Each $g_s$ is concave in $k$ and maximised at the $k$ that makes $k\,a^\star(p^{(s)})$ best *for the $\bar p$-weighting*. But for every draw $s$, $a^\star(\bar p)$ — not $a^\star(p^{(s)})$ — is the unique maximiser of $\sum_\omega\bar p(\omega)\log(1+r^\top a)$; hence $g_s(1)\le g_{\bar p}(1)$ with equality iff $a^\star(p^{(s)})=a^\star(\bar p)$. Differentiating and using Jensen exactly as in Baker–McHale gives $\Psi'(1)<0$, while $\Psi'(0^+)>0$ whenever the mean bet is +EV; with $\Psi$ concave there is a unique interior maximiser $k^\star\in(0,1)$. ∎

The bias that forces $k<1$ lives entirely in the *dispersion of the decision* $a^\star(p^{(s)})$ around $a^\star(\bar p)$ — i.e. in how sharply the optimal stakes swing as the probabilities wobble within the posterior. That dispersion is large precisely when (a) the posterior is wide, and (b) the edge is thin so stakes are near the on/off boundary (draws flip in and out of the support — the clipped-to-zero behaviour). Both are exactly the regimes where we want caution, and (U) delivers it automatically.

### 5.3 Practical recipe (MCMC / Turing.jl)

1. From the posterior, form $\bar p$ and solve (P) once for the structural portfolio $a^\star(\bar p)$.
2. Draw $S$ posterior samples $p^{(s)}$ (you already have them from MCMC).
3. For each draw solve (P) to get $a^\star(p^{(s)})$. (Subsample a few hundred draws; the solve is small and warm-startable from $a^\star(\bar p)$.)
4. Evaluate (U-MC) on a grid of $k\in(0,1]$ (or 1-D Newton, since $\Psi$ is concave); take the argmax $k^\star$.
5. Execute $k^\star\,a^\star(\bar p)$.

> A cheaper, theory-grade alternative to steps 2–4 is the **multivariate Baker–McHale Taylor approximation**: linearise $a^\star(\cdot)$ at $\bar p$ with Jacobian $J=\partial a^\star/\partial p$, so $\operatorname{Cov}(a^\star)\approx J\Sigma J^\top$, and plug into the matrix analogue of (4.1)/(B&M Eq. 14), $k^\star\approx \dfrac{a^{\star\top}H a^\star}{a^{\star\top}H a^\star+\operatorname{tr}(H\,J\Sigma J^\top)}$ with $H=-\nabla^2G$. Use it for a fast first pass; use (U-MC) when the posterior is skewed or stakes are near a support boundary, where the quadratic approximation breaks (this is the genuine advantage of integrating the real MCMC draws rather than a Gaussian/Beta surrogate).

---

## 6. The flaw in the earlier draft — diagnosed precisely

The earlier draft's unified objective (its §4) was
$$
k^\star=\arg\max_{k}\int f(p\mid\mathcal D)\Bigl[\sum_\omega p(\omega)\log\bigl(1+k\,r(\omega)^\top a^\star_{\text{naive}}\bigr)\Bigr]dp,
\tag{broken}
$$
with the stakes **frozen** at $a^\star_{\text{naive}}=a^\star(\bar p)$ and the *posterior draw used as the evaluation weight* $p(\omega)$. The integrand is **linear in $p$**, so
$$
\int f(p\mid\mathcal D)\sum_\omega p(\omega)\log(\cdot)\,dp
=\sum_\omega\Bigl[\int f(p\mid\mathcal D)\,p(\omega)\,dp\Bigr]\log(\cdot)
=\sum_\omega\bar p(\omega)\log\bigl(1+k\,r(\omega)^\top a^\star(\bar p)\bigr).
$$
The posterior has **completely integrated out** — only $\bar p$ survives. And that surviving objective is maximised at $k=1$, because $a^\star(\bar p)$ already maximises $\sum_\omega\bar p(\omega)\log(1+r^\top a)$ over all feasible scalings. Therefore:

> **The earlier draft's shrinkage optimiser returns $k^\star=1$ for every posterior, of any width. It provides no protection whatsoever.** Its "Worked Example 2" — claiming $k^\star\approx0.95$ under a tight posterior and $k^\star\approx0.15$ under a wide one — contradicts its own stated objective. §7 confirms $k^\star\approx1.0$ in both regimes for that objective.

This is not a tuning detail; it is the same fact Baker–McHale open with: *"Because the utility is linear in $p$, the expected utility is a function only of $\mathbb E(p)$, and so conventional utility maximisation ignores the uncertainty in $p$."* A single-period Bayesian who maximises expected log of next-period wealth **cannot** manufacture shrinkage — the posterior mean is a sufficient statistic. Shrinkage is a statement about the **sampling distribution of a decision that is itself estimated**, which is why the randomness must sit in $a^\star(p^{(s)})$ (objective (U)), not in the evaluation weight (objective (broken)).

Three smaller corrections to the earlier draft, for completeness:

* **Attribution.** The U1.5/BTTS *structural* resolution is Jacot–Mochkovitch (overlapping/non-mutually-exclusive bets), not Long's threshold rule. Long's closed form (2.2) applies only when each stake pays in exactly one state; the draft invoked it for compound markets, where the greedy support selection is not valid.
* **A single global $k$ is an approximation, not "McHale".** Baker–McHale explicitly flag the simultaneous-bet case as *multivariate* — "the sizes of each of a number of bets may be shrunk by differing amounts." A scalar $k$ is a defensible, conservative simplification (and what (U) optimises), but it is not the full theory; the vector/Jacobian version in §5.3 is the faithful generalisation.
* **Two different "hedges" were conflated.** Whelan's negative-EV hedges arise from the *mutual-exclusivity structure under concave utility, with probabilities known*. Baker–McHale shrinkage arises from *not knowing the probabilities*. The earlier draft's conclusion (its point 3) merged them; they are orthogonal and live in different layers (§3 vs §5).

---

## 7. Worked Example B — shrinkage, both objectives, real numbers

Same market as Example A; structural portfolio $a^\star(\bar p)=(0.1115,\,0.0808)$, cash $0.808$. Put a Dirichlet posterior on the three state probabilities centred at $\bar p=(0.40,0.25,0.35)$, at two concentrations:

| posterior | $\operatorname{sd}(p_U)$ | $\operatorname{sd}(p_O)$ | $\operatorname{sd}(p_B)$ |
|---|---|---|---|
| **tight** (low variance) | 0.017 | 0.015 | 0.017 |
| **wide** (high variance) | 0.063 | 0.054 | 0.062 |

Computing both objectives over $S=4000$ draws (script: `_verify_notes.py`, run under `uv run`):

| posterior | **(broken)** $k^\star$ (frozen stakes) | **(U)** $k^\star$ (corrected) | decision draws clipped to zero |
|---|:--:|:--:|:--:|
| tight | **1.00** | **0.88** | 0% |
| wide | **0.99** | **0.44** | 2% |

Reading the table:

* The **(broken)** objective is flat in the posterior width — $k^\star\approx1$ whether the model is confident or hopelessly uncertain. This is the linearity collapse of §6 made concrete: the variance never reaches the optimiser.
* The **corrected** objective (U) behaves exactly as Baker–McHale intend. Under the tight posterior it trims modestly ($k^\star=0.88$); under the wide posterior it halves the book and more ($k^\star=0.44$). The mechanism is visible in the last column: as the posterior widens, more posterior draws fall below the support threshold and the optimal *decision* for those draws is to bet nothing — so $a^\star(p^{(s)})$ becomes wildly dispersed, $\Psi'(1)$ turns sharply negative, and $k^\star$ drops.

Final executed stakes under the wide posterior:
$$k^\star a^\star=0.44\times(0.1115,\,0.0808)=(\mathbf{0.0491},\,\mathbf{0.0355}),\qquad \text{cash }=0.915 .$$
So a model that is genuinely uncertain ends up risking 8.5% of bankroll across the two markets instead of 19.2% — the variance protection the earlier draft *wanted* but its mathematics could not produce.

> This reproduces the earlier draft's intended *qualitative* story ("more posterior variance ⟹ more shrinkage") — but via an objective that actually has that property. The earlier draft got the right slogan attached to the wrong equation.

---

## 8. Properties of the corrected unified model

1. **Covariance awareness (structural layer, §2–3).** Stakes come from the joint concave program (P); mutual exclusivity and overlap are priced through the Hessian (2.1). No isolated-Kelly stacking, and — per Whelan — the joint solution is *more* aggressive than two-outcome Kelly on favoured bets, with negative-EV hedges admitted when they raise log-growth. Uniqueness is guaranteed by the overround + no-lay structure of real books.
2. **Honest variance protection (shrinkage layer, §5).** The shrinkage $k^\star$ from (U) responds to the *true shape* of the MCMC posterior — including skew and support-boundary effects that a Gaussian/Beta surrogate misses — because the randomness is correctly placed in the decision $a^\star(p^{(s)})$, not integrated away. It provably satisfies $k^\star<1$ under any non-degenerate posterior, and $k^\star\to1$ as the posterior concentrates.
3. **Clean separation of concerns.** Structure (known-$p$ allocation) and epistemics (parameter risk) are distinct layers with distinct justifications — not to be merged into one posterior-mean expectation, which (§6) destroys the second.
4. **Caveat emptor (Whelan §7).** Aggression is only rewarded if the edge is real. The same machinery that grows wealth fastest when $\bar p$ beats the book loses fastest when it does not; the shrinkage layer mitigates *estimation* error, not *bias*. If the model has no genuine edge over the closing line, the correct $k$ is 0, and (U) will tend there as the posterior widens to cover the no-edge region.

---

## References

* C. D. Long (2026). *Single-Event Multinomial Full Kelly via Implicit State Positions.* — closed-form threshold rule + greedy support selection for direct state bets.
* B. P. Jacot & P. V. Mochkovitch (2023). *Kelly criterion and fractional Kelly strategy for non-mutually exclusive bets.* JQAS. — the concave program (P), gradient/Hessian (2.1), and the correct framing for overlapping compound markets.
* K. Whelan (2024/2025). *On optimal betting strategies with multiple mutually exclusive outcomes.* Bulletin of Economic Research 77:67–85. — concave-utility theory, uniqueness via overround + no-lay, more-aggressive-than-two-outcome result, negative-EV hedges, and the "caveat emptor" performance analysis.
* R. D. Baker & I. G. McHale (2013). *Optimal betting under parameter uncertainty: improving the Kelly criterion.* Decision Analysis 10(3):189–199. — shrinkage theory, the bias argument behind objective (U), approximation (4.1), and the multivariate generalisation noted in §5.3.

*Computations: `docs/bets_multi/_verify_notes.py` (run `uv run _verify_notes.py`).*
