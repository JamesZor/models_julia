# RESEARCH.md — literature grounding for the Scottish in-play layer (WP0)

Deep-research pass 2026-07-14 (5 grounded workers + independent verification).
Full worker reports: `~/.antigravity-jobs/batch-20260714-121344-93aee48b/sub_0*/subreport.md`.

## Executive summary

- **NHPP with covariate multipliers is the canonical design** — Dixon & Robinson (1998)
  onward model in-play goals as a bivariate non-homogeneous birth/Poisson process with
  time- and score-dependent intensity; our observable-covariate regression is squarely
  in this tradition (and our own Ireland closed-file already ruled out latent-state filters).
- **Red-card prior anchor, verified:** Vecer, Kopriva & Ichiba (2009, JQAS) estimate
  *market-implied* effects from Betfair in-play prices: offender's rate ×2/3, opponent's
  rate **×1.2**. The ×1.2 matches our Ireland outcome-fit `man_adv=+0.18` log-rate almost
  exactly. Outcome-fitted academic estimates (Titman et al. 2015) are much larger
  (+69–84% for the advantaged team) — treat magnitude as league-dependent; estimate our
  own with a weakly-informative prior centred near log(1.2).
- **Compose-posteriors is legitimate**: it is the "cut / modular Bayes" two-stage
  posterior (Plummer 2015; Jacob et al. 2017 arXiv:1708.08719). Draw-pairing (not
  point-plug-in) is the correct uncertainty propagation; the known pitfalls
  (double-counting, feedback, uncongeniality) are all avoidable in our setup because
  the pregame and in-play modules see different data (full-match counts vs within-match
  timing given pregame rate as offset).
- **The rebalancer maths is settled convex optimisation**: log-growth − ℓ1 crossing cost
  gives the no-trade region via the subgradient at zero (discrete-time Davis–Norman);
  the Busseti–Ryu–Boyd drawdown constraint E[(1+rᵀf)^(−λ)] ≤ 1 with λ = log β / log α
  is convex and composes with the ℓ1 term in one DCP program — no new theory needed.

## 1. NHPP in-play intensity models (sub_00)

- Dixon & Robinson (1998), *A birth process model for association football matches*,
  JRSS-D 47(3): two interacting NHPPs; intensity depends on current score and time;
  scoring rates rise over the match. https://academic.oup.com/jrsssd/article-abstract/47/3/523/7091490
  (Exact parameter tables paywalled — worker could not extract; our Ireland estimates
  serve as the operative magnitudes: trailing +0.25, leading −0.24 log-rate,
  log-pregame slope ≈1.26, rate rising ~16% KO→90, post-HT spike +0.30, late surge +0.24.)
- Titman, Costain, Ridall & Gregory (2015), *Joint modelling of goals and bookings*,
  JRSS-A 178(3), doi:10.1111/rssa.12108 — joint goal/card counting process;
  red card significantly raises the non-penalised team's intensity; yellows ≈ nil.
- Zou, Song & Shi (2020) Bayesian in-play point-process updating exists but over-reacts
  (our concept map's own Zou critique: a goalless first half halving a team's rate is a
  factor-10 over-reaction vs the honest −5% update). Design: covariates only, no
  latent in-play learning.

**Design implication (WP2):** keep the Ireland l08 form
`log λ_i(t) = log λ_i^pregame + δ_time[bin] + β·state_i(t)`; δ_time hierarchical bins;
Scottish data decides the magnitudes. Terminal-slice exposure must absorb stoppage
(SofaScore clamps stoppage goals to minute 45/90 in this league's feed — r00 finding).

## 2. Red-card / game-state multipliers (sub_01)

| Source | Method | Offender | Opponent |
|---|---|---|---|
| Vecer, Kopriva & Ichiba 2009 (JQAS) — **verified** (WC2006/Euro2008 Betfair) | market-implied | ×0.67 | ×1.20 |
| Titman et al. 2015 (EPL/Championship) — plausible, single source | outcome ML fit | home red: ×0.83 / away red: ×0.58 | ×1.84 / ×1.69 |
| Ridder, Cramer & Hopstaken 1994 (JASA, *Down to Ten*) — magnitudes unverified | outcome fit | ≈×0.53 | reported +124% (suspiciously large; do not use as prior) |
| Our Ireland stream (l02/l03 GLM+Turing) | outcome fit | — | man_adv +0.18 log ≈ ×1.20 |

Sources: https://www.semanticscholar.org/paper/1c67b3e3969720cfd96ced9d579db287313201e3 ,
https://scispace.com/pdf/on-probabilistic-excitement-of-sports-games-58wbibtot6.pdf ,
https://rss.onlinelibrary.wiley.com/doi/full/10.1111/rssa.12108 ,
https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476846

**Design implication:** prior for `β_man` ~ Normal(log 1.2, 0.3) (covers Vecer AND
leaves room toward Titman); priors for trailing/leading ~ Normal(±0.25, 0.2) from Ireland.

## 3. Composing posteriors = cut / modular Bayes (sub_02)

- Plummer (2015) *Cuts in Bayesian graphical models*, Stat. Comput., doi:10.1007/s11222-014-9503-z —
  the cut operator = one-way valve: module 2 sees module 1's posterior, never feeds back.
- Jacob, Murray, Holmes & Robert (2017) *Better together?*, arXiv:1708.08719 — when a
  module is (or may be) misspecified, cutting beats the joint fit; point-plug-in is the
  pitfall (overconfident); the correct propagation is exactly our **draw pairing**:
  for each draw θ⁽ˢ⁾ from stage 1, sample/carry stage-2 parameters conditional on θ⁽ˢ⁾.
- Carmona & Nicholls (2020) *Semi-Modular Inference*, arXiv:2003.06804 — continuous
  dial between cut and full Bayes if we ever want partial feedback.
- Pitfalls mapped to us: **double-counting** — avoided (pregame module: full-match
  counts across seasons; in-play module: within-match timing/state *conditional on*
  pregame λ as fixed offset per draw; the timing decomposition p(times|N)·p(N) makes
  these near-orthogonal). **Feedback** — none by construction (we never let in-play
  slices re-estimate team strength). **Uncongeniality** (Meng 1994) — the stage-2 model
  must not assume a different marginal total than stage 1 implies; enforced by the WP3
  t=0 consistency gate (composed P_0 must reproduce pregame prices).

**Design implication (WP3):** pair each pregame draw with one multiplier draw
(independent products, S×S thinning unnecessary); report PPD quantiles per line.
The multiplier posterior is *global* (fit once, offline), so match-time cost is a
matrix product, no MCMC.

## 4. No-trade region / log-optimal rebalancing with costs (sub_03)

- Davis & Norman (1990), *Portfolio selection with transaction costs*, Math. OR 15(4):
  proportional costs ⇒ wedge-shaped no-trade region; trade only at the boundary.
- Discrete-time Kelly version: maximise E[log(growth)] − c·‖Δa‖₁. The ℓ1 subdifferential
  at Δa=0 is [−c,c]ⁿ, so no trade until the growth gradient exceeds the crossing cost —
  the no-trade region falls out of the KKT conditions, no rule needed (matches
  concept-map Concept 7 exactly).
- Sports-specific: Axén & Cortis (2020), *Hedging on betting markets* (Risks 8(3)) —
  in-play hedging as variance reduction / profit locking; bookmaker cash-out carries a
  second margin, exchange self-hedging is the fair-value route. Confirms building our
  own π(ω)-based hedger rather than using operator cash-out.

**Design implication (WP4):** solve
`max_Δa (1/S)Σ_s Σ_ω P_t^s(ω) log(W₀ + π(ω) + Σ_k Δa_k r_k(ω)) − c‖Δa‖₁`
per state change; π(ω) (payoff-by-outcome vector) is the state variable; average over
posterior draws **inside** the log (concept-map Concept 9).

## 5. Risk-constrained Kelly (sub_04) — verified against arXiv:1603.06183

- Busseti, Ryu & Boyd (2016): maximise E[log(rᵀb)] s.t. P(W_min < α) < β, made convex
  via the bound **E[(rᵀb)^(−λ)] ≤ 1 with λ = log β / log α** (both logs negative, λ>0).
- λ→∞ forces rᵀb ≥ 1 a.s. (risk-free only); λ→0 recovers unconstrained Kelly;
  moderate λ ≈ bounding variance-to-growth (quasi-Markowitz).
- Combining with the ℓ1 cost term is plain DCP composition (Boyd's multi-period
  trading line of work does exactly this) — engineering, not research.

**Design implication (WP4/later):** replace the ad-hoc Σ≤0.2 cap + fractional Kelly
with one λ from a chosen (α, β), e.g. β=0.05 of losing half the bank ⇒
λ = log 0.05 / log 0.5 ≈ 4.32. Convex constraint added to the same program.

## Unverified / open

- Exact Dixon–Robinson and Titman parameter tables (paywalled); Titman percentages and
  the RCH94 "+124%" are single-source — do **not** hard-code as priors, only as ranges.
- The dead link in sub_01 (`stat.berkeley.edu/~aldous/.../vec_ich_lau.pdf` 404s) was
  replaced by the Semantic Scholar / SciSpace copies above.
- Zou over-reaction criticism (arXiv:2106.12466 as cited by the worker) not independently
  read; our own concept-map derivation of the factor-10 over-reaction stands on its own.
- Whether the market-implied (small) vs outcome-fitted (large) red-card gap is a
  lower-league feature or an estimation artifact — our Scottish fit will add one data point.
