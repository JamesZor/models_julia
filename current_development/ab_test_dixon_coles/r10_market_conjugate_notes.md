# r10 — Inference-time market conditioning of L1 goal-rate posteriors

**Question.** Does folding the target match's own market line into Layer-1 inference —
a per-match conjugate update on the latent log goal-rate, applied *after*
`extract_parameters` — improve **betting growth G / ROI** (not just LogLoss)? And is the
optimal shrinkage `k` **contrarian** (k < 0)? This lives entirely in L1; the L3
meta-model is deprecated.

Files: `l10_market_conjugate.jl` (loader), `r10_market_conjugate.jl` (runner).

---

## 1. Mechanism (why it's conjugate)

`extract_parameters` returns, per match, posterior **draw vectors** of the log goal-rate
`θ_1 = log λ_home`, `θ_2 = log λ_away` (the Dixon–Coles score matrix reads `θ_1/θ_2`
directly — `src/predictions/score_computation/dixoncoles.jl`). Inference is market-free;
the market only enters the **training** likelihood as a Gaussian pillar in log-λ:

> Pillar C:  `log λ_s + log κ_s  ~  Normal(market_log_λ_s, σ_market)`

Because that pillar is Gaussian in log-λ, conditioning the posterior on the line is a
conjugate (location/precision) update we apply **post-hoc to fitted chains** — no
resampling — so one model is re-scored under many `k`.

For each match, side `s ∈ {home, away}`, draws `ℓ_j = θ_{s,j}` (j = 1..N):
`μ_mod = mean_j ℓ_j`, `τ²_mod = var_j ℓ_j`, anchor `m_s = log(flat_market_λ_s)`.

**Transform A — location shift (variance-preserving; PRIMARY sweep):**
```
ℓ'_j = ℓ_j + k·(m_s − μ_mod)
```
Slides the whole posterior toward the line by fraction `k`, keeping spread (preserves
Kelly uncertainty; matches the L2 "shift the whole distribution" philosophy).

**Transform B — full conjugate (variance-shrinking; SECONDARY single point):**
```
σ²       = mean(chain[:σ_market])²
k*       = (1/σ²) / (1/τ²_mod + 1/σ²)
var_post = 1 / (1/τ²_mod + 1/σ²)
μ_post   = μ_mod + k*·(m_s − μ_mod)
ℓ'_j     = μ_post + sqrt(var_post / τ²_mod)·(ℓ_j − μ_mod)
```

Then `λ'_{s,j} = exp(ℓ'_j)`; `ρ` and `true_xg_*` untouched.
Sweep `k ∈ {−0.5, −0.25, −0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.2}`.

**Important framing.** The base models (`DixonColes_Market`, 4× `DCMH_HalfLife_*`)
already carry `market_weight = 0.4` in training. So swept `k` measures **additional**
inference-time conditioning *on top of* that. `k < 0` ⇒ fade the line further;
`k = 0` ⇒ the existing r02/r06 baseline (sanity gate — must reproduce exactly).

---

## 2. Guardrail (validity)

- **Condition on** `m` = SofaScore-derived `flat_market_λ` (1X2 + O/U + BTTS inversion,
  `src/features/market_inverse_utils.jl`), built from `ds`.
- **PRIMARY evaluation = Betfair 1X2/DC** via `ds1` (`summarize_betfair_market`,
  `odds_close`) — a **different source** + de-vigged exchange close ⇒ genuinely held-out
  line. `process_signals` inner-joins on `[match_id, market_name, selection]`, so feeding
  Betfair odds automatically restricts bets to home/draw/away (+ DC). (cf. memory
  *betfair-vs-bet365-market-anchor*: anchor to the bookmaker line, execute/evaluate on Betfair.)
- **SECONDARY / flagged = SofaScore O/U + BTTS** — same source we conditioned on ⇒
  partly circular. Reported but **not** the verdict.

---

## 3. Results

Ireland, 281 target matches, walk-forward (27 folds for DC_Market). Full sweep in
`r10_market_conjugate_results.csv`. `G` = parametric hurdle growth, `Gemp` = empirical
growth, both **summed over home/draw/away** on the held-out Betfair line.

> **Loading caveat.** The saved r02/r06 experiments were serialized with the *prototype*
> engine struct, whose type-parameter & field order differs from the graduated `src/`
> struct, so JLD2 can't reconstruct `config.model`. We shimmed `JLD2.rconvert` to rebuild
> the model via its `@kwdef` keyword constructor (order-independent). The fitted **chains**
> load unchanged, so "no resampling" holds. (Shim lives only in the runner session.)

### 3.1 PRIMARY — hurdle G(k) on the held-out Betfair 1X2 line

| model (half-life) | G @ k=−0.25 | G @ k=0 | G @ k=0.5 | G @ k=0.7 | best active k | G_emp turns ≥0 at |
|---|---|---|---|---|---|---|
| DixonColes_Market (60) | −0.0646 | −0.0445 | −0.0110 | **−0.0038** | 0.7 | k≈0.4 |
| DCMH_HalfLife_14 | −0.0449 | −0.0261 | −0.0027 | **−0.0023** | 0.7 | k≈0.3 |
| DCMH_HalfLife_30 | −0.0611 | −0.0418 | −0.0007 | **+0.0021** | 0.7 | k≈0.2 |
| DCMH_HalfLife_60 | −0.0656 | −0.0378 | −0.0008 | **+0.0026** | 0.7 | k≈0.3 |
| DCMH_HalfLife_120 | −0.0341 | **−0.0093** | −0.0177 | −0.0211 | 0.0 | never (best −0.027 @ k=0) |

**Within the active-betting regime (k ≤ 0.7, ~160–300 bets), G rises monotonically as k
moves from −0.5 toward the market** for the short/mid half-lives (14/30/60), turning
slightly positive at k≈0.5–0.7. The longest half-life (120) is already maximally smoothed,
so extra conditioning *hurts* — its optimum is k=0.

**Contrarian k<0 is uniformly the worst on G** (e.g. DC_Market k=−0.5 → G=−0.097), even
though it places the *most* bets and the *highest gross profit* (k=−0.5 → profit 7.1 vs
2.7 at k=0.5). Risk-adjusted growth penalises the extra-variance contrarian bets.

**k≥1.0 is an artifact, not a result.** As k→1 the posterior collapses onto the line, the
Kelly edge `p_model−p_market`→0, and activity craters to ~30 bets. The "positive G" there
(e.g. DCMH_60 k=1.0 G=+0.027 on 28 bets) is *declining to bet*, not an edge — excluded.

### 3.2 Conjugate (Transform B) vs the G-optimum

The principled per-match conjugate point (σ² from each model's own `:σ_market`) lands at
G ≈ −0.008 to −0.027 — roughly equivalent to a **k≈0.1–0.3** shift. It **under-shrinks**
relative to the active G-optimum (k≈0.5–0.7) for the short/mid models, and **over-shrinks**
for HL120. So the variance-collapsing "principled" update is in the right *direction* but
does not hit the growth-optimal amount.

### 3.3 SECONDARY — LogLoss
Not separately tabulated: by construction the model probabilities → market probabilities
as k→1, so 1X2 LogLoss decreases monotonically toward the market line. As flagged, that is
*not* success — it is the model agreeing with a better-calibrated line, which is exactly
why the Kelly edge and bet count vanish.

---

## 4. Verdict

- **Worth it on a held-out line? Marginally, and not as an edge.** Inference-time market
  conditioning improves *risk-adjusted* growth G only by pushing **toward** the line
  (k≈0.5–0.7 for short/mid half-lives), which mechanically means **betting less and pruning
  the model's noisy 1X2 disagreements**. It does not create new profit; gross profit falls.
- **Is optimal k contrarian? No.** k<0 maximises turnover and gross ROI but *minimises* G.
  The growth-optimal k is positive (defer to the market) — the opposite of the hoped-for
  contrarian result. This corroborates [[totals-compression-is-denoising]] and
  [[staking-research-conclusions]]: the L1 engine has little exploitable 1X2 edge beyond
  the SofaScore line in this minor league.
- **Conjugate k\*** is directionally right but doesn't reach the G-optimum (under-shrinks
  for short/mid half-lives, over-shrinks for HL120).
- **Recommendation: do not ship a market k-update as an edge mechanism.** The only honest
  use of k>0 is *selective abstention* (bet fewer, more market-aligned 1X2 positions),
  which a higher `min_edge` achieves more transparently. Note this whole test is on 1X2
  where the base already carries `market_weight=0.4`; the conclusion may differ for markets
  where the model has genuine structural edge (e.g. totals dispersion / O/U fade per
  [[totals-compression-is-denoising]]) — but that must be judged on a non-circular line,
  not the SofaScore O/U we conditioned on.
