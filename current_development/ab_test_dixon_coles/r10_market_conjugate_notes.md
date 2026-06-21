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

## 3. Results  *(fill after server run)*

### 3.1 Sanity gate
- [ ] `k = 0` rows reproduce r02 `DixonColes_Market` and r06 `DCMH_*` tearsheet numbers exactly.

### 3.2 PRIMARY — hurdle G(k) on Betfair 1X2 (summed over home/draw/away)

| model | argmax_k | G_1x2 @ argmax | G_1x2 @ k=0 | ΔG | profit @ argmax | k* (conjugate) G |
|-------|----------|----------------|-------------|----|-----------------|------------------|
| DixonColes_Market |  |  |  |  |  |  |
| DCMH_HalfLife_14  |  |  |  |  |  |  |
| DCMH_HalfLife_30  |  |  |  |  |  |  |
| DCMH_HalfLife_60  |  |  |  |  |  |  |
| DCMH_HalfLife_120 |  |  |  |  |  |  |

### 3.3 SECONDARY — LogLoss-vs-market diff vs k
Expect monotone decrease toward the market as `k → 1` (NOT success on its own).

| k | logloss_diff |
|---|--------------|

---

## 4. Verdict  *(fill after server run)*

- Is inference-time market conditioning worth it on a **held-out** line? (Y/N)
- Optimal `k` (per model); **is it contrarian (k < 0)?**
- Does the principled conjugate `k*` land near the growth-optimal `k`, or do they diverge
  (i.e. the variance-collapsing "principled" update over-trusts the line vs. what maximizes G)?
- Recommendation: ship as a post-hoc L1 transform / pick a `k` / or no — and why.
