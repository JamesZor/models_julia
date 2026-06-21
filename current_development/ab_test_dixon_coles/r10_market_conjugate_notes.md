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
- **TOTALS evaluation = Betfair O/U + BTTS** via `ds1` (§3.4). After the betdb gained these
  Betfair markets for Ireland mid-study, O/U+BTTS are now priced on the exchange — a
  *different* source from the SofaScore line we condition on, so this is **also held-out**,
  not circular. (Before the DB update, O/U+BTTS were SofaScore-only ⇒ would have been
  circular; that constraint no longer applies — [[betdb-data-coverage]].)

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

### 3.4 TOTALS + BTTS on a held-out Betfair line — the real edge

The betdb was updated mid-study to price **Betfair O/U + BTTS + CorrectScore** for Ireland
(previously 1X2+DC only). This removes the circularity worry: we condition on the
**SofaScore** line and evaluate O/U+BTTS on **Betfair** odds — a genuinely different,
held-out line (anchor-SofaScore / execute-Betfair, per [[betfair-vs-bet365-market-anchor]]).
Re-ran the full sweep against the updated `ds1`. Basket = over/under {1.5,2.5,3.5} + BTTS
yes/no. Full sweep in `r10_market_conjugate_totals_results.csv`. (`Gemp` = realized
empirical growth — the trustworthy number; `G` = parametric hurdle fit, optimistic about
fade-bet tails.)

| model | metric | k=−0.25 | k=−0.1 | k=0 | k=0.2 | k=0.4 | k=0.7 | k=1.0 |
|---|---|---|---|---|---|---|---|---|
| **DixonColes_Market** | Gemp | 0.055 | **0.058** | 0.047 | 0.046 | 0.051 | −0.016 | −0.057 |
|  | G | 0.131 | 0.108 | 0.084 | 0.074 | 0.064 | −0.021 | −0.139 |
|  | profit/bets | 10.8/645 | 8.3/616 | 7.0/596 | 5.5/537 | 4.6/461 | 3.5/207 | 4.3/53 |
| DCMH_HalfLife_14 | Gemp | 0.003 | 0.016 | 0.025 | 0.033 | 0.030 | 0.049 | −0.081 |
| DCMH_HalfLife_30 | Gemp | 0.025 | 0.033 | 0.036 | **0.042** | 0.036 | 0.038 | 0.007 |
| DCMH_HalfLife_60 | Gemp | 0.017 | 0.033 | 0.035 | **0.051** | 0.048 | −0.035 | −0.108 |
| DCMH_HalfLife_120 | Gemp | −0.101 | −0.017 | −0.002 | −0.008 | −0.073 | −0.221 | −0.282 |

**Unlike 1X2 (G negative everywhere), the totals/BTTS basket is solidly positive growth
(~+0.04 to +0.06 realized) on 500–700 bets** — the model's edge lives here. Direction is
model-dependent:
- **DixonColes_Market** (standard r02 model) is **flat-to-contrarian**: realized growth is a
  broad plateau over k≈−0.25…+0.4 (Gemp ~0.05), peaking *mildly contrarian* at **k≈−0.1**
  (Gemp 0.058). Parametric G is strongly contrarian (0.17 @ −0.5) but that's optimistic —
  realized growth collapses at k=−0.5 (Gemp −0.034). Honest read: **fading the line mildly
  helps and never hurts up to ~k=0.4.**
- **DCMH time-decay models** peak *slightly toward* the market (**k≈+0.2**), not contrarian.
  HL120 is fragile (≈breakeven, blows up if faded: Gemp −0.40 @ k=−0.5).

**Where the edge is** (DixonColes_Market, per selection @ k=0): the edge concentrates in
**btts_yes** (122 bets, ROI 29%), **btts_no** (ROI 52%), and **under_15** (ROI 64%); it is
~0 on `over_25` and **negative on `over_15`/`over_35`**. Fading (k<0) amplifies the winning
BTTS/under bets *and* the losing overs — so a curated book (BTTS + unders, skip high overs)
would beat the basket aggregate. The model systematically reads games as lower/less-dispersed
than the line — the "fade market over-dispersion" thesis of [[totals-compression-is-denoising]].

---

## 4. Verdict

The answer splits by market:

- **1X2 — no edge.** Inference-time conditioning only improves risk-adjusted growth by
  pushing **toward** the line (k>0 ≈ selective abstention: bet less, defer to a
  better-calibrated line). Contrarian k<0 maximises turnover/gross profit but *minimises* G.
  Don't ship a k-update here; a higher `min_edge` expresses the same abstention transparently.

- **Totals + BTTS — real edge, evaluated on a held-out Betfair line.** The basket runs
  **~+0.04 to +0.06 realized growth on 500–700 bets**, concentrated in **BTTS and the
  unders** (the model fades the market's over-/dispersion bias). This is the model's actual
  money market, not 1X2.

- **Is optimal k contrarian?** *Market-dependent.* For the standard **DixonColes_Market**,
  growth on totals/BTTS is a **flat-to-contrarian plateau** (realized peak k≈−0.1; fading
  helps mildly, never hurts up to ~k=0.4) — opposite to 1X2 and consistent with
  [[totals-compression-is-denoising]]. For the **DCMH time-decay** models the totals optimum
  is *slightly toward* the market (k≈+0.2); HL120 is fragile. So the contrarian tilt is
  **second-order**; the first-order win is that the totals/BTTS edge exists and the market
  conditioning must **not wash it out** (keep k small, |k|≲0.3).

- **Conjugate k\*** sits in the safe small-|k| region (Gemp ≈ 0.04–0.05 on totals) but is
  not the growth-optimum for any single market.

- **Recommendation.** Don't ship the k-update as an *edge* mechanism. Best operational use:
  on **totals/BTTS**, keep conditioning light (k≈0, optionally a mild contrarian k≈−0.1 for
  DixonColes_Market) and curate selections (BTTS + unders, skip high overs); on **1X2**, lean
  on `min_edge` rather than k. Next step: a curated BTTS+unders strategy benchmarked vs
  Betfair close, and check whether the contrarian-totals tilt is specific to the stiffer
  Global model vs the hierarchical time-decay variants.

> **Data note:** the betdb gained Betfair O/U+BTTS+CorrectScore for Ireland mid-study, which
> is what made the §3.4 totals evaluation a genuinely held-out (non-circular) test. Earlier
> Ireland Betfair was 1X2+DC only ([[betdb-data-coverage]]).
