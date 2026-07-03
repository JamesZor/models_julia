# Unified staking — structural Kelly (P) + Baker–McHale shrinkage (U-MC)

Research stream implementing `docs/bets_multi/unified_kelly_postgrad_notes.md` (audited &
corrected 2026-07-01/02) against real L1 posteriors and Betfair books. Successor to the
per-bet `Signals.BayesianKelly` path; the roadmap slots this as the staking layer after
per-line recentring + coherence projection.

## Files

- `l01_structural_kelly.jl` — solver for (P) (projected gradient on {a≥0, Σa≤cap}, validated
  exactly vs Long's closed form), score-grid masks (1X2/O-U/BTTS), per-draw state probabilities
  from latents, `run_match` (book → a*(p̄) → per-draw a* → k* via U-MC), `settle`.
- `r01_r19_realmatch_example.jl` — first real-data run on the r19 `iso_flat` posterior
  (Ireland), two matches, caps {1.0, 0.2}, settled vs actual scores. Results pasted in-file.

## Findings log

### 2026-07-02 — r01: machinery validated end-to-end; risk lives in calibration, not shrinkage

- Solver ≡ Long closed form on notes Example A (0.1115/0.0808, cash 0.8077). ✔
- **Whelan −EV hedge observed live** (fav match, cap=1: over_15 EV −0.11 staked 0.25 to cushion
  the under ladder). Cap=0.2 **drops** the hedge — hedging is a luxury of a loose budget.
- **Cap and shrinkage are substitute risk controls**: when the cap binds, per-draw decisions
  pile onto the same boundary → no decision dispersion → k*=1. Uncapped k* = 0.95/0.79 (3200-draw
  posterior is tight). The dominant risk controls must be per-line recentring + the cap +
  commission-aware R, not k*. Matches [[staking-research-conclusions]].
- **Found a live near-arb artifact**: Betfair TWA close O/U 3.5 pair summed to q=0.99 (< 1) —
  (P) correctly loaded 88% of bankroll into dutching it (the §2.1 rank story in the wild).
  Commission (2–5%) + non-simultaneous TWA quotes make it fake. **TODO: put commission into R
  and screen complementary pairs with Σq < 1+commission.**
- Model's 1X2 deviations (away 0.179 vs market 0.075) are uncalibrated bias per the r13
  per-line verdict — the engine amplifies whatever bias the model carries. Recentring is a
  hard prerequisite (notes §8.4: shrinkage mitigates *estimation* error, not *bias*).
- Both example books settled profitably (W 1.19/1.08 uncapped; 1.06 capped) — anecdote, n=2.

### 2026-07-02 — r02: growth backtest, 275 matches — CURATION IS THE HINGE; curated unified WINS

`r02_growth_backtest.jl` (threaded, 16 cores, ~4 min): sequential compounding in date order,
same model p (iso_flat grids), same commission-adjusted odds d_eff (c=0.02 in decisions AND
settlement), same settlement for all strategies.

**Full book (1X2+O/U+BTTS) — everything Kelly-sized LOSES; flat stake survives:**

| strategy | terminal W | G/match | maxDD | n_bets | turnover/m |
|---|---|---|---|---|---|
| U_cap100 | 0.066 | −0.0099 | 0.996 | 1484 | 0.33 |
| U_cap02 | 0.119 | −0.0078 | 0.992 | 1093 | 0.15 |
| PB_full (per-bet full Kelly) | **0.000** | −0.120 | 1.00 | 796 | 0.29 |
| PB_cap02 | 0.573 | −0.0020 | 0.95 | 796 | 0.14 |
| FIX_1pct | **1.255** | +0.0008 | 0.31 | 796 | 0.03 |

(c=0: same ordering; PB_cap02 0.87, FIX 1.50.) PB_full bankrupt = the
[[portfolio-kelly-partial-hedge]] prediction on real books. The stream has positive edge
(flat stake profits) but Kelly sizes by the MAGNITUDE of model−market divergence, which is
dominated by uncalibrated 1X2 bias → sizing adversely selects the model's own biases
(notes §8.4: shrinkage fixes estimation error, NOT bias — k* can't save you here).

**Curated book (totals+BTTS only, the certified-edge families) — table INVERTS:**

| strategy | terminal W | G/match | maxDD |
|---|---|---|---|
| U_cap02_tot | **4.569** | +0.0055 | 0.77 |
| PB_cap02_tot | 3.563 | +0.0046 | 0.66 |
| FIX_1pct_tot | 1.354 | +0.0011 | 0.26 |

Once the bias family is removed, the structural layer genuinely earns: unified beats per-bet
by ~+28% terminal at the same cap (more selections: hedges + small edges the min_edge filter
discards). Caveats: one league, one model, n=275, maxDD 77% (needs lower cap / fractional
overlay for deployment); totals profitability = market-fade harvesting per
[[totals-compression-is-denoising]].

**Verdict:** the unified machinery adds value ONLY downstream of calibration/curation.
Priority order confirmed: (1) per-line recentring (would recover 1X2 as hedge inventory),
(2) cap sweep + fractional overlay on the curated unified engine, (3) repeat on li_smile50
posterior (the keeper cell for totals/BTTS) — iso_flat wasn't even the right model for this
book, so 4.57 is plausibly a floor.

## Next (r03+)

1. **Per-line recentring** before p enters (P) (split_market_pillar Gap 2 / l10-r21).
2. **Repeat r02-curated on li_smile50** (the totals/BTTS keeper) + cap sweep {0.05, 0.1, 0.2}
   + fractional overlay; report Sharpe/Calmar alongside G.
3. Optional: CorrectScore masks (thin/illiquid — screen by liquidity first).
