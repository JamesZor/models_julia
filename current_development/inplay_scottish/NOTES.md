# inplay_scottish — running log

In-play layer for Scottish lower leagues (56 = League One, 57 = League Two).
Plan: `~/.claude/plans/i-m-looking-at-the-shimmying-charm.md` (session 2026-07-14).

## Purpose (locked)

Fair-value in-play score matrix **P_t** for **position management of the pregame book**
(π(ω) payoff-vector rebalancer + validated exit signal), NOT in-play value betting —
the concept map closed Scottish in-play trading ("Scotland is a pre-game market").
Thin Betfair LTP is enough: we need fair value, not fills.

## Design decisions (locked with user)

1. **Bayesian form: compose posteriors.** Pregame λ posterior draws (from the Scottish
   grid winner) × NHPP multiplier posterior draws → per-draw remaining-score matrix →
   full PPD per market line. No per-match MCMC at match time. (Modular/"cut" Bayes —
   see RESEARCH.md.)
2. **Training target: outcomes via NHPP** on incident goal times (market-free).
   Betfair LTP is *evaluation only* (thin: median ~49 in-play prints MATCH_ODDS,
   ~26 OU2.5; inversion unidentified in >50% of bins per the liquidity audit).
3. **Model form settled by the Ireland stream** (`../match_inplay_explore/`, do not
   relitigate): observable-covariate regression (no latent state/filters), **global**
   (team hierarchies hurt OOS), **linear game state**, **NHPP δ_time** bins (fixes the
   late-game 3.4σ bias; post-HT spike +0.30, late surge +0.24).

## Data facts (betdb, 2026-07-14)

| | t=56 | t=57 |
|---|---|---|
| finished matches | 985 | 985 |
| with incidents | 702 — **holes: season 52605 (71/180), 77129 (16/175)** | 985 complete |
| betfair LTP | ~890 matches, 9-market ladder | **only 140** |
| bet365 closing (sofascore) | complete incl. 11-line goals ladder | same |

⇒ train on incidents (57 full + 56's 4 good seasons); evaluate vs Betfair on 56.
56 incident holes flagged to user for a re-scrape.

## Infrastructure

- **homelab archpc** (16 GB, 8c/16t) via kaimon-remote: EDA + light checks ONLY.
- **mcmc-beast** (128 GB, 16c/32t): artifacts + heavy sampling later. Currently running
  the pregame smile grid — do not disturb.
- Sync: git push (laptop) → pull (server). Artifacts move by scp over tailscale.
- Pregame winner artifact: `data/scottish_decay_grid/none_pois_hl365_hs2_20260712_212831`
  (Grid A winner hl365_hs2, 164M) — scp'd from mcmc-beast. Grid B (smile) winner: pending.

## Work packages

- **WP0** RESEARCH.md — deep-research grounding (Vecer multipliers, cut Bayes,
  Davis–Norman, risk-constrained Kelly). IN PROGRESS.
- **WP1** `r00_data_qa.jl` — incidents QA, score-path reconstruction, betfair density,
  clock-map anchoring sanity on thin prints. IN PROGRESS.
- **WP2** `l01_nhpp_scottish.jl` / `r01_nhpp_transfer.jl` — port l08 NHPP + l07 CV
  harness; paired CV pregame-only vs +δ_time vs +state; pooled 56+57 vs separate.
- **WP3** `l02_ppd_compose.jl` / `r02_ppd_calibration.jl` — draw pairing → market-line
  PPDs via `compute_market_probs(MarketOverUnder(L − current_total))`.
  Gates: t=0 reproduces pregame prices; per-bin ECE/Brier; eval vs identifiable
  Betfair bins (l01 filters: ≥6 sels + full 1X2, λ_rem<6, residual<0.06).
- **WP4** `l03_rebalancer.jl` / `r03_rebalance_backtest.jl` — π(ω) state + convex Δa*
  with ℓ1 crossing cost; race: hold vs exit-rule τ=−0.05 vs rebalancer.
  Benchmark to beat: exit-rule uplift +0.306±0.059 (t=5.21).

## Log — BBC in-play MVP stream (newest first)

Plan: `~/.claude/plans/elegant-skipping-pinwheel.md` (session 2026-07-29).
Branch `feat/inplay-bbc-mvp`, off `feat/apm-player-rating-l1` (carries `ds.bbc_events`).
Prototype only — **no `src/` changes**; WP-H is a written graduation sketch.

- 2026-07-29: **WP-B DONE — pregame source is pluggable; GATE B FAILS ON THE LETTER, for one
  fully-diagnosed scalar reason** (`l05_pregame_source.jl`, `r04b_pregame_source.jl`).
  - `AbstractPregameSource` replaces r01/r02's hard-coded `.jls` read. `ExperimentSource`
    goes through `load_experiment` → `extract_oos_predictions`; `LatentsFileSource` keeps
    the legacy file loadable with its shot fields explicitly `nothing` rather than silently
    substituted. `assemble_matches` generalises `l01.assemble_nhpp_matches` along both axes
    the race needs (event source: incidents or BBC seqs + stoppage; pregame source: full
    draw vectors, not the posterior mean).
  - **Funnel latents load clean and the thinning identity holds**: 710 matches × 4,000
    draws; λ_h 1.445 / λ_a 1.299, λ_s_h 9.72 / λ_s_a 8.74, p2 0.1488, and
    9.715 × 0.1488 = 1.445 = λ_h exactly. ~9–10 shots vs ~1.4 goals per team-match is the
    **7× count advantage MVP-1 is betting on**.
  - Coverage confirms the plan's ceiling: latents exist for **24/25 + 25/26 only** (710),
    and under `require_incidents` that is **551 matches** — the same common subset WP-F
    will race on, so Gate B is measured on the race's own sample.
  - **Gate B is NOT inherited and was refit, not rechecked.** The NHPP absorbs the pregame
    level into α, so pairing a multiplier chain trained against one pregame engine with a
    different engine's λ is the uncongeniality failure RESEARCH.md §3 warns about. Both
    arms were refit on 551 matches / 1,503 goals (max R̂ ≤ 1.005).

    | arm | kernel | max price gap | gap after removing level |
    |---|---|---|---|
    | funnel_apm_xg | 0.9856 | **0.0115** | 0.0018 |
    | funnel_winner | 0.9835 | **0.0138** | 0.0018 |

    Pre-registered thresholds were kernel ≈ 0.988 (~1.2%) and max gap < 0.01. **Both arms
    miss both.** Recorded as a FAIL, not rounded into a pass.
  - **The whole failure is one scalar.** `gate_b_decompose` re-prices the pregame reference
    with λ scaled by K: the max gap collapses **0.0138 → 0.0018** with per-selection means
    < 0.0008, i.e. the Monte Carlo floor at `n_pairs = 2000`. So the composed and pregame
    books agree in SHAPE on all 17 selections and differ only by a uniform level rescale.
    The gap is not 17 failures; it is K, twice.
  - **Interpretation: this is the funnel engines being hotter than the decay grid, which the
    NHPP is correctly reporting.** K < 1 means the NHPP, fitted on realised goal times, puts
    total intensity 1.4% (APM) / 1.6% (winner) below the pregame λ — against the old grid's
    1.2%. The signature is diagnostic: every `over_*` biased down, every `under_*` up by the
    same amount, 1X2 barely touched (max 0.003) because a uniform scaling of both sides
    cancels in the ratio. Note this is still far inside the **~5% pregame hotness already
    recorded for this stack**, so nothing new is broken — the funnel arms are simply
    0.2–0.4 pp hotter than the engine the 0.988 threshold was calibrated on.
  - **Do not "fix" this by recentring α to force K = 1.** That would make the in-play book
    inherit the pregame engine's level bias, which is worse fair value, not better. The
    honest options are to carry the offset as a known, measured property or to re-threshold
    the gate against the funnel engines; that is a call for the user, not a silent waiver.
  - **USER DECISION (2026-07-29): accept and carry the offset.** Gate B stands in the log as
    a documented marginal FAIL; K is recorded per pregame arm as a known calibration
    property of the pregame/multiplier pair (funnel_apm_xg 0.9856, funnel_winner 0.9835),
    and downstream work proceeds. Any later composed-price claim must state which K it
    carries.
  - Gotcha banked: a 111-hour-old REPL threw
    `MethodError: no method matching extract_oos_predictions(::DataStore, ::ExperimentResults)`
    with a `@world(...DataStore, 38680:39881)` annotation — Revise had rebound `DataStore`,
    so the method was compiled against a dead world age. Same family as the `const` Union
    trap, but triggered by uptime rather than by an edit. `manage_repl restart` fixes it.
  - The legacy `latents_hl365_hs2.jls` is a homelab artifact and is not on mcmc-beast; the
    source is skipped when absent, since every race arm composes against a funnel engine.

- 2026-07-29: **WP-A extension (user ask) — BBC CLOSES r00's stoppage-clamping problem.**
  r00 flagged that this SofaScore feed clamps every stoppage goal to exactly mm=45/90 with
  added_time=0, so `l01` had to proxy the H2 tail with a flat `Tend = 95` and conflate H1
  stoppage into the first H2 slice. It proposed `injury_time1`/`injury_time2` from
  `sofascore.matches` as the fix.
  - **That route is dead, confirmed not assumed:** both columns exist for all 1,970
    matches and are **zero in 99.6% / 99.4%** of them. The field is present and unpopulated
    for these leagues. BBC is the *only* stoppage source for 56/57.
  - **BBC's `half_end` clock strings are the source** — `"45'+3"`, `"90'+5"` — i.e. stoppage
    actually PLAYED, not announced. **All 2,136 parse**, covering 1,069 of 1,070 matches.
    (BBC also emits `added_time` posts carrying the announced `value`, but only 1,326 of
    them; `half_end` is both more complete and the more relevant quantity.)
  - Measured: **H1 mean 2.29 min** (median 2, 5–95% [1, 4]), **H2 mean 4.84 min**
    (median 5, [3, 7]). So the incumbent's flat 5-minute H2 tail was well *centred*
    (−0.16 min on average) but wrong per match across a 3–7 min spread, and **H1 stoppage —
    2.29 min per match of real exposure — was not modelled at all**.
  - **Two fixes, both in `l04`.** `bin_exposure` / `build_slices_bbc` put stoppage where it
    belongs, in the **offset**: a fixed 18-bin [0, 90] frame where the terminal bin of each
    half carries `Δt + at`, instead of `Tend = 95` smuggling it in as an extra bin. Keeping
    the bin structure fixed is what keeps `δ_time` identifiable across matches. And
    `event_clock` corrects the clock itself: BBC minute labels are **1-indexed** (an event
    labelled minute t happened in elapsed (t−1, t], so it belongs at t − 0.5, not t), and
    H1 stoppage is compressed into (44.5, 45) so it stays in its own half.
  - **The clock bug this exposes is real and measurable: 88 of 2,965 goals (3.0%) change
    HALF** between the incumbent clock and the corrected one. Under `time + added_time` an
    H1 stoppage goal at 45+3 sits at minute 48 — *after* the second-half kickoff labelled
    46 — so it is both ordered wrongly against other events and binned into the first H2
    slice, where it also corrupts the trailing/leading game state of every later slice.
  - Verified: all 2,965 goals bin inside [0, 90) under the new clock (none lost), max clock
    89.93, 38,520 slice rows over 1,070 matches, 19 distinct offsets.
  - `event_clock(...; mode = :incumbent)` reproduces `l01`'s convention exactly. **WP-F
    therefore splits arm 0b in two** — `0b-naive` (BBC events, incumbent clock) isolates
    *coverage*, `0b-clock` (BBC events, corrected clock + real exposure) isolates the
    *clock*. Without that split a 0b win would confound the two, which is the same mistake
    arm 0b exists to prevent for the MVPs.

- 2026-07-29: **WP-A DONE — Gate A PASS on all three blocking checks** (`l04_bbc_timeline.jl`,
  `r04a_bbc_timeline_qa.jl`). 1,070 BBC matches, 38,137 timeline events across 11 event types,
  **0 null minutes**, 0 running-score breaks. The headline is a data-quality *upgrade* over
  what the plan pre-registered:
  - **Goal reconciliation is 99.81% (1068/1070), not ~92%.** The pre-registered 92% is the
    **team-slug** route, which this run reproduces exactly at **93.4%**. Deriving the goal
    side by **differencing BBC's running `home_score`/`away_score`** instead recovers all
    **67 own goals** (every slug-less goal row in 56/57 is an own goal, confirmed from the
    text) and agrees with the slug route on 2,898/2,898 where both resolve. Own goals were
    the predicted cause; the fix is that BBC *does* carry the answer, in the score column.
    Note the `src` fetcher's "do NOT infer the side from the running score" warning is about
    **shots**, where there is no score to difference — it does not apply to goal rows.
  - **The 2 failures, enumerated and classified** — both single-match feed defects in t57,
    neither an own goal: `11395473` (23/24) BBC has 2-1, final says 2-0 → one extra away
    goal; `14035676` (25/26) BBC has 1-0, final says 1-3 → truncated commentary, 3 goals
    missing. 0.19% loss; downstream packages drop these two by the reconciliation gate.
  - **Reds cross-check vs `ds.incidents` (802 both-source matches): PASS with room.**
    136 incident reds / 142 BBC reds, 135 paired → recall 99.3%, precision 95.1%, per-match
    count agreement 99.3%. **Minute MAE 0.36, p90 = 1.0 min, only 1.5% differ by > 2 min**
    (stop threshold was 10%). Subs likewise: 5,528 / 5,654, 5,514 paired, recall 99.7%,
    MAE 0.26, 1.7% > 2 min. Given γ_man = ×1.70 is the dominant in-play effect, this is the
    check that mattered and BBC clears it.
  - **Side attribution is now 100.0% of 38,137 events.** BBC ships `post` (woodwork, 438
    rows = 2.2% of all attempts) with **no team column at all** — the one gap the three-way
    slug CASE cannot close. The club is in the free text (`"… (Dumbarton) hits the bar"`);
    slug-normalised matching recovers 412 and a leading-token fallback the last 26
    (`Inverness CT` → `inverness-caledonian-thistle`). Zero 56/57 fixtures have both teams
    sharing a leading token, so the fallback cannot mis-assign.
  - **BBC's clock is strictly better in stoppage.** BBC carries a real `added_time` (1–8)
    where this SofaScore feed clamps every stoppage goal to exactly mm=45/90 with
    added_time=0 (the r00 finding). `l04` therefore exposes both `t` (stoppage-inclusive,
    for the slicer) and `tb` (base, the only clock comparable to `ds.incidents`) — the red
    MAE above is computed on `tb`, or a pure convention difference would have scored as a
    data disagreement.
  - Deliberately NOT done: widening `src/Data/fetchers/sql/bbc_events.jl`. `build_shots(ds)`
    reads `ds.bbc_events` wholesale and the APM port is verified bit-faithful (ρ = 1.000000);
    the timeline is built by direct `LibPQ` query in the prototype instead. See WP-H.
  - Gotcha banked: `d < bd && (best, bd = j, d)` parses as a **tuple expression**, not a
    tuple assignment — it paired 0 events out of 5,650 and raised nothing. Use an explicit
    `if` block.

## Log

- 2026-07-14: stream created. Deep-research batch dispatched. r00 written.
- 2026-07-14: **r00 ran on homelab kaimon** (fresh datastore fetched from betdb, cache
  rebuilt). Verdict:
  - Training seasons: 57 all six (20/21–25/26); 56 four (20/21, 21/22, 22/23, 24/25);
    56's 23/24 (71/180) + 25/26 (16/175) excluded pending incident re-scrape.
  - Score-path reconstruction exact for 98.9% (56) / 99.4% (57) of incident matches.
  - **Stoppage clamping**: ALL stoppage goals sit at exactly mm=45 (129) or mm=90 (302)
    with added_time=0 → terminal NHPP slices need extended exposure (injury_time1/2
    exist in sofascore.matches but are NOT fetched into the DataStore; use league-mean
    stoppage or extend the fetcher).
  - Betfair (eval only): median 49 1X2 prints/match, in-match gap p50 ≈ 1.05 min,
    p90 ≈ 4.1–4.6 min; **identifiable full-1X2 5-min bins only 23% (56) / 18% (57)**
    (4-min LOCF + ≥6 selections) — harsher than the concept-map audit; confirms
    eval-only role. Goal-jump anchoring finds a price jump for 81%/76% of goals;
    off1≈3.0±1.5, off2≈18.0±2.9 (medians sit on the priors → fallback-heavy on thin
    prints; acceptable for eval binning).
  - Red cards: 305 matches (18% of incident matches), 327 total — enough to fit β_man.
- 2026-07-14: **WP0 done** → RESEARCH.md. Key: Vecer/Kopriva/Ichiba 2009 market-implied
  red-card opponent multiplier ×1.2 ≈ our Ireland man_adv fit (+0.18 log) — use as
  prior centre; compose-posteriors = cut Bayes (draw pairing, Jacob et al. 2017);
  BRB λ = log β/log α composes with ℓ1 costs in one DCP program.
- 2026-07-14: **WP2 done (r01 on homelab, 7m39s, all R̂ ≤ 1.008).** 715 matches /
  27,170 slices / 1,934 goals / 1,046 red-affected rows. **Scotland ≠ Ireland:**
  - GLM race: time>pg_only t=6.29; **state−time t=0.51 (NULL — Ireland was 3.15)**;
    +man_adv t=8.85 (dominant).
  - Posterior: α=−4.584 (net level 0.99 after shape — see r02 gate), β=0.163 (rates
    rise), γ_tr=+0.02±0.06 (nil), **γ_ld=+0.09 (sign FLIPPED vs Ireland −0.24)**,
    **γ_man=0.53 ⇒ ×1.70** (Titman-scale, crushed the ×1.2 Vecer prior — red cards
    are THE in-play repricing event in these leagues), σ_time=0.08 (first-5-min dip
    −0.12, post-HT bump +0.09).
  - Late-checkpoint remaining-goals bias: +0.020/+0.014/+0.003 (60'/75'/85') — clean.
  - Open question (r01b candidate): γ_ld>0 may be a frailty/selection artifact
    (teams outperforming their pregame λ both lead AND keep scoring).
- 2026-07-14: **WP3 gates i+ii PASSED (r02, 22s).** t=0 kernel = 0.988 (composed
  kickoff intensity within 1.2% of pregame λ; consistent with the known ~5% pregame
  hot-ness); max composed-vs-pregame price gap < 0.01 across all 17 selections.
  Outcome calibration: Brier tightens 0.14→0.05 (OU) / 0.25→0.07 (BTTS) from KO to
  80'; **in-play conditioning beats frozen-pregame at every checkpoint** (BTTS Brier
  0.246→0.074 at 80'). Per-selection biases within ~1 SE (n=300). Outstanding:
  gate iii (Betfair identifiable-bin comparison on 56 24/25) = r02b.
- 2026-07-14: **r01b (user Q: hierarchical team δ_lead?) = NULL, race CONFIRMS.**
  Non-centred team slopes on leading/trailing: σ_ld = 0.075 [0.007, 0.178],
  σ_trl = 0.10 [0.011, 0.222] — both hug zero (hierarchical-σ-null pattern again);
  shrunk team spread only −0.05…+0.06. In the hier fit the GLOBAL γ_ld CI widens to
  [−0.02, 0.19] ⇒ not credibly non-zero. **Held-out race (75/25 by match, plug-in
  posterior-mean, 179 test matches): hier − base = −0.0174 ± 0.0079 per match,
  t = −2.19 — hier is significantly WORSE OOS despite +7.2 in-sample loglik** —
  Ireland's "team hierarchies hurt OOS" reproduced at 715 matches. Verdict: keep the
  global linear spec; the γ_ld flip is most consistent with pregame-λ frailty, not
  team character. max R̂ 1.008.
- 2026-07-14: **r02b (WP3 gate iii) PASSED — with two data-validity lessons.**
  56 24/25, 143 matches, 661 identifiable bins. Raw comparison is POISONED by
  (a) settled selections and (b) ONE-SIDED thin markets (single traded side → vig-strip
  normalises p_fair to 1.0; raw market logloss 2.9 OU / 5.9 BTTS is an artifact —
  filters now built into r02b). On live two-sided rows: agreement corr 0.94/0.98/0.95
  (1X2/OU/BTTS), MAE 3.6–5.1 pts; reality-pricing is a statistical TIE (t = 0.15 / 1.05
  / −2.10, the last on only 27 matches). The composed engine is market-grade fair value
  — and prices the ~⅓ of OU rows + ~½ of BTTS rows where the exchange is one-sided or
  absent, which is exactly the coverage position management needs.
- 2026-07-14: **WP4 r03 v1 DONE — three-iteration story, each step a lesson:**
  1. **As-of fills = fake edge, amplified by the optimiser** (l04's Ireland lesson
     reproduced): LOCF fills gave REBAL e^2.44/match (×10^161 — absurd); EXIT t=3.66
     also inflated. NEVER trust as-of in-play backtests.
  2. **Forward fills (first print after decision, 1-min lag, 10% limit-order
     tolerance) + honest settlement**: unconstrained REBAL is RUINOUS (−4.06/match,
     ruin paths from lay liabilities): with NO per-line edge vs the exchange (r02b
     tie), a Kelly optimiser free to ADD is trading noise minus costs, and the
     limit-order filter adverse-selects stale quotes. "Exit yes, add no"
     rediscovered by the optimiser's failure mode.
  3. **REBAL-RO (reduce-only: box-constrained prox, may hedge/shrink held positions,
     never adds) WINS the race**: HOLD +0.0017 < EXIT τ=−0.05 +0.0044 < REBAL-RO
     +0.0067 G/match (×1.30 / ×1.95 / ×2.76 over 152 matches), no ruin, ~1.1 hedge
     trades/match, and LOWEST SE (0.0084 vs 0.0094) — the variance-reduction
     signature of correct hedging. Uplifts not yet individually significant
     (t≈1.26–1.29 at n=152) — power, not design: extend eval to 57's 140 betfair
     matches + swap in the smile-grid winner book when it lands.
  Book v1: curated totals+BTTS backs, edge≥0.03 vs Betfair pregame close, joint-Kelly
  stakes capped Σ≤0.2 (avg 1.9 bets / 0.098 stake per match; 27 matches no book).
- NEXT: (a) swap smile-grid winner book into r03 when the mcmc-beast grid lands;
  (b) extend eval set (57 betfair matches; consider latents for 56 20/21–22/23 via a
  backfill run); (c) graduation candidates once stable: l01 NHPP engine + l02 composer
  + l03 reduce-only rebalancer → src/; (d) red-card-triggered rebalance cadence
  (γ_man=×1.70 is the one event where fair value moves most).
