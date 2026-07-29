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

- 2026-07-29: **OOS EVALUATION (user ask) — the in-sample optimism is NEGLIGIBLE and now
  MEASURED; the market tie holds out of sample; the model's real error is the LEVEL, ~5% hot**
  (`r08_oos.jl`, plus `l09` gaining the `:expo` integrator).
  - **Split, on the pregame convention:** fit the NHPP on **24/25 only (360 matches, 955
    goals)**, score **25/26 (349 matches, 971 goals)**. The 25/26 pregame latents are already
    walk-forward OOS from the funnel run, so **both stages are out of sample end-to-end**.
    BBC is what makes this a real test: 25/26 has incidents for only 16 of 175 t56 matches,
    so an incidents-only holdout would have been almost entirely t57. First end-to-end use of
    the BBC clock (`kind = :expo`, 18-bin [0,90] frame + per-match stoppage).
  - **THE OPTIMISM QUESTION IS SETTLED.** Fitting the same spec on ALL data and scoring the
    same test set gives a match-clustered log-loss gap of **+0.0010 (1X2, t = 1.91), +0.0007
    (O/U, t = 0.65), −0.0007 (BTTS, t = −0.62)** — about **0.001 nats**. The monthly
    walk-forward agrees independently: mean gap **0.00044** nats/row over 15 folds, max
    0.0011, one fold negative. So every previously published in-play number (r02b's tie, r06's
    tie on 376 matches) **stands as reported**. "Should be small" was right, and is now a
    measurement rather than an assumption.
  - **The market tie HOLDS out of sample** — 269 held-out matches, 7,574 rows:
    1X2 t = **−0.04** (corr 0.931), O/U t = **+1.25** (corr 0.962), BTTS t = **−1.50**
    (corr 0.827). All |t| < 2, same verdict as in-sample. NB the pooled row-level log-loss
    puts the market marginally ahead in all three families while the match-clustered mean
    favours the model on O/U — row-weighting vs match-weighting, neither significant. Report
    it as a tie and say which weighting.
  - **THE REAL ERROR IS THE LEVEL, and it is one scalar.** At kickoff on the held-out season
    the model prices **2.920 goals against 2.782 realised — +5.0% hot**. Decomposed: the
    pregame engine alone is 2.871, i.e. **+3.2% hot**, and the NHPP fitted on 24/25 adds
    another ~1.8% (K = 1.0511 trained on 24/25 vs K = 1.0146 for the correctly-calibrated
    in-sample fit). Two lessons: (a) the level DRIFTS between seasons — 24/25 realised 2.653
    goals/match, 25/26 realised 2.782, up 4.9% — and a fixed α does not track it; (b) this is
    a single number, so periodic K recalibration on recent matches is the cheap fix. **This is
    a bigger error than anything the MVPs were chasing.**
  - **The 1X2 favourite over-confidence SURVIVES out of sample**, milder but one-directional:
    p 0.754 → y 0.702, p 0.843 → y 0.801, p 0.972 → y 0.953; and under-confident below 0.5
    (0.028 → 0.032, 0.157 → 0.173). Classic over-dispersion. **Shrink the 1X2 book toward
    0.5 before staking it.**
  - **O/U miscalibrates the OTHER way — under-confident**: p 0.139 → y 0.101, p 0.861 →
    y 0.899, p 0.652 → y 0.679. Totals are too COMPRESSED while 1X2 is too SPREAD. That is
    consistent with the repo's existing "totals compression is denoising" finding, and it
    means a single global shrinkage would fix one family and break the other — **calibrate
    per family**.
  - Posterior is stable across the split: α −4.577 vs −4.602, γ_man 0.421 vs 0.484
    (×1.52 vs ×1.62), β 0.197 vs 0.148. γ_tr flips sign (−0.111 vs −0.055) but both hug
    zero — consistent with game state being the weak term throughout. max R̂ 1.004.
  - ⚠ Reporting trap: the per-checkpoint `bias` column is **structurally zero** (~1e-18)
    because complementary selections sum to 1 within each family. Same for r06's
    `model_minus_mkt`. Neither is evidence of unbiasedness; use the level check above.

- 2026-07-29: **WP-E SKIPPED — user decision.** With MVP-1 and MVP-2 both null and the plan's
  pre-registered expectation (BBC's value is *coverage*, not a new model) holding, the user
  chose to go straight to the WP-F race rather than spend a fit on MVP-3. The plan itself
  predicted MVP-3 would be **the weakest of the three** — it spends the most parameters on
  the noisiest signal — so this drops the least informative arm.
  **Consequence to remember if it is ever revived:** the blocking `fit_shot_xg` leak fix was
  never done. `fit_shot_xg` is fitted globally, which pregame was measured as negligible
  (pillar ρ 0.9942, implied log-λ shift 0.0034) but **in-play is disqualifying** — the model
  would read the outcome of the shot it is pricing. MVP-3 cannot be run without refitting the
  xG table excluding every match in the evaluation fold. Race arm 3 is therefore ABSENT, not
  scored as null.

- 2026-07-29: **WP-D DONE — GATE D NULL on both halves; and the run caught a HARNESS BUG that
  inflates every fold-paired t in this stream by ~3×** (`l07_nowcast.jl`, `r05_nowcast.jl`).
  - **⚠ THE HARNESS FINDING IS THE IMPORTANT ONE.** `l01.paired_diff` differences the 20
    (repeat, fold) means and divides by their SD. That treats 20 numbers as independent when
    (a) each fold mean already averages ~137 held-out matches, so its variance is ~1/137 of
    the match-level variance, and (b) the 5 repeats re-score **the same 550 matches**. The
    independent unit is the MATCH.

    | comparison | t fold-paired | **t match-clustered (n=550)** |
    |---|---|---|
    | nowcast − full | 1.43 | **0.60** |
    | nowcast_opp − full | 3.17 | **1.07** |
    | nowcast_opp − nowcast | 2.43 | 0.91 |
    | full − state (man_adv control) | 8.62 | **2.43** |
    | state − time | 0.33 | 0.22 |

    Inflation ≈ **2.4–3×**, larger than the ~1.9× measured elsewhere in this repo, and it
    is analytic: `SE_fold = σ_match/√(137·20)` vs `SE_match = σ_match/√550` ⇒ ratio √(2740/550)
    = 2.23, plus the repeat correlation. **Every fold-paired t in this stream is overstated
    by roughly this factor, including the incumbent's published 6.29 / 0.51 / 8.85.** No
    published sign flips and no conclusion reverses — man_adv survives at 2.43, state stays
    null at 0.22 — but magnitudes must be read accordingly. `match_clustered_cv` / `mc_diff`
    added; `cv_race` / `paired_diff` are demoted to fast spec-screening only. **WP-F must use
    the clustered harness.**
  - Also revises WP-C's numbers downward without changing its verdict: t_state 0.33 → ~0.14,
    t_man 8.62 → 2.43. The control argument survives because it rests on the RATIO.
  - **Gate D (i): NULL.** Own surplus adds t = **0.60** over the incumbent (pre-registered
    bar was |t| > 2). γ_sf = 0.109 [0.027, 0.192] is credibly positive in-sample, but it does
    not buy held-out prediction. Adding the OPPONENT's surplus reaches t = 1.07 — the t = 3.17
    that looked significant was purely the harness bug. Recorded as a null; **not** written up
    as a near-miss.
  - **Gate D (ii): calibration is CLEAN, which does not rescue it.** Bias at 60/75/85′ =
    **+0.004 / +0.016 / +0.016** (se ~0.04/0.03/0.02) — comparable to the incumbent's
    +0.020/+0.014/+0.003 and far better than MVP-1's −0.075/−0.051/−0.024. By surplus tercile
    all nine cells sit within ~1 SE (largest: high-surplus +0.058 ± 0.069 at 60′). The gate
    required both halves to move together; only one did, so it is a null.
  - **The pre-registered confound is CLEANLY RULED OUT** — this null is not a confound
    artifact. Freeing the pregame-rate exponent retains **98.3%** of γ_sf (0.1090 → 0.1071)
    and ρ = 0.917 [0.684, 1.160] covers 1, so the model is not rescaling the pregame belief.
    Pinning the pregame rate at 1 leaves the surplus contribution unchanged (t 1.46 vs 1.43
    fold-paired). Surplus is genuinely measuring in-match deviation; there is just not much
    in it.
  - **Leak audit — kept, but it answers less than it looks like.** The future-shifted surplus
    scores t = 28.3 vs the causal 1.43. That is **tautological, not informative**: a goal IS a
    shot (`goals ⊂ shots` by construction), so letting the covariate see the slice being
    predicted cannot lose. What it does establish is that the causal build is **not leaking** —
    an off-by-one would have put the causal t near the future one instead of ~500× below it in
    mean effect. Downgraded in the code to a negative control. A real version would have to
    exclude goals from the shot count.
  - Surplus covariate audits clean pre-fit: E[shots] calibrated to 2% late in the match
    (7.65 observed vs 7.51 expected), the high clamp binds on 0.035% of rows, 15.0% of rows
    are held at surplus = 0 (the early slices where E < 1 shot), raw cor(surplus, y) = 0.020.
  - Posterior otherwise reproduces the incumbent: β 0.145, γ_man 0.493 (×1.64), max R̂ 1.005.

- 2026-07-29: **WP-C DONE — GATE C IS A CLEAN NULL on its pre-registered criterion, and the
  null is TRUSTWORTHY because a control resolved on the same data. The decomposition then
  found the real mechanism, in conversion rather than volume** (`l06_shot_flow.jl`,
  `r04_shot_flow.jl`). 550 matches, **10,202 shots vs 1,498 goals = 6.8× the counts**;
  realised 9.27 shots / 1.36 goals per team-match against pregame λ_s 9.03 and p2 0.1488
  (empirical 0.1468 — the thinning assumption holds to 1.4% before any fitting).
  - **The pre-registered question was: is the incumbent's game-state null (`state − time`
    t = 0.51 on goals) a real absence, or a resolution failure at 1.4 goals per team-match?
    Answer: a REAL ABSENCE.**

    | arm | resolution | counts | t_time | **t_state** | t_man |
    |---|---|---|---|---|---|
    | funnel_apm_xg | shots | 10,202 | 1.04 | **0.33** | 8.62 |
    | funnel_apm_xg | goals | 1,498 | 3.06 | **0.60** | 4.43 |
    | funnel_winner | shots | 10,202 | 1.04 | **−0.21** | 8.52 |
    | funnel_winner | goals | 1,498 | 3.06 | **1.17** | 4.36 |

    Win condition was |t| > 2 at shot resolution. Measured 0.33 / −0.21 — **not merely
    short of it, slightly WEAKER than at goal resolution.** Recorded as a null.
  - **Why this null is believable rather than a power failure — the control resolved.**
    `γ_man` goes **t = 4.4 (goals) → 8.6 (shots)**, i.e. the red-card effect nearly doubles
    its t-stat on exactly the 6.8× counts that left game state flat. The instrument works;
    it just finds nothing for game state. Without that control the null would be worth
    little, which is why the race was run at both resolutions on the SAME matches.
  - **BUT game state does act — on CONVERSION, not volume.** Empirical conversion by state:
    **level 13.9%, leading 17.3%, trailing 13.6%.** Match-CLUSTERED bootstrap (2,000 reps
    over 550 matches, because a Binomial likelihood treats 10k shots as independent when
    they cluster in matches): **leading − level = +0.0337 ± 0.0071, z = 4.76**, CI
    [0.020, 0.048]; trailing − level = −0.003, z = −0.48 (nil). The Turing fit agrees:
    κ_ld = +0.252 [0.146, 0.360], κ_tr = −0.028 [−0.137, 0.080].
    **A team that is ahead does not shoot more — it shoots better.** Counter-attacking
    transition chances, which is a quality effect a goal-count model cannot see.
  - **This RESOLVES the open question r01/r01b left.** r01 found `γ_ld = +0.09` on goals with
    the sign FLIPPED vs Ireland's −0.24, and r01b could only say it was "most consistent
    with pregame-λ frailty, not team character". The shot decomposition shows what it
    actually is: on shot VOLUME `γ_ld = −0.073` [−0.116, −0.030] — **negative, back in
    Ireland's direction** — while conversion is strongly positive. The goal-level `γ_ld`
    was the NET of two real, opposite-signed mechanisms, not an artifact. `γ_tr = +0.049`
    [0.005, 0.092] (trailing teams shoot slightly more, convert no better).
  - **Second part of Gate C — checkpoint bias — REGRESSED.** Remaining-goals bias at
    60/75/85′ is **−0.075 / −0.051 / −0.024** (se 0.041 / 0.033 / 0.023) against the
    incumbent's clean +0.020 / +0.014 / +0.003. MVP-1 **under**-predicts late goals.
    The diagnosis is in the same table: `t_time` falls 3.06 (goals) → 1.04 (shots), so goal
    intensity rises over the match *more* than shot intensity does — conversion rises late,
    and constant-p2 thinning structurally cannot express that. Same story as `γ_man` 0.436
    on shots (×1.55) vs the incumbent's 0.53 on goals (×1.70): conversion improves with a
    man advantage too.
  - **Verdict: MVP-1 as specified is NOT a replacement for the incumbent** — it is worse on
    the calibration check while being null on the term it was built to resolve. Its value is
    diagnostic: it separates volume from conversion and settles the γ_ld question. The
    indicated fix (state- and time-dependent p2, already fitted here and credibly non-zero)
    is a change of specification, not a tuning knob, and belongs to a later iteration rather
    than being smuggled into WP-F. Carry MVP-1 into the race as specified.
  - max R̂ 1.006 across both fits.
  - Ops note: the kaimon gate reported `failed — no activity for 10m` at 15m34s while Julia
    was still running and had already written `out/r04_gate_c.jls`. **Check artifacts and
    load average, not the gate's verdict.** Load had fallen 6.57 → 0.54, confirming
    completion; `ps` `%CPU` is a lifetime average and stays pinned at 416, so it is useless
    as a liveness signal — the load average is the live one.

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
