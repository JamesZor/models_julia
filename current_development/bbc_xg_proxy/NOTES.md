# bbc_xg_proxy — BBC-stats xG proxy for Scottish League 1/2

Started 2026-07-17. Plan: `~/.claude/plans/can-we-look-at-calm-crayon.md`.

## Goal

Train an xG-prediction model on Scottish Premiership (54) / Championship (55), where
sofascore `expectedGoals` exists (Y) and BBC-scraped match stats (Postgres schema
`bbc`) exist (X). Apply the frozen model to League One (56) / League Two (57) —
which have **zero** sofascore match statistics — to synthesize
`:flat_home_xg`/`:flat_away_xg` and unlock the Ireland-style joint
goals + xG + market engine (`split_market_pillar` keeper) for the lower leagues.

## Session scope (2026-07-17)

WP0 (data QA) → WP1 (proxy model + Prem↔Champ transfer gate) → WP2 (forward
informativeness on League 1/2 vs goals/shots baselines). **Stop after WP2.**
Bayesian integration (WP3+: ProxyXGFeature extractor, direct-SoT arm, joint-model
grid on ScottishLower) happens in later sessions if WP2 green-lights.

## Data facts (verified 2026-07-17)

- `bbc.match_meta.match_id` **is** the sofascore match_id. 4,394 matches, all
  SUCCESS; tiers 54/55/56/57, seasons 20/21–25/26. ~206 don't join to
  sofascore.matches ⋈ seasons — resolved in r00.
- `bbc.match_stats`: EAV (`stat_cat`,`stat_type`,`home_value`,`away_value`,`filled`).
  `filled=true` ⇒ value was absent upstream and imputed — **treat as missing**.
- Universal stats (n=4,394): shotsTotal, shotsOnTarget, shotsOffTarget,
  shotsBlocked, hitWoodwork, cornersWon, possessionPercentage, foulsCommitted,
  totalYellowCard (+ GK saves splits). Partial (~1.2–1.3k matches only): passes,
  touches, tackles, aerials — not universal, excluded.
- sofascore xG (`match_statistics.stat_key='expectedGoals'`, period='ALL'):
  Champ 23/24 (180), 24/25 (178), 25/26 (172); Prem 25/26 only (198).
  **Training overlap: 728 matches ≈ 1,456 team-match rows.** League 1/2: none.

## Files

- `r00_data_qa.jl` — WP0: join integrity, coverage/fill matrix, distribution
  sanity, bbc↔sofascore per-stat agreement (data-quality acceptance gate).

## WP0 verdicts (2026-07-17) — GATE PASSED

1. **Join**: 4,188/4,394 bbc matches join sofascore.matches⋈seasons. All 206
   unjoined are absent from sofascore.matches entirely: 191 kickoff-2020 (early
   20/21 gap in sofascore) + 15 kickoff-2026 (not yet fetched). No id drift.
   `scores_match` true on 4,186/4,188; the 2 mismatches are lower-league
   (10387906 Clyde–Montrose 22/23, 11395473 Spartans–Elgin 23/24) — exclude or
   ignore.
2. **`filled` semantics (corrected)**: `filled=true` ≠ imputed-zero. Filled rows
   carry nonzero values and agree well with sofascore where checkable
   (SoT cor 0.992 n=132; yellows 0.953; shotsBlocked 0.86 n=402 — noisier).
   Interpretation: backfilled from a secondary source. Policy: **use filled
   values** for all stats; shotsBlocked gets extra caution (33–65% filled in
   lower tiers).
3. **Coverage** (genuine-only rates): SoT/off/total/corners/possession/fouls
   97–100% everywhere except Champ 20/21 (80%) & 21/22 (72%). hitWoodwork is
   ~4% genuine but absence≈0 for a rare event — usable as count.
4. **bbc↔sofascore agreement (not-filled, Prem/Champ)**: cor 0.91–0.98,
   exact-match 84–95% (possession exact 40% but MAE 2.0 pp). **PASS.**
5. **Distribution sanity**: Prem 24.9 shots/match, lower tiers 18.5–19.3;
   SoT ≈ 8/match; possession sums ≈ 99; shotsTotal ≈ SoT+off+blocked (Prem
   exact; lower-tier gap explained by missing blocked). Home>away on SoT in all
   tiers. All benchmark-consistent.
6. **WP1 training set confirmed**: Prem 25/26 (198) + Champ 23/24–25/26
   (180+178+172) = **728 matches / 1,456 team rows**.

**Canonical feature vector (team-match, per side)**: shotsOnTarget,
shotsOffTarget, shotsBlocked, cornersWon, possessionPercentage, foulsCommitted,
hitWoodwork, home-flag. (shotsTotal excluded — collinear with its components.)

## WP1 verdicts (2026-07-17) — TRANSFER GATE PASSED, winner frozen

Ladder (season-blocked CV, pooled OOS on 1,454 team rows):

| model | link | R² | MAE | Spearman |
|---|---|---|---|---|
| m0 SoT-only | gamma | 0.381 | 0.500 | 0.635 |
| m1 full linear | gamma | 0.132 | 0.505 | 0.718 |
| **m2 +sqrt+SoT×poss** | **gamma** | **0.442** | **0.464** | **0.715** |

- m1 warning: raw count terms under a log link extrapolate badly OOS (R²
  collapse on held-out season blocks); the sqrt terms are what stabilize m2.
- **Transfer (tier-invariance gate)**: Prem→Champ (the downward direction that
  mimics applying to League 1/2): R² 0.408 vs 0.496 in-sample, MAE 0.452 vs
  0.443, Spearman 0.701 ≈ in-sample 0.707. Champ→Prem R² 0.335 (Prem has fatter
  xG tails). Rank signal transfers essentially intact → **PASS**.
- Calibration: deciles 1–9 clean; decile 10 over-predicts (pred 3.02 vs actual
  2.56) — known tail bias, revisit (clamp/isotonic) at WP3 integration.
- Significant terms: sqrt(sot) (z=7.0), sblock (5.6), poss (3.6), woodwork
  (2.8), sqrt(soff) (2.6), is_home (2.5), sot×poss (−3.1). corners/fouls_opp
  inert — candidates to drop at graduation.
- Frozen artifact: `proxy_model_v1.jls` (m2/gamma, all 1,454 rows, NamedTuple
  with model + formula + link + stats + provenance).

## WP2 verdict (2026-07-17) — CONDITIONAL PASS, see RESULTS_bbc_xg_proxy.md

Proxy form adds a small consistent OOS edge on League 1/2 home-win prediction
over goals+SoT forms (−0.0013/−0.0018 log-loss, all seasons, both half-lives;
paired t 1.2–1.4, underpowered), subsumes the SoT signal (form_pxf z=2.01,
SoT forms inert), adds ≈nothing on next-match goals beyond shots. Green-light
for WP3+ with the direct-SoT arm (WP3b) as mandatory comparator.

## Files (final, this session)

- `r00_data_qa.jl` — WP0 QA (tables in global `QA`).
- `l01_xg_proxy.jl` / `r01_train_proxy.jl` — WP1 proxy ladder + gates (`R1`).
- `l02_informativeness.jl` / `r02_informativeness.jl` — WP2 nested
  forward-informativeness test (`R2`).
- `proxy_model_v1.jls` — frozen m2/gamma artifact (model+formula+link+provenance).
- `RESULTS_bbc_xg_proxy.md` — full WP0–WP2 write-up + WP3 handoff.

## Log

- 2026-07-17: dir created; WP0 run + gate passed (see above). WP1 run: m2/gamma
  frozen to proxy_model_v1.jls; transfer gate passed. WP2 run: conditional pass;
  session stopped here per plan — WP3+ (ProxyXGFeature, direct-SoT arm, joint
  grid) next session.

---

# WP3 — FUNNEL CASCADE (2026-07-21)

Plan: `~/.claude/plans/funnel-cascade-smoke-curried-candle.md`.

Replaces the WP1 proxy-GLM route: instead of laundering shots through a frozen
xG model, put them straight in the observation model as a thinned Poisson
funnel.

    Shots ~ Poisson(λ_s);  SoT|Shots ~ Bin(p₁);  Goals|SoT ~ Bin(p₂)

Poisson thinning ⇒ goals marginal stays Poisson(λ_s·p₁·p₂), so the score grid,
market probs, eval-vs-close and Kelly are all untouched — the funnel only
enriches the OBSERVATION model (~7× the count volume, ≈2.7× less noise on the
log-rate).

## Files

- `l03_funnel_cascade.jl` — Stage 1 loader: `BBCFunnelFeature` (own SQL for
  `shotsTotal`/`shotsOnTarget`, `.jls`-cached, eager-loaded) + engine
  `TeamFunnelDPGoalsModel` + extraction + Poisson-grid pricing overrides.
- `r03_funnel_smoke.jl` — Stage 1 smoke vs the `none_pois` comparator.
- `l04_funnel_hier.jl` — Stage 2 loader (hierarchical per-team p₁/p₂),
  WRITTEN BUT NOT YET RUN.
- `bbc_funnel_counts.jls` — cached counts, 1968 matches (984 per league).

## Data (verified 2026-07-21, re-queried)

56: shots 10.12/9.06, SoT 4.43/4.02, goals 1.49/1.33
57: shots  9.62/8.73, SoT 4.19/3.73, goals 1.44/1.25
Coverage 100% (1968/1968); `sot > shots` 0.0%; `goals > sot` 0.74% (own goals,
routed to the marginal-Poisson fallback). Empirical p₁ 0.436, p₂ 0.337.

## THE STRUCTURAL FACT (drives everything downstream)

The cascade log-likelihood is ADDITIVELY SEPARABLE in (λ_s, p₁, p₂):

    log P = [S·log λ_s − λ_s − log S!]              ← λ_s only
          + [log C(S,T) + T·log p₁ + (S−T)·log(1−p₁)]  ← p₁ only
          + [log C(T,G) + G·log p₂ + (T−G)·log(1−p₂)]  ← p₂ only

Since α/β/HA live only inside λ_s, **goals contribute ZERO gradient to team
strength** — team strength is fitted purely to shots; goals only calibrate the
global conversion. Consequences:

- The `funnel_weight` knob does NOT interpolate to `none_pois` (down-weighting
  shots+SoT just removes team-strength information; goals never had any).
  The meaningful dial is on the GOALS ROUTING —
  `cascade_weight·logBin(G|T,p₂) + (1−cascade_weight)·logPois(G|λ_s·p₁·p₂)`,
  where 0 is exactly `none_pois` reparameterised and 1 is the pure funnel.
  NOT YET IMPLEMENTED.
- Stage 2's per-team p₂ is how goals re-enter the *goal-rate* prediction
  (through the conversion channel, not the volume channel).

## PERFORMANCE (docs/turing_ad_performance_guide.md review)

The model met every checklist item but not §6's spirit. Counts, masks and decay
weights are all data, so the weighted log-likelihood collapses onto sufficient
statistics computed once in the builder:

    Σ w·m·logPois(S|λ_s)    = Σ(w·m·S)·log λ_s − Σ(w·m)·λ_s
    Σ w·m·logBin(T|S,p₁)    = S_sot·log p₁ + S_miss·log(1−p₁)
    Σ w·c·logBin(G|T,p₂)    = S_goal·log p₂ + S_save·log(1−p₂)

No Poisson/Binomial construction and no lgamma on the tape at all. Dropped
terms are parameter-free and the routing is data-fixed ⇒ posterior EXACTLY
unchanged (verified: naive − sufficient = const −8581.9596336, spread 1.1e-11
across draws varying both rates and conversion; −2120.7388996 at fw 0.35).
Only `lp` shifts — don't compare it across engines.

Two initialisation bugs this exposed:

1. Putting the shot level in the PRIOR (`Normal(2.3, 0.3)`) fails because
   `UniformInit(-2,2)` initialises in value space ⇒ chains start at λ_s ≈ 1
   against ~10 shots. Fix: fixed `shot_scale = log(10)` offset with
   `prior_μ_base = Normal(0, 0.3)`. ε went 4e-4 → 0.22.
2. Cheaper iterations need MORE of them. The conversion posterior is ~15×
   tighter than its prior (sd ≈ 0.03 on the logit); at 200 warmup the chain was
   still in the burn-in transient (reported p₂ = 0.249 with a +339 gradient
   still pushing toward the mode — NOT a likelihood bug: the gradient at the
   data MLE is ≈[0.5, 0.5], pure prior pull). Fix: warmup 1000.

Net: gradient 0.83 ms / 95 params (guide target < 1 ms); step 0.22, mean tree
depth 4.0 at max_depth 8. Funnel trains in 27.9 min vs the first attempt not
finishing a single chain of 20 in 4.5 h.

## Stage 1 VERDICT (r03, 27/27 checks passed) — target 25/26, 5 splits, 66 OOS

Convergence: max R-hat 1.011 (funnel) / 1.011 (none_pois); p1_raw 1.0005,
p2_raw 1.0013. Posteriors: p₁ 0.443 [0.434, 0.453], p₂ 0.333 [0.320, 0.346],
λ_shots 9.89, λ_goals 1.465/1.326, δ₅₆−δ₅₇ = +0.041 on the shot scale.

**Team-strength spread — my prediction was WRONG, in the good direction.**
across-team sd of log λ_goals(home): funnel 0.1072 vs none_pois 0.0595,
ratio **1.8**. I expected compression (shot-rate spread is proportionally
tighter than goal-rate spread). What actually dominates is SHRINKAGE: with
~1.4 goals/match the goals-only model can barely separate teams and the
hierarchical prior pulls them together; 7× the counts lets the posterior
actually resolve them. The funnel is not compressing strength, it is
RESOLVING it.

LogLoss diff vs Bet365 close (negative = beats the close), family-pooled:

| model  | x12    | btts    | totals  |
|--------|--------|---------|---------|
| funnel | 0.0153 | −0.0078 | −0.0139 |
| none   | 0.0224 | −0.0082 | −0.0217 |
| Δ      | −0.0071 (funnel better) | +0.0004 (tie) | +0.0078 (funnel worse) |

Coherent with the maths: shots pin RELATIVE team strength well (1X2 improves,
closing ~1/3 of the gap to the close), but the goal LEVEL is forced through a
global conversion constant, so totals calibration degrades (both still beat the
close on totals). **n=66 OOS — a smell test, not a verdict.**

## Next

1. Stage 2 (`l04`, already written): hierarchical p₁/p₂. The read is whether
   per-team conversion recovers the totals loss while keeping the 1X2 gain.
2. Implement `cascade_weight` — the only dial that actually spans
   funnel ↔ none_pois (see the structural fact above).
3. Only then a real grid (more folds/seasons); n=66 decides nothing.

## Log

- 2026-07-21: WP3 Stage 1 built + run. 27/27 checks. Sufficient-statistic
  refactor + shot_scale/warmup fixes after the first attempt stalled.

## Stage 2 VERDICT (r04, 2026-07-21) — NULL

Hierarchical per-team p₁/p₂: σ_p1 = 0.034, σ_p2 = 0.058 against a
half-Normal(0, 0.3) prior (mean 0.239) — pulled to 1/7 and 1/4 of prior.
±1sd team finishing spans p₂ ∈ [0.320, 0.346] (pooled 0.333), a ±4% relative
spread. Team-strength spread unchanged (0.1041 vs Stage-1 0.1072). Convergence
clean (max R-hat 1.0117), p1_μ/p2_μ still on the pooled MLE — a real null, not
a fitting failure. Reproduces the Ireland hierarchical-σ null a third time.

LogLoss (x12 / btts / totals): hier 0.0140 / −0.0081 / −0.0151 vs Stage-1
0.0153 / −0.0078 / −0.0139 vs none 0.0224 / −0.0082 / −0.0217. Better on all
three by ~0.001 on n=66 — noise, for 7.9× the compute (3h 40m vs 27.9 min).
Recovers only ~15% of the totals gap to none_pois.

**Conclusion: teams differ in shot VOLUME, not conversion. Keep global p₁/p₂
(Stage 1). Per-team conversion is not the totals explanation ⇒ next lever is
`cascade_weight`.**

Cost note: 189 params ⇒ 2.89 ms/gradient (vs 0.83 ms / 95), and the 80 team
effects are badly conditioned when σ collapses toward 0 — the classic
non-centred funnel geometry. Fix that before ever revisiting per-team p.
