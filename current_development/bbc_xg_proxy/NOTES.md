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
