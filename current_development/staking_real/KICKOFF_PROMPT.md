# KICKOFF — Real-Data Staking MVP backtest (paste this as the first message of a fresh session)

Execute the approved plan at `/home/james/.claude/plans/is-it-possible-to-proud-harp.md`
("Real-Data Staking MVP") end-to-end, autonomously. Read that plan file FIRST — it has the
full design, file layout, reuse map, execution steps, and verification list. This brief only
adds the context that isn't in the plan.

## Mission (one line)

Backtest the sim-validated staking layer — per-line trust blend → coherent IPF grid tilt →
capped unified Kelly (P), with the EB trust fit as a junk-line alarm — on the REAL OOS
matches (Ireland 2025–26, ~275 matches, Betfair close) of the `src_sup40_sw40` L1 engine,
and answer: **does the EB fit pull trust w down on the markets the model is bad at
(home/away 1X2) while holding the good ones (unders, BTTS)?**

## Why this model / what to expect

- `src_sup40_sw40` = `PreGame.DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel`
  with supremacy_weight=0.4, smile_weight=0.4 — the best cell of the r21 grid
  (`current_development/split_market_pillar/r21_grid_search_src_smile.jl`; per-line backtest
  in `b21_results_of_r21_back_test.txt`).
- Its real per-line signature (b21): home −9.4% ROI / hurdle_G −0.037, away positive ROI but
  negative G (over-staking), draw/under_15/under_05/btts_yes/btts_no positive. This matches
  the staking_sim "sup-blind" world (E4) exactly, where the verdict was:
  curated per-line w ≻ EB-learned ≻ flat 0.5 ≻ raw model.
- Sim background lives in `current_development/staking_sim/` (l01/l02 + experiments.md) and
  `docs/bets_multi/staking_sim_report.pdf`. The memory note `staking-sim-mc-race` summarizes
  all four experiments.

## Critical design point (do not skip)

`src_sup40_sw40` prices O/U through the smile intensity Λ = λ_tot·φ (`SmileScoreMatrix`,
`src/predictions/score_computation/smile_poisson.jl`) — NOT the plain (λ_h, λ_a) grid. The
naive `state_draws` path silently de-smiles the totals. Fix per the plan: per-unit model
probabilities come from the Predictions PPD (`model_inference`), and the existing IPF tilt
(`coherent_multiplier` in `staking_sim/l02_strategies.jl`) imprints them onto the 144-state
grid as the w=1 targets. Verify: post-IPF grid reproduces the smile PPD probs to <1e-6.

## Decisions already made by James (do not re-ask)

- Book = the core 11 selections (1X2, O/U 1.5/2.5/3.5, BTTS) — reuse `SimMatch` + the
  staking_sim l02 machinery verbatim. Extended book (O/U 0.5/4.5/5.5, correct scores) is v2.
- Trust EB cold-starts at w≈0.5; ALL strategies bet from match 1; refit every ~25 matches.
- Commission c=0.02 into d_eff for decisions AND settlement (print a c=0 table too).
- Strategy race: FLAT_1pct · PB_BK_cap02 (BayesianKelly 0.03, Σa≤0.2 — b21-comparable
  baseline) · U_cap02 (w=1) · TRUST05_U_cap02 · CURATED05_U_cap02 (w=[0,0,0,.5,.5,.5,.5]) ·
  TRUST_EB_U_cap02. Keep the registry pluggable — testing NEW staking systems on these same
  books is an explicit goal (v2 backlog in the plan).

## Execution environment (CHANGED since the memory notes were written)

- The kaimon MCP REPL now points at the **Hetzner server: 32 threads / 16 cores, repo at
  `/root/BayesianFootball`** (verify with `pwd()`; older memory notes name other boxes/paths
  — trust the live session). UPDATE the memory notes `kaimon-repl-on-server` /
  `server-file-sync-workflow` with this once verified.
- Workflow: edit locally in `/home/james/bet_project/BayesianFootball` (branch
  `feat/split-market-pillar`) → commit+push → `git pull` on the server via kaimon `ex`
  (use `--rebase --autostash` if dirty) → include + run via kaimon → results committed
  server-side (`git -c user.name=... commit`) → push (pull --rebase first if rejected) →
  pull locally. Local ssh/scp to the server is password-only — git is the only transfer.
- kaimon gotchas (all verified this week): `ENV["JULIA_PKG_PRECOMPILE_AUTO"]="0"` before
  `using BayesianFootball`; never retry `start_session`; evals promote to background at 30s
  (`check_eval`, wait ≥30s between checks); the 10-min no-activity gate reports long runs as
  "failed" while Julia keeps running — print chunked progress lines to keep it alive, and
  recover results from files the runner serializes; `println` is stripped from `ex` output —
  return values with `q=false` (wrap file dumps in `Text(read(...))`); struct changes
  require `manage_repl` restart (plain re-include won't redefine structs).
- Preflight before writing code: confirm on the server that
  `data/double_poisson_smile_src_grid/` exists and contains the `src_sup40_sw40` experiment
  subdir (results.jld2). If absent, stop and ask James where the r21 payloads live.

## Deliverables

1. `current_development/staking_real/` — `l01_real_books.jl`, `r01_race_src_sup40.jl`,
   `experiments.md` (staking_sim run-log conventions), `results/`, `plots/`.
2. The race table (terminal W, G/match ± SE, maxDD, n_bets, turnover; c=0.02 and c=0).
3. The money shot: per-unit EB w trajectory over the season + pooled w0 — with the
   end-of-season values stated in the reads (expect home/away < 0.5, totals/BTTS ≈ 0.5).
4. Per-family G attribution per strategy (1X2 vs totals vs BTTS).
5. b21 cross-check (adapter validation): PB_BK_cap02's per-line signs match the b21
   `src_sup40_sw40` rows on shared selections.
6. Updated memory: Hetzner server note + a `staking-real-mvp` project note with the verdict.
7. Everything committed (BayesianFootball branch; results via server git as above).

Work autonomously; only stop for genuinely blocking decisions (e.g. missing experiment
payloads). James's verdict criteria: the EB alarm moving the right way on real data matters
more than the P/L of any single strategy — 275 matches is signature-reading, not ranking.
