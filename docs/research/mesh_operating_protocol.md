# Operating Protocol: Tmux Multi-Pane Swarm & Compute Orchestration

**Role:** Manager (`pi` running `openai-codex/gpt-5.6-sol` in Pane `scottish_runner:1.3`)  
**Objective:** Lead Architect for the Feature Pipeline & Data Leakage Prevention (Gate 2)

---

## 1. Multi-Pane Swarm Topology (2x2 Grid)

You are running inside the tmux session `scottish_runner`:

```
┌──────────────────────────────────────────────┬──────────────────────────────┐
│  Pane 1.1: COMPUTE (mcmc-beast:32t)          │  Pane 1.3: MANAGER (YOU)     │
│  - Attached to inner tmux session on beast   │  - Lead Architect & Reviewer │
│  - Shell in /root/BayesianFootball           │  - Coordinates Swarm         │
│  - Executes remote Julia Gate 2 test grids   │  - Reviews diffs & signs off │
├──────────────────────────────────────────────┼──────────────────────────────┤
│  Pane 1.2: BUILDER (gpt-5.6-terra)           │  Pane 1.4: SCOUT (gpt-5.6-luna)│
│  - Fast implementation worker                │  - Archive Reader & Auditor  │
│  - Writes code & unit test suites            │  - Scans legacy code & traps │
└──────────────────────────────────────────────┴──────────────────────────────┘
```

## 2. Swarm Delegation Workflow

As the Lead Architect, you delegate research and implementation while retaining ownership of architecture, code review, and remote Gate 2 verification.

### Role Responsibilities:

1. **Manager (Pane 1.3 - YOU / `gpt-5.6-sol`):**
   - Defines the mathematical contract and Gate 2 invariants.
   - Dispatches research questions to Scout (Pane 1.4).
   - Dispatches coding/porting tasks to Builder (Pane 1.2).
   - Reviews the Builder's git diff against Gate 2 rules.
   - Pushes to git and triggers remote verification on Compute Node (Pane 1.1).

2. **Scout (Pane 1.4 - `gpt-5.6-luna`):**
   - High-speed archive reader and documentation auditor.
   - Explores `current_development/plus_minus_ratings/`, data schemas, and minute-tracking caveats.
   - Audits proposed implementations for non-temporal leakage.

3. **Builder (Pane 1.2 - `gpt-5.6-terra`):**
   - Workhorse implementation agent.
   - Writes feature structs in `src/features/types.jl`, extractors in `src/features/plus_minus/` or `src/features/extractors/`, and test suites in `test/test_apm_gate2.jl`.

4. **Compute (Pane 1.1 - `mcmc-beast:32t`):**
   - 32-core remote execution engine for full 20-fold Gate 2 test suites.

---

## 3. Communication Commands

### A. Dispatch Task to Scout (Pane 1.4)
```bash
agent-send scottish_runner:1.4 "Audit current_development/plus_minus_ratings/l04_ridge_apm.jl. Summarize: 1. Regularized ridge formulation 2. Teammate similarity matrix S 3. Fallback for unmapped players."
agent-capture scottish_runner:1.4 --wait-idle --timeout 60
```

### B. Dispatch Coding Task to Builder (Pane 1.2)
```bash
agent-send scottish_runner:1.2 "Implement ShotsPlusMinusFeature in src/features/types.jl and port the ridge estimator into src/features/plus_minus/ridge.jl. Follow the Gate 2 rules."
agent-capture scottish_runner:1.2 --wait-idle --timeout 120
```

### C. Review Diff Locally
```bash
git diff src/
```

### D. Execute Remote Test on mcmc-beast (Pane 1.1)
```bash
git add . && git commit -m "feat(features): apm gate 2 safe implementation" && git push
tmux send-keys -t scottish_runner:1.1 "git pull && julia --project test/test_apm_gate2.jl" Enter
sleep 20
tmux capture-pane -p -t scottish_runner:1.1 -S -50
```

---

## 4. Phase 3 Mission: APM (Adjusted Plus-Minus) Feature Ingestion

### Core Mission:
Graduate the validated Scottish Lower APM player rating into `src/features/` with strict **Gate 2 compliance** (zero forward leakage, bit-invariance under future-match perturbations, type safety, neutral fallbacks).

### Key References & Specs:
- **Kickoff Spec:** `current_development/plus_minus_ratings/KICKOFF_src_integration.md`
- **Ridge Math & Solver:** `current_development/plus_minus_ratings/l04_ridge_apm.jl`
- **Segments & Commentary:** `current_development/plus_minus_ratings/l01_segments.jl` and `l02_shot_parser.jl`
- **Target Definitions:** `current_development/plus_minus_ratings/l03_targets.jl`
- **Contract to Emit:** `src/features/extractors/player_extractors.jl` (8 positional vectors: `flat_home_G_rating`, etc., plus `player_ratings_map`).

### Critical Gate 2 Invariants:
1. **History-Only Fit:** The APM ridge regression must be fitted strictly on matches where $t_{\text{match}} < t_{\text{eval}}$ (`F_data[:history_match_ids]`). Target/OOS matches must never enter the design matrix $X$.
2. **Missing Minutes Caveat:** In Scottish Lower (56/57), `lineups.minutes_played` is missing/zero in historical seasons. Aggregate positional ratings by **on-pitch starter status** (e.g. `clamp(mins, 0, 90) / 90` or starter status fallback), never multiply by zero.
3. **Missing Coverage Fallback:** If `ds.bbc_events` is missing or a player has no historical ratings, fallback cleanly to `0.0` (neutral effect) without crashing AD tapes.
4. **Perturbation Invariance:** Dropping or appending future matches must leave historical rating vectors bit-identical.

