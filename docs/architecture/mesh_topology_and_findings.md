# Mesh Swarm Topology & Operational Findings

**BayesianFootball.jl Swarm Architecture**  
**Version:** 1.2.0  
**Status:** Active Standard  
**Workspace:** `scottish_runner:1`

---

## 1. Five-Node Swarm Topology

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       SCOTTISH RUNNER TMUX MESH (1.1 - 1.5)                              │
├───────────────────────────────────┬───────────────────────────────────┬──────────────────────────────────┤
│ PANE 1.1: COMPUTE NODE            │ PANE 1.2: MODELLER                │ PANE 1.3: BUILDER                │
│ • Host: mcmc-beast (32 cores)     │ • Model: gpt-5.6-sol (deep math)  │ • Model: gpt-5.6-terra (builder) │
│ • Role: Heavy MCMC sampling,      │ • Role: Turing @model formulation,│ • Role: Code implementation,     │
│   Gate 0-7 runs, Pkg.test()       │   linear predictors, AD-safety    │   adapters, walkthrough files    │
├───────────────────────────────────┼───────────────────────────────────┴──────────────────────────────────┤
│ PANE 1.4: MANAGER / REVIEWER      │ PANE 1.5: SCOUT (ARCHIVE & CODEBASE AUDITOR)                         │
│ • Model: gpt-5.6-sol (lead)       │ • Model: gpt-5.6-luna (fast lookups)                                 │
│ • Role: Gate verification, code   │ • Role: Query archive/experiments, retrieve baseline parameters      │
│   reviews, test execution, commits│   and output findings to /tmp/*.md artifacts                         │
└───────────────────────────────────┴──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Inter-Agent Communication & Artifact Drop Protocol

All agents communicate via the local mesh CLI tools and structured file drops:

### A. CLI Commands
- **Send instruction:** `agent-send scottish_runner:<target_pane> "<message>"`
- **Capture status:** `agent-capture scottish_runner:<target_pane> --wait-idle --timeout <sec>`
- **Dispatch compute to Beast:** `tmux send-keys -t scottish_runner:1.1 "git pull && julia --project <test>" Enter`

### B. Scout File-Drop Handoff Pattern (`/tmp/*.md`)
Instead of reading raw, unformatted tmux pane buffers which can truncate long tables or pollute text with ANSI escapes:
1. The requester passes a target file path in the query:
   ```bash
   agent-send scottish_runner:1.5 "Investigate wealth prior in archive. Save detailed report to /tmp/scout_wealth.md"
   ```
2. The Scout writes its complete, unclipped, structured markdown report directly to `/tmp/scout_wealth.md` and signals completion.
3. The requester (Manager, Modeller, or Builder) reads `/tmp/scout_wealth.md` directly via native file reading.
4. **Key Benefits:** Full preservation of mathematical equations, code snippets, and parameter tables without buffer truncation.

---

## 3. Key Architectural Findings & Rules of Engagement

1. **State Isolation & Testing:**
   - Tests and gate runs must execute via headless CLI commands (`julia --project test/...`) rather than relying on persistent interactive REPL state. This avoids state pollution and ensures 100% reproducible exit codes.
2. **ReverseDiff AD Safety (`docs/turing_ad_performance_guide.md`):**
   - The Turing `@model` must operate in pure **log-intensity space** ($\eta = \log \lambda$), avoiding redundant `exp -> log` conversions on ReverseDiff computation tapes.
   - All feature arrays must be `Float64` / `Int`, with zero `NaN` or `missing` values.
3. **Gate 2 Anti-Leakage Invariants:**
   - Features at match $i$ must strictly depend on `fit_ids` ($t_{\text{match}} < t_{\text{eval}}$).
   - Deleting future matches must leave historical feature vectors bit-identical (perturbation invariance).
4. **Gate Progression Standard:**
   - **Phase A (Daytime / Smoke):** Model development, Gate 0 (Contract) $\to$ Gate 1 (Config) $\to$ Gate 2 (Features) $\to$ Gate 3a (Parity) $\to$ Gate 3b (AD Gradients) $\to$ Gate 3c (Smoke Convergence) $\to$ Gate 4 (Extraction) $\to$ Gate 5 (Score Matrix / Pricing).
   - **Phase B (Overnight Batch Grid):** Gate 10 (Full 20-fold MCMC Grid) $\to$ Gate 6 (OOS Evaluation vs. Bet365/Betfair) $\to$ Gate 7 (Portfolio-Kelly Simulation).
