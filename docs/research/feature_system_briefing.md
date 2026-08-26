# Mission Briefing: Feature Pipeline, Data Integrity & Leakage Prevention (Gate 2)

**Target Agent:** Manager (`pi` running `openai-codex/gpt-5.6-sol` in Pane 1.2)  
**Knowledge Partner:** Scout (`bf-scout` running `openai-codex/gpt-5.6-luna` in Pane 1.3)  
**Compute Node:** Remote `mcmc-beast:32t` (attached in Pane 1.1)  
**Date:** 2026-08-26  

---

## 1. System Topology & Multi-Pane Communication

You are operating within the `scottish_runner` tmux environment:

```
┌──────────────────────────────────────────────┬──────────────────────────────────────────────┐
│  Pane 1.1: Compute (Remote mcmc-beast)       │  Pane 1.2: Manager (YOU — Lead Architect)    │
│  - Attached to inner tmux on mcmc-beast:     │  - Model: openai-codex/gpt-5.6-sol (medium)  │
│    • julia:1 -> Bash shell for test runs     │  - Role: Architecture, planning & code QA    │
│    • julia:2 -> Julia Gate session           │                                              │
│    • julia:0 -> Kaimon server (:2828)        ├──────────────────────────────────────────────┤
│    • mbtop:0 -> btop hardware monitor        │  Pane 1.3: Scout (bf-scout — Knowledge Base) │
│  - To execute tests:                         │  - Model: openai-codex/gpt-5.6-luna (fast)   │
│    tmux send-keys -t scottish_runner:1.1     │  - Role: Fast codebase & archive lookups     │
│      "julia --project test/..." Enter        │  - Query: agent-send scout "<question>"      │
└──────────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 2. Objective & Non-Goals

### 🎯 Primary Objective
Design and implement a robust, standardized **Feature Extraction & Verification Workflow** that attaches ("monkey-patches") new feature arrays into `FeatureSet` (`F_data`), guaranteeing:
1. **Strict Zero Data Leakage:** Features available at kickoff timestamp $T$ use *only* information strictly preceding $T$.
2. **Gate 2 Feature Contract Compliance:** Type purity (`Float64`/`Int` arrays, zero `NaN`/`missing`), contiguous state indexing, and deterministic perturbation invariance (dropping future matches leaves past features bit-identical).
3. **Graceful Fallbacks:** Deterministic fallback defaults for unmapped, promoted, or missing teams/players.

### 🚫 Explicit Non-Goals (What NOT to do)
- **DO NOT modify Turing `@model` definitions or likelihoods.**
- **DO NOT retrain MCMC models or run MCMC sampling grids.**
- **DO NOT create duplicate model directories** (e.g. `01_negbin_wealth`, `01_negbin_distance`).
- We are **only** building and verifying the data ingestion and transformation layer.

---

## 3. Stepping-Stone Progression Strategy

We will use simple, well-understood features first to establish and harden the feature attachment and Gate 2 verification pipeline before tackling complex stint-based APM features:

```
┌───────────────────────────┐      ┌───────────────────────────┐      ┌───────────────────────────┐
│   STEP 1: DISTANCE        │ ──►  │   STEP 2: WEALTH          │ ──►  │   STEP 3: APM RATINGS     │
│  (Static pairwise geometry│      │ (Point-in-time valuations,│      │ (Player lineups, stints,  │
│   & OOS coordinate lookup)│      │  temporal stability check)│      │  filtered goal margins)   │
└───────────────────────────┘      └───────────────────────────┘      └───────────────────────────┘
```

---

## 4. Codebase & Archive Landscape

### A. Distance Features Archive
- **Location:** `current_development/scottish_lower/archive/distance/`
- **Core Files:**
  - `l01_distance_features.jl`: GPS stadium catalog, Great-Circle Haversine distance, log-distance standardization ($z_{\text{dist}}$), and distance tier classification.
  - `r00_eda_distance_fatigue.jl` & `EXPERIMENT_NOTES.md`: Empirical findings on away travel fatigue.
- **Key Challenges:** Handling newly promoted or unmapped stadiums in out-of-sample folds without crashing or leaking.

### B. Squad Wealth Archive
- **Location:** `current_development/scottish_lower/archive/wealth/` and `docs/models/recombination/03_squad_wealth_submodel.md`
- **Core Files:**
  - `l01_wealth_features.jl`: Starting-XI market value ratios, $\Delta W = \log W_h - \log W_a$.
- **Key Challenges:** Ensuring squad valuation reflects strictly point-in-time data at kickoff date $T$, without backward-leaking mid-season transfer valuations.

### C. APM (Adjusted Plus-Minus) Target
- **Location:** `src/features/plus_minus/` and `src/features/extractors/plus_minus_extractors.jl`
- **Core Files:**
  - Lineup stint parser, regularized ridge APM, player substitution minutes.
- **Key Challenges:** Cumulative minute filtration. For match $k$ at time $T$, player ratings must be derived *only* from stints in matches $1 \dots k-1$.

### D. Protocol Gate 2 Contract
- **Location:** `current_development/scottish_lower/_protocol/features.jl`
- **Interface:** `sl_gate_features(contract, adapter, ds)` validates 7 mandatory properties:
  1. `kickoff filtration holds`: All fitted kickoffs strictly precede OOS kickoffs.
  2. `kickoff filtration drops`: No valid training rows dropped.
  3. `perturbation test`: Truncating future matches produces bit-identical feature arrays for past matches.
  4. `type purity`: AD-safe typed vectors (`Float64`/`Int`), zero `missing`, zero `NaN`.
  5. `team_map keyed by NAME`: String-keyed maps matching runtime lookups.
  6. `contiguous model time states`: 1..K continuous time indices.
  7. `OOS team coverage`: Unmapped teams gracefully fallback to population defaults.

---

## 5. Recommended Next Actions for the Manager

1. **Query the Scout:** Send a targeted request to Scout in Pane 1.3 to inspect `current_development/scottish_lower/archive/distance/l01_distance_features.jl` and summarize its data structures and coordinate handling.
2. **Review Distance Schema:** Understand the GPS catalog and Haversine function.
3. **Design Gate 2 Distance Test:** Draft a standalone validation script to test Distance feature attachment against Gate 2 rules.

---

*Begin Phase 1 (Distance Feature Investigation).*
