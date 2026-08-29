# AGENTS.md

> **AI Agent Orchestration, Infrastructure, and Tmux Tooling Guide**  
> Guidance for AI agents (Antigravity CLI, Claude Code, etc.) operating across the `BayesianFootball.jl` distributed mesh.

---

## 1. Quick Reference & Core Guides

* **AI Agent Infrastructure & Context Guide:** [`docs/architecture/ai_agent_infrastructure_and_execution_context.md`](file:///home/james/bet_project/BayesianFootball/docs/architecture/ai_agent_infrastructure_and_execution_context.md)
* **Tmux Subagent & Persistent REPL Control Guide:** [`docs/setup/agy_tmux_agent_and_repl_control_guide.md`](file:///home/james/bet_project/BayesianFootball/docs/setup/agy_tmux_agent_and_repl_control_guide.md)
* **Remote Execution & Tmux Protocol:** [`docs/setup/agy_remote_execution_guide.md`](file:///home/james/bet_project/BayesianFootball/docs/setup/agy_remote_execution_guide.md)
* **ReverseDiff AD Performance & Safety Guide:** [`docs/turing_ad_performance_guide.md`](file:///home/james/bet_project/BayesianFootball/docs/turing_ad_performance_guide.md)
* **Julia Coding Context for AI Agents:** [`docs/guides/julia_coding_context_for_agents.md`](file:///home/james/bet_project/BayesianFootball/docs/guides/julia_coding_context_for_agents.md) — language traps, style, Turing API facts, verification ladder. **Read before writing Julia.**

---

## 2. Infrastructure & Compute Rules

1. **Topology:**
   - **Local Laptop:** Development workstation (`/home/james/bet_project/BayesianFootball`).
   - **Compute Node (`mcmc-beast`):** 16 Physical Cores (32 SMT threads), 64GB RAM (`/root/BayesianFootball`).
   - **Database Host (`archpc:5433`):** PostgreSQL `betdb`.

2. **CPU & Threads:**
   - Always launch Julia with `-t 16` on `mcmc-beast`.
   - Always run `using ThreadPinning; pinthreads(:cores)` before starting MCMC chains.
   - Always set `LinearAlgebra.BLAS.set_num_threads(1)` during sampling to prevent CPU oversubscription.
   - Ensure local inference daemons (like `ollama`) remain disabled (`systemctl disable/stop ollama`).

3. **Code Syncing & Cache Safety:**
   - Use `rsync -avz --exclude '.cache/' --exclude 'data/' ...` to push code without clobbering remote data caches.
   - Ensure Point-In-Time (PIT) feature guards accept match-row values: `stamp_ok = (stamp === nothing) || (at === nothing) || (stamp < at)`.

---

## 3. Agent-to-Agent & REPL Tmux Tooling

### Controlling Claude Subagents
```bash
# Send prompt to Claude Code subagent
tmux send-keys -t features:1.1 'Run r00_explore_poisson_models.jl on Fold 1 and report parameter posteriors.' C-m

# Inspect subagent scrollback / status
tmux capture-pane -t features:1.1 -p -S -50
```

### Controlling Persistent Julia REPL (Zero-TTFX)
```bash
# Send code evaluation into warm REPL
tmux send-keys -t scottish_runner:1.1 'include("current_development/scottish_lower/r00_explore_poisson_models.jl")' C-m

# Inspect REPL output
tmux capture-pane -t scottish_runner:1.1 -p -S -60
```
