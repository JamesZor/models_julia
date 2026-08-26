# Dynamic Tmux-Native Agent Mesh & Live REPL Architecture

## 1. Objective
Design and prototype an elastic, tmux-native multi-agent swarm architecture where persistent `pi` (or `agy`) instances run in dynamically managed tmux panes, communicating via deterministic IPC primitives rather than ephemeral CLI subagent tool loops.

---

## 2. System Topology & Role Specialization

```
                               ┌────────────────────────┐
                               │   User + Assistant     │  ◄── (Human + Pair Programmer)
                               │  (Prompting / Review)  │      High-level intent & review
                               └───────────┬────────────┘
                                           │
                               ┌───────────▼────────────┐
                               │ Orchestrator / Manager │
                               │  (Task DAG & Dispatch) │
                               └─────┬──────┬──────┬────┘
                                     │      │      │
          ┌──────────────────────────┘      │      └──────────────────────────┐
          ▼                                 ▼                                 ▼
┌──────────────────┐             ┌─────────────────────┐             ┌──────────────────┐
│  Scout (Singleton│             │   Dynamic Workers   │             │ Reviewer / Gate  │
│  Hot KV-Cache /  │ ◄─────────► │ (Worker 1..N on-dm) │ ──────────► │ (1 Dedicated QA/ │
│  Codebase Index) │             │ (Edit/Build/Fix)    │             │  Gate Verifier)  │
└──────────────────┘             └──────────┬──────────┘             └──────────────────┘
                                            │
                                 ┌──────────▼──────────┐
                                 │ Live REPL / Server  │
                                 │ (Julia / DB / GPU)  │
                                 └─────────────────────┘
```

### Roles Breakdown:
1. **Assistant (Interface / Co-Pilot):** User-facing pair programmer who formulates prompts, manages swarm directives, and evaluates outputs.
2. **Orchestrator (Manager):** Decomposes user goals into milestones, manages worker lifecycles, and routes messages.
3. **Scout (Persistent Singleton):** Holds entire repository context in hot KV cache; answers fast architectural/file lookups for workers so workers don't repeatedly read 50k tokens of files.
4. **Workers (Dynamic 0..N):** Spun up on demand in new tmux panes for parallel implementations, and retired or recycled when finished.
5. **Reviewer / QA (Singleton):** Inspects diffs and verifies gate compliance against `GATES.md`.
6. **Compute Runners (On-Demand):** Live interactive REPL sessions (e.g. Julia on `mcmc-beast:32t`, PostgreSQL shell) controlled via `send-keys`/`capture-pane` without heavy MCP overhead.

---

## 3. Deterministic Tooling & CLI Primitives to Build

Design and implement lightweight CLI/extension commands in `~/.pi/agent/bin/` or `~/.pi/agent/extensions/`:
1. `agent-spawn <role> [name]`: Dynamically launches a persistent `pi` pane in tmux with appropriate role instructions and footer.
2. `agent-send <target_pane> "<message>"`: Safely sends structured prompts/data to target stdin.
3. `agent-capture <target_pane> [--wait-idle]`: Captures output and detects thinking/idle state.
4. `agent-kill <target_pane>`: Gracefully terminates worker panes.
5. `agent-list`: Returns active agents, roles, and status (pane ID, model, state).

---

## 4. Key Architectural & Feasibility Questions to Answer
1. **Concurrency & Synchronization:** How does `pi` handle background synchronization when multiple panes communicate concurrently?
2. **Deadlock Prevention:** What is the best pattern to prevent circular review/worker loops?
3. **REPL Interactivity:** How can workers reliably detect REPL completion / errors when interacting with interactive Julia or Python shells without MCP servers?
4. **Token Economics:** Comparative token consumption and caching analysis between ephemeral tool subagents vs persistent tmux-native mesh.

---

## 5. Deliverables
1. **Architecture Specification Document:** Detailed design with state machine diagrams and message contracts.
2. **Shell Utilities / `pi` Extension Prototype:** Production-ready scripts/extensions implementing `agent-spawn`, `agent-send`, `agent-capture`, `agent-kill`, and `agent-list`.
3. **Dynamic Bootstrap Demo:** A runnable demonstration script spinning up a Scout + Worker + REPL session.
