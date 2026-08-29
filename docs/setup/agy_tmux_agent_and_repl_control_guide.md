# Antigravity (AGY) Tmux Agent & Julia REPL Control Guide

> **Protocol & Tool Reference:** Controlling Claude Code subagents and persistent remote Julia REPL sessions via Tmux.  
> **Status:** Active Standard | **Primary Use Cases:** AI-to-AI orchestration, subagent delegation, persistent Julia REPL hot-evaluation.

---

## 1. Overview & Architecture

When Antigravity (AGY) operates as the primary coordinator, it controls two types of remote interactive sessions hosted within `tmux`:
1. **Claude Code Subagents (e.g. `features:1.1`, `scottish_runner:1.1`):** Autonomous AI pair programmers running inside a terminal CLI.
2. **Persistent Julia REPLs (e.g. `scottish_runner:1.1` on `mcmc-beast`):** Long-running, warm Julia sessions where `BayesianFootball`, `Revise`, and `Turing` remain compiled in memory.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 ANTIGRAVITY COORDINATOR (AGY)                           │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │
               ┌────────────────────────────┴────────────────────────────┐
               ▼                                                         ▼
┌──────────────────────────────┐                         ┌──────────────────────────────┐
│     CLAUDE CODE SUBAGENT     │                         │    PERSISTENT JULIA REPL     │
│   (e.g. `features:1.1`)      │                         │  (`mcmc-beast:0/3:julia`)    │
├──────────────────────────────┤                         ├──────────────────────────────┤
│ • Interactive CLI agent      │                         │ • Warm JIT / Revise cache    │
│ • Edits, reviews, runs bash  │                         │ • 16 Cores Pinned (:cores)   │
│ • Controlled via send-keys   │                         │ • Instant TTFX evaluation    │
│ • Monitored via capture-pane │                         │ • Evaluates lXX / rXX scripts│
└──────────────────────────────┘                         └──────────────────────────────┘
```

---

## 2. Controlling Claude Code via Tmux

### A. Session Identification & Target Format
Tmux panes are addressed using standard `<session>:<window>.<pane>` notation:
* Target Claude Code session: `features:1.1` or `scottish_runner:1.2`.

### B. Sending Instructions to Claude Code
To send an instruction or prompt to Claude Code:
```bash
tmux send-keys -t features:1.1 'Please review the Fold 1 results in r00_explore_poisson_models.jl and run tests.' C-m
```
* `C-m` is the carriage return (Enter key).
* Wrap instructions in single quotes (`'...'`) to preserve newlines and special characters.

### C. Reading & Monitoring Claude Code State
Capture recent terminal scrollback to inspect Claude's progress:
```bash
# Capture the last 50 lines without formatting noise
tmux capture-pane -t features:1.1 -p -S -50
```

### D. Interpreting Claude Code Terminal States

| UI Indicator / Output | State Description | AGY Action |
| :--- | :--- | :--- |
| `❯ ` (flashing prompt) | **Idle / Ready for input**: Claude finished its task and awaits instructions. | Safe to send new prompt. |
| `* Actioning...` / `✢ Burrowing...` | **Thinking / Reasoning**: Claude is analyzing tools and formulating a plan. | Wait / Sleep and check back. |
| `● Running <command>...` | **Tool Execution**: Claude is executing a shell command, test, or MCMC job. | Wait for command exit code. |
| `❯ Press up to edit queued messages` | **Queued Message**: Instruction was received while Claude was busy and is queued. | Message will dequeue automatically. |
| `✻ Brewed / Crunched for ...` | **Step Completed**: Claude finished a chain of tool calls and printed its response. | Capture output and parse findings. |

---

## 3. Controlling Persistent Julia REPL as an Execution Engine

### Why Persistent REPL Control?
* **Zero TTFX (Time-To-First-Execution):** Cold CLI invocations (`julia --project -e '...'`) re-import heavy packages (`Turing`, `ReverseDiff`, `DataFrames`) every run (~30–50s overhead).
* **Hot-Reloading with `Revise.jl`:** Code edits in `src/` or `current_development/` reflect instantly in the REPL without restarting Julia.

### A. Initializing the Remote REPL on `mcmc-beast`
Inside the tmux window for Julia (e.g. window `0` or `3` on `mcmc-beast`):
```julia
using Pkg; Pkg.activate(".")
using Revise
using BayesianFootball
using ThreadPinning; pinthreads(:cores)
using LinearAlgebra; BLAS.set_num_threads(1)
```

### B. Executing Code in the REPL via Tmux
1. **Executing a Script:**
   ```bash
   tmux send-keys -t scottish_runner:1.1 'include("current_development/scottish_lower/r00_explore_poisson_models.jl")' C-m
   ```
2. **Executing a Specific Function / Gate:**
   ```bash
   tmux send-keys -t scottish_runner:1.1 'results = sl_run_experiment(SL_ALL_DS, TP04Adapter(), SL_ALL_CONTRACT; smoke=true)' C-m
   ```
3. **Interrupting a Long-Running Job:**
   ```bash
   tmux send-keys -t scottish_runner:1.1 C-c
   ```

### C. Capturing REPL Output & Return Values
Capture the last $N$ lines to verify execution status and extract values:
```bash
tmux capture-pane -t scottish_runner:1.1 -p -S -60
```
* **Success Check:** Prompt returns to `julia> `.
* **Error Detection:** Look for `ERROR: ...` and `Stacktrace:`.

---

## 4. Helper Shell Functions for Tool Wrapping

You can define these lightweight wrapper functions in your shell or script templates:

### 1. `agent-send` (Send instruction to Claude Code)
```bash
agent_send() {
    local target="$1"
    shift
    local msg="$*"
    tmux send-keys -t "$target" "$msg" C-m
}
```

### 2. `agent-capture` (Capture sanitized output)
```bash
agent_capture() {
    local target="$1"
    local lines="${2:-50}"
    tmux capture-pane -t "$target" -p -S -"$lines"
}
```

### 3. `repl-eval` (Execute command in persistent Julia REPL)
```bash
repl_eval() {
    local target="$1"
    local expr="$2"
    tmux send-keys -t "$target" "$expr" C-m
}
```

### 4. `repl-wait-idle` (Wait until REPL returns to `julia>`)
```bash
repl_wait_idle() {
    local target="$1"
    local timeout="${2:-300}"
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if tmux capture-pane -t "$target" -p -S -5 | grep -q "julia>"; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo "Timeout waiting for REPL idle on $target" >&2
    return 1
}
```

---

## 5. Best Practices & Rules of Engagement

1. **Avoid Input Collisions:** Never send keys to a tmux pane while an interactive editor (e.g. `vim`, `nano`) is open or while another prompt is half-typed.
2. **Use Bracketed Paste / Includes for Large Code:** Do not paste raw 200-line blocks directly into the terminal with `send-keys`. Always save the code to a file (`rXX_*.jl`) and send `include("...")` to the REPL.
3. **Preserve Pinned Threads:** If a REPL is restarted, always re-run `using ThreadPinning; pinthreads(:cores)` before starting MCMC chains.
4. **Scout File-Drop Pattern:** For long analytical reports from subagents, instruct them to write to a structured markdown file (e.g. `/tmp/report.md`) instead of printing 500 lines to the tmux buffer.
