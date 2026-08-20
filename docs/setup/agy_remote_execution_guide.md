# Antigravity CLI (AGY) Remote Execution & Tmux Protocol Guide

> [!IMPORTANT]
> **READ THIS BEFORE RUNNING CODE ON REMOTE COMPUTE NODES.**
> When operating as an Antigravity CLI (AGY) agent in this repository, heavy MCMC sampling and evaluations are executed on dedicated remote compute servers (`mcmc-beast`) connected via nested `tmux` panels, while PostgreSQL data services reside on `archpc`. Follow this protocol strictly to edit, sync, launch, monitor, and capture code execution safely.

---

## 1. Network & Infrastructure Topography

```
+-----------------------------------------------------------------------------------------+
|                                    TAILSCALE MESH                                       |
|                                                                                         |
|   +--------------------------+       Git Push / Pull       +------------------------+   |
|   |       LOCAL HOST         | ==========================> |       MCMC-BEAST       |   |
|   |  (AGY Agent CLI Host)    |                             |  (High-Perf Compute)   |   |
|   |  /home/james/bet_project/| <========================== |  /root/BayesianFootball|   |
|   |      BayesianFootball    |       Git Push / Pull       |  AMD Ryzen 9 (32 thr)  |   |
|   +--------------------------+                             +------------------------+   |
|                 |                                                      |                |
|                 | tmux send-keys / capture-pane                        | Database Queries|
|                 v                                                      v                |
|   +--------------------------+                             +------------------------+   |
|   |  Outer Tmux Session      |                             |         ARCHPC         |   |
|   |  `scottish_runner:1.1`   |                             |   (Postgres Server)    |   |
|   |  (SSH to mcmc-beast)     |                             |   betdb on port 5433   |   |
|   +--------------------------+                             +------------------------+   |
+-----------------------------------------------------------------------------------------+
```

### Machine Specifications & Endpoints
1. **Local Host (`/home/james/bet_project/BayesianFootball`):**
   - Development workstation where AGY runs.
   - All code edits (`lXX_*.jl`, `rXX_*.jl`, `.md` notes) must be written and committed here.
2. **Compute Server (`mcmc-beast`):**
   - **Role:** Dedicated NUTS/ADVI MCMC sampling and evaluation node.
   - **Hardware:** AMD Ryzen 9 (32 physical/logical cores, 64 GB RAM).
   - **SSH Endpoint:** `root@mcmc-beast` (via Tailscale / LAN).
   - **Repo Path:** `/root/BayesianFootball`.
3. **Database Server (`archpc`):**
   - **Role:** PostgreSQL database host for `betdb`.
   - **DB Endpoint:** `postgresql://admin:CpPhGzIZ2qHtAh6cJT%2FHHFovs0CqfTx6@archpc:5433/betdb`.
   - **Port:** `5433` (accessible over Tailscale/LAN).

---

## 2. Pre-Flight Connectivity & Health Checks

Before launching runs, execute these fast diagnostic checks from bash:

```bash
# 1. Check Tailscale Mesh Connectivity
tailscale status | grep -E "mcmc-beast|archpc"

# 2. Check Ping Latency
ping -c 1 mcmc-beast
ping -c 1 archpc

# 3. Check Postgres DB Port Reachability
nc -zv -w 3 archpc 5433
```

> [!TIP]
> If Postgres on `archpc` is temporarily unreachable, ensure runner scripts use cached DataStores (`Data.load_datastore_cached(..., max_age_hours = 10000)`) which load from `.jls` caches on disk without opening SQL connections.

---

## 3. Tmux Session Hierarchy & Window Map

The remote environment uses a **nested tmux hierarchy**:

### Outer Tmux Session: `scottish_runner`
- **Pane `scottish_runner:1.1`:** SSH terminal session connected into `mcmc-beast`.
- **Pane `scottish_runner:1.2`:** AGY CLI session.

### Inner Tmux Session (inside `mcmc-beast`):
Inside `scottish_runner:1.1`, an active tmux session named `[julia]` manages running processes:
- **Window `0:julia` or `3:julia*`:** Persistent Julia REPL launched with target threads (e.g. `julia --project -t 16` or `julia --project -t 32`) with `ThreadPinning.pinthreads(:cores)` initialized.
- **Window `1:btop`:** Live CPU core utilization, thermals, and RAM monitor.
- **Window `2:bash`:** Shell prompt on `root@mcmc-beast:~/BayesianFootball`.

---

## 4. AGY Standard Operating Procedure (SOP)

Follow this 5-step loop for every prototyping or grid execution cycle:

```
[Step 1: Edit Locally]  ---> [Step 2: Commit & Push]  ---> [Step 3: Pull on Beast]
                                                                  |
[Step 5: Capture & Eval] <--- [Step 4: Execute on Beast] <---------+
```

### Step 1: Write/Edit Code Locally
Always write models (`lXX_*.jl`) and runners (`rXX_*.jl`) locally in `current_development/<league>/...`.

### Step 2: Commit and Push Locally
```bash
git add current_development/<league>/<dir>/*.jl
git commit -m "feat(<scope>): add new runner/model"
git push origin <active-branch>
```

### Step 3: Switch to Server Shell (Window 2) & Git Pull
Send `C-b 2` to the inner tmux session to switch to `bash`, then pull:
```bash
tmux send-keys -t scottish_runner:1.1 C-b 2 && sleep 1 && \
tmux send-keys -t scottish_runner:1.1 "git pull origin <active-branch>" C-m && sleep 2 && \
tmux capture-pane -p -t scottish_runner:1.1 -S -20
```

### Step 4: Switch to Julia REPL (Window 3) & Run Code
Send `C-b 3` to the inner tmux session to switch to Julia, then execute the runner:
```bash
tmux send-keys -t scottish_runner:1.1 C-b 3 && sleep 1 && \
tmux send-keys -t scottish_runner:1.1 'include("current_development/<league>/<dir>/rXX_runner.jl")' C-m
```

### Step 5: Screen Capture & Monitor Execution
Capture pane output to check logs without interrupting the Julia process:
```bash
# Capture the last 50 lines of output
tmux capture-pane -p -t scottish_runner:1.1 -S -50

# Wait for a long task and capture
sleep 60 && tmux capture-pane -p -t scottish_runner:1.1 -S -50
```

---

## 5. Monitoring CPU & Thread Utilization in `btop`

To inspect whether MCMC sampling is fully utilizing all CPU cores:

1. **Switch to `btop` (Window 1):**
   ```bash
   tmux send-keys -t scottish_runner:1.1 C-b 1 && sleep 1 && \
   tmux capture-pane -p -t scottish_runner:1.1 -S -25
   ```
2. **Verify Thread Utilization:**
   - All 16 or 32 physical CPU cores should be pinned at $\approx 100\%$ CPU utilization.
   - If only 1 core is active during MCMC, verify `QueuedNUTSConfig(max_concurrent_tasks = 16)` is configured and Julia was launched with `-t 16`.
3. **Switch Back to Julia (Window 3):**
   ```bash
   tmux send-keys -t scottish_runner:1.1 C-b 3
   ```

---

## 6. Safe Long-Running Job Monitoring (No Polling Loops)

When monitoring long-running grid runs (10 mins to several hours):
- **Never poll in tight loops.**
- Use `sleep <N>` combined with `tmux capture-pane` inside `run_command` with a suitable `WaitMsBeforeAsync`.
- Once launched as a background task, inform the user and end your turn to await completion notifications.

---

## 7. Recovery & Troubleshooting

### Problem A: SSH Connection Dropped ("Broken Pipe")
If the SSH session in pane `1.1` disconnects:
1. Re-establish SSH connection in pane `1.1`:
   ```bash
   tmux send-keys -t scottish_runner:1.1 "ssh root@mcmc-beast" C-m && sleep 2
   ```
2. Re-attach to the inner tmux session on `mcmc-beast`:
   ```bash
   tmux send-keys -t scottish_runner:1.1 "tmux attach -t 0 || tmux a" C-m && sleep 2
   ```

### Problem B: Julia Process Interrupted / Out of Memory
If a run errors or needs a clean restart:
1. Switch to window 3 (`C-b 3`).
2. Send `C-c` (SIGINT) to interrupt current evaluation:
   ```bash
   tmux send-keys -t scottish_runner:1.1 C-c
   ```
3. To reload code with `Revise`:
   ```bash
   tmux send-keys -t scottish_runner:1.1 'using Revise; using BayesianFootball' C-m
   ```

### Problem C: Nested Keybinding Confusion
Remember:
- In nested tmux, sending keys to `scottish_runner:1.1` forwards commands directly to the active inner pane on `mcmc-beast`.
- `tmux send-keys -t scottish_runner:1.1 C-b <N>` controls the **inner** tmux window selection (`1` for btop, `2` for bash, `3` for julia).
