# Server & kaimon operating notes

Everything heavy runs on the remote box, not locally. There are **two** ways code gets there,
and they are used for different things.

| Route | Who drives it | Used for |
|---|---|---|
| nvim + `kitty-runner.nvim` → ssh'd terminal → Julia REPL | **James** | Reading and running the walkthrough blocks, poking at results, all MCMC |
| kaimon MCP (`mcp__kaimon-remote__*`) | **Claude** | Verifying cheap gates before James sees the file; never long MCMC |

## The box

- Repo checkout: `/root/BayesianFootball` — a **separate clone** from the local working copy.
- 32 hyperthreads / 16 physical cores; `Threads.nthreads() = 16` in the kaimon sessions.
- Start Julia with `julia --project -t 16`, pin cores (`using ThreadPinning; pinthreads(:cores)`),
  and set `BLAS.set_num_threads(1)` before sampling.
- Prototype artifacts written by this stream go under `data/scottish_lower/<model>/<config_hash>/`.

## Getting local edits onto the server

The server has **no git push credentials**. All commits and pushes happen locally.

```bash
# local
git add … && git commit … && git push
```

```julia
# server (REPL or via kaimon ex)
run(`git -C /root/BayesianFootball pull --rebase --autostash origin design/matchday-layer`)
```

Use `--rebase --autostash` — the server carries a dirty `Manifest.toml`/`Project.toml`, so
`--ff-only` fails. To realign hard after a local push:
`git -C /root/BayesianFootball fetch && git -C /root/BayesianFootball reset --hard origin/<branch>`
(untracked/gitignored `.jls` caches survive this).

Getting artifacts *back* from the server: there is no push route. Small text files come back
directly through an `ex` returning `Text(read(path, String))`; binaries come back base64-encoded
in ≤24k-character slices.

## Session hygiene (REPL, either route)

```julia
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"   # BEFORE `using BayesianFootball`
using BayesianFootball
```

Without that line, Julia 1.12 tries an env-wide auto-precompile which **aborts** on a broken
`LanguageServer` → `JSONRPC` dependency (`UndefVarError: Writer not defined in JSON`).
`BayesianFootball` itself precompiles fine; the failure is an unrelated tooling dep.

Revise does not pick up changes to module `include`s, exports, or struct redefinitions —
restart the session for those.

Before restarting a session, `Serialization.serialize` any expensive object (fitted chains,
extracted latents, PPDs) to a gitignored `.jls`. Reload with `using BayesianFootball` first so
the struct types resolve. Prototype types must have their defining `lXX_*.jl` included before
deserializing, or you get `UndefVarError: <ModuleName> not defined` — that is expected, not
artifact corruption.

## kaimon-specific gotchas (Claude's route)

- **`start_session` — call it once.** The MCP call can time out during the cold precompile while
  the Julia process still spawns. Retrying creates duplicate sessions that contend for CPU.
  If it times out, wait, then probe with a single cheap `ex` (`1+1`).
- **Prefer a warm session.** `ping(extended=false)` lists existing sessions; the pre-existing
  ones already have `BayesianFootball` loaded.
- **`ex` promotes to a background job at 30s.** Poll with `check_eval`.
- **There is a 10-minute no-activity gate.** A long `include` reports
  `Gate eval timed out … no activity for 10m` — **the Julia computation keeps running.**
  To recover the result, submit a trivial `ex` (e.g. `@isdefined(res)`) on the *same* session
  key; it queues behind the running include and returns once that finishes.
- **`println` is stripped** from `ex` output. Return values, or wrap in `Text(...)`. This also
  means progress output does not keep the activity gate alive.
- **`cancel_eval` is cooperative** — a `using Plots` / precompile will not stop and keeps
  blocking new evals. Cancelling mid-`savefig` truncates the PNG to 0 bytes. A
  precompile-poisoned session needs `manage_repl restart` (disk files survive).
- **Database access needs `BF_DB_URL`** in the environment; it is not set in kaimon sessions by
  default. Never write credentials into a source file.

## Division of labour for this stream

Claude verifies Gates 0–5 (config, features, equation parity, gradient diff, extraction parity,
score matrix) over kaimon so that blocks reaching James already run. James launches the Gate 3
smoke and every full grid from his own REPL, where he can watch it.
