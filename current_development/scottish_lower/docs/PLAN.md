# Plan of record — Scottish Lower pregame modelling

**Agreed:** 2026-08-25 (James + Claude). Supersedes any plan implied by `archive/`.

## Why this exists

Fast agent-written progress under `current_development/scottish_lower/` produced a leaderboard
that could not be trusted: a 2026-08-24 audit found the reported champion was priced with zero
team attack/defence effects and dropped hierarchical scales, plus several temporal leaks. The
code was also unreadable as research — James could not open a file, run it block by block, and
see what it claimed.

The response is not a better model. It is a **method**: every model walks the same seven gates,
in a file James drives himself from nvim/kitty-runner into a REPL on the server. Claude is a
coding partner, not the operator.

## Decisions

| Decision | Choice |
|---|---|
| Build order | `01_team_poisson` → `02_apm_player_poisson` → `03_open_play_recombination` (open-play last: biggest, most variants) |
| Baseline | The existing `src` engine (`DynamicGoalsTimeDecayModel`), driven through the protocol. Prototypes **dispatch into the package API**, never re-implement it |
| APM | Rebuilt from scratch — the graduated `src/features/plus_minus/` RAPM is **not** trusted until it passes the gates. Four rating variants (goals, shots, SoT, BBC proxy-xG) as configs in one folder |
| Open play | Retrofit of `archive/open_play_rebuild`. Its Stage 8 chains (38 folds, 0 divergences) may be reused **only after** Gates 4–5 pass against them |
| Execution | Claude writes and verifies Gates 0–5 over kaimon; James launches all MCMC |
| Layout | One folder per model + `_protocol/` for the shared contract + `docs/` for long-term memory |
| Held-out | `24/25` for all development and selection. `25/26` sealed |
| Gate 7 scope | Growth/CLV on `24/25` only for now, to keep iteration fast. `25/26` opened later, once |
| Book | 1X2, O/U 0.5 / 1.5 / 2.5 / 3.5, BTTS — the liquid markets only. No CorrectScore |
| Primary metric | CLV vs Betfair close. Growth `G` secondary. Proper scores per line, never aggregated across a market's selections |
| Smoke | Gate 3 ends by **persisting a real chain** through `src/experiments`; Gate 4 loads that chain |

## Sampler conventions

| Setting | Smoke (Gate 3) | Full grid |
|---|---|---|
| Chains | 4 | 4 |
| Warmup / retained | 500 / 500 | 800 / 800 |
| Folds | 1 | all `24/25` folds |
| Queue tasks | — | 16 |

Fixed seeds everywhere. Config hash in every artifact path.

## Roadmap

**Phase 0 — baseline walkthrough.** `_protocol/config.jl` (the contract only) plus
`01_team_poisson/v01_walkthrough.jl` covering Gates 0–5, with **the gates written inline and
concrete** against the `src` engine. Walking these over existing `src` code doubles as James
reading the workflow end-to-end. Any gate that fails here is a real finding about `src`.

Gate helpers are **not** written up front. See "Abstraction order" below.

**Phase 1 — baseline to a number.** Gates 6–7 on `24/25`. Produces the reference opponent that
every later model is compared against on identical fixtures.

**Phase 2 — APM.** Rebuild the ratings. The leakage risk is in the ridge fit, not the Turing
model: Gate 2's perturbation test is run **on the ratings themselves**.

**Phase 3 — open play.** Gates 4–5 against the saved Stage 8 chains; if parity passes, skip
resampling and go straight to Gates 6–7.

## Abstraction order

`_protocol/` starts holding **only the contract** — seasons, tournaments, splitter settings,
book spec, artifact root, seeds. Those are model-independent decisions that already exist.

Gate *helpers* are written inline in the first model that needs them, where their real
signatures and comparison semantics are visible. A helper is lifted into `_protocol/` only when
a **second** model needs it. On lifting, the first model is re-pointed at the lifted version and
its gates re-run: the output must be **identical**, or the abstraction dropped something.

This is deliberate. Writing a gate framework before any model exists is how the previous round
produced code nobody could read or check.

## Open items

- Baseline component selection (interception / dispersion / home-advantage / dynamics half-life)
  is presented as a menu in walkthrough block ①, with a stated default, rather than fixed here.
- Whether the `_protocol/` gate helpers eventually graduate to `src/` as a validation module.
- `25/26` unsealing — deliberately deferred.
