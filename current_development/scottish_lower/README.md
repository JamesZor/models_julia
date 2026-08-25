# Scottish Lower Leagues (tournaments 56 & 57)

Pregame modelling for Scottish League One (`56`) and League Two (`57`).

This directory was restructured on **2026-08-25**. Everything produced before that date
now lives under [`archive/`](archive/) and is reference material only — see
[`docs/ARCHIVE_NOTES.md`](docs/ARCHIVE_NOTES.md) for what is trusted and what is not.

## Start here

| If you want to... | Read |
|---|---|
| Understand the method | [`docs/PROTOCOL.md`](docs/PROTOCOL.md) |
| Build or extend a model | [`docs/WORKFLOW.md`](docs/WORKFLOW.md) |
| Run anything on the server | [`docs/SERVER_AND_KAIMON.md`](docs/SERVER_AND_KAIMON.md) |
| See what we have learned so far | [`docs/FINDINGS_INDEX.md`](docs/FINDINGS_INDEX.md) |
| Know why the old work is quarantined | [`docs/ARCHIVE_NOTES.md`](docs/ARCHIVE_NOTES.md) |

## Layout

```
scottish_lower/
├── docs/               Long-term memory: method, workflow, server notes, findings ledger
├── _protocol/          Shared, model-agnostic gate code (config, features, sampling, extraction, score matrix, eval)
├── 01_team_poisson/    Model 1 — team-level Poisson/NegBin baseline (wraps the src engine)
├── 02_apm_player_poisson/   Model 2 — player-level, own APM ratings (goals / shots / SoT / proxy-xG)
├── 03_open_play_recombination/  Model 3 — Y_total = Y_open_play + Y_pen + Y_own_goal
└── archive/            Pre-2026-08-25 work. Reference only. Do not reuse its leaderboards.
```

## The rule

A model is not "done" because it ran. It is done when it has passed gates 0–7 in
[`docs/PROTOCOL.md`](docs/PROTOCOL.md) and every result is written into that model's
`FINDINGS.md` next to its config hash.
