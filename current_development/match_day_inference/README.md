# MatchDay live execution

This directory is the active operational suite for MatchDay live execution. It prices an
entire simultaneous fixture slate, records the planned portfolio in the paper ledger, and
presents the slate through the operator console. The slate is the execution atom: reservation
is performed for the whole stake vector in one transaction before individual orders are
submitted.

## Active files

- [`QUICKSTART_LIVE.md`](QUICKSTART_LIVE.md) — operator-oriented quickstart for pricing,
  reserving, submitting, settling, and serving a paper slate.
- [`r06_slate_ledger_console.jl`](r06_slate_ledger_console.jl) — replayable end-to-end Scottish
  slate runner: canonical fit, slate pricing, paper ledger, settlement, and console snapshot.
- [`RESEARCH_MATCHDAY_ARCHITECTURE.md`](RESEARCH_MATCHDAY_ARCHITECTURE.md) — live execution
  design, dataflow, controls, and validation rationale.
- [`AI_AGENT_HANDOVER.md`](AI_AGENT_HANDOVER.md) — system state and operational context for
  follow-on work.

Run the worked example from the repository root:

```bash
julia --project -t 8 current_development/match_day_inference/r06_slate_ledger_console.jl
```

The historical exploratory inference and single-fixture runbook prototypes are retained under
[`current_development/archived/matchday/`](../archived/matchday/):
`legacy_inference/` and `legacy_runbook/`, respectively. They are archival material, not the
active live execution path.
