# Tickets

Defects and scoped pieces of work that are **deliberately not being fixed inline**,
because doing so would derail the conversation that found them.

Each ticket is a self-contained brief: a fresh Claude session should be able to open
one file, understand the problem, reproduce it, fix it, and know when it is done —
without reading the conversation that raised it.

## How to use this

**Raising one.** Copy the structure of an existing ticket. It must contain: evidence,
root cause with `file:line`, a reproduction, blast radius, proposed fix with
trade-offs, acceptance criteria, and an explicit scope guard saying what NOT to touch.
Add a row below.

**Working one.** Start a fresh session, point it at the ticket file, and let it work
only that ticket. Update `Status` here when it lands.

**Status values:** `open` · `in progress` · `blocked` · `done` · `wontfix`

## Open

| ID | Title | Severity | Area | Status | Raised |
|---|---|---|---|---|---|
| [T002](T002-scalar-taped-likelihood.md) | Engine likelihoods taped scalar-by-scalar (~20x AD work); `view` defeats vectorisation and NegBin crashes on the fast path | medium | `src/models/pregame/engines/`, `src/MyDistributions/` | open | 2026-08-26 |

## Closed

| ID | Title | Resolution | Closed |
|---|---|---|---|
| [T001](T001-pooled-tournament-clock.md) | Pooled tournament groups use a shared calendar clock with strict kickoff safety | 2026-08-25 |
