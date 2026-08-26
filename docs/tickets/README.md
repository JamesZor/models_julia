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
| [T003](T003-home-advantage-population-fallback.md) | Unmapped teams silently lose home advantage at extraction (λ_h 0.849x); 28 call sites | medium | `src/models/pregame/engines/`, `src/models/pregame/components/home_advantage.jl` | open | 2026-08-26 |
| [T004](T004-1x2-grading-disagrees-with-score.md) | `is_winner` contradicts the recorded score on 3 fixtures (2-2 draws with no 1X2 winner); no QA invariant catches it | low | `ds.odds` grading | open | 2026-08-26 |
| [T005](T005-betfair-summariser-drops-90pc.md) | `summarize_betfair_market` inner-joins an open window and silently returns 30 of 360 fixtures; breaks CLV | high | `src/Data/betfair_util.jl` | open | 2026-08-26 |

## Closed

| ID | Title | Resolution | Closed |
|---|---|---|---|
| [T001](T001-pooled-tournament-clock.md) | Pooled tournament groups use a shared calendar clock with strict kickoff safety | 2026-08-25 |
