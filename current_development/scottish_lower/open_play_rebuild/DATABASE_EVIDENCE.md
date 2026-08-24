# Read-only cross-source database evidence

**Scope:** bounded, read-only PostgreSQL research over Scottish Lower tournament IDs 56 and 57. Connection credentials are deliberately not stored here. Queries used a short statement timeout and read-only transactions.

## Source topology

- SofaScore final scores and teams: `sofascore.matches`
- SofaScore incidents and raw incident JSON: `sofascore.match_incidents`
- BBC match metadata/final scores: `bbc.match_meta`
- BBC commentary and event text: `bbc.live_text`
- BBC/SofaScore match join: shared `match_id`
- BBC team slug crosswalk: `bbc.team_map`

No separate match mapping table is required: BBC match records reference the SofaScore match ID.

## Final-score disagreement

Across tournaments 56 and 57, only two BBC-covered matches had `bbc.match_meta.scores_match=false`:

| Match | Tournament/season | SofaScore | BBC |
|---:|:---|:---:|:---:|
| `11395473` | 57 / 23/24 | 2–0 | 2–1 |
| `10387906` | 56 / 22/23 | 1–1 | 2–1 |

These remain provider disagreements. BBC commentary is corroborating evidence, not an automatic override of the model’s selected official-score source.

## Stage-2 quarantined matches

### `11395473` — The Spartans vs Elgin City

SofaScore official score is 2–0, but its incident stream contains three regular goals with score progression 1–0, 2–0, 2–1. BBC commentary also records the same three scorers and reports 2–1. No BBC text explicitly says disallowed, offside, VAR, overturned, or cancelled. Because the configured official SofaScore score and component stream disagree, the match remains quarantined rather than selecting a provider ad hoc.

### `12477131` — Cove Rangers vs Annan Athletic

SofaScore and BBC agree on 1–3. SofaScore incident `206671` duplicates incident `206670`: both are home regular goals at minute 86 with the same 1–2 score state, while `206671` lacks a player and has no BBC counterpart. This is strong evidence of a duplicate incident row. V1 still quarantines the match unless a reviewed semantic-deduplication rule is added; provider IDs alone are distinct.

## Own-goal side semantics

The database contains 110 Scottish Lower `goal/ownGoal` incidents. Bounded recent examples joined to score progression and BBC commentary support SofaScore `is_home` as the **beneficiary/scoring side**, not the own-goal player’s committing side. This agrees with the Stage-2 split result: 39 informative matches supported beneficiary and zero supported committing-side semantics.

BBC own-goal commentary often lacks a populated team field, so mandatory score reconciliation remains the governing QA check.

## Penalty semantics

SofaScore Scottish Lower incident counts:

| Incident | Count |
|:---|---:|
| `goal` with `incidentClass='penalty'` | 425 |
| `inGamePenalty` with `incidentClass='missed'` | 126 |

Thus the auditable outcome definition is:

```text
converted penalties C = goal/penalty
unsuccessful penalties U = inGamePenalty/missed
awarded penalties A = C + U
```

BBC separately labels `penalty_awarded`, `penalty_conceded`, `penalty_saved`, and `penalty_missed`. SofaScore `inGamePenalty/missed` includes kicks BBC calls `penalty_saved`; it means **not converted**, not necessarily off-target. This supports the V1 Binomial conversion likelihood `C | A,q_pen`.

## Reproducibility policy

Every database finding must retain source table, match/incident IDs, join key, and bounded SQL. Database evidence may explain or quarantine a row but must not silently rewrite the immutable modelling snapshot. Any correction/deduplication policy requires a versioned rule and rerun of component reconciliation.
