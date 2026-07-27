# Prompt: Betfair in-play liquidity audit — Scottish League 1 & 2

*Paste everything below the line into an LLM that has access to the `betdbase` Postgres database.*

---

You have read access to a Postgres database (`betdbase`) holding football match data and Betfair odds.
I need a **liquidity audit** to answer one commercial decision:

> **Are the Betfair in-play markets in Scottish League 1 and League 2 (`tournament_id` 56 and 57) liquid
> enough to trade at all — and do I need to buy a year of advanced (order-book / depth) in-play data to
> find out?**

I currently only have **last-traded-price (LTP)** data, and I suspect I am missing size/depth. My existing
in-play work is on **Ireland**, which does have tradeable markets, so **every number below must be reported
as Scottish (56/57) *versus* Ireland as a benchmark.** A ratio against a league I already trade is far more
informative than an absolute number. (Find Ireland's `tournament_id`s in
`BayesianFootball/src/Data/fetchers/segments.jl` if you have the repo; otherwise ask me.)

## Step 0 — Schema introspection (do this FIRST, do not guess)

Before writing any analytical query, introspect the schema and tell me exactly what exists:

1. List the tables holding Betfair odds and their full column lists (`information_schema.columns`).
2. State explicitly, for each, **which of these columns exist**:
   - last traded price
   - **traded volume / matched size** (this is the critical one)
   - back price / lay price (i.e. a two-sided quote)
   - back size / lay size (i.e. depth)
   - market status / in-play flag
   - tick timestamp, and the field encoding time relative to kickoff
3. Report the table holding match events (goals, cards) and its time field.

**If there is no volume/size column anywhere, say so loudly and stop guessing** — that alone answers half
the question. Report what *is* available.

Known convention from my code: the odds table has `minutes_to_kickoff = (tick_ms − kickoff_ms)/60000`.
Despite the name, **positive = in-play**; the in-play window is roughly `(0, 125]` in wall-clock minutes.
Verify this holds before relying on it.

## Step 1 — Coverage

For tournaments 56, 57, and the Ireland benchmark, over the most recent complete season:

- number of matches
- number and % of matches with **any** in-play ticks
- date range covered

## Step 2 — Tick frequency (is the feed even fast enough?)

Restricted to in-play (`0 < minutes_to_kickoff <= 130`), per league:

- median and p25/p75 **ticks per match**
- median and p90 **gap between consecutive ticks, in seconds** (per market)
- % of in-play ticks where the price **actually changed** vs. repeated the previous value
  (a market that never reprices is a dead market)

## Step 3 — Selections quoted (can I even invert the odds?)

Per league, in-play:

- distinct selections quoted per match, broken down by market type (1X2 / over-under / BTTS / correct score)
- **% of 5-minute in-play bins that carry a full 1X2 (≥ 3 selections, ideally ≥ 6 rows)**

Why this matters: a totals market alone is rank-1 — it constrains only `λ_home + λ_away` and **cannot**
identify the home/away split. I need a full 1X2 (or a handicap) in the same bin or the inversion is
unidentified. If Scottish bins rarely carry a full 1X2, the in-play inversion is impossible there regardless
of liquidity.

## Step 4 — Liquidity (only if a volume/size column exists)

Per league:

- total **matched volume per market per match**, split **pre-game vs in-play**
- distribution (median, p10, p90) of in-play matched volume per market
- the same for the top-of-book **size** if depth is available

If no size/volume column exists: state plainly that **execution cannot be backtested from this data**, and
that any in-play backtest run on LTP is unfalsifiable (I cannot know whether I would have been filled).

## Step 5 — Verdict

Give me a short table, Scottish vs Ireland, on every metric above, plus a **direct recommendation against
these thresholds**:

| Signal | Threshold for "in-play is dead here" |
|---|---|
| Median in-play tick gap | > 60 seconds |
| % in-play bins with a full 1X2 | < 50% |
| Median in-play matched volume per market | small enough that a Kelly stake moves the price |
| % of in-play ticks that reprice | very low (stale/dead book) |

Then answer, in one paragraph:

1. **Is in-play tradeable in Scottish L1/L2 at all**, based on what the data already shows?
2. **Would buying a year of advanced in-play data change that answer**, or would it just confirm a market
   that is too thin to trade regardless?

Be blunt. I would rather hear "these markets are dead, don't spend the money" than get a hedged answer.
If the data cannot settle the question, say exactly which missing field would settle it.

## Constraints

- Show me every SQL query you run.
- Do not fabricate column or table names — introspect first.
- If a query would be expensive, sample and say so.
- Report raw counts alongside percentages; I want to see the `n`.
