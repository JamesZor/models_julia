# T005 — `summarize_betfair_market` discards 90% of fixtures by requiring an open price

| | |
|---|---|
| **Status** | open |
| **Severity** | high — silently removes most of the evidence from any CLV or backtest work |
| **Area** | `src/Data/betfair_util.jl:178` |
| **Raised** | 2026-08-26, by model 01 gate 6b in `current_development/scottish_lower/` |
| **Verified on** | Scottish 56+57, 24/25, 360 OOS fixtures, 599,529 Betfair ticks |

## Summary

`summarize_betfair_market` builds an **open** summary and a **close** summary and
`innerjoin`s them. Any match without a traded price in the open window — 24 hours before
kickoff — is therefore dropped **entirely**, including its closing price.

Minor-league exchange markets routinely do not open that early. On Scottish 56+57 the
function returns closing prices for **30 of 360** fixtures, when 322 are available.

Nothing warns. The output is a well-formed DataFrame; it is simply almost empty.

## Evidence

`ds.betfair_odds` holds 599,529 ticks covering 324 of the 360 OOS fixtures.

| summarisation | fixtures (all) | **of the 360 OOS fixtures** |
|---|---|---|
| close window `(-20, 0)` alone | 1,627 | **322** (89%) |
| open window `(-1440, -1380)` alone | 212 | **35** (10%) |
| `summarize_betfair_market` (inner join of both) | 188 | **30** (8%) |

Downstream, joining a model book of 4,680 rows against that Betfair book left **96 rows
across 30 fixtures** — a 98% loss, with every remaining number looking perfectly healthy.

## Root cause

`src/Data/betfair_util.jl:178-205`:

```julia
df_open  = summarize_odds(df_long, estimator, window=open_window)    # (-1440, -1380)
df_close = summarize_odds(df_long, estimator, window=close_window)   # (-20, 0)

final_df = innerjoin(df_open[...], df_close[...],
                     on = [:match_id, :market_name, :market_line, :selection])
```

The join is the defect. The open price is needed only for line-movement metrics; every
consumer that wants a closing price pays for it with 90% of its data.

Note also that the windows are in `minutes_to_kickoff`, which on this feed is **wall-clock,
not match-clock**. That is fine for a pre-kickoff window but is a known trap in-play, and is
documented elsewhere in the research notes.

## Blast radius

Anything asking Betfair for a closing price: CLV, growth backtests, the market pillar when
anchored to Betfair, and gate 7 of the Scottish Lower protocol — where Betfair close is the
**primary discriminator**. Gate 7 would have run on 30 fixtures rather than 322 and reported
nothing unusual.

The effect is worst exactly where the exchange is thin, which is where minor-league research
lives.

## Proposed fix

Make the open side optional and default to close-only:

```julia
summarize_betfair_market(ds; require_open::Bool = false, ...)
```

with `leftjoin` (close as the left side) rather than `innerjoin`, so `odds_open` and
`overround_open` come back `missing` when the market had not opened, instead of deleting the
row. Consumers that genuinely need movement can filter on `!ismissing(odds_open)` and see
what that costs them.

Consider also whether `(-1440, -1380)` is the right open window for minor leagues at all; a
first-traded-price estimator would be more robust than a fixed 60-minute band a day out.

## Reproduction

```julia
D = BayesianFootball.Data
ds = D.load_datastore_cached(D.ScottishLower())
close_only = D.summarize_odds(ds.betfair_odds, D.TWAEstimator(); window = (-20.0, 0.0))
helper     = D.summarize_betfair_market(ds)

length(unique(close_only.match_id))   # 1627
length(unique(helper.match_id))       # 188
```

## Acceptance criteria

- [ ] Close-only summarisation returns ≥ 89% of OOS fixtures on Scottish 56+57 through the
      public helper, not only through `summarize_odds` directly.
- [ ] Rows are never dropped for missing OPEN data; `odds_open` is `missing` instead.
- [ ] Existing callers that use `odds_open` still work, or are updated with the change.
- [ ] Coverage is reported (or a warning raised) when a window matches fewer than some share
      of requested fixtures, so silent near-empty output cannot recur.
- [ ] All 403 package tests pass.

## Scope guard

Summarisation windows and joining only. Do not change the estimators, the tick ingestion, or
the grading logic. Do not fold in T004 (odds grading) — different file, different failure.
