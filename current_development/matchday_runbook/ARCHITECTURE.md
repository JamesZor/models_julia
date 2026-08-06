# src/MatchDay — system map

11 files, ~1400 lines, 9 swappable seams, 22 concrete components.

This describes the module **as built**. The pre-build design notes, including the measurements
that shaped it, are at `current_development/match_day_inference/ARCHITECTURE.md`; several claims
there were corrected by actually running the thing, and this file is the one that is true.

---

## 1. The 30-second version

```
   sofascore.events        betfair.match_meta       betfair_live.order_book_1m
   (what is on)            (who is who)             (what it costs)
         \                       |                        /
          v                      v                       v
      +--------------------------------------------------------+
      |                     FixtureCard                        |   one per fixture
      |     fixture + identity + lineup + as_of                |   I/O-bound, seconds
      +--------------------------------------------------------+
                                |
                  +-------------+-------------+
                  v                           v
             odds_df                     latents_df
        (book -> instrument           (features -> chain
          -> effective odds)           -> posterior draws)
                  |                           |
                  +-------------+-------------+
                                v
                          READINESS GATE          <- refusal is a VALUE, not an absence
                                v
        Portfolio.stake_sheet(sys, latents_df, expr, odds_df, fixtures)
                                v
                    sheet + side + venue_odds + venue_stake
                                v
                          order_ticket
```

**The one idea:** MatchDay manufactures Portfolio's two inputs for fixtures that have not been
played, and refuses loudly when it cannot. **It does no staking maths at all.**

---

## 2. File map (include order = dependency order)

```
src/MatchDay/
+- types.jl                  9 abstract types, 10 domain objects, 2 config structs
+- interfaces.jl             one contract per seam + error stubs
+- db.jl                     ALL SQL lives here + market-name mapping
+- instruments.jl            THE morphism: lay_to_back / direct_back / synthetic_back
|
+- implementations/
|  +- sources.jl             SofaScoreEvents ExplicitFixtures | MatchMetaCrosswalk
|  |                         ResolverChain | ProvisionalDB LastHistorical JsonPin SourceChain
|  +- book.jl                ArchivedOrderBook RedisLive* | BestAvailable MidPrice
|  |                         DirectBackOnly BestOfBackLay | NoMinimum FloorOrDrop FloorOrRoundUp
|  +- gates.jl               IdentityResolved MaxBookAge ConfirmedXI MaxLineupAge
|                            MinMatched GateChain
|
+- inference.jl              select_split, check_coverage, RatingsFromTracker,
|                            MarketPillarFromBook*, MaterialiserChain, matchday_latents
+- pipeline.jl               match_day, build_cards, price_cards, fixture_info,
|                            order_ticket, blocked_report
+- matchday-module.jl        module + include order + exports

* = declared seam, errors or defers if selected
```

---

## 3. The nine seams

```
SEAM                         CONTRACT                                       IMPLEMENTATIONS
===========================  =============================================  =====================
AbstractFixtureSource        fixtures(s, segment, as_of) -> [Fixture]       SofaScoreEvents
                                                                            ExplicitFixtures

AbstractIdentityResolver     resolve(r, fixture) -> Resolved|Unresolved     MatchMetaCrosswalk
                             a LOOKUP, never a matcher (see §9.1)           ResolverChain

AbstractLineupSource         lineup(s, fixture, as_of) -> Lineup|nothing    JsonPin ProvisionalDB
                             nothing = "no answer", enables SourceChain     LastHistorical
                                                                            SourceChain

AbstractBookSource           quotes(s, resolved, as_of) -> Dict{Key,Levels} ArchivedOrderBook
                             returns DEPTH, not a scalar                    RedisLive*

AbstractQuoteRule            quote_price(r, levels, side) -> Float64        BestAvailable MidPrice

AbstractInstrumentRule       instrument(r, key, comp, book, q) -> Instrument DirectBackOnly
                             THE morphism (§5)                              BestOfBackLay

AbstractStakeRounding        round_stake(r, stake, inst) -> Float64         NoMinimum FloorOrDrop
                             applies to VENUE STAKE, not risk               FloorOrRoundUp

AbstractFeatureMaterialiser  materialise!(m, Val{k}, fs, fx, ctx) -> Bool   RatingsFromTracker
                                                                            MarketPillarFromBook*
                                                                            MaterialiserChain

AbstractReadinessGate        ready(g, card) -> Ready|Blocked                IdentityResolved
                             CONJUNCTIVE, collects every reason             MaxBookAge ConfirmedXI
                                                                            MaxLineupAge
                                                                            MinMatched GateChain
```

**Two combinators, deliberately different.** `SourceChain` and `ResolverChain` are
**first-success** — return the first thing that answers. `GateChain` is **conjunctive** — run
everything, concatenate reasons. Collapsing them into one generic chain is the obvious refactor
and it is wrong.

---

## 4. Objects

```
   Fixture           m_id, home, away, kickoff::DateTime, tournament_id
        |            kickoff is a DateTime, never a Date -- every gate is a function of
        |            time-to-kickoff, and the prototype flattened all of it to today()
        v
   Resolved          fixture + bf_event_id + market_ids::Dict{String,String} + verified
   | Unresolved      fixture + reason::Symbol
        |            BOTH flow downstream. Filtering Unresolved out at stage 2 is exactly
        |            how a fixture becomes invisible instead of reported.
        v
   Lineup            home/away::[Player], confirmed::Bool, source::Symbol, scraped_at
        |            source and scraped_at are load-bearing; confirmed is not (§9.3)
        v
   FixtureCard{I}    fixture + identity + lineup + as_of + readiness
        |            parametric on Resolved|Unresolved
        v
   BookLevels        back[] back_size[] lay[] lay_size[] matched ts
        |            bid = available to BACK, ask = available to LAY. Verified by overround
        |            sign: back side sums > 1, lay side < 1. Prices unscaled from x10000 ints.
        v
   Instrument        key, odds, side, venue_odds, leverage
        |              odds     = EFFECTIVE price, denominated so 1 stake = 1 risk
        |              leverage = backer stake per unit of risk (1.0 back, 1/(d-1) lay)
        v
   MatchDayResult    sheet, cards, blocked, odds, instruments, as_of
```

---

## 5. The morphism — why Portfolio never learned what a lay is

A lay is a back on the complement **once the position is measured in risk**:

```
   lay Under at d, backer stake b   ->  risk = b(d-1),  win = b
   set risk = s, so b = s/(d-1)     ->  win/risk = 1/(d-1)
   a back at D has win/risk = D-1   ->  D = 1 + 1/(d-1) = d/(d-1)
```

So if `instrument()` emits `(effective_odds D, risk s)`, then the payoff matrix,
`KellyLogUtility`, `BakerMcHale`, `SlateDrawdown` and `FixedCap` are **already** denominated in
risk and work unchanged. `FixedCap` sums liability by construction — it never needed to be made
"liability-aware". Only the order ticket differs:

```
back :  stake        = s            at D
lay  :  backer stake = s / (d - 1)  at d        (liability = s)
```

Worked, from the real Waterford v Shelbourne O/U 2.5 book on 2026-08-02:

```
back Over  direct           4.80
lay  Under at 1.26   ->  1.26/0.26 = 4.846      +0.962%, leverage 3.85x
```

**Value, measured over 43,796 uncrossed two-sided snapshots.** The median gain is ~0 — the book
is arbitrage-free, so usually the two agree. Taking the better one is a free *option*, and
`E[max(0, gain)]` is where it shows:

```
competition             O/U1.5  O/U2.5  O/U3.5 | back overround @3.5
Scottish League Two      0.13%   1.09%   6.43% |  7.94%
Scottish League One      0.20%   0.97%   3.48% |  6.63%
Scottish Championship    0.20%   0.93%   5.07% |  4.33%
Scottish Premiership     0.29%   0.47%   1.94% |  1.71%
Irish Division 1         0.30%   0.43%   1.88% |  2.13%
Irish Premier Division   0.28%   0.37%   1.13% |  1.39%
```

It tracks book width almost monotonically — **worth most exactly where the book is worst**. And
it is asymmetric by side: quote a longshot by laying its complement, because the complement is a
near-certainty and tightly priced while the longshot's own back book is wide.

**The leverage cap.** The synthetic needs `risk/(d-1)` posted, which blows up as `d -> 1`:
laying Under 0.5 at 1.02 needs £50 for £1 of risk. Those are also the lines where the measured
gain looked implausible (O/U 5.5 at 46%, O/U 0.5 at 23% — an empty back book, not an edge).
`max_leverage = 20` rejects them **on price alone**, which is what makes skipping the depth
check safe.

**Minimum-stake interaction.** Betfair's £1 minimum applies to the venue stake, not to risk, so
a lay at a short price clears it with far less at risk. The morphism buys smaller minimum
positions as well as better prices — which matters most at a small bankroll.

---

## 6. The pipeline

```
match_day(spec, sys, segment, expr, ds; as_of, bankroll)
  |
  +- build_cards                                            STAGES 1-3
  |    fixtures(spec.fixtures, segment, as_of)
  |    resolve(spec.identity, f)          -> Resolved | Unresolved   (both kept)
  |    lineup(spec.lineups, f, as_of)     -> first source that answers
  |
  +- price_cards                                            STAGE 5'
  |    quotes(spec.book, resolved, as_of)         DISTINCT ON ... ts <= as_of
  |    complement_of(key, keys)                   two-outcome groups only
  |    instrument(spec.instrument, ...)           -> effective odds
  |    stamps book_age + max_matched onto the card for the gates
  |    -> odds_df  :match_id :market_name :market_line :selection :odds_close
  |
  +- ready(spec.gate, card) for every card                   STAGE 7
  |    conjunctive; passed / blocked
  |
  +- matchday_latents                                       STAGES 4 + 6
  |    create_id_boundaries -> select_split (§9.4)
  |    deepcopy the FeatureSet          <- never mutate a cached one
  |    materialise! over INJECTABLE_KEYS
  |    check_coverage                   <- PER FIXTURE, not per feature
  |    extract_parameters(model, frame, fs, chain)
  |
  +- Portfolio.stake_sheet(sys, latents, expr, odds, fixture_info(passed))
  |
  +- _attach_instruments!  -> side, venue_odds, risk, venue_stake; apply the minimum
```

Stage numbers are out of order on purpose — the book is built before features. That was
originally because market-pillar engines were thought to consume odds at inference; **they do
not** (§9.5). The order is harmless and stays, but it is not forced.

---

## 7. Where MatchDay ends and Portfolio begins

```
   MATCHDAY                              |  PORTFOLIO
   ------------------------------------- | ------------------------------------------
   what is on today                      |  how much to stake
   who is who on the exchange            |  which selections to hold jointly
   which XI                              |  parameter-uncertainty shrinkage
   which price, and on which side        |  the drawdown budget
   whether to bet at all                 |  the exposure cap
   what to place at the venue            |  settlement and simulation
```

Deleted from the prototype because Portfolio already does it, better: `solve_P` /
`run_match_live` / the k* grid, `calculate_betting_signals` per-bet Kelly, the per-match
`cap = 0.10`, and the manual rung-spreading and BTTS/under coherence rules the paper tracks
describe — all of which fall out of one joint allocation over 144 scorelines.

---

## 8. Two exits

```
  REPLAY (validated)                     LIVE (blocked on infrastructure)
  ==================                     ================================
  ExplicitFixtures(past fixtures)        SofaScoreEvents(horizon)
  ArchivedOrderBook, as_of in the past   ArchivedOrderBook (Redis is drained into it)
        |                                      |
  match_day(...; as_of = T-2h)           match_day(...; as_of = now())
        |                                      |
  compare snapshots -> CLV               stake sheet -> order_ticket
```

Replay is the capability the prototype never had, and the reason nothing about it was ever
validated. **Corpus today: 35 matches, 2026-05-29 .. 2026-06-26** — the intersection of
order-book coverage and a resolvable `match_id`. It grows every match week, so `r02` computes it
as a query rather than hard-coding a list. Enough to debug the pipeline; nowhere near enough to
establish an edge.

---

## 9. Things that will confuse you

```
1. A LOW RESOLUTION RATE IS A STOPPED JOB, NOT A MATCHING PROBLEM.
   betfair.match_meta resolves 100% when its job runs. Across the live feed it is
   100% to w/c 2026-06-15, 55% w/c 06-22, 0% after -- and present_but_unresolved
   is 0 in every competition, i.e. unresolved events are ABSENT from the table
   rather than failed. Do not reach for a fuzzy matcher.

2. "NO BETS" AND "BROKEN" ARE DIFFERENT, AND THE MODULE SAYS WHICH.
   Always read blocked_report(result) before concluding anything from an empty
   sheet. GateChain collects EVERY reason: "unresolved" alone is a dead resolver,
   "unresolved" + "no quotes" is a dead collector too.

3. `confirmed` IS USELESS; READ LEAD TIME INSTEAD.
   It has never been true for any match in sofascore.lineup_provisional, because
   every scrape ran 4.4-5.8h before kickoff and the XI lands ~1h out. ConfirmedXI
   is therefore non-blocking by default and MaxLineupAge is the usable gate.

4. THE SPLIT INDEX IS POSITIONAL AND DRIFTS.  (UNRESOLVED DEFECT)
   The chain is picked as training_results[N] where N = length(training_results),
   then the SAME integer indexes a boundary list rebuilt today. On src_sup40_sw40
   that is 29 vs 31: the two most recent windows are silently unused, and the
   pairing is correct at all only if the splitter appends rather than recomputes.
   select_split warns on every run. Warning is not fixing.

5. THE MARKET PILLAR IS A TRAINING FEATURE, NOT AN INFERENCE ONE.
   extract_parameters for the smile engine reads only home_team, away_team,
   match_id and optional season_idx/month_idx. The market pillar shaped the
   POSTERIOR during training. So the live book prices the bets; it does not feed
   the model. (The setup still swaps Betfair into ds.odds, because rebuilding the
   FeatureSet against a different pillar than the chain was fitted with would be
   silently wrong.)

6. REPLAY IS NOT THE BACKTEST.
   last_price_traded is NULL in 100% of order_book_1m rows, so replay prices off
   the BOOK while the Portfolio backtest settles at TRADED prices from
   betfair.odds_history. Different quantities. Never put them in one table without
   saying so.

7. required_features RETURNS CONFIG OBJECTS, NOT SYMBOLS.
   CLAUDE.md says Vector{Symbol}; this engine returns XGFeature(),
   DoublePoissonMarketFeature() and friends. Materialisers are keyed on
   INJECTABLE_KEYS -- the FeatureSet's per-match lookup maps -- because those are
   what extract_parameters actually indexes for an unseen fixture.
```

---

## 10. Where to look for what

| I want to... | Look at |
|---|---|
| see it run | `r01_quickstart.jl` |
| replay a past match day / measure CLV | `r02_replay.jl` |
| understand back vs lay | `r03_instruments.jl`, then `instruments.jl` (60 lines) |
| work out why I got no bets | `r04_diagnostics.jl`, then `blocked_report` |
| add my own component | `r05_extending.jl`, then `interfaces.jl` |
| know how a price becomes a stake | `pipeline.jl` `price_cards` + `_attach_instruments!` |
| know what is guaranteed | `test/matchday_tests.jl` (12 properties, 68 assertions) |
| see the SQL | `db.jl` — all of it, one blast radius |

---

## 11. Reference numbers

Replay of 2026-06-19 at 17:15 (Ireland Premier, `src_sup40_sw40`, 5 fixtures):

```
result                      23 bets, 5 priced, 0 blocked
selections quoted           17 per fixture (1X2 3 + BTTS 2 + O/U 0.5-5.5 12)
legs priced via lay         ~30-47% depending on the book
identity                    match 15238109 -> betfair event 35713518, 9 markets
lineup                      :provisional, 11+11, 5.8h before KO, confirmed = false
book age at as_of           0 minutes (1-minute bars run to kickoff)

£1 minimum, FloorOrDrop:    bankroll £15   ->  0 placeable bets
                            bankroll £1000 -> 17 placeable bets
```

**The £15 result is not a bug.** A 25% exposure cap over 5 fixtures leaves ~£3.75 to spread
across 23 positions, so every leg lands between 1p and 31p. £15 is below the operating threshold
for a slate this size; the fix is a bigger bankroll or a much shorter book, not a tuning change.

Upstream job health at the time of writing — all three feeds are dead, and none of them raises
anything when it stops:

```
identity resolver      last output 2026-06-22 (kickoff)
provisional lineups    last output 2026-06-26 14:07
order-book drain       last output 2026-08-02 14:57   (99 markets opened after it)
```
