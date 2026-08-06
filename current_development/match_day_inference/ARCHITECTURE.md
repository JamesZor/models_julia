# MatchDay — system design (v0, for argument)

**Status: design draft. Nothing here is built.** This is the strawman the design brief asked
for, revised against what the code and the database actually say rather than what the brief
assumed. Several of the brief's premises did not survive contact — §1 leads with those, because
two of them change what should be built.

---

## 0. Decisions taken (user, 2026-08-06)

| # | Decision | Consequence |
|---|---|---|
| 1 | Instruments are **abstract**: back-only, or back-and-lay, swappable | `AbstractInstrumentRule`. See the risk-normalisation note below — it means **no `src/Portfolio` change** |
| 2 | Canonical selection space stays **`{over, under}`**; back/lay is an execution detail | the model, payoff matrix and allocator never see a lay |
| 3 | **Bankroll is a parameter**, not £15 | `AbstractStakeRounding` must be bankroll-relative; £15 was a test only |
| 4 | **Read Postgres directly**; no stabilising view | intervals are ~1min and timing is not critical, so schema churn in the POC datastore is an accepted cost |
| 5 | Replay source is **`order_book_1m`**, not the free-API last-traded price | more depth, and the corpus grows every match week — so the replay set is a *query*, never a fixed list |
| 6 | **v1 targets Ireland**; Scottish collected in parallel | Ireland has the data and the validated engine (`src_sup40_sw40`); Scottish is where the finding-F overlay is biggest, so it follows. **But see below — "Ireland" is two segments, not one.** |

### ⚠ "Ireland" in the live feed is two tournaments, and only one has an engine

```
Data.Ireland()             -> tournament_ids = [79]   "Premier Division"
Data.IrelandFirstDivision()-> tournament_ids = [718]  "First Division"
```

They are separate `DataTournemantSegment`s, and the live Betfair feed carries **more First
Division than Premier** — 38 events vs 33. But `src_sup40_sw40` was trained on
`Data.Ireland()`, i.e. **79 only**.

This is not a naming quibble. Per the stream that opened 718, its dispersion regime is
materially different from 79 (variance/mean ≈ 1.14, negative-binomial beating Poisson by 9–12
AIC, versus 79 sitting near-Poisson), which is why the recommendation there was to *stratify*
dispersion rather than pool. Running the 79 engine over 718 fixtures would price a
higher-dispersion league with a lower-dispersion posterior — silently, and in the direction
that under-prices totals tails.

So decision 6 needs one more bit: **does v1 mean 79 only, or 79 + 718?** If 79 only, the
fixture source must filter on segment and the extra 38 live events per cycle go unused. If both,
718 needs its own trained engine before match day, not after. Recommend **79 only for v1** —
it is the one with a validated engine, a paper-track history and the `staking_layer` OOS work
behind it — and 718 as the first extension, since the live book is already being collected.

### The consequence of (1) + (2): lays need no Portfolio change at all

This is the load-bearing simplification and it is worth stating precisely.

A lay is a back on the complement **once you measure the position in units of risk**. Laying
`Under` at `d` with backer stake `b` risks liability `b(d−1)` and wins `b`. Set the liability to
`s`, so `b = s/(d−1)`; then win/risk `= 1/(d−1)`, which is exactly the win/risk of a back at
`D = 1 + 1/(d−1) = d/(d−1)`.

So if the instrument layer emits **`(effective_odds D, risk s)`**, every downstream object —
the payoff matrix, `KellyLogUtility`, `BakerMcHale`, `SlateDrawdown`, `FixedCap` — is already
denominated in risk and works **unchanged**. `FixedCap` does not under-count, because what it
sums is liability by construction.

The only thing that differs is the order ticket, at the very last step:

```
back :  place stake        = s              at D
lay  :  place backer stake = s / (d - 1)    at d       (liability = s)
```

Earlier drafts of this file said the cap "must become liability-based". That was wrong — it is
liability-based already, provided the morphism normalises before Portfolio sees anything.
`AbstractInstrumentRule` therefore returns an `Instrument`, and `Selection.odds_used` receives
`D`.

Commission stays clean for a non-obvious reason: within one market group the optimum never
covers every outcome (that is asserted, `r04` diagnostic "market groups fully covered: 0"), so
you never hold both sides of the same Betfair market and per-bet commission does not need to
become per-market netting.

---

## 1. Four findings that reshape the design

### A. Identity resolution is already solved. The resolver is dead.

The brief called the Betfair↔SofaScore join "the hardest unsolved thing in the system" at a 27%
exact-name match rate. That framing is wrong. `betfair.match_meta` is a purpose-built crosswalk:

```
match_id | betfair_event_id | betfair_event_name | kickoff_time | normalized_name
         | search_strategy_used | search_date_matched | is_verified | status
         | retry_count | error_type | error_detail | http_status | last_updated
```

8,367 verified `SUCCESS` rows via `exact_date_team_search`, 532 via `extend_date_team_search`,
spanning 2017-11-18 → 2026-06-27. Resolution rate against the live feed, bucketed by kick-off
week:

```
week of      live events   resolved    pct
2026-05-25        10          10       100%
2026-06-08         9           9       100%
2026-06-15        10          10       100%
2026-06-22        11           6        55%
2026-06-29        10           0         0%
2026-07-06         7           0         0%
2026-07-27        30           0         0%
2026-08-03         6           0         0%
```

A clean step function. And the diagnostic that settles it: across every competition,
**`present_but_unresolved = 0`** — not one live event sits in `match_meta` with a failed match.
The unresolved ones are *absent from the table entirely*. `match_meta` has no rows at all for
kick-offs after 2026-06-22.

**When the resolver runs it resolves 100%. It stopped around 2026-06-27.**

Design consequence: do **not** build a fuzzy matcher, a crosswalk table, or a manual review
queue. Those solve a problem that is already solved. Build (1) a reader for `match_meta` as the
crosswalk of record, and (2) a loud gate that reports resolver lag as an operational fault.
The fix for 0% resolution is to restart a job, not to write Julia.

### B. `stake_sheet` cannot see an unplayed fixture. It returns empty, silently.

The brief flagged this to check. It is broken, and worse than suspected.

```julia
ft = PF.fixture_table(ds)     # Ireland
(entries = 1029, unplayed_entries = 0, latest_fixture_date = 2026-08-02)
```

`fixture_table` iterates `ds.matches`, which carries **zero** rows with a missing score — it is
the curated store of finished matches. `build_book` opens with
`haskey(fixtures, m_id) || return nothing`. An upcoming fixture is not in `ds.matches`, so it
is not in `fixtures`, so every `build_book` returns `nothing`, so `build_books` returns an empty
vector, so `stake_sheet` returns `_empty_sheet()`.

The `require_result = false` branch — the entire match-day code path, its docstring, and the
`settle::Union{Nothing,Vector{Float64}}` field that forced a REPL restart to introduce — **can
never fire on a DataStore-derived fixture table.** It is unreachable code that returns a
plausible-looking empty DataFrame.

I wrote that entry point and asserted it was the match-day seam. It is not. This is the third
silent-empty failure on this project in a month, and the first one I shipped into `src/`.

Design consequence: the fixture table is a MatchDay responsibility, not a Portfolio one. Either
`build_books` takes a `Dict{Int,FixtureInfo}` directly (small change, right seam), or MatchDay
constructs a `ds`-shaped object with today's fixtures appended (larger, worse). See §9 Q3.

### C. Three collectors are down, and Postgres is the only surviving copy.

Confirmed by the user: **Redis on the homelab is down.** That is not a side issue — it is the
live path. `whatstheodds/pipeline/stream_worker.py` drains Redis into Postgres
(`INSERT INTO betfair_live.order_book_1m ... ON CONFLICT (market_id, symbol, ts) DO UPDATE`,
downsampled to the minute), so **`order_book_1m` is the persisted Redis feed, not a parallel
one.** The brief listed "same collector?" as an unproven inference; it is now proven, and it
means one `AbstractBookSource` adapter genuinely does serve both live and replay.

It also means Postgres currently holds the only copy of anything.

```
job                          last output        state
order-book drain             2026-08-02 14:57   dead (99 markets opened after, never ticked)
identity resolver            2026-06-22 KO      dead (finding A)
provisional-lineup scraper   2026-06-26 14:07   dead (finding E)
```

The live feed also now covers more than the brief recorded: Scottish Premiership,
Championship, **League One and League Two** (= `ScottishLower`, 56/57 — the modelled segment)
appeared 2026-08-01, plus Finnish Ykkösliiga. All at 0% resolution, all unticked.

Two column-level facts from the drain that change the design:

- **`last_price_traded` is NULL in 100% of rows** (0 of 428,411). `stream_worker` only writes
  prices and volumes. So the live/replay path has **no traded price at all** — it cannot
  reproduce `odds_close`, which the backtest takes from `betfair.odds_history`. Live must price
  off the book. That is the honest number anyway, but it means replay and backtest are not
  measuring the same quantity and must never be compared without saying so.
- **`market_matched` / `total_matched` are populated in 38.4% of rows** (164,515 of 428,411).
  A `MinMatched` liquidity gate would be blind on the other 62%.

### E. The lineup scraper works. It fires at the wrong hour.

The user's instinct — "we need to get the lineup differently, add a get-lineup query before the
game starts" — is right, and the precise version is narrower than a redesign: **the scraper
already exists and is correct; it has never been run inside the window where the answer changes.**

`sofascore.lineup_provisional` in full — 220 rows, 10 matches, and the number that matters:

```
match                            kickoff             scraped        hrs before KO   confirmed
drogheda-united v shelbourne     2026-06-19 18:45    13:09:45            5.6          false
galway-united v derry-city       2026-06-19 18:45    13:09:43            5.6          false
st-patricks v sligo-rovers       2026-06-19 18:45    13:09:45            5.6          false
waterford v shamrock-rovers      2026-06-19 18:45    13:09:54            5.6          false
bohemian v dundalk               2026-06-19 19:00    13:09:47            5.8          false
shelbourne v bohemian            2026-06-22 18:45    17:08:50            1.6          false
shamrock-rovers v galway         2026-06-26 18:30    14:07:27            4.4          false
derry-city v drogheda            2026-06-26 18:45    14:07:50            4.6          false
dundalk v waterford              2026-06-26 18:45    14:07:28            4.6          false
bohemian v st-patricks           2026-06-26 19:00    14:07:31            4.9          false
```

Every scrape ran **4.4–5.8 hours before kick-off** (one at 1.6h), always returning exactly 22
players. SofaScore publishes the confirmed XI roughly **one hour** before kick-off.
**`confirmed` has therefore never once been true, on any match, ever.**

`scrape_today_provisional_lineups` in `sofascrape/pipeline/orchestrator.py` even documents the
fix in its own docstring — *"Re-run as kickoff nears to refresh"*. That re-run has never happened.

Three consequences, in descending order of importance:

1. **`ConfirmedXI` cannot be a blocking gate.** On every row of evidence that exists, it would
   block 100% of fixtures. This settles §9 Q7 empirically rather than by judgement.
2. **The fix is a schedule, not a query.** A second invocation at ~T−45min, against the same
   function. The Julia side needs no new scraper — it needs to read `scraped_at`, compute
   `kickoff − scraped_at`, and carry that as lineup staleness.
3. **This is exactly the two-snapshot workflow the paper tracks already describe.** 12 Jun
   priced at 17:38 pre-lineup and re-priced at 20:11 with XIs confirmed, and found the totals
   edges held. The infrastructure only ever captured the first snapshot; the second was done by
   hand, in a browser. That gap *is* the missing feature.

### D. Portfolio already does what the paper tracks show being done by hand.

From `paper_tracks/ireland_19_06_26.md`, the user's own carried-forward lessons:

> Rungs spread across matches to avoid nesting (U1.5 ⊂ U2.5 in same game = correlated, not
> diversified).

> BTTS-Yes … contradicts Under 1.5 (mutually exclusive) … a self-cancelling straddle.
> **Carry forward: pick one coherent goal-direction per match.**

> Early exposure: £6 · Pending: £4 · Total at risk if all placed: £10 / £20 (keep ~£10 dry).

All three are correlation management, performed manually because the prototype stakes each bet
independently. `KellyLogUtility` solves one joint allocation over all 144 scorelines — nesting,
contradiction and exposure fall out of the objective. The 19 Jun counterfactual (thesis-pure
book +£1.60 on £3 vs −£0.13 as placed) is a hand-computed version of what the allocator returns.

**This is the strongest argument for the whole redesign, and it belongs at the top of the case:
the layer being replaced is not just duplicated code, it is the reason the user is doing
portfolio construction in their head at 17:15 on a Friday.**

### F. The back/lay overlay is worth most exactly where you bet.

Your instinct that "under back is the same as an over lay" is right, and it is worth more than
I expected. For a two-runner market, laying at `d` is backing the complement at `d/(d−1)`, so
every O/U position has **two instruments** and you should always take the better one:

```
effective_over  = max( best_back(Over) , lay_to_back(best_lay(Under)) )
lay_to_back(d)  = d / (d - 1)
```

Worked from a real snapshot (Waterford v Shelbourne, O/U 2.5, 2026-08-02):
back Over = 4.80 direct; lay Under at 1.26 → `1.26/0.26` = **4.846**, i.e. 0.96% better.

Measured across 43,796 uncrossed two-sided snapshots, the *median* gain is ≈0 — the book is
arbitrage-free, so usually the two instruments agree. But taking the better one is a free
option, and the option has value. `E[max(0, gain)]` per (competition, market), with the back
side's own overround alongside:

```
competition              O/U1.5  O/U2.5  O/U3.5 | back overround @ 3.5
                          over    over    over  |
Scottish League Two       0.13%   1.09%   6.43% |   7.94%
Scottish League One       0.20%   0.97%   3.48% |   6.63%
Scottish Championship     0.20%   0.93%   5.07% |   4.33%
Scottish Premiership      0.29%   0.47%   1.94% |   1.71%
Irish Division 1          0.30%   0.43%   1.88% |   2.13%
Irish Premier Division    0.28%   0.37%   1.13% |   1.39%
```

**The gain tracks book width almost monotonically.** It is worth ~1% on Ireland Premier's tight
central lines and **3.5–6.4% on Scottish League One / Two** — which is `ScottishLower`, the
segment `funnel_apm_xg` models. As you said, it depends on league and market, so it cannot be a
constant: it is a per-(competition, market, side) property that must be measured, not assumed.

It is also **asymmetric by side**. On O/U 3.5 the Over is the longshot and `opt_over` dominates;
on Irish Division 1's O/U 2.5 it inverts (`opt_under` 1.17% vs `opt_over` 0.43%). The rule is
structural: *quote a longshot by laying its complement*, because the complement is a
near-certainty and therefore tightly priced, while the longshot's own back book is wide.

Two caveats before this is treated as free money:

- **Size is unverified.** To lay Under 0.5 for £1 of *risk* the backer's stake must be
  `1/(d−1)` — large at short prices. The `ask_volumes` are in the table and this must be
  checked before the gain is believed, especially for the O/U 5.5 (46%) and O/U 0.5 (23%)
  outliers, which almost certainly reflect an empty back book rather than a real edge.
- **Exposure changes meaning.** A lay's risk is liability `(d−1)×stake`, not stake.
  `FixedCap` sums stakes. If lays are placed rather than merely priced, the cap must become
  liability-based or it silently under-counts. See §9 Q4.

---

## 2. The 30-second version

```
   sofascore.events            betfair.match_meta         betfair_live / Redis
   (what is on today)          (who is who)               (what it costs)
          \                          |                          /
           \                         |                         /
            v                        v                        v
        +-------------------------------------------------------+
        |                     FixtureCard                       |   one per fixture
        |   identity + lineup + as_of + readiness               |   CHEAP, but I/O-bound
        +-------------------------------------------------------+
                                   |
                      +------------+------------+
                      |                         |
                      v                         v
                 odds_df                   latents_df          <-- the two Portfolio wants
            (live or replayed book)    (features -> chain)         EXPENSIVE (MCMC extract)
                      |                         |
                      +------------+------------+
                                   |
                                   v
                  PF.stake_sheet(sys, latents_df, expr, odds_df, fixtures)
                                   |
                                   v
                        stake sheet, persisted with as_of
                                   |
                          reconcile later -> CLV
```

**The one idea:** MatchDay's job is to manufacture Portfolio's two inputs for fixtures that
have not been played, and to **refuse loudly** when it cannot. It does no staking maths at all.

---

## 3. What the prototype got right

Most of this document is about deletion, so it is worth being explicit that the prototype
solved several problems correctly and those solutions should survive the rewrite intact.

**The lineup priority order is right, and non-obvious.** JSON pin → provisional DB → last
historical XI. Each tier is strictly less informative than the last, the manual override sits
at the top where a human can break a tie, and the fallback never fails outright. Most systems
would have stopped at "query the DB and error if empty". Preserve the order verbatim.

**`compare_matchday_lineups` is the right diagnostic, and nothing in `src/` has an equivalent.**
It reports the *positional-sum delta* between the provisional and fallback XI — i.e. exactly
how far the model's inputs move when the lineup source changes, in the units the model actually
consumes. That is the correct way to ask "does the lineup source matter?", and the question is
still open. Port it as-is.

**Sourcing player strength from the historical tracker, not from the pre-match feed.** The
docstring on `fetch_provisional_lineup` explains that `sofascore_rating` is absent pre-match and
sets it to `0.0` deliberately, because the model keys off `player_id` against the tracker. This
is the difference between a system that works on match day and one that silently prices every
fixture as though every player were average. It is a small comment guarding a large trap.

**The `confirmed` boolean is plumbed all the way through** from `sofascore.lineup_provisional`
to the console output. The prototype knew the difference between a predicted XI and a confirmed
one and said so. That flag becomes a gate input in §5 rather than being invented from scratch.

**Debutant reporting.** `build_matchday_ratings_map` prints every player with no tracked history
and the fallback value being substituted. Unknown players are the single most likely source of a
wrong price, and the prototype makes them visible rather than silently averaging them in.

**The header comment on `unified_staking.jl`.** The file is being deleted, but its header
records the curation findings — full-book Kelly bankrupts, O/U + BTTS only, 1X2 display-only,
back-only, commission in `d_eff` — that cost real money to learn. Move the comment before
deleting the code.

**Fast-fail ordering.** `build_live_book` returns empty below 2 priced selections rather than
handing a degenerate one-leg book to a portfolio solver. Small, correct instinct.

---

## 4. What MatchDay stops doing

| Prototype | Fate | Because |
|---|---|---|
| `unified_staking.jl` — `solve_P`, `run_match_live`, k* grid | **delete** | `KellyLogUtility` + `BakerMcHale`, tested, with a KKT audit |
| `calculate_betting_signals` — per-bet EV + `BayesianKelly` | **delete** | per-bet Kelly is the thing the staking research showed bankrupts |
| per-match `cap = 0.10` | **delete** | `FixedCap` on the slate, which is where simultaneity lives |
| manual rung-spreading / coherence rules | **delete** | falls out of the joint allocation |
| `ppd_to_betfair_type` / `betfair_to_ppd_type` | **keep, move** | market-name mapping is real work; belongs next to `Data.MarketConfig` |
| lineup tiering | **keep, formalise** | genuinely good; becomes `AbstractLineupSource` |
| `inject_matchday_features!` | **keep the idea, generalise** | see §8 STAGE 4 |
| dashboards (`print_live_betting_dashboard*`) | **rewrite thin** | 400 lines of `Matrix{Any}` over what is now a DataFrame |

Net: of 1,740 prototype lines, roughly **900 are deleted outright**, ~500 are reshaped, and
~300 (the Redis/DB adapters) survive largely intact behind new interfaces.

---

## 5. The seams

Same idiom as `src/Portfolio`: abstract type, exactly one contract method, no registry.

```
SEAM                        CONTRACT                                          NOTES
==========================  ================================================  ==================
AbstractFixtureSource       fixtures(src, segment, as_of) -> Vector{Fixture}
                            SofaScoreEvents | ExplicitFixtures | ReplayDay        ReplayDay = tests

AbstractIdentityResolver    resolve(r, fixture) -> Resolved | Unresolved
                            MatchMetaCrosswalk | ResolverChain                    NOT a fuzzy matcher
                                                                                  (finding A)

AbstractLineupSource        lineup(src, fixture, as_of) -> Lineup | nothing
                            JsonPin | ProvisionalDB | LastHistorical
                            | SofaScoreAPI | SourceChain                          chain = the tiering

AbstractFeatureMaterialiser materialise(m, ::Val{:feat}, fixtures, fs, as_of)
                            RatingsFromTracker | MarketPillarFromBook             per-FEATURE, not
                            | CarryForward                                        per-model (§8.4)

AbstractBookSource          quotes(src, ref, as_of) -> DataFrame
                            RedisLive | ArchivedOrderBook | HistoricalClose       replay lives here

AbstractQuoteRule           quote_price(rule, levels) -> Float64
                            BestBack | BestLay | MidPrice | DepthWeighted         SPLIT from source
                                                                                  (§9 Q4)

AbstractInstrumentRule      instrument(rule, canonical_sel, book) -> Instrument
                            DirectBackOnly | BestOfBackLay                        the finding-F morphism.
                            | SizedBestOfBackLay                                  canonical selection stays
                                                                                  {over,under}; this picks
                                                                                  HOW to express it

AbstractStakeRounding       round_stake(r, frac, bankroll, instrument) -> Float64
                            NoMinimum | FloorOrDrop(1.0)                          the £1 floor, as two
                            | FloorOrRoundUp(1.0)                                 swappable policies

AbstractReadinessGate       ready(gate, card) -> Ready | Blocked(reason)
                            ConfirmedXI | MaxBookAge | MinMatched | MaxSpread
                            | IdentityResolved | GateChain                        loud, never silent
```

**Two deliberate departures from the strawman in the brief.**

1. `AbstractBookSource` is split into **source** (where bytes come from) and **`AbstractQuoteRule`**
   (which number in the book you take). They vary independently: replay-from-archive with
   best-back, live-Redis with mid, archive with depth-weighted. Fusing them means every new
   price rule needs a new source. The brief asked whether the rule belongs in Portfolio's
   `AbstractPricePolicy` — it does not: `settlement_odds(policy, d, overround)` receives a
   single scalar `d` and knows nothing about depth. `AbstractQuoteRule` collapses the book to
   that scalar; `AbstractPricePolicy` then de-arbs it. Different jobs, correct order.

2. `AbstractFeatureMaterialiser` dispatches on `Val{:feature_name}`, mirroring
   `Features.add_feature!`. This is what generalises `inject_matchday_features!` and it is the
   only way a market-pillar engine can ever run on match day.

`…Chain` recurs three times (lineups, identity, gates). Steal `FilterChain`'s shape from
`src/Portfolio/implementations/filters.jl` — but note lineup and identity chains are
**first-success** (return the first that works) whereas the gate chain is **conjunctive** (all
must pass, collect every reason). Two different combinators; do not paper over it.

---

## 6. Types — a sketch to argue with, not a specification

Deliberately incomplete. The parametric fields matter: abstractly-typed fields would make the
replay loop type-unstable, which is the mistake `src/Portfolio/types.jl` was careful to avoid.

```julia
# ---------------------------------------------------------------- seams
abstract type AbstractFixtureSource end        # fixtures(src, segment, as_of)
abstract type AbstractIdentityResolver end     # resolve(r, fixture)
abstract type AbstractLineupSource end         # lineup(src, fixture, as_of)
abstract type AbstractFeatureMaterialiser end  # materialise(m, ::Val{F}, fixtures, fs, as_of)
abstract type AbstractBookSource end           # quotes(src, resolved, as_of)
abstract type AbstractQuoteRule end            # quote_price(rule, levels)
abstract type AbstractReadinessGate end        # ready(gate, card)

# ---------------------------------------------------------------- domain
struct Fixture
    m_id::Int
    home::String
    away::String
    kickoff::DateTime          # NOT Date. the prototype flattened this to today()
    tournament_id::Int
end

"Depth at one selection, already unscaled from the x10000 integers."
struct BookLevels
    back::Vector{Float64}      # bid side, best first
    back_size::Vector{Float64}
    lay::Vector{Float64}       # ask side
    lay_size::Vector{Float64}
    matched::Float64           # market_matched, for MinMatched
    ts::DateTime               # for MaxBookAge -- staleness is per-selection
end

struct Resolved
    fixture::Fixture
    bf_event_id::String
    market_ids::Dict{String,String}      # "OVER_UNDER_25" => "1.260457203"
    verified::Bool
end
struct Unresolved
    fixture::Fixture
    reason::Symbol             # :absent_from_crosswalk | :resolver_stale | :no_markets
end

struct Player
    player_id::Int
    name::String
    position::Symbol           # :G :D :M :F, already cleaned
    substitute::Bool
end
struct Lineup
    home::Vector{Player}
    away::Vector{Player}
    confirmed::Bool
    source::Symbol             # :json_pin | :provisional | :last_historical
    scraped_at::DateTime       # must be <= as_of or replay leaks the future
end

"Everything known about one fixture at one instant."
struct FixtureCard{I<:Union{Resolved,Unresolved}}
    fixture::Fixture
    identity::I
    lineup::Union{Nothing,Lineup}
    as_of::DateTime
end

struct Blocked; reasons::Vector{Pair{Symbol,String}}; end
struct Ready end

# ---------------------------------------------------------------- config
Base.@kwdef struct MatchDaySpec{F<:AbstractFixtureSource, I<:AbstractIdentityResolver,
                                L<:AbstractLineupSource, M<:AbstractFeatureMaterialiser,
                                B<:AbstractBookSource,   Q<:AbstractQuoteRule,
                                G<:AbstractReadinessGate}
    fixtures::F   = SofaScoreEvents(horizon = Hour(36))
    identity::I   = MatchMetaCrosswalk(max_lag = Day(3))
    lineups::L    = SourceChain(JsonPin(), ProvisionalDB(), LastHistorical())
    features::M   = MaterialiserChain(RatingsFromTracker(), MarketPillarFromBook())
    book::B       = ArchivedOrderBook()      # RedisLive() in production
    quote_rule::Q = BestBack()
    gate::G       = GateChain(IdentityResolved(), MaxBookAge(Minute(15)),
                              MinMatched(500.0), ConfirmedXI(blocking = false))
end

# ---------------------------------------------------------------- contracts
fixtures(::AbstractFixtureSource, segment, as_of::DateTime)::Vector{Fixture} = error("...")
resolve(::AbstractIdentityResolver, ::Fixture)::Union{Resolved,Unresolved}   = error("...")
lineup(::AbstractLineupSource, ::Fixture, as_of::DateTime)                   = error("...")
materialise(::AbstractFeatureMaterialiser, ::Val, fixtures, fs, as_of)       = error("...")
quotes(::AbstractBookSource, ::Resolved, as_of::DateTime)::Dict{Tuple{String,Symbol},BookLevels} =
    error("...")
quote_price(::AbstractQuoteRule, ::BookLevels)::Float64                      = error("...")
ready(::AbstractReadinessGate, ::FixtureCard)::Union{Ready,Blocked}          = error("...")

# ---------------------------------------------------------------- entry point
"""
    match_day(spec, segment, expr, sys; as_of = now(), bankroll = 1.0)

The single parameterised entry point that replaces "copy last week's runner and edit the date".
`as_of` defaults at the CALL SITE only -- no stage reads the clock internally.
"""
function match_day(spec::MatchDaySpec, segment, expr, sys::PF.PortfolioSystem;
                   as_of::DateTime = now(), bankroll::Real = 1.0)
    # cards -> odds_df -> latents_df -> gate -> PF.stake_sheet
    # returns (sheet, cards, blocked) so a refusal is a RESULT, never an empty DataFrame
end
```

Three things to argue with:

- `FixtureCard` is parametric on `Union{Resolved,Unresolved}` so an unresolved fixture is still
  a first-class card that flows to the gate and gets reported. The alternative — filtering them
  out at stage 2 — is how they become invisible.
- `quotes` returns `Dict{(market, selection) => BookLevels}` rather than a DataFrame, so depth
  survives to the gate. The DataFrame is produced only after `quote_price` collapses it.
- `match_day` returns `(sheet, cards, blocked)`. A blocked slate is a value, not an absence.

---

## 7. Objects

```
   Fixture           m_id, home, away, kickoff::DateTime, tournament_id, segment
        |                                          ^^^^^^^^^^^^^^^^^^^^
        |            kickoff is a DateTime, never a Date. The prototype stamped
        |            match_date = today() and lost the clock entirely.
        v
   Resolved          fixture + bf_event_id + market_ids::Dict{String,String}
   | Unresolved      fixture + reason::Symbol  (:absent_from_crosswalk, :resolver_stale, ...)
        |
        v
   Lineup            home::Vector{Player}, away::Vector{Player},
                     confirmed::Bool, source::Symbol, scraped_at::DateTime
        |            source and scraped_at are NOT decoration -- they are gate inputs
        v
   FixtureCard       fixture, identity, lineup, as_of::DateTime, readiness
        |            the unit of "everything we know about one fixture at one instant"
        v
   (odds_df, latents_df)                          <-- Portfolio's contract, unchanged
        |
        v
   StakeSheet        PF.stake_sheet output + as_of + a book snapshot
                     persisted at bet time so CLV can be computed later
```

`as_of::DateTime` threads through every stage and is **never** defaulted to `now()` inside a
function. A pipeline that reads the clock internally cannot be replayed, and replay is the only
route to validating any of this (§9 Q2).

---

## 8. The staged pipeline

```
  MatchDaySpec{F,I,L,M,B,Q,G}  +  as_of::DateTime  +  segment
        |
  STAGE 1  FIXTURES ─────────────────────────────────────────────────────────────
        |   fixtures(SofaScoreEvents(), segment, as_of)
        |     SELECT ... FROM sofascore.events
        |     WHERE status_type='notstarted' AND tournament_id = ANY(...)
        |       AND start_timestamp BETWEEN as_of AND as_of + horizon
        |   ⚠ horizon, not CURRENT_DATE. A 19:45 Irish KO and a UTC midnight
        |     boundary is a real bug in the prototype's `>= CURRENT_DATE` window.
        v
     Vector{Fixture}
        |
  STAGE 2  IDENTITY ─────────────────────────────────────────────────────────────
        |   resolve(MatchMetaCrosswalk(), fixture)
        |     JOIN betfair.match_meta ON match_id
        |     -> bf_event_id, then betfair_live.market_metadata -> market_ids
        |   ⚠ 0% for kick-offs after 2026-06-22. Emit Unresolved(:resolver_stale),
        |     do NOT fall back to name matching. (finding A)
        v
     Vector{Union{Resolved,Unresolved}}     <-- both kept; Unresolved is reportable
        |
  STAGE 3  LINEUPS ──────────────────────────────────────────────────────────────
        |   lineup(SourceChain(JsonPin, ProvisionalDB, LastHistorical), fx, as_of)
        |     first success wins; records WHICH source answered
        |   ⚠ ProvisionalDB rows carry scraped_at -- reject rows newer than as_of
        |     or replay leaks the future into the past
        v
     Lineup (confirmed::Bool, source::Symbol)
        |
  STAGE 5' BOOK  (before inference -- see the ⚠ below) ───────────────────────────
        |   quotes(src, resolved, as_of) |> quote_price(rule, levels)
        |     RedisLive        : hgetall live_markets            (live only)
        |     ArchivedOrderBook: order_book_1m WHERE ts <= as_of (replay)
        |     ⚠ prices are ints scaled x10000; bid = back, ask = lay
        |       (verified: back side overround 1.0083 > 1, lay side 0.9937 < 1)
        v
     odds_df  ::  :match_id :market_name :market_line :selection :odds_close
        |
  STAGE 4  FEATURES ─────────────────────────────────────────────────────────────
        |   for f in Features.required_features(model)
        |       materialise(spec.features, Val(f), fixtures, feature_set, as_of)
        |   RatingsFromTracker    -> roll tracker history forward to as_of
        |   MarketPillarFromBook  -> CONSUMES odds_df from STAGE 5'
        |   ⚠ this is why the book comes first. src_sup40_sw40 takes market odds as
        |     a MODEL FEATURE, so inference depends on the book that staking also
        |     depends on. Same odds_df object into both -- never re-fetched.
        v
     FeatureSet (a COPY -- the prototype mutated a cached one)
        |
  STAGE 6  INFERENCE ────────────────────────────────────────────────────────────
        |   Models.PreGame.extract_parameters(model, fixtures, feature_set, chain)
        |   chain = training_results[N] where N = length(training_results)
        |   ⚠ N INDEXES A LIST THAT IS RECOMPUTED AT INFERENCE TIME. Measured on
        |     src_sup40_sw40: 29 training_results, 31 boundaries today. The rule
        |     silently drops the two most recent windows, and is only correct at
        |     all if the splitter APPENDS rather than recomputes. (§9 Q11)
        v
     latents_df  (λ draws per fixture)
        |
  STAGE 7  GATE ─────────────────────────────────────────────────────────────────
        |   ready(GateChain(IdentityResolved, ConfirmedXI, MaxBookAge(15min),
        |                   MinMatched(£500), MaxSpread(2 ticks)), card)
        |   CONJUNCTIVE: all must pass, and every failure reason is collected
        v
     Ready | Blocked([reasons])
        |
  ══════════════════════ HANDOFF ═══════════════════════════════════════════════
        v
     PF.stake_sheet(sys, latents_df, expr, odds_df, fixtures)
        ⚠ signature change required -- `fixtures`, not `ds` (finding B)
```

Note the stage numbers are deliberately out of order: **the book must be built before
features**, because market-pillar engines consume it. The brief's strawman had them the other
way round and the circularity was flagged but not resolved. This is the resolution.

---

## 9. Open questions, with recommendations

**Q1 — Identity resolution.** *Recommendation: read `betfair.match_meta`, add no matching logic,
and treat staleness as an operational alarm.* Finding A shows matching quality is 100% when the
job runs. Building a fuzzy matcher would add a second source of truth to paper over a dead cron.
**Your call:** is restarting the resolver something you can do, or does MatchDay need to own
resolution end-to-end because that job is not coming back?

**Q2 — Replay or live-only.** *Recommendation: replay from day one.* `order_book_1m` archives
minute bars back to 2026-05-29, so the pipeline can be run as-of T−120min and T−0 and compared
to settlement. Nothing about the prototype has ever been validated because it could only run on
a live Saturday. Replay is also the only way to test a gate. Cost: `as_of` in every signature —
cheap if done now, expensive later.

**Q3 — Where MatchDay ends and Portfolio begins.** *Recommendation: change `build_books` to take
`fixtures::Dict{Int,FixtureInfo}` instead of `ds`, and have MatchDay build it.* One-line change
at the Portfolio end, correct seam, and it makes the broken path in finding B representable.
The alternative — synthesising a `ds` with today's fixtures grafted onto `.matches` — puts
unplayed matches into a structure whose every other consumer assumes they are played.

**Q4 — Execution price. RESOLVED** (decision 1/2, §0). Instruments are abstract:
`DirectBackOnly` and `BestOfBackLay` are both `AbstractInstrumentRule` implementations, chosen
per run. The canonical selection stays `{over, under}`, so `Data.grade_selection` needs no
change and the payoff matrix never sees two rows for one state — my earlier worry about that
was misplaced, because the morphism collapses the pair *before* a `Selection` is constructed.

What remains open here is narrower and empirical: **is the finding-F gain real at size?**
`ask_volumes` is in the table and unqueried. Recommended default until it is checked:
`BestOfBackLay` for pricing (it is a strict improvement on the number), `DirectBackOnly` for
placement. The instrument abstraction makes flipping that a config change, which is the point.

A second open one: `SizedBestOfBackLay` should reject a synthetic price whose required backer
stake `s/(d−1)` exceeds available `ask_size`. Without it the O/U 5.5 (46%) and O/U 0.5 (23%)
figures will flow straight into stakes as though they were free.

**Q5 — Feature materialisation.** *Recommendation: per-feature dispatch on `Val{:name}`,
mirroring `Features.add_feature!`.* The special-case version cannot run `src_sup40_sw40`, which
is the engine you actually want on Ireland. Start with `:player_ratings` and `:market_pillar`;
anything unimplemented should throw, not silently carry forward a stale value.

**Q6 — Train/serve skew.** *Recommendation: delete the prototype's duplicate recursions and call
`Features.calculate_player_ratings` with a sentinel appended, taking the last element.* I checked
all four trackers line by line:

```
BayesianTracker    src out[i] = state BEFORE obs i ;  proto = state after all n   ->  CONSISTENT
EWMATracker        same structure                                                ->  CONSISTENT
WindowAverage      src [max(1,i-w) : i-1] ; proto [max(1,n-w+1) : n]             ->  CONSISTENT
                   (identical for i = n+1)
LastValueTracker   src lag(ratings) -> ratings[n]     (may be missing)
                   proto last(filter(!ismissing))     (skips trailing missings)   ->  DIVERGES
```

Plus an asymmetry that matters more than the LastValue case: the prototype has a **generic
`::AbstractRatingTracker` fallback returning `mean`**, and `src/features/` has none. A new
tracker therefore fails **loudly** at training (MethodError) and **silently** at serving (gets
the mean). That is precisely backwards. Whatever else is decided, delete that fallback.

**Q7 — Staleness and gating.** *Recommendation: `IdentityResolved` and `MaxBookAge(15min)`
blocking; `ConfirmedXI` and `MinMatched` reporting-only.* This was posed as your call; finding E
answers it with data instead. `confirmed` has never been true on any match in the table, so a
blocking `ConfirmedXI` blocks everything — it becomes usable only after the T−45min re-scrape
exists, and even then it should start as a warning. `MinMatched` is demoted for a separate
reason: `market_matched` is NULL in 62% of order-book rows, so the gate would be blind more
often than not, and a gate that silently passes on missing data is worse than no gate.

Findings A, C and E say the three things most likely to be wrong on a given Saturday are a dead
resolver, a dead drain and a stale XI. **None of them currently produces a visible symptom** —
that is the actual justification for the gate layer, more than any per-bet filtering.

**Q8 — What gets bet.** *Recommendation: config, defaulting to O/U + BTTS, 1X2 display-only.*
Both paper tracks confirm the negative-G read on 1X2, and 12 Jun found the stronger version:
every skipped 1X2 selection **drifted further from the model after lineups confirmed**. That is
the market telling you the informed close disagrees more, not less. Keep CS/DC/AH out until
something says otherwise; the staking-layer work measured CorrectScore at −20% ROI.

**Q9 — Output surface.** *Recommendation: persist first, display second.* The prototype writes
nothing, so beyond two hand-written paper tracks there is no record of what was priced when. A
`StakeSheet` row with `as_of` and the book snapshot makes CLV computable — and per §10, CLV is a
far higher-power test than P/L on this sample size.

**Q11 — Which split do we condition on?** *No recommendation; I think this one is a defect and
want you to look at it.* `inference.jl` picks the chain with
`last_split_idx = length(experiment.training_results)`, then indexes **today's** freshly built
`feature_collection` with that same integer. Measured on `src_sup40_sw40` (Ireland,
`GroupedCVConfig(Targets=["2025","2026"], Hist=2)`):

```
length(ex.training_results)              29     <- fitted 2026-07-04
length(create_id_boundaries(ds, cfg))    31     <- rebuilt 2026-08-06
```

Two problems, one certain and one conditional:

- **Certain:** it always conditions on split 29 of 31, silently discarding the two most recent
  windows. Today's fixtures are priced off a posterior and a feature set that stop two
  match-weeks short, with no warning.
- **Conditional:** the pairing is only *correct* if the splitter appends new boundaries and
  leaves old ones untouched. If `GroupedCVConfig` ever recomputes its windows when data grows,
  index 29 means a different window in the chain than in the features — a chain fitted on one
  period applied to another period's covariates, which would be silently and badly wrong.

A match-day system should name the split it wants by its **boundary**, not by a positional index
into a list whose length changes underneath it. This needs checking before anything is built on
top of it; it may also affect `Experiments.extract_oos_predictions`.

**Q10 — Module home.** *Recommendation: `src/MatchDay/` for stages 1-3 and 7, `src/features/` for
stage 4, `src/Data/` for the live-book adapter.* The materialiser is a Features concept and will
need to see internals; the book adapter is a fetcher like every other fetcher. Only the
orchestration and the gates are genuinely new.

---

## 10. Things that will confuse you

```
1. THE STAGES ARE NOT IN NUMERICAL ORDER.
   Book (5') runs before features (4) because market-pillar engines eat odds. Any
   diagram that shows a straight line is wrong.

2. "NO BETS TODAY" AND "PIPELINE BROKEN" LOOK IDENTICAL RIGHT NOW.
   Both produce an empty DataFrame. Finding B means the second is currently
   guaranteed. Every empty result must carry a reason.

3. THE £1 MINIMUM IS A FLOOR, NOT A LATTICE.  (corrected)
   An earlier draft of this file called stakes "discrete in £1 units, 6.7%
   granularity". That is wrong: £1 is the MINIMUM, and £1.01, £1.02 ... are all
   legal. Penny granularity on a £15 bankroll is 0.067% and can be ignored.
   The real constraint is the floor: a leg the allocator sizes at £0.40 must be
   ROUNDED UP to £1 (2.5x over-staked, breaking the cap) or DROPPED (losing the
   diversification the joint solve was buying). That is the design question,
   and it is a live one at £15 -- see AbstractStakeRounding in §5.
   Note lays interact with this favourably: a lay's risk is (d-1) x stake, so
   laying at 1.26 puts only £0.26 at risk for the £1 minimum. The morphism in
   finding F therefore also BUYS SMALLER MINIMUM POSITIONS on short prices.

4. p_market IS A BENCHMARK, NOT A PRICE -- BUT ON MATCH DAY IT IS ALSO A FEATURE.
   For market-pillar engines the same number enters twice, once as a model input
   and once as a scoring benchmark. If those two ever come from different fetches
   the diagnostics lie. One odds_df, threaded through.

5. THE CLOCK IS A DATETIME, NOT A DATE, EVERYWHERE.
   Kick-offs are epoch ints, lineups have scraped_at, book bars have ts. The
   prototype flattened all of it to `today()`.
```

---

## 11. Migration

```
DELETE
  deprecated_r01_matchday_runner.jl        (0 bytes)
  deprecated_runner.jl
  l00_matchday_utils.jl, l00_matchday_utils_restored.jl
  scratch_redis_{types,mappings,dashboard}.jl
  debug_script.jl, test_db.jl, clear_cache.jl
  src/unified_staking.jl                   -> Portfolio (keep the NOTES header verbatim)
  calculate_betting_signals                -> Portfolio

MOVE TO src/
  src/fixtures.jl   -> MatchDay/implementations/fixtures.jl      + horizon window, DateTime
  src/lineups.jl    -> MatchDay/implementations/lineups.jl       + SourceChain, as_of filter
  src/ratings.jl    -> DELETE the recursions, call Features      (Q6)
  src/inference.jl  -> MatchDay/inference.jl                     + per-feature materialiser
  src/live_betting.jl
      market-name mapping   -> Data (next to MarketConfig)
      Redis/order-book I/O  -> Data/fetchers  as AbstractBookSource
      dashboards            -> a runner, thin, over the stake sheet

STAY AS RUNNERS (a log, not a library)
  r00-r05, paper_tracks/*.md
  new: r01_replay.jl, r02_live_dryrun.jl, r03_clv_reconcile.jl

REQUIRES A CHANGE IN src/Portfolio
  build_books(spec, latents, expr, odds, ds) -> (..., fixtures::Dict{Int,FixtureInfo})
  and a test that an unplayed fixture produces a book with settle === nothing   (finding B)
```

---

## 12. Verified reference numbers

Measured 2026-08-06 against betdb `:5433` and a live server REPL. Re-run before trusting.

```
betfair.match_meta          8,367 SUCCESS/exact + 532 SUCCESS/extend, 2017-11-18..2026-06-27
  live-feed resolution      100% to w/c 2026-06-15, 55% w/c 06-22, 0% after     <- job stopped
  present_but_unresolved    0 in every competition                              <- never tried
betfair_live.order_book_1m  428,411 rows / 941 markets / 2026-05-29..08-02 14:57
  markets opened after      99                                                  <- collector down
  scaling                   x10000; bid = back, ask = lay
  overround check           back 1/1.25 + 1/4.80 = 1.0083 ; lay 1/1.26 + 1/5.00 = 0.9937
live competitions           Irish Div1 38, Irish Prem 33, Scot Prem 6, Scot Champ 5,
                            Scot L1 5, Scot L2 5, Fin Ykkosliiga 1
PF.fixture_table(Ireland)   1029 entries, 0 unplayed, latest 2026-08-02         <- finding B
ds.matches missing scores   0 of 1029
```

```
sofascore.lineup_provisional  220 rows / 10 matches / 2026-06-19..06-26
  confirmed = true            0 matches, ever                               <- finding E
  scrape lead time            4.4-5.8h before KO (one at 1.6h); XI lands ~1h out
order_book_1m columns         last_price_traded NULL in 428,411 of 428,411 (100%)
                              market_matched populated in 164,515 (38.4%)
```

Both brief-level unknowns are now closed:
- **Is the Redis feed alive?** No — confirmed down by the user. Postgres holds the only copy.
- **Is `order_book_1m` written by the same collector?** Yes, proven:
  `whatstheodds/pipeline/stream_worker.py` drains Redis into it with
  `ON CONFLICT (market_id, symbol, ts) DO UPDATE`. One adapter serves live and replay.

Still unverified:
- **Why did three jobs stop within days of each other in late June?** The resolver (last KO
  2026-06-22) and the lineup scraper (last run 2026-06-26) died together; the order-book drain
  survived until 2026-08-02. A common cause is worth ruling in or out before designing gates
  around them independently.
- **Does the sofascrape provisional scraper still work?** Its last successful run predates the
  outage, so "it fires at the wrong hour" (finding E) assumes it would succeed if fired at all.
