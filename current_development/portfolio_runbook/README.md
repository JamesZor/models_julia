# portfolio_runbook

Worked examples for `src/Portfolio` — how to actually use the staking manager, in the order
you would meet each piece. These are **runners only**: the loader is the `Portfolio` module
itself, so there are no `lXX` files here.

Read them in order; each builds on the previous one.

| file | what it teaches |
|---|---|
| `_setup.jl` | shared data loading (**ScottishLower** / `funnel_apm_xg`). Every runner includes it. Caches to `.jls`. |
| `_setup_ireland.jl` | same contract, **Ireland Premier** / `src_sup40_sw40`. Swap the include and every runner works unchanged. Ireland has 4x the median tick count — the better book. |
| `r01_quickstart.jl` | the whole pipeline in ~20 lines. Build books, group into slates, simulate, report. |
| `r02_policy_sweep.jl` | **the payoff of the design.** Build books once, sweep dozens of policies for free. |
| `r03_matchday_stakes.jl` | **the match-day recipe** — pricing unplayed fixtures into a stake sheet. |
| `r04_diagnostics.jl` | what to check before you believe a number. |
| `r05_extending.jl` | adding your own trust model and filter without touching `src/`. |

## Switching leagues

```julia
include("_setup.jl")            # ScottishLower / funnel_apm_xg
include("_setup_ireland.jl")    # Ireland Premier / src_sup40_sw40  -- no restart needed
ENV["RUNBOOK_RELOAD"]="1"       # force a rebuild of whichever is current
```

Both files guard on a `PORTFOLIO_LEAGUE` sentinel, not on `isdefined(Main, :ds)`. Guarding on
the latter means the second setup you include is a silent no-op and you keep the first league's
data. Book caches are named `books_<league>_<spechash>.jls` for the same reason:
`book_cache_key` hashes the *spec*, which is identical across leagues, so without the league tag
the two would share a file.

## The one idea to hold on to

The configuration is split in two, and the split is the whole design:

```
BookSpec    price, allocator, shrinkage, execution   -> determines a MatchBook -> EXPENSIVE, cache it
PolicySpec  trust, risk, cap, filter, grouping       -> pure multipliers on a book -> FREE
```

Building 628 books takes ~26 seconds because it re-solves a Kelly portfolio on 128 posterior
draws per match. Simulating a policy against those books takes milliseconds. So:

* change a **`BookSpec`** field → books must be rebuilt (`book_cache_key` changes).
* change a **`PolicySpec`** field → reuse the books you already have.

If you find yourself rebuilding books to try a different `lambda`, something has gone wrong.

## Match day in one call

```julia
sys   = PF.PortfolioSystem(PF.BookSpec(markets = MARKETS), PF.PolicySpec())
sheet = PF.stake_sheet(sys, latents_df, expr, odds_df, ds; bankroll = 1000.0)
PF.slate_summary(sheet)          # check EXPOSURE before you read the bets
```

`latents_df` for upcoming fixtures comes from `match_day_inference/src/inference.jl`
(`compute_todays_matches_latents`), not from `extract_oos_predictions`. `odds_df` needs the
`ds.odds` schema; a live feed with those columns works exactly like the historical summary.

Backtest and match-day share one code path. The only difference is that an unplayed book has
`settle == nothing`; it can be staked, and `simulate` refuses it rather than scoring a missing
result as a loss.

## The counter-intuitive bit

Once the drawdown constraint binds, **`trust` cannot change how much you stake — only what you
stake it on.** `risk_factor` solves for the factor that makes the stakes it is handed satisfy
the constraint, so handing it twice the stakes returns half the factor. `r02` demonstrates
this; `calibrate_lambda` is the dial that actually moves exposure.

## Health warning

The default policy's flat ROI on the only out-of-sample test available has a 95% bootstrap
interval that **includes zero**. `FlatTrust` is the default because every attempt to *learn*
per-selection trust lost money out of sample. Treat the non-default components as slots for
testing and rejecting ideas cheaply.
