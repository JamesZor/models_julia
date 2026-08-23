# current_development/matchday_2026_08_08/r04_walkthrough.jl
#
# ═══════════════════════════════════════════════════════════════════════════════════════════
#  A REPL WALKTHROUGH — Scottish League One + Two, Saturday 2026-08-08
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# This file is meant to be STEPPED THROUGH, not run. Paste one STEP at a time and look at what
# comes back. Every step ends with a `CHECK:` block giving the numbers this slate actually
# produces, so you can tell a working run from a broken one without knowing the code.
#
#   julia --project -t 16
#   julia> include("current_development/matchday_2026_08_08/r04_walkthrough.jl")   # or paste
#
# Running the whole file top to bottom also works and takes about a minute.
#
# ───────────────────────────────────────────────────────────────────────────────────────────
# WHY THIS SLATE
# ───────────────────────────────────────────────────────────────────────────────────────────
#
# 10 fixtures (5 in tournament 56, 5 in 57), ALL kicking off 14:00 UTC. One settlement window,
# so the drawdown budget and the exposure cap bind across the whole card — which is the only
# configuration where "portfolio" means anything.
#
# It is also the only Scottish slate this weekend that can be GRADED. As of 2026-08-09 every
# fixture in tournaments 54 and 55 is still `status_type = 'notstarted'` in sofascore.events,
# including Friday's — the result scrape has not run for the top two divisions. Their order book
# and crosswalk are fine, so they can be PRICED (see STEP 13); they just cannot be scored yet.
#
# ───────────────────────────────────────────────────────────────────────────────────────────
# THE SHAPE OF WHAT FOLLOWS
# ───────────────────────────────────────────────────────────────────────────────────────────
#
#   1-3   load        DataStore, experiment (the trained model), and what each actually holds
#   4     SAFETY      which split we condition on, and proving it is not fitted on this slate
#   5     the slate   fixtures + results, straight from SQL
#   6-7   the book    raw order-book depth, then how two prices become one tradeable instrument
#   8-9   config      MatchDaySpec (where numbers come from) + PortfolioSystem (how much to bet)
#   10    RUN         one match_day call, every intermediate inspected
#   11    the sheet   slate summary, bets, order tickets
#   12    returns     grade against the result
#   13    variations  other policies, other instants, other segments
#
# Full architecture: current_development/matchday_2026_08_08/ARCHITECTURE.md

using BayesianFootball
using DataFrames, Dates, Statistics, Printf

const DD  = BayesianFootball.Data
const EXP = BayesianFootball.Experiments
const MD  = BayesianFootball.MatchDay
const PF  = BayesianFootball.Portfolio

# include(joinpath(@__DIR__, "l02_slate_replay.jl"))
include("current_development/matchday_2026_08_08/l02_slate_replay.jl")

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 1 — the DataStore
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# A DataStore is every domain for one SEGMENT, already fetched, processed and QA'd: matches,
# odds, lineups, stats, incidents, bbc. `ScottishLower()` is the singleton for tournaments
# [56, 57]; segments are defined in src/Data/fetchers/segments.jl and nowhere else.
#
# ⚠ THE CACHE HAS A 48-HOUR TTL AND REBUILDS ITSELF SILENTLY ON LOAD. That matters more than it
#   sounds — see STEP 4. If this prints "Fetching fresh data" rather than "from local cache",
#   you have just acquired the results of the matches you are about to price.

ds = DD.load_datastore_cached(DD.ScottishLower())

@info "DataStore" segment = ds.segment matches = nrow(ds.matches) last = maximum(ds.matches.match_date)

# CHECK:  matches ≈ 1990, last = 2026-08-08.
#         `last` INCLUDING Saturday is expected and is not itself a leak — the leak question is
#         which fold we condition on, which is STEP 4.

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 2 — the model
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# An "experiment" is what `r01_train_weekend.jl` produced overnight. It is NOT a single fitted
# object — it is one MCMC chain per temporal fold, plus the config that generated them.
#
#   expr.training_results[i][1]   the chain for fold i
#   expr.config.model             the engine (its component configuration lives here)
#   expr.config.splitter          the GroupedCVConfig that cut the folds
#
# This one is the 2-layer shots→goals funnel: the only engine that can price 56/57 at all,
# because those leagues have no SofaScore stats, no xG and no player ratings. BBC shot counts
# are the entire observable.

expr = EXP.load_experiment("./data/matchday_wknd_0808/scot_lower_funnel_20260807_012812")
expr = EXP.load_experiment("./data/matchday_wknd_0808/scot_upper_poisson_outfield_20260807_011126")

@info "experiment" model = nameof(typeof(expr.config.model)) folds = length(expr.training_results)

# CHECK:  model = DynamicFunnelDoublePoissonGoalsLeagueTimeDecayModel, folds = 2.
#         folds MUST be > 1. A run reporting folds = 1 trained on history only and never saw
#         the target season — that is the r05 failure `l01.assert_splits` exists to prevent.

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 3 — what the model needs to price an UNSEEN fixture
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# A trained chain alone cannot price Saturday. `extract_parameters` looks each fixture up in the
# FeatureSet's per-match lookup maps, and a fixture played after training is in none of them.
#
# `INJECTABLE_KEYS` is the list of maps that must therefore be MATERIALISED at serving time.
# Both are read as `get(map, match_id, <default>)`, so a fixture missing from either is priced
# SILENTLY off the fallback — which is why `check_coverage` exists and why it is strict.

println("\ninjectable feature maps: ", MD.INJECTABLE_KEYS)
println("  :player_ratings_map  → per-side positional rating sums   (player-level engines)")
println("  :league_lookup       → tournament_id → training league index; zeroing it prices a")
println("                         56 fixture at the MEAN of League One and League Two")

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 4 — ⚠ THE SAFETY CHECK. Do not skip this one.
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# `select_split` chooses which chain to condition on as `idx = min(n_trained, n_rebuilt)` —
# BY POSITION, not by date. The boundary list is rebuilt from TODAY's DataStore, and because the
# cache refreshes every 48h it will have grown since training.
#
# So the question is never "is the model good", it is "was this chain fitted on the matches I am
# about to price". Answer it explicitly, every time.

bounds = DD.create_id_boundaries(ds, expr.config.splitter)
idx    = min(length(expr.training_results), length(bounds))

slate_ids = Set([16362408, 16362409, 16362410, 16362412, 16362413,
                 16362579, 16362580, 16362585, 16362588, 16362589])
leak = length(intersect(slate_ids, Set(bounds[idx][1].target_match_ids)))

@printf("  trained folds %d | rebuilt boundaries %d | conditioning on split %d\n",
        length(expr.training_results), length(bounds), idx)
@printf("  target sizes  %s\n", string([length(b[1].target_match_ids) for b in bounds]))
@printf("  slate fixtures inside split %d's TARGET window: %d  %s\n",
        idx, leak, leak == 0 ? "← leak-free" : "← LEAKAGE")

# CHECK:  trained 2, rebuilt 3, conditioning on 2, target sizes [0, 10, 20], leak = 0.
#
#         Read WHY that is safe: boundary 3's target window is 20 matches and contains all ten
#         of Saturday's. We avoid it only because min(2, 3) = 2. That is ARITHMETIC LUCK, not a
#         temporal cutoff. Retrain today and idx becomes 3, and the model would be conditioning
#         on a fold fitted on the very card it is pricing — with no error raised.
#         (Trap T2 in ARCHITECTURE.md. The fix is to select the split by as_of.)

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 5 — the slate, and its result
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# Fixtures come from SQL rather than from `ds.matches`, for two reasons: the DataStore is a
# curated store of FINISHED matches (so it cannot describe an upcoming card), and we want the
# kickoff as a DateTime — every gate is a function of time-to-kickoff.
#
# `slate_from_db` returns both the fixtures and a match_id → (home, away) score map. If the
# result scrape has not run it warns and returns an empty map: you can still price, you just
# cannot grade. That is exactly the state tournaments 54/55 are in right now.

MATCH_DAY = Date(2026, 8, 8)
fixtures, results = slate_from_db(DD.tournament_ids(DD.ScottishLower()), MATCH_DAY)

show(DataFrame(match_id = [f.m_id for f in fixtures],
               fixture  = ["$(f.home) v $(f.away)" for f in fixtures],
               tourn    = [f.tournament_id for f in fixtures],
               kickoff  = [f.kickoff for f in fixtures],
               score    = [haskey(results, f.m_id) ? "$(results[f.m_id][1])-$(results[f.m_id][2])" : "—"
                           for f in fixtures]), allrows = true, allcols = true)
println()

# CHECK:  10 fixtures, all kicking off 2026-08-08T14:00, every one with a score.
#         14:00 UTC = 15:00 BST — the traditional Saturday 3pm.

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 6 — the market book, raw
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# Before any pricing, find out what the book can actually support. `book_coverage` answers the
# question that governs everything downstream: how far back does the archive go, and how often
# did it tick?

cov = book_coverage(fixtures)
show(select(cov, :match_id, :fixture, :n_mkts, :first_tick, :last_tick, :n_snaps, :matched_close),
     allrows = true, allcols = true)
println()

# CHECK:  11 markets each, first_tick 2026-08-08T12:00, n_snaps ≈ 40.
#         first_tick is T-120, so the replayable window is the final TWO hours, and 40 snapshots
#         over 120 minutes means a 3-minute cadence — despite the table being called
#         `order_book_1m`. Always read the cadence off this table rather than assuming it.

# ── 6b. one fixture's actual depth ─────────────────────────────────────────────────────────
#
# `quotes` returns DEPTH, not a scalar. This is the rawest thing in the pipeline: what was
# actually on the exchange at a given instant.
#
#   back / back_size  the BID side — prices and sizes available to BACK
#   lay  / lay_size   the ASK side — prices and sizes available to LAY
#
# (Verified by overround sign: the back side sums above 1, the lay side below.)

kick_off_m1 = DateTime(2026, 8, 8, 14, 0)          # kickoff = the closing book
kick_off_m2 = DateTime(2026, 8, 8, 14, 20)          # kickoff = the closing book
f_ross = first(f for f in fixtures if f.home == "ross-county")
ident  = MD.resolve(MD.MatchMetaCrosswalk(), f_ross)
book1   = MD.quotes(MD.ArchivedOrderBook(), ident, kick_off_m1)
book2   = MD.quotes(MD.ArchivedOrderBook(), ident, kick_off_m2)

function show_market_book_at_time(book)
    show(sort!(DataFrame(
            market = ["$(k.group) $(k.line == 0.0 ? "" : k.line)" for k in keys(book)],
            sel    = [k.selection for k in keys(book)],
            back   = [MD.best_back(b) for b in values(book)],
            lay    = [MD.best_lay(b)  for b in values(book)],
            b_size = [isempty(b.back_size) ? 0.0 : b.back_size[1] for b in values(book)],
            l_size = [isempty(b.lay_size)  ? 0.0 : b.lay_size[1]  for b in values(book)],
            matched = [b.matched for b in values(book)]), [:market, :sel]),
         allrows = true, allcols = true)
    println()
end


show_market_book_at_time(book1)
show_market_book_at_time(book2)

# CHECK:  17 selections (1X2 ×3, BTTS ×2, O/U 0.5–5.5 ×12).
#
#         LOOK AT THE SIZE COLUMNS, not just the prices. On BTTS you will see ~£1–2 available on
#         one side. That is the binding constraint on this league and nothing in `src` reads it
#         (trap T8) — `BestAvailable` takes the price and discards the depth.


#=
julia> show_market_book_at_time(book1)
17×7 DataFrame
 Row │ market         sel       back     lay      b_size    l_size   matched
     │ String         Symbol    Float64  Float64  Float64   Float64  Float64
─────┼───────────────────────────────────────────────────────────────────────
   1 │ 1X2            away         7.8      8.4     2.0299   184.06  3777.41
   2 │ 1X2            draw         4.5      4.7    54.7       38.24  3777.41
   3 │ 1X2            home         1.5      1.55  517.63      28.0   3777.41
   4 │ BTTS           btts_no      1.99     2.04  342.1        1.0    127.08
   5 │ BTTS           btts_yes     1.96     2.02    1.91       1.03   127.08
   6 │ OverUnder 0.5  over_05      1.06     1.07  457.89      64.14   101.5
   7 │ OverUnder 0.5  under_05    16.0     18.0     3.0       74.9    101.5
   8 │ OverUnder 1.5  over_15      1.27     1.29    5.0       54.22   183.65
   9 │ OverUnder 1.5  under_15     4.5      4.9    18.0        9.92   183.65
  10 │ OverUnder 2.5  over_25      1.78     1.83    3.0       16.0    236.56
  11 │ OverUnder 2.5  under_25     2.2      2.3    17.0       93.55   236.56
  12 │ OverUnder 3.5  over_35      2.96     3.15    7.0       17.04    88.0
  13 │ OverUnder 3.5  under_35     1.46     1.5    30.0        3.0     88.0
  14 │ OverUnder 4.5  over_45      6.0      6.6     1.0        7.01   237.42
  15 │ OverUnder 4.5  under_45     1.18     1.2    66.85       4.0    237.42
  16 │ OverUnder 5.5  over_55     13.0     18.5    11.33      29.89     0.52
  17 │ OverUnder 5.5  under_55     1.06     1.09  263.65     183.95     0.52

julia> show_market_book_at_time(book2)
17×7 DataFrame
 Row │ market         sel       back     lay       b_size   l_size   matched
     │ String         Symbol    Float64  Float64   Float64  Float64  Float64
─────┼───────────────────────────────────────────────────────────────────────
   1 │ 1X2            away         8.2    10.0        9.0      2.89  5168.58
   2 │ 1X2            draw         4.3     5.2        9.0     11.0   5168.58
   3 │ 1X2            home         1.46    1.55      12.05     4.0   5168.58
   4 │ BTTS           btts_no      1.61    1.8       18.29    79.35   197.53
   5 │ BTTS           btts_yes     2.22    2.66       1.5     11.4    197.53
   6 │ OverUnder 0.5  over_05      1.09    1.11       3.15   118.0    427.4
   7 │ OverUnder 0.5  under_05    10.0    15.0        3.0      7.48   427.4
   8 │ OverUnder 1.5  over_15      1.34    1.45      13.63    49.0    204.23
   9 │ OverUnder 1.5  under_15     3.5     4.1        3.04     5.17   204.23
  10 │ OverUnder 2.5  over_25      2.14    2.36       8.19    20.0    237.37
  11 │ OverUnder 2.5  under_25     1.76    1.85       2.0      6.17   237.37
  12 │ OverUnder 3.5  over_35      1.03    4.7      194.7      3.76    88.09
  13 │ OverUnder 3.5  under_35     1.27    1.52      16.43     5.11    88.09
  14 │ OverUnder 4.5  over_45      8.8    12.0        1.88     4.0    237.42
  15 │ OverUnder 4.5  under_45     1.09    1.1399   123.0    118.0    237.42
  16 │ OverUnder 5.5  over_55     22.0   NaN         26.93     0.0      0.92
  17 │ OverUnder 5.5  under_55     1.02    1.05     235.63   257.05     0.92
=#


# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 7 — two prices become one tradeable instrument
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# THE MORPHISM, and the one piece of maths worth internalising:
#
#     backing X  ≡  laying NOT-X,  once the position is measured in RISK rather than in stake
#
#     lay Under at d with backer stake b  →  risk = b(d−1),  win = b
#     set risk = s, so b = s/(d−1)        →  win/risk = 1/(d−1)
#     a back at D has win/risk = D−1      →  D = d/(d−1)
#
# So `instrument()` emits (effective odds, side, venue price, leverage, venue runner) and
# EVERYTHING downstream — payoff matrix, Kelly, drawdown budget, exposure cap — is denominated
# in risk and never learns that lays exist. `FixedCap` sums liability by construction.
#
# It leaks in exactly one place: the order ticket, where "which runner" matters again. That is
# what `venue_key` is for.


function show_best_option_pricing(book)
  ks = collect(keys(book))
  for k in ks
      k.group == "BTTS" || (k.group == "OverUnder" && k.line == 2.5) || continue
      inst = MD.instrument(MD.BestOfBackLay(), k, MD.complement_of(k, ks), book, MD.BestAvailable())
      inst === nothing && continue
      direct = MD.instrument(MD.DirectBackOnly(), k, nothing, book, MD.BestAvailable())
      @printf("  want %-10s → %-4s %-10s @ %6.3f   effective %6.3f  (direct back %6.3f, %+0.2f%%)\n",
              k.selection, uppercase(string(inst.side)), inst.venue_key.selection,
              inst.venue_odds, inst.odds, direct.odds, 100 * (inst.odds / direct.odds - 1))
  end
end 

show_best_option_pricing(book1)
show_best_option_pricing(book2)



# CHECK:  Rows where `side = LAY` name a DIFFERENT runner than the one you asked for — that is
#         the morphism working. `btts_yes` taken by LAYING `btts_no` is the same position.
#
#         The final column is what the synthetic bought you, and it is usually a rounding error.
#         Note it is NOT free: on this fixture at T-30 the synthetic won by +0.22% of price and
#         moved from a £135-deep book to a £10-deep one. Price improvement, capacity destroyed.

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 8 — MatchDaySpec: WHERE THE NUMBERS COME FROM
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# Nine swappable seams. Every one has a default; the ones overridden here are overridden for a
# reason, and the reasons are the interesting part.
#
#   fixtures   ExplicitFixtures     `SofaScoreEvents` filters status='notstarted', so a PLAYED
#                                   day is invisible to it. Replay MUST use an explicit list.
#   identity   MatchMetaCrosswalk   answers 10/10 here. No `LiveNameMatch` fallback: a fallback
#                                   that never fires is one you cannot audit.
#   lineups    SourceChain()        EMPTY, deliberately. The funnel engine reads no lineup, so
#                                   fetching one would add a gate reason with no bearing on the
#                                   price. (A player-level engine needs the full chain — STEP 13.)
#   gate       IdentityResolved     + MaxBookAge(10min). Conjunctive: runs ALL gates and
#                                   concatenates reasons, so you can tell a dead resolver from
#                                   a dead collector.
#
# Print `spec` at the REPL — it renders as the pipeline, in execution order.

spec = replay_spec(fixtures)
display(spec)

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 9 — PortfolioSystem: HOW MUCH TO STAKE
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# Two halves, and the split is the single most useful thing to understand about this module:
#
#   BookSpec    ═══ EXPENSIVE. This IS the cache key. ~40ms per match. ═══
#     markets     which markets to price
#     price       DeArb — shrink quotes toward a fair book before settling
#     allocator   KellyLogUtility — ONE joint solve over all 144 scorelines, not a list of bets
#     shrink      BakerMcHale — re-solves on 128 posterior draws, prices parameter uncertainty
#     exec        2% commission, stake bounds, require_complete_markets
#
#   PolicySpec  ═══ CHEAP. Pure multipliers on an already-built book. Sweep freely. ═══
#     trust       how much to believe the model over the market
#     risk        the drawdown budget
#     cap         the hard ceiling on simultaneous exposure
#     filter      curation, applied LAST
#     grouping    what settles together
#
# ⚠ TRUST IS ALMOST CERTAINLY A NO-OP HERE. `risk_factor` is homogeneous of degree 0, so once
#   the drawdown constraint binds it undoes any uniform rescaling. Measured on this exact slate:
#   trust 0.25 / 0.35 / 0.5 / 1.0 give BIT-IDENTICAL books, with trust × k_risk constant at
#   0.1316. To move exposure, move `lambda`. Trust only does work when it DIFFERS between
#   selections (see STEP 13).

sys = PF.PortfolioSystem(
    PF.BookSpec(markets = MD.MatchDaySpec().markets),
    PF.PolicySpec(trust  = PF.FlatTrust(0.5),
                  risk   = PF.SlateDrawdown(lambda = 23.0, mode = :sequential),
                  cap    = PF.FixedCap(0.25),
                  filter = PF.KeepAll()))

BANKROLL1 = 1000.0

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 10 — RUN IT
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# One call does stages 1–8: fixtures → identity → lineups → book → instruments → gates →
# feature materialisation → posterior latents → joint Kelly → execution columns.
#
# `as_of` is a CALL-SITE argument. No stage reads the clock internally, which is the whole
# reason a past match day is replayable at all. Set it to `now(UTC)` and this is live pricing.

res = MD.match_day(spec, sys, DD.ScottishLower(), expr, ds; as_of = kick_off_m1, bankroll = BANKROLL1)

display(res)

# CHECK:  10 fixtures, 10 priced, 0 blocked, 45 bets, 170 quotes.
#         The `select_split` warning about boundary counts is expected — that is STEP 4's
#         min(2,3), already checked by hand.

# ── 10b. A REFUSAL IS A VALUE ──────────────────────────────────────────────────────────────
#
# Read this BEFORE concluding anything from an empty sheet. "The gate refused everything" and
# "the model found no edge" produce the same empty DataFrame otherwise.

br = MD.blocked_report(res)
isempty(br) ? println("\n  nothing blocked — the gate passed every fixture.") :
              show(br, allrows = true, allcols = true)

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 11 — the sheet
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# READ THE SLATE BEFORE THE BETS. Exposure is what can ruin you; individual stakes cannot.

sheet = res.sheet
@printf("\n  legs            %d over %d fixtures\n", nrow(sheet), length(unique(sheet.match_id)))
@printf("  total risk      £%.2f   (%.2f%% of bankroll live simultaneously)\n",
        sum(sheet.risk), 100 * sum(sheet.risk) / BANKROLL1)
@printf("  k_risk          %.4f   ← drawdown budget cut stakes to this fraction of full Kelly\n",
        first(sheet.k_risk))
@printf("  hard cap bound  %s\n", first(sheet.capped))

# CHECK:  45 legs / 9 fixtures, risk £150.85, exposure 15.08%, k_risk 0.2632, capped false.
#
#         `capped = false` with k_risk ≈ 0.26 tells you the DRAWDOWN BUDGET is holding the book
#         down, not the hard cap. Those are different knobs: to bet more, raise `lambda`; the
#         cap is not binding and raising it would change nothing.
#
#         9 fixtures, not 10 — one fixture's book was too thin to quote a complete market group
#         at this instant. `extract_selections` rejects a group unless EVERY outcome is quoted,
#         because de-vigging a partial group manufactures edge on the survivors.

disp = select(sheet, :match_id, :group, :line, :selection, :side, :venue_selection,
              :odds, :venue_odds, :p_model, :p_market, :edge, :risk, :venue_stake)
for c in (:odds, :venue_odds, :p_model, :p_market, :edge); disp[!, c] = round.(disp[!, c], digits = 3); end
for c in (:risk, :venue_stake); disp[!, c] = round.(disp[!, c], digits = 2); end
show(first(sort(disp, :risk, rev = true), 15), allrows = true, allcols = true)
println()

# CHECK:  `selection` and `venue_selection` DIFFER on every :lay row. That is the fix from
#         commit de41353 — before it, the ticket named the position while quoting the
#         complement's price, i.e. an instruction to place the opposite bet.
#
#         You will also see rows with NEGATIVE edge carrying a stake. Not a bug: this is one
#         joint solve over 144 scorelines, so a small negative-edge leg often hedges a larger
#         correlated one in the same match. Judge the sheet per match, never per row.

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 12 — order tickets, then returns
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# `order_ticket` is the last step and the only place the back/lay distinction becomes visible
# again. `selection` is what you type into the exchange; `model_selection` is the position it
# expresses and the key you grade against.

tickets = DataFrame([MD.order_ticket(r) for r in eachrow(sheet)])
show(first(sort(tickets, :stake, rev = true), 8), allrows = true, allcols = true)
println()

# ── grading ────────────────────────────────────────────────────────────────────────────────
#
# Two things here are easy to get wrong and both were wrong in earlier hand-rolled versions:
#
#   • stake `:risk`, NOT `:stake`. `stake` is what Portfolio wanted; `risk` is what survived
#     the exchange minimum. With `NoMinimum` they coincide, so the bug hides until you switch.
#   • grade at `:odds` (EFFECTIVE), not `:venue_odds`. The morphism means a lay in risk units
#     settles exactly like a back at d/(d−1), so one formula covers both sides.

graded = grade!(copy(sheet), results, sys)

@printf("\n  staked   £%.2f\n  P&L      £%+.2f\n  ROI      %+.2f%%\n  W/L/P    %d / %d / %d\n",
        sum(graded.risk), sum(graded.pnl), 100 * sum(graded.pnl) / sum(graded.risk),
        count(coalesce.(graded.graded, false)),
        count(coalesce.(.!graded.graded, false)),
        count(ismissing.(graded.graded)))

show(combine(groupby(graded, :group),
             nrow => :legs,
             :risk => (x -> round(sum(x), digits = 2)) => :risk,
             :pnl  => (x -> round(sum(x), digits = 2)) => :pnl,
             [:pnl, :risk] => ((p, r) -> round(100sum(p) / sum(r), digits = 1)) => :roi_pct),
     allcols = true)
println()

# CHECK:  staked £150.85, P&L −£44.48, ROI −29.5%.
#         Per family: 1X2 −£23.93 on £87.58, OverUnder −£18.01 on £37.63, BTTS −£2.53 on £25.63.
#
#         ⚠ THIS NUMBER MEASURES NOTHING ABOUT EDGE. n = 10 matches, one slate. The Portfolio
#           backtest's ROI confidence interval includes zero over 628 matches; one Saturday
#           cannot do better than that. What it DOES measure is that the pipeline produced a
#           complete, gradeable, correctly-ticketed book — which is a mechanical property and
#           one slate measures it fine.

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 13 — variations
# ═══════════════════════════════════════════════════════════════════════════════════════════

# ── 13a. a different policy, same book (CHEAP — pure multipliers) ──────────────────────────
#
# Everything in `PolicySpec` is a multiplier on books already built, so a sweep is milliseconds.
# Judge on `growth`, never ROI: ROI is P/L over stake, so any uniform rescaling cancels out of
# it exactly and every flat-trust cell reports the same number for very different outcomes.

latents_close, _ = MD.matchday_latents(spec, expr, ds,
                                       [c for c in res.cards if MD.is_ready(c.readiness)],
                                       res.odds, AS_OF)
close = (latents = latents_close, odds = res.odds,
         fixtures = MD.fixture_info([c for c in res.cards if MD.is_ready(c.readiness)]))

sd(l; m = :sequential) = PF.SlateDrawdown(lambda = l, mode = m)
sweep = policy_sweep(sys, expr, close, results, Pair{String,Any}[
    "base (trust .5, cap .25, λ23)" => sys.policy,
    "no 1X2 (per-family trust)"     => PF.PolicySpec(trust = family_trust(), risk = sd(23.0),
                                                    cap = PF.FixedCap(0.25), filter = PF.KeepAll()),
    "λ 40 — tighter drawdown"       => PF.PolicySpec(trust = PF.FlatTrust(0.5), risk = sd(40.0),
                                                    cap = PF.FixedCap(0.25), filter = PF.KeepAll()),
    "trust 0.25 (control)"          => PF.PolicySpec(trust = PF.FlatTrust(0.25), risk = sd(23.0),
                                                    cap = PF.FixedCap(0.25), filter = PF.KeepAll()),
]; bankroll = BANKROLL, instruments = res.instruments)
show(sweep, allrows = true, allcols = true)
println()

# CHECK:  "trust 0.25" is IDENTICAL to base on staked/P&L/ROI, with k_risk exactly doubled.
#         That is the homogeneity property, confirming itself. Two identical cells is
#         information, not a bug.
#
#         "no 1X2" stakes LESS but is not merely base-with-rows-deleted: per-family trust runs
#         BEFORE the allocator, so the drawdown budget re-solves and re-expands what is left.
#         `MarketWhitelist` would run after the cap and simply truncate (trap T7).

# ── 13b. a different instant — how the book moves into kickoff ─────────────────────────────
#
# Same call, different `as_of`. This is the whole replay capability in one line.


AS_OF = DateTime(2026, 8, 8, 14, 0)          # kickoff = the closing book
BANKROLL=1000

for t in (AS_OF - Minute(60), AS_OF - Minute(30), AS_OF)
    r = MD.match_day(spec, sys, DD.ScottishLower(), expr, ds; as_of = t, bankroll = BANKROLL)
    g = isempty(r.sheet) ? nothing : grade!(copy(r.sheet), results, sys)
    @printf("  T−%-3d  legs %2d  risk £%7.2f  P&L £%+8.2f\n",
            Dates.value(Minute(AS_OF - t)), nrow(r.sheet),
            g === nothing ? 0.0 : sum(g.risk), g === nothing ? 0.0 : sum(g.pnl))
end

# For all 41 snapshots plus CLV, churn, fill feasibility and cold-start diagnostics, run the
# full harness instead:  r03_replay_scot_lower.jl

# ── 13c. a different segment ───────────────────────────────────────────────────────────────
#
# Change four things: the segment, the experiment path, the lineup source, and the match day.
#
    ds_up   = DD.load_datastore_cached(DD.ScottishUpper(), force=true)          # tournaments [54, 55]
    expr_up = EXP.load_experiment("./data/matchday_wknd_0808/scot_upper_poisson_outfield_20260807_011126")
    fx_up, res_up = slate_from_db(DD.tournament_ids(DD.ScottishUpper()), Date(2026,8,8))
    spec_up = MD.MatchDaySpec(
        fixtures = MD.ExplicitFixtures(fx_up),
        identity = MD.MatchMetaCrosswalk(),
        lineups  = MD.SourceChain(MD.ProvisionalDB(), MD.LastHistorical(ds_up)),   # ← REQUIRED
        gate     = MD.GateChain(MD.IdentityResolved(), MD.MaxBookAge(Minute(10))))
#
# TWO THINGS WILL BITE YOU THERE, and both are worth meeting deliberately:
#
#  1. `LastHistorical(ds_up)` is NOT optional. The upper-division engine is player-level, so it
#     needs an XI. The DEFAULT spec builds `LastHistorical()` with no DataStore, which returns
#     `nothing` unconditionally — any fixture without a provisional XI then gets no lineup,
#     `RatingsFromTracker` skips it, and `check_coverage` aborts the WHOLE segment.
#
#  2. YOU CANNOT GRADE IT YET. As of 2026-08-09 every 54/55 fixture is still 'notstarted' in
#     sofascore.events — the result scrape has not run for the top two divisions, including
#     Friday's game. `slate_from_db` warns and returns an empty results map, so `grade!` marks
#     every leg `missing` and P&L is 0.0. That is the data feed, not the model.
#     Re-run STEP 12 once the scrape catches up.

println("""

  ═══════════════════════════════════════════════════════════════════════════════════
  Done. Where to go next:
    ARCHITECTURE.md            the full system map, and ten traps worth knowing
    r03_replay_scot_lower.jl   all 41 snapshots + CLV, churn, fill, cold-start, sweep
    ../portfolio_runbook/      what happens inside stake_sheet
    ../matchday_runbook/       what happens inside match_day
  ═══════════════════════════════════════════════════════════════════════════════════
""")
