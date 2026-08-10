# current_development/matchday_2026_08_08/r05_simple.jl
#
# ═══════════════════════════════════════════════════════════════════════════════════════════
#  Scottish Upper — load the model, look at the book, compare staking policies, read the ROI
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
#   julia --project -t 16
#   julia> include("current_development/matchday_2026_08_08/r05_simple.jl")
#
# Six blocks. Every one is small enough to paste on its own. All the ceremony — finding the
# experiment, rebuilding boundaries, checking for leakage, wiring the lineup chain — lives in
# l05_simple.jl and runs automatically.
#
# To price a different segment or day, change ONE line: the `matchday(...)` call in block 1.

using BayesianFootball
using DataFrames, Dates, Statistics, Printf

const DD = BayesianFootball.Data
const MD = BayesianFootball.MatchDay
const PF = BayesianFootball.Portfolio

include(joinpath(@__DIR__, "l02_slate_replay.jl"))   # slate_from_db, grade!, family_trust
include(joinpath(@__DIR__, "l05_simple.jl"))         # the façade

# ═══════════════════════════════════════════════════════════════════════════════════════════
# 1 · LOAD  — model, slate, book, and every safety check
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# Finds the newest ScottishUpper experiment under data/matchday_wknd_0808/ by itself. It prints
# which fold it will condition on and ERRORS if that fold was fitted on this slate.
#
# ⚠ The trained models are under `data/`, NOT under `current_development/`. The two directories
#   have confusingly similar names (`matchday_wknd_0808` vs `matchday_2026_08_08`); `matchday`
#   resolves it for you, and `find_experiment` says what it found.

ctx = matchday(DD.ScottishUpper(), Date(2026, 8, 8))

# EXPECT: 6 fixtures (2 in tournament 54, 4 in 55), all kicking off 14:00 UTC, all 6 results
#         present. (They were NOT present earlier on 2026-08-09 — the scrape for the top two
#         divisions lags the lower ones by hours. If it says "0/6 present", that is the feed,
#         not the model: everything below still prices, and `pnl` reads `missing` rather than
#         0.0 so the two cases stay distinguishable.)
#
#         It will also report conditioning on split 2 of 3, NOT the most recent. That is
#         correct: a DataStore rebuild regrew split 3's target window until it contained this
#         card, and split 2 is the fold whose NEXT round is this slate. It does mean the model
#         is a week behind — retrain rather than lean on the fallback.

# ═══════════════════════════════════════════════════════════════════════════════════════════
# 2 · THE MARKET BOOK  — what you could actually have traded
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# Best back and best lay, WITH the size available at each. The size columns are the point: the
# pipeline prices off the top of the book and never reads depth, so this is the only place a
# thin market announces itself.

bk = show_book(ctx; as_of = ctx.kickoff)
show(first(bk, 20), allrows = true, allcols = true); println()

@printf("\n  %d quoted selections across %d fixtures\n", nrow(bk), length(unique(bk.match_id)))
@printf("  median back size £%.2f   median lay size £%.2f\n",
        median(bk.back_size), median(bk.lay_size))

# One fixture at a time, if you prefer:
#   show_book(ctx; match_id = ctx.fixtures[1].m_id)

# ═══════════════════════════════════════════════════════════════════════════════════════════
# 3 · THE POLICIES  — what you are comparing
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# `policy(...)` takes plain numbers. The only argument that reliably moves how much you bet is
# `lambda` — trust is absorbed by the drawdown solver whenever it binds, which is nearly always.
# The "trust 0.25" row below is in the list as a CONTROL: it should come back identical to base.

POLICIES = Pair{String,Any}[
    "base λ23 cap.25"   => policy(),                                   # trust .5, λ23, cap .25
    "trust 0.25 (ctrl)" => policy(trust = 0.25),                       # expect: identical to base
    "λ 40 tighter"      => policy(lambda = 40),                        # smaller book
    "λ 10 looser"       => policy(lambda = 10),                        # bigger book
    "cap 10%"           => policy(cap = 0.10),                         # hard ceiling bites
    "no 1X2"            => policy(trust = family_trust()),             # per-family: 1X2 → 0
]

# ═══════════════════════════════════════════════════════════════════════════════════════════
# 4 · RETURNS  — every policy, at three times
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# `times` are minutes before kickoff. Costs one posterior rebuild per TIME (~10s), not per
# policy — policies are pure multipliers on a book that already exists.

res = returns(ctx, POLICIES; times = [60, 30, 0], bankroll = 1000.0)
show(res, allrows = true, allcols = true); println()

# HOW TO READ IT
#   exposure   fraction of bankroll live at once. THE number that can ruin you.
#   k_risk     what the drawdown budget cut full Kelly down to. Low + capped=false means
#              LAMBDA is the binding constraint, not the cap — so move lambda, not the cap.
#   growth     log(1 + P&L/bankroll). Rank on THIS, never on roi: ROI is P/L over stake, so a
#              uniform rescaling cancels out of it and every flat-trust cell reports the same
#              number for very different outcomes.
#   pnl        `missing` means the fixtures have no result yet. That is not zero.

# ROI as a grid, if that reads better:
if ctx.gradeable
    show(unstack(select(res, :policy, :mins_to_ko, :roi), :policy, :mins_to_ko, :roi),
         allrows = true, allcols = true)
    println()
end

# ═══════════════════════════════════════════════════════════════════════════════════════════
# 5 · STAKING COLUMNS  — which policy took which bet, side by side
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# One row per leg, one stake column per policy. 0.00 means that policy declined the leg.
# Sorted by the largest stake any policy put on it, so the positions that matter are at the top.

cols = stake_columns(ctx, POLICIES; as_of = ctx.kickoff, bankroll = 1000.0)
show(first(cols, 25), allrows = true, allcols = true); println()

# The "no 1X2" column should be 0.00 on every 1X2 row and LARGER than base elsewhere — per-family
# trust runs BEFORE the allocator, so the drawdown budget re-expands into what is left. A
# MarketWhitelist filter would run after the cap and simply truncate, leaving that capacity idle.

# ═══════════════════════════════════════════════════════════════════════════════════════════
# 6 · ONE FULL SHEET  — when you want the per-leg detail
# ═══════════════════════════════════════════════════════════════════════════════════════════

sheet = sheet_for(ctx, policy(); as_of = ctx.kickoff, bankroll = 1000.0)

if !isempty(sheet)
    show(first(select(sheet, :match_id, :group, :line, :selection, :side, :venue_selection,
                      :odds, :venue_odds, :p_model, :p_market, :edge, :risk, :venue_stake), 15),
         allrows = true, allcols = true)
    println()

    # What to actually place. `selection` is the runner the order touches; on a lay it is the
    # COMPLEMENT of the position, which is why `model_selection` is carried alongside.
    show(first(DataFrame([MD.order_ticket(r) for r in eachrow(sheet)]), 10),
         allrows = true, allcols = true)
    println()
end

# ═══════════════════════════════════════════════════════════════════════════════════════════
# 7 · PRE-GAME BREAKDOWN  — why the ROI came out where it did
# ═══════════════════════════════════════════════════════════════════════════════════════════
#
# Blocks 4-6 tell you WHAT the system did. This one tells you whether the model deserved to.
#
# Four tables: expected goals model vs market vs actual; 1X2 probabilities model vs market;
# proper scoring (log loss and Brier) of BOTH against the outcomes; and P&L per fixture.
#
# Read the scoring table first and the ROI last. On a handful of fixtures ROI is almost pure
# noise, while a one-sided goal-level gap or a collapsed 1X2 spread is visible immediately and
# is a property of the model rather than of the day.

pg = show_pregame(ctx; as_of = ctx.kickoff, bankroll = 1000.0)

# THE TWO NUMBERS TO WATCH WEEK TO WEEK
#
#  * `gap` one-sided across every fixture  → a LEVEL BIAS in expected goals. Not noise: if the
#    model is above the market on 6 of 6, that is a systematic offset you can correct.
#
#  * DISPERSION RATIO = sd(p_home model) / sd(p_home market). Below ~0.75 means the model cannot
#    separate fixtures as well as the market does, so its biggest apparent "edges" sit on the
#    underdogs of the most lopsided games — which is how ignorance gets sized as conviction.
#    Log it every week: if it climbs toward 1.0 as the season fills in, it was a cold start.
#    If it stays put, the engine is structurally under-dispersed and needs a wider prior.

println("""

  ═══════════════════════════════════════════════════════════════════════════════════
  Change the segment or the day in block 1 and everything else follows:

      ctx = matchday(DD.ScottishLower(), Date(2026, 8, 8))   # gradeable — has results
      ctx = matchday(DD.IrelandAll(),    Date(2026, 8, 7))

  More depth:
      r04_walkthrough.jl   the same thing step by step, with the internals exposed
      r03_replay_scot_lower.jl   all 41 snapshots + CLV, churn, fill, cold-start
      ARCHITECTURE.md      the system map and ten traps
  ═══════════════════════════════════════════════════════════════════════════════════
""")
