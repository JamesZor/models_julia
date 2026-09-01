# src/MatchDay/slate.jl
#
# THE SLATE IS THE UNIT OF EXECUTION.
#
# `Portfolio.stake_slate` solves one convex problem per SETTLEMENT WINDOW, not one per fixture:
# `SlateDrawdown` returns a single scalar `k` applied to every leg in the slate, and `FixedCap`
# rescales the whole vector the moment total simultaneous exposure exceeds its bound. Adding a
# 22nd fixture lowers `k` for the other 21.
#
# The consequence is a correctness property, not a batching preference:
#
#   A STAKE VECTOR IS ONLY VALID AS A VECTOR.
#
# Take 15 of 21 legs and the drawdown budget the other 6 were funding is unspent; take them in
# sequence and `k` was solved for a portfolio that never existed. `PricedSlate` therefore carries
# the slate-wide diagnostics -- `k_risk`, `slate_exposure`, `capped`, and the `λ`/cap they came
# from -- alongside the sheet, because NONE OF THEM IS RECOVERABLE AFTER THE FACT. Re-pricing the
# same legs with a different fixture list gives a different `k` for identical rows, so without
# these fields "why was this leg £26 and not £40?" has no answer.

export PricedSlate, price_slate, slate_batch_summary, leg_capacity, annotate_capacity!,
       sweep_ladder, fill_confidence, canonical_markets

"""
    PricedSlate

One pricing run over one settlement window: the sheet, the depth it was priced against, and the
slate-wide allocation diagnostics.

# Fields

* `sheet` -- one row per leg, `Portfolio.stake_sheet` plus MatchDay's execution columns
  (`side`, `venue_odds`, `venue_selection`, `risk`, `venue_stake`) and, once
  [`annotate_capacity!`](@ref) has run, the capacity columns.
* `books` -- the `BookLevels` each price was collapsed from, keyed `(match_id, SelectionKey)`.
  Carried rather than re-read: the fill model and the console ladder both need the levels, and a
  second query returns a book that has moved.
* `k_risk`, `slate_exposure`, `capped` -- read off `Portfolio.SlateAllocation` via the sheet.
* `risk_lambda`, `exposure_cap` -- the policy parameters actually in force, recorded so a slate
  is reproducible from its own row.
* `fold_idx`, `warning` -- which trained fold was conditioned on, and `select_split`'s complaint
  if it had one. An empty `warning` is a claim, so it is stored rather than printed and dropped.

`total_risk` is the sum of the `risk` column: the liability the account must reserve, in
currency. For a lay leg that is the liability, not the backer stake -- the morphism has already
denominated everything in risk by the time it reaches here.
"""
struct PricedSlate
    slate_id::UUID
    account_id::String
    window::Date
    as_of::DateTime
    bankroll::Float64
    sheet::DataFrame
    odds::DataFrame
    cards::Vector{FixtureCard}
    blocked::Vector{FixtureCard}
    instruments::Dict{Tuple{Int,SelectionKey},Instrument}
    books::Dict{Tuple{Int,SelectionKey},BookLevels}
    k_risk::Float64
    slate_exposure::Float64
    capped::Bool
    risk_lambda::Float64
    exposure_cap::Float64
    total_risk::Float64
    fold_idx::Int
    warning::String
end

n_legs(s::PricedSlate)     = nrow(s.sheet)
n_fixtures(s::PricedSlate) = length(unique(s.sheet.match_id))

"""
    canonical_markets() -> Data.MarketConfig

The market set this system prices: 1X2, BTTS, and Over/Under 0.5 / 1.5 / 2.5 / 3.5.

Deliberately **not** `MatchDaySpec`'s default, which also carries O/U 4.5. Measured on the
Scottish 26/27 book, O/U 4.5 has a median 2-4 tick spread but £0-£60 of matched volume in the
lower divisions, so it prices but does not trade. CORRECT_SCORE is excluded for the same reason
it is unmapped in `db.jl`.
"""
canonical_markets() = Data.MarketConfig(
    reduce(vcat, (Data.AbstractMarket[Data.Market1X2(), Data.MarketBTTS()],
                  [Data.MarketOverUnder(i + 0.5) for i in 0:3])))

# ===================================================================
# Capacity -- how much of a leg the book will actually take
# ===================================================================

"""
    sweep_ladder(prices, sizes, target) -> (; filled, vwap, slippage, levels)

Consume `target` currency of resting size down a price ladder, best first.

Returns what was actually filled, the volume-weighted price, the give-up against the touch as a
FRACTION of the touch price, and how many levels were consumed. `vwap` is `NaN` and `filled` is
`0.0` on an empty ladder.

The VWAP is computed in **probability space** (`Σ size / Σ (size / price)`), which is the correct
average for decimal odds: the arithmetic mean of two prices is not the price at which the
combined stake breaks even.

Only the levels the caller passes are considered. `betfair_live.order_book_1m` archives at most
**three**, verified over 635,765 rows, so a replay-derived answer is a LOWER bound on what the
live API would fill -- conservative in the safe direction.
"""
function sweep_ladder(prices::AbstractVector{<:Real}, sizes::AbstractVector{<:Real},
                      target::Real)
    n = min(length(prices), length(sizes))
    (n == 0 || target <= 0) &&
        return (; filled = 0.0, vwap = NaN, slippage = NaN, levels = 0)

    remaining = Float64(target)
    cost      = 0.0            # Σ size / price, i.e. the position in probability units
    levels    = 0
    @inbounds for i in 1:n
        p = Float64(prices[i])
        p > 1.0 || continue
        take = min(remaining, Float64(sizes[i]))
        take > 0 || continue
        cost      += take / p
        remaining -= take
        levels    += 1
        remaining <= 1e-9 && break
    end

    filled = Float64(target) - remaining
    filled <= 0 && return (; filled = 0.0, vwap = NaN, slippage = NaN, levels = 0)
    vwap  = filled / cost
    touch = Float64(prices[1])
    return (; filled, vwap, slippage = (touch - vwap) / touch, levels)
end

"""
    fill_confidence(depth_touch, venue_stake, slippage) -> Symbol

`:high`, `:medium` or `:low` -- the three-dot indicator the console renders.

* `:high` -- the touch alone covers the order. No slippage by construction.
* `:medium` -- the archived ladder covers it, giving up at most 1% against the touch.
* `:low` -- anything else: a partial fill, or a full one that costs more than 1%.

The threshold is 1% because that is where the measured edge stops surviving execution: on
Scottish League One/Two central lines the median model edge that clears staking is 2-5%, and the
§4.5 fill curve puts £100 orders at 1.2-2.1% slippage against 25-31% fill.
"""
function fill_confidence(depth_touch::Real, venue_stake::Real, slippage::Real)
    venue_stake <= 0 && return :high
    depth_touch >= venue_stake && return :high
    (isnan(slippage) || slippage > 0.01) && return :low
    return :medium
end

"""
    leg_capacity(levels::BookLevels, side::Symbol, venue_stake::Real) -> NamedTuple

What the book will do with one order, without placing it.

`side` selects which ladder is consumed, and the choice is not symmetric: a **back** order eats
the bid side (`levels.back`, prices available to back) and a **lay** order eats the ask side. The
sizes on both sides of a Betfair ladder are quoted as BACKER STAKE, which is exactly the
denomination of `venue_stake`, so the same sweep works for either.

Returns `depth_touch`, `depth_book` (the whole archived ladder), `filled`, `vwap`, `slippage`,
`levels_used`, `fillable` and `confidence`.
"""
function leg_capacity(levels::BookLevels, side::Symbol, venue_stake::Real)
    prices, sizes = side === :back ? (levels.back, levels.back_size) :
                    side === :lay  ? (levels.lay,  levels.lay_size)  :
                    error("side must be :back or :lay, got $side")

    depth_touch = isempty(sizes) ? 0.0 : Float64(sizes[1])
    depth_book  = isempty(sizes) ? 0.0 : sum(Float64, sizes)
    s = sweep_ladder(prices, sizes, venue_stake)
    fillable = s.filled >= Float64(venue_stake) - 1e-9
    conf = fillable ? fill_confidence(depth_touch, venue_stake, s.slippage) : :low
    return (; depth_touch, depth_book, filled = s.filled, vwap = s.vwap,
            slippage = s.slippage, levels_used = s.levels, fillable, confidence = conf)
end

"""
    annotate_capacity!(sheet, books) -> sheet

Add the capacity columns to a stake sheet, in place.

Adds `depth_touch`, `depth_book`, `expected_fill`, `expected_vwap`, `expected_slippage`,
`fillable` and `fill_confidence`. A row whose book is absent gets zero depth and `:low`, never a
missing value -- an unknown capacity is the same decision as no capacity, and a `missing` here
would propagate into the console's sort key.

**The order the sweep is applied to is the VENUE runner, not the model selection.** On a synthetic
those are different runners (`Instrument.venue_key`), and sweeping the model selection's ladder
for a lay would measure depth on a book the order never touches.
"""
function annotate_capacity!(sheet::DataFrame,
                            books::Dict{Tuple{Int,SelectionKey},BookLevels})
    n = nrow(sheet)
    depth_touch = zeros(Float64, n); depth_book = zeros(Float64, n)
    exp_fill    = zeros(Float64, n); exp_vwap   = fill(NaN, n)
    exp_slip    = fill(NaN, n);      fillable   = falses(n)
    conf        = fill(:low, n)

    for i in 1:n
        vkey = (group = sheet.group[i], line = sheet.line[i],
                selection = sheet.venue_selection[i])
        lv = get(books, (sheet.match_id[i], vkey), nothing)
        lv === nothing && continue
        c = leg_capacity(lv, sheet.side[i], sheet.venue_stake[i])
        depth_touch[i] = c.depth_touch; depth_book[i] = c.depth_book
        exp_fill[i]    = c.filled;      exp_vwap[i]   = c.vwap
        exp_slip[i]    = c.slippage;    fillable[i]   = c.fillable
        conf[i]        = c.confidence
    end

    sheet.depth_touch       = depth_touch
    sheet.depth_book        = depth_book
    sheet.expected_fill     = exp_fill
    sheet.expected_vwap     = exp_vwap
    sheet.expected_slippage = exp_slip
    sheet.fillable          = fillable
    sheet.fill_confidence   = conf
    return sheet
end

# ===================================================================
# The entry point
# ===================================================================

"""
    price_slate(spec, sys, segment, fit, ds; as_of, bankroll, account_id, slate_id) -> PricedSlate

Price one settlement window from a canonical fit, end to end.

    fixtures -> identity -> lineups -> BOOK -> features -> inference -> gate -> stake_sheet

`fit` may be a `Training.Fit` loaded from `mcmc_experiments` (see [`canonical_fit`](@ref)) or a
legacy `ExperimentResults`; both expose `.config.model`, `.config.splitter` and
`.training_results`, which is all the inference stage reads.

Differs from [`match_day`](@ref) in three ways, all of which exist because the ledger needs them:

1. it retains the **book** the prices were collapsed from, so the fill model needs no second read;
2. it retains the **slate diagnostics** (`k_risk`, exposure, `capped`, `λ`, cap), which are not
   recoverable from the sheet alone once the fixture list changes;
3. it annotates **capacity** per leg against that book.

`as_of` has no default here, unlike `match_day`. A live slate and a replayed one must be spelled
identically or the replay is not evidence about the live path.

Refuses a slate spanning more than one settlement window: `Portfolio` solves per window, so two
windows in one `PricedSlate` would carry one `k_risk` for two different joint problems.
"""
function price_slate(spec::MatchDaySpec, sys::Portfolio.PortfolioSystem, segment, fit, ds;
                     as_of::DateTime, bankroll::Real,
                     account_id::AbstractString = "default",
                     slate_id::UUID = uuid4())
    cards = build_cards(spec, segment, as_of)
    q     = quote_slate(spec, cards, as_of)

    for c in cards
        c.readiness = ready(spec.gate, c)
    end
    passed  = FixtureCard[c for c in cards if is_ready(c.readiness)]
    blocked = FixtureCard[c for c in cards if !is_ready(c.readiness)]

    window = isempty(cards) ? Date(as_of) : minimum(Date(c.fixture.kickoff) for c in cards)
    empty  = () -> PricedSlate(slate_id, String(account_id), window, as_of, Float64(bankroll),
                               _empty_slate_sheet(), q.odds, cards, blocked, q.instruments,
                               q.books, 1.0, 0.0, false, _policy_lambda(sys),
                               _policy_cap(sys), 0.0, 0, "")
    isempty(passed) && return empty()

    latents, diag = matchday_latents(spec, fit, ds, passed, q.odds, as_of)
    isempty(latents) && return empty()

    sheet = Portfolio.stake_sheet(sys, latents, fit, q.odds, fixture_info(passed);
                                  bankroll = bankroll)
    isempty(sheet) && return empty()

    _attach_instruments!(sheet, q.instruments, spec.rounding)
    isempty(sheet) && return empty()
    annotate_capacity!(sheet, q.books)

    windows = unique(sheet.slate)
    length(windows) == 1 || error(
        "price_slate: the sheet spans $(length(windows)) settlement windows ($(join(windows, ", "))). " *
        "Portfolio solves the drawdown budget PER WINDOW, so one PricedSlate cannot carry two -- " *
        "its single k_risk would belong to neither. Narrow `SofaScoreEvents(horizon = ...)` or " *
        "price each window separately.")

    return PricedSlate(slate_id, String(account_id), windows[1], as_of, Float64(bankroll),
                       sheet, q.odds, cards, blocked, q.instruments, q.books,
                       Float64(first(sheet.k_risk)), Float64(first(sheet.slate_exposure)),
                       Bool(first(sheet.capped)), _policy_lambda(sys), _policy_cap(sys),
                       sum(Float64, sheet.risk), diag.split, diag.warning)
end

_policy_cap(sys::Portfolio.PortfolioSystem) =
    hasproperty(sys.policy.cap, :cap) ? Float64(sys.policy.cap.cap) : NaN
_policy_lambda(sys::Portfolio.PortfolioSystem) =
    hasproperty(sys.policy.risk, :lambda) ? Float64(sys.policy.risk.lambda) : NaN

function _empty_slate_sheet()
    s = _empty_sheet()
    s.depth_touch = Float64[]; s.depth_book = Float64[]
    s.expected_fill = Float64[]; s.expected_vwap = Float64[]
    s.expected_slippage = Float64[]; s.fillable = Bool[]; s.fill_confidence = Symbol[]
    return s
end

"""
    slate_batch_summary(s::PricedSlate) -> NamedTuple

The header the operator reads BEFORE the bets, and the row `paper_slates` stores.

Exposure first, deliberately. A sheet is a list of attractive-looking prices; the only number
that says whether the vector is safe to commit is what fraction of the bankroll settles at once,
and whether `FixedCap` had to bind to get it there. `capped == true` on most slates means `λ` is
set too loose -- move `λ`, not the cap, because `risk_factor` is homogeneous of degree 0 and
rescaling the stakes is a no-op.
"""
slate_batch_summary(s::PricedSlate) = (
    slate_id       = s.slate_id,
    account_id     = s.account_id,
    window         = s.window,
    as_of          = s.as_of,
    bankroll       = s.bankroll,
    n_fixtures     = n_fixtures(s),
    n_legs         = n_legs(s),
    n_blocked      = length(s.blocked),
    total_risk     = s.total_risk,
    slate_exposure = s.slate_exposure,
    exposure_cap   = s.exposure_cap,
    k_risk         = s.k_risk,
    risk_lambda    = s.risk_lambda,
    capped         = s.capped,
    fold_idx       = s.fold_idx,
    n_low_confidence = count(==(:low), s.sheet.fill_confidence),
)
