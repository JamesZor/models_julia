# src/MatchDay/ledger/state_machine.jl
#
# `decide_order` is a PURE FUNCTION of `(order, context)`. It touches no database, no socket and
# no clock of its own. Everything impure sits in `db.jl` and `reservation.jl` around it.
#
# That split is the whole point, and it is copied deliberately from the Python collector's
# `matchday_supervisor.decide()`, which is the one piece of this stack that can be replayed as a
# table of numbers rather than only on a real Saturday. A 15:00 slate here is the same: hand
# `decide_order` a book and an account and the transition is reproducible, testable and
# arguable without any infrastructure at all.
#
# EVERY REFUSAL IS A VALUE. A leg that fails a gate returns a `CANCELLED` transition carrying the
# reason, not `nothing`. `MatchDay` already treats a refusal that way (`MatchDayResult.blocked`),
# and if the ledger did not, "no bets today" and "the pipeline is broken" would once again be the
# same empty table.

export OrderTransition, decide_order, apply_transition, reserve_plan, ReservePlan,
       EntryGates, gate_reasons

"""
    OrderTransition

What `decide_order` decided: the new state, why, and the account movement it authorises.

`delta === nothing` means the transition moves no money. That is the common case -- only
`PENDING_SUBMISSION`, the releases and settlement move any.
"""
struct OrderTransition
    order_id::UUID
    from::OrderState
    to::OrderState
    reason::String
    delta::Union{Nothing,LedgerDelta}
end

is_refusal(t::OrderTransition) = t.to in (CANCELLED, REJECTED)

"""
    EntryGates(; min_edge, max_spread, require_fillable, min_venue_stake)

The per-leg conditions checked between `TRIGGERED` and `PENDING_SUBMISSION`.

These are **not** a second staking layer. `Portfolio` has already decided the size; these decide
whether the size is *placeable*, which is a different question and is the one the allocator
cannot answer because it never sees depth.

* `min_edge` -- `p_model - p_market` floor, in probability points. `0.0` admits everything the
  allocator sized above zero.
* `max_spread` -- relative `(lay-back)/mid` at the leg's own runner. Defaults to 8%: measured,
  Scottish League Two BTTS sits at 6.0-6.8% and is the market whose fills are real and whose
  prices are not.
* `require_fillable` -- refuse a leg the archived ladder cannot fill in full. Defaults to
  `false`, because a partial fill on an authorised position is a normal outcome and refusing it
  outright loses the diversification the joint solve was buying.
* `min_venue_stake` -- the exchange minimum, applied to the VENUE stake and not to the risk. A
  lay at 1.26 clears £1 with £0.26 at risk.
"""
Base.@kwdef struct EntryGates
    min_edge::Float64        = 0.0
    max_spread::Float64      = 0.08
    require_fillable::Bool   = false
    min_venue_stake::Float64 = 1.0
end

"""
    gate_reasons(order, gates; spread = NaN, fillable = true) -> Vector{String}

Every reason this leg cannot be submitted, not the first one.

Collecting all of them matches `GateChain`: the second reason is usually the informative one, and
a leg that is both under-edge and unfillable is a different diagnosis from either alone.
"""
function gate_reasons(o::PaperOrder, g::EntryGates; spread::Float64 = NaN,
                      fillable::Bool = true)
    out = String[]
    o.risk > 0 || push!(out, "zero risk")
    o.edge < g.min_edge &&
        push!(out, "edge $(round(o.edge, digits = 4)) below $(g.min_edge)")
    o.venue_stake < g.min_venue_stake &&
        push!(out, "venue stake $(round(o.venue_stake, digits = 2)) below exchange minimum " *
                   "$(g.min_venue_stake)")
    (!isnan(spread) && spread > g.max_spread) &&
        push!(out, "spread $(round(100 * spread, digits = 2))% above " *
                   "$(round(100 * g.max_spread, digits = 2))%")
    (g.require_fillable && !fillable) &&
        push!(out, "book cannot fill $(round(o.venue_stake, digits = 2)) in full")
    return out
end

"""
    decide_order(order, gates; spread, fillable, fills, at) -> OrderTransition

The pure transition function. One call, one decision, no I/O.

`fills` is what the fill model produced (empty for a decision made before submission). The
routing:

| from | condition | to |
|---|---|---|
| `TRIGGERED` | any gate fails | `CANCELLED` (no money moved) |
| `TRIGGERED` | all gates pass | `PENDING_SUBMISSION` + `RESERVE` |
| `PENDING_SUBMISSION` | -- | `SUBMITTED` |
| `SUBMITTED` | no fills | `CANCELLED` + full `RELEASE` |
| `SUBMITTED` | filled in full | `MATCHED` |
| `SUBMITTED` | filled in part | `PARTIALLY_MATCHED` + partial `RELEASE` |

A partial release is the only subtle one: liability was reserved for `order.risk`, the position
actually taken is `filled_risk(fills)`, and the difference goes back to `balance`. Failing to
release it does not lose money but does understate free equity for the rest of the slate, which
would silently shrink the next reservation.
"""
function decide_order(o::PaperOrder, gates::EntryGates = EntryGates();
                      spread::Float64 = NaN, fillable::Bool = true,
                      fills::AbstractVector{Fill} = Fill[], at::DateTime = o.quote_ts)
    if o.state == TRIGGERED
        reasons = gate_reasons(o, gates; spread = spread, fillable = fillable)
        isempty(reasons) || return OrderTransition(o.order_id, o.state, CANCELLED,
                                                   join(reasons, "; "), nothing)
        return OrderTransition(o.order_id, o.state, PENDING_SUBMISSION, "gates passed",
                               LedgerDelta(kind = :RESERVE, account_id = o.account_id,
                                           delta_balance = -o.risk, delta_reserved = o.risk,
                                           order_id = o.order_id, slate_id = o.slate_id,
                                           note = "reserve leg"))
    elseif o.state == PENDING_SUBMISSION
        return OrderTransition(o.order_id, o.state, SUBMITTED, "submitted to venue", nothing)
    elseif o.state == SUBMITTED
        risk_filled = filled_risk(fills)
        if risk_filled <= 1e-9
            return OrderTransition(o.order_id, o.state, CANCELLED, "no fill before cut-off",
                                   LedgerDelta(kind = :RELEASE, account_id = o.account_id,
                                               delta_balance = o.risk, delta_reserved = -o.risk,
                                               order_id = o.order_id, slate_id = o.slate_id,
                                               note = "release unfilled"))
        elseif risk_filled >= o.risk - 1e-9
            return OrderTransition(o.order_id, o.state, MATCHED, "filled in full", nothing)
        else
            back = o.risk - risk_filled
            return OrderTransition(o.order_id, o.state, PARTIALLY_MATCHED,
                                   "filled $(round(risk_filled, digits = 2)) of " *
                                   "$(round(o.risk, digits = 2))",
                                   LedgerDelta(kind = :RELEASE, account_id = o.account_id,
                                               delta_balance = back, delta_reserved = -back,
                                               order_id = o.order_id, slate_id = o.slate_id,
                                               note = "release unfilled remainder"))
        end
    end
    return OrderTransition(o.order_id, o.state, o.state,
                           "terminal or unhandled state $(o.state)", nothing)
end

"""
    apply_transition(order, t) -> PaperOrder

The new order. Never mutates, and refuses a transition whose `from` does not match -- a
transition computed against a stale read is exactly the race the reservation lock exists to
prevent, and applying it silently would defeat that.
"""
function apply_transition(o::PaperOrder, t::OrderTransition)
    t.order_id == o.order_id || error(
        "apply_transition: transition is for order $(t.order_id) but the order is $(o.order_id).")
    t.from == o.state || error(
        "apply_transition: transition expects state $(t.from) but the order is in $(o.state). " *
        "This transition was computed against a stale read; recompute it.")
    return PaperOrder(order_id = o.order_id, slate_id = o.slate_id, account_id = o.account_id,
                      match_id = o.match_id, kickoff = o.kickoff,
                      bf_market_id = o.bf_market_id, market_group = o.market_group,
                      market_line = o.market_line, selection = o.selection,
                      venue_selection = o.venue_selection, side = o.side,
                      venue_odds = o.venue_odds, leverage = o.leverage,
                      effective_odds = o.effective_odds, p_model = o.p_model,
                      p_market = o.p_market, edge = o.edge,
                      stake_fraction = o.stake_fraction, risk = o.risk,
                      venue_stake = o.venue_stake, quote_ts = o.quote_ts,
                      state = t.to, reason = t.reason)
end

# ===================================================================
# The batch: one decision for the whole vector
# ===================================================================

"""
    ReservePlan

The answer to "may this slate be committed, and for how much?" -- computed before anything is
written, so the transaction that follows is a write and not a negotiation.

`admitted` and `refused` partition the legs. `total_risk` is the sum over `admitted` ONLY: the
refused legs never had liability, so reserving for them would overstate exposure and shrink the
next slate for no reason.
"""
struct ReservePlan
    slate_id::UUID
    account_id::String
    admitted::Vector{OrderTransition}
    refused::Vector{OrderTransition}
    total_risk::Float64
    exposure::Float64
    ok::Bool
    reason::String
end

"""
    reserve_plan(account, orders, gates; spreads, fillable) -> ReservePlan

Decide the whole slate at once, in one pure pass.

**This is where the slate's atomicity is enforced.** `Portfolio` solved one joint problem, so the
vector is committed whole or not at all: if admitted risk exceeds `account.max_slate_exposure`
times equity, the plan comes back `ok = false` and NOTHING is reserved. It deliberately does not
scale the vector down to fit -- `FixedCap` already had that job at pricing time with the full
book in hand, and a second, blinder rescale here would produce stakes no allocator authorised.

`spreads` and `fillable` are per-order lookups keyed by `order_id`; a missing entry is treated as
`NaN` / `true`, i.e. "not measured, do not refuse on it".
"""
function reserve_plan(account::PaperAccount, orders::AbstractVector{PaperOrder},
                      gates::EntryGates = EntryGates();
                      spreads::Dict{UUID,Float64} = Dict{UUID,Float64}(),
                      fillable::Dict{UUID,Bool}   = Dict{UUID,Bool}())
    isempty(orders) && return ReservePlan(UUID(0), account.account_id, OrderTransition[],
                                          OrderTransition[], 0.0, 0.0, false, "empty slate")
    slate_id = orders[1].slate_id
    all(o -> o.slate_id == slate_id, orders) || error(
        "reserve_plan: orders span more than one slate. The reservation is the slate's atom; " *
        "two slates in one plan would commit two joint solves under one exposure assert.")
    all(o -> o.account_id == account.account_id, orders) || error(
        "reserve_plan: orders span more than one account.")

    admitted = OrderTransition[]; refused = OrderTransition[]
    for o in orders
        t = decide_order(o, gates; spread = get(spreads, o.order_id, NaN),
                         fillable = get(fillable, o.order_id, true))
        push!(is_refusal(t) ? refused : admitted, t)
    end

    admitted_ids = Set(t.order_id for t in admitted)
    total = sum(Float64[o.risk for o in orders if o.order_id in admitted_ids]; init = 0.0)
    eq    = equity(account)
    exposure = eq > 0 ? total / eq : Inf
    limit    = account.max_slate_exposure * eq

    if total > account.balance + 1e-9
        return ReservePlan(slate_id, account.account_id, admitted, refused, total, exposure,
                           false,
                           "slate risk $(round(total, digits = 2)) exceeds free balance " *
                           "$(round(account.balance, digits = 2))")
    end
    if total > limit + 1e-9
        return ReservePlan(slate_id, account.account_id, admitted, refused, total, exposure,
                           false,
                           "slate risk $(round(total, digits = 2)) is " *
                           "$(round(100 * exposure, digits = 2))% of equity, above the " *
                           "$(round(100 * account.max_slate_exposure, digits = 2))% cap. The " *
                           "vector is committed whole or not at all -- re-price with a lower " *
                           "lambda rather than scaling these stakes.")
    end
    return ReservePlan(slate_id, account.account_id, admitted, refused, total, exposure,
                       true, "")
end
