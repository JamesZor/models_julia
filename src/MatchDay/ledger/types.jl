# src/MatchDay/ledger/types.jl
#
# The paper ledger's domain objects and its two state machines.
#
# There are TWO, nested, and conflating them is the mistake this file exists to prevent:
#
#   * `BatchState`  -- the SLATE's lifecycle. The atom. `Portfolio` solved one joint problem, so
#                      the vector is committed or it is not; there is no half-committed slate.
#   * `OrderState`  -- one leg's lifecycle INSIDE a committed batch. Legs fail individually all
#                      the time; that is a fill shortfall on an authorised position, which is a
#                      different thing from an unauthorised position.
#
# The reservation transition (`RESERVED`) is the only place money moves for the whole slate, and
# it is the only atomic step. Submission afterwards is not atomic and cannot be -- the venue takes
# one market at a time -- but by then the allocator's answer is already committed in full.

export OrderState, BatchState, PaperAccount, PaperOrder, LedgerDelta, Fill,
       ORDER_STATES, BATCH_STATES, is_terminal, has_exposure

# ===================================================================
# 1. States
# ===================================================================

"""
    OrderState

One leg's lifecycle.

    TRIGGERED ──► PENDING_SUBMISSION ──► SUBMITTED ──┬──► MATCHED ──► SETTLED
        │                 │                          ├──► PARTIALLY_MATCHED ──► SETTLED
        └──► CANCELLED    └──► CANCELLED             ├──► REJECTED
                                                     └──► VOIDED

`TRIGGERED` is written at pricing time and moves **no money**. `PENDING_SUBMISSION` is reached
only through the slate reservation, so an order in that state has liability held against it.
"""
@enum OrderState begin
    TRIGGERED
    PENDING_SUBMISSION
    SUBMITTED
    PARTIALLY_MATCHED
    MATCHED
    CANCELLED
    REJECTED
    VOIDED
    SETTLED
end

"""
    BatchState

The slate's lifecycle. `RESERVED` is the atom -- see `execute_slate_batch!`.

    PRICED ──► REVIEWED ──► RESERVED ──► SUBMITTING ──► EXECUTED ──► SETTLED
       │           │            │             │              │
       └───────────┴──► ABANDONED            └──────────────┴──► KILLED

`ABANDONED` is pre-reservation (a gate failed, no money moved); `KILLED` is post-reservation
(an operator aborted, everything released). They are distinct states because the audit question
"did we ever hold this risk?" has different answers.
"""
@enum BatchState begin
    PRICED
    REVIEWED
    RESERVED
    SUBMITTING
    EXECUTED
    BATCH_SETTLED
    KILLED
    ABANDONED
end

const ORDER_STATES = instances(OrderState)
const BATCH_STATES = instances(BatchState)

"Is this a state nothing can move out of?"
is_terminal(s::OrderState) = s in (CANCELLED, REJECTED, VOIDED, SETTLED)
is_terminal(s::BatchState) = s in (BATCH_SETTLED, KILLED, ABANDONED)

"""
    has_exposure(s::OrderState) -> Bool

Does an order in this state have account liability reserved against it?

`TRIGGERED` does **not**: pricing writes rows and moves no money, which is what lets a slate be
priced, reviewed and abandoned with the bankroll untouched.
"""
has_exposure(s::OrderState) =
    s in (PENDING_SUBMISSION, SUBMITTED, PARTIALLY_MATCHED, MATCHED)

# ===================================================================
# 2. Domain objects
# ===================================================================

"""
    PaperAccount

The bankroll of record.

The invariant that makes the whole ledger auditable:

    balance   = cash NOT committed to open risk
    reserved  = cash committed to open risk
    equity    = balance + reserved + unrealised mark-to-market

`balance` and `reserved` move only in equal and opposite pairs, except at settlement where
`reserved` falls by the filled risk and `balance` rises by the gross return net of commission.
Every move is an `account_ledger` row in the same transaction, so
`sum(delta_balance) == balance - opening_balance` is a checkable assertion rather than a hope.

`is_live` defaults to `false` and there is no constructor that defaults it to `true`.
"""
Base.@kwdef struct PaperAccount
    account_id::String
    currency::String            = "GBP"
    opening_balance::Float64
    balance::Float64
    reserved::Float64           = 0.0
    commission_rate::Float64    = 0.02
    max_slate_exposure::Float64 = 0.25
    is_live::Bool               = false
end

"Cash plus committed risk. The number a slate is sized against."
equity(a::PaperAccount) = a.balance + a.reserved

"""
    PaperOrder

One leg, in the ledger's own denomination.

Carries BOTH identities, and they are not the same on a synthetic:

* `market_group` / `market_line` / `selection` -- the MODEL's position. What we grade against.
* `venue_selection` / `side` / `venue_odds` -- the runner the order actually touches. Backing
  Over 2.5 by laying Under 2.5 names `over_25` in the first and `under_25` in the second.

`risk` is liability and is what `reserved` holds; `venue_stake = risk * leverage` is what is
placed. For a back the two coincide; for a lay at 1.26 the venue stake is 3.85x the risk.
"""
Base.@kwdef struct PaperOrder
    order_id::UUID
    slate_id::UUID
    account_id::String
    match_id::Int
    kickoff::DateTime
    bf_market_id::String        = ""
    market_group::String
    market_line::Float64
    selection::Symbol
    venue_selection::Symbol
    side::Symbol
    venue_odds::Float64
    leverage::Float64
    effective_odds::Float64
    p_model::Float64
    p_market::Float64
    edge::Float64
    stake_fraction::Float64
    risk::Float64
    venue_stake::Float64
    quote_ts::DateTime
    state::OrderState           = TRIGGERED
    reason::String              = ""
end

"""
    Fill

One execution against one order. Append-only: a partial fill is two `Fill`s, never an edit.

`risk_filled = size / leverage` is the liability actually taken, and it is the quantity the
account releases against -- not `size`, which for a lay is the backer stake and is larger.
"""
Base.@kwdef struct Fill
    order_id::UUID
    filled_at::DateTime
    price::Float64
    size::Float64
    risk_filled::Float64
    model::Symbol
    levels_used::Int = 0
end

"""
    LedgerDelta

One movement of the account, as a value.

Produced by the pure transition functions and applied by the persistence layer, so the arithmetic
is testable without a database and the database cannot invent a movement the state machine did
not authorise. `kind` is one of `:RESERVE`, `:RELEASE`, `:SETTLE`, `:COMMISSION`, `:DEPOSIT`,
`:ADJUST`.
"""
Base.@kwdef struct LedgerDelta
    kind::Symbol
    account_id::String
    delta_balance::Float64
    delta_reserved::Float64
    order_id::Union{Nothing,UUID} = nothing
    slate_id::Union{Nothing,UUID} = nothing
    note::String = ""
end

"Apply a delta to an account, returning the new account. Never mutates."
function apply_delta(a::PaperAccount, d::LedgerDelta)
    d.account_id == a.account_id || error(
        "apply_delta: delta is for account '$(d.account_id)' but the account is " *
        "'$(a.account_id)'. A ledger entry crossing accounts is a bug, not a transfer.")
    bal = a.balance  + d.delta_balance
    res = a.reserved + d.delta_reserved
    bal >= -1e-9 || error(
        "apply_delta: $(d.kind) would take balance to $(round(bal, digits = 4)). The exposure " *
        "cap exists to make this unreachable; reaching it means the reservation assert was " *
        "skipped.")
    res >= -1e-9 || error(
        "apply_delta: $(d.kind) would take reserved to $(round(res, digits = 4)), i.e. release " *
        "more liability than was ever held. Check for a double RELEASE on one order.")
    return PaperAccount(account_id = a.account_id, currency = a.currency,
                        opening_balance = a.opening_balance,
                        balance = max(bal, 0.0), reserved = max(res, 0.0),
                        commission_rate = a.commission_rate,
                        max_slate_exposure = a.max_slate_exposure, is_live = a.is_live)
end
