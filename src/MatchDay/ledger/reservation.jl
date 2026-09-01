# src/MatchDay/ledger/reservation.jl
#
# THE ATOM.
#
# `Portfolio.stake_slate` solved one convex problem for the whole settlement window: `SlateDrawdown`
# returned a single `k` applied to every leg, and `FixedCap` rescaled the whole vector. So the
# stake vector is only valid AS A VECTOR -- take 15 of 21 legs and the drawdown budget the other 6
# were funding is unspent.
#
# `execute_slate_batch!` is therefore one transaction that either commits the entire vector or
# commits nothing:
#
#     BEGIN
#       SELECT ... FROM paper_accounts WHERE account_id = $1 FOR UPDATE   -- serialise
#       assert  Σ risk <= balance  AND  Σ risk <= max_slate_exposure * equity
#       UPDATE  paper_orders  SET state = 'PENDING_SUBMISSION'  (admitted)
#       UPDATE  paper_orders  SET state = 'CANCELLED'           (refused, with reasons)
#       UPDATE  paper_accounts SET balance -= Σrisk, reserved += Σrisk
#       INSERT  account_ledger (kind = 'RESERVE')                -- unique per slate
#       UPDATE  paper_slates  SET batch_status = 'RESERVED'
#     COMMIT
#
# After it, submission is embarrassingly parallel: the submitters touch only rows they own and no
# account row at all, so there is nothing left to contend over. A submission failure is then a
# FILL SHORTFALL ON AN AUTHORISED POSITION rather than an unauthorised position, and that
# distinction is the entire safety argument.
#
# Measured scale, 2026-09-05: 36 fixtures kick off simultaneously at 14:00 UTC across six
# tournaments; Scottish alone peaks at 21 at 15:00 on eleven Saturdays this season. That is
# 80-120 orders inside a ten-minute window against one bankroll.

export execute_slate_batch!, BatchResult, kill_slate!, submit_slate!, recover_open_orders,
       set_batch_status!

"""
    BatchResult

What one reservation attempt did. A refusal is a value, with the reason attached.
"""
struct BatchResult
    slate_id::UUID
    account_id::String
    status::BatchState
    reserved::Float64
    n_admitted::Int
    n_refused::Int
    account::PaperAccount
    reason::String
end

"Set the batch's lifecycle state, stamping the timestamp that state implies."
function set_batch_status!(conn, slate_id::UUID, st::BatchState;
                           schema::AbstractString = PAPER_SCHEMA, at::DateTime = now())
    col = st == REVIEWED ? "reviewed_at" : st == RESERVED ? "reserved_at" :
          is_terminal(st) ? "terminal_at" : nothing
    sql = col === nothing ?
        "UPDATE $schema.paper_slates SET batch_status = \$2 WHERE slate_id = \$1;" :
        "UPDATE $schema.paper_slates SET batch_status = \$2, $col = \$3 WHERE slate_id = \$1;"
    LibPQ.execute(conn, sql, col === nothing ? (string(slate_id), string(st)) :
                             (string(slate_id), string(st), at))
    return st
end

"""
    execute_slate_batch!(conn, account_id, slate_id, gates; schema, spreads, fillable) -> BatchResult

Commit the whole vector, or none of it.

The lock is `SELECT ... FOR UPDATE` on the single `paper_accounts` row. That serialises every
writer of this bankroll for the duration of one slate, which is the correct granularity and the
only correct one: per-fixture locking would let two concurrent slates each pass an exposure
assert that neither passes jointly.

**It does not scale the vector down to fit.** If admitted risk exceeds the cap, the batch is
`ABANDONED` and nothing is reserved. `FixedCap` already had the rescaling job at pricing time,
with the whole book in hand; a second, blinder rescale here would produce stakes no allocator
authorised and a `k_risk` on the slate row that no longer describes them.

Idempotent by construction: the partial unique index on `account_ledger (slate_id) WHERE
kind = 'RESERVE'` means a retry raises rather than double-reserving, and the guard below turns
an already-reserved slate into a no-op result rather than an exception.
"""
function execute_slate_batch!(conn, account_id::AbstractString, slate_id::UUID,
                              gates::EntryGates = EntryGates();
                              schema::AbstractString = PAPER_SCHEMA,
                              spreads::Dict{UUID,Float64} = Dict{UUID,Float64}(),
                              fillable::Dict{UUID,Bool}   = Dict{UUID,Bool}(),
                              at::DateTime = now())
    existing = _q(conn, "SELECT batch_status FROM $schema.paper_slates WHERE slate_id = \$1;",
                  (string(slate_id),))
    nrow(existing) == 1 || error(
        "execute_slate_batch!: no slate $slate_id in $schema.paper_slates. Call insert_slate! " *
        "first -- the batch header is what the reservation is idempotent against.")
    st = String(first(existing).batch_status)
    if st != string(PRICED) && st != string(REVIEWED)
        acct = account_row(conn, account_id; schema = schema)
        return BatchResult(slate_id, String(account_id), _parse_batch(st), 0.0, 0, 0, acct,
                           "slate is already $st; reservation is a no-op")
    end

    LibPQ.execute(conn, "BEGIN;")
    try
        # THE LOCK. One row, held for the whole slate.
        locked = _q(conn, """SELECT account_id, currency, opening_balance, balance, reserved,
                                    commission_rate, max_slate_exposure, is_live
                             FROM $schema.paper_accounts
                             WHERE account_id = \$1 FOR UPDATE;""", (String(account_id),))
        nrow(locked) == 1 || error("execute_slate_batch!: no account '$account_id'.")
        r = first(locked)
        acct = PaperAccount(account_id = String(r.account_id), currency = String(r.currency),
                            opening_balance = Float64(r.opening_balance),
                            balance = Float64(r.balance), reserved = Float64(r.reserved),
                            commission_rate = Float64(r.commission_rate),
                            max_slate_exposure = Float64(r.max_slate_exposure),
                            is_live = Bool(r.is_live))

        orders = slate_orders(conn, slate_id; schema = schema)
        plan   = reserve_plan(acct, orders, gates; spreads = spreads, fillable = fillable)

        if !plan.ok
            for t in vcat(plan.admitted, plan.refused)
                update_order_state!(conn, t.order_id, CANCELLED,
                                    is_refusal(t) ? t.reason : "slate abandoned: " * plan.reason;
                                    schema = schema, at = at)
            end
            set_batch_status!(conn, slate_id, ABANDONED; schema = schema, at = at)
            LibPQ.execute(conn, "COMMIT;")
            return BatchResult(slate_id, String(account_id), ABANDONED, 0.0, 0,
                               length(plan.refused) + length(plan.admitted), acct, plan.reason)
        end

        for t in plan.refused
            update_order_state!(conn, t.order_id, CANCELLED, t.reason; schema = schema, at = at)
        end
        for t in plan.admitted
            update_order_state!(conn, t.order_id, PENDING_SUBMISSION, t.reason;
                                schema = schema, at = at)
        end

        # ONE account movement for the whole vector, and one ledger row. The partial unique
        # index makes a second one for this slate impossible rather than merely unlikely.
        upd = _q(conn, """
            UPDATE $schema.paper_accounts
               SET balance = balance - \$2, reserved = reserved + \$2, updated_at = now()
             WHERE account_id = \$1
             RETURNING balance, reserved;""", (String(account_id), plan.total_risk))
        LibPQ.execute(conn, """
            INSERT INTO $schema.account_ledger
                (account_id, at, kind, slate_id, delta_balance, delta_reserved,
                 balance_after, reserved_after, note)
            VALUES (\$1,\$2,'RESERVE',\$3,\$4,\$5,\$6,\$7,\$8);""",
            (String(account_id), at, string(slate_id), -plan.total_risk, plan.total_risk,
             Float64(first(upd).balance), Float64(first(upd).reserved),
             "reserve slate: $(length(plan.admitted)) legs"))

        set_batch_status!(conn, slate_id, RESERVED; schema = schema, at = at)
        LibPQ.execute(conn, "COMMIT;")
    catch e
        LibPQ.execute(conn, "ROLLBACK;")
        rethrow(e)
    end

    acct   = account_row(conn, account_id; schema = schema)
    after  = slate_orders(conn, slate_id; schema = schema)
    n_adm  = count(o -> o.state == PENDING_SUBMISSION, after)
    n_ref  = count(o -> o.state == CANCELLED, after)
    res    = sum(Float64[o.risk for o in after if o.state == PENDING_SUBMISSION]; init = 0.0)
    return BatchResult(slate_id, String(account_id), RESERVED, res, n_adm, n_ref, acct, "")
end

function _parse_batch(s::AbstractString)
    for st in BATCH_STATES
        string(st) == s && return st
    end
    error("_parse_batch: '$s' is not a BatchState.")
end

"""
    submit_slate!(conn, slate_id, books, model; schema, at) -> NamedTuple

Move every reserved leg through `SUBMITTED` and apply the fill model against `books`.

Sharded by market in production; here it is a loop, because the ordering is irrelevant once the
reservation has committed -- that is the property the atom buys. Each leg's release of its
unfilled remainder is its own `post_ledger!`, so a crash mid-loop leaves a consistent account
with some legs still `SUBMITTED`, which is exactly what `recover_open_orders` looks for.

`books` is keyed `(match_id, SelectionKey)` on the **venue** runner -- the same key
`PricedSlate.books` uses.
"""
function submit_slate!(conn, slate_id::UUID,
                       books::Dict{Tuple{Int,SelectionKey},BookLevels},
                       model::AbstractFillModel = TouchOnly();
                       schema::AbstractString = PAPER_SCHEMA, at::DateTime = now())
    set_batch_status!(conn, slate_id, SUBMITTING; schema = schema, at = at)
    orders = slate_orders(conn, slate_id; schema = schema)
    n_matched = 0; n_partial = 0; n_unfilled = 0; total_filled = 0.0

    for o in orders
        o.state == PENDING_SUBMISSION || continue
        t1 = decide_order(o; at = at)                       # -> SUBMITTED, no money
        o1 = apply_transition(o, t1)
        update_order_state!(conn, o1.order_id, o1.state, o1.reason; schema = schema, at = at)

        vkey  = (group = o.market_group, line = o.market_line, selection = o.venue_selection)
        lv    = get(books, (o.match_id, vkey), nothing)
        fills = lv === nothing ? Fill[] :
                attach_order(simulate_fill(model, lv, o.side, o.venue_stake, o.leverage, at),
                             o.order_id)
        isempty(fills) || record_fills!(conn, fills; schema = schema)

        t2 = decide_order(o1; fills = fills, at = at)
        o2 = apply_transition(o1, t2)
        update_order_state!(conn, o2.order_id, o2.state, o2.reason; schema = schema, at = at)
        t2.delta === nothing || post_ledger!(conn, t2.delta; schema = schema)

        total_filled += filled_risk(fills)
        o2.state == MATCHED           && (n_matched  += 1)
        o2.state == PARTIALLY_MATCHED && (n_partial  += 1)
        o2.state == CANCELLED         && (n_unfilled += 1)
    end

    set_batch_status!(conn, slate_id, EXECUTED; schema = schema, at = at)
    return (; n_matched, n_partial, n_unfilled, risk_filled = total_filled,
            account = account_row(conn, first(orders).account_id; schema = schema))
end

"""
    kill_slate!(conn, slate_id; schema, at) -> BatchResult

Operator abort. Cancels every non-terminal leg and releases its liability.

Only legs that HAVE exposure are released, and `MATCHED` legs are left alone: a filled position
cannot be un-filled by an operator pressing a key, and pretending otherwise would release
liability the account is still carrying.
"""
function kill_slate!(conn, slate_id::UUID; schema::AbstractString = PAPER_SCHEMA,
                     at::DateTime = now())
    orders  = slate_orders(conn, slate_id; schema = schema)
    isempty(orders) && error("kill_slate!: no orders for slate $slate_id.")
    account = first(orders).account_id
    released = 0.0; n = 0
    for o in orders
        (is_terminal(o.state) || o.state == MATCHED || o.state == PARTIALLY_MATCHED) && continue
        if has_exposure(o.state)
            post_ledger!(conn, LedgerDelta(kind = :RELEASE, account_id = o.account_id,
                                           delta_balance = o.risk, delta_reserved = -o.risk,
                                           order_id = o.order_id, slate_id = slate_id,
                                           note = "operator kill"); schema = schema)
            released += o.risk
        end
        update_order_state!(conn, o.order_id, CANCELLED, "operator killed the slate";
                            schema = schema, at = at)
        n += 1
    end
    set_batch_status!(conn, slate_id, KILLED; schema = schema, at = at)
    acct = account_row(conn, account; schema = schema)
    return BatchResult(slate_id, account, KILLED, -released, 0, n, acct, "operator kill")
end

"""
    recover_open_orders(conn, account_id; schema, now) -> DataFrame

What a restarted process must decide about.

Anything in `PENDING_SUBMISSION` was reserved and never submitted: submit it if still inside the
window, cancel-and-release if not. Anything `SUBMITTED` needs reconciliation against the venue
(in paper mode, the fill model at its recorded `submitted_at`). The reservation is already
durable, so nothing is lost or double-counted across a crash -- which is the entire reason
`RESERVE` precedes submission rather than following it.
"""
recover_open_orders(conn, account_id::AbstractString;
                    schema::AbstractString = PAPER_SCHEMA, at::DateTime = now()) =
    _q(conn, """SELECT * FROM $schema.paper_orders
                WHERE account_id = \$1
                  AND state IN ('PENDING_SUBMISSION','SUBMITTED','PARTIALLY_MATCHED')
                  AND kickoff > \$2
                ORDER BY kickoff;""", (String(account_id), at))
