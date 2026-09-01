# src/MatchDay/ledger/db.jl
#
# The impure shell. Everything here is a write or a read; every decision was already taken by
# `state_machine.jl`, which is why this file has no branches on prices, edges or exposure.
#
# One rule runs through it: the account is NEVER updated except through `post_ledger!`, which
# writes the `account_ledger` row and the `paper_accounts` row in the same statement pair. That
# is what makes `sum(delta_balance) == balance - opening_balance` a checkable assertion rather
# than a hope, and `reconcile_account` is the check.

export paper_connection, ensure_account!, account_row, slate_row, order_rows, fill_rows,
       ledger_rows, insert_slate!, insert_orders!, post_ledger!, update_order_state!,
       record_fills!, reconcile_account, orders_to_paper, slate_orders

"""
    paper_connection(; url = ENV["BF_DB_URL"]) -> LibPQ.Connection

Open a connection to the operational database. The caller closes it.

Deliberately not pooled and not cached: a match-day process holds one connection for the length
of one slate, and a stale pooled handle discovered at T-12 is a worse failure than the cost of
opening a socket.
"""
function paper_connection(; url::AbstractString = get(ENV, "BF_DB_URL", ""))
    isempty(url) && error(
        "paper_connection: BF_DB_URL is not set. Export it, e.g.\n" *
        "  export BF_DB_URL=\"postgresql://user:pass@host:5433/betdb\"")
    conn = Data.connect_to_db(Data.DBConfig(String(url)))
    # `timestamptz` comes back as a `ZonedDateTime` in the SESSION's zone, and every `DateTime`
    # this module writes is UTC-naive. Pinning the session to UTC is what makes a write and the
    # read that follows it the same instant -- without it a slate priced at 14:35Z reads back as
    # 15:35 in a BST session and `as_of` stops being the idempotency key it is declared to be.
    LibPQ.execute(conn, "SET TIME ZONE 'UTC';")
    return conn
end

_q(conn, sql, params = ()) = DataFrame(LibPQ.execute(conn, sql, params))

# ===================================================================
# Accounts
# ===================================================================

"""
    ensure_account!(conn, account; schema) -> PaperAccount

Create the account if it is absent; return what the database holds either way.

`ON CONFLICT DO NOTHING` rather than an upsert: an account that already exists has a balance,
and silently resetting it to the caller's opening figure would erase a track record. To change a
live account, post a `:DEPOSIT` or `:ADJUST` ledger entry.
"""
function ensure_account!(conn, a::PaperAccount; schema::AbstractString = PAPER_SCHEMA)
    LibPQ.execute(conn, """
        INSERT INTO $schema.paper_accounts
            (account_id, currency, opening_balance, balance, reserved,
             commission_rate, max_slate_exposure, is_live)
        VALUES (\$1, \$2, \$3, \$4, \$5, \$6, \$7, \$8)
        ON CONFLICT (account_id) DO NOTHING;""",
        (a.account_id, a.currency, a.opening_balance, a.balance, a.reserved,
         a.commission_rate, a.max_slate_exposure, a.is_live))
    return account_row(conn, a.account_id; schema = schema)
end

"""
    account_row(conn, account_id; schema) -> PaperAccount

Read the account. Errors when it is absent rather than returning a zeroed default -- a missing
bankroll must never look like an empty one.
"""
function account_row(conn, account_id::AbstractString; schema::AbstractString = PAPER_SCHEMA)
    df = _q(conn, "SELECT * FROM $schema.paper_accounts WHERE account_id = \$1;",
            (String(account_id),))
    nrow(df) == 1 || error("account_row: no account '$account_id' in $schema.paper_accounts. " *
                           "Call ensure_account! first.")
    r = first(df)
    return PaperAccount(account_id = String(r.account_id), currency = String(r.currency),
                        opening_balance = Float64(r.opening_balance),
                        balance = Float64(r.balance), reserved = Float64(r.reserved),
                        commission_rate = Float64(r.commission_rate),
                        max_slate_exposure = Float64(r.max_slate_exposure),
                        is_live = Bool(r.is_live))
end

# ===================================================================
# Slates and orders
# ===================================================================

"""
    insert_slate!(conn, s::PricedSlate; schema, run_name, model_run_id, ...) -> UUID

Write the batch header, or return the existing one for the same `(account, window, as_of)`.

`ON CONFLICT (account_id, slate_window, as_of) DO NOTHING` makes re-pricing the same instant a
no-op, which is what lets a crashed match-day process be restarted without double-staking.

`k_risk`, `slate_exposure`, `capped`, `risk_lambda` and `exposure_cap` are stored rather than
recomputed because **they are not recoverable after the fact**: `k` depends on the whole book, so
re-pricing the same legs with a different fixture list gives a different `k` for identical rows.
Without them, "why was this leg £26 and not £40?" has no answer.
"""
function insert_slate!(conn, s::PricedSlate; schema::AbstractString = PAPER_SCHEMA,
                       run_name::AbstractString = "", model_run_id = nothing,
                       book_spec_hash::AbstractString = "",
                       policy_spec_hash::AbstractString = "",
                       git_commit::AbstractString = "")
    blocked = JSON3.write([(match_id = c.fixture.m_id,
                            fixture = "$(c.fixture.home) v $(c.fixture.away)",
                            reasons = [string(k) * ": " * v for (k, v) in c.readiness.reasons])
                           for c in s.blocked if c.readiness isa Blocked])
    LibPQ.execute(conn, """
        INSERT INTO $schema.paper_slates
            (slate_id, account_id, slate_window, as_of, model_run_id, run_name, fold_idx,
             book_spec_hash, policy_spec_hash, bankroll, batch_status, k_risk, slate_exposure,
             exposure_cap, risk_lambda, capped, total_risk, n_fixtures, n_legs, n_blocked,
             blocked_report, git_commit)
        VALUES (\$1,\$2,\$3,\$4,\$5,\$6,\$7,\$8,\$9,\$10,\$11,\$12,\$13,\$14,\$15,\$16,\$17,
                \$18,\$19,\$20,\$21::jsonb,\$22)
        ON CONFLICT (account_id, slate_window, as_of) DO NOTHING;""",
        (string(s.slate_id), s.account_id, s.window, s.as_of,
         model_run_id === nothing ? missing : string(model_run_id), String(run_name),
         s.fold_idx, String(book_spec_hash), String(policy_spec_hash), s.bankroll,
         string(PRICED), _finite(s.k_risk, 1.0), _finite(s.slate_exposure, 0.0),
         _finite(s.exposure_cap, 0.25), _finite(s.risk_lambda, 0.0), s.capped, s.total_risk,
         n_fixtures(s), n_legs(s), length(s.blocked), blocked, String(git_commit)))

    df = _q(conn, """SELECT slate_id FROM $schema.paper_slates
                     WHERE account_id = \$1 AND slate_window = \$2 AND as_of = \$3;""",
            (s.account_id, s.window, s.as_of))
    return UUID(String(first(df).slate_id))
end

_finite(x::Real, fallback::Real) = isfinite(x) ? Float64(x) : Float64(fallback)

"""
    orders_to_paper(s::PricedSlate; slate_id) -> Vector{PaperOrder}

Turn a priced sheet into ledger rows, one per leg, in `TRIGGERED`.

The two identities are kept apart here and nowhere else: `selection` is the model's position and
`venue_selection` is the runner the order touches. On the 2026-08-08 ScottishLower slate 14 of 48
legs were synthetics, so collapsing them would mis-specify ~29% of tickets.
"""
function orders_to_paper(s::PricedSlate; slate_id::UUID = s.slate_id)
    sheet = s.sheet
    out = Vector{PaperOrder}(undef, nrow(sheet))
    for i in 1:nrow(sheet)
        mid  = sheet.match_id[i]
        card = findfirst(c -> c.fixture.m_id == mid, s.cards)
        ko   = card === nothing ? DateTime(sheet.slate[i]) : s.cards[card].fixture.kickoff
        key  = (group = sheet.group[i], line = sheet.line[i], selection = sheet.selection[i])
        inst = get(s.instruments, (mid, key), nothing)
        lev  = inst === nothing ? 1.0 : inst.leverage
        bk   = get(s.books, (mid, (group = sheet.group[i], line = sheet.line[i],
                                   selection = sheet.venue_selection[i])), nothing)
        out[i] = PaperOrder(order_id = uuid4(), slate_id = slate_id, account_id = s.account_id,
                            match_id = mid, kickoff = ko,
                            market_group = sheet.group[i], market_line = sheet.line[i],
                            selection = sheet.selection[i],
                            venue_selection = sheet.venue_selection[i],
                            side = sheet.side[i], venue_odds = sheet.venue_odds[i],
                            leverage = lev, effective_odds = sheet.odds[i],
                            p_model = sheet.p_model[i], p_market = sheet.p_market[i],
                            edge = sheet.edge[i], stake_fraction = sheet.frac[i],
                            risk = sheet.risk[i], venue_stake = sheet.venue_stake[i],
                            quote_ts = bk === nothing ? s.as_of : bk.ts)
    end
    return out
end

"Insert legs. `ON CONFLICT DO NOTHING` on `(slate_id, match_id, group, line, selection)`."
function insert_orders!(conn, orders::AbstractVector{PaperOrder};
                        schema::AbstractString = PAPER_SCHEMA)
    for o in orders
        LibPQ.execute(conn, """
            INSERT INTO $schema.paper_orders
                (order_id, slate_id, account_id, match_id, kickoff, bf_market_id, market_group,
                 market_line, selection, venue_selection, side, venue_odds, leverage,
                 effective_odds, p_model, p_market, edge, stake_fraction, risk, venue_stake,
                 quote_ts, state, reason)
            VALUES (\$1,\$2,\$3,\$4,\$5,\$6,\$7,\$8,\$9,\$10,\$11,\$12,\$13,\$14,\$15,\$16,
                    \$17,\$18,\$19,\$20,\$21,\$22,\$23)
            ON CONFLICT (slate_id, match_id, market_group, market_line, selection)
            DO NOTHING;""",
            (string(o.order_id), string(o.slate_id), o.account_id, o.match_id, o.kickoff,
             o.bf_market_id, o.market_group, o.market_line, String(o.selection),
             String(o.venue_selection), String(o.side), o.venue_odds, o.leverage,
             o.effective_odds, o.p_model, o.p_market, o.edge, o.stake_fraction, o.risk,
             o.venue_stake, o.quote_ts, string(o.state), o.reason))
    end
    return length(orders)
end

"""
    order_rows(conn, slate_id; schema) -> DataFrame

Every leg of one slate, as the database holds it. The console's read model and the recovery
path's input.
"""
order_rows(conn, slate_id::UUID; schema::AbstractString = PAPER_SCHEMA) =
    _q(conn, "SELECT * FROM $schema.paper_orders WHERE slate_id = \$1 ORDER BY match_id, " *
             "market_group, market_line, selection;", (string(slate_id),))

slate_row(conn, slate_id::UUID; schema::AbstractString = PAPER_SCHEMA) =
    _q(conn, "SELECT * FROM $schema.paper_slates WHERE slate_id = \$1;", (string(slate_id),))

fill_rows(conn, slate_id::UUID; schema::AbstractString = PAPER_SCHEMA) =
    _q(conn, """SELECT f.* FROM $schema.paper_fills f
                JOIN $schema.paper_orders o USING (order_id)
                WHERE o.slate_id = \$1 ORDER BY f.fill_id;""", (string(slate_id),))

ledger_rows(conn, account_id::AbstractString; schema::AbstractString = PAPER_SCHEMA) =
    _q(conn, "SELECT * FROM $schema.account_ledger WHERE account_id = \$1 ORDER BY entry_id;",
       (String(account_id),))

"""
    slate_orders(conn, slate_id; schema) -> Vector{PaperOrder}

`order_rows` lifted back into the domain type, so the pure state machine can be run against what
is actually stored rather than against what the caller remembers writing.
"""
function slate_orders(conn, slate_id::UUID; schema::AbstractString = PAPER_SCHEMA)
    df = order_rows(conn, slate_id; schema = schema)
    return PaperOrder[
        PaperOrder(order_id = UUID(String(r.order_id)), slate_id = UUID(String(r.slate_id)),
                   account_id = String(r.account_id), match_id = Int(r.match_id),
                   kickoff = DateTime(r.kickoff), bf_market_id = String(r.bf_market_id),
                   market_group = String(r.market_group), market_line = Float64(r.market_line),
                   selection = Symbol(r.selection), venue_selection = Symbol(r.venue_selection),
                   side = Symbol(r.side), venue_odds = Float64(r.venue_odds),
                   leverage = Float64(r.leverage), effective_odds = Float64(r.effective_odds),
                   p_model = Float64(r.p_model), p_market = Float64(r.p_market),
                   edge = Float64(r.edge), stake_fraction = Float64(r.stake_fraction),
                   risk = Float64(r.risk), venue_stake = Float64(r.venue_stake),
                   quote_ts = DateTime(r.quote_ts), state = _parse_state(String(r.state)),
                   reason = String(r.reason))
        for r in eachrow(df)]
end

function _parse_state(s::AbstractString)
    for st in ORDER_STATES
        string(st) == s && return st
    end
    error("_parse_state: '$s' is not an OrderState. The CHECK constraint should have made " *
          "this unreachable; the schema and the enum have drifted.")
end

# ===================================================================
# The one writer of the account
# ===================================================================

"""
    post_ledger!(conn, delta; schema) -> PaperAccount

Move the account, and write down why, in one transaction.

**The only function in the system that changes `paper_accounts`.** Everything else produces a
`LedgerDelta` and hands it here. The `UPDATE ... RETURNING` gives the post-state, which is stored
on the ledger row, so a reconciliation never has to reconstruct it by replaying arithmetic.

The `RESERVE` unique index means a second reservation for the same slate raises rather than
silently doubling the exposure. That is a deliberate failure: a retry that quietly re-reserved
would put the account past its cap with no row saying so.
"""
function post_ledger!(conn, d::LedgerDelta; schema::AbstractString = PAPER_SCHEMA)
    LibPQ.execute(conn, "BEGIN;")
    try
        upd = _q(conn, """
            UPDATE $schema.paper_accounts
               SET balance = balance + \$2, reserved = reserved + \$3, updated_at = now()
             WHERE account_id = \$1
             RETURNING balance, reserved;""",
            (d.account_id, d.delta_balance, d.delta_reserved))
        nrow(upd) == 1 || error("post_ledger!: no account '$(d.account_id)'.")
        LibPQ.execute(conn, """
            INSERT INTO $schema.account_ledger
                (account_id, kind, order_id, slate_id, delta_balance, delta_reserved,
                 balance_after, reserved_after, note)
            VALUES (\$1,\$2,\$3,\$4,\$5,\$6,\$7,\$8,\$9);""",
            (d.account_id, String(d.kind),
             d.order_id === nothing ? missing : string(d.order_id),
             d.slate_id === nothing ? missing : string(d.slate_id),
             d.delta_balance, d.delta_reserved,
             Float64(first(upd).balance), Float64(first(upd).reserved), d.note))
        LibPQ.execute(conn, "COMMIT;")
    catch e
        LibPQ.execute(conn, "ROLLBACK;")
        rethrow(e)
    end
    return account_row(conn, d.account_id; schema = schema)
end

"Move one order to a new state, stamping the timestamps the lifecycle needs."
function update_order_state!(conn, order_id::UUID, to::OrderState, reason::AbstractString;
                             schema::AbstractString = PAPER_SCHEMA, at::DateTime = now())
    LibPQ.execute(conn, """
        UPDATE $schema.paper_orders
           SET state = \$2, reason = \$3,
               submitted_at = CASE WHEN \$2 = 'SUBMITTED' THEN \$4 ELSE submitted_at END,
               terminal_at  = CASE WHEN \$2 IN ('CANCELLED','REJECTED','VOIDED','SETTLED')
                                   THEN \$4 ELSE terminal_at END
         WHERE order_id = \$1;""",
        (string(order_id), string(to), String(reason), at))
    return to
end

"Append fills. Never an UPDATE -- a partial fill is N rows, so the history is the record."
function record_fills!(conn, fills::AbstractVector{Fill};
                       schema::AbstractString = PAPER_SCHEMA)
    for f in fills
        LibPQ.execute(conn, """
            INSERT INTO $schema.paper_fills
                (order_id, filled_at, price, size, risk_filled, fill_model, level_depth)
            VALUES (\$1,\$2,\$3,\$4,\$5,\$6,\$7);""",
            (string(f.order_id), f.filled_at, f.price, f.size, f.risk_filled,
             String(f.model), f.levels_used))
    end
    return length(fills)
end

"""
    reconcile_account(conn, account_id; schema) -> NamedTuple

The nightly assertion: the ledger must explain the balance exactly.

    Σ delta_balance  == balance  - opening_balance
    Σ delta_reserved == reserved

A mismatch is a defect, not a rounding error -- every movement goes through `post_ledger!`, so
either a write bypassed it or a transaction committed half.

The tolerance scales with the number of entries. Money is `numeric(14,2)`, so each delta is
rounded to the penny on the way in while the running balance is rounded independently; over `n`
entries the two can drift by up to `n/2` pence. A FIXED tolerance would therefore start failing
on a long-lived account for reasons that are arithmetic rather than defects -- and, worse, would
have to be widened later, at which point it would no longer catch anything.
"""
function reconcile_account(conn, account_id::AbstractString;
                           schema::AbstractString = PAPER_SCHEMA, tol::Union{Nothing,Float64} = nothing)
    a  = account_row(conn, account_id; schema = schema)
    df = _q(conn, """SELECT COALESCE(sum(delta_balance),0) AS db,
                            COALESCE(sum(delta_reserved),0) AS dr,
                            count(*) AS n
                     FROM $schema.account_ledger WHERE account_id = \$1;""", (String(account_id),))
    db, dr = Float64(first(df).db), Float64(first(df).dr)
    n_entries = Int(first(df).n)
    tol = tol === nothing ? max(0.01, 0.005 * n_entries) : tol
    ok_balance  = abs(db - (a.balance - a.opening_balance)) <= tol
    ok_reserved = abs(dr - a.reserved) <= tol
    return (; ok = ok_balance && ok_reserved, ok_balance, ok_reserved, n_entries, tol,
            ledger_balance_delta = db, account_balance_delta = a.balance - a.opening_balance,
            ledger_reserved = dr, account_reserved = a.reserved)
end
