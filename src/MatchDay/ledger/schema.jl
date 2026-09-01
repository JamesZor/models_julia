# src/MatchDay/ledger/schema.jl
#
# The paper ledger's DDL, as a function of the schema name so a test can build the whole thing in
# `paper_test` and drop it again without touching production.
#
# It lives in `betdb` rather than `mcmc_experiments` for three reasons:
#
#   1. AVAILABILITY. The ledger is written at T-12 on a Saturday. `mcmc-beast` may be saturated
#      by a training run; `betdb` is the operational database and is already the one the
#      collector, the supervisor and the TUI all require to be up.
#   2. LOCALITY. Mark-to-market and CLV both read `betfair_live.order_book_1m`. Putting the
#      ledger beside it makes settlement a join instead of a cross-database transfer.
#   3. SEPARATION. `mcmc_experiments.portfolio_bets` is the BACKTEST ledger: one row per
#      simulated bet, no lifecycle, no time, no fills. Paper trading has a lifecycle. Overloading
#      one table with both would make "what did the backtest say" and "what did we actually do"
#      the same query, which is the one distinction the exercise exists to preserve.
#
# The link across is `paper_slates.model_run_id -> mcmc_experiments.runs.run_id`, carried as an
# opaque UUID with no foreign key. A reconciliation job asserts it resolves.

export PAPER_SCHEMA, paper_ddl, migrate_paper_schema!, drop_paper_schema!

"Default schema name. Overridden to `paper_test` by the suite."
const PAPER_SCHEMA = "paper"

"""
    paper_ddl(schema = PAPER_SCHEMA) -> Vector{String}

Every statement needed to build the ledger, in dependency order, each one idempotent.

Three idempotency mechanisms make the whole system re-runnable after a crash, and they are the
reason a re-priced slate cannot double-stake:

* `paper_slates (account_id, slate_window, as_of)` is UNIQUE -- re-pricing the same instant
  returns the existing slate rather than making a second one.
* `paper_orders (slate_id, match_id, market_group, market_line, selection)` is UNIQUE --
  re-inserting a leg is a no-op.
* `account_ledger (slate_id) WHERE kind = 'RESERVE'` is UNIQUE -- **double-reserving a slate is
  unrepresentable**, not merely guarded against. This is the single most important constraint in
  the file: it is what makes the reservation an atom even against a retry storm.
"""
function paper_ddl(schema::AbstractString = PAPER_SCHEMA)
    s = string(schema)
    return [
    "CREATE SCHEMA IF NOT EXISTS $s;",

    """
    CREATE TABLE IF NOT EXISTS $s.paper_accounts (
        account_id         text PRIMARY KEY,
        currency           char(3)       NOT NULL DEFAULT 'GBP',
        opening_balance    numeric(14,2) NOT NULL,
        balance            numeric(14,2) NOT NULL,
        reserved           numeric(14,2) NOT NULL DEFAULT 0,
        commission_rate    numeric(6,4)  NOT NULL DEFAULT 0.02,
        max_slate_exposure numeric(6,4)  NOT NULL DEFAULT 0.25,
        is_live            boolean       NOT NULL DEFAULT false,
        created_at         timestamptz   NOT NULL DEFAULT now(),
        updated_at         timestamptz   NOT NULL DEFAULT now(),
        CONSTRAINT balance_nonneg  CHECK (balance  >= 0),
        CONSTRAINT reserved_nonneg CHECK (reserved >= 0)
    );""",

    """
    CREATE TABLE IF NOT EXISTS $s.paper_slates (
        slate_id         uuid PRIMARY KEY,
        account_id       text          NOT NULL REFERENCES $s.paper_accounts,
        slate_window     date          NOT NULL,
        as_of            timestamptz   NOT NULL,
        model_run_id     uuid,
        run_name         text          NOT NULL DEFAULT '',
        fold_idx         int           NOT NULL DEFAULT 0,
        book_spec_hash   text          NOT NULL DEFAULT '',
        policy_spec_hash text          NOT NULL DEFAULT '',
        bankroll         numeric(14,2) NOT NULL,
        batch_status     text          NOT NULL DEFAULT 'PRICED',
        k_risk           numeric(12,8) NOT NULL DEFAULT 1,
        slate_exposure   numeric(9,6)  NOT NULL DEFAULT 0,
        exposure_cap     numeric(6,4)  NOT NULL DEFAULT 0.25,
        risk_lambda      numeric(8,3)  NOT NULL DEFAULT 0,
        capped           boolean       NOT NULL DEFAULT false,
        total_risk       numeric(14,2) NOT NULL DEFAULT 0,
        n_fixtures       int           NOT NULL DEFAULT 0,
        n_legs           int           NOT NULL DEFAULT 0,
        n_blocked        int           NOT NULL DEFAULT 0,
        blocked_report   jsonb         NOT NULL DEFAULT '[]'::jsonb,
        git_commit       text          NOT NULL DEFAULT '',
        reviewed_at      timestamptz,
        reserved_at      timestamptz,
        terminal_at      timestamptz,
        created_at       timestamptz   NOT NULL DEFAULT now(),
        CONSTRAINT batch_status_known CHECK (batch_status IN
            ('PRICED','REVIEWED','RESERVED','SUBMITTING','EXECUTED',
             'BATCH_SETTLED','KILLED','ABANDONED')),
        CONSTRAINT slate_once UNIQUE (account_id, slate_window, as_of)
    );""",

    """
    CREATE TABLE IF NOT EXISTS $s.paper_orders (
        order_id        uuid PRIMARY KEY,
        slate_id        uuid NOT NULL REFERENCES $s.paper_slates ON DELETE CASCADE,
        account_id      text NOT NULL REFERENCES $s.paper_accounts,
        match_id        int  NOT NULL,
        kickoff         timestamptz NOT NULL,
        bf_market_id    text NOT NULL DEFAULT '',
        market_group    text NOT NULL,
        market_line     numeric(4,1) NOT NULL,
        selection       text NOT NULL,
        venue_selection text NOT NULL,
        side            text NOT NULL CHECK (side IN ('back','lay')),
        venue_odds      numeric(10,4) NOT NULL,
        leverage        numeric(12,6) NOT NULL,
        effective_odds  numeric(12,6) NOT NULL,
        p_model         numeric(9,6) NOT NULL,
        p_market        numeric(9,6) NOT NULL,
        edge            numeric(9,6) NOT NULL,
        stake_fraction  numeric(12,9) NOT NULL,
        risk            numeric(12,2) NOT NULL,
        venue_stake     numeric(12,2) NOT NULL,
        quote_ts        timestamptz NOT NULL,
        book_snapshot   jsonb NOT NULL DEFAULT '{}'::jsonb,
        state           text NOT NULL DEFAULT 'TRIGGERED',
        reason          text NOT NULL DEFAULT '',
        submitted_at    timestamptz,
        terminal_at     timestamptz,
        CONSTRAINT order_state_known CHECK (state IN
            ('TRIGGERED','PENDING_SUBMISSION','SUBMITTED','PARTIALLY_MATCHED','MATCHED',
             'CANCELLED','REJECTED','VOIDED','SETTLED')),
        CONSTRAINT leg_once UNIQUE (slate_id, match_id, market_group, market_line, selection)
    );""",

    "CREATE INDEX IF NOT EXISTS paper_orders_acct_state ON $s.paper_orders (account_id, state);",

    """
    CREATE TABLE IF NOT EXISTS $s.paper_fills (
        fill_id     bigserial PRIMARY KEY,
        order_id    uuid NOT NULL REFERENCES $s.paper_orders ON DELETE CASCADE,
        filled_at   timestamptz NOT NULL,
        price       numeric(10,4) NOT NULL,
        size        numeric(12,2) NOT NULL CHECK (size > 0),
        risk_filled numeric(12,2) NOT NULL,
        fill_model  text NOT NULL,
        level_depth int NOT NULL DEFAULT 0
    );""",

    "CREATE INDEX IF NOT EXISTS paper_fills_order ON $s.paper_fills (order_id);",

    """
    CREATE TABLE IF NOT EXISTS $s.paper_snapshots (
        order_id       uuid NOT NULL REFERENCES $s.paper_orders ON DELETE CASCADE,
        ts             timestamptz NOT NULL,
        best_back      numeric(10,4),
        best_lay       numeric(10,4),
        back_size      numeric(12,2),
        lay_size       numeric(12,2),
        mid_prob       numeric(9,6),
        market_matched numeric(14,2),
        mtm_pnl        numeric(12,2),
        PRIMARY KEY (order_id, ts)
    );""",

    """
    CREATE TABLE IF NOT EXISTS $s.clv_audit (
        order_id       uuid PRIMARY KEY REFERENCES $s.paper_orders ON DELETE CASCADE,
        entry_prob     numeric(9,6) NOT NULL,
        close_prob     numeric(9,6) NOT NULL,
        close_ts       timestamptz  NOT NULL,
        close_source   text         NOT NULL,
        clv            numeric(9,6) NOT NULL,
        clv_pct        numeric(9,6) NOT NULL,
        beat_close     boolean      NOT NULL,
        entry_lead_min int          NOT NULL
    );""",

    """
    CREATE TABLE IF NOT EXISTS $s.paper_settlements (
        order_id      uuid PRIMARY KEY REFERENCES $s.paper_orders ON DELETE CASCADE,
        settled_at    timestamptz NOT NULL DEFAULT now(),
        result_source text NOT NULL,
        home_goals    int,
        away_goals    int,
        outcome       text NOT NULL CHECK (outcome IN ('WIN','LOSE','VOID')),
        gross_return  numeric(12,2) NOT NULL,
        commission    numeric(12,2) NOT NULL,
        net_pnl       numeric(12,2) NOT NULL
    );""",

    """
    CREATE TABLE IF NOT EXISTS $s.account_ledger (
        entry_id       bigserial PRIMARY KEY,
        account_id     text NOT NULL REFERENCES $s.paper_accounts,
        at             timestamptz NOT NULL DEFAULT now(),
        kind           text NOT NULL,
        order_id       uuid REFERENCES $s.paper_orders ON DELETE SET NULL,
        slate_id       uuid REFERENCES $s.paper_slates ON DELETE SET NULL,
        delta_balance  numeric(12,2) NOT NULL,
        delta_reserved numeric(12,2) NOT NULL,
        balance_after  numeric(14,2) NOT NULL,
        reserved_after numeric(14,2) NOT NULL,
        note           text NOT NULL DEFAULT ''
    );""",

    "CREATE INDEX IF NOT EXISTS account_ledger_acct ON $s.account_ledger (account_id, at DESC);",

    # THE constraint that makes the reservation an atom: one RESERVE per slate, ever.
    """
    CREATE UNIQUE INDEX IF NOT EXISTS paper_ledger_one_reserve_per_slate
        ON $s.account_ledger (slate_id) WHERE kind = 'RESERVE';""",
    ]
end

"""
    migrate_paper_schema!(conn; schema = PAPER_SCHEMA)

Build (or bring up to date) the ledger. Every statement is `IF NOT EXISTS`, so this is safe to
run on every start-up and is the intended way to run it -- a migration that has to be remembered
is a migration that will be forgotten on the one Saturday it mattered.
"""
function migrate_paper_schema!(conn; schema::AbstractString = PAPER_SCHEMA)
    for stmt in paper_ddl(schema)
        LibPQ.execute(conn, stmt)
    end
    return schema
end

"""
    drop_paper_schema!(conn; schema)

Destroy a ledger schema and everything in it.

Refuses `"paper"` unless `force = true`. The production ledger is the record of what was
actually done; a test that can drop it by passing the wrong default is one keystroke from
erasing the evidence.
"""
function drop_paper_schema!(conn; schema::AbstractString, force::Bool = false)
    (schema == PAPER_SCHEMA && !force) && error(
        "drop_paper_schema!: refusing to drop the production schema '$PAPER_SCHEMA'. " *
        "Pass `force = true` if that is genuinely what you mean.")
    LibPQ.execute(conn, "DROP SCHEMA IF EXISTS $(schema) CASCADE;")
    return schema
end
