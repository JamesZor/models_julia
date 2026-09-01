# PostgreSQL persistence for portfolio backtests.
# Headline metrics and individual bets remain SQL-queryable; the exact result artefact preserves
# daily states, bootstrap intervals, custom metrics and attribution for lossless reloads.

export save_portfolio_db, load_portfolio_db, portfolio_spec_hash

"Stable SHA-256 display hash for a book or policy specification."
portfolio_spec_hash(spec) = bytes2hex(SHA.sha256(string(spec)))

function _portfolio_json_metadata(result::PortfolioResult, metadata)
    base = Dict{String, Any}(
        "converged" => result.converged,
        "failed_gates" => result.failed_gates,
        "n_slates" => result.summary.n_slates,
        "span_days" => result.summary.span_days,
    )
    for (key, value) in pairs(metadata)
        base[string(key)] = value
    end
    return JSON3.write(base)
end

"Persist a portfolio result and return its portfolio-run UUID."
function save_portfolio_db(result::PortfolioResult, run_id::UUID,
                           storage::Training.PostgresStorage;
                           book_spec = nothing, policy_spec = nothing,
                           book_spec_hash::AbstractString = book_spec === nothing ?
                               "unspecified" : portfolio_spec_hash(book_spec),
                           policy_spec_hash::AbstractString = policy_spec === nothing ?
                               "unspecified" : portfolio_spec_hash(policy_spec),
                           metadata = NamedTuple())
    portfolio_run_id = uuid4()
    summary = result.summary
    conn = Training.Inference._db_connect(storage)
    try
        Training.Inference._db_exec(conn, "BEGIN;")
        try
            Training.Inference._db_exec(conn, """
                INSERT INTO portfolio_runs (
                    portfolio_run_id, model_run_id, book_spec_hash, policy_spec_hash,
                    total_return_pct, flat_roi_pct, roi_1x2_pct, max_drawdown_pct,
                    sharpe_ann, win_rate, n_bets, created_at, metadata
                ) VALUES (\$1::uuid, \$2::uuid, \$3, \$4, \$5, \$6, \$7, \$8,
                          \$9, \$10, \$11, \$12, \$13::jsonb);
            """, (string(portfolio_run_id), string(run_id), String(book_spec_hash),
                  String(policy_spec_hash), summary.total_return_pct, summary.roi,
                  Training.Inference._db_nullable(summary.roi_1x2), summary.mdd,
                  Training.Inference._db_nullable(summary.sharpe_ann),
                  Training.Inference._db_nullable(summary.win_rate), summary.n_bets,
                  now(), _portfolio_json_metadata(result, metadata)))

            bets = result.trajectory.bets
            if nrow(bets) > 0
                open_bankroll = Dict(state.date => state.bankroll_open
                                     for state in result.daily_states)
                table = (
                    portfolio_run_id = fill(string(portfolio_run_id), nrow(bets)),
                    match_id = Int.(bets.match_id),
                    kickoff_date = Date.(bets.date),
                    market_family = String.(bets.family),
                    selection = string.(bets.selection),
                    odds_close = Float64.(bets.odds),
                    stake_fraction = Float64.(bets.stake),
                    stake_amount = [Float64(bets.stake[i]) * open_bankroll[Date(bets.date[i])]
                                    for i in 1:nrow(bets)],
                    pnl = [Float64(bets.pnl[i]) * open_bankroll[Date(bets.date[i])]
                           for i in 1:nrow(bets)],
                )
                LibPQ.load!(table, conn, """
                    INSERT INTO portfolio_bets (
                        portfolio_run_id, match_id, kickoff_date, market_family, selection,
                        odds_close, stake_fraction, stake_amount, pnl
                    ) VALUES (\$1::uuid, \$2, \$3, \$4, \$5, \$6, \$7, \$8, \$9);
                """)
            end

            artifact = Training.Inference._db_artifact_blob(result)
            book_blob = book_spec === nothing ? missing :
                        Training.Inference._db_bytea(Training.Inference._db_artifact_blob(book_spec))
            policy_blob = policy_spec === nothing ? missing :
                          Training.Inference._db_bytea(Training.Inference._db_artifact_blob(policy_spec))
            Training.Inference._db_exec(conn, """
                INSERT INTO portfolio_artifacts (
                    portfolio_run_id, result_blob, book_spec_blob, policy_spec_blob
                ) VALUES (\$1::uuid, \$2::bytea, \$3::bytea, \$4::bytea);
            """, (string(portfolio_run_id), Training.Inference._db_bytea(artifact),
                  book_blob, policy_blob))
            Training.Inference._db_exec(conn, "COMMIT;")
        catch
            try
                Training.Inference._db_exec(conn, "ROLLBACK;")
            catch
            end
            rethrow()
        end
        return portfolio_run_id
    finally
        close(conn)
    end
end

save_portfolio_db(result::PortfolioResult, run_id::AbstractString,
                  storage::Training.PostgresStorage; kwargs...) =
    save_portfolio_db(result, UUID(run_id), storage; kwargs...)

"Load the exact `PortfolioResult` associated with a portfolio-run UUID."
function load_portfolio_db(portfolio_run_id::UUID, storage::Training.PostgresStorage)
    conn = Training.Inference._db_connect(storage)
    try
        rows = Training.Inference._db_rows(conn, """
            SELECT result_blob FROM portfolio_artifacts
            WHERE portfolio_run_id = \$1::uuid;
        """, (string(portfolio_run_id),))
        nrow(rows) == 1 || error(
            "load_portfolio_db: no PostgreSQL artefact for run $portfolio_run_id.")
        result = Training.Inference._db_artifact_value(rows.result_blob[1])
        result isa PortfolioResult || error(
            "load_portfolio_db: artefact for $portfolio_run_id holds $(typeof(result)).")
        return result
    finally
        close(conn)
    end
end

load_portfolio_db(portfolio_run_id::AbstractString, storage::Training.PostgresStorage) =
    load_portfolio_db(UUID(portfolio_run_id), storage)
