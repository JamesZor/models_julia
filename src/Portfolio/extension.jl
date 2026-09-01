# Incremental roll-forward of a persisted portfolio.
#
# Only posterior fixtures absent from portfolio_bets are priced.  Their simulation starts at the
# existing closing bankroll; the old and delta trajectories are then combined and all headline
# metrics are recomputed before SQL rows and the exact artefact are updated atomically.

function _portfolio_extension_row(db::Training.PostgresStorage, key)
    conn = Training.Inference._db_connect(db)
    try
        rows = if key isa Integer
            Training.Inference._db_rows(conn, """
                SELECT pr.portfolio_run_id, pr.model_run_id, pr.book_spec_hash,
                       pr.policy_spec_hash
                FROM portfolio_runs pr
                JOIN runs r ON r.run_id = pr.model_run_id
                WHERE r.experiment_name = \$1 AND (pr.id = \$2 OR r.id = \$2)
                ORDER BY CASE WHEN pr.id = \$2 THEN 0 ELSE 1 END, pr.id DESC
                LIMIT 1;
            """, (db.experiment_name, Int(key)))
        else
            text = string(key)
            Training.Inference._db_rows(conn, """
                SELECT pr.portfolio_run_id, pr.model_run_id, pr.book_spec_hash,
                       pr.policy_spec_hash
                FROM portfolio_runs pr
                JOIN runs r ON r.run_id = pr.model_run_id
                WHERE r.experiment_name = \$1
                  AND (pr.portfolio_run_id::text = \$2 OR pr.model_run_id::text = \$2 OR r.name = \$2)
                ORDER BY CASE WHEN pr.portfolio_run_id::text = \$2 THEN 0 ELSE 1 END, pr.id DESC
                LIMIT 1;
            """, (db.experiment_name, text))
        end
        nrow(rows) == 1 || error(
            "extend_portfolio: no portfolio or model run $(repr(key)) in experiment '$(db.experiment_name)'.")
        return (; portfolio_run_id = UUID(string(rows.portfolio_run_id[1])),
                model_run_id = UUID(string(rows.model_run_id[1])),
                book_hash = String(rows.book_spec_hash[1]),
                policy_hash = String(rows.policy_spec_hash[1]))
    finally
        close(conn)
    end
end

function _portfolio_artifact_spec(db::Training.PostgresStorage, portfolio_run_id::UUID,
                                  column::String)
    column in ("book_spec_blob", "policy_spec_blob") || error(
        "invalid portfolio spec column $column")
    conn = Training.Inference._db_connect(db)
    try
        rows = Training.Inference._db_rows(conn, """
            SELECT $column AS spec_blob FROM portfolio_artifacts
            WHERE portfolio_run_id = \$1::uuid;
        """, (string(portfolio_run_id),))
        (nrow(rows) == 1 && !ismissing(rows.spec_blob[1])) || return nothing
        return Training.Inference._db_artifact_value(rows.spec_blob[1])
    finally
        close(conn)
    end
end

function _portfolio_registered_spec(db::Training.PostgresStorage, kind::String, hash::String)
    hash == "unspecified" && return nothing
    conn = Training.Inference._db_connect(db)
    try
        rows = Training.Inference._db_rows(conn, """
            SELECT config_blob FROM config_registry
            WHERE experiment_name = \$1 AND config_type = \$2
            ORDER BY updated_at DESC;
        """, (db.experiment_name, kind))
        for blob in rows.config_blob
            value = Training.Inference._db_artifact_value(blob)
            portfolio_spec_hash(value) == hash && return value
        end
        return nothing
    finally
        close(conn)
    end
end

function _portfolio_extension_specs(db, row, book_spec, policy_spec)
    stored_book = book_spec === nothing ?
                  _portfolio_artifact_spec(db, row.portfolio_run_id, "book_spec_blob") : nothing
    stored_policy = policy_spec === nothing ?
                    _portfolio_artifact_spec(db, row.portfolio_run_id, "policy_spec_blob") : nothing
    book = book_spec === nothing ?
           (stored_book === nothing ?
                _portfolio_registered_spec(db, "book_spec", row.book_hash) : stored_book) : book_spec
    policy = policy_spec === nothing ?
             (stored_policy === nothing ?
                _portfolio_registered_spec(db, "policy_spec", row.policy_hash) : stored_policy) : policy_spec
    book isa BookSpec || error(
        "extend_portfolio: supply `book_spec = ...`; no registered BookSpec matches hash $(row.book_hash).")
    policy isa PolicySpec || error(
        "extend_portfolio: supply `policy_spec = ...`; no registered PolicySpec matches hash $(row.policy_hash).")
    return book, policy
end

function _portfolio_completed_matches(db, portfolio_run_id::UUID)
    conn = Training.Inference._db_connect(db)
    try
        rows = Training.Inference._db_rows(conn, """
            SELECT DISTINCT match_id FROM portfolio_bets
            WHERE portfolio_run_id = \$1::uuid;
        """, (string(portfolio_run_id),))
        return Set(Int.(rows.match_id))
    finally
        close(conn)
    end
end

function _portfolio_subset(latents::Models.CountLatents, ids::Set{Int})
    positions = Int[i for i in eachindex(latents.match_ids) if latents.match_ids[i] in ids]
    isempty(positions) && return nothing
    observation = latents.observation_params
    return observation === nothing ?
        Models.CountLatents(latents.match_ids[positions], latents.λ_home[positions, :],
                            latents.λ_away[positions, :]) :
        Models.CountLatents(latents.match_ids[positions], latents.λ_home[positions, :],
                            latents.λ_away[positions, :],
                            (; r_h = observation.r_h[positions, :],
                               r_a = observation.r_a[positions, :]))
end

function _portfolio_subset(latents::Models.AbstractPosteriorLatents, ids::Set{Int})
    error("extend_portfolio currently supports CountLatents; got $(typeof(latents)).")
end

function _portfolio_empty_delta(existing::PortfolioResult)
    return PortfolioResult(existing.daily_states, existing.summary, existing.metrics,
                           existing.bootstrap_ci, existing.trajectory, existing.attribution,
                           existing.converged, existing.failed_gates)
end

function _portfolio_combined_result(existing::PortfolioResult, delta::PortfolioResult,
                                    fit::Training.Fit)
    offset = length(existing.daily_states)
    states = copy(existing.daily_states)
    append!(states, DailyState[
        DailyState(offset + state.idx, state.date, state.n_fixtures, state.n_bets,
                   state.bankroll_open, state.bankroll_close, state.stake_frac,
                   state.pnl_frac, state.exposure, state.k_risk, state.capped)
        for state in delta.daily_states])

    old = existing.trajectory
    fresh = delta.trajectory
    scale = old.bankroll[end]
    bankroll = vcat(old.bankroll, scale .* fresh.bankroll[2:end])
    bets = isempty(old.bets) ? copy(fresh.bets) :
           isempty(fresh.bets) ? copy(old.bets) : vcat(old.bets, fresh.bets; cols = :union)
    trajectory = Trajectory(bankroll, vcat(old.dates, fresh.dates),
                            vcat(old.slate_pl, fresh.slate_pl),
                            vcat(old.k_risk, fresh.k_risk),
                            vcat(old.exposure, fresh.exposure),
                            old.n_capped + fresh.n_capped,
                            old.total_stake + fresh.total_stake,
                            old.total_pl + fresh.total_pl, bets)

    stake_1x2 = 0.0
    pnl_1x2 = 0.0
    n_wins = 0
    for row in eachrow(bets)
        row.payoff > 0 && (n_wins += 1)
        if startswith(String(row.family), "1X2")
            stake_1x2 += Float64(row.stake)
            pnl_1x2 += Float64(row.pnl)
        end
    end
    initial = existing.summary.initial_bankroll
    summary = portfolio_summary(states, trajectory, initial;
                                stake_1x2 = stake_1x2, pl_1x2 = pnl_1x2,
                                n_wins = n_wins)
    ci = if existing.bootstrap_ci === nothing
        nothing
    else
        bootstrap_portfolio(trajectory; B = existing.bootstrap_ci.B,
                            seed = existing.bootstrap_ci.seed)
    end
    return PortfolioResult(states, summary, existing.metrics, ci, trajectory,
                           attribution(trajectory), fit.diagnostics.passed,
                           copy(fit.diagnostics.failed_gates))
end

function _portfolio_insert_bets!(conn, portfolio_run_id::UUID, result::PortfolioResult)
    bets = result.trajectory.bets
    nrow(bets) == 0 && return nothing
    open_bankroll = Dict(state.date => state.bankroll_open for state in result.daily_states)
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
    return nothing
end

"""
    extend_portfolio(db, portfolio_id_or_model_run, fit, odds_df, ds;
                     book_spec=nothing, policy_spec=nothing) -> PortfolioResult

Price and simulate only latent fixtures not represented in the persisted portfolio's bet table,
continue from its closing bankroll, then atomically append bets and refresh its summary/artifact.
A model-run identifier resolves to that model's latest portfolio run.
"""
function extend_portfolio(db::Training.PostgresStorage, key, fit::Training.Fit,
                          odds_df::DataFrame, ds::Data.DataStore;
                          book_spec = nothing, policy_spec = nothing)
    row = _portfolio_extension_row(db, key)
    book, policy = _portfolio_extension_specs(db, row, book_spec, policy_spec)
    existing = load_portfolio_db(row.portfolio_run_id, db)
    fit.latents isa Models.AbstractPosteriorLatents || error(
        "extend_portfolio: Fit '$(Training.fit_name(fit))' has no posterior latents.")

    completed = _portfolio_completed_matches(db, row.portfolio_run_id)
    candidates = Set(Int(id) for id in Models.latent_match_ids(fit.latents) if Int(id) ∉ completed)
    delta_latents = _portfolio_subset(fit.latents, candidates)
    delta_latents === nothing && return _portfolio_empty_delta(existing)
    delta_ids = Set(Models.latent_match_ids(delta_latents))
    filtered_odds = DataFrames.subset(odds_df, :match_id => DataFrames.ByRow(id -> Int(id) in delta_ids))

    books, report = build_books_reported(book, delta_latents, filtered_odds, ds;
                                         require_result = true,
                                         converged = fit.diagnostics.passed,
                                         failed_gates = copy(fit.diagnostics.failed_gates),
                                         quiet = true)
    delta = simulate_portfolio(policy, books, report;
                               initial_bankroll = existing.summary.final_bankroll,
                               bootstrap = false)
    updated = _portfolio_combined_result(existing, delta, fit)

    conn = Training.Inference._db_connect(db)
    try
        Training.Inference._db_exec(conn, "BEGIN;")
        try
            locked = Training.Inference._db_rows(conn, """
                SELECT portfolio_run_id FROM portfolio_runs
                WHERE portfolio_run_id = \$1::uuid FOR UPDATE;
            """, (string(row.portfolio_run_id),))
            nrow(locked) == 1 || error(
                "extend_portfolio: portfolio $(row.portfolio_run_id) disappeared.")
            now_completed = Training.Inference._db_rows(conn, """
                SELECT DISTINCT match_id FROM portfolio_bets
                WHERE portfolio_run_id = \$1::uuid AND match_id = ANY(\$2::int[]);
            """, (string(row.portfolio_run_id), collect(delta_ids)))
            nrow(now_completed) == 0 || error(
                "extend_portfolio: candidate matches were appended concurrently; retry the extension.")

            _portfolio_insert_bets!(conn, row.portfolio_run_id, delta)
            summary = updated.summary
            Training.Inference._db_exec(conn, """
                UPDATE portfolio_runs SET
                    total_return_pct = \$2, flat_roi_pct = \$3, roi_1x2_pct = \$4,
                    max_drawdown_pct = \$5, sharpe_ann = \$6, win_rate = \$7,
                    n_bets = \$8, metadata = \$9::jsonb
                WHERE portfolio_run_id = \$1::uuid;
            """, (string(row.portfolio_run_id), summary.total_return_pct, summary.roi,
                  Training.Inference._db_nullable(summary.roi_1x2), summary.mdd,
                  Training.Inference._db_nullable(summary.sharpe_ann),
                  Training.Inference._db_nullable(summary.win_rate), summary.n_bets,
                  _portfolio_json_metadata(updated,
                      (; extended_at = string(now()), n_new_matches = length(delta_ids)))))
            Training.Inference._db_exec(conn, """
                UPDATE portfolio_artifacts SET result_blob = \$2::bytea
                WHERE portfolio_run_id = \$1::uuid;
            """, (string(row.portfolio_run_id),
                  Training.Inference._db_bytea(Training.Inference._db_artifact_blob(updated))))
            Training.Inference._db_exec(conn, "COMMIT;")
        catch
            try
                Training.Inference._db_exec(conn, "ROLLBACK;")
            catch
            end
            rethrow()
        end
    finally
        close(conn)
    end
    return updated
end
