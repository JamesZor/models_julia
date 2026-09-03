# Shared apparatus for the stochastic-control / market-capacity EDA.
#
# This file defines no top-level workflow. The two runnable scripts in `eda/` include it so the
# database addresses, market taxonomy, Betfair close, portfolio recipe, and metric definitions
# cannot drift between Parts A/B and Part C.

const SC_PF = BayesianFootball.Portfolio
const SC_EV = BayesianFootball.Evaluation

const SC_OUTPUT_DIR = joinpath(@__DIR__, "results", "stochastic_control_capacity")
const SC_DATABASE_DSN = "host=mcmc-beast user=postgres dbname=mcmc_experiments"
const SC_CAP_THRESHOLD = 0.80
const SC_CAP = 0.20
const SC_INITIAL_BANKROLL = 1_000.0

struct SCCanonicalRun
    experiment::String
    name::String
    run_uuid::UUID
    config_hash::String
    portfolio_uuid::UUID
end

# Immutable addresses from mcmc_experiments. Names are retained for readability; the scripts load
# by UUID and verify the config hash before using an artefact.
const SC_CANONICAL_RUNS = SCCanonicalRun[
    SCCanonicalRun(
        "scottish_lower_joint_2426",
        "m00_joint_baseline",
        UUID("2c6e859c-29e7-4ae7-aa0a-e88343ba7672"),
        "56dfc90e649ef06b638e8a2fdf8d9e13427172fbf8bebb633450f1ee427d9610",
        UUID("cd138387-6760-40e0-ae25-c007cba8c044"),
    ),
    SCCanonicalRun(
        "scottish_lower_joint_2426",
        "m05_joint_production_wealth",
        UUID("5eff755c-3591-48d1-a2cc-5fc2744ddf88"),
        "13dc1fc62a0d87e9d02031840e76f91418215170c0687b19b811a72903b84e39",
        UUID("c5192d31-11ca-4736-873b-1a2055c232be"),
    ),
    SCCanonicalRun(
        "scottish_lower_joint_player_2426",
        "m12_joint_hybrid_synergy",
        UUID("132df5c2-c742-4e95-8693-3aeb2b2cbaef"),
        "a11db174f91861a795c150bbdfb6f34b5814516f0a179c153ebec912eed5670d",
        UUID("d08b43b2-226a-4ecd-abd2-d9c80966ca08"),
    ),
    SCCanonicalRun(
        "scottish_lower_joint_player_2426",
        "m13_joint_composite",
        UUID("5474e824-8c9d-4613-8e39-841426c3f80f"),
        "d9fef944fad8a0d61490379aaabcdfebd512b49de9ac95d58cfe3216bcae83cf",
        UUID("ad4029f0-e1ee-4bcf-b930-5ef566558061"),
    ),
]

sc_policy_runs() = filter(r -> r.name in ("m12_joint_hybrid_synergy", "m13_joint_composite"),
                          SC_CANONICAL_RUNS)

const SC_MARKET_MENU = [
    ("1X2", Data.Market1X2()),
    ("OU0.5", Data.MarketOverUnder(0.5)),
    ("OU1.5", Data.MarketOverUnder(1.5)),
    ("OU2.5", Data.MarketOverUnder(2.5)),
    ("OU3.5", Data.MarketOverUnder(3.5)),
    ("BTTS", Data.MarketBTTS()),
]

const SC_MARKETS = Data.AbstractMarket[m for (_, m) in SC_MARKET_MENU]
const SC_MARKET_SELECTIONS = Dict(
    "1X2" => [:home, :draw, :away],
    "OU0.5" => [:over_05, :under_05],
    "OU1.5" => [:over_15, :under_15],
    "OU2.5" => [:over_25, :under_25],
    "OU3.5" => [:over_35, :under_35],
    "BTTS" => [:btts_yes, :btts_no],
)
const SC_CORE_KEYS = Set(["1X2", "OU2.5"])
const SC_FRINGE_KEYS = Set(["OU0.5", "OU1.5", "OU3.5", "BTTS"])

function sc_market_key(group::AbstractString, line::Real)
    group == "1X2" && return "1X2"
    group == "BTTS" && return "BTTS"
    group == "OverUnder" && return "OU" * string(Float64(line))
    return String(group)
end

sc_capacity_bucket(key::AbstractString) = key in SC_CORE_KEYS ? "core" : "fringe"

function sc_market_key_from_family(family::AbstractString)
    startswith(family, "1X2_") && return "1X2"
    startswith(family, "BTTS_") && return "BTTS"
    for (needle, key) in (("O/U 0.5_", "OU0.5"), ("O/U 1.5_", "OU1.5"),
                          ("O/U 2.5_", "OU2.5"), ("O/U 3.5_", "OU3.5"))
        startswith(family, needle) && return key
    end
    return "UNKNOWN"
end

"Betfair exchange close, time-weighted over the final twenty minutes before kickoff."
function sc_betfair_closing_odds(ds::Data.DataStore)
    raw = Data.summarize_odds(ds.betfair_odds, Data.TWAEstimator(); window = (-20.0, 0.0))
    odds = DataFrame(
        match_id = Int.(raw.match_id),
        market_name = String.(raw.market_name),
        market_line = Float64.(raw.market_line),
        selection = Symbol.(raw.selection),
        odds_close = Float64.(raw.odds),
    )
    filter!(row -> isfinite(row.odds_close) && row.odds_close > 1.0, odds)
    sort!(odds, [:match_id, :market_name, :market_line, :selection])
    return odds
end

"Add vig-free market probabilities and outcomes for the evaluation API."
function sc_enrich_odds(odds::DataFrame, ds::Data.DataStore)
    enriched = copy(odds)
    enriched.prob_implied_close = 1.0 ./ enriched.odds_close
    DataFrames.transform!(
        groupby(enriched, [:match_id, :market_name, :market_line]),
        :prob_implied_close => (p -> p ./ sum(p)) => :prob_fair_close,
    )
    outcomes = unique(DataFrames.select(
        ds.odds, [:match_id, :market_name, :market_line, :selection, :is_winner]))
    enriched = leftjoin(
        enriched,
        outcomes;
        on = [:match_id, :market_name, :market_line, :selection],
    )
    sort!(enriched, [:match_id, :market_name, :market_line, :selection])
    return enriched
end

"Full six-market book. Fractional sizing lives in trust for this EDA, so shrink is identity."
function sc_book_spec()
    return BookSpec(
        markets = Data.MarketConfig(copy(SC_MARKETS)),
        price = DeArb(),
        allocator = KellyLogUtility(),
        shrink = SC_PF.NoShrinkage(),
        exec = ExecutionConfig(
            commission = PerBetCommission(0.02),
            budget = 0.99,
            min_selection_stake = 0.001,
        ),
    )
end

function sc_selection_trust(; core::Real, fringe::Real)
    table = Dict{Tuple{String,Float64,Symbol},Float64}()
    for selection in SC_MARKET_SELECTIONS["1X2"]
        table[("1X2", 0.0, selection)] = Float64(core)
    end
    for selection in SC_MARKET_SELECTIONS["BTTS"]
        table[("BTTS", 0.0, selection)] = Float64(fringe)
    end
    for line in (0.5, 1.5, 2.5, 3.5)
        key = "OU" * string(line)
        weight = key in SC_CORE_KEYS ? Float64(core) : Float64(fringe)
        for selection in SC_MARKET_SELECTIONS[key]
            table[("OverUnder", line, selection)] = weight
        end
    end
    return SC_PF.SelectionTrust(table; strict = true)
end

sc_status_quo_policy() = PolicySpec(
    trust = FlatTrust(0.30),
    risk = SlateDrawdown(23.0),
    cap = FixedCap(SC_CAP),
    grouping = DailySlate(),
)

sc_hard_pruning_policy() = PolicySpec(
    trust = sc_selection_trust(core = 0.30, fringe = 0.00),
    risk = SlateDrawdown(23.0),
    cap = FixedCap(SC_CAP),
    grouping = DailySlate(),
)

sc_damped_tail_policy() = PolicySpec(
    trust = sc_selection_trust(core = 0.30, fringe = 0.05),
    risk = SlateDrawdown(23.0),
    cap = FixedCap(SC_CAP),
    grouping = DailySlate(),
)

"Causal drawdown multiplier applied to the baseline 0.30 trust."
function sc_drawdown_multiplier(drawdown::Real)
    drawdown > -0.05 && return 1.00
    drawdown > -0.10 && return 0.75
    drawdown > -0.15 && return 0.50
    return 0.25
end

"Stateful trust controller; state is updated once from each slate's opening bankroll."
mutable struct SCDrawdownAdaptiveTrust <: SC_PF.AbstractTrustModel
    base::Float64
    peak::Float64
    last_idx::Int
    current_multiplier::Float64
end

SCDrawdownAdaptiveTrust(base::Real = 0.30) =
    SCDrawdownAdaptiveTrust(Float64(base), 1.0, 0, 1.0)

function SC_PF.trust_for(t::SCDrawdownAdaptiveTrust, ::SC_PF.Selection,
                         ctx::SC_PF.SlateContext)
    if ctx.idx != t.last_idx
        ctx.idx == 1 && (t.peak = max(1.0, ctx.bankroll))
        t.peak = max(t.peak, ctx.bankroll)
        drawdown = t.peak > 0 ? ctx.bankroll / t.peak - 1.0 : -1.0
        t.current_multiplier = sc_drawdown_multiplier(drawdown)
        t.last_idx = ctx.idx
    end
    return t.base * t.current_multiplier
end

sc_drawdown_adaptive_policy() = PolicySpec(
    trust = SCDrawdownAdaptiveTrust(0.30),
    risk = SlateDrawdown(23.0),
    cap = FixedCap(SC_CAP),
    grouping = DailySlate(),
)

function sc_database_connection()
    return LibPQ.Connection(SC_DATABASE_DSN)
end

function sc_verify_run_inventory()
    conn = sc_database_connection()
    rows = NamedTuple[]
    try
        for ref in SC_CANONICAL_RUNS
            query = LibPQ.execute(conn, """
                SELECT r.experiment_name, r.name, r.id, r.run_id::text, c.config_hash,
                       r.git_commit, r.git_branch, r.status, r.created_at,
                       COUNT(fr.*) AS n_folds,
                       COUNT(*) FILTER (WHERE fr.converged) AS converged_folds
                FROM runs r
                JOIN configs c ON c.config_id = r.run_id
                LEFT JOIN fold_results fr ON fr.run_id = r.run_id
                WHERE r.run_id = \$1::uuid
                GROUP BY r.experiment_name, r.name, r.id, r.run_id, c.config_hash,
                         r.git_commit, r.git_branch, r.status, r.created_at;
            """, (string(ref.run_uuid),))
            frame = DataFrame(query)
            close(query)
            nrow(frame) == 1 || error("Canonical run $(ref.run_uuid) did not resolve exactly once.")
            row = frame[1, :]
            String(row.experiment_name) == ref.experiment || error(
                "Run $(ref.run_uuid) experiment changed: $(row.experiment_name).")
            String(row.name) == ref.name || error(
                "Run $(ref.run_uuid) name changed: $(row.name).")
            String(row.config_hash) == ref.config_hash || error(
                "Run $(ref.run_uuid) config hash changed: $(row.config_hash).")
            push!(rows, (
                experiment = ref.experiment,
                model = ref.name,
                database_id = Int(row.id),
                run_uuid = string(ref.run_uuid),
                config_hash = ref.config_hash,
                status = String(row.status),
                git_commit = String(row.git_commit),
                git_branch = String(row.git_branch),
                created_at = row.created_at,
                n_folds = Int(row.n_folds),
                converged_folds = Int(row.converged_folds),
                strict_all_folds_converged = Int(row.converged_folds) == Int(row.n_folds),
                portfolio_uuid = string(ref.portfolio_uuid),
            ))
        end
    finally
        close(conn)
    end
    return DataFrame(rows)
end

function sc_load_fit(ref::SCCanonicalRun)
    storage = PostgresStorage(ref.experiment)
    fit = load_fit(storage, ref.run_uuid)
    length(fit) == 40 || error("$(ref.name) has $(length(fit)) folds; expected 40.")
    return fit
end

"Load the exact persisted result plus the BookSpec and PolicySpec saved beside it."
function sc_load_portfolio_bundle(ref::SCCanonicalRun)
    storage = PostgresStorage(ref.experiment)
    result = load_portfolio_db(ref.portfolio_uuid, storage)
    conn = sc_database_connection()
    try
        query = LibPQ.execute(conn, """
            SELECT pr.model_run_id::text, pr.book_spec_hash, pr.policy_spec_hash,
                   pr.created_at, pr.metadata::text,
                   pa.book_spec_blob, pa.policy_spec_blob
            FROM portfolio_runs pr
            JOIN portfolio_artifacts pa USING (portfolio_run_id)
            WHERE pr.portfolio_run_id = \$1::uuid;
        """, (string(ref.portfolio_uuid),))
        frame = DataFrame(query)
        close(query)
        nrow(frame) == 1 || error("Portfolio $(ref.portfolio_uuid) did not resolve exactly once.")
        String(frame.model_run_id[1]) == string(ref.run_uuid) || error(
            "Portfolio $(ref.portfolio_uuid) is not attached to run $(ref.run_uuid).")
        book = BayesianFootball.Training.Inference._db_artifact_value(frame.book_spec_blob[1])
        policy = BayesianFootball.Training.Inference._db_artifact_value(frame.policy_spec_blob[1])
        return (;
            result,
            book,
            policy,
            book_spec_hash = String(frame.book_spec_hash[1]),
            policy_spec_hash = String(frame.policy_spec_hash[1]),
            created_at = frame.created_at[1],
            metadata = String(frame.metadata[1]),
        )
    finally
        close(conn)
    end
end

function sc_query_canonical_database_bets()
    conn = sc_database_connection()
    frames = DataFrame[]
    try
        for ref in SC_CANONICAL_RUNS
            query = LibPQ.execute(conn, """
                SELECT pb.bet_id, pb.portfolio_run_id::text, pb.match_id, pb.kickoff_date,
                       pb.market_family, pb.selection, pb.odds_close, pb.stake_fraction,
                       pb.stake_amount, pb.pnl
                FROM portfolio_bets pb
                WHERE pb.portfolio_run_id = \$1::uuid
                ORDER BY pb.kickoff_date, pb.bet_id;
            """, (string(ref.portfolio_uuid),))
            frame = DataFrame(query)
            close(query)
            frame.experiment = fill(ref.experiment, nrow(frame))
            frame.model = fill(ref.name, nrow(frame))
            frame.run_uuid = fill(string(ref.run_uuid), nrow(frame))
            frame.market_key = sc_market_key_from_family.(String.(frame.market_family))
            frame.capacity_bucket = sc_capacity_bucket.(frame.market_key)
            frame.won = frame.pnl .> 0.0
            push!(frames, frame)
        end
    finally
        close(conn)
    end
    return vcat(frames...; cols = :union)
end

"Map the internal family key back to group/line without parsing its display encoding."
function sc_family_catalog(books)
    catalog = Dict{String,NamedTuple}()
    for book in books, selection in book.sels
        get!(catalog, selection.family) do
            (;
                group = selection.group,
                line = selection.line,
                selection = selection.selection,
            )
        end
    end
    return catalog
end

"Controlled-simulation ledger in both bankroll fractions and currency."
function sc_simulation_ledger(result, books, model::AbstractString, policy::AbstractString)
    bets = copy(result.trajectory.bets)
    nrow(bets) == 0 && return DataFrame()
    opening = Dict(state.date => state.bankroll_open for state in result.daily_states)
    exposure = Dict(state.date => state.exposure for state in result.daily_states)
    catalog = sc_family_catalog(books)

    bets.model = fill(String(model), nrow(bets))
    bets.policy = fill(String(policy), nrow(bets))
    bets.group = [catalog[String(f)].group for f in bets.family]
    bets.line = [catalog[String(f)].line for f in bets.family]
    bets.market_key = [sc_market_key(catalog[String(f)].group, catalog[String(f)].line)
                       for f in bets.family]
    bets.capacity_bucket = sc_capacity_bucket.(bets.market_key)
    bets.bankroll_open = [opening[Date(d)] for d in bets.date]
    bets.exposure = [exposure[Date(d)] for d in bets.date]
    bets.constrained = bets.exposure .>= SC_CAP_THRESHOLD * SC_CAP
    bets.stake_amount = bets.stake .* bets.bankroll_open
    bets.pnl_amount = bets.pnl .* bets.bankroll_open
    bets.won = bets.payoff .> 0.0
    bets.pushed = bets.payoff .== 0.0
    return bets
end

function sc_capacity_segment_summary(ledger::DataFrame, states, model::AbstractString;
                                     cap::Real = SC_CAP)
    rows = NamedTuple[]
    for regime in ("constrained", "unconstrained", "all")
        regime_frame = regime == "all" ? ledger :
            ledger[ledger.constrained .== (regime == "constrained"), :]
        regime_states = regime == "all" ? states :
            [s for s in states if (s.exposure >= SC_CAP_THRESHOLD * cap) ==
                                  (regime == "constrained")]
        regime_stake = sum(regime_frame.stake_amount)
        for bucket in ("core", "fringe")
            frame = regime_frame[regime_frame.capacity_bucket .== bucket, :]
            stake = sum(frame.stake_amount)
            pnl = sum(frame.pnl_amount)
            push!(rows, (
                model = String(model),
                regime,
                capacity_bucket = bucket,
                n_slates = length(regime_states),
                n_bets = nrow(frame),
                stake_amount = stake,
                stake_share_pct = regime_stake > 0 ? 100 * stake / regime_stake : NaN,
                pnl_amount = pnl,
                win_rate_pct = nrow(frame) > 0 ? 100 * mean(frame.won) : NaN,
                kelly_efficiency_pct = stake > 0 ? 100 * pnl / stake : NaN,
                mean_slate_exposure = isempty(regime_states) ? NaN :
                                      mean(s.exposure for s in regime_states),
                cap = Float64(cap),
                threshold = SC_CAP_THRESHOLD * Float64(cap),
            ))
        end
    end
    return DataFrame(rows)
end

function sc_stake_by_date(ledger::DataFrame, bucket::AbstractString)
    frame = ledger[ledger.capacity_bucket .== bucket, :]
    isempty(frame) && return Dict{Date,Float64}()
    totals = combine(groupby(frame, :date), :stake => sum => :stake)
    return Dict(Date(r.date) => Float64(r.stake) for r in eachrow(totals))
end

function sc_opportunity_cost(status_result, status_ledger::DataFrame,
                             core_result, core_ledger::DataFrame,
                             model::AbstractString; cap::Real = SC_CAP)
    status_state = Dict(s.date => s for s in status_result.daily_states)
    core_state = Dict(s.date => s for s in core_result.daily_states)
    dates = sort(collect(keys(status_state)))
    Set(dates) == Set(keys(core_state)) || error(
        "$model policy simulations do not cover the same slate dates.")
    constrained = Set(d for d in dates
                      if status_state[d].exposure >= SC_CAP_THRESHOLD * cap)

    status_core = sc_stake_by_date(status_ledger, "core")
    status_fringe = sc_stake_by_date(status_ledger, "fringe")
    core_only = sc_stake_by_date(core_ledger, "core")

    status_binding_growth = prod(1.0 + status_state[d].pnl_frac for d in constrained)
    core_binding_growth = prod(1.0 + core_state[d].pnl_frac for d in constrained)
    hybrid_growth = prod(1.0 + (d in constrained ? core_state[d].pnl_frac :
                               status_state[d].pnl_frac) for d in dates)

    removed = sum(get(status_fringe, d, 0.0) for d in constrained)
    redirected = sum(get(core_only, d, 0.0) - get(status_core, d, 0.0) for d in constrained)
    pnl_delta = sum(core_state[d].pnl_frac - status_state[d].pnl_frac for d in constrained)

    return (
        model = String(model),
        n_slates = length(dates),
        n_constrained_slates = length(constrained),
        constrained_share_pct = 100 * length(constrained) / max(length(dates), 1),
        fringe_stake_removed_frac = removed,
        core_stake_change_frac = redirected,
        capacity_recaptured_pct = removed > 0 ? 100 * redirected / removed : NaN,
        status_return_constrained_pct = 100 * (status_binding_growth - 1.0),
        core_return_constrained_pct = 100 * (core_binding_growth - 1.0),
        constrained_return_delta_pp = 100 * (core_binding_growth - status_binding_growth),
        status_full_return_pct = status_result.summary.total_return_pct,
        core_full_return_pct = core_result.summary.total_return_pct,
        hybrid_binding_only_return_pct = 100 * (hybrid_growth - 1.0),
        hybrid_opportunity_cost_pp = 100 * (hybrid_growth -
                                             status_result.summary.final_bankroll /
                                             status_result.summary.initial_bankroll),
        realized_shadow_value_pct_per_stake = removed > 0 ? 100 * pnl_delta / removed : NaN,
    )
end

function sc_evaluation_context(fit, enriched_odds::DataFrame, ds::Data.DataStore)
    return SC_EV.build_evaluation_context(
        SC_EV.fit_latents(fit),
        enriched_odds,
        ds.matches,
        [SC_EV.PredictionScore()];
        markets = copy(SC_MARKETS),
        threaded = true,
    )
end

function sc_calibration_tables(fit, enriched_odds::DataFrame, ds::Data.DataStore,
                               model::AbstractString)
    context = sc_evaluation_context(fit, enriched_odds, ds)
    alignment = SC_EV.verify_alignment(context)
    rows = SC_EV.evaluation_rows(context)
    summary_rows = NamedTuple[]
    curve_rows = NamedTuple[]

    for key in first.(SC_MARKET_MENU)
        selections = SC_MARKET_SELECTIONS[key]
        scopes = [("ALL", Set(selections));
                  [(String(selection), Set([selection])) for selection in selections]]
        for (scope, wanted) in scopes
            selected = [row for row in rows if row.selection in wanted]
            for source in (:model, :market)
                score, n_obs = SC_EV.brier_score(selected; source = source)
                curve = SC_EV.calibration_curve(selected; n_bins = 10, source = source)
                ece = SC_EV.expected_calibration_error(curve)
                predicted = [source === :model ? r.model_prob : r.market_prob for r in selected]
                observed = Float64[r.outcome for r in selected]
                mean_predicted = isempty(predicted) ? NaN : mean(predicted)
                mean_observed = isempty(observed) ? NaN : mean(observed)
                push!(summary_rows, (
                    model = String(model),
                    market_key = key,
                    selection = scope,
                    source = String(source),
                    n_obs,
                    brier = score,
                    ece,
                    mean_predicted,
                    realized_win_rate = mean_observed,
                    calibration_bias = mean_observed - mean_predicted,
                ))
                for bin in 1:length(curve)
                    push!(curve_rows, (
                        model = String(model),
                        market_key = key,
                        selection = scope,
                        source = String(source),
                        bin,
                        lower = curve.edges[bin],
                        upper = curve.edges[bin + 1],
                        n = curve.counts[bin],
                        mean_predicted = curve.mean_predicted[bin],
                        realized_win_rate = curve.observed[bin],
                        calibration_gap = curve.observed[bin] - curve.mean_predicted[bin],
                    ))
                end
            end
        end
    end
    return (;
        summary = DataFrame(summary_rows),
        curves = DataFrame(curve_rows),
        alignment,
    )
end

function sc_policy_summary(result, model::AbstractString, policy_name::AbstractString)
    summary = result.summary
    return (
        model = String(model),
        policy = String(policy_name),
        initial_bankroll = summary.initial_bankroll,
        final_bankroll = summary.final_bankroll,
        total_return_pct = summary.total_return_pct,
        annual_sharpe = summary.sharpe_ann,
        max_drawdown_pct = summary.mdd,
        n_bets = summary.n_bets,
        total_turnover = summary.total_stake,
        flat_roi_pct = summary.roi,
        win_rate_pct = 100 * summary.win_rate,
        mean_exposure = summary.mean_exposure,
        max_exposure = summary.max_exposure,
        n_capped = summary.n_capped,
        n_slates = summary.n_slates,
        mean_k_risk = summary.mean_k_risk,
    )
end

function sc_policy_daily_frame(result, model::AbstractString, policy_name::AbstractString)
    frame = SC_PF.states_frame(result)
    frame.model = fill(String(model), nrow(frame))
    frame.policy = fill(String(policy_name), nrow(frame))
    peak = accumulate(max, frame.bankroll_open)
    frame.opening_drawdown = frame.bankroll_open ./ peak .- 1.0
    frame.drawdown_trust_multiplier = sc_drawdown_multiplier.(frame.opening_drawdown)
    return frame
end
