# ==============================================================================
# Gate 7 — Growth
# ==============================================================================
#
# Generic book construction and portfolio diagnostics. This layer consumes score
# matrices and closing odds only; it contains no model or likelihood logic.
#
# BookSpec -> build_books -> group(DailySlate) -> simulate -> path_metrics
#
# `DeArb` prevents an apparent arbitrage in recorded quotes from becoming a
# backtest profit. Complete markets are required so partial quote sets cannot
# create degenerate implied probabilities. Settlement is performed from the real
# score through the portfolio package, not a pre-computed winner column.

"""
    sl_betfair_odds_df(ds, contract; ids, window) -> DataFrame

Return Betfair time-weighted closing prices in the schema expected by portfolio
book construction. Only contract markets and optionally requested fixtures remain.
"""
function sl_betfair_odds_df(ds, contract::SLContract;
                            ids::Union{Nothing,AbstractSet} = nothing,
                            window = (-20.0, 0.0))
    raw = SLData.summarize_odds(ds.betfair_odds, SLData.TWAEstimator(); window = window)
    isempty(raw) && return DataFrame()

    wanted = Set(
        (SLData.market_group(m), Float64(SLData.market_line(m)))
        for m in sl_book_markets(contract)
    )
    df = filter(r -> (String(r.market_name), Float64(r.market_line)) in wanted, raw)
    ids === nothing || (df = filter(r -> Int(r.match_id) in ids, df))
    if "is_sane" in names(df)
        df = filter(r -> coalesce(r.is_sane, true), df)
    end

    return DataFrame(
        match_id = Int.(df.match_id),
        market_name = String.(df.market_name),
        market_line = Float64.(df.market_line),
        selection = Symbol.(df.selection),
        odds_close = Float64.(df.odds),
    )
end

"""
    sl_bookmaker_odds_df(ds, contract; ids) -> DataFrame

Return bookmaker closing prices in the same schema as exchange closing prices.
This is useful for coverage comparison; execution remains governed by the selected
book specification.
"""
function sl_bookmaker_odds_df(ds, contract::SLContract;
                              ids::Union{Nothing,AbstractSet} = nothing)
    wanted = Set(
        (SLData.market_group(m), Float64(SLData.market_line(m)))
        for m in sl_book_markets(contract)
    )
    df = filter(r -> (String(r.market_name), Float64(r.market_line)) in wanted, ds.odds)
    ids === nothing || (df = filter(r -> Int(r.match_id) in ids, df))

    return DataFrame(
        match_id = Int.(df.match_id),
        market_name = String.(df.market_name),
        market_line = Float64.(df.market_line),
        selection = Symbol.(df.selection),
        odds_close = Float64.(df.odds_close),
    )
end

"""
    sl_book_spec(contract; price, allocator, shrink) -> BookSpec

Declare how contract markets are priced, allocated, shrunk for posterior
uncertainty, and executed. Complete markets are mandatory.
"""
function sl_book_spec(contract::SLContract;
                      price = SLPortfolio.DeArb(),
                      allocator = SLPortfolio.KellyLogUtility(),
                      shrink = SLPortfolio.BakerMcHale())
    return SLPortfolio.BookSpec(
        markets = SLData.Markets.MarketConfig(sl_book_markets(contract)),
        price = price,
        allocator = allocator,
        shrink = shrink,
        exec = SLPortfolio.ExecutionConfig(
            commission = SLPortfolio.PerBetCommission(contract.commission),
            max_selection_stake = contract.max_selection_stake,
            require_complete_markets = true,
        ),
    )
end

"""
    sl_keep_groups(contract, groups) -> MarketWhitelist

Whitelist complete contract market groups using stable `(group, line, selection)`
keys, rather than whatever selections happen to be quoted on a fixture.
"""
function sl_keep_groups(contract::SLContract, groups::Vector{String})
    keys = Set{Tuple{String,Float64,Symbol}}()
    for market in sl_book_markets(contract)
        group = SLData.market_group(market)
        group in groups || continue
        for (_, selection) in pairs(SLData.outcomes(market))
            push!(keys, (group, Float64(SLData.market_line(market)), selection))
        end
    end
    return SLPortfolio.MarketWhitelist(keys)
end

"""
    sl_growth_policies(contract; lambda, trust) -> Vector{NamedTuple}

Return pre-declared full-book and whole-market-group curation policies. All use a
shared daily slate grouping, drawdown control, and simultaneous-exposure cap.
"""
function sl_growth_policies(contract::SLContract;
                            lambda::Float64 = contract.drawdown_lambda,
                            trust::Float64 = contract.trust_w)
    cap = SLPortfolio.FixedCap(contract.portfolio_kelly_cap)
    risk = SLPortfolio.SlateDrawdown(lambda)
    base(filter) = SLPortfolio.PolicySpec(
        trust = SLPortfolio.FlatTrust(trust),
        risk = risk,
        cap = cap,
        filter = filter,
        grouping = SLPortfolio.DailySlate(),
    )

    return [
        (name = "full book", policy = base(SLPortfolio.KeepAll())),
        (name = "totals only", policy = base(sl_keep_groups(contract, ["OverUnder"]))),
        (name = "totals + BTTS", policy = base(sl_keep_groups(contract, ["OverUnder", "BTTS"]))),
        (name = "1X2 only", policy = base(sl_keep_groups(contract, ["1X2"]))),
    ]
end

"""
    sl_gate_books(books, latents_df, odds_df; label, min_coverage) -> Vector

Check coverage, allocation convergence, KKT residuals, probability grids,
settlement, chronology, and selection availability. KKT is assessed only for books
that allocate a positive stake: a no-bet boundary optimum has no consequential KKT
residual.
"""
function sl_gate_books(books, latents_df::AbstractDataFrame, odds_df::AbstractDataFrame;
                       label::String = "", min_coverage::Float64 = 0.50)
    out = Any[]
    n_want = nrow(latents_df)
    n_have = length(books)
    priced = :match_id in propertynames(odds_df) ? length(unique(odds_df.match_id)) : 0
    coverage = n_want == 0 ? 0.0 : n_have / n_want
    push!(out, sl_result(
        "books built",
        coverage >= min_coverage,
        @sprintf("%d of %d fixtures (%.1f%%); %d had usable quotes — threshold %.0f%%",
                 n_have, n_want, 100 * coverage, priced, 100 * min_coverage),
    ))
    n_have == 0 && return out

    push!(out, sl_result(
        "Kelly solved",
        all(book.converged for book in books),
        "$(count(book -> book.converged, books)) of $n_have converged",
    ))

    betting = [book for book in books if maximum(book.a_kelly) > 0]
    worst_kkt = isempty(betting) ? 0.0 : maximum(book.kkt for book in betting)
    push!(out, sl_result(
        "KKT residual (books that allocate)",
        worst_kkt <= 1e-3,
        @sprintf("max %.3e over %d allocating books (solver tol <= 1e-3); %d books stake nothing",
                 worst_kkt, length(betting), n_have - length(betting)),
    ))

    max_grid_error = maximum(abs(sum(book.p_grid) - 1) for book in books)
    push!(out, sl_result(
        "grids normalised",
        max_grid_error <= 1e-9,
        @sprintf("max |Σp - 1| = %.3e", max_grid_error),
    ))

    push!(out, sl_result(
        "all books settled",
        all(SLPortfolio.is_settled, books),
        "$(count(SLPortfolio.is_settled, books)) of $n_have have a result",
    ))

    push!(out, sl_result(
        "chronological",
        issorted(books, by = book -> book.date),
        "$(minimum(book.date for book in books)) → $(maximum(book.date for book in books))",
    ))

    n_selections = sum(length(book.sels) for book in books)
    groups = unique(selection.group for book in books for selection in book.sels)
    push!(out, sl_result(
        "selections available",
        n_selections > 0,
        @sprintf("%d selections, %.1f per fixture, groups: %s",
                 n_selections, n_selections / n_have, join(sort(groups), ", ")),
    ))

    return out
end

"""
    sl_gate_simulation(traj, slates, contract) -> Vector

Verify bankroll survival, chronological slate processing, exposure cap, worst slate
loss, and recorded betting activity.
"""
function sl_gate_simulation(traj, slates, contract::SLContract)
    out = Any[]
    bankroll_ok = !isempty(traj.bankroll) && all(>(0), traj.bankroll)
    min_bankroll = isempty(traj.bankroll) ? NaN : minimum(traj.bankroll)
    push!(out, sl_result(
        "bankroll never non-positive",
        bankroll_ok,
        @sprintf("min bankroll %.4f over %d slates", min_bankroll, length(traj.slate_pl)),
    ))

    chronological = !isempty(slates) && issorted(slates, by = slate -> slate.window)
    slate_detail = isempty(slates) ? "no slates" : "$(length(slates)) slates, $(slates[1].window) → $(slates[end].window)"
    push!(out, sl_result("slates chronological", chronological, slate_detail))

    max_exposure = isempty(traj.exposure) ? 0.0 : maximum(traj.exposure)
    push!(out, sl_result(
        "exposure within cap",
        max_exposure <= contract.portfolio_kelly_cap + 1e-9,
        @sprintf("max slate exposure %.4f, cap %.2f, %d slates capped",
                 max_exposure, contract.portfolio_kelly_cap, traj.n_capped),
    ))

    worst_slate = isempty(traj.slate_pl) ? NaN : minimum(traj.slate_pl)
    push!(out, sl_result(
        "no slate loses the bankroll",
        !isempty(traj.slate_pl) && worst_slate > -1.0,
        @sprintf("worst slate %+.4f", worst_slate),
    ))

    push!(out, sl_result(
        "bets recorded",
        nrow(traj.bets) > 0,
        @sprintf("%d bets, %.2f staked in total", nrow(traj.bets), traj.total_stake),
    ))
    return out
end

"""
    sl_pnl_concentration(traj) -> NamedTuple

Report shares of total P&L contributed by the best 1, 5, 10, and 20 bets, together
with the winning-bet rate. Empty bet sets return missing concentration values.
"""
function sl_pnl_concentration(traj)
    isempty(traj.bets.pnl) && return (
        top1 = NaN,
        top5 = NaN,
        top10 = NaN,
        top20 = NaN,
        win_rate = NaN,
    )

    pnl = sort(traj.bets.pnl; rev = true)
    total = sum(pnl)
    share(n) = (total == 0 || length(pnl) < n) ? NaN : 100 * sum(pnl[1:n]) / total
    return (
        top1 = share(1),
        top5 = share(5),
        top10 = share(10),
        top20 = share(20),
        win_rate = mean(traj.bets.pnl .> 0),
    )
end

"""
    sl_growth_table(books, contract; lambda, trust, B) -> DataFrame

Simulate every declared policy over the same books. ROI intervals are bootstrapped
by match by the portfolio implementation, preserving within-fixture dependence.
"""
function sl_growth_table(books, contract::SLContract;
                         lambda::Float64 = contract.drawdown_lambda,
                         trust::Float64 = contract.trust_w,
                         B::Int = 4000)
    rows = NamedTuple[]
    for (name, policy) in sl_growth_policies(contract; lambda = lambda, trust = trust)
        slates = SLPortfolio.group(policy.grouping, books)
        traj = SLPortfolio.simulate(policy, slates)
        metrics = SLPortfolio.path_metrics(traj)
        interval = nrow(traj.bets) > 0 ?
                   SLPortfolio.bootstrap_roi(traj.bets; B = B) : (NaN, NaN, NaN)
        concentration = sl_pnl_concentration(traj)

        push!(rows, (
            policy = name,
            n_bets = metrics.n_bets,
            final = round(metrics.final, digits = 3),
            roi_pct = round(metrics.roi, digits = 2),
            roi_lo = round(interval[1], digits = 2),
            roi_hi = round(interval[2], digits = 2),
            top10_pct = round(concentration.top10, digits = 1),
            win_rate = round(concentration.win_rate, digits = 3),
            growth = round(metrics.growth_per_slate, digits = 5),
            mdd_pct = round(metrics.mdd, digits = 1),
            mean_expo = round(metrics.mean_exposure, digits = 4),
        ))
    end
    return DataFrame(rows)
end

"""
    sl_gate_growth(growth) -> Vector

Report policy coverage, bankroll survival, interval findings, P&L concentration,
and whether the mandatory full-book comparator is present.
"""
function sl_gate_growth(growth::AbstractDataFrame)
    isempty(growth) && return [sl_result("growth table", false, "empty")]

    out = Any[]
    full_index = findfirst(==("full book"), growth.policy)
    full = full_index === nothing ? nothing : growth[full_index, :]

    push!(out, sl_result(
        "every policy produced bets",
        all(growth.n_bets .> 0),
        join(["$(row.policy): $(row.n_bets)" for row in eachrow(growth)], ", "),
    ))
    push!(out, sl_result(
        "bankroll survived everywhere",
        all(growth.final .> 0),
        @sprintf("final wealth %.3f to %.3f", minimum(growth.final), maximum(growth.final)),
    ))

    significant = filter(row -> isfinite(row.roi_lo) && (row.roi_lo > 0 || row.roi_hi < 0), growth)
    interval_detail = isempty(significant) ?
        "no policy's ROI interval excludes zero" :
        join(["$(row.policy) $(row.roi_pct)% [$(row.roi_lo), $(row.roi_hi)]" for row in eachrow(significant)], "; ")
    push!(out, sl_result("ROI significance (finding, not a criterion)", true, interval_detail))

    finite_concentration = filter(isfinite, growth.top10_pct)
    concentration_detail = isempty(finite_concentration) ?
        "no finite top-ten P&L concentration was available" :
        @sprintf("top 10 bets are %.0f%%-%.0f%% of total P&L across policies%s",
                 minimum(finite_concentration), maximum(finite_concentration),
                 maximum(finite_concentration) >= 100 ? "; at or above 100% is concentrated profit" : "")
    push!(out, sl_result("P&L concentration (read before the ROI)", true, concentration_detail))

    full_detail = full === nothing ?
        "full book policy missing" :
        @sprintf("full book: %.2f%% ROI, final %.3f, %d bets", full.roi_pct, full.final, full.n_bets)
    push!(out, sl_result("full book reported alongside curation", full !== nothing, full_detail))
    return out
end

"""
    sl_sweep_policy(books, contract; trusts, lambdas, policy_name, B) -> DataFrame

Sweep trust and drawdown λ for one declared curation while reusing built books.
"""
function sl_sweep_policy(books, contract::SLContract;
                         trusts = [0.15, 0.3, 0.5, 1.0],
                         lambdas = [15.0, 23.0, 35.0],
                         policy_name::String = "full book",
                         B::Int = 1500)
    rows = NamedTuple[]
    for λ in lambdas, trust in trusts
        policies = sl_growth_policies(contract; lambda = λ, trust = trust)
        index = findfirst(policy -> policy.name == policy_name, policies)
        index === nothing && error("unknown policy $policy_name")
        policy = policies[index].policy
        traj = SLPortfolio.simulate(policy, SLPortfolio.group(policy.grouping, books))
        metrics = SLPortfolio.path_metrics(traj)
        concentration = sl_pnl_concentration(traj)
        interval = nrow(traj.bets) > 0 ?
                   SLPortfolio.bootstrap_roi(traj.bets; B = B) : (NaN, NaN, NaN)

        push!(rows, (
            λ = λ,
            trust = trust,
            n_bets = metrics.n_bets,
            final = round(metrics.final, digits = 3),
            roi_pct = round(metrics.roi, digits = 2),
            roi_lo = round(interval[1], digits = 2),
            roi_hi = round(interval[2], digits = 2),
            top10_pct = round(concentration.top10, digits = 1),
            mean_expo = round(metrics.mean_exposure, digits = 4),
            mdd_pct = round(metrics.mdd, digits = 1),
        ))
    end
    return DataFrame(rows)
end

"""
    sl_sweep_book(ds, contract, latents_df, expr, odds_df; variants, policy_name) -> DataFrame

Rebuild books for each pricing variant, then simulate one declared policy. Unlike a
policy sweep, a book-spec sweep cannot reuse existing books.
"""
function sl_sweep_book(ds, contract::SLContract, latents_df, expr, odds_df;
                       variants = [
                           ("DeArb (default)", SLPortfolio.DeArb()),
                           ("RawPrice", SLPortfolio.RawPrice()),
                           ("Normalise (ABLATION — manufactures edge)", SLPortfolio.Normalise()),
                       ],
                       policy_name::String = "full book")
    rows = NamedTuple[]
    for (name, price) in variants
        spec = sl_book_spec(contract; price = price)
        books = SLPortfolio.build_books(spec, latents_df, expr, odds_df, ds)
        policies = sl_growth_policies(contract)
        index = findfirst(policy -> policy.name == policy_name, policies)
        index === nothing && error("unknown policy $policy_name")
        policy = policies[index].policy
        traj = SLPortfolio.simulate(policy, SLPortfolio.group(policy.grouping, books))
        metrics = SLPortfolio.path_metrics(traj)
        concentration = sl_pnl_concentration(traj)

        push!(rows, (
            price = name,
            n_books = length(books),
            n_bets = metrics.n_bets,
            final = round(metrics.final, digits = 3),
            roi_pct = round(metrics.roi, digits = 2),
            top10_pct = round(concentration.top10, digits = 1),
            mean_expo = round(metrics.mean_exposure, digits = 4),
        ))
    end
    return DataFrame(rows)
end
