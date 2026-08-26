# ==============================================================================
# Model 00 — GATE 7 : GROWTH (Pure Poisson)
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# Assembles src/Portfolio for Model 00 (Pure Poisson):
#   BookSpec -> build_books -> group(DailySlate) -> simulate -> path_metrics
#
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Statistics
using Printf

const Pf = BayesianFootball.Portfolio


# ==============================================================================
# 1. The Odds Table
# ==============================================================================

function tp00_betfair_odds_df(ds, contract::SLContract;
                              ids::Union{Nothing,AbstractSet} = nothing,
                              window = (-20.0, 0.0))
    D   = BayesianFootball.Data
    raw = D.summarize_odds(ds.betfair_odds, D.TWAEstimator(); window = window)
    isempty(raw) && return DataFrame()

    wanted = Set((D.market_group(m), Float64(D.market_line(m))) for m in tp00_book_markets(contract))
    df = filter(r -> (String(r.market_name), Float64(r.market_line)) in wanted, raw)
    ids === nothing || (df = filter(r -> Int(r.match_id) in ids, df))
    "is_sane" in names(df) && (df = filter(r -> coalesce(r.is_sane, true), df))

    return DataFrame(
        match_id    = Int.(df.match_id),
        market_name = String.(df.market_name),
        market_line = Float64.(df.market_line),
        selection   = Symbol.(df.selection),
        odds_close  = Float64.(df.odds),
    )
end

function tp00_bookmaker_odds_df(ds, contract::SLContract; ids::Union{Nothing,AbstractSet} = nothing)
    D = BayesianFootball.Data
    wanted = Set((D.market_group(m), Float64(D.market_line(m))) for m in tp00_book_markets(contract))
    df = filter(r -> (String(r.market_name), Float64(r.market_line)) in wanted, ds.odds)
    ids === nothing || (df = filter(r -> Int(r.match_id) in ids, df))
    return DataFrame(
        match_id    = Int.(df.match_id),
        market_name = String.(df.market_name),
        market_line = Float64.(df.market_line),
        selection   = Symbol.(df.selection),
        odds_close  = Float64.(df.odds_close),
    )
end


# ==============================================================================
# 2. Specs
# ==============================================================================

function tp00_book_spec(contract::SLContract;
                        price     = Pf.DeArb(),
                        allocator = Pf.KellyLogUtility(),
                        shrink    = Pf.BakerMcHale())
    return Pf.BookSpec(
        markets   = BayesianFootball.Data.Markets.MarketConfig(tp00_book_markets(contract)),
        price     = price,
        allocator = allocator,
        shrink    = shrink,
        exec      = Pf.ExecutionConfig(
            commission               = Pf.PerBetCommission(contract.commission),
            max_selection_stake      = contract.max_selection_stake,
            require_complete_markets = true,
        ),
    )
end

function tp00_keep_groups(contract::SLContract, groups::Vector{String})
    D = BayesianFootball.Data
    keys = Set{Tuple{String,Float64,Symbol}}()
    for m in tp00_book_markets(contract)
        g = D.market_group(m)
        g in groups || continue
        for (_, sel) in pairs(D.outcomes(m))
            push!(keys, (g, Float64(D.market_line(m)), sel))
        end
    end
    return Pf.MarketWhitelist(keys)
end

function tp00_growth_policies(contract::SLContract;
                              lambda::Float64 = contract.drawdown_lambda,
                              trust::Float64  = contract.trust_w)
    cap  = Pf.FixedCap(contract.portfolio_kelly_cap)
    risk = Pf.SlateDrawdown(lambda)
    base(f) = Pf.PolicySpec(trust = Pf.FlatTrust(trust), risk = risk, cap = cap,
                            filter = f, grouping = Pf.DailySlate())

    return [
        (name = "full book",     policy = base(Pf.KeepAll())),
        (name = "totals only",   policy = base(tp00_keep_groups(contract, ["OverUnder"]))),
        (name = "totals + BTTS", policy = base(tp00_keep_groups(contract, ["OverUnder", "BTTS"]))),
        (name = "1X2 only",      policy = base(tp00_keep_groups(contract, ["1X2"]))),
    ]
end


# ==============================================================================
# 3. GATE 7a — Book Construction
# ==============================================================================

function tp00_gate_books(books, latents_df::AbstractDataFrame, odds_df::AbstractDataFrame;
                         min_coverage::Float64 = 0.50)
    out = Any[]
    n_want = nrow(latents_df)
    n_have = length(books)
    priced = length(unique(odds_df.match_id))

    cov = n_want == 0 ? 0.0 : n_have / n_want
    push!(out, (
        name   = "books built",
        pass   = cov >= min_coverage,
        detail = @sprintf("%d of %d fixtures (%.1f%%); %d had usable quotes — threshold %.0f%%",
                          n_have, n_want, 100cov, priced, 100min_coverage),
    ))
    n_have == 0 && return out

    push!(out, (
        name   = "Kelly solved",
        pass   = all(b.converged for b in books),
        detail = "$(count(b -> b.converged, books)) of $n_have converged",
    ))

    betting = [b for b in books if maximum(b.a_kelly) > 0]
    worst_kkt = isempty(betting) ? 0.0 : maximum(b.kkt for b in betting)
    push!(out, (
        name   = "KKT residual (books that allocate)",
        pass   = worst_kkt <= 1e-4,
        detail = @sprintf("max %.3e over %d allocating books (target ~1e-6); %d books stake nothing",
                          worst_kkt, length(betting), n_have - length(betting)),
    ))

    push!(out, (
        name   = "grids normalised",
        pass   = all(abs(sum(b.p_grid) - 1) <= 1e-9 for b in books),
        detail = @sprintf("max |Σp - 1| = %.3e",
                          maximum(abs(sum(b.p_grid) - 1) for b in books)),
    ))

    push!(out, (
        name   = "all books settled",
        pass   = all(Pf.is_settled(b) for b in books),
        detail = "$(count(Pf.is_settled, books)) of $n_have have a result",
    ))

    push!(out, (
        name   = "chronological",
        pass   = issorted(books, by = b -> b.date),
        detail = "$(minimum(b.date for b in books)) → $(maximum(b.date for b in books))",
    ))

    n_sel = sum(length(b.sels) for b in books)
    fams  = unique(s.group for b in books for s in b.sels)
    push!(out, (
        name   = "selections available",
        pass   = n_sel > 0,
        detail = @sprintf("%d selections, %.1f per fixture, groups: %s",
                          n_sel, n_sel / n_have, join(sort(fams), ", ")),
    ))

    return out
end


# ==============================================================================
# 4. GATE 7b — Simulation Integrity
# ==============================================================================

function tp00_gate_simulation(traj, slates, contract::SLContract)
    out = Any[]

    push!(out, (
        name   = "bankroll never non-positive",
        pass   = all(>(0), traj.bankroll),
        detail = @sprintf("min bankroll %.4f over %d slates", minimum(traj.bankroll), length(traj.slate_pl)),
    ))

    push!(out, (
        name   = "slates chronological",
        pass   = issorted(slates, by = s -> s.window),
        detail = "$(length(slates)) slates, $(slates[1].window) → $(slates[end].window)",
    ))

    mx = isempty(traj.exposure) ? 0.0 : maximum(traj.exposure)
    push!(out, (
        name   = "exposure within cap",
        pass   = mx <= contract.portfolio_kelly_cap + 1e-9,
        detail = @sprintf("max slate exposure %.4f, cap %.2f, %d slates capped",
                          mx, contract.portfolio_kelly_cap, traj.n_capped),
    ))

    push!(out, (
        name   = "no slate loses the bankroll",
        pass   = isempty(traj.slate_pl) || minimum(traj.slate_pl) > -1.0,
        detail = @sprintf("worst slate %+.4f", isempty(traj.slate_pl) ? 0.0 : minimum(traj.slate_pl)),
    ))

    push!(out, (
        name   = "bets recorded",
        pass   = nrow(traj.bets) > 0,
        detail = @sprintf("%d bets, %.2f staked in total", nrow(traj.bets), traj.total_stake),
    ))

    return out
end


# ==============================================================================
# 5. GATE 7c — Results & Concentration
# ==============================================================================

function tp00_pnl_concentration(traj)
    pnl = sort(traj.bets.pnl; rev = true)
    tot = sum(pnl)
    share(k) = (tot == 0 || length(pnl) < k) ? NaN : 100 * sum(pnl[1:k]) / tot
    return (top1 = share(1), top5 = share(5), top10 = share(10), top20 = share(20),
            win_rate = mean(traj.bets.pnl .> 0))
end

function tp00_growth_table(books::AbstractVector, contract::SLContract;
                          lambda::Float64 = contract.drawdown_lambda,
                          trust::Float64  = contract.trust_w,
                          B::Int = 4000)
    rows = NamedTuple[]
    for (name, policy) in tp00_growth_policies(contract; lambda = lambda, trust = trust)
        slates = Pf.group(policy.grouping, books)
        traj   = Pf.simulate(policy, slates)
        m      = Pf.path_metrics(traj)
        ci     = nrow(traj.bets) > 0 ? Pf.bootstrap_roi(traj.bets; B = B) : (lo = NaN, hi = NaN, sd = NaN)

        conc = tp00_pnl_concentration(traj)
        push!(rows, (
            policy      = name,
            n_bets      = m.n_bets,
            final       = round(m.final, digits = 3),
            roi_pct     = round(m.roi, digits = 2),
            roi_lo      = round(ci[1], digits = 2),
            roi_hi      = round(ci[2], digits = 2),
            top10_pct   = round(conc.top10, digits = 1),
            win_rate    = round(conc.win_rate, digits = 3),
            growth      = round(m.growth_per_slate, digits = 5),
            mdd_pct     = round(m.mdd, digits = 1),
            mean_expo   = round(m.mean_exposure, digits = 4),
        ))
    end
    return DataFrame(rows)
end

function tp00_growth_table(books_bf::Vector{BayesianFootball.Portfolio.MatchBook}, contract::SLContract)
    return tp00_growth_table(books_bf, contract; lambda = contract.drawdown_lambda, trust = contract.trust_w)
end

function tp00_gate_growth(growth::AbstractDataFrame)
    out = Any[]
    full = growth[findfirst(==("full book"), growth.policy), :]

    push!(out, (
        name   = "every policy produced bets",
        pass   = all(growth.n_bets .> 0),
        detail = join(["$(r.policy): $(r.n_bets)" for r in eachrow(growth)], ", "),
    ))

    push!(out, (
        name   = "bankroll survived everywhere",
        pass   = all(growth.final .> 0),
        detail = @sprintf("final wealth %.3f to %.3f", minimum(growth.final), maximum(growth.final)),
    ))

    sig = filter(r -> isfinite(r.roi_lo) && (r.roi_lo > 0 || r.roi_hi < 0), growth)
    push!(out, (
        name   = "ROI significance (finding, not a criterion)",
        pass   = true,
        detail = isempty(sig) ? "no policy's ROI interval excludes zero — expected at n = $(full.n_bets)" :
                 join(["$(r.policy) $(r.roi_pct)% [$(r.roi_lo), $(r.roi_hi)]" for r in eachrow(sig)], "; "),
    ))

    worst_conc = maximum(filter(isfinite, growth.top10_pct))
    push!(out, (
        name   = "P&L concentration (read before the ROI)",
        pass   = true,
        detail = @sprintf("top 10 bets are %.0f%%-%.0f%% of total P&L across policies; %s",
                          minimum(filter(isfinite, growth.top10_pct)), worst_conc,
                          worst_conc >= 100 ?
                            "AT OR ABOVE 100% — the profit is a handful of bets, not an edge" :
                            "distributed enough to be worth interpreting"),
    ))

    push!(out, (
        name   = "full book reported alongside curation",
        pass   = "full book" in growth.policy,
        detail = @sprintf("full book: %.2f%% ROI, final %.3f, %d bets",
                          full.roi_pct, full.final, full.n_bets),
    ))

    return out
end


# ==============================================================================
# 6. Policy Sweeps
# ==============================================================================

function tp00_sweep_policy(books, contract::SLContract;
                          trusts  = [0.15, 0.3, 0.5, 1.0],
                          lambdas = [15.0, 23.0, 35.0],
                          policy_name::String = "full book",
                          B::Int = 1500)
    rows = NamedTuple[]
    for λ in lambdas, w in trusts
        ps = tp00_growth_policies(contract; lambda = λ, trust = w)
        i  = findfirst(p -> p.name == policy_name, ps)
        i === nothing && error("unknown policy $policy_name")
        pol    = ps[i].policy
        slates = Pf.group(pol.grouping, books)
        traj   = Pf.simulate(pol, slates)
        m      = Pf.path_metrics(traj)
        conc   = tp00_pnl_concentration(traj)
        ci     = nrow(traj.bets) > 0 ? Pf.bootstrap_roi(traj.bets; B = B) : (NaN, NaN, NaN)
        push!(rows, (
            λ = λ, trust = w, n_bets = m.n_bets,
            final = round(m.final, digits = 3),
            roi_pct = round(m.roi, digits = 2),
            roi_lo = round(ci[1], digits = 2), roi_hi = round(ci[2], digits = 2),
            top10_pct = round(conc.top10, digits = 1),
            mean_expo = round(m.mean_exposure, digits = 4),
            mdd_pct = round(m.mdd, digits = 1),
        ))
    end
    return DataFrame(rows)
end
