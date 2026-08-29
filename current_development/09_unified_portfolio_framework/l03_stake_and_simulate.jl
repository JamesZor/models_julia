# ==============================================================================
# 09 — UNIFIED PORTFOLIO & STAKING FRAMEWORK : STAKING AND SIMULATION
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# Stages B, C and D:
#
#     Vector{MatchBook} ──group──> Vector{Slate} ──simulate_portfolio──> PortfolioResult
#
# ------------------------------------------------------------------------------
# WHAT IS REUSED, WHAT IS NEW, AND WHY THE LINE IS WHERE IT IS
# ------------------------------------------------------------------------------
#
# REUSED VERBATIM from `src/Portfolio/`, as aliases:
#
#   group / build_slates   the settlement-window partition
#   stake_slate            a_kelly → ×trust → ×shrink → ×risk → cap → filter
#   allocate               Jacot & Mochkovitch's non-mutually-exclusive Kelly solve
#   risk_factor            Busseti-Ryu-Boyd's drawdown budget
#   simulate               the legacy trajectory, so the legacy return type is IDENTICAL
#   path_metrics / bootstrap_roi / report / attribution / calibrate_*
#
# These are correct, they are covered by `test/portfolio_tests.jl`, and none of them
# is on the path this framework speeds up. The cost being removed is upstream, in the
# book builder. A second Kelly solver whose only job is to agree with the first one
# to the last bit is a liability.
#
# NEW here:
#
#   simulate_portfolio     the same forward walk, emitting `DailyState` rows and the
#                          metric set the briefing asks for
#   portfolio_summary      CAGR, Sharpe, Sortino, win rate, 1X2 ROI — none of which
#                          `Portfolio.path_metrics` has a field for
#   bootstrap_portfolio    match-clustered ROI (bit-identical to `bootstrap_roi`) plus
#                          a slate-blocked interval on log growth
#   stake_sheet            routed onto the typed-container fast path, and gated
#   run_portfolio_simulation   build → group → simulate → report, in one call
#
# `simulate_portfolio` DOES NOT call `simulate` internally, even though it produces
# the same `Trajectory`. If it did, the parity check in `l05_parity.jl` §5 would be
# checking that a function agrees with itself. It runs its own forward walk and is
# then required to agree with `src`'s, field for field.
#
# ------------------------------------------------------------------------------
# THE ONE THING TO KNOW BEFORE READING A NUMBER OUT OF THIS FILE
# ------------------------------------------------------------------------------
#
# `risk_factor` is HOMOGENEOUS OF DEGREE 0 in the stakes it is handed
# (`src/Portfolio/stake.jl:10-17`). It solves for the factor that makes those stakes
# satisfy the drawdown constraint, so handing it twice the stakes returns half the
# factor and `k .* stakes` is unchanged. Once the constraint binds, trust and
# shrinkage can only RESHAPE a book — they cannot resize it. Measured on
# ScottishLower: at λ = 20 a stake multiplier of 0.25, 1.0 or 4.0 all give mean slate
# exposure 0.1088, and trust of 0.25, 0.5 and 1.0 give identical final wealth.
#
# To move exposure, move λ (`calibrate_lambda`). Not trust, and not `scale`.
#
# ==============================================================================

include(joinpath(@__DIR__, "l02_book_builder.jl"))


# ==============================================================================
# 1. SLATES AND STAKING — ALIASED
# ==============================================================================

const build_slates    = UP_PF.build_slates
const stake_slate     = UP_PF.stake_slate
const simulate        = UP_PF.simulate
const path_metrics    = UP_PF.path_metrics
const bootstrap_roi   = UP_PF.bootstrap_roi
const report          = UP_PF.report
const attribution     = UP_PF.attribution
const slate_summary   = UP_PF.slate_summary
const calibrate_lambda = UP_PF.calibrate_lambda
const calibrate_scale  = UP_PF.calibrate_scale


# ==============================================================================
# 2. THE FORWARD WALK
# ==============================================================================

"""
    simulate_portfolio(policy, books; initial_bankroll = 1000.0, …) -> PortfolioResult

Chronological, simultaneous same-slate settlement, one compounding step per slate.

Groups `books` with `policy.grouping`, then walks forward: size the slate against the
bankroll it opened with, settle every bet in it against one scoreline per fixture,
compound once.

# Arguments

* `initial_bankroll` — a REPORTING SCALE ONLY. Stakes are fractions of bankroll and
  P&L is a fraction of bankroll, so the trajectory at `1_000` is the trajectory at
  `1.0` multiplied by 1,000. `SlateContext.bankroll` is handed the FRACTION, exactly
  as `Portfolio.simulate` hands it, so a bankroll-dependent trust or filter sees the
  same number under both simulators and the two cannot diverge.
* `use_shrink` — apply each book's `k_shrink`. `false` is the ablation.
* `scale` — a global stake multiplier applied before the risk step. Very nearly a
  no-op whenever the drawdown constraint is active; see the file header.
* `bootstrap` / `B` / `seed` — the match-clustered interval. `B = 4000` is `src`'s
  default and takes a moment on a large book; `bootstrap = false` skips it.
* `metrics` — any `BackTesting.AbstractWealthMetric`s to evaluate on the slate-level
  bankroll curve.
* `converged` / `failed_gates` — carried into the result. `simulate_portfolio(policy,
  books, report::BuildReport; …)` fills them from the build.

# Refusals

Two, both `src`'s and both kept as hard assertions rather than warnings:

* every book must be SETTLED. You cannot backtest an unplayed fixture; use
  `stake_sheet` for those.
* a slate cannot lose more than the bankroll. Guaranteed by `FixedCap`'s `(0,1)`
  constraint, asserted anyway so a future cap implementation cannot quietly break it.
  The prototype this replaced had no cap: its worst slate lost 129.5% of bankroll and
  the simulated wealth reached `-0.697`, after which the sign flipped on every
  subsequent compounding step.
"""
function simulate_portfolio(policy::PolicySpec, books::Vector{MatchBook};
                            initial_bankroll::Real = 1000.0,
                            use_shrink::Bool = true, scale::Real = 1.0,
                            bootstrap::Bool = true, B::Int = 4000, seed::Int = 1,
                            metrics::AbstractVector = [],
                            converged::Union{Nothing, Bool} = nothing,
                            failed_gates::Vector{String} = String[])
    slates = group(policy.grouping, books)
    return simulate_portfolio(policy, slates; initial_bankroll = initial_bankroll,
                              use_shrink = use_shrink, scale = scale,
                              bootstrap = bootstrap, B = B, seed = seed,
                              metrics = metrics, converged = converged,
                              failed_gates = failed_gates)
end

function simulate_portfolio(policy::PolicySpec, books::Vector{MatchBook},
                            br::BuildReport; kw...)
    return simulate_portfolio(policy, books; converged = br.converged,
                              failed_gates = br.failed_gates, kw...)
end

function simulate_portfolio(policy::PolicySpec, slates::Vector{Slate};
                            initial_bankroll::Real = 1000.0,
                            use_shrink::Bool = true, scale::Real = 1.0,
                            bootstrap::Bool = true, B::Int = 4000, seed::Int = 1,
                            metrics::AbstractVector = [],
                            converged::Union{Nothing, Bool} = nothing,
                            failed_gates::Vector{String} = String[])
    @assert issorted(slates, by = s -> s.window) "slates must be chronological"
    @assert all(is_settled(b) for sl in slates for b in sl.books) """
        simulate_portfolio needs settled books: at least one fixture has no result.
        Build with require_result = true (the default) for a backtest; unsettled books
        are for stake_sheet only."""

    B0 = Float64(initial_bankroll)
    bank = 1.0                     # FRACTION of B0 — see the docstring
    hist     = Float64[1.0]
    dates    = Date[]
    slate_pl = Float64[]
    ks       = Float64[]
    expo     = Float64[]
    n_capped = 0
    tot_stake = 0.0
    tot_pl    = 0.0
    stake_1x2 = 0.0
    pl_1x2    = 0.0
    n_wins    = 0
    rows   = NamedTuple[]
    states = DailyState[]

    for (t, sl) in enumerate(slates)
        ctx   = SlateContext(t, sl.window, bank)
        alloc = stake_slate(policy, sl, ctx; use_shrink = use_shrink,
                            scale = Float64(scale))
        push!(ks, alloc.k_risk); push!(expo, alloc.exposure)
        alloc.capped && (n_capped += 1)

        pl, stk, n_bets = 0.0, 0.0, 0
        for (i, b) in enumerate(sl.books), j in eachindex(b.sels)
            s = alloc.stakes[i][j]
            s > 0 || continue
            r = s * b.settle[j]
            stk += s; pl += r; n_bets += 1
            b.settle[j] > 0 && (n_wins += 1)
            if b.sels[j].group == "1X2"
                stake_1x2 += s; pl_1x2 += r
            end
            push!(rows, (match_id = b.m_id, date = b.date, family = b.sels[j].family,
                         selection = b.sels[j].selection, odds = b.sels[j].odds_used,
                         stake = s, pnl = r, payoff = b.settle[j],
                         p_model = b.sels[j].p_model, p_market = b.sels[j].p_market))
        end

        @assert pl > -1.0 "slate $(sl.window) lost more than the bankroll (pl = $pl); " *
                          "the exposure cap is not doing its job"
        open_bank = bank
        bank *= (1.0 + pl)
        push!(hist, bank); push!(dates, sl.window); push!(slate_pl, pl)
        tot_stake += stk; tot_pl += pl

        push!(states, DailyState(t, sl.window, length(sl.books), n_bets,
                                 B0 * open_bank, B0 * bank, stk, pl,
                                 alloc.exposure, alloc.k_risk, alloc.capped))
    end

    bets = isempty(rows) ? _up_empty_bets() : DataFrame(rows)
    traj = Trajectory(hist, dates, slate_pl, ks, expo, n_capped, tot_stake, tot_pl, bets)

    summary = portfolio_summary(states, traj, B0; stake_1x2 = stake_1x2, pl_1x2 = pl_1x2,
                                n_wins = n_wins)
    extra = NamedTuple()
    for m in metrics
        extra = merge(extra, NamedTuple{(Symbol(UP_BT.metric_name(m)),)}(
            (UP_BT.compute_metric(m, traj.bankroll),)))
    end
    ci = bootstrap ? bootstrap_portfolio(traj; B = B, seed = seed) : nothing

    return PortfolioResult(states, summary, extra, ci, traj, attribution(traj),
                           converged, failed_gates)
end

_up_empty_bets() = DataFrame(match_id = Int[], date = Date[], family = String[],
                             selection = Symbol[], odds = Float64[], stake = Float64[],
                             pnl = Float64[], payoff = Float64[],
                             p_model = Float64[], p_market = Float64[])


# ==============================================================================
# 3. THE SUMMARY
# ==============================================================================

"""
    portfolio_summary(states, trajectory, initial_bankroll; …) -> PortfolioSummary

Every number in `Portfolio.path_metrics`, computed the same way and gated against it
in `r01_demo.jl` §8, plus the six the briefing asks for that `src` has no field for.

The five conventions worth stating are on `PortfolioSummary`. The one worth repeating
here is that `sharpe` and `sortino` are on per-slate LOG returns, because a slate is
the compounding unit — a Sharpe ratio computed on flat per-bet P&L answers a question
about a strategy nobody is running.
"""
function portfolio_summary(states::Vector{DailyState}, t::Trajectory,
                           initial_bankroll::Float64;
                           stake_1x2::Float64 = 0.0, pl_1x2::Float64 = 0.0,
                           n_wins::Int = 0)
    bk = t.bankroll
    rm = accumulate(max, bk)
    dd = (bk .- rm) ./ rm .* 100
    ui = max(sqrt(mean(dd .^ 2)), 1e-9)
    tr = (bk[end] - 1.0) * 100
    mdd = minimum(dd)

    n_slates = length(t.slate_pl)
    r = n_slates == 0 ? Float64[] : log.(1.0 .+ t.slate_pl)

    days = length(t.dates) < 2 ? 0 : Dates.value(t.dates[end] - t.dates[1])
    cagr = days > 0 ? bk[end]^(365.25 / days) - 1.0 : NaN
    slates_per_year = days > 0 ? n_slates * 365.25 / days : NaN

    sharpe = length(r) < 2 ? NaN : (std(r) > 0 ? mean(r) / std(r) : NaN)
    sharpe_ann = isnan(sharpe) || isnan(slates_per_year) ? NaN :
                 sharpe * sqrt(slates_per_year)
    down = isempty(r) ? Float64[] : r[r .< 0.0]
    sortino = isempty(r) ? NaN :
              isempty(down) ? Inf : mean(r) / sqrt(mean(down .^ 2))

    n_bets = nrow(t.bets)
    return PortfolioSummary(
        initial_bankroll,
        initial_bankroll * bk[end],
        tr,
        cagr,
        isempty(t.slate_pl) ? 0.0 : mean(r),
        t.total_stake > 0 ? 100 * t.total_pl / t.total_stake : 0.0,
        stake_1x2 > 0 ? 100 * pl_1x2 / stake_1x2 : NaN,
        mdd,
        ui,
        mdd < 0 ? tr / abs(mdd) : 0.0,
        tr / ui,
        sharpe,
        sharpe_ann,
        sortino,
        n_bets > 0 ? n_wins / n_bets : NaN,
        n_slates,
        isempty(states) ? 0 : sum(s -> s.n_fixtures, states),
        n_bets,
        t.total_stake,
        t.total_pl,
        isempty(t.exposure) ? 0.0 : mean(t.exposure),
        isempty(t.exposure) ? 0.0 : maximum(t.exposure),
        isempty(t.slate_pl) ? 0.0 : minimum(t.slate_pl),
        isempty(t.k_risk) ? 1.0 : mean(t.k_risk),
        t.n_capped,
        days,
    )
end


# ==============================================================================
# 4. THE BOOTSTRAP
# ==============================================================================

"""
    bootstrap_portfolio(trajectory; B = 4000, seed = 1) -> BootstrapCI

Two intervals from one seeded stream.

**ROI, resampled BY MATCH.** Eleven selections on one fixture share one scoreline and
are strongly dependent; resampling individual bets would treat them as eleven
independent observations and divide the standard error by roughly `sqrt(11)`. On the
reference ScottishLower book that is the difference between an interval that excludes
zero and one that does not.

This half is BIT-IDENTICAL to `Portfolio.bootstrap_roi(bets; B, seed)` — the same
`MersenneTwister`, the same `rand(rng, 1:n_matches)` calls in the same order, the same
sums. It runs FIRST, so the second statistic cannot perturb it. Gated at 0 ULP in
`r01_demo.jl` §7.

**Mean per-slate log growth, resampled BY SLATE.** Growth is a path quantity: it is
defined by the order the slates compounded in, which a match-level resample destroys.
So the second interval is a separate block bootstrap over whole slates, drawn from the
continuation of the same stream.

Both are percentile intervals at 95%. `p_roi_positive` is the fraction of ROI
resamples above zero — the one-sided read, which is usually the question being asked.
"""
function bootstrap_portfolio(t::Trajectory; B::Int = 4000, seed::Int = 1)
    bets = t.bets
    if isempty(bets) || sum(bets.stake) <= 0
        return BootstrapCI(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, B, seed)
    end
    rng  = Random.MersenneTwister(seed)

    # --- ROI, clustered by match. `Portfolio.bootstrap_roi`, operation for operation.
    mids = unique(bets.match_id)
    idxs = Dict(m => findall(==(m), bets.match_id) for m in mids)
    v = Vector{Float64}(undef, B)
    for b in 1:B
        sel = Int[]
        for _ in eachindex(mids)
            append!(sel, idxs[mids[rand(rng, 1:length(mids))]])
        end
        v[b] = 100 * sum(bets.pnl[sel]) / sum(bets.stake[sel])
    end

    # --- mean log growth, blocked by slate.
    g = Vector{Float64}(undef, B)
    n_sl = length(t.slate_pl)
    if n_sl == 0
        fill!(g, 0.0)
    else
        lg = log.(1.0 .+ t.slate_pl)
        for b in 1:B
            s = 0.0
            for _ in 1:n_sl
                s += lg[rand(rng, 1:n_sl)]
            end
            g[b] = s / n_sl
        end
    end

    return BootstrapCI(quantile(v, 0.025), quantile(v, 0.975), std(v),
                       quantile(g, 0.025), quantile(g, 0.975), std(g),
                       mean(v .> 0.0), B, seed)
end

bootstrap_portfolio(r::PortfolioResult; kw...) = bootstrap_portfolio(r.trajectory; kw...)


# ==============================================================================
# 5. THE ONE-CALL ENTRY POINT
# ==============================================================================

"""
    run_portfolio_simulation(spec, policy, source, odds, fixtures; …)
        -> (result::PortfolioResult, books, build::BuildReport)

Build → group → simulate → report, in one call.

`source` is whatever `build_books` accepts: a `Fit` (gated by default), a typed
posterior container, or a legacy `(latents_df, expr)` pair passed as a tuple.

The `BuildReport` is returned alongside rather than folded into the result, because
"41 of 500 fixtures had no complete market group" is a data-quality finding and
belongs where someone will read it, not inside a metrics struct.
"""
function run_portfolio_simulation(spec::BookSpec, policy::PolicySpec, source, odds,
                                  fixtures; require_result::Bool = true,
                                  require_converged::Bool = true, quiet::Bool = false,
                                  kw...)
    books, br = if source isa Fit
        build_books_reported(spec, source, odds, fixtures;
                             require_result = require_result,
                             require_converged = require_converged, quiet = quiet)
    elseif source isa Tuple
        build_books_reported(spec, source[1], source[2], odds, fixtures;
                             require_result = require_result, quiet = quiet)
    else
        build_books_reported(spec, source, odds, fixtures;
                             require_result = require_result, quiet = quiet)
    end
    result = simulate_portfolio(policy, books, br; kw...)
    return (result, books, br)
end

run_portfolio_simulation(sys::PortfolioSystem, source, odds, fixtures; kw...) =
    run_portfolio_simulation(sys.book, sys.policy, source, odds, fixtures; kw...)


# ==============================================================================
# 6. MATCH DAY
# ==============================================================================
#
# Backtesting and match-day differ in exactly one respect: a backtest book carries a
# settlement vector and a match-day book does not. Everything else — pricing, the
# payoff matrix, the allocator, shrinkage, trust, the drawdown budget, the exposure
# cap — is identical and shared, which is the point: the sheet you bet from is
# produced by the code path that was audited against history, not by a parallel
# reimplementation.

"""
    stake_sheet(sys, source, odds_df, fixtures; bankroll = 1.0, require_converged = true)
        -> DataFrame

One row per bet to place.

`fixtures` must be a `Dict{Int, FixtureInfo}` built from a FIXTURE LIST for live use —
`ds.matches` is the curated store of FINISHED matches, so a `DataStore` here yields an
empty sheet and no error. `MatchDay.fixture_info` builds the right dictionary.

Risk is solved per slate, so every fixture settling together shares one drawdown
budget and one exposure cap. That is the whole reason this is not a per-match loop.
"""
function stake_sheet(sys::PortfolioSystem, source, odds, fixtures;
                     bankroll::Real = 1.0, require_converged::Bool = true,
                     quiet::Bool = false)
    books, _ = if source isa Fit
        build_books_reported(sys.book, source, odds, fixtures; require_result = false,
                             require_converged = require_converged, quiet = quiet)
    else
        build_books_reported(sys.book, source, odds, fixtures; require_result = false,
                             quiet = quiet)
    end
    return _stake_sheet(sys, books, bankroll)
end

"""
    stake_sheet(sys, latents_df::DataFrame, expr, odds_df, fixtures; bankroll = 1.0)

The legacy five-argument shape (`src/Portfolio/matchday.jl:34`). Same columns, same
order, same sort — routed onto the typed-container fast path when the model's family
has one.
"""
function stake_sheet(sys::PortfolioSystem, latents_df::AbstractDataFrame, expr, odds,
                     fixtures; bankroll::Real = 1.0, quiet::Bool = false)
    books, _ = build_books_reported(sys.book, latents_df, expr, odds, fixtures;
                                    require_result = false, quiet = quiet)
    return _stake_sheet(sys, books, bankroll)
end

stake_sheet(sys::PortfolioSystem, latents::AbstractPosteriorLatents, expr, odds,
            fixtures; kw...) = stake_sheet(sys, latents, odds, fixtures; kw...)

"The sheet loop itself. `src/Portfolio/matchday.jl:40-59`, column for column."
function _stake_sheet(sys::PortfolioSystem, books::Vector{MatchBook}, bankroll::Real)
    isempty(books) && return _up_empty_sheet()
    slates = group(sys.policy.grouping, books)

    rows = NamedTuple[]
    for (t, sl) in enumerate(slates)
        ctx   = SlateContext(t, sl.window, Float64(bankroll))
        alloc = stake_slate(sys.policy, sl, ctx)
        for (i, b) in enumerate(sl.books), j in eachindex(b.sels)
            f = alloc.stakes[i][j]
            f > 0 || continue
            s = b.sels[j]
            push!(rows, (slate = sl.window, match_id = b.m_id, family = s.family,
                         group = s.group, line = s.line, selection = s.selection,
                         odds_quoted = s.odds_quoted, odds = s.odds_used,
                         p_model = s.p_model, p_market = s.p_market,
                         edge = s.p_model - s.p_market,
                         frac = f, stake = f * bankroll,
                         k_risk = alloc.k_risk, slate_exposure = alloc.exposure,
                         capped = alloc.capped, settled = is_settled(b)))
        end
    end
    isempty(rows) && return _up_empty_sheet()
    return sort!(DataFrame(rows), [:slate, :stake], rev = [false, true])
end

_up_empty_sheet() = DataFrame(slate = Date[], match_id = Int[], family = String[],
                              group = String[], line = Float64[], selection = Symbol[],
                              odds_quoted = Float64[], odds = Float64[],
                              p_model = Float64[], p_market = Float64[], edge = Float64[],
                              frac = Float64[], stake = Float64[], k_risk = Float64[],
                              slate_exposure = Float64[], capped = Bool[], settled = Bool[])


# ==============================================================================
# 7. DISPLAY
# ==============================================================================

"""
    display_portfolio(result; io = stdout, max_slates = 12)

The three tables worth reading in order: the headline, the settlement windows, and
the per-family attribution.

Attribution is deliberately last and deliberately present. It is the first thing to
look at when a headline number moves: on the reference ScottishLower book 83% of the
profit came from 1X2 — a family on which the model has no measurable log-loss
advantage over the market at all.
"""
function display_portfolio(r::PortfolioResult; io::IO = stdout, max_slates::Int = 12)
    s = r.summary
    println(io)
    println(io, "  PORTFOLIO  —  ", s.n_slates, " slates, ", s.n_fixtures, " fixtures, ",
            s.n_bets, " bets over ", s.span_days, " days")
    if r.converged !== nothing
        println(io, "  posterior  —  ", r.converged ? "CONVERGED" : "NOT CONVERGED" *
                (isempty(r.failed_gates) ? "" : "  (failed: " *
                 join(r.failed_gates, ", ") * ")"))
    end
    println(io, "  " * "-"^74)
    @printf(io, "  %-22s %12.2f   %-22s %10.2f%%\n",
            "bankroll", s.final_bankroll, "total return", s.total_return_pct)
    @printf(io, "  %-22s %12s   %-22s %10.2f%%\n",
            "CAGR", isnan(s.cagr) ? "—" : @sprintf("%.2f%%", 100 * s.cagr),
            "flat ROI", s.roi)
    @printf(io, "  %-22s %12.5f   %-22s %10s\n",
            "growth / slate", s.growth_per_slate, "1X2 ROI",
            isnan(s.roi_1x2) ? "—" : @sprintf("%.2f%%", s.roi_1x2))
    @printf(io, "  %-22s %12.2f%%  %-22s %10.4f\n",
            "max drawdown", s.mdd, "Sharpe (slate)", s.sharpe)
    @printf(io, "  %-22s %12.4f   %-22s %10s\n",
            "Calmar", s.calmar, "Sharpe (annualised)",
            isnan(s.sharpe_ann) ? "—" : @sprintf("%.4f", s.sharpe_ann))
    @printf(io, "  %-22s %12.4f   %-22s %10s\n",
            "Ulcer", s.ulcer, "Sortino",
            isfinite(s.sortino) ? @sprintf("%.4f", s.sortino) : "∞")
    @printf(io, "  %-22s %12.4f   %-22s %10.2f%%\n",
            "mean exposure", s.mean_exposure, "win rate", 100 * s.win_rate)
    @printf(io, "  %-22s %12.4f   %-22s %10d\n",
            "mean k_risk", s.mean_k_risk, "slates capped", s.n_capped)
    if r.bootstrap_ci !== nothing
        c = r.bootstrap_ci
        @printf(io, "  %-22s   [%+7.2f%%, %+7.2f%%]   P(ROI > 0) = %.3f   (B = %d)\n",
                "ROI 95% CI (by match)", c.roi_lo, c.roi_hi, c.p_roi_positive, c.B)
    end

    println(io)
    println(io, "  SETTLEMENT WINDOWS")
    @printf(io, "  %-12s %5s %5s %10s %10s %9s %8s %6s\n",
            "date", "fix", "bets", "bankroll", "pnl", "exposure", "k_risk", "cap")
    println(io, "  " * "-"^74)
    shown = min(length(r.daily_states), max_slates)
    for d in r.daily_states[1:shown]
        @printf(io, "  %-12s %5d %5d %10.2f %+10.4f %9.4f %8.4f %6s\n",
                string(d.date), d.n_fixtures, d.n_bets, d.bankroll_close, d.pnl_frac,
                d.exposure, d.k_risk, d.capped ? "yes" : "—")
    end
    shown < length(r.daily_states) &&
        @printf(io, "  … %d more\n", length(r.daily_states) - shown)

    if !isempty(r.attribution)
        println(io)
        println(io, "  ATTRIBUTION BY FAMILY")
        @printf(io, "  %-22s %6s %10s %10s %9s %7s\n",
                "family", "n", "stake", "pnl", "roi %", "hit")
        println(io, "  " * "-"^74)
        for row in eachrow(r.attribution)
            @printf(io, "  %-22s %6d %10.4f %+10.4f %9.2f %7.3f\n",
                    row.family, row.n, row.stake, row.pnl, row.roi, row.hit)
        end
    end
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", r::PortfolioResult) = display_portfolio(r; io = io)
