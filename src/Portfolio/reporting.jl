# src/Portfolio/reporting.jl
#
# Stage E: what a human reads.
#
# Three tables, in the order they are worth reading -- the headline, the settlement windows, and
# the per-family attribution. Attribution is deliberately last and deliberately present: it is the
# first thing to look at when a headline number moves. On the reference ScottishLower book 83% of
# the profit came from 1X2, a family on which the model has no measurable log-loss advantage over
# the market at all.
#
# Nothing here computes a statistic. Everything is read off `PortfolioSummary`,
# `BootstrapCI` and `attribution`, so a table and a number can never disagree.

export PortfolioReport, portfolio_report, display_portfolio, daily_returns_table,
       portfolio_markdown

# ===================================================================
# 1. The terminal view
# ===================================================================

"""
    display_portfolio(result; io = stdout, max_slates = 12)

The headline block, the settlement windows and the attribution table.

`max_slates` truncates the middle table only; the headline and attribution always cover the whole
run.
"""
function display_portfolio(r::PortfolioResult; io::IO = stdout, max_slates::Int = 12)
    s = r.summary
    println(io)
    println(io, "  PORTFOLIO  --  ", s.n_slates, " slates, ", s.n_fixtures, " fixtures, ",
            s.n_bets, " bets over ", s.span_days, " days")
    if r.converged !== nothing
        println(io, "  posterior  --  ", r.converged ? "CONVERGED" :
                "NOT CONVERGED" * (isempty(r.failed_gates) ? "" :
                 "  (failed: " * join(r.failed_gates, ", ") * ")"))
    end
    println(io, "  " * "-"^74)
    @printf(io, "  %-22s %12.2f   %-22s %10.2f%%\n",
            "bankroll", s.final_bankroll, "total return", s.total_return_pct)
    @printf(io, "  %-22s %12s   %-22s %10.2f%%\n",
            "CAGR", isnan(s.cagr) ? "--" : @sprintf("%.2f%%", 100 * s.cagr),
            "flat ROI", s.roi)
    @printf(io, "  %-22s %12.5f   %-22s %10s\n",
            "growth / slate", s.growth_per_slate, "1X2 ROI",
            isnan(s.roi_1x2) ? "--" : @sprintf("%.2f%%", s.roi_1x2))
    @printf(io, "  %-22s %12.2f%%  %-22s %10.4f\n",
            "max drawdown", s.mdd, "Sharpe (slate)", s.sharpe)
    @printf(io, "  %-22s %12.4f   %-22s %10s\n",
            "Calmar", s.calmar, "Sharpe (annualised)",
            isnan(s.sharpe_ann) ? "--" : @sprintf("%.4f", s.sharpe_ann))
    @printf(io, "  %-22s %12.4f   %-22s %10s\n",
            "Ulcer", s.ulcer, "Sortino",
            isfinite(s.sortino) ? @sprintf("%.4f", s.sortino) : "inf")
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
                d.exposure, d.k_risk, d.capped ? "yes" : "--")
    end
    shown < length(r.daily_states) &&
        @printf(io, "  ... %d more\n", length(r.daily_states) - shown)

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

# ===================================================================
# 2. The daily return breakdown
# ===================================================================

"""
    daily_returns_table(result) -> DataFrame

One row per settlement window, with the derived per-window quantities a reader wants and
`DailyState` deliberately does not store because they are functions of what it does:

| column | is |
|---|---|
| `return_pct` | `100 * pnl_frac` -- the window's return on the bankroll it opened with |
| `log_growth` | `log(1 + pnl_frac)` -- the quantity Kelly maximises, and what `sharpe` averages |
| `roi_pct` | `100 * pnl_frac / stake_frac` -- return on stake, `NaN` for a window with no bets |
| `drawdown_pct` | bankroll against its own running maximum, NEGATIVE, in percent |

`drawdown_pct` is computed off `trajectory.bankroll` -- the same vector, with the same operations,
that `portfolio_summary` reduces to `mdd` -- with the opening `1.0` dropped. So `minimum` of this
column IS `summary.mdd`, bit for bit, rather than a second drawdown computed a second way from the
rounded currency series.
"""
function daily_returns_table(r::PortfolioResult)
    df = states_frame(r)
    isempty(df) && return hcat(df, DataFrame(return_pct = Float64[], log_growth = Float64[],
                                             roi_pct = Float64[], drawdown_pct = Float64[]))
    df.return_pct = 100 .* df.pnl_frac
    df.log_growth = log.(1.0 .+ df.pnl_frac)
    df.roi_pct    = [s > 0 ? 100 * p / s : NaN for (p, s) in zip(df.pnl_frac, df.stake_frac)]
    bk = r.trajectory.bankroll
    rm = accumulate(max, bk)
    df.drawdown_pct = ((bk .- rm) ./ rm .* 100)[2:end]
    return df
end

# ===================================================================
# 3. The report container
# ===================================================================

"""
    PortfolioReport

A simulation, its build provenance and its name, in one serialisable object.

Held together because the three answer one question between them and separately answer none of it:
`summary.roi` is not interpretable without `build.converged` (was the posterior fit?) and
`build.n_books` (how much of the fold survived?). A report pulled off disk carries all three.

Construct with [`portfolio_report`](@ref); render with [`portfolio_markdown`](@ref) or
`show(io, MIME"text/plain"(), r)`.
"""
struct PortfolioReport
    name::String
    result::PortfolioResult
    build::Union{Nothing, BuildReport}
    daily::DataFrame
end

"""
    portfolio_report(result; name = "portfolio", build = nothing) -> PortfolioReport

Bundle a simulation with the build that produced it. `build` is optional because a result
simulated from books loaded off disk has no live `BuildReport` to attach.
"""
portfolio_report(r::PortfolioResult; name::AbstractString = "portfolio",
                 build::Union{Nothing, BuildReport} = nothing) =
    PortfolioReport(String(name), r, build, daily_returns_table(r))

portfolio_report(r::PortfolioResult, b::BuildReport; name::AbstractString = "portfolio") =
    portfolio_report(r; name = name, build = b)

Base.show(io::IO, r::PortfolioReport) =
    print(io, "PortfolioReport(\"", r.name, "\", ", r.result, ")")

function Base.show(io::IO, ::MIME"text/plain", r::PortfolioReport)
    println(io, "PortfolioReport  --  ", r.name)
    r.build === nothing || show(io, MIME"text/plain"(), r.build)
    display_portfolio(r.result; io = io)
end

# ===================================================================
# 4. Markdown
# ===================================================================

_md_num(x::Real; digits::Int = 4) =
    isnan(x) ? "--" : isfinite(x) ? string(round(x; digits = digits)) : "inf"

"""
    portfolio_markdown(report; max_slates = 20) -> String

The report as a Markdown document: a headline table, the settlement-window breakdown, and the
per-family attribution.

Named `portfolio_markdown` rather than `markdown_report` because `Evaluation.markdown_report`
already holds that name at the package's top level and two functions answering to one name is the
failure mode this whole graduation line exists to remove.
"""
function portfolio_markdown(r::PortfolioReport; max_slates::Int = 20)
    s   = r.result.summary
    io  = IOBuffer()
    println(io, "# ", r.name)
    println(io)

    if r.build !== nothing
        b = r.build
        println(io, "Built ", b.n_books, " of ", b.n_fixtures, " fixtures",
                n_skipped(b) == 0 ? "" : " ($(n_skipped(b)) skipped)",
                b.converged === nothing ? "." :
                b.converged ? "; posterior CONVERGED." : "; posterior NOT CONVERGED.")
        println(io)
    end

    println(io, "## Headline")
    println(io)
    println(io, "| metric | value |")
    println(io, "|---|---|")
    for (k, v) in (
            "initial bankroll"   => _md_num(s.initial_bankroll; digits = 2),
            "final bankroll"     => _md_num(s.final_bankroll; digits = 2),
            "total return %"     => _md_num(s.total_return_pct; digits = 2),
            "CAGR %"             => _md_num(100 * s.cagr; digits = 2),
            "flat ROI %"         => _md_num(s.roi; digits = 2),
            "1X2 ROI %"          => _md_num(s.roi_1x2; digits = 2),
            "growth / slate"     => _md_num(s.growth_per_slate; digits = 5),
            "max drawdown %"     => _md_num(s.mdd; digits = 2),
            "Ulcer"              => _md_num(s.ulcer),
            "Calmar"             => _md_num(s.calmar),
            "Martin"             => _md_num(s.martin),
            "Sharpe (slate)"     => _md_num(s.sharpe),
            "Sharpe (annualised)"=> _md_num(s.sharpe_ann),
            "Sortino"            => _md_num(s.sortino),
            "win rate %"         => _md_num(100 * s.win_rate; digits = 2),
            "mean exposure"      => _md_num(s.mean_exposure),
            "max exposure"       => _md_num(s.max_exposure),
            "mean k_risk"        => _md_num(s.mean_k_risk),
            "slates"             => string(s.n_slates),
            "fixtures"           => string(s.n_fixtures),
            "bets"               => string(s.n_bets),
            "slates capped"      => string(s.n_capped),
            "span (days)"        => string(s.span_days))
        println(io, "| ", k, " | ", v, " |")
    end
    if r.result.bootstrap_ci !== nothing
        c = r.result.bootstrap_ci
        println(io, "| ROI 95% CI (by match) | [", _md_num(c.roi_lo; digits = 2), ", ",
                _md_num(c.roi_hi; digits = 2), "] |")
        println(io, "| P(ROI > 0) | ", _md_num(c.p_roi_positive; digits = 3), " |")
    end
    println(io)

    println(io, "## Settlement windows")
    println(io)
    println(io, "| date | fixtures | bets | bankroll | return % | ROI % | exposure | k_risk | drawdown % |")
    println(io, "|---|---|---|---|---|---|---|---|---|")
    d = r.daily
    shown = min(nrow(d), max_slates)
    for i in 1:shown
        println(io, "| ", d.date[i], " | ", d.n_fixtures[i], " | ", d.n_bets[i], " | ",
                _md_num(d.bankroll_close[i]; digits = 2), " | ",
                _md_num(d.return_pct[i]; digits = 3), " | ",
                _md_num(d.roi_pct[i]; digits = 2), " | ",
                _md_num(d.exposure[i]), " | ", _md_num(d.k_risk[i]), " | ",
                _md_num(d.drawdown_pct[i]; digits = 2), " |")
    end
    shown < nrow(d) && println(io, "| ... | | | | | | | | |")
    println(io)

    if !isempty(r.result.attribution)
        println(io, "## Attribution by family")
        println(io)
        println(io, "| family | n | stake | pnl | ROI % | hit |")
        println(io, "|---|---|---|---|---|---|")
        for row in eachrow(r.result.attribution)
            println(io, "| ", row.family, " | ", row.n, " | ", _md_num(row.stake), " | ",
                    _md_num(row.pnl), " | ", _md_num(row.roi; digits = 2), " | ",
                    _md_num(row.hit; digits = 3), " |")
        end
        println(io)
    end

    return String(take!(io))
end

portfolio_markdown(r::PortfolioResult; kw...) = portfolio_markdown(portfolio_report(r); kw...)
