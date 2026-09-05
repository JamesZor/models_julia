# ==============================================================================
# src/Calibration/diagnostics.jl — the coherence audit
# ==============================================================================
#
# The coherence claim is exact, and it is worth stating precisely what "exact" means.
# Every market family below sums the SAME 12x12 grid: 1X2 partitions it by the sign of
# (i - j), a half-integer totals line by whether i + j exceeds it, BTTS by whether both
# exceed zero. Three partitions of one tensor, so all three sums equal the grid's total
# mass — and that mass is 1 minus the truncated tail beyond 11 goals a side, which is
# O(1e-9) at football rates.
#
# So the audit reports the mass DEFICIT rather than asserting 1.0 exactly, and the number
# that matters is `max_family_spread`: the largest disagreement BETWEEN families on one
# fixture. A selection-level shift cannot produce this table at any tolerance, and that is
# the comparison being made.
# ==============================================================================

"A short, unique label per market in a list: `1x2`, `ou_25`, `btts`."
function market_family_label(m)
    m isa Data.Market1X2 && return "1x2"
    m isa Data.MarketBTTS && return "btts"
    m isa Data.MarketOverUnder && return "ou_" * replace(string(m.line), "." => "")
    return lowercase(string(typeof(m).name.name))
end

"""
    coherence_report(latents, markets; max_goals = 12, threaded = true) -> NamedTuple

Per-fixture sums of each market family, and the worst deviation from the shared grid mass.

| field | is |
|---|---|
| `max_family_spread` | the largest disagreement BETWEEN families on one fixture. **The number that matters** — generative rate calibration makes it zero to rounding |
| `max_deviation_from_one` | the largest gap from 1.0, which is the truncated tail beyond the grid, not an incoherence |
| `mean_grid_mass` | the mass a family typically sums to |
| `frame` | one row per fixture, one column per family |

A calibrated container and its raw source must produce the same `max_family_spread`,
because both are read off one tensor. That is the point of the construction, and
`test_calibration_v2.jl` T7 asserts it rather than trusting the argument.
"""
function coherence_report(l::Models.AbstractPosteriorLatents, markets;
                          max_goals::Integer = 12, threaded::Bool = true)
    p = Evaluation.market_probabilities(l, markets; keep_draws = false,
                                        max_goals = max_goals, threaded = threaded)
    nfix = length(p.match_ids)
    families = Vector{Tuple{String, Vector{Symbol}}}()
    for m in markets
        push!(families, (market_family_label(m), collect(Predictions.market_keys(m))))
    end

    sums = Matrix{Float64}(undef, nfix, length(families))
    @inbounds for (c, (_, keys_)) in enumerate(families)
        cols = [p.col_of[k] for k in keys_]
        for i in 1:nfix
            s = 0.0
            for cc in cols
                s += p.means[i, cc]
            end
            sums[i, c] = s
        end
    end

    worst_dev = 0.0
    worst_spread = 0.0
    @inbounds for i in 1:nfix
        lo = minimum(view(sums, i, :))
        hi = maximum(view(sums, i, :))
        worst_spread = max(worst_spread, hi - lo)
        worst_dev = max(worst_dev, max(abs(1.0 - lo), abs(1.0 - hi)))
    end

    frame = DataFrame(match_id = copy(p.match_ids))
    for (c, (name, _)) in enumerate(families)
        frame[!, Symbol("sum_", name)] = sums[:, c]
    end

    return (; n_fixtures = nfix,
            family_names = String[f[1] for f in families],
            max_deviation_from_one = worst_dev,
            max_family_spread = worst_spread,
            mean_grid_mass = mean(sums),
            frame = frame)
end

coherence_report(cf::CalibratedFit, markets; kw...) =
    coherence_report(Evaluation.fit_latents(cf), markets; kw...)

"""
    coherence_report(cf::CalibratedFit; kw...) -> NamedTuple

The audit over the wide book this stream reports on: 1X2, O/U 0.5/1.5/2.5/3.5, BTTS.
"""
coherence_report(cf::CalibratedFit; kw...) =
    coherence_report(cf, l2_full_direction_markets(); kw...)

"""
    l2_tradeable_markets()

The 11 directions that are safe to STAKE: 1X2, O/U 1.5/2.5/3.5, BTTS. **No O/U 0.5.**

That ladder is excluded from the staking scope because on the Scottish Lower archive it
is quoted one-sided on 574 fixtures, and a de-vigged one-sided quote is a fabricated fair
price of exactly 1.0 (see `book.jl`'s header). [`point_in_time_book`](@ref) refuses such a
market before it normalises, so a T-25 book cannot carry the artefact — but the ladder is
also thin where it IS two-sided (44 of 710 priced fixtures in the matched T-25 book), so
staking it would be staking noise.

Use [`l2_full_direction_markets`](@ref) for the diagnostic arm that documents the
exclusion rather than assuming it.
"""
l2_tradeable_markets() = Data.AbstractMarket[
    Data.Market1X2(),
    Data.MarketOverUnder(1.5),
    Data.MarketOverUnder(2.5),
    Data.MarketOverUnder(3.5),
    Data.MarketBTTS(),
]

"""
    l2_full_direction_markets()

All 13 directions including the O/U 0.5 line. **Diagnostic only** — see
[`l2_tradeable_markets`](@ref) for why that line is not staked.
"""
l2_full_direction_markets() = Data.AbstractMarket[
    Data.Market1X2(),
    Data.MarketOverUnder(0.5),
    Data.MarketOverUnder(1.5),
    Data.MarketOverUnder(2.5),
    Data.MarketOverUnder(3.5),
    Data.MarketBTTS(),
]

"""
    l2_headline_selections()

`Evaluation.DEFAULT_SCORED_MARKETS` as selection symbols: 1X2 + O/U 2.5 + BTTS.

The scope every published Gate-1 threshold was measured on. Scoring the wide book against
those numbers would be a category error, so this is the filter that keeps the comparison
honest.
"""
l2_headline_selections() =
    Symbol[:home, :draw, :away, :over_25, :under_25, :btts_yes, :btts_no]

"""
    calibration_summary(cf; io = stdout)

One screen answering "what did this calibrator do, and to how much of the book".

Prints the recipe, the inversion coverage with its refusal reasons, the weight and
variance-retention quantiles, and — when the map is not the plain pool — the
predictive-rate ratio the anchor exists to hold at 1.
"""
function calibration_summary(cf::CalibratedFit; io::IO = stdout)
    cal = cf.calibrator
    cov = cf.coverage
    ws = weight_summary(cf.rate_diagnostics)
    ds_ = dispersion_summary(cf.rate_diagnostics)

    println(io, "=" ^ 78)
    println(io, " CALIBRATION — ", Training.fit_name(cf.fit), "  x  ", cal.name)
    println(io, "=" ^ 78)
    println(io, "  law         : ", cal.law)
    println(io, "  dispersion  : ", map_label(cal.dispersion),
                "   anchor :", cal.anchor,
                is_pool_map(cal.dispersion) ? "  (zero on the pool map)" : "")
    println(io, "  book instant: T", @sprintf("%+.0f", cf.book_as_of_minutes), " min")
    println(io)
    @printf(io, "  fixtures    : %d held | %d quoted | %d inverted | %d refused | %d absent\n",
            cov.n_fixtures, cov.n_quoted, cov.n_accepted, cov.n_refused, cov.n_absent)
    @printf(io, "  coverage    : %.1f%% of all, %.1f%% of quoted\n",
            100 * cov.coverage, 100 * cov.coverage_quoted)
    refusals = inversion_refusals(cf.market_rates)
    if !isempty(refusals)
        println(io, "  refusals    :")
        for (reason, n) in refusals
            @printf(io, "      %5d  %s\n", n, reason)
        end
    end
    println(io)
    if ws.n_shifted == 0
        println(io, "  NOTHING SHIFTED — this container equals its raw source.")
    else
        @printf(io, "  weight w    : p10 %.3f | median %.3f | p90 %.3f | mean %.3f\n",
                ws.w_p10, ws.w_median, ws.w_p90, ws.w_mean)
        @printf(io, "  market share: median %.3f of the pooled location\n",
                ws.market_share_median)
        @printf(io, "  var retained: side %.3f | supremacy %.3f | totals %.3f (medians)\n",
                ds_.ret_side_median, ds_.ret_sup_median, ds_.ret_tot_median)
        if !is_pool_map(cal.dispersion)
            @printf(io, "  rate ratio  : median %.4f | p90 %.4f  (the anchor holds this at 1)\n",
                    ds_.rate_ratio_median, ds_.rate_ratio_p90)
        end
    end
    println(io, "=" ^ 78)
    return nothing
end
