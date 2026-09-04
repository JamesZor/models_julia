# ==============================================================================
# l01 — Generative rate calibration: spec, market inversion, posterior shift
# ==============================================================================
#
# Loader. Definitions only, no execution. The runners (`r01_*`, `r02_*`) load the
# data, drive the sweep and write the artefacts.
#
# ------------------------------------------------------------------------------
# WHAT THIS IS
# ------------------------------------------------------------------------------
#
# Layer-2 calibration applied at the GENERATIVE INTENSITY level rather than at the
# selection level. The whole construction is four steps:
#
#   1. invert the de-vigged closing book back to (λ_mkt_h, λ_mkt_a) by Nelder-Mead
#      on `Features.DoublePoissonMarketFeature`;
#   2. measure the log-rate discrepancy Δ = log median(λ_model) − log λ_mkt;
#   3. pool every posterior draw log-linearly at weight w(Δ):
#          log λ*⁽ᵈ⁾ = w · log λ_model⁽ᵈ⁾ + (1 − w) · log λ_mkt;
#   4. hand the shifted `CountLatents` to the SAME score-grid kernels, evaluator and
#      portfolio the raw container goes through.
#
# Step 4 is the point. Because every derivative price is read off one 12×12 score
# tensor built from the shifted rates, 1X2, every totals line and BTTS stay mutually
# coherent by construction — there is no way to shift P(Over 2.5) without moving
# P(Under 2.5) by the same amount, which is exactly the failure mode of the
# selection-level `BasicLogitShift` this stream exists to replace.
#
# ------------------------------------------------------------------------------
# WHAT THIS IS NOT
# ------------------------------------------------------------------------------
#
# * NOT a variance-preserving calibration. Log-linear pooling contracts posterior
#   log-variance by exactly w²: at w = 0.41 (the Scottish median under the failed
#   Ireland parameters) 83% of the posterior log-variance is destroyed. Kelly stake
#   size reads that variance. `calibrate_latents` returns `var_retention` per
#   fixture per side, and `weight_summary` its quantiles, so the contraction is
#   reported rather than implicit.
# * NOT a claim that (0.25, 0.25) generalises. It did not — see
#   `current_development/orderbook_layer2/research_questions_explore/notes_rqs_01.md` §4.
#   The whole purpose of the sweep is to find out whether ANY (w_base, σ) does.
# * NOT a staking-trust replacement. Trust and calibration stay separate controls
#   until r02 measures them jointly.
#
# ------------------------------------------------------------------------------
# THE FALLBACK PROTOCOL, AND WHY IT IS w = 1
# ------------------------------------------------------------------------------
#
# A fixture whose book cannot be inverted (no quotes, too few quotes, Nelder-Mead
# not converged, residual too large, implausible rates) is passed through
# UNCHANGED — the raw model draws, bit for bit, not a league-mean rate and not a
# dropped fixture. Dropping it would change which fixtures the sweep scores between
# specs and make two rows of the results table incomparable; inventing a rate would
# price a fixture from inputs the pipeline declined to use.
#
# Bit-identity is asserted rather than assumed: a fixture with w == 1.0 copies its
# raw row instead of computing exp(1·log λ + 0), because `exp(log(x)) != x` in
# Float64 and the w_base = 1.0 grid point has to reproduce the uncalibrated
# baseline EXACTLY or it is not a control.
#
# ------------------------------------------------------------------------------
# DATABASE BOUNDARY
# ------------------------------------------------------------------------------
#
# This loader touches no database. The runners read `mcmc_experiments` (posteriors)
# and `betdb` (odds, results) and write neither. `betdb.paper_runbook` is never
# opened; the live console on 8085 is not this stream's business.
# ==============================================================================

# %%
# ===================================================================
# 1. Packages and module aliases
# ===================================================================

using BayesianFootball
using DataFrames
using LinearAlgebra
using Optim
using Printf
using Statistics

const L01_FEATURES = BayesianFootball.Features
const L01_EVAL     = BayesianFootball.Evaluation


# %%
# ===================================================================
# 2. The calibration spec and the three weight laws
# ===================================================================

"""
    GenerativeCalibrationSpec(; method, w_base, sigma, w_max)

One point of the calibration hypothesis space.

| `method`             | w(0)    | w(±∞)   | reads as                                     |
|----------------------|---------|---------|----------------------------------------------|
| `:inverse_gaussian`  | w_base  | 1.0     | trust the market on noise, the model on edges |
| `:standard_gaussian` | w_max   | w_base  | optimiser's-curse shrinkage of extreme claims |
| `:static_geometric`  | w_base  | w_base  | a constant pool; `sigma` is ignored           |

`:inverse_gaussian` is the Ireland form. `:standard_gaussian` is its textbook
opposite and is in the grid precisely because the Ireland conclusion was drawn from
one league and has never survived an out-of-domain test. `:static_geometric` is the
control that says whether the Δ-dependence buys anything at all over a flat pool.

`w_base = 1.0` is the identity in every form (with `w_max = 1.0`), and is kept in
the grid deliberately: it is the uncalibrated model, priced through the identical
code path, and it must reproduce the baseline row exactly.
"""
struct GenerativeCalibrationSpec
    method::Symbol
    w_base::Float64
    sigma::Float64
    w_max::Float64

    function GenerativeCalibrationSpec(; method::Symbol = :inverse_gaussian,
                                        w_base::Real = 0.25,
                                        sigma::Real = 0.25,
                                        w_max::Real = 1.0)
        method in (:inverse_gaussian, :standard_gaussian, :static_geometric) || throw(
            ArgumentError("method must be :inverse_gaussian, :standard_gaussian or " *
                          ":static_geometric, got :$method"))
        0.0 <= w_base <= 1.0 || throw(ArgumentError("w_base must be in [0,1]: $w_base"))
        0.0 <= w_max <= 1.0 || throw(ArgumentError("w_max must be in [0,1]: $w_max"))
        sigma > 0.0 || throw(ArgumentError("sigma must be positive: $sigma"))
        method === :standard_gaussian && w_max < w_base && throw(ArgumentError(
            "standard_gaussian needs w_max >= w_base (w_max is the peak at Δ=0, " *
            "w_base the floor at large |Δ|): got w_max=$w_max, w_base=$w_base"))
        return new(method, Float64(w_base), Float64(sigma), Float64(w_max))
    end
end

"""
    calibration_weight(spec, Δ) -> Float64

The model's share of the log-linear pool at log-rate discrepancy `Δ`.

    inverse  : w = w_base + (1 − w_base)·(1 − exp(−Δ²/2σ²))
    standard : w = w_base + (w_max − w_base)·exp(−Δ²/2σ²)
    static   : w = w_base
"""
@inline function calibration_weight(spec::GenerativeCalibrationSpec, Δ::Float64)
    spec.method === :static_geometric && return spec.w_base
    isfinite(Δ) || return 1.0
    g = exp(-(Δ * Δ) / (2.0 * spec.sigma * spec.sigma))
    if spec.method === :inverse_gaussian
        return spec.w_base + (1.0 - spec.w_base) * (1.0 - g)
    end
    return spec.w_base + (spec.w_max - spec.w_base) * g
end

"A short, filename-safe and sort-stable label for one spec."
function spec_label(spec::GenerativeCalibrationSpec)
    short = spec.method === :inverse_gaussian  ? "inv" :
            spec.method === :standard_gaussian ? "std" : "sta"
    spec.method === :static_geometric &&
        return @sprintf("%s_w%.2f", short, spec.w_base)
    return @sprintf("%s_w%.2f_s%.2f", short, spec.w_base, spec.sigma)
end

"`true` when this spec is the identity map on every fixture — the raw model."
is_identity_spec(spec::GenerativeCalibrationSpec) =
    spec.method === :static_geometric ? spec.w_base == 1.0 :
    spec.method === :inverse_gaussian ? spec.w_base == 1.0 :
    (spec.w_base == 1.0 && spec.w_max == 1.0)


# %%
# ===================================================================
# 3. Market rate inversion
# ===================================================================
#
# `Features.fit_market_implied_parameters` does this fit already, but returns only
# the minimiser: no residual, no convergence flag, no target count. Every one of
# those is a gate here, so the optimisation is rebuilt around the SAME
# `Features` primitives (`build_probability_matrix`, `_calculate_error`,
# `get_initial_guess`, `extract_parameters`) rather than a second implementation of
# the objective.
#
# LINE SET. `Features.LINES` lists both sides of every totals line, and
# `_calculate_error(Val(:over_25), …)` already scores the over AND under keys, so
# the default tuple counts each totals line twice while counting 1X2 once. That is a
# silent reweighting of the objective, so this loader passes one symbol per line.

"The market-inversion line set: 1X2, BTTS, and one symbol per totals line."
const L01_INVERSION_LINES =
    (:result_1x2, :btts, :over_05, :over_15, :over_25, :over_35)

"""
    MarketInversionConfig(; feature, max_goals, min_targets, max_sse, lambda_bounds)

The inversion and its acceptance gates. A fit failing any gate is REFUSED by name
and the fixture falls back to w = 1.

| gate            | refuses                                                     |
|-----------------|-------------------------------------------------------------|
| `min_targets`   | a book too thin to identify two rates (3 = bare 1X2)        |
| `max_sse`       | a converged optimum that still does not reproduce the book  |
| `lambda_bounds` | a rate outside anything a football match produces           |
"""
Base.@kwdef struct MarketInversionConfig
    feature::L01_FEATURES.AbstractMarketFeatureConfig =
        L01_FEATURES.DoublePoissonMarketFeature(lines = L01_INVERSION_LINES)
    max_goals::Int = 10
    min_targets::Int = 3
    max_sse::Float64 = 5.0e-3
    lambda_bounds::Tuple{Float64,Float64} = (0.05, 6.0)
end

"""
    MarketRateFit

One fixture's inverted book. `accepted = false` carries the `reason`, and a refused
fit is never read by `calibrate_latents` — the fixture passes through raw.
"""
struct MarketRateFit
    match_id::Int
    λ_home::Float64
    λ_away::Float64
    sse::Float64
    n_targets::Int
    optim_converged::Bool
    accepted::Bool
    reason::String
end

"""
    market_targets(odds_df) -> Dict{Int, Dict{Symbol, Float64}}

`match_id → (selection → prob_fair_close)`, built in one O(n) scan.

Rows with a missing or non-finite `prob_fair_close` are skipped rather than
inverted against; a duplicated (match, selection) keeps the LAST row, which is the
sort order the caller established.
"""
function market_targets(odds_df::AbstractDataFrame)
    out = Dict{Int, Dict{Symbol, Float64}}()
    hasproperty(odds_df, :prob_fair_close) || error(
        "market_targets: the odds frame carries no `prob_fair_close`. Build it with " *
        "`l01_betfair_closing_odds`, which de-vigs within (match, market, line).")
    for r in eachrow(odds_df)
        p = r.prob_fair_close
        (p === missing || !isfinite(p) || p <= 0.0 || p >= 1.0) && continue
        d = get!(() -> Dict{Symbol, Float64}(), out, Int(r.match_id))
        d[Symbol(r.selection)] = Float64(p)
    end
    return out
end

"""
    invert_market_rates(odds_df; config, match_ids = nothing) -> Dict{Int, MarketRateFit}

Nelder-Mead the de-vigged closing book back to (λ_mkt_h, λ_mkt_a), one fixture per
thread. `match_ids` restricts the work to the fixtures a latent container actually
holds; `nothing` inverts every fixture in the frame.

The result is computed ONCE per odds snapshot and reused across the whole sweep —
it does not depend on the model or on the spec.
"""
function invert_market_rates(odds_df::AbstractDataFrame;
                             config::MarketInversionConfig = MarketInversionConfig(),
                             match_ids = nothing)
    targets = market_targets(odds_df)
    ids = match_ids === nothing ? collect(keys(targets)) :
                                  Int[Int(m) for m in match_ids]
    sort!(ids)

    init = L01_FEATURES.get_initial_guess(config.feature)
    lines = config.feature.lines
    lo, hi = config.lambda_bounds
    fits = Vector{MarketRateFit}(undef, length(ids))

    Threads.@threads for n in eachindex(ids)
        mid = ids[n]
        tg = get(targets, mid, nothing)

        if tg === nothing || length(tg) < config.min_targets
            k = tg === nothing ? 0 : length(tg)
            fits[n] = MarketRateFit(mid, NaN, NaN, NaN, k, false, false,
                                    "too few quoted selections ($k < $(config.min_targets))")
            continue
        end

        loss = let cfg = config.feature, tgts = tg, mg = config.max_goals, ls = lines
            θ -> begin
                P = L01_FEATURES.build_probability_matrix(cfg, θ, mg)
                sse = 0.0
                for line in ls
                    sse += L01_FEATURES._calculate_error(Val(line), P, tgts)
                end
                return sse + L01_FEATURES.compute_loss_penalty(cfg, θ)
            end
        end

        res = Optim.optimize(loss, copy(init), NelderMead())
        θ̂ = Optim.minimizer(res)
        sse = Optim.minimum(res)
        conv = Optim.converged(res)
        par = L01_FEATURES.extract_parameters(config.feature, θ̂)
        λh = Float64(par.λ_home)
        λa = Float64(par.λ_away)

        reason = if !conv
            "Nelder-Mead did not converge"
        elseif !isfinite(sse) || sse > config.max_sse
            @sprintf("residual SSE %.3e exceeds %.3e", sse, config.max_sse)
        elseif !(lo <= λh <= hi) || !(lo <= λa <= hi)
            @sprintf("implied rates (%.3f, %.3f) outside [%.2f, %.2f]", λh, λa, lo, hi)
        else
            ""
        end

        fits[n] = MarketRateFit(mid, λh, λa, sse, length(tg), conv,
                                isempty(reason), reason)
    end

    return Dict{Int, MarketRateFit}(f.match_id => f for f in fits)
end

"The inversion, as a frame: one row per fixture, refusals carrying their reason."
function inversion_frame(rates::AbstractDict{Int, MarketRateFit})
    ids = sort!(collect(keys(rates)))
    return DataFrame(
        match_id        = ids,
        lambda_mkt_h    = [rates[i].λ_home for i in ids],
        lambda_mkt_a    = [rates[i].λ_away for i in ids],
        sse             = [rates[i].sse for i in ids],
        n_targets       = [rates[i].n_targets for i in ids],
        optim_converged = [rates[i].optim_converged for i in ids],
        accepted        = [rates[i].accepted for i in ids],
        reason          = [rates[i].reason for i in ids],
    )
end

"""
    refusal_counts(frame) -> Vector{Pair{String,Int}}

Refusal reasons and their counts, most frequent first. A gate that refuses 40% of a
book for one reason is a configuration problem; one that refuses 2% across four
reasons is the book being thin. The two look identical in a coverage percentage.
"""
function refusal_counts(frame::AbstractDataFrame)
    counts = Dict{String,Int}()
    for r in eachrow(frame)
        r.accepted && continue
        counts[r.reason] = get(counts, r.reason, 0) + 1
    end
    return sort!(collect(counts), by = last, rev = true)
end

"""
    inversion_coverage(rates, match_ids) -> NamedTuple

Coverage of an inversion over the fixtures a latent container holds, reported two
ways because only one of them measures dilution.

`coverage` is against EVERY fixture, and it counts a fixture the book never quoted
as a failure. Such a fixture contributes no scored observation either —
`require_market` drops it before any metric sees it — so it dilutes nothing.
`coverage_quoted` is against the fixtures that had a book to invert, and that is
the number to read when asking how much of the measured effect the refusals ate.
"""
function inversion_coverage(rates::AbstractDict{Int, MarketRateFit}, match_ids)
    ids = Int[Int(m) for m in match_ids]
    accepted = count(m -> haskey(rates, m) && rates[m].accepted, ids)
    absent = count(m -> !haskey(rates, m), ids)
    quoted = count(m -> haskey(rates, m) && rates[m].n_targets > 0, ids)
    return (; n_fixtures = length(ids), n_accepted = accepted,
            n_refused = length(ids) - accepted - absent, n_absent = absent,
            n_quoted = quoted,
            coverage = isempty(ids) ? NaN : accepted / length(ids),
            coverage_quoted = quoted == 0 ? NaN : accepted / quoted)
end


# %%
# ===================================================================
# 4. The posterior geometric shift
# ===================================================================

"""
    calibrate_latents(l::CountLatents, rates, spec) -> (CountLatents, DataFrame)

Apply the log-linear pool to every posterior draw of every fixture, and return the
shifted container beside the per-fixture diagnostic frame.

Δ is measured against the posterior MEDIAN, not the mean: the pooling weight is a
statement about where the bulk of the posterior sits relative to the book, and the
median of a right-skewed rate posterior is the location that answers it.

Fixtures with no accepted inversion, and sides whose weight is exactly 1.0, copy
their raw draws VERBATIM. See the fallback note in the file header.
"""
function calibrate_latents(l::CountLatents{Float64},
                           rates::AbstractDict{Int, MarketRateFit},
                           spec::GenerativeCalibrationSpec)
    ids = latent_match_ids(l)
    nm, nd = size(l.λ_home)
    length(ids) == nm || error("latent container is inconsistent: $(length(ids)) ids, $nm rows.")

    med_h = Vector{Float64}(undef, nm)
    med_a = Vector{Float64}(undef, nm)
    λm_h  = fill(NaN, nm)
    λm_a  = fill(NaN, nm)
    Δ_h   = fill(NaN, nm)
    Δ_a   = fill(NaN, nm)
    w_h   = ones(Float64, nm)
    w_a   = ones(Float64, nm)
    c_h   = zeros(Float64, nm)
    c_a   = zeros(Float64, nm)
    shift_h = falses(nm)
    shift_a = falses(nm)
    inverted = falses(nm)
    reasons = fill("", nm)

    buf = Vector{Float64}(undef, nd)
    @inbounds for i in 1:nm
        copyto!(buf, view(l.λ_home, i, :))
        med_h[i] = median!(buf)
        copyto!(buf, view(l.λ_away, i, :))
        med_a[i] = median!(buf)

        f = get(rates, ids[i], nothing)
        if f === nothing
            reasons[i] = "no market inversion attempted"
            continue
        end
        reasons[i] = f.reason
        f.accepted || continue

        inverted[i] = true
        λm_h[i] = f.λ_home
        λm_a[i] = f.λ_away
        Δ_h[i] = log(med_h[i]) - log(f.λ_home)
        Δ_a[i] = log(med_a[i]) - log(f.λ_away)
        w_h[i] = calibration_weight(spec, Δ_h[i])
        w_a[i] = calibration_weight(spec, Δ_a[i])
        c_h[i] = (1.0 - w_h[i]) * log(f.λ_home)
        c_a[i] = (1.0 - w_a[i]) * log(f.λ_away)
        shift_h[i] = w_h[i] != 1.0
        shift_a[i] = w_a[i] != 1.0
    end

    λh = similar(l.λ_home)
    λa = similar(l.λ_away)
    @inbounds for k in 1:nd
        for i in 1:nm
            λh[i, k] = shift_h[i] ? exp(w_h[i] * log(l.λ_home[i, k]) + c_h[i]) :
                                    l.λ_home[i, k]
            λa[i, k] = shift_a[i] ? exp(w_a[i] * log(l.λ_away[i, k]) + c_a[i]) :
                                    l.λ_away[i, k]
        end
    end

    calibrated = CountLatents(ids, λh, λa, l.observation_params)

    diagnostics = DataFrame(
        match_id          = copy(ids),
        inverted          = collect(inverted),
        reason            = reasons,
        lambda_model_h    = med_h,
        lambda_model_a    = med_a,
        lambda_mkt_h      = λm_h,
        lambda_mkt_a      = λm_a,
        delta_h           = Δ_h,
        delta_a           = Δ_a,
        w_h               = w_h,
        w_a               = w_a,
        var_retention_h   = w_h .^ 2,
        var_retention_a   = w_a .^ 2,
        lambda_shifted_h  = [shift_h[i] ? exp(w_h[i] * log(med_h[i]) + c_h[i]) : med_h[i]
                             for i in 1:nm],
        lambda_shifted_a  = [shift_a[i] ? exp(w_a[i] * log(med_a[i]) + c_a[i]) : med_a[i]
                             for i in 1:nm],
    )

    return calibrated, diagnostics
end

"""
    restrict_latents(l::CountLatents, keep_ids) -> CountLatents

The same posterior over a SUBSET of its fixtures, in the container's own row order.

Needed because the published Gate-1 thresholds (LogLoss 0.64337, ECE 0.0100) were
measured over the 40-fold 24/25 + 25/26 study, and the canonical runs have since
been extended to 43 folds with the 26/27 August programme. Scoring the extended
fixture set against those thresholds would be comparing two different questions;
restricting the container makes the comparison exact and the exclusion explicit.

Fixtures in `keep_ids` that the container does not hold are ignored — the caller is
naming a filter, not asserting coverage.
"""
function restrict_latents(l::CountLatents{Float64}, keep_ids)
    want = Set{Int}(Int(m) for m in keep_ids)
    ids = latent_match_ids(l)
    rows = findall(i -> ids[i] in want, eachindex(ids))
    isempty(rows) && error("restrict_latents: no fixture of the container is in `keep_ids`.")
    obs = l.observation_params
    obs === nothing || error(
        "restrict_latents: this container carries observation parameters " *
        "($(typeof(obs))); subsetting them is not defined here. Extend this method " *
        "before using it on a negative-binomial posterior.")
    return CountLatents(ids[rows], l.λ_home[rows, :], l.λ_away[rows, :], nothing)
end

"""
    weight_summary(diagnostics) -> NamedTuple

Quantiles of the applied weights and of the retained posterior log-variance, over
the fixtures that were actually shifted. The Ireland post-mortem turned on exactly
these numbers: a median w of 0.41 means the market supplied most of the location and
83% of the log-variance was destroyed, and no headline score says that.
"""
function weight_summary(diagnostics::AbstractDataFrame)
    d = diagnostics[diagnostics.inverted, :]
    nrow(d) == 0 && return (; n_shifted = 0, w_p10 = NaN, w_median = NaN, w_p90 = NaN,
                            w_mean = NaN, var_retention_median = NaN,
                            market_share_median = NaN)
    w = vcat(d.w_h, d.w_a)
    return (; n_shifted = nrow(d),
            w_p10 = quantile(w, 0.10),
            w_median = median(w),
            w_p90 = quantile(w, 0.90),
            w_mean = mean(w),
            var_retention_median = median(w .^ 2),
            market_share_median = 1.0 - median(w))
end


# %%
# ===================================================================
# 5. Derivative-market coherence audit
# ===================================================================
#
# The coherence claim is exact, and it is worth stating precisely what "exact"
# means. Every family below sums the SAME 12×12 grid: 1X2 partitions it by sign of
# (i − j), a half-integer totals line by whether i + j exceeds it, BTTS by whether
# both exceed zero. Three partitions of one tensor, so all three sums equal the
# grid's total mass — and that mass is 1 minus the truncated tail beyond 11 goals
# a side, which is O(1e-9) at football rates.
#
# So the audit reports the mass DEFICIT rather than asserting 1.0 exactly. A
# selection-level shift cannot produce this table at any tolerance; that is the
# comparison being made.

"A short, unique label per market in a list: `1x2`, `ou_25`, `btts`."
function market_family_label(m)
    m isa Data.Market1X2 && return "1x2"
    m isa Data.MarketBTTS && return "btts"
    m isa Data.MarketOverUnder && return "ou_" * replace(string(m.line), "." => "")
    return lowercase(string(typeof(m).name.name))
end

"""
    coherence_report(l, markets; max_goals, threaded) -> NamedTuple

Per-fixture sums of each market family, and the worst deviation from the shared
grid mass. `max_family_spread` is the number that matters: it is the largest
disagreement BETWEEN families on one fixture, and log-linear pooling makes it zero
to rounding.
"""
function coherence_report(l::AbstractPosteriorLatents, markets;
                          max_goals::Integer = 12, threaded::Bool = true)
    p = market_probabilities(l, markets; keep_draws = false,
                             max_goals = max_goals, threaded = threaded)
    nfix = length(p.match_ids)
    families = Vector{Tuple{String, Vector{Symbol}}}()
    for m in markets
        push!(families, (market_family_label(m), collect(market_keys(m))))
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
            family_names = [f[1] for f in families],
            max_deviation_from_one = worst_dev,
            max_family_spread = worst_spread,
            mean_grid_mass = mean(sums),
            frame = frame)
end


# %%
# ===================================================================
# 6. Proper scoring, per family and stratified by edge
# ===================================================================

"The 13 directions r02 audits: 1X2, four totals lines, BTTS."
l01_wide_markets() = Data.AbstractMarket[
    Data.Market1X2(),
    Data.MarketOverUnder(0.5),
    Data.MarketOverUnder(1.5),
    Data.MarketOverUnder(2.5),
    Data.MarketOverUnder(3.5),
    Data.MarketBTTS(),
]

"""
    l01_headline_selections()

The scope the published `m12` baseline was measured on — `Evaluation`'s
`DEFAULT_SCORED_MARKETS`, i.e. 1X2 + O/U 2.5 + BTTS. Gate 1 quotes LogLoss 0.64337
and ECE 0.0100 from that scope, so any comparison to those numbers must be filtered
to these selections; scoring the wide book against them would be a category error.
"""
l01_headline_selections() =
    Symbol[:home, :draw, :away, :over_25, :under_25, :btts_yes, :btts_no]

"The market families reported separately, in leaderboard order."
const L01_FAMILIES = [
    ("1x2",   Symbol[:home, :draw, :away]),
    ("ou_05", Symbol[:over_05, :under_05]),
    ("ou_15", Symbol[:over_15, :under_15]),
    ("ou_25", Symbol[:over_25, :under_25]),
    ("ou_35", Symbol[:over_35, :under_35]),
    ("btts",  Symbol[:btts_yes, :btts_no]),
]

"""
    edge_anchor(ctx) -> Dict{Tuple{Int,Symbol}, Float64}

The RAW model's signed edge `p_model − p_market` per (fixture, selection).

The stratified-LPD buckets are anchored on this rather than on each spec's own
edges. A calibrated model's edges shrink toward the book by construction, so
self-anchored buckets would move between grid points and two rows of the sweep
would be scoring different observations under the same column heading.
"""
function edge_anchor(ctx::L01_EVAL.EvaluationContext)
    rows = evaluation_rows(ctx)
    out = Dict{Tuple{Int,Symbol}, Float64}()
    sizehint!(out, length(rows))
    for r in rows
        isfinite(r.market_prob) || continue
        out[(r.match_id, r.selection)] = r.model_prob - r.market_prob
    end
    return out
end

"""
    edge_stratified_lpd(ctx; anchor, small, large) -> NamedTuple

Mean log predictive density over the posterior, split by the size of the model's
disagreement with the closing line.

Per-row LPD is `Evaluation.calc_lpd_samples!` verbatim — the framework's own
log-sum-exp over the draw vector — so these numbers sit on the same scale as
`compute_metric(LPD(), ctx)` rather than on a private one.
"""
function edge_stratified_lpd(ctx::L01_EVAL.EvaluationContext;
                             anchor::Union{Nothing, AbstractDict} = nothing,
                             small::Float64 = 0.02, large::Float64 = 0.05)
    rows = evaluation_rows(ctx)
    p = ctx.probs
    p.keep_draws || error(
        "edge_stratified_lpd needs posterior draws; build the context with a metric " *
        "whose `needs_draws` is true (LPD()).")

    log_liks = Vector{Float64}(undef, p.n_draws)
    expbuf = Vector{Float64}(undef, p.n_draws)

    model_all = Float64[]; market_all = Float64[]
    model_sm  = Float64[]; market_sm  = Float64[]
    model_lg  = Float64[]; market_lg  = Float64[]

    for r in rows
        isfinite(r.market_prob) || continue
        y = Float64(r.outcome)
        v = view(p.draws, :, r.fixture, r.column)
        lm = L01_EVAL.calc_lpd_samples!(log_liks, expbuf, v, y)
        lk = L01_EVAL.calc_lpd_scalar(r.market_prob, y)
        push!(model_all, lm); push!(market_all, lk)

        d = anchor === nothing ? (r.model_prob - r.market_prob) :
                                 get(anchor, (r.match_id, r.selection), NaN)
        isfinite(d) || continue
        ad = abs(d)
        if ad < small
            push!(model_sm, lm); push!(market_sm, lk)
        elseif ad > large
            push!(model_lg, lm); push!(market_lg, lk)
        end
    end

    _m(v) = isempty(v) ? NaN : mean(v)
    return (; n_all = length(model_all),
            lpd_all = _m(model_all), lpd_all_market = _m(market_all),
            n_small = length(model_sm),
            lpd_small = _m(model_sm), lpd_small_market = _m(market_sm),
            n_large = length(model_lg),
            lpd_large = _m(model_lg), lpd_large_market = _m(market_lg))
end

"""
    family_scores(model, spec_id, ctx; n_bins) -> DataFrame

LogLoss, Brier and ECE per market family, model beside the closing line. Long
format: one row per (model, spec, family), which is what a facetted plot wants and
what a 40-column flat row does not give.
"""
function family_scores(model::AbstractString, spec_id::AbstractString,
                       ctx::L01_EVAL.EvaluationContext; n_bins::Integer = 10)
    rows = NamedTuple[]
    for (name, sels) in L01_FAMILIES
        sc = evaluate_predictions(ctx; selections = sels, n_bins = n_bins)
        push!(rows, (
            model = String(model), spec = String(spec_id), family = name,
            n_obs = sc.model.n_obs,
            logloss = sc.model.logloss, market_logloss = sc.market.logloss,
            brier = sc.model.brier, market_brier = sc.market.brier,
            ece = sc.model.ece, market_ece = sc.market.ece,
            mce = sc.model.mce,
        ))
    end
    return DataFrame(rows)
end

"""
    score_calibration(model, spec, latents, odds_df, matches_df; ...) -> (row, families)

Every proper score one grid point earns, plus its per-family breakdown.

ONE context, TWO scopes. The context prices the wide 13-direction book with draws
retained; the headline scores are then the same context filtered to
`l01_headline_selections()`. `evaluation_rows` walks the odds frame in its own row
order and drops what was not priced, so that filter yields exactly the rows a
narrow context would have produced — one pricing pass instead of two, with no
change to the numbers.
"""
function score_calibration(model::AbstractString,
                           spec::GenerativeCalibrationSpec,
                           l::AbstractPosteriorLatents,
                           odds_df::AbstractDataFrame,
                           matches_df::AbstractDataFrame;
                           markets = l01_wide_markets(),
                           n_bins::Integer = 10,
                           anchor::Union{Nothing, AbstractDict} = nothing,
                           edge_small::Float64 = 0.02,
                           edge_large::Float64 = 0.05,
                           weights = nothing,
                           threaded::Bool = true)
    metrics = L01_EVAL.AbstractScoringRule[
        L01_EVAL.LogLoss(), PredictionScore(), L01_EVAL.LPD()]
    ctx = build_evaluation_context(l, odds_df, matches_df, metrics;
                                   markets = markets, threaded = threaded)

    wide = evaluate_predictions(ctx; n_bins = n_bins)
    head = evaluate_predictions(ctx; selections = l01_headline_selections(),
                                n_bins = n_bins)
    crps = L01_EVAL.compute_metric(L01_EVAL.CRPS(), ctx)
    strat = edge_stratified_lpd(ctx; anchor = anchor,
                                small = edge_small, large = edge_large)
    ws = weights === nothing ?
        (; n_shifted = -1, w_p10 = NaN, w_median = NaN, w_p90 = NaN, w_mean = NaN,
           var_retention_median = NaN, market_share_median = NaN) : weights

    row = (
        model = String(model),
        spec = spec_label(spec),
        method = String(spec.method),
        w_base = spec.w_base,
        sigma = spec.method === :static_geometric ? NaN : spec.sigma,
        w_max = spec.w_max,
        identity_spec = is_identity_spec(spec),

        head_logloss = head.model.logloss,
        head_market_logloss = head.market.logloss,
        head_delta_logloss = head.model.logloss - head.market.logloss,
        head_ece = head.model.ece,
        head_market_ece = head.market.ece,
        head_mce = head.model.mce,
        head_brier = head.model.brier,
        head_rps = head.model.rps,
        head_market_rps = head.market.rps,
        head_n_obs = head.model.n_obs,

        wide_logloss = wide.model.logloss,
        wide_market_logloss = wide.market.logloss,
        wide_ece = wide.model.ece,
        wide_market_ece = wide.market.ece,
        wide_brier = wide.model.brier,
        wide_market_brier = wide.market.brier,
        wide_n_obs = wide.model.n_obs,

        crps_all = crps.all.mean,
        crps_home = crps.home.mean,
        crps_away = crps.away.mean,

        lpd_all = strat.lpd_all,
        lpd_all_market = strat.lpd_all_market,
        lpd_small = strat.lpd_small,
        lpd_small_market = strat.lpd_small_market,
        n_small = strat.n_small,
        lpd_large = strat.lpd_large,
        lpd_large_market = strat.lpd_large_market,
        n_large = strat.n_large,

        n_shifted = ws.n_shifted,
        w_median = ws.w_median,
        w_p10 = ws.w_p10,
        w_p90 = ws.w_p90,
        var_retention_median = ws.var_retention_median,
    )

    return row, family_scores(model, spec_label(spec), ctx; n_bins = n_bins)
end


# %%
# ===================================================================
# 7. The sweep grid
# ===================================================================

"""
    sweep_specs(; w_bases, sigmas, methods, w_max) -> Vector{GenerativeCalibrationSpec}

The 2D surface of §3 of the work package, deduplicated where σ cannot matter:

  * `:static_geometric` ignores σ entirely — one spec per `w_base`;
  * `w_base = 1.0` is the identity in the two Gaussian forms whatever σ is — one
    spec, kept as the in-grid control.

Order is method-major so the results table reads as three surfaces.
"""
function sweep_specs(; w_bases = [0.25, 0.40, 0.55, 0.70, 0.85, 1.00],
                       sigmas = [0.15, 0.25, 0.35, 0.50, 0.75, 1.00],
                       methods = [:inverse_gaussian, :standard_gaussian, :static_geometric],
                       w_max::Real = 1.0)
    out = GenerativeCalibrationSpec[]
    for method in methods
        for wb in w_bases
            if method === :static_geometric
                push!(out, GenerativeCalibrationSpec(method = method, w_base = wb,
                                                     sigma = first(sigmas), w_max = w_max))
                continue
            end
            if wb == 1.0
                push!(out, GenerativeCalibrationSpec(method = method, w_base = wb,
                                                     sigma = first(sigmas), w_max = w_max))
                continue
            end
            for s in sigmas
                push!(out, GenerativeCalibrationSpec(method = method, w_base = wb,
                                                     sigma = s, w_max = w_max))
            end
        end
    end
    return out
end


# %%
# ===================================================================
# 8. The closing book
# ===================================================================

"""
    l01_betfair_closing_odds(ds) -> DataFrame

The Betfair exchange close, time-weighted over [−20 min, kick-off], de-vigged
within (match, market, line) and joined to the realised outcome.

This is `r62_betfair_closing_odds` / `r63_betfair_closing_odds` from experiment 06,
verbatim in construction: the sweep is compared against that suite's published
LogLoss and ECE, and a different price snapshot would make the comparison void.
"""
function l01_betfair_closing_odds(ds)
    raw = Data.summarize_odds(ds.betfair_odds, Data.TWAEstimator(); window = (-20.0, 0.0))
    odds = DataFrame(
        match_id = Int.(raw.match_id),
        market_name = String.(raw.market_name),
        market_line = Float64.(raw.market_line),
        selection = Symbol.(raw.selection),
        odds_close = Float64.(raw.odds),
    )
    filter!(row -> isfinite(row.odds_close) && row.odds_close > 1.0, odds)
    odds.prob_implied_close = 1.0 ./ odds.odds_close
    transform!(
        groupby(odds, [:match_id, :market_name, :market_line]),
        :prob_implied_close => (p -> p ./ sum(p)) => :prob_fair_close,
    )

    outcome_cols = [:match_id, :market_name, :market_line, :selection, :is_winner]
    winners = unique(select(ds.odds, outcome_cols))
    odds = leftjoin(odds, winners;
                    on = [:match_id, :market_name, :market_line, :selection])
    sort!(odds, [:match_id, :market_name, :market_line, :selection])
    return odds
end

"An inventory of what the closing book actually quotes, for the runner's G-A gate."
function odds_inventory(odds_df::AbstractDataFrame)
    g = combine(groupby(odds_df, [:market_name, :market_line, :selection]),
                nrow => :n_rows,
                :match_id => (x -> length(unique(x))) => :n_matches,
                :prob_fair_close => (p -> mean(skipmissing(p))) => :mean_fair)
    sort!(g, [:market_name, :market_line, :selection])
    return g
end


# %%
# ===================================================================
# 9. The tradeable book and per-direction attribution  (Phase 2)
# ===================================================================
#
# WHY O/U 0.5 IS NOT TRADEABLE ON THIS BOOK. `l01_betfair_closing_odds` de-vigs by
# normalising `prob_implied_close` within `(match, market, line)`, which is right on
# a two-sided quote and DEGENERATE on a one-sided one: a lone `over_05` row is
# normalised to `prob_fair_close = 1.0`. On the Scottish Lower archive the O/U 0.5
# ladder is quoted 982 over against 408 under — 574 fixtures one-sided — and the
# symptom is unmissable: the closing line's own LogLoss on that family is 1.31832
# against the model's 0.21098. Every other scored line is paired to within three
# rows.
#
# Staking against a fabricated fair price of 1.0 would manufacture an edge out of a
# de-vigging artefact, so `l01_tradeable_markets` drops the line and
# `l01_full_direction_markets` keeps it for the diagnostic arm that documents the
# exclusion. `l01_wide_markets` is left exactly as r01 ran it so that run stays
# reproducible from the committed code.

"The 11 directions Phase 2 stakes: 1X2, O/U 1.5/2.5/3.5, BTTS. No O/U 0.5."
l01_tradeable_markets() = Data.AbstractMarket[
    Data.Market1X2(),
    Data.MarketOverUnder(1.5),
    Data.MarketOverUnder(2.5),
    Data.MarketOverUnder(3.5),
    Data.MarketBTTS(),
]

"All 13 directions of the work package, including the O/U 0.5 line. Diagnostic only."
l01_full_direction_markets() = Data.AbstractMarket[
    Data.Market1X2(),
    Data.MarketOverUnder(0.5),
    Data.MarketOverUnder(1.5),
    Data.MarketOverUnder(2.5),
    Data.MarketOverUnder(3.5),
    Data.MarketBTTS(),
]

"A `BookSpec` over `markets`, otherwise the audited production settings."
function l01_book_spec(markets)
    return BookSpec(
        markets = Data.MarketConfig(Data.AbstractMarket[m for m in markets]),
        price = DeArb(),
        allocator = KellyLogUtility(),
        shrink = BayesianFootball.Portfolio.FractionalKelly(0.30),
        exec = ExecutionConfig(
            commission = PerBetCommission(0.02),
            budget = 0.99,
            min_selection_stake = 0.001,
        ),
    )
end

"""
    l01_policy_spec(trust) -> PolicySpec

The canonical production risk settings — `SlateDrawdown(23.0)`, `FixedCap(0.25)`,
`DailySlate()` — with only the trust model varying.

`FixedCap(0.25)`, not the 0.20 of experiment 06's flat research baseline: the
`CanonicalScottishLowerTrust` champion (+155.93%) was measured at 0.25, and a
comparison that moved the cap and the trust at once would not attribute either.
"""
l01_policy_spec(trust) = PolicySpec(
    trust = trust,
    risk = SlateDrawdown(23.0),
    cap = FixedCap(0.25),
    grouping = DailySlate(),
)

"""
    direction_ledger(result) -> DataFrame

The trade ledger aggregated to one row per betting direction, in units of the
INITIAL bankroll.

`trajectory.bets.stake` and `.pnl` are fractions of the bankroll at their own
slate, so summing them raw across a compounding backtest adds different units.
Every row here is rescaled by its slate's opening bankroll first — the same
correction `MARKET_LINE_EDA_REPORT.md` §0 applies, and the reason its currency
figures and a naive `sum(bets.pnl)` disagree.

| column | is |
|---|---|
| `flat_roi` | `mean(payoff)` — what a unit-stake bettor would have made |
| `kelly_roi` | turnover-weighted return, the rate the allocator actually earned |
| `capital_share` | this direction's share of all capital staked |
| `efficiency` | `kelly_roi` over the whole book's, so 1.00 is carrying its weight |
| `calibration` | realised win rate minus mean `p_model`; positive = model UNDER-rates |
"""
function direction_ledger(result)
    bets = result.trajectory.bets
    nrow(bets) == 0 && return DataFrame()
    open_of = Dict(d.date => d.bankroll_open for d in result.daily_states)
    b = copy(bets)
    scale = [get(open_of, d, NaN) for d in b.date]
    b.stake_units = b.stake .* scale
    b.pnl_units = b.pnl .* scale
    b.won = b.payoff .> 0.0

    total_stake = sum(skipmissing(b.stake_units))
    total_pnl = sum(skipmissing(b.pnl_units))
    book_kelly_roi = total_stake == 0.0 ? NaN : total_pnl / total_stake

    g = combine(groupby(b, [:family, :selection]),
                nrow => :n_bets,
                :won => mean => :win_rate,
                :odds => mean => :mean_odds,
                :p_model => mean => :p_model,
                :p_market => mean => :p_market,
                :stake_units => sum => :stake_units,
                :pnl_units => sum => :pnl_units,
                :payoff => mean => :mean_payoff)
    # `pnl == stake * payoff` and `settle_vector` has already deducted commission, so
    # both rates below are net. They differ only in the weighting: flat weights every
    # bet equally, Kelly weights it by the capital the allocator actually committed.
    g.flat_roi = 100 .* g.mean_payoff
    g.kelly_roi = 100 .* g.pnl_units ./ g.stake_units
    g.capital_share = 100 .* g.stake_units ./ total_stake
    g.efficiency = (g.pnl_units ./ g.stake_units) ./ book_kelly_roi
    g.edge = g.p_model .- g.p_market
    g.calibration = g.win_rate .- g.p_model
    sort!(g, :capital_share; rev = true)
    return g
end

"""
    tiered_trust_from_ledger(ledger; tier1, tier2, min_bets) -> (TieredTrust, DataFrame)

Fit a `TieredTrust` from one window's direction ledger.

The tier LADDER is the audited one — `CanonicalScottishLowerTrust`'s 0.35 / 0.25 /
0.00 — and only the ASSIGNMENT is refitted. That is deliberate: experiment 06 §2.1
established that `SlateDrawdown` makes absolute trust levels irrelevant and only the
1.4 : 1.0 conviction ratio bites, so re-fitting the levels would be re-deriving a
quantity already shown not to matter. The question Phase 2 asks is whether
calibration changes WHICH directions earn which tier.

    tier 1  kelly_roi > 0  and  efficiency >= 1.00  and  n_bets >= min_bets
    tier 2  kelly_roi > 0  and  efficiency >= 0.25  and  n_bets >= min_bets
    gated   otherwise

Causality is carried by the caller: fit on the selection window, score on the
window the rule never saw. A vector fitted and scored on the same slates is not a
result, and `MARKET_LINE_EDA_REPORT.md` §5.1 is the record of that rule class
failing when it was allowed to.
"""
function tiered_trust_from_ledger(ledger::AbstractDataFrame;
                                  tier1::Float64 = 0.35, tier2::Float64 = 0.25,
                                  min_bets::Int = 50)
    table = Dict{Tuple{String,Float64,Symbol},Float64}()
    rows = NamedTuple[]
    for r in eachrow(ledger)
        group, line = _direction_key(r.family, r.selection)
        w = if r.n_bets < min_bets || !isfinite(r.kelly_roi) || r.kelly_roi <= 0
            0.0
        elseif r.efficiency >= 1.00
            tier1
        elseif r.efficiency >= 0.25
            tier2
        else
            0.0
        end
        table[(group, line, r.selection)] = w
        push!(rows, (; family = r.family, group, line, selection = r.selection,
                     n_bets = r.n_bets, kelly_roi = r.kelly_roi,
                     efficiency = r.efficiency, tier = w))
    end
    return TieredTrust(table; default = 0.0), sort!(DataFrame(rows), :tier; rev = true)
end

"`(group, line)` for a `TieredTrust` key, recovered from the ledger's trust-key string."
function _direction_key(family::AbstractString, selection::Symbol)
    s = String(selection)
    (s in ("home", "draw", "away")) && return ("1x2", 0.0)
    startswith(s, "btts_") && return ("btts", 0.0)
    for prefix in ("over_", "under_")
        if startswith(s, prefix)
            d = s[(length(prefix) + 1):end]
            return ("over_under", parse(Float64, d[1:(end - 1)] * "." * d[end]))
        end
    end
    error("_direction_key: no market family owns selection :$selection (family \"$family\").")
end
