# ==============================================================================
# src/Calibration/rate_pool.jl — market inversion and the generative rate pool
# ==============================================================================
#
# Graduated from `current_development/calibration_generative_eda/l01_generative_calibrator.jl`
# (inversion and pool) and `l03_variance_schemes.jl` (the residual map and the anchor).
#
# ------------------------------------------------------------------------------
# THE FOUR STEPS
# ------------------------------------------------------------------------------
#
#   1. invert the de-vigged tradeable book back to (lambda_mkt_h, lambda_mkt_a) by
#      Nelder-Mead on `Features.DoublePoissonMarketFeature`;
#   2. measure the log-rate discrepancy delta = log median(lambda_model) - log lambda_mkt;
#   3. pool every posterior draw at the law's weight w(delta);
#   4. hand the shifted container to the SAME score-grid kernels, evaluator and portfolio
#      the raw container goes through.
#
# Step 4 is the point. Because every derivative price is read off one 12x12 score tensor
# built from the shifted rates, 1X2, every totals line and BTTS stay mutually coherent by
# construction — there is no way to shift P(Over 2.5) without moving P(Under 2.5) by the
# same amount, which is exactly the failure mode of the selection-level `BasicLogitShift`
# this module replaces.
#
# ------------------------------------------------------------------------------
# TWO KERNELS, AND WHY
# ------------------------------------------------------------------------------
#
# `_pool_kernel!` is the plain log-linear pool written the way the validated prototype
# wrote it — `exp(w*log(lambda) + (1-w)*log(lambda_mkt))`, one expression, no
# decomposition. `_dispersion_kernel!` is the general form `exp(c + kappa + M*u)` over
# centred residuals, which the pool is an algebraic special case of.
#
# The two agree to floating-point noise and NOT bit for bit, because `w*m + (1-w)*k + w*u`
# and `w*(m + u) + (1-w)*k` are the same number in the reals and not in Float64. The pool
# path exists so that the DEFAULT calibrator reproduces the transform every published
# figure in `calibration_generative_eda/README.md` was measured on, exactly, rather than
# to 1e-16. `test_calibration_v2.jl` T2 pins that against the prototype's own code.
#
# ------------------------------------------------------------------------------
# THE FALLBACK IS w = 1, BIT FOR BIT
# ------------------------------------------------------------------------------
#
# A fixture whose book cannot be inverted (no quotes, too few quotes, Nelder-Mead not
# converged, residual too large, implausible rates) is passed through UNCHANGED — the raw
# model draws, bit for bit, not a league-mean rate and not a dropped fixture. Dropping it
# would change which fixtures two calibrators score and make their rows incomparable;
# inventing a rate would price a fixture from inputs the pipeline declined to use.
#
# A side at w == 1.0 COPIES its raw draws instead of computing exp(1*log(lambda) + 0),
# because `exp(log(x)) != x` in Float64 and the identity calibrator has to reproduce the
# baseline EXACTLY or it is not a control.
# ==============================================================================


# ==============================================================================
# 1. MARKET INVERSION
# ==============================================================================

"""
    market_targets(odds_df) -> Dict{Int, Dict{Symbol, Float64}}

`match_id -> (selection -> prob_fair_close)`, built in one O(n) scan.

Rows with a missing or non-finite `prob_fair_close` are skipped rather than inverted
against; a duplicated (match, selection) keeps the LAST row, which is the sort order the
caller established.
"""
function market_targets(odds_df::AbstractDataFrame)
    hasproperty(odds_df, :prob_fair_close) || error(
        "market_targets: the odds frame carries no `prob_fair_close`. Build it with " *
        "`point_in_time_book`, which de-vigs within (match, market, line) AFTER checking " *
        "the market's selection set is complete.")
    out = Dict{Int, Dict{Symbol, Float64}}()
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
    invert_market_rates(cal, odds_df; match_ids = nothing)   -> Dict{Int, MarketRateFit}

Nelder-Mead the de-vigged book back to `(lambda_mkt_h, lambda_mkt_a)`, one fixture per
thread. `match_ids` restricts the work to the fixtures a latent container actually holds;
`nothing` inverts every fixture in the frame.

The result depends on the BOOK ONLY — not on the model and not on the calibrator's law —
so a sweep over laws computes it once and reuses it. Every refusal carries its reason;
see [`MarketInversionConfig`](@ref) for the four gates.
"""
function invert_market_rates(odds_df::AbstractDataFrame;
                             config::MarketInversionConfig = MarketInversionConfig(),
                             match_ids = nothing)
    targets = market_targets(odds_df)
    ids = match_ids === nothing ? collect(keys(targets)) : Int[Int(m) for m in match_ids]
    sort!(ids)

    init = Features.get_initial_guess(config.feature)
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
            theta -> begin
                P = Features.build_probability_matrix(cfg, theta, mg)
                sse = 0.0
                for line in ls
                    sse += Features._calculate_error(Val(line), P, tgts)
                end
                return sse + Features.compute_loss_penalty(cfg, theta)
            end
        end

        res = Optim.optimize(loss, copy(init), Optim.NelderMead())
        theta_hat = Optim.minimizer(res)
        sse = Optim.minimum(res)
        conv = Optim.converged(res)
        par = Features.extract_parameters(config.feature, theta_hat)
        lh = Float64(par.λ_home)
        la = Float64(par.λ_away)

        reason = if !conv
            "Nelder-Mead did not converge"
        elseif !isfinite(sse) || sse > config.max_sse
            @sprintf("residual SSE %.3e exceeds %.3e", sse, config.max_sse)
        elseif !(lo <= lh <= hi) || !(lo <= la <= hi)
            @sprintf("implied rates (%.3f, %.3f) outside [%.2f, %.2f]", lh, la, lo, hi)
        else
            ""
        end

        fits[n] = MarketRateFit(mid, lh, la, sse, length(tg), conv, isempty(reason), reason)
    end

    return Dict{Int, MarketRateFit}(f.match_id => f for f in fits)
end

invert_market_rates(cal::AbstractGenerativeRateCalibrator, odds_df::AbstractDataFrame;
                    match_ids = nothing) =
    invert_market_rates(odds_df; config = cal.inversion, match_ids = match_ids)

"The inversion, as a frame: one row per fixture, refusals carrying their reason."
function inversion_frame(rates::AbstractDict{Int, MarketRateFit})
    ids = sort!(collect(keys(rates)))
    return DataFrame(
        match_id        = ids,
        lambda_mkt_h    = [rates[i].lambda_home for i in ids],
        lambda_mkt_a    = [rates[i].lambda_away for i in ids],
        sse             = [rates[i].sse for i in ids],
        n_targets       = [rates[i].n_targets for i in ids],
        optim_converged = [rates[i].optim_converged for i in ids],
        accepted        = [rates[i].accepted for i in ids],
        reason          = [rates[i].reason for i in ids],
    )
end

"""
    inversion_refusals(frame_or_rates) -> Vector{Pair{String, Int}}

Refusal reasons and their counts, most frequent first.

A gate that refuses 40% of a book for one reason is a configuration problem; one that
refuses 2% across four reasons is the book being thin. The two look identical in a
coverage percentage.
"""
function inversion_refusals(frame::AbstractDataFrame)
    counts = Dict{String, Int}()
    for r in eachrow(frame)
        r.accepted && continue
        counts[r.reason] = get(counts, r.reason, 0) + 1
    end
    return sort!(collect(counts), by = last, rev = true)
end

inversion_refusals(rates::AbstractDict{Int, MarketRateFit}) =
    inversion_refusals(inversion_frame(rates))

"""
    inversion_coverage(rates, match_ids) -> NamedTuple

Coverage of an inversion over the fixtures a latent container holds, reported two ways
because only one of them measures dilution.

`coverage` is against EVERY fixture, and it counts a fixture the book never quoted as a
failure. Such a fixture contributes no scored observation either — the evaluator drops it
before any metric sees it — so it dilutes nothing. `coverage_quoted` is against the
fixtures that had a book to invert, and that is the number to read when asking how much
of a measured effect the refusals ate.
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


# ==============================================================================
# 2. THE POSTERIOR SHIFT
# ==============================================================================

"The diagnostic frame's schema, so both kernels fill exactly the same columns."
function _empty_rate_diagnostics(n::Int, ids::Vector{Int})
    return DataFrame(
        match_id          = copy(ids),
        inverted          = falses(n),
        reason            = fill("", n),
        lambda_model_h    = fill(NaN, n),
        lambda_model_a    = fill(NaN, n),
        lambda_mkt_h      = fill(NaN, n),
        lambda_mkt_a      = fill(NaN, n),
        delta_h           = fill(NaN, n),
        delta_a           = fill(NaN, n),
        w_h               = ones(Float64, n),
        w_a               = ones(Float64, n),
        kappa_h           = zeros(Float64, n),
        kappa_a           = zeros(Float64, n),
        var_retention_h   = ones(Float64, n),
        var_retention_a   = ones(Float64, n),
        var_retention_sup = ones(Float64, n),
        var_retention_tot = ones(Float64, n),
        rate_ratio_h      = ones(Float64, n),
        rate_ratio_a      = ones(Float64, n),
        lambda_shifted_h  = fill(NaN, n),
        lambda_shifted_a  = fill(NaN, n),
    )
end

"""
    calibrate_latents(cal, latents, rates) -> (latents2, diagnostics::DataFrame)

Apply the calibrator to every posterior draw of every fixture, and return the shifted
container beside the per-fixture diagnostic frame.

`latents2` is the SAME CONCRETE TYPE as `latents`. See [`CalibratedLatents`](@ref) for why
that matters and why there is no wrapper.

`delta` is measured against the posterior MEDIAN, not the mean: the pooling weight is a
statement about where the bulk of the posterior sits relative to the book, and the median
of a right-skewed rate posterior is the location that answers it.

# The diagnostic frame

One row per fixture, in the container's row order.

| column | is |
|---|---|
| `inverted` | did this fixture get an accepted market inversion |
| `reason` | why not, when it did not |
| `lambda_model_h/a` | the raw posterior MEDIAN rate |
| `lambda_mkt_h/a` | the inverted market rate |
| `delta_h/a` | `log(model median) - log(market)` |
| `w_h/a` | the applied pool weight |
| `kappa_h/a` | the Jensen anchor; identically zero on `PoolDispersion` |
| `var_retention_h/a` | retained posterior log-variance per side |
| `var_retention_sup/tot` | the same in the supremacy and totals bases |
| `rate_ratio_h/a` | calibrated draw-mean rate over the plain pool's — what the anchor sets to 1 |
| `lambda_shifted_h/a` | the calibrated posterior median rate |

The (s, t) retentions are reported separately because a map can retain 100% of the
per-side variance and still have changed what the totals market sees — any `M` with
off-diagonal terms does — and only those two columns show it.
"""
function calibrate_latents(cal::AbstractGenerativeRateCalibrator,
                           l::Models.CountLatents{Float64},
                           rates::AbstractDict{Int, MarketRateFit})
    ids = Models.latent_match_ids(l)
    nm, nd = size(l.λ_home)
    length(ids) == nm ||
        error("calibrate_latents: container is inconsistent — $(length(ids)) ids, $nm rows.")

    _assert_no_refusals(cal, ids, rates)

    diag = _empty_rate_diagnostics(nm, ids)
    lam_h = similar(l.λ_home)
    lam_a = similar(l.λ_away)

    if is_pool_map(cal.dispersion)
        _pool_kernel!(lam_h, lam_a, diag, cal, l, rates, ids, nm, nd)
    else
        nd > 1 || error(
            "calibrate_latents: a dispersion map other than PoolDispersion needs more " *
            "than one posterior draw to have a residual to map; got $nd.")
        _dispersion_kernel!(lam_h, lam_a, diag, cal, l, rates, ids, nm, nd)
    end

    return Models.CountLatents(copy(ids), lam_h, lam_a, l.observation_params), diag
end

"`fallback = :refuse` turns an un-invertible fixture into an error naming the fixtures."
function _assert_no_refusals(cal::AbstractGenerativeRateCalibrator, ids, rates)
    cal.fallback === :refuse || return nothing
    bad = Int[m for m in ids if !(haskey(rates, m) && rates[m].accepted)]
    isempty(bad) && return nothing
    shown = length(bad) <= 12 ? string(bad) : string(bad[1:12]) * " … (+$(length(bad) - 12))"
    error("calibrate_latents: calibrator \"$(cal.name)\" has fallback = :refuse and " *
          "$(length(bad)) of $(length(ids)) fixtures have no accepted market inversion: " *
          shown * ". Use fallback = :identity to pass them through raw.")
end

# ------------------------------------------------------------------------------
# 2.1 The pool kernel — the production transform, written as the prototype wrote it
# ------------------------------------------------------------------------------

function _pool_kernel!(lam_h, lam_a, diag, cal, l, rates, ids, nm::Int, nd::Int)
    med_h = diag.lambda_model_h
    med_a = diag.lambda_model_a
    w_h = diag.w_h
    w_a = diag.w_a
    c_h = zeros(Float64, nm)
    c_a = zeros(Float64, nm)
    shift_h = falses(nm)
    shift_a = falses(nm)

    buf = Vector{Float64}(undef, nd)
    @inbounds for i in 1:nm
        copyto!(buf, view(l.λ_home, i, :)); med_h[i] = median!(buf)
        copyto!(buf, view(l.λ_away, i, :)); med_a[i] = median!(buf)

        f = get(rates, ids[i], nothing)
        if f === nothing
            diag.reason[i] = "no market inversion attempted"
            continue
        end
        diag.reason[i] = f.reason
        f.accepted || continue

        diag.inverted[i] = true
        diag.lambda_mkt_h[i] = f.lambda_home
        diag.lambda_mkt_a[i] = f.lambda_away
        diag.delta_h[i] = log(med_h[i]) - log(f.lambda_home)
        diag.delta_a[i] = log(med_a[i]) - log(f.lambda_away)
        w_h[i] = calibration_weight(cal.law, diag.delta_h[i])
        w_a[i] = calibration_weight(cal.law, diag.delta_a[i])
        c_h[i] = (1.0 - w_h[i]) * log(f.lambda_home)
        c_a[i] = (1.0 - w_a[i]) * log(f.lambda_away)
        shift_h[i] = w_h[i] != 1.0
        shift_a[i] = w_a[i] != 1.0
    end

    @inbounds for k in 1:nd
        for i in 1:nm
            lam_h[i, k] = shift_h[i] ? exp(w_h[i] * log(l.λ_home[i, k]) + c_h[i]) :
                                       l.λ_home[i, k]
            lam_a[i, k] = shift_a[i] ? exp(w_a[i] * log(l.λ_away[i, k]) + c_a[i]) :
                                       l.λ_away[i, k]
        end
    end

    # Retentions and the predictive-rate ratio, measured on the draws rather than assumed
    # from `w^2`: the (s, t) bases are not diagonal in the per-side weights when
    # w_h != w_a, and `rate_ratio` is exactly 1 here only because this IS the pool.
    _fill_retentions!(diag, l, lam_h, lam_a, nm, nd)
    @inbounds for i in 1:nm
        diag.lambda_shifted_h[i] = shift_h[i] ?
            exp(w_h[i] * log(med_h[i]) + c_h[i]) : med_h[i]
        diag.lambda_shifted_a[i] = shift_a[i] ?
            exp(w_a[i] * log(med_a[i]) + c_a[i]) : med_a[i]
    end
    return nothing
end

# ------------------------------------------------------------------------------
# 2.2 The general kernel — location frozen, residual mapped, rate anchored
# ------------------------------------------------------------------------------
#
#     log lambda_new^(d) = c + kappa + [M * u^(d)]
#
# with c the SAME pooled location the pool kernel produces. Only M and kappa move, which
# is what makes a dispersion comparison an experiment about dispersion and not a re-run of
# the weight sweep with extra steps.

function _dispersion_kernel!(lam_h, lam_a, diag, cal, l, rates, ids, nm::Int, nd::Int)
    uh = Vector{Float64}(undef, nd)
    ua = Vector{Float64}(undef, nd)
    med = Vector{Float64}(undef, nd)
    anchored = cal.anchor === :pool_mean

    @inbounds for i in 1:nm
        for k in 1:nd
            uh[k] = log(l.λ_home[i, k])
            ua[k] = log(l.λ_away[i, k])
        end
        mh = mean(uh); ma = mean(ua)
        for k in 1:nd
            uh[k] -= mh
            ua[k] -= ma
        end

        f = get(rates, ids[i], nothing)
        if f === nothing || !f.accepted
            diag.reason[i] = f === nothing ? "no market inversion attempted" : f.reason
            for k in 1:nd
                lam_h[i, k] = l.λ_home[i, k]
                lam_a[i, k] = l.λ_away[i, k]
            end
            copyto!(med, view(l.λ_home, i, :)); diag.lambda_model_h[i] = median!(med)
            copyto!(med, view(l.λ_away, i, :)); diag.lambda_model_a[i] = median!(med)
            diag.lambda_shifted_h[i] = diag.lambda_model_h[i]
            diag.lambda_shifted_a[i] = diag.lambda_model_a[i]
            continue
        end
        diag.reason[i] = f.reason
        diag.inverted[i] = true
        diag.lambda_mkt_h[i] = f.lambda_home
        diag.lambda_mkt_a[i] = f.lambda_away

        copyto!(med, view(l.λ_home, i, :)); mdh = median!(med); diag.lambda_model_h[i] = mdh
        copyto!(med, view(l.λ_away, i, :)); mda = median!(med); diag.lambda_model_a[i] = mda
        diag.delta_h[i] = log(mdh) - log(f.lambda_home)
        diag.delta_a[i] = log(mda) - log(f.lambda_away)
        wh = calibration_weight(cal.law, diag.delta_h[i])
        wa = calibration_weight(cal.law, diag.delta_a[i])
        diag.w_h[i] = wh; diag.w_a[i] = wa
        ch = wh * mh + (1.0 - wh) * log(f.lambda_home)
        ca = wa * ma + (1.0 - wa) * log(f.lambda_away)

        m11, m12, m21, m22 = residual_map(cal.dispersion, wh, wa)

        # The Jensen anchor. `c` cancels out of the ratio, so this is a statement about
        # the residuals alone: how much hotter the new dispersion made the predictive
        # rate, relative to the plain pool's own.
        if anchored
            sA = 0.0; sB = 0.0; tA = 0.0; tB = 0.0
            for k in 1:nd
                sA += exp(wh * uh[k]); sB += exp(m11 * uh[k] + m12 * ua[k])
                tA += exp(wa * ua[k]); tB += exp(m21 * uh[k] + m22 * ua[k])
            end
            diag.kappa_h[i] = log(sA / sB)
            diag.kappa_a[i] = log(tA / tB)
        end

        kh = diag.kappa_h[i]; ka = diag.kappa_a[i]
        pool_h = 0.0; pool_a = 0.0
        for k in 1:nd
            nh = m11 * uh[k] + m12 * ua[k]
            na = m21 * uh[k] + m22 * ua[k]
            lam_h[i, k] = exp(ch + kh + nh)
            lam_a[i, k] = exp(ca + ka + na)
            pool_h += exp(ch + wh * uh[k])
            pool_a += exp(ca + wa * ua[k])
        end
        s_h = 0.0; s_a = 0.0
        for k in 1:nd
            s_h += lam_h[i, k]
            s_a += lam_a[i, k]
        end
        diag.rate_ratio_h[i] = s_h / pool_h
        diag.rate_ratio_a[i] = s_a / pool_a
        diag.lambda_shifted_h[i] = exp(ch + kh + m11 * (log(mdh) - mh) + m12 * (log(mda) - ma))
        diag.lambda_shifted_a[i] = exp(ca + ka + m21 * (log(mdh) - mh) + m22 * (log(mda) - ma))
    end

    _fill_retentions!(diag, l, lam_h, lam_a, nm, nd)
    return nothing
end

"""
    _fill_retentions!(diag, raw, lam_h, lam_a, nm, nd)

Retained log-variance in three bases — per side, supremacy `u_h - u_a`, totals
`u_h + u_a` — measured on the draws.

Residuals are re-centred here rather than reused from the kernel, because the pool kernel
never forms them. Raw second moments of exactly centred residuals ARE the variance, so no
mean correction is owed and none is applied.
"""
function _fill_retentions!(diag, raw, lam_h, lam_a, nm::Int, nd::Int)
    nd > 1 || return nothing
    uh = Vector{Float64}(undef, nd); ua = Vector{Float64}(undef, nd)
    vh = Vector{Float64}(undef, nd); va = Vector{Float64}(undef, nd)
    @inbounds for i in 1:nm
        for k in 1:nd
            uh[k] = log(raw.λ_home[i, k]); ua[k] = log(raw.λ_away[i, k])
            vh[k] = log(lam_h[i, k]);      va[k] = log(lam_a[i, k])
        end
        mh = mean(uh); ma = mean(ua); nh = mean(vh); na = mean(va)
        h_raw = 0.0; a_raw = 0.0; s_raw = 0.0; t_raw = 0.0
        h_new = 0.0; a_new = 0.0; s_new = 0.0; t_new = 0.0
        for k in 1:nd
            du = uh[k] - mh; da = ua[k] - ma
            dv = vh[k] - nh; dw = va[k] - na
            h_raw += du * du;         a_raw += da * da
            s_raw += (du - da)^2;     t_raw += (du + da)^2
            h_new += dv * dv;         a_new += dw * dw
            s_new += (dv - dw)^2;     t_new += (dv + dw)^2
        end
        diag.var_retention_h[i]   = h_raw > 0 ? h_new / h_raw : 1.0
        diag.var_retention_a[i]   = a_raw > 0 ? a_new / a_raw : 1.0
        diag.var_retention_sup[i] = s_raw > 0 ? s_new / s_raw : 1.0
        diag.var_retention_tot[i] = t_raw > 0 ? t_new / t_raw : 1.0
    end
    return nothing
end

# ------------------------------------------------------------------------------
# 2.3 Containers this transform is NOT defined for
# ------------------------------------------------------------------------------

function calibrate_latents(::AbstractGenerativeRateCalibrator, l::Models.SmileLatents, ::AbstractDict)
    error("""
    calibrate_latents: a smile container prices Over/Under from its OWN per-strike
    intensity `lambda_tot * phi(K)`, not from the (lambda_home, lambda_away) grid
    (`src/predictions/score_computation/smile_poisson.jl`). Pooling only the grid rates
    would leave a container whose 1X2 is calibrated and whose totals ladder is not — the
    exact derivative incoherence this module exists to remove, arrived at from the other
    direction.

    A smile calibrator has to pool `lambda_tot` against the market's totals intensity and
    derive what that does to `phi`. That derivation has not been done and is not in
    `calibration_generative_eda/` (README §8.12 records `observation_params === nothing`
    throughout). When it is, it belongs here as a `calibrate_latents(::…, ::SmileLatents, …)`
    method: the container contract already accommodates it, because a calibrator returns
    the type it was given.

    Got: $(typeof(l)).""")
end

function calibrate_latents(::AbstractGenerativeRateCalibrator, l::Models.RecombLatents, ::AbstractDict)
    error("""
    calibrate_latents: a recombination container decomposes the goal rate into three
    independent Poisson channels (open play, penalties, own goals). Pooling the TOTAL
    against a market rate is one equation in three unknowns, and any split of the shift
    across the channels would be a modelling assumption arrived at silently.

    Got: $(typeof(l)).""")
end


# ==============================================================================
# 3. SUMMARIES
# ==============================================================================

"""
    weight_summary(diagnostics) -> NamedTuple

Quantiles of the applied weights and of the retained posterior log-variance, over the
fixtures that were actually shifted.

The Ireland post-mortem turned on exactly these numbers: a median `w` of 0.41 means the
market supplied most of the location and 83% of the log-variance was destroyed, and no
headline score says that. Report it beside every result.
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
            var_retention_median = median(vcat(d.var_retention_h, d.var_retention_a)),
            market_share_median = 1.0 - median(w))
end

"""
    dispersion_summary(diagnostics) -> NamedTuple

Median retained log-variance in each basis, and the median predictive-rate inflation,
over the fixtures the calibrator actually touched.

`rate_ratio` is the number that separates "wider" from "hotter". A map with
`var_retention_tot = 1.00` and `rate_ratio = 1.04` did two things, and only its anchored
twin does one. On `PoolDispersion` `rate_ratio` is exactly 1 by construction.
"""
function dispersion_summary(diagnostics::AbstractDataFrame)
    d = diagnostics[diagnostics.inverted, :]
    nrow(d) == 0 && return (; n_shifted = 0, w_median = NaN, ret_side_median = NaN,
                            ret_sup_median = NaN, ret_tot_median = NaN,
                            rate_ratio_median = NaN, rate_ratio_p90 = NaN)
    rr = vcat(d.rate_ratio_h, d.rate_ratio_a)
    return (; n_shifted = nrow(d),
            w_median = median(vcat(d.w_h, d.w_a)),
            ret_side_median = median(vcat(d.var_retention_h, d.var_retention_a)),
            ret_sup_median = median(d.var_retention_sup),
            ret_tot_median = median(d.var_retention_tot),
            rate_ratio_median = median(rr),
            rate_ratio_p90 = quantile(rr, 0.90))
end

"""
    restrict_latents(l::CountLatents, keep_ids) -> CountLatents

The same posterior over a SUBSET of its fixtures, in the container's own row order.

Needed whenever a published threshold was measured on one fixture set and the run has
since been extended: scoring the extended set against those thresholds compares two
different questions, and restricting makes the comparison exact and the exclusion
explicit.

Fixtures in `keep_ids` that the container does not hold are ignored — the caller is
naming a filter, not asserting coverage.
"""
function restrict_latents(l::Models.CountLatents{Float64}, keep_ids)
    want = Set{Int}(Int(m) for m in keep_ids)
    ids = Models.latent_match_ids(l)
    rows = findall(i -> ids[i] in want, eachindex(ids))
    isempty(rows) && error("restrict_latents: no fixture of the container is in `keep_ids`.")
    obs = l.observation_params
    obs === nothing || error(
        "restrict_latents: this container carries observation parameters ($(typeof(obs))); " *
        "subsetting them is not defined here. Extend this method before using it on a " *
        "negative-binomial posterior.")
    return Models.CountLatents(ids[rows], l.λ_home[rows, :], l.λ_away[rows, :], nothing)
end
