# ==============================================================================
# l03 — Dispersion transforms for the generative rate pool
# ==============================================================================
#
# Loader. Definitions only. Pairs with `r05_variance_experiments.jl`.
#
# ------------------------------------------------------------------------------
# WHY THIS FILE EXISTS
# ------------------------------------------------------------------------------
#
# `l01`'s `calibrate_latents` applies the log-linear opinion pool
#
#     log λ̃⁽ˢ⁾ = w · log λ⁽ˢ⁾ + (1 − w) · log λ_mkt                          (A)
#
# to every posterior draw s. Two things happen at once and only one of them was
# ever asked for:
#
#   1. the LOCATION moves toward the market by (1 − w)·Δ, which is the point;
#   2. the DISPERSION contracts by w², which is a side effect of the algebra.
#
# At the T−25 optimum (`std_w0.40_s0.15`, median w ≈ 0.84) that is ~70% of the
# posterior log-variance retained; at the inverse form's optimum it is ~8.5%.
# Fractional Kelly is a function of the whole predictive distribution, so a
# contracted posterior stakes smaller, and README §7.7 recorded the open question:
# is the contraction load-bearing, or is it an artefact that a differently-shaped
# transform should not have?
#
# This file makes the location and the dispersion INDEPENDENTLY CONTROLLABLE so
# the question can be answered by experiment rather than by argument.
#
# ------------------------------------------------------------------------------
# THE DECOMPOSITION
# ------------------------------------------------------------------------------
#
# Write, per fixture i and side ∈ {h, a}, over the D posterior draws:
#
#     m    = mean_s( log λ⁽ˢ⁾ )              the raw posterior log-location
#     u⁽ˢ⁾ = log λ⁽ˢ⁾ − m                    the centred residual
#     Δ    = log( median_s λ⁽ˢ⁾ ) − log λ_mkt
#     w    = calibration_weight(spec, Δ)     l01's weight law, unchanged
#     c    = w·m + (1 − w)·log λ_mkt         the POOLED LOCATION (A) produces
#
# Then (A) is exactly
#
#     log λ̃⁽ˢ⁾ = c + w · u⁽ˢ⁾
#
# and every scheme in this file is
#
#     log λ̃⁽ˢ⁾ = c + κ + [ M · u⁽ˢ⁾ ]                                        (★)
#
# with the SAME c. The location law is frozen at l01's; only the 2×2 residual map
# M and the scalar anchor κ move. That is what makes the comparison an experiment
# about dispersion and not a re-run of the weight sweep with extra steps.
#
# ------------------------------------------------------------------------------
# WHY M IS 2×2 AND NOT TWO SCALARS
# ------------------------------------------------------------------------------
#
# Because the two things a football posterior is uncertain about are not λ_home
# and λ_away. They are
#
#     supremacy   s = u_h − u_a        which 1X2 and Asian handicaps price
#     totals      t = u_h + u_a        which O/U and BTTS price
#
# and the Jensen distortion `eda/README.md` Discovery 2 warns about lives almost
# entirely in t: the predictive zero-goal mass is E[e^(−Λ_h−Λ_a)] ≥ e^(−E[Λ_h+Λ_a)],
# and the gap is driven by Var(t), not by Var(s). A scheme that preserves supremacy
# dispersion while letting totals dispersion contract therefore keeps the Kelly
# sizing on the market the model is good at and declines the tail inflation on the
# market it is not. That hypothesis is unrepresentable with two per-side scalars
# and is the reason for the matrix.
#
# In the (s, t) basis a retention pair (ρ_s, ρ_t) maps back to
#
#     M = [ (ρ_t + ρ_s)/2   (ρ_t − ρ_s)/2 ]
#         [ (ρ_t − ρ_s)/2   (ρ_t + ρ_s)/2 ]
#
# which is symmetric, and reduces to ρ·I when ρ_s = ρ_t = ρ.
#
# ------------------------------------------------------------------------------
# WHY κ EXISTS — THE JENSEN ANCHOR
# ------------------------------------------------------------------------------
#
# Restoring dispersion at a fixed log-location RAISES the predictive rate:
# E[Λ] = E[e^(log Λ)] grows with Var(log Λ). So a variance-preserving scheme is not
# only wider than the pool, it is also HOTTER — it predicts more goals than the
# calibrated location said. Those are two different changes and they must not be
# allowed to hide in one another.
#
# `:pool_mean` anchoring removes the second by choosing κ so the draw-mean rate of
# the scheme equals the draw-mean rate scheme A would have produced:
#
#     κ = log mean_s exp(w·u⁽ˢ⁾)  −  log mean_s exp([M·u]⁽ˢ⁾)
#
# (c cancels, so this is a property of the residuals alone). It is computed on the
# draws, not from a log-normal formula, so it is exact whatever the posterior shape.
# An anchored scheme and its unanchored twin differ ONLY in first predictive moment,
# which is precisely the H2 contrast.
#
# ------------------------------------------------------------------------------
# WHAT SCHEME C ACTUALLY IS
# ------------------------------------------------------------------------------
#
# Not an arbitrary midpoint. If the posterior log-rate is N(m, σ²) and the market
# is an INDEPENDENT noisy observation of the same log-rate with precision τ, the
# conjugate posterior is
#
#     mean = (σ⁻²·m + τ·log λ_mkt)/(σ⁻² + τ),      variance = 1/(σ⁻² + τ) = w·σ²
#
# where w = σ⁻²/(σ⁻² + τ) is exactly the pool weight. So `C_sqrt` (ρ = √w, hence
# Var = w·σ²) is the coherent Bayesian update under the market-as-likelihood
# reading, and the log-linear pool's w²σ² is what you get instead when the same
# information is counted twice — once in the location, once in the width.
# `C_sqrt` is therefore the only scheme here with a generative justification rather
# than a mechanical one, and it is worth reading its result before B's.
#
# ------------------------------------------------------------------------------
# WHAT THIS FILE DOES NOT DO
# ------------------------------------------------------------------------------
#
# It does not touch `l01`. `calibrate_latents` stays exactly as r01–r04 ran it, and
# `A_pool` here must reproduce it to floating-point noise — `assert_scheme_a_matches`
# is the gate that says so, and the runner calls it before anything else is read.
# ==============================================================================

# %%
# ===================================================================
# 1. Packages
# ===================================================================

using BayesianFootball
using DataFrames
using Printf
using Statistics


# %%
# ===================================================================
# 2. The scheme
# ===================================================================

"""
    DispersionScheme(; id, family, residual_map, anchor, note)

One dispersion transform: the residual map `M` of (★) and its anchor.

| field | is |
|---|---|
| `id` | short, filename-safe, sort-stable. The key every CSV joins on |
| `family` | `"A"`…`"D"`, the work-package scheme letter this belongs to |
| `residual_map` | `(w_h, w_a) -> (m11, m12, m21, m22)`, the 2×2 map in ROW-MAJOR order |
| `anchor` | `:none`, or `:pool_mean` for the Jensen anchor described in the header |
| `note` | one line, printed by the runner and copied into the README table |

`residual_map` is called once per fixture, not once per draw, so a `Function` field
costs nothing measurable here and keeps each scheme readable as its own algebra.
"""
struct DispersionScheme
    id::String
    family::String
    residual_map::Function
    anchor::Symbol
    note::String

    function DispersionScheme(; id::AbstractString, family::AbstractString,
                               residual_map::Function, anchor::Symbol = :none,
                               note::AbstractString = "")
        anchor in (:none, :pool_mean) || throw(ArgumentError(
            "anchor must be :none or :pool_mean, got :$anchor"))
        return new(String(id), String(family), residual_map, anchor, String(note))
    end
end

"""
    sup_tot_map(ρ_s, ρ_t) -> (m11, m12, m21, m22)

The residual map that scales the supremacy residual `u_h − u_a` by `ρ_s` and the
totals residual `u_h + u_a` by `ρ_t`. See the header for the derivation.
"""
@inline function sup_tot_map(ρ_s::Float64, ρ_t::Float64)
    a = 0.5 * (ρ_t + ρ_s)
    b = 0.5 * (ρ_t - ρ_s)
    return (a, b, b, a)
end

"The mean pool weight of a fixture — the scalar a (ρ_s, ρ_t) pair contracts toward."
@inline mean_weight(w_h::Float64, w_a::Float64) = 0.5 * (w_h + w_a)

"""
    l03_schemes() -> Vector{DispersionScheme}

The benchmark set, in reporting order.

`A_pool` is the production baseline and must reproduce `l01.calibrate_latents`.
`D_tot` is a FALSIFICATION CONTROL, not a candidate: it preserves exactly the
dispersion `D_sup` discards and discards exactly the dispersion `D_sup` preserves.
If both help, the mechanism is "more variance is more Kelly" and the supremacy
story is wrong; if only `D_sup` helps, the asymmetry is real. Reporting `D_sup`
without it would be reporting half an experiment.
"""
function l03_schemes()
    return DispersionScheme[
        DispersionScheme(
            id = "A_pool", family = "A",
            residual_map = (wh, wa) -> (wh, 0.0, 0.0, wa),
            anchor = :none,
            note = "log-linear pool, Var = w²σ² (production baseline)"),

        DispersionScheme(
            id = "B_full", family = "B",
            residual_map = (wh, wa) -> (1.0, 0.0, 0.0, 1.0),
            anchor = :none,
            note = "full mean-shift variance preservation, Var = σ²"),

        DispersionScheme(
            id = "B_anch", family = "B",
            residual_map = (wh, wa) -> (1.0, 0.0, 0.0, 1.0),
            anchor = :pool_mean,
            note = "B_full with the predictive rate anchored to A_pool's"),

        DispersionScheme(
            id = "C_sqrt", family = "C",
            residual_map = (wh, wa) -> (sqrt(wh), 0.0, 0.0, sqrt(wa)),
            anchor = :none,
            note = "conjugate update, Var = wσ² (market as noisy observation)"),

        DispersionScheme(
            id = "D_sup", family = "D",
            residual_map = (wh, wa) -> sup_tot_map(1.0, mean_weight(wh, wa)),
            anchor = :none,
            note = "supremacy dispersion preserved, totals dispersion contracted"),

        DispersionScheme(
            id = "D_sup_anch", family = "D",
            residual_map = (wh, wa) -> sup_tot_map(1.0, mean_weight(wh, wa)),
            anchor = :pool_mean,
            note = "D_sup with the predictive rate anchored to A_pool's"),

        DispersionScheme(
            id = "D_tot", family = "D",
            residual_map = (wh, wa) -> sup_tot_map(mean_weight(wh, wa), 1.0),
            anchor = :none,
            note = "totals dispersion preserved, supremacy contracted (control)"),
    ]
end


# %%
# ===================================================================
# 3. The transform
# ===================================================================

"""
    apply_dispersion(l, rates, spec, scheme) -> (CountLatents, DataFrame)

Apply (★) to every posterior draw of every fixture, and return the transformed
container beside a per-fixture diagnostic frame.

The location `c` is `l01`'s and is identical across schemes; only `M` and `κ` move.

A fixture with no accepted market inversion copies its raw draws VERBATIM, exactly
as `calibrate_latents` does — the two functions must agree on the fallback set or
`A_pool` would not reproduce the baseline.

The diagnostic frame carries the retained log-variance in three bases — per side,
supremacy and totals — because a scheme can retain 100% of the per-side variance
and still have changed what the totals market sees (any `M` with off-diagonal
terms does), and only the (s, t) columns show it.
"""
function apply_dispersion(l::CountLatents{Float64},
                          rates::AbstractDict{Int, MarketRateFit},
                          spec::GenerativeCalibrationSpec,
                          scheme::DispersionScheme)
    ids = latent_match_ids(l)
    nm, nd = size(l.λ_home)
    length(ids) == nm || error(
        "apply_dispersion: container is inconsistent — $(length(ids)) ids, $nm rows.")
    nd > 1 || error("apply_dispersion: needs more than one posterior draw; got $nd.")

    λh = similar(l.λ_home)
    λa = similar(l.λ_away)

    inverted = falses(nm)
    w_h = ones(Float64, nm);        w_a = ones(Float64, nm)
    κ_h = zeros(Float64, nm);       κ_a = zeros(Float64, nm)
    ret_h = ones(Float64, nm);      ret_a = ones(Float64, nm)
    ret_sup = ones(Float64, nm);    ret_tot = ones(Float64, nm)
    rate_ratio_h = ones(Float64, nm); rate_ratio_a = ones(Float64, nm)
    sd_raw_tot = fill(NaN, nm);     sd_new_tot = fill(NaN, nm)

    uh = Vector{Float64}(undef, nd)
    ua = Vector{Float64}(undef, nd)
    med = Vector{Float64}(undef, nd)

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
            for k in 1:nd
                λh[i, k] = l.λ_home[i, k]
                λa[i, k] = l.λ_away[i, k]
            end
            continue
        end
        inverted[i] = true

        copyto!(med, view(l.λ_home, i, :));  mdh = median!(med)
        copyto!(med, view(l.λ_away, i, :));  mda = median!(med)
        wh = calibration_weight(spec, log(mdh) - log(f.λ_home))
        wa = calibration_weight(spec, log(mda) - log(f.λ_away))
        w_h[i] = wh; w_a[i] = wa
        ch = wh * mh + (1.0 - wh) * log(f.λ_home)
        ca = wa * ma + (1.0 - wa) * log(f.λ_away)

        m11, m12, m21, m22 = scheme.residual_map(wh, wa)

        # The Jensen anchor. `c` cancels out of the ratio, so this is a statement
        # about the residuals alone: how much hotter the new dispersion made the
        # predictive rate, relative to the pool's own.
        if scheme.anchor === :pool_mean
            sA = 0.0; sB = 0.0; tA = 0.0; tB = 0.0
            for k in 1:nd
                sA += exp(wh * uh[k]);  sB += exp(m11 * uh[k] + m12 * ua[k])
                tA += exp(wa * ua[k]);  tB += exp(m21 * uh[k] + m22 * ua[k])
            end
            κ_h[i] = log(sA / sB)
            κ_a[i] = log(tA / tB)
        end

        vh_raw = 0.0; va_raw = 0.0; vs_raw = 0.0; vt_raw = 0.0
        vh_new = 0.0; va_new = 0.0; vs_new = 0.0; vt_new = 0.0
        rate_h_raw = 0.0; rate_a_raw = 0.0; rate_h_new = 0.0; rate_a_new = 0.0
        for k in 1:nd
            nh = m11 * uh[k] + m12 * ua[k]
            na = m21 * uh[k] + m22 * ua[k]
            λh[i, k] = exp(ch + κ_h[i] + nh)
            λa[i, k] = exp(ca + κ_a[i] + na)

            vh_raw += uh[k]^2;              va_raw += ua[k]^2
            vs_raw += (uh[k] - ua[k])^2;    vt_raw += (uh[k] + ua[k])^2
            vh_new += nh^2;                 va_new += na^2
            vs_new += (nh - na)^2;          vt_new += (nh + na)^2
            rate_h_raw += exp(ch + wh * uh[k]); rate_a_raw += exp(ca + wa * ua[k])
            rate_h_new += λh[i, k];             rate_a_new += λa[i, k]
        end
        # Residuals are exactly centred by construction, so a raw second moment IS
        # the variance; no mean correction is owed and none is applied.
        ret_h[i] = vh_raw > 0 ? vh_new / vh_raw : 1.0
        ret_a[i] = va_raw > 0 ? va_new / va_raw : 1.0
        ret_sup[i] = vs_raw > 0 ? vs_new / vs_raw : 1.0
        ret_tot[i] = vt_raw > 0 ? vt_new / vt_raw : 1.0
        sd_raw_tot[i] = sqrt(vt_raw / nd)
        sd_new_tot[i] = sqrt(vt_new / nd)
        rate_ratio_h[i] = rate_h_new / rate_h_raw
        rate_ratio_a[i] = rate_a_new / rate_a_raw
    end

    out = CountLatents(copy(ids), λh, λa, l.observation_params)

    diagnostics = DataFrame(
        match_id            = copy(ids),
        scheme              = fill(scheme.id, nm),
        inverted            = collect(inverted),
        w_h                 = w_h,
        w_a                 = w_a,
        kappa_h             = κ_h,
        kappa_a             = κ_a,
        var_retention_h     = ret_h,
        var_retention_a     = ret_a,
        var_retention_sup   = ret_sup,
        var_retention_tot   = ret_tot,
        sd_log_total_raw    = sd_raw_tot,
        sd_log_total_new    = sd_new_tot,
        rate_ratio_h        = rate_ratio_h,
        rate_ratio_a        = rate_ratio_a,
    )
    return out, diagnostics
end

"""
    dispersion_summary(diagnostics) -> NamedTuple

Median retained log-variance in each basis, and the median predictive-rate
inflation, over the fixtures the scheme actually touched.

`rate_ratio` is the number that separates "wider" from "hotter". A scheme with
`var_retention_tot = 1.00` and `rate_ratio = 1.04` did two things, and only the
anchored twin does one.
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


# %%
# ===================================================================
# 4. The gate that keeps `A_pool` honest
# ===================================================================

"""
    assert_scheme_a_matches(l, rates, spec; rtol) -> NamedTuple

Refuse to proceed unless `A_pool` reproduces `l01.calibrate_latents` on every draw.

Every claim this stream makes is a difference between a scheme and `A_pool`, so if
`A_pool` is not the production pool the whole table is measuring something else.
The two code paths compute the same quantity by different algebra —
`exp(w·log λ + (1−w)·log λ_mkt)` against `exp(c + w·(log λ − m))` — so exact
equality is not owed and a relative tolerance is; `rtol = 1e-9` is roughly seven
orders of magnitude tighter than any difference that could move a fourth decimal
place in a score.
"""
function assert_scheme_a_matches(l::CountLatents{Float64},
                                 rates::AbstractDict{Int, MarketRateFit},
                                 spec::GenerativeCalibrationSpec;
                                 rtol::Float64 = 1e-9)
    ref, _ = calibrate_latents(l, rates, spec)
    got, _ = apply_dispersion(l, rates, spec,
                              first(s for s in l03_schemes() if s.id == "A_pool"))
    latent_match_ids(ref) == latent_match_ids(got) || error(
        "assert_scheme_a_matches: the two paths returned different fixture orders.")
    dh = maximum(abs.(got.λ_home .- ref.λ_home) ./ ref.λ_home)
    da = maximum(abs.(got.λ_away .- ref.λ_away) ./ ref.λ_away)
    worst = max(dh, da)
    worst <= rtol || error(@sprintf(
        "assert_scheme_a_matches: A_pool departs from l01.calibrate_latents by %.3e " *
        "relative (bound %.1e). The dispersion decomposition is wrong and no " *
        "downstream comparison is interpretable.", worst, rtol))
    return (; max_rel_home = dh, max_rel_away = da, rtol = rtol)
end


# %%
# ===================================================================
# 5. Jensen diagnostics — market-free, full fixture coverage
# ===================================================================
#
# These do NOT go through the odds book. `eda/README.md` Discovery 2 is a claim
# about the PREDICTIVE DISTRIBUTION, not about any price, and scoring it against
# the book would restrict it to the fixtures the book happens to quote and mix the
# question up with de-vigging. Computed here straight off the draws, at full
# fixture coverage, from the same double-Poisson mixture the score grid uses when
# `observation_params === nothing`.

"P(N ≤ n) for a Poisson of rate λ, n small — a direct sum, no `Distributions` call."
@inline function _poisson_cdf(λ::Float64, n::Int)
    term = exp(-λ)
    acc = term
    for k in 1:n
        term *= λ / k
        acc += term
    end
    return acc
end

"""
    jensen_diagnostics(l; label) -> DataFrame

Per fixture, the mixture predictive tail masses beside their PLUG-IN counterparts.

    mixture : E_s[ P(N ≤ n | Λ⁽ˢ⁾) ]      what the model actually predicts
    plugin  : P(N ≤ n | E_s[Λ])           what it would predict with no posterior width

The gap between them IS the Jensen term. `p_under_05` is `E[e^(−Λ)]` against
`e^(−E[Λ])` written out, which is Discovery 2's inequality in the units the O/U
0.5 ladder trades in. `p_over_35` is the same inequality from the other end, where
it points the other way.

Totals are the sum of the two arms DRAW-WISE, which is the only correct way to do
it: the arms share a posterior, so summing their marginal summaries would discard
the within-draw dependence the goalless-draw mass is most sensitive to.
"""
function jensen_diagnostics(l::CountLatents{Float64}; label::AbstractString = "")
    ids = latent_match_ids(l)
    nm, nd = size(l.λ_home)
    cols = (:p_under_05, :p_under_15, :p_under_25, :p_under_35)
    mix = [Vector{Float64}(undef, nm) for _ in cols]
    plug = [Vector{Float64}(undef, nm) for _ in cols]
    mean_tot = Vector{Float64}(undef, nm)
    sd_tot = Vector{Float64}(undef, nm)
    cv_tot = Vector{Float64}(undef, nm)

    @inbounds for i in 1:nm
        acc = zeros(Float64, 4)
        s1 = 0.0; s2 = 0.0
        for k in 1:nd
            t = l.λ_home[i, k] + l.λ_away[i, k]
            s1 += t; s2 += t * t
            e = exp(-t)
            c0 = e
            c1 = c0 + e * t
            c2 = c1 + e * t * t / 2
            c3 = c2 + e * t * t * t / 6
            acc[1] += c0; acc[2] += c1; acc[3] += c2; acc[4] += c3
        end
        for j in 1:4
            mix[j][i] = acc[j] / nd
        end
        mt = s1 / nd
        mean_tot[i] = mt
        v = max(s2 / nd - mt * mt, 0.0)
        sd_tot[i] = sqrt(v)
        cv_tot[i] = mt > 0 ? sqrt(v) / mt : NaN
        for (j, n) in enumerate(0:3)
            plug[j][i] = _poisson_cdf(mt, n)
        end
    end

    df = DataFrame(match_id = copy(ids), scheme = fill(String(label), nm),
                   mean_total = mean_tot, sd_total = sd_tot, cv_total = cv_tot)
    for (j, c) in enumerate(cols)
        df[!, c] = mix[j]
        df[!, Symbol(string(c), "_plugin")] = plug[j]
        df[!, Symbol(string(c), "_jensen")] = mix[j] .- plug[j]
    end
    df.p_over_35 = 1.0 .- df.p_under_35
    df.p_over_35_plugin = 1.0 .- df.p_under_35_plugin
    df.p_over_35_jensen = df.p_over_35 .- df.p_over_35_plugin
    return df
end

"""
    realised_totals(matches, ids) -> NamedTuple

The realised base rates the mixture predictions are answerable to, over exactly
the fixtures in `ids` that have a settled score.
"""
function realised_totals(matches::AbstractDataFrame, ids)
    want = Set{Int}(Int(i) for i in ids)
    tot = Int[]
    for r in eachrow(matches)
        Int(r.match_id) in want || continue
        (ismissing(r.home_score) || ismissing(r.away_score)) && continue
        (isfinite(r.home_score) && isfinite(r.away_score)) || continue
        push!(tot, Int(r.home_score) + Int(r.away_score))
    end
    n = length(tot)
    n == 0 && return (; n = 0, under_05 = NaN, under_15 = NaN, under_25 = NaN,
                      over_35 = NaN, mean_total = NaN)
    return (; n,
            under_05 = count(==(0), tot) / n,
            under_15 = count(<=(1), tot) / n,
            under_25 = count(<=(2), tot) / n,
            over_35 = count(>=(4), tot) / n,
            mean_total = mean(tot))
end

"""
    jensen_summary(diag, realised; label) -> NamedTuple

One row per scheme: mean predicted tail mass, the mean Jensen gap that produced
it, and the realised frequency it is answerable to.

`bias_*` is prediction minus realisation. It is the number H2 lives or dies on: if
restoring dispersion inflates `p_under_05` past the realised goalless-draw rate,
the extra mass is manufactured and Kelly will bet it.
"""
function jensen_summary(diag::AbstractDataFrame, realised::NamedTuple;
                        label::AbstractString = "")
    return (; scheme = String(label),
            n_fixtures = nrow(diag),
            mean_total = mean(diag.mean_total),
            mean_sd_total = mean(diag.sd_total),
            mean_cv_total = mean(diag.cv_total),
            p_under_05 = mean(diag.p_under_05),
            p_under_05_plugin = mean(diag.p_under_05_plugin),
            jensen_under_05 = mean(diag.p_under_05_jensen),
            realised_under_05 = realised.under_05,
            bias_under_05 = mean(diag.p_under_05) - realised.under_05,
            p_under_15 = mean(diag.p_under_15),
            jensen_under_15 = mean(diag.p_under_15_jensen),
            realised_under_15 = realised.under_15,
            bias_under_15 = mean(diag.p_under_15) - realised.under_15,
            p_under_25 = mean(diag.p_under_25),
            realised_under_25 = realised.under_25,
            bias_under_25 = mean(diag.p_under_25) - realised.under_25,
            p_over_35 = mean(diag.p_over_35),
            jensen_over_35 = mean(diag.p_over_35_jensen),
            realised_over_35 = realised.over_35,
            bias_over_35 = mean(diag.p_over_35) - realised.over_35)
end


# %%
# ===================================================================
# 6. Book spec with a movable Kelly fraction  (Scheme E)
# ===================================================================

"""
    l03_book_spec(markets; kelly_fraction) -> BookSpec

`l01_book_spec` with `FractionalKelly` exposed.

Scheme E is the counter-hypothesis (H3): leave the posterior alone and turn the
two knobs that were built for exactly this — the Kelly fraction in the book spec
and `SlateDrawdown`'s λ in the policy spec. Those knobs live in two different
specs, `l01_book_spec` hard-codes the first at 0.30, and a sweep that could only
move λ would be testing half of H3. Everything else is `l01_book_spec` verbatim.
"""
function l03_book_spec(markets; kelly_fraction::Real = 0.30)
    0.0 < kelly_fraction <= 1.0 || throw(ArgumentError(
        "kelly_fraction must be in (0, 1]: $kelly_fraction"))
    return BookSpec(
        markets = Data.MarketConfig(Data.AbstractMarket[m for m in markets]),
        price = DeArb(),
        allocator = KellyLogUtility(),
        shrink = BayesianFootball.Portfolio.FractionalKelly(Float64(kelly_fraction)),
        exec = ExecutionConfig(
            commission = PerBetCommission(0.02),
            budget = 0.99,
            min_selection_stake = 0.001,
        ),
    )
end
