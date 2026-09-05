# ==============================================================================
# src/Calibration/types.jl — the Layer-2 calibrator hierarchy
# ==============================================================================
#
# Two hierarchies live in this file and they do NOT meet.
#
#   §1–§6  GENERATIVE RATE CALIBRATION (v2, production). Shifts the posterior
#          log-INTENSITY, so every derivative price is read off one score tensor
#          and 1X2 / totals / BTTS cannot disagree.
#
#   §9     THE LEGACY SELECTION-LEVEL HIERARCHY (`AbstractLayerTwoModel`,
#          `BasicLogitShift`). Retained, deprecated, and kept working.
#
# Design record: `docs/architecture/rfc_layer2_calibration_v2.md`.
# Evidence:     `current_development/calibration_generative_eda/README.md`.
# ==============================================================================


# ==============================================================================
# 1. THE ROOTS
# ==============================================================================

"""
    AbstractCalibrator

Anything that maps a fitted posterior onto a calibrated one **of the same kind**.

The contract is deliberately narrow, and it is a contract about the RETURN TYPE:

    calibrate_latents(cal, latents, rates) -> (latents2, diagnostics::DataFrame)

`latents2` must be the *same concrete container type* as `latents`. That is what lets the
calibrated posterior go through `Predictions`, `Evaluation` and `Portfolio` with no new
methods anywhere — see §3.1 of the RFC for the `SmileLatents` dispatch that a wrapper
type would have silently broken.
"""
abstract type AbstractCalibrator end

"""
    AbstractGenerativeRateCalibrator <: AbstractCalibrator

A calibrator that works on the generative intensity: it inverts a tradeable book back to
`(lambda_mkt_h, lambda_mkt_a)` and pools the model's log-rate draws with it.

Coherence across derivative markets is then STRUCTURAL — 1X2, every totals line and BTTS
are three partitions of one 12x12 tensor built from the pooled rates — rather than a
property that has to be checked after the fact. `coherence_report` checks it anyway,
once, because "structural" is a claim about code.
"""
abstract type AbstractGenerativeRateCalibrator <: AbstractCalibrator end


# ==============================================================================
# 2. THE LOCATION LAW — how far the pool moves toward the book
# ==============================================================================
#
# One method, `calibration_weight(law, delta)`, returning the MODEL's share of the
# log-linear pool at log-rate discrepancy delta. Adding a law is a struct plus that
# method; no existing file changes.

function _check_unit(what::AbstractString, name::AbstractString, v::Real)
    (isfinite(v) && 0.0 <= v <= 1.0) ||
        throw(ArgumentError("$what: $name must be in [0, 1], got $v."))
    return nothing
end

"""
    AbstractCalibrationWeightLaw

`w(delta)` in `[0, 1]` — the weight the MODEL keeps when its log-rate disagrees with the
market's by `delta`. `w = 1` is the raw model, `w = 0` is the book.

| law | `w(0)` | `w(+-inf)` | reads as |
|---|---|---|---|
| [`InverseGaussianLaw`](@ref)  | `w_base` | 1.0      | trust the market on noise, the model on structural edges |
| [`StandardGaussianLaw`](@ref) | `w_max`  | `w_base` | optimiser's-curse shrinkage of extreme claims |
| [`StaticGeometricLaw`](@ref)  | `w`      | `w`      | a constant pool — the control for whether delta-dependence buys anything |

**Which form wins depends on the sharpness of the book being pooled with, and that is
measured rather than assumed.** Against the Betfair close the standard (shrinkage) form
wins on LogLoss; against the softer T-25 book the inverse (conviction) form wins,
reversing the ordering (stream README §7.3). Parameters do not transfer between price
instants — a close-fitted pick gives up 0.0015-0.0020 LogLoss when moved to T-25 rates —
which is why [`GenerativeRateCalibrator`](@ref) records the instant it was fitted at.
"""
abstract type AbstractCalibrationWeightLaw end

"""
    StandardGaussianLaw(; w_base = 0.40, sigma = 0.15, w_max = 1.0)

    w = w_base + (w_max - w_base) * exp(-delta^2 / (2 sigma^2))

Optimiser's-curse shrinkage: full trust in the model where it agrees with the book,
falling to `w_base` where it disagrees most. The T-25 ECE optimum on both Scottish Lower
candidates (`m12` 0.0093, `m05` 0.0094, against the T-25 book's own 0.0183).

`w_max >= w_base` is required and checked: `w_max` is the PEAK at `delta = 0` and
`w_base` the FLOOR at large `|delta|`, and a pair the other way round is a different law
written by accident.
"""
struct StandardGaussianLaw <: AbstractCalibrationWeightLaw
    w_base::Float64
    sigma::Float64
    w_max::Float64

    function StandardGaussianLaw(; w_base::Real = 0.40, sigma::Real = 0.15,
                                  w_max::Real = 1.0)
        _check_unit("StandardGaussianLaw", "w_base", w_base)
        _check_unit("StandardGaussianLaw", "w_max", w_max)
        sigma > 0.0 ||
            throw(ArgumentError("StandardGaussianLaw: sigma must be positive, got $sigma."))
        w_max >= w_base || throw(ArgumentError(
            "StandardGaussianLaw needs w_max >= w_base (w_max is the peak at delta = 0, " *
            "w_base the floor at large |delta|); got w_max=$w_max, w_base=$w_base."))
        return new(Float64(w_base), Float64(sigma), Float64(w_max))
    end
end

"""
    InverseGaussianLaw(; w_base = 0.25, sigma = 0.35)

    w = w_base + (1 - w_base) * (1 - exp(-delta^2 / (2 sigma^2)))

Conviction: the book supplies most of the location where the two nearly agree, the model
keeps its own where it disagrees hard. The Ireland Premier form, which failed on Scottish
Lower **at closing prices** and wins there **at T-25** (README §7.3) — the cleanest
available demonstration that a calibration law is a statement about a price instant and
not about a league.

Retains about 8.5% of the posterior log-variance at its T-25 optimum. That contraction is
reported per fixture (`var_retention_h` / `_a`) and, measured, costs nothing: restoring
up to 11.8x of it moves staked exposure by 0.4% (README §8.5).
"""
struct InverseGaussianLaw <: AbstractCalibrationWeightLaw
    w_base::Float64
    sigma::Float64

    function InverseGaussianLaw(; w_base::Real = 0.25, sigma::Real = 0.35)
        _check_unit("InverseGaussianLaw", "w_base", w_base)
        sigma > 0.0 ||
            throw(ArgumentError("InverseGaussianLaw: sigma must be positive, got $sigma."))
        return new(Float64(w_base), Float64(sigma))
    end
end

"""
    StaticGeometricLaw(; w = 0.40)

    w = w   (constant)

The control. If a delta-dependent law does not beat this, the delta-dependence bought
nothing — and at T-25 it very nearly does not: `sta_w0.40` scores LogLoss 0.63089 against
the inverse form's 0.63064 on `m12` (README §7.3).

`w = 1.0` is the IDENTITY calibrator, and is the in-grid control every sweep should
carry: it prices the raw model through the identical code path and must reproduce the
uncalibrated baseline exactly.
"""
struct StaticGeometricLaw <: AbstractCalibrationWeightLaw
    w::Float64

    function StaticGeometricLaw(; w::Real = 0.40)
        _check_unit("StaticGeometricLaw", "w", w)
        return new(Float64(w))
    end
end

"""
    calibration_weight(law, delta) -> Float64

The model's share of the log-linear pool at log-rate discrepancy `delta`.

A non-finite `delta` returns `1.0` — the raw model — because a discrepancy that could not
be measured is not evidence for moving toward the book.
"""
@inline calibration_weight(law::StaticGeometricLaw, ::Float64) = law.w

@inline function calibration_weight(law::InverseGaussianLaw, delta::Float64)
    isfinite(delta) || return 1.0
    g = exp(-(delta * delta) / (2.0 * law.sigma * law.sigma))
    return law.w_base + (1.0 - law.w_base) * (1.0 - g)
end

@inline function calibration_weight(law::StandardGaussianLaw, delta::Float64)
    isfinite(delta) || return 1.0
    g = exp(-(delta * delta) / (2.0 * law.sigma * law.sigma))
    return law.w_base + (law.w_max - law.w_base) * g
end

"`true` when the law is `w == 1` everywhere — the identity map on every fixture."
is_identity_law(l::StaticGeometricLaw) = l.w == 1.0
is_identity_law(l::InverseGaussianLaw) = l.w_base == 1.0
is_identity_law(l::StandardGaussianLaw) = l.w_base == 1.0 && l.w_max == 1.0

"A short, filename-safe, sort-stable label for a law. The key every artefact joins on."
law_label(l::StaticGeometricLaw) = @sprintf("sta_w%.2f", l.w)
law_label(l::InverseGaussianLaw) = @sprintf("inv_w%.2f_s%.2f", l.w_base, l.sigma)
law_label(l::StandardGaussianLaw) =
    l.w_max == 1.0 ? @sprintf("std_w%.2f_s%.2f", l.w_base, l.sigma) :
                     @sprintf("std_w%.2f_s%.2f_m%.2f", l.w_base, l.sigma, l.w_max)


# ==============================================================================
# 3. THE DISPERSION MAP — what happens to the posterior WIDTH
# ==============================================================================
#
# Decompose, per fixture and side, over the D draws:
#
#     m       = mean_d log lambda^(d)              raw posterior log-location
#     u^(d)   = log lambda^(d) - m                 centred residual
#     c       = w*m + (1 - w)*log lambda_mkt       the POOLED LOCATION, frozen
#
# Every map in this section is
#
#     log lambda_new^(d) = c + kappa + [M * u^(d)]                           (STAR)
#
# and differs only in the 2x2 residual map M and the anchor kappa. M is 2x2 rather than
# two scalars because the two things a football posterior is uncertain about are not
# lambda_h and lambda_a — they are SUPREMACY (u_h - u_a), which 1X2 prices, and TOTALS
# (u_h + u_a), which O/U and BTTS price. A map that preserves one and contracts the other
# is unrepresentable with per-side scalars.
#
# READ THIS BEFORE CHANGING THE DEFAULT. Dispersion was measured and it is not the lever:
# restoring up to 11.8x the posterior log-variance moves staked exposure by 0.4%
# (README §8.5), and the Jensen tail term it drives is at most 0.0012 of probability
# against a P(under 0.5) bias of +0.0065 that no map changes (§8.4). `PoolDispersion` is
# the default because it reproduces the validated production transform bit for bit.

"""
    AbstractDispersionMap

The residual map `M` of (STAR). One method:

    residual_map(map, w_h, w_a) -> (m11, m12, m21, m22)      # ROW-MAJOR

called once per fixture, not once per draw.
"""
abstract type AbstractDispersionMap end

"""
    PoolDispersion()

`M = diag(w_h, w_a)` — the plain log-linear opinion pool,
`log lambda_new = w*log lambda + (1 - w)*log lambda_mkt`, whose posterior log-variance is
`w^2 * sigma^2`.

**The default, and the only map that reproduces the prototype's `calibrate_latents` bit
for bit.** The `w^2` contraction is a side effect of the algebra rather than a modelling
choice, and it was measured not to matter (README §8.5). `test_calibration_v2.jl` T2 pins
the equivalence against `l01_generative_calibrator.jl` itself.
"""
struct PoolDispersion <: AbstractDispersionMap end

"""
    PreservedDispersion()

`M = I` — a pure mean shift. The location moves; the posterior keeps its full width
`sigma^2`.

With `anchor = :pool_mean` this is scheme `B_anch`, which won README §8.6 on both models,
both location laws, at the production budget, at matched risk and out of sample — by 1.5
points of return. Read that ordering honestly: `B_full` and `B_anch` have IDENTICAL
dispersion in every basis and differ only by 0.13-0.50% of predictive rate, so the anchor
is doing all the work and the variance preservation none. 1.5 points over 99 slates is a
consistent ordering, not a resolved one.
"""
struct PreservedDispersion <: AbstractDispersionMap end

"""
    ConjugateDispersion()

`M = diag(sqrt(w_h), sqrt(w_a))` — posterior log-variance `w * sigma^2`.

If the market is an INDEPENDENT noisy observation of the same log-rate with precision
`tau`, the conjugate posterior has mean `(s^-2 m + tau log lambda_mkt)/(s^-2 + tau)` —
exactly the pool's location, with `w = s^-2/(s^-2 + tau)` — and variance `w sigma^2`. The
pool's `w^2 sigma^2` is what you get instead when the same information is counted twice,
once in the location and once in the width. So this is the only map here with a
generative justification rather than a mechanical one.

Measured: indistinguishable from `PoolDispersion` on every score and every portfolio
number to three significant figures (README §8.6). The double-counting is real and, on
this data, unmeasurable — which is worth knowing before anyone rebuilds the calibrator to
fix it.
"""
struct ConjugateDispersion <: AbstractDispersionMap end

"""
    SupremacyDispersion(; rho_s = 1.0, rho_t = :pool)

Scale the supremacy residual `u_h - u_a` by `rho_s` and the totals residual `u_h + u_a`
by `rho_t`, which maps back to the symmetric

    M = [ (rho_t + rho_s)/2   (rho_t - rho_s)/2 ]
        [ (rho_t - rho_s)/2   (rho_t + rho_s)/2 ]

`rho_t = :pool` (the default) means "this fixture's mean pool weight `(w_h + w_a)/2`",
which is scheme `D_sup`: keep the Kelly sizing on the market the model is good at,
decline the Jensen tail inflation on the market it is not.

`SupremacyDispersion(rho_s = :pool, rho_t = 1.0)` is `D_tot`, the FALSIFICATION CONTROL —
it preserves exactly what `D_sup` discards. It reversed the expected sign twice: the
proper-score cost of dispersion tracks SUPREMACY, while the Jensen tail term tracks
TOTALS. Half that experiment would have supported a confident and wrong conclusion, which
is why the control ships beside the candidate.
"""
struct SupremacyDispersion <: AbstractDispersionMap
    rho_s::Union{Float64, Symbol}
    rho_t::Union{Float64, Symbol}

    function SupremacyDispersion(; rho_s = 1.0, rho_t = :pool)
        for (nm, v) in (("rho_s", rho_s), ("rho_t", rho_t))
            if v isa Symbol
                v === :pool || throw(ArgumentError(
                    "SupremacyDispersion: the only symbolic $nm is :pool (this fixture's " *
                    "mean pool weight); got :$v."))
            else
                (isfinite(v) && v >= 0.0) || throw(ArgumentError(
                    "SupremacyDispersion: $nm must be a finite non-negative number or " *
                    ":pool, got $v."))
            end
        end
        return new(rho_s isa Symbol ? rho_s : Float64(rho_s),
                   rho_t isa Symbol ? rho_t : Float64(rho_t))
    end
end

"""
    residual_map(map, w_h, w_a) -> (m11, m12, m21, m22)

The 2x2 residual map of (STAR), row-major, for one fixture's pool weights.
"""
@inline residual_map(::PoolDispersion, w_h::Float64, w_a::Float64) = (w_h, 0.0, 0.0, w_a)
@inline residual_map(::PreservedDispersion, ::Float64, ::Float64) = (1.0, 0.0, 0.0, 1.0)
@inline residual_map(::ConjugateDispersion, w_h::Float64, w_a::Float64) =
    (sqrt(w_h), 0.0, 0.0, sqrt(w_a))

@inline function residual_map(m::SupremacyDispersion, w_h::Float64, w_a::Float64)
    wbar = 0.5 * (w_h + w_a)
    rho_s = m.rho_s isa Symbol ? wbar : m.rho_s
    rho_t = m.rho_t isa Symbol ? wbar : m.rho_t
    a = 0.5 * (rho_t + rho_s)
    b = 0.5 * (rho_t - rho_s)
    return (a, b, b, a)
end

"""
    is_pool_map(map) -> Bool

`true` when the map is `diag(w_h, w_a)` for every weight pair — i.e. the plain pool.

Load-bearing: the Jensen anchor is identically zero on such a map (the two draw-sums it
compares are the same sum), so `calibrate_latents` skips computing it, and
`is_identity_calibrator` can promise bit-identity.
"""
is_pool_map(::PoolDispersion) = true
is_pool_map(::AbstractDispersionMap) = false

"A short, filename-safe label for a dispersion map."
map_label(::PoolDispersion) = "A_pool"
map_label(::PreservedDispersion) = "B_full"
map_label(::ConjugateDispersion) = "C_sqrt"
function map_label(m::SupremacyDispersion)
    f(v) = v isa Symbol ? "pool" : @sprintf("%.2f", v)
    return "D_s" * f(m.rho_s) * "_t" * f(m.rho_t)
end


# ==============================================================================
# 4. MARKET INVERSION — the config and its acceptance gates
# ==============================================================================

"""
    L2_INVERSION_LINES

The inversion line set: 1X2, BTTS, and **one symbol per totals line**.

`Features.LINES` lists both sides of every totals line, and
`Features._calculate_error(Val(:over_25), ...)` already scores the over AND the under
key, so a both-sides tuple counts each totals line twice while counting 1X2 once. That is
a silent reweighting of the objective, invisible in any result table.
"""
const L2_INVERSION_LINES = (:result_1x2, :btts, :over_05, :over_15, :over_25, :over_35)

"""
    MarketInversionConfig(; feature, max_goals, min_targets, max_sse, lambda_bounds)

The Nelder-Mead inversion of a de-vigged book back to `(lambda_mkt_h, lambda_mkt_a)`, and
its four acceptance gates. A fit failing any gate is REFUSED BY NAME and its fixture
falls back to `w = 1`.

| gate | refuses |
|---|---|
| `min_targets` | a book too thin to identify two rates (3 = a bare 1X2) |
| convergence | Nelder-Mead that did not converge |
| `max_sse` | a converged optimum that still does not reproduce the book |
| `lambda_bounds` | a rate outside anything a football match produces |

`Features.fit_market_implied_parameters` performs this fit already but returns only the
minimiser — no residual, no convergence flag, no target count. Every one of those is a
gate here, so the optimisation is rebuilt around the SAME `Features` primitives
(`build_probability_matrix`, `_calculate_error`, `get_initial_guess`,
`compute_loss_penalty`, `extract_parameters`) rather than as a second implementation of
the objective.
"""
Base.@kwdef struct MarketInversionConfig
    feature::Features.AbstractMarketFeatureConfig =
        Features.DoublePoissonMarketFeature(lines = L2_INVERSION_LINES)
    max_goals::Int = 10
    min_targets::Int = 3
    max_sse::Float64 = 5.0e-3
    lambda_bounds::Tuple{Float64, Float64} = (0.05, 6.0)
end

"""
    MarketRateFit

One fixture's inverted book.

`accepted = false` carries the `reason`, and a refused fit is never read by
`calibrate_latents` — the fixture passes through raw. The reason is kept rather than
collapsed to a boolean because "40% of the book refused for one reason" is a
configuration problem and "2% across four reasons" is the book being thin, and the two
look identical in a coverage percentage.
"""
struct MarketRateFit
    match_id::Int
    lambda_home::Float64
    lambda_away::Float64
    sse::Float64
    n_targets::Int
    optim_converged::Bool
    accepted::Bool
    reason::String
end


# ==============================================================================
# 5. THE CALIBRATOR
# ==============================================================================

"""
    GenerativeRateCalibrator(; name, law, dispersion = PoolDispersion(),
                               anchor = :pool_mean, fallback = :identity,
                               inversion = MarketInversionConfig(),
                               book_as_of_minutes = -25.0)

One calibration recipe: a location law, a dispersion map, an anchor, a fallback protocol,
the inversion gates, and **the price instant it was fitted at**.

| field | is |
|---|---|
| `name` | the registry key. Stable, and it should say the instant: `"scot_lower_t25_inv"` |
| `law` | [`AbstractCalibrationWeightLaw`](@ref) — where the pooled location sits |
| `dispersion` | [`AbstractDispersionMap`](@ref) — what happens to the posterior width |
| `anchor` | `:pool_mean` or `:none` — the Jensen anchor, see below |
| `fallback` | `:identity` (raw draws pass through) or `:refuse` (throw, naming the fixtures) |
| `inversion` | [`MarketInversionConfig`](@ref) |
| `book_as_of_minutes` | minutes to kick-off of the book this calibrator was fitted against; `0.0` for a close |

# The anchor

`kappa` is chosen so the calibrated draw-mean rate equals the rate the plain pool would
have produced:

    kappa = log mean_d exp(w * u^(d)) - log mean_d exp([M u]^(d))

computed on the draws, not from a log-normal formula, so it is exact whatever the
posterior shape. It exists because restoring dispersion at a fixed log-location makes the
container not only WIDER but HOTTER (`E[L] = E[exp(log L)]` grows with `Var(log L)`), and
those are two different changes that must not hide in one another.

**On `PoolDispersion` it is identically zero** — `M = diag(w_h, w_a)` makes the two sums
the same sum — so the default calibrator is rate-anchored by construction and the field
costs nothing. It becomes load-bearing the moment a non-pool map is selected: on
`PreservedDispersion` the anchor is worth +8.8 points of Over 2.5 flat ROI and +1.5 of
return, both measured against its unanchored twin (README §8.9, §8.6). The default is
`:pool_mean` so that switching the map cannot silently lose it.

# `book_as_of_minutes` is not decoration

Calibration parameters DO NOT TRANSFER between price instants, and the winning functional
form flips with the sharpness of the book (README §7.3). `calibrate_fit` asserts the book
it is handed was built at this instant, so a T-25 calibrator meeting a closing book is an
error at the call site rather than a plausible wrong number three weeks later.

# Example

```julia
cal = GenerativeRateCalibrator(
    name = "scot_lower_t25_inv",
    law  = InverseGaussianLaw(w_base = 0.25, sigma = 0.35),
    book_as_of_minutes = -25.0,
)
```
"""
struct GenerativeRateCalibrator{L <: AbstractCalibrationWeightLaw,
                                D <: AbstractDispersionMap} <: AbstractGenerativeRateCalibrator
    name::String
    law::L
    dispersion::D
    anchor::Symbol
    fallback::Symbol
    inversion::MarketInversionConfig
    book_as_of_minutes::Float64

    function GenerativeRateCalibrator(; name::AbstractString,
                                       law::L,
                                       dispersion::D = PoolDispersion(),
                                       anchor::Symbol = :pool_mean,
                                       fallback::Symbol = :identity,
                                       inversion::MarketInversionConfig = MarketInversionConfig(),
                                       book_as_of_minutes::Real = -25.0) where {L, D}
        isempty(strip(name)) &&
            throw(ArgumentError("GenerativeRateCalibrator: name must not be empty."))
        anchor in (:pool_mean, :none) || throw(ArgumentError(
            "GenerativeRateCalibrator: anchor must be :pool_mean or :none, got :$anchor."))
        fallback in (:identity, :refuse) || throw(ArgumentError(
            "GenerativeRateCalibrator: fallback must be :identity (pass the raw draws " *
            "through) or :refuse (throw, naming the fixtures), got :$fallback. There is " *
            "deliberately no :market or :league_mean — inventing a rate would price a " *
            "fixture from inputs the pipeline declined to use."))
        isfinite(book_as_of_minutes) || throw(ArgumentError(
            "GenerativeRateCalibrator: book_as_of_minutes must be finite; it is the " *
            "instant this calibrator was fitted at, and calibration parameters do not " *
            "transfer between instants."))
        return new{L, D}(String(name), law, dispersion, anchor, fallback, inversion,
                         Float64(book_as_of_minutes))
    end
end

"""
    is_identity_calibrator(cal) -> Bool

`true` when the calibrator is the identity map on every fixture, so the calibrated
container must be BIT-IDENTICAL to the raw one.

Needs both: `w == 1` everywhere, and the pool map (which is `diag(1,1) = I` at `w = 1`).
The anchor cannot break it — it is zero on the pool map by construction.

This is the in-grid control every sweep should carry, and `calibrate_latents`
short-circuits to a copy on it rather than computing `exp(1*log(lambda) + 0)`, because
`exp(log(x)) != x` in Float64 and a control that reproduces the baseline to 1e-16 is not
a control.
"""
is_identity_calibrator(cal::GenerativeRateCalibrator) =
    is_identity_law(cal.law) && is_pool_map(cal.dispersion)

"""
    calibrator_label(cal) -> String

A short, filename-safe, sort-stable label: `"inv_w0.25_s0.35__A_pool__anch"`. The key
every artefact and every results table joins on.
"""
function calibrator_label(cal::GenerativeRateCalibrator)
    base = law_label(cal.law) * "__" * map_label(cal.dispersion)
    return cal.anchor === :pool_mean ? base * "__anch" : base
end

"""
    calibrator_hash(cal) -> String

SHA-256 of the calibrator's canonical description. The identity that
`calibration_runs.calibrator_hash` stores, and the key that says two runs used the same
recipe.

The name is EXCLUDED. Two calibrators that differ only in what they are called are the
same transform, and a rename should not orphan a run's lineage.
"""
function calibrator_hash(cal::GenerativeRateCalibrator)
    canonical = join(("calibrator", string(cal.law), string(cal.dispersion),
                      string(cal.anchor), string(cal.fallback), string(cal.inversion),
                      @sprintf("%.6f", cal.book_as_of_minutes)), "")
    return bytes2hex(SHA.sha256(canonical))
end

"""
    calibrator_json(cal) -> Dict{String, Any}

A JSON-safe, SQL-queryable description. Every scalar a calibration sweep would want to
`GROUP BY` is a top-level key rather than a string to be parsed back out.
"""
function calibrator_json(cal::GenerativeRateCalibrator)
    law = Dict{String, Any}("type" => string(nameof(typeof(cal.law))))
    for f in fieldnames(typeof(cal.law))
        law[string(f)] = getfield(cal.law, f)
    end
    disp = Dict{String, Any}("type" => string(nameof(typeof(cal.dispersion))))
    for f in fieldnames(typeof(cal.dispersion))
        v = getfield(cal.dispersion, f)
        disp[string(f)] = v isa Symbol ? string(v) : v
    end
    inv = cal.inversion
    return Dict{String, Any}(
        "type" => "calibrator",
        "kind" => string(nameof(typeof(cal))),
        "name" => cal.name,
        "label" => calibrator_label(cal),
        "law" => law,
        "law_label" => law_label(cal.law),
        "dispersion" => disp,
        "dispersion_label" => map_label(cal.dispersion),
        "anchor" => string(cal.anchor),
        "fallback" => string(cal.fallback),
        "book_as_of_minutes" => cal.book_as_of_minutes,
        "identity" => is_identity_calibrator(cal),
        "inversion" => Dict{String, Any}(
            "feature" => string(inv.feature),
            "max_goals" => inv.max_goals,
            "min_targets" => inv.min_targets,
            "max_sse" => inv.max_sse,
            "lambda_bounds" => collect(inv.lambda_bounds),
        ),
    )
end

function Base.show(io::IO, ::MIME"text/plain", cal::GenerativeRateCalibrator)
    print(io, "GenerativeRateCalibrator(\"", cal.name, "\")")
    print(io, "\n  law         : ", cal.law)
    print(io, "\n  dispersion  : ", cal.dispersion, "  [", map_label(cal.dispersion), "]")
    print(io, "\n  anchor      : :", cal.anchor,
          is_pool_map(cal.dispersion) ? "  (identically zero on the pool map)" : "")
    print(io, "\n  fallback    : :", cal.fallback)
    print(io, "\n  book instant: T", @sprintf("%+.0f", cal.book_as_of_minutes), " min")
    print(io, "\n  label       : ", calibrator_label(cal))
    is_identity_calibrator(cal) &&
        print(io, "\n  IDENTITY — reproduces the raw container bit for bit")
    return nothing
end

Base.show(io::IO, cal::GenerativeRateCalibrator) =
    print(io, "GenerativeRateCalibrator(\"", cal.name, "\", ", calibrator_label(cal), ")")


# ==============================================================================
# 6. CALIBRATED LATENTS — an alias, deliberately, and not a wrapper
# ==============================================================================

"""
    CalibratedLatents

The container a calibrator returns: **the same concrete type it was given**.
`CountLatents` in, `CountLatents` out; `SmileLatents` in, `SmileLatents` out.

This is an alias for `Models.AbstractPosteriorLatents` rather than a wrapper struct, and
that is a decision with a reason. `src/Portfolio/pricing.jl` dispatches on the container:

    if l isa Models.SmileLatents
        ...  # build a SmileScoreGrid so O/U prices off the model's own lambda_tot * phi(K)

A wrapper fails that `isa`. The book would still build, still price and still stake — and
it would silently **de-smile** the totals ladder, pricing every O/U line off the score
grid instead of the per-strike intensity. That is exactly the failure
`src/predictions/score_computation/smile_poisson.jl` warns about, it produces a plausible
number, and no test that checks "the portfolio ran" would catch it.

So there is no wrapper, no new kernel method, and no `src/Portfolio/` change. The
provenance a wrapper would have carried lives on [`CalibratedFit`](@ref), which is the
object a caller actually holds.
"""
const CalibratedLatents = Models.AbstractPosteriorLatents


# ==============================================================================
# 9. LEGACY — the selection-level hierarchy, deprecated and kept working
# ==============================================================================
#
# `BasicLogitShift` moves P(Home) and P(Over 2.5) with independently fitted GLM offsets,
# so the shifted board is not a valid scoreline distribution and derivative markets can
# contradict each other: P*(over_25) + P*(under_25) != 1 by construction. Everything
# above exists to replace it. It is retained, warned, and left working, because a legacy
# script that stops running is a worse outcome than one that prints a line.

"""
    AbstractLayerTwoModel

**DEPRECATED.** The selection-level shift hierarchy. Use
[`GenerativeRateCalibrator`](@ref), which shifts the generative intensity and is coherent
across derivative markets by construction.
"""
abstract type AbstractLayerTwoModel end

"""
    CalibrationConfig

**DEPRECATED.** The recipe for a legacy Layer-2 recalibration experiment.
"""
Base.@kwdef struct CalibrationConfig <: AbstractLayerTwoModelConfig
    name::String
    model::AbstractLayerTwoModel

    prob_col::Symbol = :prob_mean
    # --- Backtesting / Windowing Controls ---
    min_history_splits::Integer = 4          # Wait for 4 periods of L1 OOS data before applying L2
    max_history_splits::Integer = 0          # 0 = expanding window. >0 = rolling window

    min_market_train::Integer = 10
    min_market_predict::Integer = 0
end

"""
    CalibrationResults

**DEPRECATED.** Target market -> split id -> fitted legacy shift model.
"""
struct CalibrationResults
    config::CalibrationConfig
    fitted_models_history::Dict{Symbol, Dict{String, Any}}
end

"""
    fit_calibrator(model::AbstractLayerTwoModel, data, config)

**DEPRECATED.** Trains a legacy selection-level shift on historical PPDs and outcomes.
"""
function fit_calibrator(model::AbstractLayerTwoModel, data::DataFrame, config::CalibrationConfig)
    error("Not implemented for $(typeof(model))")
end

"**DEPRECATED.** Applies a learned selection-level shift to unobserved PPDs."
function apply_shift(fitted_model, new_data::DataFrame)
    error("Not implemented for $(typeof(fitted_model))")
end

function apply_calibration(fitted_model, new_data::DataFrame)
    error("Not implemented for $(typeof(fitted_model))")
end
