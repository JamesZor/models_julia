#=
l10_calibrate.jl — make the in-play book staking-ready: level correction + per-family
probability calibration.

WHAT THE OOS RUN FOUND, AND WHY IT NEEDS TWO SEPARATE FIXES:

  1. **The level is ~5% hot.** At kickoff on the held-out season the model priced 2.920 goals
     against 2.782 realised; the pregame engine alone accounts for +3.2%. It is ONE SCALAR,
     and it drifts between seasons (24/25 realised 2.653 goals/match, 25/26 2.782, +4.9%).
     Fixed by scaling the remaining intensity → `fit_level` + `with_level`.

  2. **The two families miscalibrate in OPPOSITE directions.** 1X2 is over-dispersed
     (p 0.754 → y 0.702, and under-confident below 0.5); O/U is under-dispersed
     (p 0.139 → y 0.101, p 0.861 → y 0.899). **A single global shrinkage would fix one and
     break the other**, so calibration is per family → `fit_family_calibrator`.

The probability fix is logit-scale temperature scaling, `logit p' = a + b·logit p`, fitted by
logistic regression per family. `b < 1` shrinks toward 0.5 (cures over-confidence), `b > 1`
sharpens. Two parameters per family — deliberately the smallest thing that can express
"too spread" vs "too compressed" without inventing structure.

**RENORMALISATION IS NOT OPTIONAL.** Calibrating selections independently breaks the
adding-up constraints — home+draw+away must sum to 1, and each O/U pair and the BTTS pair
must sum to 1. `apply_calibration` renormalises within each market group afterwards. Skipping
it hands the staking layer a book with fake arbitrage in it.

FITTING DISCIPLINE. The calibrator is fitted on the NHPP's TRAINING season and applied to the
held-out season, so it is a clean temporal holdout for the calibrator itself. The NHPP's own
in-sample optimism on its training season was measured at ~0.001 nats (r08), which is small
enough that cross-fitting the NHPP underneath the calibrator would not move these numbers.

Requires l09.
=#

using DataFrames, Statistics, GLM

const CAL_SELS = vcat([:home, :draw, :away, :btts_yes, :btts_no],
                      [Symbol("over_$(k)5") for k in 0:3],
                      [Symbol("under_$(k)5") for k in 0:3])

famof(s::Symbol) = s in (:home, :draw, :away) ? :x12 :
                   s in (:btts_yes, :btts_no) ? :btts : :ou

# ---------------------------------------------------------------------------
# 1. Level
# ---------------------------------------------------------------------------

"""
    fit_level(m, ms_list; t0 = 0.0) -> Float64

The scalar that makes expected total goals match realised total goals over `ms_list`.
Apply with `with_level(m, fit_level(...))`.

Estimate it on RECENT matches, not on all history: the level drifts, and a stale estimate is
what produced the +5% error in the first place.
"""
function fit_level(m::InGameModel, ms_list; t0::Float64 = 0.0, n_pairs::Int = 400)
    pred = 0.0; real = 0.0
    for ms in ms_list
        st = ingame_state(ms, ms.mid, t0)
        rem = ingame_remaining(m, ms.mid, t0; gh = st.gh, ga = st.ga, rh = st.rh, ra = st.ra,
                               n_pairs = n_pairs)
        pred += mean(rem.Λ_h) + mean(rem.Λ_a)
        real += count(g -> g.t >= t0, ms.goals)
    end
    return pred <= 0 ? 1.0 : real / pred
end

"""
    rolling_level(m, ms_sorted, i; window = 120) -> Float64

Level from the `window` matches immediately BEFORE index `i` — the deployable form, which
tracks drift instead of freezing last season's number. Falls back to 1.0 until there is
enough history.
"""
function rolling_level(m::InGameModel, ms_sorted, i::Int; window::Int = 120)
    lo = max(1, i - window); hi = i - 1
    hi - lo < 40 && return 1.0
    return fit_level(m, view(ms_sorted, lo:hi))
end

# ---------------------------------------------------------------------------
# 2. Per-family probability calibration
# ---------------------------------------------------------------------------

_logit(p) = log(clamp(p, 1e-6, 1 - 1e-6) / (1 - clamp(p, 1e-6, 1 - 1e-6)))
_sigm(x) = 1 / (1 + exp(-clamp(x, -30.0, 30.0)))

struct FamilyCalibrator
    a::Dict{Symbol, Float64}
    b::Dict{Symbol, Float64}
    n::Dict{Symbol, Int}
end

"Identity calibrator — the null to race against."
identity_calibrator(fams = (:x12, :ou, :btts)) =
    FamilyCalibrator(Dict(f => 0.0 for f in fams), Dict(f => 1.0 for f in fams),
                     Dict(f => 0 for f in fams))

"""
    fit_family_calibrator(rows; min_n = 200) -> FamilyCalibrator

`rows` needs `:family`, `:p` (model probability) and `:y` (0/1 outcome). Families with fewer
than `min_n` rows keep the identity map rather than fitting two parameters to noise — BTTS is
the thin one and it is better left alone than over-fitted.
"""
function fit_family_calibrator(rows::DataFrame; min_n::Int = 200)
    a = Dict{Symbol, Float64}(); b = Dict{Symbol, Float64}(); n = Dict{Symbol, Int}()
    for g in groupby(rows, :family)
        f = g.family[1]; n[f] = nrow(g)
        if nrow(g) < min_n || length(unique(g.y)) < 2
            a[f] = 0.0; b[f] = 1.0; continue
        end
        df = DataFrame(y = Float64.(g.y), z = _logit.(g.p))
        fit = glm(@formula(y ~ z), df, Binomial(), LogitLink())
        c = coef(fit); a[f] = c[1]; b[f] = c[2]
    end
    return FamilyCalibrator(a, b, n)
end

calibrate_p(cal::FamilyCalibrator, fam::Symbol, p::Real) =
    _sigm(get(cal.a, fam, 0.0) + get(cal.b, fam, 1.0) * _logit(p))

# Which selections must sum to 1 together.
_group_of(s::Symbol) = (t = String(s);
    startswith(t, "over_")  ? "ou_" * t[6:end] :
    startswith(t, "under_") ? "ou_" * t[7:end] :
    s in (:btts_yes, :btts_no) ? "btts" : "x12")

"""
    apply_calibration(cal, book) -> Dict{Symbol, Float64}

Calibrate each selection, then RENORMALISE within each market group so the adding-up
constraints survive (see the header — skipping this puts fake arbitrage in the book).
"""
function apply_calibration(cal::FamilyCalibrator, book::AbstractDict)
    out = Dict{Symbol, Float64}()
    for (s, p) in book
        out[s] = calibrate_p(cal, famof(s), p)
    end
    groups = Dict{String, Vector{Symbol}}()
    for s in keys(out); push!(get!(groups, _group_of(s), Symbol[]), s); end
    for (_, sels) in groups
        length(sels) < 2 && continue
        tot = sum(out[s] for s in sels)
        tot > 0 && for s in sels; out[s] /= tot; end
    end
    return out
end

"""
    calibrated_book(m, cal, mid, t; gh, ga, rh, ra, ...) -> Dict{Symbol, Float64}

The staking-ready price: model → level-corrected intensity → per-family calibration →
renormalised book. This is the function a staking layer should call.
"""
function calibrated_book(m::InGameModel, cal::FamilyCalibrator, mid::Integer, t::Real;
                         gh::Int = 0, ga::Int = 0, rh::Int = 0, ra::Int = 0,
                         n_pairs::Int = 2000, rng = Xoshiro(Int(mid)))
    apply_calibration(cal, ingame_book(m, mid, t; gh = gh, ga = ga, rh = rh, ra = ra,
                                       n_pairs = n_pairs, rng = rng))
end

# ---------------------------------------------------------------------------
# 3. Scoring helper shared by the calibration runners
# ---------------------------------------------------------------------------

"""
    score_book(m, ms_list; cal = nothing, checkpoints) -> DataFrame

One row per (match, checkpoint, selection) with the (optionally calibrated) probability and
the realised outcome.
"""
function score_book(m::InGameModel, ms_list; cal::Union{Nothing, FamilyCalibrator} = nothing,
                    checkpoints = (0.0, 30.0, 60.0, 80.0), n_pairs::Int = 800)
    rows = DataFrame(mid = Int[], t0 = Float64[], sel = Symbol[], p = Float64[], y = Int[])
    for ms in ms_list
        fh = count(g -> g.home, ms.goals); fa = count(g -> !g.home, ms.goals)
        tr = truth_of(fh, fa)
        for t0 in checkpoints
            st = ingame_state(ms, ms.mid, t0)
            bk = ingame_book(m, ms.mid, t0; gh = st.gh, ga = st.ga, rh = st.rh, ra = st.ra,
                             n_pairs = n_pairs, rng = Xoshiro(ms.mid + Int(t0)))
            cal === nothing || (bk = apply_calibration(cal, bk))
            for s in CAL_SELS
                haskey(bk, s) && haskey(tr, s) && push!(rows, (ms.mid, t0, s, bk[s], tr[s]))
            end
        end
    end
    rows.family = famof.(rows.sel)
    return rows
end

nll(p, y) = -(y * log(clamp(p, 1e-9, 1)) + (1 - y) * log(clamp(1 - p, 1e-9, 1)))

"Per-family log-loss / Brier, and the match-clustered paired t against a baseline."
function score_summary(rows::DataFrame)
    combine(groupby(rows, :family), nrow => :n,
        [:p, :y] => ((p, y) -> mean(nll.(p, y))) => :logloss,
        [:p, :y] => ((p, y) -> mean((p .- y) .^ 2)) => :brier)
end

function paired_vs(rows_a::DataFrame, rows_b::DataFrame)
    j = innerjoin(select(rows_a, :mid, :t0, :sel, :family, :p => :pa, :y),
                  select(rows_b, :mid, :t0, :sel, :p => :pb), on = [:mid, :t0, :sel])
    combine(groupby(j, :family)) do g
        per = combine(groupby(DataFrame(mid = g.mid,
                d = nll.(g.pb, g.y) .- nll.(g.pa, g.y)), :mid), :d => mean => :d)
        (; n_matches = nrow(per), gain = mean(per.d),
           t = mean(per.d) / (std(per.d) / sqrt(nrow(per))))
    end
end
