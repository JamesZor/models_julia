# src/features/pxg.jl
#
# POINT-IN-TIME TEAM-LEVEL PROXY EXPECTED GOALS (pxG).
#
# WHAT THIS IS, AND HOW IT DIFFERS FROM `OpenPlayPxGFeature`.
# `OpenPlayPxGFeature` (extractors/open_play_extractors.jl) emits THIS match's own pxG — it is an
# OBSERVATION, consumed by the recombination engines as a second likelihood route. `PxGFeature`
# emits a PRE-MATCH FORM covariate: for every fixture it reports what the two sides' attacking and
# defensive pxG rates looked like using ONLY matches that kicked off strictly earlier. The two are
# complementary and deliberately carry different keys; nothing here touches `:flat_pxg_h/a`.
#
# THE MEASUREMENT LADDER (per match, per side)
#   1. BBC live-text commentary  — the zonal shot model in plus_minus/shot_parser.jl, summed over
#      every attempt the side took (open play at its cell rate, penalties at the constant). This is
#      the 0.817-team-correlation proxy the bbc_xg_proxy stream validated against SofaScore xG.
#   2. BBC match-page shot counts (`ds.bbc.shots_h/a`) x the league's own pxG-per-shot, for matches
#      with a scoreline page but no live text (live text only starts in 23/24).
#   3. Goals. Always available, badly overdispersed, but an honest last resort.
# `config.fallback` sets how far down the ladder we are willing to go. Which rung each match used is
# reported in `:pxg_source_counts`, so a runner can see the mix rather than assume it.
#
# WHY DEVIATIONS, NOT LEVELS. Every emitted quantity is a deviation from the running league mean, so
# a team with no history contributes EXACTLY 0.0 and the covariate term `w * 0` vanishes. That is the
# builder's "missingness is a zero, not a mask" contract (builder/components.jl §2) satisfied without
# a second vector, and it means the cold start of a season degrades smoothly instead of cliff-edging.
#
# PIT SAFETY IS STRUCTURAL, NOT ASSERTED. Matches are walked in kickoff order and matches sharing a
# calendar day are emitted as a GROUP before any of their observations update the state. A fixture
# can therefore never see itself, and same-day fixtures never see each other. There is no
# fold-dependence in that walk, which is why the same lookup serves both
# `covariate_column` (in-fold) and `covariate_oos` (future fixtures).
#
# The one fold-dependent piece is the shot-xG CELL TABLE, which is fitted on the fold's permitted
# matches when the builder supplies `F_data[:history_match_ids]`. It carries no team or player
# identity — it is `P(goal | zone, body part, context)` — but fitting it on history keeps the whole
# feature Gate-2 clean at negligible cost.

using DataFrames
using Dates
using Statistics

# ==========================================
# 1. CONFIG
# ==========================================
"""
    PxGFeature <: AbstractFeatureConfig

Point-in-time team attacking/defensive proxy-xG form.

  * `lookback_matches`   — number of recent matches averaged when `decay = :window`.
                           `0` means "every earlier match". `lookback` is retained as a
                           backwards-compatible keyword alias.
  * `decay`              — `:window` (flat k-match mean) or `:exponential` (0.5^(j/half_life)).
  * `half_life_matches`  — the exponential decay half-life, in matches, used when `decay = :exponential`.
  * `prior_weight`       — pseudo-matches of the league baseline mixed into every team mean. This is
                           the shrinkage that keeps a two-match sample from swinging the covariate.
  * `min_matches`        — below this many earlier matches a side is held at exactly 0.0 deviation.
  * `k`                  — empirical-Bayes pseudo-count of the shot-xG cell table.
  * `fallback`           — how far down the measurement ladder to go: `:none` (commentary only),
                           `:shots` (+ BBC shot counts), `:goals` (+ goals; the default).
  * `scale`              — divisor applied to the emitted columns. The natural unit is xG per match,
                           so `1.0` is already interpretable; raise it to shrink the prior's reach.
"""
Base.@kwdef struct PxGFeature <: AbstractFeatureConfig
    # `lookback_matches = lookback` lets old `lookback=...` call sites retain their
    # exact window while making the public field name explicit.
    lookback::Int = 8
    lookback_matches::Int = lookback
    decay::Symbol = :exponential
    half_life_matches::Float64 = 16.0
    prior_weight::Float64 = 3.0
    min_matches::Int = 2
    k::Float64 = 25.0
    fallback::Symbol = :goals
    scale::Float64 = 1.0
end

const _PXG_DECAYS   = (:window, :exponential)
const _PXG_FALLBACKS = (:none, :shots, :goals)

function _pxg_validate(config::PxGFeature)
    config.decay in _PXG_DECAYS ||
        error("PxGFeature.decay must be one of $(_PXG_DECAYS); got :$(config.decay)")
    config.fallback in _PXG_FALLBACKS ||
        error("PxGFeature.fallback must be one of $(_PXG_FALLBACKS); got :$(config.fallback)")
    config.lookback >= 0 || error("PxGFeature.lookback must be >= 0")
    config.lookback_matches >= 0 || error("PxGFeature.lookback_matches must be >= 0")
    isfinite(config.half_life_matches) && config.half_life_matches > 0.0 ||
        error("PxGFeature.half_life_matches must be finite and > 0")
    isfinite(config.prior_weight) && config.prior_weight >= 0.0 ||
        error("PxGFeature.prior_weight must be finite and >= 0")
    config.min_matches >= 0 || error("PxGFeature.min_matches must be >= 0")
    isfinite(config.k) && config.k >= 0.0 || error("PxGFeature.k must be finite and >= 0")
    isfinite(config.scale) && config.scale > 0.0 ||
        error("PxGFeature.scale must be finite and > 0")
    return nothing
end

# ==========================================
# 2. THE MATCH CLOCK
# ==========================================
"""
    _pxg_kickoff(row) -> DateTime

The fixture's kickoff, preferring an explicit `start_timestamp` (LibPQ hands these back as
`ZonedDateTime`) and otherwise reconstructing it from `match_date` plus `match_hour`. Every
point-in-time comparison in this file is against this value.
"""
function _pxg_kickoff(row)
    cols = propertynames(row)
    if :start_timestamp in cols && !ismissing(row.start_timestamp)
        v = row.start_timestamp
        v isa DateTime && return v
        v isa Date && return DateTime(v)
        if hasproperty(v, :zone)                      # TimeZones.ZonedDateTime
            return DateTime(v, Dates.UTC)
        end
        parsed = tryparse(DateTime, string(v))
        parsed === nothing || return parsed
    end
    if :match_date in cols && !ismissing(row.match_date)
        d = row.match_date isa Date ? row.match_date : Date(string(row.match_date))
        hour = (:match_hour in cols && !ismissing(row.match_hour)) ?
               clamp(Int(row.match_hour), 0, 23) : 0
        return DateTime(d) + Hour(hour)
    end
    # No usable clock: sort such a row last so it can never precede a dated fixture.
    return DateTime(9999, 12, 31)
end

function _pxg_num(x, default::Float64 = 0.0)
    (ismissing(x) || x === nothing) && return default
    v = try
        Float64(x)
    catch
        return default
    end
    return isfinite(v) ? v : default
end

# ==========================================
# 3. THE MEASUREMENT LADDER
# ==========================================
"""
    pxg_match_observations(ds, config; fit_ids = nothing) -> Dict{Int, NamedTuple}

One `(h, a, source)` record per match for which a pxG measurement could be made at all.
`fit_ids` restricts the matches the shot-xG cell table may be fitted on; `nothing` uses every shot
in the store.

This is an OBSERVATION table — the value for a match describes that match. Turning it into a
pre-match covariate is `_pxg_rolling_lookup`'s job, and only that function enforces the clock.
"""
function pxg_match_observations(ds::Data.DataStore, config::PxGFeature;
                                fit_ids::Union{Nothing,Set{Int}} = nothing)
    out = Dict{Int, NamedTuple{(:h, :a, :source), Tuple{Float64, Float64, Symbol}}}()

    # --- rung 1: live-text commentary -----------------------------------------------------
    shots = build_shots(ds)
    per_shot = NaN
    if nrow(shots) > 0
        fit_rows = fit_ids === nothing ? shots : shots[in.(Int.(shots.match_id), Ref(fit_ids)), :]
        model = fit_shot_xg(nrow(fit_rows) > 0 ? fit_rows : shots; k = config.k)
        pred = predict_xg(model, shots)

        totals = Dict{Int, Tuple{Float64, Float64}}()
        n_shots = 0
        sum_xg = 0.0
        for (i, r) in enumerate(eachrow(shots))
            ismissing(r.is_home) && continue
            mid = Int(r.match_id)
            xg = _pxg_num(pred[i])
            h, a = get(totals, mid, (0.0, 0.0))
            totals[mid] = r.is_home === true ? (h + xg, a) : (h, a + xg)
            n_shots += 1
            sum_xg += xg
        end
        for (mid, (h, a)) in totals
            out[mid] = (h = h, a = a, source = :commentary)
        end
        n_shots > 0 && (per_shot = sum_xg / n_shots)
    end

    config.fallback === :none && return out

    # --- rung 2: BBC match-page shot counts x the league's own pxG-per-shot ----------------
    bbc = ds.bbc
    if isfinite(per_shot) && per_shot > 0.0 && nrow(bbc) > 0 &&
       all(c -> String(c) in names(bbc), (:match_id, :shots_h, :shots_a))
        for r in eachrow(bbc)
            mid = Int(r.match_id)
            haskey(out, mid) && continue
            sh = _pxg_num(r.shots_h, NaN)
            sa = _pxg_num(r.shots_a, NaN)
            (isfinite(sh) && isfinite(sa) && sh >= 0.0 && sa >= 0.0) || continue
            out[mid] = (h = sh * per_shot, a = sa * per_shot, source = :shot_counts)
        end
    end

    config.fallback === :shots && return out

    # --- rung 3: goals ---------------------------------------------------------------------
    matches = ds.matches
    if nrow(matches) > 0 && all(c -> String(c) in names(matches), (:home_score, :away_score))
        for r in eachrow(matches)
            mid = Int(r.match_id)
            haskey(out, mid) && continue
            (ismissing(r.home_score) || ismissing(r.away_score)) && continue
            gh = _pxg_num(r.home_score, NaN)
            ga = _pxg_num(r.away_score, NaN)
            (isfinite(gh) && isfinite(ga) && gh >= 0.0 && ga >= 0.0) || continue
            out[mid] = (h = gh, a = ga, source = :goals)
        end
    end
    return out
end

# ==========================================
# 4. THE POINT-IN-TIME WALK
# ==========================================
"""
    _pxg_weighted_mean(vals, config, baseline) -> Float64

The shrunk team mean over `vals`, which are ordered oldest-first. An empty history returns the
baseline exactly, which is what makes the emitted deviation exactly `0.0`.
"""
function _pxg_weighted_mean(vals::Vector{Float64}, config::PxGFeature, baseline::Float64)
    n = length(vals)
    num = 0.0
    den = 0.0
    if n > 0
        if config.decay === :exponential
            log_two = log(2.0)
            for j in 0:(n - 1)
                w = exp(-log_two * j / config.half_life_matches)
                num += w * vals[n - j]
                den += w
            end
        else
            m = config.lookback_matches <= 0 ? n : min(config.lookback_matches, n)
            for j in 0:(m - 1)
                num += vals[n - j]
                den += 1.0
            end
        end
    end
    total = den + config.prior_weight
    total <= 0.0 && return baseline
    return (num + config.prior_weight * baseline) / total
end

"""
    _pxg_rolling_lookup(observations, matches, config) -> Dict{Int, NamedTuple}

The pre-match form table: for every fixture in `matches`, the two sides' attacking and defensive
pxG deviations built from strictly earlier kickoffs, plus the two assembled design quantities.

    supremacy = (att_h + def_a) - (att_a + def_h)
    level     = (att_h + def_a) + (att_a + def_h)

where `att_x` is x's pxG-scored deviation and `def_x` its pxG-CONCEDED deviation. Read `att_h + def_a`
as "home's expected pxG in this fixture, relative to a league-average pairing": home creating more
than average and away conceding more than average both push it up. The supremacy form is that
quantity for home minus the same for away, so it is the log-odds-shaped term the engine's
`SupremacyRole` wants; the level form is their sum, which moves the total and leaves the result alone.

NOTE ON THE SPEC. PXG_RAPM_SPEC.md §2.A writes the supremacy role as
`(pxG_h,att - pxG_a,def) - (pxG_a,att - pxG_h,def)`. Taken literally that SUBTRACTS the opponent's
concession rate from your own creation rate, so a home side facing a leaky defence is scored DOWN
for it and the two effects cancel exactly when they should reinforce. The sign used here is the one
that makes the term an expected-pxG difference; it is the only difference from the written spec.
"""
function _pxg_rolling_lookup(observations::Dict{Int, <:NamedTuple},
                             matches::AbstractDataFrame,
                             config::PxGFeature)
    out = Dict{Int, NamedTuple{(:att_h, :att_a, :def_h, :def_a, :supremacy, :level, :available),
                               Tuple{Float64, Float64, Float64, Float64, Float64, Float64, Float64}}}()
    nrow(matches) == 0 && return out

    rows = [(id = Int(r.match_id), kickoff = _pxg_kickoff(r),
             home = String(r.home_team), away = String(r.away_team))
            for r in eachrow(matches)]
    sort!(rows, by = r -> (r.kickoff, r.id))

    scored   = Dict{String, Vector{Float64}}()
    conceded = Dict{String, Vector{Float64}}()
    base_sum = 0.0
    base_n   = 0

    i = 1
    n_rows = length(rows)
    while i <= n_rows
        # Every fixture sharing this calendar day is emitted BEFORE any of them updates the state.
        # This is stricter than merely sorting by kickoff and prevents an early match from becoming
        # an input to a later fixture on the same card.
        j = i
        kickoff_day = Date(rows[i].kickoff)
        while j <= n_rows && Date(rows[j].kickoff) == kickoff_day
            j += 1
        end
        baseline = base_n == 0 ? 0.0 : base_sum / base_n

        for t in i:(j - 1)
            r = rows[t]
            h_for = get(scored, r.home, Float64[])
            h_ag  = get(conceded, r.home, Float64[])
            a_for = get(scored, r.away, Float64[])
            a_ag  = get(conceded, r.away, Float64[])

            ready = length(h_for) >= config.min_matches && length(a_for) >= config.min_matches
            if !ready
                out[r.id] = (att_h = 0.0, att_a = 0.0, def_h = 0.0, def_a = 0.0,
                             supremacy = 0.0, level = 0.0, available = 0.0)
                continue
            end

            att_h = _pxg_weighted_mean(h_for, config, baseline) - baseline
            def_h = _pxg_weighted_mean(h_ag,  config, baseline) - baseline
            att_a = _pxg_weighted_mean(a_for, config, baseline) - baseline
            def_a = _pxg_weighted_mean(a_ag,  config, baseline) - baseline

            home_side = att_h + def_a
            away_side = att_a + def_h
            out[r.id] = (att_h = att_h / config.scale, att_a = att_a / config.scale,
                         def_h = def_h / config.scale, def_a = def_a / config.scale,
                         supremacy = (home_side - away_side) / config.scale,
                         level     = (home_side + away_side) / config.scale,
                         available = 1.0)
        end

        for t in i:(j - 1)
            r = rows[t]
            obs = get(observations, r.id, nothing)
            obs === nothing && continue
            (isfinite(obs.h) && isfinite(obs.a)) || continue
            push!(get!(scored,   r.home, Float64[]), obs.h)
            push!(get!(conceded, r.home, Float64[]), obs.a)
            push!(get!(scored,   r.away, Float64[]), obs.a)
            push!(get!(conceded, r.away, Float64[]), obs.h)
            base_sum += obs.h + obs.a
            base_n   += 2
        end
        i = j
    end
    return out
end

# ==========================================
# 5. THE EXTRACTOR
# ==========================================
"""
    add_feature!(F_data, config::PxGFeature, ordered_ids, team_map, ds)

Emits the fold's pre-match pxG design columns plus a whole-store causal bridge for prediction-time
extraction. Every match in the bridge is evaluated against its own kickoff, so a future fixture can
be priced from it without any earlier match having seen the future.

Degrades to all-zero columns (never an error) on a store with no matches — i.e. every segment
without BBC coverage still builds, it simply contributes nothing.
"""
function add_feature!(F_data::Dict, config::PxGFeature, ordered_ids, team_map::Dict,
                      ds::Data.DataStore)
    _pxg_validate(config)
    n = length(ordered_ids)

    fit_ids = if haskey(F_data, :history_match_ids) && !isempty(F_data[:history_match_ids])
        Set(Int.(F_data[:history_match_ids]))
    else
        nothing
    end

    observations = pxg_match_observations(ds, config; fit_ids = fit_ids)
    lookup = _pxg_rolling_lookup(observations, ds.matches, config)

    neutral = (att_h = 0.0, att_a = 0.0, def_h = 0.0, def_a = 0.0,
               supremacy = 0.0, level = 0.0, available = 0.0)
    pick(id) = get(lookup, Int(id), neutral)

    F_data[:flat_pxg_att_home]  = Float64[pick(id).att_h for id in ordered_ids]
    F_data[:flat_pxg_att_away]  = Float64[pick(id).att_a for id in ordered_ids]
    F_data[:flat_pxg_def_home]  = Float64[pick(id).def_h for id in ordered_ids]
    F_data[:flat_pxg_def_away]  = Float64[pick(id).def_a for id in ordered_ids]
    F_data[:flat_pxg_supremacy] = Float64[pick(id).supremacy for id in ordered_ids]
    F_data[:flat_pxg_level]     = Float64[pick(id).level for id in ordered_ids]
    F_data[:flat_pxg_available] = Float64[pick(id).available for id in ordered_ids]
    F_data[:flat_pxg_form_fallback] = Int[pick(id).available > 0.0 ? 0 : 1 for id in ordered_ids]

    # The bridge is the SAME point-in-time walk, kept over every match in the store so that an
    # out-of-sample fixture — which is by construction not in `ordered_ids` — can still be priced.
    F_data[:pxg_supremacy_by_match_id] = Dict{Int,Float64}(
        mid => v.supremacy for (mid, v) in lookup)
    F_data[:pxg_level_by_match_id] = Dict{Int,Float64}(
        mid => v.level for (mid, v) in lookup)

    counts = Dict{Symbol,Int}(:commentary => 0, :shot_counts => 0, :goals => 0)
    for v in values(observations)
        counts[v.source] = get(counts, v.source, 0) + 1
    end
    F_data[:pxg_source_counts] = counts
    F_data[:pxg_observations_by_match_id] = observations
    return nothing
end
