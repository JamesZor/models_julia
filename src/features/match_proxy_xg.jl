# src/features/match_proxy_xg.jl
#
# MATCH-LEVEL PROXY xG AS AN *OBSERVATION*.
#
# WHERE THIS SITS IN THE pxG FAMILY. There are now three team-level pxG feeds in `src/features/`,
# and they are deliberately different things:
#
#   `PxGFeature`            (pxg.jl)                     PRE-MATCH FORM covariate. Strictly earlier
#                                                        matches only. Emits deviations, so an
#                                                        uncovered fixture contributes exactly 0.0.
#   `OpenPlayPxGFeature`    (extractors/open_play_…)     THIS match's OPEN-PLAY pxG, penalties
#                                                        excluded, for the recombination engines.
#   `MatchProxyXGFeature`   (this file)                  THIS match's TOTAL pxG — open play at its
#                                                        cell rate plus penalties at the constant —
#                                                        for the two-arm joint likelihood, whose
#                                                        Gamma arm reads it as a direct measurement
#                                                        of the same latent μ the goals arm reads.
#
# WHY A SEPARATE FEATURE RATHER THAN A KEYWORD ON `OpenPlayPxGFeature`. The joint model's Gamma arm
# is an observation of TOTAL scoring intensity, so it must include penalties — the Poisson arm it is
# tied to counts penalty goals. `OpenPlayPxGFeature` filters penalties out by construction, because
# recombination prices them on their own channel. Sharing one extractor would mean one of the two
# consumers silently getting the wrong quantity.
#
# THE MEASUREMENT LADDER, AND WHY IT STOPS SHORT OF GOALS. `pxg_match_observations` (pxg.jl §3) is
# reused verbatim, so the cell table, the penalty constant and the fold-safe `fit_ids` contract are
# the same code the form covariate validated against SofaScore xG at r = 0.817. But this feature
# REFUSES the ladder's third rung. Falling back to goals would hand the Gamma arm the very counts
# the Poisson arm is already reading, and the joint likelihood would then count each goal twice with
# no way to tell from the posterior that it had. `fallback = :goals` is an error here, not an option.
#
# MISSINGNESS IS A MASK, NOT A ZERO. The builder's covariate contract imputes an absent covariate to
# 0.0 because `w * 0 == 0` makes the term vanish exactly. That trick does not transfer: the Gamma
# density is not linear in its observation, and `Gamma(ν, μ/ν)` has no support at 0 at all. So this
# feature emits a genuine binary mask alongside a finite dummy value, which is what the AD guide
# asks for whenever the zero-imputation shortcut does not apply.
#
# Live BBC commentary starts in 23/24. Everything earlier is mask 0 and contributes to the Poisson
# arm only, which is the whole point of the two-arm design: the goals arm keeps the full history,
# the proxy arm sharpens the seasons that have text.

using DataFrames
using Statistics

# ==========================================
# 1. CONFIG
# ==========================================
"""
    MatchProxyXGFeature <: AbstractFeatureConfig

Match-level total proxy xG (open play + penalties), emitted as a masked observation for the
two-arm joint likelihood.

  * `k`         — empirical-Bayes pseudo-count of the shot-xG cell table, passed through to
                  `fit_shot_xg`. A cell with `k` shots is pulled halfway to the base rate.
  * `fallback`  — how far down `pxg_match_observations`' ladder to go. `:none` uses live-text
                  commentary only; `:shots` also accepts BBC match-page shot counts times the
                  league's own pxG-per-shot. `:goals` is REFUSED — see the file header.
  * `floor`     — the smallest pxG a covered side may report. `Gamma` has no support at 0, and a
                  side that registered no attempt at all measures exactly 0.0, so the alternative to
                  a floor is masking out an otherwise perfectly good match. 0.05 sits below the
                  least dangerous single attempt in the cell table.
  * `dummy`     — the value written where the mask is 0. Never read by the likelihood (it is
                  multiplied by a zero weight); it exists so the design vector is finite and the
                  compiled tape sees one shape for every match.
"""
Base.@kwdef struct MatchProxyXGFeature <: AbstractFeatureConfig
    k::Float64 = 25.0
    fallback::Symbol = :none
    floor::Float64 = 0.05
    dummy::Float64 = 1.0
end

const _MPXG_FALLBACKS = (:none, :shots)

function _mpxg_validate(config::MatchProxyXGFeature)
    if config.fallback === :goals
        error("MatchProxyXGFeature.fallback = :goals is refused. The Gamma arm would then observe " *
              "the same goal counts the Poisson arm observes, double-counting every goal in the " *
              "joint likelihood. Use :none (commentary) or :shots (+ BBC shot counts).")
    end
    config.fallback in _MPXG_FALLBACKS ||
        error("MatchProxyXGFeature.fallback must be one of $(_MPXG_FALLBACKS); got :$(config.fallback)")
    isfinite(config.k) && config.k >= 0.0 || error("MatchProxyXGFeature.k must be finite and >= 0")
    isfinite(config.floor) && config.floor > 0.0 ||
        error("MatchProxyXGFeature.floor must be finite and > 0; Gamma has no support at 0")
    isfinite(config.dummy) && config.dummy > 0.0 ||
        error("MatchProxyXGFeature.dummy must be finite and > 0")
    return nothing
end

"""
    _mpxg_ladder_config(config) -> PxGFeature

The `PxGFeature` whose measurement ladder this feature borrows. Only the two fields the ladder
actually reads (`k`, `fallback`) carry over; every rolling-window field is irrelevant here because
`pxg_match_observations` returns the raw per-match observation table, before any point-in-time walk.
"""
_mpxg_ladder_config(config::MatchProxyXGFeature) =
    PxGFeature(k = config.k, fallback = config.fallback)

# ==========================================
# 2. THE EXTRACTOR
# ==========================================
"""
    add_feature!(F_data, config::MatchProxyXGFeature, ordered_ids, team_map, ds)

Emits, in `ordered_match_ids` order:

    :flat_pxg_home            Float64, strictly positive everywhere
    :flat_pxg_away            Float64, strictly positive everywhere
    :flat_pxg_obs_available   Float64, exactly 0.0 or 1.0

A match is available only when BOTH sides carry a finite measurement from an allowed rung. A
one-sided measurement would make the two arms disagree about which matches they are describing,
and the Gamma arm's shape parameter is shared across both sides.

KEY NAMING. The work package specifies `:flat_pxg_available` for the mask. `PxGFeature` (pxg.jl)
already owns that key for a DIFFERENT quantity — whether its pre-match FORM window was ready — and
`test/test_pxg_rapm_features.jl` asserts on it. A model carrying both feeds would have had one
silently overwrite the other, with the loser's meaning inherited by the winner's consumer. The mask
is therefore `:flat_pxg_obs_available`; `obs` marks it as observation coverage rather than form
coverage. Nothing else about the work package changed.

Degrades to an all-zero mask (never an error) on a store with no BBC coverage, so a segment without
live text still builds and simply contributes no proxy arm.
"""
function add_feature!(F_data::Dict, config::MatchProxyXGFeature, ordered_ids, team_map::Dict,
                      ds::Data.DataStore)
    _mpxg_validate(config)

    # The cell table is fitted on the fold's permitted matches when the builder supplies them, for
    # the same reason `PxGFeature` does it: the table carries no team identity, but fitting it on
    # history keeps the whole feature Gate-2 clean at negligible cost.
    fit_ids = if haskey(F_data, :history_match_ids) && !isempty(F_data[:history_match_ids])
        Set(Int.(F_data[:history_match_ids]))
    else
        nothing
    end

    observations = pxg_match_observations(ds, _mpxg_ladder_config(config); fit_ids = fit_ids)

    n = length(ordered_ids)
    pxg_home  = Vector{Float64}(undef, n)
    pxg_away  = Vector{Float64}(undef, n)
    available = Vector{Float64}(undef, n)
    counts    = Dict{Symbol,Int}(:commentary => 0, :shot_counts => 0, :none => 0)

    for (i, id) in enumerate(ordered_ids)
        obs = get(observations, Int(id), nothing)
        covered = obs !== nothing && isfinite(obs.h) && isfinite(obs.a) &&
                  obs.h >= 0.0 && obs.a >= 0.0
        if covered
            pxg_home[i]  = max(config.floor, obs.h)
            pxg_away[i]  = max(config.floor, obs.a)
            available[i] = 1.0
            counts[obs.source] = get(counts, obs.source, 0) + 1
        else
            pxg_home[i]  = config.dummy
            pxg_away[i]  = config.dummy
            available[i] = 0.0
            counts[:none] += 1
        end
    end

    F_data[:flat_pxg_home]          = pxg_home
    F_data[:flat_pxg_away]          = pxg_away
    F_data[:flat_pxg_obs_available] = available
    F_data[:pxg_obs_source_counts]  = counts
    F_data[:pxg_obs_by_match_id]    = Dict{Int,Tuple{Float64,Float64}}(
        Int(id) => (pxg_home[i], pxg_away[i])
        for (i, id) in enumerate(ordered_ids) if available[i] == 1.0)
    return nothing
end
