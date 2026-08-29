# ==============================================================================
# 06 — TYPED POSTERIOR LATENTS : THE EXTRACTION LAYER
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# ------------------------------------------------------------------------------
# THE ONE DESIGN DECISION IN THIS FILE
# ------------------------------------------------------------------------------
#
# `extract_latents` does NOT re-derive any linear predictor. Wherever the engine
# already has a working `PreGame.extract_parameters(model, df, feature_set, chain)`,
# this file CALLS IT and re-lays-out the result.
#
# That is the whole point. The posterior arithmetic — zero-sum recentring, the
# clamp bounds, the season/month index resolution, the unknown-team fallback — is
# modelling, and it must have exactly one implementation. Copying it here to "make
# the typed path independent" would create a second copy that drifts, and the parity
# harness would then be testing that two copies of the same code agree, which is a
# tautology dressed as evidence.
#
# What IS being tested, and is not a tautology:
#
#   * the PACKING is lossless — `l04_parity.jl` §4 re-reads every matrix cell back
#     against the source vectors;
#   * the SCORE GRIDS and MARKET PRICES computed from the typed layout are bit-
#     identical to the ones the legacy `Predictions` kernels compute from the
#     DataFrame layout (`l04_parity.jl` §5-6). That is a genuine comparison of two
#     independent kernels over the same numbers, and it is the claim that actually
#     has to hold for the container swap to be safe.
#
# ------------------------------------------------------------------------------
# ONE EXCEPTION, AND WHY
# ------------------------------------------------------------------------------
#
# `DynamicPxGRecombModel` gets a from-scratch extractor here (§4), because BOTH of
# its `extract_parameters` methods are dead code:
#
#     src/models/pregame/engines/team_level/time_decay/recombined_pxg.jl:191
#     src/models/pregame/engines/team_level/time_decay/recombined_pxg.jl:214
#         dyn = extract_dynamics(chain, model.dynamics_config, n_teams)
#
# There is no 3-argument `extract_dynamics`. Every method in
# `src/models/pregame/components/dynamics/` takes `(chain, config, prefix::String, n_teams)`.
# Both calls raise `MethodError` on first use. `r01_demo.jl` §6 demonstrates this
# against the live method table rather than asserting it from a reading.
#
# The extractor below is the legacy body with that one call corrected, and nothing
# else changed. Its arithmetic is verified against the legacy PREDICTION reader —
# `Predictions.extract_params(::AbstractRecombinationModels, row)`
# (recombination.jl:18-40), which is live, is what actually prices recombination
# fixtures today, and re-derives the total from the same channel components.
#
# ==============================================================================

using DataFrames
using Dates
using MCMCChains
using Statistics

using BayesianFootball

include(joinpath(@__DIR__, "l01_latents.jl"))

const TPL_PG   = BayesianFootball.Models.PreGame
const TPL_TI   = BayesianFootball.TypesInterfaces
const TPL_Pred = BayesianFootball.Predictions


# ==============================================================================
# 0. A COMPATIBILITY SHIM, AND AN OPEN DEFECT
# ==============================================================================
#
# MCMCChains 7.7.0 does not define `haskey(::Chains, ::Symbol)`; 7.6 did. Nine live
# `src` call sites use it to test whether an optional site is present in a chain:
#
#     src/models/pregame/components/squad_wealth.jl:46
#     src/models/pregame/components/pxg_observation.jl:34
#     src/models/pregame/components/recombination.jl:66, 67, 68
#     src/models/pregame/engines/.../recombined_pxg.jl:196, 239, 249
#     src/models/pregame/engines/.../recombined_goals.jl:189
#
# Every one of them raises `MethodError` on the installed version, which makes the
# squad-wealth, pxG-observation and recombination extractors — and therefore both
# recombination ENGINES — uncallable. `Project.toml` currently allows
# `MCMCChains = "7.6, 7.7"`, so which of these works depends on what a given machine
# resolved. `r01_demo.jl` §6 reports it.
#
# THE SHIM IS TYPE PIRACY AND IS DELIBERATE. Defining a `Base` method on a foreign
# type is normally the wrong answer, and it is taken here because the alternative is
# worse: without it this file would have to re-implement four `src` extractors, and
# the parity claim would degrade from "the typed path calls the real code" to "the
# typed path agrees with a copy of the real code that I also wrote". The method is
# ADDITIVE (there is none to override) and restores exactly the 7.6 semantics.
#
# WHERE IT BELONGS: not here. Either `src` moves to `sym in names(chain)` at those
# nine sites, or `Project.toml` pins `MCMCChains = "7.6"` until it does. Both are
# out of scope for a container prototype.
if !hasmethod(Base.haskey, Tuple{Chains, Symbol})
    Base.haskey(c::Chains, s::Symbol) = s in names(c)
end

"""
    tpl_haskey_is_shimmed() -> Bool

Whether `haskey(::Chains, ::Symbol)` is being provided by the shim above rather than
by MCMCChains itself. Read by `r01_demo.jl` §6 so the transcript records which side of
the version boundary the run happened on.
"""
tpl_haskey_is_shimmed() =
    parentmodule(which(Base.haskey, Tuple{Chains, Symbol})) === @__MODULE__


# ==============================================================================
# 1. THE FAMILY TRAIT
# ==============================================================================
#
# Which container a model's posterior belongs in is a property of the MODEL, not of
# the columns that happened to be written. Resolving it by dispatch is what removes
# the `hasproperty` chains in `negativebinomial.jl` and `recombination.jl`: by the
# time a container exists, the question is already answered and cannot be re-asked
# with a different answer at a different call site.

abstract type AbstractLatentFamily end

"Double-Poisson marginals. `CountLatents{T, Nothing}`."
struct PoissonCountFamily <: AbstractLatentFamily end

"Double-negative-binomial marginals. `CountLatents{T, <:NamedTuple}`."
struct NegBinCountFamily <: AbstractLatentFamily end

"Additive open-play / penalty / own-goal channels. `RecombLatents{T}`."
struct RecombinationFamily <: AbstractLatentFamily end

"Poisson grid plus a market smile curve. `SmileLatents{T, Nothing}`."
struct SmilePoissonFamily <: AbstractLatentFamily end

"NegBin grid plus a market smile curve. `SmileLatents{T, <:NamedTuple}`."
struct SmileNegBinFamily <: AbstractLatentFamily end

"""
    latent_family(model) -> AbstractLatentFamily

The posterior container family for `model`.

Resolved from the `src` ABSTRACT supertypes wherever possible, so a newly built
model inherits the right family without being named here. In particular every
`ComposableCountModel` from `05_composable_count_builder` is covered by the two
abstract methods below — `PoissonCountModel <: AbstractPoissonModel` and
`NegBinCountModel <: AbstractNegBinModel` — and this file never mentions the
prototype, never includes it, and does not have to be edited when a new covariate
or observation is added to it.

Concrete methods are needed only where a model's supertype is a poor guide, which
happens because the `src` hierarchy routes *everything* team- or player-level
through `AbstractNegBinModel` (`src/models/pregame/types.jl:23-24`) regardless of
its actual observation density. Recombination and smile engines are both in that
position.
"""
latent_family(m) = error(
    "No latent family registered for $(typeof(m)).\n" *
    "  Add a `latent_family(::YourModel) = <Family>()` method in l02_extract.jl §1,\n" *
    "  or make the model subtype AbstractPoissonModel / AbstractNegBinModel.")

latent_family(::TPL_TI.AbstractPoissonModel) = PoissonCountFamily()
latent_family(::TPL_TI.AbstractNegBinModel)  = NegBinCountFamily()

latent_family(::TPL_PG.DynamicPxGRecombModel)      = RecombinationFamily()
latent_family(::TPL_PG.DynamicRecombinedGoalsModel) = RecombinationFamily()

latent_family(::TPL_PG.DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel) = SmilePoissonFamily()
latent_family(::TPL_PG.DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel) = SmilePoissonFamily()


# ==============================================================================
# 2. THE PUBLIC ENTRY POINT
# ==============================================================================

"""
    extract_latents(model, chain, oos_fixtures, feature_set) -> AbstractPosteriorLatents

Posterior latents for every fixture in `oos_fixtures`, in a dense typed container.

  * `chain`        — the fitted `MCMCChains.Chains` for this fold.
  * `oos_fixtures` — a DataFrame of held-out fixtures. Must carry `:match_id`,
                     `:home_team`, `:away_team`, `:match_date`; individual engines
                     read more (`:season_idx`, covariate columns).
  * `feature_set`  — the `FeatureSet` the fold was FITTED on. It supplies `team_map`,
                     `n_teams`, `n_seasons` and the point-in-time covariate bridges,
                     so a fixture's parameters are resolved through the same team
                     indexing the likelihood used.

ROW ORDER IS `oos_fixtures.match_id` ORDER. `extract_parameters` returns a `Dict`,
whose iteration order is a hash artefact and changes between runs; a container whose
row order came from that would serialise differently every time and make two runs
impossible to diff. The DataFrame's own order is the one the caller can see.
"""
extract_latents(model, chain::Chains, oos_fixtures::AbstractDataFrame, feature_set) =
    extract_latents(latent_family(model), model, chain, oos_fixtures, feature_set)


"""
    tpl_ordered_ids(oos_fixtures) -> Vector{Int}

The fixture ids in DataFrame order, checked for duplicates. Extracted as its own
function because every family needs exactly this and a silent duplicate would
produce a container whose row `i` and row `j` are the same fixture with two
different posteriors.
"""
function tpl_ordered_ids(oos_fixtures::AbstractDataFrame)
    hasproperty(oos_fixtures, :match_id) ||
        error("oos_fixtures needs a :match_id column; got $(propertynames(oos_fixtures)).")
    ids = Int.(oos_fixtures.match_id)
    allunique(ids) || error(
        "oos_fixtures has duplicate match_ids: " *
        "$(join(unique([i for i in ids if count(==(i), ids) > 1]), ", ")).")
    return ids
end

"""
    tpl_stack(raw, ids, field) -> Matrix{Float64}

Copy one parameter out of the legacy `Dict{Int, NamedTuple}` into a dense
`(n_matches × n_draws)` matrix, in `ids` order.

This single function is the entire "boxing" fix. On the way in, `raw[id][field]` is
one separately-allocated `Vector{Float64}` per fixture; on the way out there is one
matrix. Everything downstream — serialisation size, cache behaviour, the ability of
the score-grid kernel to specialise — follows from this copy.
"""
function tpl_stack(raw::AbstractDict{Int, <:NamedTuple}, ids::Vector{Int}, field::Symbol)
    isempty(ids) && return Matrix{Float64}(undef, 0, 0)

    probe = get(raw, first(ids)) do
        error("extract_parameters returned no entry for match_id $(first(ids)).")
    end
    haskey(probe, field) || error(
        "extract_parameters returned $(keys(probe)) for match_id $(first(ids)); " *
        "this family needs :$field.")

    nd  = length(probe[field])
    out = Matrix{Float64}(undef, length(ids), nd)
    for (i, id) in enumerate(ids)
        row = get(raw, id) do
            error("extract_parameters returned no entry for match_id $id.")
        end
        v = row[field]
        length(v) == nd || error(
            "match_id $id has $(length(v)) draws of :$field but match_id $(first(ids)) " *
            "has $nd. Every fixture in a fold must come from the same posterior sweep.")
        @inbounds for k in 1:nd
            out[i, k] = v[k]
        end
    end
    return out
end

"""
    tpl_dispersion_fields(raw, ids) -> (; r_h, r_a)

Resolve the two dispersion matrices from whichever schema the engine wrote.

`GlobalDispersion` engines emit a single shared `:r` (or a pair of identical
`:r_h`/`:r_a`); `HomeAwayDispersion` and `AdvancedVolatilityDispersion` emit an
asymmetric pair. `negativebinomial.jl:9-27` re-discovers which, at prediction time,
on every single row. This resolves it ONCE, here, on the boundary — and after this
line the distinction does not exist anywhere downstream.
"""
function tpl_dispersion_fields(raw::AbstractDict{Int, <:NamedTuple}, ids::Vector{Int})
    probe = raw[first(ids)]
    if haskey(probe, :r_h) && haskey(probe, :r_a)
        return (; r_h = tpl_stack(raw, ids, :r_h), r_a = tpl_stack(raw, ids, :r_a))
    elseif haskey(probe, :r)
        shared = tpl_stack(raw, ids, :r)
        # Materialised into both slots rather than aliased: aliasing would make an
        # in-place edit of one side silently change the other.
        return (; r_h = shared, r_a = copy(shared))
    end
    error("A negative-binomial family needs :r or :r_h/:r_a from extract_parameters; " *
          "got $(keys(probe)). If this engine is really Poisson, register it with " *
          "`latent_family(::YourModel) = PoissonCountFamily()`.")
end


# ==============================================================================
# 3. COUNT FAMILIES
# ==============================================================================

function extract_latents(::PoissonCountFamily, model, chain::Chains,
                         oos_fixtures::AbstractDataFrame, feature_set)
    ids = tpl_ordered_ids(oos_fixtures)
    raw = TPL_PG.extract_parameters(model, oos_fixtures, feature_set, chain)
    return CountLatents(ids,
                        tpl_stack(raw, ids, :λ_h),
                        tpl_stack(raw, ids, :λ_a),
                        nothing)
end

function extract_latents(::NegBinCountFamily, model, chain::Chains,
                         oos_fixtures::AbstractDataFrame, feature_set)
    ids = tpl_ordered_ids(oos_fixtures)
    raw = TPL_PG.extract_parameters(model, oos_fixtures, feature_set, chain)
    return CountLatents(ids,
                        tpl_stack(raw, ids, :λ_h),
                        tpl_stack(raw, ids, :λ_a),
                        tpl_dispersion_fields(raw, ids))
end


# ==============================================================================
# 4. RECOMBINATION FAMILY
# ==============================================================================
#
# The one from-scratch extractor. See the file header for why.

"""
    tpl_recomb_components(model, chain, oos_fixtures, feature_set) -> NamedTuple

The RAW recombination channel components, one `(n_matches × n_draws)` matrix each:

    μ_open_h, μ_open_a   open-play pxG intensity, before finishing
    κ_h, κ_a             team finishing factors
    λ_pen_raw_h/a        penalties AWARDED per match
    q_pen                penalty conversion probability
    og_rate              own-goal intensity

Kept separate from `extract_latents` so that the parity harness can build the LEGACY
DataFrame and the TYPED container from one common set of numbers. Both consumers then
do their own arithmetic on identical inputs, which is what makes `r01_demo.jl` §8b
a real comparison rather than a round trip.

The body is `recombined_pxg.jl:206-283` with one correction: `extract_dynamics` is
called with its actual 4-argument signature `(chain, config, prefix, n_teams)`. Every
other line — the clamp bounds, the zero-sum recentrings, the `-1` unknown-team
sentinel, the `min(l_idx, n_leagues)` league guard — is the legacy body verbatim,
because those are modelling decisions and this file has no standing to change them.
"""
function tpl_recomb_components(model::TPL_PG.DynamicPxGRecombModel,
                               chain::Chains,
                               oos_fixtures::AbstractDataFrame,
                               feature_set)
    data      = feature_set.data
    n_teams   = Int(data[:n_teams])
    team_map  = data[:team_map]
    n_leagues = Int(get(data, :n_leagues, 1))

    inter  = TPL_PG.extract_interception(chain, model.interception_config, 1)
    ha_mat = TPL_PG.extract_home_advantage(chain, model.homeadvantage_config, n_teams)
    # THE CORRECTION. Legacy passes 3 args; no such method exists.
    dyn    = TPL_PG.extract_dynamics(chain, model.dynamics_config, "dyn", n_teams)
    w_val  = TPL_PG.extract_squad_wealth(chain, model.wealth_config)
    recomb = TPL_PG.extract_recombination(chain, model.recomb_config)

    n_samples = size(chain, 1) * size(chain, 3)

    # Finishing factor κ, zero-sum in log space then bounded — legacy lines 219-227.
    κ_sym = [Symbol("log_κ_raw[$i]") for i in 1:n_teams]
    if all(haskey(chain, s) for s in κ_sym)
        raw_mat  = Array(chain[κ_sym])
        centered = raw_mat .- mean(raw_mat, dims = 2)
        κ_mat    = exp.(clamp.(centered, -0.50, 0.50))
    else
        κ_mat = ones(Float64, n_samples, n_teams)
    end

    league_sym = [Symbol("δ_league_raw[$i]") for i in 1:n_leagues]
    if all(haskey(chain, s) for s in league_sym)
        raw_l        = Array(chain[league_sym])
        delta_league = raw_l .- mean(raw_l, dims = 2)
    else
        delta_league = zeros(Float64, n_samples, n_leagues)
    end

    wealth_lookup = get(data, :wealth_lookup, Dict{Int, Float64}())
    league_lookup = get(data, :league_lookup, Dict{Int, Int}())

    ids = tpl_ordered_ids(oos_fixtures)
    n_m = length(ids)
    alloc() = Matrix{Float64}(undef, n_m, n_samples)
    μ_open_h, μ_open_a = alloc(), alloc()
    κ_h_m,    κ_a_m    = alloc(), alloc()
    pen_h_m,  pen_a_m  = alloc(), alloc()
    q_pen_m,  og_m     = alloc(), alloc()

    # Officiating is match-invariant in the current engine (the referee effect is
    # dropped at prediction time because the OOS referee is unknown), but it is
    # materialised per fixture so that a per-referee prediction is a change of these
    # eight lines and not a change of the container.
    q_pen   = hasproperty(recomb, :pen_conv) ? recomb.pen_conv : fill(0.768, n_samples)
    og_rate = hasproperty(recomb, :og_rate)  ? recomb.og_rate  : fill(0.0276, n_samples)
    if hasproperty(recomb, :pen_base_μ)
        λ_pen_h = exp.(clamp.(recomb.pen_base_μ .+ recomb.ha_pen, -6.0, 2.0))
        λ_pen_a = exp.(clamp.(recomb.pen_base_μ, -6.0, 2.0))
    else
        λ_pen_h = fill(0.207, n_samples)
        λ_pen_a = fill(0.207, n_samples)
    end

    for (i, row) in enumerate(eachrow(oos_fixtures))
        mid   = Int(row.match_id)
        h_idx = get(team_map, row.home_team, -1)
        a_idx = get(team_map, row.away_team, -1)

        α_h = h_idx > 0 ? dyn.α[:, h_idx] : zeros(n_samples)
        β_h = h_idx > 0 ? dyn.β[:, h_idx] : zeros(n_samples)
        α_a = a_idx > 0 ? dyn.α[:, a_idx] : zeros(n_samples)
        β_a = a_idx > 0 ? dyn.β[:, a_idx] : zeros(n_samples)

        γ_h = h_idx > 0 ? ha_mat[:, h_idx] : zeros(n_samples)
        κ_h = h_idx > 0 ? κ_mat[:, h_idx]  : ones(n_samples)
        κ_a = a_idx > 0 ? κ_mat[:, a_idx]  : ones(n_samples)

        m_idx = Dates.month(row.match_date)
        l_idx = get(league_lookup, mid, 1)
        dw    = get(wealth_lookup, mid, 0.0)

        w_shift    = w_val .* dw
        inter_match = inter.μ_base[:, 1] .+ inter.δ_month[:, m_idx] .+
                      delta_league[:, min(l_idx, n_leagues)]

        log_μ_h = clamp.(inter_match .+ γ_h .+ α_h .- β_a .+ w_shift, -5.0, 4.0)
        log_μ_a = clamp.(inter_match .+        α_a .- β_h .- w_shift, -5.0, 4.0)

        @inbounds for k in 1:n_samples
            μ_open_h[i, k] = exp(log_μ_h[k])
            μ_open_a[i, k] = exp(log_μ_a[k])
            κ_h_m[i, k]    = κ_h[k]
            κ_a_m[i, k]    = κ_a[k]
            pen_h_m[i, k]  = λ_pen_h[k]
            pen_a_m[i, k]  = λ_pen_a[k]
            q_pen_m[i, k]  = q_pen[k]
            og_m[i, k]     = og_rate[k]
        end
    end

    return (; match_ids = ids,
              μ_open_h, μ_open_a,
              κ_h = κ_h_m, κ_a = κ_a_m,
              λ_pen_raw_h = pen_h_m, λ_pen_raw_a = pen_a_m,
              q_pen = q_pen_m, og_rate = og_m)
end

"""
    recomb_latents_from_components(c) -> RecombLatents

Fold the raw channel components into the container's GOAL-rate convention:

    λ_open = μ_open · κ          finishing applied
    λ_pen  = q_pen  · λ_pen_raw  conversion applied
    λ_og   = og_rate
    pxg    = μ_open              carried un-finished, for diagnostics

Each product is formed with the same two operands in the same order as
`recombination.jl:36-37`, so `λ_open + λ_pen + λ_og` reproduces the legacy total
bit-for-bit and not merely to rounding.
"""
function recomb_latents_from_components(c::NamedTuple)
    return RecombLatents(
        c.match_ids,
        c.μ_open_h .* c.κ_h,
        c.μ_open_a .* c.κ_a,
        c.q_pen .* c.λ_pen_raw_h,
        c.q_pen .* c.λ_pen_raw_a,
        c.og_rate,
        copy(c.og_rate),          # copied, not aliased — see tpl_dispersion_fields
        c.μ_open_h,
        c.μ_open_a,
    )
end

extract_latents(::RecombinationFamily, model, chain::Chains,
                oos_fixtures::AbstractDataFrame, feature_set) =
    recomb_latents_from_components(
        tpl_recomb_components(model, chain, oos_fixtures, feature_set))


# ==============================================================================
# 5. SMILE FAMILIES
# ==============================================================================

"""
    tpl_stack_smile(raw, ids, n_draws) -> (φ, strikes)

Re-lay the per-fixture `φ` matrices into one `(n_matches × n_strikes × n_draws)`
array and derive the strike ladder from its width.

The engines store φ as `n_draws × n_strikes` (`goals_smile_league.jl:208-212`) and
put the SAME matrix object on every row, because the smile is global. That sharing is
an implicit contract nothing checks — a per-league smile would break it silently. The
copy here makes the per-fixture dimension explicit; §5 of `l01_latents.jl` explains
why paying for it is worth doing.
"""
function tpl_stack_smile(raw::AbstractDict{Int, <:NamedTuple}, ids::Vector{Int}, nd::Int)
    probe = raw[first(ids)]
    haskey(probe, :φ) || error(
        "A smile family needs :φ from extract_parameters; got $(keys(probe)).")

    φ_probe = probe.φ
    ndims(φ_probe) == 2 || error("φ must be a (n_draws × n_strikes) matrix; got $(size(φ_probe)).")
    size(φ_probe, 1) == nd || error(
        "φ has $(size(φ_probe, 1)) rows but λ_h has $nd draws. The engine writes φ " *
        "as (n_draws × n_strikes) — see goals_smile_league.jl:208.")
    nK = size(φ_probe, 2)

    φ = Array{Float64, 3}(undef, length(ids), nK, nd)
    for (i, id) in enumerate(ids)
        m = raw[id].φ
        size(m) == (nd, nK) || error(
            "match_id $id has φ of size $(size(m)); expected $((nd, nK)).")
        @inbounds for k in 1:nd, s in 1:nK
            φ[i, s, k] = m[k, s]
        end
    end

    # Column s is the shape at integer threshold K = s - 1, whose market line is K + 0.5.
    strikes = [(s - 1) + 0.5 for s in 1:nK]
    return φ, strikes
end

function extract_latents(::SmilePoissonFamily, model, chain::Chains,
                         oos_fixtures::AbstractDataFrame, feature_set)
    ids = tpl_ordered_ids(oos_fixtures)
    raw = TPL_PG.extract_parameters(model, oos_fixtures, feature_set, chain)
    λ_h = tpl_stack(raw, ids, :λ_h)
    λ_a = tpl_stack(raw, ids, :λ_a)
    λ_t = tpl_stack(raw, ids, :λ_tot)
    φ, strikes = tpl_stack_smile(raw, ids, size(λ_h, 2))
    return SmileLatents(ids, λ_h, λ_a, nothing, λ_t, φ, strikes)
end

function extract_latents(::SmileNegBinFamily, model, chain::Chains,
                         oos_fixtures::AbstractDataFrame, feature_set)
    ids = tpl_ordered_ids(oos_fixtures)
    raw = TPL_PG.extract_parameters(model, oos_fixtures, feature_set, chain)
    λ_h = tpl_stack(raw, ids, :λ_h)
    λ_a = tpl_stack(raw, ids, :λ_a)
    λ_t = tpl_stack(raw, ids, :λ_tot)
    φ, strikes = tpl_stack_smile(raw, ids, size(λ_h, 2))
    return SmileLatents(ids, λ_h, λ_a, tpl_dispersion_fields(raw, ids), λ_t, φ, strikes)
end


# ==============================================================================
# 6. THE MIGRATION BRIDGE
# ==============================================================================
#
# Two directions, both needed, for different reasons.
#
#   latents_from_legacy_dataframe  reads an ALREADY-CACHED `oos_latents.jls`.
#                                  There are folds on disk that cost hours of NUTS to
#                                  produce; a migration that required refitting them
#                                  would not be adopted.
#
#   to_legacy_dataframe            feeds a typed container to a consumer that has not
#                                  been migrated yet (and, in `l04_parity.jl`, to the
#                                  legacy kernels being compared against). It is the
#                                  escape hatch that lets the swap happen one consumer
#                                  at a time instead of all at once.

"""
    latents_from_legacy_dataframe(model, df) -> AbstractPosteriorLatents

Build a typed container from a legacy `LatentStates.df` (or any DataFrame with the
same column convention), without touching a chain.

`model` selects the family; the columns must match what that family needs.
"""
latents_from_legacy_dataframe(model, df::AbstractDataFrame) =
    latents_from_legacy_dataframe(latent_family(model), df)

# A family with no reader falls here, not back into the generic method above — which
# would re-resolve `latent_family` on the FAMILY object and report a confusing
# "no latent family registered for PoissonCountFamily".
latents_from_legacy_dataframe(f::AbstractLatentFamily, ::AbstractDataFrame) = error(
    "No legacy-DataFrame reader for $(nameof(typeof(f))).\n" *
    "  Add a `latents_from_legacy_dataframe(::$(nameof(typeof(f))), df)` method in " *
    "l02_extract.jl §6,\n  or build the container from a chain with `extract_latents`.")

function _tpl_col_matrix(df::AbstractDataFrame, col::Symbol)
    hasproperty(df, col) || error(
        "legacy latents.df has no :$col column; it has $(propertynames(df)).")
    vs = df[!, col]
    nd = length(first(vs))
    out = Matrix{Float64}(undef, nrow(df), nd)
    for i in 1:nrow(df)
        v = vs[i]
        length(v) == nd || error(
            "legacy :$col row $i has $(length(v)) draws, row 1 has $nd.")
        @inbounds for k in 1:nd
            out[i, k] = v[k]
        end
    end
    return out
end

latents_from_legacy_dataframe(::PoissonCountFamily, df::AbstractDataFrame) =
    CountLatents(Int.(df.match_id),
                 _tpl_col_matrix(df, :λ_h), _tpl_col_matrix(df, :λ_a), nothing)

function latents_from_legacy_dataframe(::NegBinCountFamily, df::AbstractDataFrame)
    obs = if hasproperty(df, :r_h) && hasproperty(df, :r_a)
        (; r_h = _tpl_col_matrix(df, :r_h), r_a = _tpl_col_matrix(df, :r_a))
    elseif hasproperty(df, :r)
        shared = _tpl_col_matrix(df, :r)
        (; r_h = shared, r_a = copy(shared))
    else
        error("legacy latents.df for a NegBin family needs :r or :r_h/:r_a; " *
              "it has $(propertynames(df)).")
    end
    return CountLatents(Int.(df.match_id),
                        _tpl_col_matrix(df, :λ_h), _tpl_col_matrix(df, :λ_a), obs)
end

"""
    _tpl_smile_from_df(df, obs) -> SmileLatents

Rebuild a smile container from a legacy frame carrying `:λ_h, :λ_a, :λ_tot, :φ`.

The strike ladder is DERIVED from the width of the `φ` cells, not read from anywhere,
because no legacy frame records it: the engines put a bare `n_draws × nK` matrix on
every row and leave `K = column - 1` as an unwritten convention
(goals_smile_league.jl:208-212). Recovering it by counting columns is the only thing
available — and the `SmileLatents` constructor then enforces `strikes[s] == (s-1)+0.5`,
so a frame whose φ was built under some other convention fails loudly here instead of
pricing Over 2.5 off the Over 1.5 curve.
"""
function _tpl_smile_from_df(df::AbstractDataFrame, obs)
    hasproperty(df, :φ) || error(
        "legacy latents.df for a smile family needs a :φ column; it has $(propertynames(df)).")
    λ_h = _tpl_col_matrix(df, :λ_h)
    λ_a = _tpl_col_matrix(df, :λ_a)
    λ_t = _tpl_col_matrix(df, :λ_tot)

    nm, nd = size(λ_h)
    cells  = df[!, :φ]
    nK     = size(first(cells), 2)
    φ = Array{Float64, 3}(undef, nm, nK, nd)
    for i in 1:nm
        m = cells[i]
        size(m) == (nd, nK) || error(
            "legacy :φ row $i is $(size(m)); expected $((nd, nK)) — the engines write " *
            "φ as (n_draws × n_strikes).")
        @inbounds for k in 1:nd, s in 1:nK
            φ[i, s, k] = m[k, s]
        end
    end
    return SmileLatents(Int.(df.match_id), λ_h, λ_a, obs, λ_t, φ,
                        [(s - 1) + 0.5 for s in 1:nK])
end

latents_from_legacy_dataframe(::SmilePoissonFamily, df::AbstractDataFrame) =
    _tpl_smile_from_df(df, nothing)

function latents_from_legacy_dataframe(::SmileNegBinFamily, df::AbstractDataFrame)
    obs = if hasproperty(df, :r_h) && hasproperty(df, :r_a)
        (; r_h = _tpl_col_matrix(df, :r_h), r_a = _tpl_col_matrix(df, :r_a))
    elseif hasproperty(df, :r)
        shared = _tpl_col_matrix(df, :r)
        (; r_h = shared, r_a = copy(shared))
    else
        error("legacy latents.df for a NegBin smile family needs :r or :r_h/:r_a; " *
              "it has $(propertynames(df)).")
    end
    return _tpl_smile_from_df(df, obs)
end

"""
    latents_from_legacy_dataframe(::RecombinationFamily, df)

REFUSED, and this is the correct behaviour rather than a missing feature.

A `RecombLatents` is a DECOMPOSITION, and no legacy frame contains one. The engine
writes `μ_open_h/a, κ_h/a, λ_pen_h/a, λ_h/a` (recombined_pxg.jl:275-285) — but neither
`q_pen` nor `og_rate`. So `λ_total − μ_open·κ` is recoverable and its split into a
penalty channel and an own-goal channel is not:

    λ_total_h − λ_open_h = q_pen·λ_pen_h + og_rate      one equation, two unknowns

Guessing the split would put own goals in the penalty channel. Nothing downstream of
`compute_score_grid` currently reads the channels, so the guess would be invisible until
somebody used them — which is the worst possible time to find out. Rebuild from the
chain with `extract_latents` instead; `tpl_recomb_components` is where the split comes
from and it needs the posterior, not the frame.
"""
latents_from_legacy_dataframe(::RecombinationFamily, ::AbstractDataFrame) = error(
    "A RecombLatents cannot be rebuilt from a legacy latents.df: the frame carries the " *
    "recombined totals but not q_pen or og_rate, so the penalty and own-goal channels " *
    "cannot be separated (one equation, two unknowns).\n" *
    "  Use `extract_latents(model, chain, oos_fixtures, feature_set)`, which reads the " *
    "split from the posterior.\n" *
    "  If only the total is needed, `CountLatents(ids, λ_h, λ_a)` prices identically — " *
    "the recombination grid is a Poisson of the summed rate.")

"""
    to_legacy_dataframe(l) -> DataFrame

Render a typed container back into the column convention the legacy
`Predictions.extract_params` readers expect, so an unmigrated consumer keeps working.

Each family emits the columns of the branch its legacy reader takes:

  * `CountLatents` Poisson → `:λ_h, :λ_a`            (poisson.jl:24)
  * `CountLatents` NegBin  → `+ :r_h, :r_a`          (negativebinomial.jl:17-22)
  * `RecombLatents`        → `:λ_h, :λ_a`            (recombination.jl:20, branch 1 —
                             the ALREADY-RECOMBINED total, so the reader does not
                             re-derive it from channels this container has already
                             folded together)
  * `SmileLatents`         → `+ :λ_tot, :φ`          (smile_poisson.jl:60)

Note the round trip is lossy in one direction only: `RecombLatents` loses its channel
decomposition, because branch 1 of the legacy reader has nowhere to put it. Nothing
downstream of `extract_params` reads the channels, so nothing is lost that is used —
but a caller who needs them should hold the container, not the DataFrame.
"""
function to_legacy_dataframe end

_tpl_rows(m::Matrix{Float64}) = [collect(view(m, i, :)) for i in 1:size(m, 1)]

to_legacy_dataframe(l::CountLatents{T, Nothing}) where {T} = DataFrame(
    :match_id => l.match_ids,
    :λ_h => _tpl_rows(l.λ_home),
    :λ_a => _tpl_rows(l.λ_away),
)

to_legacy_dataframe(l::CountLatents{T, <:NamedTuple}) where {T} = DataFrame(
    :match_id => l.match_ids,
    :λ_h => _tpl_rows(l.λ_home),
    :λ_a => _tpl_rows(l.λ_away),
    :r_h => _tpl_rows(l.observation_params.r_h),
    :r_a => _tpl_rows(l.observation_params.r_a),
)

function to_legacy_dataframe(l::RecombLatents)
    nm, nd = n_matches(l), n_draws(l)
    λ_h = Matrix{Float64}(undef, nm, nd)
    λ_a = Matrix{Float64}(undef, nm, nd)
    @inbounds for k in 1:nd, i in 1:nm
        λ_h[i, k] = recomb_total_home(l, i, k)
        λ_a[i, k] = recomb_total_away(l, i, k)
    end
    return DataFrame(:match_id => l.match_ids,
                     :λ_h => _tpl_rows(λ_h), :λ_a => _tpl_rows(λ_a))
end

"φ back to the engine's (n_draws × n_strikes) per-fixture layout."
function _tpl_phi_rows(φ::Array{Float64, 3})
    nm, nK, nd = size(φ)
    return [Float64[φ[i, s, k] for k in 1:nd, s in 1:nK] for i in 1:nm]
end

function to_legacy_dataframe(l::SmileLatents{T, Nothing}) where {T}
    return DataFrame(:match_id => l.match_ids,
                     :λ_h => _tpl_rows(l.λ_home), :λ_a => _tpl_rows(l.λ_away),
                     :λ_tot => _tpl_rows(l.λ_tot), :φ => _tpl_phi_rows(l.φ))
end

function to_legacy_dataframe(l::SmileLatents{T, <:NamedTuple}) where {T}
    return DataFrame(:match_id => l.match_ids,
                     :λ_h => _tpl_rows(l.λ_home), :λ_a => _tpl_rows(l.λ_away),
                     :λ_tot => _tpl_rows(l.λ_tot), :φ => _tpl_phi_rows(l.φ),
                     :r_h => _tpl_rows(l.observation_params.r_h),
                     :r_a => _tpl_rows(l.observation_params.r_a))
end
