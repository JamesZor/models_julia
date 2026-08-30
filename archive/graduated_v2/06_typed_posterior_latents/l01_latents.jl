# ==============================================================================
# 06 — TYPED POSTERIOR LATENTS : THE TYPE HIERARCHY
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# ------------------------------------------------------------------------------
# WHAT THIS FILE REPLACES
# ------------------------------------------------------------------------------
#
# `Experiments.LatentStates` (src/experiments/post_processing.jl:18) wraps a
# `DataFrame` whose cells each hold a full posterior sample VECTOR:
#
#     match_id │ λ_h                     │ λ_a                     │ …
#     ─────────┼─────────────────────────┼─────────────────────────┼───
#     1234     │ [1.31, 1.29, …] (3200)  │ [0.94, 0.97, …] (3200)  │ …
#
# It is built by `_latent_state_dict_to_df` (post_processing.jl:205), which
# constructs its columns as `Vector{Any}`. That one line is the root of every
# problem this prototype exists to fix:
#
#   1. TYPE INSTABILITY. `Vector{Any}` means every `row.λ_h` is a dynamic dispatch
#      returning a boxed `Any`. `Predictions.extract_params` (interface.jl:10) then
#      hands that `Any` to `compute_score_matrix`, which cannot specialise on it, so
#      the score-grid kernel — the hottest loop in the whole prediction path — is
#      compiled against `Any` element access.
#
#   2. FRAGMENTATION. N_matches × N_parameters separately-allocated vectors, each
#      with its own header, scattered across the heap. 500 fixtures × 4 parameters
#      × 3200 draws is 6,400,000 Float64 in 2,000 allocations instead of 4.
#
#   3. AD-HOC SCHEMA. Every engine invents its own column names, and every consumer
#      re-discovers them with `hasproperty` chains:
#        - negativebinomial.jl:9-27   `:r` or `:r_h`/`:r_a`?
#        - recombination.jl:18-40     `:λ_h`? `:μ_total_h`? `:μ_open_h` + 5 more?
#      Each `hasproperty` branch is a schema that is checked at PREDICTION time,
#      three hours after the run that would have to be repeated to fix it.
#
# ------------------------------------------------------------------------------
# WHAT REPLACES IT
# ------------------------------------------------------------------------------
#
# One dense `Matrix{Float64}` per parameter, shared by every fixture:
#
#     λ_home :: Matrix{Float64}   # (n_matches × n_draws)
#
# The schema is now the TYPE. `CountLatents{Float64, Nothing}` is a Poisson model's
# posterior and cannot be anything else; `CountLatents{Float64, <:NamedTuple}` carries
# dispersion and cannot be read as if it did not. The `hasproperty` chains above
# become method dispatch, resolved at compile time, and a missing parameter is a
# `MethodError` at construction rather than a wrong price at settlement.
#
# ------------------------------------------------------------------------------
# THE LAYOUT DECISION, AND ITS ONE COST
# ------------------------------------------------------------------------------
#
# The briefing specifies `(n_matches, n_draws)`. Julia is column-major, so the draws
# for ONE fixture are strided by `n_matches`, not contiguous. That is a real cost and
# it is worth being explicit about why it is the right trade anyway:
#
#   * The score-grid kernel reads TWO scalars per draw (`λ_home[i,k]`, `λ_away[i,k]`)
#     and writes `max_goals²` = 144 contiguous cells. The write side dominates by ~70x,
#     so the strided read is noise. §11 of r01_demo.jl measures this rather than
#     asserting it.
#   * The alternative layout `(n_draws, n_matches)` makes a single fixture contiguous
#     but makes the far more common cross-fixture reductions (calibration, CLV,
#     portfolio aggregation — all of which sweep every fixture at a fixed draw) strided
#     instead. Those sweeps have no 144-cell write to hide behind.
#
# So: `(n_matches, n_draws)`, direct scalar indexing `A[i, k]` in the kernels, and NO
# per-fixture `view`. A `SubArray` would be the natural way to express "this fixture's
# draws", but constructing one inside the kernel is a heap allocation the escape
# analyser is not guaranteed to elide — and the zero-allocation claim in §10 is only
# worth making if it does not depend on an optimisation that might not fire.
#
# ==============================================================================

using LinearAlgebra
using Printf


# ==============================================================================
# 1. THE ABSTRACT CONTRACT
# ==============================================================================

"""
    AbstractPosteriorLatents

A dense, typed container for the posterior draws of every out-of-sample fixture in
one training fold.

Every concrete subtype must satisfy:

  * `latent_match_ids(l) -> Vector{Int}`, in the row order of the parameter matrices.
  * `n_matches(l)`, `n_draws(l)` — the two dimensions every parameter matrix shares.
  * `latent_matrices(l) -> NamedTuple` — every stored `(n_matches × n_draws)` matrix,
    by name. Used by validation, memory accounting and the parity harness; NOT on
    any hot path.
  * A `compute_score_grid!` method in `l03_score_grids.jl`.

The interface deliberately does NOT include a "give me fixture i's parameters"
accessor. Any such method must either allocate or return a `SubArray`, and the
kernels are written to need neither.
"""
abstract type AbstractPosteriorLatents end


"""
    n_matches(l::AbstractPosteriorLatents) -> Int

Number of out-of-sample fixtures held in the container.
"""
n_matches(l::AbstractPosteriorLatents) = length(latent_match_ids(l))

"""
    n_draws(l::AbstractPosteriorLatents) -> Int

Number of posterior draws per fixture — `n_chains × n_samples` of the source
`MCMCChains.Chains`, flattened.
"""
function n_draws end

"""
    latent_match_ids(l::AbstractPosteriorLatents) -> Vector{Int}

The fixture ids, in the row order of every parameter matrix.
"""
latent_match_ids(l::AbstractPosteriorLatents) = l.match_ids

"""
    latent_matrices(l::AbstractPosteriorLatents) -> NamedTuple

Every `(n_matches × n_draws)` parameter matrix the container holds, keyed by the
name the engine gave it. Diagnostics only.
"""
function latent_matrices end

"""
    match_index(l::AbstractPosteriorLatents, match_id::Integer) -> Int

Row index of `match_id`, or `0` if the fixture is not in this container.

LINEAR SCAN, ON PURPOSE. The briefing fixes the struct fields, and a cached
`Dict{Int,Int}` is not one of them. It is also not needed: this is a setup-time
lookup (a caller resolves a fixture id once, then loops over draws), and every hot
path in `l03_score_grids.jl` takes the integer row index directly. A fold holds
hundreds of fixtures, not millions.
"""
function match_index(l::AbstractPosteriorLatents, match_id::Integer)
    idx = findfirst(==(Int(match_id)), latent_match_ids(l))
    return idx === nothing ? 0 : idx
end

"""
    latent_bytes(l::AbstractPosteriorLatents) -> Int

Total heap bytes held by the parameter matrices. The number to compare against the
equivalent `latents.df`, which additionally pays one `Vector` header (40 bytes) plus
allocator rounding per fixture per parameter.
"""
latent_bytes(l::AbstractPosteriorLatents) = sum(sizeof, values(latent_matrices(l)); init = 0)

"""
    latent_allocations(l::AbstractPosteriorLatents) -> Int

Number of distinct heap objects backing the parameters. This is the headline
comparison: a `CountLatents` is 2 (or 4) allocations regardless of fold size, where
`latents.df` is `n_matches × n_parameters`.
"""
latent_allocations(l::AbstractPosteriorLatents) = length(latent_matrices(l))


# ==============================================================================
# 2. VALIDATION
# ==============================================================================
#
# Run once, at construction, on the boundary between the un-typed world (a `Chains`
# object, a legacy DataFrame) and the typed one. Everything downstream is then free
# to assume it.
#
# These checks are not defensive padding. Each one corresponds to a failure that
# currently surfaces as a plausible-looking PRICE:
#
#   mismatched rows  -> fixture i priced with fixture j's posterior
#   non-finite λ     -> `pdf(Poisson(NaN), k)` is NaN, the grid sums to NaN, the
#                       market probability is NaN, and Kelly stakes it as zero edge
#   λ ≤ 0            -> `Poisson(-0.3)` throws three hours into a backtest

"""
    tpl_validate_panel(what, match_ids, named_matrices)

Check that every parameter matrix is `(length(match_ids) × n_draws)` for one common
`n_draws`, and that every entry is finite. Throws with the offending name and the
first offending index.
"""
function tpl_validate_panel(what::AbstractString,
                            match_ids::Vector{Int},
                            named_matrices)
    isempty(named_matrices) && error("$what: no parameter matrices supplied")

    n_m = length(match_ids)
    allunique(match_ids) ||
        error("$what: match_ids contains duplicates — a fixture would be priced twice " *
              "from two different posteriors")

    first_name, first_mat = first(named_matrices)
    n_d = size(first_mat, 2)
    n_d > 0 || error("$what: $(first_name) has zero draws")

    for (name, mat) in named_matrices
        size(mat, 1) == n_m || error(
            "$what: $(name) has $(size(mat, 1)) rows but there are $n_m match_ids. " *
            "A row/id misalignment prices fixture i with fixture j's posterior.")
        size(mat, 2) == n_d || error(
            "$what: $(name) has $(size(mat, 2)) draws but $(first_name) has $n_d. " *
            "Every parameter must come from the same posterior sweep.")
        bad = findfirst(!isfinite, mat)
        bad === nothing || error(
            "$what: $(name) has a non-finite entry at $(Tuple(bad)) " *
            "(value $(mat[bad])). NaN propagates silently through the score grid " *
            "into a NaN market probability.")
    end
    return nothing
end

"""
    tpl_validate_positive(what, named_matrices)

Additionally require strict positivity. Applied to intensities (λ) and dispersions
(r), which are arguments to `Poisson`/`RobustNegativeBinomial` constructors.
"""
function tpl_validate_positive(what::AbstractString, named_matrices)
    for (name, mat) in named_matrices
        bad = findfirst(<=(zero(eltype(mat))), mat)
        bad === nothing || error(
            "$what: $(name) has a non-positive entry at $(Tuple(bad)) " *
            "(value $(mat[bad])). Rates and dispersions must be strictly positive.")
    end
    return nothing
end

"""
    tpl_validate_observation(obs, panel_size, what)

The observation-parameter slot accepts exactly two shapes, and nothing else:

  * `nothing`        — Poisson. No dispersion parameter exists.
  * `(; r_h, r_a)`   — Negative binomial. Two `(n_matches × n_draws)` dispersion
                       matrices, per-side because `HomeAwayDispersion` and
                       `AdvancedVolatilityDispersion` both produce asymmetric r.

MATERIALISATION, AND WHAT IT COSTS. `GlobalDispersion` and `HomeAwayDispersion` produce
an r that is the SAME for every fixture — `reconstruct_dispersion` hands back the same
two vector objects on every row (dispersion.jl:151-153), so the legacy DataFrame stores
one object referenced `n_matches` times. This container copies it into two full
`(n_matches × n_draws)` matrices instead, which on a 500-fixture fold is ~25 MB where
the shared form would be ~51 KB.

That is a real cost, taken deliberately, for two reasons:

  1. `AdvancedVolatilityDispersion` genuinely varies r per fixture (it composes team and
     month volatility terms), so the matrix shape is required in general. A container
     that were sometimes-shared and sometimes-not would reintroduce, as a memory-layout
     convention, exactly the implicit schema this file exists to delete — and it would
     be an UNCHECKED one, since nothing in the type would say which.
  2. The uniform shape is what lets one grid kernel read `r_h[i, k]` with no branch and
     no dispatch, which is where the measured speedup in `r01_demo.jl` §11 comes from.

`r01_demo.jl` §11 reports the byte cost rather than hiding it: for the NegBin and smile
families the typed container is LARGER than the legacy frame, and that is the correct
number to publish.
"""
tpl_validate_observation(::Nothing, ::Tuple{Int,Int}, ::AbstractString) = nothing

function tpl_validate_observation(obs::NamedTuple, panel::Tuple{Int,Int}, what::AbstractString)
    keys(obs) == (:r_h, :r_a) || error(
        "$what: observation_params must be `nothing` (Poisson) or `(; r_h, r_a)` " *
        "(negative binomial); got fields $(keys(obs)).")
    for name in (:r_h, :r_a)
        size(obs[name]) == panel || error(
            "$what: observation_params.$(name) is $(size(obs[name])), expected $panel.")
    end
    tpl_validate_panel(what, collect(1:panel[1]), (:r_h => obs.r_h, :r_a => obs.r_a))
    tpl_validate_positive(what, (:r_h => obs.r_h, :r_a => obs.r_a))
    return nothing
end

tpl_validate_observation(obs, ::Tuple{Int,Int}, what::AbstractString) = error(
    "$what: observation_params must be `nothing` or a `(; r_h, r_a)` NamedTuple; " *
    "got a $(typeof(obs)).")


# ==============================================================================
# 3. COUNT LATENTS — the double-intensity family
# ==============================================================================

"""
    CountLatents(match_ids, λ_home, λ_away, observation_params = nothing)

Posterior latents for any model whose score grid is the outer product of two
independent count marginals.

Covers, today:

  * every `ComposableCountModel` from `05_composable_count_builder`
    (Poisson and negative-binomial observations, any covariate set);
  * `AbstractPoissonModel` / `AbstractNegBinModel` team- and player-level engines
    in `src/models/pregame/engines/`.

| field                | shape                  | meaning                                  |
|----------------------|------------------------|------------------------------------------|
| `match_ids`          | `n_matches`            | fixture ids, row order of the matrices   |
| `λ_home`             | `n_matches × n_draws`  | home scoring intensity                   |
| `λ_away`             | `n_matches × n_draws`  | away scoring intensity                   |
| `observation_params` | —                      | `nothing`, or `(; r_h, r_a)` dispersions |

The observation slot is a TYPE PARAMETER, so `CountLatents{Float64,Nothing}` and
`CountLatents{Float64,<:NamedTuple}` reach different `compute_score_grid!` methods
with no runtime branch — the compile-time replacement for
`negativebinomial.jl:9-27`'s `hasproperty` chain.
"""
struct CountLatents{T<:Real, Obs} <: AbstractPosteriorLatents
    match_ids::Vector{Int}
    λ_home::Matrix{T}
    λ_away::Matrix{T}
    observation_params::Obs

    function CountLatents(match_ids::AbstractVector{<:Integer},
                          λ_home::AbstractMatrix{T},
                          λ_away::AbstractMatrix{T},
                          observation_params::Obs = nothing) where {T<:Real, Obs}
        ids = Vector{Int}(match_ids)
        λh  = Matrix{T}(λ_home)
        λa  = Matrix{T}(λ_away)
        mats = (:λ_home => λh, :λ_away => λa)
        tpl_validate_panel("CountLatents", ids, mats)
        tpl_validate_positive("CountLatents", mats)
        tpl_validate_observation(observation_params, size(λh), "CountLatents")
        return new{T, Obs}(ids, λh, λa, observation_params)
    end
end

n_draws(l::CountLatents) = size(l.λ_home, 2)

latent_matrices(l::CountLatents{T, Nothing}) where {T} =
    (; λ_home = l.λ_home, λ_away = l.λ_away)

latent_matrices(l::CountLatents{T, <:NamedTuple}) where {T} =
    (; λ_home = l.λ_home, λ_away = l.λ_away,
       r_h = l.observation_params.r_h, r_a = l.observation_params.r_a)

"The count density this container's grid kernel will use. Dispatch, not a flag."
observation_family(::CountLatents{T, Nothing}) where {T} = :poisson
observation_family(::CountLatents{T, <:NamedTuple}) where {T} = :negbin


# ==============================================================================
# 4. RECOMBINATION LATENTS — the additive-channel family
# ==============================================================================

"""
    RecombLatents(match_ids, λ_open_h, λ_open_a, λ_pen_h, λ_pen_a,
                  λ_og_h, λ_og_a, pxg_h, pxg_a)

Posterior latents for the recombination engines (`DynamicPxGRecombModel`,
`DynamicRecombinedGoalsModel`), which decompose a team's goal rate into three
independent Poisson channels:

    λ_total = λ_open + λ_pen + λ_og

CHANNEL CONVENTIONS — these are GOAL rates, not event rates. The distinction is the
one place a recombination model is easy to get silently wrong:

| field      | is                                                  | is NOT                        |
|------------|-----------------------------------------------------|-------------------------------|
| `λ_open_h` | open-play GOALS `μ_open · κ` (finishing applied)     | the shot/pxG intensity        |
| `λ_pen_h`  | penalty GOALS `q_pen · λ_pen_awarded` (conversion)   | penalties AWARDED             |
| `λ_og_h`   | own goals conceded by the opponent, benefiting home  | own goals scored BY home      |
| `pxg_h`    | open-play proxy-xG intensity `μ_open`, pre-finishing | a goal rate                   |

`pxg_h`/`pxg_a` are carried for diagnostics and for the pxG-vs-goals residual work;
they are NOT summed into the total. `λ_open = pxg · κ` recovers the finishing factor
without storing κ, which no downstream consumer reads on its own.

The sum is formed left-to-right as `λ_open + λ_pen + λ_og`, matching
`Predictions.extract_params(::AbstractRecombinationModels, row)`
(src/predictions/score_computation/recombination.jl:36-37) term for term. Floating-point
addition is not associative, so reproducing the ORDER — not merely the algebra — is what
makes the parity in `l04_parity.jl` exact rather than approximate.
"""
struct RecombLatents{T<:Real} <: AbstractPosteriorLatents
    match_ids::Vector{Int}
    λ_open_h::Matrix{T}
    λ_open_a::Matrix{T}
    λ_pen_h::Matrix{T}
    λ_pen_a::Matrix{T}
    λ_og_h::Matrix{T}
    λ_og_a::Matrix{T}
    pxg_h::Matrix{T}
    pxg_a::Matrix{T}

    function RecombLatents(match_ids::AbstractVector{<:Integer},
                           λ_open_h::AbstractMatrix{T}, λ_open_a::AbstractMatrix{T},
                           λ_pen_h::AbstractMatrix{T},  λ_pen_a::AbstractMatrix{T},
                           λ_og_h::AbstractMatrix{T},   λ_og_a::AbstractMatrix{T},
                           pxg_h::AbstractMatrix{T},    pxg_a::AbstractMatrix{T}) where {T<:Real}
        ids = Vector{Int}(match_ids)
        m   = map(Matrix{T}, (λ_open_h, λ_open_a, λ_pen_h, λ_pen_a,
                              λ_og_h, λ_og_a, pxg_h, pxg_a))
        names = (:λ_open_h, :λ_open_a, :λ_pen_h, :λ_pen_a,
                 :λ_og_h, :λ_og_a, :pxg_h, :pxg_a)
        mats = Tuple(names[i] => m[i] for i in eachindex(names))
        tpl_validate_panel("RecombLatents", ids, mats)
        # Channels may legitimately be zero (a league with no recorded own goals),
        # so positivity is required of the TOTAL, not of each channel.
        tpl_validate_positive("RecombLatents",
                              (:λ_total_h => m[1] .+ m[3] .+ m[5],
                               :λ_total_a => m[2] .+ m[4] .+ m[6],
                               :pxg_h => m[7], :pxg_a => m[8]))
        return new{T}(ids, m...)
    end
end

n_draws(l::RecombLatents) = size(l.λ_open_h, 2)

latent_matrices(l::RecombLatents) = (;
    λ_open_h = l.λ_open_h, λ_open_a = l.λ_open_a,
    λ_pen_h  = l.λ_pen_h,  λ_pen_a  = l.λ_pen_a,
    λ_og_h   = l.λ_og_h,   λ_og_a   = l.λ_og_a,
    pxg_h    = l.pxg_h,    pxg_a    = l.pxg_a,
)

observation_family(::RecombLatents) = :recombination

"""
    recomb_total_home(l, i, k) -> Float64
    recomb_total_away(l, i, k) -> Float64

The recombined goal intensity for fixture row `i`, draw `k`. Summed in the same
left-to-right order the legacy reader uses, so the result is bit-identical and not
merely equal to rounding.
"""
@inline recomb_total_home(l::RecombLatents, i::Int, k::Int) =
    @inbounds l.λ_open_h[i, k] + l.λ_pen_h[i, k] + l.λ_og_h[i, k]

@inline recomb_total_away(l::RecombLatents, i::Int, k::Int) =
    @inbounds l.λ_open_a[i, k] + l.λ_pen_a[i, k] + l.λ_og_a[i, k]


# ==============================================================================
# 5. SMILE LATENTS — the grid-plus-pricing-curve family
# ==============================================================================

"""
    SmileLatents(match_ids, λ_home, λ_away, observation_params,
                 λ_tot, φ, strikes)

Posterior latents for the market-smile engines
(`DynamicSmileDoublePoisson…`, `DynamicSmileDoubleNegBin…`).

A smile model prices TWO WAYS AT ONCE and the container has to carry both:

  * 1X2 / BTTS / correct-score come from the ordinary `(λ_home, λ_away)` grid;
  * per-line Over/Under comes from its OWN per-strike intensity
    `Λ(K) = λ_tot · φ(K)`, priced as `P(N ≤ K) = cdf(Poisson(Λ(K)), K)`.

Collapsing this into a `CountLatents` would silently price O/U off the grid and
"de-smile" the model — the exact failure `src/predictions/score_computation/smile_poisson.jl`
warns about in its header. Making it a separate TYPE makes that collapse impossible
rather than merely discouraged.

| field                | shape                             | meaning                          |
|----------------------|-----------------------------------|----------------------------------|
| `λ_home`, `λ_away`   | `n_matches × n_draws`             | grid intensities                 |
| `observation_params` | —                                 | `nothing`, or `(; r_h, r_a)`     |
| `λ_tot`              | `n_matches × n_draws`             | market total intensity           |
| `φ`                  | `n_matches × n_strikes × n_draws` | smile shape curve                |
| `strikes`            | `n_strikes`                       | the O/U lines φ is indexed by    |

STRIKE INDEXING. `strikes[s]` is the market line (0.5, 1.5, …), and column `s` of `φ`
is the shape at integer threshold `K = s - 1` — i.e. `strikes[s] == K + 0.5`. This
mirrors `smile_poisson.jl:70`'s `S.Λ[K + 1, s]` exactly, and the constructor enforces
it, because an off-by-one here prices Over 2.5 with the Over 1.5 curve and produces a
perfectly plausible, systematically wrong number.

φ IS PER-FIXTURE HERE, GLOBAL IN THE ENGINES. Today's smile engines learn ONE global
`log_φ` and every fixture shares it (goals_smile_league.jl:208-212 builds one
`n_samples × nK` matrix and stores the same object on every row). The 3-D field costs
`n_matches ×` more memory than that, and buys a per-league or per-fixture smile without
a container change. The alternative — a 2-D field plus a "which fixtures share this"
convention — is the kind of implicit schema this prototype exists to delete.
"""
struct SmileLatents{T<:Real, Obs} <: AbstractPosteriorLatents
    match_ids::Vector{Int}
    λ_home::Matrix{T}
    λ_away::Matrix{T}
    observation_params::Obs
    λ_tot::Matrix{T}
    φ::Array{T, 3}
    strikes::Vector{Float64}

    function SmileLatents(match_ids::AbstractVector{<:Integer},
                          λ_home::AbstractMatrix{T},
                          λ_away::AbstractMatrix{T},
                          observation_params::Obs,
                          λ_tot::AbstractMatrix{T},
                          φ::AbstractArray{T, 3},
                          strikes::AbstractVector{<:Real}) where {T<:Real, Obs}
        ids = Vector{Int}(match_ids)
        λh  = Matrix{T}(λ_home)
        λa  = Matrix{T}(λ_away)
        λt  = Matrix{T}(λ_tot)
        φ3  = Array{T, 3}(φ)
        ks  = Vector{Float64}(strikes)

        mats = (:λ_home => λh, :λ_away => λa, :λ_tot => λt)
        tpl_validate_panel("SmileLatents", ids, mats)
        tpl_validate_positive("SmileLatents", mats)
        tpl_validate_observation(observation_params, size(λh), "SmileLatents")

        size(φ3, 1) == length(ids) || error(
            "SmileLatents: φ has $(size(φ3, 1)) fixtures but there are $(length(ids)) match_ids.")
        size(φ3, 3) == size(λh, 2) || error(
            "SmileLatents: φ has $(size(φ3, 3)) draws but λ_home has $(size(λh, 2)).")
        size(φ3, 2) == length(ks) || error(
            "SmileLatents: φ has $(size(φ3, 2)) strike columns but $(length(ks)) strikes were given.")
        all(isfinite, φ3) || error("SmileLatents: φ has non-finite entries.")
        all(>(0), φ3)     || error("SmileLatents: φ must be strictly positive (it multiplies λ_tot).")

        # The K = s - 1 convention, enforced rather than documented-and-hoped-for.
        for (s, line) in enumerate(ks)
            line ≈ (s - 1) + 0.5 || error(
                "SmileLatents: strikes[$s] = $line but column $s of φ is the shape at " *
                "K = $(s - 1), whose market line is $((s - 1) + 0.5). Strikes must be " *
                "0.5, 1.5, … and contiguous from K = 0 — see smile_poisson.jl:70.")
        end

        return new{T, Obs}(ids, λh, λa, observation_params, λt, φ3, ks)
    end
end

n_draws(l::SmileLatents) = size(l.λ_home, 2)

"Number of O/U strikes the learned smile covers. Lines beyond it fall back to the grid."
n_strikes(l::SmileLatents) = length(l.strikes)

latent_matrices(l::SmileLatents{T, Nothing}) where {T} =
    (; λ_home = l.λ_home, λ_away = l.λ_away, λ_tot = l.λ_tot, φ = l.φ)

latent_matrices(l::SmileLatents{T, <:NamedTuple}) where {T} =
    (; λ_home = l.λ_home, λ_away = l.λ_away, λ_tot = l.λ_tot, φ = l.φ,
       r_h = l.observation_params.r_h, r_a = l.observation_params.r_a)

observation_family(::SmileLatents{T, Nothing}) where {T} = :smile_poisson
observation_family(::SmileLatents{T, <:NamedTuple}) where {T} = :smile_negbin

"""
    smile_intensity(l::SmileLatents, i, s, k) -> Float64

`Λ(K) = λ_tot · φ(K)` for fixture row `i`, strike column `s` (K = s-1), draw `k`.

Same product, same operand order as `smile_poisson.jl:81`'s
`transpose(params.λ_tot .* params.φ)`, so the O/U prices agree to the last bit.
"""
@inline smile_intensity(l::SmileLatents, i::Int, s::Int, k::Int) =
    @inbounds l.λ_tot[i, k] * l.φ[i, s, k]


# ==============================================================================
# 6. DISPLAY
# ==============================================================================
#
# A container printed in a REPL should answer "what is in here, how big is it, and
# what will it price with" without a second command.

function _tpl_human_bytes(b::Integer)
    b < 1024        && return @sprintf("%d B", b)
    b < 1024^2      && return @sprintf("%.1f KiB", b / 1024)
    b < 1024^3      && return @sprintf("%.1f MiB", b / 1024^2)
    return @sprintf("%.2f GiB", b / 1024^3)
end

function _tpl_show_common(io::IO, l::AbstractPosteriorLatents, extra::String = "")
    print(io, nameof(typeof(l)), "(", observation_family(l), ")")
    print(io, "\n  fixtures    : ", n_matches(l))
    print(io, "\n  draws       : ", n_draws(l))
    print(io, "\n  parameters  : ", join(string.(keys(latent_matrices(l))), ", "))
    isempty(extra) || print(io, extra)
    print(io, "\n  memory      : ", _tpl_human_bytes(latent_bytes(l)),
              " in ", latent_allocations(l), " heap object(s)")
    return nothing
end

Base.show(io::IO, ::MIME"text/plain", l::CountLatents)  = _tpl_show_common(io, l)
Base.show(io::IO, ::MIME"text/plain", l::RecombLatents) = _tpl_show_common(io, l)

Base.show(io::IO, ::MIME"text/plain", l::SmileLatents) =
    _tpl_show_common(io, l, "\n  strikes     : " * join(string.(l.strikes), ", "))
