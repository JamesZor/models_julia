# ==============================================================================
# 05 — Composable Count Model Builder : THE BUILDER AND ITS REFEREE
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# THE ONE IDEA. A model is assembled by repeatedly calling ONE generic function
#
#     add!(builder, component)
#
# which dispatches on the component's ABSTRACT type. There is no `with_dynamics`,
# no `with_interception`, no `with_covariate` — the component already knows what
# it is, so asking the caller to say it again is just a second place to get it
# wrong. Adding a whole new component family is one `add!` method.
#
#     b = CountModelBuilder(:scottish_count)
#     add!(b, GlobalInterception(), TimeDecayDynamics(), GlobalHomeAdvantage())
#     add!(b, WealthCovariate(), DistanceCovariate())
#     model = build(b)
#
# or, if you prefer the pipeline:
#
#     model = CountModelBuilder(:scottish_count) |>
#             add(GlobalInterception(), TimeDecayDynamics(), GlobalHomeAdvantage()) |>
#             add(WealthCovariate()) |>
#             add(DistanceCovariate()) |>
#             build
#
# THE TWO PHASES, AND WHY THEY ARE SEPARATE.
#
#   CountModelBuilder     mutable, abstractly typed, order-remembering, INVALID by
#                         default. Cheap to hold half-built. Never sampled from.
#   build(builder)        runs the referee, then freezes the covariate list into a
#                         TUPLE and returns an immutable, fully concretely-typed
#                         model. This call is the ONE type-unstable boundary in the
#                         whole design; everything downstream of it — the engine,
#                         the tape, extraction — is concrete and static.
#
# ==============================================================================

# Dependencies and component definitions are loaded by builder-module.jl.


# ==============================================================================
# 1. THE ASSEMBLED MODELS
# ==============================================================================
#
# Two structs with identical fields. They differ ONLY in supertype, and the
# supertype is not decoration: `src/predictions/score_computation/` dispatches the
# score grid on it. `AbstractPoissonModel` reaches the double-Poisson grid,
# `AbstractNegBinModel` the negative-binomial grid. A single struct carrying the
# observation as a type parameter cannot do this, because a Julia type's supertype
# cannot depend on its parameters.
#
# This is the real reason a "one model type to rule them all" refactor fails, and
# the resolution is: ONE ENGINE, one struct per PREDICTION FAMILY (not per feature
# combination). Four families instead of twenty-eight engines.

"Composable count model with a Poisson observation. Routes to the double-Poisson score grid."
struct PoissonCountModel{
    I<:CB_PG.AbstractInterceptionConfig,
    T<:CB_PG.AbstractDynamicsConfig,
    H<:CB_PG.AbstractHomeAdvantageConfig,
    C<:Tuple,
    O<:CBPoissonFamilyObservation,
    G<:AbstractRateGuard,
} <: CB_TI.AbstractPoissonModel
    interception::I
    dynamics::T
    home_advantage::H
    covariates::C
    observation::O
    guard::G
end

"Composable count model with a negative-binomial observation. Routes to the NegBin score grid."
struct NegBinCountModel{
    I<:CB_PG.AbstractInterceptionConfig,
    T<:CB_PG.AbstractDynamicsConfig,
    H<:CB_PG.AbstractHomeAdvantageConfig,
    C<:Tuple,
    O<:NegativeBinomialObservation,
    G<:AbstractRateGuard,
} <: CB_TI.AbstractNegBinModel
    interception::I
    dynamics::T
    home_advantage::H
    covariates::C
    observation::O
    guard::G
end

const ComposableCountModel = Union{PoissonCountModel, NegBinCountModel}

# `build` picks the struct from the observation config. Dispatch, not a branch.
_assemble(o::PoissonObservation, i, t, h, c, g)          = PoissonCountModel(i, t, h, c, o, g)
# The joint two-arm observation prices from its Poisson goals arm, so it belongs to the SAME
# prediction family as a plain Poisson. Its Gamma arm is a fit-time likelihood and never touches a
# score grid — which is why widening `O` above is safe rather than a loophole.
_assemble(o::JointGammaPoissonObservation, i, t, h, c, g) = PoissonCountModel(i, t, h, c, o, g)
_assemble(o::NegativeBinomialObservation, i, t, h, c, g) = NegBinCountModel(i, t, h, c, o, g)


# ==============================================================================
# 2. THE BUILDER
# ==============================================================================

"""
    CountModelBuilder(kind = :scottish_count)

A mutable, order-remembering, deliberately-invalid-until-`build`ed accumulator for
one count model.

`kind` is a label, not a dispatch key. It travels into the referee's report so a
transcript says which family of model was being assembled; it does not select an
engine. Selecting an engine from a label is exactly the indirection this design
removes.

`covariates` is a Vector during assembly (order matters — it fixes the parameter
layout) and becomes a Tuple at `build`.

`provenance` records every accepted `add!` in order, so a built model can explain
itself without the caller having kept the script.
"""
mutable struct CountModelBuilder
    kind::Symbol
    interception::Union{Nothing, CB_PG.AbstractInterceptionConfig}
    dynamics::Union{Nothing, CB_PG.AbstractDynamicsConfig}
    home_advantage::Union{Nothing, CB_PG.AbstractHomeAdvantageConfig}
    observation::Union{Nothing, AbstractObservationConfig}
    guard::Union{Nothing, AbstractRateGuard}
    covariates::Vector{AbstractPredictorTerm}
    provenance::Vector{String}
end

CountModelBuilder(kind::Symbol = :scottish_count) =
    CountModelBuilder(kind, nothing, nothing, nothing, nothing, nothing,
                      AbstractPredictorTerm[], String[])

function Base.show(io::IO, b::CountModelBuilder)
    slot(x) = x === nothing ? "—" : string(nameof(typeof(x)))
    print(io, "CountModelBuilder(:", b.kind, ")")
    print(io, "\n  interception   : ", slot(b.interception))
    print(io, "\n  dynamics       : ", slot(b.dynamics))
    print(io, "\n  home advantage : ", slot(b.home_advantage))
    print(io, "\n  observation    : ", b.observation === nothing ?
              "— (defaults to PoissonObservation)" : slot(b.observation))
    print(io, "\n  rate guard     : ", b.guard === nothing ?
              "— (defaults to ClampGuard)" : guard_describe(b.guard))
    print(io, "\n  predictors     : ", isempty(b.covariates) ? "none" :
              join(string.(predictor_name.(b.covariates)), ", "))
end


# ==============================================================================
# 3. add! — ONE GENERIC FUNCTION, DISPATCHED ON THE ABSTRACT COMPONENT TYPE
# ==============================================================================

"""
    add!(builder, component) -> builder
    add!(builder, c1, c2, c3...) -> builder

Attach a component. The method is selected by the component's abstract supertype,
so the caller never names the slot.

Structural slots hold exactly one component and REFUSE to be silently overwritten:
a second dynamics config is a modelling mistake far more often than it is a
revision, and a builder that quietly keeps the last one turns that mistake into a
result. Use `replace!` to mean it.
"""
function add! end

add!(b::CountModelBuilder, c::CB_PG.AbstractInterceptionConfig)   = _set!(b, :interception, c)
add!(b::CountModelBuilder, c::CB_PG.AbstractDynamicsConfig)       = _set!(b, :dynamics, c)
add!(b::CountModelBuilder, c::CB_PG.AbstractHomeAdvantageConfig)  = _set!(b, :home_advantage, c)
add!(b::CountModelBuilder, c::AbstractObservationConfig)          = _set!(b, :observation, c)
add!(b::CountModelBuilder, c::AbstractRateGuard)                  = _set!(b, :guard, c)

# Convenience: the raw `src` observation-shaping components name their own
# observation family unambiguously, so accept them directly and wrap.
add!(b::CountModelBuilder, c::CB_PG.AbstractDispersionConfig) = add!(b, NegativeBinomialObservation(c))
add!(b::CountModelBuilder, c::CB_PG.AbstractDixonColesConfig) = add!(b, DixonColesCorrelation(c))
add!(b::CountModelBuilder, c::CB_PG.AbstractCopulaConfig)     = add!(b, FrankCopulaCorrelation(c))

function add!(b::CountModelBuilder, c::AbstractPredictorTerm)
    push!(b.covariates, c)
    push!(b.provenance, "add! predictor  $(predictor_name(c)) :: $(nameof(typeof(c)))")
    return b
end

# Variadic. `foreach` in insertion order — the covariate order IS the parameter
# layout, so it must not be reordered for convenience anywhere.
function add!(b::CountModelBuilder, c1, c2, cs...)
    foreach(c -> add!(b, c), (c1, c2, cs...))
    return b
end

# Anything else is a mistake worth a loud, specific message.
add!(b::CountModelBuilder, c) = error(
    "CountModelBuilder does not know where to put a $(typeof(c)).\n" *
    "  Components are routed by ABSTRACT supertype. Accepted families:\n" *
    "    AbstractInterceptionConfig, AbstractDynamicsConfig, AbstractHomeAdvantageConfig,\n" *
    "    AbstractObservationConfig (or a raw AbstractDispersionConfig / AbstractDixonColesConfig /\n" *
    "    AbstractCopulaConfig), AbstractRateGuard, AbstractPredictorTerm.\n" *
    "  To add a new family, write one `add!` method — see builder/builder.jl §3.")

"""
    add(components...) -> builder -> builder

Curried `add!`, so a builder can be piped:
`CountModelBuilder() |> add(dyn) |> add(cov) |> build`.
"""
add(cs...) = b -> add!(b, cs...)

"""
    replace!(builder, component) -> builder

Overwrite an occupied structural slot on purpose. Recorded in the provenance.
"""
function Base.replace!(b::CountModelBuilder, c)
    slot = _slot_of(b, c)
    slot === nothing && return add!(b, c)
    old = getfield(b, slot)
    setfield!(b, slot, c)
    push!(b.provenance, "replace! $(slot)  $(nameof(typeof(old))) -> $(nameof(typeof(c)))")
    return b
end

_slot_of(::CountModelBuilder, ::CB_PG.AbstractInterceptionConfig)  = :interception
_slot_of(::CountModelBuilder, ::CB_PG.AbstractDynamicsConfig)      = :dynamics
_slot_of(::CountModelBuilder, ::CB_PG.AbstractHomeAdvantageConfig) = :home_advantage
_slot_of(::CountModelBuilder, ::AbstractObservationConfig)         = :observation
_slot_of(::CountModelBuilder, ::AbstractRateGuard)                = :guard
_slot_of(::CountModelBuilder, ::Any)                               = nothing

function _set!(b::CountModelBuilder, slot::Symbol, c)
    occupant = getfield(b, slot)
    occupant === nothing || error(
        "$(slot) is already set to $(nameof(typeof(occupant))); refusing to overwrite it with " *
        "$(nameof(typeof(c))).\n  Use `replace!(builder, component)` if the overwrite is intended, " *
        "or start a fresh CountModelBuilder.")
    setfield!(b, slot, c)
    push!(b.provenance, "add! $(rpad(String(slot), 14)) $(nameof(typeof(c)))")
    return b
end


# ==============================================================================
# 4. THE REFEREE
# ==============================================================================
#
# `validate` never throws: it returns a table, so a runner can print the whole
# thing and see every problem at once. `build` throws on the first failing row.
#
# Every rule here exists because breaking it produces a model that RUNS. None of
# these are type errors; all of them are silent-wrong-answer errors.

"Uniform validation row, shaped like the protocol's `sl_result` so `sl_gate_table` prints it."
cb_result(name, pass::Bool, detail) = (; name = String(name), pass, detail = String(detail))

"""
    _cb_specialised(hook, argtypes) -> Bool

Does `hook` have a method for this covariate that is NOT the abstract fallback?

`hasmethod` alone is useless here: the contract's fallbacks are defined on
`AbstractCovariateConfig`, so every covariate "has" every method — the fallback
just throws when called. A method is a real implementation only if the type it
dispatches on is narrower than the abstract type the fallback claims.
"""
function _cb_specialised(hook, argtypes::Type{<:Tuple})
    m = try
        which(hook, argtypes)
    catch
        return false
    end
    return m.sig.parameters[2] !== AbstractCovariateConfig
end

"Chain-site prefixes owned by the structural layer. A covariate may not shadow one."
const CB_RESERVED_PREFIXES = (:inter, :ha, :dyn, :disp, :dc, :cop, :obs, :cov)

_cb_interception_supported(::CB_PG.AbstractInterceptionConfig) = false
_cb_interception_supported(::Union{
    CB_PG.GlobalInterception,
    CB_PG.SeasonalInterception,
    CB_PG.HierarchicalMonthlyInterception,
}) = true

_cb_dynamics_supported(::CB_PG.AbstractDynamicsConfig) = false
_cb_dynamics_supported(::Union{
    CB_PG.TimeDecayDynamics,CB_PG.StaticZeroDynamics,
}) = true

_cb_home_advantage_supported(::CB_PG.AbstractHomeAdvantageConfig) = false
_cb_home_advantage_supported(::Union{
    CB_PG.GlobalHomeAdvantage,
    CB_PG.HierarchicalTeamHomeAdvantage,
}) = true

"""
    validate(builder) -> Vector of (name, pass, detail)

The structural invariants of a composable count model. Returns the full table.
"""
const BuilderValidationRow = NamedTuple{
    (:name, :pass, :detail),
    Tuple{String, Bool, String},
}

function validate(b::CountModelBuilder)
    out = BuilderValidationRow[]
    obs   = b.observation === nothing ? PoissonObservation() : b.observation
    guard = b.guard === nothing ? ClampGuard() : b.guard

    # --- R1-R3: the linear predictor must be completely specified --------------
    # Every one of these has a defensible "sensible default", and every one of
    # those defaults is a modelling decision. A model that silently acquired
    # `GlobalHomeAdvantage()` because nobody mentioned home advantage is a model
    # nobody chose.
    for (slot, label) in ((:interception, "interception"), (:dynamics, "dynamics"),
                          (:home_advantage, "home advantage"))
        c = getfield(b, slot)
        push!(out, cb_result("exactly one $label", c !== nothing,
            c === nothing ? "MISSING — add!(builder, <an $(label) config>)" :
                            string(nameof(typeof(c)))))
    end

    # The shared predictor has an explicit structural contract. Other existing
    # PreGame components remain fully backward-compatible in their legacy engines,
    # but are refused here until an adapter defines the α/β or indexing semantics.
    supported = (
        (:interception, b.interception,
         b.interception !== nothing && _cb_interception_supported(b.interception)),
        (:dynamics, b.dynamics,
         b.dynamics !== nothing && _cb_dynamics_supported(b.dynamics)),
        (:home_advantage, b.home_advantage,
         b.home_advantage !== nothing && _cb_home_advantage_supported(b.home_advantage)),
    )
    for (slot, component, pass) in supported
        detail = component === nothing ? "missing" : pass ?
                 string(nameof(typeof(component))) :
                 "$(nameof(typeof(component))) has no composable predictor adapter"
        push!(out, cb_result("$slot supported by composable engine", pass, detail))
    end

    # --- R4: likelihood weighting is explicit per dynamics family --------------
    dyn = b.dynamics
    valid_weighting = dyn !== nothing && _dynamics_weighting_valid(dyn)
    weighting_detail = dyn === nothing ? "no dynamics config" :
                       _dynamics_weighting_detail(dyn)
    push!(out, cb_result("dynamics likelihood weighting is valid",
        valid_weighting, weighting_detail))

    lineup_terms = [t for t in b.covariates if t isa PlayerLineupPillar]
    lineup_count_valid = length(lineup_terms) <= 1
    push!(out, cb_result("at most one player-lineup pillar", lineup_count_valid,
        lineup_count_valid ? "$(length(lineup_terms)) lineup pillar(s)" :
        "multiple lineup pillars share one feature bridge; configure at most one"))

    player_priors_valid = all(t ->
        t.w_att_prior isa ContinuousUnivariateDistribution &&
        t.w_def_prior isa ContinuousUnivariateDistribution &&
        (t.w_bench_prior === nothing ||
         t.w_bench_prior isa ContinuousUnivariateDistribution), lineup_terms)
    push!(out, cb_result("player-lineup priors are continuous univariate",
        player_priors_valid,
        isempty(lineup_terms) ? "no player-lineup pillar" :
        player_priors_valid ? "attack, defence, and optional bench priors are valid" :
        "w_att_prior, w_def_prior, and optional w_bench_prior must be continuous univariate"))

    bench_weight_valid = all(t ->
        !(t.aggregation isa BenchWeightedPlayerAggregation) ||
        (isfinite(t.aggregation.w_bench) && 0.0 <= t.aggregation.w_bench <= 1.0),
        lineup_terms)
    push!(out, cb_result("fixed bench weight is in [0, 1]",
        bench_weight_valid,
        isempty(lineup_terms) ? "no player-lineup pillar" :
        bench_weight_valid ? "all fixed bench weights are in [0, 1]" :
        "a fixed bench weight is outside [0, 1]"))

    # --- R5: the observation family must actually have an engine method --------
    push!(out, cb_result("observation is wired", observation_wired(obs),
        observation_wired(obs) ?
            "$(nameof(typeof(obs))) → :$(observation_family(obs)) score grid" :
            "$(nameof(typeof(obs))) is declared but NOT implemented — $(observation_gap(obs))"))

    # --- R5b: the numerical guard is stated, never inherited silently ---------
    guard_valid = guard isa NoGuard ||
                  (guard isa ClampGuard && isfinite(guard.lo) && isfinite(guard.hi) &&
                   guard.lo < guard.hi)
    push!(out, cb_result("rate guard declared and valid", guard_valid,
        !guard_valid ? "ClampGuard bounds must be finite with lo < hi" :
        b.guard === nothing ? "ClampGuard() by default — $(guard_describe(guard))" :
                              guard_describe(guard)))

    # RobustNegativeBinomial floors μ at 1e-6. Rather than put a value branch in
    # the compiled tape, require the rate guard to make that floor unreachable.
    negbin_guard_valid = !(obs isa NegativeBinomialObservation) ||
        (guard isa ClampGuard && guard.lo >= log(1e-6))
    push!(out, cb_result("NegBin guard stays above the legacy mean floor",
        negbin_guard_valid,
        !(obs isa NegativeBinomialObservation) ? "not a NegBin observation" :
        negbin_guard_valid ? "minimum η = $(guard.lo), above log(1e-6)" :
        "NegBin requires ClampGuard(lo >= $(log(1e-6))); NoGuard can cross the legacy μ floor"))

    # The Gamma arm evaluates `exp(-η)`, which is unbounded as η -> -infinity. With NoGuard an early
    # warmup trajectory can drive that to Inf and take the log-density to -Inf on a COMPILED tape,
    # where there is no branch left to catch it. Requiring a finite lower clamp is the same
    # reasoning as the NegBin mean-floor rule above: keep the pathology out of the tape rather than
    # putting a value branch inside it.
    joint_guard_valid = !(obs isa JointGammaPoissonObservation) ||
        (guard isa ClampGuard && isfinite(guard.lo))
    push!(out, cb_result("joint Gamma arm has a finite η floor",
        joint_guard_valid,
        !(obs isa JointGammaPoissonObservation) ? "not a joint observation" :
        joint_guard_valid ? "minimum η = $(guard.lo), so exp(-η) <= $(exp(-guard.lo))" :
        "JointGammaPoissonObservation requires a ClampGuard with a finite lo; the Gamma arm's " *
        "exp(-η) term is unbounded below and NoGuard leaves it uncontrolled"))

    # ν indexes a Gamma SHAPE. A shape at or below 0 is not a density; a shape that can reach 0
    # gives a spike at the origin that no amount of warmup recovers from.
    joint_priors_valid = !(obs isa JointGammaPoissonObservation) ||
        (minimum(obs.shape_prior) > 0.0 && obs.feature isa CB_Features.MatchProxyXGFeature)
    push!(out, cb_result("joint observation priors and feed are well posed",
        joint_priors_valid,
        !(obs isa JointGammaPoissonObservation) ? "not a joint observation" :
        joint_priors_valid ?
            "ν > $(minimum(obs.shape_prior)), fed by $(nameof(typeof(obs.feature)))" :
        minimum(obs.shape_prior) > 0.0 ?
            "feature must be a MatchProxyXGFeature; got $(nameof(typeof(obs.feature)))" :
            "shape_prior must have strictly positive support; got minimum $(minimum(obs.shape_prior))"))

    # --- R6: covariate names are unique ---------------------------------------
    # Two covariates named `:wealth` would both sample the site `wealth.w`. Turing
    # does not stop you; the second silently overwrites the first's contribution in
    # extraction and the posterior is quietly wrong.
    names = Symbol[]
    dup_ok = true
    for c in b.covariates
        n = predictor_name(c)
        n in names && (dup_ok = false)
        push!(names, n)
    end
    push!(out, cb_result("covariate names unique", dup_ok,
        isempty(names) ? "no covariates" :
        dup_ok ? join(string.(names), ", ") :
        "DUPLICATE: $(join(string.([n for n in unique(names) if count(==(n), names) > 1]), ", "))"))

    # --- R7: covariate names may not shadow a structural prefix ---------------
    shadow = [n for n in names if n in CB_RESERVED_PREFIXES]
    push!(out, cb_result("no reserved site prefixes", isempty(shadow),
        isempty(shadow) ? "clear of $(join(string.(CB_RESERVED_PREFIXES), ", "))" :
        "SHADOWED: $(join(string.(shadow), ", "))"))

    # --- R8: the observation layer's own prefixes are clear too ---------------
    clash = [n for n in names if n in observation_prefixes(obs)]
    push!(out, cb_result("no observation site clash", isempty(clash),
        isempty(clash) ? "$(nameof(typeof(obs))) owns $(observation_prefixes(obs))" :
        "covariate(s) $(clash) collide with the observation layer"))

    # --- R9: every covariate implements the whole contract --------------------
    # Checked by calling the config-only half of the contract. The data-dependent
    # half (`covariate_column`, `covariate_oos`) cannot be called without a
    # FeatureSet, so its presence is checked by method lookup instead.
    incomplete = String[]
    for c in b.covariates
        c isa AbstractCovariateConfig || continue
        for hook in (covariate_name, covariate_role, covariate_prior, covariate_features)
            try
                hook(c)
            catch
                push!(incomplete, "$(nameof(typeof(c))).$(nameof(hook))")
            end
        end
        _cb_specialised(covariate_column, Tuple{typeof(c), Any}) ||
            push!(incomplete, "$(nameof(typeof(c))).covariate_column")
        _cb_specialised(covariate_oos, Tuple{typeof(c), Any, Any}) ||
            push!(incomplete, "$(nameof(typeof(c))).covariate_oos")
    end
    push!(out, cb_result("covariate contract complete", isempty(incomplete),
        isempty(incomplete) ? "$(length(b.covariates)) covariate(s), 6/6 methods each" :
        "missing: $(join(incomplete, ", "))"))

    # --- R10: priors on covariate weights are univariate continuous -----------
    bad_prior = String[]
    for c in b.covariates
        c isa AbstractCovariateConfig || continue
        p = try covariate_prior(c) catch; nothing end
        (p isa ContinuousUnivariateDistribution) ||
            push!(bad_prior, String(covariate_name(c)))
    end
    push!(out, cb_result("covariate priors continuous univariate", isempty(bad_prior),
        isempty(bad_prior) ? "all weights are continuous scalars" :
        "not continuous univariate: $(join(bad_prior, ", "))"))

    # --- R11: every covariate declares at least one feature -------------------
    # A covariate whose design vector comes from nowhere is a covariate whose
    # design vector is all zeros, and an all-zero column samples its prior and
    # reports it as a posterior.
    featureless = [String(predictor_name(c)) for c in b.covariates
                   if isempty(try predictor_features(c) catch; [] end)]
    push!(out, cb_result("predictors declare features", isempty(featureless),
        isempty(featureless) ? "$(sum(length(predictor_features(c)) for c in b.covariates; init=0)) feature config(s) contributed" :
        "declare no features: $(join(featureless, ", "))"))

    return out
end

"""
    build_count_model(builder) -> PoissonCountModel | NegBinCountModel

Run the referee, then freeze. Throws with the full failing table if any invariant
is broken; the model can otherwise be trusted to have a well-defined parameter
layout, a non-colliding chain schema, and a score grid that matches its likelihood.

`Tuple(b.covariates)` is where the abstract Vector becomes a concrete Tuple type.
Everything after this line is type-stable, which is what lets the engine unroll the
covariate walk at compile time.
"""
function build_count_model(b::CountModelBuilder)
    report = validate(b)
    failed = [r for r in report if !r.pass]
    if !isempty(failed)
        lines = ["  [FAIL]  $(r.name): $(r.detail)" for r in failed]
        error("CountModelBuilder validation failed ($(length(failed)) of $(length(report)) rules):\n" *
              join(lines, "\n"))
    end
    obs   = b.observation === nothing ? PoissonObservation() : b.observation
    guard = b.guard === nothing ? ClampGuard() : b.guard
    return _assemble(obs, b.interception, b.dynamics, b.home_advantage,
                     Tuple(b.covariates), guard)
end

"Build a count model directly from a label and component sequence."
function build_count_model(kind::Symbol, components...)
    builder = CountModelBuilder(kind)
    isempty(components) || add!(builder, components...)
    return build_count_model(builder)
end

# Prototype-compatible spelling retained inside the builder API. The package root
# exports the less collision-prone `build_count_model` name.
build(b::CountModelBuilder) = build_count_model(b)


# ==============================================================================
# 5. WHAT THE ASSEMBLED MODEL CAN SAY ABOUT ITSELF
# ==============================================================================
#
# All derived from the components. None of it is written per model, which is the
# whole point: a new covariate changes these answers without any of these
# functions being touched.

"All predictor terms, in parameter-layout order."
cb_predictor_terms(m::ComposableCountModel) = m.covariates

"The ordinary scalar covariates, retained as a compatibility view."
cb_covariates(m::ComposableCountModel) = Tuple(t for t in m.covariates if t isa AbstractCovariateConfig)

"Predictor names, in parameter-layout order."
cb_predictor_names(m::ComposableCountModel) = map(predictor_name, m.covariates)

"Ordinary covariate names, in parameter-layout order."
cb_covariate_names(m::ComposableCountModel) = map(covariate_name, cb_covariates(m))

"""
    cb_varinfo_sites(model) -> Vector{Symbol}

The grouped `VarInfo` site names this model produces, derived from the components.
Ordered exactly as the engine declares them, which is the θ layout.
"""
function cb_varinfo_sites(m::ComposableCountModel)
    sites = Symbol[]
    append!(sites, _sites_interception(m.interception))
    append!(sites, _sites_home_advantage(m.home_advantage))
    append!(sites, _sites_dynamics(m.dynamics))
    for term in m.covariates
        append!(sites, predictor_sites(term))
    end
    append!(sites, _sites_observation(m.observation))
    return sites
end

_sites_interception(::CB_PG.GlobalInterception)               = [Symbol("inter.μ")]
_sites_interception(::CB_PG.SeasonalInterception)             = [Symbol("inter.μ")]
_sites_interception(::CB_PG.HierarchicalMonthlyInterception)  =
    [Symbol("inter.μ_base"), Symbol("inter.σ_month"), Symbol("inter.raw_month")]

_sites_home_advantage(::CB_PG.GlobalHomeAdvantage)              = [Symbol("ha.γ_global")]
_sites_home_advantage(::CB_PG.HierarchicalTeamHomeAdvantage)    =
    [Symbol("ha.γ_base"), Symbol("ha.σ_γ"), Symbol("ha.γ_team_raw")]
_sites_dynamics(::CB_PG.TimeDecayDynamics)  =
    [Symbol("dyn.σ_a"), Symbol("dyn.σ_d"), Symbol("dyn.raw_a"), Symbol("dyn.raw_d")]
_sites_dynamics(::CB_PG.StaticZeroDynamics) = Symbol[]

_sites_observation(::PoissonObservation) = Symbol[]
# Declaration order inside `_joint_gamma_poisson_params`, which is the θ layout.
_sites_observation(::JointGammaPoissonObservation) = [Symbol("obs.ν"), Symbol("obs.log_κ")]
_sites_observation(o::NegativeBinomialObservation) = _sites_dispersion(o.dispersion)
_sites_dispersion(::CB_PG.GlobalDispersion)   = [Symbol("disp.log_r")]
_sites_dispersion(::CB_PG.HomeAwayDispersion) = [Symbol("disp.log_r"), Symbol("disp.δ_r_home")]

"""
    cb_chain_columns(model, n_teams; n_seasons=1) -> Vector{String}

The EXPANDED chain column names (vector sites unrolled into `name[i]`). `n_seasons`
is required for multi-season interception components; it defaults to one for the
single-season Scottish protocol and backward compatibility.
"""
function cb_chain_columns(m::ComposableCountModel, n_teams::Int; n_seasons::Int=1)
    n_teams > 0 || error("n_teams must be positive; got $n_teams")
    n_seasons > 0 || error("n_seasons must be positive; got $n_seasons")

    cols = String[]
    for site in cb_varinfo_sites(m)
        name = String(site)
        width = _cb_site_width(m, site, n_teams, n_seasons)
        if width == 1
            push!(cols, name)
        else
            append!(cols, ["$name[$i]" for i in 1:width])
        end
    end
    return cols
end

_cb_site_width(m::ComposableCountModel, site::Symbol, n_teams::Int, n_seasons::Int) =
    _cb_site_width(m.interception, m.home_advantage, site, n_teams, n_seasons)

function _cb_site_width(interception, home_advantage, site::Symbol,
                        n_teams::Int, n_seasons::Int)
    site === Symbol("inter.μ") && interception isa CB_PG.SeasonalInterception &&
        return n_seasons
    site === Symbol("inter.μ_base") &&
        interception isa CB_PG.HierarchicalMonthlyInterception && return n_seasons
    site === Symbol("inter.raw_month") && return 12
    site === Symbol("ha.γ_team_raw") &&
        home_advantage isa CB_PG.HierarchicalTeamHomeAdvantage && return n_teams
    (site === Symbol("dyn.raw_a") || site === Symbol("dyn.raw_d")) && return n_teams
    return 1
end

# Positional convenience for callers that already carry both dimensions.
cb_chain_columns(m::ComposableCountModel, n_teams::Int, n_seasons::Int) =
    cb_chain_columns(m, n_teams; n_seasons)

"""
    cb_parameter_count(model, n_teams; n_seasons=1) -> Int

Number of scalar parameters. Derived from the expanded chain schema.
"""
cb_parameter_count(m::ComposableCountModel, n_teams::Int; n_seasons::Int=1) =
    length(cb_chain_columns(m, n_teams; n_seasons))
cb_parameter_count(m::ComposableCountModel, n_teams::Int, n_seasons::Int) =
    cb_parameter_count(m, n_teams; n_seasons)

function Base.show(io::IO, m::ComposableCountModel)
    fam = string(nameof(typeof(m.observation)))
    covs = isempty(m.covariates) ? "none" : join(string.(cb_predictor_names(m)), " + ")
    print(io, nameof(typeof(m)), "(", fam, "; ",
          nameof(typeof(m.interception)), " + ", nameof(typeof(m.home_advantage)), " + ",
          nameof(typeof(m.dynamics)), "; predictors: ", covs,
          "; guard: ", nameof(typeof(m.guard)), ")")
end
