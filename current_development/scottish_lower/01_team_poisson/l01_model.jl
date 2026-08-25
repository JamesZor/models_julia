# ==============================================================================
# Model 01 — Team-level baseline : CONFIG  (stage ① of the protocol)
# ==============================================================================
#
# Loader. Definitions only, no execution.
#
# This file builds `Models.PreGame.DynamicGoalsTimeDecayModel` from `src`. It does
# NOT reimplement the engine — see docs/PROTOCOL.md § "extend the package, never
# re-implement it". The whole point of model 01 is to drive existing package code
# through the gates.
#
# Read alongside MODEL.md, which states the equations this config selects.
#
# ==============================================================================

using BayesianFootball
using Distributions

const PG       = BayesianFootball.Models.PreGame
const Features = BayesianFootball.Features


# ==============================================================================
# 1. The model constructor
# ==============================================================================

"""
    tp_model(; kwargs...) -> DynamicGoalsTimeDecayModel

The baseline engine. Every keyword is a modelling decision; there are no hidden
defaults beyond the component priors, which are printed by `tp_describe`.

    half_life_days   exponential decay half-life on the likelihood weights.
                     180 is the `src` default. It is NOT a Scottish-derived number
                     — see MODEL.md. Provisional until a gate-6 sweep.
    interception     match-level scoring level (intercept)
    dispersion       negative-binomial dispersion r
    home_advantage   home intensity bump
    sigma_att/def    hyperpriors on the spread of team attack/defence ratings

Swap any component in the REPL and re-run walkthrough blocks ②–⑤; nothing else
needs changing.
"""
function tp_model(;
    half_life_days = 180.0,
    interception   = PG.GlobalInterception(μ = Normal(0.2, 0.1)),
    dispersion     = PG.GlobalDispersion(log_r = Normal(3.1, 0.4)),
    home_advantage = PG.GlobalHomeAdvantage(γ_global = Normal(0.2, 0.2)),
    sigma_att      = Gamma(2.0, 0.15),
    sigma_def      = Gamma(2.0, 0.15),
)
    dynamics = PG.TimeDecayDynamics(
        days_half_life = half_life_days,
        σ_att          = sigma_att,
        σ_def          = sigma_def,
    )

    return PG.DynamicGoalsTimeDecayModel(
        interception_config  = interception,
        dynamics_config      = dynamics,
        dispersion_config    = dispersion,
        homeadvantage_config = home_advantage,
    )
end


# ==============================================================================
# 2. Reporting
# ==============================================================================

"""
    tp_menu()

Print the component menu. The walkthrough calls this at block ① so the available
choices are visible in the transcript, not buried in `src`.
"""
function tp_menu()
    println("=" ^ 74)
    println("MODEL 01 — COMPONENT MENU   (default marked *)")
    println("=" ^ 74)
    println("  Interception     * PG.GlobalInterception(μ)")
    println("                     PG.SeasonalInterception(μ)                  per-season level")
    println("                     PG.HierarchicalMonthlyInterception(...)     + month effects")
    println()
    println("  Dispersion       * PG.GlobalDispersion(log_r)                  r_h = r_a")
    println("                     PG.HomeAwayDispersion(log_r, δ_r_home)")
    println("                     PG.AdvancedVolatilityDispersion(...)        team + month volatility")
    println()
    println("  Home advantage   * PG.GlobalHomeAdvantage(γ_global)")
    println("                     PG.HierarchicalTeamHomeAdvantage(γ_base, σ_γ)")
    println("                     PG.HierarchicalLeagueHomeAdvantage(γ_base, σ_γ)")
    println()
    println("  Dynamics         * PG.TimeDecayDynamics(days_half_life = 180)")
    println("                     half-life is PROVISIONAL — see MODEL.md")
    println("=" ^ 74)
    return nothing
end

"""
    tp_component_fields(label, cfg)

Print a component AND its priors.

The package defines a compact `show` for these configs (`pregame/display.jl`), so
`GlobalInterception()` prints with no fields at all. Gate 1 promises that nothing
about the configuration is hidden, so the fields are read out explicitly rather
than left to `show`.
"""
function tp_component_fields(label::AbstractString, cfg)
    println("  ", rpad(label, 18), ": ", typeof(cfg).name.name)
    for fname in fieldnames(typeof(cfg))
        println("      ", rpad(String(fname), 16), " = ", getfield(cfg, fname))
    end
    return nothing
end

"""
    tp_describe(model)

Print every hyperparameter the model carries, priors included. Gate 1 requires
that nothing about the config is invisible.
"""
function tp_describe(model)
    println("=" ^ 74)
    println("MODEL 01 — CONFIG   [$(sl_hash(model))]")
    println("=" ^ 74)
    println("  engine            : $(typeof(model).name.name)")
    println("-" ^ 74)
    tp_component_fields("interception",   model.interception_config)
    tp_component_fields("dispersion",     model.dispersion_config)
    tp_component_fields("home advantage", model.homeadvantage_config)
    tp_component_fields("dynamics",       model.dynamics_config)
    println("-" ^ 74)
    println("  required features : ")
    for f in Features.required_features(model)
        println("      - $(typeof(f).name.name)")
    end
    println("=" ^ 74)
    return nothing
end


"""
    tp_sampled_sites() -> Vector{String}

The chain variable names this configuration is expected to produce. Gate 4 checks
the fitted chain against this list in both directions: a site here that the chain
lacks, and a site in the chain that extraction never reads.

Valid for the DEFAULT components only. Swap a component and this list changes —
that is deliberate, so the swap cannot pass gate 4 unnoticed.
"""
function tp_sampled_sites(n_teams::Int)
    sites = String["inter.μ", "disp.log_r", "ha.γ_global", "dyn.σ_a", "dyn.σ_d"]
    append!(sites, ["dyn.raw_a[$i]" for i in 1:n_teams])
    append!(sites, ["dyn.raw_d[$i]" for i in 1:n_teams])
    return sites
end
