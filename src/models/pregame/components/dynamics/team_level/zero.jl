# src/models/pregame/components/dynamics/team_level/zero.jl
#
# TEAM DYNAMICS SWITCHED OFF: α ≡ β ≡ 0, no sampled parameters.
#
# WHY THIS EXISTS. It is not a model anyone would run on its own — it is the ablation that turns a
# team-dynamics engine into a PURE PLAYER engine, so the two hypotheses can be compared inside one
# code path instead of two engine files that differ in a dozen incidental ways:
#
#   H1  "players ADJUST team strength"  — free per-team α/β PLUS a player pillar. The player term
#       can only earn its keep on the lineup-varying residual, because α/β absorb the team-level
#       component. This is what DynamicGoalsPlusMinusLeagueTimeDecayModel does by default.
#   H2  "players ARE team strength"     — no α/β at all; a team is good because its players are
#       good. This is the form the nine Ireland `outfield_*` xG engines take, via
#       OutfieldPlayerDynamicsConfig, which samples four global positional weights and nothing
#       per-team.
#
# Swapping `dynamics_config = StaticZeroDynamics()` into a team-dynamics engine gives H2 while
# holding EVERYTHING else fixed — same likelihood, same interception, same home advantage, same
# league offset, same pillar parameterisation. A difference between the two arms is then
# attributable to the team-dynamics term alone.
#
# `days_half_life` is carried because engine builders read it off the dynamics config to form the
# likelihood decay weights; it has nothing to do with α/β here.

# ==========================================
# 1. CONFIGURATION
# ==========================================
Base.@kwdef struct StaticZeroDynamics <: AbstractDynamicsConfig
    days_half_life::Real = 180
end

# ==========================================
# 2. TURING SUBMODEL  (samples nothing)
# ==========================================
@model function build_dynamics(config::StaticZeroDynamics, n_teams::Int)
    # Returned as plain Float64 zeros. `view(dyn.α, indices)` in the calling engine then
    # contributes an exact structural zero to the log-rate, and ReverseDiff records no tape node
    # for it — so this really is the engine minus the term, not the term pinned near zero.
    return (; α = zeros(n_teams), β = zeros(n_teams))
end

# ==========================================
# 3. EXTRACTOR
# ==========================================
function extract_dynamics(chain::Chains, ::StaticZeroDynamics, prefix::String, n_teams::Int)
    n_samples = size(chain, 1) * size(chain, 3)
    return (; α = zeros(n_samples, n_teams), β = zeros(n_samples, n_teams))
end
