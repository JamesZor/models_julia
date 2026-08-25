# ==============================================================================
# Scottish Lower — the shared experiment contract
# ==============================================================================
#
# This file holds ONLY decisions that are independent of any model: which
# tournaments and seasons, how folds are cut, which markets form the book, where
# artifacts go, and the sampler budgets.
#
# It deliberately contains NO gate helpers. Gates are written inside the first
# model that needs them and lifted here only on a second use — see
# docs/PLAN.md § "Abstraction order".
#
# Usage (server REPL):
#     include("current_development/scottish_lower/_protocol/config.jl")
#     contract = sl_contract()
#     sl_describe(contract)
#
# ==============================================================================

using SHA
using BayesianFootball

const Data = BayesianFootball.Data


# ==============================================================================
# 1. The contract
# ==============================================================================

"""
    SLContract

Every decision shared by all Scottish Lower models. Anything a model is free to
choose (components, priors, half-life) belongs in the model's own config, not here.

`dev_seasons` is what you are allowed to look at. `sealed_seasons` is not —
see `sl_assert_not_sealed`.
"""
Base.@kwdef struct SLContract
    # --- Data scope -----------------------------------------------------------
    tournaments::Vector{Int}       = [56, 57]      # League One, League Two — pooled
    dev_seasons::Vector{String}    = ["24/25"]     # development + selection
    sealed_seasons::Vector{String} = ["25/26", "26/27"]  # untouched until selection is frozen

    # --- Fold construction ----------------------------------------------------
    history_seasons::Int  = 2
    dynamics_col::Symbol  = :match_biweek          # postponements can still cross a biweek
    warmup_period::Int    = 0
    stop_early::Bool      = true

    # --- Sampler budgets ------------------------------------------------------
    smoke_chains::Int  = 4                         # gate 3: one fold, persisted
    smoke_samples::Int = 500
    smoke_warmup::Int  = 500

    grid_chains::Int   = 4                         # gates 6-7: all dev folds
    grid_samples::Int  = 800
    grid_warmup::Int   = 800
    queue_tasks::Int   = 16                        # physical cores on the server

    accept_rate::Float64 = 0.65
    max_depth::Int       = 10
    seed::Int            = 20260825

    # --- Pricing --------------------------------------------------------------
    max_goals::Int              = 12               # score-matrix truncation
    totals_lines::Vector{Float64} = [0.5, 1.5, 2.5, 3.5]
    markets::Vector{Symbol}     = [:match_odds, :over_under, :btts]

    # --- Staking (gate 7) -----------------------------------------------------
    commission::Float64          = 0.02            # Betfair, net
    portfolio_kelly_cap::Float64 = 0.20            # Σ simultaneous stakes

    # --- Paths ----------------------------------------------------------------
    artifact_root::String = "data/scottish_lower"
end

sl_contract(; kwargs...) = SLContract(; kwargs...)


# ==============================================================================
# 2. Derived objects
# ==============================================================================

"""
    sl_splitter(contract) -> Data.GroupedCVConfig

The walk-forward splitter for the DEVELOPMENT seasons, with 56 and 57 pooled into
one tournament group.

Note what the resulting boundaries mean, because it is easy to get wrong:
`history_match_ids` + `target_match_ids` are ALL observations through step `t`, and
all of them are fitted. The genuinely held-out fixtures are step `t+1`, obtained
separately via `Data.get_next_matches(ds, (boundary, meta), splitter)`.
"""
function sl_splitter(contract::SLContract = sl_contract())
    return Data.GroupedCVConfig(
        tournament_groups = [contract.tournaments],
        target_seasons    = contract.dev_seasons,
        history_seasons   = contract.history_seasons,
        dynamics_col      = contract.dynamics_col,
        warmup_period     = contract.warmup_period,
        stop_early        = contract.stop_early,
    )
end

"""
    sl_hash(x) -> String

Short deterministic fingerprint of any config object. Goes into every artifact
path, so an artifact can never be silently reused across a config change.
"""
sl_hash(x)::String = bytes2hex(sha256(repr(x)))[1:8]

"""
    sl_artifact_dir(contract, model_name, config_hash) -> String

`data/scottish_lower/<model_name>/<config_hash>/`. Created on demand.
"""
function sl_artifact_dir(contract::SLContract, model_name::AbstractString, config_hash::AbstractString)
    dir = joinpath(contract.artifact_root, String(model_name), String(config_hash))
    mkpath(dir)
    return dir
end

"""
    sl_assert_not_sealed(contract, seasons)

Hard stop if any sealed season appears where development data is expected.
`25/26` is opened once, deliberately, after model selection is frozen — not by
a runner that happened to widen its season list.
"""
function sl_assert_not_sealed(contract::SLContract, seasons::AbstractVector{<:AbstractString})
    leaked = intersect(Set(seasons), Set(contract.sealed_seasons))
    if !isempty(leaked)
        error("SEALED SEASON TOUCHED: $(collect(leaked)). See docs/PLAN.md — 25/26 is held out.")
    end
    return true
end


# ==============================================================================
# 3. Reporting
# ==============================================================================

"""
    sl_describe(contract)

Print the contract. Run this at the top of every walkthrough so the terms of the
experiment are visible in the transcript rather than buried in a struct.
"""
function sl_describe(contract::SLContract = sl_contract())
    println("=" ^ 74)
    println("SCOTTISH LOWER — EXPERIMENT CONTRACT   [$(sl_hash(contract))]")
    println("=" ^ 74)
    println("  tournaments        : $(contract.tournaments)  (pooled)")
    println("  development        : $(contract.dev_seasons)")
    println("  SEALED             : $(contract.sealed_seasons)")
    println("  history seasons    : $(contract.history_seasons)")
    println("  fold column        : $(contract.dynamics_col)   stop_early=$(contract.stop_early)")
    println("-" ^ 74)
    println("  smoke  (gate 3)    : $(contract.smoke_chains) chains x $(contract.smoke_warmup)/$(contract.smoke_samples)")
    println("  grid   (gate 6-7)  : $(contract.grid_chains) chains x $(contract.grid_warmup)/$(contract.grid_samples), $(contract.queue_tasks) tasks")
    println("  accept/depth/seed  : $(contract.accept_rate) / $(contract.max_depth) / $(contract.seed)")
    println("-" ^ 74)
    println("  book               : 1X2, O/U $(contract.totals_lines), BTTS")
    println("  score matrix       : 0..$(contract.max_goals) goals")
    println("  commission / cap   : $(contract.commission) / $(contract.portfolio_kelly_cap)")
    println("  artifacts          : $(contract.artifact_root)/<model>/<config_hash>/")
    println("=" ^ 74)
    return nothing
end


"""
    sl_gate_table(title, results) -> Bool

Print a PASS/FAIL table and return whether everything passed.

`results` is a vector of NamedTuples with fields `name`, `pass`, and `detail`.
Every gate block ends by calling this and then asserting the return value: you
read the table, the assert is only the tripwire.
"""
function sl_gate_table(title::AbstractString, results::AbstractVector)
    width = maximum(length(String(r.name)) for r in results; init = 20)
    println()
    println("-" ^ 74)
    println("GATE  $title")
    println("-" ^ 74)
    for r in results
        mark = r.pass ? "PASS" : "FAIL"
        println("  [$mark]  ", rpad(String(r.name), width), "  ", r.detail)
    end
    n_pass = count(r -> r.pass, results)
    println("-" ^ 74)
    println("  $n_pass / $(length(results)) passed")
    println()
    return n_pass == length(results)
end
