# ==============================================================================
# Experiment 06 · l66 — shared loader for the hierarchical-kappa evaluation
# ==============================================================================
#
# Loader. Definitions and shared state only; r66 (scores, GLM edge, finishing
# factor) and r67 (portfolio backtest) execute.
#
# It carries three things both runners need and neither should own:
#
#   §0  an artefact compatibility shim, described at length below;
#   §1  the run manifest — the four run UUIDs the study is defined over;
#   §2  the Betfair TWA closing-line frame and the fit loader, so the two runners
#       cannot drift into scoring different prices or different artefacts.
#
# It includes `l64_hierarchical_kappa_loader.jl`, which in turn includes
# `l60_loader.jl`, so `ds`, `db`, both candidate sets, the splitter, the book and
# the policy all arrive from the files that already define them.
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using Serialization
using UUIDs

include(joinpath(@__DIR__, "l64_hierarchical_kappa_loader.jl"))

const L66_PG = BayesianFootball.Models.PreGame
const L66_INF = BayesianFootball.Training.Inference

# %%
# ==============================================================================
# 0. ARTEFACT COMPATIBILITY — reading the shared-κ controls back
# ==============================================================================
#
# THE PROBLEM. `HierarchicalKappa` was added by giving `JointGammaPoissonObservation`
# a fourth type parameter for the finishing mode:
#
#     before   JointGammaPoissonObservation{F,S,K}
#     after    JointGammaPoissonObservation{F,S,K,M}
#
# The two shared-κ controls were fitted on 2026-09-02, BEFORE that change, and
# `fit_artifacts.fit_blob` is a `Serialization` stream that stores the concrete type
# it was written with. Reading one now reconstructs `JointGammaPoissonObservation{F,S,K}`,
# which is no longer a concrete type but a `UnionAll` with `M` still free, and
# `deserialize(::AbstractSerializer, ::DataType)` does not dispatch on a `UnionAll`.
# The control artefacts therefore raise `MethodError` on load, and the hierarchical
# ones — written after the change — load normally.
#
# WHY THIS IS NOT A DATA PROBLEM. `match_latents` is relational: `load_fit`
# reconstructs the held-out posterior panel from that table, not from the blob, so
# every score and every portfolio number in r66/r67 would survive without the blob at
# all. What the blob carries that the tables do not is the CHAINS, and the chains are
# where `obs.log_κ` lives. Without them the controls have no league finishing factor to
# compare the hierarchical fits against, which is the one comparison §12 of r66 exists
# to make. So the shim is not a convenience; it is what makes the control a control.
#
# WHY THIS FORM IS SAFE.
#   * It dispatches on `Type{<:JointGammaPoissonObservation}` and IMMEDIATELY delegates
#     to the stock `DataType` method whenever the type is concrete. Every artefact
#     written by the current code — including both hierarchical runs — takes the
#     untouched path, byte for byte.
#   * Only the `UnionAll` case, which is unreachable for anything the current code
#     writes, reads the three-field layout and reinstates `SharedKappa()`. That is not
#     a guess about the old object: a three-parameter observation had exactly one
#     finishing mode, and `SharedKappa` is it.
#   * The field read mirrors the immutable branch of `Serialization.deserialize`
#     (`handle_deserialize` per field, no cycle registration, `new` at the end),
#     because the struct is immutable and its fields are read in declaration order.
#
# THIS IS A READ-SIDE SHIM AND IT STAYS HERE. It must not move into `src/`: `src` should
# describe the model as it is, and the day these two controls are refitted this file
# becomes dead code. Compare `current_development/06_typed_posterior_latents/l02_extract.jl` §0,
# which carries a comparable deliberate shim for the same reason and the same scope.
function Serialization.deserialize(s::Serialization.AbstractSerializer,
                                   T::Type{<:L66_PG.JointGammaPoissonObservation})
    T isa DataType && return invoke(
        Serialization.deserialize,
        Tuple{Serialization.AbstractSerializer, DataType}, s, T)

    fields = Any[]
    for _ in 1:3
        tag = Int32(read(s.io, UInt8)::UInt8)
        push!(fields, Serialization.handle_deserialize(s, tag))
    end
    return L66_PG.JointGammaPoissonObservation(
        fields[1], fields[2], fields[3], L66_PG.SharedKappa())
end

# %%
# ==============================================================================
# 1. The run manifest
# ==============================================================================
#
# Runs are addressed by UUID, not by name. `load_fit(db, name)` takes the newest row
# with that name, which is right in a workflow and wrong in a report: a report has to be
# reproducible against the exact artefacts the work package names.
const L66_RUNS = [
    (name = "m05_joint_production_wealth",
     run_id = "ed541a7c-01e2-447e-a771-783517728d47", kappa = :shared,       arm = "m05"),
    (name = "m05_hierarchical_kappa",
     run_id = "b3e19ad4-f755-4b89-addd-ff7592787deb", kappa = :hierarchical, arm = "m05"),
    (name = "m12_joint_hybrid_synergy",
     run_id = "132df5c2-c742-4e95-8693-3aeb2b2cbaef", kappa = :shared,       arm = "m12"),
    (name = "m12_hierarchical_kappa",
     run_id = "a0847873-de69-4e25-824f-c03e4a4fd8c4", kappa = :hierarchical, arm = "m12"),
]

const L66_NAMES = [r.name for r in L66_RUNS]
const L66_ARMS = ["m05", "m12"]
const L66_CONTROL_NAME = Dict("m05" => "m05_joint_production_wealth",
                              "m12" => "m12_joint_hybrid_synergy")
const L66_CANDIDATE_NAME = Dict("m05" => "m05_hierarchical_kappa",
                                "m12" => "m12_hierarchical_kappa")

const L66_EXPECTED_FOLDS = 40
const L66_EXPECTED_OOS = 710

const L66_OUTPUT_DIR = joinpath(@__DIR__, "results", "hierarchical_kappa")

"Every candidate model object, hierarchical and shared, addressable by run name."
l66_models_by_name() = Dict{String,Any}(
    name => model for (name, model) in vcat(l60_candidate_models, l64_candidate_models))

# %%
# ==============================================================================
# 2. Shared data contracts
# ==============================================================================
"""
    l66_load_fit(db, name) -> (Fit, UUID)

The named run, checked against the manifest UUID before it is returned.

Resolving by name and then ASSERTING the UUID is deliberate: it keeps the runner
readable (`m12_hierarchical_kappa`, not a hex string) while making a silent swap — a
re-run, a smoke fit that took the name — impossible rather than merely unlikely.
"""
function l66_load_fit(db, name::AbstractString)
    entry = only(filter(r -> r.name == name, L66_RUNS))
    resolved = L66_INF._run_uuid(db, name)
    string(resolved) == entry.run_id || error(
        "$name resolves to run $resolved but the manifest names $(entry.run_id); " *
        "refusing to report numbers from an artefact the report does not identify")
    fit = load_fit(resolved, db)
    length(fit.folds) == L66_EXPECTED_FOLDS || error(
        "$name has $(length(fit.folds)) folds; expected $L66_EXPECTED_FOLDS")
    n_matches(fit.latents) == L66_EXPECTED_OOS || error(
        "$name has $(n_matches(fit.latents)) OOS latents; expected $L66_EXPECTED_OOS")
    return fit, resolved
end

"""
    l66_betfair_closing_odds(ds) -> DataFrame

Betfair time-weighted-average closes over (-20m, 0m], de-vigged within each market.

`prob_fair_close` is the proportional de-vig — implied probabilities divided by the
market's own overround. It is the benchmark every score in r66 is read against and the
price every stake in r67 is struck at, which is why both runners take it from here
rather than each building their own.
"""
function l66_betfair_closing_odds(ds)
    raw = Data.summarize_odds(ds.betfair_odds, Data.TWAEstimator(); window = (-20.0, 0.0))
    odds = DataFrame(
        match_id = Int.(raw.match_id),
        market_name = String.(raw.market_name),
        market_line = Float64.(raw.market_line),
        selection = Symbol.(raw.selection),
        odds_close = Float64.(raw.odds),
    )
    filter!(row -> isfinite(row.odds_close) && row.odds_close > 1.0, odds)
    odds.prob_implied_close = 1.0 ./ odds.odds_close
    transform!(
        groupby(odds, [:match_id, :market_name, :market_line]),
        :prob_implied_close => (p -> p ./ sum(p)) => :prob_fair_close,
    )
    outcome_cols = [:match_id, :market_name, :market_line, :selection, :is_winner]
    winners = unique(select(ds.odds, outcome_cols))
    odds = leftjoin(odds, winners; on = [:match_id, :market_name, :market_line, :selection])
    sort!(odds, [:match_id, :market_name, :market_line, :selection])
    return odds
end

"The market family a scored selection belongs to, for per-market cuts."
function l66_family(market_name::AbstractString, line::Real)
    m = lowercase(market_name)
    m == "1x2" && return "1X2"
    occursin("btts", m) && return "BTTS"
    (occursin("over", m) || occursin("under", m) || occursin("total", m)) &&
        return "OU" * string(line)
    return market_name
end

"`match_id => season` and `match_id => kickoff date`, for the temporal splits."
l66_season_of(ds) = Dict(Int(r.match_id) => String(r.season) for r in eachrow(ds.matches))
l66_date_of(ds) = Dict(Int(r.match_id) => r.match_date for r in eachrow(ds.matches))

mkpath(L66_OUTPUT_DIR)
