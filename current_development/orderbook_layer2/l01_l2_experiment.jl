# current_development/orderbook_layer2/l01_l2_experiment.jl
#
# The Layer-2 experiment: a config, a task, a result, and the one axis Layer 1 does not have.
#
# ---------------------------------------------------------------------------------------------
# WHY THIS FILE EXISTS
# ---------------------------------------------------------------------------------------------
#
# Layer 1 is not a pile of runners. It is `ExperimentConfig` -> `ExperimentTask` ->
# `ExperimentResults`, with `run_experiment` / `save_experiment` / `load_experiment` /
# `list_experiments` around it, and `BacktestLedger` + pluggable metrics to judge the output.
# That structure is why an L1 question is a config rather than a script, and why two L1 results
# from different months are comparable at all.
#
# Layer 2 had neither half. This file is the first half; `l02_l2_ledger.jl` is the second.
#
# ---------------------------------------------------------------------------------------------
# THE THREE-TIER COST MODEL (this is the whole design)
# ---------------------------------------------------------------------------------------------
#
# The runbook's central insight is that `BookSpec` is an expensive cache key and `PolicySpec` is
# a free multiplier, which is what lets `r02_policy_sweep` sweep 24 policies in 0.9s. Layer 2 has
# the same structure with one more tier, because it added a time axis:
#
#   TIER 1  SNAPSHOTS   expensive   DB reads + latent extraction, once per (slate, as_of)
#                                   -> `L2Snapshots`, built by `l04_corpus_replay.jl`
#   TIER 2  STAKING     cheap       build_books + stake_slate for a given `PortfolioSystem`
#                                   -> re-run per trust model / filter  (WP5, WP6)
#   TIER 3  ENTRY       free        select rows from the resulting ledger
#                                   -> re-run per entry rule            (WP4)
#
# Getting this wrong is expensive in the literal sense: replaying ~20 slates x ~40 snapshots is
# ~800 `match_day` calls. Doing that once per trust setting would be a day of compute for an
# answer that is a `groupby` away.
#
# ---------------------------------------------------------------------------------------------
# THE ONE CORRECTNESS TRAP IN `AbstractEntryRule`
# ---------------------------------------------------------------------------------------------
#
# `FixedLead` and `AtClose` fire a whole slate at ONE instant, so the Kelly solve, the drawdown
# factor and the exposure cap all still hold: the book that was sized together is the book that
# is taken.
#
# `FirstQualifying` and `BestPrice` do not. They assemble legs from DIFFERENT instants, and the
# portfolio constraints were solved per-snapshot -- so the assembled book can breach the very cap
# that made it legal. Summing those stakes and reporting the total is a real bug, and it is
# invisible unless you look for it: the numbers are all individually correct.
#
# `apply_entry` therefore re-applies the slate cap after assembly, and every rule reports
# `recapped` so the reader can see when it bound. The Kelly weights remain only LOCALLY optimal
# in those two rules -- that is inherent to picking legs across time, not a defect of the fix,
# and it is the reason `BestPrice` is labelled an oracle rather than a strategy.

using DataFrames, Dates, Statistics, Printf, JLD2, JSON3, Random

# ===================================================================
# 1. The entry-time seam
# ===================================================================

"""
    AbstractEntryRule

When to fire. This is the axis Layer 1 has no analogue for: `src/Portfolio` carries exactly one
price per bet (`:odds_close`) and no `DateTime` anywhere, so "when" was never a parameter until
`MatchDay` supplied `as_of`.

A rule is a pure selection over an already-replayed ledger — see the cost model above. Implement
`apply_entry(rule, ledger)` returning at most one row per `(match_id, group, line, selection)`.
"""
abstract type AbstractEntryRule end

"""
    apply_entry(rule, ledger) -> DataFrame

Pick the rows this rule actually fires, from a ledger holding every leg at every snapshot.
Must return at most one row per `(match_id, group, line, selection)`.
"""
apply_entry(r::AbstractEntryRule, ::AbstractDataFrame) =
    error("apply_entry not implemented for $(typeof(r))")

entry_name(r::AbstractEntryRule) = string(nameof(typeof(r)))

"""
    FixedLead(lead)

Fire the whole slate `lead` before the earliest kick-off, e.g. `FixedLead(Minute(90))`.

Snaps to the nearest available snapshot rather than requiring an exact match, because the
collector's cadence is its own business and a grid point can miss a tick. The realised lead is
reported in `:mins_to_ko`, so a rule that could not be honoured is visible rather than silent.
"""
struct FixedLead <: AbstractEntryRule
    lead::Period
end
entry_name(r::FixedLead) = "FixedLead($(Dates.value(Minute(r.lead)))m)"

function apply_entry(r::FixedLead, led::AbstractDataFrame)
    isempty(led) && return copy(led)
    target = Dates.value(Minute(r.lead))
    return _pick_per_leg(led, sub -> argmin(abs.(sub.mins_to_ko .- target)))
end

"""
    AtClose()

The last snapshot before kick-off — the tightest spread and the deepest book, and the de facto
behaviour of every existing runner in the repo (`close_window = (-20, 0)`). The baseline every
other entry rule is measured against.
"""
struct AtClose <: AbstractEntryRule end

function apply_entry(::AtClose, led::AbstractDataFrame)
    isempty(led) && return copy(led)
    return _pick_per_leg(led, sub -> argmin(sub.mins_to_ko))
end

"""
    FirstQualifying(edge; max_lead = Minute(360))

Fire each leg the moment its edge first clears `edge`, scanning forward from `max_lead`.

Unlike the others this is a genuine STRATEGY, not a clock setting: it says "take the price when
the model likes it", which is what a human actually does. It is also the rule most exposed to
the assembly trap in the header — its legs come from many instants — so it is the one to read
`recapped` on.
"""
struct FirstQualifying <: AbstractEntryRule
    edge::Float64
    max_lead::Period
end
FirstQualifying(edge::Real; max_lead::Period = Minute(360)) =
    FirstQualifying(Float64(edge), max_lead)
entry_name(r::FirstQualifying) = @sprintf("FirstQualifying(%.3f)", r.edge)

function apply_entry(r::FirstQualifying, led::AbstractDataFrame)
    isempty(led) && return copy(led)
    lim = Dates.value(Minute(r.max_lead))
    return _pick_per_leg(led, function (sub)
        ok = findall(i -> sub.mins_to_ko[i] <= lim && sub.edge[i] >= r.edge, 1:nrow(sub))
        isempty(ok) && return nothing
        return ok[argmax(sub.mins_to_ko[ok])]      # earliest qualifying instant
    end)
end

"""
    BestPrice(; max_lead = Minute(360))

**Oracle, not a strategy.** For each leg, the best price that was actually available anywhere in
the window — knowable only after the fact.

Run this FIRST. It is an upper bound on what any timing rule could ever be worth, so if the gap
between `BestPrice` and `AtClose` is small, the entire entry-time question is closed for the cost
of one `groupby` and the corpus can be spent on something that moves.
"""
struct BestPrice <: AbstractEntryRule
    max_lead::Period
end
BestPrice(; max_lead::Period = Minute(360)) = BestPrice(max_lead)
entry_name(::BestPrice) = "BestPrice(oracle)"

function apply_entry(r::BestPrice, led::AbstractDataFrame)
    isempty(led) && return copy(led)
    lim = Dates.value(Minute(r.max_lead))
    return _pick_per_leg(led, function (sub)
        ok = findall(<=(lim), sub.mins_to_ko)
        isempty(ok) && return nothing
        return ok[argmax(sub.odds[ok])]
    end)
end

"""
    RandomEntry(seed; max_lead = Minute(360))

**Null control for the oracle, not a strategy.** Fire each leg at a uniformly random instant in
the window.

This exists because `BestPrice` is guaranteed to beat `AtClose` even when entry time carries no
information at all. If prices are a driftless random walk, the maximum over N snapshots exceeds
the last one purely by sampling — and that gap GROWS with N, i.e. with how densely we happened to
sample the book. Read on its own, a large oracle gap would look like a large timing opportunity
when it is an artefact of the grid.

`RandomEntry` separates the two. Under a pure random walk it lands on the close's price in
expectation, so:

    BestPrice >> AtClose  and  RandomEntry ~ AtClose   =>  no trend; the gap is hindsight noise
    BestPrice >> AtClose  and  RandomEntry >  AtClose   =>  a real drift toward kickoff

Average several seeds — one draw is one sample of a rule that is deliberately noisy.
"""
struct RandomEntry <: AbstractEntryRule
    seed::Int
    max_lead::Period
end
RandomEntry(seed::Integer = 1; max_lead::Period = Minute(360)) = RandomEntry(Int(seed), max_lead)
entry_name(r::RandomEntry) = "RandomEntry(seed=$(r.seed))"

function apply_entry(r::RandomEntry, led::AbstractDataFrame)
    isempty(led) && return copy(led)
    lim = Dates.value(Minute(r.max_lead))
    rng = Random.MersenneTwister(r.seed)
    return _pick_per_leg(led, function (sub)
        ok = findall(<=(lim), sub.mins_to_ko)
        isempty(ok) && return nothing
        return ok[rand(rng, 1:length(ok))]
    end)
end

"""
    _pick_per_leg(ledger, chooser) -> DataFrame

Group by leg identity and keep the one row `chooser` selects. `chooser` returns an index into
the group, or `nothing` to drop the leg entirely (a leg that never qualified was never bet — it
must not silently fall through to some default row).
"""
function _pick_per_leg(led::AbstractDataFrame, chooser)
    out = DataFrame[]
    for sub in groupby(led, [:match_id, :group, :line, :selection])
        i = chooser(sub)
        i === nothing && continue
        push!(out, DataFrame(sub[i:i, :]))
    end
    return isempty(out) ? similar(led, 0) : reduce(vcat, out)
end

# ===================================================================
# 2. Re-capping an assembled book
# ===================================================================

"""
    cap_fraction(cap) -> Float64

The largest slate exposure `cap` permits, as a bankroll fraction — the number `recap_slates!`
needs.

Exists because the cap types do not share a field name: `FixedCap` stores `cap`, `VolTargetCap`
stores `ceiling` (`src/Portfolio/implementations/caps.jl:22,48`). An earlier `run_l2_experiment`
guarded on `hasproperty(cap, :c)`, which is true of NEITHER, so the guard was silently false,
`recap_slates!` never ran, and multi-instant entry rules kept books breaching the very constraint
that made each of their parts legal — with no error and no column to notice it by.

The fallback throws rather than guessing. A cap read wrongly here is invisible in the output,
which is exactly the class of bug that should be loud.
"""
cap_fraction(c) = error("cap_fraction: no method for $(typeof(c)) — add one rather than " *
                        "letting recap_slates! silently skip")
cap_fraction(c::BayesianFootball.Portfolio.FixedCap)     = c.cap
cap_fraction(c::BayesianFootball.Portfolio.VolTargetCap) = c.ceiling

"""
    recap_slates!(picked, cap_frac) -> DataFrame

Re-apply the slate exposure cap after an entry rule has assembled legs from different instants.
Take `cap_frac` from `cap_fraction(policy.cap)` rather than reaching for a field by name.

See the header: the cap and the drawdown factor were solved per-snapshot, so a book assembled
across time can breach the constraint that made each of its parts legal. Scaling the whole slate
back is the conservative repair — it preserves the relative Kelly weights, which is the part
that carries the information, and gives up only the scale, which the cap owns anyway.

Adds `:recapped` (did it bind) and `:recap_factor`. A rule that fires at one instant will always
show `recapped = false`, which is the check that this function is not silently reshaping the
baselines.
"""
function recap_slates!(picked::DataFrame, cap_frac::Float64)
    if isempty(picked)
        picked.recapped = Bool[]; picked.recap_factor = Float64[]
        return picked
    end
    picked.recapped     = falses(nrow(picked))
    picked.recap_factor = ones(nrow(picked))

    for sub in groupby(picked, :slate)
        total = sum(sub.stake)
        total <= cap_frac && continue
        f = cap_frac / total
        sub.stake        .*= f
        sub.recap_factor .= f
        sub.recapped     .= true
        if hasproperty(sub, :risk)
            sub.risk .*= f
        end
    end
    return picked
end

# ===================================================================
# 3. Config / Task / Results  (mirrors src/experiments/types.jl)
# ===================================================================

"""
    L2Config

The immutable recipe for a Layer-2 experiment — the analogue of `Experiments.ExperimentConfig`.

`sys` is the Tier-2 cache key (changing it forces a re-stake) and `entry` is Tier 3 (free).
`arm` selects which latent series the ledger was built from:

  * `:frozen` — latents computed once per slate and reused at every snapshot. All movement in
    the trace is then the BOOK moving, which is the only way to attribute a timing effect.
  * `:live`   — latents recomputed per snapshot, so the announced XI moves them. This is the
    operationally realistic arm, and `live - frozen` is the measured value of team news.

The distinction is not cosmetic: `src_sup40_sw40` is a PLAYER-level engine, so unlike the funnel
engine the existing single-slate harness was built on, its latents genuinely do move with the
clock. Reporting one arm alone confounds model drift with market drift.
"""
Base.@kwdef struct L2Config
    name::String
    sys::Any                                  # Portfolio.PortfolioSystem
    entry::AbstractEntryRule = AtClose()
    arm::Symbol              = :frozen
    tags::Vector{String}     = String[]
    description::String      = ""
    save_dir::String         = "./data/l2_experiments"
end

"The full definition of a Layer-2 run: the replayed snapshots plus the recipe."
struct L2Task
    corpus::Any                               # L2Corpus
    snapshots::Any                            # L2Snapshots  (see l04_corpus_replay.jl)
    config::L2Config
end

"""
    L2Results

Artifacts of one Layer-2 run — the analogue of `Experiments.ExperimentResults`.

`ledger` is the long frame every metric reads (see `l02_l2_ledger.jl`); `trajectory` is the
compounded `Portfolio.Trajectory` the wealth metrics need, and is `nothing` when the config's
entry rule assembled legs across instants without a settled slate sequence to compound over.
"""
struct L2Results
    config::L2Config
    ledger::DataFrame
    trajectory::Any                           # Union{Nothing, Portfolio.Trajectory}
    diagnostics::Dict{Symbol,Any}
    save_path::String
end

# ===================================================================
# 4. Persistence  (mirrors src/experiments/runner.jl)
# ===================================================================

"""
    save_l2_experiment(r::L2Results; path = nothing) -> String

Writes `results.jld2` plus a human-readable `config.json` and `meta.json` into
`<save_dir>/<name>_<yyyymmdd_HHMMSS>/`, matching the Layer-1 layout so the same habits
(and the same `list_*` browsing) work across both layers.

The config is stringified rather than serialised structurally, exactly as Layer 1 does: a
`PortfolioSystem` contains parametric structs whose fields are functions of a research runner,
and a JSON sidecar that cannot be reloaded is still worth having as the thing you read six weeks
later to remember what a directory was.
"""
function save_l2_experiment(r::L2Results; path = nothing, quiet::Bool = false)
    target = path === nothing ? r.save_path : path
    mkpath(target)

    jldsave(joinpath(target, "results.jld2"); results = r)

    open(joinpath(target, "config.json"), "w") do io
        JSON3.pretty(io, Dict(
            "name"        => r.config.name,
            "entry"       => entry_name(r.config.entry),
            "arm"         => String(r.config.arm),
            "system"      => sprint(show, r.config.sys),
            "tags"        => r.config.tags,
            "description" => r.config.description))
    end

    open(joinpath(target, "meta.json"), "w") do io
        JSON3.pretty(io, Dict(
            "name"      => r.config.name,
            "n_legs"    => nrow(r.ledger),
            "n_matches" => isempty(r.ledger) ? 0 : length(unique(r.ledger.match_id)),
            "timestamp" => string(now()),
            "diagnostics" => Dict(string(k) => string(v) for (k, v) in r.diagnostics)))
    end

    quiet || println("saved L2 experiment -> $target")
    return target
end

"Load an `L2Results` from a directory or a `.jld2` path. Mirrors `Experiments.load_experiment`."
function load_l2_experiment(path::String)
    f = endswith(path, ".jld2") ? path : joinpath(path, "results.jld2")
    isfile(f) || error("load_l2_experiment: no results.jld2 at $path")
    return load(f)["results"]
end

"List saved Layer-2 experiments, newest first. Mirrors `Experiments.list_experiments`."
function list_l2_experiments(dir::String = "./data/l2_experiments")
    isdir(dir) || return String[]
    paths = [joinpath(dir, d) for d in readdir(dir) if isdir(joinpath(dir, d))]
    sort!(paths, by = mtime, rev = true)
    for p in paths
        @printf("  %-55s  %s\n", basename(p), Dates.format(unix2datetime(mtime(p)), "yyyy-mm-dd HH:MM"))
    end
    return paths
end

# ===================================================================
# 5. Display
# ===================================================================

function Base.show(io::IO, ::MIME"text/plain", c::L2Config)
    println(io, "L2Config \"$(c.name)\"")
    println(io, "├─ entry   $(entry_name(c.entry))")
    println(io, "├─ arm     $(c.arm)")
    isempty(c.tags) || println(io, "├─ tags    $(join(c.tags, ", "))")
    print(io,   "└─ system  $(sprint(show, c.sys))")
end

function Base.show(io::IO, ::MIME"text/plain", r::L2Results)
    println(io, "L2Results \"$(r.config.name)\"  [$(entry_name(r.config.entry)), $(r.config.arm)]")
    println(io, "├─ legs      $(nrow(r.ledger))")
    if !isempty(r.ledger)
        println(io, "├─ matches   $(length(unique(r.ledger.match_id)))")
        if hasproperty(r.ledger, :recapped)
            println(io, "├─ recapped  $(count(r.ledger.recapped)) legs")
        end
    end
    print(io, "└─ saved     $(r.save_path)")
end

Base.show(io::IO, c::L2Config)  = print(io, "L2Config(\"$(c.name)\", $(entry_name(c.entry)), :$(c.arm))")
Base.show(io::IO, r::L2Results) = print(io, "L2Results(\"$(r.config.name)\", $(nrow(r.ledger)) legs)")
