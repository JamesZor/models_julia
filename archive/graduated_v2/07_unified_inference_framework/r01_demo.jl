# ==============================================================================
# 07 — UNIFIED INFERENCE FRAMEWORK : THE PROOF
# ==============================================================================
#
# WHAT THIS IS
#   A deterministic verification that the unified `Fit` / `FitConfig` / `InGameFit`
#   lifecycle does everything `src/training/` and `src/experiments/` do, plus the
#   convergence audit and the typed-latent extraction they leave to the user, and that
#   every legacy call site keeps working verbatim.
#
#   Seven things are established, in this order:
#
#     §5   the convergence telemetry PASSES a healthy run and FAILS each of four
#          specific pathologies, one per gate, on chains built to have them
#     §6   FitConfig → fit_model → Fit, and `fit[1].chain` reaches what
#          `exp.training_results.items[1][1]` used to
#     §7   the typed latents (06) survive the fold merge with 0 ULP, and the pricing
#          kernels still allocate 0 bytes through the framework
#     §8   the in-game bridge computes Λ(t → 90) and prices a live state at 0 bytes,
#          and reduces EXACTLY to the pre-game price under an identity kernel
#     §9   a legacy call site — its body copied verbatim — runs unmodified
#     §10  save → load → save round-trips, the JSON sidecar is complete, and a
#          genuine legacy `ExperimentResults` upgrades to a `Fit` on load
#     §11  the cost, measured rather than asserted
#
# WHAT THIS IS NOT
#   Not a model-fitting run, not an evaluation, not a betting study. No MCMC is run and
#   no database is touched. Chains are drawn from priors with a fixed seed, exactly as
#   in `06_typed_posterior_latents/r01_demo.jl` — see `l04_parity.jl` §9 there for what
#   that does and does not test. Nothing here claims any model fits anything.
#
#   The convergence chains in §5 are built to be pathological ON PURPOSE. A NUTS run
#   that produced them would be broken; that is the point of the section.
#
# THE COMPARABILITY CONTRACT
#   Every parity row compares TWO IMPLEMENTATIONS OVER ONE SET OF NUMBERS. The
#   framework's path and the reference path are fed the same chains and the same
#   fixtures, so a difference is the implementation's and nothing else's.
#
#   The pre-game reference side always goes through the live `06` kernels
#   (`compute_score_grid!` → `price_market!`), which in turn are held at 0 ULP against
#   the live `src` `Predictions` path by that prototype's own runner. This file does
#   not re-derive that chain; it extends it by one link.
#
# WHAT WOULD FALSIFY THE CLAIMS
#   Any non-zero ULP in §7 or §8. Any non-zero byte count in §7's or §8's steady-state
#   rows. Any legacy expression in §9 raising. Any field missing from §10's sidecar.
#
# PERSISTENCE
#   This runner DOES write files — §10 is about persistence — but only into a
#   `mktempdir()` that it removes on the way out. Nothing under `data/` is touched.
#
# USAGE
#   julia --project current_development/07_unified_inference_framework/r01_demo.jl
#
#   or, in a REPL:
#     include("current_development/07_unified_inference_framework/r01_demo.jl")
#
#   Runs in seconds. Exits non-zero if any gate fails, so it is usable in CI.
#
# ==============================================================================


# %% ===========================================================================
# 1. Packages and implementation
# ==============================================================================

import BayesianFootball
using DataFrames
using Dates
using Distributions
using JSON3
using JLD2
using MCMCChains
using Printf
using Random
using Serialization
using Statistics

# The thing under test. `l06_compat_bridge.jl` opens `module UnifiedInference`, whose
# include chain pulls in l05 → l04 → l03 → l02 → l01 and, through l01, the whole of
# `06_typed_posterior_latents` and `05_composable_count_builder`.
include(joinpath(@__DIR__, "l06_compat_bridge.jl"))

const UIF = UnifiedInference
const BF  = BayesianFootball
const D   = BayesianFootball.Data
const PG  = BayesianFootball.Models.PreGame

using .UnifiedInference


# %% ===========================================================================
# 2. Configuration
# ==============================================================================

const DEMO_N_TEAMS   = 8
const DEMO_N_FOLDS   = 3
const DEMO_PER_FOLD  = 8
const DEMO_N_MATCHES = DEMO_N_FOLDS * DEMO_PER_FOLD    # 24 held-out fixtures
const DEMO_N_DRAWS   = 300                              # posterior draws PER CHAIN
const DEMO_N_CHAINS  = 4                                # exercises the flattening
const DEMO_TOTAL     = DEMO_N_DRAWS * DEMO_N_CHAINS     # 1200
const DEMO_N_SEASONS = 2
const DEMO_MAX_GOALS = UIF.TPL_MAX_GOALS
const DEMO_SEED      = 20240601

# The three markets a live book actually prices. `MarketOverUnder(2.5)` is the one whose
# live form differs most from its pre-game form, because the goals already scored move
# the whole distribution across the line rather than scaling it.
const DEMO_MARKETS = (D.Market1X2(), D.MarketBTTS(), D.MarketOverUnder(2.5))

# The in-game NHPP: 5-minute bins over [0, 95], matching `inplay_scottish`'s NHPPXConfig.
const DEMO_INGAME_MODEL = UIF.NHPPIntensityModel(name = :demo_nhpp, Δt = 5.0, Tend = 95.0)

# Every gate's verdict lands here; §12 is the only place that decides pass/fail.
const DEMO_GATES = Pair{String, Bool}[]
demo_gate!(name, ok) = (push!(DEMO_GATES, name => Bool(ok)); Bool(ok))

function demo_banner(n, title)
    println()
    println("=" ^ 78)
    @printf("%d. %s\n", n, uppercase(title))
    println("=" ^ 78)
end

# The scratch directory §10 writes into. Removed in §12, whatever happens above it.
const DEMO_TMP = mktempdir(; prefix = "uif_demo_")


# %% ===========================================================================
# 3. Deterministic fixtures, folds and feature sets
# ==============================================================================
#
# Three walk-forward folds of eight held-out fixtures each. Three rather than one
# because every interesting thing in this framework is a REDUCTION over folds — the
# convergence summary, the latent merge, the legacy `training_results.items` — and a
# one-fold run would let a reduction bug pass by being the identity.

demo_banner(3, "deterministic fixtures and folds")

const DEMO_FIXTURES = UIF.tpl_synthetic_fixtures(DEMO_N_MATCHES;
                                                 n_teams = DEMO_N_TEAMS, seed = DEMO_SEED)
const DEMO_TEAM_MAP = UIF.tpl_team_map(DEMO_N_TEAMS)

"Fold `k`'s slice of the fixture frame."
demo_fold_fixtures(k::Int) =
    DEMO_FIXTURES[((k - 1) * DEMO_PER_FOLD + 1):(k * DEMO_PER_FOLD), :]

const DEMO_OOS = [demo_fold_fixtures(k) for k in 1:DEMO_N_FOLDS]

# The FeatureSet the composable engine's OOS extractor reads: three keys, nothing else.
# An extractor reaching for a key this does not supply raises a `KeyError` here, at the
# top of a five-second demo, rather than three hours into a real fold.
const DEMO_FS = UIF.tpl_feature_set(n_teams = DEMO_N_TEAMS,
                                    n_seasons = DEMO_N_SEASONS,
                                    team_map = DEMO_TEAM_MAP)

# Real `SplitMetaData`, not a stand-in: `FoldFit`'s `M` parameter is bounded by
# `AbstractSplitMetaData`, so a fake would not type-check.
const DEMO_METAS = [D.SplitMetaData(1, "23/24", "24/25", 2, k, 0) for k in 1:DEMO_N_FOLDS]

const DEMO_FEATURE_SETS = [(DEMO_FS, DEMO_METAS[k]) for k in 1:DEMO_N_FOLDS]

# A real splitter instance, for the same reason.
const DEMO_SPLITTER = D.CVConfig(tournament_ids = [1], target_seasons = ["24/25"],
                                 history_seasons = 2, dynamics_col = :match_week)

@printf("  fixtures        : %d over %d teams, %d folds × %d\n",
        DEMO_N_MATCHES, DEMO_N_TEAMS, DEMO_N_FOLDS, DEMO_PER_FOLD)
@printf("  posterior draws : %d samples × %d chains = %d flattened\n",
        DEMO_N_DRAWS, DEMO_N_CHAINS, DEMO_TOTAL)
@printf("  date span       : %s .. %s\n",
        minimum(DEMO_FIXTURES.match_date), maximum(DEMO_FIXTURES.match_date))
@printf("  scratch dir     : %s\n", DEMO_TMP)

# The model: the Scottish Lower baseline from `05_composable_count_builder`, plus a
# squad-wealth supremacy covariate. Its chain schema is DERIVED from the assembled
# components, so this runner never hand-writes a site name.
function demo_count_model()
    b = UIF.CountModelBuilder(:uif_demo_poisson)
    UIF.add!(b, PG.GlobalInterception(),
                PG.TimeDecayDynamics(days_half_life = 365.0),
                PG.GlobalHomeAdvantage())
    UIF.add!(b, UIF.WealthCovariate())
    return UIF.build(b)
end

const DEMO_MODEL = demo_count_model()
const DEMO_COLS  = UIF.cb_chain_columns(DEMO_MODEL, DEMO_N_TEAMS)

println("\n  model           : ", DEMO_MODEL)
@printf("  chain sites     : %d  (%s …)\n",
        length(DEMO_COLS), join(DEMO_COLS[1:min(4, end)], ", "))

demo_gate!("3a model is a pre-game model", UIF.is_pregame(DEMO_MODEL))
demo_gate!("3b model is not an in-game model", !UIF.is_ingame(DEMO_MODEL))
demo_gate!("3c existing src engines satisfy AbstractPreGameModel",
           PG.DynamicGoalsModel <: UIF.AbstractPreGameModel)


# %% ===========================================================================
# 4. Synthetic NUTS chains, with the internals a real one carries
# ==============================================================================
#
# `tpl_synthetic_chain` (06) produces the PARAMETER sites and nothing else, which is all
# a container test needs. The convergence audit reads the INTERNALS — `numerical_error`,
# `tree_depth`, `hamiltonian_energy` — so this runner builds chains that have them,
# under the names AdvancedHMC records them through Turing.
#
# The energy series is an AR(1) with a controllable coefficient, because BFMI has a
# closed form under AR(1) that §5 checks the implementation against:
#
#     E-BFMI = Σ(ΔE)² / Σ(E − Ē)²  →  2(1 − φ)  as n → ∞
#
# So `φ = 0.50` gives BFMI ≈ 1.0 (healthy) and `φ = 0.97` gives ≈ 0.06 (a run that
# needs reparameterising). That is a REAL relationship, not a hand-tuned constant, and
# §5b tests the estimator against it rather than against a number this file chose.

demo_banner(4, "synthetic chains with HMC internals")

const DEMO_INTERNALS = ["lp", "n_steps", "is_accept", "acceptance_rate", "step_size",
                        "tree_depth", "numerical_error", "hamiltonian_energy"]

"""
    demo_nuts_chain(colnames; …) -> Chains

A deterministic `Chains` with the requested parameter sites AND the eight internals a
Turing NUTS run records.

The knobs are the four pathologies §5 needs, each isolated so that a chain can fail
exactly one gate:

  * `chain_offset` — a per-chain shift of every parameter. Breaks R-hat and nothing else.
  * `ar` — AR(1) coefficient on the PARAMETERS. Breaks ESS by autocorrelation.
  * `div_rate` — fraction of transitions flagged `numerical_error`.
  * `energy_ar` — AR(1) coefficient on the energy. Breaks BFMI (see above).
  * `cap_rate` — fraction of transitions saturating `max_depth`.
"""
function demo_nuts_chain(colnames::Vector{String};
                         n_draws::Int = DEMO_N_DRAWS,
                         n_chains::Int = DEMO_N_CHAINS,
                         seed::Int = DEMO_SEED,
                         chain_offset::Float64 = 0.0,
                         ar::Float64 = 0.0,
                         div_rate::Float64 = 0.0,
                         energy_ar::Float64 = 0.50,
                         cap_rate::Float64 = 0.0,
                         max_depth::Int = 10)
    rng = MersenneTwister(seed)
    p = length(colnames)
    all_names = vcat(colnames, DEMO_INTERNALS)
    vals = Array{Float64, 3}(undef, n_draws, length(all_names), n_chains)

    for c in 1:n_chains
        for (j, nm) in enumerate(colnames)
            v = UIF.tpl_synthetic_site(nm, n_draws, rng)
            if ar != 0.0
                # An AR(1) smoothing that preserves the marginal scale, so the site is
                # still in its prior's range and only its AUTOCORRELATION has changed.
                s = std(v)
                for i in 2:n_draws
                    v[i] = ar * v[i - 1] + sqrt(1 - ar^2) * v[i]
                end
                s > 0 && (v .*= s / max(std(v), eps()))
            end
            vals[:, j, c] = v .+ chain_offset * (c - 1)
        end

        E = Vector{Float64}(undef, n_draws)
        E[1] = 40.0 + randn(rng)
        for i in 2:n_draws
            E[i] = 40.0 + energy_ar * (E[i - 1] - 40.0) +
                   sqrt(1 - energy_ar^2) * randn(rng)
        end

        depth = fill(Float64(max_depth - 3), n_draws)
        n_cap = round(Int, cap_rate * n_draws)
        n_cap > 0 && (depth[1:n_cap] .= Float64(max_depth))

        divs = zeros(n_draws)
        n_div = round(Int, div_rate * n_draws)
        n_div > 0 && (divs[1:n_div] .= 1.0)

        base = p
        vals[:, base + 1, c] = -520.0 .+ randn(rng, n_draws)      # lp
        vals[:, base + 2, c] = 2 .^ depth                          # n_steps
        vals[:, base + 3, c] .= 1.0                                # is_accept
        vals[:, base + 4, c] = 0.80 .+ 0.05 .* rand(rng, n_draws)  # acceptance_rate
        vals[:, base + 5, c] .= 0.05                               # step_size
        vals[:, base + 6, c] = depth                               # tree_depth
        vals[:, base + 7, c] = divs                                # numerical_error
        vals[:, base + 8, c] = E                                   # hamiltonian_energy
    end

    return Chains(vals, Symbol.(all_names),
                  Dict(:parameters => Symbol.(colnames),
                       :internals  => Symbol.(DEMO_INTERNALS)))
end

# One healthy chain per fold. Different seeds, so the folds are genuinely different
# posteriors and a merge that dropped one would change the numbers.
const DEMO_CHAINS = [demo_nuts_chain(DEMO_COLS; seed = DEMO_SEED + k)
                     for k in 1:DEMO_N_FOLDS]

@printf("  chain shape     : %d draws × %d sites × %d chains\n", size(DEMO_CHAINS[1])...)
@printf("  parameters      : %d    internals: %d\n",
        length(MCMCChains.names(DEMO_CHAINS[1], :parameters)),
        length(MCMCChains.names(DEMO_CHAINS[1], :internals)))

demo_gate!("4a chains carry the HMC internals the audit reads",
           all(s -> s in MCMCChains.names(DEMO_CHAINS[1], :internals),
               (:numerical_error, :tree_depth, :hamiltonian_energy)))


# %% ===========================================================================
# 5. Convergence telemetry
# ==============================================================================
#
# The section that matters most, because it is the one thing `src/experiments/
# diagnostics/` cannot do at all: turn a set of chains into a verdict, with no
# `DataStore` anywhere in the call.
#
# Four pathologies, each built to break a NAMED gate. A telemetry module that reported
# "FAIL" for everything would pass a test that only checked healthy-versus-broken; these
# rows check that the right gate fires, and print every gate that did.
#
# TWO OF THE FOUR ARE ISOLATED AND TWO ARE NOT, and that is a property of the statistics
# rather than a defect. Divergences and BFMI read the sampler's own internals, so a chain
# built to break one breaks only it. R-hat and ESS both read BETWEEN-CHAIN variance under
# rank normalisation — offsetting the chains raises R-hat AND collapses bulk/tail ESS, and
# an AR(1) that destroys ESS also leaves the chains disagreeing. The table below prints
# every gate each pathology tripped rather than claiming a cleanliness that is not there;
# the gate asserts that the TARGETED one is among them.

demo_banner(5, "convergence telemetry")

const DEMO_HEALTHY_FOLDS =
    [UIF.FoldFit(k, DEMO_CHAINS[k], DEMO_METAS[k]) for k in 1:DEMO_N_FOLDS]

const DEMO_DIAG = UIF.audit_convergence(DEMO_HEALTHY_FOLDS)

show(stdout, MIME"text/plain"(), DEMO_DIAG)
println()
UIF.convergence_table(DEMO_DIAG)

demo_gate!("5a healthy run passes every gate", DEMO_DIAG.passed)
demo_gate!("5b audit needs no DataStore, splitter or model",
           DEMO_DIAG isa UIF.ConvergenceSummary && DEMO_DIAG.n_applicable == DEMO_N_FOLDS)
demo_gate!("5c divergences are counted, not assumed absent",
           DEMO_DIAG.n_transitions == DEMO_N_FOLDS * DEMO_TOTAL)
demo_gate!("5d no gate abstained on a full NUTS chain", isempty(DEMO_DIAG.abstained))

# --- 5.1 one pathology per gate ------------------------------------------------

println("\n  a pathology per gate — the targeted gate must be among those that fire:")
@printf("    %-38s %-8s  %s\n", "pathology", "target", "gates that fired")

const DEMO_PATHOLOGIES = [
    ("R-hat      (chains offset by 0.8)",
     demo_nuts_chain(DEMO_COLS; seed = DEMO_SEED + 90, chain_offset = 0.8),
     "R-hat"),
    ("ESS        (AR(1) φ = 0.985)",
     demo_nuts_chain(DEMO_COLS; seed = DEMO_SEED + 91, ar = 0.985),
     "bulk ESS"),
    ("divergence (2% of transitions)",
     demo_nuts_chain(DEMO_COLS; seed = DEMO_SEED + 92, div_rate = 0.02),
     "divergences"),
    ("BFMI       (energy AR(1) φ = 0.97)",
     demo_nuts_chain(DEMO_COLS; seed = DEMO_SEED + 93, energy_ar = 0.97),
     "BFMI"),
]

for (label, ch, target) in DEMO_PATHOLOGIES
    s = UIF.audit_convergence([UIF.FoldFit(1, ch, DEMO_METAS[1])])
    hit = target in s.failed_gates
    @printf("    %-38s %-8s  %-34s %s\n", label, split(target)[end],
            isempty(s.failed_gates) ? "(none)" : join(s.failed_gates, ", "),
            hit ? "" : "  TARGET GATE DID NOT FIRE")
    demo_gate!("5e $(strip(split(label, "(")[1])) pathology trips its own gate",
               hit && !s.passed)
end

# The two that ARE isolated, stated as a gate rather than left to the table above: a
# divergence-only chain must not also fail R-hat, and a BFMI-only chain must not either.
# If they did, the telemetry would be reporting one underlying defect four times and
# `failed_gates` would be useless for triage.
let div_only = UIF.audit_convergence(
        [UIF.FoldFit(1, DEMO_PATHOLOGIES[3][2], DEMO_METAS[1])]),
    bfmi_only = UIF.audit_convergence(
        [UIF.FoldFit(1, DEMO_PATHOLOGIES[4][2], DEMO_METAS[1])])

    demo_gate!("5e' the divergence pathology fails ONLY the divergence gate",
               div_only.failed_gates == ["divergences"])
    demo_gate!("5e'' the BFMI pathology fails ONLY the BFMI gate",
               bfmi_only.failed_gates == ["BFMI"])
end

# --- 5.2 BFMI against its closed form -----------------------------------------
#
# Not "is the number plausible" but "is it the number".
#
# For an AR(1) energy series with coefficient φ, `E-BFMI → 2(1 − φ)`. So a series built
# with a known φ has a known answer, and the estimator can be checked against it rather
# than against a number this file picked.
#
# TWO THINGS THE CHECK HAS TO GET RIGHT, both of them properties of the STATISTIC and
# not of the implementation:
#
#   1. It uses a LONG series (`DEMO_BFMI_N`), not the 300 draws the rest of the runner
#      uses. E-BFMI is a ratio whose denominator is a sample variance, and at φ = 0.97
#      a 300-draw series holds only `N(1−φ)/(1+φ) ≈ 5` effectively independent points.
#      The expectation of that ratio is not the ratio of the expectations — it comes out
#      near 0.11 against a true 0.06 — and no tolerance that accepts it would be
#      meaningful. At 20,000 draws the effective count is ~300 and the two agree to
#      under a percent. (The 300-draw series is still the right thing to GATE on: §5.1's
#      φ = 0.97 chain fails the BFMI gate on the strength of that same 0.11, which is
#      well under the 0.30 threshold. A biased estimate of a broken run is still a
#      broken run.)
#
#   2. It compares the MEAN over chains. `audit_convergence` reports the MINIMUM, which
#      is correct for a diagnostic — the worst chain is the one that matters — but the
#      minimum of four draws sits about a standard deviation below their mean by
#      construction, and testing an estimator against its limit means testing its centre.

const DEMO_BFMI_N = 20_000

"An AR(1) energy series with a known coefficient: `E_n − μ = φ(E_{n−1} − μ) + √(1−φ²)·ε`."
function demo_ar1_energy(φ::Float64, n::Int, n_chains::Int, seed::Int)
    rng = MersenneTwister(seed)
    E = Matrix{Float64}(undef, n, n_chains)
    for c in 1:n_chains
        E[1, c] = randn(rng)
        for i in 2:n
            E[i, c] = φ * E[i - 1, c] + sqrt(1 - φ^2) * randn(rng)
        end
    end
    return E
end

println("\n  BFMI estimator against its AR(1) limit  2(1 − φ)   ",
        "($(DEMO_BFMI_N) draws × $(DEMO_N_CHAINS) chains):")
@printf("    %-8s %12s %12s %10s %10s\n", "φ", "2(1-φ)", "measured", "rel. err", "n_eff")
for φ in (0.50, 0.80, 0.97)
    E = demo_ar1_energy(φ, DEMO_BFMI_N, DEMO_N_CHAINS, DEMO_SEED + 77)
    got = mean(UIF.bfmi(E))
    want = 2 * (1 - φ)
    rel = abs(got - want) / want
    ok = rel < 0.10
    @printf("    %-8.2f %12.4f %12.4f %9.1f%% %10.0f  %s\n",
            φ, want, got, 100 * rel, DEMO_BFMI_N * (1 - φ) / (1 + φ), ok ? "ok" : "OFF")
    demo_gate!("5f BFMI matches 2(1-φ) at φ=$φ", ok)
end

# --- 5.3 abstention, and a point-estimate fold --------------------------------
#
# A gate whose metric was never recorded must ABSTAIN, not pass. A sampler that emits
# no divergence record would otherwise earn a clean bill of health by recording nothing.

let bare = UIF.tpl_synthetic_chain(DEMO_COLS; n_draws = DEMO_N_DRAWS,
                                   n_chains = DEMO_N_CHAINS, seed = DEMO_SEED),
    s = UIF.audit_convergence([UIF.FoldFit(1, bare, DEMO_METAS[1])])

    @printf("\n  chain with no internals  → abstained on: %s\n", join(s.abstained, ", "))
    demo_gate!("5g absent metrics abstain rather than pass",
               ("divergences" in s.abstained) && ("BFMI" in s.abstained))

    point = UIF.tpl_synthetic_chain(DEMO_COLS; n_draws = 1, n_chains = 1, seed = DEMO_SEED)
    sp = UIF.audit_convergence([UIF.FoldFit(1, point, DEMO_METAS[1])])
    @printf("  point-estimate fold      → applicable folds: %d of %d\n",
            sp.n_applicable, sp.n_folds)
    demo_gate!("5h a point-estimate fold is excluded, not averaged over",
               sp.n_applicable == 0 && sp.n_folds == 1)
end


# %% ===========================================================================
# 6. The pre-game pipeline
# ==============================================================================

demo_banner(6, "pre-game pipeline — FitConfig → fit_model → Fit")

const DEMO_SAMPLER = UIF.ReplaySampler(DEMO_CHAINS)

const DEMO_CONFIG = UIF.FitConfig(
    name = "uif_demo",
    model = DEMO_MODEL,
    splitter = DEMO_SPLITTER,
    sampler = DEMO_SAMPLER,
    tags = ["prototype", "07"],
    description = "Deterministic verification of the unified inference framework.",
    save_dir = joinpath(DEMO_TMP, "fits"),
)

const DEMO_FIT = UIF.fit_model(DEMO_CONFIG;
                               feature_sets = DEMO_FEATURE_SETS,
                               oos_fixtures = DEMO_OOS,
                               quiet = false)

println()
show(stdout, MIME"text/plain"(), DEMO_FIT)
println()

# --- 6.1 ergonomic access ------------------------------------------------------

println("\n  the four-hop legacy read, and its replacement:")
println("    legacy : exp_results.training_results.items[1][1]")
println("    now    : fit[1].chain")

demo_gate!("6a fit[i] yields fold i", DEMO_FIT[1] isa UIF.FoldFit && DEMO_FIT[1].fold == 1)
demo_gate!("6b length(fit) is the fold count", length(DEMO_FIT) == DEMO_N_FOLDS)
demo_gate!("6c fit[1].chain === the chain that was sampled",
           DEMO_FIT[1].chain === DEMO_CHAINS[1])
demo_gate!("6d fit iterates its folds",
           [f.fold for f in DEMO_FIT] == collect(1:DEMO_N_FOLDS))
demo_gate!("6e fit.diagnostics is a field, not a function call",
           DEMO_FIT.diagnostics isa UIF.ConvergenceSummary && DEMO_FIT.diagnostics.passed)
demo_gate!("6f fit.latents was extracted by the run itself",
           DEMO_FIT.latents isa UIF.CountLatents)
demo_gate!("6g latents cover every fold's fixtures",
           UIF.n_matches(DEMO_FIT.latents) == DEMO_N_MATCHES)
demo_gate!("6h latents carry the flattened draw count",
           UIF.n_draws(DEMO_FIT.latents) == DEMO_TOTAL)
demo_gate!("6i provenance recorded", DEMO_FIT.metadata.n_threads == Threads.nthreads() &&
                                     DEMO_FIT.metadata.julia_version == VERSION)
demo_gate!("6j elapsed time tagged",
           any(startswith(t, "time:") for t in DEMO_FIT.config.tags))

@printf("\n  fit[1].chain            : %s %s\n",
        nameof(typeof(DEMO_FIT[1].chain)), size(DEMO_FIT[1].chain))
@printf("  fit.latents             : %s\n", typeof(DEMO_FIT.latents))
@printf("  fit.diagnostics.max_rhat: %.5f\n", DEMO_FIT.diagnostics.max_rhat)
@printf("  fit.metadata.git_commit : %s\n", DEMO_FIT.metadata.git_commit)

# --- 6.2 the fold merge is order-preserving ------------------------------------
#
# The merged container's row order must be fold 1's fixtures, then fold 2's, then fold
# 3's, in the order each fold's extractor produced them. A merge that sorted, or that
# concatenated in the wrong order, would price fixture i with fixture j's posterior —
# and every downstream number would still look entirely reasonable.

const DEMO_EXPECTED_IDS = reduce(vcat, UIF.tpl_ordered_ids(fx) for fx in DEMO_OOS)

demo_gate!("6k merged latents preserve fold-then-fixture order",
           UIF.latent_match_ids(DEMO_FIT.latents) == DEMO_EXPECTED_IDS)

# --- 6.3 execution strategies agree --------------------------------------------
#
# Three executors, one answer. `QueuedExecution` on a 1-chain sampler falls back to the
# threaded one by design (l03 §2), so the three rows below are two code paths and one
# fallback — which is exactly what needs checking, because the fallback is the branch a
# reader is most likely to assume is untested.

let seq = UIF.run_folds(DEMO_MODEL, DEMO_SAMPLER, UIF.SequentialExecution(),
                        DEMO_FEATURE_SETS),
    thr = UIF.run_folds(DEMO_MODEL, DEMO_SAMPLER,
                        UIF.ThreadedExecution(max_concurrent_splits = 4),
                        DEMO_FEATURE_SETS),
    que = UIF.run_folds(DEMO_MODEL, DEMO_SAMPLER,
                        UIF.QueuedExecution(max_concurrent_tasks = 8),
                        DEMO_FEATURE_SETS)

    @printf("\n  executors: sequential %d folds | threaded %d | queued %d\n",
            count(!isnothing, seq), count(!isnothing, thr), count(!isnothing, que))
    demo_gate!("6l sequential, threaded and queued executors agree",
               seq == thr == que == DEMO_CHAINS)
end

demo_gate!("6m AutoExecution resolves off the sampler type",
           UIF.resolve_execution(UIF.AutoExecution(),
                                 BF.Samplers.QueuedNUTSConfig()) isa UIF.QueuedExecution)


# %% ===========================================================================
# 7. Typed latents through the framework
# ==============================================================================
#
# `06` established that its containers price identically to `src`. What is new here is
# the MERGE: three per-fold containers concatenated into one. This section shows the
# merge changes no number, and that the kernels still allocate nothing when the
# container reached them through `fit_model` rather than through a direct
# `extract_latents` call.

demo_banner(7, "typed latents through the framework")

const DEMO_PER_FOLD_LATENTS =
    [UIF.extract_latents(DEMO_MODEL, DEMO_CHAINS[k], DEMO_OOS[k], DEMO_FS)
     for k in 1:DEMO_N_FOLDS]

const DEMO_L = DEMO_FIT.latents

let rows = UIF.ParityRow[]
    # Every parameter matrix, fold by fold, against its slice of the merged container.
    for (name, get) in ((:λ_home, l -> getfield(l, :λ_home)),
                        (:λ_away, l -> getfield(l, :λ_away)))
        offset = 0
        merged = get(DEMO_L)
        for k in 1:DEMO_N_FOLDS
            src = get(DEMO_PER_FOLD_LATENTS[k])
            rows = push!(rows, UIF.tpl_compare(
                "merge $(name) fold $k",
                vec(src), vec(merged[(offset + 1):(offset + size(src, 1)), :])))
            k == DEMO_N_FOLDS || (offset += size(src, 1))
        end
    end
    demo_gate!("7a fold merge is bit-identical to the per-fold containers",
               UIF.tpl_parity_table(rows; title = "7a. FOLD MERGE PARITY (0 ULP)"))
end

# --- 7.1 pricing through the framework's container ------------------------------

let rows = UIF.ParityRow[]
    ws = UIF.GridWorkspace(DEMO_MAX_GOALS)
    S_merged = UIF.alloc_score_grid(DEMO_L, DEMO_MAX_GOALS)
    S_fold   = UIF.alloc_score_grid(DEMO_PER_FOLD_LATENTS[2], DEMO_MAX_GOALS)

    # Fixture 3 of fold 2 is row 3 there and row DEMO_PER_FOLD + 3 in the merge.
    i_fold = 3
    i_merged = DEMO_PER_FOLD + 3

    UIF.compute_score_grid!(S_merged, ws, DEMO_L, i_merged)
    UIF.compute_score_grid!(S_fold, ws, DEMO_PER_FOLD_LATENTS[2], i_fold)
    push!(rows, UIF.tpl_compare("score grid  (merged vs per-fold)", vec(S_merged), vec(S_fold)))

    for m in DEMO_MARKETS
        bm = UIF.alloc_market_book(m, UIF.n_draws(DEMO_L))
        bf = UIF.alloc_market_book(m, UIF.n_draws(DEMO_PER_FOLD_LATENTS[2]))
        UIF.price_market!(bm, S_merged, m)
        UIF.price_market!(bf, S_fold, m)
        for (j, k) in enumerate(UIF.market_keys(m))
            push!(rows, UIF.tpl_compare("price $(k)", bm[j], bf[j]))
        end
    end
    demo_gate!("7b prices from the merged container are bit-identical",
               UIF.tpl_parity_table(rows; title = "7b. PRICING PARITY (0 ULP)"))
end

# --- 7.2 the kernels still allocate nothing ------------------------------------

let rows = UIF.allocation_audit(DEMO_L; max_goals = DEMO_MAX_GOALS,
                                markets = DEMO_MARKETS)
    demo_gate!("7c pre-game kernels allocate 0 bytes on a framework container",
               UIF.tpl_alloc_table(rows; title = "7c. PRE-GAME ALLOCATION AUDIT"))
end

# --- 7.3 the legacy `.df` bridge ------------------------------------------------

let df = DEMO_L.df
    @printf("\n  latents.df  : %d × %d DataFrame, columns %s\n",
            nrow(df), ncol(df), join(string.(propertynames(df)), ", "))
    demo_gate!("7d latents.df gives the legacy frame back",
               df isa AbstractDataFrame && nrow(df) == DEMO_N_MATCHES)
    demo_gate!("7e nrow(latents) is the fixture count", nrow(DEMO_L) == DEMO_N_MATCHES)
end


# %% ===========================================================================
# 8. The in-game pipeline
# ==============================================================================

demo_banner(8, "in-game pipeline — Λ(t → 90) and live pricing")

# The in-play chain's sites, under the names `inplay_scottish/l01_nhpp_scottish.jl`
# samples them, so a chain fitted there replays here without translation.
const DEMO_NB = UIF.n_time_bins(DEMO_INGAME_MODEL)
const DEMO_INGAME_COLS = vcat(["α", "β", "γ_tr", "γ_ld", "γ_man", "σ_time"],
                              ["z_time[$b]" for b in 1:DEMO_NB])

"""
Draw the in-play sites from priors that put the composed kernel in a realistic place.

`α = log(1/90)` centres the per-minute multiplier so that `∫₀⁹⁰ exp(α) dt ≈ 1` — i.e.
the whole-match in-play intensity reproduces the pre-game rate rather than some multiple
of it. `γ_tr > 0 > γ_ld` is the documented game-state effect: a trailing team attacks.
"""
function demo_ingame_site(name::AbstractString, n::Int, rng::AbstractRNG)
    name == "α"      && return log(1 / 90) .+ 0.05 .* randn(rng, n)
    name == "β"      && return 0.15 .+ 0.05 .* randn(rng, n)
    name == "γ_tr"   && return 0.20 .+ 0.05 .* randn(rng, n)
    name == "γ_ld"   && return -0.10 .+ 0.05 .* randn(rng, n)
    name == "γ_man"  && return 0.18 .+ 0.05 .* randn(rng, n)
    name == "σ_time" && return 0.10 .+ 0.03 .* abs.(randn(rng, n))
    return randn(rng, n)                                   # z_time[b]
end

function demo_ingame_chain(; n_draws::Int = DEMO_N_DRAWS, n_chains::Int = DEMO_N_CHAINS,
                             seed::Int = DEMO_SEED + 500)
    rng = MersenneTwister(seed)
    p = length(DEMO_INGAME_COLS)
    vals = Array{Float64, 3}(undef, n_draws, p, n_chains)
    for c in 1:n_chains, (j, nm) in enumerate(DEMO_INGAME_COLS)
        vals[:, j, c] = demo_ingame_site(nm, n_draws, rng)
    end
    return Chains(vals, Symbol.(DEMO_INGAME_COLS))
end

const DEMO_INGAME_CHAIN = demo_ingame_chain()

const DEMO_INGAME_CONFIG = UIF.InGameFitConfig(
    name = "uif_demo_ingame",
    model = DEMO_INGAME_MODEL,
    pregame = DEMO_FIT,                       # the Fit from §6, not a loose container
    splitter = DEMO_SPLITTER,
    sampler = UIF.ReplaySampler([DEMO_INGAME_CHAIN]),
    save_dir = joinpath(DEMO_TMP, "inplay_fits"),
)

const DEMO_INGAME_FIT = UIF.fit_model(DEMO_INGAME_CONFIG;
                                      feature_sets = [(DEMO_FS, DEMO_METAS[1])],
                                      quiet = false)

println()
show(stdout, MIME"text/plain"(), DEMO_INGAME_FIT)
println()

demo_gate!("8a in-game fit resolved its pre-game baseline",
           DEMO_INGAME_FIT.pregame_latents === DEMO_FIT.latents)
demo_gate!("8b baseline recorded in the tags",
           any(startswith(t, "baseline:") for t in DEMO_INGAME_FIT.config.tags))
demo_gate!("8c a Fit with no latents is refused as a baseline",
           (try
                UIF.pregame_latents(UIF.Fit(DEMO_CONFIG, UIF.FoldFit[], nothing,
                                            DEMO_DIAG, DEMO_FIT.metadata, ""))
                false
            catch
                true
            end))

# --- 8.1 the kernel, and the intensity -----------------------------------------

const DEMO_K = UIF.live_kernel(DEMO_INGAME_FIT)
const DEMO_Λh, DEMO_Λa = UIF.alloc_intensity(DEMO_K)

demo_gate!("8d kernel is paired to the pre-game draw count",
           UIF.kernel_n_draws(DEMO_K) == DEMO_TOTAL)
demo_gate!("8e kernel covers the whole clock",
           UIF.kernel_n_bins(DEMO_K) == DEMO_NB && DEMO_K.edges[end] == 95.0)

const DEMO_I = 5     # the fixture priced throughout this section

println("\n  Λ(t → 90) for fixture ", UIF.latent_match_ids(DEMO_L)[DEMO_I],
        " (pre-game λ_h = ",
        @sprintf("%.3f", mean(getfield(DEMO_L, :λ_home)[DEMO_I, :])), "):")
println("    minute   state      Λ_home   Λ_away")

const DEMO_TRACE = [
    (UIF.MatchState(0.0,  0, 0, 0, 0), "0-0 kickoff"),
    (UIF.MatchState(30.0, 0, 0, 0, 0), "0-0"),
    (UIF.MatchState(30.0, 1, 0, 0, 0), "1-0"),
    (UIF.MatchState(30.0, 0, 1, 0, 0), "0-1"),
    (UIF.MatchState(30.0, 0, 0, 0, 1), "0-0, away red"),
    (UIF.MatchState(75.0, 1, 1, 0, 0), "1-1 late"),
    (UIF.MatchState(95.0, 1, 1, 0, 0), "full time"),
]

const DEMO_TRACE_MEANS = map(DEMO_TRACE) do (st, label)
    UIF.remaining_intensity!(DEMO_Λh, DEMO_Λa, DEMO_K, DEMO_L, DEMO_I, st)
    m = (mean(DEMO_Λh), mean(DEMO_Λa))
    @printf("    %6.1f   %-14s %.4f   %.4f\n", st.t, label, m[1], m[2])
    m
end

demo_gate!("8f Λ falls monotonically as the clock runs",
           DEMO_TRACE_MEANS[1][1] > DEMO_TRACE_MEANS[2][1] > DEMO_TRACE_MEANS[6][1])
demo_gate!("8g Λ is zero at full time",
           DEMO_TRACE_MEANS[7][1] == 0.0 && DEMO_TRACE_MEANS[7][2] == 0.0)
demo_gate!("8h a trailing side attacks harder than a level one",
           DEMO_TRACE_MEANS[4][1] > DEMO_TRACE_MEANS[2][1])
demo_gate!("8i a leading side attacks less than a level one",
           DEMO_TRACE_MEANS[3][1] < DEMO_TRACE_MEANS[2][1])
demo_gate!("8j a man advantage raises the advantaged side's Λ",
           DEMO_TRACE_MEANS[5][1] > DEMO_TRACE_MEANS[2][1] &&
           DEMO_TRACE_MEANS[5][2] < DEMO_TRACE_MEANS[2][2])

# --- 8.2 live pricing, and the state that shifts it ----------------------------

println("\n  live 1X2 (posterior mean) as the state moves:")
println("    minute   state            home     draw     away")
let book = UIF.alloc_live_book(D.Market1X2(), DEMO_TOTAL; max_goals = DEMO_MAX_GOALS)
    for (st, label) in DEMO_TRACE[1:6]
        UIF.remaining_intensity!(DEMO_Λh, DEMO_Λa, DEMO_K, DEMO_L, DEMO_I, st)
        UIF.price_live_market!(book, DEMO_Λh, DEMO_Λa, st, D.Market1X2())
        @printf("    %6.1f   %-16s %.4f   %.4f   %.4f\n", st.t, label,
                mean(book[1]), mean(book[2]), mean(book[3]))
    end

    # A 1-0 lead at minute 30 must price home higher than 0-0 at minute 30. This is the
    # thing a pre-game pricer structurally CANNOT express — it has no state — and it is
    # why the live pricer is a separate kernel rather than the same one with a smaller λ.
    UIF.remaining_intensity!(DEMO_Λh, DEMO_Λa, DEMO_K, DEMO_L, DEMO_I, DEMO_TRACE[2][1])
    UIF.price_live_market!(book, DEMO_Λh, DEMO_Λa, DEMO_TRACE[2][1], D.Market1X2())
    level = mean(book[1])
    UIF.remaining_intensity!(DEMO_Λh, DEMO_Λa, DEMO_K, DEMO_L, DEMO_I, DEMO_TRACE[3][1])
    UIF.price_live_market!(book, DEMO_Λh, DEMO_Λa, DEMO_TRACE[3][1], D.Market1X2())
    ahead = mean(book[1])
    demo_gate!("8k goals already scored move the price, not just the intensity",
               ahead > level + 0.20)

    # BTTS with both sides on the board is certain, whatever Λ is. A pricer that applied
    # the state only to the intensity would return something well under 1 here.
    #
    # The exact statement is `no == 0`, not `yes == 1`: NO cell of the grid can reach
    # the `no` accumulator once both sides have scored, so that side is exactly zero,
    # bit for bit. `yes` comes out a few times 1e-8 SHORT of 1, and that shortfall is
    # not the pricer's — it is the 12-goal grid truncation every price in this
    # repository is computed under. `P(N ≥ 12 | Λ = 1.46) ≈ 2.5e-8` per side is missing
    # from the marginal PMFs, so the whole grid sums to `1 − 5e-8` before any market is
    # read off it. Gating on `yes > 1 − 1e-9` would be gating on `max_goals`, not on
    # this kernel, so the truncation is measured and reported instead.
    both = UIF.MatchState(70.0, 1, 1, 0, 0)
    b2 = UIF.alloc_live_book(D.MarketBTTS(), DEMO_TOTAL; max_goals = DEMO_MAX_GOALS)
    UIF.remaining_intensity!(DEMO_Λh, DEMO_Λa, DEMO_K, DEMO_L, DEMO_I, both)
    UIF.price_live_market!(b2, DEMO_Λh, DEMO_Λa, both, D.MarketBTTS())
    @printf("\n  BTTS at 1-1, 70': no = %.1f exactly | yes = 1 − %.2e (the %d-goal grid truncation at max Λ = %.3f)\n",
            maximum(b2[2]), 1 - minimum(b2[1]), DEMO_MAX_GOALS, maximum(DEMO_Λh))
    demo_gate!("8l BTTS `no` is exactly zero once both sides have scored",
               all(iszero, b2[2]))
    demo_gate!("8l' the `yes` shortfall is only the grid truncation",
               all(p -> p > 1 - 1e-6, b2[1]))
end

# --- 8.3 the identity kernel: exact reduction to the pre-game price -------------
#
# The strongest statement available about the live pricer, and it is exact rather than
# approximate.
#
# A kernel with α = β = γ = δ = 0 over a single bin [0, 1] integrates to exactly 1.0 —
# `exp(0.0) * 1.0`, with no rounding anywhere — so `Λ_h[k] == λ_home[i, k]` bit for bit.
# Pricing THAT at an empty kickoff state must therefore reproduce the pre-game price,
# and not merely to a tolerance: the live pricer accumulates the same 144 products into
# the same three accumulators in the same order as `06`'s pre-game pricer.
#
# 0 ULP here means the live path is the pre-game path plus a state, with no arithmetic
# drift introduced along the way. Any reassociation — an `exp(a) * exp(b)` where the
# reference has `exp(a + b)`, a different loop nest — shows up immediately.

let nd = DEMO_TOTAL,
    z = zeros(nd),
    identity_K = UIF.LiveKernel(z, copy(z), copy(z), copy(z), copy(z),
                                zeros(1, nd), [0.0, 1.0], 1.0),
    Λh = Vector{Float64}(undef, nd), Λa = Vector{Float64}(undef, nd),
    ws = UIF.GridWorkspace(DEMO_MAX_GOALS),
    S = UIF.alloc_score_grid(DEMO_L, DEMO_MAX_GOALS),
    rows = UIF.ParityRow[]

    UIF.remaining_intensity!(Λh, Λa, identity_K, DEMO_L, DEMO_I, UIF.kickoff_state())
    push!(rows, UIF.tpl_compare("Λ_home == pre-game λ_home",
                                Λh, getfield(DEMO_L, :λ_home)[DEMO_I, :]))
    push!(rows, UIF.tpl_compare("Λ_away == pre-game λ_away",
                                Λa, getfield(DEMO_L, :λ_away)[DEMO_I, :]))

    UIF.compute_score_grid!(S, ws, DEMO_L, DEMO_I)
    for m in DEMO_MARKETS
        pre = UIF.alloc_market_book(m, nd)
        UIF.price_market!(pre, S, m)
        live = UIF.alloc_live_book(m, nd; max_goals = DEMO_MAX_GOALS)
        UIF.price_live_market!(live, Λh, Λa, UIF.kickoff_state(), m)
        for (j, k) in enumerate(UIF.market_keys(m))
            push!(rows, UIF.tpl_compare("live $(k) == pre-game $(k)", live[j], pre[j]))
        end
    end

    demo_gate!("8m identity kernel at kickoff reproduces the pre-game price exactly",
               UIF.tpl_parity_table(rows;
                   title = "8m. LIVE vs PRE-GAME UNDER AN IDENTITY KERNEL (0 ULP)"))
end

# --- 8.4 zero allocations on the hot paths --------------------------------------

let rows = UIF.AllocRow[]
    st = UIF.MatchState(37.0, 1, 1, 0, 1)
    push!(rows, UIF.tpl_measure_alloc("baseline (empty closure)", 0, () -> nothing))
    push!(rows, UIF.tpl_measure_alloc("remaining_intensity!  [CountLatents]", 0,
              () -> UIF.remaining_intensity!(DEMO_Λh, DEMO_Λa, DEMO_K, DEMO_L, DEMO_I, st)))

    λh = getfield(DEMO_L, :λ_home)[DEMO_I, :]
    λa = getfield(DEMO_L, :λ_away)[DEMO_I, :]
    push!(rows, UIF.tpl_measure_alloc("remaining_intensity!  [raw vectors]", 0,
              () -> UIF.remaining_intensity!(DEMO_Λh, DEMO_Λa, DEMO_K, λh, λa, st)))

    UIF.remaining_intensity!(DEMO_Λh, DEMO_Λa, DEMO_K, DEMO_L, DEMO_I, st)
    for m in DEMO_MARKETS
        b = UIF.alloc_live_book(m, DEMO_TOTAL; max_goals = DEMO_MAX_GOALS)
        push!(rows, UIF.tpl_measure_alloc("price_live_market!  $(m)", 0,
                  () -> UIF.price_live_market!(b, DEMO_Λh, DEMO_Λa, st, m)))
    end

    demo_gate!("8n in-game intensity and live pricing allocate 0 bytes",
               UIF.tpl_alloc_table(rows; title = "8n. IN-GAME ALLOCATION AUDIT"))
end

# --- 8.5 the whole repricing loop, allocation-free -------------------------------
#
# The audit above measures the kernels one at a time. What a match-day process actually
# runs is the LOOP: recompute Λ, then reprice every market, on every tick. If any of the
# plumbing between them allocated, the rows above would still read 0 and the loop would
# still schedule the garbage collector.

let (K, Λh, Λa, books) = UIF.live_book(DEMO_INGAME_FIT, DEMO_MARKETS)
    states = [UIF.MatchState(Float64(t), t ÷ 40, t ÷ 60, 0, 0) for t in 0:5:85]

    function tick_all()
        for st in states
            UIF.remaining_intensity!(Λh, Λa, K, DEMO_L, DEMO_I, st)
            UIF.price_live_market!(books[1], Λh, Λa, st, DEMO_MARKETS[1])
            UIF.price_live_market!(books[2], Λh, Λa, st, DEMO_MARKETS[2])
            UIF.price_live_market!(books[3], Λh, Λa, st, DEMO_MARKETS[3])
        end
        return nothing
    end

    row = UIF.tpl_measure_alloc("$(length(states)) ticks × 3 markets, full loop", 0, tick_all)
    demo_gate!("8o the whole repricing loop allocates 0 bytes",
               UIF.tpl_alloc_table([row]; title = "8o. REPRICING LOOP"))
end


# %% ===========================================================================
# 9. Backward compatibility
# ==============================================================================
#
# The module below is the point of this section. Its body is the legacy call pattern
# COPIED, not adapted: every expression in it appears in `src/experiments/` or in a
# runner under `current_development/`. Only the import line differs, and the header of
# `l06_compat_bridge.jl` explains why it has to.

demo_banner(9, "backward compatibility — a legacy call site, unmodified")

module LegacyCallSite

using ..UnifiedInference.Legacy      # binds `Experiments` and `Training`
using DataFrames

"The legacy read that took four hops. `exp_results.training_results.items[i][1]`."
fold_chain(res, i) = res.training_results.items[i][1]

"The legacy fold count. `r02_train_ireland.jl:156`."
n_folds(res) = length(res.training_results.items)

"The legacy iteration. `r03_pipeline_smoke.jl:214`."
function walk_folds(res)
    out = Int[]
    for (chain, meta) in res.training_results.items
        push!(out, size(chain, 1) * size(chain, 3))
    end
    return out
end

"The legacy sampler read that `save_experiment` performs. `runner.jl:142`."
sampler_name(res) = string(nameof(typeof(res.config.training_config.sampler)))

"The legacy OOS call, and the `.df` that always follows it."
function oos_frame(ds, res)
    latents = Experiments.extract_oos_predictions(ds, res)
    return latents.df
end

"The legacy dead field. `types.jl:42`."
vocabulary(res) = res.vocabulary

"The legacy run. `Experiments.run_experiment(ds, config)`."
run(ds, config; kwargs...) = Experiments.run_experiment(ds, config; kwargs...)

"The legacy save and load."
save(res; kwargs...) = Experiments.save_experiment(res; kwargs...)
load(path) = Experiments.load_experiment(path)

"The legacy training loop. `Training.train(model, config, feature_sets)`."
train(model, cfg, fss) = Training.train(model, cfg, fss; quiet = true)

end # module LegacyCallSite

const LCS = LegacyCallSite

@printf("  fold_chain(res, 1)              : %s %s\n",
        nameof(typeof(LCS.fold_chain(DEMO_FIT, 1))), size(LCS.fold_chain(DEMO_FIT, 1)))
@printf("  n_folds(res)                    : %d\n", LCS.n_folds(DEMO_FIT))
@printf("  walk_folds(res)                 : %s\n", LCS.walk_folds(DEMO_FIT))
@printf("  res.config.training_config      : %s\n", DEMO_FIT.config.training_config)
@printf("  sampler_name(res)               : %s\n", LCS.sampler_name(DEMO_FIT))
@printf("  vocabulary(res)                 : %s\n", LCS.vocabulary(DEMO_FIT))

demo_gate!("9a res.training_results.items[i][1] is the chain",
           LCS.fold_chain(DEMO_FIT, 1) === DEMO_CHAINS[1])
demo_gate!("9b length(res.training_results.items) is the fold count",
           LCS.n_folds(DEMO_FIT) == DEMO_N_FOLDS)
demo_gate!("9c `for (chain, meta) in res.training_results.items` iterates",
           LCS.walk_folds(DEMO_FIT) == fill(DEMO_TOTAL, DEMO_N_FOLDS))
demo_gate!("9d res.config.training_config.sampler resolves",
           LCS.sampler_name(DEMO_FIT) == "ReplaySampler")
demo_gate!("9e res.vocabulary answers nothing, as it always did",
           LCS.vocabulary(DEMO_FIT) === nothing)

let df = LCS.oos_frame(nothing, DEMO_FIT)
    demo_gate!("9f extract_oos_predictions(ds, res).df is the legacy frame",
               df isa AbstractDataFrame && nrow(df) == DEMO_N_MATCHES &&
               :match_id in propertynames(df))
end

let tr = LCS.train(DEMO_MODEL,
                   BF.Training.TrainingConfig(sampler = DEMO_SAMPLER,
                                              strategy = BF.Training.Independent()),
                   DEMO_FEATURE_SETS)
    @printf("  Training.train(...)             : %d folds, items[2][1] is a %s\n",
            length(tr.items), nameof(typeof(tr.items[2][1])))
    demo_gate!("9g Training.train returns the legacy shape",
               length(tr.items) == DEMO_N_FOLDS && tr.items[2][1] === DEMO_CHAINS[2])
end

# The legacy CONFIG construction — `ExperimentConfig(; …, training_config = …)` — and
# the alias identity that makes `res isa ExperimentResults` still true.
let legacy_cfg = UIF.ExperimentConfig(
        name = "legacy_shaped",
        model = DEMO_MODEL,
        splitter = DEMO_SPLITTER,
        training_config = BF.Training.TrainingConfig(
            sampler = DEMO_SAMPLER,
            strategy = BF.Training.Independent(parallel = true, max_concurrent_tasks = 6)),
        save_dir = joinpath(DEMO_TMP, "legacy"))

    @printf("  ExperimentConfig(…)             : sampler %s, execution %s\n",
            nameof(typeof(legacy_cfg.sampler)), legacy_cfg.execution)
    demo_gate!("9h legacy ExperimentConfig construction unpacks training_config",
               legacy_cfg.sampler === DEMO_SAMPLER)
    demo_gate!("9i the Independent strategy's cap survives the unpack",
               legacy_cfg.execution isa UIF.AutoExecution &&
               legacy_cfg.execution.max_concurrent_tasks == 6)
    demo_gate!("9j ExperimentResults is Fit, ExperimentConfig is FitConfig",
               UIF.ExperimentResults === UIF.Fit && UIF.ExperimentConfig === UIF.FitConfig)
    demo_gate!("9k res isa ExperimentResults", DEMO_FIT isa UIF.ExperimentResults)
end

# `ExperimentTask(ds, config)` — the construction the briefing's NamedTuple alias could
# not have supported (l06 §2).
let task = UIF.ExperimentTask(nothing, DEMO_CONFIG)
    demo_gate!("9l ExperimentTask(ds, config) still constructs",
               task.config === DEMO_CONFIG && task isa UIF.FitTask)
end


# %% ===========================================================================
# 10. Persistence and metadata
# ==============================================================================

demo_banner(10, "persistence, sidecars and the legacy upgrade path")

const DEMO_SAVED = UIF.save_fit(DEMO_FIT; quiet = false)

let files = sort(readdir(DEMO_SAVED))
    println("\n  artefacts: ", join(files, ", "))
    demo_gate!("10a all four artefacts written",
               files == ["config.json", "meta.json", "oos_latents.jls", "results.jld2"])
    demo_gate!("10b no temporary files left behind",
               !any(f -> occursin(".tmp.", f), files))
end

# --- 10.1 the sidecar answers the question a scan is run to answer ---------------

let meta = JSON3.read(read(joinpath(DEMO_SAVED, UIF.UIF_META_FILE), String))
    println("\n  meta.json:")
    for k in (:name, :model, :sampler, :n_folds, :n_draws, :n_oos_fixtures,
              :converged, :max_rhat, :min_ess, :n_divergent, :time_taken, :git_commit)
        @printf("    %-16s %s\n", string(k), repr(get(meta, k, nothing)))
    end

    required = (:kind, :name, :model, :splitter, :sampler, :timestamp, :time_taken,
                :n_folds, :n_draws, :n_oos_fixtures, :has_oos_latents, :converged,
                :max_rhat, :min_ess, :n_divergent, :git_commit)
    demo_gate!("10c sidecar carries every scannable field",
               all(k -> haskey(meta, k), required))
    demo_gate!("10d sidecar records the convergence verdict the legacy one omitted",
               meta[:converged] === true && meta[:n_folds] == DEMO_N_FOLDS &&
               meta[:n_oos_fixtures] == DEMO_N_MATCHES)
    demo_gate!("10e sidecar is valid JSON with no NaN",
               !occursin("NaN", read(joinpath(DEMO_SAVED, UIF.UIF_META_FILE), String)))
end

# --- 10.2 round trip --------------------------------------------------------------

const DEMO_RELOADED = UIF.load_fit(DEMO_SAVED)

println()
show(stdout, MIME"text/plain"(), DEMO_RELOADED)
println()

demo_gate!("10f a saved Fit reloads as a Fit", DEMO_RELOADED isa UIF.Fit)
demo_gate!("10g every fold survives the round trip",
           length(DEMO_RELOADED) == DEMO_N_FOLDS)
demo_gate!("10h the chains survive bit-for-bit",
           all(k -> Array(DEMO_RELOADED[k].chain) == Array(DEMO_CHAINS[k]),
               1:DEMO_N_FOLDS))
demo_gate!("10i the diagnostics survive",
           DEMO_RELOADED.diagnostics.passed == DEMO_FIT.diagnostics.passed &&
           DEMO_RELOADED.diagnostics.max_rhat == DEMO_FIT.diagnostics.max_rhat)
demo_gate!("10j the latents survive",
           getfield(DEMO_RELOADED.latents, :λ_home) == getfield(DEMO_L, :λ_home))
demo_gate!("10k the metadata survives",
           DEMO_RELOADED.metadata.git_commit == DEMO_FIT.metadata.git_commit)

# The cached container, read through the legacy filename by the legacy accessor.
demo_gate!("10l has_oos_predictions finds the cache",
           UIF.has_oos_predictions(DEMO_SAVED))
demo_gate!("10m load_oos_predictions returns the typed container",
           UIF.load_oos_predictions(DEMO_SAVED) isa UIF.CountLatents)

# --- 10.3 atomicity ---------------------------------------------------------------
#
# Not a claim about `mv` — that is the operating system's — but a check that the write
# path actually goes through it and leaves nothing behind on the way.

let probe = joinpath(DEMO_TMP, "atomic_probe.bin")
    UIF.atomic_write(probe) do tmp
        write(tmp, "payload")
    end
    leftovers = filter(f -> occursin("atomic_probe.bin.tmp", f), readdir(DEMO_TMP))
    demo_gate!("10n atomic_write leaves the target and no scratch",
               read(probe, String) == "payload" && isempty(leftovers))

    failed = joinpath(DEMO_TMP, "atomic_fail.bin")
    threw = try
        UIF.atomic_write(failed) do tmp
            write(tmp, "half")
            error("simulated interrupt")
        end
        false
    catch
        true
    end
    demo_gate!("10o a failed write leaves no partial file",
               threw && !isfile(failed) &&
               isempty(filter(f -> occursin("atomic_fail", f), readdir(DEMO_TMP))))
end

# --- 10.4 the legacy upgrade path ---------------------------------------------------
#
# A genuine `BayesianFootball.Experiments.ExperimentResults`, saved the way the legacy
# runner saves one, loaded by this framework. Not a mock of the legacy type — the type
# itself, from `src`.

const DEMO_LEGACY_DIR = joinpath(DEMO_TMP, "legacy_run_20240601_120000")

let cfg = BF.Experiments.ExperimentConfig(
        name = "legacy_run",
        model = DEMO_MODEL,
        splitter = DEMO_SPLITTER,
        training_config = BF.Training.TrainingConfig(
            sampler = BF.Samplers.NUTSConfig(n_samples = DEMO_N_DRAWS,
                                             n_chains = DEMO_N_CHAINS),
            strategy = BF.Training.Independent()),
        tags = ["time:2m 30s"],
        save_dir = DEMO_TMP),
    tr = BF.Training.TrainingResults([(DEMO_CHAINS[k], DEMO_METAS[k])
                                      for k in 1:DEMO_N_FOLDS]),
    res = BF.Experiments.ExperimentResults(cfg, tr, nothing, DEMO_LEGACY_DIR)

    mkpath(DEMO_LEGACY_DIR)
    jldsave(joinpath(DEMO_LEGACY_DIR, "results.jld2"); results = res)
    println("\n  wrote a genuine legacy ExperimentResults to ", basename(DEMO_LEGACY_DIR))
end

const DEMO_UPGRADED = UIF.load_fit(DEMO_LEGACY_DIR)

println()
show(stdout, MIME"text/plain"(), DEMO_UPGRADED)
println()

demo_gate!("10p a legacy ExperimentResults loads as a Fit", DEMO_UPGRADED isa UIF.Fit)
demo_gate!("10q its folds and chains are recovered",
           length(DEMO_UPGRADED) == DEMO_N_FOLDS &&
           Array(DEMO_UPGRADED[2].chain) == Array(DEMO_CHAINS[2]))
demo_gate!("10r the nested training_config.sampler is flattened",
           DEMO_UPGRADED.config.sampler isa BF.Samplers.NUTSConfig)
demo_gate!("10s it is RE-AUDITED on load, gaining diagnostics it never had",
           DEMO_UPGRADED.diagnostics isa UIF.ConvergenceSummary &&
           DEMO_UPGRADED.diagnostics.passed)
demo_gate!("10t the legacy `time:` tag is recovered as seconds",
           DEMO_UPGRADED.metadata.elapsed_seconds == 150.0)
demo_gate!("10u the timestamp is recovered from the directory name",
           DEMO_UPGRADED.metadata.timestamp == DateTime(2024, 6, 1, 12, 0, 0))
demo_gate!("10v an upgraded Fit answers the legacy properties too",
           LCS.fold_chain(DEMO_UPGRADED, 1) === DEMO_UPGRADED[1].chain)

# --- 10.5 discovery ----------------------------------------------------------------

const DEMO_LISTED = UIF.list_fits(joinpath(DEMO_TMP, "fits"))

demo_gate!("10w list_fits finds the run without opening its binary",
           length(DEMO_LISTED) == 1 && DEMO_LISTED[1].n_folds == DEMO_N_FOLDS &&
           DEMO_LISTED[1].converged === true)

let paths = UIF.list_experiments(joinpath(DEMO_TMP, "fits"))
    demo_gate!("10x list_experiments returns paths, as it always did",
               paths isa Vector{String} && length(paths) == 1 &&
               UIF.load_experiment(paths, 1) isa UIF.Fit)
end


# %% ===========================================================================
# 11. What it costs
# ==============================================================================
#
# Reported rather than asserted. The framework's job is to remove work, and the honest
# way to show that is to time the work it removes.

demo_banner(11, "cost")

let rows = UIF.TimingRow[]
    n = 200
    st = UIF.MatchState(37.0, 1, 1, 0, 1)
    ws = UIF.GridWorkspace(DEMO_MAX_GOALS)
    S = UIF.alloc_score_grid(DEMO_L, DEMO_MAX_GOALS)
    m = D.Market1X2()
    pre_book = UIF.alloc_market_book(m, DEMO_TOTAL)
    live_book_1x2 = UIF.alloc_live_book(m, DEMO_TOTAL; max_goals = DEMO_MAX_GOALS)

    push!(rows, UIF.tpl_time("pre-game  grid fill", n,
              () -> (for _ in 1:n; UIF.compute_score_grid!(S, ws, DEMO_L, DEMO_I); end)))
    push!(rows, UIF.tpl_time("pre-game  price 1X2", n,
              () -> (for _ in 1:n; UIF.price_market!(pre_book, S, m); end)))
    push!(rows, UIF.tpl_time("in-game   remaining Λ", n,
              () -> (for _ in 1:n
                         UIF.remaining_intensity!(DEMO_Λh, DEMO_Λa, DEMO_K, DEMO_L,
                                                  DEMO_I, st)
                     end)))
    push!(rows, UIF.tpl_time("in-game   price 1X2", n,
              () -> (for _ in 1:n
                         UIF.price_live_market!(live_book_1x2, DEMO_Λh, DEMO_Λa, st, m)
                     end)))

    UIF.tpl_timing_table(rows;
        title = "11. TIMING  ($(DEMO_TOTAL) draws, $(DEMO_MAX_GOALS)² grid)")
    println("  (the right-hand column is µs per CALL — one fixture, all $(DEMO_TOTAL) " *
            "draws — not per fixture-draw.)")
end

println()
@printf("  latent container : %d heap object(s), %s\n",
        UIF.latent_allocations(DEMO_L),
        UIF._tpl_human_bytes(UIF.latent_bytes(DEMO_L)))
@printf("  live kernel      : %d draws × %d bins, %s\n",
        UIF.kernel_n_draws(DEMO_K), UIF.kernel_n_bins(DEMO_K),
        UIF._tpl_human_bytes(sizeof(DEMO_K.δ_time) + 5 * sizeof(DEMO_K.α)))
@printf("  saved run        : %s on disk\n",
        UIF._tpl_human_bytes(sum(filesize(joinpath(DEMO_SAVED, f))
                                 for f in readdir(DEMO_SAVED))))
println()
println("  Work the legacy path does and this one does not:")
println("    * one re-derivation of boundaries + feature sets per `extract_oos_predictions`")
println("    * a second one per `Diagnostics.extract_chains`")
println("    * both need a live DataStore; `audit_convergence` and `load_fit` need none")


# %% ===========================================================================
# 12. Final report
# ==============================================================================

demo_banner(12, "final report")

try
    rm(DEMO_TMP; recursive = true, force = true)
catch e
    @warn "Could not remove scratch directory $DEMO_TMP" exception = e
end

let width = maximum(length(first(g)) for g in DEMO_GATES; init = 20)
    all_ok = all(last, DEMO_GATES)
    for (name, ok) in DEMO_GATES
        @printf("  %-*s  %s\n", width, name, ok ? "pass" : "FAIL")
    end
    println("  ", "-"^(width + 8))
    @printf("  %-*s  %d / %d\n", width, "gates passed",
            count(last, DEMO_GATES), length(DEMO_GATES))
    println()

    if all_ok
        println("  RESULT: PASS")
        println()
        println("  The unified lifecycle — split → sample → AUDIT → EXTRACT → Fit — runs")
        println("  end to end. The convergence telemetry passes a healthy run and fails")
        println("  each of four pathologies at its own gate, with no DataStore in the call.")
        println("  The fold merge and every price through it are bit-identical (0 ULP) to")
        println("  the per-fold containers, and the pre-game kernels still allocate zero.")
        println("  The in-game bridge computes Λ(t → 90) and prices a live state at zero")
        println("  bytes, and under an identity kernel reproduces the pre-game price to")
        println("  the last bit. A legacy call site runs verbatim, and a genuine legacy")
        println("  `ExperimentResults` upgrades to a `Fit` — with diagnostics it never had.")
        println()
        println("  NOT SHOWN, and not claimed: that any of these models fits anything, or")
        println("  that any price here is good. The posteriors are prior draws with a")
        println("  fixed seed (06/l04_parity.jl §9).")
    else
        println("  RESULT: FAIL — see the failing gates above.")
    end

    # Non-zero exit so this runner is usable as a CI check, but only when run as a
    # script: an `include` from a REPL should not kill the session.
    if abspath(PROGRAM_FILE) == @__FILE__
        exit(all_ok ? 0 : 1)
    end
end
