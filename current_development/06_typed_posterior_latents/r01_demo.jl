# ==============================================================================
# 06 — TYPED POSTERIOR LATENTS : THE PROOF
# ==============================================================================
#
# WHAT THIS IS
#   A deterministic verification that replacing `Experiments.LatentStates`' untyped
#   `DataFrame` with dense typed matrix containers changes NO PRICE, and that the
#   resulting score-grid and market-pricing kernels allocate nothing.
#
#   Three container families are exercised end to end, each against the live `src`
#   prediction path it must replace:
#
#     CountLatents    Poisson and NegBin   ← 05_composable_count_builder engines
#     RecombLatents   open play + pens + own goals ← DynamicPxGRecombModel
#     SmileLatents    grid + market smile  ← DynamicSmileDoublePoissonGoalsLeague…
#                                            and the smile_negbin prototype
#
# WHAT THIS IS NOT
#   Not a model-fitting run, not an evaluation, not a betting study. Nothing here
#   claims any model is good, converged, or profitable. The only claim is that the
#   posterior can be moved out of a DataFrame without any number changing.
#
#   No MCMC is run. Chains are drawn from priors with a fixed seed — see
#   `l04_parity.jl` §9 for exactly what that does and does not test. Everything
#   downstream of the draws is the real code path.
#
# THE COMPARABILITY CONTRACT
#   Every parity row compares TWO KERNELS OVER ONE SET OF NUMBERS. The typed container
#   and the legacy DataFrame are built from the same `extract_parameters` call (or, for
#   the recombination family, from one common set of channel components), so any
#   difference the tables show is the kernel's and nothing else's.
#
#   The legacy side always goes through the live `src` entry points
#   (`Predictions.extract_params` → `compute_score_matrix` → `compute_market_probs`),
#   never a transcription of them. If `src` changes, these tables change with it.
#
# WHAT WOULD FALSIFY THE CLAIM
#   Any non-zero ULP in §8 or §9. "Within 1e-12" is the briefing's threshold and is
#   enforced, but the standard actually being held to is bit-identity: two
#   implementations of one density evaluated on one set of parameters must agree to the
#   last bit, and a systematic reassociation would sail through a 1e-12 tolerance.
#
#   Any non-zero byte count in §10's steady-state rows.
#
# DEFECTS THIS RUN REPRODUCES (§6)
#   Three pre-existing `src` bugs, demonstrated against the live method table and the
#   live MCMCChains version rather than asserted from a reading. None is fixed here —
#   see §6 for why, and for the one compatibility shim that had to be taken anyway.
#
# PERSISTENCE
#   None. This runner writes no files and reads no cache. It is safe to re-run and
#   safe to interrupt.
#
# USAGE
#   julia --project current_development/06_typed_posterior_latents/r01_demo.jl
#
#   or, in a REPL:
#     include("current_development/06_typed_posterior_latents/r01_demo.jl")
#
#   Runs in seconds. Exits non-zero if any gate fails, so it is usable in CI.
#
# ==============================================================================


# %% ===========================================================================
# 1. Packages and implementation
# ==============================================================================

using BayesianFootball
using DataFrames
using Dates
using MCMCChains
using Printf
using Random
using Statistics

# The thing under test. `l04_parity.jl` transitively includes l03 → l02 → l01, so this
# one line loads the whole prototype.
include(joinpath(@__DIR__, "l04_parity.jl"))

# The composable count builder supplies the CountLatents family's models. Loading it
# also pulls in the wealth and distance feature loaders it wraps; neither is called
# here (the fixtures carry a materialised `delta_wealth_logsum` column instead), so
# this runner needs no cached valuation or geocode data.
include(joinpath(@__DIR__, "..", "scottish_lower", "05_composable_count_builder",
                 "l03_engine.jl"))

# The smile NegBin prototype supplies the fourth pricing kernel — the one legacy
# reference for `SmileLatents{Float64,<:NamedTuple}`, which has no `src` engine.
include(joinpath(@__DIR__, "..", "smile_negbin", "l01_smile_negbin_engine.jl"))
include(joinpath(@__DIR__, "..", "smile_negbin", "l02_smile_negbin_predict.jl"))

const DEMO_PG = BayesianFootball.Models.PreGame
const DEMO_D  = BayesianFootball.Data

# Registered here rather than in l02 because the model type lives in a prototype
# directory: l02_extract.jl must not depend on `current_development/smile_negbin/`
# being loaded.
latent_family(::DynamicSmileDoubleNegBinXGOutfieldPlayerTimeDecayModel) = SmileNegBinFamily()


# %% ===========================================================================
# 2. Configuration
# ==============================================================================

const DEMO_N_MATCHES = 24     # held-out fixtures per family
const DEMO_N_TEAMS   = 8
const DEMO_N_DRAWS   = 200    # posterior draws PER CHAIN
const DEMO_N_CHAINS  = 2      # exercises the size(chain,1)*size(chain,3) flattening
const DEMO_N_SEASONS = 2
const DEMO_N_LEAGUES = 2
const DEMO_SMILE_KMAX = 4     # φ covers K = 0..4, i.e. lines 0.5 .. 4.5
const DEMO_MAX_GOALS = TPL_MAX_GOALS
const DEMO_TOL       = 1e-12  # the briefing's threshold; the ULP column is the real one
const DEMO_SEED      = 20240601

# One O/U line beyond the smile ladder (5.5 → K=5, outside K=0..4) so the smile
# container's documented grid FALLBACK is exercised, not just its smile route.
const DEMO_MARKETS = (
    DEMO_D.Market1X2(),
    DEMO_D.MarketBTTS(),
    DEMO_D.MarketOverUnder(1.5),
    DEMO_D.MarketOverUnder(2.5),
    DEMO_D.MarketOverUnder(3.5),
    DEMO_D.MarketOverUnder(5.5),
)

# Every gate's verdict lands here; §12 is the only place that decides pass/fail.
const DEMO_GATES = Pair{String, Bool}[]
demo_gate!(name, ok) = (push!(DEMO_GATES, name => ok); ok)

function demo_banner(n, title)
    println()
    println("=" ^ 78)
    @printf("%d. %s\n", n, uppercase(title))
    println("=" ^ 78)
end


# %% ===========================================================================
# 3. Deterministic fixtures and metadata
# ==============================================================================

demo_banner(3, "deterministic fixtures")

const DEMO_FIXTURES = tpl_synthetic_fixtures(DEMO_N_MATCHES;
                                             n_teams = DEMO_N_TEAMS, seed = DEMO_SEED)
const DEMO_TEAM_MAP = tpl_team_map(DEMO_N_TEAMS)

# match_id → league index. Half the fixtures in each division, so the zero-sum league
# offset in the recombination and smile engines is actually non-degenerate.
const DEMO_LEAGUE_LOOKUP = Dict(
    Int(r.match_id) => (i <= DEMO_N_MATCHES ÷ 2 ? 1 : 2)
    for (i, r) in enumerate(eachrow(DEMO_FIXTURES))
)

# match_id → squad-wealth differential, for the recombination engine's own wealth term
# (which reads a lookup, where the composable builder reads a DataFrame column).
const DEMO_WEALTH_LOOKUP = Dict(
    Int(r.match_id) => r.delta_wealth_logsum for r in eachrow(DEMO_FIXTURES)
)

@printf("  fixtures        : %d over %d teams\n", DEMO_N_MATCHES, DEMO_N_TEAMS)
@printf("  posterior draws : %d samples × %d chain(s) = %d flattened\n",
        DEMO_N_DRAWS, DEMO_N_CHAINS, DEMO_N_DRAWS * DEMO_N_CHAINS)
@printf("  date span       : %s .. %s (%d distinct months)\n",
        minimum(DEMO_FIXTURES.match_date), maximum(DEMO_FIXTURES.match_date),
        length(unique(Dates.month.(DEMO_FIXTURES.match_date))))
@printf("  score grid      : %d × %d\n", DEMO_MAX_GOALS, DEMO_MAX_GOALS)
println("\n  first three fixtures:")
show(stdout, MIME"text/plain"(), first(DEMO_FIXTURES, 3))
println()


# %% ===========================================================================
# 4. Family A — CountLatents, from the composable count builder
# ==============================================================================
#
# Two models, identical linear predictor, different observation density. That is the
# whole point of `05_composable_count_builder`, and it is exactly the pair that the
# typed container has to keep distinct: `CountLatents{Float64,Nothing}` and
# `CountLatents{Float64,<:NamedTuple}` reach different grid kernels by DISPATCH, where
# the legacy reader reaches them by a `hasproperty(row, :r)` test performed once per
# fixture per prediction.

demo_banner(4, "family A — CountLatents (Poisson and NegBin)")

"The Poisson arm: the Scottish Lower baseline plus a squad-wealth supremacy covariate."
function demo_poisson_count_model()
    b = CountModelBuilder(:tpl_demo_poisson)
    add!(b, DEMO_PG.GlobalInterception(),
            DEMO_PG.TimeDecayDynamics(days_half_life = 365.0),
            DEMO_PG.GlobalHomeAdvantage())
    add!(b, WealthCovariate())
    return build(b)
end

"""
The NegBin arm: the same predictor, with `HomeAwayDispersion`.

`HomeAwayDispersion` rather than `GlobalDispersion` on purpose — it makes r_h ≠ r_a, so
the container's two dispersion matrices carry genuinely different numbers and a kernel
that silently used one for both sides would fail §8 instead of passing by coincidence.
"""
function demo_negbin_count_model()
    b = CountModelBuilder(:tpl_demo_negbin)
    add!(b, DEMO_PG.GlobalInterception(),
            DEMO_PG.TimeDecayDynamics(days_half_life = 365.0),
            DEMO_PG.GlobalHomeAdvantage())
    add!(b, WealthCovariate())
    add!(b, DEMO_PG.HomeAwayDispersion())
    return build(b)
end

const DEMO_POIS_MODEL   = demo_poisson_count_model()
const DEMO_NEGBIN_MODEL = demo_negbin_count_model()

println("  ", DEMO_POIS_MODEL)
println("  ", DEMO_NEGBIN_MODEL)

# The chain schema is DERIVED from the assembled components — `cb_chain_columns` walks
# the same component list the engine declares its sites from — so this runner never
# hand-writes a site name for these two models.
const DEMO_POIS_COLS   = cb_chain_columns(DEMO_POIS_MODEL, DEMO_N_TEAMS)
const DEMO_NEGBIN_COLS = cb_chain_columns(DEMO_NEGBIN_MODEL, DEMO_N_TEAMS)

@printf("\n  Poisson chain sites : %d  (%s …)\n",
        length(DEMO_POIS_COLS), join(DEMO_POIS_COLS[1:min(4, end)], ", "))
@printf("  NegBin  chain sites : %d  (adds %s)\n",
        length(DEMO_NEGBIN_COLS), join(setdiff(DEMO_NEGBIN_COLS, DEMO_POIS_COLS), ", "))

const DEMO_POIS_CHAIN = tpl_synthetic_chain(DEMO_POIS_COLS;
    n_draws = DEMO_N_DRAWS, n_chains = DEMO_N_CHAINS, seed = DEMO_SEED)
const DEMO_NEGBIN_CHAIN = tpl_synthetic_chain(DEMO_NEGBIN_COLS;
    n_draws = DEMO_N_DRAWS, n_chains = DEMO_N_CHAINS, seed = DEMO_SEED + 1)

# The composable engine's OOS extractor reads exactly these three keys.
const DEMO_COUNT_FS = tpl_feature_set(
    n_teams   = DEMO_N_TEAMS,
    n_seasons = DEMO_N_SEASONS,
    team_map  = DEMO_TEAM_MAP,
)

# One `extract_parameters` call per model, feeding BOTH sides of every comparison:
# the typed container is packed from it, and the legacy DataFrame is built from it by
# the same constructor `src` uses. Same numbers, two layouts.
const DEMO_POIS_RAW = DEMO_PG.extract_parameters(
    DEMO_POIS_MODEL, DEMO_FIXTURES, DEMO_COUNT_FS, DEMO_POIS_CHAIN)
const DEMO_NEGBIN_RAW = DEMO_PG.extract_parameters(
    DEMO_NEGBIN_MODEL, DEMO_FIXTURES, DEMO_COUNT_FS, DEMO_NEGBIN_CHAIN)

const DEMO_IDS = tpl_ordered_ids(DEMO_FIXTURES)

const DEMO_POIS_LATENTS = extract_latents(
    DEMO_POIS_MODEL, DEMO_POIS_CHAIN, DEMO_FIXTURES, DEMO_COUNT_FS)
const DEMO_NEGBIN_LATENTS = extract_latents(
    DEMO_NEGBIN_MODEL, DEMO_NEGBIN_CHAIN, DEMO_FIXTURES, DEMO_COUNT_FS)

const DEMO_POIS_LEGACY_DF   = tpl_legacy_latents_df(DEMO_POIS_RAW, DEMO_IDS)
const DEMO_NEGBIN_LEGACY_DF = tpl_legacy_latents_df(DEMO_NEGBIN_RAW, DEMO_IDS)

println()
show(stdout, MIME"text/plain"(), DEMO_POIS_LATENTS)
println()
show(stdout, MIME"text/plain"(), DEMO_NEGBIN_LATENTS)
println()

@printf("\n  λ_home range : [%.4f, %.4f]   λ_away range : [%.4f, %.4f]\n",
        extrema(DEMO_POIS_LATENTS.λ_home)..., extrema(DEMO_POIS_LATENTS.λ_away)...)
@printf("  r_h range    : [%.4f, %.4f]   r_a range    : [%.4f, %.4f]  (r_h ≠ r_a: %s)\n",
        extrema(DEMO_NEGBIN_LATENTS.observation_params.r_h)...,
        extrema(DEMO_NEGBIN_LATENTS.observation_params.r_a)...,
        DEMO_NEGBIN_LATENTS.observation_params.r_h !=
            DEMO_NEGBIN_LATENTS.observation_params.r_a)

# The container's schema IS its type, and that is the claim worth stating in the
# transcript: it is checkable by a reader, and by a compiler.
@printf("\n  Poisson container type : %s\n", typeof(DEMO_POIS_LATENTS))
@printf("  NegBin  container type : %s\n", typeof(DEMO_NEGBIN_LATENTS))

demo_gate!("A1 Poisson family resolved by dispatch",
           observation_family(DEMO_POIS_LATENTS) === :poisson)
demo_gate!("A2 NegBin family resolved by dispatch",
           observation_family(DEMO_NEGBIN_LATENTS) === :negbin)
demo_gate!("A3 legacy DataFrame columns are Vector{Any} (as src builds them)",
           all(c -> c isa Vector{Any}, eachcol(DEMO_POIS_LEGACY_DF)))


# %% ===========================================================================
# 5. Family B — RecombLatents
# ==============================================================================
#
# The additive-channel family. Its container carries the DECOMPOSITION (open play,
# penalties, own goals, and the pre-finishing pxG intensity) where the legacy row
# carries only whichever subset the engine happened to write — which is why
# `recombination.jl:18-40` needs a three-branch `hasproperty` cascade with six
# hard-coded empirical fallbacks to read it back.

demo_banner(5, "family B — RecombLatents")

const DEMO_RECOMB_MODEL = DEMO_PG.DynamicPxGRecombModel()

"""
Chain sites for `DynamicPxGRecombModel` with its default components.

Written out rather than derived: unlike the composable builder, this engine has no
component-driven schema function, and the site names are only discoverable by reading
`build_recombined_pxg_engine`. That asymmetry is itself an argument for the builder.
"""
function demo_recomb_chain_columns(n_teams::Int, n_leagues::Int)
    cols = String[]
    push!(cols, "inter.μ_base[1]", "inter.σ_month")                 # HierarchicalMonthly
    append!(cols, ["inter.raw_month[$i]" for i in 1:12])
    push!(cols, "ha.γ_global")                                       # GlobalHomeAdvantage
    push!(cols, "dyn.σ_a", "dyn.σ_d")                                # TimeDecayDynamics
    append!(cols, ["dyn.raw_a[$i]" for i in 1:n_teams])
    append!(cols, ["dyn.raw_d[$i]" for i in 1:n_teams])
    push!(cols, "wealth.w_wealth")                                   # LinearSquadWealth
    push!(cols, "pxg.ν_xg")                                          # GammaPxGObservation
    append!(cols, ["log_κ_raw[$i]" for i in 1:n_teams])              # finishing factors
    append!(cols, ["δ_league_raw[$i]" for i in 1:n_leagues])
    push!(cols, "officiating.pen_base_μ", "officiating.ha_pen", "officiating.σ_ref")
    return cols
end

const DEMO_RECOMB_COLS = demo_recomb_chain_columns(DEMO_N_TEAMS, DEMO_N_LEAGUES)

# ONE CHAIN, deliberately. See §6 defect 2 and `tpl_multichain_warning()`.
const DEMO_RECOMB_CHAIN = tpl_synthetic_chain(DEMO_RECOMB_COLS;
    n_draws = DEMO_N_DRAWS, n_chains = 1, seed = DEMO_SEED + 2)

const DEMO_RECOMB_FS = tpl_feature_set(
    n_teams       = DEMO_N_TEAMS,
    n_leagues     = DEMO_N_LEAGUES,
    team_map      = DEMO_TEAM_MAP,
    league_lookup = DEMO_LEAGUE_LOOKUP,
    wealth_lookup = DEMO_WEALTH_LOOKUP,
)

# The raw channel components, extracted ONCE. Both sides of the §8 comparison are built
# from this single object, so the comparison is between two arithmetic paths over one
# set of inputs — not a round trip of one path through itself.
const DEMO_RECOMB_COMPONENTS = tpl_recomb_components(
    DEMO_RECOMB_MODEL, DEMO_RECOMB_CHAIN, DEMO_FIXTURES, DEMO_RECOMB_FS)

const DEMO_RECOMB_LATENTS = recomb_latents_from_components(DEMO_RECOMB_COMPONENTS)

"""
The legacy latents row for a recombination fixture, in the CHANNEL form.

`Predictions.extract_params(::AbstractRecombinationModels, row)` has three branches;
this frame deliberately omits `:λ_h`/`:λ_a` so the reader takes its THIRD branch and
re-derives the total from components (recombination.jl:26-38). Handing it a
pre-summed `:λ_h` would make it take branch one and the comparison would test nothing
but a copy.
"""
function demo_recomb_legacy_df(c::NamedTuple)
    rows(m) = [collect(view(m, i, :)) for i in 1:size(m, 1)]
    return DataFrame(
        :match_id => Vector{Any}(c.match_ids),
        :μ_open_h => Any[r for r in rows(c.μ_open_h)],
        :μ_open_a => Any[r for r in rows(c.μ_open_a)],
        :κ_h      => Any[r for r in rows(c.κ_h)],
        :κ_a      => Any[r for r in rows(c.κ_a)],
        :q_pen    => Any[r for r in rows(c.q_pen)],
        :λ_pen_h  => Any[r for r in rows(c.λ_pen_raw_h)],
        :λ_pen_a  => Any[r for r in rows(c.λ_pen_raw_a)],
        :λ_og     => Any[r for r in rows(c.og_rate)],
    )
end

const DEMO_RECOMB_LEGACY_DF = demo_recomb_legacy_df(DEMO_RECOMB_COMPONENTS)

println()
show(stdout, MIME"text/plain"(), DEMO_RECOMB_LATENTS)
println()

let l = DEMO_RECOMB_LATENTS
    tot_h = l.λ_open_h .+ l.λ_pen_h .+ l.λ_og_h
    @printf("\n  channel shares (mean over all draws and fixtures):\n")
    @printf("    open play : %.4f  (%.1f%%)\n", mean(l.λ_open_h), 100 * mean(l.λ_open_h ./ tot_h))
    @printf("    penalties : %.4f  (%.1f%%)\n", mean(l.λ_pen_h),  100 * mean(l.λ_pen_h ./ tot_h))
    @printf("    own goals : %.4f  (%.1f%%)\n", mean(l.λ_og_h),   100 * mean(l.λ_og_h ./ tot_h))
    @printf("    λ_total_h range : [%.4f, %.4f]\n", extrema(tot_h)...)
    @printf("    pxG (pre-finishing) range : [%.4f, %.4f]\n", extrema(l.pxg_h)...)
    demo_gate!("B1 all three channels contribute", mean(l.λ_open_h) > 0 &&
                                                    mean(l.λ_pen_h) > 0 &&
                                                    mean(l.λ_og_h) > 0)
end


# %% ===========================================================================
# 6. Defects reproduced, not fixed
# ==============================================================================
#
# Both are pre-existing `src` bugs found while wiring the recombination family. They
# are demonstrated here against the live method table and the live source, so the
# finding survives being read months later, and neither is fixed in this prototype:
# a container change that also silently repaired a sampler-facing extractor would make
# every parity table above incomparable to the production path it is validating.

demo_banner(6, "defects reproduced")

# --- Defect 1: DynamicPxGRecombModel's own OOS extractor cannot be called ------
println("  DEFECT 1  src/models/pregame/engines/team_level/time_decay/recombined_pxg.jl:191,214")
println("            `extract_dynamics(chain, config, n_teams)` — a 3-argument call.")
println("            Every method takes (chain, config, prefix::String, n_teams).")
println()
println("            live method table for PreGame.extract_dynamics:")
for m in methods(DEMO_PG.extract_dynamics)
    sig = m.sig.parameters[2:end]
    @printf("              (%s)\n", join(sig, ", "))
end

const DEMO_DEFECT1_HIT = try
    DEMO_PG.extract_parameters(DEMO_RECOMB_MODEL, DEMO_FIXTURES,
                               DEMO_RECOMB_FS, DEMO_RECOMB_CHAIN)
    nothing
catch err
    err
end

if DEMO_DEFECT1_HIT isa MethodError
    @printf("\n            calling it raises: MethodError for %s\n",
            DEMO_DEFECT1_HIT.f)
    println("            → `l02_extract.jl` §4 reimplements the body with the 4-arg call.")
elseif DEMO_DEFECT1_HIT === nothing
    println("\n            NOT REPRODUCED — the legacy extractor now runs. It may have")
    println("            been fixed; l02_extract.jl §4 should be re-checked against it.")
else
    @printf("\n            raised a different error: %s\n", typeof(DEMO_DEFECT1_HIT))
end
demo_gate!("F1 defect 1 recorded (MethodError, or a note that it is gone)",
           DEMO_DEFECT1_HIT isa MethodError || DEMO_DEFECT1_HIT === nothing)

# --- Defect 2: three extractors mis-size their output under multiple chains ----
println()
println("  DEFECT 2  ", tpl_multichain_warning())
let two_chain = tpl_synthetic_chain(DEMO_RECOMB_COLS;
                                    n_draws = DEMO_N_DRAWS, n_chains = 2, seed = DEMO_SEED + 3)
    flat = size(two_chain, 1) * size(two_chain, 3)
    got  = length(DEMO_PG.extract_recombination(two_chain, DEMO_RECOMB_MODEL.recomb_config).pen_conv)
    @printf("            flattened draws = %d, extract_recombination returned %d\n", flat, got)
    println("            → this runner uses ONE chain for the recombination family only.")
    println("              (Families A and C run on $(DEMO_N_CHAINS) chains and flatten correctly.)")
    demo_gate!("F2 defect 2 reproduced (single-chain sizing)", got == size(two_chain, 1))
end

# --- Defect 3: nine src call sites use a `haskey` MCMCChains 7.7 removed -------
println()
println("  DEFECT 3  `haskey(::Chains, ::Symbol)` is not defined in MCMCChains 7.7.0")
println("            (it was in 7.6). Nine live src call sites depend on it:")
for (file, lines) in (("src/models/pregame/components/squad_wealth.jl", "46"),
                      ("src/models/pregame/components/pxg_observation.jl", "34"),
                      ("src/models/pregame/components/recombination.jl", "66, 67, 68"),
                      ("src/models/pregame/engines/.../recombined_pxg.jl", "196, 239, 249"),
                      ("src/models/pregame/engines/.../recombined_goals.jl", "189"))
    @printf("              %-52s  %s\n", file, lines)
end
@printf("\n            installed MCMCChains : %s\n",
        pkgversion(MCMCChains) === nothing ? "unknown" : string(pkgversion(MCMCChains)))
@printf("            Project.toml allows  : %s\n", "MCMCChains = \"7.6, 7.7\"")
if tpl_haskey_is_shimmed()
    println("            → this run is using the compat shim in l02_extract.jl §0.")
    println("              Without it, extract_squad_wealth / extract_pxg_observation /")
    println("              extract_recombination — and both recombination engines — raise")
    println("              MethodError. Fix belongs in src or in the MCMCChains bound.")
else
    println("            → NOT REPRODUCED on this machine: MCMCChains supplies haskey.")
    println("              The nine call sites are still version-fragile.")
end
demo_gate!("F3 defect 3 recorded (haskey shim state reported)", true)


# %% ===========================================================================
# 7. Family C — SmileLatents
# ==============================================================================
#
# The family that MUST NOT be collapsed into `CountLatents`. A smile model prices O/U
# from `Λ(K) = λ_tot · φ(K)` and everything else from the grid; a container that
# carried only λ_h and λ_a would price O/U off the grid, produce entirely plausible
# numbers, and quietly delete the pillar the model exists for.

demo_banner(7, "family C — SmileLatents")

const DEMO_SMILE_MODEL = DEMO_PG.DynamicSmileDoublePoissonGoalsLeagueTimeDecayModel(
    smile_feature = BayesianFootball.Features.MarketSmileFeature(Kmax = DEMO_SMILE_KMAX),
)

function demo_smile_chain_columns(n_teams::Int, n_seasons::Int, n_leagues::Int, nK::Int)
    cols = String[]
    append!(cols, ["inter.μ_base[$i]" for i in 1:n_seasons])         # HierarchicalMonthly
    push!(cols, "inter.σ_month")
    append!(cols, ["inter.raw_month[$i]" for i in 1:12])
    push!(cols, "ha.γ_base", "ha.σ_γ")                                # HierarchicalTeamHA
    append!(cols, ["ha.γ_team_raw[$i]" for i in 1:n_teams])
    push!(cols, "dyn.σ_a", "dyn.σ_d")                                 # TimeDecayDynamics
    append!(cols, ["dyn.raw_a[$i]" for i in 1:n_teams])
    append!(cols, ["dyn.raw_d[$i]" for i in 1:n_teams])
    append!(cols, ["δ_league_raw[$i]" for i in 1:n_leagues])
    append!(cols, ["log_φ[$k]" for k in 1:nK])                        # the smile itself
    return cols
end

const DEMO_SMILE_NK   = DEMO_SMILE_KMAX + 1
const DEMO_SMILE_COLS = demo_smile_chain_columns(
    DEMO_N_TEAMS, DEMO_N_SEASONS, DEMO_N_LEAGUES, DEMO_SMILE_NK)

const DEMO_SMILE_CHAIN = tpl_synthetic_chain(DEMO_SMILE_COLS;
    n_draws = DEMO_N_DRAWS, n_chains = DEMO_N_CHAINS, seed = DEMO_SEED + 4)

const DEMO_SMILE_FS = tpl_feature_set(
    n_teams       = DEMO_N_TEAMS,
    n_seasons     = DEMO_N_SEASONS,
    n_leagues     = DEMO_N_LEAGUES,
    team_map      = DEMO_TEAM_MAP,
    league_lookup = DEMO_LEAGUE_LOOKUP,
    smile_Kmax    = DEMO_SMILE_KMAX,
)

const DEMO_SMILE_RAW = DEMO_PG.extract_parameters(
    DEMO_SMILE_MODEL, DEMO_FIXTURES, DEMO_SMILE_FS, DEMO_SMILE_CHAIN)
const DEMO_SMILE_LATENTS = extract_latents(
    DEMO_SMILE_MODEL, DEMO_SMILE_CHAIN, DEMO_FIXTURES, DEMO_SMILE_FS)
const DEMO_SMILE_LEGACY_DF = tpl_legacy_latents_df(DEMO_SMILE_RAW, DEMO_IDS)

println()
show(stdout, MIME"text/plain"(), DEMO_SMILE_LATENTS)
println()

@printf("\n  φ(K) posterior mean by strike:\n")
for (s, line) in enumerate(DEMO_SMILE_LATENTS.strikes)
    @printf("    K = %d  (line %.1f) : φ = %.5f   Λ = λ_tot·φ = %.4f\n",
            s - 1, line, mean(view(DEMO_SMILE_LATENTS.φ, :, s, :)),
            mean(DEMO_SMILE_LATENTS.λ_tot .* view(DEMO_SMILE_LATENTS.φ, :, s, :)))
end

# The NegBin smile: no `src` engine has one, so the container is built directly from
# the Poisson smile's intensities plus a synthetic dispersion panel. This exercises
# `SmileLatents{Float64,<:NamedTuple}` and its own grid kernel (which multiplies two
# `RobustNegativeBinomial` pdfs, where the plain NegBin kernel exps a sum of logs —
# l03 §5 explains why those must not be merged).
#
# Only the model IDENTITY matters here — it selects `Pred.extract_params` and
# `Pred.compute_score_matrix`, and neither reads a component config. The components are
# supplied because the constructor requires them, and are copied from
# `smile_negbin/r01_smoke.jl:165-178` so this instance is the one that prototype fits.
const DEMO_SMILE_NB_MODEL = DynamicSmileDoubleNegBinXGOutfieldPlayerTimeDecayModel(
    interception_config    = DEMO_PG.HierarchicalMonthlyInterception(),
    player_dynamics_config = DEMO_PG.OutfieldPlayerDynamicsConfig(days_half_life = 60.0),
    dispersion_config      = DEMO_PG.HomeAwayDispersion(),
    homeadvantage_config   = DEMO_PG.HierarchicalTeamHomeAdvantage(),
    kappa_config           = DEMO_PG.HierarchicalTeamKappa(),
    player_ratings_feature = BayesianFootball.Features.PlayerRatingsFeature(
                                 BayesianFootball.Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)),
    smile_feature          = BayesianFootball.Features.MarketSmileFeature(Kmax = DEMO_SMILE_KMAX),
)

const DEMO_SMILE_NB_LATENTS = let
    rng = MersenneTwister(DEMO_SEED + 5)
    nm, nd = n_matches(DEMO_SMILE_LATENTS), n_draws(DEMO_SMILE_LATENTS)
    r_h = 4.0 .+ 0.5 .* abs.(randn(rng, nm, nd))
    r_a = 5.0 .+ 0.5 .* abs.(randn(rng, nm, nd))
    SmileLatents(DEMO_SMILE_LATENTS.match_ids,
                 DEMO_SMILE_LATENTS.λ_home, DEMO_SMILE_LATENTS.λ_away,
                 (; r_h, r_a),
                 DEMO_SMILE_LATENTS.λ_tot, DEMO_SMILE_LATENTS.φ,
                 DEMO_SMILE_LATENTS.strikes)
end

const DEMO_SMILE_NB_LEGACY_DF = to_legacy_dataframe(DEMO_SMILE_NB_LATENTS)

println()
show(stdout, MIME"text/plain"(), DEMO_SMILE_NB_LATENTS)
println()

demo_gate!("C1 smile strikes are 0.5 .. $(DEMO_SMILE_KMAX + 0.5)",
           DEMO_SMILE_LATENTS.strikes == [(s - 1) + 0.5 for s in 1:DEMO_SMILE_NK])
demo_gate!("C2 smile family distinct from count family",
           observation_family(DEMO_SMILE_LATENTS) === :smile_poisson &&
           observation_family(DEMO_SMILE_NB_LATENTS) === :smile_negbin)
demo_gate!("C3 O/U 5.5 is outside the learned ladder (exercises grid fallback)",
           5 + 1 > DEMO_SMILE_NK)


# %% ===========================================================================
# 8. Parity — packing, score grids, grid mass
# ==============================================================================
#
# Three layers, in the order a failure would propagate:
#   8a  did the re-layout preserve every number?
#   8b  do the two grid kernels agree, bit for bit?
#   8c  is the grid a probability distribution at all? (a check both kernels could
#       fail together, which 8b by construction cannot detect)

demo_banner(8, "parity — packing and score grids")

const DEMO_CASES = (
    (name = "Poisson count", model = DEMO_POIS_MODEL,      latents = DEMO_POIS_LATENTS,
     legacy = DEMO_POIS_LEGACY_DF,   raw = DEMO_POIS_RAW,
     fields = [:λ_home => :λ_h, :λ_away => :λ_a]),
    (name = "NegBin count",  model = DEMO_NEGBIN_MODEL,    latents = DEMO_NEGBIN_LATENTS,
     legacy = DEMO_NEGBIN_LEGACY_DF, raw = DEMO_NEGBIN_RAW,
     fields = [:λ_home => :λ_h, :λ_away => :λ_a, :r_h => :r_h, :r_a => :r_a]),
    (name = "Smile Poisson", model = DEMO_SMILE_MODEL,     latents = DEMO_SMILE_LATENTS,
     legacy = DEMO_SMILE_LEGACY_DF,  raw = DEMO_SMILE_RAW,
     fields = [:λ_home => :λ_h, :λ_away => :λ_a, :λ_tot => :λ_tot]),
)

# --- 8a. packing ---------------------------------------------------------------
let rows = ParityRow[]
    for c in DEMO_CASES
        append!(rows, parity_packing(c.latents, c.raw, DEMO_IDS, c.fields; tol = DEMO_TOL))
    end
    demo_gate!("P1 packing is lossless",
               tpl_parity_table(rows; title = "8a. PACKING — typed matrix vs source sample vectors"))
end

# The smile's φ is packed by a different function (a 3-D transpose, not a 2-D copy),
# so it is checked separately against the engine's own (n_draws × n_strikes) matrices.
let rows = ParityRow[]
    typed  = Float64[]
    legacy = Float64[]
    for (i, id) in enumerate(DEMO_IDS)
        m = DEMO_SMILE_RAW[id].φ
        for k in 1:n_draws(DEMO_SMILE_LATENTS), s in 1:DEMO_SMILE_NK
            push!(typed,  DEMO_SMILE_LATENTS.φ[i, s, k])
            push!(legacy, m[k, s])
        end
    end
    push!(rows, tpl_compare("pack φ (n_m × n_K × n_d) <- (n_d × n_K)", typed, legacy; tol = DEMO_TOL))
    demo_gate!("P2 smile φ transpose is lossless",
               tpl_parity_table(rows; title = "8a'. PACKING — smile shape curve"))
end

# --- 8b. score grids -----------------------------------------------------------
for c in DEMO_CASES
    rows = parity_score_grids(c.model, c.latents, c.legacy;
                              max_goals = DEMO_MAX_GOALS, tol = DEMO_TOL)
    demo_gate!("P3 score grid parity — $(c.name)",
               tpl_parity_table(rows; title = "8b. SCORE GRID — $(c.name)"))
end

# The recombination family's legacy side takes the CHANNEL branch of
# `extract_params`, so this row compares two independent recombinations of the same
# components rather than a copy of one total.
let rows = parity_score_grids(DEMO_RECOMB_MODEL, DEMO_RECOMB_LATENTS, DEMO_RECOMB_LEGACY_DF;
                              max_goals = DEMO_MAX_GOALS, tol = DEMO_TOL)
    demo_gate!("P3 score grid parity — Recombination",
               tpl_parity_table(rows; title = "8b. SCORE GRID — Recombination (channel branch)"))
end

let rows = parity_score_grids(DEMO_SMILE_NB_MODEL, DEMO_SMILE_NB_LATENTS, DEMO_SMILE_NB_LEGACY_DF;
                              max_goals = DEMO_MAX_GOALS, tol = DEMO_TOL)
    demo_gate!("P3 score grid parity — Smile NegBin",
               tpl_parity_table(rows; title = "8b. SCORE GRID — Smile NegBin"))
end

# --- 8c. grid mass -------------------------------------------------------------
let rows = ParityRow[]
    for (name, l) in (("Poisson count", DEMO_POIS_LATENTS),
                      ("NegBin count",  DEMO_NEGBIN_LATENTS),
                      ("Recombination", DEMO_RECOMB_LATENTS),
                      ("Smile Poisson", DEMO_SMILE_LATENTS),
                      ("Smile NegBin",  DEMO_SMILE_NB_LATENTS))
        r = parity_grid_mass(l; max_goals = DEMO_MAX_GOALS)[1]
        push!(rows, ParityRow("$name  $(r.check)", r.n, r.max_abs, r.max_ulp,
                              r.tol, r.ulp_budget, r.pass))
    end
    demo_gate!("P4 every grid is a probability distribution",
               tpl_parity_table(rows; title = "8c. GRID MASS — independent of the legacy path"))
end

# --- 8d. does the harness detect a wrong answer? --------------------------------
#
# A parity table full of zeros is only evidence if a NON-zero is reachable. This
# perturbs one λ of one fixture by ONE ULP — the smallest change representable — and
# requires that exactly that fixture's row goes red and no other's does.
#
# It is also the check that caught a real weakness in this harness. Gating only on the
# briefing's |Δ| < 1e-12 threshold, this mutation reported max |Δ| = 1.4e-17 and PASSED,
# because sixteen units in the last place of a score-grid cell are four orders of
# magnitude inside that tolerance. The ULP budget in `ParityRow` exists because of this
# section, and this section stays so it cannot silently regress.

let l = DEMO_POIS_LATENTS
    perturbed = CountLatents(l.match_ids, copy(l.λ_home), copy(l.λ_away), nothing)
    target_i, target_k = 2, 3
    before = perturbed.λ_home[target_i, target_k]
    perturbed.λ_home[target_i, target_k] = nextfloat(before)

    rows = parity_score_grids(DEMO_POIS_MODEL, perturbed, DEMO_POIS_LEGACY_DF;
                              max_goals = DEMO_MAX_GOALS, tol = DEMO_TOL)
    @printf("\n  mutation: λ_home[%d,%d]  %.17g → %.17g  (1 ULP)\n",
            target_i, target_k, before, perturbed.λ_home[target_i, target_k])

    tpl_parity_table(rows; title = "8d. NEGATIVE CONTROL — one 1-ULP mutation (fixture 2 MUST fail)")

    only_target_failed = !rows[target_i].pass &&
                         all(rows[i].pass for i in eachindex(rows) if i != target_i)
    would_pass_on_tol_alone = rows[target_i].max_abs <= DEMO_TOL

    @printf("\n  the mutated fixture's max |Δ| is %.3e, which is %s the briefing's %.0e\n",
            rows[target_i].max_abs,
            would_pass_on_tol_alone ? "INSIDE" : "outside", DEMO_TOL)
    println("  → a tolerance-only gate would have called this a pass. The ULP budget is")
    println("    what makes every green row above mean bit-identity.")

    demo_gate!("P4b harness fails the mutated fixture and only that one", only_target_failed)
end

# --- 8e. the migration bridge and the allocating wrappers -----------------------
#
# `latents_from_legacy_dataframe` is the function that makes adoption possible: there
# are folds on disk that cost hours of NUTS, and a migration requiring them to be
# refitted would not be taken. It is exercised here on the REAL legacy frames — the
# `Vector{Any}` ones built the way `src` builds them — not on a tidied-up round trip.
#
# The allocating wrappers (`compute_score_grid`, `compute_score_grid(::SmileLatents)`)
# are checked against their own bang forms in the same breath, because a convenience
# form that quietly disagreed with the kernel it wraps would be worse than not having
# one.

let rows = ParityRow[]
    for c in DEMO_CASES
        # Legacy `oos_latents.jls` → typed container, no chain involved.
        from_df = latents_from_legacy_dataframe(c.model, c.legacy)
        push!(rows, tpl_compare("from legacy df: λ_home  [$(c.name)]",
                                from_df.λ_home, c.latents.λ_home; tol = DEMO_TOL))
        push!(rows, tpl_compare("from legacy df: λ_away  [$(c.name)]",
                                from_df.λ_away, c.latents.λ_away; tol = DEMO_TOL))

        # Typed container → legacy frame → back again.
        round_trip = latents_from_legacy_dataframe(c.model, to_legacy_dataframe(c.latents))
        push!(rows, tpl_compare("round trip:     λ_home  [$(c.name)]",
                                round_trip.λ_home, c.latents.λ_home; tol = DEMO_TOL))
    end

    # Dispersion and the smile curve are the two parts a naive reader would drop, so
    # they are asserted explicitly rather than left to the λ rows above.
    nb = latents_from_legacy_dataframe(DEMO_NEGBIN_MODEL, DEMO_NEGBIN_LEGACY_DF)
    push!(rows, tpl_compare("from legacy df: r_h     [NegBin count]",
                            nb.observation_params.r_h,
                            DEMO_NEGBIN_LATENTS.observation_params.r_h; tol = DEMO_TOL))

    sm = latents_from_legacy_dataframe(DEMO_SMILE_MODEL, DEMO_SMILE_LEGACY_DF)
    push!(rows, tpl_compare("from legacy df: φ       [Smile Poisson]",
                            sm.φ, DEMO_SMILE_LATENTS.φ; tol = DEMO_TOL))

    # Allocating wrapper vs the bang form it wraps.
    ws = GridWorkspace(DEMO_MAX_GOALS)
    S  = alloc_score_grid(DEMO_POIS_LATENTS, DEMO_MAX_GOALS)
    compute_score_grid!(S, ws, DEMO_POIS_LATENTS, 5)
    push!(rows, tpl_compare("compute_score_grid vs bang form",
                            compute_score_grid(DEMO_POIS_LATENTS, 5; max_goals = DEMO_MAX_GOALS),
                            S; tol = DEMO_TOL))

    Ssm = alloc_score_grid(DEMO_SMILE_LATENTS, DEMO_MAX_GOALS)
    compute_score_grid!(Ssm, ws, DEMO_SMILE_LATENTS, 5)
    smile_g = compute_score_grid(DEMO_SMILE_LATENTS, 5; max_goals = DEMO_MAX_GOALS)
    push!(rows, tpl_compare("SmileScoreGrid.grid vs bang form", smile_g.grid, Ssm; tol = DEMO_TOL))
    push!(rows, tpl_compare("SmileScoreGrid.λ_tot vs container", smile_g.λ_tot,
                            [DEMO_SMILE_LATENTS.λ_tot[5, k]
                             for k in 1:n_draws(DEMO_SMILE_LATENTS)]; tol = DEMO_TOL))

    demo_gate!("P6 migration bridge and allocating wrappers agree",
               tpl_parity_table(rows; title = "8e. MIGRATION BRIDGE — legacy df ⇄ typed container"))
end

# The recombination family is the one that CANNOT come back from a legacy frame, and the
# bridge has to say so rather than guessing. `λ_total − λ_open = q_pen·λ_pen + og_rate`
# is one equation in two unknowns; any reader that returned a container anyway would be
# putting own goals in the penalty channel and nothing downstream would notice.
let refusal = try
        latents_from_legacy_dataframe(DEMO_RECOMB_MODEL, to_legacy_dataframe(DEMO_RECOMB_LATENTS))
        nothing
    catch err
        err
    end
    println("\n  recombination round trip is refused, with a reason:")
    println("    ", refusal === nothing ? "NOT REFUSED — the bridge silently guessed a split." :
                    first(split(sprint(showerror, refusal), "\n")))
    demo_gate!("P8 recombination refuses a lossy round trip", refusal isa ErrorException)
end

# `match_index` is the only id→row lookup the container offers, and every consumer that
# resolves a fixture by id depends on it. Cheap to check, expensive to have wrong.
let l = DEMO_POIS_LATENTS
    resolves   = all(match_index(l, l.match_ids[i]) == i for i in 1:n_matches(l))
    absent_is0 = match_index(l, -1) == 0
    demo_gate!("P7 match_index resolves every id, and 0 for an absent one",
               resolves && absent_is0)
    @printf("\n  match_index: %d/%d ids resolve to their own row; unknown id → %d\n",
            count(i -> match_index(l, l.match_ids[i]) == i, 1:n_matches(l)),
            n_matches(l), match_index(l, -1))
end


# %% ===========================================================================
# 9. Parity — market prices
# ==============================================================================
#
# The end of the chain, and the only numbers that ever get staked. Compared per
# (market, outcome) so a failure names the selection.

demo_banner(9, "parity — market prices")

for c in DEMO_CASES
    rows = parity_market_prices(c.model, c.latents, c.legacy, DEMO_MARKETS;
                                max_goals = DEMO_MAX_GOALS, tol = DEMO_TOL)
    demo_gate!("P5 market parity — $(c.name)",
               tpl_parity_table(rows; title = "9. MARKETS — $(c.name)"))
end

let rows = parity_market_prices(DEMO_RECOMB_MODEL, DEMO_RECOMB_LATENTS,
                                DEMO_RECOMB_LEGACY_DF, DEMO_MARKETS;
                                max_goals = DEMO_MAX_GOALS, tol = DEMO_TOL)
    demo_gate!("P5 market parity — Recombination",
               tpl_parity_table(rows; title = "9. MARKETS — Recombination"))
end

let rows = parity_market_prices(DEMO_SMILE_NB_MODEL, DEMO_SMILE_NB_LATENTS,
                                DEMO_SMILE_NB_LEGACY_DF, DEMO_MARKETS;
                                max_goals = DEMO_MAX_GOALS, tol = DEMO_TOL)
    demo_gate!("P5 market parity — Smile NegBin",
               tpl_parity_table(rows; title = "9. MARKETS — Smile NegBin"))
end

# The smile route is only being tested if it is actually TAKEN. If λ_tot·φ(K) happened
# to equal what the grid produces, every O/U row above would pass without the smile
# ever being consulted. This shows the two routes disagree — which is the model's
# entire point — and that the out-of-ladder line falls back to the grid exactly.
let l = DEMO_SMILE_LATENTS
    ws = GridWorkspace(DEMO_MAX_GOALS)
    S  = alloc_score_grid(l, DEMO_MAX_GOALS)
    g  = _tpl_typed_container(l, S, ws, 1)
    println("\n  smile route vs grid route, fixture $(l.match_ids[1]), mean P(over):")
    @printf("    %6s %14s %14s %12s\n", "line", "smile route", "grid route", "difference")
    for line in (1.5, 2.5, 3.5, 5.5)
        m = DEMO_D.MarketOverUnder(line)
        smile_p = mean(price_market(g, m)[DEMO_D.outcomes(m).over])
        grid_p  = mean(price_market(g.grid, m)[DEMO_D.outcomes(m).over])
        tag = line > DEMO_SMILE_KMAX + 0.5 ? "  (fallback)" : ""
        @printf("    %6.1f %14.6f %14.6f %12.2e%s\n", line, smile_p, grid_p, smile_p - grid_p, tag)
    end
    m55 = DEMO_D.MarketOverUnder(5.5)
    demo_gate!("C4 out-of-ladder line falls back to the grid exactly",
               price_market(g, m55)[DEMO_D.outcomes(m55).over] ==
               price_market(g.grid, m55)[DEMO_D.outcomes(m55).over])
    m25 = DEMO_D.MarketOverUnder(2.5)
    demo_gate!("C5 in-ladder line is priced by the smile, not the grid",
               price_market(g, m25)[DEMO_D.outcomes(m25).over] !=
               price_market(g.grid, m25)[DEMO_D.outcomes(m25).over])
end


# %% ===========================================================================
# 10. Allocation audit
# ==============================================================================
#
# The steady-state claim: with a caller-owned destination grid and workspace, the
# score-grid and market-pricing kernels allocate ZERO bytes, for every family.
#
# The `baseline` row measures an empty closure through the identical `@allocated`
# path. It must also read 0; if it does not, the measurement harness is contributing
# and every other row in the table is suspect.

demo_banner(10, "allocation audit")

for (name, l) in (("Poisson count", DEMO_POIS_LATENTS),
                  ("NegBin count",  DEMO_NEGBIN_LATENTS),
                  ("Recombination", DEMO_RECOMB_LATENTS),
                  ("Smile Poisson", DEMO_SMILE_LATENTS),
                  ("Smile NegBin",  DEMO_SMILE_NB_LATENTS))
    rows = allocation_audit(l; max_goals = DEMO_MAX_GOALS, markets = DEMO_MARKETS)
    demo_gate!("Z1 zero allocations — $name",
               tpl_alloc_table(rows; title = "10. ALLOCATIONS — $name"))
end


# %% ===========================================================================
# 11. Memory and timing
# ==============================================================================
#
# Not a benchmark suite; two questions only. What does the typed layout cost in memory,
# and is the typed kernel at least as fast as the one it replaces?
#
# READ THE OBJECT COLUMN, NOT THE BYTE COLUMN. The byte ratio goes BOTH WAYS below, and
# the reasons are worth knowing before the table rather than after:
#
#   * The legacy frames carry `true_xg_h`/`true_xg_a`, which the count engines set to
#     the SAME OBJECTS as `λ_h`/`λ_a` (`_cb_rates`,
#     05_composable_count_builder/l03_engine.jl:386-387). Counted once, as
#     `tpl_dataframe_bytes` does, they are free; the typed container simply drops them.
#
#   * Conversely, the legacy frames SHARE one dispersion vector, and one smile `φ`
#     matrix, across every fixture. The typed container materialises both per fixture,
#     so it is genuinely LARGER for the NegBin and smile families. `l01_latents.jl` §3
#     and §5 explain why that is the right trade and what it buys; this table is where
#     the bill arrives.
#
# The object count is the number that does not move: two to four heap objects, whatever
# the fold size, against `n_matches × n_parameters` boxed vectors.
#
# The timing comparison is deliberately conservative. The legacy side pays its
# `Vector{Any}` unboxing — which is the cost being removed — but BOTH sides allocate a
# fresh score tensor per fixture, because the legacy kernel has no in-place form to
# compare against. The typed path's other advantage, reusing one destination grid
# across a whole fold, is therefore NOT in these numbers.

demo_banner(11, "memory and timing")

memory_comparison("Poisson count", DEMO_POIS_LATENTS,   DEMO_POIS_LEGACY_DF)
memory_comparison("NegBin count",  DEMO_NEGBIN_LATENTS, DEMO_NEGBIN_LEGACY_DF)
memory_comparison("Smile Poisson", DEMO_SMILE_LATENTS,  DEMO_SMILE_LEGACY_DF)

let rows = TimingRow[]
    n  = DEMO_N_MATCHES
    ws = GridWorkspace(DEMO_MAX_GOALS)
    S  = alloc_score_grid(DEMO_POIS_LATENTS, DEMO_MAX_GOALS)

    push!(rows, tpl_time("legacy  grid  (Poisson)", n, function ()
        for i in 1:n
            legacy_score_tensor(DEMO_POIS_MODEL, DEMO_POIS_LEGACY_DF, i;
                                max_goals = DEMO_MAX_GOALS)
        end
    end))
    push!(rows, tpl_time("typed   grid  (Poisson)", n, function ()
        for i in 1:n
            compute_score_grid!(S, ws, DEMO_POIS_LATENTS, i)
        end
    end))

    Snb = alloc_score_grid(DEMO_NEGBIN_LATENTS, DEMO_MAX_GOALS)
    push!(rows, tpl_time("legacy  grid  (NegBin)", n, function ()
        for i in 1:n
            legacy_score_tensor(DEMO_NEGBIN_MODEL, DEMO_NEGBIN_LEGACY_DF, i;
                                max_goals = DEMO_MAX_GOALS)
        end
    end))
    push!(rows, tpl_time("typed   grid  (NegBin)", n, function ()
        for i in 1:n
            compute_score_grid!(Snb, ws, DEMO_NEGBIN_LATENTS, i)
        end
    end))

    m1x2 = DEMO_D.Market1X2()
    book = alloc_market_book(m1x2, n_draws(DEMO_POIS_LATENTS))
    compute_score_grid!(S, ws, DEMO_POIS_LATENTS, 1)
    legacy_S = BayesianFootball.Predictions.ScoreMatrix(copy(S))
    push!(rows, tpl_time("legacy  price 1X2", n, function ()
        for _ in 1:n
            BayesianFootball.Predictions.compute_market_probs(legacy_S, m1x2)
        end
    end))
    push!(rows, tpl_time("typed   price 1X2", n, function ()
        for _ in 1:n
            price_market!(book, S, m1x2)
        end
    end))

    tpl_timing_table(rows; title = "11. TIMING  ($(n) fixtures × $(n_draws(DEMO_POIS_LATENTS)) draws)")
end


# %% ===========================================================================
# 12. Final report
# ==============================================================================

demo_banner(12, "final report")

let width = maximum(length(first(g)) for g in DEMO_GATES; init = 20)
    all_ok = true
    for (name, ok) in DEMO_GATES
        ok || (all_ok = false)
        @printf("  %-*s  %s\n", width, name, ok ? "pass" : "FAIL")
    end
    println("  ", "-"^(width + 8))
    @printf("  %-*s  %d / %d\n", width, "gates passed",
            count(last, DEMO_GATES), length(DEMO_GATES))
    println()

    if all_ok
        println("  RESULT: PASS")
        println()
        println("  Every price computed from a typed container is bit-identical (0 ULP) to")
        println("  the price the live `src` kernels compute from the equivalent legacy")
        println("  latents.df, across all five model families, six markets, and every")
        println("  posterior draw. The steady-state kernels allocate zero bytes.")
        println()
        println("  NOT SHOWN, and not claimed: that any of these models fits anything.")
        println("  The posteriors here are prior draws with a fixed seed (l04 §9).")
    else
        println("  RESULT: FAIL — see the failing gates above.")
    end

    # Non-zero exit so this runner is usable as a CI check, but only when run as a
    # script: an `include` from a REPL should not kill the session.
    if abspath(PROGRAM_FILE) == @__FILE__
        exit(all_ok ? 0 : 1)
    end
end
