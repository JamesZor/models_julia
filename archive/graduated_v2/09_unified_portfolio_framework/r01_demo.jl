# ==============================================================================
# 09 — UNIFIED PORTFOLIO & STAKING FRAMEWORK : THE PROOF
# ==============================================================================
#
# WHAT THIS IS
#   A deterministic verification that the rebuilt staking pipeline produces the same
#   numbers as `src/Portfolio/`, that its hot path allocates nothing, that convergence
#   gating actually gates, and that every legacy portfolio call site keeps working.
#
#   Nine things are established, in this order:
#
#     §4   the odds index and the fixture table are the legacy ones, exactly — the
#          same rows in the same order, and `fixture_table(ds)` is `==` to `src`'s
#     §5   the hot scoring path allocates ZERO BYTES, for two container families and
#          every market in the spec, against an empty-closure baseline that reads 0
#     §6   mathematical parity against the live `BayesianFootball.Portfolio` builder
#          at 0 ULP — score grids, selections, prices, payoff matrices, settlement
#          vectors, Kelly allocations, shrinkage factors and KKT residuals — plus a
#          negative control sized to fire
#     §7   multi-slate staking and simulation: the bankroll identities hold, the
#          result is reproducible, `initial_bankroll` is a pure scale, and evaluating
#          never perturbs the caller's RNG
#     §8   the trajectory, the path metrics and the bootstrap interval are IDENTICAL
#          to `Portfolio.simulate`, `path_metrics` and `bootstrap_roi`
#     §9   the convergence gate REFUSES an unconverged fit by default, BUILDS and
#          FLAGS it on request, treats an unaudited container as unconverged, and
#          changes no arithmetic when it admits a build
#     §10  a legacy call site — its body copied verbatim from a real runner — runs
#          unmodified, including through a serialisation round trip
#     §11  the four causes of a dropped fixture, separated and named, where `src`
#          returns `nothing` for all of them
#     §12  the cost, measured rather than asserted
#
# WHAT THIS IS NOT
#   Not a model-fitting run and not a betting study. No MCMC is run and no database is
#   touched. Chains are prior draws with a fixed seed (`06/l04_parity.jl` §9); the odds
#   are the model's own prices, perturbed and vigged (`08/l05_parity.jl` §6), with one
#   deliberate distortion so the price policy is exercised (`l05_parity.jl` §6).
#
#   NOTHING HERE SAYS ANY STRATEGY MAKES MONEY. The synthetic market is built FROM the
#   model, so the model beats it by construction and every positive ROI below is an
#   artefact of the fixture. On the only real out-of-sample evaluation this repository
#   has — ScottishLower, 628 matches — the default policy returns a flat ROI whose
#   match-clustered 95% interval INCLUDES ZERO.
#
# THE COMPARABILITY CONTRACT
#   Every parity row compares TWO BUILDERS OVER ONE SET OF NUMBERS. The legacy side is
#   the live `src` path — real `extract_params`, real `compute_score_matrix`, real
#   `compute_market_probs`, real `Portfolio.extract_selections`, real
#   `Portfolio.allocate`, real `Portfolio.simulate` — fed a `DataFrame` built from the
#   same typed container the new side reads. Nothing is transcribed.
#
# WHAT WOULD FALSIFY THE CLAIMS
#   Any parity row above 0 ULP. Any non-zero byte count in §5. A converged fit refused
#   or an unconverged one admitted in §9. Any legacy expression in §10 raising, or
#   returning a different book.
#
# USAGE
#   julia --project current_development/09_unified_portfolio_framework/r01_demo.jl
#
#   or, in a REPL:
#     include("current_development/09_unified_portfolio_framework/r01_demo.jl")
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
using Printf
using Random
using Serialization
using Statistics

# The thing under test. `l04_compat_bridge.jl` opens `module UnifiedPortfolio`, whose
# include chain pulls in l03 → l02 → l01 and, through l01, the whole of
# `08_unified_evaluation_framework`, `07_unified_inference_framework`,
# `06_typed_posterior_latents` and `05_composable_count_builder`.
include(joinpath(@__DIR__, "l04_compat_bridge.jl"))

const UP = UnifiedPortfolio
const UE = UnifiedPortfolio.UnifiedEvaluation      # `05`–`08`, for the fixtures
const BF = BayesianFootball
const PF = BayesianFootball.Portfolio              # the legacy side, named explicitly
const D  = BayesianFootball.Data
const PG = BayesianFootball.Models.PreGame

using .UnifiedPortfolio


# %% ===========================================================================
# 2. Configuration
# ==============================================================================

const DEMO_N_TEAMS   = 8
const DEMO_N_FOLDS   = 3
const DEMO_PER_FOLD  = 8
const DEMO_N_MATCHES = DEMO_N_FOLDS * DEMO_PER_FOLD     # 24 held-out fixtures
const DEMO_N_DRAWS   = 300                              # posterior draws PER CHAIN
const DEMO_N_CHAINS  = 4                                # exercises the flattening
const DEMO_TOTAL     = DEMO_N_DRAWS * DEMO_N_CHAINS     # 1200
const DEMO_SEED      = 20240808

# 8 fixtures per settlement window, 7 days apart. The slate size is load-bearing: the
# drawdown budget is solved across everything settling together and the exposure cap
# bounds their total, and both are trivial when a slate holds one match.
const DEMO_PER_SLATE = 8
const DEMO_GAP_DAYS  = 7

const DEMO_MARKETS = D.AbstractMarket[
    D.Market1X2(), D.MarketBTTS(),
    D.MarketOverUnder(1.5), D.MarketOverUnder(2.5), D.MarketOverUnder(3.5),
]

# Every gate's verdict lands here; §13 is the only place that decides pass/fail.
const DEMO_GATES = Pair{String, Bool}[]
demo_gate!(name, ok) = (push!(DEMO_GATES, name => Bool(ok)); Bool(ok))

function demo_checks!(prefix, checks)
    all_ok = true
    for (k, v) in checks
        demo_gate!("$prefix $k", v)
        v || (all_ok = false)
        @printf("    %-6s %s\n", v ? "ok" : "FAIL", k)
    end
    return all_ok
end

function demo_banner(n, title)
    println()
    println("=" ^ 78)
    @printf("%d. %s\n", n, uppercase(title))
    println("=" ^ 78)
end


# %% ===========================================================================
# 3. Deterministic posteriors, fits and a synthetic DataStore
# ==============================================================================
#
# Two engines with the same linear predictor and different observation layers, so the
# score-grid kernel reaches two different families BY DISPATCH — a
# `CountLatents{Float64, Nothing}` goes to the double-Poisson grid and a
# `CountLatents{Float64, <:NamedTuple}` to the double-negative-binomial one, with no
# runtime branch. `HomeAwayDispersion` rather than `GlobalDispersion`, so `r_h ≠ r_a`
# and a kernel that silently used one for both sides fails §6 rather than passing by
# coincidence.
#
# A THIRD fit is built with a per-chain offset. Between-chain variance with no
# within-chain justification is exactly what R-hat measures, so it fails the gate while
# remaining a perfectly well-formed `Chains` — that is §9's subject.

demo_banner(3, "deterministic posteriors, fits and a synthetic DataStore")

function demo_poisson_model()
    b = UE.CountModelBuilder(:up_demo_poisson)
    UE.add!(b, PG.GlobalInterception(),
               PG.TimeDecayDynamics(days_half_life = 365.0),
               PG.GlobalHomeAdvantage())
    UE.add!(b, UE.WealthCovariate())
    return UE.build(b)
end

function demo_negbin_model()
    b = UE.CountModelBuilder(:up_demo_negbin)
    UE.add!(b, PG.GlobalInterception(),
               PG.TimeDecayDynamics(days_half_life = 365.0),
               PG.GlobalHomeAdvantage())
    UE.add!(b, UE.WealthCovariate())
    UE.add!(b, PG.HomeAwayDispersion())
    return UE.build(b)
end

const DEMO_POIS_MODEL   = demo_poisson_model()
const DEMO_NEGBIN_MODEL = demo_negbin_model()

println("  Poisson engine : ", DEMO_POIS_MODEL)
println("  NegBin  engine : ", DEMO_NEGBIN_MODEL)

const DEMO_FIXTURES = UE.tpl_synthetic_fixtures(DEMO_N_MATCHES;
                                                n_teams = DEMO_N_TEAMS, seed = DEMO_SEED)
const DEMO_TEAM_MAP = UE.tpl_team_map(DEMO_N_TEAMS)
const DEMO_FS  = UE.tpl_feature_set(n_teams = DEMO_N_TEAMS, n_seasons = 2,
                                    team_map = DEMO_TEAM_MAP)
const DEMO_OOS = [DEMO_FIXTURES[((k - 1) * DEMO_PER_FOLD + 1):(k * DEMO_PER_FOLD), :]
                  for k in 1:DEMO_N_FOLDS]
const DEMO_METAS = [D.SplitMetaData(1, "23/24", "24/25", 2, k, 0) for k in 1:DEMO_N_FOLDS]
const DEMO_FEATURE_SETS = [(DEMO_FS, DEMO_METAS[k]) for k in 1:DEMO_N_FOLDS]
const DEMO_SPLITTER = D.CVConfig(tournament_ids = [1], target_seasons = ["24/25"],
                                 history_seasons = 0, warmup_period = 1,
                                 dynamics_col = :match_week)

function demo_fit(name, model, cols; seed, chain_offset = 0.0)
    chs = [UE.demo_nuts_chain(cols; n_draws = DEMO_N_DRAWS, n_chains = DEMO_N_CHAINS,
                              seed = seed + k, chain_offset = chain_offset)
           for k in 1:DEMO_N_FOLDS]
    cfg = UE.FitConfig(name = name, model = model, splitter = DEMO_SPLITTER,
                       sampler = UE.ReplaySampler(chs), save_dir = tempdir())
    return UE.fit_model(cfg; feature_sets = DEMO_FEATURE_SETS,
                        oos_fixtures = DEMO_OOS, quiet = true)
end

const DEMO_POIS_FIT = demo_fit("poisson_baseline", DEMO_POIS_MODEL,
                               UE.cb_chain_columns(DEMO_POIS_MODEL, DEMO_N_TEAMS);
                               seed = DEMO_SEED)
const DEMO_NEGBIN_FIT = demo_fit("negbin_dispersed", DEMO_NEGBIN_MODEL,
                                 UE.cb_chain_columns(DEMO_NEGBIN_MODEL, DEMO_N_TEAMS);
                                 seed = DEMO_SEED + 100)
const DEMO_BROKEN_FIT = demo_fit("poisson_unconverged", DEMO_POIS_MODEL,
                                 UE.cb_chain_columns(DEMO_POIS_MODEL, DEMO_N_TEAMS);
                                 seed = DEMO_SEED + 200, chain_offset = 3.0)

const DEMO_POIS_L   = DEMO_POIS_FIT.latents
const DEMO_NEGBIN_L = DEMO_NEGBIN_FIT.latents

@printf("\n  fixtures        : %d over %d teams, %d folds × %d\n",
        DEMO_N_MATCHES, DEMO_N_TEAMS, DEMO_N_FOLDS, DEMO_PER_FOLD)
@printf("  posterior draws : %d × %d chains = %d flattened\n",
        DEMO_N_DRAWS, DEMO_N_CHAINS, DEMO_TOTAL)
@printf("  Poisson latents : %s\n", typeof(DEMO_POIS_L))
@printf("  NegBin  latents : %s\n", typeof(DEMO_NEGBIN_L))
@printf("  convergence     : %s / %s / %s\n",
        DEMO_POIS_FIT.diagnostics.passed ? "PASS" : "FAIL",
        DEMO_NEGBIN_FIT.diagnostics.passed ? "PASS" : "FAIL",
        DEMO_BROKEN_FIT.diagnostics.passed ? "PASS" : "FAIL")

demo_gate!("3a Poisson fit yields CountLatents{Float64,Nothing}",
           DEMO_POIS_L isa UP.CountLatents{Float64, Nothing})
demo_gate!("3b NegBin fit yields CountLatents{Float64,<:NamedTuple}",
           DEMO_NEGBIN_L isa UP.CountLatents{Float64, <:NamedTuple})
demo_gate!("3c r_h ≠ r_a — HomeAwayDispersion is genuinely asymmetric",
           DEMO_NEGBIN_L.observation_params.r_h != DEMO_NEGBIN_L.observation_params.r_a)
demo_gate!("3d both containers cover the same fixtures",
           UP.latent_match_ids(DEMO_POIS_L) == UP.latent_match_ids(DEMO_NEGBIN_L))
demo_gate!("3e the two baseline fits converged", DEMO_POIS_FIT.diagnostics.passed &&
                                                 DEMO_NEGBIN_FIT.diagnostics.passed)
demo_gate!("3f the offset-chain fit did NOT converge", !DEMO_BROKEN_FIT.diagnostics.passed)

# The store. Scores are simulated from the Poisson container's own marginals; the odds
# are the model's own prices, perturbed and vigged; every fourth match is quoted at an
# overround BELOW one so `DeArb` is exercised rather than being the identity everywhere.
const DEMO_DS, DEMO_SCORES = UP.portfolio_datastore(
    DEMO_POIS_L, DEMO_MARKETS; seed = DEMO_SEED + 1, fixtures = DEMO_FIXTURES,
    per_slate = DEMO_PER_SLATE, gap_days = DEMO_GAP_DAYS, thin_every = 4)

const DEMO_OVERROUNDS = collect(skipmissing(DEMO_DS.odds.overround_close))

@printf("\n  ds.matches      : %d fixtures over %d settlement windows\n",
        nrow(DEMO_DS.matches), length(unique(DEMO_DS.matches.match_date)))
@printf("  ds.odds         : %d quotes, %d selections, %d markets\n",
        nrow(DEMO_DS.odds), length(unique(DEMO_DS.odds.selection)), length(DEMO_MARKETS))
@printf("  overround       : %.4f … %.4f  (%.0f%% below 1.0 — DeArb bites there)\n",
        minimum(DEMO_OVERROUNDS), maximum(DEMO_OVERROUNDS),
        100 * mean(DEMO_OVERROUNDS .< 1.0))

demo_gate!("3g fixtures cluster into multi-match settlement windows",
           length(unique(DEMO_DS.matches.match_date)) == DEMO_N_FOLDS &&
           all(d -> count(==(d), DEMO_DS.matches.match_date) == DEMO_PER_SLATE,
               unique(DEMO_DS.matches.match_date)))
demo_gate!("3h every fixture is quoted in every market group",
           nrow(DEMO_DS.odds) == DEMO_N_MATCHES * sum(UP.market_arity, DEMO_MARKETS))
demo_gate!("3i the store carries books on BOTH sides of overround 1.0",
           any(DEMO_OVERROUNDS .< 1.0) && any(DEMO_OVERROUNDS .> 1.0))


# %% ===========================================================================
# 3b. The two specs and the policy
# ==============================================================================
#
# TWO specs, because they measure different things:
#
#   `DEMO_SPEC`    BakerMcHale — the production default. Its 16 re-solves of the
#                  allocator per fixture dominate both the time and the heap, which is
#                  why §12's whole-build row understates the container change.
#   `DEMO_SPEC_FK` FractionalKelly — a constant. Removes that term, so §12's second row
#                  shows what the build itself costs.
#
# `BakerMcHale(n_draws = 16)` rather than the default 128: 16 is enough to make the
# factor non-trivial and this runner is meant to finish in seconds. The parity claim is
# unaffected — both builders are handed the same object.

const DEMO_EXEC = UP.ExecutionConfig(commission = UP.PerBetCommission(0.02))

const DEMO_SPEC = UP.BookSpec(markets   = D.MarketConfig(DEMO_MARKETS),
                              price     = UP.DeArb(),
                              allocator = UP.KellyLogUtility(),
                              shrink    = UP.BakerMcHale(n_draws = 16),
                              exec      = DEMO_EXEC)

const DEMO_SPEC_FK = UP.BookSpec(markets   = D.MarketConfig(DEMO_MARKETS),
                                 price     = UP.DeArb(),
                                 allocator = UP.KellyLogUtility(),
                                 shrink    = UP.FractionalKelly(0.25),
                                 exec      = DEMO_EXEC)

const DEMO_POLICY = UP.PolicySpec(trust    = UP.FlatTrust(0.25),
                                  risk     = UP.SlateDrawdown(23.0),
                                  cap      = UP.FixedCap(0.25),
                                  filter   = UP.KeepAll(),
                                  grouping = UP.DailySlate())

const DEMO_SYSTEM = UP.PortfolioSystem(DEMO_SPEC, DEMO_POLICY)


# %% ===========================================================================
# 4. The odds index and the fixture table
# ==============================================================================
#
# These two replace the per-fixture full-frame scans in `extract_selections` and the
# `DataStore`-only fixture table. Both are pure reindexings, so both are checked for
# EQUALITY with what `src` does rather than for approximate agreement — a reindexing
# that is only approximately the same reindexing is a bug with no upside.

demo_banner(4, "the odds index and the fixture table")

const DEMO_OI  = UP.build_odds_index(DEMO_DS.odds)
const DEMO_FXS = UP.fixture_table(DEMO_DS)

println("  ", DEMO_OI)
@printf("  fixture table   : %d fixtures, %d with a result\n",
        length(DEMO_FXS), count(v -> v.score !== nothing, values(DEMO_FXS)))

demo_gate!("4a fixture_table(ds) is `==` to Portfolio.fixture_table(ds)",
           DEMO_FXS == PF.fixture_table(DEMO_DS))
demo_gate!("4b fixture_table(ds.matches) agrees with the DataStore method",
           UP.fixture_table(DEMO_DS.matches) == DEMO_FXS)
demo_gate!("4c an already-built table passes through unchanged",
           UP.fixture_table(DEMO_FXS) === DEMO_FXS)
demo_gate!("4d every quoted match is indexed",
           Set(keys(DEMO_OI.rows)) == Set(Int.(DEMO_DS.odds.match_id)))
demo_gate!("4e no quote is lost or duplicated",
           length(DEMO_OI) == nrow(DEMO_DS.odds))

# The row-order claim, checked against the legacy `view` it replaces. `extract_selections`
# builds a `Dict{Symbol,Float64}` whose LAST write wins on a duplicate quote, and the
# resulting selection order fixes the column order of `R` — so this is not cosmetic.
const DEMO_ORDER_OK = all(Int.(DEMO_DS.matches.match_id)) do m_id
    legacy_rows = view(DEMO_DS.odds, DEMO_DS.odds.match_id .== m_id, :)
    rng = DEMO_OI.rows[m_id]
    length(rng) == nrow(legacy_rows) &&
        all(DEMO_OI.selection[p] === Symbol(legacy_rows.selection[j])
            for (j, p) in enumerate(rng)) &&
        all(DEMO_OI.odds_close[p] == Float64(legacy_rows.odds_close[j])
            for (j, p) in enumerate(rng))
end
demo_gate!("4f the index yields each match's rows in the legacy `view` order",
           DEMO_ORDER_OK)

# A `missing` market line is refused by name rather than raising three frames down.
const DEMO_BAD_ODDS = let df = copy(DEMO_DS.odds[1:12, :])
    df.market_line = Vector{Union{Missing, Float64}}(df.market_line)
    df.market_line[3] = missing
    df
end
demo_gate!("4g a `missing` market line is refused with an actionable message",
           try
               UP.build_odds_index(DEMO_BAD_ODDS); false
           catch e
               occursin("market_line", sprint(showerror, e))
           end)
demo_gate!("4h a frame missing a required column is refused by column name",
           try
               UP.build_odds_index(select(DEMO_DS.odds, Not(:odds_close))); false
           catch e
               occursin("odds_close", sprint(showerror, e))
           end)


# %% ===========================================================================
# 5. Zero allocations on the hot scoring path
# ==============================================================================
#
# `06`'s RULE 2, one level up. The claim is not merely that `compute_score_grid!`
# allocates nothing — that is `06`'s — but that a WHOLE FIXTURE can be priced across
# every market in the spec without touching the heap.
#
# The baseline row is not decoration. `@allocated` around an empty closure must read 0,
# or the measurement apparatus is itself allocating and every other row is unreadable.

demo_banner(5, "zero allocations on the hot scoring path")

const DEMO_W_POIS   = UP.BookWorkspace(DEMO_SPEC, DEMO_POIS_L; quiet = true)
const DEMO_W_NEGBIN = UP.BookWorkspace(DEMO_SPEC, DEMO_NEGBIN_L; quiet = true)

println("  ", DEMO_W_POIS)
@printf("  one workspace, %d fixtures: %.1f KiB held, %.1f KiB/fixture avoided\n",
        DEMO_N_MATCHES, UP.workspace_bytes(DEMO_W_POIS) / 1024,
        (144 * DEMO_TOTAL * 8) / 1024)

function demo_alloc(f, args...; reps = 3)
    f(args...)
    best = typemax(Int)
    for _ in 1:reps
        best = min(best, Int(@allocated f(args...)))
    end
    return best
end

const DEMO_ALLOC_ROWS = UP.AllocationRow[
    UP.AllocationRow("baseline (empty closure)", UP.baseline_allocations(), 0),
    UP.AllocationRow("compute_score_grid!  Poisson",
        demo_alloc(UP.compute_score_grid!, DEMO_W_POIS.S, DEMO_W_POIS.ws, DEMO_POIS_L, 1), 0),
    UP.AllocationRow("compute_score_grid!  NegBin",
        demo_alloc(UP.compute_score_grid!, DEMO_W_NEGBIN.S, DEMO_W_NEGBIN.ws, DEMO_NEGBIN_L, 1), 0),
    UP.AllocationRow("price_market!  1X2",
        demo_alloc(UP.price_market!, DEMO_W_POIS.slots_1x2[1].book, DEMO_W_POIS.S,
                   DEMO_W_POIS.slots_1x2[1].market), 0),
    UP.AllocationRow("price_market!  BTTS",
        demo_alloc(UP.price_market!, DEMO_W_POIS.slots_btts[1].book, DEMO_W_POIS.S,
                   DEMO_W_POIS.slots_btts[1].market), 0),
    UP.AllocationRow("price_market!  O/U 2.5",
        demo_alloc(UP.price_market!, DEMO_W_POIS.slots_ou[2].book, DEMO_W_POIS.S,
                   DEMO_W_POIS.slots_ou[2].market), 0),
    UP.AllocationRow("price_fixture!  Poisson, 5 markets",
        UP.scoring_allocations(DEMO_W_POIS, DEMO_POIS_L, 1), 0),
    UP.AllocationRow("price_fixture!  NegBin, 5 markets",
        UP.scoring_allocations(DEMO_W_NEGBIN, DEMO_NEGBIN_L, 1), 0),
    UP.AllocationRow("price_fixture!  every fixture, worst",
        maximum(UP.scoring_allocations(DEMO_W_POIS, DEMO_POIS_L, i)
                for i in 1:UP.n_matches(DEMO_POIS_L)), 0),
]

demo_gate!("5a the hot scoring path allocates 0 bytes",
           UP.allocation_table(DEMO_ALLOC_ROWS;
               title = "ALLOCATION — one fixture, $(DEMO_TOTAL) draws, 5 markets"))

# The workspace is allocated ONCE. The legacy path allocates a tensor of the same size
# per fixture, so the ratio is the fold size — which is the property that matters.
demo_gate!("5b the workspace is the only per-fold allocation of grid size",
           UP.workspace_bytes(DEMO_W_POIS) < 2 * (144 * DEMO_TOTAL * 8) + 512 * 1024)
demo_gate!("5c the spec's markets are all on the zero-allocation path",
           isempty(UP.fallback_market_names(DEMO_W_POIS)))


# %% ===========================================================================
# 6. Mathematical parity against `src/Portfolio/`
# ==============================================================================
#
# The legacy side is the live builder, run here, over the same container through
# `to_legacy_dataframe` — the `Vector{Any}` frame `Experiments` would have produced.
#
# THE GATE IS 0 ULP. `06` established why a 1e-12 tolerance is the wrong gate for a
# price: a one-ULP perturbation of a single λ shows up as ~1e-17 absolute and 16 ULP,
# comfortably inside the tolerance and unmistakable in ULP. §6d is the negative control
# that proves this harness can tell the difference.

demo_banner(6, "mathematical parity against src/Portfolio/")

println("\n  --- 6a. Poisson container, BakerMcHale shrinkage ---")
const DEMO_BOOKS_LEGACY = UP.legacy_build(DEMO_SPEC, DEMO_POIS_L, DEMO_POIS_MODEL,
                                          DEMO_DS.odds, DEMO_DS)
const DEMO_BOOKS, DEMO_BUILD = UP.build_books_reported(DEMO_SPEC, DEMO_POIS_FIT,
                                                       DEMO_DS.odds, DEMO_DS; quiet = true)

@printf("  legacy %d books, new %d books, %d selections each side\n",
        length(DEMO_BOOKS_LEGACY), length(DEMO_BOOKS),
        sum(b -> length(b.sels), DEMO_BOOKS))

demo_gate!("6a1 Poisson / BakerMcHale — every number is bit-identical",
           UP.tpl_parity_table(UP.book_parity_rows(DEMO_BOOKS_LEGACY, DEMO_BOOKS);
                               title = "PARITY — Poisson container, BakerMcHale"))
println()
demo_checks!("6a2", UP.book_structure_checks(DEMO_BOOKS_LEGACY, DEMO_BOOKS))

println("\n  --- 6b. NegBin container, FractionalKelly shrinkage ---")
const DEMO_NB_LEGACY = UP.legacy_build(DEMO_SPEC_FK, DEMO_NEGBIN_L, DEMO_NEGBIN_MODEL,
                                       DEMO_DS.odds, DEMO_DS)
const DEMO_NB_BOOKS  = UP.build_books(DEMO_SPEC_FK, DEMO_NEGBIN_L, DEMO_DS.odds, DEMO_DS;
                                      quiet = true)

demo_gate!("6b1 NegBin / FractionalKelly — every number is bit-identical",
           UP.tpl_parity_table(UP.book_parity_rows(DEMO_NB_LEGACY, DEMO_NB_BOOKS);
                               title = "PARITY — NegBin container, FractionalKelly"))
println()
demo_checks!("6b2", UP.book_structure_checks(DEMO_NB_LEGACY, DEMO_NB_BOOKS))

# The two containers must NOT agree with each other. If they did, the dispersion
# parameters would be reaching no kernel and §6b would be re-checking §6a.
demo_gate!("6c the two families price differently — dispersion reaches a kernel",
           any(DEMO_BOOKS[i].p_grid != DEMO_NB_BOOKS[i].p_grid
               for i in 1:length(DEMO_BOOKS)))

# --- 6d. the negative control -------------------------------------------------
#
# A harness that cannot fail is not evidence. One λ, moved by one ULP — the smallest
# change representable — must break the parity table for the fixture that holds it.

println("\n  --- 6d. negative control: one λ, moved by one ULP ---")
const DEMO_PERTURBED_L = let λh = copy(DEMO_POIS_L.λ_home), λa = copy(DEMO_POIS_L.λ_away)
    λh[1, 1] = nextfloat(λh[1, 1])
    UP.CountLatents(copy(UP.latent_match_ids(DEMO_POIS_L)), λh, λa, nothing)
end
const DEMO_CTRL_BOOKS = UP.build_books(DEMO_SPEC_FK, DEMO_PERTURBED_L, DEMO_DS.odds,
                                       DEMO_DS; quiet = true)
const DEMO_CTRL_CLEAN = UP.build_books(DEMO_SPEC_FK, DEMO_POIS_L, DEMO_DS.odds,
                                       DEMO_DS; quiet = true)
const DEMO_CTRL_ROWS = UP.book_parity_rows(DEMO_CTRL_CLEAN, DEMO_CTRL_BOOKS)

@printf("  p_grid max |Δ| = %.3e, max ULP = %d  (this row MUST fail)\n",
        DEMO_CTRL_ROWS[1].max_abs, DEMO_CTRL_ROWS[1].max_ulp)
demo_gate!("6d1 a one-ULP λ perturbation breaks the parity gate",
           !DEMO_CTRL_ROWS[1].pass)
demo_gate!("6d2 …and it is invisible to a 1e-12 tolerance, which is why the gate is ULP",
           DEMO_CTRL_ROWS[1].max_abs < 1e-12)


# %% ===========================================================================
# 7. Kelly staking and multi-slate simulation
# ==============================================================================

demo_banner(7, "kelly staking and multi-slate simulation")

const DEMO_RESULT = UP.simulate_portfolio(DEMO_POLICY, DEMO_BOOKS, DEMO_BUILD;
                                          initial_bankroll = 1000.0, B = 2000, seed = 1)
UP.display_portfolio(DEMO_RESULT)

const DEMO_STATES = DEMO_RESULT.daily_states
const DEMO_SUM    = DEMO_RESULT.summary

println("""
  Three numbers above are artefacts of a three-slate fixture and not results:

    * CAGR annualises a 14-day span, so it reads in the hundreds of percent. It is
      correct arithmetic on a sample far too short to annualise; §7d checks that it
      compounds back to the realised wealth, which is all it claims.
    * MDD is 0.00% and Sortino is ∞ because no window lost. Both are the honest output
      of a run with no downside, and both exercise the branch that produces them.
    * mean k_risk is 1.0000 — the drawdown budget never bound at λ = 23 on a book this
      small, so `SlateDrawdown` is present but slack. `calibrate_lambda` is the dial
      that moves it; trust and `scale` are not (see l03's header).
""")

println()
demo_checks!("7a", [
    "one DailyState per settlement window" =>
        (length(DEMO_STATES) == length(unique(b.date for b in DEMO_BOOKS))),
    "states are chronological" => issorted(DEMO_STATES, by = d -> d.date),
    "bankroll_close == bankroll_open × (1 + pnl)" =>
        all(isapprox(d.bankroll_close, d.bankroll_open * (1 + d.pnl_frac); rtol = 1e-12)
            for d in DEMO_STATES),
    "each window opens where the last one closed" =>
        all(DEMO_STATES[i].bankroll_open == DEMO_STATES[i - 1].bankroll_close
            for i in 2:length(DEMO_STATES)),
    "no window loses more than the bankroll" =>
        all(d.pnl_frac > -1.0 for d in DEMO_STATES),
    "exposure respects the FixedCap" =>
        all(d.exposure <= DEMO_POLICY.cap.cap + 1e-12 for d in DEMO_STATES),
    "bet count matches the trajectory's bet frame" =>
        (sum(d -> d.n_bets, DEMO_STATES) == nrow(DEMO_RESULT.trajectory.bets)),
    "fixture count matches the book count" =>
        (DEMO_SUM.n_fixtures == length(DEMO_BOOKS)),
    "the MCMC verdict was carried, not recomputed" =>
        (DEMO_RESULT.converged === true),
])

# `initial_bankroll` is a REPORTING SCALE. `SlateContext.bankroll` is handed the
# fraction, exactly as `Portfolio.simulate` hands it, so a bankroll-dependent trust or
# filter sees the same number under both and the trajectories cannot diverge.
const DEMO_RESULT_UNIT = UP.simulate_portfolio(DEMO_POLICY, DEMO_BOOKS;
                                               initial_bankroll = 1.0, bootstrap = false)
demo_checks!("7b", [
    "initial_bankroll is a pure scale — the trajectory is bit-identical" =>
        (DEMO_RESULT_UNIT.trajectory.bankroll == DEMO_RESULT.trajectory.bankroll),
    "…and ROI, a ratio, is unchanged" =>
        (DEMO_RESULT_UNIT.summary.roi == DEMO_SUM.roi),
    "…and final wealth scales exactly by 1000" =>
        isapprox(DEMO_RESULT.summary.final_bankroll,
                 1000 * DEMO_RESULT_UNIT.summary.final_bankroll; rtol = 1e-12),
])

# Reproducibility, and RNG hygiene. `BakerMcHale` and the bootstrap both sample; neither
# may reach the caller's stream. A backtest that silently advanced the global RNG would
# make every downstream simulation in the same session unreproducible.
Random.seed!(4242)
const DEMO_RNG_BEFORE = rand(3)
Random.seed!(4242)
const DEMO_REPEAT = UP.simulate_portfolio(DEMO_POLICY, DEMO_BOOKS;
                                          initial_bankroll = 1000.0, B = 2000, seed = 1)
const DEMO_RNG_AFTER = rand(3)

demo_checks!("7c", [
    "the same inputs give the same trajectory twice" =>
        (DEMO_REPEAT.trajectory.bankroll == DEMO_RESULT.trajectory.bankroll),
    "…the same bootstrap interval twice" =>
        (DEMO_REPEAT.bootstrap_ci.roi_lo == DEMO_RESULT.bootstrap_ci.roi_lo &&
         DEMO_REPEAT.bootstrap_ci.roi_hi == DEMO_RESULT.bootstrap_ci.roi_hi),
    "simulating never perturbs the caller's global RNG" =>
        (DEMO_RNG_AFTER == DEMO_RNG_BEFORE),
])

# The six metrics `src` has no field for, checked for internal consistency rather than
# against a reference — there is no reference. Each is checked against the definition it
# is documented with, from the same trajectory.
const DEMO_LG = log.(1.0 .+ DEMO_RESULT.trajectory.slate_pl)
demo_checks!("7d", [
    "CAGR compounds to the realised final wealth" =>
        isapprox((1 + DEMO_SUM.cagr)^(DEMO_SUM.span_days / 365.25),
                 DEMO_SUM.final_bankroll / DEMO_SUM.initial_bankroll; rtol = 1e-10),
    "Sharpe is mean/sd of per-slate LOG returns" =>
        isapprox(DEMO_SUM.sharpe, mean(DEMO_LG) / std(DEMO_LG); rtol = 1e-12),
    "Sortino uses downside deviation only" =>
        (isempty(DEMO_LG[DEMO_LG .< 0]) ? !isfinite(DEMO_SUM.sortino) :
         isapprox(DEMO_SUM.sortino,
                  mean(DEMO_LG) / sqrt(mean(DEMO_LG[DEMO_LG .< 0] .^ 2)); rtol = 1e-12)),
    "win rate is over staked selections" =>
        isapprox(DEMO_SUM.win_rate,
                 mean(DEMO_RESULT.trajectory.bets.payoff .> 0); rtol = 1e-12),
    "1X2 ROI is the 1X2 subset of the flat ROI" =>
        (isnan(DEMO_SUM.roi_1x2) || isapprox(DEMO_SUM.roi_1x2,
            100 * sum(DEMO_RESULT.trajectory.bets.pnl[
                        startswith.(DEMO_RESULT.trajectory.bets.family, "1X2")]) /
                  sum(DEMO_RESULT.trajectory.bets.stake[
                        startswith.(DEMO_RESULT.trajectory.bets.family, "1X2")]);
            rtol = 1e-12)),
    "a single-slate span reports CAGR as NaN rather than inventing a year" =>
        isnan(UP.simulate_portfolio(DEMO_POLICY,
                  filter(b -> b.date == DEMO_BOOKS[1].date, DEMO_BOOKS);
                  bootstrap = false).summary.cagr),
])

# The refusals.
demo_gate!("7e simulating an unsettled book is refused, not silently zero-filled",
           try
               unsettled = UP.build_books(DEMO_SPEC_FK, DEMO_POIS_L, DEMO_DS.odds,
                   Dict(k => (date = v.date, score = nothing)
                        for (k, v) in DEMO_FXS);
                   require_result = false, quiet = true)
               UP.simulate_portfolio(DEMO_POLICY, unsettled; bootstrap = false)
               false
           catch e
               occursin("settled", sprint(showerror, e))
           end)


# %% ===========================================================================
# 8. Simulation parity against `Portfolio.simulate`
# ==============================================================================
#
# `simulate_portfolio` runs its OWN forward walk. It does not call `simulate` and then
# decorate the result — if it did, this section would be checking that a function agrees
# with itself.

demo_banner(8, "simulation parity against Portfolio.simulate")

const DEMO_SLATES     = PF.group(DEMO_POLICY.grouping, DEMO_BOOKS)
const DEMO_TRAJ_LEGACY = PF.simulate(DEMO_POLICY, DEMO_SLATES)
const DEMO_PATH_LEGACY = PF.path_metrics(DEMO_TRAJ_LEGACY)
const DEMO_CI_LEGACY   = PF.bootstrap_roi(DEMO_TRAJ_LEGACY.bets; B = 2000, seed = 1)

demo_gate!("8a the trajectory is bit-identical to Portfolio.simulate's",
           UP.tpl_parity_table(
               UP.trajectory_parity_rows(DEMO_TRAJ_LEGACY, DEMO_RESULT.trajectory);
               title = "PARITY — trajectory, src vs simulate_portfolio"))
println()
demo_checks!("8b", UP.trajectory_structure_checks(DEMO_TRAJ_LEGACY, DEMO_RESULT.trajectory))

demo_gate!("8c every path metric src computes is bit-identical",
           UP.tpl_parity_table(UP.summary_parity_rows(DEMO_SUM, DEMO_PATH_LEGACY);
                               title = "PARITY — PortfolioSummary vs path_metrics"))
println()
demo_checks!("8d", [
    "the match-clustered ROI interval is bit-identical to bootstrap_roi" =>
        (DEMO_CI_LEGACY.lo == DEMO_RESULT.bootstrap_ci.roi_lo &&
         DEMO_CI_LEGACY.hi == DEMO_RESULT.bootstrap_ci.roi_hi &&
         DEMO_CI_LEGACY.sd == DEMO_RESULT.bootstrap_ci.roi_sd),
    "the slate-blocked growth interval brackets the realised growth" =>
        (DEMO_RESULT.bootstrap_ci.growth_lo <= DEMO_SUM.growth_per_slate <=
         DEMO_RESULT.bootstrap_ci.growth_hi),
    "attribution is Portfolio.attribution, on the same trajectory" =>
        (DEMO_RESULT.attribution == PF.attribution(DEMO_TRAJ_LEGACY)),
    "grouping agrees with Portfolio.group" =>
        (length(DEMO_SLATES) == length(DEMO_STATES) &&
         all(DEMO_SLATES[i].window == DEMO_STATES[i].date for i in eachindex(DEMO_SLATES))),
])


# %% ===========================================================================
# 9. The convergence gate
# ==============================================================================
#
# The gate this pipeline has never had. `src/Portfolio/` will stake real bankroll on a
# chain that did not mix, and an unconverged posterior does not merely add noise: it is
# too NARROW and biased toward wherever the sampler stuck, so every model probability
# looks more confident than the evidence supports and every `p_model - p_market` edge
# looks larger. Kelly stake size is monotone in that edge.

demo_banner(9, "the convergence gate")

@printf("  %-22s %8s %10s %10s  %s\n", "fit", "verdict", "max R-hat", "min ESS",
        "failed gates")
println("  ", "-"^76)
for f in (DEMO_POIS_FIT, DEMO_NEGBIN_FIT, DEMO_BROKEN_FIT)
    d = f.diagnostics
    @printf("  %-22s %8s %10.4f %10.1f  %s\n", UP.fit_name(f),
            d.passed ? "PASS" : "FAIL", d.max_rhat, d.min_ess_bulk,
            isempty(d.failed_gates) ? "—" : join(d.failed_gates, ", "))
end

const DEMO_REFUSED = try
    UP.build_books(DEMO_SPEC_FK, DEMO_BROKEN_FIT, DEMO_DS.odds, DEMO_DS; quiet = true)
    nothing
catch e
    e
end

# Building the same fit with the gate lifted. The books must be IDENTICAL: the gate
# refuses, it does not change arithmetic.
const DEMO_UNGATED, DEMO_UNGATED_REPORT = UP.build_books_reported(
    DEMO_SPEC_FK, DEMO_BROKEN_FIT, DEMO_DS.odds, DEMO_DS;
    require_converged = false, quiet = true)
const DEMO_UNGATED_DIRECT = UP.build_books(DEMO_SPEC_FK, DEMO_BROKEN_FIT.latents,
                                           DEMO_DS.odds, DEMO_DS; quiet = true)

# A `Fit` carrying no audit. Not a crash and not a pass — "unknown" counts as failed,
# for the same reason `07` abstains on an unmeasured gate.
const DEMO_UNAUDITED = UP.Fit(getfield(DEMO_POIS_FIT, :config),
                              getfield(DEMO_POIS_FIT, :folds),
                              getfield(DEMO_POIS_FIT, :latents),
                              nothing,
                              getfield(DEMO_POIS_FIT, :metadata),
                              getfield(DEMO_POIS_FIT, :save_path))

println()
demo_checks!("9a", [
    "an unconverged Fit is REFUSED by default" =>
        (DEMO_REFUSED isa UP.ConvergenceRefusal),
    "…and the refusal names the gates it failed" =>
        (DEMO_REFUSED isa UP.ConvergenceRefusal &&
         !isempty(DEMO_REFUSED.failed_gates) &&
         occursin("R-hat", sprint(showerror, DEMO_REFUSED))),
    "…and says how to proceed anyway" =>
        (DEMO_REFUSED isa UP.ConvergenceRefusal &&
         occursin("require_converged = false", sprint(showerror, DEMO_REFUSED))),
    "require_converged = false BUILDS it" => !isempty(DEMO_UNGATED),
    "…and records the verdict on the BuildReport" =>
        (DEMO_UNGATED_REPORT.converged === false &&
         !isempty(DEMO_UNGATED_REPORT.failed_gates)),
    "…and the books are bit-identical to the ungated container route" =>
        all(DEMO_UNGATED[i].a_kelly == DEMO_UNGATED_DIRECT[i].a_kelly &&
            DEMO_UNGATED[i].p_grid  == DEMO_UNGATED_DIRECT[i].p_grid
            for i in eachindex(DEMO_UNGATED)),
    "a converged Fit is ADMITTED and flagged as such" =>
        (DEMO_BUILD.converged === true && DEMO_BUILD.gated),
    "a Fit with no audit counts as NOT converged" =>
        !first(UP.convergence_verdict(DEMO_UNAUDITED)),
    "…and is refused rather than crashing inside the gate" =>
        (try
             UP.build_books(DEMO_SPEC_FK, DEMO_UNAUDITED, DEMO_DS.odds, DEMO_DS;
                            quiet = true); false
         catch e
             e isa UP.ConvergenceRefusal && "no audit" in e.failed_gates
         end),
    "the verdict travels into the PortfolioResult" =>
        (UP.simulate_portfolio(DEMO_POLICY, DEMO_UNGATED, DEMO_UNGATED_REPORT;
                               bootstrap = false).converged === false),
    "stake_sheet is gated the same way" =>
        (try
             UP.stake_sheet(DEMO_SYSTEM, DEMO_BROKEN_FIT, DEMO_DS.odds, DEMO_FXS;
                            quiet = true); false
         catch e
             e isa UP.ConvergenceRefusal
         end),
])


# %% ===========================================================================
# 10. Backward compatibility
# ==============================================================================
#
# The strongest form of the claim available: this framework declares no domain or
# configuration type of its own, so old and new are THE SAME TYPES and there is nothing
# for a bridge to get wrong.

demo_banner(10, "backward compatibility")

"""
A legacy call site, its body copied VERBATIM from
`current_development/scottish_lower/02_poisson_wealth/r03_growth_clv.jl:76-94`.

Only the import line differs, and it has to: `BayesianFootball` exports the name
`Portfolio`, and Julia refuses to rebind an imported name.
"""
module LegacyCallSite

import BayesianFootball
using ..UnifiedPortfolio.Legacy         # binds `Portfolio`
using DataFrames

function run(book_spec, latents_df, expr, odds_df, ds, policy)
    books    = Portfolio.build_books(book_spec, latents_df, expr, odds_df, ds)
    slates   = Portfolio.group(policy.grouping, books)
    traj     = Portfolio.simulate(policy, slates)
    metrics  = Portfolio.path_metrics(traj)
    ci       = Portfolio.bootstrap_roi(traj.bets; B = 500, seed = 7)
    attrib   = Portfolio.attribution(traj)
    key      = Portfolio.book_cache_key(book_spec)
    return (; books, slates, traj, metrics, ci, attrib, key)
end

end # module LegacyCallSite

const DEMO_LEGACY_DF  = UP.to_legacy_dataframe(DEMO_POIS_L)
const DEMO_LEGACY_OUT = LegacyCallSite.run(DEMO_SPEC_FK, DEMO_LEGACY_DF,
                                           UP.LegacyExpr(DEMO_POIS_MODEL),
                                           DEMO_DS.odds, DEMO_DS, DEMO_POLICY)

# The same call, straight through `src`, for comparison.
const DEMO_SRC_OUT_BOOKS = PF.build_books(DEMO_SPEC_FK, DEMO_LEGACY_DF,
                                          UP.LegacyExpr(DEMO_POIS_MODEL),
                                          DEMO_DS.odds, DEMO_FXS)

@printf("  legacy call site : %d books, %d slates, %d bets, final %.4f\n",
        length(DEMO_LEGACY_OUT.books), length(DEMO_LEGACY_OUT.slates),
        nrow(DEMO_LEGACY_OUT.traj.bets), DEMO_LEGACY_OUT.metrics.final)

# A serialisation round trip: a book cached to disk by the old code and read by the new,
# or the reverse. Nothing in either direction needs a version check.
const DEMO_ROUNDTRIP = let io = IOBuffer()
    serialize(io, DEMO_SRC_OUT_BOOKS)
    seekstart(io)
    deserialize(io)
end

println()
demo_checks!("10a", [
    "MatchBook is the src type, not a look-alike" => (UP.MatchBook === PF.MatchBook),
    "Selection is the src type" => (UP.Selection === PF.Selection),
    "BookSpec is the src type" => (UP.BookSpec === PF.BookSpec),
    "PolicySpec is the src type" => (UP.PolicySpec === PF.PolicySpec),
    "Trajectory is the src type" => (UP.Trajectory === PF.Trajectory),
    "the abstract seams are the src hierarchy" =>
        (UP.AbstractAllocator === PF.AbstractAllocator &&
         UP.KellyLogUtility <: PF.AbstractAllocator),
])

demo_checks!("10b", [
    "a legacy call site runs unmodified" => (length(DEMO_LEGACY_OUT.books) > 0),
    "…and produces exactly the books src produces" =>
        UP.tpl_compare("", reduce(vcat, [b.a_kelly for b in DEMO_LEGACY_OUT.books]),
                       reduce(vcat, [b.a_kelly for b in DEMO_SRC_OUT_BOOKS])).pass,
    "…including through the fast path this framework routes it onto" =>
        UP.tpl_compare("", reduce(vcat, [b.p_grid for b in DEMO_LEGACY_OUT.books]),
                       reduce(vcat, [b.p_grid for b in DEMO_CTRL_CLEAN])).pass,
    "book_cache_key is unchanged, so an existing cache still HITS" =>
        (DEMO_LEGACY_OUT.key == PF.book_cache_key(DEMO_SPEC_FK)),
    "equal specs give equal keys (the BakerMcHale case)" =>
        (UP.book_cache_key(DEMO_SPEC) ==
         PF.book_cache_key(UP.BookSpec(markets = D.MarketConfig(DEMO_MARKETS),
                                       price = UP.DeArb(),
                                       allocator = UP.KellyLogUtility(),
                                       shrink = UP.BakerMcHale(n_draws = 16),
                                       exec = DEMO_EXEC))),
])

demo_checks!("10c", [
    "src functions accept books built here" =>
        (nrow(PF.simulate(DEMO_POLICY, PF.group(DEMO_POLICY.grouping, DEMO_BOOKS)).bets) ==
         nrow(DEMO_RESULT.trajectory.bets)),
    "this framework accepts books built by src" =>
        (UP.simulate_portfolio(DEMO_POLICY, DEMO_SRC_OUT_BOOKS;
                               bootstrap = false).summary.n_bets ==
         nrow(PF.simulate(DEMO_POLICY,
                          PF.group(DEMO_POLICY.grouping, DEMO_SRC_OUT_BOOKS)).bets)),
    "…including after a serialisation round trip" =>
        (UP.simulate_portfolio(DEMO_POLICY, DEMO_ROUNDTRIP;
                               bootstrap = false).trajectory.bankroll ==
         PF.simulate(DEMO_POLICY, PF.group(DEMO_POLICY.grouping,
                                           DEMO_SRC_OUT_BOOKS)).bankroll),
    "src's report/attribution read this framework's trajectory" =>
        (PF.report(DEMO_RESULT.trajectory).final == DEMO_PATH_LEGACY.final),
])

# The legacy match-day surface, and the legacy primitives a caller may compose by hand.
const DEMO_SHEET = UP.stake_sheet(DEMO_SYSTEM, DEMO_LEGACY_DF,
                                  UP.LegacyExpr(DEMO_POIS_MODEL), DEMO_DS.odds,
                                  DEMO_FXS; bankroll = 1000.0, quiet = true)
const DEMO_SHEET_SRC = PF.stake_sheet(DEMO_SYSTEM, DEMO_LEGACY_DF,
                                      UP.LegacyExpr(DEMO_POIS_MODEL), DEMO_DS.odds,
                                      DEMO_FXS; bankroll = 1000.0)
const DEMO_LEGACY_SELS = UP.extract_selections(
    DEMO_DS.odds, Int(DEMO_DS.matches.match_id[1]), DEMO_SPEC_FK,
    Dict(string(m) => UP.price_market(UP.compute_score_grid(DEMO_POIS_L, 1), m)
         for m in DEMO_MARKETS))

demo_checks!("10d", [
    "stake_sheet keeps src's columns, in src's order" =>
        (names(DEMO_SHEET) == names(DEMO_SHEET_SRC)),
    "…and its rows" =>
        (nrow(DEMO_SHEET) == nrow(DEMO_SHEET_SRC) &&
         DEMO_SHEET.stake == DEMO_SHEET_SRC.stake),
    "slate_summary is src's, on this framework's sheet" =>
        (nrow(UP.slate_summary(DEMO_SHEET)) == length(DEMO_STATES)),
    "the legacy 4-arg extract_selections still works" =>
        (length(DEMO_LEGACY_SELS) == length(DEMO_BOOKS[1].sels)),
    "the legacy single-fixture build_book still works" =>
        (UP.build_book(DEMO_SPEC_FK, DEMO_LEGACY_DF[1, :], UP.LegacyExpr(DEMO_POIS_MODEL),
                       DEMO_DS.odds, DEMO_FXS) isa UP.MatchBook),
    "the briefing's field accessors read the src fields" =>
        (UP.book_payoff(DEMO_BOOKS[1]) === DEMO_BOOKS[1].R &&
         UP.book_alloc(DEMO_BOOKS[1]) === DEMO_BOOKS[1].a_kelly &&
         UP.book_match_id(DEMO_BOOKS[1]) === DEMO_BOOKS[1].m_id &&
         UP.book_selections(DEMO_BOOKS[1]) === DEMO_BOOKS[1].sels &&
         UP.book_settle(DEMO_BOOKS[1]) === DEMO_BOOKS[1].settle &&
         UP.book_shrink(DEMO_BOOKS[1]) === DEMO_BOOKS[1].k_shrink &&
         UP.sel_odds_close(DEMO_BOOKS[1].sels[1]) === DEMO_BOOKS[1].sels[1].odds_quoted &&
         UP.sel_odds_settle(DEMO_BOOKS[1].sels[1]) === DEMO_BOOKS[1].sels[1].odds_used),
])

# --- 10e. the rest of the exported surface, exercised once each ---------------
#
# Not depth — coverage. A name this framework exports and never calls is a name nobody
# has checked compiles, and `run_portfolio_simulation` in particular is a briefing
# requirement that exists nowhere in `src` to be compared against.

const DEMO_ONECALL, DEMO_OC_BOOKS, DEMO_OC_BUILD =
    UP.run_portfolio_simulation(DEMO_SPEC_FK, DEMO_POLICY, DEMO_POIS_FIT,
                                DEMO_DS.odds, DEMO_DS; bootstrap = false, quiet = true)
const DEMO_SYS_ONECALL = first(
    UP.run_portfolio_simulation(UP.PortfolioSystem(DEMO_SPEC_FK, DEMO_POLICY),
                                (DEMO_LEGACY_DF, UP.LegacyExpr(DEMO_POIS_MODEL)),
                                DEMO_DS.odds, DEMO_DS; bootstrap = false, quiet = true))

demo_checks!("10e", [
    "run_portfolio_simulation builds, groups, simulates and reports in one call" =>
        (length(DEMO_OC_BOOKS) == DEMO_N_MATCHES && DEMO_OC_BUILD.converged === true &&
         DEMO_ONECALL.summary.n_bets > 0),
    "…and its PortfolioSystem form takes the legacy (latents_df, expr) pair" =>
        (DEMO_SYS_ONECALL.trajectory.bankroll == DEMO_ONECALL.trajectory.bankroll),
    "states_frame is one row per settlement window, with every DailyState field" =>
        (nrow(UP.states_frame(DEMO_RESULT)) == length(DEMO_STATES) &&
         Symbol.(names(UP.states_frame(DEMO_RESULT))) ==
             collect(fieldnames(UP.DailyState))),
    "as_namedtuple round-trips the summary" =>
        (UP.as_namedtuple(DEMO_SUM).roi === DEMO_SUM.roi),
    "log_growth is the per-window log return" =>
        isapprox(UP.log_growth(DEMO_STATES[1]), log(1 + DEMO_STATES[1].pnl_frac);
                 rtol = 1e-15),
    "PortfolioResult indexes and iterates its DailyStates" =>
        (DEMO_RESULT[1] === DEMO_STATES[1] &&
         collect(DEMO_RESULT) == DEMO_STATES),
    "unsettled_books names exactly the books simulate would refuse" =>
        (isempty(UP.unsettled_books(DEMO_BOOKS)) &&
         length(UP.unsettled_books(UP.build_books(
             DEMO_SPEC_FK, DEMO_POIS_L, DEMO_DS.odds,
             Dict(k => (date = v.date, score = nothing) for (k, v) in DEMO_FXS);
             require_result = false, quiet = true))) == DEMO_N_MATCHES),
    "build_slates and group agree" =>
        (length(UP.build_slates(DEMO_BOOKS)) == length(DEMO_STATES)),
    "grid_shrink_factor over a shared grid equals src's over a ScoreMatrix" =>
        (UP.grid_shrink_factor(UP.BakerMcHale(n_draws = 8), DEMO_W_POIS.S,
                               DEMO_BOOKS[1].R, DEMO_BOOKS[1].p_grid,
                               UP.KellyLogUtility(), DEMO_EXEC; seed_offset = 7) ==
         PF.shrink_factor(UP.BakerMcHale(n_draws = 8),
                          BF.Predictions.ScoreMatrix(DEMO_W_POIS.S), DEMO_BOOKS[1].R,
                          DEMO_BOOKS[1].p_grid, UP.KellyLogUtility(), DEMO_EXEC;
                          seed_offset = 7)),
])


# %% ===========================================================================
# 11. What the builder reports that `src` drops
# ==============================================================================
#
# `src`'s builder returns `nothing` for five distinct causes and filters them all out
# (`book.jl:151`), so a data outage and a clean fold look identical from the outside.
# The books here are the same books — this adds a second return value.

demo_banner(11, "what the builder reports that src drops")

# Five fixtures, damaged five different ways, so all four skip causes fire at once and
# a fifth case — a partial market group inside an otherwise fine fixture — shows that a
# refusal is per GROUP and not per fixture.
const DEMO_ID_PARTIAL  = Int(DEMO_DS.matches.match_id[1])   # one 1X2 leg missing
const DEMO_ID_NOQUOTE  = Int(DEMO_DS.matches.match_id[2])   # every quote gone
const DEMO_ID_NOSELS   = Int(DEMO_DS.matches.match_id[3])   # one leg gone from EVERY group
const DEMO_ID_NOFIXT   = Int(DEMO_DS.matches.match_id[4])   # absent from the fixture table
const DEMO_ID_UNPLAYED = Int(DEMO_DS.matches.match_id[5])   # quoted, but not yet played

const DEMO_DAMAGED = let df = copy(DEMO_DS.odds)
    # One leg of one 1X2 group, gone. Vig removal divides by the sum over the legs
    # PRESENT, so admitting the survivors would inflate the model's apparent edge over
    # them by up to 20 points. `require_complete_markets` is the only thing in the way.
    UP.drop_market_leg!(df, DEMO_ID_PARTIAL, "1X2", 0.0, :draw)
    # Every quote for one fixture, gone — an odds feed that lost a match.
    deleteat!(df, findall(==(DEMO_ID_NOQUOTE), Int.(df.match_id)))
    # One leg gone from every group, so no complete group survives anywhere.
    for (grp, ln, sel) in (("1X2", 0.0, :draw), ("BTTS", 0.0, :btts_no),
                           ("OverUnder", 1.5, :under_15), ("OverUnder", 2.5, :under_25),
                           ("OverUnder", 3.5, :under_35))
        UP.drop_market_leg!(df, DEMO_ID_NOSELS, grp, ln, sel)
    end
    df
end

# The fixture table, damaged the other two ways.
const DEMO_DMG_FXS = let d = copy(DEMO_FXS)
    delete!(d, DEMO_ID_NOFIXT)
    d[DEMO_ID_UNPLAYED] = (date = d[DEMO_ID_UNPLAYED].date, score = nothing)
    d
end

const DEMO_DMG_BOOKS, DEMO_DMG_REPORT = UP.build_books_reported(
    DEMO_SPEC_FK, DEMO_POIS_L, DEMO_DAMAGED, DEMO_DMG_FXS; quiet = true)
const DEMO_DMG_LEGACY = UP.legacy_build(DEMO_SPEC_FK, DEMO_POIS_L, DEMO_POIS_MODEL,
                                        DEMO_DAMAGED, DEMO_DMG_FXS)

show(stdout, MIME"text/plain"(), DEMO_DMG_REPORT)

const DEMO_DMG_1 = only(b for b in DEMO_DMG_BOOKS if b.m_id == DEMO_ID_PARTIAL)

println()
demo_checks!("11a", [
    "cause 1 — a fixture absent from the fixture table is named" =>
        (DEMO_DMG_REPORT.skipped_no_fixture == [DEMO_ID_NOFIXT]),
    "cause 2 — an unplayed fixture is named, not confused with a data gap" =>
        (DEMO_DMG_REPORT.skipped_unplayed == [DEMO_ID_UNPLAYED]),
    "cause 3 — a fixture with no quotes at all is named" =>
        (DEMO_DMG_REPORT.skipped_no_quotes == [DEMO_ID_NOQUOTE]),
    "cause 4 — a fixture whose every market group is partial is named" =>
        (DEMO_DMG_REPORT.skipped_no_selections == [DEMO_ID_NOSELS]),
    "…and `src` returns a bare `nothing` for all four" =>
        (length(DEMO_DMG_LEGACY) == DEMO_N_MATCHES - 4),
    "a PARTIAL group is refused per group, not per fixture" =>
        (!any(s -> s.group == "1X2", DEMO_DMG_1.sels) && length(DEMO_DMG_1.sels) == 8),
    "the report accounts for every fixture" =>
        (DEMO_DMG_REPORT.n_books + UP.n_skipped(DEMO_DMG_REPORT) ==
         DEMO_DMG_REPORT.n_fixtures),
    "no fixture errored" => isempty(DEMO_DMG_REPORT.errored),
    "the books are still bit-identical to src's on the same damaged inputs" =>
        UP.tpl_parity_table(UP.book_parity_rows(DEMO_DMG_LEGACY, DEMO_DMG_BOOKS);
                            title = "PARITY — damaged odds frame and fixture table"),
])


# %% ===========================================================================
# 12. Cost, measured
# ==============================================================================

demo_banner(12, "cost, measured")

const DEMO_COST = UP.CostRow[
    UP.measure_pricing_cost("pricing only (grid + 5 markets)", DEMO_SPEC,
                            DEMO_POIS_L, DEMO_POIS_MODEL),
    UP.measure_build_cost("full build, FractionalKelly", DEMO_SPEC_FK, DEMO_POIS_L,
                          DEMO_POIS_MODEL, DEMO_DS.odds, DEMO_DS),
    UP.measure_build_cost("full build, BakerMcHale(16)", DEMO_SPEC, DEMO_POIS_L,
                          DEMO_POIS_MODEL, DEMO_DS.odds, DEMO_DS),
]
UP.cost_table(DEMO_COST;
              title = "COST — $(DEMO_N_MATCHES) fixtures × $(DEMO_TOTAL) draws, 5 markets")

println("""
  Read the first row, not the third.

  Row 1 isolates what this framework changed: a score grid and five market books per
  fixture. The legacy side allocates a fresh (12 × 12 × $(DEMO_TOTAL)) tensor and five
  dictionaries every time; the new side allocates one workspace for the whole fold and
  then nothing. Its byte column is the workspace, ONCE — so the ratio grows with the
  fold, and on a 500-fixture production fold it is ~20× what it reads here.

  Rows 2 and 3 add the convex solve, which both sides pay identically. `BakerMcHale`
  re-solves the allocator 16 times per fixture (128 in production) and dominates
  everything, which is why row 3's speedup is near 1. That is not a disappointing
  result — it is a correct one, and it says where the next optimisation belongs.

  No gate is attached to this section. A timing on a $(DEMO_N_MATCHES)-fixture synthetic
  fold is an indication, not a measurement of production cost, and gating CI on it would
  make the run flaky for a reason unrelated to correctness.
""")


# %% ===========================================================================
# 13. Gate summary
# ==============================================================================

demo_banner(13, "gate summary")

const DEMO_FAILED = [k for (k, v) in DEMO_GATES if !v]

@printf("\n  %d gates, %d passed, %d failed\n\n",
        length(DEMO_GATES), length(DEMO_GATES) - length(DEMO_FAILED), length(DEMO_FAILED))

for (k, v) in DEMO_GATES
    @printf("  %-6s %s\n", v ? "ok" : "FAIL", k)
end

println()
if isempty(DEMO_FAILED)
    println("  ALL GATES PASS.")
    println()
    println("  Established: the rebuilt builder is bit-identical to src/Portfolio/ across")
    println("  every layer of a book and every step of a simulation; its scoring path")
    println("  allocates nothing; the convergence gate refuses what it should and changes")
    println("  no arithmetic when it admits; and a legacy call site runs verbatim.")
    println()
    println("  NOT established: that any model fits anything, or that any strategy makes")
    println("  money. The market here is the model's own prices — it is beaten by")
    println("  construction, and every positive ROI above is an artefact of the fixture.")
else
    println("  FAILED GATES:")
    for k in DEMO_FAILED
        println("    - ", k)
    end
end
println()

exit(isempty(DEMO_FAILED) ? 0 : 1)
