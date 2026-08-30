# ==============================================================================
# 08 — UNIFIED EVALUATION FRAMEWORK : THE PROOF
# ==============================================================================
#
# WHAT THIS IS
#   A deterministic verification that the typed evaluation kernels compute the same
#   numbers as `src/evaluation/`, that convergence gating actually gates, and that every
#   legacy evaluation call site keeps working verbatim.
#
#   Eight things are established, in this order:
#
#     §4   the three typed indexes are correct, and the ONE floating-point assumption
#          the whole parity story rests on — that `mean(view(M, i, :))` is bit-identical
#          to `mean` of the `Vector` the legacy frame stored — is measured, not assumed
#     §5   all six scoring rules compute, on a Poisson container and a NegBin one,
#          reaching different marginals BY DISPATCH
#     §6   mathematical parity against the live `BayesianFootball.Evaluation` kernels,
#          leaf by leaf, for every metric that `src` can compute at all
#     §7   the same inputs give the same numbers twice — including RQR, which in `src`
#          they do not
#     §8   the convergence gate EXCLUDES an unconverged fit by default, FLAGS it on
#          request, and REFUSES a single-fit call that asks for gating
#     §9   a legacy call site — its body copied verbatim — runs unmodified, and produces
#          column names identical to `src`'s, character for character
#     §10  three live defects in `src/evaluation/` reproduced, not merely described
#     §11  the cost, measured rather than asserted
#
# WHAT THIS IS NOT
#   Not a model-fitting run, not a betting study. No MCMC is run and no database is
#   touched. Chains are prior draws with a fixed seed (`06/l04_parity.jl` §9) and the
#   odds are the model's own prices, perturbed and vigged (`l05_parity.jl` §6).
#
#   NOTHING HERE SAYS ANY MODEL IS GOOD. The synthetic market is built FROM the model,
#   so the model beats it by construction and every `diff_ll` below is an artefact of
#   the fixture. The claim is that two implementations of six metrics agree, and that
#   the gate in front of them works.
#
# THE COMPARABILITY CONTRACT
#   Every parity row compares TWO IMPLEMENTATIONS OVER ONE SET OF NUMBERS. The legacy
#   side is the live `src` path — a real `ExperimentResults`, a real `LatentStates`, the
#   real `model_inference` and the real four-column `innerjoin` — fed a `DataFrame`
#   built from the same typed container the new side reads. Nothing is transcribed.
#
# WHAT WOULD FALSIFY THE CLAIMS
#   Any parity leaf in §6 more than 1e-12 from `src`'s. Any non-identical repeat in §7.
#   An unconverged fit appearing in §8's default scorecard. Any legacy expression in §9
#   raising, or producing a column name `src` does not.
#
#   §6's gate is a TOLERANCE and not bit-identity, which is a weaker claim than
#   `06_typed_posterior_latents` makes about prices, deliberately and for a stated
#   reason — see `MetricParityRow` in `l05_parity.jl`. §6.4 is a pair of negative
#   controls that show where that gate fires and where it stops.
#
# USAGE
#   julia --project current_development/08_unified_evaluation_framework/r01_demo.jl
#
#   or, in a REPL:
#     include("current_development/08_unified_evaluation_framework/r01_demo.jl")
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
using MCMCChains
using Printf
using Random
using Statistics

# The thing under test. `l04_compat_bridge.jl` opens `module UnifiedEvaluation`, whose
# include chain pulls in l03 → l02 → l01 and, through l01, the whole of
# `07_unified_inference_framework`, `06_typed_posterior_latents` and
# `05_composable_count_builder`.
include(joinpath(@__DIR__, "l04_compat_bridge.jl"))

const UE = UnifiedEvaluation
const BF = BayesianFootball
const D  = BayesianFootball.Data
const PG = BayesianFootball.Models.PreGame

using .UnifiedEvaluation


# %% ===========================================================================
# 2. Configuration
# ==============================================================================

const DEMO_N_TEAMS   = 8
const DEMO_N_FOLDS   = 3
const DEMO_PER_FOLD  = 8
const DEMO_N_MATCHES = DEMO_N_FOLDS * DEMO_PER_FOLD     # 24 held-out fixtures
const DEMO_N_DRAWS   = 300                               # posterior draws PER CHAIN
const DEMO_N_CHAINS  = 4                                 # exercises the flattening
const DEMO_TOTAL     = DEMO_N_DRAWS * DEMO_N_CHAINS      # 1200
const DEMO_N_SEASONS = 2
const DEMO_SEED      = 20240808

# The markets the synthetic odds frame carries. Five, not the forty-odd of
# `DEFAULT_MARKET_CONFIG`: these are the ones `MIQResult` reports on, and a store that
# carried more would make the unfiltered parity rows compare different row SETS rather
# than different implementations (`parity_scope_ok`, checked in §6).
const DEMO_MARKETS = D.AbstractMarket[
    D.Market1X2(), D.MarketBTTS(),
    D.MarketOverUnder(1.5), D.MarketOverUnder(2.5), D.MarketOverUnder(3.5),
]

# Every gate's verdict lands here; §12 is the only place that decides pass/fail.
const DEMO_GATES = Pair{String, Bool}[]
demo_gate!(name, ok) = (push!(DEMO_GATES, name => Bool(ok)); Bool(ok))

function demo_banner(n, title)
    println()
    println("=" ^ 78)
    @printf("%d. %s\n", n, uppercase(title))
    println("=" ^ 78)
end


# %% ===========================================================================
# 3. Deterministic posteriors, fits, and a synthetic DataStore
# ==============================================================================
#
# Two engines with the SAME linear predictor and different observation layers, which is
# the pair the whole `marginals` dispatch exists to keep apart: a `PoissonObservation`
# builds a `CountLatents{Float64, Nothing}` and a `NegativeBinomialObservation` builds a
# `CountLatents{Float64, <:NamedTuple}`, and §5 shows them reaching different densities
# with no runtime branch.
#
# `HomeAwayDispersion` rather than `GlobalDispersion`, so `r_h ≠ r_a` and a kernel that
# silently used one for both sides fails §6 instead of passing by coincidence.

demo_banner(3, "deterministic posteriors, fits and a synthetic DataStore")

function demo_poisson_model()
    b = UE.CountModelBuilder(:ue_demo_poisson)
    UE.add!(b, PG.GlobalInterception(),
               PG.TimeDecayDynamics(days_half_life = 365.0),
               PG.GlobalHomeAdvantage())
    UE.add!(b, UE.WealthCovariate())
    return UE.build(b)
end

function demo_negbin_model()
    b = UE.CountModelBuilder(:ue_demo_negbin)
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
const DEMO_FS = UE.tpl_feature_set(n_teams = DEMO_N_TEAMS, n_seasons = DEMO_N_SEASONS,
                                   team_map = DEMO_TEAM_MAP)
const DEMO_OOS = [DEMO_FIXTURES[((k - 1) * DEMO_PER_FOLD + 1):(k * DEMO_PER_FOLD), :]
                  for k in 1:DEMO_N_FOLDS]
const DEMO_METAS = [D.SplitMetaData(1, "23/24", "24/25", 2, k, 0) for k in 1:DEMO_N_FOLDS]
const DEMO_FEATURE_SETS = [(DEMO_FS, DEMO_METAS[k]) for k in 1:DEMO_N_FOLDS]
const DEMO_SPLITTER = D.CVConfig(tournament_ids = [1], target_seasons = ["24/25"],
                                 history_seasons = 0, warmup_period = 1, dynamics_col = :match_week)

const DEMO_POIS_COLS   = UE.cb_chain_columns(DEMO_POIS_MODEL, DEMO_N_TEAMS)
const DEMO_NEGBIN_COLS = UE.cb_chain_columns(DEMO_NEGBIN_MODEL, DEMO_N_TEAMS)

"""
Build a `Fit` from replayed chains. `chain_offset` is the knob §8 uses: a per-chain
constant shift is between-chain variance with no within-chain justification, which is
exactly what R-hat measures.
"""
function demo_fit(name, model, cols; seed, chain_offset = 0.0)
    chs = [UE.demo_nuts_chain(cols; n_draws = DEMO_N_DRAWS, n_chains = DEMO_N_CHAINS,
                              seed = seed + k, chain_offset = chain_offset)
           for k in 1:DEMO_N_FOLDS]
    cfg = UE.FitConfig(name = name, model = model, splitter = DEMO_SPLITTER,
                       sampler = UE.ReplaySampler(chs), save_dir = tempdir())
    return (UE.fit_model(cfg; feature_sets = DEMO_FEATURE_SETS,
                         oos_fixtures = DEMO_OOS, quiet = true), chs)
end

const DEMO_POIS_FIT,   DEMO_POIS_CHAINS   = demo_fit("poisson_baseline", DEMO_POIS_MODEL,
                                                     DEMO_POIS_COLS; seed = DEMO_SEED)
const DEMO_NEGBIN_FIT, DEMO_NEGBIN_CHAINS = demo_fit("negbin_dispersed", DEMO_NEGBIN_MODEL,
                                                     DEMO_NEGBIN_COLS; seed = DEMO_SEED + 100)
const DEMO_BROKEN_FIT, DEMO_BROKEN_CHAINS = demo_fit("poisson_unconverged", DEMO_POIS_MODEL,
                                                     DEMO_POIS_COLS; seed = DEMO_SEED + 200,
                                                     chain_offset = 3.0)

const DEMO_POIS_L   = DEMO_POIS_FIT.latents
const DEMO_NEGBIN_L = DEMO_NEGBIN_FIT.latents

@printf("\n  fixtures        : %d over %d teams, %d folds × %d\n",
        DEMO_N_MATCHES, DEMO_N_TEAMS, DEMO_N_FOLDS, DEMO_PER_FOLD)
@printf("  posterior draws : %d × %d chains = %d flattened\n",
        DEMO_N_DRAWS, DEMO_N_CHAINS, DEMO_TOTAL)
@printf("  Poisson latents : %s\n", typeof(DEMO_POIS_L))
@printf("  NegBin  latents : %s\n", typeof(DEMO_NEGBIN_L))

demo_gate!("3a Poisson fit yields CountLatents{Float64,Nothing}",
           DEMO_POIS_L isa UE.CountLatents{Float64, Nothing})
demo_gate!("3b NegBin fit yields CountLatents{Float64,<:NamedTuple}",
           DEMO_NEGBIN_L isa UE.CountLatents{Float64, <:NamedTuple})
demo_gate!("3c the two containers cover the same fixtures",
           UE.latent_match_ids(DEMO_POIS_L) == UE.latent_match_ids(DEMO_NEGBIN_L))
demo_gate!("3d draws are the flattened chains×samples count",
           UE.n_draws(DEMO_POIS_L) == DEMO_TOTAL)
demo_gate!("3e r_h ≠ r_a — HomeAwayDispersion is genuinely asymmetric",
           DEMO_NEGBIN_L.observation_params.r_h != DEMO_NEGBIN_L.observation_params.r_a)

# The store. Scores are simulated from the Poisson container's own marginals, so the
# model is correctly specified for this data — which is why §5's RQR lands near N(0,1)
# and is a property of the fixture, not a result.
const DEMO_DS, DEMO_SCORES = UE.synthetic_datastore(DEMO_POIS_L, DEMO_MARKETS;
                                                    seed = DEMO_SEED + 1,
                                                    fixtures = DEMO_FIXTURES)

@printf("\n  ds.matches      : %d fixtures, scores %s\n",
        nrow(DEMO_DS.matches),
        string(extrema(DEMO_DS.matches.home_score .+ DEMO_DS.matches.away_score)))
@printf("  ds.odds         : %d rows, %d selections, %d markets\n",
        nrow(DEMO_DS.odds), length(unique(DEMO_DS.odds.selection)),
        length(DEMO_MARKETS))
@printf("  overround       : %.4f (flat 5%% vig)\n",
        mean(skipmissing(DEMO_DS.odds.overround_close)))

demo_gate!("3f the odds frame carries every enriched column the metrics read",
           all(c -> c in names(DEMO_DS.odds),
               ["prob_fair_close", "prob_implied_close", "odds_close", "is_winner",
                "clm_prob", "fair_odds_close"]))
demo_gate!("3g every fixture has a settled row in every market",
           nrow(DEMO_DS.odds) == DEMO_N_MATCHES * sum(UE.market_arity, DEMO_MARKETS))


# %% ===========================================================================
# 4. The typed indexes
# ==============================================================================
#
# This section carries the load-bearing floating-point claim. Everything in §6 reduces
# to "the same numbers, accumulated in the same order"; §4d is where the "same numbers"
# half is established, because the legacy frame stores a posterior as a contiguous
# `Vector{Float64}` and the typed container stores it as a ROW of a matrix. If
# `mean(view(M, i, :))` and `mean(M[i, :])` ever differed in the last bit, every CRPS
# and RQR row in §6 would fail and the reason would be three files away.

demo_banner(4, "the typed indexes")

const DEMO_ODDS_VIEW = UE.OddsView(DEMO_DS.odds)
const DEMO_OUTCOMES  = UE.MatchOutcomes(DEMO_DS.matches)
const DEMO_PROBS     = UE.market_probabilities(DEMO_POIS_L, DEMO_MARKETS; threaded = false)

println("  ", DEMO_ODDS_VIEW)
println("  ", DEMO_OUTCOMES)
println("  ", DEMO_PROBS)

demo_gate!("4a OddsView keeps every row and its order",
           DEMO_ODDS_VIEW.n == nrow(DEMO_DS.odds) &&
           DEMO_ODDS_VIEW.match_id == Int.(DEMO_DS.odds.match_id) &&
           DEMO_ODDS_VIEW.selection == DEMO_DS.odds.selection)
demo_gate!("4b every odds row has a fair price and a settled outcome",
           all(DEMO_ODDS_VIEW.has_fair) && all(>=(0), DEMO_ODDS_VIEW.is_winner))
demo_gate!("4c MatchOutcomes covers every fixture",
           length(DEMO_OUTCOMES) == DEMO_N_MATCHES &&
           all(i -> UE.outcome_of(DEMO_OUTCOMES, UE.latent_match_ids(DEMO_POIS_L)[i]) ==
                    DEMO_SCORES[i], 1:DEMO_N_MATCHES))

# --- 4.1 the floating-point assumption ------------------------------------------

let M = DEMO_POIS_L.λ_home,
    identical = all(i -> UE.posterior_mean(M, i) === mean(M[i, :]), 1:size(M, 1))
    demo_gate!("4d mean(view(M,i,:)) is BIT-IDENTICAL to mean(M[i,:])", identical)
    identical || @warn "the parity rows for CRPS and RQR in §6 depend on this"
end

let D3 = DEMO_PROBS.draws,
    ok = true
    for i in 1:size(D3, 2), c in 1:size(D3, 3)
        v = view(D3, :, i, c)
        (mean(v) === DEMO_PROBS.means[i, c] && mean(v) === mean(collect(v))) || (ok = false)
    end
    demo_gate!("4e the draw-major tensor reduces exactly like a Vector", ok)
end

# --- 4.2 the prices are 06's prices ----------------------------------------------

let ws = UE.GridWorkspace(UE.TPL_MAX_GOALS),
    S  = UE.alloc_score_grid(DEMO_POIS_L),
    worst = 0
    for i in 1:UE.n_matches(DEMO_POIS_L)
        UE.compute_score_grid!(S, ws, DEMO_POIS_L, i)
        for m in DEMO_MARKETS
            book = UE.price_market(S, m)
            for k in UE.market_keys(m)
                a = book[k]
                b = UE.prob_draws(DEMO_PROBS, UE.latent_match_ids(DEMO_POIS_L)[i], k)
                for j in eachindex(a)
                    u = UE.ulp_distance(a[j], b[j])
                    u > worst && (worst = u)
                end
            end
        end
    end
    @printf("\n  worst ULP, tensor vs a direct 06 price_market call : %d\n", worst)
    demo_gate!("4f the tensor holds exactly 06's prices (0 ULP)", worst == 0)
end

# --- 4.3 threading and the means-only build ---------------------------------------

let par = UE.market_probabilities(DEMO_POIS_L, DEMO_MARKETS; threaded = true)
    demo_gate!("4g threaded and sequential sweeps are bit-identical",
               par.means == DEMO_PROBS.means && par.draws == DEMO_PROBS.draws)
end

let lean = UE.market_probabilities(DEMO_POIS_L, DEMO_MARKETS;
                                   keep_draws = false, threaded = false)
    demo_gate!("4h keep_draws = false allocates no draw tensor",
               isempty(lean.draws) && lean.means == DEMO_PROBS.means)
    demo_gate!("4i and refuses to fake a draw vector",
               (try UE.prob_draws(lean, UE.latent_match_ids(DEMO_POIS_L)[1], :home)
                    false
                catch; true end))
    @printf("  tensor with draws : %.2f MiB     means only : %.2f MiB\n",
            UE.probability_bytes(DEMO_PROBS) / 1024^2,
            UE.probability_bytes(lean) / 1024^2)
end

# --- 4.4 the context builds only what the metrics asked for -------------------------

let only_counts = UE.evaluation_context(DEMO_POIS_L, DEMO_DS.odds, DEMO_DS.matches,
                                        [UE.CRPS(), UE.RQR()]; threaded = false),
    only_market = UE.evaluation_context(DEMO_POIS_L, DEMO_DS.odds, DEMO_DS.matches,
                                        [UE.LogLoss()]; threaded = false)
    demo_gate!("4j a CRPS+RQR batch prices no markets at all",
               isempty(only_counts.probs.selections) && only_counts.odds.n == 0)
    demo_gate!("4k a LogLoss batch keeps means only (needs_draws is false)",
               !only_market.probs.keep_draws && isempty(only_market.probs.draws))
    demo_gate!("4l a LogLoss batch prices only the three default markets",
               length(only_market.probs.markets) == 3)
end


# %% ===========================================================================
# 5. The six scoring rules
# ==============================================================================

demo_banner(5, "the six scoring rules")

const DEMO_ALL_METRICS = UE.AbstractScoringRule[
    UE.LogLoss(markets = DEMO_MARKETS),
    UE.LPD(markets = DEMO_MARKETS),
    UE.CRPS(),
    UE.RQR(n_sims = 1, seed = DEMO_SEED),
    UE.GLMEdge(markets = DEMO_MARKETS),
    UE.MIQ(),
]

const DEMO_POIS_CTX = UE.evaluation_context(DEMO_POIS_L, DEMO_DS.odds, DEMO_DS.matches,
                                            DEMO_ALL_METRICS; threaded = false)
const DEMO_NEGBIN_CTX = UE.evaluation_context(DEMO_NEGBIN_L, DEMO_DS.odds, DEMO_DS.matches,
                                              DEMO_ALL_METRICS; threaded = false)

const DEMO_POIS_RESULTS   = [UE.compute_metric(m, DEMO_POIS_CTX) for m in DEMO_ALL_METRICS]
const DEMO_NEGBIN_RESULTS = [UE.compute_metric(m, DEMO_NEGBIN_CTX) for m in DEMO_ALL_METRICS]

let ll = DEMO_POIS_RESULTS[1].overall,
    lp = DEMO_POIS_RESULTS[2].overall,
    cr = DEMO_POIS_RESULTS[3],
    rq = DEMO_POIS_RESULTS[4],
    ge = DEMO_POIS_RESULTS[5],
    mq = DEMO_POIS_RESULTS[6]

    println("\n  Poisson container")
    @printf("    logloss  model %.6f  market %.6f  diff %+.6f   (n = %d)\n",
            ll.model_ll, ll.market_ll, ll.diff_ll, ll.n_obs)
    @printf("    lpd      model %.6f  market %.6f  diff %+.6f   elpd %.3f\n",
            lp.model_lpd, lp.market_lpd, lp.diff_lpd, lp.elpd)
    @printf("    crps     home %.5f   away %.5f   all %.5f\n",
            cr.home.mean, cr.away.mean, cr.all.mean)
    @printf("    rqr      all: mean %+.4f  sd %.4f  skew %+.4f  kurt %+.4f  W %.4f  p %.4f\n",
            rq.all.mean, rq.all.std, rq.all.skewness, rq.all.kurtosis,
            rq.all.shapiro_w, rq.all.shapiro_p)
    @printf("    glmedge  intercept %+.4f   prob_fair %+.4f   spread_fair %+.4f (p %.4f)\n",
            ge.intercept.coef, ge.prob_fair.coef, ge.spread_fair.coef,
            ge.spread_fair.p_value)
    @printf("    miq      all: gap %+.4f  KS-D %.4f  p %.4f  (%d won / %d lost)\n",
            mq.all.mean_gap, mq.all.ks_d_stat, mq.all.p_value,
            mq.all.n_winners, mq.all.n_losers)

    n_rows = nrow(DEMO_DS.odds)
    demo_gate!("5a LogLoss scores every settled odds row", ll.n_obs == n_rows)
    demo_gate!("5b LPD scores the same rows and its elpd is their sum",
               lp.n_obs == n_rows && isapprox(lp.elpd, lp.model_lpd * lp.n_obs; rtol = 1e-12))
    demo_gate!("5c LogLoss and LPD agree on direction",
               sign(ll.diff_ll) == -sign(lp.diff_lpd))
    demo_gate!("5d CRPS is positive and `all` is the home/away average",
               cr.home.mean > 0 && cr.away.mean > 0 &&
               isapprox(cr.all.mean, (cr.home.mean + cr.away.mean) / 2; rtol = 1e-12))
    demo_gate!("5e RQR on a correctly-specified fixture is near N(0,1)",
               abs(rq.all.mean) < 0.35 && 0.6 < rq.all.std < 1.5)
    demo_gate!("5f GLMEdge fits and reports three finite coefficients",
               isfinite(ge.intercept.coef) && isfinite(ge.prob_fair.coef) &&
               isfinite(ge.spread_fair.coef) && ge.n_obs == n_rows)
    demo_gate!("5g MIQ partitions every scored row into winners and losers",
               mq.all.n_winners + mq.all.n_losers == n_rows)
    demo_gate!("5h MIQ reports all twelve selection groups",
               all(s -> getproperty(mq, s).n_winners + getproperty(mq, s).n_losers > 0,
                   UE.MIQ_FIELD_SELECTIONS))
end

# --- 5.1 the two containers reach different marginals, by dispatch -----------------

let (ph, pa) = UE.marginals(DEMO_POIS_L, 1),
    (nh, na) = UE.marginals(DEMO_NEGBIN_L, 1)
    @printf("\n  marginals(CountLatents{_,Nothing},      1) → %s / %s\n",
            nameof(typeof(ph)), nameof(typeof(pa)))
    @printf("  marginals(CountLatents{_,<:NamedTuple}, 1) → %s / %s\n",
            nameof(typeof(nh)), nameof(typeof(na)))
    demo_gate!("5i the Poisson container reaches Poisson",
               ph isa Poisson && pa isa Poisson)
    demo_gate!("5j the NegBin container reaches NegativeBinomial",
               nh isa NegativeBinomial && na isa NegativeBinomial)
    demo_gate!("5k the NegBin marginal uses p = r/(r+λ) per side",
               isapprox(mean(nh), UE.posterior_mean(DEMO_NEGBIN_L.λ_home, 1); rtol = 1e-10))
    demo_gate!("5l NegBin CRPS differs from Poisson CRPS on the same scores",
               DEMO_NEGBIN_RESULTS[3].all.mean != DEMO_POIS_RESULTS[3].all.mean)
end

# --- 5.2 the briefing's score-target LPD -------------------------------------------

let joint = UE.compute_metric(UE.LPD(target = :score), DEMO_POIS_CTX).overall,
    marg  = DEMO_POIS_RESULTS[2].overall
    @printf("\n  LPD(target = :score)  model %.6f  elpd %.3f  (n = %d)\n",
            joint.model_lpd, joint.elpd, joint.n_obs)
    demo_gate!("5m the score-target LPD scores every fixture once",
               joint.n_obs == DEMO_N_MATCHES)
    demo_gate!("5n it is a joint density, so strictly below any marginal market LPD",
               joint.model_lpd < marg.model_lpd)
    demo_gate!("5o it declares no market baseline rather than inventing one",
               isnan(joint.market_lpd) && isnan(joint.diff_lpd))
    demo_gate!("5p and gets its own column name",
               UE.metric_column_suffix(UE.LPD(target = :score)) == "_score")
end

# --- 5.3 the selection filter --------------------------------------------------------

let over = UE.compute_metric(UE.LogLoss(:over_25), DEMO_POIS_L,
                             DEMO_DS.odds, DEMO_DS.matches).overall
    @printf("\n  LogLoss(:over_25)  model %.6f  market %.6f  (n = %d)\n",
            over.model_ll, over.market_ll, over.n_obs)
    demo_gate!("5q a legacy selection filter scores exactly its own rows",
               over.n_obs == DEMO_N_MATCHES)
    demo_gate!("5r and tells the pricer it needs only that market",
               UE.scored_markets(UE.LogLoss(:over_25)) == [D.MarketOverUnder(2.5)])
end


# %% ===========================================================================
# 6. Mathematical parity against src/evaluation/
# ==============================================================================
#
# The legacy side is the LIVE `BayesianFootball.Evaluation` path. `legacy_experiment`
# builds a real `Experiments.ExperimentResults` — the type `compute_metric`'s signature
# demands — and `legacy_latent_states` wraps the same typed container's
# `to_legacy_dataframe` output, which `06/r01_demo.jl` holds at 0 ULP against a frame
# built by `src`'s own `_latent_state_dict_to_df`.

demo_banner(6, "mathematical parity against src/evaluation/")

const DEMO_LEGACY_DIR = mktempdir(; prefix = "ue_demo_legacy_")

const DEMO_LEGACY_POIS = UE.legacy_experiment("poisson_baseline", DEMO_POIS_MODEL,
                                              DEMO_POIS_CHAINS, DEMO_METAS, DEMO_SPLITTER)
const DEMO_LEGACY_NEGBIN = UE.legacy_experiment("negbin_dispersed", DEMO_NEGBIN_MODEL,
                                                DEMO_NEGBIN_CHAINS, DEMO_METAS,
                                                DEMO_SPLITTER; save_dir = DEMO_LEGACY_DIR)

# The cached `oos_latents.jls` a legacy run leaves behind when it was saved with
# `compute_oos = true`. §9.4 loads a genuine `ExperimentResults` through the bridge, and
# this is the middle branch of `_ue_as_fit`'s three-way search — the one that lets an
# archived run be re-scored without a live database.
UE.save_latents(DEMO_LEGACY_DIR, DEMO_NEGBIN_L)

# --- 6.1 scope alignment, checked before anything is compared ------------------------

let market_metrics = [UE.LogLoss(markets = DEMO_MARKETS), UE.LPD(markets = DEMO_MARKETS),
                      UE.GLMEdge(markets = DEMO_MARKETS), UE.MIQ(),
                      UE.LogLoss(:over_25)],
    all_ok = true
    println()
    for m in market_metrics
        ok, offenders = UE.parity_scope_ok(m, DEMO_DS.odds)
        ok || (all_ok = false)
        @printf("  scope  %-24s %s%s\n",
                UE.get_metric_method_name(m), ok ? "aligned" : "MISMATCH ",
                ok ? "" : string(offenders))
    end
    demo_gate!("6a src and this framework would score the same rows", all_ok)
end

# --- 6.2 the NegBin arm: all six metrics -----------------------------------------------
#
# NegBin first, because it is the ONLY arm on which `src` can compute all six — see
# §10a for why CRPS and RQR cannot be computed for a Poisson model at all.

const DEMO_PARITY_METRICS = UE.AbstractScoringRule[
    UE.LogLoss(markets = DEMO_MARKETS),
    UE.LPD(markets = DEMO_MARKETS),
    UE.CRPS(),
    UE.RQR(n_sims = 1, seed = DEMO_SEED),
    UE.GLMEdge(markets = DEMO_MARKETS),
    UE.MIQ(),
    UE.LogLoss(:over_25),
    UE.LPD([:btts_yes, :btts_no]),
]

const DEMO_PARITY_NEGBIN = UE.parity_report(DEMO_PARITY_METRICS, DEMO_NEGBIN_L,
                                            DEMO_LEGACY_NEGBIN, DEMO_DS;
                                            model = DEMO_NEGBIN_MODEL)

demo_gate!("6b every metric agrees with src on the NegBin container",
           UE.metric_parity_table(DEMO_PARITY_NEGBIN;
                                  title = "PARITY — NegBin container, src vs typed kernels"))

# --- 6.3 the Poisson arm: the four market metrics ---------------------------------------

const DEMO_PARITY_POIS = UE.parity_report(
    UE.AbstractScoringRule[UE.LogLoss(markets = DEMO_MARKETS),
                           UE.LPD(markets = DEMO_MARKETS),
                           UE.GLMEdge(markets = DEMO_MARKETS),
                           UE.MIQ()],
    DEMO_POIS_L, DEMO_LEGACY_POIS, DEMO_DS; model = DEMO_POIS_MODEL)

demo_gate!("6c and on the Poisson container",
           UE.metric_parity_table(DEMO_PARITY_POIS;
                                  title = "PARITY — Poisson container, src vs typed kernels"))

# --- 6.4 negative controls, sized to the gate ------------------------------------------
#
# A parity harness that passes everything proves nothing. Two controls, and the pair is
# the point — one shows the gate FIRES, the other shows honestly where it stops.
#
# The gate is `max |Δ| ≤ 1e-12` and NOT 0 ULP, for the reason in `MetricParityRow`'s
# docstring: `src` sums its rows in an order `innerjoin` does not guarantee, so the last
# bit of a mean is not reproducible by construction. A harness that hid that behind a
# loosened tolerance and called it bit-exact would be lying; one that quoted the
# tolerance without showing its resolution would be evading. So both are measured.

let base_ids = UE.latent_match_ids(DEMO_POIS_L),
    metric = UE.LogLoss(markets = DEMO_MARKETS),
    perturb(f) = begin
        λ = copy(DEMO_POIS_L.λ_home)
        f(λ)
        UE.CountLatents(base_ids, λ, DEMO_POIS_L.λ_away)
    end,
    # ONE SIDE ONLY. `src` scores the reference container, this framework scores the
    # perturbed one. Perturbing both — which is what feeding the perturbed container to
    # `parity_report` would do — makes the two agree about the perturbed posterior and
    # the control reports "pass" having tested nothing.
    control(f) = UE.parity_control(metric, DEMO_POIS_L, perturb(f), DEMO_LEGACY_POIS,
                                   DEMO_DS; model = DEMO_POIS_MODEL)

    detectable = control(λ -> (λ[1, :] .*= (1 + 1e-9)))
    one_ulp    = control(λ -> (λ[1, 1] = nextfloat(λ[1, 1])))

    println()
    @printf("  control: fixture 1's λ_home scaled by 1 + 1e-9  → max |Δ| %.3e   %s\n",
            detectable.max_abs, detectable.pass ? "PASSED (bad!)" : "caught")
    @printf("  control: one λ draw moved by one ULP            → max |Δ| %.3e   %s\n",
            one_ulp.max_abs,
            one_ulp.pass ? "below the 1e-12 gate — stated, not hidden" : "caught")

    demo_gate!("6d a 1e-9 relative change in one fixture's λ is caught", !detectable.pass)
    demo_gate!("6e a single-ULP change is below the gate, and the harness says so",
               one_ulp.max_abs < 1e-12)
end


# %% ===========================================================================
# 7. Determinism and reproducibility
# ==============================================================================

demo_banner(7, "determinism and reproducibility")

let a = UE.compute_metric(UE.RQR(n_sims = 1, seed = 99), DEMO_POIS_CTX),
    b = UE.compute_metric(UE.RQR(n_sims = 1, seed = 99), DEMO_POIS_CTX)
    demo_gate!("7a RQR with a fixed seed is bit-reproducible",
               a.all.shapiro_w === b.all.shapiro_w && a.all.mean === b.all.mean &&
               a.home.skewness === b.home.skewness)
end

let one  = UE.compute_metric(UE.RQR(n_sims = 1,  seed = 99), DEMO_POIS_CTX),
    many = UE.compute_metric(UE.RQR(n_sims = 40, seed = 99), DEMO_POIS_CTX)
    @printf("  n_sims =  1  →  W %.5f   p %.5f\n", one.all.shapiro_w, one.all.shapiro_p)
    @printf("  n_sims = 40  →  W %.5f   p %.5f   (Monte-Carlo noise averaged out)\n",
            many.all.shapiro_w, many.all.shapiro_p)
    demo_gate!("7b replicates change the number and keep it a number",
               many.all.shapiro_w != one.all.shapiro_w &&
               isfinite(many.all.shapiro_w) && 0 <= many.all.shapiro_w <= 1)
    demo_gate!("7c a different seed gives a different randomisation",
               UE.compute_metric(UE.RQR(n_sims = 1, seed = 7), DEMO_POIS_CTX
                                ).all.shapiro_w != one.all.shapiro_w)
end

let first  = UE.evaluate_fits(DEMO_ALL_METRICS, [DEMO_POIS_FIT, DEMO_NEGBIN_FIT], DEMO_DS;
                              quiet = true, show_tables = false),
    second = UE.evaluate_fits(DEMO_ALL_METRICS, [DEMO_POIS_FIT, DEMO_NEGBIN_FIT], DEMO_DS;
                              quiet = true, show_tables = false)
    same = names(first.rows) == names(second.rows) &&
           all(c -> isequal(first.rows[!, c], second.rows[!, c]), names(first.rows))
    demo_gate!("7d a whole scorecard is reproducible, column for column", same)
    demo_gate!("7e evaluating never perturbs the caller's global RNG",
               (Random.seed!(1234); x = rand();
                UE.compute_metric(UE.RQR(n_sims = 3), DEMO_POIS_CTX);
                Random.seed!(1234); x === rand()))
end


# %% ===========================================================================
# 8. Convergence gating
# ==============================================================================
#
# The one thing `src/evaluation/` cannot do at all. `Fit.diagnostics` is a field (07),
# so this needs no chains, no DataStore and no re-audit.

demo_banner(8, "convergence gating")

@printf("  %-22s %-6s  max R-hat %8.4f   failed gates: %s\n", "poisson_baseline",
        DEMO_POIS_FIT.diagnostics.passed ? "PASS" : "FAIL",
        DEMO_POIS_FIT.diagnostics.max_rhat,
        join(DEMO_POIS_FIT.diagnostics.failed_gates, ", "))
@printf("  %-22s %-6s  max R-hat %8.4f   failed gates: %s\n", "poisson_unconverged",
        DEMO_BROKEN_FIT.diagnostics.passed ? "PASS" : "FAIL",
        DEMO_BROKEN_FIT.diagnostics.max_rhat,
        join(DEMO_BROKEN_FIT.diagnostics.failed_gates, ", "))

demo_gate!("8a the healthy fits pass their gates",
           DEMO_POIS_FIT.diagnostics.passed && DEMO_NEGBIN_FIT.diagnostics.passed)
demo_gate!("8b the offset-chain fit fails, on R-hat",
           !DEMO_BROKEN_FIT.diagnostics.passed &&
           "R-hat" in DEMO_BROKEN_FIT.diagnostics.failed_gates)

const DEMO_FITS = [DEMO_POIS_FIT, DEMO_NEGBIN_FIT, DEMO_BROKEN_FIT]

const DEMO_SCORECARD = UE.evaluate_fits(DEMO_ALL_METRICS, DEMO_FITS, DEMO_DS)

demo_gate!("8c an unconverged fit is EXCLUDED by default",
           nrow(DEMO_SCORECARD.rows) == 2 &&
           DEMO_SCORECARD.excluded == ["poisson_unconverged"] &&
           !("poisson_unconverged" in DEMO_SCORECARD.rows.model))
demo_gate!("8d but it still has a row in the convergence frame",
           nrow(DEMO_SCORECARD.convergence) == 3 &&
           "poisson_unconverged" in DEMO_SCORECARD.convergence.model)
demo_gate!("8e no metric raised", isempty(DEMO_SCORECARD.errors))
demo_gate!("8f every metric produced its columns",
           all(f -> any(n -> startswith(n, f), names(DEMO_SCORECARD.rows)),
               ("logloss", "lpd", "crps", "rqr", "glmedge", "miq")))
demo_gate!("8g the verdict travels with the numbers",
           all(c -> c in names(DEMO_SCORECARD.rows), ["converged", "max_rhat"]) &&
           all(DEMO_SCORECARD.rows.converged))

let flagged = UE.evaluate_fits(DEMO_ALL_METRICS, DEMO_FITS, DEMO_DS;
                               require_converged = false, quiet = true, show_tables = false)
    demo_gate!("8h require_converged = false scores it and flags it",
               nrow(flagged.rows) == 3 && isempty(flagged.excluded) &&
               count(!, flagged.rows.converged) == 1)
end

let threw = try
        UE.compute_metric(UE.LogLoss(), DEMO_BROKEN_FIT, DEMO_DS; require_converged = true)
        false
    catch e
        e isa UE.ConvergenceRefusal
    end
    demo_gate!("8i a single-fit call can refuse outright", threw)
    demo_gate!("8j and scores it when asked to",
               UE.compute_metric(UE.LogLoss(), DEMO_BROKEN_FIT, DEMO_DS
                                ).overall.n_obs > 0)
end

let unaudited = UE.Fit(DEMO_POIS_FIT.config, DEMO_POIS_FIT.folds, DEMO_POIS_L,
                       nothing, DEMO_POIS_FIT.metadata, DEMO_POIS_FIT.save_path),
    (passed, gates, _) = UE.convergence_verdict(unaudited)
    demo_gate!("8k an UNAUDITED container is not treated as converged",
               !passed && gates == ["no audit"])
end

println("\n  leaderboard on the market LPD:")
display(UE.leaderboard(DEMO_SCORECARD, :lpd_overall_diff_lpd; higher_is_better = true))
println()


# %% ===========================================================================
# 9. Backward compatibility
# ==============================================================================
#
# `LegacyCallSite`'s body is the pattern every evaluation runner in this repository
# writes, copied verbatim. Only its import line differs — see the header of
# `l04_compat_bridge.jl` for why nothing can make two modules answer to the name
# `Evaluation`.

demo_banner(9, "backward compatibility")

module LegacyCallSite

import BayesianFootball
using ..UnifiedEvaluation.Legacy        # binds `Evaluation`
using DataFrames

"The legacy batch call, unchanged."
function run_batch(experiments, ds)
    metrics = [Evaluation.LogLoss(),
               Evaluation.LPD(:over_25),
               Evaluation.CRPS(),
               Evaluation.GLMEdge(:home)]
    return Evaluation.evaluate_experiments(metrics, experiments, ds)
end

"The legacy single-metric call with latents already in hand."
single(metric, exp, ds, latents) = Evaluation.compute_metric(metric, exp, ds, latents)

"The legacy flattener."
row(exp, metric, result) = Evaluation.to_dataframe_row(exp, metric, result)
row(exp, result) = Evaluation.to_dataframe_row(exp, result)

"The legacy display."
summarise(df, family) = Evaluation.display_summary_metric(df, family)

"The legacy names, referenced the legacy way."
const TRIGGERS = (Evaluation.LogLoss, Evaluation.LPD, Evaluation.CRPS,
                  Evaluation.RQR, Evaluation.GLMEdge, Evaluation.MIQ)
const RESULTS  = (Evaluation.LogLossResult, Evaluation.LPDResult,
                  Evaluation.CRPSResults, Evaluation.CRPSResult,
                  Evaluation.RQRResult, Evaluation.GLMEdgeResult, Evaluation.MIQResult)

name_of(x) = Evaluation.get_metric_method_name(x)

end # module LegacyCallSite

const LCS = LegacyCallSite

demo_gate!("9a the legacy `Evaluation` name binds",
           LCS.TRIGGERS[1] === UE.LogLoss && length(LCS.RESULTS) == 7)
demo_gate!("9b `CRPSResult` and `CRPSResults` are the same type",
           UE.CRPSResult === UE.CRPSResults)

const DEMO_LEGACY_DF = LCS.run_batch([DEMO_POIS_FIT, DEMO_NEGBIN_FIT], DEMO_DS)

println("\n  evaluate_experiments returned a ", nrow(DEMO_LEGACY_DF), "×",
        ncol(DEMO_LEGACY_DF), " DataFrame")
println("  columns: ", join(names(DEMO_LEGACY_DF), ", "))

demo_gate!("9c evaluate_experiments returns the legacy wide DataFrame",
           DEMO_LEGACY_DF isa DataFrame && nrow(DEMO_LEGACY_DF) == 2 &&
           DEMO_LEGACY_DF.model == sort(["poisson_baseline", "negbin_dispersed"]))
demo_gate!("9d and no convergence columns unless asked",
           !any(c -> c in names(DEMO_LEGACY_DF), ["converged", "max_rhat", "min_ess_bulk"]))
demo_gate!("9e the legacy column names are unchanged",
           all(c -> c in names(DEMO_LEGACY_DF),
               ["logloss_overall_model_ll", "logloss_overall_diff_ll",
                "lpd_over_25_overall_diff_lpd", "crps_all_mean",
                "glmedge_home_spread_fair_coef"]))

# --- 9.1 column-name identity, against src's own translator --------------------------
#
# The strongest form of the compatibility claim: run BOTH translators over BOTH sides'
# results and require the produced column names to be identical.

let legacy_ll = UE.legacy_compute(UE.legacy_metric(UE.LogLoss(:over_25)),
                                  DEMO_LEGACY_POIS, DEMO_DS,
                                  UE.legacy_latent_states(DEMO_POIS_L, DEMO_POIS_MODEL)),
    new_ll = UE.compute_metric(UE.LogLoss(:over_25), DEMO_POIS_CTX),
    legacy_row = UE.UE_Eval.to_dataframe_row(DEMO_LEGACY_POIS,
                                             UE.UE_Eval.LogLoss([:over_25]), legacy_ll),
    new_row = LCS.row("poisson_baseline", UE.LogLoss(:over_25), new_ll)
    demo_gate!("9f to_dataframe_row produces src's column names exactly",
               propertynames(legacy_row) == propertynames(new_row))
    demo_gate!("9g and src's values",
               all(k -> k === :model || getproperty(legacy_row, k) ≈ getproperty(new_row, k),
                   propertynames(new_row)))
end

let two_arg = LCS.row("poisson_baseline", DEMO_POIS_RESULTS[1])
    demo_gate!("9h the 2-argument to_dataframe_row drops the selection suffix",
               :logloss_overall_model_ll in propertynames(two_arg))
end

# --- 9.2 the four-argument compute_metric, on every latents shape ----------------------

let typed  = LCS.single(UE.CRPS(), DEMO_NEGBIN_FIT, DEMO_DS, DEMO_NEGBIN_L),
    wrapped = LCS.single(UE.CRPS(), DEMO_NEGBIN_FIT, DEMO_DS,
                         UE.LatentStates(DEMO_NEGBIN_L, DEMO_NEGBIN_MODEL)),
    legacy_ls = LCS.single(UE.CRPS(), DEMO_NEGBIN_FIT, DEMO_DS,
                           UE.legacy_latent_states(DEMO_NEGBIN_L, DEMO_NEGBIN_MODEL)),
    frame = LCS.single(UE.CRPS(), DEMO_NEGBIN_FIT, DEMO_DS,
                       UE.to_legacy_dataframe(DEMO_NEGBIN_L))
    demo_gate!("9i the 4-arg form accepts a typed container, both LatentStates, and a frame",
               typed.all.mean === wrapped.all.mean === legacy_ls.all.mean === frame.all.mean)
end

# --- 9.3 display_summary_metric ------------------------------------------------------

for fam in (:logloss, :lpd, :crps, :glmedge)
    LCS.summarise(DEMO_LEGACY_DF, fam)
end
println()

demo_gate!("9j display_summary_metric handles every legacy family",
           all(fam -> UE._ue_summary_columns(DEMO_LEGACY_DF, fam)[1] !== nothing,
               (:rqr, :logloss, :glmedge, :crps, :lpd)))
demo_gate!("9k and rejects an unknown one the way src does",
           UE._ue_summary_columns(DEMO_LEGACY_DF, :nonsense)[1] === nothing)
demo_gate!("9l :miq is a family here and is not in src",
           UE._ue_summary_columns(DEMO_LEGACY_DF, :miq)[1] !== nothing)

# --- 9.4 a genuine legacy ExperimentResults, evaluated -----------------------------------

let df = UE.evaluate_experiments([UE.LogLoss(markets = DEMO_MARKETS)],
                                 [DEMO_LEGACY_NEGBIN], DEMO_DS;
                                 quiet = true, show_tables = false)
    demo_gate!("9m a real src ExperimentResults is upgraded, audited and scored",
               nrow(df) == 1 && df.model[1] == "negbin_dispersed" &&
               isfinite(df.logloss_overall_model_ll[1]))
    demo_gate!("9n and its numbers match the Fit's",
               df.logloss_overall_model_ll[1] ≈
               DEMO_NEGBIN_RESULTS[1].overall.model_ll)
end


# %% ===========================================================================
# 10. Live defects in src/evaluation/
# ==============================================================================
#
# Reproduced, not described. If one of these is fixed upstream, the probe stops
# returning a failure and the gate below flips — which is the point: a claim about
# `src` that cannot go stale silently.

demo_banner(10, "live defects in src/evaluation/")

let (raised, msg) = UE.probe_poisson_latent_columns(DEMO_POIS_MODEL, DEMO_POIS_L)
    println("  1. CRPS and RQR cannot be computed for a Poisson model")
    println("     Predictions.get_latent_column_symbols has methods for")
    println("     AbstractNegBinModel only (negativebinomial.jl:29, frank_copula.jl:77);")
    println("     crps.jl:69 and rqr.jl:89 call it unconditionally.")
    println("     → ", msg)
    demo_gate!("10a reproduced: src CRPS/RQR raise on a Poisson model", raised)
end

# A store carrying 1X2 only. `MIQResult` still reports twelve selections, so eight of
# them come back all-`missing` — which is the ordinary case for any league whose odds
# feed does not quote Over/Under 1.5 and 3.5, not a contrived one.
const DEMO_DS_1X2 = D.DataStore(
    DEMO_DS.segment, DEMO_DS.matches, DataFrame(),
    filter(:market_name => ==("1X2"), DEMO_DS.odds),
    DataFrame(), DataFrame(), DataFrame(), DataFrame(), DataFrame())

let miq_sparse = UE.UE_Eval.compute_metric(UE.UE_Eval.MIQ(), DEMO_LEGACY_POIS, DEMO_DS_1X2,
                                           UE.legacy_latent_states(DEMO_POIS_L,
                                                                   DEMO_POIS_MODEL)),
    (raised, msg) = UE.probe_miq_translator(DEMO_LEGACY_POIS, miq_sparse),
    ours = UE.compute_metric(UE.MIQ(markets = [D.Market1X2()]),
                             DEMO_POIS_L, DEMO_DS_1X2.odds, DEMO_DS_1X2.matches),
    ours_row = UE.to_dataframe_row("poisson_baseline", UE.MIQ(), ours)

    println("\n  2. An MIQResult with an empty selection group cannot be flattened")
    println("     MIQStats' fields are Union{Missing,Float64} (miq.jl:12-18) and")
    println("     Evaluation.unroll has no Missing method (translator.jl:6,11).")
    println("     On a 1X2-only store eight of the twelve groups are empty.")
    println("     → ", msg)
    demo_gate!("10b reproduced: src's translator raises on a missing MIQ field", raised)
    demo_gate!("10b′ this framework flattens the same result",
               ours_row isa NamedTuple &&
               ismissing(getproperty(ours_row, :miq_over_15_mean_gap)))
end

let (differs, δ) = UE.probe_rqr_nondeterminism(DEMO_LEGACY_NEGBIN, DEMO_DS,
                                               UE.legacy_latent_states(DEMO_NEGBIN_L,
                                                                       DEMO_NEGBIN_MODEL))
    println("\n  3. src's RQR is not reproducible")
    println("     rqr.jl:50 draws from the unseeded global RNG, so two consecutive")
    println("     calls on identical inputs disagree.")
    @printf("     → two calls differ by %.6e in (mean + Shapiro-W)\n", δ)
    demo_gate!("10c reproduced: src's RQR differs between two identical calls", differs)
end

println("\n  None of the three is fixed in `src` by this prototype. Each is avoided")
println("  here by construction — see README.md § What this framework does differently.")


# %% ===========================================================================
# 11. What it costs
# ==============================================================================
#
# A 24-fixture synthetic fold is an indication, not a production measurement, and no
# gate is attached to these numbers. The SHAPE is what matters: the legacy path's cost
# is per-metric and covers forty markets, the new path's is per-batch and covers five.

demo_banner(11, "what it costs")

const DEMO_COSTS = UE.CostRow[
    UE.measure_cost("1 metric  (LogLoss)", [UE.LogLoss(markets = DEMO_MARKETS)],
                    DEMO_NEGBIN_L, DEMO_LEGACY_NEGBIN, DEMO_DS; model = DEMO_NEGBIN_MODEL),
    UE.measure_cost("6 metrics (all)", DEMO_ALL_METRICS,
                    DEMO_NEGBIN_L, DEMO_LEGACY_NEGBIN, DEMO_DS; model = DEMO_NEGBIN_MODEL),
]

UE.cost_table(DEMO_COSTS; title = "COST — src/evaluation vs typed kernels, 24 fixtures × 1200 draws")

println()
println("  The bytes column is the posterior-probability materialisation on each side:")
println("  `src`'s PPD frame over every market in DEFAULT_MARKET_CONFIG, against this")
println("  framework's tensor over the five markets the metrics named. On a real fold")
println("  the ratio grows with the number of markets the store carries and shrinks")
println("  with the number a metric batch actually wants.")

let ctx = UE.evaluation_context(DEMO_NEGBIN_L, DEMO_DS.odds, DEMO_DS.matches,
                                DEMO_ALL_METRICS; threaded = false)
    demo_gate!("11a the batch prices the posterior exactly once for six metrics",
               length(ctx.probs.markets) == length(DEMO_MARKETS))
    demo_gate!("11b and the new path is not slower than the legacy one",
               all(r -> UE.speedup(r) >= 1.0, DEMO_COSTS))
end


# %% ===========================================================================
# 12. Final report
# ==============================================================================

demo_banner(12, "final report")

# The only thing this runner wrote outside memory: §6's cached `oos_latents.jls`.
try
    rm(DEMO_LEGACY_DIR; recursive = true, force = true)
catch e
    @warn "Could not remove scratch directory $DEMO_LEGACY_DIR" exception = e
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
        println("  Six scoring rules, computed from typed posterior containers with no")
        println("  PPD frame, no four-column join and no `dropmissing` copy, agree with")
        println("  `src/evaluation/`'s answers leaf by leaf — on both a Poisson and a")
        println("  negative-binomial container, which reach different marginals by")
        println("  DISPATCH rather than by probing a DataFrame for an `r` column.")
        println()
        println("  Convergence is a gate, not a footnote: an unconverged fit is excluded")
        println("  from the leaderboard by default, flagged on request, and refused")
        println("  outright by a single-fit call that asked for gating — while still")
        println("  appearing, with its failed gates named, in the convergence frame.")
        println()
        println("  Every legacy call site runs verbatim and produces `src`'s column")
        println("  names character for character. Three live defects in `src/evaluation/`")
        println("  are reproduced rather than described.")
        println()
        println("  NOT SHOWN, and not claimed: that any of these models fits anything, or")
        println("  that any of these numbers is good. The posteriors are prior draws with")
        println("  a fixed seed and the market is the model's own prices, perturbed.")
    else
        println("  RESULT: FAIL — see the failing gates above.")
    end

    # Non-zero exit so this runner is usable as a CI check, but only when run as a
    # script: an `include` from a REPL should not kill the session.
    if abspath(PROGRAM_FILE) == @__FILE__
        exit(all_ok ? 0 : 1)
    end
end
