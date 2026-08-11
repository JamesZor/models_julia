# current_development/orderbook_layer2/r02_train_ireland.jl
#
# WP2. Train `src_sup40_sw40` on Ireland 79 and 718, independently.
#
# ---------------------------------------------------------------------------------------------
# WHAT THIS IS AND IS NOT
# ---------------------------------------------------------------------------------------------
#
# This is NOT a Layer-1 experiment. The engine, its hyperparameters and its sampler settings are
# taken as SETTLED from `split_market_pillar/r21` (`src_sup40_sw40`, the best residual-edge
# coefficient in that 17-cell grid at essentially tied LogLoss/LPD). Nothing here re-opens that.
#
# The only job is to produce two `ExperimentResults` whose folds cover the order-book window, so
# the Layer-2 stream has latents to price with.
#
# ---------------------------------------------------------------------------------------------
# THE ONE PLACE THE TWO LEAGUES DIFFER, AND WHY
# ---------------------------------------------------------------------------------------------
#
# r21 trained the market pillar on SofaScore `ds.odds` and evaluated CLV against Betfair, which
# keeps the two feeds cleanly separated. That works for 79. It CANNOT work for 718:
#
#   SofaScore O/U ladder, 718     0 matches from 2022 onward
#   SofaScore 1X2, 718 in 2025    27 / 180
#
# `MarketSmileFeature(Kmax=4)` inverts the O/U ladder per strike. On 718's SofaScore feed there
# is no ladder to invert, so `smile_weight = 0.4` would be anchoring to an empty feature — a
# silently mis-specified model, not a degraded one.
#
# So 718 trains on **Betfair** (`betfair.odds_history`, which does carry the full 0.5–5.5 ladder)
# and 79 stays r21-exact on SofaScore.
#
# ### The window, and why it is not the close
#
# `market_extractors.jl:71` reads `prob_fair_close` — so whatever is passed as `close_window` IS
# the training pillar, whatever it is named. Training on the actual close would be circular: the
# Layer-2 evaluation measures CLV *against* the close, so the model would be scored against its
# own input.
#
# `close_window = (-360, -180)` ends exactly where the WP4 decision grid begins (T-180), so the
# pillar is strictly outside the window the model is evaluated on. Measured cost of that choice:
#
#   window            matches with an O/U ladder (718, 2023-26)
#   (-1440, -360)      71     <- too thin to use
#   ( -720, -360)     232
#   ( -360, -180)     276     <- chosen: principled separation, acceptable coverage
#   ( -180,  -60)     312     <- more data, but overlaps the decision window
#
# ### The asymmetry this leaves, stated plainly
#
# 79's pillar is SofaScore at ~100% coverage; 718's is Betfair at ~54%. The two models are
# therefore anchored to different feeds at different densities. **WP4-WP6 must report per league,
# never pooled** — a pooled difference would be confounded by the feed rather than by the league.
# This was a deliberate choice (the alternative was dropping 718 and halving the corpus); it is
# recorded here so no downstream reader has to rediscover it.
#
# ---------------------------------------------------------------------------------------------
# DATASTORE PINNING (trap T1)
# ---------------------------------------------------------------------------------------------
#
# `load_datastore_cached` has a 48h TTL and rebuilds SILENTLY when it expires — it did exactly
# that to 718 during the WP0 preflight. If the store grows between training and
# `extract_oos_predictions`, the latter rebuilds boundaries from the NEW store and zips them
# positionally against the OLD `training_results`, so folds mis-pair without erroring.
#
# This runner therefore serialises the exact stores it trained on and every downstream stage
# loads THOSE, not the cache.
#
# ---------------------------------------------------------------------------------------------
# USAGE (server, 32 threads)
# ---------------------------------------------------------------------------------------------
#
#   include("current_development/orderbook_layer2/r02_train_ireland.jl")

using BayesianFootball
using DataFrames, Dates, Distributions, Statistics, Printf, Serialization

const PreGame     = BayesianFootball.Models.PreGame
const Features    = BayesianFootball.Features
const Experiments = BayesianFootball.Experiments
const Diagnostics = BayesianFootball.Experiments.Diagnostics
const Data        = BayesianFootball.Data

const OUT_DIR = "./data/l2_ireland_engines"
mkpath(OUT_DIR)

# ===================================================================
# 1. Pinned DataStores
# ===================================================================

"""
    pinned_datastore(segment, tag) -> DataStore

Load from a pin if one exists, otherwise build from the cache and pin it.

The pin is the contract between this runner and everything downstream: as long as WP3 loads the
same `.jls`, `length(boundaries) == length(training_results)` holds by construction rather than
by luck.
"""
function pinned_datastore(segment, tag::String)
    path = joinpath(OUT_DIR, "ds_$(tag).jls")
    if isfile(path)
        @info "pinned_datastore: loading pin $path"
        return deserialize(path)
    end
    @info "pinned_datastore: building $tag and pinning to $path"
    ds = Data.load_datastore_cached(segment)
    serialize(path, ds)
    return ds
end

# ===================================================================
# 2. The engine — r21's winning cell, verbatim
# ===================================================================
#
# NOTE: `supremacy_weight` and `smile_weight` MUST be passed. src still ships the older keeper
# defaults (1.0 / 0.5), so omitting them silently trains a different model.

src_sup40_sw40() = PreGame.DynamicSmileDoublePoissonXGOutfieldPlayerTimeDecayModel(
    interception_config    = PreGame.HierarchicalMonthlyInterception(),
    player_dynamics_config = PreGame.OutfieldPlayerDynamicsConfig(days_half_life = 60.0),
    dispersion_config      = PreGame.HomeAwayDispersion(),
    homeadvantage_config   = PreGame.HierarchicalTeamHomeAdvantage(),
    kappa_config           = PreGame.HierarchicalTeamKappa(),
    player_ratings_feature = Features.PlayerRatingsFeature(
                                 Features.BayesianTracker(6.5, 1.0, 0.5, 0.01)),
    market_feature_config  = Features.DoublePoissonMarketFeature(),
    smile_feature          = Features.MarketSmileFeature(Kmax = 4),
    market_on              = true,
    supremacy_weight       = 0.4,
    smile_weight           = 0.4,
)

# Sampler / splitter, identical to r21 for comparability with b21.
const TARGET_SEASONS = ["2025", "2026"]
const DYNAMICS_COL   = :match_biweek     # biweekly refit; the cadence IS this column
const SAMPLES, WARMUP, CHAINS, MAX_DEPTH = 800, 300, 4, 10

# The Betfair training window for 718. See the header for why it is not the close.
const BF_OPEN_WINDOW  = (-4320.0, -360.0)
const BF_CLOSE_WINDOW = (-360.0, -180.0)

"""
    with_betfair_pillar(ds) -> DataStore

Rebuild a `DataStore` with `summarize_betfair_market` swapped into the `odds` slot.

Positional reconstruction is the documented idiom for swapping one domain
(`src/Data/types.jl:41-43`). `betfair_odds` is left in place so downstream CLV work still has
the raw tick series.
"""
function with_betfair_pillar(ds)
    odds_bf = Data.summarize_betfair_market(ds;
                  open_window = BF_OPEN_WINDOW, close_window = BF_CLOSE_WINDOW)
    isempty(odds_bf) && error("with_betfair_pillar: summarize_betfair_market returned nothing")
    return Data.DataStore(ds.segment, ds.matches, ds.statistics, odds_bf, ds.lineups,
                          ds.incidents, ds.betfair_odds, ds.bbc, ds.bbc_events)
end

"Coverage of the market pillar over the seasons that matter, so a thin pillar is visible."
function pillar_report(ds, label::String)
    m = select(ds.matches, :match_id, :season)
    j = dropmissing(leftjoin(select(ds.odds, :match_id, :market_name), m, on = :match_id), :season)
    j = filter(:season => s -> s in ("2023", "2024", "2025", "2026"), j)
    tot = combine(groupby(filter(:season => s -> s in ("2023","2024","2025","2026"), m), :season),
                  nrow => :matches)
    cov = combine(groupby(j, [:season, :market_name]), :match_id => (x -> length(unique(x))) => :n)
    println("\n--- market pillar coverage: $label ---")
    show(stdout, MIME"text/plain"(), sort(unstack(cov, :season, :market_name, :n), :season))
    println("\ntotal matches per season:")
    show(stdout, MIME"text/plain"(), sort(tot, :season))
    println()
end

# ===================================================================
# 3. Train
# ===================================================================

"""
    train_one(ds, name) -> ExperimentResults

One league, one engine, saved to `OUT_DIR`. Returns the results so the gates can run in-process.
"""
function train_one(ds, name::String)
    println("\n", "#"^90, "\n# TRAIN: $name\n", "#"^90)
    task = Experiments.create_experiment_task(
        ds, src_sup40_sw40(), name, OUT_DIR;
        target_seasons  = TARGET_SEASONS,
        history_seasons = 2,
        warmup_period   = 0,
        dynamics_col    = DYNAMICS_COL,
        samples         = SAMPLES,
        warmup          = WARMUP,
        chains          = CHAINS,
        use_queue       = true,
        max_depth       = MAX_DEPTH,
    )
    t0 = time()
    res = Experiments.run_experiment(task)
    Experiments.save_experiment(res)
    @printf("[%s] done in %.1f min, %d splits\n", name, (time() - t0) / 60,
            length(res.training_results.items))
    return res
end

# ===================================================================
# 4. Gates
# ===================================================================

"""
    gate_experiment(res, ds, label) -> NamedTuple

Three checks, in the order a failure would matter.

G-A **split count**. The queued trainer drops a split that fails NUTS initialisation WITHOUT
raising, so a short `training_results` is the only symptom. Compared against the fold count the
splitter says it should have produced.

G-B **convergence**. Max R-hat across every split and parameter.

G-C **coverage of the order-book window**. Folds must reach the biweeks the Layer-2 corpus lives
in (2026-05-29 .. 2026-08-09), or WP3 has latents for matches nobody can price.
"""
function gate_experiment(res, ds, label::String)
    println("\n", "="^90, "\nGATES: $label\n", "="^90)

    bounds  = Data.create_id_boundaries(ds, res.config.splitter)
    n_built = length(bounds)
    n_kept  = length(res.training_results.items)
    ga = n_built == n_kept
    @printf("G-A  splits kept       %3d / %3d built            %s\n", n_kept, n_built,
            ga ? "PASS" : "FAIL  <- queued trainer dropped splits SILENTLY")

    # Reuse the shipped diagnostics rather than re-deriving R-hat: `extract_chains` already
    # walks the folds and pairs each chain with its feature set, and `check_convergence` already
    # knows to drop NaN rows. Rolling our own would be a second opinion that can disagree.
    maxr, n_par, worst = NaN, 0, ""
    try
        diag = Diagnostics.check_convergence(Diagnostics.extract_chains(ds, res)).df
        if !isempty(diag)
            maxr  = maximum(diag.rhat)
            n_par = nrow(diag)
            worst = string(diag[argmax(diag.rhat), :parameter])
        end
    catch e
        @warn "convergence extraction failed" exception = e
    end
    gb = !isnan(maxr) && maxr < 1.01
    @printf("G-B  max R-hat         %.4f  (n=%d, worst=%s)   %s\n", maxr, n_par, worst,
            gb ? "PASS" : "FAIL")

    # which biweeks did the folds actually reach, in the 2026 target season?
    steps_2026 = sort(unique([md.time_step for (_, md) in bounds
                              if string(md.target_season) == "2026"]))
    ob = filter(r -> !ismissing(r.match_date) &&
                     Date(2026, 5, 29) <= r.match_date <= Date(2026, 8, 9), ds.matches)
    need = isempty(ob) ? Int[] : sort(unique(skipmissing(ob.match_biweek)))
    gc = !isempty(need) && all(n -> n in steps_2026, need)
    @printf("G-C  order-book window  biweeks needed %s\n", string(need))
    @printf("     folds reach 2026 biweeks %s        %s\n",
            string(steps_2026), gc ? "PASS" : "FAIL")

    return (label = label, splits_kept = n_kept, splits_built = n_built,
            max_rhat = maxr, ga = ga, gb = gb, gc = gc, pass = ga && gb && gc)
end

# ===================================================================
# 5. Run
# ===================================================================

println("\n", "="^90)
println("WP2  —  training src_sup40_sw40 on Ireland 79 and 718")
println("="^90)

ds79_raw  = pinned_datastore(Data.Ireland(), "ire79")
ds718_raw = pinned_datastore(Data.IrelandFirstDivision(), "ire718")

# 79 keeps SofaScore (r21-exact); 718 gets the Betfair ladder it otherwise has none of.
ds79  = ds79_raw
ds718 = with_betfair_pillar(ds718_raw)

pillar_report(ds79,  "79 (SofaScore, r21-exact)")
pillar_report(ds718, "718 (Betfair, close_window = $(BF_CLOSE_WINDOW))")

# Pin the pillar-swapped 718 too — WP3 must feed the model the same odds frame it trained on.
serialize(joinpath(OUT_DIR, "ds_ire718_bfpillar.jls"), ds718)

res79  = train_one(ds79,  "l2_ire79_sup40_sw40")
res718 = train_one(ds718, "l2_ire718_sup40_sw40")

g79  = gate_experiment(res79,  ds79,  "79 Ireland Premier")
g718 = gate_experiment(res718, ds718, "718 Ireland First Division")

println("\n", "="^90)
for g in (g79, g718)
    @printf("%-32s splits %3d/%3d   max R-hat %.4f   %s\n",
            g.label, g.splits_kept, g.splits_built, g.max_rhat, g.pass ? "PASS" : "FAIL")
end
println("="^90)
println("\nSaved to $OUT_DIR — WP3 must load the PINNED stores, not load_datastore_cached.")

nothing
