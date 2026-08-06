# current_development/matchday_2026_08_08/r01_train_weekend.jl
#
# Train the three engines needed to price the 7-9 Aug 2026 slate. Overnight run, 16 cores.
#
#   julia --project -t 16 current_development/matchday_2026_08_08/r01_train_weekend.jl
#
# THE SLATE (31 fixtures)
#   Fri 07 Aug   79   x3    718 x5    55 x1
#   Sat 08 Aug   54   x2    55  x4    56  x5    57 x5
#   Sun 09 Aug   54   x4    79  x2
#
# THE THREE RUNS
#   IrelandAll   [79, 718]   poisson outfield   season "2026"    Fri + Sun Irish card
#   ScottishUpper[54, 55]    poisson outfield   season "26/27"   Sat + Sun Scottish top two
#   ScottishLower[56, 57]    funnel             season "26/27"   Sat League One + Two
#
# WHY THREE AND NOT FOUR. 56/57 have zero SofaScore stats, zero xG and zero player ratings, so no
# player-level engine can run there; the funnel's BBC shot counts are the only observable. 54/55
# and 79/718 both have ratings and xG, so they take the same engine.
#
# WHY POOLED. Promoted/relegated teams keep one rating across the tier boundary — dundalk
# (718->79) and cork-city (79->718) play this weekend, as do livingston (55->54) and st-johnstone
# (54->55). Pooling ScottishLower additionally pulls airdrieonians and ross-county (both 55->56)
# into the team_map; without the 26/27 target season they are absent and `MatchDay.check_coverage`
# refuses the whole League One card.
#
# CAVEAT worth carrying into the results: the player-level engine has NO zero-sum δ_league term
# (only the ...League... team engines do), so on the pooled segments the tier gap is absorbed into
# team α/β rather than a league offset. Harmless while every fixture is within-tier, but it means
# IrelandAll and ScottishUpper are not pooled the same way ScottishLower is.
#
# This is a SMOKE TEST of the match-day workflow, not a bid for accuracy. Paper only.

using BayesianFootball
using DataFrames, Dates
using Turing, MCMCChains
using ThreadPinning

pinthreads(:cores)
@info "threads" n = Threads.nthreads()

const Data        = BayesianFootball.Data
const Experiments = BayesianFootball.Experiments

include(joinpath(@__DIR__, "l01_weekend_training.jl"))

const SAVE_DIR = "./data/matchday_wknd_0808/"

# 16 cores. The queued sampler flattens folds x chains into one global queue, so the useful knob
# is `max_concurrent_tasks`, not `max_concurrent_splits`. Each run below produces 2-3 folds, i.e.
# 16-24 queued chains against 16 workers.
const SAMPLER = (samples = 1000, warmup = 500, chains = 8, max_concurrent_tasks = 16)

# ===================================================================
# 1. Load. Do NOT rebuild these caches between now and inference.
# ===================================================================
# `MatchDay.select_split` pairs a trained chain with a REBUILT boundary list by index. Playing
# another round changes the boundary count, and the pairing then silently conditions on a stale
# window (the Q11 warning). The caches were rebuilt 2026-08-06; leave them.

ds_ire = Data.load_datastore_cached(Data.IrelandAll())        # first call builds it (~15 min)
ds_up  = Data.load_datastore_cached(Data.ScottishUpper())
ds_low = Data.load_datastore_cached(Data.ScottishLower())

for (nm, ds) in (("IrelandAll", ds_ire), ("ScottishUpper", ds_up), ("ScottishLower", ds_low))
    @info nm matches = nrow(ds.matches) last = maximum(ds.matches.match_date)
end

# ===================================================================
# 2. Tasks. Warmup is derived per segment, never typed -- see l01.
# ===================================================================
task_ire = weekend_task(ds_ire, poisson_outfield_model(), "ire_pooled_poisson_outfield",
                        SAVE_DIR, "2026"; SAMPLER...)

task_up  = weekend_task(ds_up,  poisson_outfield_model(), "scot_upper_poisson_outfield",
                        SAVE_DIR, "26/27"; SAMPLER...)

task_low = weekend_task(ds_low, funnel_model(),           "scot_lower_funnel",
                        SAVE_DIR, "26/27"; SAMPLER...)

# ===================================================================
# 3. Run. Sequential across segments -- the queue already saturates all 16 cores within a run,
#    so overlapping them would only trade wall-clock for contention. Each is saved as it lands,
#    so a failure in run 3 does not cost runs 1 and 2.
# ===================================================================
for (label, task) in (("IrelandAll [79,718]",   task_ire),
                      ("ScottishUpper [54,55]", task_up),
                      ("ScottishLower [56,57]", task_low))
    @info "=== training $label ===" started = now()
    t0 = time()
    try
        res = Experiments.run_experiment(task)
        Experiments.save_experiment(res)
        @info "done $label" folds = length(res.training_results) minutes = round((time()-t0)/60, digits=1)
    catch e
        @error "FAILED $label -- continuing to the next segment" exception = (e, catch_backtrace())
    end
end

# ===================================================================
# 4. What to check in the morning, before trusting any of it
# ===================================================================
# The single most important line: `folds` above must be > 1 for every segment. A run reporting
# `folds = 1` is the r05 failure -- history only, target season never seen. l01's `assert_splits`
# should have refused it before sampling, but check the number that actually came back.

@info "saved experiments" list = Experiments.list_experiments(SAVE_DIR, data_dir = "")

# Then, per experiment:
#   chains_df = Experiments.Diagnostics.extract_chains(ds, expr)
#   Experiments.Diagnostics.check_convergence(chains_df)     # R-hat, ESS
#
# Inference for the weekend is r02, once the order-book drain and the lineup scrape are up.
# Note the funnel run needs NO lineups, so ScottishLower can be priced even if the XI scrape is
# still down on Saturday morning.
