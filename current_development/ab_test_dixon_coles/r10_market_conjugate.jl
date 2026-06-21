# current_development/ab_test_dixon_coles/r10_market_conjugate.jl
#
# Runner: does inference-time market conditioning of the L1 goal-rate posterior
# improve BETTING growth G / ROI (not just LogLoss)?  And is optimal k contrarian?
#
# Reuses ALREADY-TRAINED chains (no MCMC resampling):
#   - DixonColes_Market           (data/dixon_coles_ab/,            r02, market_weight=0.4)
#   - DCMH_HalfLife_{14,30,60,120}(data/dixon_coles_halflife_grid/, r06, market_weight=0.4)
#
# Condition on:  SofaScore-derived flat_market_λ (built from `ds`)
# Evaluate on:   Betfair 1X2/DC line (`ds1`)  <-- held-out, different source (GUARDRAIL)
#
# Run on the mcmc-beast server (kaimon REPL). These data/ dirs live on the server.

using Revise
using BayesianFootball
using DataFrames
using Statistics
using ThreadPinning
pinthreads(:cores)

const Data        = BayesianFootball.Data
const Experiments = BayesianFootball.Experiments
const BackTesting = BayesianFootball.BackTesting
const Signals     = BayesianFootball.Signals

include("current_development/ab_test_dixon_coles/l10_market_conjugate.jl")

# ==========================================================================
# 1. DATA
# ==========================================================================
println("[INFO] Loading Ireland DataStore...")
ds = Data.load_datastore_cached(Data.Ireland())

# Betfair evaluation DataStore (held-out 1X2/DC line) — same construction as r02/r06.
odds = Data.summarize_betfair_market(ds; open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
ds1  = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)

# ==========================================================================
# 2. LOAD ALREADY-TRAINED EXPERIMENTS  (no retraining)
# ==========================================================================
function load_by_name(save_dir::String, name::String)
    paths = Experiments.list_experiments(save_dir; data_dir="")
    for p in paths
        exp = Experiments.load_experiment(p)
        if exp.config.name == name
            return exp
        end
    end
    error("Experiment '$name' not found in $save_dir")
end

exp_dc_m = load_by_name("./data/dixon_coles_ab/", "DixonColes_Market")
exp_hl14  = load_by_name("./data/dixon_coles_halflife_grid/", "DCMH_HalfLife_14")
exp_hl30  = load_by_name("./data/dixon_coles_halflife_grid/", "DCMH_HalfLife_30")
exp_hl60  = load_by_name("./data/dixon_coles_halflife_grid/", "DCMH_HalfLife_60")
exp_hl120 = load_by_name("./data/dixon_coles_halflife_grid/", "DCMH_HalfLife_120")

experiments = [exp_dc_m, exp_hl14, exp_hl30, exp_hl60, exp_hl120]

# ==========================================================================
# 3. K-SWEEP  (Transform A = location shift; + one Transform B conjugate point)
# ==========================================================================
# k=0 reproduces the r02/r06 baseline EXACTLY (sanity gate). k<0 = fade the line
# further (contrarian); k>0 = extra inference-time pull toward the SofaScore line
# ON TOP of the 0.4 training weight already baked into these models.
ks      = [-0.5, -0.25, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.2]
signals = [Signals.BayesianKelly(0.03)]                # min_edge per staking-research memory
mcfg    = Data.Markets.DEFAULT_MARKET_CONFIG

tearsheets = Dict{String, DataFrame}()
sweeps     = Dict{String, Any}()

for exp in experiments
    nm = exp.config.name
    println("\n=== Sweeping $nm ===")
    sw = sweep_experiment(ds, ds1, exp, ks, signals; market_config=mcfg, add_conjugate=true)
    ts = BackTesting.generate_tearsheet(BackTesting.BacktestLedger(sw.ledger))
    tearsheets[nm] = ts
    sweeps[nm]     = sw
end

# ==========================================================================
# 4. PRIMARY RESULT: hurdle growth G(k) on the held-out Betfair 1X2 line
# ==========================================================================
# Betfair-backed selections only (the genuine held-out line). Aggregate G across
# the 1X2 book per (model, k) by summing per-selection hurdle_G.
primary_sel = [:home, :draw, :away]

println("\n", "="^90)
println("📈 PRIMARY: hurdle G vs k  (Betfair 1X2 held-out line, summed over home/draw/away)")
println("="^90)

for exp in experiments
    nm = exp.config.name
    ts = tearsheets[nm]
    sub = subset(ts, :selection => ByRow(in(primary_sel)))
    g = combine(groupby(sub, :model_name),
                :hurdle_G       => sum => :G_1x2,
                :hurdle_G_emp   => sum => :G_emp_1x2,
                :profit         => sum => :profit,
                :roi_pct        => mean => :roi_mean,
                :bets_placed    => sum => :bets)
    # sort so the k-ordering is readable (conjugate row floats to the end by name)
    sort!(g, :model_name)
    println("\n--- $nm ---")
    show(g, allrows=true, allcols=true, truncate=0)
    println()
    best = g[argmax(g.G_1x2), :]
    println(">>> argmax_k  ($nm):  $(best.model_name)   G_1x2=$(round(best.G_1x2, digits=5))   profit=$(round(best.profit, digits=3))   bets=$(best.bets)")
end

# Full per-selection tearsheet for the headline model (DixonColes_Market)
cols = [:model_name, :selection, :bets_placed, :turnover, :profit, :roi_pct,
        :win_rate_pct, :hurdle_n_bets, :hurdle_E_R, :hurdle_sharpe, :hurdle_G, :hurdle_G_emp]
println("\n", "="^90)
println("Full per-selection tearsheet: DixonColes_Market (held-out Betfair line)")
println("="^90)
show(subset(tearsheets["DixonColes_Market"], :selection => ByRow(in(primary_sel)))[:, cols];
     allrows=true, truncate=0)
println()

# ==========================================================================
# 5. SECONDARY (flagged, optional): SofaScore O/U+BTTS eval — partly circular
#    Same source we conditioned on; report but DO NOT use for the verdict.
# ==========================================================================
# To run, swap ds_eval -> ds in sweep_experiment for an exp and inspect over/under +
# btts selections. Left commented to keep the primary run lean.
#
# sw_soft = sweep_experiment(ds, ds, exp_dc_m, ks, signals; market_config=mcfg, add_conjugate=false)
# ts_soft = BackTesting.generate_tearsheet(BackTesting.BacktestLedger(sw_soft.ledger))
# show(subset(ts_soft, :selection => ByRow(in([:over_25,:under_25,:btts_yes,:btts_no])))[:, cols]; allrows=true)

println("\nDone. Fill r10_market_conjugate_notes.md with the G(k) table + verdict.")
