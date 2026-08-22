# current_development/scottish_lower/open_play/r10_betfair_backtest_all_models.jl
#
# BETFAIR EXCHANGE HISTORICAL BACKTEST: All 6 Models across 24/25 & 25/26 Seasons
#
# Models:
# 1. goals_pois_ctl_hl365_hs2
# 2. goals_pois_open_play_hl365_hs2
# 3. recomb_pois_integrated_hl365_hs2
# 4. goals_negbin_ctl_hl365_hs2
# 5. goals_negbin_open_play_hl365_hs2
# 6. recomb_negbin_integrated_hl365_hs2

using Revise
using BayesianFootball
using DataFrames, Dates, Statistics, Printf, LinearAlgebra, Serialization

const PreGame     = BayesianFootball.Models.PreGame
const Experiments = BayesianFootball.Experiments
const Portfolio   = BayesianFootball.Portfolio
const Data        = BayesianFootball.Data

const ROOT = pkgdir(BayesianFootball)
include("l01_open_play_feature.jl")
include("l02_open_play_engines.jl")
include("l03_recombination_models.jl")

println("\n", "="^95)
println("🚀 BETFAIR EXCHANGE HISTORICAL BACKTEST: ALL 6 MODELS (24/25 & 25/26, 2% COMM, BM 800)")
println("="^95)

# 1. Load DataStore & Betfair Closed Odds
ds = Data.load_datastore_cached(Data.ScottishLower(), max_age_hours = 10000)
bf_odds = Data.summarize_betfair_market(ds; open_window = (-100000.0, -10.0), close_window = (-20.0, 0.0))
println("✓ Loaded Betfair Historical Odds: $(nrow(bf_odds)) rows")

# 2. Discover & Load Experiments
ctl_folders = Experiments.list_experiments("scottish_negbin_grid"; data_dir = joinpath(ROOT, "data"))
op_folders  = Experiments.list_experiments("scottish_open_play_grid"; data_dir = joinpath(ROOT, "data"))
all_folders = vcat(ctl_folders, op_folders)
all_loaded  = Experiments.load_experiments(all_folders)

target_models = [
    "goals_pois_ctl_hl365_hs2",
    "goals_pois_open_play_hl365_hs2",
    "recomb_pois_integrated_hl365_hs2",
    "goals_negbin_ctl_hl365_hs2",
    "goals_negbin_open_play_hl365_hs2",
    "recomb_negbin_integrated_hl365_hs2"
]

experiments_dict = Dict{String, Any}()
for exp in all_loaded
    for t in target_models
        if startswith(exp.config.name, t)
            experiments_dict[t] = exp
        end
    end
end
experiments = [experiments_dict[t] for t in target_models if haskey(experiments_dict, t)]
println("✓ Loaded $(length(experiments))/$(length(target_models)) target experiments")

# 3. Filter Target Match IDs (Seasons 24/25 & 25/26)
target_matches = filter(r -> hasproperty(r, :season) && (r.season == "24/25" || r.season == "25/26"), ds.matches)
target_match_ids = Set(target_matches.match_id)
println("✓ Target OOS Test Matches: $(length(target_match_ids)) matches across 24/25 & 25/26")

# 4. Multi-Market Portfolio Spec
MARKETS = Data.MarketConfig(reduce(vcat, (
    Data.AbstractMarket[Data.Market1X2(), Data.MarketBTTS()],
    [Data.MarketOverUnder(i + 0.5) for i in 0:4],
)))

spec = Portfolio.BookSpec(
    markets   = MARKETS,
    price     = Portfolio.DeArb(),
    allocator = Portfolio.KellyLogUtility(),
    shrink    = Portfolio.BakerMcHale(n_draws = 800),
    exec      = Portfolio.ExecutionConfig(
                    commission = Portfolio.PerBetCommission(0.02),
                    max_selection_stake = 0.50,
                    budget = 0.99,
                    require_complete_markets = true
                )
)

CACHE_DIR = joinpath(@__DIR__, "cache")
mkpath(CACHE_DIR)

books_map = Dict{String, Vector{Portfolio.MatchBook}}()
for exp in experiments
    m_name = exp.config.name
    cache_file = joinpath(CACHE_DIR, "books_bf_$(m_name)_bm800.jls")
    
    if isfile(cache_file) && get(ENV, "REBUILD_BOOKS", "0") != "1"
        @info "Reusing cached Betfair MatchBooks for: $m_name" cache_file
        books_map[m_name] = deserialize(cache_file)
    else
        @info "Building Betfair MatchBooks for: $m_name..."
        oos_latents = Experiments.extract_oos_predictions(ds, exp)
        target_df = filter(r -> r.match_id in target_match_ids, oos_latents.df)
        t0 = time()
        b = Portfolio.build_books(spec, target_df, exp, bf_odds, ds)
        elapsed = round(time() - t0, digits = 1)
        @info "Completed Betfair MatchBooks for $m_name in $(elapsed)s" n_books=length(b)
        serialize(cache_file, b)
        books_map[m_name] = b
    end
end

all_slates = Dict{String, Vector{Portfolio.Slate}}()
for (m_name, b) in books_map
    all_slates[m_name] = Portfolio.group(Portfolio.DailySlate(), b)
end

policies = [
    ("Balanced Growth (Cap 15%, λ=15)", Portfolio.PolicySpec(
        trust    = Portfolio.FlatTrust(0.25),
        risk     = Portfolio.SlateDrawdown(15.0),
        cap      = Portfolio.FixedCap(0.15),
        filter   = Portfolio.KeepAll(),
        grouping = Portfolio.DailySlate()
    )),
    ("Conservative (Cap 10%, λ=23)", Portfolio.PolicySpec(
        trust    = Portfolio.FlatTrust(0.25),
        risk     = Portfolio.SlateDrawdown(23.0),
        cap      = Portfolio.FixedCap(0.10),
        filter   = Portfolio.KeepAll(),
        grouping = Portfolio.DailySlate()
    )),
    ("Aggressive (Cap 25%, λ=10)", Portfolio.PolicySpec(
        trust    = Portfolio.FlatTrust(0.50),
        risk     = Portfolio.SlateDrawdown(10.0),
        cap      = Portfolio.FixedCap(0.25),
        filter   = Portfolio.KeepAll(),
        grouping = Portfolio.DailySlate()
    ))
]

for (pol_name, pol) in policies
    println("\n", "="^95)
    println("BETFAIR EXCHANGE SIMULATION: $pol_name (2% Comm, BM 800 Draws, 710 Matches)")
    println("="^95)
    
    port_df = DataFrame(
        model        = String[],
        final_wealth = Float64[],
        growth_slate = Float64[],
        roi_pct      = Float64[],
        mean_expo    = Float64[],
        mdd_pct      = Float64[],
        sharpe       = Float64[],
        total_bets   = Int[]
    )
    
    for exp in experiments
        m_name = exp.config.name
        haskey(all_slates, m_name) || continue
        slates = all_slates[m_name]
        traj = Portfolio.simulate(pol, slates; use_shrink = true)
        m = Portfolio.path_metrics(traj)
        
        ret_series = traj.slate_pl
        sh = length(ret_series) > 1 && std(ret_series) > 1e-6 ? (mean(ret_series) / std(ret_series)) * sqrt(35) : 0.0
        
        total_bets = isempty(books_map[m_name]) ? 0 : sum(sum(bk.a_kelly .> 1e-5) for bk in books_map[m_name]; init = 0)
        
        push!(port_df, (
            model        = m_name,
            final_wealth = round(m.final, digits=3),
            growth_slate = round(m.growth_per_slate, digits=5),
            roi_pct      = round(m.roi, digits=2),
            mean_expo    = round(m.mean_exposure * 100, digits=1),
            mdd_pct      = round(m.mdd, digits=2),
            sharpe       = round(sh, digits=2),
            total_bets   = total_bets
        ))
    end
    sort!(port_df, :final_wealth, rev = true)
    show(stdout, MIME("text/plain"), port_df)
    println()
end

println("\n", "="^95)
println("✓ BETFAIR EXCHANGE SIMULATION COMPLETE!")
println("="^95)
