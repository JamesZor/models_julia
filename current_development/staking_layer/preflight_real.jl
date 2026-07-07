#=
PREFLIGHT — build (or reload) the real L1 inputs for the src_sup40_sw40 smile engine.

Produces `lat`, `ppd`, `odds_bf`, `ds1` in Main and caches (lat, ppd, odds_bf) to
`results/_lat_ppd_cache.jls` (gitignored) so the real races reload in seconds. Recipe mirrors the
validated staking_real preflight (experiments.md §Preflight): Ireland datastore → Betfair-swapped
eval store → load the smile experiment → extract_oos_predictions → model_inference.

Needs the experiment payload `data/double_poisson_smile_src_grid/src_sup40_sw40_*/` present (on the
kaimon server; not shipped locally). Run once per session before r02/r03/r04/r05.
=#
using BayesianFootball
using DataFrames
using Serialization
const Data        = BayesianFootball.Data
const Experiments = BayesianFootball.Experiments
const Predictions = BayesianFootball.Predictions

const _CACHE = joinpath(@__DIR__, "results", "_lat_ppd_cache.jls")

function build_real_inputs(; rebuild::Bool=false)
    ds = Data.load_datastore_cached(Data.Ireland())
    odds = Data.summarize_betfair_market(ds, open_window=(-100000.0, -10.0), close_window=(-20.0, 0.0))
    ds1 = Data.DataStore(ds.segment, ds.matches, ds.statistics, odds, ds.lineups, ds.incidents, ds.betfair_odds)

    if !rebuild && isfile(_CACHE)
        lat, ppd, odds_bf = deserialize(_CACHE)
        @info "preflight: reloaded lat/ppd/odds_bf from cache" n_odds=nrow(odds_bf)
        return (; lat, ppd, odds_bf, ds1)
    end

    hits = filter(isdir, readdir(joinpath(@__DIR__, "..", "..", "data", "double_poisson_smile_src_grid"), join=true))
    src_dir = findfirst(p -> occursin("src_sup40_sw40", p), hits)
    src_dir === nothing && error("preflight: no src_sup40_sw40 experiment payload found under data/double_poisson_smile_src_grid/")
    res = Experiments.load_experiment(hits[src_dir])

    lat = Experiments.extract_oos_predictions(ds1, res)
    ppd = Predictions.model_inference(lat)
    odds_bf = odds
    mkpath(dirname(_CACHE)); serialize(_CACHE, (lat, ppd, odds_bf))
    @info "preflight: built + cached lat/ppd/odds_bf" n_odds=nrow(odds_bf)
    return (; lat, ppd, odds_bf, ds1)
end
