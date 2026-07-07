#=
staking_layer — dev-module loader. Includes the src/ files in dependency order so a runner just
does `include(".../src/loader.jl")`. Graduates to a proper `src/` module once validated.

    l01 book schema + grid geometry + StakingMatch     (source-agnostic foundation)
    l02 Kelly solver (P)                               (pure math)
    l03 coherent pricing: blend + IPF tilt
    l04 trust interface: AbstractTrustModel + TrustHist + FlatTrust/CuratedTrust
    l05 EBTrust                                         (parity target)
    l06 BayesianTrust (Turing) + HierarchicalTrust stub
    l07 staking policies                               (needs BayesianFootball.Signals)
    l08 SimSource                                       (simulator world)
    l09 RealSource + extended book                     (needs BayesianFootball.Data.Markets)
    l10 run_race / run_ext_race + metrics + reporting
=#
const _SL = @__DIR__
"Absolute path to the staking_layer/ module dir — use this (not @__DIR__) in runners so they
resolve correctly whether included, run as a script, or pasted line-by-line at the REPL."
const STAKING_LAYER_DIR = dirname(_SL)
include(joinpath(_SL, "l01_book_schema.jl"))
include(joinpath(_SL, "l02_kelly.jl"))
include(joinpath(_SL, "l03_coherent_pricing.jl"))
include(joinpath(_SL, "l04_trust_interface.jl"))
include(joinpath(_SL, "l05_trust_eb.jl"))
include(joinpath(_SL, "l06_trust_bayes.jl"))
include(joinpath(_SL, "l07_policy.jl"))
include(joinpath(_SL, "l08_sim_source.jl"))
include(joinpath(_SL, "l09_real_source.jl"))
include(joinpath(_SL, "l10_runner.jl"))
